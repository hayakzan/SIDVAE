#include "PluginProcessor.h"
#include "PluginEditor.h"

#include "core/FallbackTables.h"
#include "core/SidCondition.h"
#include "core/SidVaeDecoder.h"
#include "core/TableUtils.h"

#include <algorithm>
#include <cmath>

//==============================================================================
// Model discovery + metadata
//==============================================================================
namespace
{
    const juce::String kDefaultModelFile = "sid_vae_decoder.onnx";
    const juce::String kMetaFile         = "sid_vae_meta.json";

    // Where the decoder is looked for, in order.
    juce::Array<juce::File> candidateModelDirs()
    {
        juce::Array<juce::File> dirs;

        // 1. Explicit override, handy while experimenting with different models.
        const auto env = juce::SystemStats::getEnvironmentVariable ("SIDVAE_MODEL_DIR", {});
        if (env.isNotEmpty() && juce::File::isAbsolutePath (env))
            dirs.add (juce::File (env));

        // 2. Inside the plugin bundle (CMake copies Models/ here).
        //    macOS: <bundle>/Contents/MacOS/<binary>  ->  <bundle>/Contents/Resources/Models
        const auto exeDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
        dirs.add (exeDir.getParentDirectory().getChildFile ("Resources").getChildFile ("Models"));
        dirs.add (exeDir.getChildFile ("Models"));

        // 3. Per-user install location.
        auto userData = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory);
       #if JUCE_MAC
        userData = userData.getChildFile ("Application Support");
       #endif
        dirs.add (userData.getChildFile ("SIDVae").getChildFile ("Models"));

        // 4. The source tree's Models/ folder (development builds).
       #ifdef SIDVAE_DEV_MODEL_DIR
        dirs.add (juce::File (SIDVAE_DEV_MODEL_DIR));
       #endif
        return dirs;
    }

    struct ModelMeta
    {
        int latentDim = 3;
        int condDim   = 6;
        std::vector<std::string> condOrder = sidvae::defaultCondOrder();
        sidvae::LogF0Norm f0Norm;
        juce::String decoderFile = kDefaultModelFile;
    };

    // Reads sid_vae_meta.json. Missing optional keys keep their defaults.
    bool parseMeta (const juce::File& file, ModelMeta& m, juce::String& error)
    {
        const auto json = juce::JSON::parse (file);
        if (! json.isObject())
        {
            error = "could not parse " + file.getFileName();
            return false;
        }

        if (json.hasProperty ("latent_dim"))   m.latentDim   = (int) json["latent_dim"];
        if (json.hasProperty ("cond_dim"))     m.condDim     = (int) json["cond_dim"];
        if (json.hasProperty ("decoder_file")) m.decoderFile = json["decoder_file"].toString();

        if (auto* arr = json["cond_order"].getArray())
        {
            m.condOrder.clear();
            for (const auto& v : *arr)
                m.condOrder.push_back (v.toString().toStdString());
        }

        if (json.hasProperty ("log_f0_mean") && json.hasProperty ("log_f0_std"))
        {
            m.f0Norm.enabled = true;
            m.f0Norm.mean    = (float) (double) json["log_f0_mean"];
            m.f0Norm.std     = (float) (double) json["log_f0_std"];
        }

        if ((int) m.condOrder.size() != m.condDim)
        {
            error = "cond_order has " + juce::String ((int) m.condOrder.size())
                  + " names but cond_dim is " + juce::String (m.condDim);
            return false;
        }
        if (m.latentDim <= 0 || m.condDim <= 0)
        {
            error = "latent_dim / cond_dim must be positive";
            return false;
        }
        return true;
    }
}

//==============================================================================
// TableBuilder: owns the decoder, watches the parameters, publishes tables.
// Runs on its own thread so ONNX inference never touches the audio thread.
//==============================================================================
class SIDVaeAudioProcessor::TableBuilder : public juce::Thread
{
public:
    explicit TableBuilder (SIDVaeAudioProcessor& p)
        : juce::Thread ("SIDVae table builder"), proc (p) {}

    ~TableBuilder() override { stopThread (4000); }

    void run() override
    {
        loadModel();

        Snapshot last;
        bool haveLast = false;

        while (! threadShouldExit())
        {
            const bool forced = proc.rebuildRequested.exchange (false);
            const Snapshot s  = readParams();

            if (forced || ! haveLast || ! s.sameAs (last, decoder.isReady()))
            {
                float* dst = proc.tables.writeBuffer();
                buildTable (s, dst);
                {
                    const juce::ScopedLock sl (proc.infoLock);
                    std::copy (dst, dst + kTableLen, proc.displayTable.begin());
                }
                proc.tables.publish();
                last = s;
                haveLast = true;
            }

            wait (5);   // ~200 Hz control rate; decodes only when something changed
        }
    }

private:
    struct Snapshot
    {
        int   waveA = 0, waveB = 1;
        float morph = 0.0f, pw = 0.5f, vel = 1.0f, freq = 220.0f;
        float z[3] = { 0.0f, 0.0f, 0.0f };

        bool sameAs (const Snapshot& o, bool freqMatters) const
        {
            auto close = [] (float a, float b) { return std::abs (a - b) < 1.0e-5f; };
            if (waveA != o.waveA || waveB != o.waveB) return false;
            if (! close (morph, o.morph) || ! close (pw, o.pw) || ! close (vel, o.vel)) return false;
            for (int i = 0; i < 3; ++i) if (! close (z[i], o.z[i])) return false;
            // f0 is a decoder input (log_f0), so pitch changes re-decode in VAE mode.
            if (freqMatters && std::abs (freq - o.freq) > 0.001f * o.freq) return false;
            return true;
        }
    };

    Snapshot readParams() const
    {
        Snapshot s;
        s.waveA = juce::jlimit (0, sidvae::kNumWaves - 1, (int) proc.pWaveA->load());
        s.waveB = juce::jlimit (0, sidvae::kNumWaves - 1, (int) proc.pWaveB->load());
        s.morph = juce::jlimit (0.0f, 1.0f, proc.pMorph->load());
        s.pw    = proc.pPw->load();
        s.vel   = proc.pVel->load();
        s.freq  = proc.pFreq->load();
        for (int i = 0; i < 3; ++i) s.z[i] = proc.pZ[i]->load();
        return s;
    }

    void loadModel()
    {
        juce::File modelDir;
        for (const auto& dir : candidateModelDirs())
        {
            if (dir.getChildFile (kDefaultModelFile).existsAsFile()
                || dir.getChildFile (kMetaFile).existsAsFile())
            {
                modelDir = dir;
                break;
            }
        }

        if (modelDir == juce::File())
        {
            loadError = "no " + kDefaultModelFile + " found (see README: Model files)";
            refreshStatus();
            return;
        }

        // Meta next to the model, or one level up 
        auto metaFile = modelDir.getChildFile (kMetaFile);
        if (! metaFile.existsAsFile())
            metaFile = modelDir.getParentDirectory().getChildFile (kMetaFile);

        if (metaFile.existsAsFile())
        {
            juce::String metaErr;
            if (! parseMeta (metaFile, meta, metaErr))
            {
                loadError = metaErr;
                refreshStatus();
                return;
            }
        }
        else
        {
            metaNote = "no " + kMetaFile + ", using defaults";
        }

        const auto modelFile = modelDir.getChildFile (meta.decoderFile);
        const sidvae::SidVaeDecoder::Config cfg { meta.latentDim, meta.condDim, kTableLen };

        std::string err;
        if (! decoder.load (modelFile.getFullPathName().toStdString(), cfg, err))
        {
            loadError = juce::String (err);
            refreshStatus();
            return;
        }

        // Warn about any conditioning names this build doesn't know how to fill.
        std::vector<std::string> unknown;
        std::vector<float> scratch (meta.condOrder.size());
        sidvae::buildCondition ({}, meta.condOrder, meta.f0Norm, scratch.data(), &unknown);
        for (const auto& u : unknown) unknownCond.add (juce::String (u));

        if (meta.latentDim != 3)
            metaNote = "latent_dim is " + juce::String (meta.latentDim) + "; only the first 3 have knobs";

        modelInfo = modelFile.getFileName() + " | " + juce::String (decoder.ioSummary());
        refreshStatus();
    }

    void buildTable (const Snapshot& s, float* dst)
    {
        // Waveform mix at this morph position: one-hot at the poles.
        float w[sidvae::kNumWaves] = { 0.0f, 0.0f, 0.0f };
        w[s.waveA] += 1.0f - s.morph;
        w[s.waveB] += s.morph;

        if (decoder.isReady())
        {
            sidvae::SidControls c;
            c.pw = s.pw;  c.f0Hz = s.freq;  c.vel = s.vel;
            c.waveTri = w[0];  c.waveSaw = w[1];  c.wavePulse = w[2];

            cond.resize (meta.condOrder.size());
            sidvae::buildCondition (c, meta.condOrder, meta.f0Norm, cond.data());

            z.assign ((std::size_t) meta.latentDim, 0.0f);
            for (int i = 0; i < std::min (3, meta.latentDim); ++i) z[(std::size_t) i] = s.z[i];

            std::string err;
            if (decoder.decode (z.data(), cond.data(), decoded, &err))
            {
                sidvae::removeDC   (decoded.data(), decoded.size());
                sidvae::limitPeak  (decoded.data(), decoded.size(), 1.0f);
                std::copy (decoded.begin(), decoded.end(), dst);
                if (decodeError.isNotEmpty()) { decodeError.clear(); refreshStatus(); }
                return;
            }
            if (decodeError != juce::String (err)) { decodeError = juce::String (err); refreshStatus(); }
        }

        // Fallback: plain weighted sum of reference shapes (shape-domain crossfade).
        ensureFallbackTables (s.pw);
        for (int i = 0; i < kTableLen; ++i)
        {
            const auto k = (std::size_t) i;
            dst[i] = w[0] * fallback[0][k] + w[1] * fallback[1][k] + w[2] * fallback[2][k];
        }
        sidvae::limitPeak (dst, (std::size_t) kTableLen, 1.0f);
    }

    void ensureFallbackTables (float pw)
    {
        if (fallback.empty())
        {
            fallback.push_back (sidvae::makeTriangle (kTableLen));
            fallback.push_back (sidvae::makeSaw (kTableLen));
            fallback.push_back (sidvae::makePulse (kTableLen, pw));
            fallbackPw = pw;
        }
        else if (std::abs (pw - fallbackPw) > 1.0e-4f)
        {
            fallback[2] = sidvae::makePulse (kTableLen, pw);
            fallbackPw = pw;
        }
    }

    void refreshStatus()
    {
        const bool vae = decoder.isReady() && decodeError.isEmpty();
        proc.vaeActive.store (vae);

        juce::String s;
        if (vae)
        {
            s = "VAE  |  " + modelInfo;
            if (unknownCond.size() > 0) s << "  |  unknown cond: " << unknownCond.joinIntoString (", ");
        }
        else
        {
            s = "FALLBACK: plain crossfade, no VAE  |  "
              + (decodeError.isNotEmpty() ? "decode failed: " + decodeError : loadError);
        }
        if (metaNote.isNotEmpty()) s << "  |  " << metaNote;

        proc.setStatus (s);
    }

    SIDVaeAudioProcessor& proc;

    sidvae::SidVaeDecoder decoder;
    ModelMeta meta;
    juce::String modelInfo, loadError, decodeError, metaNote;
    juce::StringArray unknownCond;

    std::vector<float> z, cond, decoded;
    std::vector<std::vector<float>> fallback;
    float fallbackPw = -1.0f;
};

//==============================================================================
// Parameters
//==============================================================================
SIDVaeAudioProcessor::APVTS::ParameterLayout SIDVaeAudioProcessor::createParameterLayout()
{
    using namespace juce;
    std::vector<std::unique_ptr<RangedAudioParameter>> p;

    auto decimals = [] (int places, const String& label = {})
    {
        return AudioParameterFloatAttributes()
            .withLabel (label)
            .withStringFromValueFunction ([places] (float v, int) { return String (v, places); });
    };

    // Index order must match sidvae::Wave (Triangle, Saw, Pulse).
    const StringArray waves { "Triangle", "Saw", "Pulse" };
    p.push_back (std::make_unique<AudioParameterChoice> (ParameterID { "waveA", 1 }, "Wave A", waves, 0));
    p.push_back (std::make_unique<AudioParameterChoice> (ParameterID { "waveB", 1 }, "Wave B", waves, 1));

    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { "morph", 1 }, "Morph",
                                                        NormalisableRange<float> (0.0f, 1.0f), 0.0f, decimals (2)));
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { "pw", 1 }, "Pulse Width",
                                                        NormalisableRange<float> (0.02f, 0.98f), 0.5f, decimals (2)));

    for (int i = 1; i <= 3; ++i)
        p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { "z" + String (i), 1 }, "Latent " + String (i),
                                                            NormalisableRange<float> (-3.0f, 3.0f), 0.0f, decimals (2)));

    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { "vel", 1 }, "Velocity",
                                                        NormalisableRange<float> (0.0f, 1.0f), 1.0f, decimals (2)));

    NormalisableRange<float> freqRange (20.0f, 5000.0f);
    freqRange.setSkewForCentre (300.0f);
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { "freq", 1 }, "Frequency", freqRange, 220.0f,
                                                        decimals (1, "Hz")));

    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { "out", 1 }, "Output",
                                                        NormalisableRange<float> (-60.0f, 0.0f), -12.0f,
                                                        decimals (1, "dB")));

    return { p.begin(), p.end() };
}

//==============================================================================
SIDVaeAudioProcessor::SIDVaeAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMS", createParameterLayout())
{
    pWaveA = apvts.getRawParameterValue ("waveA");
    pWaveB = apvts.getRawParameterValue ("waveB");
    pMorph = apvts.getRawParameterValue ("morph");
    pPw    = apvts.getRawParameterValue ("pw");
    for (int i = 0; i < 3; ++i)
        pZ[i] = apvts.getRawParameterValue ("z" + juce::String (i + 1));
    pVel   = apvts.getRawParameterValue ("vel");
    pFreq  = apvts.getRawParameterValue ("freq");
    pOutDb = apvts.getRawParameterValue ("out");

    tables.prepare (kTableLen);
    displayTable.assign ((size_t) kTableLen, 0.0f);

    builder = std::make_unique<TableBuilder> (*this);
    builder->startThread();
}

SIDVaeAudioProcessor::~SIDVaeAudioProcessor()
{
    builder.reset();   // stops the thread before anything it touches is destroyed
}

void SIDVaeAudioProcessor::prepareToPlay (double sampleRate, int)
{
    osc.prepare (sampleRate, kTableLen);
    rebuildRequested.store (true);   // make sure a fresh table arrives after the reset
}

bool SIDVaeAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::disabled())
        return false;
    const auto out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

void SIDVaeAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused (midi);

    const int n = buffer.getNumSamples();
    if (buffer.getNumChannels() == 0 || n == 0)
        return;

    if (tables.acquire())
        osc.setTargetTable (tables.readBuffer());

    const float gain = juce::Decibels::decibelsToGain (pOutDb->load(), -60.0f);
    osc.render (buffer.getWritePointer (0), n, pFreq->load(), gain);

    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        buffer.copyFrom (ch, 0, buffer, 0, 0, n);
}

//==============================================================================
void SIDVaeAudioProcessor::setStatus (const juce::String& s)
{
    const juce::ScopedLock sl (infoLock);
    statusText = s;
}

juce::String SIDVaeAudioProcessor::getStatusText() const
{
    const juce::ScopedLock sl (infoLock);
    return statusText;
}

void SIDVaeAudioProcessor::copyDisplayTable (std::vector<float>& dst) const
{
    const juce::ScopedLock sl (infoLock);
    dst = displayTable;
}

//==============================================================================
void SIDVaeAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void SIDVaeAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        if (xml->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessorEditor* SIDVaeAudioProcessor::createEditor()
{
    return new SIDVaeAudioProcessorEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SIDVaeAudioProcessor();
}
