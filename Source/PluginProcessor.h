#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <atomic>
#include <memory>
#include <vector>

#include "core/TripleBuffer.h"
#include "core/WavetableOsc.h"

class SIDVaeAudioProcessor : public juce::AudioProcessor
{
public:
    using APVTS = juce::AudioProcessorValueTreeState;
    static constexpr int kTableLen = 2048;

    SIDVaeAudioProcessor();
    ~SIDVaeAudioProcessor() override;

    //==================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override  { return true; }   // instrument for hosts; MIDI not used yet
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==================================================================
    // For the editor (message thread).
    bool isVaeActive() const noexcept { return vaeActive.load(); }
    juce::String getStatusText() const;
    void copyDisplayTable (std::vector<float>& dst) const;

    static APVTS::ParameterLayout createParameterLayout();
    APVTS apvts;

private:
    class TableBuilder;          // background thread: runs the decoder, publishes tables
    friend class TableBuilder;

    void setStatus (const juce::String& s);

    // Parameters (raw atomics, safe to read from any thread)
    std::atomic<float>* pWaveA  = nullptr;
    std::atomic<float>* pWaveB  = nullptr;
    std::atomic<float>* pMorph  = nullptr;
    std::atomic<float>* pPw     = nullptr;
    std::atomic<float>* pZ[3]   = { nullptr, nullptr, nullptr };
    std::atomic<float>* pVel    = nullptr;
    std::atomic<float>* pFreq   = nullptr;
    std::atomic<float>* pOutDb  = nullptr;

    // Audio path
    sidvae::TableTripleBuffer tables;   // builder thread -> audio thread
    sidvae::WavetableOsc      osc;
    std::atomic<bool>         rebuildRequested { true };

    // Status / display (message thread reads, builder thread writes)
    std::atomic<bool>          vaeActive { false };
    juce::CriticalSection      infoLock;
    juce::String               statusText { "Starting..." };
    std::vector<float>         displayTable;

    std::unique_ptr<TableBuilder> builder;   // declared last: started last, stopped first

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SIDVaeAudioProcessor)
};
