#include "PluginEditor.h"
#include "PluginProcessor.h"

namespace
{
    const juce::Colour kBg       { 0xff121418 };
    const juce::Colour kPanel    { 0xff1b1e24 };
    const juce::Colour kText     { 0xffd8dce4 };
    const juce::Colour kDim      { 0xff8a91a0 };
    const juce::Colour kVae      { 0xff6fd3a0 };   // VAE active
    const juce::Colour kFallback { 0xffe8b04a };   // plain crossfade fallback
}

//==============================================================================
void SIDVaeAudioProcessorEditor::WaveformView::setTable (const std::vector<float>& t, bool vae)
{
    table   = t;
    vaeMode = vae;
    repaint();
}

void SIDVaeAudioProcessorEditor::WaveformView::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    g.setColour (kPanel);
    g.fillRoundedRectangle (r, 6.0f);

    r = r.reduced (10.0f, 12.0f);
    g.setColour (kDim.withAlpha (0.35f));
    g.drawHorizontalLine ((int) r.getCentreY(), r.getX(), r.getRight());

    if (table.size() < 2)
        return;

    juce::Path path;
    const int   n    = (int) table.size();
    const int   step = juce::jmax (1, n / (int) juce::jmax (1.0f, r.getWidth() * 2.0f));
    for (int i = 0; i <= n; i += step)
    {
        const float v = table[(size_t) (i % n)];
        const float x = r.getX() + r.getWidth() * (float) i / (float) n;
        const float y = r.getCentreY() - juce::jlimit (-1.0f, 1.0f, v) * r.getHeight() * 0.5f;
        if (i == 0) path.startNewSubPath (x, y);
        else        path.lineTo (x, y);
    }

    g.setColour (vaeMode ? kVae : kFallback);
    g.strokePath (path, juce::PathStrokeType (1.8f, juce::PathStrokeType::curved));
}

//==============================================================================
SIDVaeAudioProcessorEditor::SIDVaeAudioProcessorEditor (SIDVaeAudioProcessor& p)
    : juce::AudioProcessorEditor (&p), proc (p)
{
    status.setJustificationType (juce::Justification::centredRight);
    status.setFont (juce::FontOptions (12.0f));
    status.setMinimumHorizontalScale (0.6f);
    addAndMakeVisible (status);
    addAndMakeVisible (waveView);

    setupChoice (waveA, "waveA", "Wave A");
    setupKnob   (morph, "morph", "Morph");
    setupChoice (waveB, "waveB", "Wave B");
    setupKnob   (pw,    "pw",    "Pulse W");
    setupKnob   (vel,   "vel",   "Velocity");

    setupKnob   (z1,    "z1",    "Latent 1");
    setupKnob   (z2,    "z2",    "Latent 2");
    setupKnob   (z3,    "z3",    "Latent 3");
    setupKnob   (freq,  "freq",  "Freq");
    setupKnob   (out,   "out",   "Output");

    setSize (760, 420);
    timerCallback();
    startTimerHz (30);
}

SIDVaeAudioProcessorEditor::~SIDVaeAudioProcessorEditor()
{
    stopTimer();
}

void SIDVaeAudioProcessorEditor::setupKnob (Knob& k, const juce::String& paramId, const juce::String& text)
{
    k.slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    k.slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 70, 18);
    k.slider.setColour (juce::Slider::rotarySliderFillColourId, kVae.withAlpha (0.8f));
    k.slider.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible (k.slider);

    k.label.setText (text, juce::dontSendNotification);
    k.label.setJustificationType (juce::Justification::centred);
    k.label.setColour (juce::Label::textColourId, kDim);
    addAndMakeVisible (k.label);

    k.attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (proc.apvts, paramId, k.slider);
}

void SIDVaeAudioProcessorEditor::setupChoice (Choice& c, const juce::String& paramId, const juce::String& text)
{
    // Items must exist before the attachment is created.
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*> (proc.apvts.getParameter (paramId)))
        c.box.addItemList (param->choices, 1);
    addAndMakeVisible (c.box);

    c.label.setText (text, juce::dontSendNotification);
    c.label.setJustificationType (juce::Justification::centred);
    c.label.setColour (juce::Label::textColourId, kDim);
    addAndMakeVisible (c.label);

    c.attachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (proc.apvts, paramId, c.box);
}

//==============================================================================
void SIDVaeAudioProcessorEditor::timerCallback()
{
    const bool vae = proc.isVaeActive();
    status.setText (proc.getStatusText(), juce::dontSendNotification);
    status.setColour (juce::Label::textColourId, vae ? kVae : kFallback);

    proc.copyDisplayTable (scratch);
    waveView.setTable (scratch, vae);
}

void SIDVaeAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (kBg);
    g.setColour (kText);
    g.setFont (juce::FontOptions (20.0f, juce::Font::bold));
    g.drawText ("SIDVae", 14, 8, 120, 26, juce::Justification::centredLeft);
}

void SIDVaeAudioProcessorEditor::layoutKnob (Knob& k, juce::Rectangle<int> r)
{
    k.label.setBounds (r.removeFromTop (18));
    k.slider.setBounds (r.reduced (4, 0));
}

void SIDVaeAudioProcessorEditor::layoutChoice (Choice& c, juce::Rectangle<int> r)
{
    c.label.setBounds (r.removeFromTop (18));
    c.box.setBounds (r.withSizeKeepingCentre (juce::jmin (r.getWidth() - 12, 120), 26));
}

void SIDVaeAudioProcessorEditor::resized()
{
    auto r = getLocalBounds().reduced (12, 8);

    auto header = r.removeFromTop (28);
    header.removeFromLeft (120);
    status.setBounds (header);

    r.removeFromTop (6);
    waveView.setBounds (r.removeFromTop (130));
    r.removeFromTop (10);

    const int rowH = r.getHeight() / 2;
    auto row1 = r.removeFromTop (rowH);
    auto row2 = r;
    const int w = row1.getWidth() / 5;

    layoutChoice (waveA, row1.removeFromLeft (w));
    layoutKnob   (morph, row1.removeFromLeft (w));
    layoutChoice (waveB, row1.removeFromLeft (w));
    layoutKnob   (pw,    row1.removeFromLeft (w));
    layoutKnob   (vel,   row1);

    layoutKnob   (z1,    row2.removeFromLeft (w));
    layoutKnob   (z2,    row2.removeFromLeft (w));
    layoutKnob   (z3,    row2.removeFromLeft (w));
    layoutKnob   (freq,  row2.removeFromLeft (w));
    layoutKnob   (out,   row2);
}
