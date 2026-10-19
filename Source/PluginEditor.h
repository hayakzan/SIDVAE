#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <memory>
#include <vector>

class SIDVaeAudioProcessor;

class SIDVaeAudioProcessorEditor : public juce::AudioProcessorEditor,
                                   private juce::Timer
{
public:
    explicit SIDVaeAudioProcessorEditor (SIDVaeAudioProcessor&);
    ~SIDVaeAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    // Renders the active single-cycle wavetable and indicates VAE vs. fallback status.
    class WaveformView : public juce::Component
    {
    public:
        void setTable (const std::vector<float>& t, bool vae);
        void paint (juce::Graphics&) override;
    private:
        std::vector<float> table;
        bool vaeMode = false;
    };

    struct Knob
    {
        juce::Slider slider;
        juce::Label  label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
    };

    struct Choice
    {
        juce::ComboBox box;
        juce::Label    label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> attachment;
    };

    void setupKnob   (Knob&,   const juce::String& paramId, const juce::String& text);
    void setupChoice (Choice&, const juce::String& paramId, const juce::String& text);

    static void layoutKnob   (Knob&,   juce::Rectangle<int>);
    static void layoutChoice (Choice&, juce::Rectangle<int>);

    SIDVaeAudioProcessor& proc;

    juce::Label  status;
    WaveformView waveView;

    Choice waveA, waveB;
    Knob   morph, pw, vel, z1, z2, z3, freq, out;

    std::vector<float> scratch;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SIDVaeAudioProcessorEditor)
};
