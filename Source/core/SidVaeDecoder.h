#pragma once
// Thin wrapper around the conditional VAE decoder exported to ONNX.
//
//   table = decoder(z, cond)      z: latent_dim floats, cond: cond_dim floats
//
// It inspects the model's inputs when loading and supports both export styles:
//   * two inputs  -> one of size latent_dim (z), one of size cond_dim (cond),
//                    matched by size, or by name ("cond"/"label") if sizes are ambiguous
//   * one input   -> size latent_dim + cond_dim, fed as [z..., cond...]
// The first output is taken as one waveform cycle and resampled (cyclically) to
// tableLen if its length differs.
//
// Not thread-safe: load() and decode() must be called from the same thread
// (the plugin's table-builder thread). NEVER call from the audio thread.
// No JUCE dependency, so it can be unit-tested on its own.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#ifndef SIDVAE_WITH_ONNX
 #define SIDVAE_WITH_ONNX 0
#endif

namespace sidvae
{
    class SidVaeDecoder
    {
    public:
        struct Config
        {
            int latentDim = 3;
            int condDim   = 6;
            int tableLen  = 2048;
        };

        SidVaeDecoder();
        ~SidVaeDecoder();

        static constexpr bool compiledWithOnnx() noexcept { return SIDVAE_WITH_ONNX != 0; }

        // Loads the model and runs one test decode. On failure returns false and fills `error`.
        // The .onnx.data file (external weights) must sit next to the .onnx file.
        bool load (const std::filesystem::path& modelFile, const Config& cfg, std::string& error);

        bool isReady() const noexcept { return ready; }
        const Config& config() const noexcept { return cfg; }

        // Human-readable summary of the model I/O, e.g. "z[1,3] + cond[1,6] -> 2048".
        const std::string& ioSummary() const noexcept { return summary; }

        // z must hold latentDim floats, cond condDim floats. `out` is resized to tableLen.
        bool decode (const float* z, const float* cond, std::vector<float>& out, std::string* error = nullptr);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
        Config cfg;
        bool ready = false;
        std::string summary;
        std::size_t lastOutputLength = 0;
    };
}
