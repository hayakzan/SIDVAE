# SIDVAE

A C++/JUCE synthesizer plugin (AU / VST3 / Standalone) that generates single-cycle waveforms via a conditional Variational Autoencoder (cVAE) trained on MOS 6581/8580 SID recordings (rendered from Plogue chipsynth C64). Waveform interpolation is performed directly in the decoder's conditioning and latent space prior to inference rather than crossfading fixed audio tables.

## Architecture & Signal Path

The ONNX decoder maps a 3D latent vector and a 6D conditioning vector to a 2048-sample single-cycle wavetable: `table = decoder(z, cond)`.

| Tensor | Shape | Parameter Mapping |
|--------|-------|-------------------|
| `z` | `[1, 3]` | **Latent 1–3** (`0.0` = standard-normal prior mean) |
| `cond` | `[1, 6]` | `[pw_norm, log_f0, vel01, wave_tri, wave_saw, wave_pulse]` defined in `Models/sid_vae_meta.json` |

* **Latent & Conditioning Morph:** At `Morph = 0.0`, waveform conditioning is one-hot for **Wave A**; at `Morph = 1.0`, one-hot for **Wave B**. Intermediate positions linearly interpolate conditioning weights and re-evaluate the decoder to synthesize continuous hybrid timbres.
* **Lock-Free Audio Handoff:** Parameter changes trigger asynchronous inference on a background worker thread via the ONNX Runtime C++ API (`SidVaeDecoder`). Decoded wavetables are passed to the real-time audio thread via a lock-free `TripleBuffer`, where `WavetableOsc` applies an 8 ms crossfade between consecutive tables to prevent phase discontinuities.
* **Diagnostic Fallback:** If the ONNX runtime or model weights are unavailable (`-DSIDVAE_WITH_ONNX=OFF`), the engine falls back to band-limited reference tables (`FallbackTables.h`) and flags `FALLBACK` in the UI status bar.

## Repository Structure

```text
CMakeLists.txt
Source/
  PluginProcessor.*      APVTS parameter management, background inference worker, audio callback
  PluginEditor.*         Waveform visualizer, status bar, parameter controls
  core/                  JUCE-independent DSP and inference primitives
    SidVaeDecoder.*      ONNX Runtime C++ wrapper with dynamic I/O shape resolution
    SidCondition.h       Conditioning vector builder mapped to sid_vae_meta.json
    WavetableOsc.h       Wavetable oscillator with 8 ms table-swap crossfading
    TripleBuffer.h       Lock-free producer/consumer handoff (inference thread -> audio thread)
    FallbackTables.h     Band-limited reference waveforms (triangle, saw, pulse)
    TableUtils.h         Resampling and normalization utilities
Models/
  sid_vae_decoder.onnx       Exported PyTorch cVAE decoder graph
  sid_vae_decoder.onnx.data  External model weights
  sid_vae_meta.json          Conditioning schema and normalization constants
tools/
  inspect_model.py       CLI utility to inspect ONNX tensor shapes and plot spectral morph curves
```

## Build Instructions (macOS)

Requirements: Xcode Command Line Tools, CMake >= 3.22.

1. **ONNX Runtime Dependency:**  
   Download `onnxruntime-osx-arm64-1.31.0.tgz` (or `universal2` / `x86_64` for Intel Macs) from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) and extract into `third_party/onnxruntime`:

   ```bash
   mkdir -p third_party
   tar xzf onnxruntime-osx-arm64-1.31.0.tgz -C third_party
   mv third_party/onnxruntime-osx-arm64-1.31.0 third_party/onnxruntime
   ```

2. **Configure and Compile:**  
   CMake fetches JUCE 9.0.4 automatically via `FetchContent` (or uses a local `./JUCE` directory if present), bundles `Models/` and `libonnxruntime` into the `.vst3` / `.component` / `.app` packages, ad-hoc signs the binaries, and installs to `~/Library/Audio/Plug-Ins/`:

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

   To build without ONNX Runtime (reference wavetables only):
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIDVAE_WITH_ONNX=OFF
   ```

## Model Inspection Utility

`tools/inspect_model.py` verifies the exported ONNX graph outside the plugin and measures non-linearity and spectral distance across morph steps:

```bash
pip install onnxruntime numpy matplotlib
python tools/inspect_model.py Models/sid_vae_decoder.onnx --a tri --b saw --plot morph.png
```

## Current Scope

- Pitch is currently driven via the continuous `Freq` parameter (`MIDI` input is enabled at the bus level for host compatibility).
- Monophonic signal path; exposes the primary 3 latent dimensions (`z1..z3`).
