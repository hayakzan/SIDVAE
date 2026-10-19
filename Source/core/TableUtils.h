#pragma once
// Small, allocation-free helpers for single-cycle wavetables. No JUCE dependency.

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sidvae
{
    // Linear resampling of one *periodic* cycle: wraps at the end instead of clamping,
    // so the last sample interpolates back towards the first.
    inline void resampleCyclic (const float* src, std::size_t srcLen, float* dst, std::size_t dstLen) noexcept
    {
        if (srcLen == 0 || dstLen == 0)
            return;

        const double step = (double) srcLen / (double) dstLen;
        for (std::size_t i = 0; i < dstLen; ++i)
        {
            const double pos  = (double) i * step;
            const auto   i0   = (std::size_t) pos % srcLen;
            const auto   i1   = (i0 + 1) % srcLen;
            const float  frac = (float) (pos - std::floor (pos));
            dst[i] = src[i0] + frac * (src[i1] - src[i0]);
        }
    }

    // A wavetable oscillator turns any DC in the table into a constant offset, remove it.
    inline void removeDC (float* t, std::size_t n) noexcept
    {
        if (n == 0) return;
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += t[i];
        const float mean = (float) (sum / (double) n);
        for (std::size_t i = 0; i < n; ++i) t[i] -= mean;
    }

    inline float peakAbs (const float* t, std::size_t n) noexcept
    {
        float p = 0.0f;
        for (std::size_t i = 0; i < n; ++i) p = std::max (p, std::abs (t[i]));
        return p;
    }

    // Only scales *down*: keeps whatever level differences the model produces,
    // but stops a bad decode from blasting the output.
    inline void limitPeak (float* t, std::size_t n, float maxPeak = 1.0f) noexcept
    {
        const float p = peakAbs (t, n);
        if (p > maxPeak && p > 0.0f)
        {
            const float g = maxPeak / p;
            for (std::size_t i = 0; i < n; ++i) t[i] *= g;
        }
    }

    inline bool allFinite (const float* t, std::size_t n) noexcept
    {
        for (std::size_t i = 0; i < n; ++i)
            if (! std::isfinite (t[i])) return false;
        return true;
    }
}
