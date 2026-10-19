#pragma once
// Band-limited reference waveforms used ONLY when the VAE is unavailable.
// In that mode the plugin crossfades these in the shape domain, and the UI says so.

#include <cmath>
#include <vector>
#include "TableUtils.h"

namespace sidvae
{
    enum class Wave : int { Triangle = 0, Saw = 1, Pulse = 2 };
    constexpr int kNumWaves = 3;

    namespace detail
    {
        constexpr double kPi    = 3.14159265358979323846;
        constexpr double kTwoPi = 2.0 * kPi;
    }

    inline std::vector<float> makeTriangle (int n, int harmonics = 64)
    {
        std::vector<float> t ((std::size_t) n, 0.0f);
        for (int i = 0; i < n; ++i)
        {
            const double p = (double) i / (double) n;
            double s = 0.0;
            for (int k = 0; k < harmonics / 2; ++k)
            {
                const int h = 2 * k + 1;
                s += ((k % 2 == 0) ? 1.0 : -1.0) * std::sin (detail::kTwoPi * h * p) / (double) (h * h);
            }
            t[(std::size_t) i] = (float) (8.0 / (detail::kPi * detail::kPi) * s);
        }
        return t;
    }

    inline std::vector<float> makeSaw (int n, int harmonics = 64)
    {
        std::vector<float> t ((std::size_t) n, 0.0f);
        for (int i = 0; i < n; ++i)
        {
            const double p = (double) i / (double) n;
            double s = 0.0;
            for (int h = 1; h <= harmonics; ++h)
                s += std::sin (detail::kTwoPi * h * p) / (double) h;
            t[(std::size_t) i] = (float) (-2.0 / detail::kPi * s); // rising ramp
        }
        const float pk = peakAbs (t.data(), t.size());   // Gibbs overshoot -> normalise to 1
        if (pk > 0.0f)
            for (auto& v : t) v /= pk;
        return t;
    }

    // Pulse with duty cycle `pw` (0..1), DC removed, peak-normalised to 1.
    inline std::vector<float> makePulse (int n, float pw, int harmonics = 64)
    {
        const double d = std::fmin (0.98, std::fmax (0.02, (double) pw));
        std::vector<float> t ((std::size_t) n, 0.0f);
        for (int i = 0; i < n; ++i)
        {
            const double p = (double) i / (double) n;
            double s = 0.0;
            for (int h = 1; h <= harmonics; ++h)
                s += (2.0 / (detail::kPi * h)) * std::sin (detail::kPi * h * d)
                     * std::cos (detail::kTwoPi * h * (p - 0.5 * d));
            t[(std::size_t) i] = (float) s;
        }
        removeDC (t.data(), t.size());
        const float pk = peakAbs (t.data(), t.size());
        if (pk > 0.0f)
            for (auto& v : t) v /= pk;
        return t;
    }
}
