#pragma once
// Single-cycle wavetable oscillator with linear crossfading across table swaps to prevent phase discontinuities.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sidvae
{
    class WavetableOsc
    {
    public:
        void prepare (double sampleRate, int tableLen, double fadeMs = 8.0)
        {
            sr  = sampleRate > 0.0 ? sampleRate : 44100.0;
            len = std::max (16, tableLen);
            cur .assign ((std::size_t) len, 0.0f);
            prev.assign ((std::size_t) len, 0.0f);
            fadeLen  = std::max (1, (int) std::lround (sr * fadeMs * 0.001));
            fadePos  = fadeLen;  // not fading
            phase    = 0.0;
            gainSmoothed = 0.0f;
            gainCoeff = (float) (1.0 - std::exp (-1.0 / (0.010 * sr))); // ~10 ms
        }

        // Audio thread. `src` must hold tableLen samples.
        void setTargetTable (const float* src) noexcept
        {
            // Bake whatever is audible right now into `prev` so a table that lands
            // mid-fade continues from the current mix instead of jumping.
            const float f = fadeAmount();
            if (f < 1.0f)
                for (int i = 0; i < len; ++i)
                    prev[(std::size_t) i] += f * (cur[(std::size_t) i] - prev[(std::size_t) i]);
            else
                std::swap (prev, cur);   // vector swap: pointer exchange, no allocation

            std::memcpy (cur.data(), src, sizeof (float) * (std::size_t) len);
            fadePos = 0;
        }

        void render (float* dst, int numSamples, float freqHz, float gainLinear) noexcept
        {
            const double f   = std::clamp ((double) freqHz, 0.0, 0.49 * sr);
            const double inc = (double) len * f / sr;

            for (int n = 0; n < numSamples; ++n)
            {
                const int   i0   = (int) phase;
                const int   i1   = (i0 + 1 == len) ? 0 : i0 + 1;
                const float frac = (float) (phase - (double) i0);

                const float c = cur[(std::size_t) i0] + frac * (cur[(std::size_t) i1] - cur[(std::size_t) i0]);
                float s = c;
                if (fadePos < fadeLen)
                {
                    const float p = prev[(std::size_t) i0] + frac * (prev[(std::size_t) i1] - prev[(std::size_t) i0]);
                    const float a = (float) fadePos / (float) fadeLen;
                    s = p + a * (c - p);
                    ++fadePos;
                }

                gainSmoothed += gainCoeff * (gainLinear - gainSmoothed);
                dst[n] = s * gainSmoothed;

                phase += inc;
                if (phase >= (double) len) phase -= (double) len;
            }
        }

        // Copy of the table currently being faded towards (for tests / debugging).
        const std::vector<float>& currentTable() const noexcept { return cur; }

    private:
        float fadeAmount() const noexcept { return fadePos >= fadeLen ? 1.0f : (float) fadePos / (float) fadeLen; }

        std::vector<float> cur, prev;
        double sr = 44100.0;
        int    len = 2048;
        int    fadeLen = 1, fadePos = 1;
        double phase = 0.0;
        float  gainSmoothed = 0.0f, gainCoeff = 0.01f;
    };
}
