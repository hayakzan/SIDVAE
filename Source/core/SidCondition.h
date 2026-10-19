#pragma once
// Maps synthesis parameters to the conditioning vector layout defined in sid_vae_meta.json.
//
//   pw_norm    : pulse width, 0..1
//   log_f0     : natural log of f0 in Hz; if the meta has "log_f0_mean" and
//                "log_f0_std" it is standardised as (ln f0 - mean) / std
//   vel01      : velocity, 0..1
//   wave_tri / wave_saw / wave_pulse : waveform flags (one-hot at the poles)

#include <cmath>
#include <string>
#include <vector>

namespace sidvae
{
    struct SidControls
    {
        float pw      = 0.5f;
        float f0Hz    = 220.0f;
        float vel     = 1.0f;
        float waveTri = 1.0f, waveSaw = 0.0f, wavePulse = 0.0f;
    };

    struct LogF0Norm
    {
        bool  enabled = false;
        float mean = 0.0f, std = 1.0f;
    };

    inline std::vector<std::string> defaultCondOrder()
    {
        return { "pw_norm", "log_f0", "vel01", "wave_tri", "wave_saw", "wave_pulse" };
    }

    // Writes order.size() values into `out`. Names it doesn't recognise are set to 0
    // and appended to `unknown` (if given) so the UI can warn about them.
    inline void buildCondition (const SidControls& c,
                                const std::vector<std::string>& order,
                                const LogF0Norm& norm,
                                float* out,
                                std::vector<std::string>* unknown = nullptr)
    {
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            const std::string& k = order[i];
            float v = 0.0f;

            if      (k == "pw_norm")    v = c.pw;
            else if (k == "vel01")      v = c.vel;
            else if (k == "wave_tri")   v = c.waveTri;
            else if (k == "wave_saw")   v = c.waveSaw;
            else if (k == "wave_pulse") v = c.wavePulse;
            else if (k == "log_f0")
            {
                v = std::log (std::fmax (1.0f, c.f0Hz));
                if (norm.enabled && std::abs (norm.std) > 1.0e-12f)
                    v = (v - norm.mean) / norm.std;
            }
            else if (unknown != nullptr)
                unknown->push_back (k);

            out[i] = v;
        }
    }
}
