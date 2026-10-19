#pragma once
// Lock-free single-producer / single-consumer triple buffer for wavetables.
// The decode thread writes into its own buffer and publishes; the audio thread
// picks up the newest complete table at the start of a block. Neither side ever
// waits on the other, and nothing allocates after prepare().

#include <array>
#include <atomic>
#include <vector>

namespace sidvae
{
    class TableTripleBuffer
    {
    public:
        // Call before either thread starts using the buffer.
        void prepare (int tableLen)
        {
            for (auto& b : bufs)
                b.assign ((std::size_t) tableLen, 0.0f);
            front = 0;
            middle.store (1);
            back = 2;
        }

        int size() const noexcept { return (int) bufs[0].size(); }

        // ---- producer (decode thread) ----
        float* writeBuffer() noexcept { return bufs[(std::size_t) back].data(); }

        void publish() noexcept
        {
            back = middle.exchange (back | kDirty, std::memory_order_acq_rel) & kIndexMask;
        }

        // ---- consumer (audio thread) ----
        // Returns true if a newer table was swapped in since the last call.
        bool acquire() noexcept
        {
            if ((middle.load (std::memory_order_acquire) & kDirty) == 0)
                return false;
            front = middle.exchange (front, std::memory_order_acq_rel) & kIndexMask;
            return true;
        }

        const float* readBuffer() const noexcept { return bufs[(std::size_t) front].data(); }

    private:
        static constexpr int kDirty     = 4;
        static constexpr int kIndexMask = 3;

        std::array<std::vector<float>, 3> bufs;
        int front = 0;               // audio thread only
        std::atomic<int> middle { 1 };
        int back = 2;                // decode thread only
    };
}
