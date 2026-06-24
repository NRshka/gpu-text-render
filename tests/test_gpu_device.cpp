#include "gpu_device.h"

#include <cstdlib>
#include <iostream>

using namespace fac;

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT(cond)                                                  \
    do {                                                              \
        if (cond) {                                                   \
            ++g_passed;                                               \
        } else {                                                      \
            ++g_failed;                                               \
            std::cerr << "  FAIL  " << #cond                         \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        }                                                             \
    } while (0)

#define SECTION(name) std::cout << "\n[" << name << "]\n"

int main()
{
    SECTION("Tuning for Ampere-class GPU");
    {
        GpuDeviceInfo info;
        info.compute_major = 8;
        info.compute_minor = 6;
        info.sm_count = 48;

        const GpuKernelTuning tuning = SelectGpuKernelTuning(info);
        EXPECT(tuning.block_x == 16);
        EXPECT(tuning.block_y == 16);
        EXPECT(tuning.max_commands_per_launch == 131072u);
        EXPECT(tuning.prefer_large_batches);
    }

    SECTION("Tuning for Hopper-class GPU");
    {
        GpuDeviceInfo info;
        info.compute_major = 9;
        info.compute_minor = 0;
        info.sm_count = 114;

        const GpuKernelTuning tuning = SelectGpuKernelTuning(info);
        EXPECT(tuning.block_x == 32);
        EXPECT(tuning.block_y == 8);
        EXPECT(tuning.max_commands_per_launch == 262144u);
        EXPECT(tuning.prefer_large_batches);
    }

    SECTION("Query real device when available");
    {
        if (!HasCudaDevice())
        {
            std::cout << "  No CUDA device available, skipping\n";
            EXPECT(true);
        }
        else
        {
            const GpuDeviceInfo info = QueryCudaDevice();
            EXPECT(!info.name.empty());
            EXPECT(info.compute_major >= 1);
            EXPECT(info.warp_size == 32);
        }
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
