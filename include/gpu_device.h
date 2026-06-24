#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace fac {

struct GpuDeviceInfo
{
    int device_index = -1;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int sm_count = 0;
    int warp_size = 32;
    std::size_t global_mem_bytes = 0;
    std::size_t shared_mem_per_block = 0;
    bool concurrent_kernels = false;
    bool async_engine = false;
};

struct GpuKernelTuning
{
    int block_x = 16;
    int block_y = 16;
    uint32_t max_commands_per_launch = 65535u;
    bool prefer_large_batches = false;
};

bool HasCudaDevice() noexcept;
GpuDeviceInfo QueryCudaDevice(int device_index = -1);
GpuKernelTuning SelectGpuKernelTuning(const GpuDeviceInfo& info) noexcept;

} // namespace fac
