#include "gpu_device.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace fac {

namespace {

static void CheckCuda(cudaError_t err, const char* context)
{
    if (err == cudaSuccess)
        return;

    std::ostringstream ss;
    ss << context << ": " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
}

} // namespace

bool HasCudaDevice() noexcept
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

GpuDeviceInfo QueryCudaDevice(int device_index)
{
    if (device_index < 0)
        CheckCuda(cudaGetDevice(&device_index), "QueryCudaDevice cudaGetDevice");

    cudaDeviceProp prop{};
    CheckCuda(cudaGetDeviceProperties(&prop, device_index),
              "QueryCudaDevice cudaGetDeviceProperties");

    GpuDeviceInfo out;
    out.device_index = device_index;
    out.name = prop.name;
    out.compute_major = prop.major;
    out.compute_minor = prop.minor;
    out.sm_count = prop.multiProcessorCount;
    out.warp_size = prop.warpSize;
    out.global_mem_bytes = prop.totalGlobalMem;
    out.shared_mem_per_block = prop.sharedMemPerBlock;
    out.concurrent_kernels = prop.concurrentKernels != 0;
    out.async_engine = prop.asyncEngineCount > 0;
    return out;
}

GpuKernelTuning SelectGpuKernelTuning(const GpuDeviceInfo& info) noexcept
{
    GpuKernelTuning out;

    const bool is_hopper_or_newer = info.compute_major >= 9;
    const bool is_ampere_or_newer = info.compute_major >= 8;

    if (is_hopper_or_newer)
    {
        out.block_x = 32;
        out.block_y = 8;
        out.max_commands_per_launch = 262144u;
        out.prefer_large_batches = true;
        return out;
    }

    if (is_ampere_or_newer)
    {
        out.block_x = 16;
        out.block_y = 16;
        out.max_commands_per_launch = 131072u;
        out.prefer_large_batches = info.sm_count >= 48;
        return out;
    }

    out.block_x = 16;
    out.block_y = 8;
    out.max_commands_per_launch = 65535u;
    out.prefer_large_batches = false;
    return out;
}

} // namespace fac
