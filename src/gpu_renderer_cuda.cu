#include "gpu_renderer_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

static cudaStream_t AsStream(void* stream) noexcept
{
    return reinterpret_cast<cudaStream_t>(stream);
}

static void RecordElapsed(cudaEvent_t start,
                          cudaEvent_t stop,
                          double& out_ms)
{
    float ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&ms, start, stop),
              "GpuRendererCuda cudaEventElapsedTime");
    out_ms += (double)ms;
}

static void DestroyEvent(cudaEvent_t& evt) noexcept
{
    if (evt != nullptr)
        cudaEventDestroy(evt);
    evt = nullptr;
}

template <typename T>
static void EnsureDeviceCapacity(T*& ptr,
                                 std::size_t& capacity_bytes,
                                 std::size_t required_bytes,
                                 const char* context)
{
    if (required_bytes <= capacity_bytes)
        return;

    if (ptr != nullptr)
        CheckCuda(cudaFree(ptr), context);

    if (required_bytes == 0)
    {
        ptr = nullptr;
        capacity_bytes = 0;
        return;
    }

    CheckCuda(cudaMalloc((void**)&ptr, required_bytes), context);
    capacity_bytes = required_bytes;
}

static void EnsureDeviceCapacityVoid(void*& ptr,
                                     std::size_t& capacity_bytes,
                                     std::size_t required_bytes,
                                     const char* context)
{
    if (required_bytes <= capacity_bytes)
        return;

    if (ptr != nullptr)
        CheckCuda(cudaFree(ptr), context);

    if (required_bytes == 0)
    {
        ptr = nullptr;
        capacity_bytes = 0;
        return;
    }

    CheckCuda(cudaMalloc(&ptr, required_bytes), context);
    capacity_bytes = required_bytes;
}

static void EnsurePinnedCommandCapacity(GpuRenderCommand*& ptr,
                                        std::size_t& capacity_count,
                                        std::size_t required_count,
                                        const char* context)
{
    if (required_count <= capacity_count)
        return;

    if (ptr != nullptr)
        CheckCuda(cudaFreeHost(ptr), context);

    if (required_count == 0)
    {
        ptr = nullptr;
        capacity_count = 0;
        return;
    }

    CheckCuda(cudaHostAlloc((void**)&ptr,
                            required_count * sizeof(GpuRenderCommand),
                            cudaHostAllocDefault),
              context);
    capacity_count = required_count;
}

__device__ static float Clamp01(float v)
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ static float FetchGlyphCoverage(const uint8_t* atlas_pixels,
                                           uint32_t atlas_width,
                                           const GpuRenderCommand& cmd,
                                           int gx,
                                           int gy)
{
    if (gx < 0 || gy < 0
        || gx >= (int)cmd.atlas_w
        || gy >= (int)cmd.atlas_h)
    {
        return 0.0f;
    }

    const uint32_t sx = cmd.atlas_x + (uint32_t)gx;
    const uint32_t sy = cmd.atlas_y + (uint32_t)gy;
    return (float)atlas_pixels[(std::size_t)sy * atlas_width + sx] / 255.0f;
}

__device__ static float SampleGlyphCoverageBilinear(const uint8_t* atlas_pixels,
                                                    uint32_t atlas_width,
                                                    const GpuRenderCommand& cmd,
                                                    float local_x,
                                                    float local_y)
{
    const float sample_x = local_x - 0.5f;
    const float sample_y = local_y - 0.5f;
    const int x0 = (int)floorf(sample_x);
    const int y0 = (int)floorf(sample_y);
    const float fx = sample_x - (float)x0;
    const float fy = sample_y - (float)y0;

    const float c00 = FetchGlyphCoverage(atlas_pixels, atlas_width, cmd, x0,     y0);
    const float c10 = FetchGlyphCoverage(atlas_pixels, atlas_width, cmd, x0 + 1, y0);
    const float c01 = FetchGlyphCoverage(atlas_pixels, atlas_width, cmd, x0,     y0 + 1);
    const float c11 = FetchGlyphCoverage(atlas_pixels, atlas_width, cmd, x0 + 1, y0 + 1);

    const float cx0 = c00 + (c10 - c00) * fx;
    const float cx1 = c01 + (c11 - c01) * fx;
    return cx0 + (cx1 - cx0) * fy;
}

__global__ void RgbToRgbaKernel(uchar4* dst,
                                const uint8_t* src,
                                uint32_t pixel_count)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count)
        return;

    const std::size_t src_idx = (std::size_t)idx * 3u;
    dst[idx] = make_uchar4(src[src_idx + 0],
                           src[src_idx + 1],
                           src[src_idx + 2],
                           255u);
}

__global__ void RgbaToRgbKernel(uint8_t* dst,
                                const uchar4* src,
                                uint32_t pixel_count)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count)
        return;

    const std::size_t dst_idx = (std::size_t)idx * 3u;
    const uchar4 pixel = src[idx];
    dst[dst_idx + 0] = pixel.x;
    dst[dst_idx + 1] = pixel.y;
    dst[dst_idx + 2] = pixel.z;
}

__global__ void RenderGlyphBatchKernel(uchar4* image,
                                       uint32_t image_width,
                                       uint32_t image_height,
                                       const uint8_t* atlas_pixels,
                                       uint32_t atlas_width,
                                       const GpuRenderCommand* commands,
                                       uint32_t command_count)
{
    const uint32_t cmd_index = blockIdx.x;
    if (cmd_index >= command_count)
        return;

    __shared__ GpuRenderCommand cmd;
    __shared__ int bbox_x0, bbox_y0, bbox_x1, bbox_y1;
    __shared__ float inv00, inv01, inv10, inv11;
    __shared__ float src_r, src_g, src_b, src_a_base;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cmd = commands[cmd_index];

        const float det = cmd.m00 * cmd.m11 - cmd.m01 * cmd.m10;
        inv00 =  cmd.m11 / det;
        inv01 = -cmd.m01 / det;
        inv10 = -cmd.m10 / det;
        inv11 =  cmd.m00 / det;

        float min_x = cmd.origin_x;
        float max_x = cmd.origin_x;
        float min_y = cmd.origin_y;
        float max_y = cmd.origin_y;

        const float w = (float)cmd.atlas_w;
        const float h = (float)cmd.atlas_h;
        const float corners[4][2] = {
            {0.0f, 0.0f},
            {w,    0.0f},
            {0.0f, h   },
            {w,    h   },
        };

        for (int i = 0; i < 4; ++i)
        {
            const float ix = cmd.origin_x + cmd.m00 * corners[i][0] + cmd.m01 * corners[i][1];
            const float iy = cmd.origin_y + cmd.m10 * corners[i][0] + cmd.m11 * corners[i][1];
            min_x = fminf(min_x, ix);
            max_x = fmaxf(max_x, ix);
            min_y = fminf(min_y, iy);
            max_y = fmaxf(max_y, iy);
        }

        bbox_x0 = max(0, (int)floorf(min_x));
        bbox_y0 = max(0, (int)floorf(min_y));
        bbox_x1 = min((int)image_width - 1, (int)ceilf(max_x));
        bbox_y1 = min((int)image_height - 1, (int)ceilf(max_y));

        src_r = (float)((cmd.rgba >> 24) & 0xFFu) / 255.0f;
        src_g = (float)((cmd.rgba >> 16) & 0xFFu) / 255.0f;
        src_b = (float)((cmd.rgba >> 8) & 0xFFu) / 255.0f;
        src_a_base = (float)(cmd.rgba & 0xFFu) / 255.0f;
    }
    __syncthreads();

    if (cmd.atlas_w == 0 || cmd.atlas_h == 0)
        return;

    for (int y = bbox_y0 + (int)threadIdx.y; y <= bbox_y1; y += blockDim.y)
    {
        for (int x = bbox_x0 + (int)threadIdx.x; x <= bbox_x1; x += blockDim.x)
        {
            const float dx = (float)x + 0.5f - cmd.origin_x;
            const float dy = (float)y + 0.5f - cmd.origin_y;

            const float local_x = inv00 * dx + inv01 * dy;
            const float local_y = inv10 * dx + inv11 * dy;

            if (local_x < 0.0f || local_y < 0.0f
                || local_x >= (float)cmd.atlas_w
                || local_y >= (float)cmd.atlas_h)
                continue;

            const float coverage =
                SampleGlyphCoverageBilinear(atlas_pixels, atlas_width, cmd, local_x, local_y);
            if (coverage <= 0.0f)
                continue;

            const float src_a = src_a_base * coverage;

            const std::size_t pixel_index = (std::size_t)y * image_width + (std::size_t)x;
            uchar4* p = image + pixel_index;

            const float dst_r = (float)p->x / 255.0f;
            const float dst_g = (float)p->y / 255.0f;
            const float dst_b = (float)p->z / 255.0f;
            const float dst_a = (float)p->w / 255.0f;

            const float out_r = src_a * src_r + (1.0f - src_a) * dst_r;
            const float out_g = src_a * src_g + (1.0f - src_a) * dst_g;
            const float out_b = src_a * src_b + (1.0f - src_a) * dst_b;
            const float out_a = src_a + (1.0f - src_a) * dst_a;

            p->x = (uint8_t)lrintf(Clamp01(out_r) * 255.0f);
            p->y = (uint8_t)lrintf(Clamp01(out_g) * 255.0f);
            p->z = (uint8_t)lrintf(Clamp01(out_b) * 255.0f);
            p->w = (uint8_t)lrintf(Clamp01(out_a) * 255.0f);
        }
    }
}

static cudaStream_t SelectStream(void* preferred_stream,
                                 void* owned_stream) noexcept
{
    return preferred_stream != nullptr ? AsStream(preferred_stream)
                                       : AsStream(owned_stream);
}

} // namespace

GpuRendererCuda::~GpuRendererCuda()
{
    Clear();
}

GpuRendererCuda::GpuRendererCuda(GpuRendererCuda&& other) noexcept
    : m_device_index(other.m_device_index)
    , m_device_image(other.m_device_image)
    , m_device_image_bytes(other.m_device_image_bytes)
    , m_device_rgb(other.m_device_rgb)
    , m_device_rgb_bytes(other.m_device_rgb_bytes)
    , m_device_commands(other.m_device_commands)
    , m_device_command_capacity(other.m_device_command_capacity)
    , m_host_command_staging(other.m_host_command_staging)
    , m_host_command_capacity(other.m_host_command_capacity)
    , m_stream(other.m_stream)
{
    other.m_device_index = -2;
    other.m_device_image = nullptr;
    other.m_device_image_bytes = 0;
    other.m_device_rgb = nullptr;
    other.m_device_rgb_bytes = 0;
    other.m_device_commands = nullptr;
    other.m_device_command_capacity = 0;
    other.m_host_command_staging = nullptr;
    other.m_host_command_capacity = 0;
    other.m_stream = nullptr;
}

GpuRendererCuda& GpuRendererCuda::operator=(GpuRendererCuda&& other) noexcept
{
    if (this == &other)
        return *this;

    Clear();
    m_device_index = other.m_device_index;
    m_device_image = other.m_device_image;
    m_device_image_bytes = other.m_device_image_bytes;
    m_device_rgb = other.m_device_rgb;
    m_device_rgb_bytes = other.m_device_rgb_bytes;
    m_device_commands = other.m_device_commands;
    m_device_command_capacity = other.m_device_command_capacity;
    m_host_command_staging = other.m_host_command_staging;
    m_host_command_capacity = other.m_host_command_capacity;
    m_stream = other.m_stream;

    other.m_device_index = -2;
    other.m_device_image = nullptr;
    other.m_device_image_bytes = 0;
    other.m_device_rgb = nullptr;
    other.m_device_rgb_bytes = 0;
    other.m_device_commands = nullptr;
    other.m_device_command_capacity = 0;
    other.m_host_command_staging = nullptr;
    other.m_host_command_capacity = 0;
    other.m_stream = nullptr;
    return *this;
}

void GpuRendererCuda::Initialize(int device_index)
{
    if (device_index >= 0)
        CheckCuda(cudaSetDevice(device_index), "GpuRendererCuda::Initialize cudaSetDevice");
    else
        CheckCuda(cudaGetDevice(&device_index), "GpuRendererCuda::Initialize cudaGetDevice");

    if (m_device_index == device_index && m_stream != nullptr)
        return;

    Clear();
    m_device_index = device_index;

    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "GpuRendererCuda::Initialize cudaStreamCreateWithFlags");
    m_stream = stream;
}

void GpuRendererCuda::Clear() noexcept
{
    if (m_device_index >= 0)
        cudaSetDevice(m_device_index);

    if (m_device_image)
        cudaFree(m_device_image);
    if (m_device_rgb)
        cudaFree(m_device_rgb);
    if (m_device_commands)
        cudaFree(m_device_commands);
    if (m_host_command_staging)
        cudaFreeHost(m_host_command_staging);
    if (m_stream)
        cudaStreamDestroy(AsStream(m_stream));

    m_device_image = nullptr;
    m_device_image_bytes = 0;
    m_device_rgb = nullptr;
    m_device_rgb_bytes = 0;
    m_device_commands = nullptr;
    m_device_command_capacity = 0;
    m_host_command_staging = nullptr;
    m_host_command_capacity = 0;
    m_stream = nullptr;
    m_device_index = -2;
}

GpuRenderStats GpuRendererCuda::RenderDeviceRgba(const DeviceRgbaImageView& image,
                                                 const GpuAtlasManager& atlas_manager,
                                                 const GpuCommandBuffer& buffer,
                                                 void* stream_ptr)
{
    if (!image.IsValid())
        throw std::runtime_error("GpuRendererCuda::RenderDeviceRgba: image view is invalid");

    Initialize(m_device_index);

    GpuRenderStats stats;
    stats.device = QueryCudaDevice(m_device_index);
    stats.tuning = SelectGpuKernelTuning(stats.device);

    const std::size_t command_count = buffer.commands.size();
    EnsureDeviceCapacity(m_device_commands,
                         m_device_command_capacity,
                         command_count * sizeof(GpuRenderCommand),
                         "GpuRendererCuda::RenderDeviceRgba cudaMalloc commands");
    EnsurePinnedCommandCapacity(m_host_command_staging,
                                m_host_command_capacity,
                                command_count,
                                "GpuRendererCuda::RenderDeviceRgba cudaHostAlloc commands");

    cudaEvent_t evt_cmd_start = nullptr;
    cudaEvent_t evt_cmd_stop = nullptr;
    cudaEvent_t evt_kernel_start = nullptr;
    cudaEvent_t evt_kernel_stop = nullptr;

    try
    {
        CheckCuda(cudaEventCreate(&evt_cmd_start),
                  "GpuRendererCuda::RenderDeviceRgba cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_cmd_stop),
                  "GpuRendererCuda::RenderDeviceRgba cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_kernel_start),
                  "GpuRendererCuda::RenderDeviceRgba cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_kernel_stop),
                  "GpuRendererCuda::RenderDeviceRgba cudaEventCreate");

        const cudaStream_t stream = SelectStream(stream_ptr, m_stream);

        if (command_count > 0)
        {
            std::memcpy(m_host_command_staging,
                        buffer.commands.data(),
                        command_count * sizeof(GpuRenderCommand));

            CheckCuda(cudaEventRecord(evt_cmd_start, stream),
                      "GpuRendererCuda::RenderDeviceRgba cudaEventRecord cmd_start");
            CheckCuda(cudaMemcpyAsync(m_device_commands,
                                      m_host_command_staging,
                                      command_count * sizeof(GpuRenderCommand),
                                      cudaMemcpyHostToDevice,
                                      stream),
                      "GpuRendererCuda::RenderDeviceRgba cudaMemcpyAsync commands");
            CheckCuda(cudaEventRecord(evt_cmd_stop, stream),
                      "GpuRendererCuda::RenderDeviceRgba cudaEventRecord cmd_stop");
        }

        const dim3 block((unsigned)stats.tuning.block_x, (unsigned)stats.tuning.block_y);

        CheckCuda(cudaEventRecord(evt_kernel_start, stream),
                  "GpuRendererCuda::RenderDeviceRgba cudaEventRecord kernel_start");

        for (const GpuRenderBatch& batch : buffer.batches)
        {
            const GpuAtlasHandle* atlas = atlas_manager.GetAtlas(batch.atlas_render_size);
            if (!atlas)
            {
                throw std::runtime_error(
                    "GpuRendererCuda::RenderDeviceRgba: missing uploaded atlas for render size "
                    + std::to_string(batch.atlas_render_size));
            }

            uint32_t remaining = batch.command_count;
            uint32_t offset = batch.command_offset;

            while (remaining > 0)
            {
                const uint32_t chunk =
                    remaining > stats.tuning.max_commands_per_launch
                        ? stats.tuning.max_commands_per_launch
                        : remaining;

                RenderGlyphBatchKernel<<<chunk, block, 0, stream>>>(
                    reinterpret_cast<uchar4*>(image.pixels),
                    image.width,
                    image.height,
                    atlas->device_pixels,
                    atlas->atlas_width,
                    m_device_commands + offset,
                    chunk);
                CheckCuda(cudaGetLastError(),
                          "GpuRendererCuda::RenderDeviceRgba kernel launch");

                remaining -= chunk;
                offset += chunk;
                ++stats.batches_rendered;
                stats.glyphs_rendered += chunk;
            }
        }

        CheckCuda(cudaEventRecord(evt_kernel_stop, stream),
                  "GpuRendererCuda::RenderDeviceRgba cudaEventRecord kernel_stop");
        CheckCuda(cudaStreamSynchronize(stream),
                  "GpuRendererCuda::RenderDeviceRgba cudaStreamSynchronize");

        if (command_count > 0)
            RecordElapsed(evt_cmd_start, evt_cmd_stop, stats.command_upload_ms);
        RecordElapsed(evt_kernel_start, evt_kernel_stop, stats.kernel_ms);
    }
    catch (...)
    {
        DestroyEvent(evt_cmd_start);
        DestroyEvent(evt_cmd_stop);
        DestroyEvent(evt_kernel_start);
        DestroyEvent(evt_kernel_stop);
        throw;
    }

    DestroyEvent(evt_cmd_start);
    DestroyEvent(evt_cmd_stop);
    DestroyEvent(evt_kernel_start);
    DestroyEvent(evt_kernel_stop);
    return stats;
}

GpuRenderStats GpuRendererCuda::Render(ImageRgba8& image,
                                       const GpuAtlasManager& atlas_manager,
                                       const GpuCommandBuffer& buffer)
{
    if (image.Empty())
        throw std::runtime_error("GpuRendererCuda::Render: image is empty");

    Initialize(m_device_index);
    EnsureDeviceCapacityVoid(m_device_image,
                             m_device_image_bytes,
                             image.pixels.size(),
                             "GpuRendererCuda::Render cudaMalloc image");

    cudaEvent_t evt_upload_start = nullptr;
    cudaEvent_t evt_upload_stop = nullptr;
    cudaEvent_t evt_download_start = nullptr;
    cudaEvent_t evt_download_stop = nullptr;

    try
    {
        CheckCuda(cudaEventCreate(&evt_upload_start), "GpuRendererCuda::Render cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_upload_stop), "GpuRendererCuda::Render cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_download_start), "GpuRendererCuda::Render cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_download_stop), "GpuRendererCuda::Render cudaEventCreate");

        const cudaStream_t stream = AsStream(m_stream);

        CheckCuda(cudaEventRecord(evt_upload_start, stream),
                  "GpuRendererCuda::Render cudaEventRecord upload_start");
        CheckCuda(cudaMemcpyAsync(m_device_image,
                                  image.pixels.data(),
                                  image.pixels.size(),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  "GpuRendererCuda::Render cudaMemcpyAsync image");
        CheckCuda(cudaEventRecord(evt_upload_stop, stream),
                  "GpuRendererCuda::Render cudaEventRecord upload_stop");

        GpuRenderStats stats = RenderDeviceRgba(
            DeviceRgbaImageView{image.width, image.height, m_device_image, image.pixels.size()},
            atlas_manager,
            buffer,
            m_stream);

        CheckCuda(cudaEventRecord(evt_download_start, stream),
                  "GpuRendererCuda::Render cudaEventRecord download_start");
        CheckCuda(cudaMemcpyAsync(image.pixels.data(),
                                  m_device_image,
                                  image.pixels.size(),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  "GpuRendererCuda::Render cudaMemcpyAsync image->host");
        CheckCuda(cudaEventRecord(evt_download_stop, stream),
                  "GpuRendererCuda::Render cudaEventRecord download_stop");
        CheckCuda(cudaStreamSynchronize(stream),
                  "GpuRendererCuda::Render cudaStreamSynchronize");

        RecordElapsed(evt_upload_start, evt_upload_stop, stats.image_upload_ms);
        RecordElapsed(evt_download_start, evt_download_stop, stats.image_download_ms);

        DestroyEvent(evt_upload_start);
        DestroyEvent(evt_upload_stop);
        DestroyEvent(evt_download_start);
        DestroyEvent(evt_download_stop);
        return stats;
    }
    catch (...)
    {
        DestroyEvent(evt_upload_start);
        DestroyEvent(evt_upload_stop);
        DestroyEvent(evt_download_start);
        DestroyEvent(evt_download_stop);
        throw;
    }
}

GpuRenderStats GpuRendererCuda::RenderRgb(uint8_t* output_rgb,
                                          GpuBufferMemoryType output_memory,
                                          const uint8_t* input_rgb,
                                          GpuBufferMemoryType input_memory,
                                          uint32_t width,
                                          uint32_t height,
                                          const GpuAtlasManager& atlas_manager,
                                          const GpuCommandBuffer& buffer,
                                          void* stream_ptr)
{
    if (output_rgb == nullptr || input_rgb == nullptr || width == 0 || height == 0)
    {
        throw std::runtime_error("GpuRendererCuda::RenderRgb: input and output buffers must be valid");
    }

    Initialize(m_device_index);

    const std::size_t rgb_bytes = (std::size_t)width * height * 3u;
    const std::size_t rgba_bytes = (std::size_t)width * height * 4u;

    EnsureDeviceCapacityVoid(m_device_image,
                             m_device_image_bytes,
                             rgba_bytes,
                             "GpuRendererCuda::RenderRgb cudaMalloc rgba");
    EnsureDeviceCapacity(m_device_rgb,
                         m_device_rgb_bytes,
                         rgb_bytes,
                         "GpuRendererCuda::RenderRgb cudaMalloc rgb");

    cudaEvent_t evt_upload_start = nullptr;
    cudaEvent_t evt_upload_stop = nullptr;
    cudaEvent_t evt_download_start = nullptr;
    cudaEvent_t evt_download_stop = nullptr;

    try
    {
        CheckCuda(cudaEventCreate(&evt_upload_start), "GpuRendererCuda::RenderRgb cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_upload_stop), "GpuRendererCuda::RenderRgb cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_download_start), "GpuRendererCuda::RenderRgb cudaEventCreate");
        CheckCuda(cudaEventCreate(&evt_download_stop), "GpuRendererCuda::RenderRgb cudaEventCreate");

        const cudaStream_t stream = SelectStream(stream_ptr, m_stream);
        const uint32_t pixel_count = width * height;
        const uint32_t convert_block = 256u;
        const uint32_t convert_grid =
            pixel_count == 0 ? 1u : (pixel_count + convert_block - 1u) / convert_block;

        const uint8_t* device_rgb_input = input_rgb;
        if (input_memory == GpuBufferMemoryType::Host)
        {
            CheckCuda(cudaEventRecord(evt_upload_start, stream),
                      "GpuRendererCuda::RenderRgb cudaEventRecord upload_start");
            CheckCuda(cudaMemcpyAsync(m_device_rgb,
                                      input_rgb,
                                      rgb_bytes,
                                      cudaMemcpyHostToDevice,
                                      stream),
                      "GpuRendererCuda::RenderRgb cudaMemcpyAsync rgb->device");
            CheckCuda(cudaEventRecord(evt_upload_stop, stream),
                      "GpuRendererCuda::RenderRgb cudaEventRecord upload_stop");
            device_rgb_input = m_device_rgb;
        }

        RgbToRgbaKernel<<<convert_grid, convert_block, 0, stream>>>(
            reinterpret_cast<uchar4*>(m_device_image),
            device_rgb_input,
            pixel_count);
        CheckCuda(cudaGetLastError(), "GpuRendererCuda::RenderRgb rgb->rgba kernel");

        GpuRenderStats stats = RenderDeviceRgba(
            DeviceRgbaImageView{width, height, m_device_image, rgba_bytes},
            atlas_manager,
            buffer,
            stream_ptr != nullptr ? stream_ptr : m_stream);

        if (output_memory == GpuBufferMemoryType::Device)
        {
            RgbaToRgbKernel<<<convert_grid, convert_block, 0, stream>>>(
                output_rgb,
                reinterpret_cast<const uchar4*>(m_device_image),
                pixel_count);
            CheckCuda(cudaGetLastError(), "GpuRendererCuda::RenderRgb rgba->rgb kernel");
            CheckCuda(cudaStreamSynchronize(stream),
                      "GpuRendererCuda::RenderRgb cudaStreamSynchronize device output");
        }
        else
        {
            RgbaToRgbKernel<<<convert_grid, convert_block, 0, stream>>>(
                m_device_rgb,
                reinterpret_cast<const uchar4*>(m_device_image),
                pixel_count);
            CheckCuda(cudaGetLastError(), "GpuRendererCuda::RenderRgb rgba->rgb kernel");

            CheckCuda(cudaEventRecord(evt_download_start, stream),
                      "GpuRendererCuda::RenderRgb cudaEventRecord download_start");
            CheckCuda(cudaMemcpyAsync(output_rgb,
                                      m_device_rgb,
                                      rgb_bytes,
                                      cudaMemcpyDeviceToHost,
                                      stream),
                      "GpuRendererCuda::RenderRgb cudaMemcpyAsync rgb->host");
            CheckCuda(cudaEventRecord(evt_download_stop, stream),
                      "GpuRendererCuda::RenderRgb cudaEventRecord download_stop");
            CheckCuda(cudaStreamSynchronize(stream),
                      "GpuRendererCuda::RenderRgb cudaStreamSynchronize host output");
        }

        if (input_memory == GpuBufferMemoryType::Host)
            RecordElapsed(evt_upload_start, evt_upload_stop, stats.image_upload_ms);
        if (output_memory == GpuBufferMemoryType::Host)
            RecordElapsed(evt_download_start, evt_download_stop, stats.image_download_ms);

        DestroyEvent(evt_upload_start);
        DestroyEvent(evt_upload_stop);
        DestroyEvent(evt_download_start);
        DestroyEvent(evt_download_stop);
        return stats;
    }
    catch (...)
    {
        DestroyEvent(evt_upload_start);
        DestroyEvent(evt_upload_stop);
        DestroyEvent(evt_download_start);
        DestroyEvent(evt_download_stop);
        throw;
    }
}

GpuRenderStats RenderCommandBufferCuda(ImageRgba8& image,
                                       const GpuAtlasManager& atlas_manager,
                                       const GpuCommandBuffer& buffer,
                                       int device_index)
{
    GpuRendererCuda renderer;
    renderer.Initialize(device_index);
    return renderer.Render(image, atlas_manager, buffer);
}

GpuRenderStats RenderCommandBufferCudaRgb(uint8_t* output_rgb,
                                          GpuBufferMemoryType output_memory,
                                          const uint8_t* input_rgb,
                                          GpuBufferMemoryType input_memory,
                                          uint32_t width,
                                          uint32_t height,
                                          const GpuAtlasManager& atlas_manager,
                                          const GpuCommandBuffer& buffer,
                                          int device_index,
                                          void* stream)
{
    GpuRendererCuda renderer;
    renderer.Initialize(device_index);
    return renderer.RenderRgb(output_rgb,
                              output_memory,
                              input_rgb,
                              input_memory,
                              width,
                              height,
                              atlas_manager,
                              buffer,
                              stream);
}

GpuRenderStats RenderPlanCuda(ImageRgba8& image,
                              const FontDatabase& db,
                              const RenderPlan& plan,
                              int device_index)
{
    GpuAtlasManager atlas_manager;
    if (device_index >= 0)
        CheckCuda(cudaSetDevice(device_index), "RenderPlanCuda cudaSetDevice");
    atlas_manager.Upload(db);

    const GpuCommandBuffer buffer = BuildGpuCommandBuffer(db, plan);
    return RenderCommandBufferCuda(image, atlas_manager, buffer, device_index);
}

GpuRenderStats RenderPlanCudaRgb(uint8_t* output_rgb,
                                 GpuBufferMemoryType output_memory,
                                 const uint8_t* input_rgb,
                                 GpuBufferMemoryType input_memory,
                                 uint32_t width,
                                 uint32_t height,
                                 const FontDatabase& db,
                                 const RenderPlan& plan,
                                 int device_index,
                                 void* stream)
{
    GpuAtlasManager atlas_manager;
    if (device_index >= 0)
        CheckCuda(cudaSetDevice(device_index), "RenderPlanCudaRgb cudaSetDevice");
    atlas_manager.Upload(db);

    const GpuCommandBuffer buffer = BuildGpuCommandBuffer(db, plan);
    return RenderCommandBufferCudaRgb(output_rgb,
                                      output_memory,
                                      input_rgb,
                                      input_memory,
                                      width,
                                      height,
                                      atlas_manager,
                                      buffer,
                                      device_index,
                                      stream);
}

} // namespace fac
