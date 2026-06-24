#pragma once

#include "gpu_atlas_manager.h"
#include "gpu_command_buffer.h"
#include "gpu_device.h"
#include "text_renderer_cpu.h"

#include <cstddef>

namespace fac {

enum class GpuBufferMemoryType
{
    Host,
    Device,
};

struct DeviceRgbaImageView
{
    uint32_t width = 0;
    uint32_t height = 0;
    void* pixels = nullptr;
    std::size_t pixel_bytes = 0;

    bool IsValid() const noexcept
    {
        return width > 0
            && height > 0
            && pixels != nullptr
            && pixel_bytes >= (std::size_t)width * height * 4u;
    }
};

struct GpuRenderStats
{
    GpuDeviceInfo device;
    GpuKernelTuning tuning;
    std::size_t batches_rendered = 0;
    std::size_t glyphs_rendered = 0;
    double image_upload_ms = 0.0;
    double command_upload_ms = 0.0;
    double kernel_ms = 0.0;
    double image_download_ms = 0.0;
};

class GpuRendererCuda
{
public:
    GpuRendererCuda() = default;
    ~GpuRendererCuda();

    GpuRendererCuda(const GpuRendererCuda&) = delete;
    GpuRendererCuda& operator=(const GpuRendererCuda&) = delete;
    GpuRendererCuda(GpuRendererCuda&& other) noexcept;
    GpuRendererCuda& operator=(GpuRendererCuda&& other) noexcept;

    void Initialize(int device_index = -1);
    void Clear() noexcept;

    GpuRenderStats Render(ImageRgba8& image,
                          const GpuAtlasManager& atlas_manager,
                          const GpuCommandBuffer& buffer);

    GpuRenderStats RenderDeviceRgba(const DeviceRgbaImageView& image,
                                    const GpuAtlasManager& atlas_manager,
                                    const GpuCommandBuffer& buffer,
                                    void* stream = nullptr);

    GpuRenderStats RenderRgb(uint8_t* output_rgb,
                             GpuBufferMemoryType output_memory,
                             const uint8_t* input_rgb,
                             GpuBufferMemoryType input_memory,
                             uint32_t width,
                             uint32_t height,
                             const GpuAtlasManager& atlas_manager,
                             const GpuCommandBuffer& buffer,
                             void* stream = nullptr);

private:
    int m_device_index = -2;
    void* m_device_image = nullptr;
    std::size_t m_device_image_bytes = 0;
    uint8_t* m_device_rgb = nullptr;
    std::size_t m_device_rgb_bytes = 0;
    GpuRenderCommand* m_device_commands = nullptr;
    std::size_t m_device_command_capacity = 0;
    GpuRenderCommand* m_host_command_staging = nullptr;
    std::size_t m_host_command_capacity = 0;
    void* m_stream = nullptr;
};

GpuRenderStats RenderCommandBufferCuda(ImageRgba8& image,
                                       const GpuAtlasManager& atlas_manager,
                                       const GpuCommandBuffer& buffer,
                                       int device_index = -1);

GpuRenderStats RenderCommandBufferCudaRgb(uint8_t* output_rgb,
                                          GpuBufferMemoryType output_memory,
                                          const uint8_t* input_rgb,
                                          GpuBufferMemoryType input_memory,
                                          uint32_t width,
                                          uint32_t height,
                                          const GpuAtlasManager& atlas_manager,
                                          const GpuCommandBuffer& buffer,
                                          int device_index = -1,
                                          void* stream = nullptr);

GpuRenderStats RenderPlanCuda(ImageRgba8& image,
                              const FontDatabase& db,
                              const RenderPlan& plan,
                              int device_index = -1);

GpuRenderStats RenderPlanCudaRgb(uint8_t* output_rgb,
                                 GpuBufferMemoryType output_memory,
                                 const uint8_t* input_rgb,
                                 GpuBufferMemoryType input_memory,
                                 uint32_t width,
                                 uint32_t height,
                                 const FontDatabase& db,
                                 const RenderPlan& plan,
                                 int device_index = -1,
                                 void* stream = nullptr);

} // namespace fac
