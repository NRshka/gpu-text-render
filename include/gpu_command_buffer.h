#pragma once

#include "font_database.h"
#include "render_plan.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fac {

constexpr int32_t kGpuCommandBufferAbiVersion = 2;

// Packed per-glyph command for GPU execution.
// The 2x2 matrix already includes glyph scale, so the GPU renderer can apply:
// image = origin + M * glyph_local
struct GpuRenderCommand
{
    float origin_x = 0.0f;
    float origin_y = 0.0f;

    float m00 = 1.0f;
    float m01 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;

    uint32_t rgba = 0xFFFFFFFFu;

    uint16_t atlas_x = 0;
    uint16_t atlas_y = 0;
    uint16_t atlas_w = 0;
    uint16_t atlas_h = 0;
};

// One batch references a single uploaded atlas texture and a contiguous
// command range within the flattened command buffer.
struct GpuRenderBatch
{
    uint32_t atlas_render_size = 0;
    uint32_t atlas_width       = 0;
    uint32_t atlas_height      = 0;
    uint32_t command_offset    = 0;
    uint32_t command_count     = 0;
};

struct GpuCommandBuffer
{
    std::vector<GpuRenderBatch> batches;
    std::vector<GpuRenderCommand> commands;
    std::size_t total_regions = 0;
    std::size_t total_glyphs  = 0;
};

// Versioned command format for batched GPU rendering. image_index addresses
// the target image slice inside a contiguous [B, H, W, 4] staging buffer.
struct GpuRenderCommandV2
{
    float origin_x = 0.0f;
    float origin_y = 0.0f;

    float m00 = 1.0f;
    float m01 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;

    uint32_t rgba = 0xFFFFFFFFu;
    uint32_t image_index = 0;

    uint16_t atlas_x = 0;
    uint16_t atlas_y = 0;
    uint16_t atlas_w = 0;
    uint16_t atlas_h = 0;
};

struct GpuRenderBatchV2
{
    uint32_t atlas_render_size = 0;
    uint32_t atlas_width       = 0;
    uint32_t atlas_height      = 0;
    uint32_t command_offset    = 0;
    uint32_t command_count     = 0;
};

struct GpuCommandBufferV2
{
    std::vector<GpuRenderBatchV2> batches;
    std::vector<GpuRenderCommandV2> commands;
    std::size_t total_regions = 0;
    std::size_t total_glyphs  = 0;
    uint32_t total_images     = 0;
};

GpuCommandBuffer BuildGpuCommandBuffer(const FontDatabase& db,
                                       const RenderPlan& plan);

GpuCommandBufferV2 BuildGpuCommandBufferV2(const FontDatabase& db,
                                           const RenderPlan& plan,
                                           uint32_t image_index = 0);

GpuCommandBufferV2 BuildCombinedGpuCommandBufferV2(
    const FontDatabase& db,
    const std::vector<RenderPlan>& plans);

GpuCommandBufferV2 CombineGpuCommandBuffersV2(
    const std::vector<GpuCommandBufferV2>& buffers);

std::vector<uint8_t> SerializeGpuRenderCommandsV2(
    const std::vector<GpuRenderCommandV2>& commands);

std::vector<uint8_t> SerializeGpuRenderBatchesV2(
    const std::vector<GpuRenderBatchV2>& batches);

std::vector<GpuRenderCommandV2> DeserializeGpuRenderCommandsV2(
    const uint8_t* bytes,
    std::size_t byte_size);

std::vector<GpuRenderBatchV2> DeserializeGpuRenderBatchesV2(
    const uint8_t* bytes,
    std::size_t byte_size);

GpuCommandBufferV2 DeserializeGpuCommandBufferV2(const uint8_t* command_bytes,
                                                 std::size_t command_byte_size,
                                                 const uint8_t* batch_bytes,
                                                 std::size_t batch_byte_size,
                                                 uint32_t total_images);

void ValidateGpuCommandBufferV2(const GpuCommandBufferV2& buffer,
                                uint32_t total_images);

} // namespace fac
