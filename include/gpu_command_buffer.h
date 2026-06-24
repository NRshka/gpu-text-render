#pragma once

#include "font_database.h"
#include "render_plan.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fac {

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

GpuCommandBuffer BuildGpuCommandBuffer(const FontDatabase& db,
                                       const RenderPlan& plan);

} // namespace fac
