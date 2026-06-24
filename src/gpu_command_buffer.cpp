#include "gpu_command_buffer.h"

#include <stdexcept>

namespace fac {

GpuCommandBuffer BuildGpuCommandBuffer(const FontDatabase& db,
                                       const RenderPlan& plan)
{
    GpuCommandBuffer out;
    out.total_regions = plan.fitted_regions;
    out.total_glyphs = plan.total_glyphs;
    out.commands.reserve(plan.total_glyphs);
    out.batches.reserve(plan.batches.size());

    for (const RenderBatch& batch : plan.batches)
    {
        const AtlasEntry* atlas = db.GetEntry(batch.atlas_render_size);
        if (!atlas)
        {
            throw std::runtime_error("BuildGpuCommandBuffer: missing atlas for render size "
                                     + std::to_string(batch.atlas_render_size));
        }

        GpuRenderBatch gpu_batch;
        gpu_batch.atlas_render_size = batch.atlas_render_size;
        gpu_batch.atlas_width = atlas->atlas_width;
        gpu_batch.atlas_height = atlas->atlas_height;
        gpu_batch.command_offset = (uint32_t)out.commands.size();
        gpu_batch.command_count = (uint32_t)batch.glyphs.size();

        out.commands.reserve(out.commands.size() + batch.glyphs.size());

        for (const RenderGlyph& glyph : batch.glyphs)
        {
            const GlyphInfo& info = atlas->GetGlyph(glyph.atlas_codepoint);

            GpuRenderCommand cmd;
            cmd.origin_x = glyph.origin_x;
            cmd.origin_y = glyph.origin_y;
            cmd.m00 = glyph.scale * glyph.basis_ux;
            cmd.m01 = glyph.scale * glyph.basis_vx;
            cmd.m10 = glyph.scale * glyph.basis_uy;
            cmd.m11 = glyph.scale * glyph.basis_vy;
            cmd.rgba = glyph.rgba;
            cmd.atlas_x = info.atlas_x;
            cmd.atlas_y = info.atlas_y;
            cmd.atlas_w = info.atlas_w;
            cmd.atlas_h = info.atlas_h;
            out.commands.push_back(cmd);
        }

        out.batches.push_back(gpu_batch);
    }

    return out;
}

} // namespace fac
