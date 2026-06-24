#include "text_renderer_cpu.h"

#include <algorithm>
#include <cmath>

namespace fac {

namespace {

struct ColorF
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

static ColorF UnpackRgba(uint32_t rgba) noexcept
{
    return ColorF{
        (float)((rgba >> 24) & 0xFFu) / 255.0f,
        (float)((rgba >> 16) & 0xFFu) / 255.0f,
        (float)((rgba >>  8) & 0xFFu) / 255.0f,
        (float)( rgba        & 0xFFu) / 255.0f,
    };
}

static uint8_t ToByte(float v) noexcept
{
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    return (uint8_t)std::lround(clamped * 255.0f);
}

static float FetchGlyphCoverage(const AtlasEntry& atlas,
                                const GlyphInfo& glyph,
                                int gx,
                                int gy) noexcept
{
    if (gx < 0 || gy < 0
        || gx >= (int)glyph.atlas_w
        || gy >= (int)glyph.atlas_h)
    {
        return 0.0f;
    }

    const uint32_t sx = glyph.atlas_x + (uint32_t)gx;
    const uint32_t sy = glyph.atlas_y + (uint32_t)gy;
    return (float)atlas.pixels[(std::size_t)sy * atlas.atlas_width + sx] / 255.0f;
}

static float SampleGlyphCoverageBilinear(const AtlasEntry& atlas,
                                         const GlyphInfo& glyph,
                                         float local_x,
                                         float local_y) noexcept
{
    const float sample_x = local_x - 0.5f;
    const float sample_y = local_y - 0.5f;
    const int x0 = (int)std::floor(sample_x);
    const int y0 = (int)std::floor(sample_y);
    const float fx = sample_x - (float)x0;
    const float fy = sample_y - (float)y0;

    const float c00 = FetchGlyphCoverage(atlas, glyph, x0,     y0);
    const float c10 = FetchGlyphCoverage(atlas, glyph, x0 + 1, y0);
    const float c01 = FetchGlyphCoverage(atlas, glyph, x0,     y0 + 1);
    const float c11 = FetchGlyphCoverage(atlas, glyph, x0 + 1, y0 + 1);

    const float cx0 = c00 + (c10 - c00) * fx;
    const float cx1 = c01 + (c11 - c01) * fx;
    return cx0 + (cx1 - cx0) * fy;
}

static void FillImage(ImageRgba8& image, uint32_t rgba)
{
    const ColorF c = UnpackRgba(rgba);
    image.pixels.resize((std::size_t)image.width * image.height * 4u);

    for (uint32_t y = 0; y < image.height; ++y)
    {
        for (uint32_t x = 0; x < image.width; ++x)
        {
            uint8_t* p = image.PixelPtr(x, y);
            p[0] = ToByte(c.r);
            p[1] = ToByte(c.g);
            p[2] = ToByte(c.b);
            p[3] = ToByte(c.a);
        }
    }
}

} // namespace

ImageRgba8::ImageRgba8(uint32_t w, uint32_t h, uint32_t rgba)
    : width(w), height(h)
{
    FillImage(*this, rgba);
}

void RenderGlyphsCpu(ImageRgba8& image,
                     const AtlasEntry& atlas,
                     const std::vector<RenderGlyph>& commands)
{
    if (image.Empty())
        return;

    for (const RenderGlyph& cmd : commands)
    {
        const GlyphInfo& glyph = atlas.GetGlyph(cmd.atlas_codepoint);
        if (glyph.atlas_w == 0 || glyph.atlas_h == 0 || cmd.scale <= 0.0f)
            continue;

        // Transform from glyph-local pixels into image space:
        // image = origin + scale * [u v] * local
        const float m00 = cmd.scale * cmd.basis_ux;
        const float m01 = cmd.scale * cmd.basis_vx;
        const float m10 = cmd.scale * cmd.basis_uy;
        const float m11 = cmd.scale * cmd.basis_vy;

        const float det = m00 * m11 - m01 * m10;
        if (std::fabs(det) < 1e-8f)
            continue;

        const float inv00 =  m11 / det;
        const float inv01 = -m01 / det;
        const float inv10 = -m10 / det;
        const float inv11 =  m00 / det;

        float min_x = cmd.origin_x;
        float max_x = cmd.origin_x;
        float min_y = cmd.origin_y;
        float max_y = cmd.origin_y;

        const float w = (float)glyph.atlas_w;
        const float h = (float)glyph.atlas_h;
        const float corners[4][2] = {
            {0.0f, 0.0f},
            {w,    0.0f},
            {0.0f, h   },
            {w,    h   },
        };

        for (const auto& c : corners)
        {
            const float ix = cmd.origin_x + m00 * c[0] + m01 * c[1];
            const float iy = cmd.origin_y + m10 * c[0] + m11 * c[1];
            min_x = std::min(min_x, ix);
            max_x = std::max(max_x, ix);
            min_y = std::min(min_y, iy);
            max_y = std::max(max_y, iy);
        }

        const int x0 = std::max(0, (int)std::floor(min_x));
        const int y0 = std::max(0, (int)std::floor(min_y));
        const int x1 = std::min((int)image.width - 1, (int)std::ceil(max_x));
        const int y1 = std::min((int)image.height - 1, (int)std::ceil(max_y));
        if (x0 > x1 || y0 > y1)
            continue;

        const ColorF src_color = UnpackRgba(cmd.rgba);

        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                const float dx = (float)x + 0.5f - cmd.origin_x;
                const float dy = (float)y + 0.5f - cmd.origin_y;

                const float local_x = inv00 * dx + inv01 * dy;
                const float local_y = inv10 * dx + inv11 * dy;

                if (local_x < 0.0f || local_y < 0.0f
                    || local_x >= w || local_y >= h)
                    continue;

                const float coverage =
                    SampleGlyphCoverageBilinear(atlas, glyph, local_x, local_y);
                if (coverage <= 0.0f)
                    continue;

                const float src_a = src_color.a * coverage;

                uint8_t* p = image.PixelPtr((uint32_t)x, (uint32_t)y);
                const float dst_r = (float)p[0] / 255.0f;
                const float dst_g = (float)p[1] / 255.0f;
                const float dst_b = (float)p[2] / 255.0f;
                const float dst_a = (float)p[3] / 255.0f;

                const float out_r = src_a * src_color.r + (1.0f - src_a) * dst_r;
                const float out_g = src_a * src_color.g + (1.0f - src_a) * dst_g;
                const float out_b = src_a * src_color.b + (1.0f - src_a) * dst_b;
                const float out_a = src_a + (1.0f - src_a) * dst_a;

                p[0] = ToByte(out_r);
                p[1] = ToByte(out_g);
                p[2] = ToByte(out_b);
                p[3] = ToByte(out_a);
            }
        }
    }
}

void RenderPlanCpu(ImageRgba8& image,
                   const FontDatabase& db,
                   const RenderPlan& plan)
{
    for (const RenderBatch& batch : plan.batches)
    {
        const AtlasEntry* atlas = db.GetEntry(batch.atlas_render_size);
        if (!atlas)
            continue;
        RenderGlyphsCpu(image, *atlas, batch.glyphs);
    }
}

} // namespace fac
