#pragma once

#include "font_database.h"
#include "render_plan.h"
#include "text_render_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fac {

// Simple CPU-side RGBA8 image buffer used by the reference renderer.
// Pixels are stored in row-major order as 4 bytes per pixel: R, G, B, A.
struct ImageRgba8
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;

    ImageRgba8() = default;
    ImageRgba8(uint32_t w, uint32_t h, uint32_t rgba = 0x000000FFu);

    bool Empty() const noexcept
    {
        return width == 0 || height == 0 || pixels.size() != (std::size_t)width * height * 4u;
    }

    uint8_t* PixelPtr(uint32_t x, uint32_t y) noexcept
    {
        return pixels.data() + ((std::size_t)y * width + x) * 4u;
    }

    const uint8_t* PixelPtr(uint32_t x, uint32_t y) const noexcept
    {
        return pixels.data() + ((std::size_t)y * width + x) * 4u;
    }
};

// Renders atlas glyphs into the image using the same OBB-aware command data
// that the future GPU path will consume. The packed color is 0xRRGGBBAA.
void RenderGlyphsCpu(ImageRgba8& image,
                     const AtlasEntry& atlas,
                     const std::vector<RenderGlyph>& commands);

void RenderPlanCpu(ImageRgba8& image,
                   const FontDatabase& db,
                   const RenderPlan& plan);

} // namespace fac
