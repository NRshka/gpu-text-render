#pragma once

#include "text_render_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fac {

struct PolygonRegionTensorData
{
    std::vector<std::string> region_texts;
    std::vector<std::string> region_original_texts;
    std::vector<int32_t> region_vertex_counts;
    std::vector<Vec2f> region_vertices;
    std::vector<uint8_t> region_has_rgba;
    std::vector<uint32_t> region_rgba;
};

struct RgbRenderRequest
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> image_rgb;
    std::vector<TextRegion> regions;

    bool HasValidImage() const noexcept
    {
        return width > 0
            && height > 0
            && image_rgb.size() == (std::size_t)width * height * 3u;
    }
};

bool IsValidUtf8(std::string_view text) noexcept;

std::vector<TextRegion> BuildTextRegionsFromPolygonTensorData(
    const PolygonRegionTensorData& data);

RgbRenderRequest BuildRgbRenderRequest(uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> image_rgb,
                                       const PolygonRegionTensorData& region_data);

} // namespace fac
