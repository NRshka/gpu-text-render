#include "render_request.h"

#include "text_renderer_cpu.h"

#include <limits>
#include <stdexcept>

namespace fac {

namespace {

static void ValidateRegionCounts(const PolygonRegionTensorData& data)
{
    const std::size_t region_count = data.region_texts.size();
    if (data.region_original_texts.size() != region_count
        || data.region_vertex_counts.size() != region_count
        || data.region_has_rgba.size() != region_count
        || data.region_rgba.size() != region_count)
    {
        throw std::runtime_error(
            "BuildTextRegionsFromPolygonTensorData: region tensor lengths must match");
    }

    std::size_t total_vertices = 0;
    for (int32_t count : data.region_vertex_counts)
    {
        if (count < 3)
        {
            throw std::runtime_error(
                "BuildTextRegionsFromPolygonTensorData: each polygon must contain at least 3 vertices");
        }

        const std::size_t count_size = (std::size_t)count;
        if (count_size > std::numeric_limits<std::size_t>::max() - total_vertices)
        {
            throw std::runtime_error(
                "BuildTextRegionsFromPolygonTensorData: vertex count overflow");
        }
        total_vertices += count_size;
    }

    if (total_vertices != data.region_vertices.size())
    {
        throw std::runtime_error(
            "BuildTextRegionsFromPolygonTensorData: vertex counts do not match the vertex tensor");
    }
}

} // namespace

bool IsValidUtf8(std::string_view text) noexcept
{
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t size = text.size();

    for (std::size_t i = 0; i < size; ++i)
    {
        const unsigned char c0 = bytes[i];
        if (c0 <= 0x7Fu)
            continue;

        uint32_t codepoint = 0;
        std::size_t extra = 0;
        uint32_t min_value = 0;

        if ((c0 & 0xE0u) == 0xC0u)
        {
            codepoint = c0 & 0x1Fu;
            extra = 1;
            min_value = 0x80u;
        }
        else if ((c0 & 0xF0u) == 0xE0u)
        {
            codepoint = c0 & 0x0Fu;
            extra = 2;
            min_value = 0x800u;
        }
        else if ((c0 & 0xF8u) == 0xF0u)
        {
            codepoint = c0 & 0x07u;
            extra = 3;
            min_value = 0x10000u;
        }
        else
        {
            return false;
        }

        if (i + extra >= size)
            return false;

        for (std::size_t j = 0; j < extra; ++j)
        {
            const unsigned char cx = bytes[i + 1 + j];
            if ((cx & 0xC0u) != 0x80u)
                return false;
            codepoint = (codepoint << 6) | (uint32_t)(cx & 0x3Fu);
        }

        if (codepoint < min_value
            || codepoint > 0x10FFFFu
            || (codepoint >= 0xD800u && codepoint <= 0xDFFFu))
        {
            return false;
        }

        i += extra;
    }

    return true;
}

std::vector<TextRegion> BuildTextRegionsFromPolygonTensorData(
    const PolygonRegionTensorData& data)
{
    ValidateRegionCounts(data);

    std::vector<TextRegion> regions;
    regions.reserve(data.region_texts.size());

    std::size_t vertex_offset = 0;
    for (std::size_t i = 0; i < data.region_texts.size(); ++i)
    {
        const std::string& text = data.region_texts[i];
        const std::string& original_text = data.region_original_texts[i];
        if (!IsValidUtf8(text))
        {
            throw std::runtime_error(
                "BuildTextRegionsFromPolygonTensorData: region_texts must contain valid UTF-8");
        }
        if (!IsValidUtf8(original_text))
        {
            throw std::runtime_error(
                "BuildTextRegionsFromPolygonTensorData: region_original_texts must contain valid UTF-8");
        }

        const std::size_t vertex_count = (std::size_t)data.region_vertex_counts[i];

        TextRegion region;
        region.text = text;
        region.original_text = original_text;
        region.has_polygon = true;
        region.polygon.assign(data.region_vertices.begin() + (std::ptrdiff_t)vertex_offset,
                              data.region_vertices.begin()
                                  + (std::ptrdiff_t)(vertex_offset + vertex_count));

        if (data.region_has_rgba[i] != 0)
        {
            region.has_explicit_rgba = true;
            region.rgba = data.region_rgba[i];
        }

        regions.push_back(std::move(region));
        vertex_offset += vertex_count;
    }

    return regions;
}

RgbRenderRequest BuildRgbRenderRequest(uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> image_rgb,
                                       const PolygonRegionTensorData& region_data)
{
    if (width == 0 || height == 0)
        throw std::runtime_error("BuildRgbRenderRequest: image dimensions must be positive");

    if (image_rgb.size() != (std::size_t)width * height * 3u)
    {
        throw std::runtime_error(
            "BuildRgbRenderRequest: RGB image byte size does not match width * height * 3");
    }

    RgbRenderRequest request;
    request.width = width;
    request.height = height;
    request.image_rgb = std::move(image_rgb);
    request.regions = BuildTextRegionsFromPolygonTensorData(region_data);
    return request;
}

LumaRenderRequest BuildLumaRenderRequest(uint32_t width,
                                         uint32_t height,
                                         std::vector<uint8_t> image_luma,
                                         const PolygonRegionTensorData& region_data)
{
    if (width == 0 || height == 0)
        throw std::runtime_error("BuildLumaRenderRequest: image dimensions must be positive");

    if (image_luma.size() != (std::size_t)width * height)
    {
        throw std::runtime_error(
            "BuildLumaRenderRequest: luma image byte size does not match width * height");
    }

    LumaRenderRequest request;
    request.width = width;
    request.height = height;
    request.image_luma = std::move(image_luma);
    request.regions = BuildTextRegionsFromPolygonTensorData(region_data);
    return request;
}

PlannedGpuRenderRequest BuildPlannedGpuRenderRequest(
    const FontDatabase& db,
    const LumaRenderRequest& request,
    uint32_t image_index,
    const RenderPlanOptions& options)
{
    if (!request.HasValidImage())
    {
        throw std::runtime_error(
            "BuildPlannedGpuRenderRequest: request image dimensions do not match the luma buffer");
    }

    ImageRgba8 brightness_image(request.width, request.height, 0x000000FFu);
    for (uint32_t y = 0; y < request.height; ++y)
    {
        for (uint32_t x = 0; x < request.width; ++x)
        {
            const uint8_t value = request.image_luma[(std::size_t)y * request.width + x];
            uint8_t* pixel = brightness_image.PixelPtr(x, y);
            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
            pixel[3] = 255u;
        }
    }

    const RenderPlan plan = BuildRenderPlan(db, brightness_image, request.regions, options);

    PlannedGpuRenderRequest out;
    out.width = request.width;
    out.height = request.height;
    out.image_index = image_index;
    out.command_buffer = BuildGpuCommandBufferV2(db, plan, image_index);
    out.command_bytes = SerializeGpuRenderCommandsV2(out.command_buffer.commands);
    out.batch_bytes = SerializeGpuRenderBatchesV2(out.command_buffer.batches);
    return out;
}

} // namespace fac
