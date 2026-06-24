#pragma once

#include "font_database.h"
#include "text_render_types.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace fac {

enum class HorizontalAlign
{
    Left,
    Center,
};

enum class VerticalAlign
{
    Top,
    Center,
};

struct TextFitOptions
{
    float padding_x = 0.0f;
    float padding_y = 0.0f;
    float max_scale = 0.0f;
    HorizontalAlign horizontal_align = HorizontalAlign::Center;
    VerticalAlign vertical_align     = VerticalAlign::Center;
};

struct TextMeasurement
{
    float min_x         = 0.0f;
    float max_x         = 0.0f;
    float ascent        = 0.0f;
    float descent       = 0.0f;
    float width         = 0.0f;
    float advance_width = 0.0f;
    float ink_height    = 0.0f;
    std::size_t glyph_count = 0;
};

struct TextFitResult
{
    const AtlasEntry* atlas = nullptr;
    float scale             = 0.0f;
    float start_x           = 0.0f;
    float baseline_y        = 0.0f;
    float text_width        = 0.0f;
    float line_height       = 0.0f;

    bool IsValid() const noexcept
    {
        return atlas != nullptr && scale > 0.0f;
    }
};

struct CurveFitResult
{
    const AtlasEntry* atlas = nullptr;
    float scale             = 0.0f;
    float start_offset      = 0.0f;
    float normal_offset     = 0.0f;
    float text_width        = 0.0f;
    float line_height       = 0.0f;
    float curve_length      = 0.0f;

    bool IsValid() const noexcept
    {
        return atlas != nullptr && scale > 0.0f && curve_length > 0.0f;
    }
};

std::vector<uint32_t> DecodeUtf8(std::string_view text);

TextMeasurement MeasureTextLine(const AtlasEntry& atlas,
                                std::string_view text);

TextFitResult FitTextToBox(const FontDatabase& db,
                           std::string_view text,
                           const OrientedBox& box,
                           const TextFitOptions& options = {});

CurveFitResult FitTextToCurve(const FontDatabase& db,
                              std::string_view text,
                              const CurvedTextPath& curve,
                              const TextFitOptions& options = {});

std::vector<GlyphPlacement> LayoutTextLine(const AtlasEntry& atlas,
                                           std::string_view text,
                                           const TextFitResult& fit);

std::vector<RenderGlyph> LayoutTextOnCurve(const AtlasEntry& atlas,
                                           std::string_view text,
                                           const CurvedTextPath& curve,
                                           const CurveFitResult& fit,
                                           uint32_t rgba = 0xFFFFFFFFu);

RenderGlyph MakeRenderGlyph(const GlyphPlacement& placement,
                            const OrientedBox& box,
                            uint32_t rgba = 0xFFFFFFFFu) noexcept;

} // namespace fac
