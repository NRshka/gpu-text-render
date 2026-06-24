#include "text_layout.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fac {

namespace {

constexpr float kCurveSampleStepPx = 6.0f;
constexpr int kMinCurveSamplesPerSegment = 16;

struct CurveSample
{
    float offset = 0.0f;
    Vec2f point{};
    Vec2f tangent{1.0f, 0.0f};
};

TextMeasurement MeasureDecodedText(const AtlasEntry& atlas,
                                   const std::vector<uint32_t>& codepoints)
{
    TextMeasurement out;
    out.glyph_count = codepoints.size();

    if (codepoints.empty())
        return out;

    float pen_x = 0.0f;
    float min_x = 0.0f;
    float max_x = 0.0f;
    float min_y = 0.0f;
    float max_y = 0.0f;
    bool has_ink = false;

    for (uint32_t cp : codepoints)
    {
        const GlyphInfo& g = atlas.GetGlyph(cp);

        if (g.atlas_w > 0 && g.atlas_h > 0)
        {
            const float left   = pen_x + g.bearing_x;
            const float right  = left + (float)g.atlas_w;
            const float top    = -g.bearing_y;
            const float bottom = top + (float)g.atlas_h;

            if (!has_ink)
            {
                min_x = left;
                max_x = right;
                min_y = top;
                max_y = bottom;
                has_ink = true;
            }
            else
            {
                min_x = std::min(min_x, left);
                max_x = std::max(max_x, right);
                min_y = std::min(min_y, top);
                max_y = std::max(max_y, bottom);
            }
        }

        pen_x += g.advance;
    }

    out.advance_width = pen_x;

    if (has_ink)
    {
        out.min_x      = min_x;
        out.max_x      = std::max(max_x, pen_x);
        out.ascent     = -min_y;
        out.descent    = std::max(0.0f, max_y);
        out.ink_height = max_y - min_y;
    }
    else
    {
        out.min_x      = 0.0f;
        out.max_x      = pen_x;
        out.ascent     = 0.0f;
        out.descent    = 0.0f;
        out.ink_height = 0.0f;
    }

    out.width = out.max_x - out.min_x;
    return out;
}

const AtlasEntry* SelectAtlasForHeight(const FontDatabase& db,
                                       float target_font_height_px) noexcept
{
    const std::vector<uint32_t>& sizes = db.LoadedSizes();
    if (sizes.empty())
        return nullptr;

    const AtlasEntry* fallback = db.GetEntry(sizes.back());
    for (uint32_t sz : sizes)
    {
        const AtlasEntry* entry = db.GetEntry(sz);
        if (!entry)
            continue;

        if ((float)entry->font_height_px >= target_font_height_px)
            return entry;
    }

    return fallback;
}

static Vec2f EvaluateBezier(const CubicBezierSegment& segment, float t) noexcept
{
    const float u = 1.0f - t;
    const float b0 = u * u * u;
    const float b1 = 3.0f * u * u * t;
    const float b2 = 3.0f * u * t * t;
    const float b3 = t * t * t;
    return segment.p0 * b0
         + segment.p1 * b1
         + segment.p2 * b2
         + segment.p3 * b3;
}

static Vec2f EvaluateBezierDerivative(const CubicBezierSegment& segment,
                                      float t) noexcept
{
    const float u = 1.0f - t;
    return (segment.p1 - segment.p0) * (3.0f * u * u)
         + (segment.p2 - segment.p1) * (6.0f * u * t)
         + (segment.p3 - segment.p2) * (3.0f * t * t);
}

static std::vector<CurveSample> BuildCurveSamples(const CurvedTextPath& curve)
{
    std::vector<CurveSample> out;
    if (!curve.IsValid())
        return out;

    std::size_t reserve_count = 1;
    for (const CubicBezierSegment& segment : curve.segments)
    {
        const float chord = Length(segment.p3 - segment.p0);
        reserve_count += (std::size_t)std::max(
            kMinCurveSamplesPerSegment,
            (int)std::ceil(chord / kCurveSampleStepPx));
    }
    out.reserve(reserve_count);

    CurveSample first;
    first.offset = 0.0f;
    first.point = curve.segments.front().p0;
    first.tangent = Normalize(EvaluateBezierDerivative(curve.segments.front(), 0.0f));
    out.push_back(first);

    float total = 0.0f;
    Vec2f prev = first.point;

    for (const CubicBezierSegment& segment : curve.segments)
    {
        const float chord = Length(segment.p3 - segment.p0);
        const int steps = std::max(
            kMinCurveSamplesPerSegment,
            (int)std::ceil(chord / kCurveSampleStepPx));

        for (int i = 1; i <= steps; ++i)
        {
            const float t = (float)i / (float)steps;
            const Vec2f point = EvaluateBezier(segment, t);
            total += Length(point - prev);

            CurveSample sample;
            sample.offset = total;
            sample.point = point;
            sample.tangent = Normalize(EvaluateBezierDerivative(segment, t));
            out.push_back(sample);
            prev = point;
        }
    }

    if (out.size() >= 2 && out.back().offset <= 0.0f)
        out.back().offset = Length(out.back().point - out.front().point);

    return out;
}

static float MeasureCurveLength(const CurvedTextPath& curve)
{
    const std::vector<CurveSample> samples = BuildCurveSamples(curve);
    if (samples.empty())
        return 0.0f;
    return samples.back().offset;
}

static CurveSample SampleCurveAtOffset(const std::vector<CurveSample>& samples,
                                       float offset) noexcept
{
    if (samples.empty())
        return CurveSample{};

    if (offset <= 0.0f)
        return samples.front();

    if (offset >= samples.back().offset)
        return samples.back();

    auto it = std::lower_bound(
        samples.begin(),
        samples.end(),
        offset,
        [](const CurveSample& sample, float value) {
            return sample.offset < value;
        });

    if (it == samples.begin())
        return *it;

    const CurveSample& b = *it;
    const CurveSample& a = *(it - 1);
    const float span = std::max(1e-6f, b.offset - a.offset);
    const float t = (offset - a.offset) / span;

    CurveSample out;
    out.offset = offset;
    out.point = a.point + (b.point - a.point) * t;
    out.tangent = Normalize(a.tangent + (b.tangent - a.tangent) * t);
    return out;
}

static Vec2f CurveNormal(const Vec2f& tangent, CurveNormalSide side) noexcept
{
    if (side == CurveNormalSide::Left)
        return Normalize(Vec2f{tangent.y, -tangent.x});
    return Normalize(Vec2f{-tangent.y, tangent.x});
}

} // namespace

std::vector<uint32_t> DecodeUtf8(std::string_view text)
{
    std::vector<uint32_t> out;
    out.reserve(text.size());

    const unsigned char* s =
        reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t n = text.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        const unsigned char c = s[i];

        if (c < 0x80)
        {
            out.push_back(c);
            continue;
        }

        uint32_t cp = 0xFFFD;
        std::size_t need = 0;
        uint32_t min_cp = 0;

        if ((c & 0xE0) == 0xC0)
        {
            cp = c & 0x1Fu;
            need = 1;
            min_cp = 0x80u;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            cp = c & 0x0Fu;
            need = 2;
            min_cp = 0x800u;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            cp = c & 0x07u;
            need = 3;
            min_cp = 0x10000u;
        }
        else
        {
            out.push_back(0xFFFDu);
            continue;
        }

        if (i + need >= n)
        {
            out.push_back(0xFFFDu);
            break;
        }

        bool valid = true;
        for (std::size_t j = 1; j <= need; ++j)
        {
            const unsigned char cc = s[i + j];
            if ((cc & 0xC0) != 0x80)
            {
                valid = false;
                break;
            }
            cp = (cp << 6) | (uint32_t)(cc & 0x3Fu);
        }

        if (!valid
            || cp < min_cp
            || cp > 0x10FFFFu
            || (cp >= 0xD800u && cp <= 0xDFFFu))
        {
            out.push_back(0xFFFDu);
            continue;
        }

        out.push_back(cp);
        i += need;
    }

    return out;
}

TextMeasurement MeasureTextLine(const AtlasEntry& atlas,
                                std::string_view text)
{
    return MeasureDecodedText(atlas, DecodeUtf8(text));
}

TextFitResult FitTextToBox(const FontDatabase& db,
                           std::string_view text,
                           const OrientedBox& box,
                           const TextFitOptions& options)
{
    TextFitResult out;

    if (!box.HasArea() || db.Empty())
        return out;

    const float inner_w = std::max(0.0f, box.width - 2.0f * options.padding_x);
    const float inner_h = std::max(0.0f, box.height - 2.0f * options.padding_y);
    if (inner_w <= 0.0f || inner_h <= 0.0f)
        return out;

    const AtlasEntry* atlas = SelectAtlasForHeight(db, inner_h);
    if (!atlas || atlas->font_height_px == 0)
        return out;

    const TextMeasurement m = MeasureTextLine(*atlas, text);
    if (m.glyph_count == 0)
        return out;

    const float width_scale = (m.width > 0.0f)
        ? (inner_w / m.width)
        : std::numeric_limits<float>::infinity();
    const float height_scale = inner_h / (float)atlas->font_height_px;
    float scale = std::min(width_scale, height_scale);
    if (options.max_scale > 0.0f)
        scale = std::min(scale, options.max_scale);

    if (!(scale > 0.0f))
        return out;

    const float scaled_width = m.width * scale;
    const float scaled_line_h = (float)atlas->font_height_px * scale;

    const float local_left = -0.5f * box.width + options.padding_x;
    const float local_top  = -0.5f * box.height + options.padding_y;

    out.atlas       = atlas;
    out.scale       = scale;
    out.text_width  = scaled_width;
    out.line_height = scaled_line_h;

    if (options.horizontal_align == HorizontalAlign::Left)
        out.start_x = local_left;
    else
        out.start_x = local_left + 0.5f * (inner_w - scaled_width);

    const float line_top = (options.vertical_align == VerticalAlign::Top)
        ? local_top
        : local_top + 0.5f * (inner_h - scaled_line_h);

    out.baseline_y = line_top + m.ascent * scale;
    return out;
}

CurveFitResult FitTextToCurve(const FontDatabase& db,
                              std::string_view text,
                              const CurvedTextPath& curve,
                              const TextFitOptions& options)
{
    CurveFitResult out;

    if (!curve.IsValid() || db.Empty())
        return out;

    const float inner_w = std::max(0.0f, MeasureCurveLength(curve) - 2.0f * options.padding_x);
    const float inner_h = std::max(0.0f, curve.band_height - 2.0f * options.padding_y);
    if (inner_w <= 0.0f || inner_h <= 0.0f)
        return out;

    const AtlasEntry* atlas = SelectAtlasForHeight(db, inner_h);
    if (!atlas || atlas->font_height_px == 0)
        return out;

    const TextMeasurement m = MeasureTextLine(*atlas, text);
    if (m.glyph_count == 0)
        return out;

    const float width_scale = (m.width > 0.0f)
        ? (inner_w / m.width)
        : std::numeric_limits<float>::infinity();
    const float height_scale = inner_h / (float)atlas->font_height_px;
    float scale = std::min(width_scale, height_scale);
    if (options.max_scale > 0.0f)
        scale = std::min(scale, options.max_scale);
    if (!(scale > 0.0f))
        return out;

    out.atlas = atlas;
    out.scale = scale;
    out.text_width = m.width * scale;
    out.line_height = (float)atlas->font_height_px * scale;
    out.curve_length = inner_w + 2.0f * options.padding_x;

    if (options.horizontal_align == HorizontalAlign::Left)
        out.start_offset = options.padding_x;
    else
        out.start_offset = options.padding_x + 0.5f * (inner_w - out.text_width);

    const float line_top = (options.vertical_align == VerticalAlign::Top)
        ? (-0.5f * curve.band_height + options.padding_y)
        : (-0.5f * curve.band_height + options.padding_y + 0.5f * (inner_h - out.line_height));
    out.normal_offset = line_top + m.ascent * scale;
    return out;
}

std::vector<GlyphPlacement> LayoutTextLine(const AtlasEntry& atlas,
                                           std::string_view text,
                                           const TextFitResult& fit)
{
    std::vector<uint32_t> codepoints = DecodeUtf8(text);
    std::vector<GlyphPlacement> out;
    out.reserve(codepoints.size());

    if (!fit.IsValid() || fit.atlas != &atlas)
        return out;

    const TextMeasurement m = MeasureDecodedText(atlas, codepoints);
    float pen_x = 0.0f;

    for (uint32_t cp : codepoints)
    {
        const GlyphInfo& g = atlas.GetGlyph(cp);

        GlyphPlacement p;
        p.codepoint = cp;
        p.scale = fit.scale;
        p.local_x = fit.start_x + (pen_x + g.bearing_x - m.min_x) * fit.scale;
        p.local_y = fit.baseline_y - g.bearing_y * fit.scale;
        out.push_back(p);

        pen_x += g.advance;
    }

    return out;
}

std::vector<RenderGlyph> LayoutTextOnCurve(const AtlasEntry& atlas,
                                           std::string_view text,
                                           const CurvedTextPath& curve,
                                           const CurveFitResult& fit,
                                           uint32_t rgba)
{
    std::vector<uint32_t> codepoints = DecodeUtf8(text);
    std::vector<RenderGlyph> out;
    out.reserve(codepoints.size());

    if (!curve.IsValid() || !fit.IsValid() || fit.atlas != &atlas)
        return out;

    const std::vector<CurveSample> samples = BuildCurveSamples(curve);
    if (samples.size() < 2 || samples.back().offset <= 0.0f)
        return out;

    const TextMeasurement m = MeasureDecodedText(atlas, codepoints);
    float pen_x = 0.0f;

    for (uint32_t cp : codepoints)
    {
        const GlyphInfo& g = atlas.GetGlyph(cp);
        const float glyph_offset =
            fit.start_offset + (pen_x + g.bearing_x - m.min_x) * fit.scale;
        const CurveSample sample = SampleCurveAtOffset(samples, glyph_offset);
        const Vec2f tangent = Normalize(sample.tangent);
        const Vec2f normal = CurveNormal(tangent, curve.normal_side);
        const Vec2f origin = sample.point + normal * (fit.normal_offset - g.bearing_y * fit.scale);

        RenderGlyph glyph;
        glyph.atlas_codepoint = cp;
        glyph.origin_x = origin.x;
        glyph.origin_y = origin.y;
        glyph.basis_ux = tangent.x;
        glyph.basis_uy = tangent.y;
        glyph.basis_vx = normal.x;
        glyph.basis_vy = normal.y;
        glyph.scale = fit.scale;
        glyph.rgba = rgba;
        out.push_back(glyph);

        pen_x += g.advance;
    }

    return out;
}

RenderGlyph MakeRenderGlyph(const GlyphPlacement& placement,
                            const OrientedBox& box,
                            uint32_t rgba) noexcept
{
    const Vec2f origin = box.LocalToImage(placement.local_x, placement.local_y);

    RenderGlyph out;
    out.atlas_codepoint = placement.codepoint;
    out.origin_x = origin.x;
    out.origin_y = origin.y;
    out.basis_ux = box.ux;
    out.basis_uy = box.uy;
    out.basis_vx = box.vx;
    out.basis_vy = box.vy;
    out.scale    = placement.scale;
    out.rgba     = rgba;
    return out;
}

} // namespace fac
