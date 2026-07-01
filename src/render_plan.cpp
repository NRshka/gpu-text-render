#include "render_plan.h"

#include "text_renderer_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>

namespace fac {

namespace {

constexpr float kLineAngleThresholdDeg = 10.0f;
constexpr float kBrightnessThreshold = 128.0f;
constexpr float kStraightCurveDeviationPx = 4.0f;
constexpr float kCurveSampleStepPx = 6.0f;
constexpr float kVisualFitHeightFill = 0.90f;
constexpr float kVisualFitFontHeightFloor = 0.65f;
constexpr int kMinCurveSamplesPerSegment = 16;

enum class GeometryKind
{
    Straight,
    Curved,
};

enum class RegionPlannerKind
{
    FastStraight,
    Quality,
};

struct LocalBounds
{
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;

    float Width() const noexcept
    {
        return max_x - min_x;
    }

    float Height() const noexcept
    {
        return max_y - min_y;
    }

    float CenterX() const noexcept
    {
        return 0.5f * (min_x + max_x);
    }

    float CenterY() const noexcept
    {
        return 0.5f * (min_y + max_y);
    }
};

struct CurveSample
{
    float offset = 0.0f;
    Vec2f point{};
};

static std::vector<CurveSample> BuildCurvePolyline(const CurvedTextPath& curve);

struct ResolvedGeometry
{
    GeometryKind kind = GeometryKind::Straight;
    OrientedBox box;
    CurvedTextPath curve;
};

struct PlannedRegion
{
    const TextRegion* region = nullptr;
    std::size_t original_index = 0;
    ResolvedGeometry geometry;
    TextFitResult fit;
    CurveFitResult curve_fit;
    TextMeasurement measurement;
    uint32_t resolved_rgba = 0xFFFFFFFFu;
    bool valid = false;

    bool IsCurved() const noexcept
    {
        return geometry.kind == GeometryKind::Curved;
    }
};

struct RegionFeatures
{
    RegionPlannerKind initial_kind = RegionPlannerKind::Quality;
    OrientedBox straight_box;
    bool brightness_ambiguous = false;
    float translated_ratio = 1.0f;
};

struct GlyphQuad
{
    const AtlasEntry* atlas = nullptr;
    uint32_t codepoint = 0;
    std::array<Vec2f, 4> corners{};

    Vec2f Center() const noexcept
    {
        Vec2f center{};
        for (const Vec2f& point : corners)
            center = center + point;
        return center * 0.25f;
    }
};

struct PlacedFootprint
{
    std::vector<GlyphQuad> quads;
    bool overflowed = false;
};

struct AcceptedPlacement
{
    bool valid = false;
    bool overflowed = false;
    bool is_curved = false;
    uint32_t render_size = 0;
    std::vector<RenderGlyph> commands;
    OrientedBox placed_box;
    PlacedFootprint footprint;
};

struct PlacementCandidate
{
    bool valid = false;
    bool overflowed = false;
    std::size_t outside_score = 0;
    float displacement2 = 0.0f;
    float tangent_delta = 0.0f;
    float normal_delta = 0.0f;
    TextFitResult fit;
    CurveFitResult curve_fit;
    std::vector<RenderGlyph> glyphs;
    std::vector<GlyphQuad> quads;
};

static PlacementCandidate ResolveOverflowPlacement(const PlannedRegion& region,
                                                   const RenderPlanOptions& options,
                                                   const std::vector<PlacedFootprint>& placed);

static TextFitResult ResolveOverlaps(const PlannedRegion& region,
                                     const RenderPlanOptions& options,
                                     const std::vector<OrientedBox>& placed);

static float Clamp(float v, float lo, float hi) noexcept
{
    return std::max(lo, std::min(v, hi));
}

static Vec2f Sub(const Vec2f& a, const Vec2f& b) noexcept
{
    return Vec2f{a.x - b.x, a.y - b.y};
}

static std::size_t CountTrimmedCodepoints(std::string_view text)
{
    const std::vector<uint32_t> cps = DecodeUtf8(text);
    std::size_t first = 0;
    std::size_t last = cps.size();

    const auto is_ws = [](uint32_t cp) {
        return cp == 0x20u || cp == 0x09u || cp == 0x0Au || cp == 0x0Du;
    };

    while (first < last && is_ws(cps[first]))
        ++first;
    while (last > first && is_ws(cps[last - 1]))
        --last;

    return last - first;
}

static float PolygonArea(const std::vector<Vec2f>& polygon) noexcept
{
    if (polygon.size() < 3)
        return 0.0f;

    double area2 = 0.0;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
        area2 += (double)polygon[j].x * polygon[i].y - (double)polygon[i].x * polygon[j].y;
    return (float)(std::fabs(area2) * 0.5);
}

static const AtlasEntry* SelectAtlasForHeight(const FontDatabase& db,
                                              float target_font_height_px) noexcept
{
    const std::vector<uint32_t>& sizes = db.LoadedSizes();
    if (sizes.empty())
        return nullptr;

    const AtlasEntry* fallback = db.GetEntry(sizes.back());
    for (uint32_t size : sizes)
    {
        const AtlasEntry* entry = db.GetEntry(size);
        if (!entry)
            continue;
        if ((float)entry->font_height_px >= target_font_height_px)
            return entry;
    }
    return fallback;
}

static float VisualFitHeight(const AtlasEntry& atlas,
                             const TextMeasurement& measurement) noexcept
{
    const float font_floor =
        (float)atlas.font_height_px * kVisualFitFontHeightFloor;
    return std::max(measurement.ink_height, font_floor);
}

static TextFitOptions MakeFitOptions(const RenderPlanOptions& options) noexcept
{
    return options.fit;
}

static LocalBounds ComputeBounds(const TextFitResult& fit,
                                 const TextMeasurement& measurement) noexcept
{
    LocalBounds bounds;
    bounds.min_x = fit.start_x;
    bounds.max_x = fit.start_x + fit.text_width;
    bounds.min_y = fit.baseline_y - measurement.ascent * fit.scale;
    bounds.max_y = fit.baseline_y + measurement.descent * fit.scale;
    return bounds;
}

static void RecenterFit(const OrientedBox& box,
                        const TextMeasurement& measurement,
                        const TextFitOptions& options,
                        TextFitResult& fit) noexcept
{
    if (!fit.IsValid() || fit.atlas == nullptr)
        return;

    const float inner_w = std::max(0.0f, box.width - 2.0f * options.padding_x);
    const float inner_h = std::max(0.0f, box.height - 2.0f * options.padding_y);
    const float local_left = -0.5f * box.width + options.padding_x;
    const float local_top  = -0.5f * box.height + options.padding_y;

    fit.text_width = measurement.width * fit.scale;
    fit.line_height = VisualFitHeight(*fit.atlas, measurement) * fit.scale;

    if (options.horizontal_align == HorizontalAlign::Left)
        fit.start_x = local_left;
    else
        fit.start_x = local_left + 0.5f * (inner_w - fit.text_width);

    const float line_top = (options.vertical_align == VerticalAlign::Top)
        ? local_top
        : local_top + 0.5f * (inner_h - fit.line_height);
    fit.baseline_y = line_top + measurement.ascent * fit.scale;
}

static void ScaleFit(const OrientedBox& box,
                     const TextMeasurement& measurement,
                     const TextFitOptions& options,
                     float scale_factor,
                     TextFitResult& fit) noexcept
{
    if (!fit.IsValid() || !(scale_factor > 0.0f))
        return;

    fit.scale *= scale_factor;
    RecenterFit(box, measurement, options, fit);
}

static void RecenterCurveFit(const CurvedTextPath& curve,
                             const TextMeasurement& measurement,
                             const TextFitOptions& options,
                             CurveFitResult& fit) noexcept
{
    if (!fit.IsValid() || fit.atlas == nullptr)
        return;

    const float inner_w = std::max(0.0f, fit.curve_length - 2.0f * options.padding_x);
    const float inner_h = std::max(0.0f, curve.band_height - 2.0f * options.padding_y);

    fit.text_width = measurement.width * fit.scale;
    fit.line_height = VisualFitHeight(*fit.atlas, measurement) * fit.scale;

    if (options.horizontal_align == HorizontalAlign::Left)
        fit.start_offset = options.padding_x;
    else
        fit.start_offset = options.padding_x + 0.5f * (inner_w - fit.text_width);

    const float line_top = (options.vertical_align == VerticalAlign::Top)
        ? (-0.5f * curve.band_height + options.padding_y)
        : (-0.5f * curve.band_height + options.padding_y
           + 0.5f * (inner_h - fit.line_height));
    fit.normal_offset = line_top + measurement.ascent * fit.scale;
}

static void ScaleCurveFit(const CurvedTextPath& curve,
                          const TextMeasurement& measurement,
                          const TextFitOptions& options,
                          float scale_factor,
                          CurveFitResult& fit) noexcept
{
    if (!fit.IsValid() || !(scale_factor > 0.0f))
        return;

    fit.scale *= scale_factor;
    RecenterCurveFit(curve, measurement, options, fit);
}

static TextFitResult MakeReadableFit(const FontDatabase& db,
                                     std::string_view text,
                                     const OrientedBox& box,
                                     const TextFitOptions& options)
{
    TextFitResult fit;
    if (!box.HasArea() || db.Empty())
        return fit;

    const float inner_h = std::max(0.0f, box.height - 2.0f * options.padding_y);
    if (inner_h <= 0.0f)
        return fit;

    const AtlasEntry* atlas = SelectAtlasForHeight(db, inner_h);
    if (!atlas || atlas->font_height_px == 0)
        return fit;

    const TextMeasurement measurement = MeasureTextLine(*atlas, text);
    if (measurement.glyph_count == 0)
        return fit;

    fit.atlas = atlas;
    fit.scale =
        (inner_h * kVisualFitHeightFill) / VisualFitHeight(*atlas, measurement);
    if (options.max_scale > 0.0f)
        fit.scale = std::min(fit.scale, options.max_scale);
    if (!(fit.scale > 0.0f))
        return TextFitResult{};

    RecenterFit(box, measurement, options, fit);
    return fit;
}

static CurveFitResult MakeReadableCurveFit(const FontDatabase& db,
                                           std::string_view text,
                                           const CurvedTextPath& curve,
                                           const TextFitOptions& options)
{
    CurveFitResult fit;
    if (!curve.IsValid() || db.Empty())
        return fit;

    const float inner_h = std::max(0.0f, curve.band_height - 2.0f * options.padding_y);
    if (inner_h <= 0.0f)
        return fit;

    const AtlasEntry* atlas = SelectAtlasForHeight(db, inner_h);
    if (!atlas || atlas->font_height_px == 0)
        return fit;

    const TextMeasurement measurement = MeasureTextLine(*atlas, text);
    if (measurement.glyph_count == 0)
        return fit;

    fit.atlas = atlas;
    fit.scale =
        (inner_h * kVisualFitHeightFill) / VisualFitHeight(*atlas, measurement);
    if (options.max_scale > 0.0f)
        fit.scale = std::min(fit.scale, options.max_scale);
    if (!(fit.scale > 0.0f))
        return CurveFitResult{};

    fit.curve_length = 0.0f;
    const std::vector<CurveSample> polyline = BuildCurvePolyline(curve);
    if (!polyline.empty())
        fit.curve_length = polyline.back().offset;
    if (!(fit.curve_length > 0.0f))
        return CurveFitResult{};

    RecenterCurveFit(curve, measurement, options, fit);
    return fit;
}

static void ShiftFit(TextFitResult& fit, float dx, float dy) noexcept
{
    fit.start_x += dx;
    fit.baseline_y += dy;
}

static void ShiftCurveFit(CurveFitResult& fit, float tangent_delta, float normal_delta) noexcept
{
    fit.start_offset += tangent_delta;
    fit.normal_offset += normal_delta;
}

static bool ClampShiftInsideBox(const OrientedBox& box,
                                const TextMeasurement& measurement,
                                const TextFitOptions& options,
                                TextFitResult& fit,
                                float desired_dx,
                                float desired_dy) noexcept
{
    const float box_left = -0.5f * box.width + options.padding_x;
    const float box_top = -0.5f * box.height + options.padding_y;
    const float box_right = 0.5f * box.width - options.padding_x;
    const float box_bottom = 0.5f * box.height - options.padding_y;

    const LocalBounds bounds = ComputeBounds(fit, measurement);
    const float min_dx = box_left - bounds.min_x;
    const float max_dx = box_right - bounds.max_x;
    const float min_dy = box_top - bounds.min_y;
    const float max_dy = box_bottom - bounds.max_y;

    const float dx = Clamp(desired_dx, min_dx, max_dx);
    const float dy = Clamp(desired_dy, min_dy, max_dy);
    if (std::fabs(dx) <= 1e-6f && std::fabs(dy) <= 1e-6f)
        return false;

    ShiftFit(fit, dx, dy);
    return true;
}

static void AlignBaselineFraction(const OrientedBox& box,
                                  const TextMeasurement& measurement,
                                  const TextFitOptions& options,
                                  float baseline_fraction,
                                  TextFitResult& fit) noexcept
{
    const float desired = -0.5f * box.height + baseline_fraction * box.height;
    const float delta = desired - fit.baseline_y;
    ClampShiftInsideBox(box, measurement, options, fit, 0.0f, delta);
}

static OrientedBox MakePlacedBox(const OrientedBox& box,
                                 const TextFitResult& fit,
                                 const TextMeasurement& measurement) noexcept
{
    const LocalBounds bounds = ComputeBounds(fit, measurement);

    OrientedBox placed = box;
    placed.cx = box.cx + bounds.CenterX() * box.ux + bounds.CenterY() * box.vx;
    placed.cy = box.cy + bounds.CenterX() * box.uy + bounds.CenterY() * box.vy;
    placed.width = std::max(0.0f, bounds.Width());
    placed.height = std::max(0.0f, bounds.Height());
    return placed;
}

static std::array<Vec2f, 4> BoxCorners(const OrientedBox& box) noexcept
{
    const float hw = 0.5f * box.width;
    const float hh = 0.5f * box.height;
    return {
        box.LocalToImage(-hw, -hh),
        box.LocalToImage( hw, -hh),
        box.LocalToImage( hw,  hh),
        box.LocalToImage(-hw,  hh),
    };
}

static bool OverlapsOnAxis(const std::array<Vec2f, 4>& a,
                           const std::array<Vec2f, 4>& b,
                           const Vec2f& axis) noexcept
{
    float a_min = Dot(a[0], axis);
    float a_max = a_min;
    float b_min = Dot(b[0], axis);
    float b_max = b_min;

    for (std::size_t i = 1; i < 4; ++i)
    {
        const float ap = Dot(a[i], axis);
        const float bp = Dot(b[i], axis);
        a_min = std::min(a_min, ap);
        a_max = std::max(a_max, ap);
        b_min = std::min(b_min, bp);
        b_max = std::max(b_max, bp);
    }

    return !(a_max < b_min || b_max < a_min);
}

static std::array<Vec2f, 4> ExpandCorners(const std::array<Vec2f, 4>& corners,
                                          float margin) noexcept
{
    Vec2f center{};
    for (const Vec2f& p : corners)
    {
        center.x += p.x;
        center.y += p.y;
    }
    center.x *= 0.25f;
    center.y *= 0.25f;

    std::array<Vec2f, 4> expanded = corners;
    for (Vec2f& p : expanded)
    {
        const Vec2f dir = Normalize(Sub(p, center));
        p.x += dir.x * margin;
        p.y += dir.y * margin;
    }

    return expanded;
}

static bool BoxesOverlap(const OrientedBox& a,
                         const OrientedBox& b,
                         float margin) noexcept
{
    if (!a.HasArea() || !b.HasArea())
        return false;

    const std::array<Vec2f, 4> a_corners = ExpandCorners(BoxCorners(a), margin);
    const std::array<Vec2f, 4> b_corners = ExpandCorners(BoxCorners(b), margin);

    const Vec2f axes[4] = {
        Normalize(Vec2f{-a.uy, a.ux}),
        Normalize(Vec2f{-a.vy, a.vx}),
        Normalize(Vec2f{-b.uy, b.ux}),
        Normalize(Vec2f{-b.vy, b.vx}),
    };

    for (const Vec2f& axis : axes)
    {
        if (!OverlapsOnAxis(a_corners, b_corners, axis))
            return false;
    }

    return true;
}

static bool OverlapsAny(const OrientedBox& candidate,
                        const std::vector<OrientedBox>& placed,
                        float margin) noexcept
{
    for (const OrientedBox& other : placed)
    {
        if (BoxesOverlap(candidate, other, margin))
            return true;
    }
    return false;
}

static std::size_t FindNearestPlaced(const OrientedBox& candidate,
                                     const std::vector<OrientedBox>& placed) noexcept
{
    std::size_t best_index = 0;
    float best_dist2 = std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < placed.size(); ++i)
    {
        const float dx = candidate.cx - placed[i].cx;
        const float dy = candidate.cy - placed[i].cy;
        const float dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2)
        {
            best_dist2 = dist2;
            best_index = i;
        }
    }

    return best_index;
}

static bool TryResolveByNudging(const OrientedBox& box,
                                const TextMeasurement& measurement,
                                const RenderPlanOptions& options,
                                const std::vector<OrientedBox>& placed,
                                TextFitResult& fit) noexcept
{
    if (placed.empty())
        return false;

    const OrientedBox base_box = MakePlacedBox(box, fit, measurement);
    const OrientedBox& nearest = placed[FindNearestPlaced(base_box, placed)];
    const Vec2f away = Normalize(Vec2f{base_box.cx - nearest.cx, base_box.cy - nearest.cy});
    const float dir_local_x = away.x * box.ux + away.y * box.uy;
    const float dir_local_y = away.x * box.vx + away.y * box.vy;
    const float dir_len = std::sqrt(dir_local_x * dir_local_x + dir_local_y * dir_local_y);

    const float nx = (dir_len > 1e-6f) ? (dir_local_x / dir_len) : 1.0f;
    const float ny = (dir_len > 1e-6f) ? (dir_local_y / dir_len) : 0.0f;
    const float max_dx = options.max_nudge_frac * box.width;
    const float max_dy = options.max_nudge_frac * box.height;

    for (int step = 1; step <= std::max(1, options.nudge_steps); ++step)
    {
        const float t = (float)step / (float)std::max(1, options.nudge_steps);

        TextFitResult candidate = fit;
        if (!ClampShiftInsideBox(box,
                                 measurement,
                                 options.fit,
                                 candidate,
                                 nx * max_dx * t,
                                 ny * max_dy * t))
        {
            continue;
        }

        const OrientedBox candidate_box = MakePlacedBox(box, candidate, measurement);
        if (!OverlapsAny(candidate_box, placed, options.overlap_margin_px))
        {
            fit = candidate;
            return true;
        }
    }

    return false;
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

static std::vector<CurveSample> BuildCurvePolyline(const CurvedTextPath& curve)
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
    first.point = curve.segments.front().p0;
    out.push_back(first);

    Vec2f prev = first.point;
    float total = 0.0f;
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
            out.push_back(sample);
            prev = point;
        }
    }

    return out;
}

static float DistancePointToSegment(const Vec2f& p,
                                    const Vec2f& a,
                                    const Vec2f& b) noexcept
{
    const Vec2f ab = b - a;
    const float len2 = LengthSquared(ab);
    if (len2 <= 1e-12f)
        return Length(p - a);

    const float t = Clamp(Dot(p - a, ab) / len2, 0.0f, 1.0f);
    const Vec2f closest = a + ab * t;
    return Length(p - closest);
}

static bool PointInPolygon(const std::vector<Vec2f>& polygon,
                           const Vec2f& p) noexcept
{
    bool inside = false;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
    {
        const Vec2f& a = polygon[i];
        const Vec2f& b = polygon[j];
        const bool crosses =
            ((a.y > p.y) != (b.y > p.y))
            && (p.x < (b.x - a.x) * (p.y - a.y) / std::max(b.y - a.y, 1e-12f) + a.x);
        if (crosses)
            inside = !inside;
    }
    return inside;
}

static GlyphQuad MakeGlyphQuad(const AtlasEntry& atlas,
                               const RenderGlyph& glyph) noexcept
{
    const GlyphInfo& info = atlas.GetGlyph(glyph.atlas_codepoint);

    GlyphQuad quad;
    quad.atlas = &atlas;
    quad.codepoint = glyph.atlas_codepoint;

    const float m00 = glyph.scale * glyph.basis_ux;
    const float m01 = glyph.scale * glyph.basis_vx;
    const float m10 = glyph.scale * glyph.basis_uy;
    const float m11 = glyph.scale * glyph.basis_vy;
    const float w = (float)info.atlas_w;
    const float h = (float)info.atlas_h;

    quad.corners[0] = Vec2f{glyph.origin_x, glyph.origin_y};
    quad.corners[1] = Vec2f{glyph.origin_x + m00 * w, glyph.origin_y + m10 * w};
    quad.corners[2] = Vec2f{glyph.origin_x + m00 * w + m01 * h, glyph.origin_y + m10 * w + m11 * h};
    quad.corners[3] = Vec2f{glyph.origin_x + m01 * h, glyph.origin_y + m11 * h};
    return quad;
}

static bool QuadsOverlap(const GlyphQuad& a,
                         const GlyphQuad& b,
                         float margin) noexcept
{
    const std::array<Vec2f, 4> a_corners = ExpandCorners(a.corners, margin);
    const std::array<Vec2f, 4> b_corners = ExpandCorners(b.corners, margin);
    const Vec2f axes[4] = {
        Normalize(a_corners[1] - a_corners[0]),
        Normalize(a_corners[3] - a_corners[0]),
        Normalize(b_corners[1] - b_corners[0]),
        Normalize(b_corners[3] - b_corners[0]),
    };

    for (const Vec2f& axis : axes)
    {
        Vec2f normal{-axis.y, axis.x};
        normal = Normalize(normal);
        if (!OverlapsOnAxis(a_corners, b_corners, normal))
            return false;
    }
    return true;
}

static bool OverlapsPlacedQuads(const std::vector<GlyphQuad>& candidate,
                                const std::vector<PlacedFootprint>& placed,
                                float margin) noexcept
{
    for (const GlyphQuad& quad : candidate)
    {
        for (const PlacedFootprint& footprint : placed)
        {
            for (const GlyphQuad& other : footprint.quads)
            {
                if (QuadsOverlap(quad, other, margin))
                    return true;
            }
        }
    }
    return false;
}

static std::vector<float> BuildOffsetFractions(int steps)
{
    std::vector<float> out;
    out.push_back(0.0f);
    for (int i = 1; i <= std::max(0, steps); ++i)
    {
        const float t = (float)i / (float)std::max(1, steps);
        out.push_back(t);
        out.push_back(-t);
    }
    return out;
}

static std::size_t CountOutsideSamples(const std::vector<Vec2f>& polygon,
                                       const std::vector<GlyphQuad>& quads) noexcept
{
    if (polygon.size() < 3)
        return 0;

    std::size_t outside = 0;
    for (const GlyphQuad& quad : quads)
    {
        const Vec2f center = quad.Center();
        if (!PointInPolygon(polygon, center))
            ++outside;
        for (const Vec2f& corner : quad.corners)
        {
            if (!PointInPolygon(polygon, corner))
                ++outside;
        }
    }
    return outside;
}

static float EffectiveTangentOverflow(const OrientedBox& box,
                                      const RenderPlanOptions& options) noexcept
{
    return std::min(options.max_tangent_overflow_frac * box.width,
                    options.max_tangent_overflow_px);
}

static float EffectiveNormalOverflow(const OrientedBox& box,
                                     const RenderPlanOptions& options) noexcept
{
    return std::min(options.max_normal_overflow_frac * box.height,
                    options.max_normal_overflow_px);
}

static float EffectiveTangentOverflow(const CurveFitResult& fit,
                                      const RenderPlanOptions& options) noexcept
{
    return std::min(options.max_tangent_overflow_frac * fit.curve_length,
                    options.max_tangent_overflow_px);
}

static float EffectiveNormalOverflow(const CurvedTextPath& curve,
                                     const RenderPlanOptions& options) noexcept
{
    return std::min(options.max_normal_overflow_frac * curve.band_height,
                    options.max_normal_overflow_px);
}

static bool FitsWithinExtendedBox(const OrientedBox& box,
                                  const TextMeasurement& measurement,
                                  const TextFitOptions& fit_options,
                                  const RenderPlanOptions& options,
                                  const TextFitResult& fit) noexcept
{
    const LocalBounds bounds = ComputeBounds(fit, measurement);
    const float tangent_overflow = options.allow_overflow
        ? EffectiveTangentOverflow(box, options)
        : 0.0f;
    const float normal_overflow = options.allow_overflow
        ? EffectiveNormalOverflow(box, options)
        : 0.0f;

    const float left = -0.5f * box.width + fit_options.padding_x - tangent_overflow;
    const float right = 0.5f * box.width - fit_options.padding_x + tangent_overflow;
    const float top = -0.5f * box.height + fit_options.padding_y - normal_overflow;
    const float bottom = 0.5f * box.height - fit_options.padding_y + normal_overflow;
    return bounds.min_x >= left
        && bounds.max_x <= right
        && bounds.min_y >= top
        && bounds.max_y <= bottom;
}

static bool OverflowedBox(const OrientedBox& box,
                          const TextMeasurement& measurement,
                          const TextFitOptions& fit_options,
                          const TextFitResult& fit) noexcept
{
    const LocalBounds bounds = ComputeBounds(fit, measurement);
    const float left = -0.5f * box.width + fit_options.padding_x;
    const float right = 0.5f * box.width - fit_options.padding_x;
    const float top = -0.5f * box.height + fit_options.padding_y;
    const float bottom = 0.5f * box.height - fit_options.padding_y;
    return bounds.min_x < left || bounds.max_x > right
        || bounds.min_y < top || bounds.max_y > bottom;
}

static bool FitsWithinExtendedCurve(const CurvedTextPath& curve,
                                    const TextMeasurement& measurement,
                                    const TextFitOptions& fit_options,
                                    const RenderPlanOptions& options,
                                    const CurveFitResult& fit) noexcept
{
    const float tangent_overflow = options.allow_overflow
        ? EffectiveTangentOverflow(fit, options)
        : 0.0f;
    const float normal_overflow = options.allow_overflow
        ? EffectiveNormalOverflow(curve, options)
        : 0.0f;
    const float inner_start = fit_options.padding_x - tangent_overflow;
    const float inner_end = fit.curve_length - fit_options.padding_x + tangent_overflow;
    const float top = -0.5f * curve.band_height + fit_options.padding_y - normal_overflow;
    const float bottom = 0.5f * curve.band_height - fit_options.padding_y + normal_overflow;
    const float min_y = fit.normal_offset - measurement.ascent * fit.scale;
    const float max_y = fit.normal_offset + measurement.descent * fit.scale;
    return fit.start_offset >= inner_start
        && fit.start_offset + fit.text_width <= inner_end
        && min_y >= top
        && max_y <= bottom;
}

static bool OverflowedCurve(const CurvedTextPath& curve,
                            const TextMeasurement& measurement,
                            const TextFitOptions& fit_options,
                            const CurveFitResult& fit) noexcept
{
    const float min_y = fit.normal_offset - measurement.ascent * fit.scale;
    const float max_y = fit.normal_offset + measurement.descent * fit.scale;
    const float inner_start = fit_options.padding_x;
    const float inner_end = fit.curve_length - fit_options.padding_x;
    const float top = -0.5f * curve.band_height + fit_options.padding_y;
    const float bottom = 0.5f * curve.band_height - fit_options.padding_y;
    return fit.start_offset < inner_start
        || fit.start_offset + fit.text_width > inner_end
        || min_y < top
        || max_y > bottom;
}

static double SampleBrightnessFromBox(const ImageRgba8& image,
                                      const OrientedBox& box,
                                      std::size_t& samples)
{
    samples = 0;
    if (image.Empty() || !box.HasArea())
        return kBrightnessThreshold;

    const std::array<Vec2f, 4> corners = BoxCorners(box);
    float min_x = corners[0].x;
    float min_y = corners[0].y;
    float max_x = corners[0].x;
    float max_y = corners[0].y;
    for (std::size_t i = 1; i < corners.size(); ++i)
    {
        min_x = std::min(min_x, corners[i].x);
        min_y = std::min(min_y, corners[i].y);
        max_x = std::max(max_x, corners[i].x);
        max_y = std::max(max_y, corners[i].y);
    }

    const int x0 = std::max(0, (int)std::floor(min_x));
    const int y0 = std::max(0, (int)std::floor(min_y));
    const int x1 = std::min((int)image.width - 1, (int)std::ceil(max_x));
    const int y1 = std::min((int)image.height - 1, (int)std::ceil(max_y));

    double brightness_sum = 0.0;
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            const float dx = (float)x + 0.5f - box.cx;
            const float dy = (float)y + 0.5f - box.cy;
            const float local_x = dx * box.ux + dy * box.uy;
            const float local_y = dx * box.vx + dy * box.vy;
            if (std::fabs(local_x) > 0.5f * box.width
                || std::fabs(local_y) > 0.5f * box.height)
            {
                continue;
            }

            const uint8_t* pixel = image.PixelPtr((uint32_t)x, (uint32_t)y);
            brightness_sum += ((double)pixel[0] + (double)pixel[1] + (double)pixel[2]) / 3.0;
            ++samples;
        }
    }

    return (samples > 0) ? (brightness_sum / (double)samples) : kBrightnessThreshold;
}

static double SampleBrightnessFromBoundingBox(const ImageRgba8& image,
                                              const std::vector<Vec2f>& polygon,
                                              std::size_t& samples)
{
    samples = 0;
    if (image.Empty() || polygon.size() < 3)
        return kBrightnessThreshold;

    float min_x = polygon.front().x;
    float min_y = polygon.front().y;
    float max_x = polygon.front().x;
    float max_y = polygon.front().y;
    for (const Vec2f& point : polygon)
    {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }

    const int x0 = std::max(0, (int)std::floor(min_x));
    const int y0 = std::max(0, (int)std::floor(min_y));
    const int x1 = std::min((int)image.width - 1, (int)std::ceil(max_x));
    const int y1 = std::min((int)image.height - 1, (int)std::ceil(max_y));
    if (x0 > x1 || y0 > y1)
        return kBrightnessThreshold;

    double brightness_sum = 0.0;
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            const uint8_t* pixel = image.PixelPtr((uint32_t)x, (uint32_t)y);
            brightness_sum += ((double)pixel[0] + (double)pixel[1] + (double)pixel[2]) / 3.0;
            ++samples;
        }
    }

    return (samples > 0) ? (brightness_sum / (double)samples) : kBrightnessThreshold;
}

static double SampleBrightnessFromLumaBoundingBox(uint32_t width,
                                                  uint32_t height,
                                                  const std::vector<uint8_t>& image_luma,
                                                  const std::vector<Vec2f>& polygon,
                                                  std::size_t& samples)
{
    samples = 0;
    if (width == 0 || height == 0
        || image_luma.size() != (std::size_t)width * height
        || polygon.size() < 3)
    {
        return kBrightnessThreshold;
    }

    float min_x = polygon.front().x;
    float min_y = polygon.front().y;
    float max_x = polygon.front().x;
    float max_y = polygon.front().y;
    for (const Vec2f& point : polygon)
    {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }

    const int x0 = std::max(0, (int)std::floor(min_x));
    const int y0 = std::max(0, (int)std::floor(min_y));
    const int x1 = std::min((int)width - 1, (int)std::ceil(max_x));
    const int y1 = std::min((int)height - 1, (int)std::ceil(max_y));
    if (x0 > x1 || y0 > y1)
        return kBrightnessThreshold;

    double brightness_sum = 0.0;
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            brightness_sum += image_luma[(std::size_t)y * width + (std::size_t)x];
            ++samples;
        }
    }

    return (samples > 0) ? (brightness_sum / (double)samples) : kBrightnessThreshold;
}

static double SampleBrightnessFromPolygon(const ImageRgba8& image,
                                          const std::vector<Vec2f>& polygon,
                                          std::size_t& samples)
{
    samples = 0;
    if (image.Empty() || polygon.size() < 3)
        return kBrightnessThreshold;

    float min_x = polygon.front().x;
    float min_y = polygon.front().y;
    float max_x = polygon.front().x;
    float max_y = polygon.front().y;
    for (const Vec2f& point : polygon)
    {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }

    const int x0 = std::max(0, (int)std::floor(min_x));
    const int y0 = std::max(0, (int)std::floor(min_y));
    const int x1 = std::min((int)image.width - 1, (int)std::ceil(max_x));
    const int y1 = std::min((int)image.height - 1, (int)std::ceil(max_y));

    double brightness_sum = 0.0;
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            const Vec2f p{(float)x + 0.5f, (float)y + 0.5f};
            if (!PointInPolygon(polygon, p))
                continue;

            const uint8_t* pixel = image.PixelPtr((uint32_t)x, (uint32_t)y);
            brightness_sum += ((double)pixel[0] + (double)pixel[1] + (double)pixel[2]) / 3.0;
            ++samples;
        }
    }

    return (samples > 0) ? (brightness_sum / (double)samples) : kBrightnessThreshold;
}

static double SampleBrightnessFromCurveStrip(const ImageRgba8& image,
                                             const CurvedTextPath& curve,
                                             std::size_t& samples)
{
    samples = 0;
    if (image.Empty() || !curve.IsValid())
        return kBrightnessThreshold;

    const std::vector<CurveSample> polyline = BuildCurvePolyline(curve);
    if (polyline.size() < 2)
        return kBrightnessThreshold;

    float min_x = polyline.front().point.x;
    float min_y = polyline.front().point.y;
    float max_x = polyline.front().point.x;
    float max_y = polyline.front().point.y;
    for (const CurveSample& sample : polyline)
    {
        min_x = std::min(min_x, sample.point.x);
        min_y = std::min(min_y, sample.point.y);
        max_x = std::max(max_x, sample.point.x);
        max_y = std::max(max_y, sample.point.y);
    }

    const float radius = 0.5f * curve.band_height;
    const int x0 = std::max(0, (int)std::floor(min_x - radius));
    const int y0 = std::max(0, (int)std::floor(min_y - radius));
    const int x1 = std::min((int)image.width - 1, (int)std::ceil(max_x + radius));
    const int y1 = std::min((int)image.height - 1, (int)std::ceil(max_y + radius));

    double brightness_sum = 0.0;
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            const Vec2f p{(float)x + 0.5f, (float)y + 0.5f};
            float best = std::numeric_limits<float>::infinity();
            for (std::size_t i = 1; i < polyline.size(); ++i)
            {
                best = std::min(best, DistancePointToSegment(
                    p,
                    polyline[i - 1].point,
                    polyline[i].point));
            }
            if (best > radius)
                continue;

            const uint8_t* pixel = image.PixelPtr((uint32_t)x, (uint32_t)y);
            brightness_sum += ((double)pixel[0] + (double)pixel[1] + (double)pixel[2]) / 3.0;
            ++samples;
        }
    }

    return (samples > 0) ? (brightness_sum / (double)samples) : kBrightnessThreshold;
}

static Vec2f PrincipalAxisFromPolygon(const std::vector<Vec2f>& polygon) noexcept
{
    Vec2f centroid{};
    for (const Vec2f& point : polygon)
        centroid = centroid + point;
    centroid = centroid / (float)polygon.size();

    float xx = 0.0f;
    float xy = 0.0f;
    float yy = 0.0f;
    for (const Vec2f& point : polygon)
    {
        const Vec2f d = point - centroid;
        xx += d.x * d.x;
        xy += d.x * d.y;
        yy += d.y * d.y;
    }

    if (std::fabs(xy) <= 1e-6f)
        return (xx >= yy) ? Vec2f{1.0f, 0.0f} : Vec2f{0.0f, 1.0f};

    const float theta = 0.5f * std::atan2(2.0f * xy, xx - yy);
    return Normalize(Vec2f{std::cos(theta), std::sin(theta)});
}

static OrientedBox MakeStraightBoxFromPolygon(const std::vector<Vec2f>& polygon) noexcept
{
    const Vec2f axis = PrincipalAxisFromPolygon(polygon);
    Vec2f normal{-axis.y, axis.x};
    Vec2f use_axis = axis;
    if (normal.y < 0.0f || (std::fabs(normal.y) <= 1e-6f && normal.x < 0.0f))
    {
        use_axis = use_axis * -1.0f;
        normal = normal * -1.0f;
    }

    float min_u = Dot(polygon.front(), use_axis);
    float max_u = min_u;
    float min_v = Dot(polygon.front(), normal);
    float max_v = min_v;
    for (const Vec2f& point : polygon)
    {
        const float u = Dot(point, use_axis);
        const float v = Dot(point, normal);
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    const float mid_u = 0.5f * (min_u + max_u);
    const float mid_v = 0.5f * (min_v + max_v);
    const Vec2f center = use_axis * mid_u + normal * mid_v;

    OrientedBox box;
    box.cx = center.x;
    box.cy = center.y;
    box.ux = use_axis.x;
    box.uy = use_axis.y;
    box.vx = normal.x;
    box.vy = normal.y;
    box.width = std::max(0.0f, max_u - min_u);
    box.height = std::max(0.0f, max_v - min_v);
    return box;
}

static std::vector<Vec2f> ExtractChain(const std::vector<Vec2f>& polygon,
                                       std::size_t start,
                                       std::size_t end,
                                       int step)
{
    std::vector<Vec2f> chain;
    if (polygon.empty())
        return chain;

    const std::size_t n = polygon.size();
    chain.push_back(polygon[start]);
    std::size_t index = start;

    for (std::size_t iter = 0; iter < n; ++iter)
    {
        if (index == end)
            break;
        index = (std::size_t)(((int)index + step + (int)n) % (int)n);
        chain.push_back(polygon[index]);
    }

    if (chain.empty() || chain.back().x != polygon[end].x || chain.back().y != polygon[end].y)
        chain.push_back(polygon[end]);
    return chain;
}

static std::vector<Vec2f> ResampleChain(const std::vector<Vec2f>& chain,
                                        int sample_count)
{
    std::vector<Vec2f> out;
    if (chain.empty() || sample_count <= 0)
        return out;

    if (chain.size() == 1 || sample_count == 1)
        return std::vector<Vec2f>((std::size_t)sample_count, chain.front());

    std::vector<float> cumulative(chain.size(), 0.0f);
    for (std::size_t i = 1; i < chain.size(); ++i)
        cumulative[i] = cumulative[i - 1] + Length(chain[i] - chain[i - 1]);

    const float total = cumulative.back();
    if (total <= 1e-6f)
        return std::vector<Vec2f>((std::size_t)sample_count, chain.front());

    out.reserve((std::size_t)sample_count);
    for (int i = 0; i < sample_count; ++i)
    {
        const float target = total * (float)i / (float)(sample_count - 1);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), target);
        if (it == cumulative.begin())
        {
            out.push_back(chain.front());
            continue;
        }
        if (it == cumulative.end())
        {
            out.push_back(chain.back());
            continue;
        }

        const std::size_t index = (std::size_t)(it - cumulative.begin());
        const float a = cumulative[index - 1];
        const float b = cumulative[index];
        const float t = (target - a) / std::max(b - a, 1e-6f);
        out.push_back(chain[index - 1] + (chain[index] - chain[index - 1]) * t);
    }

    return out;
}

static float DistancePointToLine(const Vec2f& p,
                                 const Vec2f& a,
                                 const Vec2f& b) noexcept
{
    const Vec2f ab = b - a;
    const float len = Length(ab);
    if (len <= 1e-6f)
        return Length(p - a);
    return std::fabs(ab.y * p.x - ab.x * p.y + b.x * a.y - b.y * a.x) / len;
}

static float Quantile(std::vector<float> values, float q)
{
    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());
    const float clamped = Clamp(q, 0.0f, 1.0f);
    const std::size_t index = (std::size_t)std::floor(clamped * (float)(values.size() - 1));
    return values[index];
}

static Vec2f ComputeCentripetalTangent(const Vec2f& p0,
                                       const Vec2f& p1,
                                       const Vec2f& p2,
                                       const Vec2f& p3) noexcept
{
    const float t0 = 0.0f;
    const float t1 = t0 + std::sqrt(std::max(Length(p1 - p0), 1e-4f));
    const float t2 = t1 + std::sqrt(std::max(Length(p2 - p1), 1e-4f));
    (void)p3;

    Vec2f tangent =
        (p2 - p1) / std::max(t2 - t1, 1e-4f)
        - (p2 - p0) / std::max(t2 - t0, 1e-4f)
        + (p1 - p0) / std::max(t1 - t0, 1e-4f);
    tangent = tangent * std::max(t2 - t1, 1e-4f);
    return tangent;
}

static CurvedTextPath MakeCurveFromCenterline(const std::vector<Vec2f>& points,
                                              float band_height)
{
    CurvedTextPath curve;
    curve.band_height = band_height;
    curve.normal_side = CurveNormalSide::Right;

    if (points.size() < 2 || !(band_height > 0.0f))
        return curve;

    curve.segments.reserve(points.size() - 1);
    if (points.size() == 2)
    {
        CubicBezierSegment segment;
        segment.p0 = points[0];
        segment.p1 = points[0] + (points[1] - points[0]) / 3.0f;
        segment.p2 = points[0] + (points[1] - points[0]) * (2.0f / 3.0f);
        segment.p3 = points[1];
        curve.segments.push_back(segment);
        return curve;
    }

    for (std::size_t i = 0; i + 1 < points.size(); ++i)
    {
        const Vec2f& p1 = points[i];
        const Vec2f& p2 = points[i + 1];
        const Vec2f& p0 = (i > 0) ? points[i - 1] : p1;
        const Vec2f& p3 = (i + 2 < points.size()) ? points[i + 2] : p2;

        const Vec2f m1 = ComputeCentripetalTangent(p0, p1, p2, p3);
        const Vec2f m2 = ComputeCentripetalTangent(p1, p2, p3, p3);

        CubicBezierSegment segment;
        segment.p0 = p1;
        segment.p1 = p1 + m1 / 3.0f;
        segment.p2 = p2 - m2 / 3.0f;
        segment.p3 = p2;
        curve.segments.push_back(segment);
    }

    return curve;
}

static ResolvedGeometry ResolveGeometry(const TextRegion& region)
{
    ResolvedGeometry geometry;

    if (region.has_curve && region.curve.IsValid())
    {
        geometry.kind = GeometryKind::Curved;
        geometry.curve = region.curve;
        return geometry;
    }

    if (region.has_polygon && region.polygon.size() >= 3)
    {
        geometry.box = MakeStraightBoxFromPolygon(region.polygon);

        const Vec2f axis{geometry.box.ux, geometry.box.uy};
        std::size_t min_index = 0;
        std::size_t max_index = 0;
        float min_proj = Dot(region.polygon.front(), axis);
        float max_proj = min_proj;
        for (std::size_t i = 1; i < region.polygon.size(); ++i)
        {
            const float proj = Dot(region.polygon[i], axis);
            if (proj < min_proj)
            {
                min_proj = proj;
                min_index = i;
            }
            if (proj > max_proj)
            {
                max_proj = proj;
                max_index = i;
            }
        }

        if (min_index == max_index)
            return geometry;

        const std::vector<Vec2f> chain_a =
            ExtractChain(region.polygon, min_index, max_index, +1);
        const std::vector<Vec2f> chain_b =
            ExtractChain(region.polygon, min_index, max_index, -1);
        if (chain_a.size() < 2 || chain_b.size() < 2)
            return geometry;

        const int sample_count = std::max(8, (int)std::ceil(geometry.box.width / 12.0f));
        const std::vector<Vec2f> a_samples = ResampleChain(chain_a, sample_count);
        const std::vector<Vec2f> b_samples = ResampleChain(chain_b, sample_count);
        if (a_samples.size() != b_samples.size() || a_samples.size() < 2)
            return geometry;

        std::vector<Vec2f> centerline;
        std::vector<float> band_heights;
        centerline.reserve(a_samples.size());
        band_heights.reserve(a_samples.size());

        for (std::size_t i = 0; i < a_samples.size(); ++i)
        {
            centerline.push_back((a_samples[i] + b_samples[i]) * 0.5f);
            band_heights.push_back(Length(a_samples[i] - b_samples[i]));
        }

        const Vec2f chord_a = centerline.front();
        const Vec2f chord_b = centerline.back();
        float max_deviation = 0.0f;
        for (const Vec2f& point : centerline)
            max_deviation = std::max(max_deviation, DistancePointToLine(point, chord_a, chord_b));

        if (max_deviation < kStraightCurveDeviationPx)
            return geometry;

        const float band_height = Quantile(band_heights, 0.2f);
        CurvedTextPath curve = MakeCurveFromCenterline(centerline, band_height);
        if (curve.IsValid())
        {
            geometry.kind = GeometryKind::Curved;
            geometry.curve = std::move(curve);
        }
    }
    else
    {
        geometry.box = region.box;
    }

    if (geometry.kind == GeometryKind::Straight && !geometry.box.HasArea())
        geometry.box = region.box;
    return geometry;
}

static PlannedRegion MakeProvisionalRegion(const FontDatabase& db,
                                           const TextRegion& region,
                                           std::size_t original_index,
                                           const ImageRgba8& image,
                                           const RenderPlanOptions& options)
{
    PlannedRegion out;
    out.region = &region;
    out.original_index = original_index;
    out.geometry = ResolveGeometry(region);

    TextFitOptions fit_options = MakeFitOptions(options);
    if (out.IsCurved())
    {
        out.curve_fit = options.allow_overflow
            ? MakeReadableCurveFit(db, region.text, out.geometry.curve, fit_options)
            : FitTextToCurve(db, region.text, out.geometry.curve, fit_options);
        if (!out.curve_fit.IsValid() || out.curve_fit.atlas == nullptr)
            return out;
        out.measurement = MeasureTextLine(*out.curve_fit.atlas, region.text);
    }
    else
    {
        out.fit = options.allow_overflow
            ? MakeReadableFit(db, region.text, out.geometry.box, fit_options)
            : FitTextToBox(db, region.text, out.geometry.box, fit_options);
        if (!out.fit.IsValid() || out.fit.atlas == nullptr)
            return out;
        out.measurement = MeasureTextLine(*out.fit.atlas, region.text);
    }

    if (out.measurement.glyph_count == 0)
        return out;

    const std::size_t original_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.original_text));
    const std::size_t translated_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.text));
    if (translated_len > original_len)
    {
        const float factor =
            std::pow((float)original_len / (float)translated_len, 0.6f);
        if (out.IsCurved())
            ScaleCurveFit(out.geometry.curve, out.measurement, options.fit, factor, out.curve_fit);
        else
            ScaleFit(out.geometry.box, out.measurement, options.fit, factor, out.fit);
    }

    if (region.has_explicit_rgba)
    {
        out.resolved_rgba = region.rgba;
    }
    else
    {
        std::size_t samples = 0;
        const double brightness = region.has_polygon
            ? SampleBrightnessFromPolygon(image, region.polygon, samples)
            : (out.IsCurved()
                ? SampleBrightnessFromCurveStrip(image, out.geometry.curve, samples)
                : SampleBrightnessFromBox(image, out.geometry.box, samples));
        out.resolved_rgba = (brightness < kBrightnessThreshold)
            ? 0xFFFFFFFFu
            : 0x000000FFu;
    }

    out.valid = out.IsCurved() ? out.curve_fit.IsValid() : out.fit.IsValid();
    return out;
}

static RegionFeatures ComputeRegionFeatures(const TextRegion& region)
{
    RegionFeatures features;

    const std::size_t original_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.original_text));
    const std::size_t translated_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.text));
    features.translated_ratio = (float)translated_len / (float)original_len;

    if (!region.has_polygon || region.polygon.size() < 3 || region.has_curve)
        return features;

    features.straight_box = MakeStraightBoxFromPolygon(region.polygon);
    if (!features.straight_box.HasArea() || !features.straight_box.HasFiniteBasis())
        return features;

    if (region.polygon.size() > 8u || features.translated_ratio > 1.8f)
        return features;

    const float polygon_area = PolygonArea(region.polygon);
    const float box_area = features.straight_box.width * features.straight_box.height;
    if (!(polygon_area > 0.0f) || !(box_area > 0.0f))
        return features;

    const float area_ratio = polygon_area / box_area;
    if (area_ratio < 0.72f)
        return features;

    features.initial_kind = RegionPlannerKind::FastStraight;
    return features;
}

static void AppendPlacement(RenderPlan& plan,
                            std::unordered_map<uint32_t, std::size_t>& batch_by_size,
                            const AcceptedPlacement& placement,
                            std::vector<OrientedBox>& placed_boxes,
                            std::vector<PlacedFootprint>& placed_footprints)
{
    auto [it, inserted] = batch_by_size.emplace(placement.render_size, plan.batches.size());
    if (inserted)
    {
        RenderBatch batch;
        batch.atlas_render_size = placement.render_size;
        plan.batches.push_back(std::move(batch));
    }

    RenderBatch& batch = plan.batches[it->second];
    batch.glyphs.reserve(batch.glyphs.size() + placement.commands.size());
    batch.glyphs.insert(batch.glyphs.end(), placement.commands.begin(), placement.commands.end());

    if (!placement.is_curved)
        placed_boxes.push_back(placement.placed_box);
    placed_footprints.push_back(placement.footprint);

    ++plan.fitted_regions;
    if (placement.overflowed)
        ++plan.overflowed_regions;
    plan.total_glyphs += placement.commands.size();
}

static float ComputeBaselineFraction(const PlannedRegion& region) noexcept
{
    return (region.fit.baseline_y + 0.5f * region.geometry.box.height)
        / std::max(region.geometry.box.height, 1e-6f);
}

static float Median(std::vector<float> values)
{
    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u)
        return values[mid];
    return 0.5f * (values[mid - 1] + values[mid]);
}

static bool AreLineCompatible(const PlannedRegion& a,
                              const PlannedRegion& b,
                              const RenderPlanOptions& options) noexcept
{
    if (!a.valid || !b.valid || a.IsCurved() || b.IsCurved())
        return false;

    const float dot_u = Clamp(std::fabs(a.geometry.box.ux * b.geometry.box.ux
                                      + a.geometry.box.uy * b.geometry.box.uy),
                              -1.0f,
                              1.0f);
    const float angle_deg = std::acos(dot_u) * (180.0f / 3.14159265358979323846f);
    if (angle_deg > kLineAngleThresholdDeg)
        return false;

    Vec2f avg_normal{
        a.geometry.box.vx + b.geometry.box.vx,
        a.geometry.box.vy + b.geometry.box.vy,
    };
    avg_normal = Normalize(avg_normal);

    const Vec2f delta{
        b.geometry.box.cx - a.geometry.box.cx,
        b.geometry.box.cy - a.geometry.box.cy,
    };
    const float normal_distance = std::fabs(Dot(delta, avg_normal));
    const float threshold = std::max(
        options.line_y_tolerance * std::min(a.geometry.box.height, b.geometry.box.height),
        options.min_line_group_px);
    return normal_distance < threshold;
}

static void ApplyLineAlignment(std::vector<PlannedRegion>& planned,
                               const RenderPlanOptions& options)
{
    std::vector<bool> grouped(planned.size(), false);

    for (std::size_t i = 0; i < planned.size(); ++i)
    {
        if (grouped[i] || !planned[i].valid || planned[i].IsCurved())
            continue;

        std::vector<std::size_t> group = {i};
        grouped[i] = true;

        for (std::size_t j = i + 1; j < planned.size(); ++j)
        {
            if (grouped[j] || !planned[j].valid || planned[j].IsCurved())
                continue;
            if (!AreLineCompatible(planned[i], planned[j], options))
                continue;

            grouped[j] = true;
            group.push_back(j);
        }

        if (group.size() < 2)
            continue;

        std::vector<float> line_heights;
        std::vector<float> baseline_fractions;
        line_heights.reserve(group.size());
        baseline_fractions.reserve(group.size());

        for (std::size_t index : group)
        {
            line_heights.push_back(planned[index].fit.line_height);
            baseline_fractions.push_back(ComputeBaselineFraction(planned[index]));
        }

        const float target_line_height = Median(line_heights);
        const float target_baseline_fraction = Median(baseline_fractions);

        for (std::size_t index : group)
        {
            PlannedRegion& entry = planned[index];
            if (!(entry.fit.line_height > 0.0f))
                continue;

            const float scale_factor = std::min(1.0f, target_line_height / entry.fit.line_height);
            ScaleFit(entry.geometry.box, entry.measurement, options.fit, scale_factor, entry.fit);
            AlignBaselineFraction(entry.geometry.box,
                                  entry.measurement,
                                  options.fit,
                                  target_baseline_fraction,
                                  entry.fit);
        }
    }
}

static bool IsClustered(const PlannedRegion& region) noexcept
{
    return region.region != nullptr
        && region.region->cluster_id != TextRegion::kUnclustered;
}

static float PlannedLineHeight(const PlannedRegion& region) noexcept
{
    if (!region.valid)
        return 0.0f;
    return region.IsCurved() ? region.curve_fit.line_height : region.fit.line_height;
}

static void ScalePlannedRegion(PlannedRegion& region,
                               const RenderPlanOptions& options,
                               float scale_factor) noexcept
{
    if (region.IsCurved())
        ScaleCurveFit(region.geometry.curve,
                      region.measurement,
                      options.fit,
                      scale_factor,
                      region.curve_fit);
    else
        ScaleFit(region.geometry.box,
                 region.measurement,
                 options.fit,
                 scale_factor,
                 region.fit);
}

static void ScalePlannedRegionToLineHeight(PlannedRegion& region,
                                           const RenderPlanOptions& options,
                                           float target_line_height) noexcept
{
    const float current = PlannedLineHeight(region);
    if (!(current > 0.0f) || !(target_line_height > 0.0f))
        return;
    ScalePlannedRegion(region, options, std::min(1.0f, target_line_height / current));
}

static void ApplyClusterInitialSizing(std::vector<PlannedRegion>& planned,
                                      const RenderPlanOptions& options)
{
    for (std::size_t i = 0; i < planned.size(); ++i)
    {
        if (!planned[i].valid || !IsClustered(planned[i]))
            continue;

        const std::size_t cluster_id = planned[i].region->cluster_id;
        std::vector<std::size_t> group;
        float target_line_height = std::numeric_limits<float>::infinity();

        for (std::size_t j = i; j < planned.size(); ++j)
        {
            if (!planned[j].valid || !IsClustered(planned[j])
                || planned[j].region->cluster_id != cluster_id)
            {
                continue;
            }

            group.push_back(j);
            target_line_height = std::min(target_line_height, PlannedLineHeight(planned[j]));
        }

        if (group.size() < 2 || !std::isfinite(target_line_height))
            continue;

        for (std::size_t index : group)
            ScalePlannedRegionToLineHeight(planned[index], options, target_line_height);
    }
}

static Vec2f GeometryAnchor(const PlannedRegion& region) noexcept
{
    if (!region.IsCurved())
        return Vec2f{region.geometry.box.cx, region.geometry.box.cy};

    if (!region.region->polygon.empty())
    {
        Vec2f center{};
        for (const Vec2f& point : region.region->polygon)
            center = center + point;
        return center / (float)region.region->polygon.size();
    }

    const std::vector<CurveSample> polyline = BuildCurvePolyline(region.geometry.curve);
    if (polyline.empty())
        return Vec2f{};
    return polyline[polyline.size() / 2].point;
}

static float MobilityScore(const PlannedRegion& region) noexcept
{
    const std::size_t translated_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.region->text));
    const float area = region.IsCurved()
        ? (region.curve_fit.curve_length * region.geometry.curve.band_height)
        : (region.geometry.box.width * region.geometry.box.height);
    return area / (float)translated_len;
}

static bool PlannedRegionLess(const PlannedRegion& a,
                              const PlannedRegion& b) noexcept
{
    if (a.IsCurved() != b.IsCurved())
        return a.IsCurved() && !b.IsCurved();

    const float a_mobility = MobilityScore(a);
    const float b_mobility = MobilityScore(b);
    if (std::fabs(a_mobility - b_mobility) > 1e-6f)
        return a_mobility < b_mobility;

    const Vec2f a_anchor = GeometryAnchor(a);
    const Vec2f b_anchor = GeometryAnchor(b);
    if (std::fabs(a_anchor.y - b_anchor.y) > 1e-6f)
        return a_anchor.y < b_anchor.y;
    if (std::fabs(a_anchor.x - b_anchor.x) > 1e-6f)
        return a_anchor.x < b_anchor.x;
    if (a.region->text != b.region->text)
        return a.region->text < b.region->text;
    if (a.region->original_text != b.region->original_text)
        return a.region->original_text < b.region->original_text;
    return a.original_index < b.original_index;
}

static bool BetterCandidateAtSameScale(const PlacementCandidate& a,
                                       const PlacementCandidate& b) noexcept
{
    if (!b.valid)
        return a.valid;
    if (!a.valid)
        return false;
    if (a.outside_score != b.outside_score)
        return a.outside_score < b.outside_score;
    if (std::fabs(a.displacement2 - b.displacement2) > 1e-6f)
        return a.displacement2 < b.displacement2;
    if (a.overflowed != b.overflowed)
        return !a.overflowed;
    if (std::fabs(a.tangent_delta - b.tangent_delta) > 1e-6f)
        return a.tangent_delta < b.tangent_delta;
    return a.normal_delta < b.normal_delta;
}

static std::vector<RenderGlyph> BuildStraightGlyphs(const PlannedRegion& region,
                                                    const TextFitResult& fit)
{
    std::vector<RenderGlyph> glyphs;
    if (!fit.IsValid() || fit.atlas == nullptr)
        return glyphs;

    const std::vector<GlyphPlacement> placements =
        LayoutTextLine(*fit.atlas, region.region->text, fit);
    glyphs.reserve(placements.size());
    for (const GlyphPlacement& placement : placements)
        glyphs.push_back(MakeRenderGlyph(placement, region.geometry.box, region.resolved_rgba));
    return glyphs;
}

static std::vector<GlyphQuad> BuildGlyphQuads(const AtlasEntry& atlas,
                                              const std::vector<RenderGlyph>& glyphs)
{
    std::vector<GlyphQuad> quads;
    quads.reserve(glyphs.size());
    for (const RenderGlyph& glyph : glyphs)
    {
        const GlyphInfo& info = atlas.GetGlyph(glyph.atlas_codepoint);
        if (info.atlas_w == 0 || info.atlas_h == 0 || glyph.scale <= 0.0f)
            continue;
        quads.push_back(MakeGlyphQuad(atlas, glyph));
    }
    return quads;
}

static AcceptedPlacement TryAcceptFastStraightRegion(const FontDatabase& db,
                                                     const TextRegion& region,
                                                     std::size_t original_index,
                                                     const RegionFeatures& features,
                                                     const RenderPlanOptions& options,
                                                     const std::vector<OrientedBox>& placed_boxes,
                                                     const std::vector<PlacedFootprint>& placed_footprints,
                                                     double brightness)
{
    (void)placed_footprints;

    AcceptedPlacement accepted;
    PlannedRegion planned;
    planned.region = &region;
    planned.original_index = original_index;
    planned.geometry.kind = GeometryKind::Straight;
    planned.geometry.box = features.straight_box;

    planned.fit = FitTextToBox(db, region.text, planned.geometry.box, MakeFitOptions(options));
    if (!planned.fit.IsValid() || planned.fit.atlas == nullptr)
        return accepted;

    planned.measurement = MeasureTextLine(*planned.fit.atlas, region.text);
    if (planned.measurement.glyph_count == 0)
        return accepted;

    const std::size_t original_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.original_text));
    const std::size_t translated_len =
        std::max<std::size_t>(1, CountTrimmedCodepoints(region.text));
    if (translated_len > original_len)
    {
        const float factor =
            std::pow((float)original_len / (float)translated_len, 0.6f);
        ScaleFit(planned.geometry.box, planned.measurement, options.fit, factor, planned.fit);
    }

    if (!planned.fit.IsValid() || planned.fit.line_height < options.min_line_height_px)
        return accepted;

    if (std::fabs(brightness - kBrightnessThreshold) <= 18.0)
        return accepted;

    planned.resolved_rgba = region.has_explicit_rgba
        ? region.rgba
        : ((brightness < kBrightnessThreshold) ? 0xFFFFFFFFu : 0x000000FFu);

    RenderPlanOptions fast_options = options;
    fast_options.allow_overflow = false;
    fast_options.resolve_overlaps = true;
    fast_options.nudge_steps = 4;
    fast_options.shrink_factor = 0.97f;

    planned.fit = ResolveOverlaps(planned, fast_options, placed_boxes);
    if (!planned.fit.IsValid() || planned.fit.line_height < fast_options.min_line_height_px)
        return accepted;

    const OrientedBox accepted_box = MakePlacedBox(planned.geometry.box, planned.fit, planned.measurement);
    if (OverlapsAny(accepted_box, placed_boxes, fast_options.overlap_margin_px))
        return accepted;

    std::vector<RenderGlyph> glyphs = BuildStraightGlyphs(planned, planned.fit);
    if (glyphs.empty())
        return accepted;

    std::vector<GlyphQuad> quads = BuildGlyphQuads(*planned.fit.atlas, glyphs);
    const std::size_t outside_score = region.has_polygon
        ? CountOutsideSamples(region.polygon, quads)
        : 0u;
    if (outside_score > 2u)
        return accepted;

    accepted.valid = true;
    accepted.overflowed = OverflowedBox(planned.geometry.box, planned.measurement, fast_options.fit, planned.fit);
    accepted.is_curved = false;
    accepted.render_size = planned.fit.atlas->render_size;
    accepted.commands = std::move(glyphs);
    accepted.placed_box = accepted_box;
    accepted.footprint.overflowed = accepted.overflowed;
    accepted.footprint.quads = std::move(quads);
    return accepted;
}

static AcceptedPlacement TryAcceptQualityRegion(const FontDatabase& db,
                                                const TextRegion& region,
                                                std::size_t original_index,
                                                const ImageRgba8& image,
                                                const RenderPlanOptions& options,
                                                const std::vector<OrientedBox>& placed_boxes,
                                                const std::vector<PlacedFootprint>& placed_footprints)
{
    AcceptedPlacement accepted;
    PlannedRegion entry = MakeProvisionalRegion(db, region, original_index, image, options);
    if (!entry.valid)
        return accepted;

    std::vector<RenderGlyph> commands;
    uint32_t render_size = 0;
    bool overflowed = false;
    PlacedFootprint footprint;

    if (options.allow_overflow)
    {
        const PlacementCandidate placement =
            ResolveOverflowPlacement(entry, options, placed_footprints);
        if (!placement.valid)
            return accepted;

        commands = placement.glyphs;
        overflowed = placement.overflowed;
        render_size = entry.IsCurved()
            ? placement.curve_fit.atlas->render_size
            : placement.fit.atlas->render_size;
        if (entry.IsCurved())
            entry.curve_fit = placement.curve_fit;
        else
            entry.fit = placement.fit;
        footprint.overflowed = placement.overflowed;
        footprint.quads = placement.quads;
    }
    else
    {
        if (entry.IsCurved())
        {
            if (!entry.curve_fit.IsValid() || entry.curve_fit.atlas == nullptr)
                return accepted;

            commands = LayoutTextOnCurve(*entry.curve_fit.atlas,
                                         entry.region->text,
                                         entry.geometry.curve,
                                         entry.curve_fit,
                                         entry.resolved_rgba);
            render_size = entry.curve_fit.atlas->render_size;
        }
        else
        {
            if (!entry.fit.IsValid() || entry.fit.atlas == nullptr)
                return accepted;

            entry.fit = ResolveOverlaps(entry, options, placed_boxes);
            if (!entry.fit.IsValid())
                return accepted;

            commands = BuildStraightGlyphs(entry, entry.fit);
            render_size = entry.fit.atlas->render_size;
        }

        if (commands.empty())
            return accepted;
        footprint.overflowed = false;
        if (entry.IsCurved())
            footprint.quads = BuildGlyphQuads(*entry.curve_fit.atlas, commands);
        else
            footprint.quads = BuildGlyphQuads(*entry.fit.atlas, commands);
    }

    accepted.valid = true;
    accepted.overflowed = overflowed;
    accepted.is_curved = entry.IsCurved();
    accepted.render_size = render_size;
    accepted.commands = std::move(commands);
    if (!entry.IsCurved())
        accepted.placed_box = MakePlacedBox(entry.geometry.box, entry.fit, entry.measurement);
    accepted.footprint = std::move(footprint);
    return accepted;
}

static ImageRgba8 BuildBrightnessImageFromLuma(uint32_t width,
                                               uint32_t height,
                                               const std::vector<uint8_t>& image_luma)
{
    ImageRgba8 brightness_image(width, height, 0x000000FFu);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint8_t value = image_luma[(std::size_t)y * width + x];
            uint8_t* pixel = brightness_image.PixelPtr(x, y);
            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
            pixel[3] = 255u;
        }
    }
    return brightness_image;
}

static PlacementCandidate MakeStraightCandidate(const PlannedRegion& region,
                                                const RenderPlanOptions& options,
                                                const std::vector<PlacedFootprint>& placed,
                                                const TextFitResult& fit,
                                                float tangent_delta,
                                                float normal_delta)
{
    PlacementCandidate candidate;
    candidate.fit = fit;
    candidate.tangent_delta = tangent_delta;
    candidate.normal_delta = normal_delta;
    candidate.displacement2 = tangent_delta * tangent_delta + normal_delta * normal_delta;

    if (!candidate.fit.IsValid() || candidate.fit.atlas == nullptr)
        return candidate;
    if (!FitsWithinExtendedBox(region.geometry.box,
                               region.measurement,
                               options.fit,
                               options,
                               candidate.fit))
    {
        return candidate;
    }

    candidate.glyphs = BuildStraightGlyphs(region, candidate.fit);
    if (candidate.glyphs.empty())
        return candidate;
    candidate.quads = BuildGlyphQuads(*candidate.fit.atlas, candidate.glyphs);
    if (options.resolve_overlaps
        && OverlapsPlacedQuads(candidate.quads, placed, options.collision_margin_px))
    {
        return candidate;
    }

    candidate.overflowed = OverflowedBox(region.geometry.box,
                                         region.measurement,
                                         options.fit,
                                         candidate.fit);
    candidate.outside_score = region.region->has_polygon
        ? CountOutsideSamples(region.region->polygon, candidate.quads)
        : 0u;
    candidate.valid = true;
    return candidate;
}

static PlacementCandidate MakeCurveCandidate(const PlannedRegion& region,
                                             const RenderPlanOptions& options,
                                             const std::vector<PlacedFootprint>& placed,
                                             const CurveFitResult& fit,
                                             float tangent_delta,
                                             float normal_delta)
{
    PlacementCandidate candidate;
    candidate.curve_fit = fit;
    candidate.tangent_delta = tangent_delta;
    candidate.normal_delta = normal_delta;
    candidate.displacement2 = tangent_delta * tangent_delta + normal_delta * normal_delta;

    if (!candidate.curve_fit.IsValid() || candidate.curve_fit.atlas == nullptr)
        return candidate;
    if (!FitsWithinExtendedCurve(region.geometry.curve,
                                 region.measurement,
                                 options.fit,
                                 options,
                                 candidate.curve_fit))
    {
        return candidate;
    }

    candidate.glyphs = LayoutTextOnCurve(*candidate.curve_fit.atlas,
                                         region.region->text,
                                         region.geometry.curve,
                                         candidate.curve_fit,
                                         region.resolved_rgba);
    if (candidate.glyphs.empty())
        return candidate;
    candidate.quads = BuildGlyphQuads(*candidate.curve_fit.atlas, candidate.glyphs);
    if (options.resolve_overlaps
        && OverlapsPlacedQuads(candidate.quads, placed, options.collision_margin_px))
    {
        return candidate;
    }

    candidate.overflowed = OverflowedCurve(region.geometry.curve,
                                           region.measurement,
                                           options.fit,
                                           candidate.curve_fit);
    candidate.outside_score = region.region->has_polygon
        ? CountOutsideSamples(region.region->polygon, candidate.quads)
        : 0u;
    candidate.valid = true;
    return candidate;
}

static PlacementCandidate ResolveOverflowPlacementAtScale(
    const PlannedRegion& region,
    const RenderPlanOptions& options,
    const std::vector<PlacedFootprint>& placed,
    float scale_factor)
{
    const std::vector<float> tangent_fracs =
        BuildOffsetFractions(options.candidate_tangent_steps);
    const std::vector<float> normal_fracs =
        BuildOffsetFractions(options.candidate_normal_steps);

    PlacementCandidate best_at_scale;

    if (region.IsCurved())
    {
        CurveFitResult scaled = region.curve_fit;
        ScaleCurveFit(region.geometry.curve,
                      region.measurement,
                      options.fit,
                      scale_factor,
                      scaled);
        if (!scaled.IsValid() || scaled.line_height < options.min_line_height_px)
            return best_at_scale;

        const float tangent_budget = EffectiveTangentOverflow(scaled, options);
        const float normal_budget = EffectiveNormalOverflow(region.geometry.curve, options);
        for (float tangent_frac : tangent_fracs)
        {
            for (float normal_frac : normal_fracs)
            {
                CurveFitResult candidate_fit = scaled;
                const float tangent_delta = tangent_frac * tangent_budget;
                const float normal_delta = normal_frac * normal_budget;
                ShiftCurveFit(candidate_fit, tangent_delta, normal_delta);
                PlacementCandidate candidate = MakeCurveCandidate(region,
                                                                  options,
                                                                  placed,
                                                                  candidate_fit,
                                                                  tangent_delta,
                                                                  normal_delta);
                if (BetterCandidateAtSameScale(candidate, best_at_scale))
                    best_at_scale = candidate;
            }
        }
    }
    else
    {
        TextFitResult scaled = region.fit;
        ScaleFit(region.geometry.box,
                 region.measurement,
                 options.fit,
                 scale_factor,
                 scaled);
        if (!scaled.IsValid() || scaled.line_height < options.min_line_height_px)
            return best_at_scale;

        const float tangent_budget = EffectiveTangentOverflow(region.geometry.box, options);
        const float normal_budget = EffectiveNormalOverflow(region.geometry.box, options);
        for (float tangent_frac : tangent_fracs)
        {
            for (float normal_frac : normal_fracs)
            {
                TextFitResult candidate_fit = scaled;
                const float tangent_delta = tangent_frac * tangent_budget;
                const float normal_delta = normal_frac * normal_budget;
                ShiftFit(candidate_fit, tangent_delta, normal_delta);
                PlacementCandidate candidate = MakeStraightCandidate(region,
                                                                     options,
                                                                     placed,
                                                                     candidate_fit,
                                                                     tangent_delta,
                                                                     normal_delta);
                if (BetterCandidateAtSameScale(candidate, best_at_scale))
                    best_at_scale = candidate;
            }
        }
    }

    return best_at_scale;
}

static PlacementCandidate ResolveOverflowPlacement(const PlannedRegion& region,
                                                   const RenderPlanOptions& options,
                                                   const std::vector<PlacedFootprint>& placed)
{
    PlacementCandidate best_any;
    float scale_factor = 1.0f;

    while (true)
    {
        PlacementCandidate best_at_scale =
            ResolveOverflowPlacementAtScale(region, options, placed, scale_factor);
        if (BetterCandidateAtSameScale(best_at_scale, best_any))
            best_any = best_at_scale;
        if (best_at_scale.valid)
            return best_at_scale;
        if (PlannedLineHeight(region) * scale_factor < options.min_line_height_px)
            break;
        if (!(options.scale_step_factor > 0.0f) || !(options.scale_step_factor < 1.0f))
            break;
        scale_factor *= options.scale_step_factor;
    }

    if (!options.skip_if_unplaceable && best_any.valid)
        return best_any;
    return PlacementCandidate{};
}

static TextFitResult ResolveOverlaps(const PlannedRegion& region,
                                     const RenderPlanOptions& options,
                                     const std::vector<OrientedBox>& placed)
{
    TextFitResult accepted = region.fit;
    if (!accepted.IsValid())
        return accepted;

    const OrientedBox accepted_box =
        MakePlacedBox(region.geometry.box, accepted, region.measurement);
    if (placed.empty() || !options.resolve_overlaps
        || !OverlapsAny(accepted_box, placed, options.overlap_margin_px))
    {
        return accepted;
    }

    if (TryResolveByNudging(region.geometry.box,
                            region.measurement,
                            options,
                            placed,
                            accepted))
    {
        return accepted;
    }

    TextFitResult fallback = accepted;
    TextFitResult current = region.fit;

    while (true)
    {
        ScaleFit(region.geometry.box,
                 region.measurement,
                 options.fit,
                 options.shrink_factor,
                 current);
        if (current.line_height < options.min_line_height_px)
            break;

        fallback = current;

        const OrientedBox current_box =
            MakePlacedBox(region.geometry.box, current, region.measurement);
        if (!OverlapsAny(current_box, placed, options.overlap_margin_px))
            return current;

        TextFitResult nudged = current;
        if (TryResolveByNudging(region.geometry.box,
                                region.measurement,
                                options,
                                placed,
                                nudged))
        {
            return nudged;
        }
    }

    return fallback;
}

static AcceptedPlacement MakeAcceptedPlacementFromCandidate(
    const PlannedRegion& entry,
    PlacementCandidate placement)
{
    AcceptedPlacement accepted;
    if (!placement.valid || placement.glyphs.empty())
        return accepted;

    accepted.valid = true;
    accepted.overflowed = placement.overflowed;
    accepted.is_curved = entry.IsCurved();
    accepted.render_size = entry.IsCurved()
        ? placement.curve_fit.atlas->render_size
        : placement.fit.atlas->render_size;
    accepted.commands = std::move(placement.glyphs);
    if (!entry.IsCurved())
        accepted.placed_box =
            MakePlacedBox(entry.geometry.box, placement.fit, entry.measurement);
    accepted.footprint.overflowed = accepted.overflowed;
    accepted.footprint.quads = std::move(placement.quads);
    return accepted;
}

static AcceptedPlacement MakeAcceptedPlacement(PlannedRegion& entry,
                                               PlacementCandidate placement)
{
    if (entry.IsCurved())
        entry.curve_fit = placement.curve_fit;
    else
        entry.fit = placement.fit;
    return MakeAcceptedPlacementFromCandidate(entry, std::move(placement));
}

static AcceptedPlacement TryAcceptPlannedQualityRegion(
    PlannedRegion& entry,
    const RenderPlanOptions& options,
    const std::vector<OrientedBox>& placed_boxes,
    const std::vector<PlacedFootprint>& placed_footprints)
{
    AcceptedPlacement accepted;
    if (!entry.valid)
        return accepted;

    if (options.allow_overflow)
    {
        return MakeAcceptedPlacement(
            entry,
            ResolveOverflowPlacement(entry, options, placed_footprints));
    }

    if (entry.IsCurved())
    {
        if (!entry.curve_fit.IsValid() || entry.curve_fit.atlas == nullptr)
            return accepted;

        accepted.commands = LayoutTextOnCurve(*entry.curve_fit.atlas,
                                              entry.region->text,
                                              entry.geometry.curve,
                                              entry.curve_fit,
                                              entry.resolved_rgba);
        accepted.render_size = entry.curve_fit.atlas->render_size;
        accepted.footprint.quads = BuildGlyphQuads(*entry.curve_fit.atlas, accepted.commands);
    }
    else
    {
        if (!entry.fit.IsValid() || entry.fit.atlas == nullptr)
            return accepted;

        entry.fit = ResolveOverlaps(entry, options, placed_boxes);
        if (!entry.fit.IsValid())
            return accepted;

        accepted.commands = BuildStraightGlyphs(entry, entry.fit);
        accepted.render_size = entry.fit.atlas->render_size;
        accepted.placed_box = MakePlacedBox(entry.geometry.box, entry.fit, entry.measurement);
        accepted.footprint.quads = BuildGlyphQuads(*entry.fit.atlas, accepted.commands);
    }

    if (accepted.commands.empty())
        return AcceptedPlacement{};

    accepted.valid = true;
    accepted.overflowed = false;
    accepted.is_curved = entry.IsCurved();
    accepted.footprint.overflowed = false;
    return accepted;
}

static bool TryAcceptClusterAtScale(
    const std::vector<PlannedRegion*>& group,
    const RenderPlanOptions& options,
    const std::vector<PlacedFootprint>& placed_footprints,
    float scale_factor,
    std::vector<AcceptedPlacement>& out)
{
    out.clear();
    std::vector<PlacedFootprint> attempt_footprints = placed_footprints;
    out.reserve(group.size());

    for (PlannedRegion* entry : group)
    {
        PlacementCandidate candidate =
            ResolveOverflowPlacementAtScale(*entry, options, attempt_footprints, scale_factor);
        if (!candidate.valid)
            return false;

        AcceptedPlacement accepted =
            MakeAcceptedPlacementFromCandidate(*entry, std::move(candidate));
        if (!accepted.valid)
            return false;

        attempt_footprints.push_back(accepted.footprint);
        out.push_back(std::move(accepted));
    }

    return true;
}

static std::vector<AcceptedPlacement> TryAcceptClusteredQualityRegions(
    const std::vector<PlannedRegion*>& group,
    const RenderPlanOptions& options,
    const std::vector<PlacedFootprint>& placed_footprints)
{
    std::vector<AcceptedPlacement> accepted;
    if (group.empty())
        return accepted;

    std::vector<AcceptedPlacement> best_partial;
    float scale_factor = 1.0f;

    while (true)
    {
        std::vector<AcceptedPlacement> attempt;
        if (TryAcceptClusterAtScale(group,
                                    options,
                                    placed_footprints,
                                    scale_factor,
                                    attempt))
        {
            return attempt;
        }

        if (!options.skip_if_unplaceable && attempt.size() > best_partial.size())
            best_partial = std::move(attempt);

        float next_line_height = std::numeric_limits<float>::infinity();
        for (const PlannedRegion* entry : group)
            next_line_height = std::min(next_line_height,
                                        PlannedLineHeight(*entry)
                                            * scale_factor
                                            * options.scale_step_factor);
        if (next_line_height < options.min_line_height_px)
            break;

        scale_factor *= options.scale_step_factor;
        if (!(options.scale_step_factor > 0.0f) || !(options.scale_step_factor < 1.0f))
            break;
    }

    return best_partial;
}

static bool HasClusteredRegions(const std::vector<TextRegion>& regions) noexcept
{
    for (const TextRegion& region : regions)
    {
        if (region.cluster_id != TextRegion::kUnclustered)
            return true;
    }
    return false;
}

} // namespace

static RenderPlan BuildQualityRenderPlanInternal(const FontDatabase& db,
                                                 const ImageRgba8& image,
                                                 const std::vector<TextRegion>& regions,
                                                 const RenderPlanOptions& options)
{
    RenderPlan plan;
    plan.total_regions = regions.size();

    std::vector<PlannedRegion> planned;
    planned.reserve(regions.size());

    for (std::size_t i = 0; i < regions.size(); ++i)
        planned.push_back(MakeProvisionalRegion(db, regions[i], i, image, options));

    if (options.align_lines)
        ApplyLineAlignment(planned, options);
    ApplyClusterInitialSizing(planned, options);

    if (options.allow_overflow)
        std::sort(planned.begin(), planned.end(), PlannedRegionLess);

    std::unordered_map<uint32_t, std::size_t> batch_by_size;
    std::vector<OrientedBox> placed_boxes;
    placed_boxes.reserve(regions.size());
    std::vector<PlacedFootprint> placed_footprints;
    placed_footprints.reserve(regions.size());
    std::vector<bool> processed(planned.size(), false);

    for (std::size_t index = 0; index < planned.size(); ++index)
    {
        if (processed[index])
            continue;

        PlannedRegion& entry = planned[index];
        if (!entry.valid)
        {
            processed[index] = true;
            ++plan.unplaced_regions;
            continue;
        }

        if (IsClustered(entry))
        {
            const std::size_t cluster_id = entry.region->cluster_id;
            std::vector<PlannedRegion*> group;
            std::vector<std::size_t> group_indices;
            for (std::size_t j = index; j < planned.size(); ++j)
            {
                if (processed[j] || !planned[j].valid || !IsClustered(planned[j])
                    || planned[j].region->cluster_id != cluster_id)
                {
                    continue;
                }

                group.push_back(&planned[j]);
                group_indices.push_back(j);
            }

            for (std::size_t group_index : group_indices)
                processed[group_index] = true;

            if (group.size() > 1)
            {
                std::vector<AcceptedPlacement> accepted =
                    TryAcceptClusteredQualityRegions(group, options, placed_footprints);
                if (accepted.size() != group.size() && options.skip_if_unplaceable)
                {
                    plan.unplaced_regions += group.size();
                    continue;
                }

                for (AcceptedPlacement& placement : accepted)
                {
                    ++plan.quality_regions;
                    AppendPlacement(plan,
                                    batch_by_size,
                                    placement,
                                    placed_boxes,
                                    placed_footprints);
                }

                if (accepted.size() < group.size())
                    plan.unplaced_regions += group.size() - accepted.size();
                continue;
            }
        }

        processed[index] = true;
        AcceptedPlacement placement = TryAcceptPlannedQualityRegion(entry,
                                                                    options,
                                                                    placed_boxes,
                                                                    placed_footprints);
        if (!placement.valid)
        {
            ++plan.unplaced_regions;
            continue;
        }

        ++plan.quality_regions;
        AppendPlacement(plan, batch_by_size, placement, placed_boxes, placed_footprints);
    }

    return plan;
}

static RenderPlan BuildAdaptiveRenderPlanInternal(const FontDatabase& db,
                                                  const ImageRgba8& image,
                                                  const std::vector<TextRegion>& regions,
                                                  const RenderPlanOptions& options)
{
    RenderPlan plan;
    plan.total_regions = regions.size();

    std::unordered_map<uint32_t, std::size_t> batch_by_size;
    std::vector<OrientedBox> placed_boxes;
    placed_boxes.reserve(regions.size());
    std::vector<PlacedFootprint> placed_footprints;
    placed_footprints.reserve(regions.size());

    for (std::size_t i = 0; i < regions.size(); ++i)
    {
        const TextRegion& region = regions[i];
        const RegionFeatures features = ComputeRegionFeatures(region);

        AcceptedPlacement placement;
        if (features.initial_kind == RegionPlannerKind::FastStraight)
        {
            std::size_t samples = 0;
            const double brightness =
                SampleBrightnessFromBoundingBox(image, region.polygon, samples);
            placement = TryAcceptFastStraightRegion(db,
                                                    region,
                                                    i,
                                                    features,
                                                    options,
                                                    placed_boxes,
                                                    placed_footprints,
                                                    brightness);
            if (placement.valid)
            {
                ++plan.fast_regions;
                AppendPlacement(plan, batch_by_size, placement, placed_boxes, placed_footprints);
                continue;
            }

            ++plan.escalated_regions;
        }

        placement = TryAcceptQualityRegion(db,
                                           region,
                                           i,
                                           image,
                                           options,
                                           placed_boxes,
                                           placed_footprints);
        if (!placement.valid)
        {
            ++plan.unplaced_regions;
            continue;
        }

        ++plan.quality_regions;
        AppendPlacement(plan, batch_by_size, placement, placed_boxes, placed_footprints);
    }

    return plan;
}

static RenderPlan BuildAdaptiveRenderPlanFromLumaInternal(const FontDatabase& db,
                                                          uint32_t width,
                                                          uint32_t height,
                                                          const std::vector<uint8_t>& image_luma,
                                                          const std::vector<TextRegion>& regions,
                                                          const RenderPlanOptions& options)
{
    RenderPlan plan;
    plan.total_regions = regions.size();

    std::unordered_map<uint32_t, std::size_t> batch_by_size;
    std::vector<OrientedBox> placed_boxes;
    placed_boxes.reserve(regions.size());
    std::vector<PlacedFootprint> placed_footprints;
    placed_footprints.reserve(regions.size());
    std::unique_ptr<ImageRgba8> quality_image;

    const auto ensure_quality_image = [&]() -> const ImageRgba8& {
        if (!quality_image)
            quality_image = std::make_unique<ImageRgba8>(
                BuildBrightnessImageFromLuma(width, height, image_luma));
        return *quality_image;
    };

    for (std::size_t i = 0; i < regions.size(); ++i)
    {
        const TextRegion& region = regions[i];
        const RegionFeatures features = ComputeRegionFeatures(region);

        AcceptedPlacement placement;
        if (features.initial_kind == RegionPlannerKind::FastStraight)
        {
            std::size_t samples = 0;
            const double brightness = SampleBrightnessFromLumaBoundingBox(width,
                                                                          height,
                                                                          image_luma,
                                                                          region.polygon,
                                                                          samples);
            placement = TryAcceptFastStraightRegion(db,
                                                    region,
                                                    i,
                                                    features,
                                                    options,
                                                    placed_boxes,
                                                    placed_footprints,
                                                    brightness);
            if (placement.valid)
            {
                ++plan.fast_regions;
                AppendPlacement(plan, batch_by_size, placement, placed_boxes, placed_footprints);
                continue;
            }

            ++plan.escalated_regions;
        }

        placement = TryAcceptQualityRegion(db,
                                           region,
                                           i,
                                           ensure_quality_image(),
                                           options,
                                           placed_boxes,
                                           placed_footprints);
        if (!placement.valid)
        {
            ++plan.unplaced_regions;
            continue;
        }

        ++plan.quality_regions;
        AppendPlacement(plan, batch_by_size, placement, placed_boxes, placed_footprints);
    }

    return plan;
}

RenderPlan BuildRenderPlan(const FontDatabase& db,
                           const ImageRgba8& image,
                           const std::vector<TextRegion>& regions,
                           const RenderPlanOptions& options)
{
    if (options.profile == PlannerProfile::Adaptive && !HasClusteredRegions(regions))
        return BuildAdaptiveRenderPlanInternal(db, image, regions, options);
    return BuildQualityRenderPlanInternal(db, image, regions, options);
}

RenderPlan BuildAdaptiveRenderPlan(const FontDatabase& db,
                                   uint32_t width,
                                   uint32_t height,
                                   const std::vector<uint8_t>& image_luma,
                                   const std::vector<TextRegion>& regions,
                                   const RenderPlanOptions& options)
{
    if (!HasClusteredRegions(regions))
        return BuildAdaptiveRenderPlanFromLumaInternal(db, width, height, image_luma, regions, options);

    const ImageRgba8 brightness_image = BuildBrightnessImageFromLuma(width, height, image_luma);
    return BuildQualityRenderPlanInternal(db, brightness_image, regions, options);
}

} // namespace fac
