#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fac {

struct Vec2f
{
    float x = 0.0f;
    float y = 0.0f;
};

inline Vec2f operator+(const Vec2f& a, const Vec2f& b) noexcept
{
    return Vec2f{a.x + b.x, a.y + b.y};
}

inline Vec2f operator-(const Vec2f& a, const Vec2f& b) noexcept
{
    return Vec2f{a.x - b.x, a.y - b.y};
}

inline Vec2f operator*(const Vec2f& v, float s) noexcept
{
    return Vec2f{v.x * s, v.y * s};
}

inline Vec2f operator*(float s, const Vec2f& v) noexcept
{
    return v * s;
}

inline Vec2f operator/(const Vec2f& v, float s) noexcept
{
    return Vec2f{v.x / s, v.y / s};
}

inline float Dot(const Vec2f& a, const Vec2f& b) noexcept
{
    return a.x * b.x + a.y * b.y;
}

inline float LengthSquared(const Vec2f& v) noexcept
{
    return Dot(v, v);
}

inline float Length(const Vec2f& v) noexcept
{
    return std::sqrt(LengthSquared(v));
}

inline Vec2f Normalize(const Vec2f& v) noexcept
{
    const float len2 = LengthSquared(v);
    if (len2 <= 1e-12f)
        return Vec2f{1.0f, 0.0f};
    return v / std::sqrt(len2);
}

enum class CurveNormalSide
{
    Right,
    Left,
};

struct CubicBezierSegment
{
    Vec2f p0{};
    Vec2f p1{};
    Vec2f p2{};
    Vec2f p3{};

    bool HasFinitePoints() const noexcept
    {
        return std::isfinite(p0.x) && std::isfinite(p0.y)
            && std::isfinite(p1.x) && std::isfinite(p1.y)
            && std::isfinite(p2.x) && std::isfinite(p2.y)
            && std::isfinite(p3.x) && std::isfinite(p3.y);
    }
};

struct CurvedTextPath
{
    std::vector<CubicBezierSegment> segments;
    float band_height = 0.0f;
    CurveNormalSide normal_side = CurveNormalSide::Right;

    bool IsValid() const noexcept
    {
        if (!(band_height > 0.0f) || segments.empty())
            return false;

        for (const CubicBezierSegment& segment : segments)
        {
            if (!segment.HasFinitePoints())
                return false;
        }

        return true;
    }
};

// Oriented bounding box in image space.
// The local x axis points along the text baseline direction.
// The local y axis points downward in text-local space.
struct OrientedBox
{
    float cx     = 0.0f;
    float cy     = 0.0f;
    float ux     = 1.0f;
    float uy     = 0.0f;
    float vx     = 0.0f;
    float vy     = 1.0f;
    float width  = 0.0f;
    float height = 0.0f;

    bool HasArea() const noexcept
    {
        return width > 0.0f && height > 0.0f;
    }

    bool HasFiniteBasis() const noexcept
    {
        return std::isfinite(ux) && std::isfinite(uy)
            && std::isfinite(vx) && std::isfinite(vy);
    }

    Vec2f LocalToImage(float local_x, float local_y) const noexcept
    {
        return Vec2f{
            cx + local_x * ux + local_y * vx,
            cy + local_x * uy + local_y * vy,
        };
    }

    Vec2f LocalTopLeft() const noexcept
    {
        return Vec2f{
            -0.5f * width,
            -0.5f * height,
        };
    }

    Vec2f TopLeftInImage() const noexcept
    {
        const Vec2f local = LocalTopLeft();
        return LocalToImage(local.x, local.y);
    }
};

struct TextRegion
{
    static constexpr std::size_t kUnclustered =
        std::numeric_limits<std::size_t>::max();

    std::string text;
    std::string original_text;
    std::size_t cluster_id = kUnclustered;
    OrientedBox box;
    std::vector<Vec2f> polygon;
    CurvedTextPath curve;
    uint32_t rgba = 0xFFFFFFFFu;
    bool has_explicit_rgba = false;
    bool has_polygon = false;
    bool has_curve = false;
};

// Glyph placement produced by the CPU layout stage in OBB-local coordinates.
// local_x/local_y denote the glyph origin in text-local space before the box
// transform is applied.
struct GlyphPlacement
{
    uint32_t codepoint = 0;
    float local_x      = 0.0f;
    float local_y      = 0.0f;
    float scale        = 1.0f;
};

// GPU-facing render command. origin_* is the glyph origin in image space;
// basis_u and basis_v map glyph-local coordinates into image coordinates.
struct RenderGlyph
{
    uint32_t atlas_codepoint = 0;
    float origin_x           = 0.0f;
    float origin_y           = 0.0f;
    float basis_ux           = 1.0f;
    float basis_uy           = 0.0f;
    float basis_vx           = 0.0f;
    float basis_vy           = 1.0f;
    float scale              = 1.0f;
    uint32_t rgba            = 0xFFFFFFFFu;
};

} // namespace fac
