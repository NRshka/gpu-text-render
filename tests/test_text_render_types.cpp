#include "text_render_types.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace fac;

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT(cond)                                                  \
    do {                                                              \
        if (cond) {                                                   \
            ++g_passed;                                               \
        } else {                                                      \
            ++g_failed;                                               \
            std::cerr << "  FAIL  " << #cond                         \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        }                                                             \
    } while (0)

#define EXPECT_NEAR(a, b, eps) EXPECT(std::fabs((a) - (b)) <= (eps))
#define SECTION(name) std::cout << "\n[" << name << "]\n"

int main()
{
    SECTION("Default box");
    {
        OrientedBox box;
        EXPECT(box.HasFiniteBasis());
        EXPECT(!box.HasArea());

        const Vec2f p = box.LocalToImage(10.0f, 20.0f);
        EXPECT_NEAR(p.x, 10.0f, 1e-6f);
        EXPECT_NEAR(p.y, 20.0f, 1e-6f);
    }

    SECTION("Axis-aligned transform");
    {
        OrientedBox box;
        box.cx = 100.0f;
        box.cy = 50.0f;
        box.width = 40.0f;
        box.height = 20.0f;

        EXPECT(box.HasArea());

        const Vec2f top_left = box.TopLeftInImage();
        EXPECT_NEAR(top_left.x, 80.0f, 1e-6f);
        EXPECT_NEAR(top_left.y, 40.0f, 1e-6f);

        const Vec2f p = box.LocalToImage(-20.0f, -10.0f);
        EXPECT_NEAR(p.x, top_left.x, 1e-6f);
        EXPECT_NEAR(p.y, top_left.y, 1e-6f);
    }

    SECTION("Rotated transform");
    {
        OrientedBox box;
        box.cx = 10.0f;
        box.cy = 20.0f;
        box.ux = 0.0f;
        box.uy = 1.0f;
        box.vx = -1.0f;
        box.vy = 0.0f;
        box.width = 12.0f;
        box.height = 6.0f;

        const Vec2f p = box.LocalToImage(4.0f, 2.0f);
        EXPECT_NEAR(p.x, 8.0f, 1e-6f);
        EXPECT_NEAR(p.y, 24.0f, 1e-6f);

        const Vec2f top_left = box.TopLeftInImage();
        EXPECT_NEAR(top_left.x, 13.0f, 1e-6f);
        EXPECT_NEAR(top_left.y, 14.0f, 1e-6f);
    }

    SECTION("Region and render command defaults");
    {
        TextRegion region;
        EXPECT(region.text.empty());
        EXPECT(region.original_text.empty());
        EXPECT(region.rgba == 0xFFFFFFFFu);
        EXPECT(!region.has_explicit_rgba);
        EXPECT(!region.has_polygon);
        EXPECT(!region.has_curve);
        EXPECT(region.polygon.empty());
        EXPECT(!region.curve.IsValid());

        GlyphPlacement placement;
        EXPECT(placement.codepoint == 0u);
        EXPECT_NEAR(placement.scale, 1.0f, 1e-6f);

        RenderGlyph cmd;
        EXPECT(cmd.atlas_codepoint == 0u);
        EXPECT_NEAR(cmd.origin_x, 0.0f, 1e-6f);
        EXPECT_NEAR(cmd.origin_y, 0.0f, 1e-6f);
        EXPECT_NEAR(cmd.basis_ux, 1.0f, 1e-6f);
        EXPECT_NEAR(cmd.basis_vy, 1.0f, 1e-6f);
        EXPECT(cmd.rgba == 0xFFFFFFFFu);
    }

    SECTION("Curved path validity");
    {
        CurvedTextPath curve;
        curve.band_height = 12.0f;
        EXPECT(!curve.IsValid());

        CubicBezierSegment segment;
        segment.p0 = Vec2f{0.0f, 0.0f};
        segment.p1 = Vec2f{5.0f, 0.0f};
        segment.p2 = Vec2f{10.0f, 4.0f};
        segment.p3 = Vec2f{15.0f, 4.0f};
        curve.segments.push_back(segment);
        EXPECT(curve.IsValid());
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
