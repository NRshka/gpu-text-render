#include "text_layout.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

static void WriteLayoutTestAtlas(const std::filesystem::path& path)
{
    AtlasHeader hdr{};
    hdr.magic          = ATLAS_MAGIC;
    hdr.version        = ATLAS_VERSION_2;
    hdr.width          = 8;
    hdr.height         = 8;
    hdr.glyph_count    = 3;
    hdr.pixel_offset   = sizeof(AtlasHeader) + 3u * sizeof(AtlasGlyphRecord);
    hdr.render_size    = 16;
    hdr.font_height_px = 12;

    AtlasGlyphRecord records[3]{};

    records[0].codepoint = 'A';
    records[0].atlas_x   = 0;
    records[0].atlas_y   = 0;
    records[0].atlas_w   = 8;
    records[0].atlas_h   = 10;
    records[0].advance   = 10.0f;
    records[0].bearing_x = 0.0f;
    records[0].bearing_y = 9.0f;

    records[1].codepoint = 'B';
    records[1].atlas_x   = 0;
    records[1].atlas_y   = 0;
    records[1].atlas_w   = 5;
    records[1].atlas_h   = 8;
    records[1].advance   = 6.0f;
    records[1].bearing_x = 0.0f;
    records[1].bearing_y = 7.0f;

    records[2].codepoint = ' ';
    records[2].atlas_x   = 0;
    records[2].atlas_y   = 0;
    records[2].atlas_w   = 0;
    records[2].atlas_h   = 0;
    records[2].advance   = 4.0f;
    records[2].bearing_x = 0.0f;
    records[2].bearing_y = 0.0f;

    std::vector<uint8_t> pixels(hdr.width * hdr.height, 0);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = (uint8_t)(i & 0xFFu);

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path.string());

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(records), sizeof(records));
    f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

int main()
{
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
    fs::create_directories(dir);

    const fs::path atlas_path = dir / "atlas_layout.bin";
    WriteLayoutTestAtlas(atlas_path);

    FontDatabase db;
    db.LoadAtlas(atlas_path);
    const AtlasEntry* atlas = db.GetEntry(16);
    if (!atlas) {
        std::cerr << "Atlas failed to load\n";
        return EXIT_FAILURE;
    }

    SECTION("UTF-8 decoding");
    {
        const std::vector<uint32_t> cps = DecodeUtf8("A\xe1\x83\x90");
        EXPECT(cps.size() == 2u);
        EXPECT(cps[0] == (uint32_t)'A');
        EXPECT(cps[1] == 0x10D0u);
    }

    SECTION("Text measurement");
    {
        const TextMeasurement m = MeasureTextLine(*atlas, "AB");
        EXPECT(m.glyph_count == 2u);
        EXPECT_NEAR(m.width, 16.0f, 1e-6f);
        EXPECT_NEAR(m.advance_width, 16.0f, 1e-6f);
        EXPECT_NEAR(m.ascent, 9.0f, 1e-6f);
        EXPECT_NEAR(m.descent, 1.0f, 1e-6f);
    }

    SECTION("Center fit and layout");
    {
        OrientedBox box;
        box.width = 20.0f;
        box.height = 20.0f;

        const TextFitResult fit = FitTextToBox(db, "AB", box);
        EXPECT(fit.IsValid());
        EXPECT(fit.atlas == atlas);
        EXPECT_NEAR(fit.scale, 1.25f, 1e-6f);
        EXPECT_NEAR(fit.start_x, -10.0f, 1e-6f);
        EXPECT_NEAR(fit.baseline_y, 3.75f, 1e-6f);
        EXPECT_NEAR(fit.text_width, 20.0f, 1e-6f);
        EXPECT_NEAR(fit.line_height, 15.0f, 1e-6f);

        const std::vector<GlyphPlacement> placements =
            LayoutTextLine(*atlas, "AB", fit);
        EXPECT(placements.size() == 2u);
        EXPECT_NEAR(placements[0].local_x, -10.0f, 1e-6f);
        EXPECT_NEAR(placements[0].local_y, -7.5f, 1e-6f);
        EXPECT_NEAR(placements[1].local_x, 2.5f, 1e-6f);
        EXPECT_NEAR(placements[1].local_y, -5.0f, 1e-6f);
    }

    SECTION("Left/top alignment and render transform");
    {
        OrientedBox box;
        box.cx = 100.0f;
        box.cy = 50.0f;
        box.ux = 0.0f;
        box.uy = 1.0f;
        box.vx = -1.0f;
        box.vy = 0.0f;
        box.width = 40.0f;
        box.height = 20.0f;

        TextFitOptions options;
        options.padding_x = 2.0f;
        options.padding_y = 1.0f;
        options.horizontal_align = HorizontalAlign::Left;
        options.vertical_align = VerticalAlign::Top;

        const TextFitResult fit = FitTextToBox(db, "A B", box, options);
        EXPECT(fit.IsValid());
        EXPECT_NEAR(fit.start_x, -18.0f, 1e-6f);
        EXPECT_NEAR(fit.baseline_y, 4.5f, 1e-6f);

        const std::vector<GlyphPlacement> placements =
            LayoutTextLine(*atlas, "A B", fit);
        EXPECT(placements.size() == 3u);
        EXPECT_NEAR(placements[0].local_x, -18.0f, 1e-6f);

        const RenderGlyph cmd = MakeRenderGlyph(placements[0], box, 0xAABBCCDDu);
        EXPECT(cmd.atlas_codepoint == (uint32_t)'A');
        EXPECT_NEAR(cmd.origin_x, 109.0f, 1e-6f);
        EXPECT_NEAR(cmd.origin_y, 32.0f, 1e-6f);
        EXPECT_NEAR(cmd.basis_ux, 0.0f, 1e-6f);
        EXPECT_NEAR(cmd.basis_uy, 1.0f, 1e-6f);
        EXPECT_NEAR(cmd.basis_vx, -1.0f, 1e-6f);
        EXPECT_NEAR(cmd.basis_vy, 0.0f, 1e-6f);
        EXPECT(cmd.rgba == 0xAABBCCDDu);
    }

    SECTION("Curve fit and centered layout");
    {
        CurvedTextPath curve;
        curve.band_height = 20.0f;
        curve.normal_side = CurveNormalSide::Right;

        CubicBezierSegment segment;
        segment.p0 = Vec2f{0.0f, 10.0f};
        segment.p1 = Vec2f{13.333333f, 10.0f};
        segment.p2 = Vec2f{26.666666f, 10.0f};
        segment.p3 = Vec2f{40.0f, 10.0f};
        curve.segments.push_back(segment);

        const CurveFitResult fit = FitTextToCurve(db, "AB", curve);
        EXPECT(fit.IsValid());
        EXPECT(fit.atlas == atlas);
        EXPECT_NEAR(fit.scale, 20.0f / 12.0f, 1e-4f);
        EXPECT_NEAR(fit.start_offset, (40.0f - 16.0f * fit.scale) * 0.5f, 1e-4f);

        const std::vector<RenderGlyph> glyphs =
            LayoutTextOnCurve(*atlas, "AB", curve, fit, 0x12345678u);
        EXPECT(glyphs.size() == 2u);
        EXPECT_NEAR(glyphs[0].basis_ux, 1.0f, 1e-3f);
        EXPECT_NEAR(glyphs[0].basis_uy, 0.0f, 1e-3f);
        EXPECT_NEAR(glyphs[0].basis_vx, 0.0f, 1e-3f);
        EXPECT(glyphs[0].origin_x > 0.0f);
        EXPECT(glyphs[1].origin_x > glyphs[0].origin_x);
        EXPECT(glyphs[0].rgba == 0x12345678u);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
