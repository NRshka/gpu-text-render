#include "render_plan.h"
#include "text_renderer_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

static void WriteAtlas(const std::filesystem::path& path,
                       uint32_t render_size,
                       uint32_t font_height_px,
                       uint16_t glyph_w,
                       uint16_t glyph_h,
                       float advance)
{
    AtlasHeader hdr{};
    hdr.magic          = ATLAS_MAGIC;
    hdr.version        = ATLAS_VERSION_2;
    hdr.width          = 32;
    hdr.height         = 16;
    hdr.glyph_count    = 4;
    hdr.pixel_offset   = sizeof(AtlasHeader) + 4u * sizeof(AtlasGlyphRecord);
    hdr.render_size    = render_size;
    hdr.font_height_px = font_height_px;

    AtlasGlyphRecord records[4]{};

    records[0].codepoint = 'A';
    records[0].atlas_x   = 0;
    records[0].atlas_y   = 0;
    records[0].atlas_w   = glyph_w;
    records[0].atlas_h   = glyph_h;
    records[0].advance   = advance;
    records[0].bearing_x = 0.0f;
    records[0].bearing_y = (float)glyph_h;

    records[1].codepoint = 'B';
    records[1].atlas_x   = (uint16_t)(glyph_w + 1);
    records[1].atlas_y   = 0;
    records[1].atlas_w   = glyph_w;
    records[1].atlas_h   = glyph_h;
    records[1].advance   = advance;
    records[1].bearing_x = 0.0f;
    records[1].bearing_y = (float)glyph_h;

    records[2].codepoint = 0x10D0u; // Georgian ა
    records[2].atlas_x   = (uint16_t)(2 * (glyph_w + 1));
    records[2].atlas_y   = 0;
    records[2].atlas_w   = glyph_w;
    records[2].atlas_h   = glyph_h;
    records[2].advance   = advance;
    records[2].bearing_x = 0.0f;
    records[2].bearing_y = (float)glyph_h;

    records[3].codepoint = ' ';
    records[3].atlas_x   = 0;
    records[3].atlas_y   = 0;
    records[3].atlas_w   = 0;
    records[3].atlas_h   = 0;
    records[3].advance   = advance * 0.5f;
    records[3].bearing_x = 0.0f;
    records[3].bearing_y = 0.0f;

    std::vector<uint8_t> pixels((std::size_t)hdr.width * hdr.height, 0);
    for (uint16_t y = 0; y < glyph_h; ++y)
    {
        for (uint16_t x = 0; x < glyph_w; ++x)
        {
            pixels[(std::size_t)y * hdr.width + x] = 255;
            pixels[(std::size_t)y * hdr.width + (glyph_w + 1 + x)] = 255;
            pixels[(std::size_t)y * hdr.width + (2 * (glyph_w + 1) + x)] = 255;
        }
    }

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path.string());

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(records), sizeof(records));
    f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

static int CountChangedPixels(const ImageRgba8& image)
{
    int changed = 0;
    for (uint32_t y = 0; y < image.height; ++y)
    {
        for (uint32_t x = 0; x < image.width; ++x)
        {
            const uint8_t* p = image.PixelPtr(x, y);
            if (!(p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 255))
                ++changed;
        }
    }
    return changed;
}

static std::vector<std::pair<uint32_t, RenderGlyph>> CollectGlyphs(const RenderPlan& plan)
{
    std::vector<std::pair<uint32_t, RenderGlyph>> out;
    out.reserve(plan.total_glyphs);
    for (const RenderBatch& batch : plan.batches)
    {
        for (const RenderGlyph& glyph : batch.glyphs)
            out.emplace_back(batch.atlas_render_size, glyph);
    }
    return out;
}

static float EffectiveLineHeight(const FontDatabase& db,
                                 const std::pair<uint32_t, RenderGlyph>& glyph)
{
    const AtlasEntry* entry = db.GetEntry(glyph.first);
    if (!entry)
        throw std::runtime_error("Missing atlas entry");
    return (float)entry->font_height_px * glyph.second.scale;
}

int main()
{
    namespace fs = std::filesystem;

    const fs::path dir =
        fs::temp_directory_path() / "gpu_font_rendering_tests" / "render_plan";
    fs::create_directories(dir);

    WriteAtlas(dir / "atlas_16.bin", 16, 8, 2, 2, 3.0f);
    WriteAtlas(dir / "atlas_32.bin", 32, 20, 4, 4, 6.0f);

    FontDatabase db;
    db.LoadDirectory(dir);

    SECTION("Build batches by atlas size");
    {
        std::vector<TextRegion> regions;

        TextRegion small;
        small.text = u8"Aა";
        small.original_text = u8"Aა";
        small.box.width = 8.0f;
        small.box.height = 8.0f;
        small.rgba = 0xFFFFFFFFu;
        regions.push_back(small);

        TextRegion large;
        large.text = u8"აა";
        large.original_text = u8"აა";
        large.box.cx = 20.0f;
        large.box.cy = 20.0f;
        large.box.width = 20.0f;
        large.box.height = 20.0f;
        large.rgba = 0xFF0000FFu;
        regions.push_back(large);

        const ImageRgba8 base(32, 32, 0x000000FFu);
        const RenderPlan plan = BuildRenderPlan(db, base, regions);
        EXPECT(plan.total_regions == 2u);
        EXPECT(plan.fitted_regions == 2u);
        EXPECT(plan.total_glyphs == 4u);
        EXPECT(plan.batches.size() == 2u);

        bool saw_16 = false;
        bool saw_32 = false;
        for (const RenderBatch& batch : plan.batches)
        {
            if (batch.atlas_render_size == 16u) {
                saw_16 = true;
                EXPECT(batch.glyphs.size() == 2u);
            }
            if (batch.atlas_render_size == 32u) {
                saw_32 = true;
                EXPECT(batch.glyphs.size() == 2u);
            }
        }
        EXPECT(saw_16);
        EXPECT(saw_32);

        ImageRgba8 image = base;
        RenderPlanCpu(image, db, plan);
        EXPECT(CountChangedPixels(image) > 0);
    }

    SECTION("Explicit rgba preserves provided color");
    {
        TextRegion region;
        region.text = "A";
        region.original_text = "A";
        region.box.cx = 8.0f;
        region.box.cy = 8.0f;
        region.box.width = 10.0f;
        region.box.height = 10.0f;
        region.rgba = 0xFF00FFFFu;
        region.has_explicit_rgba = true;

        const RenderPlan dark_plan =
            BuildRenderPlan(db, ImageRgba8(16, 16, 0x000000FFu), {region});
        const std::vector<std::pair<uint32_t, RenderGlyph>> dark_glyphs =
            CollectGlyphs(dark_plan);
        EXPECT(dark_glyphs.size() == 1u);
        EXPECT(dark_glyphs[0].second.rgba == 0xFF00FFFFu);

        const RenderPlan bright_plan =
            BuildRenderPlan(db, ImageRgba8(16, 16, 0xFFFFFFFFu), {region});
        const std::vector<std::pair<uint32_t, RenderGlyph>> bright_glyphs =
            CollectGlyphs(bright_plan);
        EXPECT(bright_glyphs.size() == 1u);
        EXPECT(bright_glyphs[0].second.rgba == 0xFF00FFFFu);
    }

    SECTION("Auto color chooses white on dark and black on bright");
    {
        TextRegion region;
        region.text = "A";
        region.original_text = "A";
        region.box.cx = 8.0f;
        region.box.cy = 8.0f;
        region.box.width = 10.0f;
        region.box.height = 10.0f;

        const RenderPlan dark_plan =
            BuildRenderPlan(db, ImageRgba8(16, 16, 0x000000FFu), {region});
        const std::vector<std::pair<uint32_t, RenderGlyph>> dark_glyphs =
            CollectGlyphs(dark_plan);
        EXPECT(dark_glyphs.size() == 1u);
        EXPECT(dark_glyphs[0].second.rgba == 0xFFFFFFFFu);

        const RenderPlan bright_plan =
            BuildRenderPlan(db, ImageRgba8(16, 16, 0xFFFFFFFFu), {region});
        const std::vector<std::pair<uint32_t, RenderGlyph>> bright_glyphs =
            CollectGlyphs(bright_plan);
        EXPECT(bright_glyphs.size() == 1u);
        EXPECT(bright_glyphs[0].second.rgba == 0x000000FFu);
    }

    SECTION("Length-aware sizing shrinks longer translations");
    {
        TextRegion equal;
        equal.text = "A A A";
        equal.original_text = "A A A";
        equal.box.cx = 40.0f;
        equal.box.cy = 12.0f;
        equal.box.width = 80.0f;
        equal.box.height = 20.0f;

        TextRegion longer = equal;
        longer.original_text = "A";

        const ImageRgba8 base(96, 24, 0x000000FFu);
        RenderPlanOptions options;
        options.min_line_height_px = 2.0f;
        const std::vector<std::pair<uint32_t, RenderGlyph>> equal_glyphs =
            CollectGlyphs(BuildRenderPlan(db, base, {equal}, options));
        const std::vector<std::pair<uint32_t, RenderGlyph>> longer_glyphs =
            CollectGlyphs(BuildRenderPlan(db, base, {longer}, options));

        EXPECT(equal_glyphs.size() == 5u);
        EXPECT(longer_glyphs.size() == 5u);
        EXPECT(longer_glyphs[0].second.scale < equal_glyphs[0].second.scale);
    }

    SECTION("Overflow-aware placement can keep a larger readable scale");
    {
        TextRegion region;
        region.text = "A A A";
        region.original_text = "A A A";
        region.box.cx = 40.0f;
        region.box.cy = 12.0f;
        region.box.width = 18.0f;
        region.box.height = 20.0f;

        RenderPlanOptions strict;
        strict.allow_overflow = false;
        const std::vector<std::pair<uint32_t, RenderGlyph>> strict_glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(96, 24, 0x000000FFu), {region}, strict));

        RenderPlanOptions overflow;
        const std::vector<std::pair<uint32_t, RenderGlyph>> overflow_glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(96, 24, 0x000000FFu), {region}, overflow));

        EXPECT(strict_glyphs.size() == 5u);
        EXPECT(overflow_glyphs.size() == 5u);
        EXPECT(overflow_glyphs[0].second.scale > strict_glyphs[0].second.scale);
    }

    SECTION("Overlap resolution nudges overlapping regions");
    {
        TextRegion a;
        a.text = "A";
        a.original_text = "A";
        a.box.cx = 16.0f;
        a.box.cy = 16.0f;
        a.box.width = 16.0f;
        a.box.height = 16.0f;

        TextRegion b = a;
        b.box.cx = 18.0f;

        const ImageRgba8 base(40, 40, 0x000000FFu);

        RenderPlanOptions no_resolve;
        no_resolve.resolve_overlaps = false;
        no_resolve.min_line_height_px = 2.0f;
        const std::vector<std::pair<uint32_t, RenderGlyph>> plain =
            CollectGlyphs(BuildRenderPlan(db, base, {a, b}, no_resolve));

        RenderPlanOptions resolve;
        resolve.min_line_height_px = 2.0f;
        const std::vector<std::pair<uint32_t, RenderGlyph>> fixed =
            CollectGlyphs(BuildRenderPlan(db, base, {a, b}, resolve));

        EXPECT(plain.size() == 2u);
        EXPECT(fixed.size() == 2u);
        EXPECT(std::fabs(fixed[1].second.origin_x - plain[1].second.origin_x) > 0.1f
            || std::fabs(fixed[1].second.origin_y - plain[1].second.origin_y) > 0.1f);
    }

    SECTION("Rotated overlap resolution stays deterministic");
    {
        TextRegion a;
        a.text = "A";
        a.original_text = "A";
        a.box.cx = 32.0f;
        a.box.cy = 32.0f;
        a.box.ux = 0.70710678f;
        a.box.uy = 0.70710678f;
        a.box.vx = -0.70710678f;
        a.box.vy = 0.70710678f;
        a.box.width = 16.0f;
        a.box.height = 14.0f;

        TextRegion b = a;
        b.box.cx = 34.0f;
        b.box.cy = 34.0f;

        const ImageRgba8 base(64, 64, 0x000000FFu);
        RenderPlanOptions options;
        options.min_line_height_px = 2.0f;
        const std::vector<std::pair<uint32_t, RenderGlyph>> fixed =
            CollectGlyphs(BuildRenderPlan(db, base, {a, b}, options));
        EXPECT(fixed.size() == 2u);
        EXPECT(fixed[0].second.origin_x != fixed[1].second.origin_x
            || fixed[0].second.origin_y != fixed[1].second.origin_y);
    }

    SECTION("Placement ordering stays stable across input order");
    {
        TextRegion a;
        a.text = "A";
        a.original_text = "A";
        a.box.cx = 16.0f;
        a.box.cy = 16.0f;
        a.box.width = 16.0f;
        a.box.height = 16.0f;

        TextRegion b = a;
        b.text = "B";
        b.original_text = "B";
        b.box.cx = 18.0f;

        RenderPlanOptions options;
        options.min_line_height_px = 2.0f;
        const std::vector<std::pair<uint32_t, RenderGlyph>> ab =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(32, 32, 0x000000FFu), {a, b}, options));
        const std::vector<std::pair<uint32_t, RenderGlyph>> ba =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(32, 32, 0x000000FFu), {b, a}, options));

        EXPECT(ab.size() == 2u);
        EXPECT(ba.size() == 2u);

        std::vector<std::pair<float, float>> ab_origins;
        std::vector<std::pair<float, float>> ba_origins;
        for (const auto& glyph : ab)
            ab_origins.emplace_back(glyph.second.origin_x, glyph.second.origin_y);
        for (const auto& glyph : ba)
            ba_origins.emplace_back(glyph.second.origin_x, glyph.second.origin_y);
        std::sort(ab_origins.begin(), ab_origins.end());
        std::sort(ba_origins.begin(), ba_origins.end());
        EXPECT(ab_origins == ba_origins);
    }

    SECTION("Minimum-size exhaustion marks the region unplaced");
    {
        TextRegion region;
        region.text = "A A A A A";
        region.original_text = "A A A A A";
        region.box.cx = 16.0f;
        region.box.cy = 16.0f;
        region.box.width = 4.0f;
        region.box.height = 4.0f;

        RenderPlanOptions options;
        options.min_line_height_px = 12.0f;

        const RenderPlan plan =
            BuildRenderPlan(db, ImageRgba8(32, 32, 0x000000FFu), {region}, options);
        EXPECT(plan.fitted_regions == 0u);
        EXPECT(plan.unplaced_regions == 1u);
        EXPECT(plan.total_glyphs == 0u);
    }

    SECTION("Optional line alignment reduces line-height mismatch");
    {
        TextRegion a;
        a.text = "A";
        a.original_text = "A";
        a.box.cx = 20.0f;
        a.box.cy = 24.0f;
        a.box.width = 20.0f;
        a.box.height = 16.0f;

        TextRegion b = a;
        b.box.cx = 48.0f;
        b.box.width = 30.0f;
        b.box.height = 30.0f;

        const ImageRgba8 base(96, 48, 0x000000FFu);

        RenderPlanOptions no_align;
        no_align.align_lines = false;
        const std::vector<std::pair<uint32_t, RenderGlyph>> loose =
            CollectGlyphs(BuildRenderPlan(db, base, {a, b}, no_align));

        RenderPlanOptions align;
        align.align_lines = true;
        const std::vector<std::pair<uint32_t, RenderGlyph>> aligned =
            CollectGlyphs(BuildRenderPlan(db, base, {a, b}, align));

        EXPECT(loose.size() == 2u);
        EXPECT(aligned.size() == 2u);

        const float loose_diff =
            std::fabs(EffectiveLineHeight(db, loose[0]) - EffectiveLineHeight(db, loose[1]));
        const float aligned_diff =
            std::fabs(EffectiveLineHeight(db, aligned[0]) - EffectiveLineHeight(db, aligned[1]));
        EXPECT(aligned_diff < loose_diff);
    }

    SECTION("Rectangle polygon falls back to straight layout");
    {
        TextRegion region;
        region.text = "AB";
        region.original_text = "AB";
        region.has_polygon = true;
        region.polygon = {
            Vec2f{8.0f, 8.0f},
            Vec2f{40.0f, 8.0f},
            Vec2f{40.0f, 20.0f},
            Vec2f{8.0f, 20.0f},
        };

        const std::vector<std::pair<uint32_t, RenderGlyph>> glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {region}));
        EXPECT(glyphs.size() == 2u);
        EXPECT_NEAR(glyphs[0].second.basis_ux, glyphs[1].second.basis_ux, 1e-6f);
        EXPECT_NEAR(glyphs[0].second.basis_uy, glyphs[1].second.basis_uy, 1e-6f);
    }

    SECTION("Polygon scoring prefers the better-contained offset");
    {
        TextRegion polygon_region;
        polygon_region.text = "ABAB";
        polygon_region.original_text = "ABAB";
        polygon_region.has_polygon = true;
        polygon_region.polygon = {
            Vec2f{8.0f, 8.0f},
            Vec2f{40.0f, 8.0f},
            Vec2f{40.0f, 20.0f},
            Vec2f{22.0f, 20.0f},
            Vec2f{8.0f, 12.0f},
        };

        TextRegion box_region;
        box_region.text = polygon_region.text;
        box_region.original_text = polygon_region.original_text;
        box_region.box.cx = 24.0f;
        box_region.box.cy = 14.0f;
        box_region.box.width = 32.0f;
        box_region.box.height = 12.0f;

        const std::vector<std::pair<uint32_t, RenderGlyph>> polygon_glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {polygon_region}));
        const std::vector<std::pair<uint32_t, RenderGlyph>> box_glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {box_region}));

        EXPECT(polygon_glyphs.size() == 4u);
        EXPECT(box_glyphs.size() == 4u);
        EXPECT(polygon_glyphs[0].second.origin_x > box_glyphs[0].second.origin_x);
    }

    SECTION("Explicit curve honors normal side");
    {
        TextRegion region;
        region.text = "AB";
        region.original_text = "AB";
        region.has_curve = true;
        region.curve.band_height = 18.0f;
        region.curve.normal_side = CurveNormalSide::Left;

        CubicBezierSegment segment;
        segment.p0 = Vec2f{8.0f, 16.0f};
        segment.p1 = Vec2f{20.0f, 16.0f};
        segment.p2 = Vec2f{32.0f, 16.0f};
        segment.p3 = Vec2f{44.0f, 16.0f};
        region.curve.segments.push_back(segment);

        const std::vector<std::pair<uint32_t, RenderGlyph>> glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {region}));
        EXPECT(glyphs.size() == 2u);
        EXPECT(glyphs[0].second.basis_vy < 0.0f);
    }

    SECTION("Curved overlap resolution shifts neighboring curves");
    {
        TextRegion a;
        a.text = "AB";
        a.original_text = "AB";
        a.has_curve = true;
        a.curve.band_height = 18.0f;
        CubicBezierSegment seg_a;
        seg_a.p0 = Vec2f{8.0f, 16.0f};
        seg_a.p1 = Vec2f{20.0f, 16.0f};
        seg_a.p2 = Vec2f{32.0f, 16.0f};
        seg_a.p3 = Vec2f{44.0f, 16.0f};
        a.curve.segments.push_back(seg_a);

        TextRegion b = a;
        CubicBezierSegment seg_b = seg_a;
        seg_b.p0.x += 2.0f;
        seg_b.p1.x += 2.0f;
        seg_b.p2.x += 2.0f;
        seg_b.p3.x += 2.0f;
        b.curve.segments.clear();
        b.curve.segments.push_back(seg_b);

        RenderPlanOptions no_resolve;
        no_resolve.resolve_overlaps = false;
        const std::vector<std::pair<uint32_t, RenderGlyph>> plain =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {a, b}, no_resolve));

        RenderPlanOptions resolve;
        const std::vector<std::pair<uint32_t, RenderGlyph>> fixed =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {a, b}, resolve));

        EXPECT(plain.size() == 4u);
        EXPECT(fixed.size() == 4u);
        EXPECT(std::fabs(fixed[2].second.origin_x - plain[2].second.origin_x) > 0.1f
            || std::fabs(fixed[2].second.origin_y - plain[2].second.origin_y) > 0.1f);
    }

    SECTION("Arced polygon resolves to curved layout");
    {
        TextRegion region;
        region.text = "AAA";
        region.original_text = "AAA";
        region.has_polygon = true;
        region.polygon = {
            Vec2f{8.0f, 20.0f},
            Vec2f{16.0f, 12.0f},
            Vec2f{28.0f, 8.0f},
            Vec2f{40.0f, 12.0f},
            Vec2f{48.0f, 20.0f},
            Vec2f{48.0f, 30.0f},
            Vec2f{40.0f, 24.0f},
            Vec2f{28.0f, 22.0f},
            Vec2f{16.0f, 24.0f},
            Vec2f{8.0f, 28.0f},
        };

        const std::vector<std::pair<uint32_t, RenderGlyph>> glyphs =
            CollectGlyphs(BuildRenderPlan(db, ImageRgba8(64, 40, 0x000000FFu), {region}));
        EXPECT(glyphs.size() == 3u);
        EXPECT(std::fabs(glyphs.front().second.origin_y - glyphs.back().second.origin_y) > 0.1f
            || std::fabs(glyphs.front().second.basis_uy - glyphs.back().second.basis_uy) > 0.02f);
    }

    SECTION("Strict mode preserves analytic fit inside the box");
    {
        TextRegion region;
        region.text = "AB";
        region.original_text = "AB";
        region.box.cx = 20.0f;
        region.box.cy = 12.0f;
        region.box.width = 10.0f;
        region.box.height = 8.0f;

        RenderPlanOptions options;
        options.allow_overflow = false;
        const RenderPlan plan =
            BuildRenderPlan(db, ImageRgba8(40, 24, 0x000000FFu), {region}, options);
        const std::vector<std::pair<uint32_t, RenderGlyph>> glyphs = CollectGlyphs(plan);
        const TextFitResult fit = FitTextToBox(db, region.text, region.box, options.fit);

        EXPECT(glyphs.size() == 2u);
        EXPECT(fit.IsValid());
        EXPECT_NEAR(glyphs[0].second.scale, fit.scale, 1e-6f);
        EXPECT(plan.overflowed_regions == 0u);
    }

    SECTION("Adaptive planner fast-paths simple straight polygons");
    {
        TextRegion region;
        region.text = "AB";
        region.original_text = "AB";
        region.has_polygon = true;
        region.polygon = {
            Vec2f{8.0f, 8.0f},
            Vec2f{40.0f, 8.0f},
            Vec2f{40.0f, 20.0f},
            Vec2f{8.0f, 20.0f},
        };

        RenderPlanOptions options;
        options.profile = PlannerProfile::Adaptive;
        options.allow_overflow = false;
        options.min_line_height_px = 2.0f;

        const RenderPlan plan =
            BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {region}, options);
        EXPECT(plan.fitted_regions == 1u);
        EXPECT(plan.fast_regions == 1u);
        EXPECT(plan.quality_regions == 0u);
        EXPECT(plan.escalated_regions == 0u);
    }

    SECTION("Adaptive planner routes explicit curves to quality");
    {
        TextRegion region;
        region.text = "AB";
        region.original_text = "AB";
        region.has_curve = true;
        region.curve.band_height = 18.0f;
        region.curve.normal_side = CurveNormalSide::Left;

        CubicBezierSegment segment;
        segment.p0 = Vec2f{8.0f, 16.0f};
        segment.p1 = Vec2f{20.0f, 16.0f};
        segment.p2 = Vec2f{32.0f, 16.0f};
        segment.p3 = Vec2f{44.0f, 16.0f};
        region.curve.segments.push_back(segment);

        RenderPlanOptions options;
        options.profile = PlannerProfile::Adaptive;

        const RenderPlan plan =
            BuildRenderPlan(db, ImageRgba8(64, 32, 0x000000FFu), {region}, options);
        EXPECT(plan.fitted_regions == 1u);
        EXPECT(plan.fast_regions == 0u);
        EXPECT(plan.quality_regions == 1u);
        EXPECT(plan.escalated_regions == 0u);
    }

    SECTION("Adaptive planner escalates ambiguous brightness regions");
    {
        TextRegion region;
        region.text = "AB";
        region.original_text = "AB";
        region.has_polygon = true;
        region.polygon = {
            Vec2f{8.0f, 8.0f},
            Vec2f{40.0f, 8.0f},
            Vec2f{40.0f, 20.0f},
            Vec2f{8.0f, 20.0f},
        };

        RenderPlanOptions options;
        options.profile = PlannerProfile::Adaptive;
        options.allow_overflow = false;
        options.min_line_height_px = 2.0f;

        const RenderPlan plan =
            BuildRenderPlan(db, ImageRgba8(64, 32, 0x808080FFu), {region}, options);
        EXPECT(plan.fitted_regions == 1u);
        EXPECT(plan.fast_regions == 0u);
        EXPECT(plan.quality_regions == 1u);
        EXPECT(plan.escalated_regions == 1u);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
