#include "text_layout.h"
#include "text_renderer_cpu.h"

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

#define SECTION(name) std::cout << "\n[" << name << "]\n"

static void WriteRendererTestAtlas(const std::filesystem::path& path)
{
    AtlasHeader hdr{};
    hdr.magic          = ATLAS_MAGIC;
    hdr.version        = ATLAS_VERSION_2;
    hdr.width          = 8;
    hdr.height         = 8;
    hdr.glyph_count    = 3;
    hdr.pixel_offset   = sizeof(AtlasHeader) + 3u * sizeof(AtlasGlyphRecord);
    hdr.render_size    = 16;
    hdr.font_height_px = 8;

    AtlasGlyphRecord records[3]{};

    records[0].codepoint = 'A';
    records[0].atlas_x   = 0;
    records[0].atlas_y   = 0;
    records[0].atlas_w   = 2;
    records[0].atlas_h   = 2;
    records[0].advance   = 3.0f;
    records[0].bearing_x = 0.0f;
    records[0].bearing_y = 2.0f;

    records[1].codepoint = 0x10D0u; // Georgian ა
    records[1].atlas_x   = 3;
    records[1].atlas_y   = 0;
    records[1].atlas_w   = 2;
    records[1].atlas_h   = 2;
    records[1].advance   = 3.0f;
    records[1].bearing_x = 0.0f;
    records[1].bearing_y = 2.0f;

    records[2].codepoint = ' ';
    records[2].atlas_x   = 0;
    records[2].atlas_y   = 0;
    records[2].atlas_w   = 0;
    records[2].atlas_h   = 0;
    records[2].advance   = 1.0f;
    records[2].bearing_x = 0.0f;
    records[2].bearing_y = 0.0f;

    std::vector<uint8_t> pixels(hdr.width * hdr.height, 0);

    // 'A' coverage block.
    pixels[0] = 255;
    pixels[1] = 255;
    pixels[8] = 255;
    pixels[9] = 255;

    // Georgian 'ა' coverage block.
    pixels[3]  = 255;
    pixels[4]  = 255;
    pixels[11] = 255;
    pixels[12] = 255;

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

static bool HasExactPixel(const ImageRgba8& image,
                          uint32_t x0, uint32_t y0,
                          uint32_t x1, uint32_t y1,
                          uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    for (uint32_t y = y0; y <= y1 && y < image.height; ++y)
    {
        for (uint32_t x = x0; x <= x1 && x < image.width; ++x)
        {
            const uint8_t* p = image.PixelPtr(x, y);
            if (p[0] == r && p[1] == g && p[2] == b && p[3] == a)
                return true;
        }
    }
    return false;
}

static bool HasPartialAlphaPixel(const ImageRgba8& image,
                                 uint32_t x0, uint32_t y0,
                                 uint32_t x1, uint32_t y1)
{
    for (uint32_t y = y0; y <= y1 && y < image.height; ++y)
    {
        for (uint32_t x = x0; x <= x1 && x < image.width; ++x)
        {
            const uint8_t* p = image.PixelPtr(x, y);
            if (p[3] > 0 && p[3] < 255)
                return true;
        }
    }
    return false;
}

int main()
{
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
    fs::create_directories(dir);

    const fs::path atlas_path = dir / "atlas_renderer.bin";
    WriteRendererTestAtlas(atlas_path);

    FontDatabase db;
    db.LoadAtlas(atlas_path);
    const AtlasEntry* atlas = db.GetEntry(16);
    if (!atlas) {
        std::cerr << "Atlas failed to load\n";
        return EXIT_FAILURE;
    }

    SECTION("Mixed English and Georgian rendering");
    {
        OrientedBox box;
        box.cx = 8.0f;
        box.cy = 8.0f;
        box.width = 8.0f;
        box.height = 8.0f;

        const std::string text = u8"Aა";
        const TextFitResult fit = FitTextToBox(db, text, box);
        EXPECT(fit.IsValid());

        const std::vector<GlyphPlacement> placements =
            LayoutTextLine(*atlas, text, fit);
        EXPECT(placements.size() == 2u);

        std::vector<RenderGlyph> commands;
        commands.reserve(placements.size());
        for (const GlyphPlacement& p : placements)
            commands.push_back(MakeRenderGlyph(p, box, 0xFF0000FFu));

        ImageRgba8 image(16, 16, 0x000000FFu);
        RenderGlyphsCpu(image, *atlas, commands);

        EXPECT(CountChangedPixels(image) >= 8);
        EXPECT(HasExactPixel(image, 3, 5, 6, 10, 255, 0, 0, 255));
        EXPECT(HasExactPixel(image, 8, 5, 11, 10, 255, 0, 0, 255));
    }

    SECTION("Rotated rendering and alpha blending");
    {
        OrientedBox box;
        box.cx = 10.0f;
        box.cy = 10.0f;
        box.ux = 0.0f;
        box.uy = 1.0f;
        box.vx = -1.0f;
        box.vy = 0.0f;
        box.width = 6.0f;
        box.height = 6.0f;

        const TextFitResult fit = FitTextToBox(db, "A", box);
        EXPECT(fit.IsValid());

        const std::vector<GlyphPlacement> placements =
            LayoutTextLine(*atlas, "A", fit);
        EXPECT(placements.size() == 1u);

        const RenderGlyph cmd = MakeRenderGlyph(placements[0], box, 0x00FF0080u);
        ImageRgba8 image(20, 20, 0x0000FFFFu);
        RenderGlyphsCpu(image, *atlas, std::vector<RenderGlyph>{cmd});

        const int changed = CountChangedPixels(image);
        EXPECT(changed > 0);

        bool found_blended = false;
        for (uint32_t y = 0; y < image.height && !found_blended; ++y)
        {
            for (uint32_t x = 0; x < image.width; ++x)
            {
                const uint8_t* p = image.PixelPtr(x, y);
                if (p[1] > 0 && p[2] > 0 && p[1] < 255 && p[2] < 255)
                {
                    found_blended = true;
                    break;
                }
            }
        }
        EXPECT(found_blended);
    }

    SECTION("Magnified glyph edges are smoothly filtered");
    {
        RenderGlyph cmd;
        cmd.atlas_codepoint = 'A';
        cmd.origin_x = 2.0f;
        cmd.origin_y = 2.0f;
        cmd.scale = 4.0f;
        cmd.rgba = 0xFFFFFFFFu;

        ImageRgba8 image(16, 16, 0x00000000u);
        RenderGlyphsCpu(image, *atlas, std::vector<RenderGlyph>{cmd});

        EXPECT(HasExactPixel(image, 4, 4, 7, 7, 255, 255, 255, 255));
        EXPECT(HasPartialAlphaPixel(image, 2, 2, 3, 3));
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
