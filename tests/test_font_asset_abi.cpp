#include "font_asset.h"
#include "font_database.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

static void WriteTestAtlas(const std::filesystem::path& path,
                           uint32_t version,
                           uint32_t font_height_px,
                           uint16_t glyph_w,
                           uint16_t glyph_h)
{
    AtlasHeader hdr{};
    hdr.magic          = ATLAS_MAGIC;
    hdr.version        = version;
    hdr.width          = 4;
    hdr.height         = 4;
    hdr.glyph_count    = 1;
    hdr.pixel_offset   = sizeof(AtlasHeader) + sizeof(AtlasGlyphRecord);
    hdr.render_size    = 16;
    hdr.font_height_px = font_height_px;

    AtlasGlyphRecord rec{};
    rec.codepoint = 'A';
    rec.atlas_x   = 0;
    rec.atlas_y   = 0;
    rec.atlas_w   = glyph_w;
    rec.atlas_h   = glyph_h;
    rec.advance   = 8.0f;
    rec.bearing_x = 1.0f;
    rec.bearing_y = 2.0f;

    const std::vector<uint8_t> pixels = {
        255, 0,   0,   0,
        0,   255, 0,   0,
        0,   0,   255, 0,
        0,   0,   0,   255,
    };

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path.string());

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

int main()
{
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
    fs::create_directories(dir);

    SECTION("v1 atlases stay loadable");
    {
        const fs::path atlas_path = dir / "atlas_v1.bin";
        WriteTestAtlas(atlas_path, ATLAS_VERSION_1, 0, 2, 3);

        FontDatabase db;
        db.LoadAtlas(atlas_path);
        const AtlasEntry* entry = db.GetEntry(16);

        EXPECT(entry != nullptr);
        EXPECT(entry->font_height_px == 3u);
        EXPECT(entry->glyph_count == 1u);
        EXPECT(entry->HasGlyph('A'));
    }

    SECTION("v2 atlases expose exact font height");
    {
        const fs::path atlas_path = dir / "atlas_v2.bin";
        WriteTestAtlas(atlas_path, ATLAS_VERSION_2, 19, 2, 3);

        FontDatabase db;
        db.LoadAtlas(atlas_path);
        const AtlasEntry* entry = db.GetEntry(16);

        EXPECT(entry != nullptr);
        EXPECT(entry->font_height_px == 19u);
        EXPECT(entry->glyph_count == 1u);
        EXPECT(entry->GetGlyph('A').advance == 8.0f);
    }

    SECTION("unsupported version is rejected");
    {
        const fs::path atlas_path = dir / "atlas_bad.bin";
        WriteTestAtlas(atlas_path, 999u, 0, 2, 3);

        FontDatabase db;
        bool threw = false;
        try {
            db.LoadAtlas(atlas_path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        EXPECT(threw);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
