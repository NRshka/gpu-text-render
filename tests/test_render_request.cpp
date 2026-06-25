#include "render_request.h"

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

static void WritePlannerTestAtlas(const std::filesystem::path& path)
{
    AtlasHeader hdr{};
    hdr.magic          = ATLAS_MAGIC;
    hdr.version        = ATLAS_VERSION_2;
    hdr.width          = 8;
    hdr.height         = 8;
    hdr.glyph_count    = 2;
    hdr.pixel_offset   = sizeof(AtlasHeader) + 2u * sizeof(AtlasGlyphRecord);
    hdr.render_size    = 16;
    hdr.font_height_px = 8;

    AtlasGlyphRecord records[2]{};

    records[0].codepoint = 'A';
    records[0].atlas_x   = 0;
    records[0].atlas_y   = 0;
    records[0].atlas_w   = 2;
    records[0].atlas_h   = 2;
    records[0].advance   = 3.0f;
    records[0].bearing_x = 0.0f;
    records[0].bearing_y = 2.0f;

    records[1].codepoint = ' ';
    records[1].atlas_x   = 0;
    records[1].atlas_y   = 0;
    records[1].atlas_w   = 0;
    records[1].atlas_h   = 0;
    records[1].advance   = 1.0f;
    records[1].bearing_x = 0.0f;
    records[1].bearing_y = 0.0f;

    std::vector<uint8_t> pixels(hdr.width * hdr.height, 0u);
    pixels[0] = 255u;
    pixels[1] = 255u;
    pixels[8] = 255u;
    pixels[9] = 255u;

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path.string());
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(records), sizeof(records));
    f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

int main()
{
    SECTION("Build polygon regions from tensorized request data");
    {
        PolygonRegionTensorData data;
        data.region_texts = {u8"hello", u8"გთხოვ"};
        data.region_original_texts = {u8"orig", u8"თხოვნა"};
        data.region_vertex_counts = {4, 3};
        data.region_vertices = {
            {0.0f, 0.0f},
            {4.0f, 0.0f},
            {4.0f, 2.0f},
            {0.0f, 2.0f},
            {10.0f, 10.0f},
            {14.0f, 11.0f},
            {12.0f, 14.0f},
        };
        data.region_has_rgba = {0u, 1u};
        data.region_rgba = {0u, 0xAABBCCDDu};

        const std::vector<TextRegion> regions = BuildTextRegionsFromPolygonTensorData(data);
        EXPECT(regions.size() == 2u);
        EXPECT(regions[0].text == u8"hello");
        EXPECT(regions[0].original_text == u8"orig");
        EXPECT(regions[0].has_polygon);
        EXPECT(regions[0].polygon.size() == 4u);
        EXPECT(!regions[0].has_explicit_rgba);
        EXPECT(regions[1].polygon.size() == 3u);
        EXPECT(regions[1].has_explicit_rgba);
        EXPECT(regions[1].rgba == 0xAABBCCDDu);
    }

    SECTION("Mismatched tensor lengths are rejected");
    {
        PolygonRegionTensorData data;
        data.region_texts = {"a"};
        data.region_original_texts = {"a"};
        data.region_vertex_counts = {3};
        data.region_vertices = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
        data.region_has_rgba = {};
        data.region_rgba = {0u};

        bool threw = false;
        try
        {
            (void)BuildTextRegionsFromPolygonTensorData(data);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        EXPECT(threw);
    }

    SECTION("Vertex count sum mismatch is rejected");
    {
        PolygonRegionTensorData data;
        data.region_texts = {"a"};
        data.region_original_texts = {"a"};
        data.region_vertex_counts = {4};
        data.region_vertices = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
        data.region_has_rgba = {0u};
        data.region_rgba = {0u};

        bool threw = false;
        try
        {
            (void)BuildTextRegionsFromPolygonTensorData(data);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        EXPECT(threw);
    }

    SECTION("Polygons with fewer than three points are rejected");
    {
        PolygonRegionTensorData data;
        data.region_texts = {"a"};
        data.region_original_texts = {"a"};
        data.region_vertex_counts = {2};
        data.region_vertices = {{0.0f, 0.0f}, {1.0f, 0.0f}};
        data.region_has_rgba = {0u};
        data.region_rgba = {0u};

        bool threw = false;
        try
        {
            (void)BuildTextRegionsFromPolygonTensorData(data);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        EXPECT(threw);
    }

    SECTION("Invalid UTF-8 is rejected");
    {
        PolygonRegionTensorData data;
        data.region_texts = {std::string("\xFF", 1)};
        data.region_original_texts = {"orig"};
        data.region_vertex_counts = {3};
        data.region_vertices = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
        data.region_has_rgba = {0u};
        data.region_rgba = {0u};

        bool threw = false;
        try
        {
            (void)BuildTextRegionsFromPolygonTensorData(data);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        EXPECT(threw);
    }

    SECTION("Empty-region request stays valid");
    {
        PolygonRegionTensorData data;
        const RgbRenderRequest request =
            BuildRgbRenderRequest(2, 1, std::vector<uint8_t>{1, 2, 3, 4, 5, 6}, data);
        EXPECT(request.HasValidImage());
        EXPECT(request.regions.empty());
        EXPECT(request.image_rgb.size() == 6u);
    }

    SECTION("Luma planner request preserves auto-color and serializes commands");
    {
        namespace fs = std::filesystem;

        const fs::path dir =
            fs::temp_directory_path() / "gpu_font_rendering_tests" / "render_request";
        fs::create_directories(dir);
        WritePlannerTestAtlas(dir / "atlas_16.bin");

        FontDatabase db;
        db.LoadDirectory(dir);

        PolygonRegionTensorData data;
        data.region_texts = {"A"};
        data.region_original_texts = {"A"};
        data.region_vertex_counts = {4};
        data.region_vertices = {
            {0.0f, 0.0f},
            {15.0f, 0.0f},
            {15.0f, 15.0f},
            {0.0f, 15.0f},
        };
        data.region_has_rgba = {0u};
        data.region_rgba = {0u};

        const LumaRenderRequest request =
            BuildLumaRenderRequest(16, 16, std::vector<uint8_t>(16u * 16u, 0u), data);
        EXPECT(request.HasValidImage());
        EXPECT(request.regions.size() == 1u);

        const PlannedGpuRenderRequest planned =
            BuildPlannedGpuRenderRequest(db, request, 3u);
        EXPECT(planned.width == 16u);
        EXPECT(planned.height == 16u);
        EXPECT(planned.image_index == 3u);
        EXPECT(!planned.command_bytes.empty());
        EXPECT(!planned.batch_bytes.empty());
        EXPECT(planned.command_buffer.commands.size() == 1u);
        EXPECT(planned.command_buffer.commands[0].image_index == 3u);
        EXPECT(planned.command_buffer.commands[0].rgba == 0xFFFFFFFFu);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
