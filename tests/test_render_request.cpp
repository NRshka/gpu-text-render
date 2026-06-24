#include "render_request.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
