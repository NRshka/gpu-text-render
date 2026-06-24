#include "render_scene.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

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

static std::filesystem::path RepoRoot()
{
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

int main()
{
    namespace fs = std::filesystem;

    SECTION("Load repository sample");
    {
        const fs::path sample_path = RepoRoot() / "assets" / "sample.json";
        const RenderScene scene = LoadRenderSceneJson(sample_path);

        EXPECT(scene.image_path.filename() == "sample_001.png");
        EXPECT(scene.regions.size() == 9u);
        EXPECT(!scene.regions.empty());
        EXPECT(!scene.regions.front().text.empty());
        EXPECT(!scene.regions.front().original_text.empty());
        EXPECT(scene.regions.front().box.HasArea());
    }

    SECTION("Support obb spelling");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_obb.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "original_text": "test",
                    "rgba": 4294967295,
                    "obb": {
                        "cx": 10.0,
                        "cy": 20.0,
                        "ux": 1.0,
                        "uy": 0.0,
                        "vx": 0.0,
                        "vy": 1.0,
                        "width": 30.0,
                        "height": 15.0
                    }
                }
            ]
        })";
        f.close();

        const RenderScene scene = LoadRenderSceneJson(json_path);
        EXPECT(scene.regions.size() == 1u);
        EXPECT(scene.regions[0].text == "test");
        EXPECT(scene.regions[0].original_text == "test");
        EXPECT(scene.regions[0].has_explicit_rgba);
        EXPECT(scene.regions[0].rgba == 4294967295u);
        EXPECT(scene.regions[0].box.cx == 10.0f);
        EXPECT(scene.image_path.filename() == "img.png");
    }

    SECTION("Missing rgba enables auto color");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_no_rgba.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "original_text": "test",
                    "obb": {
                        "cx": 10.0,
                        "cy": 20.0,
                        "ux": 1.0,
                        "uy": 0.0,
                        "vx": 0.0,
                        "vy": 1.0,
                        "width": 30.0,
                        "height": 15.0
                    }
                }
            ]
        })";
        f.close();

        const RenderScene scene = LoadRenderSceneJson(json_path);
        EXPECT(scene.regions.size() == 1u);
        EXPECT(!scene.regions[0].has_explicit_rgba);
        EXPECT(scene.regions[0].rgba == 0xFFFFFFFFu);
    }

    SECTION("Polygon-only region loads");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_polygon.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "original_text": "orig",
                    "polygon": [[0, 0], [20, 0], [20, 10], [0, 10]]
                }
            ]
        })";
        f.close();

        const RenderScene scene = LoadRenderSceneJson(json_path);
        EXPECT(scene.regions.size() == 1u);
        EXPECT(scene.regions[0].has_polygon);
        EXPECT(scene.regions[0].polygon.size() == 4u);
        EXPECT(!scene.regions[0].has_curve);
    }

    SECTION("Curve-only region loads");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_curve.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "original_text": "orig",
                    "curve": {
                        "band_height": 18.0,
                        "normal_side": "left",
                        "segments": [
                            {
                                "p0": [0, 10],
                                "p1": [10, 10],
                                "p2": [20, 20],
                                "p3": [30, 20]
                            }
                        ]
                    }
                }
            ]
        })";
        f.close();

        const RenderScene scene = LoadRenderSceneJson(json_path);
        EXPECT(scene.regions.size() == 1u);
        EXPECT(scene.regions[0].has_curve);
        EXPECT(scene.regions[0].curve.segments.size() == 1u);
        EXPECT(scene.regions[0].curve.normal_side == CurveNormalSide::Left);
    }

    SECTION("Curve and polygon both load");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_curve_polygon.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "original_text": "orig",
                    "polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
                    "curve": {
                        "band_height": 18.0,
                        "segments": [
                            {
                                "p0": [0, 10],
                                "p1": [10, 10],
                                "p2": [20, 10],
                                "p3": [30, 10]
                            }
                        ]
                    },
                    "obb": {
                        "cx": 10.0,
                        "cy": 20.0,
                        "ux": 1.0,
                        "uy": 0.0,
                        "vx": 0.0,
                        "vy": 1.0,
                        "width": 30.0,
                        "height": 15.0
                    }
                }
            ]
        })";
        f.close();

        const RenderScene scene = LoadRenderSceneJson(json_path);
        EXPECT(scene.regions.size() == 1u);
        EXPECT(scene.regions[0].has_polygon);
        EXPECT(scene.regions[0].has_curve);
        EXPECT(scene.regions[0].box.width == 30.0f);
    }

    SECTION("Require original_text");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_missing_original_text.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "rgba": 4294967295,
                    "obb": {
                        "cx": 10.0,
                        "cy": 20.0,
                        "ux": 1.0,
                        "uy": 0.0,
                        "vx": 0.0,
                        "vy": 1.0,
                        "width": 30.0,
                        "height": 15.0
                    }
                }
            ]
        })";
        f.close();

        bool threw = false;
        try {
            (void)LoadRenderSceneJson(json_path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        EXPECT(threw);
    }

    SECTION("Reject malformed curve");
    {
        const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests";
        fs::create_directories(dir);

        const fs::path json_path = dir / "scene_bad_curve.json";
        std::ofstream f(json_path);
        if (!f)
            throw std::runtime_error("Cannot open " + json_path.string());

        f << R"({
            "image": "img.png",
            "regions": [
                {
                    "text": "test",
                    "original_text": "orig",
                    "curve": {
                        "band_height": 18.0,
                        "segments": []
                    }
                }
            ]
        })";
        f.close();

        bool threw = false;
        try {
            (void)LoadRenderSceneJson(json_path);
        } catch (const std::exception&) {
            threw = true;
        }
        EXPECT(threw);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
