#include "font_database.h"
#include "gpu_command_buffer.h"
#include "gpu_device.h"
#include "gpu_renderer_cuda.h"
#include "image_io.h"
#include "render_plan.h"
#include "render_scene.h"
#include "text_renderer_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace fac {

namespace {

using Clock = std::chrono::steady_clock;

struct Bounds2f
{
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
};

enum class Backend
{
    Cpu,
    Cuda,
};

struct BenchmarkOptions
{
    fs::path scene_path;
    fs::path atlas_dir;
    Backend backend = Backend::Cpu;
    int iterations = 50;
};

static void PrintUsage()
{
    std::cerr
        << "Usage: benchmark_render_scene <scene.json> <atlas_dir>"
        << " [--backend cpu|cuda] [--iterations N]\n";
}

static Backend ParseBackend(const std::string& value)
{
    if (value == "cpu") return Backend::Cpu;
    if (value == "cuda") return Backend::Cuda;
    throw std::runtime_error("Unknown backend: " + value);
}

static BenchmarkOptions ParseArgs(int argc, char* argv[])
{
    if (argc < 3)
    {
        PrintUsage();
        throw std::runtime_error("Not enough arguments");
    }

    BenchmarkOptions opts;
    opts.scene_path = argv[1];
    opts.atlas_dir = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc)
        {
            opts.backend = ParseBackend(argv[++i]);
        }
        else if (arg == "--iterations" && i + 1 < argc)
        {
            opts.iterations = std::max(1, std::stoi(argv[++i]));
        }
        else
        {
            PrintUsage();
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}

static Bounds2f ComputeSceneBounds(const std::vector<TextRegion>& regions)
{
    Bounds2f out{};
    bool init = false;

    for (const TextRegion& region : regions)
    {
        const OrientedBox& box = region.box;
        const float hw = 0.5f * box.width;
        const float hh = 0.5f * box.height;
        const float corners[4][2] = {
            {-hw, -hh},
            { hw, -hh},
            {-hw,  hh},
            { hw,  hh},
        };

        for (const auto& c : corners)
        {
            const Vec2f p = box.LocalToImage(c[0], c[1]);
            if (!init)
            {
                out.min_x = out.max_x = p.x;
                out.min_y = out.max_y = p.y;
                init = true;
            }
            else
            {
                out.min_x = std::min(out.min_x, p.x);
                out.min_y = std::min(out.min_y, p.y);
                out.max_x = std::max(out.max_x, p.x);
                out.max_y = std::max(out.max_y, p.y);
            }
        }
    }

    return out;
}

static ImageRgba8 CreateBaseImage(const RenderScene& scene)
{
    try
    {
        return LoadImageRgba8(scene.image_path);
    }
    catch (const std::exception&)
    {
        uint32_t width = 0;
        uint32_t height = 0;
        if (!TryReadImageDimensions(scene.image_path, width, height))
        {
            const Bounds2f bounds = ComputeSceneBounds(scene.regions);
            width = (uint32_t)std::max(1.0f, std::ceil(bounds.max_x));
            height = (uint32_t)std::max(1.0f, std::ceil(bounds.max_y));
        }
        return ImageRgba8(width, height, 0x202020FFu);
    }
}

static double MillisecondsSince(const Clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

} // namespace

} // namespace fac

int main(int argc, char* argv[])
{
    using namespace fac;

    try
    {
        const BenchmarkOptions opts = ParseArgs(argc, argv);
        const RenderScene scene = LoadRenderSceneJson(opts.scene_path);

        FontDatabase db;
        db.LoadDirectory(opts.atlas_dir);
        const ImageRgba8 base = CreateBaseImage(scene);
        const RenderPlan plan = BuildRenderPlan(db, base, scene.regions);

        std::cout << "Scene        : " << opts.scene_path << "\n";
        std::cout << "Atlas dir    : " << opts.atlas_dir << "\n";
        std::cout << "Canvas       : " << base.width << "x" << base.height << "\n";
        std::cout << "Regions      : " << plan.fitted_regions << "/" << plan.total_regions << "\n";
        std::cout << "Glyphs       : " << plan.total_glyphs << "\n";
        std::cout << "Iterations   : " << opts.iterations << "\n";

        if (opts.backend == Backend::Cuda)
        {
            if (!HasCudaDevice())
                throw std::runtime_error("CUDA backend requested, but no CUDA device is visible");

            const GpuCommandBuffer command_buffer = BuildGpuCommandBuffer(db, plan);

            const Clock::time_point atlas_start = Clock::now();
            GpuAtlasManager atlas_manager;
            atlas_manager.Upload(db);
            const double atlas_upload_ms = MillisecondsSince(atlas_start);

            GpuRendererCuda renderer;
            renderer.Initialize();

            double total_wall_ms = 0.0;
            double total_img_up_ms = 0.0;
            double total_cmd_up_ms = 0.0;
            double total_kernel_ms = 0.0;
            double total_img_down_ms = 0.0;
            GpuRenderStats last_stats;

            for (int i = 0; i < opts.iterations; ++i)
            {
                ImageRgba8 image = base;
                const Clock::time_point iter_start = Clock::now();
                last_stats = renderer.Render(image, atlas_manager, command_buffer);
                total_wall_ms += MillisecondsSince(iter_start);
                total_img_up_ms += last_stats.image_upload_ms;
                total_cmd_up_ms += last_stats.command_upload_ms;
                total_kernel_ms += last_stats.kernel_ms;
                total_img_down_ms += last_stats.image_download_ms;
            }

            std::cout << "Backend      : cuda\n";
            std::cout << "Device       : " << last_stats.device.name
                      << " (sm_" << last_stats.device.compute_major
                      << last_stats.device.compute_minor << ")\n";
            std::cout << "Block        : " << last_stats.tuning.block_x
                      << "x" << last_stats.tuning.block_y << "\n";
            std::cout << "Atlas upload : " << atlas_upload_ms << " ms (one-time)\n";
            std::cout << "Avg wall     : " << (total_wall_ms / opts.iterations) << " ms\n";
            std::cout << "Avg img up   : " << (total_img_up_ms / opts.iterations) << " ms\n";
            std::cout << "Avg cmd up   : " << (total_cmd_up_ms / opts.iterations) << " ms\n";
            std::cout << "Avg kernel   : " << (total_kernel_ms / opts.iterations) << " ms\n";
            std::cout << "Avg img down : " << (total_img_down_ms / opts.iterations) << " ms\n";
            return 0;
        }

        double total_wall_ms = 0.0;
        for (int i = 0; i < opts.iterations; ++i)
        {
            ImageRgba8 image = base;
            const Clock::time_point iter_start = Clock::now();
            RenderPlanCpu(image, db, plan);
            total_wall_ms += MillisecondsSince(iter_start);
        }

        std::cout << "Backend      : cpu\n";
        std::cout << "Avg wall     : " << (total_wall_ms / opts.iterations) << " ms\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "benchmark_render_scene: " << e.what() << "\n";
        return 1;
    }
}
