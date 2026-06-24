#include "font_database.h"
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

struct PreviewOptions
{
    fs::path scene_path;
    fs::path atlas_dir;
    fs::path output_dir;
    Backend backend = Backend::Cpu;
    bool compare = false;
};

struct DiffStats
{
    std::size_t differing_pixels = 0;
    uint8_t max_channel_diff = 0;
};

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

static void PrintUsage()
{
    std::cerr
        << "Usage: render_scene_preview <scene.json> <atlas_dir> <output_dir>"
        << " [--backend cpu|cuda] [--compare]\n";
}

static Backend ParseBackend(const std::string& value)
{
    if (value == "cpu")
        return Backend::Cpu;
    if (value == "cuda")
        return Backend::Cuda;

    throw std::runtime_error("Unknown backend: " + value);
}

static PreviewOptions ParseArgs(int argc, char* argv[])
{
    if (argc < 4)
    {
        PrintUsage();
        throw std::runtime_error("Not enough arguments");
    }

    PreviewOptions opts;
    opts.scene_path = argv[1];
    opts.atlas_dir = argv[2];
    opts.output_dir = argv[3];

    for (int i = 4; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--compare")
        {
            opts.compare = true;
        }
        else if (arg == "--backend" && i + 1 < argc)
        {
            opts.backend = ParseBackend(argv[++i]);
        }
        else
        {
            PrintUsage();
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}

static ImageRgba8 CreateBaseImage(const RenderScene& scene, bool& loaded_source)
{
    loaded_source = false;

    try
    {
        ImageRgba8 image = LoadImageRgba8(scene.image_path);
        loaded_source = true;
        return image;
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

static ImageRgba8 MakeDiffImage(const ImageRgba8& a,
                                const ImageRgba8& b,
                                DiffStats& stats)
{
    if (a.width != b.width || a.height != b.height)
        throw std::runtime_error("MakeDiffImage: image sizes differ");

    ImageRgba8 diff(a.width, a.height, 0x000000FFu);
    stats = {};

    for (uint32_t y = 0; y < a.height; ++y)
    {
        for (uint32_t x = 0; x < a.width; ++x)
        {
            const uint8_t* pa = a.PixelPtr(x, y);
            const uint8_t* pb = b.PixelPtr(x, y);
            uint8_t* pd = diff.PixelPtr(x, y);

            uint8_t max_diff = 0;
            for (int c = 0; c < 4; ++c)
            {
                const uint8_t d = (uint8_t)std::abs((int)pa[c] - (int)pb[c]);
                max_diff = std::max(max_diff, d);
            }

            if (max_diff > 0)
                ++stats.differing_pixels;
            stats.max_channel_diff = std::max(stats.max_channel_diff, max_diff);

            pd[0] = max_diff;
            pd[1] = 0;
            pd[2] = 0;
            pd[3] = 255;
        }
    }

    return diff;
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
        const PreviewOptions opts = ParseArgs(argc, argv);
        const RenderScene scene = LoadRenderSceneJson(opts.scene_path);

        FontDatabase db;
        db.LoadDirectory(opts.atlas_dir);

        bool loaded_source = false;
        const ImageRgba8 base = CreateBaseImage(scene, loaded_source);
        const RenderPlan plan = BuildRenderPlan(db, base, scene.regions);

        fs::create_directories(opts.output_dir);

        std::cout << "Scene        : " << opts.scene_path << "\n";
        std::cout << "Atlas dir    : " << opts.atlas_dir << "\n";
        std::cout << "Source image : " << scene.image_path << "\n";
        std::cout << "Loaded src   : " << (loaded_source ? "yes" : "no") << "\n";
        std::cout << "Canvas       : " << base.width << "x" << base.height << "\n";
        std::cout << "Regions      : " << plan.fitted_regions << "/" << plan.total_regions << "\n";
        std::cout << "Glyphs       : " << plan.total_glyphs << "\n";

        if (opts.compare)
        {
            if (!HasCudaDevice())
                throw std::runtime_error("CUDA compare requested, but no CUDA device is visible");

            ImageRgba8 cpu_image = base;
            const Clock::time_point cpu_start = Clock::now();
            RenderPlanCpu(cpu_image, db, plan);
            const double cpu_ms = MillisecondsSince(cpu_start);

            ImageRgba8 gpu_image = base;
            const Clock::time_point gpu_start = Clock::now();
            const GpuRenderStats gpu_stats = RenderPlanCuda(gpu_image, db, plan);
            const double gpu_ms = MillisecondsSince(gpu_start);

            DiffStats diff_stats;
            const ImageRgba8 diff_image = MakeDiffImage(cpu_image, gpu_image, diff_stats);

            const fs::path cpu_path = opts.output_dir / "render_preview_cpu.png";
            const fs::path gpu_path = opts.output_dir / "render_preview_cuda.png";
            const fs::path diff_path = opts.output_dir / "render_preview_diff.png";
            const fs::path selected_path =
                opts.output_dir /
                (opts.backend == Backend::Cuda ? "render_preview.png" : "render_preview.png");

            WriteImagePng(cpu_image, cpu_path);
            WriteImagePng(gpu_image, gpu_path);
            WriteImagePng(diff_image, diff_path);
            WriteImagePng(opts.backend == Backend::Cuda ? gpu_image : cpu_image, selected_path);

            std::cout << "CPU output   : " << cpu_path << "\n";
            std::cout << "CUDA output  : " << gpu_path << "\n";
            std::cout << "Diff output  : " << diff_path << "\n";
            std::cout << "Selected out : " << selected_path << "\n";
            std::cout << "CPU ms       : " << cpu_ms << "\n";
            std::cout << "CUDA ms      : " << gpu_ms << "\n";
            std::cout << "CUDA device  : " << gpu_stats.device.name
                      << " (sm_" << gpu_stats.device.compute_major
                      << gpu_stats.device.compute_minor << ")\n";
            std::cout << "CUDA block   : " << gpu_stats.tuning.block_x
                      << "x" << gpu_stats.tuning.block_y << "\n";
            std::cout << "CUDA img up  : " << gpu_stats.image_upload_ms << " ms\n";
            std::cout << "CUDA cmd up  : " << gpu_stats.command_upload_ms << " ms\n";
            std::cout << "CUDA kernel  : " << gpu_stats.kernel_ms << " ms\n";
            std::cout << "CUDA img down: " << gpu_stats.image_download_ms << " ms\n";
            std::cout << "Diff pixels  : " << diff_stats.differing_pixels << "\n";
            std::cout << "Max diff     : " << (int)diff_stats.max_channel_diff << "\n";
            return 0;
        }

        ImageRgba8 image = base;
        const Clock::time_point start = Clock::now();

        if (opts.backend == Backend::Cuda)
        {
            if (!HasCudaDevice())
                throw std::runtime_error("CUDA backend requested, but no CUDA device is visible");

            const GpuRenderStats gpu_stats = RenderPlanCuda(image, db, plan);
            const double gpu_ms = MillisecondsSince(start);
            const fs::path out_path = opts.output_dir / "render_preview.png";
            WriteImagePng(image, out_path);

            std::cout << "Backend      : cuda\n";
            std::cout << "Output image : " << out_path << "\n";
            std::cout << "CUDA ms      : " << gpu_ms << "\n";
            std::cout << "CUDA device  : " << gpu_stats.device.name
                      << " (sm_" << gpu_stats.device.compute_major
                      << gpu_stats.device.compute_minor << ")\n";
            std::cout << "CUDA block   : " << gpu_stats.tuning.block_x
                      << "x" << gpu_stats.tuning.block_y << "\n";
            std::cout << "CUDA img up  : " << gpu_stats.image_upload_ms << " ms\n";
            std::cout << "CUDA cmd up  : " << gpu_stats.command_upload_ms << " ms\n";
            std::cout << "CUDA kernel  : " << gpu_stats.kernel_ms << " ms\n";
            std::cout << "CUDA img down: " << gpu_stats.image_download_ms << " ms\n";
            std::cout << "Batches      : " << gpu_stats.batches_rendered << "\n";
            return 0;
        }

        RenderPlanCpu(image, db, plan);
        const double cpu_ms = MillisecondsSince(start);
        const fs::path out_path = opts.output_dir / "render_preview.png";
        WriteImagePng(image, out_path);

        std::cout << "Backend      : cpu\n";
        std::cout << "Output image : " << out_path << "\n";
        std::cout << "CPU ms       : " << cpu_ms << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "render_scene_preview: " << e.what() << "\n";
        return 1;
    }
}
