#include "gpu_renderer_cuda.h"
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

    records[1].codepoint = 0x10D0u;
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
    pixels[0] = 255; pixels[1] = 255; pixels[8] = 255; pixels[9] = 255;
    pixels[3] = 255; pixels[4] = 255; pixels[11] = 255; pixels[12] = 255;

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

    SECTION("GPU render matches CPU for simple scene");
    {
        if (!HasCudaDevice())
        {
            std::cout << "  No CUDA device available, skipping\n";
            EXPECT(true);
        }
        else
        {
            const fs::path dir =
                fs::temp_directory_path() / "gpu_font_rendering_tests" / "gpu_renderer_cuda";
            fs::create_directories(dir);

            const fs::path atlas_path = dir / "atlas_16.bin";
            WriteRendererTestAtlas(atlas_path);

            FontDatabase db;
            db.LoadAtlas(atlas_path);

            TextRegion region;
            region.text = u8"Aა";
            region.original_text = u8"Aა";
            region.rgba = 0xFF0000FFu;
            region.box.cx = 8.0f;
            region.box.cy = 8.0f;
            region.box.width = 8.0f;
            region.box.height = 8.0f;

            const ImageRgba8 base(16, 16, 0x000000FFu);
            const RenderPlan plan = BuildRenderPlan(db, base, std::vector<TextRegion>{region});
            EXPECT(plan.total_glyphs == 2u);

            ImageRgba8 cpu_image = base;
            RenderPlanCpu(cpu_image, db, plan);

            ImageRgba8 gpu_image = base;
            const GpuRenderStats stats = RenderPlanCuda(gpu_image, db, plan);

            EXPECT(stats.glyphs_rendered == 2u);
            EXPECT(stats.batches_rendered == 1u);
            EXPECT(gpu_image.pixels == cpu_image.pixels);
        }
    }

    SECTION("GPU render matches CPU for magnified glyph");
    {
        if (!HasCudaDevice())
        {
            std::cout << "  No CUDA device available, skipping\n";
            EXPECT(true);
        }
        else
        {
            const fs::path dir =
                fs::temp_directory_path() / "gpu_font_rendering_tests" / "gpu_renderer_cuda_large";
            fs::create_directories(dir);

            const fs::path atlas_path = dir / "atlas_16.bin";
            WriteRendererTestAtlas(atlas_path);

            FontDatabase db;
            db.LoadAtlas(atlas_path);

            TextRegion region;
            region.text = "A";
            region.original_text = "A";
            region.rgba = 0xFFFFFFFFu;
            region.box.cx = 16.0f;
            region.box.cy = 16.0f;
            region.box.width = 24.0f;
            region.box.height = 24.0f;

            const ImageRgba8 base(32, 32, 0x00000000u);
            const RenderPlan plan = BuildRenderPlan(db, base, std::vector<TextRegion>{region});
            EXPECT(plan.total_glyphs == 1u);

            ImageRgba8 cpu_image = base;
            RenderPlanCpu(cpu_image, db, plan);

            ImageRgba8 gpu_image = base;
            const GpuRenderStats stats = RenderPlanCuda(gpu_image, db, plan);

            EXPECT(stats.glyphs_rendered == 1u);
            EXPECT(stats.batches_rendered == 1u);
            EXPECT(gpu_image.pixels == cpu_image.pixels);
        }
    }

    SECTION("GPU render matches CPU for curved text");
    {
        if (!HasCudaDevice())
        {
            std::cout << "  No CUDA device available, skipping\n";
            EXPECT(true);
        }
        else
        {
            const fs::path dir =
                fs::temp_directory_path() / "gpu_font_rendering_tests" / "gpu_renderer_cuda_curve";
            fs::create_directories(dir);

            const fs::path atlas_path = dir / "atlas_16.bin";
            WriteRendererTestAtlas(atlas_path);

            FontDatabase db;
            db.LoadAtlas(atlas_path);

            TextRegion region;
            region.text = u8"Aა";
            region.original_text = u8"Aა";
            region.rgba = 0xFF0000FFu;
            region.has_curve = true;
            region.curve.band_height = 16.0f;

            CubicBezierSegment segment;
            segment.p0 = Vec2f{4.0f, 12.0f};
            segment.p1 = Vec2f{10.0f, 8.0f};
            segment.p2 = Vec2f{18.0f, 8.0f};
            segment.p3 = Vec2f{24.0f, 12.0f};
            region.curve.segments.push_back(segment);

            const ImageRgba8 base(32, 24, 0x000000FFu);
            const RenderPlan plan = BuildRenderPlan(db, base, std::vector<TextRegion>{region});
            EXPECT(plan.total_glyphs == 2u);

            ImageRgba8 cpu_image = base;
            RenderPlanCpu(cpu_image, db, plan);

            ImageRgba8 gpu_image = base;
            const GpuRenderStats stats = RenderPlanCuda(gpu_image, db, plan);

            EXPECT(stats.glyphs_rendered == 2u);
            EXPECT(stats.batches_rendered == 1u);
            EXPECT(gpu_image.pixels == cpu_image.pixels);
        }
    }

    SECTION("GPU RGB pipeline matches CPU RGB result");
    {
        if (!HasCudaDevice())
        {
            std::cout << "  No CUDA device available, skipping\n";
            EXPECT(true);
        }
        else
        {
            const fs::path dir =
                fs::temp_directory_path() / "gpu_font_rendering_tests" / "gpu_renderer_cuda_rgb";
            fs::create_directories(dir);

            const fs::path atlas_path = dir / "atlas_16.bin";
            WriteRendererTestAtlas(atlas_path);

            FontDatabase db;
            db.LoadAtlas(atlas_path);

            TextRegion region;
            region.text = u8"Aა";
            region.original_text = u8"Aა";
            region.rgba = 0x00FF00FFu;
            region.box.cx = 8.0f;
            region.box.cy = 8.0f;
            region.box.width = 8.0f;
            region.box.height = 8.0f;

            const ImageRgba8 base_rgba(16, 16, 0x102030FFu);
            const RenderPlan plan = BuildRenderPlan(db, base_rgba, std::vector<TextRegion>{region});

            ImageRgba8 cpu_rgba = base_rgba;
            RenderPlanCpu(cpu_rgba, db, plan);

            std::vector<uint8_t> rgb_in(base_rgba.width * base_rgba.height * 3u, 0u);
            std::vector<uint8_t> rgb_out(rgb_in.size(), 0u);
            std::vector<uint8_t> rgb_cpu(rgb_in.size(), 0u);

            for (uint32_t y = 0; y < base_rgba.height; ++y)
            {
                for (uint32_t x = 0; x < base_rgba.width; ++x)
                {
                    const std::size_t rgb_idx =
                        ((std::size_t)y * base_rgba.width + x) * 3u;
                    const uint8_t* base_pixel = base_rgba.PixelPtr(x, y);
                    const uint8_t* cpu_pixel = cpu_rgba.PixelPtr(x, y);
                    rgb_in[rgb_idx + 0u] = base_pixel[0];
                    rgb_in[rgb_idx + 1u] = base_pixel[1];
                    rgb_in[rgb_idx + 2u] = base_pixel[2];
                    rgb_cpu[rgb_idx + 0u] = cpu_pixel[0];
                    rgb_cpu[rgb_idx + 1u] = cpu_pixel[1];
                    rgb_cpu[rgb_idx + 2u] = cpu_pixel[2];
                }
            }

            const GpuRenderStats stats = RenderPlanCudaRgb(rgb_out.data(),
                                                           GpuBufferMemoryType::Host,
                                                           rgb_in.data(),
                                                           GpuBufferMemoryType::Host,
                                                           base_rgba.width,
                                                           base_rgba.height,
                                                           db,
                                                           plan);

            EXPECT(stats.glyphs_rendered == 2u);
            EXPECT(rgb_out == rgb_cpu);
        }
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
