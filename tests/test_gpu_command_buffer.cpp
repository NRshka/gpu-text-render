#include "gpu_command_buffer.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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
                       uint16_t glyph_w,
                       uint16_t glyph_h)
{
    AtlasHeader hdr{};
    hdr.magic          = ATLAS_MAGIC;
    hdr.version        = ATLAS_VERSION_2;
    hdr.width          = 16;
    hdr.height         = 16;
    hdr.glyph_count    = 2;
    hdr.pixel_offset   = sizeof(AtlasHeader) + 2u * sizeof(AtlasGlyphRecord);
    hdr.render_size    = render_size;
    hdr.font_height_px = 12;

    AtlasGlyphRecord records[2]{};

    records[0].codepoint = 'A';
    records[0].atlas_x   = 1;
    records[0].atlas_y   = 2;
    records[0].atlas_w   = glyph_w;
    records[0].atlas_h   = glyph_h;
    records[0].advance   = 4.0f;
    records[0].bearing_x = 0.0f;
    records[0].bearing_y = (float)glyph_h;

    records[1].codepoint = 0x10D0u;
    records[1].atlas_x   = 5;
    records[1].atlas_y   = 6;
    records[1].atlas_w   = glyph_w;
    records[1].atlas_h   = glyph_h;
    records[1].advance   = 4.0f;
    records[1].bearing_x = 0.0f;
    records[1].bearing_y = (float)glyph_h;

    std::vector<uint8_t> pixels(hdr.width * hdr.height, 0);
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

    const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests" / "gpu_command_buffer";
    fs::create_directories(dir);

    WriteAtlas(dir / "atlas_16.bin", 16, 2, 3);
    WriteAtlas(dir / "atlas_32.bin", 32, 4, 5);

    FontDatabase db;
    db.LoadDirectory(dir);

    SECTION("Pack render plan into GPU command buffer");
    {
        RenderPlan plan;
        plan.total_regions = 2;
        plan.fitted_regions = 2;
        plan.total_glyphs = 2;

        RenderBatch batch16;
        batch16.atlas_render_size = 16;
        RenderGlyph g0;
        g0.atlas_codepoint = 'A';
        g0.origin_x = 10.0f;
        g0.origin_y = 20.0f;
        g0.basis_ux = 1.0f;
        g0.basis_uy = 0.0f;
        g0.basis_vx = 0.0f;
        g0.basis_vy = 1.0f;
        g0.scale = 2.0f;
        g0.rgba = 0xAABBCCDDu;
        batch16.glyphs.push_back(g0);

        RenderBatch batch32;
        batch32.atlas_render_size = 32;
        RenderGlyph g1;
        g1.atlas_codepoint = 0x10D0u;
        g1.origin_x = 30.0f;
        g1.origin_y = 40.0f;
        g1.basis_ux = 0.0f;
        g1.basis_uy = 1.0f;
        g1.basis_vx = -1.0f;
        g1.basis_vy = 0.0f;
        g1.scale = 0.5f;
        g1.rgba = 0x11223344u;
        batch32.glyphs.push_back(g1);

        plan.batches.push_back(batch16);
        plan.batches.push_back(batch32);

        const GpuCommandBuffer gpu = BuildGpuCommandBuffer(db, plan);
        EXPECT(gpu.total_regions == 2u);
        EXPECT(gpu.total_glyphs == 2u);
        EXPECT(gpu.batches.size() == 2u);
        EXPECT(gpu.commands.size() == 2u);

        EXPECT(gpu.batches[0].atlas_render_size == 16u);
        EXPECT(gpu.batches[0].atlas_width == 16u);
        EXPECT(gpu.batches[0].atlas_height == 16u);
        EXPECT(gpu.batches[0].command_offset == 0u);
        EXPECT(gpu.batches[0].command_count == 1u);

        EXPECT(gpu.batches[1].atlas_render_size == 32u);
        EXPECT(gpu.batches[1].command_offset == 1u);
        EXPECT(gpu.batches[1].command_count == 1u);

        EXPECT_NEAR(gpu.commands[0].origin_x, 10.0f, 1e-6f);
        EXPECT_NEAR(gpu.commands[0].origin_y, 20.0f, 1e-6f);
        EXPECT_NEAR(gpu.commands[0].m00, 2.0f, 1e-6f);
        EXPECT_NEAR(gpu.commands[0].m11, 2.0f, 1e-6f);
        EXPECT(gpu.commands[0].rgba == 0xAABBCCDDu);
        EXPECT(gpu.commands[0].atlas_x == 1u);
        EXPECT(gpu.commands[0].atlas_y == 2u);
        EXPECT(gpu.commands[0].atlas_w == 2u);
        EXPECT(gpu.commands[0].atlas_h == 3u);

        EXPECT_NEAR(gpu.commands[1].m00, 0.0f, 1e-6f);
        EXPECT_NEAR(gpu.commands[1].m01, -0.5f, 1e-6f);
        EXPECT_NEAR(gpu.commands[1].m10, 0.5f, 1e-6f);
        EXPECT_NEAR(gpu.commands[1].m11, 0.0f, 1e-6f);
        EXPECT(gpu.commands[1].atlas_x == 5u);
        EXPECT(gpu.commands[1].atlas_y == 6u);
        EXPECT(gpu.commands[1].atlas_w == 4u);
        EXPECT(gpu.commands[1].atlas_h == 5u);
    }

    SECTION("Serialize and combine batched GPU command buffer v2");
    {
        RenderPlan plan_a;
        plan_a.total_regions = 1;
        plan_a.fitted_regions = 1;
        plan_a.total_glyphs = 1;

        RenderBatch batch_a;
        batch_a.atlas_render_size = 16;
        RenderGlyph glyph_a;
        glyph_a.atlas_codepoint = 'A';
        glyph_a.origin_x = 11.0f;
        glyph_a.origin_y = 21.0f;
        glyph_a.basis_ux = 1.0f;
        glyph_a.basis_uy = 0.0f;
        glyph_a.basis_vx = 0.0f;
        glyph_a.basis_vy = 1.0f;
        glyph_a.scale = 1.5f;
        glyph_a.rgba = 0x01020304u;
        batch_a.glyphs.push_back(glyph_a);
        plan_a.batches.push_back(batch_a);

        RenderPlan plan_b;
        plan_b.total_regions = 1;
        plan_b.fitted_regions = 1;
        plan_b.total_glyphs = 1;

        RenderBatch batch_b;
        batch_b.atlas_render_size = 16;
        RenderGlyph glyph_b;
        glyph_b.atlas_codepoint = 'A';
        glyph_b.origin_x = 31.0f;
        glyph_b.origin_y = 41.0f;
        glyph_b.basis_ux = 0.0f;
        glyph_b.basis_uy = 1.0f;
        glyph_b.basis_vx = -1.0f;
        glyph_b.basis_vy = 0.0f;
        glyph_b.scale = 0.75f;
        glyph_b.rgba = 0xA0B0C0D0u;
        batch_b.glyphs.push_back(glyph_b);
        plan_b.batches.push_back(batch_b);

        const GpuCommandBufferV2 combined =
            BuildCombinedGpuCommandBufferV2(db, {plan_a, plan_b});
        EXPECT(combined.total_images == 2u);
        EXPECT(combined.total_regions == 2u);
        EXPECT(combined.total_glyphs == 2u);
        EXPECT(combined.batches.size() == 1u);
        EXPECT(combined.batches[0].atlas_render_size == 16u);
        EXPECT(combined.batches[0].command_offset == 0u);
        EXPECT(combined.batches[0].command_count == 2u);
        EXPECT(combined.commands.size() == 2u);
        EXPECT(combined.commands[0].image_index == 0u);
        EXPECT(combined.commands[1].image_index == 1u);

        const std::vector<uint8_t> command_bytes =
            SerializeGpuRenderCommandsV2(combined.commands);
        const std::vector<uint8_t> batch_bytes =
            SerializeGpuRenderBatchesV2(combined.batches);
        const GpuCommandBufferV2 roundtrip = DeserializeGpuCommandBufferV2(
            command_bytes.data(),
            command_bytes.size(),
            batch_bytes.data(),
            batch_bytes.size(),
            combined.total_images);

        EXPECT(roundtrip.total_images == 2u);
        EXPECT(roundtrip.total_glyphs == 2u);
        EXPECT(roundtrip.batches.size() == 1u);
        EXPECT(roundtrip.commands.size() == 2u);
        EXPECT(roundtrip.commands[0].image_index == 0u);
        EXPECT(roundtrip.commands[1].image_index == 1u);
        EXPECT_NEAR(roundtrip.commands[0].m00, 1.5f, 1e-6f);
        EXPECT_NEAR(roundtrip.commands[1].m01, -0.75f, 1e-6f);
    }

    SECTION("Combine prebuilt GPU command buffers v2");
    {
        RenderPlan plan_a;
        plan_a.total_regions = 1;
        plan_a.fitted_regions = 1;
        plan_a.total_glyphs = 1;

        RenderBatch batch_a;
        batch_a.atlas_render_size = 16;
        RenderGlyph glyph_a;
        glyph_a.atlas_codepoint = 'A';
        glyph_a.origin_x = 7.0f;
        glyph_a.origin_y = 9.0f;
        glyph_a.basis_ux = 1.0f;
        glyph_a.basis_uy = 0.0f;
        glyph_a.basis_vx = 0.0f;
        glyph_a.basis_vy = 1.0f;
        glyph_a.scale = 1.0f;
        glyph_a.rgba = 0x55667788u;
        batch_a.glyphs.push_back(glyph_a);
        plan_a.batches.push_back(batch_a);

        RenderPlan plan_b;
        plan_b.total_regions = 1;
        plan_b.fitted_regions = 1;
        plan_b.total_glyphs = 1;

        RenderBatch batch_b;
        batch_b.atlas_render_size = 32;
        RenderGlyph glyph_b;
        glyph_b.atlas_codepoint = 0x10D0u;
        glyph_b.origin_x = 17.0f;
        glyph_b.origin_y = 19.0f;
        glyph_b.basis_ux = 1.0f;
        glyph_b.basis_uy = 0.0f;
        glyph_b.basis_vx = 0.0f;
        glyph_b.basis_vy = 1.0f;
        glyph_b.scale = 2.0f;
        glyph_b.rgba = 0x99AABBCCu;
        batch_b.glyphs.push_back(glyph_b);
        plan_b.batches.push_back(batch_b);

        const GpuCommandBufferV2 single_a = BuildGpuCommandBufferV2(db, plan_a, 0);
        const GpuCommandBufferV2 single_b = BuildGpuCommandBufferV2(db, plan_b, 0);
        const GpuCommandBufferV2 combined =
            CombineGpuCommandBuffersV2({single_a, single_b});

        EXPECT(combined.total_images == 2u);
        EXPECT(combined.total_regions == 2u);
        EXPECT(combined.total_glyphs == 2u);
        EXPECT(combined.batches.size() == 2u);
        EXPECT(combined.commands.size() == 2u);
        EXPECT(combined.commands[0].image_index == 0u);
        EXPECT(combined.commands[1].image_index == 1u);
        EXPECT(combined.batches[0].command_count == 1u);
        EXPECT(combined.batches[1].command_count == 1u);
        EXPECT(combined.batches[0].atlas_render_size == 16u);
        EXPECT(combined.batches[1].atlas_render_size == 32u);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
