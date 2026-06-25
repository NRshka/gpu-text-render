#include "gpu_command_buffer.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>


namespace fac {

namespace {

static_assert(std::is_trivially_copyable<GpuRenderCommandV2>::value,
              "GpuRenderCommandV2 must remain trivially copyable");
static_assert(std::is_trivially_copyable<GpuRenderBatchV2>::value,
              "GpuRenderBatchV2 must remain trivially copyable");

template <typename T>
std::vector<uint8_t> SerializePodVector(const std::vector<T>& values)
{
    if (values.empty())
        return {};

    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template <typename T>
std::vector<T> DeserializePodVector(const uint8_t* bytes,
                                    std::size_t byte_size,
                                    const char* label)
{
    if (byte_size == 0)
        return {};

    if (bytes == nullptr)
    {
        throw std::runtime_error(std::string(label) + ": bytes pointer is null");
    }

    if (byte_size % sizeof(T) != 0)
    {
        throw std::runtime_error(std::string(label) + ": byte size is misaligned");
    }

    std::vector<T> values(byte_size / sizeof(T));
    std::memcpy(values.data(), bytes, byte_size);
    return values;
}

GpuRenderCommand MakeGpuRenderCommand(const GlyphInfo& info,
                                      const RenderGlyph& glyph)
{
    GpuRenderCommand cmd;
    cmd.origin_x = glyph.origin_x;
    cmd.origin_y = glyph.origin_y;
    cmd.m00 = glyph.scale * glyph.basis_ux;
    cmd.m01 = glyph.scale * glyph.basis_vx;
    cmd.m10 = glyph.scale * glyph.basis_uy;
    cmd.m11 = glyph.scale * glyph.basis_vy;
    cmd.rgba = glyph.rgba;
    cmd.atlas_x = info.atlas_x;
    cmd.atlas_y = info.atlas_y;
    cmd.atlas_w = info.atlas_w;
    cmd.atlas_h = info.atlas_h;
    return cmd;
}

GpuRenderCommandV2 MakeGpuRenderCommandV2(const GlyphInfo& info,
                                          const RenderGlyph& glyph,
                                          uint32_t image_index)
{
    GpuRenderCommandV2 cmd;
    cmd.origin_x = glyph.origin_x;
    cmd.origin_y = glyph.origin_y;
    cmd.m00 = glyph.scale * glyph.basis_ux;
    cmd.m01 = glyph.scale * glyph.basis_vx;
    cmd.m10 = glyph.scale * glyph.basis_uy;
    cmd.m11 = glyph.scale * glyph.basis_vy;
    cmd.rgba = glyph.rgba;
    cmd.image_index = image_index;
    cmd.atlas_x = info.atlas_x;
    cmd.atlas_y = info.atlas_y;
    cmd.atlas_w = info.atlas_w;
    cmd.atlas_h = info.atlas_h;
    return cmd;
}

} // namespace

GpuCommandBuffer BuildGpuCommandBuffer(const FontDatabase& db,
                                       const RenderPlan& plan)
{
    GpuCommandBuffer out;
    out.total_regions = plan.fitted_regions;
    out.total_glyphs = plan.total_glyphs;
    out.commands.reserve(plan.total_glyphs);
    out.batches.reserve(plan.batches.size());

    for (const RenderBatch& batch : plan.batches)
    {
        const AtlasEntry* atlas = db.GetEntry(batch.atlas_render_size);
        if (!atlas)
        {
            throw std::runtime_error("BuildGpuCommandBuffer: missing atlas for render size "
                                     + std::to_string(batch.atlas_render_size));
        }

        GpuRenderBatch gpu_batch;
        gpu_batch.atlas_render_size = batch.atlas_render_size;
        gpu_batch.atlas_width = atlas->atlas_width;
        gpu_batch.atlas_height = atlas->atlas_height;
        gpu_batch.command_offset = (uint32_t)out.commands.size();
        gpu_batch.command_count = (uint32_t)batch.glyphs.size();

        out.commands.reserve(out.commands.size() + batch.glyphs.size());

        for (const RenderGlyph& glyph : batch.glyphs)
        {
            const GlyphInfo& info = atlas->GetGlyph(glyph.atlas_codepoint);
            out.commands.push_back(MakeGpuRenderCommand(info, glyph));
        }

        out.batches.push_back(gpu_batch);
    }

    return out;
}

GpuCommandBufferV2 BuildGpuCommandBufferV2(const FontDatabase& db,
                                           const RenderPlan& plan,
                                           uint32_t image_index)
{
    GpuCommandBufferV2 out;
    out.total_regions = plan.fitted_regions;
    out.total_glyphs = plan.total_glyphs;
    out.total_images = 1;
    out.commands.reserve(plan.total_glyphs);
    out.batches.reserve(plan.batches.size());

    for (const RenderBatch& batch : plan.batches)
    {
        const AtlasEntry* atlas = db.GetEntry(batch.atlas_render_size);
        if (!atlas)
        {
            throw std::runtime_error("BuildGpuCommandBufferV2: missing atlas for render size "
                                     + std::to_string(batch.atlas_render_size));
        }

        GpuRenderBatchV2 gpu_batch;
        gpu_batch.atlas_render_size = batch.atlas_render_size;
        gpu_batch.atlas_width = atlas->atlas_width;
        gpu_batch.atlas_height = atlas->atlas_height;
        gpu_batch.command_offset = (uint32_t)out.commands.size();
        gpu_batch.command_count = (uint32_t)batch.glyphs.size();

        out.commands.reserve(out.commands.size() + batch.glyphs.size());
        for (const RenderGlyph& glyph : batch.glyphs)
        {
            const GlyphInfo& info = atlas->GetGlyph(glyph.atlas_codepoint);
            out.commands.push_back(MakeGpuRenderCommandV2(info, glyph, image_index));
        }

        out.batches.push_back(gpu_batch);
    }
    return out;
}

GpuCommandBufferV2 BuildCombinedGpuCommandBufferV2(
    const FontDatabase& db,
    const std::vector<RenderPlan>& plans)
{
    std::vector<GpuCommandBufferV2> buffers;
    buffers.reserve(plans.size());
    for (const RenderPlan& plan : plans)
    {
        buffers.push_back(BuildGpuCommandBufferV2(db, plan, 0));
    }

    return CombineGpuCommandBuffersV2(buffers);
}

GpuCommandBufferV2 CombineGpuCommandBuffersV2(
    const std::vector<GpuCommandBufferV2>& buffers)
{
    struct BatchBucket
    {
        GpuRenderBatchV2 batch;
        std::vector<GpuRenderCommandV2> commands;
    };

    GpuCommandBufferV2 out;
    uint32_t image_base = 0;

    std::vector<BatchBucket> buckets;
    std::vector<uint32_t> atlas_order;
    std::unordered_map<uint32_t, std::size_t> atlas_to_bucket;

    for (const GpuCommandBufferV2& buffer : buffers)
    {
        ValidateGpuCommandBufferV2(buffer, buffer.total_images);
        out.total_images += buffer.total_images;
        out.total_regions += buffer.total_regions;
        out.total_glyphs += buffer.total_glyphs;

        for (const GpuRenderBatchV2& batch : buffer.batches)
        {
            auto it = atlas_to_bucket.find(batch.atlas_render_size);
            if (it == atlas_to_bucket.end())
            {
                const std::size_t bucket_index = buckets.size();
                atlas_to_bucket.emplace(batch.atlas_render_size, bucket_index);
                atlas_order.push_back(batch.atlas_render_size);
                buckets.push_back(BatchBucket{batch, {}});
                buckets.back().batch.command_offset = 0;
                buckets.back().batch.command_count = 0;
                it = atlas_to_bucket.find(batch.atlas_render_size);
            }

            BatchBucket& bucket = buckets[it->second];
            const std::size_t begin = batch.command_offset;
            const std::size_t end = begin + batch.command_count;
            bucket.commands.reserve(bucket.commands.size() + batch.command_count);
            for (std::size_t command_index = begin; command_index < end; ++command_index)
            {
                GpuRenderCommandV2 command = buffer.commands[command_index];
                command.image_index += image_base;
                bucket.commands.push_back(command);
            }
        }

        image_base += buffer.total_images;
    }

    out.batches.reserve(atlas_order.size());
    out.commands.reserve(out.total_glyphs);
    for (uint32_t atlas_render_size : atlas_order)
    {
        BatchBucket& bucket = buckets[atlas_to_bucket[atlas_render_size]];
        bucket.batch.command_offset = (uint32_t)out.commands.size();
        bucket.batch.command_count = (uint32_t)bucket.commands.size();
        out.commands.insert(out.commands.end(), bucket.commands.begin(), bucket.commands.end());
        out.batches.push_back(bucket.batch);
    }

    ValidateGpuCommandBufferV2(out, out.total_images);
    return out;
}

std::vector<uint8_t> SerializeGpuRenderCommandsV2(
    const std::vector<GpuRenderCommandV2>& commands)
{
    return SerializePodVector(commands);
}

std::vector<uint8_t> SerializeGpuRenderBatchesV2(
    const std::vector<GpuRenderBatchV2>& batches)
{
    return SerializePodVector(batches);
}

std::vector<GpuRenderCommandV2> DeserializeGpuRenderCommandsV2(
    const uint8_t* bytes,
    std::size_t byte_size)
{
    return DeserializePodVector<GpuRenderCommandV2>(
        bytes, byte_size, "DeserializeGpuRenderCommandsV2");
}

std::vector<GpuRenderBatchV2> DeserializeGpuRenderBatchesV2(
    const uint8_t* bytes,
    std::size_t byte_size)
{
    return DeserializePodVector<GpuRenderBatchV2>(
        bytes, byte_size, "DeserializeGpuRenderBatchesV2");
}

GpuCommandBufferV2 DeserializeGpuCommandBufferV2(const uint8_t* command_bytes,
                                                 std::size_t command_byte_size,
                                                 const uint8_t* batch_bytes,
                                                 std::size_t batch_byte_size,
                                                 uint32_t total_images)
{
    GpuCommandBufferV2 out;
    out.commands = DeserializeGpuRenderCommandsV2(command_bytes, command_byte_size);
    out.batches = DeserializeGpuRenderBatchesV2(batch_bytes, batch_byte_size);
    out.total_glyphs = out.commands.size();
    out.total_images = total_images;
    ValidateGpuCommandBufferV2(out, total_images);
    return out;
}

void ValidateGpuCommandBufferV2(const GpuCommandBufferV2& buffer,
                                uint32_t total_images)
{
    for (const GpuRenderCommandV2& command : buffer.commands)
    {
        if (command.image_index >= total_images)
        {
            throw std::runtime_error(
                "ValidateGpuCommandBufferV2: command image_index is out of bounds");
        }
    }

    for (const GpuRenderBatchV2& batch : buffer.batches)
    {
        const std::size_t offset = batch.command_offset;
        const std::size_t count = batch.command_count;
        if (offset > buffer.commands.size() || count > buffer.commands.size() - offset)
        {
            throw std::runtime_error(
                "ValidateGpuCommandBufferV2: batch command range is out of bounds");
        }
    }
}

} // namespace fac
