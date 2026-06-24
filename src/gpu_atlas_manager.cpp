#include "gpu_atlas_manager.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace fac {

namespace {

static void CheckCuda(cudaError_t err, const char* context)
{
    if (err == cudaSuccess)
        return;

    std::ostringstream ss;
    ss << context << ": " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
}

static void FreeHandle(GpuAtlasHandle& handle) noexcept
{
    if (handle.device_pixels)
        cudaFree(handle.device_pixels);
    handle.device_pixels = nullptr;
    handle.pixel_bytes = 0;
}

} // namespace

GpuAtlasManager::~GpuAtlasManager()
{
    Clear();
}

GpuAtlasManager::GpuAtlasManager(GpuAtlasManager&& other) noexcept
    : m_atlases(std::move(other.m_atlases))
{
    other.m_atlases.clear();
}

GpuAtlasManager& GpuAtlasManager::operator=(GpuAtlasManager&& other) noexcept
{
    if (this == &other)
        return *this;

    Clear();
    m_atlases = std::move(other.m_atlases);
    other.m_atlases.clear();
    return *this;
}

void GpuAtlasManager::Upload(const FontDatabase& db)
{
    Clear();

    for (uint32_t render_size : db.LoadedSizes())
    {
        const AtlasEntry* atlas = db.GetEntry(render_size);
        if (!atlas)
            continue;

        GpuAtlasHandle handle;
        handle.render_size = render_size;
        handle.atlas_width = atlas->atlas_width;
        handle.atlas_height = atlas->atlas_height;
        handle.font_height_px = atlas->font_height_px;
        handle.pixel_bytes = atlas->pixels.size();

        if (handle.pixel_bytes > 0)
        {
            CheckCuda(cudaMalloc((void**)&handle.device_pixels, handle.pixel_bytes),
                      "GpuAtlasManager::Upload cudaMalloc");
            CheckCuda(cudaMemcpy(handle.device_pixels,
                                 atlas->pixels.data(),
                                 handle.pixel_bytes,
                                 cudaMemcpyHostToDevice),
                      "GpuAtlasManager::Upload cudaMemcpy");
        }

        m_atlases.emplace(render_size, handle);
    }
}

void GpuAtlasManager::Clear() noexcept
{
    for (auto& [_, handle] : m_atlases)
        FreeHandle(handle);
    m_atlases.clear();
}

const GpuAtlasHandle* GpuAtlasManager::GetAtlas(uint32_t render_size) const noexcept
{
    auto it = m_atlases.find(render_size);
    return it != m_atlases.end() ? &it->second : nullptr;
}

} // namespace fac
