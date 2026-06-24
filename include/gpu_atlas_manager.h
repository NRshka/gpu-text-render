#pragma once

#include "font_database.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace fac {

struct GpuAtlasHandle
{
    uint32_t render_size    = 0;
    uint32_t atlas_width    = 0;
    uint32_t atlas_height   = 0;
    uint32_t font_height_px = 0;
    std::size_t pixel_bytes = 0;
    uint8_t* device_pixels  = nullptr;
};

class GpuAtlasManager
{
public:
    GpuAtlasManager() = default;
    ~GpuAtlasManager();

    GpuAtlasManager(const GpuAtlasManager&) = delete;
    GpuAtlasManager& operator=(const GpuAtlasManager&) = delete;
    GpuAtlasManager(GpuAtlasManager&& other) noexcept;
    GpuAtlasManager& operator=(GpuAtlasManager&& other) noexcept;

    void Upload(const FontDatabase& db);
    void Clear() noexcept;

    const GpuAtlasHandle* GetAtlas(uint32_t render_size) const noexcept;
    bool Empty() const noexcept { return m_atlases.empty(); }

private:
    std::unordered_map<uint32_t, GpuAtlasHandle> m_atlases;
};

} // namespace fac
