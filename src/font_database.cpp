#include "font_database.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fac {

static bool IsSupportedAtlasVersion(uint32_t version) noexcept
{
    return version == ATLAS_VERSION_1 || version == ATLAS_VERSION_2;
}

// ─────────────────────────────────────────────────────────────────────────────
//  AtlasEntry
// ─────────────────────────────────────────────────────────────────────────────

void AtlasEntry::BuildLookup(const std::vector<AtlasGlyphRecord>& records)
{
    m_glyphs.reserve(records.size());

    for (const auto& rec : records)
    {
        GlyphInfo g{};
        g.codepoint = rec.codepoint;
        g.atlas_x   = rec.atlas_x;
        g.atlas_y   = rec.atlas_y;
        g.atlas_w   = rec.atlas_w;
        g.atlas_h   = rec.atlas_h;
        g.advance   = rec.advance;
        g.bearing_x = rec.bearing_x;
        g.bearing_y = rec.bearing_y;

        m_glyphs.emplace(rec.codepoint, g);
    }

    // Choose fallback: prefer U+FFFD (REPLACEMENT CHARACTER), then '?'.
    auto it = m_glyphs.find(0xFFFD);
    if (it == m_glyphs.end())
        it = m_glyphs.find(U'?');

    if (it != m_glyphs.end())
        m_fallback = it->second;
    // else: m_fallback stays zero-initialised — renderer treats it as invisible.
}

const GlyphInfo& AtlasEntry::GetGlyph(uint32_t codepoint) const noexcept
{
    auto it = m_glyphs.find(codepoint);
    if (it != m_glyphs.end())
        return it->second;
    return m_fallback;
}

bool AtlasEntry::HasGlyph(uint32_t codepoint) const noexcept
{
    auto it = m_glyphs.find(codepoint);
    return it != m_glyphs.end() && it->second.atlas_w > 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FontDatabase — loading
// ─────────────────────────────────────────────────────────────────────────────

void FontDatabase::LoadAtlas(const std::filesystem::path& bin_path)
{
    std::ifstream f(bin_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("FontDatabase: cannot open " + bin_path.string());

    // ── Read and validate header ──────────────────────────────────────────────
    AtlasHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f)
        throw std::runtime_error("FontDatabase: truncated header in "
                                 + bin_path.string());

    if (hdr.magic != ATLAS_MAGIC)
    {
        std::ostringstream ss;
        ss << "FontDatabase: bad magic 0x" << std::hex << hdr.magic
           << " in " << bin_path.string()
           << " (expected 0x" << ATLAS_MAGIC << ")";
        throw std::runtime_error(ss.str());
    }

    if (!IsSupportedAtlasVersion(hdr.version))
    {
        std::ostringstream ss;
        ss << "FontDatabase: unsupported atlas version " << hdr.version
           << " in " << bin_path.string()
           << " (supported: " << ATLAS_VERSION_1
           << " or " << ATLAS_VERSION_2 << ")";
        throw std::runtime_error(ss.str());
    }

    if (hdr.width == 0 || hdr.height == 0 || hdr.glyph_count == 0)
        throw std::runtime_error("FontDatabase: zero-size atlas in "
                                 + bin_path.string());

    // ── Read glyph records ────────────────────────────────────────────────────
    std::vector<AtlasGlyphRecord> records(hdr.glyph_count);
    f.read(reinterpret_cast<char*>(records.data()),
           hdr.glyph_count * sizeof(AtlasGlyphRecord));
    if (!f)
        throw std::runtime_error("FontDatabase: truncated glyph records in "
                                 + bin_path.string());

    // ── Read pixel data ───────────────────────────────────────────────────────
    // Seek to pixel_offset in case there's future padding between records and
    // pixels (e.g. a v2 header with extra fields).
    f.seekg(hdr.pixel_offset, std::ios::beg);
    if (!f)
        throw std::runtime_error("FontDatabase: cannot seek to pixels in "
                                 + bin_path.string());

    const size_t pixel_count = (size_t)hdr.width * hdr.height;
    std::vector<uint8_t> pixels(pixel_count);
    f.read(reinterpret_cast<char*>(pixels.data()), pixel_count);
    if (!f)
        throw std::runtime_error("FontDatabase: truncated pixel data in "
                                 + bin_path.string());

    // ── Populate AtlasEntry ───────────────────────────────────────────────────
    AtlasEntry entry;
    entry.render_size    = hdr.render_size;
    entry.atlas_width    = hdr.width;
    entry.atlas_height   = hdr.height;
    entry.glyph_count    = hdr.glyph_count;
    entry.pixels         = std::move(pixels);

    if (hdr.version >= ATLAS_VERSION_2 && hdr.font_height_px > 0)
    {
        entry.font_height_px = hdr.font_height_px;
    }
    else
    {
        // v1 files do not store exact font height. Fall back to the tallest
        // glyph bitmap so legacy atlases remain loadable.
        uint32_t max_h = 0;
        for (const auto& rec : records)
            max_h = std::max(max_h, (uint32_t)rec.atlas_h);
        entry.font_height_px = max_h;
    }

    entry.BuildLookup(records);

    // ── Insert / replace ──────────────────────────────────────────────────────
    const uint32_t sz = hdr.render_size;
    m_entries[sz]     = std::move(entry);

    // Keep m_sizes sorted and deduplicated.
    auto pos = std::lower_bound(m_sizes.begin(), m_sizes.end(), sz);
    if (pos == m_sizes.end() || *pos != sz)
        m_sizes.insert(pos, sz);
}

int FontDatabase::LoadDirectory(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;

    if (!fs::is_directory(dir))
        throw std::runtime_error("FontDatabase: not a directory: " + dir.string());

    // Collect matching files, then sort so loading order is deterministic.
    std::vector<fs::path> candidates;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        const auto& p = entry.path();
        if (p.extension() != ".bin") continue;

        const std::string stem = p.stem().string();
        if (stem.rfind("atlas_", 0) == 0)   // starts with "atlas_"
            candidates.push_back(p);
    }
    std::sort(candidates.begin(), candidates.end());

    for (const auto& p : candidates)
        LoadAtlas(p);

    return (int)candidates.size();
}

// ─────────────────────────────────────────────────────────────────────────────
//  FontDatabase — lookup
// ─────────────────────────────────────────────────────────────────────────────

const AtlasEntry* FontDatabase::GetEntry(uint32_t render_size) const noexcept
{
    auto it = m_entries.find(render_size);
    return it != m_entries.end() ? &it->second : nullptr;
}

const AtlasEntry* FontDatabase::GetEntryForScale(float scale,
                                                  uint32_t base_render_size) const noexcept
{
    if (m_sizes.empty()) return nullptr;

    // Desired pixel size at this scale.
    const float desired_px = scale * (float)base_render_size;

    // Find the smallest loaded size that is >= desired_px (ceiling search).
    // This ensures we never upscale a smaller atlas, which would look blurry.
    for (uint32_t sz : m_sizes)           // m_sizes is sorted ascending
    {
        if ((float)sz >= desired_px)
        {
            auto it = m_entries.find(sz);
            return it != m_entries.end() ? &it->second : nullptr;
        }
    }

    // All loaded sizes are smaller than desired — return the largest available.
    auto it = m_entries.find(m_sizes.back());
    return it != m_entries.end() ? &it->second : nullptr;
}

} // namespace fac
