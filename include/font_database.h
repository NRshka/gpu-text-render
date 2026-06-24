#pragma once

#include "font_asset.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>

// ─────────────────────────────────────────────────────────────────────────────
//  FontDatabase
//
//  Runtime counterpart of the Font Asset Compiler.  Loads atlas_<size>.bin
//  files produced offline and exposes O(1) glyph lookup by codepoint.
//
//  Lifecycle:
//      FontDatabase db;
//      db.LoadAtlas("assets/atlas_32.bin");
//      db.LoadAtlas("assets/atlas_64.bin");
//      // ...
//      const AtlasEntry* entry = db.GetEntry(32);   // exact size
//      const GlyphInfo&  g     = entry->GetGlyph(0x10D0); // Georgian ა
//
//  Size selection:
//      db.GetEntryForScale(scale, base_render_size)
//      returns the atlas whose render_size best covers the requested scale
//      without the glyph appearing blurry (always picks the next size up).
//
//  Thread safety:
//      All const member functions are safe to call concurrently once loading
//      is complete.  Do not call LoadAtlas() from multiple threads.
// ─────────────────────────────────────────────────────────────────────────────

namespace fac {

// ── Per-size atlas loaded into RAM ───────────────────────────────────────────

class AtlasEntry
{
public:
    // Header fields exposed for layout math.
    uint32_t render_size    = 0;  // pixel size used at compile time
    uint32_t atlas_width    = 0;
    uint32_t atlas_height   = 0;
    uint32_t glyph_count    = 0;
    uint32_t font_height_px = 0;  // ascender − descender at render_size

    // Raw 8-bit alpha pixels (atlas_width × atlas_height).
    // Kept in RAM so the GPU manager (Task 7) can upload without re-reading disk.
    std::vector<uint8_t> pixels;

    // ── Glyph lookup ─────────────────────────────────────────────────────────

    // Returns the GlyphInfo for the given Unicode codepoint.
    // If the codepoint was not compiled into the atlas, returns the fallback
    // glyph (U+FFFD REPLACEMENT CHARACTER, or U+003F '?' if that is also
    // absent).  Never throws; never returns nullptr.
    const GlyphInfo& GetGlyph(uint32_t codepoint) const noexcept;

    // Returns true if the codepoint has a non-zero bitmap in this atlas.
    bool HasGlyph(uint32_t codepoint) const noexcept;

    // ── UV helpers ───────────────────────────────────────────────────────────

    // Normalised UV coordinates [0,1] of the glyph's top-left corner.
    float ULeft (const GlyphInfo& g) const noexcept
    { return (float)g.atlas_x / (float)atlas_width;  }

    float URight(const GlyphInfo& g) const noexcept
    { return (float)(g.atlas_x + g.atlas_w) / (float)atlas_width;  }

    float VTop  (const GlyphInfo& g) const noexcept
    { return (float)g.atlas_y / (float)atlas_height; }

    float VBottom(const GlyphInfo& g) const noexcept
    { return (float)(g.atlas_y + g.atlas_h) / (float)atlas_height; }

private:
    friend class FontDatabase;

    std::unordered_map<uint32_t, GlyphInfo> m_glyphs;
    GlyphInfo                               m_fallback{};  // U+FFFD or '?'

    void BuildLookup(const std::vector<AtlasGlyphRecord>& records);
};

// ── FontDatabase ─────────────────────────────────────────────────────────────

class FontDatabase
{
public:
    FontDatabase()  = default;
    ~FontDatabase() = default;

    // Non-copyable (owns pixel buffers); movable.
    FontDatabase(const FontDatabase&)            = delete;
    FontDatabase& operator=(const FontDatabase&) = delete;
    FontDatabase(FontDatabase&&)                 = default;
    FontDatabase& operator=(FontDatabase&&)      = default;

    // ── Loading ───────────────────────────────────────────────────────────────

    // Load a single atlas_<size>.bin file.
    // Throws std::runtime_error on any file / format error.
    // Can be called multiple times with different sizes.
    void LoadAtlas(const std::filesystem::path& bin_path);

    // Convenience: load every atlas_*.bin found in a directory, sorted by size.
    // Returns the number of atlases loaded.
    int LoadDirectory(const std::filesystem::path& dir);

    // ── Size lookup ───────────────────────────────────────────────────────────

    // Returns the AtlasEntry for the given render_size, or nullptr if not loaded.
    const AtlasEntry* GetEntry(uint32_t render_size) const noexcept;

    // Returns the smallest loaded atlas whose render_size >= target_px.
    // If all loaded atlases are smaller than target_px, returns the largest one.
    // target_px = scale * base_render_size  (base_render_size is arbitrary, e.g. 32).
    const AtlasEntry* GetEntryForScale(float scale,
                                       uint32_t base_render_size = 32) const noexcept;

    // ── Inspection ────────────────────────────────────────────────────────────

    // Sorted list of all loaded render sizes.
    const std::vector<uint32_t>& LoadedSizes() const noexcept { return m_sizes; }

    bool Empty() const noexcept { return m_entries.empty(); }

private:
    // Keyed by render_size for O(1) exact lookup.
    std::unordered_map<uint32_t, AtlasEntry> m_entries;

    // Sorted render sizes for nearest-size selection.
    std::vector<uint32_t> m_sizes;
};

} // namespace fac
