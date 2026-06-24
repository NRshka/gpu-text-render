#pragma once

#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  font_asset.h  —  shared ABI between Font Asset Compiler and runtime loader
//
//  All binary files written by the compiler are read back by the runtime using
//  these exact structs.  Do NOT add virtual methods or non-trivial members.
// ─────────────────────────────────────────────────────────────────────────────

namespace fac {

// ── Binary atlas file ────────────────────────────────────────────────────────
//
//  Layout of  atlas_<size>.bin :
//
//    [ AtlasHeader            ]   (fixed, 32 bytes)
//    [ AtlasGlyphRecord × N   ]   (N = header.glyph_count)
//    [ pixel data             ]   (header.width × header.height bytes, 8-bit α)
//
static constexpr uint32_t ATLAS_MAGIC   = 0x46415431; // "FAT1"
static constexpr uint32_t ATLAS_VERSION_1 = 1;
static constexpr uint32_t ATLAS_VERSION_2 = 2;
static constexpr uint32_t ATLAS_VERSION   = ATLAS_VERSION_2;

#pragma pack(push, 1)

struct AtlasHeader
{
    uint32_t magic;         // must equal ATLAS_MAGIC
    uint32_t version;       // current writer emits ATLAS_VERSION
    uint32_t width;         // texture width  in pixels
    uint32_t height;        // texture height in pixels
    uint32_t glyph_count;   // number of AtlasGlyphRecord entries that follow
    uint32_t pixel_offset;  // byte offset from file start to first pixel
    uint32_t render_size;   // FreeType pixel size used when rasterising
    // Exact font cell height (ascender - descender) at render_size.
    // In v1 files this slot is reserved padding and should be treated as 0.
    uint32_t font_height_px;
};

// Per-glyph metrics and atlas placement — everything the renderer needs.
// Coordinates in the atlas are in pixels; metrics are in 26.6 fixed-point
// converted to float at compile time so the runtime never needs FreeType.
struct AtlasGlyphRecord
{
    uint32_t codepoint;     // Unicode codepoint

    // ── atlas location ───────────────────────────────────────────────────
    uint16_t atlas_x;       // top-left x in atlas texture
    uint16_t atlas_y;       // top-left y in atlas texture
    uint16_t atlas_w;       // glyph bitmap width  (may be 0 for whitespace)
    uint16_t atlas_h;       // glyph bitmap height (may be 0 for whitespace)

    // ── layout metrics (at render_size px, pre-scaled to floats) ────────
    float advance;          // horizontal advance in pixels
    float bearing_x;        // left side bearing
    float bearing_y;        // distance from baseline to top of glyph bitmap
};

#pragma pack(pop)

// ── Convenience: what the runtime keeps in RAM per glyph ─────────────────────
struct GlyphInfo
{
    uint32_t codepoint;

    // Atlas UV coordinates (pixel coords; divide by atlas w/h for normalised).
    uint16_t atlas_x, atlas_y;
    uint16_t atlas_w, atlas_h;

    // Layout metrics at the nominal render size stored in the atlas.
    float advance;
    float bearing_x;
    float bearing_y;
};

} // namespace fac
