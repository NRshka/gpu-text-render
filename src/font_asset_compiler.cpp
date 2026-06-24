// font_asset_compiler.cpp
//
// Offline tool: converts a TTF/OTF font into a set of GPU-ready atlas binaries
// and a JSON metadata file.  No FreeType dependency at runtime.
//
// Build:
//   g++ -std=c++17 -O2 -Iinclude \
//       $(pkg-config --cflags freetype2) \
//       src/font_asset_compiler.cpp \
//       $(pkg-config --libs freetype2) \
//       -o font_asset_compiler
//
// Usage:
//   ./font_asset_compiler <font.ttf> <output_dir> [--sizes 16,24,32,48,64]
//                                                  [--atlas-size 1024]
//                                                  [--ranges latin,cjk,cyrillic]

#include "font_asset.h"
#include "skyline_packer.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
//  Unicode range definitions
// ─────────────────────────────────────────────────────────────────────────────

struct UnicodeRange { uint32_t first, last; const char* name; };

static const UnicodeRange RANGE_LATIN[] = {
    {0x0020, 0x007E},  // Basic Latin (printable ASCII)
    {0x00A0, 0x00FF},  // Latin-1 Supplement
    {0x0100, 0x017F},  // Latin Extended-A
    {0x0180, 0x024F},  // Latin Extended-B
    {0x2000, 0x206F},  // General Punctuation
    {0x2070, 0x209F},  // Superscripts and Subscripts
    {0x20A0, 0x20CF},  // Currency Symbols
    {0x2190, 0x21FF},  // Arrows
    {0x2200, 0x22FF},  // Mathematical Operators
};

static const UnicodeRange RANGE_CYRILLIC[] = {
    {0x0400, 0x04FF},  // Cyrillic
    {0x0500, 0x052F},  // Cyrillic Supplement
};

static const UnicodeRange RANGE_GEORGIAN[] = {
    {0x10A0, 0x10FF},  // Georgian (Mkhedruli + Asomtavruli capitals)
    {0x2D00, 0x2D2F},  // Georgian Supplement (Nuskhuri)
    // {0x1C90, 0x1CBF}, // Georgian Extended (Unicode 11+) — not in DejaVu
};

static const UnicodeRange RANGE_CJK[] = {
    {0x3000, 0x303F},  // CJK Symbols and Punctuation
    {0x3040, 0x309F},  // Hiragana
    {0x30A0, 0x30FF},  // Katakana
    {0x4E00, 0x9FFF},  // CJK Unified Ideographs (core)
    {0xF900, 0xFAFF},  // CJK Compatibility Ideographs
    {0xFF00, 0xFFEF},  // Halfwidth / Fullwidth Forms
};

// ─────────────────────────────────────────────────────────────────────────────
//  Compiler options
// ─────────────────────────────────────────────────────────────────────────────

struct CompilerOptions
{
    std::string              font_path;
    std::string              output_dir;
    std::vector<uint32_t>    sizes       = {16, 24, 32, 48, 64};
    uint32_t                 atlas_size  = 1024;
    uint16_t                 padding     = 1;
    std::set<std::string>    ranges      = {"latin"};
};

// ─────────────────────────────────────────────────────────────────────────────
//  Glyph raster (intermediate, before packing)
// ─────────────────────────────────────────────────────────────────────────────

struct GlyphRaster
{
    uint32_t codepoint = 0;

    // bitmap (may be empty for space / whitespace)
    uint32_t bmp_w = 0, bmp_h = 0;
    std::vector<uint8_t> pixels; // grayscale α, row-major

    // metrics (float, already converted from 26.6 fixed)
    float advance   = 0.f;
    float bearing_x = 0.f;
    float bearing_y = 0.f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Build codepoint set from requested range names
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<uint32_t>
BuildCodepointList(const std::set<std::string>& range_names)
{
    std::vector<std::pair<const UnicodeRange*, size_t>> active;

    auto push = [&](const UnicodeRange* arr, size_t n) {
        active.push_back({arr, n});
    };

    for (auto& name : range_names) {
        if (name == "latin")    push(RANGE_LATIN,    std::size(RANGE_LATIN));
        if (name == "cyrillic") push(RANGE_CYRILLIC, std::size(RANGE_CYRILLIC));
        if (name == "georgian") push(RANGE_GEORGIAN, std::size(RANGE_GEORGIAN));
        if (name == "cjk")      push(RANGE_CJK,      std::size(RANGE_CJK));
    }

    std::set<uint32_t> seen;
    std::vector<uint32_t> result;
    for (auto& [arr, n] : active) {
        for (size_t i = 0; i < n; ++i)
            for (uint32_t cp = arr[i].first; cp <= arr[i].last; ++cp)
                if (seen.insert(cp).second)
                    result.push_back(cp);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rasterise all glyphs for one font size
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<GlyphRaster>
RasteriseGlyphs(FT_Face face, uint32_t size_px,
                const std::vector<uint32_t>& codepoints,
                uint32_t& out_font_height_px)
{
    FT_Error err = FT_Set_Pixel_Sizes(face, 0, size_px);
    if (err)
        throw std::runtime_error("FT_Set_Pixel_Sizes failed");

    // Ascender + descender give the full cell height in pixels.
    out_font_height_px = (uint32_t)((face->size->metrics.ascender
                                   - face->size->metrics.descender) >> 6);

    std::vector<GlyphRaster> rasters;
    rasters.reserve(codepoints.size());

    for (uint32_t cp : codepoints)
    {
        FT_UInt glyph_index = FT_Get_Char_Index(face, cp);

        GlyphRaster r;
        r.codepoint = cp;

        if (glyph_index == 0)
        {
            // Codepoint not in font — emit a zero-advance placeholder so the
            // metadata still contains an entry and layout can fall back.
            r.advance = 0.f;
            rasters.push_back(std::move(r));
            continue;
        }

        err = FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT);
        if (err) continue;

        err = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
        if (err) continue;

        FT_GlyphSlot g = face->glyph;
        FT_Bitmap&   bm = g->bitmap;

        r.advance   = (float)(g->advance.x >> 6);
        r.bearing_x = (float)g->bitmap_left;
        r.bearing_y = (float)g->bitmap_top;

        if (bm.width > 0 && bm.rows > 0)
        {
            r.bmp_w  = bm.width;
            r.bmp_h  = bm.rows;
            r.pixels.resize(bm.width * bm.rows);

            // FreeType may use a pitch != width (e.g. padded rows).
            for (uint32_t row = 0; row < (uint32_t)bm.rows; ++row)
                std::memcpy(r.pixels.data() + row * bm.width,
                            bm.buffer  + row * std::abs(bm.pitch),
                            bm.width);
        }

        rasters.push_back(std::move(r));
    }

    return rasters;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pack rasters into an atlas and blit pixels
// ─────────────────────────────────────────────────────────────────────────────

struct PackedAtlas
{
    uint32_t                        width, height;
    uint32_t                        render_size;
    uint32_t                        font_height_px;
    std::vector<fac::AtlasGlyphRecord> records;
    std::vector<uint8_t>            pixels; // width × height, 8-bit α
    float                           occupancy;
};

static PackedAtlas
PackAtlas(const std::vector<GlyphRaster>& rasters,
          uint32_t atlas_size, uint32_t render_size,
          uint32_t font_height_px, uint16_t padding)
{
    fac::SkylinePacker packer(atlas_size, atlas_size);

    // Two-pass: first pack, then blit (avoids re-rendering).
    struct Slot { uint16_t x, y; };
    std::vector<Slot> slots(rasters.size(), {0, 0});

    for (size_t i = 0; i < rasters.size(); ++i)
    {
        auto& r = rasters[i];
        if (r.bmp_w == 0 || r.bmp_h == 0) continue;

        auto pos = packer.Insert((uint16_t)r.bmp_w,
                                 (uint16_t)r.bmp_h,
                                 padding);
        if (!pos)
            throw std::runtime_error(
                "Atlas too small — increase --atlas-size or reduce glyph set");

        slots[i] = {pos->x, pos->y};
    }

    // Allocate texture and blit.
    PackedAtlas out;
    out.width          = atlas_size;
    out.height         = atlas_size;
    out.render_size    = render_size;
    out.font_height_px = font_height_px;
    out.pixels.assign(atlas_size * atlas_size, 0);
    out.occupancy      = packer.Occupancy();

    for (size_t i = 0; i < rasters.size(); ++i)
    {
        auto& r   = rasters[i];
        auto& s   = slots[i];

        fac::AtlasGlyphRecord rec{};
        rec.codepoint = r.codepoint;
        rec.atlas_x   = s.x;
        rec.atlas_y   = s.y;
        rec.atlas_w   = (uint16_t)r.bmp_w;
        rec.atlas_h   = (uint16_t)r.bmp_h;
        rec.advance   = r.advance;
        rec.bearing_x = r.bearing_x;
        rec.bearing_y = r.bearing_y;

        out.records.push_back(rec);

        // Blit pixels.
        for (uint32_t row = 0; row < r.bmp_h; ++row)
        {
            std::memcpy(
                out.pixels.data() + (s.y + row) * atlas_size + s.x,
                r.pixels.data()   + row * r.bmp_w,
                r.bmp_w);
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Write binary atlas file
// ─────────────────────────────────────────────────────────────────────────────

static void
WriteAtlasBin(const PackedAtlas& atlas, const fs::path& path)
{
    // pixel_offset = sizeof(AtlasHeader) + N * sizeof(AtlasGlyphRecord)
    uint32_t pixel_offset = sizeof(fac::AtlasHeader)
                          + (uint32_t)atlas.records.size()
                            * sizeof(fac::AtlasGlyphRecord);

    fac::AtlasHeader hdr{};
    hdr.magic        = fac::ATLAS_MAGIC;
    hdr.version      = fac::ATLAS_VERSION;
    hdr.width        = atlas.width;
    hdr.height       = atlas.height;
    hdr.glyph_count  = (uint32_t)atlas.records.size();
    hdr.pixel_offset = pixel_offset;
    hdr.render_size  = atlas.render_size;
    hdr.font_height_px = atlas.font_height_px;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path.string());

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(atlas.records.data()),
            atlas.records.size() * sizeof(fac::AtlasGlyphRecord));
    f.write(reinterpret_cast<const char*>(atlas.pixels.data()),
            atlas.pixels.size());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Build JSON metadata
// ─────────────────────────────────────────────────────────────────────────────

static json
BuildMetadata(const CompilerOptions& opts,
              const std::map<uint32_t, PackedAtlas>& atlases,
              const std::string& font_family,
              const std::string& font_style)
{
    json meta;
    meta["font_family"] = font_family;
    meta["font_style"]  = font_style;
    meta["atlas_size"]  = opts.atlas_size;
    meta["padding"]     = opts.padding;
    meta["ranges"]      = json::array();
    for (auto& r : opts.ranges) meta["ranges"].push_back(r);

    json sizes_arr = json::array();
    for (auto& [sz, atlas] : atlases)
    {
        json entry;
        entry["size"]           = sz;
        entry["atlas_file"]     = "atlas_" + std::to_string(sz) + ".bin";
        entry["atlas_width"]    = atlas.width;
        entry["atlas_height"]   = atlas.height;
        entry["glyph_count"]    = atlas.records.size();
        entry["font_height_px"] = atlas.font_height_px;
        entry["occupancy"]      = atlas.occupancy;

        // Per-glyph table for human inspection (runtime uses binary).
        json glyphs = json::array();
        for (auto& rec : atlas.records)
        {
            if (rec.atlas_w == 0) continue; // skip invisible glyphs
            json g;
            g["cp"]  = rec.codepoint;
            g["x"]   = rec.atlas_x;
            g["y"]   = rec.atlas_y;
            g["w"]   = rec.atlas_w;
            g["h"]   = rec.atlas_h;
            g["adv"] = rec.advance;
            g["bx"]  = rec.bearing_x;
            g["by"]  = rec.bearing_y;
            glyphs.push_back(g);
        }
        entry["glyphs"] = glyphs;
        sizes_arr.push_back(entry);
    }
    meta["atlases"] = sizes_arr;
    return meta;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────

static CompilerOptions
ParseArgs(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: font_asset_compiler <font.ttf> <output_dir>"
                     " [--sizes 16,24,32,48,64]"
                     " [--atlas-size 1024]"
                     " [--ranges latin,cjk,cyrillic]\n";
        std::exit(1);
    }

    CompilerOptions opts;
    opts.font_path   = argv[1];
    opts.output_dir  = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc)
        {
            opts.sizes.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ','))
                opts.sizes.push_back((uint32_t)std::stoul(tok));
        }
        else if (arg == "--atlas-size" && i + 1 < argc)
        {
            opts.atlas_size = (uint32_t)std::stoul(argv[++i]);
        }
        else if (arg == "--ranges" && i + 1 < argc)
        {
            opts.ranges.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ','))
                opts.ranges.insert(tok);
        }
    }

    return opts;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    CompilerOptions opts = ParseArgs(argc, argv);

    fs::create_directories(opts.output_dir);

    // ── Init FreeType ────────────────────────────────────────────────────────
    FT_Library ft_lib;
    if (FT_Init_FreeType(&ft_lib))
        throw std::runtime_error("FreeType init failed");

    FT_Face face;
    if (FT_New_Face(ft_lib, opts.font_path.c_str(), 0, &face))
        throw std::runtime_error("Cannot load font: " + opts.font_path);

    std::string font_family = face->family_name  ? face->family_name  : "Unknown";
    std::string font_style  = face->style_name   ? face->style_name   : "Regular";

    std::cout << "Font      : " << font_family << " " << font_style << "\n";
    std::cout << "Output    : " << opts.output_dir << "\n";
    std::cout << "Sizes     : ";
    for (auto s : opts.sizes) std::cout << s << " ";
    std::cout << "\nAtlas size: " << opts.atlas_size << "×"
              << opts.atlas_size << "\n\n";

    // ── Build codepoint set ──────────────────────────────────────────────────
    auto codepoints = BuildCodepointList(opts.ranges);
    std::cout << "Codepoints: " << codepoints.size() << "\n";

    // ── Process each size ────────────────────────────────────────────────────
    std::map<uint32_t, PackedAtlas> atlases;

    for (uint32_t sz : opts.sizes)
    {
        std::cout << "  Compiling size " << sz << "px ... " << std::flush;

        uint32_t font_height_px = 0;
        auto rasters = RasteriseGlyphs(face, sz, codepoints, font_height_px);

        auto atlas = PackAtlas(rasters, opts.atlas_size, sz,
                               font_height_px, opts.padding);

        fs::path bin_path = fs::path(opts.output_dir)
                          / ("atlas_" + std::to_string(sz) + ".bin");
        WriteAtlasBin(atlas, bin_path);

        std::cout << atlas.records.size() << " glyphs packed, "
                  << "occupancy " << (int)(atlas.occupancy * 100.f) << "%\n";

        atlases[sz] = std::move(atlas);
    }

    // ── Write metadata.json ──────────────────────────────────────────────────
    json meta = BuildMetadata(opts, atlases, font_family, font_style);
    fs::path json_path = fs::path(opts.output_dir) / "metadata.json";
    std::ofstream jf(json_path);
    jf << meta.dump(2) << "\n";

    std::cout << "\nmetadata.json written to " << json_path << "\n";

    // ── Cleanup ──────────────────────────────────────────────────────────────
    FT_Done_Face(face);
    FT_Done_FreeType(ft_lib);

    return 0;
}
