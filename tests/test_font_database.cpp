// test_font_database.cpp
//
// Build (from font_compiler/):
//   g++ -std=c++17 -O2 -Iinclude \
//       src/font_database.cpp \
//       tests/test_font_database.cpp \
//       -o test_font_database
//
// Run:
//   ./test_font_database <atlas_dir>
//   e.g.  ./test_font_database ./output_georgian

#include "font_database.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace fac;

// ── Minimal test harness ──────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: test_font_database <atlas_dir>\n";
        return 1;
    }
    const std::string dir = argv[1];

    // ── 1. LoadDirectory ─────────────────────────────────────────────────────
    SECTION("LoadDirectory");
    {
        FontDatabase db;
        EXPECT(db.Empty());

        int n = db.LoadDirectory(dir);
        std::cout << "  Loaded " << n << " atlases: ";
        for (uint32_t sz : db.LoadedSizes())
            std::cout << sz << " ";
        std::cout << "\n";

        EXPECT(n > 0);
        EXPECT(!db.Empty());
        EXPECT(db.LoadedSizes().size() == (size_t)n);

        // Sizes must be in ascending order.
        const auto& sizes = db.LoadedSizes();
        for (size_t i = 1; i < sizes.size(); ++i)
            EXPECT(sizes[i] > sizes[i - 1]);
    }

    // ── 2. GetEntry (exact size) ──────────────────────────────────────────────
    SECTION("GetEntry — exact size");
    {
        FontDatabase db;
        db.LoadDirectory(dir);

        for (uint32_t sz : db.LoadedSizes())
        {
            const AtlasEntry* e = db.GetEntry(sz);
            EXPECT(e != nullptr);
            EXPECT(e->render_size == sz);
            EXPECT(e->atlas_width  > 0);
            EXPECT(e->atlas_height > 0);
            EXPECT(e->glyph_count  > 0);
            EXPECT(e->pixels.size() == e->atlas_width * e->atlas_height);
        }

        // Non-existent size must return nullptr.
        EXPECT(db.GetEntry(0)     == nullptr);
        EXPECT(db.GetEntry(99999) == nullptr);
    }

    // ── 3. GetEntryForScale ───────────────────────────────────────────────────
    SECTION("GetEntryForScale");
    {
        FontDatabase db;
        db.LoadDirectory(dir);

        const auto& sizes = db.LoadedSizes();
        uint32_t smallest = sizes.front();
        uint32_t largest  = sizes.back();

        // scale=1.0 with base=smallest → should return smallest or next-up.
        {
            const AtlasEntry* e = db.GetEntryForScale(1.0f, smallest);
            EXPECT(e != nullptr);
            EXPECT(e->render_size >= smallest);
        }

        // Very small scale → smallest atlas.
        {
            const AtlasEntry* e = db.GetEntryForScale(0.1f, 32);
            EXPECT(e != nullptr);
            EXPECT(e->render_size == smallest);
        }

        // Very large scale → largest atlas (no upscaling beyond what's loaded).
        {
            const AtlasEntry* e = db.GetEntryForScale(100.0f, 32);
            EXPECT(e != nullptr);
            EXPECT(e->render_size == largest);
        }

        // scale such that desired_px equals an exact loaded size.
        // base=32, scale=1.5 → desired=48; if 48 is loaded it must be returned.
        if (db.GetEntry(48))
        {
            const AtlasEntry* e = db.GetEntryForScale(1.5f, 32);
            EXPECT(e != nullptr);
            EXPECT(e->render_size == 48);
        }

        // Never returns nullptr when db is non-empty.
        for (float s : {0.01f, 0.5f, 1.0f, 2.0f, 10.0f})
            EXPECT(db.GetEntryForScale(s, 32) != nullptr);
    }

    // ── 4. GetGlyph — ASCII ───────────────────────────────────────────────────
    SECTION("GetGlyph — printable ASCII");
    {
        FontDatabase db;
        db.LoadDirectory(dir);
        const AtlasEntry* e = db.GetEntry(db.LoadedSizes().front());
        if (!e) { std::cerr << "  (no atlas loaded, skipping)\n"; goto skip_ascii; }

        // Every printable ASCII character must have a positive advance.
        for (uint32_t cp = 0x20; cp <= 0x7E; ++cp)
        {
            const GlyphInfo& g = e->GetGlyph(cp);
            EXPECT(g.advance > 0.0f);
        }
        std::cout << "  ASCII advance spot-check: "
                  << "'A'=" << e->GetGlyph('A').advance << "px  "
                  << "'a'=" << e->GetGlyph('a').advance << "px  "
                  << "' '=" << e->GetGlyph(' ').advance << "px\n";
    }
    skip_ascii:;

    // ── 5. GetGlyph — Georgian ────────────────────────────────────────────────
    SECTION("GetGlyph — Georgian (U+10D0–U+10D5)");
    {
        FontDatabase db;
        db.LoadDirectory(dir);

        // Use 32px atlas for this check; fall back to any loaded size.
        const AtlasEntry* e = db.GetEntry(32);
        if (!e) e = db.GetEntry(db.LoadedSizes().front());
        if (!e) { std::cerr << "  (no atlas loaded, skipping)\n"; goto skip_geo; }

        // ა ბ გ დ ე ვ
        const uint32_t georgian[] = {0x10D0, 0x10D1, 0x10D2, 0x10D3, 0x10D4, 0x10D5};
        int found = 0;
        for (uint32_t cp : georgian)
        {
            if (e->HasGlyph(cp))
            {
                const GlyphInfo& g = e->GetGlyph(cp);
                EXPECT(g.atlas_w > 0);
                EXPECT(g.atlas_h > 0);
                EXPECT(g.advance > 0.0f);
                // Must not overflow atlas bounds.
                EXPECT(g.atlas_x + g.atlas_w <= e->atlas_width);
                EXPECT(g.atlas_y + g.atlas_h <= e->atlas_height);
                ++found;
            }
        }
        std::cout << "  Georgian glyphs with bitmap: " << found
                  << "/" << (int)(sizeof(georgian)/sizeof(georgian[0])) << "\n";
        EXPECT(found > 0);
    }
    skip_geo:;

    // ── 6. UV helpers ─────────────────────────────────────────────────────────
    SECTION("UV helpers — values in [0,1] and ULeft < URight");
    {
        FontDatabase db;
        db.LoadDirectory(dir);
        const AtlasEntry* e = db.GetEntry(db.LoadedSizes().front());
        if (!e) { std::cerr << "  (no atlas loaded, skipping)\n"; goto skip_uv; }

        for (uint32_t cp : {(uint32_t)'A', (uint32_t)'g', (uint32_t)0x10D0u})
        {
            if (!e->HasGlyph(cp)) continue;
            const GlyphInfo& g = e->GetGlyph(cp);

            float ul = e->ULeft(g),  ur = e->URight(g);
            float vt = e->VTop(g),   vb = e->VBottom(g);

            EXPECT(ul >= 0.f && ul <= 1.f);
            EXPECT(ur >= 0.f && ur <= 1.f);
            EXPECT(vt >= 0.f && vt <= 1.f);
            EXPECT(vb >= 0.f && vb <= 1.f);
            EXPECT(ul < ur);
            EXPECT(vt < vb);
        }
    }
    skip_uv:;

    // ── 7. Fallback glyph ─────────────────────────────────────────────────────
    SECTION("Fallback glyph — missing codepoint does not crash");
    {
        FontDatabase db;
        db.LoadDirectory(dir);
        const AtlasEntry* e = db.GetEntry(db.LoadedSizes().front());
        if (!e) goto skip_fallback;

        // Private Use Area — should never be in the atlas.
        const GlyphInfo& g = e->GetGlyph(0xE000);
        // Must not return a dangling reference; advance may be 0 for fallback.
        (void)g.codepoint;
        EXPECT(true);  // reaching here without crash is the test
        std::cout << "  Fallback glyph for U+E000: advance="
                  << g.advance << "px\n";
    }
    skip_fallback:;

    // ── 8. Double-load idempotency ────────────────────────────────────────────
    SECTION("Double-load — same atlas replaces cleanly");
    {
        FontDatabase db;
        db.LoadDirectory(dir);
        size_t n_before = db.LoadedSizes().size();

        // Load again — size list must not grow.
        db.LoadDirectory(dir);
        EXPECT(db.LoadedSizes().size() == n_before);
    }

    // ── 9. Bad file ───────────────────────────────────────────────────────────
    SECTION("Error handling — bad file throws");
    {
        FontDatabase db;
        bool threw = false;
        try { db.LoadAtlas("/nonexistent/path/atlas_32.bin"); }
        catch (const std::runtime_error&) { threw = true; }
        EXPECT(threw);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? 1 : 0;
}
