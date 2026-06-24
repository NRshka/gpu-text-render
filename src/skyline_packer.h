#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <algorithm>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
//  SkylinePacker
//
//  Classic skyline bottom-left bin-packing algorithm.
//  Deterministic: identical input → identical layout.
//  Achieves >80% occupancy for typical glyph sets.
//
//  Usage:
//      SkylinePacker packer(1024, 1024);
//      auto slot = packer.Insert(w, h, padding);
//      if (slot) { /* slot->x, slot->y */ }
// ─────────────────────────────────────────────────────────────────────────────

namespace fac {

struct PackRect
{
    uint16_t x, y;
};

class SkylinePacker
{
public:
    SkylinePacker(uint32_t width, uint32_t height)
        : m_width(width), m_height(height)
    {
        // Start with a single flat segment spanning the full width at y=0.
        m_skyline.push_back({0, 0, width});
    }

    // Insert a rectangle of (w × h) with optional padding on each side.
    // Returns the top-left position, or nullopt if it doesn't fit.
    std::optional<PackRect> Insert(uint16_t w, uint16_t h, uint16_t padding = 1)
    {
        uint16_t pw = w + padding;
        uint16_t ph = h + padding;

        uint32_t best_x    = 0;
        uint32_t best_y    = std::numeric_limits<uint32_t>::max();
        int      best_seg  = -1;

        for (int i = 0; i < (int)m_skyline.size(); ++i)
        {
            uint32_t x, y;
            if (!FitsAt(i, pw, ph, x, y))
                continue;
            if (y < best_y || (y == best_y && x < best_x))
            {
                best_x   = x;
                best_y   = y;
                best_seg = i;
            }
        }

        if (best_seg < 0)
            return std::nullopt;

        AddSkylineLevel(best_seg, best_x, best_y, pw, ph);
        return PackRect{(uint16_t)best_x, (uint16_t)best_y};
    }

    float Occupancy() const
    {
        uint64_t used = 0;
        for (auto& s : m_skyline)
            used += (uint64_t)s.w * s.y;
        return (float)used / ((float)m_width * m_height);
    }

private:
    struct Segment { uint32_t x, y, w; };

    uint32_t            m_width, m_height;
    std::vector<Segment> m_skyline;

    // Check whether a rect of (rw × rh) fits starting at segment i.
    // Fills out_x / out_y with placement position.
    bool FitsAt(int i, uint16_t rw, uint16_t rh,
                uint32_t& out_x, uint32_t& out_y) const
    {
        out_x = m_skyline[i].x;
        if (out_x + rw > m_width)
            return false;

        out_y = m_skyline[i].y;
        uint32_t remaining = rw;

        for (int j = i; j < (int)m_skyline.size() && remaining > 0; ++j)
        {
            out_y     = std::max(out_y, m_skyline[j].y);
            remaining = (remaining > m_skyline[j].w)
                        ? remaining - m_skyline[j].w
                        : 0;
        }

        return (out_y + rh <= m_height);
    }

    void AddSkylineLevel(int i, uint32_t x, uint32_t y,
                         uint16_t rw, uint16_t rh)
    {
        Segment new_seg{x, y + rh, rw};
        m_skyline.insert(m_skyline.begin() + i, new_seg);

        // Shrink / remove segments now covered by the new rect.
        for (int j = i + 1; j < (int)m_skyline.size(); )
        {
            auto& prev = m_skyline[j - 1];
            auto& cur  = m_skyline[j];

            if (cur.x < prev.x + prev.w)
            {
                uint32_t shrink = prev.x + prev.w - cur.x;
                if (shrink >= cur.w)
                {
                    m_skyline.erase(m_skyline.begin() + j);
                    continue;
                }
                cur.x += shrink;
                cur.w -= shrink;
            }
            break;
        }

        // Merge adjacent segments at the same y.
        for (int j = 0; j + 1 < (int)m_skyline.size(); )
        {
            if (m_skyline[j].y == m_skyline[j + 1].y)
            {
                m_skyline[j].w += m_skyline[j + 1].w;
                m_skyline.erase(m_skyline.begin() + j + 1);
            }
            else
                ++j;
        }
    }
};

} // namespace fac
