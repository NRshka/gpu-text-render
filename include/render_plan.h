#pragma once

#include "font_database.h"
#include "text_layout.h"
#include "text_render_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fac {

enum class PlannerProfile
{
    Quality,
    Adaptive,
};

struct ImageRgba8;

struct RenderBatch
{
    uint32_t atlas_render_size = 0;
    std::vector<RenderGlyph> glyphs;
};

struct RenderPlan
{
    std::vector<RenderBatch> batches;
    std::size_t total_regions     = 0;
    std::size_t fitted_regions    = 0;
    std::size_t overflowed_regions = 0;
    std::size_t unplaced_regions  = 0;
    std::size_t total_glyphs      = 0;
    std::size_t fast_regions      = 0;
    std::size_t quality_regions   = 0;
    std::size_t escalated_regions = 0;
};

struct RenderPlanOptions
{
    PlannerProfile profile = PlannerProfile::Quality;
    TextFitOptions fit;
    bool align_lines = false;
    bool resolve_overlaps = true;
    bool allow_overflow = true;
    bool skip_if_unplaceable = true;
    float line_y_tolerance = 0.55f;
    float min_line_group_px = 8.0f;
    float overlap_margin_px = 4.0f;
    float collision_margin_px = 4.0f;
    float max_nudge_frac = 0.35f;
    int nudge_steps = 12;
    float shrink_factor = 0.95f;
    float scale_step_factor = 0.94f;
    float max_tangent_overflow_frac = 0.35f;
    float max_tangent_overflow_px = 48.0f;
    float max_normal_overflow_frac = 0.20f;
    float max_normal_overflow_px = 12.0f;
    int candidate_tangent_steps = 8;
    int candidate_normal_steps = 4;
    float min_line_height_px = 8.0f;
};

RenderPlan BuildRenderPlan(const FontDatabase& db,
                           const ImageRgba8& image,
                           const std::vector<TextRegion>& regions,
                           const RenderPlanOptions& options = {});

RenderPlan BuildAdaptiveRenderPlan(const FontDatabase& db,
                                   uint32_t width,
                                   uint32_t height,
                                   const std::vector<uint8_t>& image_luma,
                                   const std::vector<TextRegion>& regions,
                                   const RenderPlanOptions& options = {});

} // namespace fac
