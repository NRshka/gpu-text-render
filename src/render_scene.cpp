#include "render_scene.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fac {

namespace {

using json = nlohmann::json;

static float ReadFloat(const json& obj, const char* key)
{
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_number())
        throw std::runtime_error(std::string("RenderScene: missing numeric field '")
                                 + key + "'");
    return it->get<float>();
}

static bool TryReadUint32(const json& obj, const char* key, uint32_t& out)
{
    auto it = obj.find(key);
    if (it == obj.end())
        return false;
    if (!it->is_number_unsigned())
        throw std::runtime_error(std::string("RenderScene: field '")
                                 + key + "' must be an unsigned integer");
    out = it->get<uint32_t>();
    return true;
}

static Vec2f ParsePoint(const json& point_json)
{
    if (point_json.is_array() && point_json.size() == 2
        && point_json[0].is_number() && point_json[1].is_number())
    {
        return Vec2f{
            point_json[0].get<float>(),
            point_json[1].get<float>(),
        };
    }

    if (point_json.is_object())
    {
        return Vec2f{
            ReadFloat(point_json, "x"),
            ReadFloat(point_json, "y"),
        };
    }

    throw std::runtime_error("RenderScene: point must be [x, y] or {x, y}");
}

static bool TryReadBoxObject(const json& region, OrientedBox& out)
{
    auto it = region.find("obb");
    if (it == region.end() || !it->is_object())
    {
        it = region.find("oob");
        if (it == region.end() || !it->is_object())
            return false;
    }

    out = OrientedBox{};
    const json& box_json = *it;
    out.cx     = ReadFloat(box_json, "cx");
    out.cy     = ReadFloat(box_json, "cy");
    out.ux     = ReadFloat(box_json, "ux");
    out.uy     = ReadFloat(box_json, "uy");
    out.vx     = ReadFloat(box_json, "vx");
    out.vy     = ReadFloat(box_json, "vy");
    out.width  = ReadFloat(box_json, "width");
    out.height = ReadFloat(box_json, "height");

    if (!out.HasArea() || !out.HasFiniteBasis())
        throw std::runtime_error("RenderScene: invalid oriented box");

    const float u_len2 = out.ux * out.ux + out.uy * out.uy;
    const float v_len2 = out.vx * out.vx + out.vy * out.vy;
    const float dot    = out.ux * out.vx + out.uy * out.vy;
    if (std::fabs(u_len2 - 1.0f) > 1e-2f
        || std::fabs(v_len2 - 1.0f) > 1e-2f
        || std::fabs(dot) > 1e-2f)
    {
        throw std::runtime_error("RenderScene: oriented box basis must be orthonormal");
    }

    return true;
}

static std::vector<Vec2f> ParsePolygon(const json& polygon_json)
{
    if (!polygon_json.is_array())
        throw std::runtime_error("RenderScene: 'polygon' must be an array");

    std::vector<Vec2f> points;
    points.reserve(polygon_json.size());
    for (const auto& point_json : polygon_json)
    {
        const Vec2f point = ParsePoint(point_json);
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
            throw std::runtime_error("RenderScene: polygon points must be finite");
        points.push_back(point);
    }

    if (points.size() < 3)
        throw std::runtime_error("RenderScene: polygon must contain at least 3 points");

    return points;
}

static CurvedTextPath ParseCurve(const json& curve_json)
{
    if (!curve_json.is_object())
        throw std::runtime_error("RenderScene: 'curve' must be an object");

    auto segments_it = curve_json.find("segments");
    if (segments_it == curve_json.end() || !segments_it->is_array())
        throw std::runtime_error("RenderScene: curve missing 'segments' array");

    CurvedTextPath curve;
    curve.band_height = ReadFloat(curve_json, "band_height");

    auto normal_it = curve_json.find("normal_side");
    if (normal_it != curve_json.end())
    {
        if (!normal_it->is_string())
            throw std::runtime_error("RenderScene: curve 'normal_side' must be a string");

        const std::string side = normal_it->get<std::string>();
        if (side == "left")
            curve.normal_side = CurveNormalSide::Left;
        else if (side == "right")
            curve.normal_side = CurveNormalSide::Right;
        else
            throw std::runtime_error("RenderScene: curve 'normal_side' must be 'left' or 'right'");
    }

    curve.segments.reserve(segments_it->size());
    for (const auto& segment_json : *segments_it)
    {
        if (!segment_json.is_object())
            throw std::runtime_error("RenderScene: each curve segment must be an object");

        CubicBezierSegment segment;
        segment.p0 = ParsePoint(segment_json.at("p0"));
        segment.p1 = ParsePoint(segment_json.at("p1"));
        segment.p2 = ParsePoint(segment_json.at("p2"));
        segment.p3 = ParsePoint(segment_json.at("p3"));
        curve.segments.push_back(segment);
    }

    if (!curve.IsValid())
        throw std::runtime_error("RenderScene: invalid curve geometry");

    return curve;
}

static TextRegion ParseRegionObject(const json& region_json, std::size_t cluster_id)
{
    if (!region_json.is_object())
        throw std::runtime_error("RenderScene: each region must be an object");

    auto text_it = region_json.find("text");
    if (text_it == region_json.end() || !text_it->is_string())
        throw std::runtime_error("RenderScene: region missing 'text' string");

    auto original_text_it = region_json.find("original_text");
    if (original_text_it == region_json.end() || !original_text_it->is_string())
        throw std::runtime_error("RenderScene: region missing 'original_text' string");

    TextRegion region;
    region.text = text_it->get<std::string>();
    region.original_text = original_text_it->get<std::string>();
    region.cluster_id = cluster_id;
    region.has_explicit_rgba = TryReadUint32(region_json, "rgba", region.rgba);

    auto polygon_it = region_json.find("polygon");
    if (polygon_it != region_json.end())
    {
        region.polygon = ParsePolygon(*polygon_it);
        region.has_polygon = true;
    }

    auto curve_it = region_json.find("curve");
    if (curve_it != region_json.end())
    {
        region.curve = ParseCurve(*curve_it);
        region.has_curve = true;
    }

    if (TryReadBoxObject(region_json, region.box))
    {
    }
    else if (!region.has_curve && !region.has_polygon)
    {
        throw std::runtime_error(
            "RenderScene: region must contain 'curve', 'polygon', or 'obb'");
    }

    return region;
}

} // namespace

RenderScene LoadRenderSceneJson(const std::filesystem::path& json_path)
{
    std::ifstream f(json_path);
    if (!f)
        throw std::runtime_error("RenderScene: cannot open " + json_path.string());

    json root;
    f >> root;

    if (!root.is_object())
        throw std::runtime_error("RenderScene: root JSON must be an object");

    auto image_it = root.find("image");
    if (image_it == root.end() || !image_it->is_string())
        throw std::runtime_error("RenderScene: missing 'image' string");

    auto regions_it = root.find("regions");
    if (regions_it == root.end() || !regions_it->is_array())
        throw std::runtime_error("RenderScene: missing 'regions' array");

    RenderScene scene;
    scene.image_path = json_path.parent_path() / image_it->get<std::string>();
    scene.regions.reserve(regions_it->size());

    std::size_t cluster_id = 0;
    for (const auto& item_json : *regions_it)
    {
        if (item_json.is_object())
        {
            scene.regions.push_back(ParseRegionObject(
                item_json,
                TextRegion::kUnclustered));
            continue;
        }

        if (!item_json.is_array())
            throw std::runtime_error(
                "RenderScene: each regions entry must be a region object or an array of region objects");

        for (const auto& region_json : item_json)
            scene.regions.push_back(ParseRegionObject(region_json, cluster_id));
        ++cluster_id;
    }

    return scene;
}

} // namespace fac
