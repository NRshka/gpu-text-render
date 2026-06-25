#include "font_database.h"
#include "render_request.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr ssize_t kExpectedHeight = 1200;
constexpr ssize_t kExpectedWidth = 900;
constexpr ssize_t kExpectedChannels = 3;

std::vector<std::string> ResolveOriginalTexts(
    const std::vector<std::string>& texts,
    const py::object& original_texts_obj)
{
    if (original_texts_obj.is_none())
        return texts;

    const std::vector<std::string> original_texts =
        original_texts_obj.cast<std::vector<std::string>>();
    if (original_texts.size() != texts.size())
        throw py::value_error("original_texts must match texts length");

    return original_texts;
}

void FillPolygonRegionData(
    fac::PolygonRegionTensorData& out,
    const std::vector<std::string>& texts,
    const std::vector<std::vector<std::array<float, 2>>>& polygons,
    const std::vector<std::string>& original_texts,
    const py::object& rgba_obj)
{
    if (texts.size() != polygons.size())
        throw py::value_error("texts and polygons must have the same number of regions");

    out.region_texts = texts;
    out.region_original_texts = original_texts;
    out.region_vertex_counts.reserve(polygons.size());

    std::size_t total_points = 0;
    for (const auto& polygon : polygons)
    {
        if (polygon.size() < 3u)
            throw py::value_error("Each polygon must contain at least 3 points");

        out.region_vertex_counts.push_back((int32_t)polygon.size());
        total_points += polygon.size();
    }

    out.region_vertices.reserve(total_points);
    for (const auto& polygon : polygons)
    {
        for (const auto& point : polygon)
            out.region_vertices.push_back(fac::Vec2f{point[0], point[1]});
    }

    const std::size_t region_count = texts.size();
    if (rgba_obj.is_none())
    {
        out.region_has_rgba.assign(region_count, 0u);
        out.region_rgba.assign(region_count, 0u);
        return;
    }

    const std::vector<uint32_t> rgba = rgba_obj.cast<std::vector<uint32_t>>();
    if (rgba.size() != region_count)
        throw py::value_error("rgba must match texts length when provided");

    out.region_has_rgba.assign(region_count, 1u);
    out.region_rgba = rgba;
}

std::vector<uint8_t> BuildLumaFromRgb(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image_rgb)
{
    const py::buffer_info info = image_rgb.request();
    if (info.ndim != 3
        || info.shape[0] != kExpectedHeight
        || info.shape[1] != kExpectedWidth
        || info.shape[2] != kExpectedChannels)
    {
        throw py::value_error(
            "image_rgb must have shape (1200, 900, 3) and dtype uint8");
    }

    const uint8_t* rgb = static_cast<const uint8_t*>(info.ptr);
    std::vector<uint8_t> luma((std::size_t)kExpectedHeight * (std::size_t)kExpectedWidth);
    for (std::size_t i = 0; i < luma.size(); ++i)
    {
        const std::size_t offset = i * 3u;
        const uint32_t value = (uint32_t)rgb[offset + 0u]
                             + (uint32_t)rgb[offset + 1u]
                             + (uint32_t)rgb[offset + 2u];
        luma[i] = (uint8_t)(value / 3u);
    }

    return luma;
}

class GpuRenderPlanner
{
public:
    explicit GpuRenderPlanner(const std::string& atlas_dir)
    {
        try
        {
            const int atlas_count = m_font_database.LoadDirectory(atlas_dir);
            if (atlas_count <= 0 || m_font_database.Empty())
                throw std::runtime_error("GpuRenderPlanner: no atlas_*.bin files found in " + atlas_dir);
        }
        catch (const py::error_already_set&)
        {
            throw;
        }
        catch (const std::exception& e)
        {
            throw py::value_error(e.what());
        }
    }

    std::pair<py::bytes, py::bytes> Plan(
        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image_rgb,
        const std::vector<std::string>& texts,
        const std::vector<std::vector<std::array<float, 2>>>& polygons,
        const py::object& original_texts_obj,
        const py::object& rgba_obj) const
    {
        try
        {
            const std::vector<std::string> original_texts =
                ResolveOriginalTexts(texts, original_texts_obj);

            fac::PolygonRegionTensorData region_data;
            FillPolygonRegionData(region_data, texts, polygons, original_texts, rgba_obj);

            std::vector<uint8_t> luma = BuildLumaFromRgb(image_rgb);

            fac::PlannedGpuRenderRequest planned;
            {
                py::gil_scoped_release release;
                fac::LumaRenderRequest request = fac::BuildLumaRenderRequest(
                    (uint32_t)kExpectedWidth,
                    (uint32_t)kExpectedHeight,
                    std::move(luma),
                    region_data);
                planned = fac::BuildPlannedGpuRenderRequest(m_font_database, request, 0u);
            }

            return {
                py::bytes(reinterpret_cast<const char*>(planned.command_bytes.data()),
                          (py::ssize_t)planned.command_bytes.size()),
                py::bytes(reinterpret_cast<const char*>(planned.batch_bytes.data()),
                          (py::ssize_t)planned.batch_bytes.size()),
            };
        }
        catch (const py::error_already_set&)
        {
            throw;
        }
        catch (const std::exception& e)
        {
            throw py::value_error(e.what());
        }
    }

private:
    fac::FontDatabase m_font_database;
};

} // namespace

PYBIND11_MODULE(gpu_font_planner, m)
{
    m.doc() = "Python planner bridge for text_renderer_gpu command buffers";

    py::class_<GpuRenderPlanner>(m, "GpuRenderPlanner")
        .def(py::init<const std::string&>(), py::arg("atlas_dir"))
        .def("plan",
             &GpuRenderPlanner::Plan,
             py::arg("image_rgb"),
             py::arg("texts"),
             py::arg("polygons"),
             py::arg("original_texts") = py::none(),
             py::arg("rgba") = py::none());
}
