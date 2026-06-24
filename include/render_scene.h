#pragma once

#include "text_render_types.h"

#include <filesystem>
#include <vector>

namespace fac {

struct RenderScene
{
    std::filesystem::path image_path;
    std::vector<TextRegion> regions;
};

RenderScene LoadRenderSceneJson(const std::filesystem::path& json_path);

} // namespace fac
