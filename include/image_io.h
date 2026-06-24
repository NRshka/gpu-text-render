#pragma once

#include "text_renderer_cpu.h"

#include <cstdint>
#include <filesystem>

namespace fac {

ImageRgba8 LoadImageRgba8(const std::filesystem::path& path);

bool TryReadPngDimensions(const std::filesystem::path& path,
                          uint32_t& out_width,
                          uint32_t& out_height) noexcept;

bool TryReadWebpDimensions(const std::filesystem::path& path,
                           uint32_t& out_width,
                           uint32_t& out_height) noexcept;

bool TryReadImageDimensions(const std::filesystem::path& path,
                            uint32_t& out_width,
                            uint32_t& out_height) noexcept;

void WriteImagePng(const ImageRgba8& image,
                   const std::filesystem::path& path);

void WriteImageWebpLossless(const ImageRgba8& image,
                            const std::filesystem::path& path);

void WriteImageFile(const ImageRgba8& image,
                    const std::filesystem::path& path);

void WriteImagePpm(const ImageRgba8& image,
                   const std::filesystem::path& path);

} // namespace fac
