#include "image_io.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace fac;

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

static void WriteBytes(const std::filesystem::path& path,
                       const std::vector<unsigned char>& bytes)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            (std::streamsize)bytes.size());
}

static ImageRgba8 MakeTinyImage()
{
    ImageRgba8 image(2, 2, 0x00000000u);

    uint8_t* p00 = image.PixelPtr(0, 0);
    p00[0] = 255; p00[1] = 0;   p00[2] = 0;   p00[3] = 255;

    uint8_t* p10 = image.PixelPtr(1, 0);
    p10[0] = 0;   p10[1] = 255; p10[2] = 0;   p10[3] = 255;

    uint8_t* p01 = image.PixelPtr(0, 1);
    p01[0] = 0;   p01[1] = 0;   p01[2] = 255; p01[3] = 255;

    uint8_t* p11 = image.PixelPtr(1, 1);
    p11[0] = 255; p11[1] = 255; p11[2] = 255; p11[3] = 128;

    return image;
}

int main()
{
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "gpu_font_rendering_tests" / "image_io";
    fs::create_directories(dir);

    SECTION("PNG dimensions");
    {
        const fs::path path = dir / "tiny.png";
        const std::vector<unsigned char> bytes = {
            0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n',
            0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R',
            0x00, 0x00, 0x03, 0x84, // 900
            0x00, 0x00, 0x04, 0xB0, // 1200
        };
        WriteBytes(path, bytes);

        uint32_t w = 0, h = 0;
        EXPECT(TryReadPngDimensions(path, w, h));
        EXPECT(w == 900u);
        EXPECT(h == 1200u);
        EXPECT(TryReadImageDimensions(path, w, h));
    }

    SECTION("WEBP VP8X dimensions");
    {
        const fs::path path = dir / "vp8x.webp";
        const std::vector<unsigned char> bytes = {
            'R','I','F','F', 22,0,0,0, 'W','E','B','P',
            'V','P','8','X', 10,0,0,0,
            0, 0,0,0,
            0x83,0x03,0x00, // 900 - 1
            0xAF,0x04,0x00  // 1200 - 1
        };
        WriteBytes(path, bytes);

        uint32_t w = 0, h = 0;
        EXPECT(TryReadWebpDimensions(path, w, h));
        EXPECT(w == 900u);
        EXPECT(h == 1200u);
        EXPECT(TryReadImageDimensions(path, w, h));
    }

    SECTION("WEBP VP8L dimensions");
    {
        const fs::path path = dir / "vp8l.webp";
        const std::vector<unsigned char> bytes = {
            'R','I','F','F', 17,0,0,0, 'W','E','B','P',
            'V','P','8','L', 5,0,0,0,
            0x2F,
            0x83, 0xC3, 0x2B, 0x01
        };
        WriteBytes(path, bytes);

        uint32_t w = 0, h = 0;
        EXPECT(TryReadWebpDimensions(path, w, h));
        EXPECT(w == 900u);
        EXPECT(h == 1200u);
    }

    SECTION("WEBP VP8 dimensions");
    {
        const fs::path path = dir / "vp8.webp";
        const std::vector<unsigned char> bytes = {
            'R','I','F','F', 22,0,0,0, 'W','E','B','P',
            'V','P','8',' ', 10,0,0,0,
            0,0,0,
            0x9D,0x01,0x2A,
            0x84,0x03, // 900
            0xB0,0x04  // 1200
        };
        WriteBytes(path, bytes);

        uint32_t w = 0, h = 0;
        EXPECT(TryReadWebpDimensions(path, w, h));
        EXPECT(w == 900u);
        EXPECT(h == 1200u);
    }

    SECTION("PNG round trip");
    {
        const fs::path path = dir / "roundtrip.png";
        const ImageRgba8 src = MakeTinyImage();
        WriteImagePng(src, path);

        uint32_t w = 0, h = 0;
        EXPECT(TryReadImageDimensions(path, w, h));
        EXPECT(w == 2u);
        EXPECT(h == 2u);

        const ImageRgba8 decoded = LoadImageRgba8(path);
        EXPECT(decoded.width == 2u);
        EXPECT(decoded.height == 2u);
        EXPECT(decoded.pixels == src.pixels);
    }

    SECTION("WEBP lossless round trip");
    {
        const fs::path path = dir / "roundtrip.webp";
        const ImageRgba8 src = MakeTinyImage();
        WriteImageWebpLossless(src, path);

        uint32_t w = 0, h = 0;
        EXPECT(TryReadImageDimensions(path, w, h));
        EXPECT(w == 2u);
        EXPECT(h == 2u);

        const ImageRgba8 decoded = LoadImageRgba8(path);
        EXPECT(decoded.width == 2u);
        EXPECT(decoded.height == 2u);
        EXPECT(decoded.pixels == src.pixels);
    }

    std::cout << "\n────────────────────────────────\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
