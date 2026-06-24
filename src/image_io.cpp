#include "image_io.h"

#include <png.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <array>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fac {

namespace {

static uint32_t ReadBe32(const unsigned char* p) noexcept
{
    return ((uint32_t)p[0] << 24)
         | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] << 8)
         |  (uint32_t)p[3];
}

static uint16_t ReadLe16(const unsigned char* p) noexcept
{
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static uint32_t ReadLe24(const unsigned char* p) noexcept
{
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16);
}

static std::string LowerExt(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    for (char& c : ext)
        c = (char)std::tolower((unsigned char)c);
    return ext;
}

static std::vector<uint8_t> ReadWholeFile(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("LoadImageRgba8: cannot open " + path.string());

    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 0)
        throw std::runtime_error("LoadImageRgba8: cannot size " + path.string());
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes((std::size_t)size);
    f.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!f)
        throw std::runtime_error("LoadImageRgba8: cannot read " + path.string());
    return bytes;
}

} // namespace

ImageRgba8 LoadImageRgba8(const std::filesystem::path& path)
{
    const std::string ext = LowerExt(path);

    if (ext == ".png")
    {
        png_image img{};
        img.version = PNG_IMAGE_VERSION;

        if (!png_image_begin_read_from_file(&img, path.c_str()))
            throw std::runtime_error("LoadImageRgba8: cannot read PNG " + path.string());

        img.format = PNG_FORMAT_RGBA;

        ImageRgba8 out;
        out.width = img.width;
        out.height = img.height;
        out.pixels.resize(PNG_IMAGE_SIZE(img));

        if (!png_image_finish_read(&img, nullptr, out.pixels.data(), 0, nullptr))
        {
            png_image_free(&img);
            throw std::runtime_error("LoadImageRgba8: PNG decode failed for " + path.string());
        }

        png_image_free(&img);
        return out;
    }

    if (ext == ".webp")
    {
        const std::vector<uint8_t> bytes = ReadWholeFile(path);

        int width = 0;
        int height = 0;
        if (!WebPGetInfo(bytes.data(), bytes.size(), &width, &height)
            || width <= 0 || height <= 0)
        {
            throw std::runtime_error("LoadImageRgba8: invalid WebP " + path.string());
        }

        uint8_t* decoded = WebPDecodeRGBA(bytes.data(), bytes.size(), &width, &height);
        if (!decoded)
            throw std::runtime_error("LoadImageRgba8: WebP decode failed for " + path.string());

        ImageRgba8 out;
        out.width = (uint32_t)width;
        out.height = (uint32_t)height;
        out.pixels.assign(decoded, decoded + (std::size_t)width * height * 4u);
        WebPFree(decoded);
        return out;
    }

    throw std::runtime_error("LoadImageRgba8: unsupported image type " + path.string());
}

bool TryReadPngDimensions(const std::filesystem::path& path,
                          uint32_t& out_width,
                          uint32_t& out_height) noexcept
{
    out_width = 0;
    out_height = 0;

    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    std::array<unsigned char, 24> header{};
    f.read(reinterpret_cast<char*>(header.data()), (std::streamsize)header.size());
    if (!f)
        return false;

    static constexpr unsigned char kPngSig[8] = {
        0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'
    };

    for (int i = 0; i < 8; ++i)
    {
        if (header[(std::size_t)i] != kPngSig[i])
            return false;
    }

    // The first chunk must be IHDR for a valid PNG.
    if (!(header[12] == 'I' && header[13] == 'H'
       && header[14] == 'D' && header[15] == 'R'))
        return false;

    out_width = ReadBe32(header.data() + 16);
    out_height = ReadBe32(header.data() + 20);
    return out_width > 0 && out_height > 0;
}

bool TryReadWebpDimensions(const std::filesystem::path& path,
                           uint32_t& out_width,
                           uint32_t& out_height) noexcept
{
    out_width = 0;
    out_height = 0;

    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    std::array<unsigned char, 30> header{};
    f.read(reinterpret_cast<char*>(header.data()), (std::streamsize)header.size());
    const std::streamsize n = f.gcount();
    if (n < 16)
        return false;

    if (!(header[0] == 'R' && header[1] == 'I'
       && header[2] == 'F' && header[3] == 'F'
       && header[8] == 'W' && header[9] == 'E'
       && header[10] == 'B' && header[11] == 'P'))
        return false;

    if (header[12] == 'V' && header[13] == 'P'
     && header[14] == '8' && header[15] == 'X')
    {
        if (n < 30)
            return false;
        out_width = 1u + ReadLe24(header.data() + 24);
        out_height = 1u + ReadLe24(header.data() + 27);
        return out_width > 0 && out_height > 0;
    }

    if (header[12] == 'V' && header[13] == 'P'
     && header[14] == '8' && header[15] == 'L')
    {
        if (n < 25 || header[20] != 0x2Fu)
            return false;

        out_width = 1u + (((uint32_t)header[21]
                         | ((uint32_t)header[22] << 8)) & 0x3FFFu);
        out_height = 1u + ((((uint32_t)header[22] >> 6)
                          | ((uint32_t)header[23] << 2)
                          | ((uint32_t)header[24] << 10)) & 0x3FFFu);
        return out_width > 0 && out_height > 0;
    }

    if (header[12] == 'V' && header[13] == 'P'
     && header[14] == '8' && header[15] == ' ')
    {
        if (n < 30)
            return false;
        if (!(header[23] == 0x9Du && header[24] == 0x01u && header[25] == 0x2Au))
            return false;

        out_width = (uint32_t)(ReadLe16(header.data() + 26) & 0x3FFFu);
        out_height = (uint32_t)(ReadLe16(header.data() + 28) & 0x3FFFu);
        return out_width > 0 && out_height > 0;
    }

    return false;
}

bool TryReadImageDimensions(const std::filesystem::path& path,
                            uint32_t& out_width,
                            uint32_t& out_height) noexcept
{
    return TryReadPngDimensions(path, out_width, out_height)
        || TryReadWebpDimensions(path, out_width, out_height);
}

void WriteImagePng(const ImageRgba8& image,
                   const std::filesystem::path& path)
{
    if (image.Empty())
        throw std::runtime_error("WriteImagePng: image is empty");

    png_image img{};
    img.version = PNG_IMAGE_VERSION;
    img.width = image.width;
    img.height = image.height;
    img.format = PNG_FORMAT_RGBA;

    if (!png_image_write_to_file(&img, path.c_str(), 0,
                                 image.pixels.data(), 0, nullptr))
    {
        png_image_free(&img);
        throw std::runtime_error("WriteImagePng: failed to write " + path.string());
    }

    png_image_free(&img);
}

void WriteImageWebpLossless(const ImageRgba8& image,
                            const std::filesystem::path& path)
{
    if (image.Empty())
        throw std::runtime_error("WriteImageWebpLossless: image is empty");

    uint8_t* encoded = nullptr;
    const size_t size = WebPEncodeLosslessRGBA(image.pixels.data(),
                                               (int)image.width,
                                               (int)image.height,
                                               (int)image.width * 4,
                                               &encoded);
    if (size == 0 || encoded == nullptr)
        throw std::runtime_error("WriteImageWebpLossless: encode failed");

    std::ofstream f(path, std::ios::binary);
    if (!f)
    {
        WebPFree(encoded);
        throw std::runtime_error("WriteImageWebpLossless: cannot open " + path.string());
    }

    f.write(reinterpret_cast<const char*>(encoded), (std::streamsize)size);
    WebPFree(encoded);
    if (!f)
        throw std::runtime_error("WriteImageWebpLossless: failed while writing " + path.string());
}

void WriteImageFile(const ImageRgba8& image,
                    const std::filesystem::path& path)
{
    const std::string ext = LowerExt(path);
    if (ext == ".png")
        return WriteImagePng(image, path);
    if (ext == ".webp")
        return WriteImageWebpLossless(image, path);
    if (ext == ".ppm")
        return WriteImagePpm(image, path);

    throw std::runtime_error("WriteImageFile: unsupported image extension " + path.string());
}

void WriteImagePpm(const ImageRgba8& image,
                   const std::filesystem::path& path)
{
    if (image.Empty())
        throw std::runtime_error("WriteImagePpm: image is empty");

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("WriteImagePpm: cannot open " + path.string());

    f << "P6\n" << image.width << " " << image.height << "\n255\n";

    for (uint32_t y = 0; y < image.height; ++y)
    {
        for (uint32_t x = 0; x < image.width; ++x)
        {
            const uint8_t* p = image.PixelPtr(x, y);
            f.put((char)p[0]);
            f.put((char)p[1]);
            f.put((char)p[2]);
        }
    }

    if (!f)
        throw std::runtime_error("WriteImagePpm: failed while writing " + path.string());
}

} // namespace fac
