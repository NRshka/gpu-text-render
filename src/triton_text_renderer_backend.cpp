#include "gpu_atlas_manager.h"
#include "gpu_command_buffer.h"
#include "gpu_device.h"
#include "gpu_renderer_cuda.h"
#include "render_plan.h"
#include "render_request.h"

#include <cuda_runtime_api.h>
#include <triton/core/tritonbackend.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace fac {
namespace triton_backend {

namespace {

constexpr const char* kBackendName = "text_renderer";
constexpr const char* kInputImageName = "inpainted_image";
constexpr const char* kInputTextsName = "region_texts";
constexpr const char* kInputOriginalTextsName = "region_original_texts";
constexpr const char* kInputVertexCountsName = "region_vertex_counts";
constexpr const char* kInputVerticesName = "region_vertices";
constexpr const char* kInputHasRgbaName = "region_has_rgba";
constexpr const char* kInputRgbaName = "region_rgba";
constexpr const char* kOutputImageName = "rendered_image";

#define RETURN_IF_ERROR(EXPR)            \
    do {                                 \
        TRITONSERVER_Error* err__ = (EXPR); \
        if (err__ != nullptr)            \
            return err__;                \
    } while (0)

TRITONSERVER_Error* BackendError(TRITONSERVER_Error_Code code,
                                 const std::string& message)
{
    return TRITONSERVER_ErrorNew(code, message.c_str());
}

template <typename T>
TRITONSERVER_Error* CopyBufferToHost(const void* src,
                                     std::size_t byte_size,
                                     TRITONSERVER_MemoryType memory_type,
                                     std::vector<T>& out)
{
    if (byte_size % sizeof(T) != 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "tensor byte size is not aligned with the requested element type");
    }

    out.resize(byte_size / sizeof(T));
    if (byte_size == 0)
        return nullptr;

    if (memory_type == TRITONSERVER_MEMORY_CPU)
    {
        std::memcpy(out.data(), src, byte_size);
        return nullptr;
    }

    if (memory_type == TRITONSERVER_MEMORY_GPU)
    {
        cudaError_t err = cudaMemcpy(out.data(), src, byte_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            return BackendError(TRITONSERVER_ERROR_INTERNAL,
                                std::string("cudaMemcpy device->host failed: ")
                                    + cudaGetErrorString(err));
        }
        return nullptr;
    }

    return BackendError(TRITONSERVER_ERROR_UNSUPPORTED,
                        "unsupported Triton memory type");
}

TRITONSERVER_Error* CollectTensorBytes(TRITONBACKEND_Input* input,
                                       std::vector<uint8_t>& out,
                                       TRITONSERVER_MemoryType* single_buffer_memory_type,
                                       int64_t* single_buffer_memory_id,
                                       const void** single_buffer_ptr)
{
    const char* input_name = nullptr;
    TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &dtype, &shape, &dims_count, &byte_size, &buffer_count));
    (void)input_name;
    (void)dtype;
    (void)shape;
    (void)dims_count;

    if (buffer_count == 1)
    {
        uint64_t chunk_size = 0;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t memory_id = 0;
        const void* buffer = nullptr;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            input, 0, &buffer, &chunk_size, &memory_type, &memory_id));

        if (chunk_size != byte_size)
        {
            return BackendError(TRITONSERVER_ERROR_INTERNAL,
                                "single input buffer size does not match tensor byte size");
        }

        if (single_buffer_memory_type != nullptr)
            *single_buffer_memory_type = memory_type;
        if (single_buffer_memory_id != nullptr)
            *single_buffer_memory_id = memory_id;
        if (single_buffer_ptr != nullptr)
            *single_buffer_ptr = buffer;
        out.clear();
        return nullptr;
    }

    out.clear();
    out.reserve((std::size_t)byte_size);
    for (uint32_t i = 0; i < buffer_count; ++i)
    {
        const void* buffer = nullptr;
        uint64_t chunk_size = 0;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t memory_id = 0;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            input, i, &buffer, &chunk_size, &memory_type, &memory_id));
        (void)memory_id;

        if (memory_type == TRITONSERVER_MEMORY_CPU)
        {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(buffer);
            out.insert(out.end(), bytes, bytes + chunk_size);
            continue;
        }

        if (memory_type == TRITONSERVER_MEMORY_GPU)
        {
            const std::size_t old_size = out.size();
            out.resize(old_size + (std::size_t)chunk_size);
            cudaError_t err = cudaMemcpy(out.data() + old_size,
                                         buffer,
                                         chunk_size,
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                return BackendError(TRITONSERVER_ERROR_INTERNAL,
                                    std::string("cudaMemcpy device->host failed: ")
                                        + cudaGetErrorString(err));
            }
            continue;
        }

        return BackendError(TRITONSERVER_ERROR_UNSUPPORTED,
                            "unsupported Triton memory type");
    }

    if (out.size() != (std::size_t)byte_size)
    {
        return BackendError(TRITONSERVER_ERROR_INTERNAL,
                            "input buffers did not reconstruct the expected byte size");
    }

    if (single_buffer_memory_type != nullptr)
        *single_buffer_memory_type = TRITONSERVER_MEMORY_CPU;
    if (single_buffer_memory_id != nullptr)
        *single_buffer_memory_id = 0;
    if (single_buffer_ptr != nullptr)
        *single_buffer_ptr = out.data();
    return nullptr;
}

template <typename T>
TRITONSERVER_Error* ReadNumericTensor(TRITONBACKEND_Request* request,
                                      const char* input_name,
                                      TRITONSERVER_DataType expected_type,
                                      std::vector<T>& out,
                                      std::vector<int64_t>* shape_out)
{
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, input_name, &input));

    const char* actual_name = nullptr;
    TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &actual_name, &dtype, &shape, &dims_count, &byte_size, &buffer_count));
    (void)actual_name;
    (void)buffer_count;

    if (dtype != expected_type)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("unexpected datatype for input '") + input_name + "'");
    }

    if (shape_out != nullptr)
        shape_out->assign(shape, shape + dims_count);

    std::vector<uint8_t> host_bytes;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    const void* buffer = nullptr;
    RETURN_IF_ERROR(
        CollectTensorBytes(input, host_bytes, &memory_type, &memory_id, &buffer));
    (void)memory_id;

    if (!host_bytes.empty())
        buffer = host_bytes.data();

    return CopyBufferToHost(buffer, byte_size, memory_type, out);
}

TRITONSERVER_Error* ParseBytesTensor(const std::vector<uint8_t>& raw,
                                     std::vector<std::string>& out)
{
    out.clear();
    std::size_t offset = 0;
    while (offset < raw.size())
    {
        if (offset + 4u > raw.size())
        {
            return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                                "malformed BYTES tensor: truncated length prefix");
        }

        const uint32_t element_size =
            (uint32_t)raw[offset + 0]
            | ((uint32_t)raw[offset + 1] << 8)
            | ((uint32_t)raw[offset + 2] << 16)
            | ((uint32_t)raw[offset + 3] << 24);
        offset += 4u;

        if (offset + element_size > raw.size())
        {
            return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                                "malformed BYTES tensor: element extends beyond input size");
        }

        out.emplace_back(reinterpret_cast<const char*>(raw.data() + offset), element_size);
        offset += element_size;
    }

    return nullptr;
}

TRITONSERVER_Error* ReadBytesTensor(TRITONBACKEND_Request* request,
                                    const char* input_name,
                                    std::vector<std::string>& out,
                                    std::vector<int64_t>* shape_out)
{
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, input_name, &input));

    const char* actual_name = nullptr;
    TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &actual_name, &dtype, &shape, &dims_count, &byte_size, &buffer_count));
    (void)actual_name;
    (void)buffer_count;

    if (dtype != TRITONSERVER_TYPE_BYTES)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("unexpected datatype for input '") + input_name + "'");
    }

    if (shape_out != nullptr)
        shape_out->assign(shape, shape + dims_count);

    std::vector<uint8_t> host_bytes;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    const void* buffer = nullptr;
    RETURN_IF_ERROR(
        CollectTensorBytes(input, host_bytes, &memory_type, &memory_id, &buffer));
    (void)memory_id;

    if (!host_bytes.empty())
        buffer = host_bytes.data();

    std::vector<uint8_t> contiguous;
    RETURN_IF_ERROR(CopyBufferToHost(buffer, byte_size, memory_type, contiguous));
    return ParseBytesTensor(contiguous, out);
}

TRITONSERVER_Error* TryReadOptionalNumericTensor(TRITONBACKEND_Request* request,
                                                 const char* input_name,
                                                 TRITONSERVER_DataType expected_type,
                                                 bool& present,
                                                 std::vector<uint8_t>& raw,
                                                 std::vector<int64_t>* shape_out)
{
    TRITONBACKEND_Input* input = nullptr;
    TRITONSERVER_Error* err = TRITONBACKEND_RequestInput(request, input_name, &input);
    if (err != nullptr)
    {
        const std::string message = TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_ErrorDelete(err);
        if (message.find("not found") != std::string::npos)
        {
            present = false;
            raw.clear();
            if (shape_out != nullptr)
                shape_out->clear();
            return nullptr;
        }
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG, message);
    }

    present = true;

    const char* actual_name = nullptr;
    TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &actual_name, &dtype, &shape, &dims_count, &byte_size, &buffer_count));
    (void)actual_name;
    (void)buffer_count;

    if (dtype != expected_type)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("unexpected datatype for input '") + input_name + "'");
    }

    if (shape_out != nullptr)
        shape_out->assign(shape, shape + dims_count);

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    const void* buffer = nullptr;
    RETURN_IF_ERROR(CollectTensorBytes(input, raw, &memory_type, &memory_id, &buffer));
    (void)memory_id;

    if (!raw.empty())
    {
        return nullptr;
    }

    return CopyBufferToHost(buffer, byte_size, memory_type, raw);
}

template <typename T>
std::vector<T> ReinterpretBytes(const std::vector<uint8_t>& raw)
{
    if (raw.empty())
        return {};

    if (raw.size() % sizeof(T) != 0)
        throw std::runtime_error("raw tensor byte size is misaligned");

    std::vector<T> out(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

struct ModelState
{
    std::string name;
    fs::path atlas_dir;
};

struct InstanceState
{
    explicit InstanceState(std::shared_ptr<ModelState> model_state)
        : model(std::move(model_state))
    {
    }

    std::shared_ptr<ModelState> model;
    int device_id = 0;
    FontDatabase font_database;
    GpuAtlasManager atlas_manager;
    GpuRendererCuda renderer;
};

std::shared_ptr<ModelState> SharedModelState(TRITONBACKEND_Model* model)
{
    void* state = nullptr;
    TRITONBACKEND_ModelState(model, &state);
    return *reinterpret_cast<std::shared_ptr<ModelState>*>(state);
}

InstanceState* GetInstanceState(TRITONBACKEND_ModelInstance* instance)
{
    void* state = nullptr;
    TRITONBACKEND_ModelInstanceState(instance, &state);
    return reinterpret_cast<InstanceState*>(state);
}

TRITONSERVER_Error* ReadImageTensor(TRITONBACKEND_Request* request,
                                    uint32_t& width,
                                    uint32_t& height,
                                    const uint8_t*& input_ptr,
                                    GpuBufferMemoryType& input_memory,
                                    std::vector<uint8_t>& host_fallback)
{
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, kInputImageName, &input));

    const char* actual_name = nullptr;
    TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &actual_name, &dtype, &shape, &dims_count, &byte_size, &buffer_count));
    (void)actual_name;

    if (dtype != TRITONSERVER_TYPE_UINT8)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "inpainted_image must have TYPE_UINT8");
    }

    if (dims_count != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "inpainted_image must have shape [H, W, 3]");
    }

    width = (uint32_t)shape[1];
    height = (uint32_t)shape[0];
    const std::size_t expected_size = (std::size_t)width * height * 3u;
    if (byte_size != expected_size)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "inpainted_image byte size does not match shape [H, W, 3]");
    }

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    const void* buffer = nullptr;
    RETURN_IF_ERROR(
        CollectTensorBytes(input, host_fallback, &memory_type, &memory_id, &buffer));
    (void)memory_id;

    if (!host_fallback.empty())
    {
        input_ptr = host_fallback.data();
        input_memory = GpuBufferMemoryType::Host;
        return nullptr;
    }

    if (memory_type == TRITONSERVER_MEMORY_GPU)
    {
        input_ptr = reinterpret_cast<const uint8_t*>(buffer);
        input_memory = GpuBufferMemoryType::Device;
        return nullptr;
    }

    input_ptr = reinterpret_cast<const uint8_t*>(buffer);
    input_memory = GpuBufferMemoryType::Host;
    return nullptr;
}

TRITONSERVER_Error* BuildRegionTensorData(TRITONBACKEND_Request* request,
                                          PolygonRegionTensorData& data)
{
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(ReadBytesTensor(request, kInputTextsName, data.region_texts, &shape));
    if (shape.size() != 1 || shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_texts must have shape [N]");
    }

    RETURN_IF_ERROR(
        ReadBytesTensor(request, kInputOriginalTextsName, data.region_original_texts, &shape));
    if (shape.size() != 1 || shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_original_texts must have shape [N]");
    }

    RETURN_IF_ERROR(
        ReadNumericTensor(request, kInputVertexCountsName, TRITONSERVER_TYPE_INT32,
                          data.region_vertex_counts, &shape));
    if (shape.size() != 1 || shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_vertex_counts must have shape [N]");
    }

    std::vector<float> vertices_raw;
    RETURN_IF_ERROR(
        ReadNumericTensor(request, kInputVerticesName, TRITONSERVER_TYPE_FP32,
                          vertices_raw, &shape));
    if (shape.size() != 2 || shape[0] < 0 || shape[1] != 2)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_vertices must have shape [V, 2]");
    }
    data.region_vertices.resize(vertices_raw.size() / 2u);
    for (std::size_t i = 0; i < data.region_vertices.size(); ++i)
    {
        data.region_vertices[i] = Vec2f{
            vertices_raw[i * 2u + 0u],
            vertices_raw[i * 2u + 1u],
        };
    }

    bool has_rgba_present = false;
    std::vector<uint8_t> has_rgba_raw;
    std::vector<int64_t> optional_shape;
    RETURN_IF_ERROR(TryReadOptionalNumericTensor(
        request, kInputHasRgbaName, TRITONSERVER_TYPE_BOOL,
        has_rgba_present, has_rgba_raw, &optional_shape));

    bool rgba_present = false;
    std::vector<uint8_t> rgba_raw;
    RETURN_IF_ERROR(TryReadOptionalNumericTensor(
        request, kInputRgbaName, TRITONSERVER_TYPE_UINT32,
        rgba_present, rgba_raw, &shape));

    const std::size_t region_count = data.region_texts.size();
    if (has_rgba_present != rgba_present)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_has_rgba and region_rgba must either both be present or both be absent");
    }

    if (!has_rgba_present)
    {
        data.region_has_rgba.assign(region_count, 0u);
        data.region_rgba.assign(region_count, 0u);
        return nullptr;
    }

    if (optional_shape.size() != 1 || optional_shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_has_rgba must have shape [N]");
    }
    if (shape.size() != 1 || shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "region_rgba must have shape [N]");
    }

    data.region_has_rgba = has_rgba_raw;
    data.region_rgba = ReinterpretBytes<uint32_t>(rgba_raw);
    return nullptr;
}

TRITONSERVER_Error* ProcessRequest(InstanceState& instance_state,
                                   TRITONBACKEND_Request* request)
{
    uint32_t width = 0;
    uint32_t height = 0;
    const uint8_t* input_rgb = nullptr;
    GpuBufferMemoryType input_memory = GpuBufferMemoryType::Host;
    std::vector<uint8_t> host_image;
    RETURN_IF_ERROR(
        ReadImageTensor(request, width, height, input_rgb, input_memory, host_image));

    PolygonRegionTensorData region_data;
    RETURN_IF_ERROR(BuildRegionTensorData(request, region_data));

    const std::vector<TextRegion> regions = BuildTextRegionsFromPolygonTensorData(region_data);
    ImageRgba8 cpu_reference(width, height, 0x000000FFu);
    if (input_memory == GpuBufferMemoryType::Host)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                const std::size_t src_idx = ((std::size_t)y * width + x) * 3u;
                uint8_t* p = cpu_reference.PixelPtr(x, y);
                p[0] = input_rgb[src_idx + 0u];
                p[1] = input_rgb[src_idx + 1u];
                p[2] = input_rgb[src_idx + 2u];
                p[3] = 255u;
            }
        }
    }
    else
    {
        cpu_reference = ImageRgba8(width, height, 0x000000FFu);
    }

    const RenderPlan plan =
        BuildRenderPlan(instance_state.font_database, cpu_reference, regions);
    const GpuCommandBuffer buffer =
        BuildGpuCommandBuffer(instance_state.font_database, plan);

    TRITONBACKEND_Response* response = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

    const int64_t output_shape[3] = {
        (int64_t)height,
        (int64_t)width,
        3,
    };
    TRITONBACKEND_Output* output = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        response, &output, kOutputImageName, TRITONSERVER_TYPE_UINT8, output_shape, 3));

    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
    int64_t output_memory_id = instance_state.device_id;
    void* output_buffer = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output,
                                               &output_buffer,
                                               (uint64_t)width * height * 3u,
                                               &output_memory_type,
                                               &output_memory_id));

    const GpuBufferMemoryType render_output_memory =
        (output_memory_type == TRITONSERVER_MEMORY_GPU)
            ? GpuBufferMemoryType::Device
            : GpuBufferMemoryType::Host;

    instance_state.renderer.RenderRgb(reinterpret_cast<uint8_t*>(output_buffer),
                                      render_output_memory,
                                      input_rgb,
                                      input_memory,
                                      width,
                                      height,
                                      instance_state.atlas_manager,
                                      buffer);

    RETURN_IF_ERROR(
        TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));
    return nullptr;
}

} // namespace

} // namespace triton_backend
} // namespace fac

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
    (void)backend;
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
    (void)backend;
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
    using namespace fac::triton_backend;

    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &name));

    const char* repository_path = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelRepositoryPath(model, &repository_path));

    int64_t version = 0;
    RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

    auto state = std::make_shared<ModelState>();
    state->name = name != nullptr ? name : kBackendName;
    state->atlas_dir = fs::path(repository_path) / std::to_string(version) / "atlas";

    auto* boxed_state = new std::shared_ptr<ModelState>(std::move(state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, boxed_state));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
    void* state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &state));
    delete reinterpret_cast<std::shared_ptr<fac::triton_backend::ModelState>*>(state);
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, nullptr));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance)
{
    using namespace fac::triton_backend;

    TRITONBACKEND_Model* model = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    auto state = std::make_unique<InstanceState>(SharedModelState(model));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &state->device_id));

    if (!fac::HasCudaDevice())
    {
        return BackendError(TRITONSERVER_ERROR_UNAVAILABLE,
                            "text_renderer Triton backend requires a visible CUDA device");
    }

    if (!fs::exists(state->model->atlas_dir))
    {
        return BackendError(TRITONSERVER_ERROR_NOT_FOUND,
                            "atlas directory does not exist: "
                                + state->model->atlas_dir.string());
    }

    state->font_database.LoadDirectory(state->model->atlas_dir);
    state->atlas_manager.Upload(state->font_database);
    state->renderer.Initialize(state->device_id);

    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceSetState(instance, state.release()));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance)
{
    using namespace fac::triton_backend;

    InstanceState* state = GetInstanceState(instance);
    delete state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, nullptr));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    using namespace fac::triton_backend;

    InstanceState* state = GetInstanceState(instance);
    for (uint32_t i = 0; i < request_count; ++i)
    {
        TRITONSERVER_Error* err = nullptr;
        try
        {
            err = ProcessRequest(*state, requests[i]);
        }
        catch (const std::exception& e)
        {
            err = BackendError(TRITONSERVER_ERROR_INTERNAL, e.what());
        }

        if (err != nullptr)
        {
            TRITONBACKEND_Response* response = nullptr;
            TRITONBACKEND_ResponseNew(&response, requests[i]);
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
        }

        TRITONBACKEND_RequestRelease(
            requests[i], TRITONSERVER_REQUEST_RELEASE_ALL);
    }

    return nullptr;
}

} // extern "C"
