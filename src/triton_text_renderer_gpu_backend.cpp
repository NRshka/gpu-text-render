#include "gpu_atlas_manager.h"
#include "gpu_command_buffer.h"
#include "gpu_device.h"
#include "gpu_renderer_cuda.h"

#include <cuda_runtime_api.h>
#include <triton/core/tritonbackend.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace fac {
namespace triton_gpu_backend {

namespace {

constexpr const char* kBackendName = "text_renderer_gpu";
constexpr const char* kInputImagesName = "input_images";
constexpr const char* kInputCommandVersionName = "command_version";
constexpr const char* kInputCommandBytesName = "command_bytes";
constexpr const char* kInputBatchBytesName = "batch_bytes";
constexpr const char* kOutputImagesName = "rendered_image";
constexpr uint32_t kExpectedHeight = 1200;
constexpr uint32_t kExpectedWidth = 900;

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

fs::path ResolveAtlasDir(const fs::path& repository_location, uint64_t version)
{
    const fs::path versioned_atlas_dir =
        repository_location / std::to_string(version) / "atlas";
    if (fs::exists(versioned_atlas_dir))
        return versioned_atlas_dir;

    const fs::path direct_atlas_dir = repository_location / "atlas";
    return direct_atlas_dir;
}

TRITONSERVER_Error* ReadImageBatchTensor(TRITONBACKEND_Request* request,
                                         uint32_t& batch_size,
                                         uint32_t& width,
                                         uint32_t& height,
                                         const uint8_t*& input_ptr,
                                         GpuBufferMemoryType& input_memory,
                                         std::vector<uint8_t>& host_fallback)
{
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, kInputImagesName, &input));

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

    if (dtype != TRITONSERVER_TYPE_UINT8)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "input_images must have TYPE_UINT8");
    }

    if (dims_count != 4
        || shape[0] <= 0
        || shape[1] != (int64_t)kExpectedHeight
        || shape[2] != (int64_t)kExpectedWidth
        || shape[3] != 3)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "input_images must have shape [B, 1200, 900, 3]");
    }

    batch_size = (uint32_t)shape[0];
    height = (uint32_t)shape[1];
    width = (uint32_t)shape[2];
    const std::size_t expected_size = (std::size_t)batch_size * width * height * 3u;
    if (byte_size != expected_size)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "input_images byte size does not match shape [B, 1200, 900, 3]");
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

    input_ptr = reinterpret_cast<const uint8_t*>(buffer);
    input_memory = (memory_type == TRITONSERVER_MEMORY_GPU)
        ? GpuBufferMemoryType::Device
        : GpuBufferMemoryType::Host;
    return nullptr;
}

TRITONSERVER_Error* ReadCommandVersion(TRITONBACKEND_Request* request,
                                       int32_t& version)
{
    std::vector<int32_t> values;
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(ReadNumericTensor(
        request, kInputCommandVersionName, TRITONSERVER_TYPE_INT32, values, &shape));

    if (shape.empty())
    {
        if (values.size() != 1u)
        {
            return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                                "command_version scalar must contain exactly one value");
        }
    }
    else if (!(shape.size() == 1 && shape[0] == 1))
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "command_version must be a scalar or shape [1]");
    }

    if (values.size() != 1u)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "command_version must contain exactly one value");
    }

    version = values[0];
    return nullptr;
}

TRITONSERVER_Error* ProcessRequest(InstanceState& instance_state,
                                   TRITONBACKEND_Request* request)
{
    uint32_t batch_size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    const uint8_t* input_rgb = nullptr;
    GpuBufferMemoryType input_memory = GpuBufferMemoryType::Host;
    std::vector<uint8_t> host_image;
    RETURN_IF_ERROR(ReadImageBatchTensor(
        request, batch_size, width, height, input_rgb, input_memory, host_image));

    int32_t command_version = 0;
    RETURN_IF_ERROR(ReadCommandVersion(request, command_version));
    if (command_version != kGpuCommandBufferAbiVersion)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "unsupported command_version; expected 2");
    }

    std::vector<uint8_t> command_bytes;
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(ReadNumericTensor(
        request, kInputCommandBytesName, TRITONSERVER_TYPE_UINT8, command_bytes, &shape));
    if (shape.size() != 1 || shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "command_bytes must have shape [N]");
    }

    std::vector<uint8_t> batch_bytes;
    RETURN_IF_ERROR(ReadNumericTensor(
        request, kInputBatchBytesName, TRITONSERVER_TYPE_UINT8, batch_bytes, &shape));
    if (shape.size() != 1 || shape[0] < 0)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG,
                            "batch_bytes must have shape [N]");
    }

    GpuCommandBufferV2 buffer;
    try
    {
        buffer = DeserializeGpuCommandBufferV2(command_bytes.data(),
                                               command_bytes.size(),
                                               batch_bytes.data(),
                                               batch_bytes.size(),
                                               batch_size);
    }
    catch (const std::exception& e)
    {
        return BackendError(TRITONSERVER_ERROR_INVALID_ARG, e.what());
    }

    TRITONBACKEND_Response* response = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

    const int64_t output_shape[4] = {
        (int64_t)batch_size,
        (int64_t)height,
        (int64_t)width,
        3,
    };
    TRITONBACKEND_Output* output = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        response, &output, kOutputImagesName, TRITONSERVER_TYPE_UINT8, output_shape, 4));

    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
    int64_t output_memory_id = instance_state.device_id;
    void* output_buffer = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
        output,
        &output_buffer,
        (uint64_t)batch_size * width * height * 3u,
        &output_memory_type,
        &output_memory_id));

    const GpuBufferMemoryType render_output_memory =
        (output_memory_type == TRITONSERVER_MEMORY_GPU)
            ? GpuBufferMemoryType::Device
            : GpuBufferMemoryType::Host;

    instance_state.renderer.RenderRgbBatch(reinterpret_cast<uint8_t*>(output_buffer),
                                           render_output_memory,
                                           input_rgb,
                                           input_memory,
                                           batch_size,
                                           width,
                                           height,
                                           instance_state.atlas_manager,
                                           buffer);

    RETURN_IF_ERROR(
        TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));
    return nullptr;
}

} // namespace

} // namespace triton_gpu_backend
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
    using namespace fac::triton_gpu_backend;

    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &name));

    TRITONBACKEND_ArtifactType artifact_type = TRITONBACKEND_ARTIFACT_FILESYSTEM;
    const char* repository_path = nullptr;
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelRepository(model, &artifact_type, &repository_path));

    if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM)
    {
        return BackendError(TRITONSERVER_ERROR_UNSUPPORTED,
                            "text_renderer_gpu Triton backend only supports filesystem model artifacts");
    }

    uint64_t version = 0;
    RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

    auto state = std::make_shared<ModelState>();
    state->name = name != nullptr ? name : kBackendName;
    state->atlas_dir = ResolveAtlasDir(fs::path(repository_path), version);

    auto* boxed_state = new std::shared_ptr<ModelState>(std::move(state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, boxed_state));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
    void* state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &state));
    delete reinterpret_cast<std::shared_ptr<fac::triton_gpu_backend::ModelState>*>(state);
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, nullptr));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance)
{
    using namespace fac::triton_gpu_backend;

    TRITONBACKEND_Model* model = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    auto state = std::make_unique<InstanceState>(SharedModelState(model));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &state->device_id));

    if (!fac::HasCudaDevice())
    {
        return BackendError(TRITONSERVER_ERROR_UNAVAILABLE,
                            "text_renderer_gpu Triton backend requires a visible CUDA device");
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

    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, state.release()));
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance)
{
    using namespace fac::triton_gpu_backend;

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
    using namespace fac::triton_gpu_backend;

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
