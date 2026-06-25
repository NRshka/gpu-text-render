# Triton Backend Layout

This repository can build an optional custom Triton backend shared library when
configured with:

```bash
sudo apt-get install -y libfreetype-dev nlohmann-json3-dev cmake libwebp-dev

cmake -S . -B build \
  -DGPU_FONT_BUILD_TRITON_BACKEND=ON \
  -DTRITON_BACKEND_INCLUDE_DIR=/opt/tritonserver/include
cmake --build build --target text_renderer text_renderer_gpu font_asset_compiler
```

`TRITON_BACKEND_INCLUDE_DIR` should normally be the include root, not the
header file itself. The build now also accepts the full
`.../triton/core/tritonbackend.h` path and normalizes it automatically.

Both backends expect atlas assets in the Triton model version directory:

```text
model_repository/
  text_renderer/
    config.pbtxt
    1/
      atlas/
        atlas_16.bin
        atlas_24.bin
        ...
  text_renderer_gpu/
    config.pbtxt
    1/
      atlas/
        atlas_16.bin
        atlas_24.bin
        ...
```

`text_renderer` is the compatibility backend. Its contract matches the original
pipeline plan:

- input `inpainted_image`: `TYPE_UINT8`, dims `[ -1, -1, 3 ]`
- input `region_texts`: `TYPE_BYTES`, dims `[ -1 ]`
- input `region_original_texts`: `TYPE_BYTES`, dims `[ -1 ]`
- input `region_vertex_counts`: `TYPE_INT32`, dims `[ -1 ]`
- input `region_vertices`: `TYPE_FP32`, dims `[ -1, 2 ]`
- optional input `region_has_rgba`: `TYPE_BOOL`, dims `[ -1 ]`
- optional input `region_rgba`: `TYPE_UINT32`, dims `[ -1 ]`
- output `rendered_image`: `TYPE_UINT8`, dims `[ -1, -1, 3 ]`

`text_renderer_gpu` is the throughput-oriented render-only backend. It now
expects one image plus one prebuilt command buffer per Triton request:

- input `input_images`: `TYPE_UINT8`, dims `[ 1200, 900, 3 ]`
- input `command_version`: `TYPE_INT32`, dims `[ 1 ]`, currently `2`
- input `command_bytes`: `TYPE_UINT8`, dims `[ -1 ]`
- input `batch_bytes`: `TYPE_UINT8`, dims `[ -1 ]`
- output `rendered_image`: `TYPE_UINT8`, dims `[ 1200, 900, 3 ]`

The packed command ABI is versioned and validated in-process:

- `GpuRenderCommandV2` extends the old per-glyph command with `image_index`
- `GpuRenderBatchV2` keeps atlas-grouped command ranges
- `command_bytes.size()` must be divisible by `sizeof(GpuRenderCommandV2)`
- `batch_bytes.size()` must be divisible by `sizeof(GpuRenderBatchV2)`
- every command `image_index` must be `< 1` for a single request
- every batch range must stay within the flattened command buffer

The shared planner-side helpers now live in the normal C++ API:

- `BuildLumaRenderRequest(...)`
- `BuildPlannedGpuRenderRequest(...)`
- `BuildCombinedGpuCommandBufferV2(...)`
- `CombineGpuCommandBuffersV2(...)`

These helpers preserve the current brightness-based auto-color rule by
planning from a single-channel luma image where `luma = (R + G + B) / 3`.

## Recommended Deployment Shape

For correctness work, debugging, or ad hoc integration:

- use `text_renderer`
- send polygon and text tensors directly to Triton

For maximum throughput on large fixed-size workloads:

- run planning outside Triton on CPU
- build one `GpuCommandBufferV2` payload per image from luma images plus regions
- send many single-image requests to `text_renderer_gpu`
- let Triton dynamic batching merge ready requests on the fly

This keeps the expensive CPU-side geometry resolution, UTF-8 validation,
brightness sampling, overlap handling, and atlas-size selection out of the GPU
render backend.

## Planner-to-Renderer Flow

The intended production flow is:

1. Upstream produces the erased RGB image and translation metadata.
2. A CPU planner stage derives a single-channel luma plane using:
   - `luma = (R + G + B) / 3`
3. The planner builds `TextRegion[]` from polygons and texts.
4. The planner calls:
   - `BuildLumaRenderRequest(...)`
   - `BuildPlannedGpuRenderRequest(...)`
5. The planner sends one Triton request per image with:
   - `input_images`
   - `command_version`
   - `command_bytes`
   - `batch_bytes`
6. Triton dynamic batching groups compatible requests and
   `text_renderer_gpu` flattens them into one render batch internally.
7. The GPU backend performs only:
   - RGB -> RGBA staging
   - atlas-backed glyph rasterization
   - RGBA -> RGB output conversion

## Current Batching Semantics

`text_renderer`

- Triton config remains `max_batch_size: 0`
- requests are unbatched at the API level
- one request contains one image plus region tensors
- Triton may pass multiple requests to one execute call, but they are processed
  independently

`text_renderer_gpu`

- Triton config uses `max_batch_size: 8` plus `dynamic_batching {}`
- one Triton request contains exactly one image and one command payload
- Triton may pass multiple requests to one execute call
- the backend combines those requests into one contiguous image batch
- each request-local command buffer is validated with `image_index == 0`
- combined execution rebases `image_index` and merges atlas buckets internally

This means the optimized path now uses Triton scheduler-level dynamic batching
instead of caller-formed explicit micro-batches.

## Build Outputs

When Triton backend build support is enabled, CMake now produces:

- `text_renderer`
  - compatibility Triton backend shared library
- `text_renderer_gpu`
  - render-only Triton backend shared library
- `font_asset_compiler`
  - offline atlas generator

## Current Limitations

- `text_renderer_gpu` requires fixed per-request image shape `[1200, 900, 3]`
- packed command payloads must be built by a trusted planner using the same ABI
- each request must carry a single-image command buffer; explicit multi-image payloads are no longer the Triton contract
- the repo still uses the current codepoint/atlas pipeline and does not add a shaping engine
