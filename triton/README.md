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

`text_renderer_gpu` is the throughput-oriented render-only backend. It expects
explicit micro-batches plus prebuilt render commands:

- input `input_images`: `TYPE_UINT8`, dims `[ B, 1200, 900, 3 ]`
- input `command_version`: `TYPE_INT32`, dims `[ 1 ]`, currently `2`
- input `command_bytes`: `TYPE_UINT8`, dims `[ -1 ]`
- input `batch_bytes`: `TYPE_UINT8`, dims `[ -1 ]`
- output `rendered_image`: `TYPE_UINT8`, dims `[ B, 1200, 900, 3 ]`

The packed command ABI is versioned and validated in-process:

- `GpuRenderCommandV2` extends the old per-glyph command with `image_index`
- `GpuRenderBatchV2` keeps atlas-grouped command ranges
- `command_bytes.size()` must be divisible by `sizeof(GpuRenderCommandV2)`
- `batch_bytes.size()` must be divisible by `sizeof(GpuRenderBatchV2)`
- every command `image_index` must be `< B`
- every batch range must stay within the flattened command buffer

The shared planner-side helpers now live in the normal C++ API:

- `BuildLumaRenderRequest(...)`
- `BuildPlannedGpuRenderRequest(...)`
- `BuildCombinedGpuCommandBufferV2(...)`

These helpers preserve the current brightness-based auto-color rule by
planning from a single-channel luma image where `luma = (R + G + B) / 3`.
