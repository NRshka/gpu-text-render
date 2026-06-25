cmake_minimum_required(VERSION 3.16)
project(font_asset_compiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_FONT_CUDA_ARCHITECTURES "75;80;86;89;90"
    CACHE STRING "CUDA architectures to compile for")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_FONT_CUDA_ARCHITECTURES})

option(GPU_FONT_BUILD_TRITON_BACKEND "Build the optional Triton backend shared library" OFF)
option(GPU_FONT_BUILD_PYTHON_PLANNER "Build the optional Python planner extension module" OFF)
set(TRITON_BACKEND_INCLUDE_DIR ""
    CACHE PATH "Directory containing Triton backend API headers")

enable_testing()

# ── Dependencies ──────────────────────────────────────────────────────────────

find_package(Freetype REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PNG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)

# ── font_asset_compiler ───────────────────────────────────────────────────────

add_executable(font_asset_compiler
    src/font_asset_compiler.cpp
)

target_include_directories(font_asset_compiler PRIVATE
    include
    ${FREETYPE_INCLUDE_DIRS}
)

target_link_libraries(font_asset_compiler PRIVATE
    Freetype::Freetype
    nlohmann_json::nlohmann_json
)

target_compile_options(font_asset_compiler PRIVATE
    -Wall -Wextra -O2
)

add_library(font_database STATIC src/font_database.cpp)
target_include_directories(font_database PUBLIC include)

add_library(text_layout STATIC src/text_layout.cpp)
target_include_directories(text_layout PUBLIC include)
target_link_libraries(text_layout PUBLIC font_database)
target_compile_options(text_layout PRIVATE -Wall -Wextra -O2)

add_library(render_request STATIC src/render_request.cpp)
target_include_directories(render_request PUBLIC include)
target_link_libraries(render_request PUBLIC
    gpu_command_buffer
    text_renderer_cpu
)
target_compile_options(render_request PRIVATE -Wall -Wextra -O2)

add_library(render_plan STATIC src/render_plan.cpp)
target_include_directories(render_plan PUBLIC include)
target_link_libraries(render_plan PUBLIC text_layout)
target_compile_options(render_plan PRIVATE -Wall -Wextra -O2)

add_library(gpu_command_buffer STATIC src/gpu_command_buffer.cpp)
target_include_directories(gpu_command_buffer PUBLIC include)
target_link_libraries(gpu_command_buffer PUBLIC render_plan)
target_compile_options(gpu_command_buffer PRIVATE -Wall -Wextra -O2)

add_library(gpu_device STATIC src/gpu_device.cpp)
target_include_directories(gpu_device PUBLIC include)
target_link_libraries(gpu_device PUBLIC CUDA::cudart)
target_compile_options(gpu_device PRIVATE -Wall -Wextra -O2)

add_library(gpu_atlas_manager STATIC src/gpu_atlas_manager.cpp)
target_include_directories(gpu_atlas_manager PUBLIC include)
target_link_libraries(gpu_atlas_manager PUBLIC font_database CUDA::cudart)
target_compile_options(gpu_atlas_manager PRIVATE -Wall -Wextra -O2)

add_library(gpu_renderer_cuda STATIC src/gpu_renderer_cuda.cu)
target_include_directories(gpu_renderer_cuda PUBLIC include)
target_link_libraries(gpu_renderer_cuda PUBLIC
    gpu_atlas_manager
    gpu_command_buffer
    gpu_device
    CUDA::cudart
)
set_target_properties(gpu_renderer_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(render_scene STATIC src/render_scene.cpp)
target_include_directories(render_scene PUBLIC include)
target_link_libraries(render_scene PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(render_scene PRIVATE -Wall -Wextra -O2)

add_library(image_io STATIC src/image_io.cpp)
target_include_directories(image_io PUBLIC include)
target_link_libraries(image_io PUBLIC text_renderer_cpu PNG::PNG PkgConfig::WEBP)
target_compile_options(image_io PRIVATE -Wall -Wextra -O2)

add_library(text_renderer_cpu STATIC src/text_renderer_cpu.cpp)
target_include_directories(text_renderer_cpu PUBLIC include)
target_link_libraries(text_renderer_cpu PUBLIC render_plan)
target_compile_options(text_renderer_cpu PRIVATE -Wall -Wextra -O2)

add_executable(render_scene_preview src/render_scene_preview.cpp)
target_link_libraries(render_scene_preview PRIVATE image_io render_scene gpu_renderer_cuda)
target_compile_options(render_scene_preview PRIVATE -Wall -Wextra -O2)

add_executable(benchmark_render_scene src/benchmark_render_scene.cpp)
target_link_libraries(benchmark_render_scene PRIVATE image_io render_scene gpu_renderer_cuda)
target_compile_options(benchmark_render_scene PRIVATE -Wall -Wextra -O2)

add_executable(test_font_database tests/test_font_database.cpp)
target_link_libraries(test_font_database PRIVATE font_database)
target_compile_options(test_font_database PRIVATE -Wall -Wextra -O2)

add_executable(test_text_render_types tests/test_text_render_types.cpp)
target_include_directories(test_text_render_types PRIVATE include)
target_compile_options(test_text_render_types PRIVATE -Wall -Wextra -O2)

add_executable(test_font_asset_abi tests/test_font_asset_abi.cpp)
target_link_libraries(test_font_asset_abi PRIVATE font_database)
target_compile_options(test_font_asset_abi PRIVATE -Wall -Wextra -O2)

add_executable(test_text_layout tests/test_text_layout.cpp)
target_link_libraries(test_text_layout PRIVATE text_layout)
target_compile_options(test_text_layout PRIVATE -Wall -Wextra -O2)

add_executable(test_render_request tests/test_render_request.cpp)
target_link_libraries(test_render_request PRIVATE render_request)
target_compile_options(test_render_request PRIVATE -Wall -Wextra -O2)

add_executable(test_text_renderer_cpu tests/test_text_renderer_cpu.cpp)
target_link_libraries(test_text_renderer_cpu PRIVATE text_renderer_cpu)
target_compile_options(test_text_renderer_cpu PRIVATE -Wall -Wextra -O2)

add_executable(test_render_scene tests/test_render_scene.cpp)
target_link_libraries(test_render_scene PRIVATE render_scene)
target_compile_options(test_render_scene PRIVATE -Wall -Wextra -O2)

add_executable(test_render_plan tests/test_render_plan.cpp)
target_link_libraries(test_render_plan PRIVATE text_renderer_cpu)
target_compile_options(test_render_plan PRIVATE -Wall -Wextra -O2)

add_executable(test_image_io tests/test_image_io.cpp)
target_link_libraries(test_image_io PRIVATE image_io)
target_compile_options(test_image_io PRIVATE -Wall -Wextra -O2)

add_executable(test_gpu_command_buffer tests/test_gpu_command_buffer.cpp)
target_link_libraries(test_gpu_command_buffer PRIVATE gpu_command_buffer)
target_compile_options(test_gpu_command_buffer PRIVATE -Wall -Wextra -O2)

add_executable(test_gpu_device tests/test_gpu_device.cpp)
target_link_libraries(test_gpu_device PRIVATE gpu_device)
target_compile_options(test_gpu_device PRIVATE -Wall -Wextra -O2)

add_executable(test_gpu_renderer_cuda tests/test_gpu_renderer_cuda.cpp)
target_link_libraries(test_gpu_renderer_cuda PRIVATE gpu_renderer_cuda text_renderer_cpu)
target_compile_options(test_gpu_renderer_cuda PRIVATE -Wall -Wextra -O2)

if (GPU_FONT_BUILD_PYTHON_PLANNER)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
        OUTPUT_VARIABLE PYBIND11_CMAKEDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKEDIR}")
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
    endif()

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(gpu_font_planner
        src/python_gpu_font_planner.cpp
    )
    target_include_directories(gpu_font_planner PRIVATE include)
    target_link_libraries(gpu_font_planner PRIVATE render_request)
    target_compile_options(gpu_font_planner PRIVATE -Wall -Wextra -O2)
    set_target_properties(gpu_font_planner PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
        OUTPUT_NAME "gpu_font_planner"
    )

    add_test(
        NAME test_gpu_font_planner_python
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_gpu_font_planner.py"
    )
    set_tests_properties(test_gpu_font_planner_python PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}"
    )
endif()

if (GPU_FONT_BUILD_TRITON_BACKEND)
    if (TRITON_BACKEND_INCLUDE_DIR STREQUAL "")
        message(FATAL_ERROR
            "GPU_FONT_BUILD_TRITON_BACKEND=ON requires TRITON_BACKEND_INCLUDE_DIR")
    endif()

    set(TRITON_BACKEND_INCLUDE_ROOT "${TRITON_BACKEND_INCLUDE_DIR}")
    if (EXISTS "${TRITON_BACKEND_INCLUDE_ROOT}"
        AND NOT IS_DIRECTORY "${TRITON_BACKEND_INCLUDE_ROOT}")
        get_filename_component(TRITON_BACKEND_INCLUDE_ROOT
            "${TRITON_BACKEND_INCLUDE_ROOT}" DIRECTORY)
        get_filename_component(TRITON_BACKEND_INCLUDE_ROOT
            "${TRITON_BACKEND_INCLUDE_ROOT}" DIRECTORY)
        get_filename_component(TRITON_BACKEND_INCLUDE_ROOT
            "${TRITON_BACKEND_INCLUDE_ROOT}" DIRECTORY)
    endif()

    if (NOT EXISTS "${TRITON_BACKEND_INCLUDE_ROOT}/triton/core/tritonbackend.h")
        message(FATAL_ERROR
            "TRITON_BACKEND_INCLUDE_DIR must be /path/to/include or the full "
            "path to tritonbackend.h so that "
            "'${TRITON_BACKEND_INCLUDE_ROOT}/triton/core/tritonbackend.h' exists")
    endif()

    add_library(triton_text_renderer_backend SHARED
        src/triton_text_renderer_backend.cpp
    )
    target_include_directories(triton_text_renderer_backend PRIVATE
        include
        ${TRITON_BACKEND_INCLUDE_ROOT}
    )
    target_link_libraries(triton_text_renderer_backend PRIVATE
        render_request
        gpu_renderer_cuda
        gpu_command_buffer
        gpu_atlas_manager
        render_plan
        text_renderer_cpu
        font_database
        CUDA::cudart
    )
    target_compile_options(triton_text_renderer_backend PRIVATE -Wall -Wextra -O2)
    set_target_properties(triton_text_renderer_backend PROPERTIES
        PREFIX ""
        OUTPUT_NAME "text_renderer"
    )

    add_custom_target(text_renderer
        DEPENDS triton_text_renderer_backend
    )

    add_library(triton_text_renderer_gpu_backend SHARED
        src/triton_text_renderer_gpu_backend.cpp
    )
    target_include_directories(triton_text_renderer_gpu_backend PRIVATE
        include
        ${TRITON_BACKEND_INCLUDE_ROOT}
    )
    target_link_libraries(triton_text_renderer_gpu_backend PRIVATE
        gpu_renderer_cuda
        gpu_command_buffer
        gpu_atlas_manager
        font_database
        CUDA::cudart
    )
    target_compile_options(triton_text_renderer_gpu_backend PRIVATE -Wall -Wextra -O2)
    set_target_properties(triton_text_renderer_gpu_backend PROPERTIES
        PREFIX ""
        OUTPUT_NAME "text_renderer_gpu"
    )

    add_custom_target(text_renderer_gpu
        DEPENDS triton_text_renderer_gpu_backend
    )
endif()

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS font_asset_compiler DESTINATION bin)
install(FILES
    include/font_asset.h
    include/gpu_atlas_manager.h
    include/gpu_command_buffer.h
    include/gpu_device.h
    include/gpu_renderer_cuda.h
    include/image_io.h
    include/render_plan.h
    include/render_request.h
    include/render_scene.h
    include/text_layout.h
    include/text_renderer_cpu.h
    include/text_render_types.h
    include/skyline_packer.h
    DESTINATION include
)
