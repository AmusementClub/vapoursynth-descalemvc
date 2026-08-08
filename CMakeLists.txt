cmake_minimum_required(VERSION 3.24)

project(dsmvc VERSION 0.1.0 LANGUAGES CXX)

if(MSVC)
    # The plugin boundary is a C ABI. Static CRT linkage prevents the host's
    # bundled, potentially older MSVC runtime from changing STL synchronization ABI.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include(CTest)

set(DSMVC_VAPOURSYNTH_SDK "" CACHE PATH "Path to the VapourSynth SDK directory")
set(DSMVC_VS_PYTHON "" CACHE FILEPATH "Python executable from the target VapourSynth environment")
set(DSMVC_BASELINE_PLUGIN "" CACHE FILEPATH "Baseline descale plugin used by integration tests")
set(DSMVC_BASELINE_WRAPPER "" CACHE FILEPATH "Baseline descale.py used by integration tests")
option(DSMVC_ENABLE_CUDA
       "Build the optional CUDA backend on Windows or Linux" OFF)
set(DSMVC_CUDA_MIN_ARCHITECTURE "75" CACHE STRING
    "Minimum compute capability accepted by the CUDA backend")
set(DSMVC_CUDA_ARCHITECTURES
    "75;80;86;87;88;89;90;100;103;110;120;121" CACHE STRING
    "Native SM targets embedded in the CUDA fatbin")
set(DSMVC_CUDA_PTX_ARCHITECTURES "75;121" CACHE STRING
    "Virtual compute targets embedded as CUDA PTX fallbacks")
option(DSMVC_BUILD_BENCHMARKS "Build the fixed-recipe CPU benchmark" OFF)

set(DSMVC_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _dsmvc_osx_arch_count)
    if(NOT _dsmvc_osx_arch_count EQUAL 1)
        message(FATAL_ERROR
            "Universal macOS builds require separate arm64 and x86_64 builds")
    endif()
    list(GET CMAKE_OSX_ARCHITECTURES 0 DSMVC_TARGET_PROCESSOR)
endif()
string(TOLOWER "${DSMVC_TARGET_PROCESSOR}" DSMVC_TARGET_PROCESSOR)

set(_dsmvc_metal_default OFF)
if(APPLE AND DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_dsmvc_metal_default ON)
endif()
if(DEFINED CACHE{DSMVC_BUILD_METAL_EXPERIMENTS})
    set(_dsmvc_legacy_metal_option_defined ON)
else()
    set(_dsmvc_legacy_metal_option_defined OFF)
endif()
option(DSMVC_ENABLE_METAL
    "Build the Apple Silicon UMA Metal backend" ${_dsmvc_metal_default})
option(DSMVC_BUILD_METAL_EXPERIMENTS
    "Deprecated alias for DSMVC_ENABLE_METAL" ${DSMVC_ENABLE_METAL})
if(_dsmvc_legacy_metal_option_defined)
    set(DSMVC_ENABLE_METAL ${DSMVC_BUILD_METAL_EXPERIMENTS})
else()
    set(DSMVC_BUILD_METAL_EXPERIMENTS ${DSMVC_ENABLE_METAL})
endif()

if(APPLE AND DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    add_compile_options(
        "$<$<CONFIG:Release>:-O3;-flto=full>"
        "$<$<CONFIG:RelWithDebInfo>:-O3;-flto=full>")
    add_link_options(
        "$<$<CONFIG:Release>:-flto=full>"
        "$<$<CONFIG:RelWithDebInfo>:-flto=full>")
endif()

if(DSMVC_VAPOURSYNTH_SDK)
    set(_dsmvc_vs_hints "${DSMVC_VAPOURSYNTH_SDK}/include")
endif()
find_path(DSMVC_VAPOURSYNTH_INCLUDE_DIR VapourSynth4.h
    HINTS
        ${_dsmvc_vs_hints}
        "$ENV{VAPOURSYNTH_SDK}/include"
        "D:/okegui/OKEGui/tools/vapoursynth/sdk/include"
)
if(NOT DSMVC_VAPOURSYNTH_INCLUDE_DIR)
    message(FATAL_ERROR "VapourSynth API4 headers were not found; set DSMVC_VAPOURSYNTH_SDK")
endif()

set(DSMVC_ENGINE_SOURCES
    src/axis_plan.cpp
    src/backend.cpp
    src/cpu_executor.cpp
    src/executor.cpp
)

set(DSMVC_CAN_BUILD_AVX2 OFF)
set(DSMVC_CAN_BUILD_NEON OFF)
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND DSMVC_TARGET_PROCESSOR MATCHES "^(amd64|x86_64)$")
    set(DSMVC_CAN_BUILD_AVX2 ON)
    list(APPEND DSMVC_ENGINE_SOURCES src/cpu_executor_avx2.cpp)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8 AND DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(DSMVC_CAN_BUILD_NEON ON)
    list(APPEND DSMVC_ENGINE_SOURCES src/cpu_executor_neon.cpp)
endif()

if(DSMVC_ENABLE_CUDA)
    if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "DSMVC_ENABLE_CUDA currently requires Windows or Linux")
    endif()
    find_program(DSMVC_NVCC_EXECUTABLE
        NAMES nvcc.exe nvcc
        HINTS "$ENV{CUDA_PATH}/bin" "/usr/local/cuda/bin")
    find_program(DSMVC_CUOBJDUMP_EXECUTABLE
        NAMES cuobjdump.exe cuobjdump
        HINTS "$ENV{CUDA_PATH}/bin" "/usr/local/cuda/bin")
    find_path(DSMVC_CUDA_INCLUDE_DIR
        NAMES cuda.h
        HINTS "$ENV{CUDA_PATH}/include" "/usr/local/cuda/include")
    if(NOT DSMVC_NVCC_EXECUTABLE OR NOT DSMVC_CUOBJDUMP_EXECUTABLE
       OR NOT DSMVC_CUDA_INCLUDE_DIR)
        message(FATAL_ERROR
            "DSMVC_ENABLE_CUDA requires a CUDA Toolkit with nvcc, cuobjdump, and cuda.h")
    endif()
    if(NOT DSMVC_CUDA_MIN_ARCHITECTURE MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "DSMVC_CUDA_MIN_ARCHITECTURE must be one numeric SM value")
    endif()
    if(DSMVC_CUDA_ARCHITECTURES STREQUAL ""
       OR DSMVC_CUDA_PTX_ARCHITECTURES STREQUAL "")
        message(FATAL_ERROR "CUDA native and PTX architecture lists must not be empty")
    endif()
    list(REMOVE_DUPLICATES DSMVC_CUDA_ARCHITECTURES)
    list(REMOVE_DUPLICATES DSMVC_CUDA_PTX_ARCHITECTURES)
    foreach(_architecture IN LISTS
            DSMVC_CUDA_ARCHITECTURES DSMVC_CUDA_PTX_ARCHITECTURES)
        if(NOT _architecture MATCHES "^[0-9]+$")
            message(FATAL_ERROR "CUDA architecture lists must contain numeric SM values")
        endif()
        if(_architecture LESS DSMVC_CUDA_MIN_ARCHITECTURE)
            message(FATAL_ERROR
                "CUDA architecture ${_architecture} is below the configured minimum ${DSMVC_CUDA_MIN_ARCHITECTURE}")
        endif()
    endforeach()
    list(FIND DSMVC_CUDA_ARCHITECTURES
         "${DSMVC_CUDA_MIN_ARCHITECTURE}" _minimum_native_index)
    list(FIND DSMVC_CUDA_PTX_ARCHITECTURES
         "${DSMVC_CUDA_MIN_ARCHITECTURE}" _minimum_ptx_index)
    if(_minimum_native_index EQUAL -1 OR _minimum_ptx_index EQUAL -1)
        message(FATAL_ERROR
            "The minimum CUDA architecture must be present in native and PTX lists")
    endif()

    set(_dsmvc_cuda_code_args)
    foreach(_architecture IN LISTS DSMVC_CUDA_ARCHITECTURES)
        list(APPEND _dsmvc_cuda_code_args
            "--generate-code=arch=compute_${_architecture},code=sm_${_architecture}")
    endforeach()
    foreach(_architecture IN LISTS DSMVC_CUDA_PTX_ARCHITECTURES)
        list(APPEND _dsmvc_cuda_code_args
            "--generate-code=arch=compute_${_architecture},code=compute_${_architecture}")
    endforeach()

    set(_dsmvc_cuda_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda")
    set(_dsmvc_cuda_fatbin "${_dsmvc_cuda_generated_dir}/dsmvc_cuda.fatbin")
    set(_dsmvc_cuda_header "${_dsmvc_cuda_generated_dir}/dsmvc_cuda_fatbin.hpp")
    set(_dsmvc_cuda_resource "${_dsmvc_cuda_generated_dir}/dsmvc_cuda_fatbin.rc")
    set(_dsmvc_cuda_host_options)
    if(CMAKE_HOST_WIN32)
        list(APPEND _dsmvc_cuda_host_options -Xcompiler=/utf-8)
    endif()
    add_custom_command(
        OUTPUT "${_dsmvc_cuda_fatbin}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_dsmvc_cuda_generated_dir}"
        COMMAND "${DSMVC_NVCC_EXECUTABLE}"
                --fatbin --cudart=none --machine=64 --std=c++17 -O3 -lineinfo
                ${_dsmvc_cuda_host_options}
                ${_dsmvc_cuda_code_args}
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/dsmvc_cuda.cu"
                -o "${_dsmvc_cuda_fatbin}"
        DEPENDS
            src/cuda/dsmvc_cuda.cu
            src/cuda/cuda_kernel.hpp
        COMMENT "Building the dsmvc CUDA fatbin"
        VERBATIM)
    if(WIN32)
        enable_language(RC)
        configure_file(
            cmake/cuda_fatbin_resource.hpp.in
            "${_dsmvc_cuda_header}" @ONLY)
        configure_file(
            cmake/cuda_fatbin.rc.in
            "${_dsmvc_cuda_resource}" @ONLY)
        set_source_files_properties(
            "${_dsmvc_cuda_resource}" PROPERTIES OBJECT_DEPENDS "${_dsmvc_cuda_fatbin}")
        set(_dsmvc_cuda_embed_outputs "${_dsmvc_cuda_header}")
        list(APPEND DSMVC_ENGINE_SOURCES src/cuda/cuda_fatbin_resource.cpp)
    else()
        add_custom_command(
            OUTPUT "${_dsmvc_cuda_header}"
            COMMAND "${CMAKE_COMMAND}"
                    "-DINPUT=${_dsmvc_cuda_fatbin}"
                    "-DOUTPUT=${_dsmvc_cuda_header}"
                    -DSYMBOL=dsmvc_cuda_fatbin
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS
                "${_dsmvc_cuda_fatbin}"
                cmake/embed_binary.cmake
            VERBATIM)
        set(_dsmvc_cuda_embed_outputs "${_dsmvc_cuda_header}")
    endif()
    add_custom_target(dsmvc_cuda_fatbin
        DEPENDS "${_dsmvc_cuda_fatbin}" ${_dsmvc_cuda_embed_outputs})
    list(APPEND DSMVC_ENGINE_SOURCES
        src/cuda/cuda_driver.cpp
        src/cuda/cuda_executor.cpp
        "${_dsmvc_cuda_header}")
endif()

add_library(dsmvc_engine STATIC ${DSMVC_ENGINE_SOURCES})
target_compile_features(dsmvc_engine PUBLIC cxx_std_23)
target_include_directories(dsmvc_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties(dsmvc_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DSMVC_ENABLE_CUDA)
    add_dependencies(dsmvc_engine dsmvc_cuda_fatbin)
    target_compile_definitions(dsmvc_engine PRIVATE
        DSMVC_HAS_CUDA=1
        DSMVC_CUDA_MIN_ARCHITECTURE=${DSMVC_CUDA_MIN_ARCHITECTURE})
    target_include_directories(dsmvc_engine PRIVATE
        "${DSMVC_CUDA_INCLUDE_DIR}"
        "${_dsmvc_cuda_generated_dir}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(dsmvc_engine PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

if(DSMVC_CAN_BUILD_AVX2)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_AVX2_OBJECT=1)
    if(MSVC)
        set_source_files_properties(src/cpu_executor_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2;/Ob3;/fp:fast")
    else()
        set_source_files_properties(src/cpu_executor_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
elseif(DSMVC_CAN_BUILD_NEON)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_NEON_OBJECT=1)
endif()

add_library(dsmvc MODULE src/vs_plugin.cpp)
target_compile_features(dsmvc PRIVATE cxx_std_23)
target_include_directories(dsmvc PRIVATE "${DSMVC_VAPOURSYNTH_INCLUDE_DIR}")
target_link_libraries(dsmvc PRIVATE dsmvc_engine)
set_target_properties(dsmvc PROPERTIES PREFIX "" OUTPUT_NAME "dsmvc")
if(APPLE)
    target_link_options(dsmvc PRIVATE
        "LINKER:-exported_symbol,_VapourSynthPluginInit2")
endif()
if(DSMVC_ENABLE_CUDA AND WIN32)
    target_sources(dsmvc PRIVATE "${_dsmvc_cuda_resource}")
    add_dependencies(dsmvc dsmvc_cuda_fatbin)
endif()
if(DSMVC_CAN_BUILD_NEON)
    target_compile_definitions(dsmvc PRIVATE DSMVC_HAS_NEON_OBJECT=1)
endif()

foreach(_target dsmvc_engine dsmvc)
    if(MSVC)
        target_compile_options(${_target} PRIVATE /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
        target_compile_definitions(${_target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()
endforeach()

install(TARGETS dsmvc LIBRARY DESTINATION vapoursynth RUNTIME DESTINATION vapoursynth)
install(FILES dsmvc.py DESTINATION python)
install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md DESTINATION share/dsmvc)

if(BUILD_TESTING)
    add_executable(dsmvc_engine_tests tests/engine_tests.cpp)
    target_compile_features(dsmvc_engine_tests PRIVATE cxx_std_23)
    target_link_libraries(dsmvc_engine_tests PRIVATE dsmvc_engine)
    if(DSMVC_ENABLE_CUDA AND WIN32)
        target_sources(dsmvc_engine_tests PRIVATE "${_dsmvc_cuda_resource}")
        add_dependencies(dsmvc_engine_tests dsmvc_cuda_fatbin)
    endif()
    add_test(NAME dsmvc_engine_tests COMMAND dsmvc_engine_tests)

    if(DSMVC_ENABLE_CUDA)
        add_test(NAME dsmvc_cuda_artifact_inventory
            COMMAND "${CMAKE_COMMAND}"
                "-DFATBIN=${_dsmvc_cuda_fatbin}"
                "-DCUOBJDUMP=${DSMVC_CUOBJDUMP_EXECUTABLE}"
                "-DNATIVE_ARCHITECTURES=${DSMVC_CUDA_ARCHITECTURES}"
                "-DPTX_ARCHITECTURES=${DSMVC_CUDA_PTX_ARCHITECTURES}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_cuda_fatbin.cmake")
    endif()

    if(NOT DSMVC_VS_PYTHON AND EXISTS "D:/okegui/OKEGui/tools/vapoursynth/python.exe")
        set(DSMVC_VS_PYTHON "D:/okegui/OKEGui/tools/vapoursynth/python.exe")
    endif()
    if(NOT DSMVC_BASELINE_PLUGIN AND EXISTS "D:/okegui/OKEGui/tools/vapoursynth/vapoursynth64/plugins/descale.dll")
        set(DSMVC_BASELINE_PLUGIN
            "D:/okegui/OKEGui/tools/vapoursynth/vapoursynth64/plugins/descale.dll")
    endif()
    if(DSMVC_VS_PYTHON AND DSMVC_BASELINE_PLUGIN)
        set(_dsmvc_vs_integration_args
            --plugin "$<TARGET_FILE:dsmvc>"
            --old-plugin "${DSMVC_BASELINE_PLUGIN}"
            --repo-root "${CMAKE_CURRENT_SOURCE_DIR}")
        if(DSMVC_BASELINE_WRAPPER)
            list(APPEND _dsmvc_vs_integration_args
                --old-wrapper "${DSMVC_BASELINE_WRAPPER}")
        endif()
        if(DSMVC_ENABLE_CUDA)
            list(APPEND _dsmvc_vs_integration_args --cuda-enabled)
        endif()
        add_test(
            NAME dsmvc_vs_integration
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_integration.py"
                ${_dsmvc_vs_integration_args}
        )
        set_tests_properties(dsmvc_vs_integration PROPERTIES TIMEOUT 60)
    endif()
    if(DSMVC_VS_PYTHON AND NOT DSMVC_ENABLE_METAL)
        add_test(
            NAME dsmvc_vs_api4_smoke
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_metal_integration.py"
                --plugin "$<TARGET_FILE:dsmvc>"
                --expect-metal disabled)
        set_tests_properties(dsmvc_vs_api4_smoke PROPERTIES TIMEOUT 60)
    endif()
    if(DSMVC_ENABLE_METAL AND DSMVC_VS_PYTHON)
        add_test(
            NAME dsmvc_vs_metal_integration
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_metal_integration.py"
                --plugin "$<TARGET_FILE:dsmvc>"
                --expect-metal enabled)
        set_tests_properties(dsmvc_vs_metal_integration PROPERTIES
            TIMEOUT 300
            RESOURCE_LOCK dsmvc_metal_device)
    endif()
endif()

if(DSMVC_BUILD_BENCHMARKS)
    add_executable(dsmvc_cpu_profile_benchmark
        benchmarks/cpu_profile_benchmark.cpp)
    target_compile_features(dsmvc_cpu_profile_benchmark PRIVATE cxx_std_23)
    target_link_libraries(dsmvc_cpu_profile_benchmark PRIVATE dsmvc_engine)
    if(MSVC)
        target_compile_options(dsmvc_cpu_profile_benchmark PRIVATE
            /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
        target_compile_definitions(dsmvc_cpu_profile_benchmark PRIVATE
            _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(dsmvc_cpu_profile_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()
endif()

if(DSMVC_ENABLE_METAL)
    if(NOT APPLE OR NOT DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
        message(FATAL_ERROR
            "DSMVC_ENABLE_METAL requires a native Apple ARM64 build")
    endif()

    find_program(DSMVC_XCRUN xcrun REQUIRED)
    execute_process(
        COMMAND "${DSMVC_XCRUN}" metal --version
        RESULT_VARIABLE DSMVC_METAL_COMPILER_RESULT
        OUTPUT_QUIET ERROR_QUIET)
    execute_process(
        COMMAND "${DSMVC_XCRUN}" metallib --version
        RESULT_VARIABLE DSMVC_METALLIB_RESULT
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT DSMVC_METAL_COMPILER_RESULT EQUAL 0
       OR NOT DSMVC_METALLIB_RESULT EQUAL 0)
        message(FATAL_ERROR
            "DSMVC_ENABLE_METAL requires the Xcode Metal toolchain")
    endif()

    enable_language(OBJCXX)
    find_library(DSMVC_METAL_FRAMEWORK Metal REQUIRED)
    find_library(DSMVC_FOUNDATION_FRAMEWORK Foundation REQUIRED)

    set(DSMVC_METAL_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/metal-routes")
    set(DSMVC_METAL_AIR
        "${DSMVC_METAL_GENERATED_DIR}/dsmvc_metal_routes.air")
    set(DSMVC_METALLIB
        "${DSMVC_METAL_GENERATED_DIR}/dsmvc_metal_routes.metallib")
    set(DSMVC_METALLIB_HEADER
        "${DSMVC_METAL_GENERATED_DIR}/dsmvc_metal_routes_metallib.hpp")

    add_custom_command(
        OUTPUT "${DSMVC_METAL_AIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${DSMVC_METAL_GENERATED_DIR}"
        COMMAND "${DSMVC_XCRUN}" -sdk macosx metal
                -Werror -fno-fast-math -ffp-contract=off
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/metal_routes.metal"
                -o "${DSMVC_METAL_AIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/metal_routes.metal"
        VERBATIM)
    add_custom_command(
        OUTPUT "${DSMVC_METALLIB}"
        COMMAND "${DSMVC_XCRUN}" -sdk macosx metallib
                "${DSMVC_METAL_AIR}" -o "${DSMVC_METALLIB}"
        DEPENDS "${DSMVC_METAL_AIR}"
        VERBATIM)
    add_custom_command(
        OUTPUT "${DSMVC_METALLIB_HEADER}"
        COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${DSMVC_METALLIB}"
                "-DOUTPUT=${DSMVC_METALLIB_HEADER}"
                -DSYMBOL=dsmvc_metal_routes_metallib
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
        DEPENDS "${DSMVC_METALLIB}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
        VERBATIM)

    target_sources(dsmvc PRIVATE
        src/metal_scheduler_apple.mm
        "${DSMVC_METALLIB_HEADER}")
    target_sources(dsmvc_engine PRIVATE src/metal_device_apple.mm)
    target_include_directories(dsmvc PRIVATE "${DSMVC_METAL_GENERATED_DIR}")
    target_compile_definitions(dsmvc PRIVATE DSMVC_HAS_METAL=1)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_METAL=1)
    set_source_files_properties(
        src/metal_scheduler_apple.mm src/metal_device_apple.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(dsmvc_engine PRIVATE
        "${DSMVC_METAL_FRAMEWORK}"
        "${DSMVC_FOUNDATION_FRAMEWORK}")
    target_link_libraries(dsmvc PRIVATE
        "${DSMVC_METAL_FRAMEWORK}"
        "${DSMVC_FOUNDATION_FRAMEWORK}")

    if(DSMVC_BUILD_BENCHMARKS)
        add_executable(dsmvc_metal_route_benchmark
            benchmarks/metal_route_benchmark.mm
            "${DSMVC_METALLIB_HEADER}")
        target_compile_features(dsmvc_metal_route_benchmark PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_route_benchmark PRIVATE
            "${DSMVC_METAL_GENERATED_DIR}")
        target_compile_options(dsmvc_metal_route_benchmark PRIVATE
            -fobjc-arc -Wall -Wextra -Wpedantic -ffp-contract=off)
        target_link_libraries(dsmvc_metal_route_benchmark PRIVATE
            dsmvc_engine
            "${DSMVC_METAL_FRAMEWORK}"
            "${DSMVC_FOUNDATION_FRAMEWORK}")

        add_executable(dsmvc_metal_yuv_benchmark
            benchmarks/metal_yuv_benchmark.mm
            "${DSMVC_METALLIB_HEADER}")
        target_compile_features(dsmvc_metal_yuv_benchmark PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_yuv_benchmark PRIVATE
            "${DSMVC_METAL_GENERATED_DIR}")
        target_compile_options(dsmvc_metal_yuv_benchmark PRIVATE
            -fobjc-arc -Wall -Wextra -Wpedantic -ffp-contract=off)
        target_link_libraries(dsmvc_metal_yuv_benchmark PRIVATE
            dsmvc_engine
            "${DSMVC_METAL_FRAMEWORK}"
            "${DSMVC_FOUNDATION_FRAMEWORK}")
    endif()

    if(BUILD_TESTING)
        add_test(NAME dsmvc_metal_artifact_inventory
            COMMAND "${CMAKE_COMMAND}"
                "-DMETALLIB=${DSMVC_METALLIB}"
                "-DEXPECTED_SYMBOLS=${CMAKE_CURRENT_SOURCE_DIR}/cmake/metal_entrypoints.txt"
                "-DXCRUN=${DSMVC_XCRUN}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_metal_metallib.cmake")
        set_tests_properties(
            dsmvc_metal_artifact_inventory PROPERTIES TIMEOUT 30)

        add_executable(dsmvc_hybrid_batch_coordinator_tests
            tests/hybrid_batch_coordinator_tests.cpp
            src/hybrid_batch_coordinator.cpp)
        target_compile_features(
            dsmvc_hybrid_batch_coordinator_tests PRIVATE cxx_std_23)
        target_compile_options(dsmvc_hybrid_batch_coordinator_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME dsmvc_hybrid_batch_coordinator_tests
            COMMAND dsmvc_hybrid_batch_coordinator_tests)

        add_executable(dsmvc_metal_yuv_staging_tests
            tests/metal_yuv_staging_tests.cpp)
        target_compile_features(dsmvc_metal_yuv_staging_tests PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_yuv_staging_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_options(dsmvc_metal_yuv_staging_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME dsmvc_metal_yuv_staging_tests
            COMMAND dsmvc_metal_yuv_staging_tests)

        add_executable(dsmvc_metal_scheduler_tests
            tests/metal_scheduler_tests.cpp
            src/metal_scheduler_apple.mm
            "${DSMVC_METALLIB_HEADER}")
        target_compile_features(dsmvc_metal_scheduler_tests PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_scheduler_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DSMVC_METAL_GENERATED_DIR}")
        target_compile_options(dsmvc_metal_scheduler_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
        target_link_libraries(dsmvc_metal_scheduler_tests PRIVATE
            dsmvc_engine
            "${DSMVC_METAL_FRAMEWORK}"
            "${DSMVC_FOUNDATION_FRAMEWORK}")
        add_test(NAME dsmvc_metal_scheduler_tests
            COMMAND dsmvc_metal_scheduler_tests)
        set_tests_properties(dsmvc_metal_scheduler_tests PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK dsmvc_metal_device)
    endif()
endif()
