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
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
    set(DSMVC_CAN_BUILD_AVX2 ON)
    list(APPEND DSMVC_ENGINE_SOURCES src/cpu_executor_avx2.cpp)
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
    add_custom_target(dsmvc_cuda_fatbin
        DEPENDS "${_dsmvc_cuda_fatbin}" "${_dsmvc_cuda_header}")
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
endif()

add_library(dsmvc MODULE src/vs_plugin.cpp)
target_compile_features(dsmvc PRIVATE cxx_std_23)
target_include_directories(dsmvc PRIVATE "${DSMVC_VAPOURSYNTH_INCLUDE_DIR}")
target_link_libraries(dsmvc PRIVATE dsmvc_engine)
set_target_properties(dsmvc PROPERTIES PREFIX "" OUTPUT_NAME "dsmvc")

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
endif()
