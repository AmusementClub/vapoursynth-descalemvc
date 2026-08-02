cmake_minimum_required(VERSION 3.24)

project(dsmvc VERSION 0.1.0 LANGUAGES CXX)

if(MSVC)
    # API3 exposes a C ABI only. Static CRT linkage prevents the host's bundled,
    # potentially older MSVC runtime from changing STL synchronization ABI.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include(CTest)
include(FetchContent)

set(DSMVC_VAPOURSYNTH_SDK "" CACHE PATH "Path to the VapourSynth SDK directory")
set(DSMVC_GETNATIVE_SOURCE_DIR "" CACHE PATH "Path to a pinned GetNative-VF checkout")
set(DSMVC_VS_PYTHON "" CACHE FILEPATH "Python executable from the target VapourSynth environment")
set(DSMVC_BASELINE_PLUGIN "" CACHE FILEPATH "Baseline descale plugin used by integration tests")

if(DSMVC_GETNATIVE_SOURCE_DIR)
    set(DSMVC_GETNATIVE_ROOT "${DSMVC_GETNATIVE_SOURCE_DIR}")
else()
    FetchContent_Declare(
        dsmvc_getnative
        GIT_REPOSITORY https://github.com/MysteryDove/GetNative-VF.git
        GIT_TAG d64af6caa6c18d670eda1f25ee69fb47f5313b69
        GIT_SHALLOW FALSE
    )
    FetchContent_GetProperties(dsmvc_getnative)
    if(NOT dsmvc_getnative_POPULATED)
        FetchContent_Populate(dsmvc_getnative)
    endif()
    set(DSMVC_GETNATIVE_ROOT "${dsmvc_getnative_SOURCE_DIR}")
endif()

if(NOT EXISTS "${DSMVC_GETNATIVE_ROOT}/engine/include/getnative/axis_plan.hpp")
    message(FATAL_ERROR "DSMVC_GETNATIVE_SOURCE_DIR is not a GetNative-VF checkout")
endif()

if(DSMVC_VAPOURSYNTH_SDK)
    set(_dsmvc_vs_hints "${DSMVC_VAPOURSYNTH_SDK}/include")
endif()
find_path(DSMVC_VAPOURSYNTH_INCLUDE_DIR VapourSynth.h
    HINTS
        ${_dsmvc_vs_hints}
        "$ENV{VAPOURSYNTH_SDK}/include"
        "D:/okegui/OKEGui/tools/vapoursynth/sdk/include"
)
if(NOT DSMVC_VAPOURSYNTH_INCLUDE_DIR)
    message(FATAL_ERROR "VapourSynth API3 headers were not found; set DSMVC_VAPOURSYNTH_SDK")
endif()

set(DSMVC_ENGINE_SOURCES
    src/axis_plan.cpp
    src/backend.cpp
    src/cpu_executor.cpp
    "${DSMVC_GETNATIVE_ROOT}/engine/src/planner/filter.cpp"
)

set(DSMVC_CAN_BUILD_AVX2 OFF)
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
    set(DSMVC_CAN_BUILD_AVX2 ON)
    list(APPEND DSMVC_ENGINE_SOURCES src/cpu_executor_avx2.cpp)
endif()

add_library(dsmvc_engine STATIC ${DSMVC_ENGINE_SOURCES})
target_compile_features(dsmvc_engine PUBLIC cxx_std_23)
target_include_directories(dsmvc_engine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${DSMVC_GETNATIVE_ROOT}/engine/include"
    PRIVATE
        "${DSMVC_GETNATIVE_ROOT}/engine/src/planner"
)
set_target_properties(dsmvc_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
install(FILES python/dsmvc.py DESTINATION python)
install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md DESTINATION share/dsmvc)

if(BUILD_TESTING)
    add_executable(dsmvc_engine_tests tests/engine_tests.cpp)
    target_compile_features(dsmvc_engine_tests PRIVATE cxx_std_23)
    target_link_libraries(dsmvc_engine_tests PRIVATE dsmvc_engine)
    add_test(NAME dsmvc_engine_tests COMMAND dsmvc_engine_tests)

    if(NOT DSMVC_VS_PYTHON AND EXISTS "D:/okegui/OKEGui/tools/vapoursynth/python.exe")
        set(DSMVC_VS_PYTHON "D:/okegui/OKEGui/tools/vapoursynth/python.exe")
    endif()
    if(NOT DSMVC_BASELINE_PLUGIN AND EXISTS "D:/okegui/OKEGui/tools/vapoursynth/vapoursynth64/plugins/descale.dll")
        set(DSMVC_BASELINE_PLUGIN
            "D:/okegui/OKEGui/tools/vapoursynth/vapoursynth64/plugins/descale.dll")
    endif()
    if(DSMVC_VS_PYTHON AND DSMVC_BASELINE_PLUGIN)
        add_test(
            NAME dsmvc_vs_integration
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_integration.py"
                --plugin "$<TARGET_FILE:dsmvc>"
                --old-plugin "${DSMVC_BASELINE_PLUGIN}"
                --repo-root "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        set_tests_properties(dsmvc_vs_integration PROPERTIES TIMEOUT 60)
    endif()
endif()
