include_guard(GLOBAL)

function(dsmvc_add_numerical_contract_test target)
    set(options)
    set(one_value_args BACKEND)
    set(multi_value_args SOURCES LIBRARIES DEFINITIONS INCLUDE_DIRECTORIES)
    cmake_parse_arguments(
        DSMVC_NUMERICAL "${options}" "${one_value_args}"
        "${multi_value_args}" ${ARGN})
    if(NOT DSMVC_NUMERICAL_SOURCES)
        message(FATAL_ERROR
            "${target}: dsmvc_add_numerical_contract_test requires SOURCES")
    endif()
    if(NOT DSMVC_NUMERICAL_BACKEND)
        set(DSMVC_NUMERICAL_BACKEND shared)
    endif()

    add_executable(${target} ${DSMVC_NUMERICAL_SOURCES})
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/tests"
        ${DSMVC_NUMERICAL_INCLUDE_DIRECTORIES})
    target_link_libraries(${target} PRIVATE
        dsmvc_engine ${DSMVC_NUMERICAL_LIBRARIES})
    if(DSMVC_NUMERICAL_DEFINITIONS)
        target_compile_definitions(${target} PRIVATE
            ${DSMVC_NUMERICAL_DEFINITIONS})
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    endif()

    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
        LABELS "numerical-contract;backend-${DSMVC_NUMERICAL_BACKEND}")
endfunction()
