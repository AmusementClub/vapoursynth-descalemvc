# Integrator inventory for the backend-local Vulkan Float64 lane.
# Top-level registration remains integrator-owned.

set(DSMVC_VULKAN_F64_SHADER_VARIANTS
    "transpose_f64|src/vulkan/shaders/transpose_f64.comp|vulkan_transpose_f64_spv"
    "rhs_f64|src/vulkan/shaders/rhs_f64.comp|vulkan_rhs_f64_spv"
    "inverse_f64|src/vulkan/shaders/inverse_f64.comp|vulkan_inverse_f64_spv|-DDSMVC_SOLVE_ONLY=0"
    "solve_f64|src/vulkan/shaders/inverse_f64.comp|vulkan_solve_f64_spv|-DDSMVC_SOLVE_ONLY=1"
    "convert_f64|src/vulkan/shaders/convert_f64.comp|vulkan_convert_f64_spv")

set(DSMVC_VULKAN_F64_BACKEND_TESTS
    src/vulkan/vulkan_f64_policy_tests.cpp
    src/vulkan/vulkan_f64_runtime_tests.cpp)

set(DSMVC_VULKAN_F64_BACKEND_BENCHMARKS
    src/vulkan/vulkan_f64_benchmark.cpp
    src/vulkan/vulkan_f64_benchmark_pair.py
    src/vulkan/vulkan_f64_plugin_benchmark.py
    src/vulkan/vulkan_f64_plugin_benchmark.vpy)

set(DSMVC_VULKAN_F64_ARTIFACT_SCRIPT
    src/vulkan/vulkan_f64_artifacts.sh)
