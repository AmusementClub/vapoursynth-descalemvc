if(NOT DEFINED FATBIN OR NOT EXISTS "${FATBIN}")
    message(FATAL_ERROR "CUDA fatbin was not found: ${FATBIN}")
endif()
if(NOT DEFINED CUOBJDUMP OR NOT EXISTS "${CUOBJDUMP}")
    message(FATAL_ERROR "cuobjdump was not found: ${CUOBJDUMP}")
endif()

execute_process(
    COMMAND "${CUOBJDUMP}" --list-elf "${FATBIN}"
    RESULT_VARIABLE elf_result
    OUTPUT_VARIABLE elf_output
    ERROR_VARIABLE elf_error)
if(NOT elf_result EQUAL 0)
    message(FATAL_ERROR "cuobjdump --list-elf failed: ${elf_error}")
endif()
string(REGEX MATCHALL "\\.sm_[0-9]+\\.cubin" elf_entries "${elf_output}")
set(actual_native_architectures)
foreach(elf_entry IN LISTS elf_entries)
    string(REGEX REPLACE "^\\.sm_([0-9]+)\\.cubin$" "\\1"
        architecture "${elf_entry}")
    list(APPEND actual_native_architectures "${architecture}")
endforeach()
list(REMOVE_DUPLICATES actual_native_architectures)
set(expected_native_architectures ${NATIVE_ARCHITECTURES})
list(SORT actual_native_architectures COMPARE NATURAL)
list(SORT expected_native_architectures COMPARE NATURAL)
if(NOT actual_native_architectures STREQUAL expected_native_architectures)
    message(FATAL_ERROR
        "CUDA native inventory mismatch: expected [${expected_native_architectures}], found [${actual_native_architectures}]")
endif()

execute_process(
    COMMAND "${CUOBJDUMP}" --list-ptx "${FATBIN}"
    RESULT_VARIABLE ptx_result
    OUTPUT_VARIABLE ptx_output
    ERROR_VARIABLE ptx_error)
if(NOT ptx_result EQUAL 0)
    message(FATAL_ERROR "cuobjdump --list-ptx failed: ${ptx_error}")
endif()
string(REGEX MATCHALL "\\.sm_[0-9]+\\.ptx" ptx_entries "${ptx_output}")
set(actual_ptx_architectures)
foreach(ptx_entry IN LISTS ptx_entries)
    string(REGEX REPLACE "^\\.sm_([0-9]+)\\.ptx$" "\\1"
        architecture "${ptx_entry}")
    list(APPEND actual_ptx_architectures "${architecture}")
endforeach()
list(REMOVE_DUPLICATES actual_ptx_architectures)
set(expected_ptx_architectures ${PTX_ARCHITECTURES})
list(SORT actual_ptx_architectures COMPARE NATURAL)
list(SORT expected_ptx_architectures COMPARE NATURAL)
if(NOT actual_ptx_architectures STREQUAL expected_ptx_architectures)
    message(FATAL_ERROR
        "CUDA PTX inventory mismatch: expected [${expected_ptx_architectures}], found [${actual_ptx_architectures}]")
endif()

execute_process(
    COMMAND "${CUOBJDUMP}" --dump-resource-usage "${FATBIN}"
    RESULT_VARIABLE resource_result
    OUTPUT_VARIABLE resource_output
    ERROR_VARIABLE resource_error)
if(NOT resource_result EQUAL 0)
    message(FATAL_ERROR
        "cuobjdump --dump-resource-usage failed: ${resource_error}")
endif()
if(resource_output MATCHES "STACK:[1-9][0-9]*"
   OR resource_output MATCHES "LOCAL:[1-9][0-9]*")
    message(FATAL_ERROR "CUDA fatbin contains a kernel with stack or local memory")
endif()
