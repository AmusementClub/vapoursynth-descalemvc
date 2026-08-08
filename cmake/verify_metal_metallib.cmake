if(NOT DEFINED METALLIB OR NOT EXISTS "${METALLIB}")
    message(FATAL_ERROR "METALLIB must name an existing Metal library")
endif()
if(NOT DEFINED EXPECTED_SYMBOLS OR NOT EXISTS "${EXPECTED_SYMBOLS}")
    message(FATAL_ERROR
        "EXPECTED_SYMBOLS must name the Metal entrypoint inventory")
endif()
if(NOT DEFINED XCRUN OR NOT EXISTS "${XCRUN}")
    message(FATAL_ERROR "XCRUN must name the xcrun executable")
endif()

file(STRINGS "${EXPECTED_SYMBOLS}" expected_symbols)
list(FILTER expected_symbols EXCLUDE REGEX "^[ \t]*(#.*)?$")
list(REMOVE_DUPLICATES expected_symbols)
list(SORT expected_symbols)
list(LENGTH expected_symbols expected_count)
if(NOT expected_count EQUAL 32)
    message(FATAL_ERROR
        "Metal inventory must contain 32 unique entrypoints, found ${expected_count}")
endif()

execute_process(
    COMMAND "${XCRUN}" --find metal-nm
    RESULT_VARIABLE find_result
    OUTPUT_VARIABLE metal_nm
    ERROR_VARIABLE find_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT find_result EQUAL 0 OR NOT EXISTS "${metal_nm}")
    message(FATAL_ERROR "metal-nm lookup failed: ${find_error}")
endif()

execute_process(
    COMMAND "${metal_nm}" "${METALLIB}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error)
if(NOT nm_result EQUAL 0)
    message(FATAL_ERROR "metal-nm failed: ${nm_error}")
endif()

string(REPLACE "\r\n" "\n" nm_output "${nm_output}")
string(REPLACE "\n" ";" nm_lines "${nm_output}")
set(actual_symbols)
foreach(line IN LISTS nm_lines)
    string(STRIP "${line}" line)
    if(line MATCHES
       "^[0-9A-Fa-f]+[ \t]+[Tt][ \t]+([A-Za-z_][A-Za-z0-9_]*)$")
        list(APPEND actual_symbols "${CMAKE_MATCH_1}")
    endif()
endforeach()
list(REMOVE_DUPLICATES actual_symbols)
list(SORT actual_symbols)
list(LENGTH actual_symbols actual_count)

if(NOT "${actual_symbols}" STREQUAL "${expected_symbols}")
    string(JOIN ", " expected_text ${expected_symbols})
    string(JOIN ", " actual_text ${actual_symbols})
    message(FATAL_ERROR
        "Metal entrypoint inventory differs\n"
        "expected (${expected_count}): ${expected_text}\n"
        "actual (${actual_count}): ${actual_text}")
endif()

message(STATUS
    "Metal entrypoint inventory matches (${actual_count}/${expected_count})")
