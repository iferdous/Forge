cmake_minimum_required(VERSION 3.25)

if(FORGE_LLVM_PLATFORM STREQUAL "linux-x64")
    set(asset "LLVM-23.1.6-Linux-X64.tar.zst")
    set(sha256 "59e14234f145927f982b50627e8bafaf2fca50151b7d973d6a0b40796ddcfe9a")
elseif(FORGE_LLVM_PLATFORM STREQUAL "windows-x64")
    set(asset "clang+llvm-23.1.6-x86_64-pc-windows-msvc.tar.zst")
    set(sha256 "7623f582089983c1cb51030cf1996514b19464e5669099571b399e8543770155")
else()
    message(FATAL_ERROR "Set FORGE_LLVM_PLATFORM to linux-x64 or windows-x64.")
endif()

set(local "${CMAKE_CURRENT_LIST_DIR}/../.local")
set(prefix "${local}/toolchains/llvm-23.1.6-${FORGE_LLVM_PLATFORM}")
set(archive "${local}/cache/${asset}")
set(marker "${prefix}/llvm-dir.txt")
set(output "${local}/llvm-dir-${FORGE_LLVM_PLATFORM}.txt")
file(MAKE_DIRECTORY "${local}")

# Publish the marker only after the complete archive has been verified and extracted.
if(EXISTS "${marker}")
    file(STRINGS "${marker}" cached_config LIMIT_COUNT 1)
    if(EXISTS "${prefix}/${cached_config}/LLVMConfig.cmake")
        get_filename_component(llvm_dir "${prefix}/${cached_config}" ABSOLUTE)
        file(WRITE "${output}" "${llvm_dir}\n")
        message(STATUS "Using cached LLVM SDK at ${llvm_dir}")
        return()
    endif()
endif()

set(actual "")
file(MAKE_DIRECTORY "${local}/cache")
if(EXISTS "${archive}")
    file(SHA256 "${archive}" actual)
endif()

if(NOT actual STREQUAL sha256)
    find_program(CURL_EXECUTABLE NAMES curl REQUIRED)
    execute_process(
        COMMAND "${CURL_EXECUTABLE}" --fail --silent --show-error --location
            --retry 2 --retry-all-errors --retry-max-time 360 --max-time 240
            --output "${archive}.partial"
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.6/${asset}"
        COMMAND_ERROR_IS_FATAL ANY
    )
    file(SHA256 "${archive}.partial" actual)
    if(NOT actual STREQUAL sha256)
        file(REMOVE "${archive}.partial")
        message(FATAL_ERROR "SHA-256 mismatch for ${asset}: expected ${sha256}, got ${actual}")
    endif()
    file(RENAME "${archive}.partial" "${archive}")
endif()

message(STATUS "Verified SHA-256 ${sha256} for ${asset}")
set(staging "${prefix}.extracting")
file(REMOVE_RECURSE "${staging}")
file(MAKE_DIRECTORY "${staging}")
file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${staging}")
file(GLOB_RECURSE configs RELATIVE "${staging}" "${staging}/LLVMConfig.cmake")
list(LENGTH configs config_count)
if(NOT config_count EQUAL 1)
    message(FATAL_ERROR "Expected one LLVMConfig.cmake in ${asset}, found ${config_count}: ${configs}")
endif()

cmake_path(GET configs PARENT_PATH relative_config)
file(WRITE "${staging}/llvm-dir.txt" "${relative_config}\n")
file(REMOVE_RECURSE "${prefix}")
file(RENAME "${staging}" "${prefix}")
get_filename_component(llvm_dir "${prefix}/${relative_config}" ABSOLUTE)
file(WRITE "${output}" "${llvm_dir}\n")
message(STATUS "Installed LLVM SDK at ${llvm_dir}")
