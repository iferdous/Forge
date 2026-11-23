cmake_minimum_required(VERSION 3.25)

project(Forge LANGUAGES C CXX)

option(FORGE_ENABLE_JIT "Build with the pinned LLVM SDK" ON)
option(FORGE_BUILD_PYTHON "Build the Python frontend" ON)
option(FORGE_BUILD_BENCHMARKS "Build the native benchmark harness" OFF)

if(FORGE_ENABLE_JIT)
    if(WIN32)
        # LLVM's finder expects zstd_static.lib. The vcpkg config exports the same
        # static CMake target from zstd.lib, so load it before the LLVM SDK.
        find_package(zstd CONFIG QUIET)
    endif()

    find_package(LLVM 23.1.6 EXACT REQUIRED CONFIG)
    message(STATUS "Forge LLVM SDK: ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
    message(STATUS "LLVM host triple: ${LLVM_HOST_TRIPLE}")
endif()

function(forge_target_defaults target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(forge_graph STATIC src/graph.cpp src/kernel_plan.cpp)
target_include_directories(forge_graph PUBLIC src)
set_target_properties(forge_graph PROPERTIES POSITION_INDEPENDENT_CODE ON)
forge_target_defaults(forge_graph)

if(FORGE_ENABLE_JIT)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET zstd::libzstd_shared)
        # The SDK exports static zstd, but Ubuntu's archive cannot link into a
        # Python extension. Use the shared library's same public zstd ABI.
        get_target_property(llvm_support_dependencies LLVMSupport INTERFACE_LINK_LIBRARIES)
        string(REPLACE "zstd::libzstd_static" "zstd::libzstd_shared"
            llvm_support_dependencies "${llvm_support_dependencies}")
        set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES "${llvm_support_dependencies}")
    endif()
    llvm_map_components_to_libnames(llvm_libraries Core OrcJIT Passes native X86)
    separate_arguments(llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
    add_library(forge_llvm INTERFACE)
    target_include_directories(forge_llvm SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})
    target_compile_options(forge_llvm INTERFACE ${llvm_definitions})
    target_link_libraries(forge_llvm INTERFACE ${llvm_libraries})
    if(NOT LLVM_ENABLE_RTTI)
        # LLVM consumers must match the SDK's RTTI setting, including under UBSan.
        if(MSVC)
            target_compile_options(forge_llvm INTERFACE /GR-)
        else()
            target_compile_options(forge_llvm INTERFACE -fno-rtti)
        endif()
    endif()

    add_library(forge_jit STATIC src/jit.cpp src/llvm_lowering.cpp)
    set_target_properties(forge_jit PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(forge_jit PUBLIC forge_graph PRIVATE forge_llvm)
    forge_target_defaults(forge_jit)
endif()

if(FORGE_BUILD_BENCHMARKS)
    if(NOT FORGE_ENABLE_JIT)
        message(FATAL_ERROR "Native benchmarks require FORGE_ENABLE_JIT=ON")
    endif()
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(forge_benchmark_version 1.10.0)
    FetchContent_Declare(
        benchmark
        URL https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v${forge_benchmark_version}
        URL_HASH SHA256=acc5c7c0fbed855c4aaa2d7c3c36fb82b0c15c28f4e151e3f679c2fc0aa404af
    )
    FetchContent_MakeAvailable(benchmark)
    add_executable(forge_native_bench bench/native.cpp)
    add_executable(forge_native_suite bench/native_suite.cpp)
    foreach(target forge_native_bench forge_native_suite)
        target_link_libraries(${target} PRIVATE forge_jit benchmark::benchmark)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_BINARY_DIR}/bench>")
        forge_target_defaults(${target})
    endforeach()
    if(MSVC)
        target_compile_options(forge_native_suite PRIVATE /fp:strict)
    else()
        target_compile_options(forge_native_suite PRIVATE -ffp-contract=off)
    endif()
endif()

if(FORGE_BUILD_PYTHON)
    include(FetchContent)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    set(forge_pybind11_version 3.1.0)
    FetchContent_Declare(
        pybind11
        URL https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v${forge_pybind11_version}
        URL_HASH SHA256=ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665
    )
    FetchContent_MakeAvailable(pybind11)

    set(forge_python_dir "${CMAKE_CURRENT_BINARY_DIR}/python")
    foreach(source __init__.py __main__.py _cache.py _language.py _reference.py)
        configure_file("python/forge/${source}" "${forge_python_dir}/forge/${source}" COPYONLY)
    endforeach()
    pybind11_add_module(_forge_ir MODULE NO_EXTRAS python/bindings.cpp)
    set_target_properties(_forge_ir PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${forge_python_dir}/forge>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${forge_python_dir}/forge>"
    )
    forge_target_defaults(_forge_ir)
    if(FORGE_ENABLE_JIT)
        target_sources(_forge_ir PRIVATE python/jit_bindings.cpp)
        target_compile_definitions(_forge_ir PRIVATE FORGE_ENABLE_JIT)
        target_link_libraries(_forge_ir PRIVATE forge_jit)
    else()
        target_link_libraries(_forge_ir PRIVATE forge_graph)
    endif()
endif()

if(FORGE_BUILD_BENCHMARKS)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bench/toolchain-$<CONFIG>.json" CONTENT
        "{\"cmake\": \"${CMAKE_VERSION}\", \"compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\", \"configuration\": \"$<CONFIG>\", \"llvm\": \"${LLVM_PACKAGE_VERSION}\", \"pybind11\": \"${forge_pybind11_version}\", \"google_benchmark\": \"${forge_benchmark_version}\"}\n")
endif()

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)

    # Preserve the parent project's MSVC runtime selection in GoogleTest.
    set(gtest_force_shared_crt ON)
    set(INSTALL_GTEST OFF)
    FetchContent_Declare(
        googletest
        URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    )
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

    add_executable(forge_graph_tests tests/graph_test.cpp tests/kernel_plan_test.cpp)
    target_link_libraries(forge_graph_tests PRIVATE forge_graph GTest::gtest_main)
    forge_target_defaults(forge_graph_tests)
    gtest_discover_tests(forge_graph_tests)

    if(FORGE_BUILD_PYTHON)
        add_test(NAME forge_python_reference
            COMMAND "${Python_EXECUTABLE}" -m unittest discover
                -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_reference.py -v)
        add_test(NAME forge_python_properties
            COMMAND "${Python_EXECUTABLE}" -m unittest discover
                -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_properties.py -v)
        set_tests_properties(forge_python_reference forge_python_properties PROPERTIES
            ENVIRONMENT "PYTHONPATH=${forge_python_dir};PYTHONDONTWRITEBYTECODE=1")
    endif()

    if(FORGE_ENABLE_JIT)
        add_executable(forge_jit_smoke tests/jit_smoke_test.cpp)
        target_link_libraries(forge_jit_smoke PRIVATE GTest::gtest_main forge_llvm)
        forge_target_defaults(forge_jit_smoke)
        gtest_discover_tests(forge_jit_smoke)

        add_executable(forge_jit_tests tests/jit_test.cpp tests/simd_test.cpp)
        target_link_libraries(forge_jit_tests PRIVATE GTest::gtest_main forge_jit)
        forge_target_defaults(forge_jit_tests)
        gtest_discover_tests(forge_jit_tests)

        if(FORGE_BUILD_PYTHON)
            add_test(NAME forge_python_jit
                COMMAND "${Python_EXECUTABLE}" -m unittest discover
                    -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_jit.py -v)
            set_tests_properties(forge_python_jit PROPERTIES
                ENVIRONMENT "PYTHONPATH=${forge_python_dir};PYTHONDONTWRITEBYTECODE=1")
        endif()
    endif()
endif()
