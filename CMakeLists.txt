cmake_minimum_required(VERSION 3.25)

project(Forge LANGUAGES CXX)

option(FORGE_ENABLE_JIT "Build with the pinned LLVM SDK" ON)

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

add_library(forge_graph STATIC src/graph.cpp)
target_include_directories(forge_graph PUBLIC src)
set_target_properties(forge_graph PROPERTIES POSITION_INDEPENDENT_CODE ON)
forge_target_defaults(forge_graph)

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

    add_executable(forge_graph_tests tests/graph_test.cpp)
    target_link_libraries(forge_graph_tests PRIVATE forge_graph GTest::gtest_main)
    forge_target_defaults(forge_graph_tests)
    gtest_discover_tests(forge_graph_tests)

    if(FORGE_ENABLE_JIT)
        llvm_map_components_to_libnames(llvm_libraries Core OrcJIT native)
        add_executable(forge_jit_smoke tests/jit_smoke_test.cpp)
        target_include_directories(forge_jit_smoke SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
        target_link_libraries(forge_jit_smoke PRIVATE GTest::gtest_main ${llvm_libraries})
        separate_arguments(llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
        target_compile_options(forge_jit_smoke PRIVATE ${llvm_definitions})
        forge_target_defaults(forge_jit_smoke)
        gtest_discover_tests(forge_jit_smoke)
    endif()
endif()
