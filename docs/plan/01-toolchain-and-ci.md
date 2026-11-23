# 01 · Qualify the toolchain and CI

[Plan](../../README.md) · [Next: language and reference](02-language-and-reference.md)

**Outcome.** A clean checkout can compile a C++20 program, link the LLVM libraries, JIT a trivial function on Ubuntu and Windows x64, and run a GoogleTest assertion on the result.

## Build

1. Keep the workstation checkout at `/Volumes/SSK/Forge`. Put downloaded SDKs, toolchains, Python environments, caches, and working artifacts in `.local/`. Use `.local/toolchains/`, `.local/cache/`, `.local/tmp/`, and `.local/venv/`. Set `TMPDIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, and the CMake dependency/build directories to paths inside the checkout for project commands. If used, point `CCACHE_DIR` there too. Keep the checked-in tree portable with relative paths and configurable presets.
2. Introduce a small CMake project with `src/`, `tests/`, `python/`, and `bench/` as their first targets appear. Use an explicit `LLVM_DIR` or package prefix. Make configure fail with an actionable diagnostic if the installed LLVM version differs. Record CMake, generator, compiler, Python minor, NumPy, pybind11, GoogleTest and Google Benchmark versions in the build instructions or lock file once tested. Use the same Python minor in both CI jobs.
3. Try the pinned binary SDK first on each x64 OS. Inspect architecture, `LLVMConfig.cmake`, headers, available targets, and library files. Check the SHA-256 before unpacking. Verify CMake can actually link core, native codegen and ORC against the selected compiler and C++ runtime. On Windows use the developer archive and a compatible MSVC runtime. A `clang` executable on PATH alone establishes neither SDK completeness nor ABI compatibility.
4. If an SDK fails the link check, capture the configure/link error. Build the same LLVM tag from source with the minimal backend set. Use `X86` on x86 and `AArch64;X86` if building a native Mac SDK for cross-emission. Record the install prefix and flags, including any Windows runtime settings, the CMake build type, and the backend list. Cache verified installs using OS, architecture, compiler, version, and build flags in the key. A full source build is a fallback because of disk and CI time.
5. Add CI jobs with explicit x64 labels, starting with `ubuntu-24.04` and `windows-2025`. Log host architecture and OS-enabled AVX/AVX2 support before running SIMD tests. A scalar job can still pass without AVX2. Show skipped SIMD tests as skips, and qualify at least one AVX2 execution host before Step 04. Use Debug and Release presets as appropriate. Run core tests in both OS jobs. Add coverage on a supported Linux configuration during release hardening.
6. Include a minimal LLJIT test such as a generated `float(float)` add function. Verify the module, add it to LLJIT, look up its callable address, and run it. Capture the triple and data layout; a compile-only check can miss object-linker and executable-memory failures. On the M3, start with frontend work and assembly inspection. Make native ARM execution an optional later capability.

## LLVM release inputs

Use the upstream release metadata to double-check these compressed assets and their SHA-256 digests. [LLVM 23.1.6 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.6)

| Platform | Asset | SHA-256 |
| --- | --- | --- |
| Linux x64 | `LLVM-23.1.6-Linux-X64.tar.zst` | `59e14234f145927f982b50627e8bafaf2fca50151b7d973d6a0b40796ddcfe9a` |
| Windows x64 | `clang+llvm-23.1.6-x86_64-pc-windows-msvc.tar.zst` | `7623f582089983c1cb51030cf1996514b19464e5669099571b399e8543770155` |
| Source | `llvm-project-23.1.6.src.tar.xz` | `d3b86ab29cfb4445a6652e5c2642ccc75e04e38084dde3701d323320f35a31da` |

## Exit evidence

- Two clean configure, link, and test logs with installed versions, CPU features, and OS.
- A passing JIT function smoke test and a route to native x86 AVX2 execution. The host that will produce publishable timing can be qualified in Step 07.
- A recorded toolchain decision if prebuilt packages or the Windows linker fail, with the exact error and the tested fallback.
