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
| Linux x64 | `LLVM-23.1.6-Linux-X64.tar.xz` | `bd4d1c434ff69ef06d19e7934c04708da677e5f2eb51f5bcf052ab91818f0094` |
| Windows x64 | `clang+llvm-23.1.6-x86_64-pc-windows-msvc.tar.xz` | `16af05919cf36958dfb4ddaa4fc4d9d4ff577f5daeb76bc112ea49cd8ced82b2` |
| Source | `llvm-project-23.1.6.src.tar.xz` | `d3b86ab29cfb4445a6652e5c2642ccc75e04e38084dde3701d323320f35a31da` |

## Reproduce the smoke test

Use a native x86-64 machine with CMake 3.25 or newer, curl, and a C++20 toolchain. Use a CMake release that supports the installed Visual Studio version. On Windows, use the Visual Studio C++ desktop workload with the DIA SDK and a bootstrapped vcpkg checkout. Set `VCPKG_ROOT` to that checkout. The Windows SDK uses a static MSVC runtime, so use the `x64-windows-static` triplet for the dependencies in `vcpkg.json`. The SDK script verifies the compressed download before extraction and records the path to `LLVMConfig.cmake`. The build and test presets keep output in `.local/build/`.

Ubuntu shell

```sh
mkdir -p .local/tmp .local/cache
export TMPDIR="$PWD/.local/tmp" XDG_CACHE_HOME="$PWD/.local/cache"
cmake -DFORGE_LLVM_PLATFORM=linux-x64 -P cmake/ProvisionLLVM.cmake
cmake --preset release "-DLLVM_DIR=$(cat .local/llvm-dir-linux-x64.txt)"
cmake --build --preset release --parallel
ctest --preset release --verbose
```

Windows PowerShell

```powershell
New-Item -ItemType Directory -Force .local/tmp, .local/cache | Out-Null
$env:TEMP = (Resolve-Path .local/tmp).Path
$env:TMP = $env:TEMP
cmake -DFORGE_LLVM_PLATFORM=windows-x64 -P cmake/ProvisionLLVM.cmake
$llvmDir = Get-Content .local/llvm-dir-windows-x64.txt
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
cmake --preset release "-DLLVM_DIR=$llvmDir" -A x64 `
  "-DMSVC_DIA_SDK_DIR=$vs/DIA SDK" `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build --preset release --parallel
ctest --preset release --verbose
```

The hosted build uses Release to match the prebuilt SDK. The Debug presets require an SDK compatible with that platform's debug configuration. The Stage 01 build consists of LLVM 23.1.6 and GoogleTest 1.17.0. Python and the benchmark dependencies enter with the first stages that exercise them.

## SDK compatibility findings

- CMake's archive reader rejected the release's Zstandard frames with `Frame requires too much memory for decoding` on both hosted runners. The SHA-256-verified XZ archives unpack on both platforms and preserve the LLVM 23.1.6 pin.
- The Windows archive ships LLVM's libraries and headers. Its link targets also require the static Zlib, Zstandard, and LibXml2 libraries and Visual Studio's DIA SDK. The pinned vcpkg manifest provides the three libraries. CMake loads Zstandard's own package config first because LLVM's finder expects the name `zstd_static.lib` while vcpkg names its static library `zstd.lib`.

## Verified result

The [Stage 01 CI run](https://github.com/iferdous/Forge/actions/runs/45096400510) passed on commit `0515e71`. Both runners built against LLVM 23.1.6, verified the module, linked it in-process with LLJIT, and returned `42.5` for `40.5` and `-3.0` for `-5.0` from the generated `float(float)` function.

| Runner | CMake | C++ compiler | JIT target | LLVM host CPU | AVX2 available | Smoke test |
| --- | --- | --- | --- | --- | --- | --- |
| Ubuntu 24.04 x64 | 3.31.6 | GCC 13.3.0 | `x86_64-unknown-linux-gnu` | `znver3` | Yes | Passed |
| Windows Server 2025 x64 | 4.5.0 | MSVC 19.52.37029.0 | `x86_64-pc-windows-msvc` | `znver5` | Yes | Passed |

The job logs contain the LLVM data layouts and full test output. The smoke test runs a scalar kernel. CPU feature detection confirms an AVX2 execution route for Step 04, where the generated vector code will be exercised. A stable performance host is a Step 07 deliverable.
