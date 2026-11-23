# 03 · Build the scalar JIT

[Previous: language and reference](02-language-and-reference.md) · [Plan](../../README.md) · [Next: SIMD and baselines](04-simd-and-baselines.md)

**Outcome.** Execute the entire example on real buffers on Ubuntu and Windows x64 with one scalar loop per operation. Keep this path as the correctness baseline for later optimizations.

## Lower and execute

1. Use `IRBuilder` for loop bounds, loads, `fadd`, `fmul`, `llvm.sqrt.f32`, and stores. Emit verified LLVM IR with a target triple and data layout matching LLJIT. Use scalar LLVM math in the order specified by each node. Keep fast-math flags clear and contraction disabled. Keep loop and SLP vectorization disabled in the scalar baseline and inspect the result.
2. Start with a compact C-compatible internal ABI. Pass raw typed array addresses and an element count using natural-width integers fixed at the API boundary. Use a documented element alignment no greater than the buffers guarantee. Keep zero-length buffers out of dereference paths. Allocate intermediates in the runtime so each operation forms an independent kernel with a visible store.
3. Add the module using `ThreadSafeModule`, perform symbol lookup, and keep LLJIT and any resource tracker alive while the resulting function is in use. Match the function pointer signature and calling convention on both OSes. Treat IR verification, module add, linking, and lookup errors as reported failures. Preserve ownership if compilation or execution setup fails midway.
4. Test from a small C++ harness and expose the minimum Python binding needed to call one compiled expression on a contiguous NumPy array. Pass stable input pointers for the full call and return an owned array. Keep the full caching and public API work for Step 07.
5. Build an assembly or object emission route with an explicit cross target for offline inspection on the M3. Keep foreign-target objects out of the native LLJIT execution path. Include the triple, layout, CPU and features with any evidence.

## Run and inspect the scalar kernel

Build with the pinned LLVM SDK and run `ctest --preset release` on x64. For the current SSK Mac, the LLVM-enabled Python module was built under `.local/build/macos-llvm/python`. The Homebrew LLVM 23.1.6 bottle and its relocated Z3 dependency are under `.local/toolchains/macos-bottle/`. Match this bottle's macOS deployment target with `-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0` when configuring the Mac build.

Open a Python session with `PYTHONPATH="$PWD/.local/build/macos-llvm/python" .local/venv/bin/python -i` on the Mac, or set `PYTHONPATH` to `.local/build/release/python` in the x64 build.

~~~python
import forge
import numpy as np

x = np.array([1, 10, 25, 46], dtype=np.float32)
program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
kernel = program.jit(x, target="scalar", fusion=False)
print(kernel(x).tolist())  # [3.0, 6.0, 9.0, 12.0]
print(kernel.info.ir)
print(program.emit_x86(x, fusion=False).text)
~~~

The entry point receives the element count and separate input and output pointer tables. Each tensor operation writes its result through a separate loop. Literal broadcasts are scalar parameters, and intermediate arrays occupy a reusable workspace. Outputs own their storage by default. `run_into` accepts a writable, correctly shaped output disjoint from the inputs. The LLVM module is verified before and after optimization, and the kernel keeps the LLJIT owner alive for every call.

On Linux, Forge links LLVM's Zstandard dependency through the distribution's shared library. The SDK's static Zstandard archive linked into executables, but its relocations could not link into the Python extension. Windows continues to use its compatible static SDK dependencies.

## Exit evidence

- The same example passes C++ and Python comparisons against the reference on Ubuntu and Windows. Include an empty array, one element, a short odd length, and the proposed demo size.
- The scalar IR is readable and passes LLVM verification. A test demonstrates a clear unsupported-target or bad-input error.
- Record the emitted IR, Linux and Windows results, and any errors discovered at the ABI boundary. Compilation alone gives limited evidence here. [LLVM ORC guide](https://llvm.org/docs/ORCv2.html)

The [Stage 03 run](https://github.com/iferdous/Forge/actions/runs/45100241317) passed the C++ and Python scalar checks on Ubuntu 24.04 and Windows Server 2025 x64 at `5496dfd`. The M3 also executed the scalar kernel with LLVM 23.1.6 and emitted Linux x86-64 assembly.
