# 03 · Build the scalar JIT

[Previous: language and reference](02-language-and-reference.md) · [Plan](../../README.md) · [Next: SIMD and baselines](04-simd-and-baselines.md)

**Outcome.** Execute the entire example on real buffers on Ubuntu and Windows x64 with one scalar loop per operation. Keep this path as the correctness baseline for later optimizations.

## Lower and execute

1. Use `IRBuilder` for loop bounds, loads, `fadd`, `fmul`, `llvm.sqrt.f32`, and stores. Emit verified LLVM IR with a target triple and data layout matching LLJIT. Use scalar LLVM math in the order specified by each node. Keep fast-math flags clear and contraction disabled. Keep loop and SLP vectorization disabled in the scalar baseline and inspect the result.
2. Start with a compact C-compatible internal ABI. Pass raw typed array addresses and an element count using natural-width integers fixed at the API boundary. Use a documented element alignment no greater than the buffers guarantee. Keep zero-length buffers out of dereference paths. Allocate intermediates in the runtime so each operation forms an independent kernel with a visible store.
3. Add the module using `ThreadSafeModule`, perform symbol lookup, and keep LLJIT and any resource tracker alive while the resulting function is in use. Match the function pointer signature and calling convention on both OSes. Treat IR verification, module add, linking, and lookup errors as reported failures. Preserve ownership if compilation or execution setup fails midway.
4. Test from a small C++ harness and expose the minimum Python binding needed to call one compiled expression on a contiguous NumPy array. Pass stable input pointers for the full call and return an owned array. Keep the full caching and public API work for Step 07.
5. Build an assembly or object emission route with an explicit cross target for offline inspection on the M3. Keep foreign-target objects out of the native LLJIT execution path. Include the triple, layout, CPU and features with any evidence.

## Exit evidence

- The same example passes C++ and Python comparisons against the reference on Ubuntu and Windows. Include an empty array, one element, a short odd length, and the proposed demo size.
- The scalar IR is readable and passes LLVM verification. A test demonstrates a clear unsupported-target or bad-input error.
- Record the emitted IR, Linux and Windows results, and any errors discovered at the ABI boundary. Compilation alone gives limited evidence here. [LLVM ORC guide](https://llvm.org/docs/ORCv2.html)
