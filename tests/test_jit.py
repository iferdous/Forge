import unittest

import numpy as np

import forge
from forge import _forge_ir


@unittest.skipUnless(hasattr(_forge_ir, "CompiledKernel"), "LLVM JIT is disabled")
class ScalarJitTests(unittest.TestCase):
    def test_demo_matches_reference_and_exposes_ir(self):
        program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
        for shape in ((0,), (2, 0), (1,), (9,), (5, 200)):
            with self.subTest(shape=shape):
                x = np.resize(np.array([1, 10, 25, 46], dtype=np.float32), shape)
                kernel = program.jit(x)
                result = kernel(x)
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(result.flags.owndata)
                np.testing.assert_array_equal(
                    result.view(np.uint32), program.reference(x).view(np.uint32)
                )
                self.assertIn("llvm.sqrt.f32", kernel.info.ir)
                self.assertTrue(kernel.object)
        cross = program.emit_x86(np.ones(9, dtype=np.float32))
        self.assertEqual(cross.info.triple, "x86_64-unknown-linux-gnu")
        self.assertIn("sqrtss", cross.text)

    def test_identity_returns_own_storage_and_buffers_are_checked(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        kernel = forge.compile("x", inputs=("x",)).jit(x)
        result = kernel(x)
        np.testing.assert_array_equal(result, x)
        self.assertTrue(result.flags.owndata)
        self.assertFalse(np.shares_memory(x, result))
        with self.assertRaises(ValueError):
            kernel(x.reshape(3, 2))
        with self.assertRaises(TypeError):
            kernel(x.astype(np.float64))
        with self.assertRaises(ValueError):
            kernel.run_into(x, x)
        result.setflags(write=False)
        with self.assertRaises(ValueError):
            kernel.run_into(result, x)

    def test_scalar_splat_rounded_operations_and_aliasing_read_inputs(self):
        x = np.array([0x80000000, 0x00000001, 0xBF800000, 0x7F800000, 0x7FC00000],
                     dtype=np.uint32).view(np.float32)
        program = forge.compile("sqrt(a * b + sqrt(4.0))", inputs=("a", "b"))
        actual = program.jit(x, x)(x, x)
        expected = program.reference(x, x)
        nan = np.isnan(expected)
        np.testing.assert_array_equal(np.isnan(actual), nan)
        np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])


@unittest.skipUnless(hasattr(_forge_ir, "CompiledKernel"), "LLVM JIT is disabled")
class Avx2JitTests(unittest.TestCase):
    def test_cross_assembly_and_invalid_targets(self):
        x = np.ones(9, dtype=np.float32)
        program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
        emitted = program.emit_x86(x, target="avx2")
        self.assertIn("+avx2", emitted.info.features)
        self.assertIn("<8 x float>", emitted.info.ir)
        self.assertTrue(emitted.info.remarks)
        for mnemonic in ("vaddps", "vmulps", "vsqrtps", "vsqrtss"):
            self.assertIn(mnemonic, emitted.text)
        with self.assertRaisesRegex(ValueError, "scalar.*avx2"):
            program.jit(x, target="typo")
        if not forge.has_avx2():
            with self.assertRaisesRegex(ValueError, "usable AVX2"):
                program.jit(x, target="avx2")

    def test_native_avx2_bits_match_scalar_with_misalignment_and_special_values(self):
        if not forge.has_avx2():
            self.skipTest("Native AVX2 is unavailable")

        program = forge.compile("sqrt(a * b + c)", inputs=("a", "b", "c"))
        special_bits = np.array([0x80000000, 0x00000001, 0x3F800001, 0x7F800000,
                                 0x7FC00000, 0xBF800000, 0x7F7FFFFF], dtype=np.uint32)
        for size in (7, 8, 9, 15, 16, 17, 1023, 1024, 1025):
            with self.subTest(size=size):
                storage = np.empty(size * 4 + 1, dtype=np.uint8)
                left = np.ndarray(size, dtype=np.float32, buffer=storage, offset=1)
                left.view(np.uint32)[:] = np.resize(special_bits, size)
                self.assertNotEqual(left.ctypes.data % 32, 0)
                right = np.resize(np.array([0.0, 0.5, 1, 2], dtype=np.float32), size)
                bias = np.resize(np.array([-0.0, 0.0, 1.0], dtype=np.float32), size)
                scalar = program.jit(left, right, bias)(left, right, bias)
                avx2 = program.jit(left, right, bias, target="avx2")
                output = np.empty_like(left)
                avx2.run_into(output, left, right, bias)
                reference = program.reference(left, right, bias)
                nan = np.isnan(reference)
                np.testing.assert_array_equal(np.isnan(scalar), nan)
                np.testing.assert_array_equal(np.isnan(output), nan)
                np.testing.assert_array_equal(output.view(np.uint32)[~nan],
                                              scalar.view(np.uint32)[~nan])
                np.testing.assert_array_equal(output.view(np.uint32)[~nan],
                                              reference.view(np.uint32)[~nan])


if __name__ == "__main__":
    unittest.main()
