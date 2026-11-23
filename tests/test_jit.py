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


if __name__ == "__main__":
    unittest.main()
