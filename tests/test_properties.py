"""Seeded differential tests and bounded parser and shape-validator fuzzing."""

import random
import unittest

import numpy as np

import forge
from forge import _forge_ir

SEED = 0xF047E
POINTWISE_SHAPES = ((0,), (1,), (7,), (8,), (9,), (17,), (33,), (0, 5), (3, 7), (3, 11))
BINARY = {"+": np.add, "*": np.multiply}


def pointwise_tree(rng, inputs, depth):
    """Generate source and evaluate the same ordered operations independently."""
    if depth == 0 or rng.randrange(4) == 0:
        name = rng.choice(tuple(inputs))
        return name, inputs[name]
    left, lhs = pointwise_tree(rng, inputs, depth - 1)
    op = rng.choice(("+", "*", "sqrt"))
    if op == "sqrt":
        return f"sqrt({left})", np.sqrt(lhs, dtype=np.float32)
    if rng.randrange(3) == 0:
        right = rng.choice(("-0.0", "0.0", "-2", "0.125", "3"))
        rhs = np.float32(right)
    elif rng.randrange(4) == 0:
        right, rhs = left, lhs
    else:
        right, rhs = pointwise_tree(rng, inputs, depth - 1)
    return f"({left} {op} {right})", BINARY[op](lhs, rhs, dtype=np.float32)


def ordered_sum(values):
    result = np.float32(0)
    for value in values.ravel(order="C"):
        result = np.float32(result + value)
    return result


def ordered_matmul(a, b):
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        prod = np.multiply(a[:, k, None], b[k, None, :], dtype=np.float32)
        np.add(result, prod, out=result)
    return result


class SeededProperties(unittest.TestCase):
    def check_bits(self, actual, expected):
        self.assertEqual((actual.shape, actual.dtype), (expected.shape, np.dtype(np.float32)))
        nan = np.isnan(expected)
        np.testing.assert_array_equal(np.isnan(actual), nan)
        np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])

    def check_program(self, source, names, arguments, expected, case):
        with self.subTest(seed=SEED, case=case, source=source,
                          shapes=tuple(array.shape for array in arguments)):
            program = forge.compile(source, inputs=names)
            self.check_bits(program.reference(*arguments), expected)
            if hasattr(_forge_ir, "CompiledKernel"):
                for target in ("scalar", "avx2") if forge.has_avx2() else ("scalar",):
                    for fusion in (False, True):
                        with self.subTest(target=target, fusion=fusion):
                            kernel = program.jit(*arguments, target=target, fusion=fusion)
                            out = np.empty_like(expected)
                            kernel.run_into(out, *arguments)
                            self.check_bits(out, expected)

    def test_pointwise_trees_against_ordered_numpy_operations(self):
        rng = random.Random(SEED)
        data = np.random.default_rng(SEED)
        special = np.array([0x80000000, 0x00000001, 0x7F800000, 0xFF800000, 0x7FC00000],
                           dtype=np.uint32).view(np.float32)
        for case in range(48):
            shape = POINTWISE_SHAPES[case % len(POINTWISE_SHAPES)]
            x, y = (data.standard_normal(shape, dtype=np.float32) for _ in range(2))
            if case % 3 == 0:
                count = min(x.size, special.size)
                x.flat[:count] = special[:count]
            with np.errstate(all="ignore"):
                source, expected = pointwise_tree(rng, {"x": x, "y": y}, rng.randint(1, 5))
            self.check_program(source, ("x", "y"), (x, y), expected, case)

    def test_matmul_reduction_and_broadcast_compositions(self):
        rng = random.Random(SEED)
        data = np.random.default_rng(SEED)
        edge_shapes = ((0, 3, 5), (4, 0, 5), (4, 3, 0), (17, 33, 35))
        for case in range(20):
            m, k, n = edge_shapes[case] if case < len(edge_shapes) else tuple(
                rng.choice((1, 7, 8, 9, 16, 17, 32, 33)) for _ in range(3)
            )
            a = data.standard_normal((m, k), dtype=np.float32)
            b = data.standard_normal((k, n), dtype=np.float32)
            product = ordered_matmul(a, b)
            if case % 2:
                src = "sum((a @ b) * 2)"
                expected = np.asarray(ordered_sum(product * np.float32(2)))
            else:
                src = "sum(a @ b) + a"
                expected = np.add(ordered_sum(product), a, dtype=np.float32)
            self.check_program(src, ("a", "b"), (a, b), expected, case)

    def test_rejects_bounded_malformed_and_excessive_expressions(self):
        rng = random.Random(SEED)
        for case in range(96):
            literal = rng.randint(0, 1 << 48)
            bad = rng.choice((f"f(x + {literal})", f"x[{literal}]", f"x ^ {literal}",
                              f"sqrt(x, {literal})", f"sqrt(x={literal})", f"x + ) {literal}",
                              f"x + non_ascii_{chr(0x100 + case)}", f"x\x00 + {literal}"))
            with self.subTest(seed=SEED, case=case, source=bad), self.assertRaises(SyntaxError):
                forge.compile(bad, inputs=("x",))
        for source in ("x" + " " * 4096, "sqrt(" * 33 + "x" + ")" * 33):
            with self.assertRaises(SyntaxError):
                forge.compile(source, inputs=("x",))
        balanced = "x"
        for _ in range(8):
            balanced = f"({balanced}+{balanced})"
        with self.assertRaises(SyntaxError):
            forge.compile(balanced, inputs=("x",))

    def test_rejects_seeded_invalid_shapes_storage_and_types(self):
        rng = random.Random(SEED)
        pointwise = forge.compile("x + y", inputs=("x", "y"))
        matrix = forge.compile("x @ y", inputs=("x", "y"))
        for case in range(32):
            m, k, n = (rng.randrange(2, 12) for _ in range(3))
            x = np.zeros((m, k), dtype=np.float32)
            bad_layout = np.empty((m, 2 * k), dtype=np.float32)[:, ::2]
            with self.subTest(seed=SEED, case=case, shape=x.shape):
                for invalid in (bad_layout, x.ravel(), np.empty((m, k + 1), dtype=np.float32)):
                    with self.assertRaises(ValueError):
                        pointwise.graph(x, invalid)
                with self.assertRaises(ValueError):
                    matrix.graph(x, np.empty((k + 1, n), dtype=np.float32))
                with self.assertRaises(TypeError):
                    pointwise.graph(x, x.astype(np.float64))


if __name__ == "__main__":
    unittest.main()
