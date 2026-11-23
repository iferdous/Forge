import sys
import unittest

import numpy as np

import forge


class ReferenceTests(unittest.TestCase):
    def assertFloat32BitsEqual(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        nan = np.isnan(expected)
        np.testing.assert_array_equal(np.isnan(actual), nan)
        np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])

    def test_demo_and_stable_native_graph(self):
        program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
        x = np.array([1, 10, 25, 46], dtype=np.float32)
        self.assertFloat32BitsEqual(
            program.reference(x), np.array([3, 6, 9, 12], dtype=np.float32)
        )
        expected = (
            "%0 = input(0) : f32[4] @1:7 uses=1\n"
            "%1 = literal(0x40000000) : f32 @1:11 uses=1\n"
            "%2 = splat(%1) : f32[4] @1:11 uses=1\n"
            "%3 = add(%0, %2) : f32[4] @1:7 uses=1\n"
            "%4 = literal(0x40400000) : f32 @1:16 uses=1\n"
            "%5 = splat(%4) : f32[4] @1:16 uses=1\n"
            "%6 = multiply(%3, %5) : f32[4] @1:6 uses=1\n"
            "%7 = sqrt(%6) : f32[4] @1:1 uses=0\n"
            "return %7\n"
        )
        self.assertEqual(program.graph(x).dump(), expected)
        self.assertEqual(program.graph(x.copy()).dump(), expected)

    def test_binding_preserves_input_order_and_bitwise_literal_identity(self):
        x = np.array([2], dtype=np.float32)
        y = np.array([3], dtype=np.float32)
        program = forge.compile("y + x * 2", inputs=("x", "y"))
        self.assertEqual(program.graph(x, y).nodes[0].input_index, 1)
        self.assertFloat32BitsEqual(program.reference(x, y), np.array([7], dtype=np.float32))
        zeros = forge.compile("(x * -0.0) + (x * 0.0)", inputs=("x",)).graph(x).dump()
        self.assertIn("literal(0x80000000)", zeros)
        self.assertIn("literal(0x00000000)", zeros)

    def test_identical_subexpressions_share_a_node(self):
        program = forge.compile("(x + 2) * (x + 2)", inputs=("x",))
        x = np.array([0, 3], dtype=np.float32)
        graph = program.graph(x)
        output = graph.nodes[graph.output]
        left, right = output.operands
        self.assertEqual(left, right)
        self.assertEqual(graph.nodes[left].use_count, 2)
        self.assertFloat32BitsEqual(
            program.reference(x), np.array([4, 25], dtype=np.float32)
        )

    def test_empty_singleton_and_owned_output(self):
        for shape in ((0,), (0, 2), (2, 0), (1,), (1, 1)):
            with self.subTest(shape=shape):
                x = np.ones(shape, dtype=np.float32)
                x.setflags(write=False)
                identity = forge.compile("x", inputs=("x",)).reference(x)
                expression = forge.compile("sqrt(x * 4)", inputs=("x",)).reference(x)
                self.assertFloat32BitsEqual(identity, x)
                self.assertFloat32BitsEqual(expression, x * np.float32(2))
                for result in (identity, expression):
                    self.assertIsNot(result, x)
                    self.assertTrue(result.flags.owndata)
                    self.assertTrue(result.flags.c_contiguous)

    def test_exact_float32_steps_and_exceptional_values(self):
        a = np.array([0x3F800001], dtype=np.uint32).view(np.float32)
        b = np.array([0x3F7FFFFE], dtype=np.uint32).view(np.float32)
        c = np.array([-1], dtype=np.float32)
        rounded = forge.compile("a * b + c", inputs=("a", "b", "c")).reference(a, b, c)
        self.assertEqual(rounded.view(np.uint32).item(), 0)

        x = np.array([0x80000000, 0x00000001, 0xBF800000, 0x7F800000, 0x7FC00000],
                     dtype=np.uint32).view(np.float32)
        with np.errstate(all="raise"):
            root = forge.compile("sqrt(x)", inputs=("x",)).reference(x)
        expected = np.array([0x80000000, 0x1A3504F3, 0x7FC00000, 0x7F800000, 0x7FC00000],
                            dtype=np.uint32).view(np.float32)
        self.assertFloat32BitsEqual(root, expected)

        tiny = x[1:2]
        down = forge.compile("x * 0.5", inputs=("x",)).reference(tiny)
        up = forge.compile("x + x", inputs=("x",)).reference(tiny)
        self.assertEqual(down.view(np.uint32).item(), 0)
        self.assertEqual(up.view(np.uint32).item(), 2)
        largest = np.array([0x7F7FFFFF], dtype=np.uint32).view(np.float32)
        overflow = forge.compile("x * 2", inputs=("x",)).reference(largest)
        self.assertEqual(overflow.view(np.uint32).item(), 0x7F800000)

    def test_integer_literals_round_directly_to_float32(self):
        halfway = (1 << 60) + (1 << 36)
        x = np.array([0], dtype=np.float32)
        for value, expected in ((halfway, 0x5D800000), (halfway + 1, 0x5D800001),
                                (-(halfway + 1), 0xDD800001)):
            with self.subTest(value=value):
                result = forge.compile(f"x + {value}", inputs=("x",)).reference(x)
                self.assertEqual(result.view(np.uint32).item(), expected)

    def test_seeded_inputs_match_stepwise_numpy(self):
        random = np.random.default_rng(20261122)
        program = forge.compile("sqrt(a * a + 2.5 * b * b)", inputs=("a", "b"))
        for shape in ((17,), (3, 5), (1000,)):
            for _ in range(6):
                a = random.standard_normal(shape, dtype=np.float32)
                b = random.standard_normal(shape, dtype=np.float32)
                expected = np.sqrt(a * a + (np.float32(2.5) * b) * b)
                self.assertFloat32BitsEqual(program.reference(a, b), expected)

    def test_rejects_unsupported_syntax_and_reports_location(self):
        for source in ("x.attr", "x[0]", "x ** 2", "x - 2", "-x", "f(x)",
                       "sqrt(x, x)", "sqrt(x=x)", "sqrt(*x)", "sqrt()", "True",
                       "'word'", "1j", "1e400 + x", "4e38 + x", "x = 2",
                       "(lambda: x)()", "unknown + x", "x + K"):
            with self.subTest(source=source), self.assertRaises(SyntaxError):
                forge.compile(source, inputs=("x",))
        with self.assertRaises(SyntaxError) as raised:
            forge.compile("(x +\n missing)", inputs=("x",))
        self.assertEqual((raised.exception.lineno, raised.exception.offset), (2, 2))

    def test_parser_bounds(self):
        balanced = "x"
        for _ in range(8):
            balanced = f"({balanced}+{balanced})"
        too_deep = "sqrt(" * 33 + "x" + ")" * 33
        for source in ("x" + " " * 4096, balanced, too_deep):
            with self.subTest(length=len(source)), self.assertRaises(SyntaxError):
                forge.compile(source, inputs=("x",))

    def test_rejects_input_type_rank_layout_and_broadcast(self):
        x = np.ones((2, 3), dtype=np.float32)
        program = forge.compile("x + y", inputs=("x", "y"))
        foreign_f32 = ">f4" if sys.byteorder == "little" else "<f4"
        for bad in ([1, 2], np.float32(1), np.ones((2, 3)), x.astype(foreign_f32)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                program.reference(x, bad)
        for bad in (np.array(1, dtype=np.float32), x.reshape(1, 2, 3), x[:, ::2],
                    np.asfortranarray(x), np.ones((1, 3), dtype=np.float32)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                program.reference(x, bad)
        with self.assertRaises(TypeError):
            program.reference(x)
        with self.assertRaises(ValueError):
            forge.compile("2 * 3", inputs=("x",)).reference(x)
        for names in ((), ("x", "x"), ("sqrt",), ("class",), ("x y",), ("K",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                forge.compile("x", inputs=names)


if __name__ == "__main__":
    unittest.main()
