"""Parse bounded array expressions and bind them to the native tensor graph."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import keyword
import math
import re
import struct

import numpy as np

from . import _forge_ir
from ._cache import get_or_compile
from ._forge_ir import Graph, GraphBuilder, Op, SourceLocation
from ._reference import evaluate

_MAX_SOURCE_LENGTH = 4096
_MAX_DEPTH = 32
_MAX_NODES = 256
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BINARY = {ast.Add: Op.ADD, ast.Mult: Op.MULTIPLY, ast.MatMult: Op.MATMUL}
_FUNCTIONS = {"sqrt": Op.SQRT, "sum": Op.SUM, "max": Op.MAX, "mean": Op.MEAN}


def _round_integer(value: int) -> int:
    """Round the magnitude to the 24-bit binary32 significand before conversion."""
    magnitude = abs(value)
    shift = magnitude.bit_length() - 24
    if shift <= 0:
        return value

    unit = 1 << shift
    significand, remainder = divmod(magnitude, unit)
    half = unit // 2
    if remainder > half or (remainder == half and significand & 1):
        significand += 1
    return (-significand if value < 0 else significand) * unit


@dataclass(frozen=True)
class _Instruction:
    op: Op
    location: tuple[int, int]
    operands: tuple[int, ...] = ()
    payload: int | float | None = None


class _Parser:
    def __init__(self, source: str, inputs: tuple[str, ...]):
        self.lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.inputs = {name: index for index, name in enumerate(inputs)}
        self.instructions: list[_Instruction] = []
        self.visited = 0
        self.interned: dict[tuple, int] = {}

    def error(self, node: ast.expr, message: str) -> SyntaxError:
        return SyntaxError(message, ("<forge>", node.lineno, node.col_offset + 1,
                                     self.lines[node.lineno - 1]))

    def emit(
        self, node: ast.expr, op: Op, operands: tuple[int, ...] = (),
        payload: int | float | None = None,
    ) -> int:
        # Preserve the sign of zero when sharing identical literal nodes.
        key_payload = struct.pack("!f", payload) if op == Op.LITERAL else payload
        key = op, operands, key_payload
        if key not in self.interned:
            self.interned[key] = len(self.instructions)
            self.instructions.append(
                _Instruction(op, (node.lineno, node.col_offset + 1), operands, payload)
            )
        return self.interned[key]

    def literal(self, node: ast.expr, value: object, sign: int = 1) -> int:
        if type(value) not in (int, float):
            raise self.error(node, "Expected a real numeric literal")
        signed = sign * value
        if type(signed) is int:
            signed = _round_integer(signed)
        try:
            with np.errstate(over="ignore"):
                normalized = float(np.float32(signed))
        except OverflowError:
            raise self.error(node, "Literal must be finite as float32") from None
        if not math.isfinite(normalized):
            raise self.error(node, "Literal must be finite as float32")
        return self.emit(node, Op.LITERAL, payload=normalized)

    def visit(self, node: ast.expr, depth: int = 1) -> int:
        self.visited += 1
        if depth > _MAX_DEPTH or self.visited > _MAX_NODES:
            raise self.error(node, "Expression exceeds the depth or node limit")

        match node:
            case ast.Name(id=name):
                if name not in self.inputs:
                    raise self.error(node, f"Unknown input {name!r}")
                return self.emit(node, Op.INPUT, payload=self.inputs[name])
            case ast.Constant(value=value):
                return self.literal(node, value)
            case ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=value)):
                return self.literal(node, value)
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=value)):
                return self.literal(node, value, sign=-1)
            case ast.BinOp(left=left, right=right, op=operator) if type(operator) in _BINARY:
                operands = self.visit(left, depth + 1), self.visit(right, depth + 1)
                return self.emit(node, _BINARY[type(operator)], operands)
            case ast.Call(func=ast.Name(id=name), args=[operand], keywords=[]) if name in _FUNCTIONS:
                return self.emit(node, _FUNCTIONS[name], (self.visit(operand, depth + 1),))
            case _:
                raise self.error(node, "Expected an input, literal, +, *, @, sqrt, sum, max, or mean")


@dataclass(frozen=True)
class Program:
    """A parsed expression that can be bound to concrete input shapes."""

    inputs: tuple[str, ...]
    _instructions: tuple[_Instruction, ...]
    _output: int
    _cache_graph: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Source locations and input names affect diagnostics, while operand
        # order, input indices and the exact literal bits affect code generation.
        object.__setattr__(self, "_cache_graph", tuple(
            (ins.op, ins.operands, struct.pack("!f", ins.payload) if ins.op == Op.LITERAL else ins.payload)
            for ins in self._instructions
        ))

    def _shapes(self, arrays):
        if len(arrays) != len(self.inputs):
            raise TypeError(f"Expected {len(self.inputs)} input arrays, got {len(arrays)}")
        for name, array in zip(self.inputs, arrays, strict=True):
            if type(array) is not np.ndarray or array.dtype != np.dtype(np.float32):
                raise TypeError(f"{name!r} must be a native-endian float32 NumPy array")
            if array.ndim not in (1, 2):
                raise ValueError(f"{name!r} must have rank one or two")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name!r} must be C-contiguous")
        return tuple(array.shape for array in arrays)

    def _bind(self, shapes) -> Graph:
        builder = GraphBuilder()
        node_ids = []
        for instruction in self._instructions:
            op = instruction.op
            location = SourceLocation(*instruction.location)
            operands = [node_ids[index] for index in instruction.operands]
            if op == Op.INPUT:
                index = instruction.payload
                node_id = builder.input(index, shapes[index], location)
            elif op == Op.LITERAL:
                node_id = builder.literal(instruction.payload, location)
            elif op == Op.SQRT:
                node_id = builder.sqrt(operands[0], location)
            elif op == Op.MATMUL:
                node_id = builder.matmul(*operands, location)
            elif op in (Op.SUM, Op.MAX, Op.MEAN):
                node_id = builder.reduce(op, operands[0], location)
            else:
                node_id = builder.binary(op, *operands, location)
            node_ids.append(node_id)
        return builder.finish(node_ids[self._output])

    def graph(self, *arrays: np.ndarray) -> Graph:
        """Bind validated arrays to the native, shape-specialized tensor graph."""
        return self._bind(self._shapes(arrays))

    def reference(self, *arrays: np.ndarray) -> np.ndarray:
        """Evaluate the typed graph using rounded float32 NumPy operations."""
        return evaluate(self.graph(*arrays), arrays)

    def plan(self, *example_arrays: np.ndarray, fusion: bool = True):
        """Describe the scheduled loops and intermediate arrays without invoking LLVM."""
        return _forge_ir.plan_kernel(self.graph(*example_arrays), len(self.inputs), fusion)

    def jit(self, *example_arrays: np.ndarray, target: str = "scalar", fusion: bool = True,
            cache: bool = False):
        """Compile a reusable native kernel, optionally sharing cached machine code."""
        if not hasattr(_forge_ir, "CompiledKernel"):
            raise RuntimeError("LLVM JIT support requires a build with FORGE_ENABLE_JIT=ON")
        shapes = self._shapes(example_arrays)
        if not cache:
            return _forge_ir.CompiledKernel(self._bind(shapes), len(self.inputs), target, fusion)
        if not isinstance(target, str) or type(fusion) is not bool:
            raise TypeError("A cached kernel requires a string target and boolean fusion mode")
        key = (self._cache_graph, self._output, np.dtype(np.float32).str, shapes, target, fusion)
        code = get_or_compile(key, lambda: _forge_ir._compile_code(
            self._bind(shapes), len(self.inputs), target, fusion
        ))
        return _forge_ir.CompiledKernel(code)

    def emit_x86(self, *example_arrays: np.ndarray, target: str = "scalar", fusion: bool = True):
        """Inspect Linux x86-64 assembly without executing it."""
        if not hasattr(_forge_ir, "emit_x86_linux"):
            raise RuntimeError("LLVM assembly emission requires a build with FORGE_ENABLE_JIT=ON")
        return _forge_ir.emit_x86_linux(self.graph(*example_arrays), len(self.inputs), target, fusion)


def compile(source: str, *, inputs: tuple[str, ...] | list[str]) -> Program:
    """Parse a Forge array expression. Input names specify positional argument order."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if len(source) > _MAX_SOURCE_LENGTH or not source.isascii():
        raise SyntaxError("Expected an ASCII expression of at most 4096 characters")
    if not isinstance(inputs, (tuple, list)):
        raise TypeError("inputs must be a tuple or list of names")
    if not 1 <= len(inputs) <= _MAX_NODES:
        raise ValueError("Expected between 1 and 256 input names")
    names = tuple(inputs)
    for name in names:
        if not isinstance(name, str):
            raise TypeError("Each input name must be a string")
        if not _IDENTIFIER.fullmatch(name) or keyword.iskeyword(name) or name in _FUNCTIONS:
            raise ValueError(f"Invalid or reserved input name {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("Input names must be unique")

    parser = _Parser(source, names)
    syntax = ast.parse(source, filename="<forge>", mode="eval")
    output = parser.visit(syntax.body)
    return Program(names, tuple(parser.instructions), output)
