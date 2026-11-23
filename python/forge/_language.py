"""Parse bounded array expressions and bind them to the native tensor graph."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import keyword
import math
import re
import struct

import numpy as np

from . import _forge_ir
from ._forge_ir import Graph, GraphBuilder, Op, SourceLocation
from ._reference import evaluate

_MAX_SOURCE_LENGTH = 4096
_MAX_DEPTH = 32
_MAX_NODES = 256
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
            case ast.BinOp(left=left, right=right, op=(ast.Add() | ast.Mult()) as operator):
                operands = self.visit(left, depth + 1), self.visit(right, depth + 1)
                op = Op.ADD if isinstance(operator, ast.Add) else Op.MULTIPLY
                return self.emit(node, op, operands)
            case ast.Call(func=ast.Name(id="sqrt"), args=[operand], keywords=[]):
                return self.emit(node, Op.SQRT, (self.visit(operand, depth + 1),))
            case _:
                raise self.error(node, "Expected an input, literal, +, *, or sqrt(expression)")


@dataclass(frozen=True)
class Program:
    """A parsed expression that can be bound to concrete input shapes."""

    inputs: tuple[str, ...]
    _instructions: tuple[_Instruction, ...]
    _output: int

    def graph(self, *arrays: np.ndarray) -> Graph:
        """Bind validated arrays to the native, shape-specialized tensor graph."""
        if len(arrays) != len(self.inputs):
            raise TypeError(f"Expected {len(self.inputs)} input arrays, got {len(arrays)}")
        for name, array in zip(self.inputs, arrays, strict=True):
            if type(array) is not np.ndarray or array.dtype != np.dtype(np.float32):
                raise TypeError(f"{name!r} must be a native-endian float32 NumPy array")
            if array.ndim not in (1, 2):
                raise ValueError(f"{name!r} must have rank one or two")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name!r} must be C-contiguous")

        builder = GraphBuilder()
        node_ids = []
        for instruction in self._instructions:
            op = instruction.op
            location = SourceLocation(*instruction.location)
            operands = [node_ids[index] for index in instruction.operands]
            if op == Op.INPUT:
                index = instruction.payload
                node_id = builder.input(index, arrays[index].shape, location)
            elif op == Op.LITERAL:
                node_id = builder.literal(instruction.payload, location)
            elif op == Op.SQRT:
                node_id = builder.sqrt(operands[0], location)
            else:
                node_id = builder.binary(op, *operands, location)
            node_ids.append(node_id)
        return builder.finish(node_ids[self._output])

    def reference(self, *arrays: np.ndarray) -> np.ndarray:
        """Evaluate the typed graph using rounded float32 NumPy operations."""
        return evaluate(self.graph(*arrays), arrays)

    def jit(self, *example_arrays: np.ndarray):
        """Compile a reusable scalar kernel for the example input shapes."""
        if not hasattr(_forge_ir, "CompiledKernel"):
            raise RuntimeError("LLVM JIT support requires a build with FORGE_ENABLE_JIT=ON")
        return _forge_ir.CompiledKernel(self.graph(*example_arrays), len(self.inputs))

    def emit_x86(self, *example_arrays: np.ndarray):
        """Inspect Linux x86-64 assembly without executing it."""
        if not hasattr(_forge_ir, "emit_x86_linux"):
            raise RuntimeError("LLVM assembly emission requires a build with FORGE_ENABLE_JIT=ON")
        return _forge_ir.emit_x86_linux(self.graph(*example_arrays), len(self.inputs))


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
        if not _IDENTIFIER.fullmatch(name) or keyword.iskeyword(name) or name == "sqrt":
            raise ValueError(f"Invalid or reserved input name {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("Input names must be unique")

    parser = _Parser(source, names)
    syntax = ast.parse(source, filename="<forge>", mode="eval")
    output = parser.visit(syntax.body)
    return Program(names, tuple(parser.instructions), output)
