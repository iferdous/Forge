"""Evaluate a native tensor graph with a float32 result at each operation."""

import numpy as np

from ._forge_ir import Graph, Op


def evaluate(graph: Graph, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
    nodes = graph.nodes
    values = []
    # Forge exposes numerical results. NumPy's warning and exception policy is separate.
    with np.errstate(all="ignore"):
        for node in nodes:
            op = node.op
            operands = [values[index] for index in node.operands]
            if op == Op.INPUT:
                value = arrays[node.input_index]
            elif op == Op.LITERAL:
                value = np.float32(node.literal_value)
            elif op == Op.SPLAT:
                value = np.broadcast_to(operands[0], node.shape)
            elif op == Op.ADD:
                value = np.add(*operands, dtype=np.float32)
            elif op == Op.MULTIPLY:
                value = np.multiply(*operands, dtype=np.float32)
            elif op == Op.SQRT:
                value = np.sqrt(operands[0], dtype=np.float32)
            else:
                raise RuntimeError(f"Unexpected graph operation {op}")
            values.append(value)

    result = values[graph.output]
    # An expression can return an input directly. Its result still owns its storage.
    return result.copy(order="C") if nodes[graph.output].op == Op.INPUT else result
