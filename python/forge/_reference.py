"""Evaluate a native tensor graph with a float32 result at each operation."""

import numpy as np

from ._forge_ir import Graph, Op


def _reduce(op, array):
    flat = array.ravel()
    if op == Op.MAX:
        value = np.max(flat)
        # NumPy's maximum can keep either zero sign on a tie.
        if value == 0:
            has_positive_zero = np.any((flat == 0) & ~np.signbit(flat))
            return np.float32(0.0 if has_positive_zero else -0.0)
        return value

    # Prepend the specified initial value before the ordered float32 fold.
    terms = np.empty(flat.size + 1, dtype=np.float32)
    terms[0] = 0
    terms[1:] = flat
    total = np.add.accumulate(terms)[-1]
    if op == Op.MEAN:
        return np.float32(np.nan) if flat.size == 0 else np.divide(total, np.float32(flat.size))
    return total


def _matmul(left, right):
    shape = left.shape[0], right.shape[1]
    result = np.zeros(shape, dtype=np.float32)
    if result.size:
        product = np.empty_like(result)
        for k in range(left.shape[1]):
            np.multiply(left[:, k, None], right[k, None, :], out=product)
            np.add(result, product, out=result)
    return result


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
            elif op in (Op.SUM, Op.MAX, Op.MEAN):
                value = _reduce(op, operands[0])
            elif op == Op.MATMUL:
                value = _matmul(*operands)
            else:
                raise RuntimeError(f"Unexpected graph operation {op}")
            values.append(value)

    result = values[graph.output]
    # An expression can return an input directly. Its result still owns its storage.
    return result.copy(order="C") if nodes[graph.output].op == Op.INPUT else np.asarray(result)
