"""Expressions and dimensions shared by the runner and report."""

from dataclasses import dataclass

LENGTHS = (32, 1000, 1024, 65536, 1048576, 8388608)
DEMO_LENGTHS = (1009, 67108864)
MATRICES = ((16, 16, 16), (31, 63, 17), (64, 64, 64), (256, 256, 256), (512, 512, 512))


@dataclass(frozen=True)
class Workload:
    key: str
    expression: str
    inputs: tuple[str, ...]


WORKLOADS = (
    Workload("E1", "x + y", ("x", "y")),
    Workload("E2", "x * 3", ("x",)),
    Workload("E3", "sqrt(x)", ("x",)),
    Workload("E4", "x * y + z", ("x", "y", "z")),
    Workload("E5", "sqrt((x + 2) * 3)", ("x",)),
    Workload("E6", "((x + 1) * 2 + 3) * 4", ("x",)),
    Workload("R1", "sum(x)", ("x",)),
    Workload("R2", "max(x)", ("x",)),
    Workload("R3", "mean(x)", ("x",)),
    Workload("M1", "A @ B", ("A", "B")),
)


def dimensions(workload, quick):
    if workload.key == "M1":
        return MATRICES[:1] if quick else MATRICES
    if quick:
        return ((LENGTHS[0],),)
    lengths = sorted((*LENGTHS, *DEMO_LENGTHS)) if workload.key == "E5" else LENGTHS
    return ((length,) for length in lengths)
