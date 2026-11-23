"""Forge's array expression frontend."""

from . import _forge_ir
from ._cache import cache_info, clear_cache
from ._language import Program, compile

__all__ = ["Program", "compile", "has_avx2", "cache_info", "clear_cache"]


def has_avx2() -> bool:
    """Check whether this build can execute native AVX2 kernels."""
    return hasattr(_forge_ir, "has_avx2") and _forge_ir.has_avx2()
