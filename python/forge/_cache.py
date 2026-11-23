"""Bounded process-local cache of compiled code without execution workspaces."""

from collections import OrderedDict, namedtuple
from threading import Lock

_CAPACITY = 16
_entries = OrderedDict()
_lock = Lock()
_hits = 0
_misses = 0
CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


def get_or_compile(key, compile_code):
    global _hits, _misses
    with _lock:
        code = _entries.get(key)
        if code is not None:
            _hits += 1
            _entries.move_to_end(key)
            return code

        _misses += 1
        code = compile_code()
        _entries[key] = code
        if len(_entries) > _CAPACITY:
            _entries.popitem(last=False)
        return code


def cache_info():
    """Return process-local hit, miss, capacity and entry counts."""
    with _lock:
        return CacheInfo(_hits, _misses, _CAPACITY, len(_entries))


def clear_cache():
    """Clear cached code and counters. Existing handles retain their code."""
    global _hits, _misses
    with _lock:
        _entries.clear()
        _hits = _misses = 0
