"""Thread-local debug trace collector for pipeline diagnostics."""

import threading

_local = threading.local()


def _get_trace() -> list[str]:
    if not hasattr(_local, "trace"):
        _local.trace = []
    return _local.trace


def trace(msg: str) -> None:
    """Append a debug line to the current request's trace."""
    _get_trace().append(msg)


def get_trace() -> list[str]:
    """Return the current trace and reset it."""
    t = _get_trace()
    result = list(t)
    t.clear()
    return result
