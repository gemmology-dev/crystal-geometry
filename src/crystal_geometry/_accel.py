"""
Lazy-loading acceleration wrapper.

Provides transparent fallback between native C++ implementations (when available)
and pure Python implementations. The native module is only loaded on first use.

Usage:
    from crystal_geometry._accel import prefer_native, get_backend

    @prefer_native
    def my_function(x, y):
        # Pure Python implementation
        return x + y

    # Check which backend is active
    print(get_backend())  # "native" or "python"
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

# Lazy-loaded state
_NATIVE_AVAILABLE: bool | None = None
_native_module: Any = None

F = TypeVar("F", bound=Callable[..., Any])


def _check_native() -> bool:
    """Check if native C++ module is available.

    Returns:
        True if native module loaded successfully and has required functions, False otherwise
    """
    global _NATIVE_AVAILABLE, _native_module

    if _NATIVE_AVAILABLE is None:
        try:
            from . import _native as native

            # Verify the module has the expected functions (not just a directory import)
            if hasattr(native, "halfspace_intersection") or hasattr(native, "__version__"):
                _native_module = native
                _NATIVE_AVAILABLE = True
            else:
                # Module imported but doesn't have expected functions
                # This happens when _native/ is a directory without compiled extension
                _NATIVE_AVAILABLE = False
        except ImportError:
            _NATIVE_AVAILABLE = False

    return _NATIVE_AVAILABLE


def prefer_native(python_impl: F) -> F:
    """Decorator that prefers native implementation when available.

    If a native C++ function with the same name exists in the _native module,
    it will be used instead of the Python implementation.

    Args:
        python_impl: The Python implementation to wrap

    Returns:
        Wrapped function that dispatches to native or Python implementation
    """

    @functools.wraps(python_impl)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _check_native():
            native_fn = getattr(_native_module, python_impl.__name__, None)
            if native_fn is not None:
                return native_fn(*args, **kwargs)
        return python_impl(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def get_backend() -> str:
    """Get the currently active backend name.

    Returns:
        "native" if C++ module is available and loaded, "python" otherwise
    """
    return "native" if _check_native() else "python"


def get_backend_info() -> dict[str, Any]:
    """Get detailed information about the active backend.

    Returns:
        Dictionary with backend details including version and capabilities
    """
    info: dict[str, Any] = {
        "backend": get_backend(),
        "python_available": True,
    }

    if _check_native() and _native_module is not None:
        info["native_available"] = True
        info["native_version"] = getattr(_native_module, "__version__", "unknown")
        info["native_functions"] = [
            name
            for name in dir(_native_module)
            if not name.startswith("_") and callable(getattr(_native_module, name))
        ]

        # Check for OpenMP support
        if hasattr(_native_module, "get_num_threads"):
            info["openmp_enabled"] = True
            info["num_threads"] = _native_module.get_num_threads()
        else:
            info["openmp_enabled"] = False
    else:
        info["native_available"] = False

    return info


def set_num_threads(n: int | None = None) -> None:
    """Set number of threads for parallel operations.

    Only has effect when native module is available and compiled with OpenMP.

    Args:
        n: Number of threads to use, or None for automatic (uses all cores)
    """
    if _check_native() and _native_module is not None:
        if hasattr(_native_module, "set_num_threads"):
            _native_module.set_num_threads(n or 0)


def get_num_threads() -> int:
    """Get current number of threads for parallel operations.

    Returns:
        Number of threads, or 1 if native module not available
    """
    if _check_native() and _native_module is not None:
        if hasattr(_native_module, "get_num_threads"):
            return _native_module.get_num_threads()
    return 1
