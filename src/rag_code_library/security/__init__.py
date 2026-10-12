"""Security helpers for the RAG Code Library.

:mod:`path_safety` is the single choke point for filesystem writes (NFR-3). No
CLI command, component, or script may open a file for writing except through
:class:`~rag_code_library.security.path_safety.SafeDestination`.
"""

from __future__ import annotations

from .path_safety import (
    MAX_NATIVE_PATH,
    MAX_PATH,
    RESERVED_NAMES,
    SafeDestination,
    UnsafePathError,
)

__all__ = [
    "MAX_NATIVE_PATH",
    "MAX_PATH",
    "RESERVED_NAMES",
    "SafeDestination",
    "UnsafePathError",
]
