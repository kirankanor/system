"""RAG Code Library — copy-ready, production-oriented RAG components.

The library ships three things:

* **Components** under ``components/``: framework-neutral Python with a canonical
  ``manifest.json``, a README, a runnable example, and tests.
* **The ``ragkit`` CLI**: lists components, shows their details, and copies them
  into a destination project without touching that project's dependencies.
* **A local cookbook website**: generated from the same manifests, opened from
  disk with no backend.

Import name is ``rag_code_library``; the console script is ``ragkit``.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Package version. Kept in sync with ``pyproject.toml``; bump both together.
__version__ = "0.1.0"
