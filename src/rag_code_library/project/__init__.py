"""Destination-project helpers: root, package name, and destination templates.

Split by question, not by size: *is this a directory I may write into*, *what is
it called as a module*, and *what does this manifest's template become here*.
Each answer is needed at a different point in ``ragkit add``, and conflating
them is how a destination root ends up being re-resolved in more than one place.
"""

from __future__ import annotations

from .detector import ProjectRootError, project_root
from .package_detector import detect_package, normalize_package_name
from .path_resolver import (
    PACKAGE_PLACEHOLDER,
    DestinationTemplateError,
    destination_base,
    resolve_destination,
)

__all__ = [
    "PACKAGE_PLACEHOLDER",
    "DestinationTemplateError",
    "ProjectRootError",
    "destination_base",
    "detect_package",
    "normalize_package_name",
    "project_root",
    "resolve_destination",
]
