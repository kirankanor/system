"""Fill a manifest's ``destination`` template.

The one substitution the registry cannot do itself
--------------------------------------------------
Manifests declare ``src/{project_package}/rag/chunking/fixed_size_chunker.py``.
The placeholder is the destination project's package name, which only exists
once a destination has been chosen, so it is resolved here at add time. This is
the mirror of ``resolver.py``'s decision *not* to substitute ``destination``:
that module had no destination to substitute against, this one does.

This is not a containment check
-------------------------------
Rejecting a path that escapes the destination root belongs to
``security/path_safety.py``, which decides it on the resolved real path against
a root cached for the whole operation (NFR-3). Re-deriving containment here
would mean two places could disagree, and the one that lost would be the one
nobody was watching.
"""

from __future__ import annotations

from pathlib import PurePosixPath

__all__ = ["PACKAGE_PLACEHOLDER", "DestinationTemplateError", "destination_base", "resolve_destination"]

#: The only placeholder a manifest may contain.
PACKAGE_PLACEHOLDER = "{project_package}"


class DestinationTemplateError(RuntimeError):
    """A destination names a placeholder this CLI does not know how to fill."""


def resolve_destination(template: str, package: str) -> str:
    """Substitute the package name into ``template``.

    A brace left over after substitution names a placeholder nobody can fill.
    Left alone it would create a real directory literally called ``{foo}``,
    which is a wrong result that looks like a successful run, so it is refused.
    """
    resolved = template.replace(PACKAGE_PLACEHOLDER, package)
    if "{" in resolved or "}" in resolved:
        raise DestinationTemplateError(
            f"destination {template!r} names a placeholder this CLI cannot fill; "
            f"the only supported placeholder is {PACKAGE_PLACEHOLDER}"
        )
    return resolved


def destination_base(destination: str) -> PurePosixPath:
    """The directory a component's files are rooted at.

    A destination naming a directory ends in ``/``; one naming a file is its
    own parent. Both reduce to a directory, which keeps this to one rule rather
    than two.

    Note what this does *not* decide: where each declared file lands within it.
    That is the planner's call, and it has to be, because the source goes to
    its import path while the rest goes somewhere collision-free -- see
    ``planner.plan``.
    """
    path = PurePosixPath(destination)
    return path if destination.endswith("/") else path.parent
