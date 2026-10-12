"""Transitive dependency resolution, and the report `ragkit add` prints (FR-15).

U2 resolved: ``core-types`` ships as a component like any other
--------------------------------------------------------------
Shared contracts are not injected by the CLI's own package -- a destination
project that received a chunker and had to import ``Chunk`` from
``rag_code_library`` would depend on the library it is supposed to be copying
*away* from. So ``core-types`` is a component, and every component that needs it
declares it, which makes the graph the single source of what a component needs
(DECISION-fcd5b130, FR-12).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..registry import ComponentManifest, Registry, UnknownComponentError

__all__ = ["DependencyError", "PackageReport", "package_report", "render_report", "resolve_order"]


class DependencyError(RuntimeError):
    """A component's dependency graph cannot be satisfied.

    Raised rather than returned so ``main`` renders it as one readable line
    (FR-19, NFR-7). The message carries the whole report: there is no second
    piece of state for the caller to go and fetch.
    """


def resolve_order(component_id: str, registry: Registry) -> tuple[str, ...]:
    """Every component to add, dependencies before the components needing them.

    Depth-first, post-order: a component is appended only once everything it
    depends on has been. ``ragkit add`` copies in this order so that a written
    module never imports one that has not been written yet.

    A cycle is reported with the path that closes it, not merely as "there is a
    cycle" -- "a -> b -> a" names the two manifests to edit, and the other name
    does not.
    """
    order: list[str] = []
    # 1 == on the current path, 2 == finished. Absent == not yet visited.
    state: dict[str, int] = {}
    path: list[str] = []

    def visit(current: str) -> None:
        marker = state.get(current)
        if marker == 2:
            return
        if marker == 1:
            cycle = path[path.index(current) :] + [current]
            raise DependencyError(
                "dependency cycle: " + " -> ".join(cycle)
                + "; break it by removing one of those dependencies"
            )
        state[current] = 1
        path.append(current)
        # ``Registry.get`` returns None rather than raising, so the unknown-id
        # error is raised here -- the same one ``info`` produces, naming the ids
        # that do exist, rather than an AttributeError on a None entry.
        entry = registry.get(current)
        if entry is None:
            raise UnknownComponentError(current, registry.ids) from None
        for dependency in entry.manifest.dependencies:
            visit(dependency)
        path.pop()
        state[current] = 2
        order.append(current)

    visit(component_id)
    return tuple(order)


@dataclass(frozen=True, slots=True)
class PackageReport:
    """What the added components need, and the commands to install it.

    Data, not prose: the CLI prints it and nothing else reads it, but keeping
    it as fields means the FR-15 requirement ("prints the component's required
    and optional packages ... plus any external service or API key") is
    satisfied by construction rather than by a string that happens to mention
    them today.
    """

    required: tuple[str, ...]
    """Distributions every added component imports, as ``name`` + specifier."""

    optional_by_component: tuple[tuple[str, str], ...]
    """``(distribution, component_id)`` for imports guarded by an ImportError."""

    services: tuple[tuple[str, str, tuple[str, ...]], ...]
    """``(service, notes, environment variables)`` the components expect."""


def package_report(manifests: Iterable[ComponentManifest]) -> PackageReport:
    """Aggregate the dependencies of every component being added.

    Required and optional are kept apart on purpose: merging them would make
    ``ragkit add`` look like it demands a cross-encoder for a chunker that
    merely offers one, which is the impression FR-15's separation exists to
    avoid.
    """
    # A dict keyed by "name<specifier>" with a None value, rather than a set:
    # insertion order keeps the printed report stable across runs, and sets of
    # strings iterate in hash order. Keys only for `required` -- the optional
    # block below shows which component offered a package, because that changes
    # what the user has to install, whereas "core-types and settings both need
    # pydantic" is a line answering a question nobody asked.
    required: dict[str, None] = {}
    optional: dict[str, str] = {}
    services: dict[str, tuple[str, tuple[str, ...]]] = {}

    for manifest in manifests:
        for package in manifest.required_packages:
            required[f"{package.name}{package.specifier}"] = None
        for package in manifest.optional_packages:
            optional.setdefault(f"{package.name}{package.specifier}", manifest.id)
        for service in manifest.external_services:
            services.setdefault(service.name, (service.notes, service.env_vars))

    return PackageReport(
        required=tuple(required),
        optional_by_component=tuple(optional.items()),
        services=tuple(
            (name, notes, env_vars) for name, (notes, env_vars) in services.items()
        ),
    )


def _install_command(distributions: Sequence[str]) -> str:
    """The command a human runs; ragkit prints it and never executes it (FR-15)."""
    quoted = " ".join(f'"{distribution}"' for distribution in distributions)
    return f"uv add {quoted}"


def render_report(report: PackageReport) -> str:
    """The FR-15 block, printed after a successful add.

    Everything here is a suggestion for the human to run. ``ragkit add`` never
    installs a package and never touches ``pyproject.toml``, ``uv.lock`` or
    ``requirements*.txt`` -- not even with ``--force`` -- because a component
    copy that silently changed a project's dependency set would be the one
    operation nobody could review afterwards.
    """
    lines = ["Packages required by the added components:"]
    lines.append(f"  {', '.join(report.required) or '(none)'}")
    if report.required:
        lines.append(f"  install with: {_install_command(report.required)}")

    lines.append("Packages optional -- add only the ones you use:")
    if report.optional_by_component:
        width = max(len(d) for d, _ in report.optional_by_component)
        for distribution, component in report.optional_by_component:
            lines.append(f"  {distribution:<{width}}  ({component})")
        lines.append(
            "  install with: "
            + _install_command([d for d, _ in report.optional_by_component])
        )
    else:
        lines.append("  (none)")

    lines.append("External services these components expect:")
    if report.services:
        for name, notes, env_vars in report.services:
            lines.append(f"  {name}: {notes}")
            lines.append(
                f"    environment: {', '.join(env_vars) if env_vars else '(none)'}"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines)
