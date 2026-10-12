"""``ragkit`` — the RAG Code Library command-line interface.

This module owns argument parsing, exit-code policy, and error presentation.
``ragkit add`` delegates to :mod:`rag_code_library.generator`; ``list`` and
``info`` are rendered here because they are pure presentation over fields the
registry already owns. The registry stays the single source of component
metadata (FR-12) -- nothing below re-describes a component, it formats one.

Exit-code policy (stable contract, relied on by scripts and CI):

==========  ==========================================================
``0``       Success.
``1``       A handled domain error (unknown component, unsafe path,
            dependency conflict, unreadable manifest). The message names
            the component, path, or setting at fault and the fix.
``2``       Argument misuse — argparse's own "invalid choice" / missing
            argument handling.
==========  ==========================================================

Expected failures never print a raw traceback: ``main`` catches the library's
domain errors and renders them as a single readable line. Programming errors
still raise, because swallowing those hides bugs.

Subcommand implementations are imported lazily *inside* each handler. That keeps
``ragkit --help`` instant and lets each area ship independently: a subcommand
whose backing module is not yet present reports a clean, actionable error rather
than an import-time crash of the whole CLI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:
    from .registry import Package, Registry

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_DOMAIN_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Kept as a separate function so tests can construct the parser, inspect its
    choices, and assert on ``--help`` output without spawning a process.
    """
    parser = argparse.ArgumentParser(
        prog="ragkit",
        description="List, inspect, and add copy-ready RAG components to a project.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument(
        "--category",
        metavar="NAME",
        help="Only show components in this category (for example: chunking)",
    )

    info_parser = subparsers.add_parser("info", help="Show details for one component")
    info_parser.add_argument("id", metavar="ID", help="Component id, for example fixed-size-chunker")

    add_parser = subparsers.add_parser("add", help="Add a component to a destination project")
    add_parser.add_argument("id", metavar="ID", help="Component id to add")
    add_parser.add_argument(
        "--destination",
        metavar="PATH",
        default=".",
        help="Destination project root (default: current directory)",
    )
    add_parser.add_argument(
        "--package",
        metavar="NAME",
        help=(
            "Python package the component lands in; detected from "
            "pyproject.toml when omitted"
        ),
    )
    add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the file operations that would run, without writing anything",
    )
    add_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files inside the destination root",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run the ingestion pipeline over a source directory",
        description=(
            "Run the stages named by RAG_PIPELINE__STAGES over every document "
            "in SOURCE, writing results and a run summary into DESTINATION."
        ),
    )
    ingest_parser.add_argument(
        "source", metavar="SOURCE", help="Directory holding the documents to ingest"
    )
    ingest_parser.add_argument(
        "destination",
        metavar="DESTINATION",
        help="Directory the pipeline writes into; also where the run lock is taken",
    )
    ingest_parser.add_argument(
        "--stages",
        metavar="MODULE:FACTORY",
        help=(
            "Override RAG_PIPELINE__STAGES: a dotted reference to a function "
            "returning the stages to run"
        ),
    )
    ingest_parser.add_argument(
        "--failure-policy",
        metavar="POLICY",
        choices=("stop", "skip", "quarantine"),
        help=(
            "What one document's failure does to the rest of the run "
            "(default: skip, or RAG_PIPELINE__FAILURE_POLICY)"
        ),
    )
    ingest_parser.add_argument(
        "--fingerprint",
        metavar="TEXT",
        help=(
            "Configuration fingerprint recorded with each document; a document "
            "whose fingerprint differs is reprocessed rather than skipped"
        ),
    )
    ingest_parser.add_argument(
        "--summary",
        metavar="PATH",
        help="Where to write the run summary, relative to DESTINATION",
    )
    ingest_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the run summary to stdout (it is still written)",
    )

    return parser


class _CliError(RuntimeError):
    """A failure the CLI detected itself.

    Raised from here rather than re-used from the registry, so ``main``'s
    "does this exception belong to this package" check recognises it and
    renders one readable line instead of a traceback (NFR-7).
    """


def _components_root() -> Path:
    """Locate the ``components`` tree, walking up from the current directory.

    The documented invocation is from the repository root, but a tool that
    stops working the moment you ``cd`` into a subdirectory is a support
    ticket, and walking up costs four lines.
    """
    here = Path.cwd()
    for base in (here, *here.parents):
        candidate = base / "components"
        if candidate.is_dir() and any(candidate.rglob("manifest.json")):
            return candidate
    raise _CliError(
        f"no 'components' directory found above {here.as_posix()}; run ragkit "
        "from inside the RAG Code Library repository"
    )


def _registry(root: Path | None = None) -> Registry:
    """Load the registry, or fail with one readable reason (FR-11, NFR-7).

    ``root`` is accepted so that ``add`` can walk up to the components tree
    once rather than twice; it resolves the same path either way.
    """
    # Imported here so ``ragkit --help`` does not pay for pydantic.
    from .registry import Registry

    registry = Registry.load(_components_root() if root is None else root)
    if not registry.ok:
        for issue in registry.issues:
            print(f"ragkit: {issue}", file=sys.stderr)
        raise _CliError(
            f"{len(registry.issues)} manifest problem(s) must be fixed first; "
            "see the lines above"
        )
    return registry


def _cmd_list(category: str | None) -> int:
    """Print one line per component: id, category, status, description (FR-13).

    ``--category`` is validated rather than silently returning nothing: an
    empty result for a mistyped category reads as "this library has no
    chunking components", which is the opposite of the truth (NFR-7).
    """
    registry = _registry()
    manifests = sorted(registry.manifests, key=lambda manifest: manifest.id)

    if category is not None:
        known = {manifest.category for manifest in registry.manifests}
        if category not in known:
            raise _CliError(
                f"unknown category {category!r}; known categories: "
                f"{', '.join(sorted(known))}"
            )
        manifests = [m for m in manifests if m.category == category]

    # Unreachable while the registry is ok (zero entries always yields an
    # issue), but a traceback here would cost more than the branch.
    if not manifests:
        return EXIT_OK

    width = max(len(manifest.id) for manifest in manifests)
    print(f"{'id':<{width}}  {'category':<11}  {'status':<13}  description")
    print(f"{'-' * width}  {'-' * 11}  {'-' * 13}  {'-' * 11}")
    for manifest in manifests:
        print(
            f"{manifest.id:<{width}}  {manifest.category:<11}  "
            f"{manifest.status:<13}  {manifest.description}"
        )
    return EXIT_OK


def _example_file(directory: Path, documentation: Path | None) -> str:
    """Name the example this component ships (FR-14, AC-8).

    FR-6 gives every sub-component its own example, so naming the file is
    normally all that is needed. ``settings`` predates that rule and ships an
    environment template instead. Where there is genuinely no example, saying
    so and pointing at the README beats printing a path that does not exist.
    """
    for name in ("example.py", ".env.example"):
        if (directory / name).is_file():
            return name
    if documentation is not None:
        return f"(no example file -- see {documentation.name})"
    return "(none)"


def _cmd_info(component_id: str) -> int:
    """Print one component's full details (FR-14, FR-35, AC-8).

    An unknown id is deliberately *not* caught here: ``UnknownComponentError``
    is a domain error, so ``main`` prints it as one line naming the id and the
    ids that do exist, then exits 1.
    """
    registry = _registry()
    resolved = registry.resolve(component_id)
    manifest = resolved.manifest

    def packages(items: Sequence[Package]) -> str:
        return ", ".join(f"{item.name}{item.specifier}" for item in items) or "(none)"

    interpreter = f"python >= {manifest.requires_python.min}"
    if manifest.requires_python.max:
        interpreter += f", < {manifest.requires_python.max}"

    print(f"{manifest.id} - {manifest.name}")
    print(
        f"category: {manifest.category}   status: {manifest.status}   "
        f"version: {manifest.version}"
    )
    print()
    print("Description")
    print(f"  {manifest.description}")
    print()
    print("Dependencies")
    print(
        f"  {', '.join(manifest.dependencies) if manifest.dependencies else '(none)'}"
    )
    print()
    print("Requires")
    print(f"  {interpreter}")
    print(f"  packages: {packages(manifest.required_packages)}")
    print(f"  optional: {packages(manifest.optional_packages)}")
    print(
        "  services: "
        + (", ".join(s.name for s in manifest.external_services) or "(none)")
    )
    print()
    print("Example")
    print(f"  {_example_file(resolved.directory, resolved.documentation)}")
    print()
    print("Usage notes")
    # ``install`` stores "Copy with: ragkit add <id>" for components that need
    # no packages -- the placeholder is there to be resolved, and the id is
    # known here (unlike ``destination``'s {project_package}, which only
    # ``ragkit add`` can substitute).
    print(f"  {manifest.install.replace('<id>', manifest.id)}")
    print()
    print("Trade-offs")
    print(f"  {manifest.tradeoffs or '(not recorded in this manifest)'}")
    return EXIT_OK


def _cmd_add(args: argparse.Namespace) -> int:
    """Copy a component and everything it depends on into a project (FR-15).

    The import is isolated from the calls below it: a broad ``except
    ImportError`` around both would swallow an ImportError raised *inside* a
    component -- precisely the missing-optional-dependency case FR-3 requires
    to be reported with the distribution name and the install command -- and
    mislabel it as "not available in this build".
    """
    try:
        from .generator import AddError, PartialFailure, add_component, render
    except ImportError as exc:
        raise _NotReady("add", exc) from exc

    from .security.path_safety import UnsafePathError

    library_root = _components_root()
    try:
        result = add_component(
            args.id,
            destination=args.destination,
            package=args.package,
            dry_run=args.dry_run,
            force=args.force,
            registry=_registry(library_root),
            library_root=library_root,
        )
    except (AddError, PartialFailure, UnsafePathError) as exc:
        print(f"ragkit: {exc}", file=sys.stderr)
        return EXIT_DOMAIN_ERROR

    verb = "would copy" if result.dry_run else "copied"
    print(f"{verb} {len(result.operations)} file(s) for {result.component_id}:")
    print(render(result.operations))
    print()
    # Said plainly because it is the part of FR-15 most likely to be doubted:
    # nothing below installs anything or edits a dependency file.
    print("Installed nothing, and changed no dependency file.")
    print(result.report)
    return EXIT_OK


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline over a source directory (FR-51).

    This function names no parser, no chunker and no storage backend, and
    imports no component: every stage-shaped decision arrives through
    configuration as a callable (FR-54). What it does here is the part that is
    genuinely generic -- check the arguments, resolve the configured factory,
    run, report the outcome and write the summary.
    """
    import json

    from .config import load_pipeline_settings, resolve_stages, with_overrides
    from .pipeline import CheckpointStore, Pipeline
    from .security.path_safety import SafeDestination

    source = Path(args.source)
    if not source.is_dir():
        raise _CliError(
            f"source {str(source)!r} is not a directory; ragkit ingest reads "
            "documents from a directory"
        )

    overrides = {
        key: value
        for key, value in (
            ("stages", args.stages),
            ("failure_policy", args.failure_policy),
            ("fingerprint", args.fingerprint),
            ("summary_file", args.summary),
        )
        if value is not None
    }
    try:
        settings = with_overrides(load_pipeline_settings(), **overrides)
        stages = resolve_stages(settings)
    except ValueError as exc:
        # A configuration mistake is the one failure a user reaches by typing,
        # so it gets its message rather than a traceback.
        raise _CliError(str(exc)) from exc

    pipeline = Pipeline(
        stages,
        failure_policy=settings.policy(),
        fingerprint=settings.fingerprint,
        destination=args.destination,
        lock_file=settings.lock_file,
        # Held on disk rather than in memory, so a second `ragkit ingest` over
        # an unchanged tree skips the work the first one did (FR-53) instead of
        # quietly redoing it.
        checkpoints=CheckpointStore(args.destination),
    )
    summary = pipeline.run(source)

    # Written through the same guarded path as everything else (NFR-3), so a
    # summary path that tries to escape the destination is refused rather than
    # quietly honoured.
    written = SafeDestination(args.destination).write_text(
        settings.summary_file,
        json.dumps(summary.to_dict(), indent=2) + "\n",
        force=True,
    )

    if not args.quiet:
        print(summary.report())
        print(f"  summary written to {written}")

    # Rejected and skipped documents are normal outcomes, not failures: an
    # unchanged second run is a success. Only documents the pipeline could not
    # finish make the run non-zero.
    if summary.documents_failed or summary.documents_quarantined:
        return EXIT_DOMAIN_ERROR
    return EXIT_OK


def _dispatch(args: argparse.Namespace) -> int:
    """Route a parsed command to its implementation.

    Imports are lazy so that a missing or half-built area degrades into a clean
    domain error instead of breaking every ``ragkit`` invocation at import time.
    """
    # ``list`` and ``info`` are implemented above, so nothing is imported for
    # them. ``add`` imports its own module inside ``_cmd_add``, for the reason
    # that function documents: an ImportError raised *inside* a component must
    # surface as FR-3's message, not be relabelled "not available in this build".
    if args.command == "list":
        return _cmd_list(category=args.category)
    if args.command == "info":
        return _cmd_info(args.id)
    if args.command == "add":
        return _cmd_add(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    return EXIT_USAGE_ERROR


class _NotReady(RuntimeError):
    """A subcommand's backing module is not importable in this build."""


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ragkit`` console script.

    Returns the process exit code rather than calling ``sys.exit`` so tests can
    assert on it in-process. ``__main__`` is the only place that exits.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 0 for --help, 2 for misuse
        return int(exc.code or EXIT_OK)

    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        return _dispatch(args)
    except _NotReady as exc:
        print(
            f"ragkit: the '{args.command}' command is not available in this build "
            f"(its module could not be imported: {exc.__cause__})",
            file=sys.stderr,
        )
        return EXIT_DOMAIN_ERROR
    except _CliError as exc:
        # Explicit, not left to the module-name check below: under
        # ``python -m rag_code_library.cli`` this module *is* ``__main__``, so
        # ``type(exc).__module__`` is "__main__" and the check would miss it,
        # turning a readable failure into a traceback (NFR-7).
        print(f"ragkit: {exc}", file=sys.stderr)
        return EXIT_DOMAIN_ERROR
    except ImportError as exc:
        # FR-3: a component that needs an optional library raises ImportError
        # naming the distribution, the component, and the install command.
        # That is an expected user-facing failure, so NFR-7 requires a clean
        # line rather than a traceback. The message text is the component's to
        # author; the CLI only presents it.
        print(f"ragkit: {exc}", file=sys.stderr)
        return EXIT_DOMAIN_ERROR
    except Exception as exc:
        # Domain errors are typed by the library and carry a readable message.
        # Anything else is a programming error and must stay loud.
        message = getattr(exc, "message", None) or str(exc)
        if type(exc).__module__.startswith("rag_code_library"):
            print(f"ragkit: {message}", file=sys.stderr)
            return EXIT_DOMAIN_ERROR
        raise


if __name__ == "__main__":  # pragma: no cover - exercised via console script
    sys.exit(main())
