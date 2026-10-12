"""Render and check citation spans so a model cannot cite what it was not shown.

What goes wrong without this
----------------------------
A generator writes ``[3]`` into its answer because that is the number that
looked right. If only two passages were in the context, ``[3]`` refers to
nothing -- and a reader who follows it finds either a dead end or, far worse,
a different source that happens to be numbered third elsewhere. Citation
numbers are an interface between the model and the reader, and an interface
with no validation is a lie waiting to happen.

This component is the check. It finds every span, resolves it against the
``Context``'s own ``chunks`` -- which are, by construction in
``context-selector``, exactly the passages that were shown, in order -- and
refuses to render text whose spans do not resolve.

Numbering is 1-based
--------------------
``[1]`` means ``chunks[0]``. Nobody writes ``[0]`` in prose, and the off-by-one
between "first item" and "index zero" is the single most common defect in
citation code. It is decided here, once, and stated in every message.

``format`` appends nothing
-------------------------
A natural thing for a "formatter" to do is append a numbered bibliography.
It does not, deliberately: appending text to a ``Context`` makes it longer
than the budget ``context-selector`` just enforced, so the caller would send
more than the model has room for and never find out. Sources come back
separately from :meth:`CitationFormatter.bibliography`, to be rendered
wherever the application wants them -- a footnote, a sidebar, a hover card --
by code that knows what those extra tokens cost.

Text is returned unchanged, so ``len`` and token counts stay true.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from core_types import Context, ScoredDocument

__all__ = [
    "CitationFormatter",
    "CitationInputError",
    "CitationOrphanSpanError",
    "DEFAULT_PATTERN",
]

#: Read once at import, for the documented messages.
MANIFEST_MESSAGE = json.loads(
    (Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8")
)["missing_dependency_message"]

#: A bracketed integer. Anything else in brackets -- ``[note]``, ``[1.5]`` --
#: is ordinary text and is left alone.
DEFAULT_PATTERN = r"\[(\d+)\]"


class CitationError(Exception):
    """Base for every failure in this component."""


class CitationInputError(CitationError):
    """The caller passed something this component cannot render."""


class CitationOrphanSpanError(CitationError):
    """A citation span names a passage that was not in the context.

    Carries ``spans`` -- every unresolved ``(number, position)`` pair -- so a
    caller can report all of them at once instead of fixing one per run.
    """

    def __init__(self, message: str, *, spans: Sequence[tuple[int, int]]) -> None:
        super().__init__(message)
        self.spans = tuple(spans)


class CitationFormatter:
    """Resolve ``[n]`` spans against a ``Context``'s chunks.

    ``pattern`` must contain exactly one capture group yielding the citation
    number, so callers can adopt a house style -- ``(1)``, ``§1`` -- without
    rewriting this component.
    """

    def __init__(
        self,
        *,
        pattern: str = DEFAULT_PATTERN,
        one_based: bool = True,
        name: str = "citation-formatter",
    ) -> None:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise CitationInputError(
                f"citation-formatter: pattern is not a valid regular "
                f"expression: {exc}"
            ) from exc
        if compiled.groups != 1:
            raise CitationInputError(
                f"citation-formatter: pattern must have exactly one capture "
                f"group yielding the citation number, got {compiled.groups}"
            )
        if not isinstance(one_based, bool):
            raise CitationInputError(
                f"citation-formatter: one_based must be a bool, got "
                f"{type(one_based).__name__}"
            )
        if not isinstance(name, str) or not name:
            raise CitationInputError(
                f"citation-formatter: name must be a non-empty string, got {name!r}"
            )
        self._pattern = compiled
        self.one_based = one_based
        self.name = name

    def _validate(self, context: object) -> Context:
        if not isinstance(context, Context):
            raise CitationInputError(
                f"citation-formatter: context must be a Context, got "
                f"{type(context).__name__}"
            )
        return context

    def _index_of(self, number: int) -> int:
        return number - 1 if self.one_based else number

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Every ``(number, character_offset)`` span in ``text``, in order."""
        if not isinstance(text, str):
            raise CitationInputError(
                f"citation-formatter: text must be a string, got "
                f"{type(text).__name__}"
            )
        return [
            (int(match.group(1)), match.start())
            for match in self._pattern.finditer(text)
        ]

    def references(self, context: Context) -> list[int]:
        """Citation numbers that appear in ``text``, in order, with repeats."""
        return [number for number, _ in self.spans(self._validate(context).text)]

    def validate(self, context: Context) -> None:
        """Raise ``CitationOrphanSpanError`` if any span is unresolvable.

        A span is resolvable when it lands inside ``context.chunks``. Text
        with no spans at all is valid -- a model is allowed to answer without
        citing, and inventing citations to satisfy a validator would be worse.
        """
        ctx = self._validate(context)
        orphans = [
            (number, offset)
            for number, offset in self.spans(ctx.text)
            if not 0 <= self._index_of(number) < len(ctx.chunks)
        ]
        if not orphans:
            return
        listed = ", ".join(f"[{n}] at offset {o}" for n, o in orphans)
        raise CitationOrphanSpanError(
            f"citation-formatter: {len(orphans)} citation span(s) name a "
            f"passage that was not in the context: {listed}. The context holds "
            f"{len(ctx.chunks)} passage(es), so "
            f"{self._span_range(len(ctx.chunks))} are the only resolvable "
            f"numbers. A span that does not resolve is a source the model was "
            f"never shown",
            spans=orphans,
        )

    def _span_range(self, count: int) -> str:
        if count == 0:
            return "none at all"
        if self.one_based:
            return f"[1] through [{count}]"
        return f"[0] through [{count - 1}]"

    def format(self, context: Context) -> str:
        """Validate ``context`` and return its text unchanged.

        Unchanged matters: see the module docstring. This never grows the
        text, so a budget enforced upstream stays true.
        """
        self.validate(context)
        return self._validate(context).text

    def bibliography(self, context: Context) -> list[str]:
        """One label per distinct cited passage, in first-appearance order.

        Uncited passages are omitted -- a bibliography lists what was cited.
        A span resolving to a chunk with no ``source_name`` falls back to its
        ``document_id``, then its ``chunk_id``, then a positional label, so a
        citation always renders as *something* traceable rather than blank.
        """
        ctx = self._validate(context)
        ordered: list[str] = []
        seen: set[int] = set()
        for number, _ in self.spans(ctx.text):
            index = self._index_of(number)
            if not 0 <= index < len(ctx.chunks) or index in seen:
                continue
            seen.add(index)
            ordered.append(self._label_for(ctx.chunks[index], number))
        return ordered

    def _label_for(self, chunk: ScoredDocument, number: int) -> str:
        for candidate in (chunk.source_name, chunk.document_id, chunk.chunk_id):
            if candidate:
                return str(candidate)
        return f"passage {number}"

    def __repr__(self) -> str:
        return (
            f"CitationFormatter(pattern={self._pattern.pattern!r}, "
            f"one_based={self.one_based}, name={self.name!r})"
        )
