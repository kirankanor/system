"""Runnable example for `core-types`.

    python example.py

Three stages hand the SAME result type to each other with no conversion:
a chunker produces `Chunk`, a retriever scores it into `ScoredDocument`,
and a context builder assembles a `Context`. Nothing here imports anything
outside this file plus pydantic, so it runs standalone.

The chunker/retriever/builder below are deliberately tiny fakes: the point of
this example is the *shape* of the data flowing between stages, not how good
the algorithms are.
"""

from __future__ import annotations

from collections.abc import Sequence

from core_types import (
    Chunk,
    ChunkingProtocol,
    Context,
    ContextBuilderProtocol,
    ScoredDocument,
)

ARTICLE = (
    "Retrieval augmented generation grounds an answer in retrieved text. "
    "The retriever finds candidate passages, a reranker orders them by "
    "relevance, and the context builder packs the best of them into a budget. "
    "Everything downstream depends on those stages agreeing on one result type."
)


class SlidingWindowChunker:
    """Implements ChunkingProtocol: fixed windows with an explicit overlap."""

    def chunk(self, text: str, *, size: int, overlap: int, **kw: object) -> list[Chunk]:
        if overlap >= size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than size ({size}); "
                "silently clamping would produce chunks the caller did not ask for"
            )
        words = text.split()
        if not words:
            return []
        step = size - overlap
        out: list[Chunk] = []
        start = 0
        while start < len(words):
            window = words[start : start + size]
            out.append(Chunk(chunk_index=len(out), text=" ".join(window)))
            if start + size >= len(words):
                break
            start += step
        return out


class NaiveScorer:
    """Implements ContextBuilderProtocol: keep whole chunks until the budget runs out."""

    def build(
        self, query: str, chunks: Sequence[ScoredDocument], budget: int
    ) -> Context:
        wanted = {w.lower().strip("?,.") for w in query.split()}
        scored = sorted(chunks, key=lambda c: -self._overlap(wanted, c.text))

        kept: list[ScoredDocument] = []
        tokens = 0
        truncated = False
        for candidate in scored:
            cost = len(candidate.text.split())
            if tokens + cost > budget:
                truncated = True
                break
            kept.append(candidate)
            tokens += cost

        return Context(
            text="\n".join(c.text for c in kept),
            chunks=tuple(kept),
            token_count=tokens,
            budget=budget,
            truncated=truncated,
        )

    @staticmethod
    def _overlap(wanted: set[str], text: str) -> int:
        return len(wanted & {w.lower() for w in text.split()})


def retrieve(chunks: Sequence[Chunk], query: str) -> list[ScoredDocument]:
    """Stand-in retriever: score by query-term overlap, id = position."""
    terms = {w.lower() for w in query.split()}
    results = []
    for chunk in chunks:
        hits = len(terms & {w.lower() for w in chunk.text.split()})
        results.append(
            ScoredDocument.from_chunk(
                chunk,
                score=hits / max(len(terms), 1),
                document_id="example-article",
                source_name="example.txt",
            )
        )
    return results


def main() -> None:
    query = "how does the context builder pack a budget"

    chunker: ChunkingProtocol = SlidingWindowChunker()
    builder: ContextBuilderProtocol = NaiveScorer()

    chunks = chunker.chunk(ARTICLE, size=18, overlap=4)
    print(f"chunked into {len(chunks)} chunks")

    candidates = retrieve(chunks, query)
    for c in sorted(candidates, key=lambda c: -c.score)[:3]:
        print(f"  score={c.score:.2f} src={c.source_name} :: {c.text[:52]}...")

    ctx = builder.build(query, candidates, budget=40)
    print(
        f"\ncontext: {len(ctx.chunks)} chunks, {ctx.token_count}/{ctx.budget} tokens, "
        f"truncated={ctx.truncated}"
    )
    print(f"provenance: {[c.source_name for c in ctx.chunks]}")
    print(f"\n---\n{ctx.text}")

    # Failure path, shown rather than hidden: overlap >= size is rejected with
    # a typed message instead of being clamped behind the caller's back (FR-7).
    try:
        chunker.chunk(ARTICLE, size=10, overlap=10)
    except ValueError as exc:
        print(f"\nrejected as expected: {exc}")


if __name__ == "__main__":
    main()
