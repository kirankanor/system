"""Gate tests for the reranking area (T-09b).

Two components, both implementing ``RerankerProtocol`` from core-types:
``relevance-ranking`` (deterministic, dependency-free) and
``cross-encoder-reranker`` (model-backed, wheel guarded).

The three properties the task names as observable are asserted here by name:

1. score ordering is correct,
2. ties resolve deterministically,
3. a missing cross-encoder library raises ImportError naming the package.

The cross-encoder is exercised against an injected fake, because
``sentence-transformers`` is not installed here and installing it would modify
dependency files -- which this library forbids. The real model is unexecuted;
the component README says so plainly. What *is* proven is the property that
matters without it: the module imports, loads lazily, and fails with the
manifest's message verbatim.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RERANKING = ROOT / "components" / "reranking"
CORE_TYPES = ROOT / "components" / "storage" / "core_types"

for directory in (CORE_TYPES, RERANKING / "relevance_ranking", RERANKING / "cross_encoder"):
    sys.path.insert(0, str(directory))

import core_types as core  # noqa: E402
from cross_encoder_reranker import (  # noqa: E402
    MANIFEST_MESSAGE,
    CrossEncoderReranker,
    MissingDependency,
    RerankingError,
    RerankingInputError,
)
from relevance_ranking import (  # noqa: E402
    DEFAULT_STOPWORDS,
    RankingError,
    RankingInputError,
    RelevanceRanker,
    score_documents,
    tokenize,
)

import pytest  # noqa: E402


def scored(cid: str, text: str, **kwargs: Any) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    return core.ScoredDocument(chunk_id=cid, text=text, score=0.0, **kwargs)


@pytest.fixture
def candidates():
    return [
        scored("c0", "Python error handling with exceptions", chunk_index=0),
        scored("c1", "Cooking pasta with salt", chunk_index=1),
        scored("c2", "Java exceptions and Python compared", chunk_index=2),
    ]


class FakeCrossEncoder:
    """Speaks the slice of sentence_transformers.CrossEncoder we call."""

    def __init__(self, scores):
        self._scores = scores
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs):
        self.calls.append([list(pair) for pair in pairs])
        return self._scores


def cross_encoder(scores, **kwargs) -> CrossEncoderReranker:
    return CrossEncoderReranker("cross-encoder/ms-marco", model=FakeCrossEncoder(scores), **kwargs)


# =========================================================================== #
# Observable 1: score ordering is correct
# =========================================================================== #


def test_the_relevant_candidate_ranks_first(candidates):
    out = RelevanceRanker().rerank("python exception", candidates, 3)
    assert out[0].chunk_id in ("c0", "c2")
    assert out[-1].chunk_id == "c1"  # pasta covers no query term


def test_an_irrelevant_candidate_scores_zero(candidates):
    out = RelevanceRanker().score("xylophone", candidates)
    assert all(d.score == 0.0 for d in out)


def test_scores_are_bounded_to_the_unit_interval(candidates):
    for document in RelevanceRanker().score("python exception handling salt", candidates):
        assert 0.0 <= document.score <= 1.0


def test_a_document_covering_every_query_term_scores_one():
    out = RelevanceRanker().score(
        "python exception", [scored("c0", "a python exception and another python")]
    )
    assert out[0].score == pytest.approx(1.0)


def test_coverage_is_the_fraction_of_query_weight_covered():
    """A candidate covering half the query's IDF weight scores about half."""
    docs = [
        scored("c0", "python alpha", chunk_index=0),
        scored("c1", "python beta", chunk_index=1),
        scored("c2", "gamma beta", chunk_index=2),
    ]
    out = {d.chunk_id: d.score for d in RelevanceRanker().score("python beta", docs)}
    assert out["c0"] == pytest.approx(0.5)  # covers python only
    assert out["c1"] == pytest.approx(1.0)  # covers both
    assert out["c2"] == pytest.approx(0.5)  # covers beta only
    assert out["c0"] == out["c2"]


def test_a_rare_term_outranks_a_universal_one():
    """IDF over the candidate set: rarity is what separates candidates.

    'shared' appears in two of three candidates, 'unique' in one, so 'unique'
    carries more weight and a passage containing only it wins. The two
    'shared'-only passages tie and break on chunk_index.
    """
    docs = [
        scored("c0", "shared", chunk_index=0),
        scored("c1", "unique", chunk_index=1),
        scored("c2", "shared", chunk_index=2),
    ]
    ranked = RelevanceRanker().rerank("shared unique", docs, 3)
    assert [d.chunk_id for d in ranked] == ["c1", "c0", "c2"]
    assert ranked[0].score > ranked[1].score == ranked[2].score


def test_the_cross_encoder_orders_by_the_model_score(candidates):
    out = cross_encoder([0.9, 0.1, 0.5]).rerank("q", candidates, 3)
    assert [d.chunk_id for d in out] == ["c0", "c2", "c1"]
    assert [d.score for d in out] == [0.9, 0.5, 0.1]


def test_the_cross_encoder_scores_every_candidate_then_truncates(candidates):
    """Scoring a pre-truncated list would bake the first retriever in."""
    fake = FakeCrossEncoder([0.1, 0.9, 0.5])
    CrossEncoderReranker("m", model=fake).rerank("q", candidates, 1)
    assert len(fake.calls[0]) == 3  # all three were scored, not one


def test_the_query_is_paired_with_each_document_text_in_order(candidates):
    fake = FakeCrossEncoder([0.1, 0.2, 0.3])
    CrossEncoderReranker("m", model=fake).rerank("how do I", candidates, 3)
    pairs = fake.calls[0]
    assert [p[0] for p in pairs] == ["how do I"] * 3
    assert [p[1] for p in pairs] == [c.text for c in candidates]


def test_reranking_replaces_the_score_and_keeps_the_document(candidates):
    out = cross_encoder([0.9, 0.1, 0.5]).rerank("q", candidates, 3)
    by_id = {d.chunk_id: d for d in out}
    assert by_id["c0"].text == "Python error handling with exceptions"
    assert by_id["c0"].score == 0.9


# =========================================================================== #
# Observable 2: ties resolve deterministically
# =========================================================================== #


def test_a_tied_relevance_score_breaks_on_chunk_index():
    tied = [
        scored("z", "identical words", chunk_index=1),
        scored("a", "identical words", chunk_index=0),
    ]
    assert [d.chunk_id for d in RelevanceRanker().rerank("identical words", tied, 2)] == ["a", "z"]


def test_a_tied_model_score_breaks_on_chunk_index(candidates):
    out = cross_encoder([0.5, 0.5, 0.5]).rerank("q", candidates, 3)
    assert [d.chunk_id for d in out] == ["c0", "c1", "c2"]


def test_a_tie_never_breaks_on_input_order(candidates):
    """Permuting the input must not permute a tied ranking."""
    forward = [d.chunk_id for d in cross_encoder([0.5, 0.5, 0.5]).rerank("q", candidates, 3)]
    backward = [d.chunk_id for d in cross_encoder([0.5, 0.5, 0.5]).rerank("q", list(reversed(candidates)), 3)]
    assert forward == backward == ["c0", "c1", "c2"]


def test_the_same_query_returns_the_same_ranking_every_time(candidates):
    ranker = RelevanceRanker()
    first = [d.chunk_id for d in ranker.rerank("python exception", candidates, 3)]
    for _ in range(10):
        assert [d.chunk_id for d in ranker.rerank("python exception", candidates, 3)] == first


def test_relevance_ranking_is_independent_of_input_order(candidates):
    ranker = RelevanceRanker()
    assert [d.chunk_id for d in ranker.rerank("python exception", candidates, 3)] == [
        d.chunk_id for d in ranker.rerank("python exception", list(reversed(candidates)), 3)
    ]


def test_a_tie_breaks_on_chunk_id_when_indices_also_match():
    tied = [
        scored("b", "same", chunk_index=0),
        scored("a", "same", chunk_index=0),
    ]
    assert [d.chunk_id for d in RelevanceRanker().rerank("same", tied, 2)] == ["a", "b"]


def test_scoring_preserves_input_order(candidates):
    """score() scores; it does not reorder. Ordering belongs to rerank()."""
    assert [d.chunk_id for d in RelevanceRanker().score("python", candidates)] == ["c0", "c1", "c2"]


# =========================================================================== #
# Observable 3: a missing cross-encoder library raises ImportError naming it
# =========================================================================== #


@pytest.fixture
def manifest():
    return json.loads((RERANKING / "cross_encoder" / "manifest.json").read_text(encoding="utf-8"))


def test_a_missing_wheel_raises_import_error(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        CrossEncoderReranker("cross-encoder/ms-marco").rerank("q", candidates, 2)


def test_the_missing_dependency_error_is_an_import_error(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert isinstance(exc.value, ImportError)


def test_the_message_names_the_package(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert "sentence-transformers" in str(exc.value)


def test_the_message_is_the_manifests_verbatim(candidates, manifest, monkeypatch):
    """A paraphrase drifts; this is copied from the manifest."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert str(exc.value) == MANIFEST_MESSAGE
    assert str(exc.value) == manifest["missing_dependency_message"]


def test_the_missing_dependency_error_keeps_its_cause(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert isinstance(exc.value.__cause__, ImportError)


def test_the_wheel_is_never_imported_at_module_top_level():
    """DECISION-bf2566ea: heavy libraries are never top-level imports."""
    module = sys.modules[CrossEncoderReranker.__module__]
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:  # direct children of the module only
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "sentence_transformers" not in top_level


def test_the_module_imports_with_the_wheel_absent(monkeypatch):
    """Importing must not require the dependency at all."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert hasattr(sys.modules[CrossEncoderReranker.__module__], "CrossEncoderReranker")


def test_the_reranker_constructs_with_the_wheel_absent(monkeypatch):
    """A caller may register the strategy before deciding to run it."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker("cross-encoder/ms-marco")
    assert reranker.is_loaded() is False


def test_loading_is_deferred_until_the_first_rerank(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker("m")
    assert reranker.is_loaded() is False  # no import attempted yet
    with pytest.raises(ImportError):
        reranker.rerank("q", candidates, 1)


def test_the_manifest_declares_the_optional_dependency(manifest):
    assert manifest["optional_packages"] == [
        {"name": "sentence-transformers", "specifier": ">=2.2.0"}
    ]
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == ["core-types"]


# =========================================================================== #
# Both components honour RerankerProtocol
# =========================================================================== #


def test_the_relevance_ranker_satisfies_the_protocol(candidates):
    assert isinstance(RelevanceRanker(), core.RerankerProtocol)


def test_the_cross_encoder_reranker_satisfies_the_protocol(candidates):
    assert isinstance(cross_encoder([0.1, 0.2, 0.3]), core.RerankerProtocol)


def test_the_protocol_signature_is_the_documented_one():
    import inspect

    for reranker in (RelevanceRanker(), cross_encoder([0.1, 0.2, 0.3])):
        signature = inspect.signature(reranker.rerank)
        assert list(signature.parameters) == ["query", "documents", "top_k"]


# =========================================================================== #
# Requesting more results than exist
# =========================================================================== #


def test_relevance_returning_more_than_candidates_returns_what_exists(candidates):
    assert len(RelevanceRanker().rerank("python", candidates, 100)) == 3


def test_cross_encoder_returning_more_than_candidates_returns_what_exists(candidates):
    assert len(cross_encoder([0.1, 0.2, 0.3]).rerank("q", candidates, 100)) == 3


def test_an_empty_candidate_list_is_not_an_error(candidates):
    assert RelevanceRanker().rerank("python", [], 3) == []
    assert cross_encoder([]).rerank("q", [], 3) == []


def test_a_one_candidate_list_returns_it(candidates):
    one = candidates[:1]
    assert [d.chunk_id for d in RelevanceRanker().rerank("python", one, 5)] == ["c0"]
    assert [d.chunk_id for d in cross_encoder([0.7]).rerank("q", one, 5)] == ["c0"]


# =========================================================================== #
# Failure paths -- relevance
# =========================================================================== #


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(query, candidates):
    with pytest.raises(RankingInputError) as exc:
        RelevanceRanker().rerank(query, candidates, 1)
    assert "relevance-ranking" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"]])
def test_a_non_string_query_is_rejected(query, candidates):
    with pytest.raises(RankingInputError):
        RelevanceRanker().rerank(query, candidates, 1)


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_is_rejected(top_k, candidates):
    with pytest.raises(RankingInputError) as exc:
        RelevanceRanker().rerank("q", candidates, top_k)
    assert "at least 1" in str(exc.value)


@pytest.mark.parametrize("top_k", [True, 1.5, "3", None])
def test_a_non_int_top_k_is_rejected(top_k, candidates):
    with pytest.raises(RankingInputError):
        RelevanceRanker().rerank("q", candidates, top_k)


def test_a_non_document_in_the_list_is_rejected_by_position(candidates):
    with pytest.raises(RankingInputError) as exc:
        RelevanceRanker().rerank("q", candidates + ["nope"], 3)
    assert "[3]" in str(exc.value)


@pytest.mark.parametrize("bad", ["list", 3, None, {"a": 1}])
def test_a_non_sequence_of_documents_is_rejected(bad):
    with pytest.raises(RankingInputError):
        RelevanceRanker().rerank("q", bad, 1)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(RankingInputError):
        RelevanceRanker(name=name)


def test_tokenize_rejects_a_non_string():
    with pytest.raises(RankingInputError):
        tokenize(3)


# =========================================================================== #
# Failure paths -- cross-encoder
# =========================================================================== #


@pytest.mark.parametrize("model_name", ["", "   ", None, 5])
def test_an_unusable_model_name_is_rejected(model_name):
    """No silent default: picking a model is the caller's decision."""
    with pytest.raises(RerankingInputError):
        CrossEncoderReranker(model_name)


@pytest.mark.parametrize("device", [3, ["cpu"], {}])
def test_an_unusable_device_is_rejected(device):
    with pytest.raises(RerankingInputError):
        CrossEncoderReranker("m", device=device)


@pytest.mark.parametrize("name", ["", 5])
def test_an_explicit_unusable_name_is_rejected(name):
    with pytest.raises(RerankingInputError):
        CrossEncoderReranker("m", model=FakeCrossEncoder([]), name=name)


def test_name_none_derives_the_default_rather_than_failing():
    """`None` is the documented 'not specified' value, not an invalid one."""
    derived = CrossEncoderReranker("m", model=FakeCrossEncoder([]), name=None)
    assert derived.name == "cross-encoder:m"


@pytest.mark.parametrize("query", ["", "  ", None, 3])
def test_an_unusable_query_is_rejected(candidates, query):
    with pytest.raises(RerankingInputError):
        cross_encoder([0.1, 0.2, 0.3]).rerank(query, candidates, 1)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "3"])
def test_an_unusable_top_k_is_rejected(candidates, top_k):
    with pytest.raises(RerankingInputError):
        cross_encoder([0.1, 0.2, 0.3]).rerank("q", candidates, top_k)


def test_a_non_document_in_the_list_is_rejected_by_position(candidates):
    with pytest.raises(RerankingInputError) as exc:
        cross_encoder([0.1, 0.2, 0.3, 0.4]).rerank("q", candidates + ["nope"], 4)
    assert "[3]" in str(exc.value)


def test_a_model_returning_the_wrong_number_of_scores_is_rejected(candidates):
    """Otherwise candidates would be dropped from the ranking silently."""
    with pytest.raises(RerankingError) as exc:
        cross_encoder([0.5]).rerank("q", candidates, 3)
    assert "1 scores for 3 document(s)" in str(exc.value)


def test_a_model_returning_nothing_is_rejected(candidates):
    with pytest.raises(RerankingError):
        cross_encoder([]).rerank("q", candidates, 3)


def test_a_broken_model_is_wrapped_with_the_model_name(candidates):
    class Broken:
        def predict(self, pairs):
            raise RuntimeError("cuda out of memory")

    with pytest.raises(RerankingError) as exc:
        CrossEncoderReranker("ms-marco-MiniLM", model=Broken()).rerank("q", candidates, 3)
    assert "ms-marco-MiniLM" in str(exc.value)
    assert "cuda out of memory" in str(exc.value)


def test_a_broken_model_keeps_its_cause(candidates):
    class Broken:
        def predict(self, pairs):
            raise RuntimeError("nope")

    with pytest.raises(RerankingError) as exc:
        CrossEncoderReranker("m", model=Broken()).rerank("q", candidates, 3)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_model_returning_a_non_iterable_is_wrapped(candidates):
    class Bad:
        def predict(self, pairs):
            return "not scores"

    with pytest.raises(RerankingError) as exc:
        CrossEncoderReranker("m", model=Bad()).rerank("q", candidates, 3)
    assert "does not yield one number per pair" in str(exc.value)


# =========================================================================== #
# Composition with the rest of the pipeline
# =========================================================================== #


def test_scoring_does_not_mutate_the_callers_documents(candidates):
    before = [d.score for d in candidates]
    RelevanceRanker().score("python", candidates)
    assert [d.score for d in candidates] == before  # frozen, so replaced


def test_the_results_are_scored_documents(candidates):
    for item in RelevanceRanker().rerank("python", candidates, 2):
        assert isinstance(item, core.ScoredDocument)
    for item in cross_encoder([0.1, 0.2, 0.3]).rerank("q", candidates, 2):
        assert isinstance(item, core.ScoredDocument)


def test_stopwords_are_dropped_from_the_query():
    assert "the" not in tokenize("the python and the snake")
    assert DEFAULT_STOPWORDS and "the" in DEFAULT_STOPWORDS


def test_stopwords_can_be_overridden():
    assert tokenize("the python", stopwords=frozenset()) == frozenset({"the", "python"})


def test_no_stemming_a_query_term_does_not_match_its_plural():
    """Documented limitation, pinned so it cannot be mistaken for a bug."""
    docs = [scored("c0", "handling exceptions gracefully")]
    out = RelevanceRanker().score("exception", docs)
    assert out[0].score == 0.0  # 'exception' never matches 'exceptions'


def test_a_tokenizer_match_is_a_whole_word_not_a_substring():
    docs = [scored("c0", "pythonic code")]
    assert RelevanceRanker().score("python", docs)[0].score == 0.0


def test_the_relevance_errors_form_a_hierarchy():
    assert issubclass(RankingInputError, RankingError)
    assert not issubclass(RankingError, ValueError)


def test_the_cross_encoder_errors_form_a_hierarchy():
    assert issubclass(RerankingInputError, RerankingError)
    assert not issubclass(RerankingError, ValueError)
    assert not issubclass(MissingDependency, RerankingInputError)


def test_the_manifests_declare_the_core_types_dependency():
    for directory in ("relevance_ranking", "cross_encoder"):
        manifest = json.loads((RERANKING / directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dependencies"] == ["core-types"], directory
        assert manifest["category"] == "reranking", directory


def test_the_relevance_manifest_declares_no_optional_packages():
    manifest = json.loads((RERANKING / "relevance_ranking" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []


def test_no_reranking_component_imports_the_other():
    """Each is copy-ready on its own."""
    module_names = ("relevance_ranking", "cross_encoder_reranker")
    for name in module_names:
        module = sys.modules[name]
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for other in module_names:
            assert other not in imported, f"{name} imports {other}"


def test_both_rerankers_compose_in_one_pipeline(candidates):
    """Lexical first to cut the list, then the model to order what remains."""
    lexical = RelevanceRanker().rerank("python exception", candidates, 2)
    assert len(lexical) == 2
    modelled = cross_encoder([0.2, 0.8]).rerank("python exception", lexical, 2)
    assert modelled[0].chunk_id in ("c0", "c2")
