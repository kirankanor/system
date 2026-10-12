# Product specification — RAG Code Library

> Status: draft. A coding agent must not implement product code until this specification is approved through Genesis.

## Problem

Teams and individuals building RAG systems re-write the same plumbing — loaders, chunkers, embedders, stores, retrievers, rerankers, context builders — for every project, usually as untested glue copied between repos. The result is inconsistent quality, no regression safety, and no way to compare strategies.

This project delivers a **reusable library of production-oriented RAG components**: each component ships as copy-ready Python with documentation, an example, tests, and an explicit statement of when to use it and what it costs. Components are discovered and installed into a target project through a `ragkit` CLI and browsed through a local cookbook website.

Desired outcome: the human can start a new RAG project, browse the catalog, and `ragkit add` a tested, documented, framework-neutral component into their own package in seconds — with the same metadata driving the CLI, the tests, and the website, so nothing drifts.

## Users

- **Primary (v1): the repository owner.** An experienced Python engineer running locally on Windows with `uv`, VS Code, and Python 3.12+. Consumes the library from their own projects, extends it, and reviews every change. Not a consumer of a hosted service; expects no accounts, no network dependency, and no telemetry.
- **Secondary: the human's future project code.** Downstream code imports added components directly. It is a consumer with a right to expect stable, typed, documented interfaces and predictable errors.
- **Affected third parties: none.** No public users, no data subjects, no tenants in v1.

## Definitions

- **Implemented:** a source file contains real executable logic (not a comment stub), imports resolve, and it is covered by a manifest.
- **Tested:** a test exercises at least one success path and at least one failure path, and passes in an executed run.
- **Verified:** an executable gate ran in the current session against the current source hash and produced a passing, non-stale, non-invalid proof.
- **Full test suite:** `python -m pytest` from the repository root, with the committed `pyproject.toml` `[tool.pytest.ini_options]` (`testpaths = ["tests", "components"]`, `--import-mode=importlib`), collecting every test module under both roots.
- **Failed:** a check or test that ran and did not pass.
- **Not tested:** a requirement with no executed evidence; never reported as passing and never reported as failed.
- **Blocked:** a check that could not run for a recorded reason (missing authorization, missing dependency, environment limitation).
- **Production-grade:** standalone, reusable, validated, testable code with explicit inputs, stable outputs, controlled failures, and minimal unnecessary coupling.

## Functional requirements

### Packaging and entry points

- FR-1: The project installs into a `uv`-managed environment as a Python package exposing the `ragkit` console entry point, importable as `rag_code_library`.
- FR-2: `ragkit` requires Python 3.12 or newer and declares its dependencies explicitly in `pyproject.toml`; a `uv.lock` generated from that declaration is committed and valid.
- FR-3: The runtime core depends only on `pydantic` and `pydantic-settings`, and no core module imports an optional component library. A component needing a heavier third-party library declares it as an optional package in its manifest and raises, at first use, an `ImportError` whose message contains: the missing distribution name, the component id that needs it, and the exact installation command (`uv add <distribution>`). The message is asserted verbatim by a test.

### Component library

- FR-4: Every component in the v1 core pipeline path has a real implementation — not a stub — with type hints, docstrings, input validation, explicit typed errors, and framework-neutral contracts: chunking (fixed-size, recursive, token-based, markdown-aware, structure-aware), embeddings (protocol, local, remote, batch), storage (document store, vector-store protocol, adapters, core types), retrieval (dense, sparse/BM25, hybrid, metadata filter, result normalizer), reranking (cross-encoder, relevance), context (token budget, context selector, citation formatter), generation (prompt builder, grounded-answer contract), security (input validation, metadata validation, retrieved-content safety), ingestion (text normalizer, PDF extraction sub-components), and evaluation (retrieval metrics, answer metrics, evaluation dataset).
- FR-5: Components in an area that share a contract implement a common protocol (chunking, embedding, vector-store, document-store, reranker, context-builder, metric) so they are interchangeable without changing caller code.
- FR-6: An area may be split into selectable sub-components when implementations differ by library or technique — for example PDF extraction split into `pymupdf` and `ocr` sub-components. Each sub-component carries its own manifest, README, example, and tests, and the human selects which one to add.
- FR-7: Every chunker documents and enforces its overlap semantics; overlap larger than chunk size is rejected with a typed error rather than silently clamped.
- FR-8: Retrieval, reranking, and context building operate on a documented, provider-neutral result type carrying text, score, source id, and metadata, so components compose without adapters.
- FR-9: Evaluation components provide retrieval metrics (at least hit rate, MRR, recall@k), answer metrics (at least faithfulness/groundedness and answer relevance computed from provided evidence), and a dataset format with regression-test support.

### Manifests, registry, and schema

- FR-10: Every component has a canonical `manifest.json` validated against `schemas/component-manifest.schema.json`; the schema's category enum covers all areas in use, including `config`, `generation`, `observability`, and `integrations`.
- FR-11: A single registry loader discovers manifests, resolves ids, and reports duplicates, missing files, and schema violations with actionable messages rather than tracebacks.
- FR-12: Component metadata (id, name, category, description, dependencies, destination, docs, tests) is read from manifests by the CLI, the tests, and the cookbook generator. No component definition is maintained in more than one place.

### ragkit CLI

- FR-13: `ragkit list` lists components, optionally filtered by category, and reports each component's id, category, and one-line description.
- FR-14: `ragkit info <id>` shows full details: description, dependencies, requirements, example, usage notes, and trade-offs.
- FR-15: `ragkit add <id>` copies a component's source, docs, tests, and manifest into a destination project, creating required directories and writing only inside the resolved destination. It **never** installs packages and **never** modifies the destination's `pyproject.toml`, `uv.lock`, `requirements*.txt`, or any other dependency file. After copying it prints the component's required and optional packages with the commands to install them, plus any external service or API key the component expects.
- FR-16: `ragkit add` supports `--dry-run` printing the exact file operations that would occur, and performs no writes.
- FR-17: `ragkit add` refuses unsafe destinations — path traversal, escaping the project root, or symlinks resolving outside it — with a typed error, and never overwrites an existing file without an explicit `--force`.
- FR-18: Conflicts are reported clearly: which file exists, which component owns it, and what the human can do (choose another destination, pass `--force`, or remove the file).
- FR-19: The CLI exits 0 on success, 1 on a handled domain error, and 2 on misuse of arguments; error output names the failing component or path and never prints a raw traceback for expected failures.

### Cookbook website

- FR-20: `cookbook/` is a local static site (HTML, CSS, JavaScript) with no backend, opened directly from disk: top navigation, a grouped collapsible sidebar, a home page, one section per pipeline area, and a page or view per component.
- FR-21: Each component view shows description, code example, dependencies, use cases, and trade-offs, plus a working copy-code button.
- FR-22: The site supports search and filtering across components, and shows breadcrumbs and clear navigation between sections.
- FR-23: `scripts/build_cookbook.py` generates the site's component data from canonical manifests; regenerating against unchanged manifests produces identical output, and the generated data is committed so the site works without a build step at view time.

### Supporting tooling and docs

- FR-24: `scripts/validate_components.py` validates every manifest against the schema and fails with a non-zero exit listing each invalid component and reason; it is runnable standalone and from CI.
- FR-25: `scripts/verify_generated_code.py` verifies that files written by `ragkit add` match their source components byte-for-byte.
- FR-26: The root README documents setup, commands, examples, architecture, and limitations, and each component's README documents its purpose, API, example, dependencies, and trade-offs.
- FR-27: `docs/` covers architecture, CLI reference, component authoring, security, versioning, and troubleshooting, consistent with the shipped behavior.

### Dependencies, transactions, and lifecycle (added in review)

- FR-28: The manifest schema records, per component: required packages with version constraints, optional packages with version constraints, minimum and maximum supported Python versions, external services and the environment variables they need, human-readable installation instructions, the dependency-missing error message text, and a lifecycle status.
- FR-29: `dependencies` in a manifest refers to other *component ids* and is resolved transitively by `ragkit add`; package names are never placed in that field. `ragkit add` resolves and copies required component dependencies before the requested component, and reports dependency cycles, missing dependencies, and version conflicts clearly.
- FR-30: Package metadata (`pyproject.toml`), manifests, generated cookbook data, and dependency validation agree on the same distribution names and version constraints; a check fails on any disagreement.
- FR-31: The dry-run plan and the real execution of `ragkit add` are produced by one planner, so the operations printed by `--dry-run` are byte-identical to the operations the real run performs, verified by a test that diffs both.
- FR-32: `ragkit add` is transactional per file set: on any failure it reports which files were written, which were not, and offers the exact commands to remove the partial result. It never leaves an unreported partial state.
- FR-33: `--force` permits overwriting files inside the destination root only. It can never widen the authorized root, and it never converts a rejected unsafe path into a permitted one.
- FR-34: Hostile-path tests run against the real Windows filesystem of this host, creating actual symlinks and junctions where the platform permits, and skipping with an explicit recorded reason where it does not.
- FR-35: `ragkit list` and `ragkit info` display each component's lifecycle status (`stable`, `v2-deferred`, `experimental`), and `ragkit add` refuses `v2-deferred` components with a message naming the deferral.

## Non-functional requirements

- NFR-1: **Test collection succeeds.** `python -m pytest` collects and runs the full suite with no collection errors, no import errors, and no unexplained skips; the duplicate `tests/test_component.py` basename collision is resolved by configuration, not by weakening collection scope.
- NFR-2: **Honest results.** No test is reported as passing unless it was executed in the same run against the current source. Every verification report states, as separate lines, at least: interpreter and version, exit code, collected count, passed, failed, collection errors, skipped (each with its reason), xfail, and deselected. "Passing" means exit code 0 with zero failures and zero collection errors; a non-zero exit, a collection error, or a missing command is reported as a failure, never as an incomplete result. A requirement that was never executed is reported as **not tested**, which is distinct from **failed**.
- NFR-3: **Path safety.** Every filesystem write reachable from the CLI goes through a single path-safety module. No CLI command, component, or script may open a file for writing except through it. The module must reject, with a typed error and zero writes: relative traversal (`..`) at any depth; absolute paths outside the resolved destination root; drive-letter paths (`C:\`, `c:/`); drive-relative paths (`C:foo`); UNC paths (`\\server\share`, `\\?\UNC\...`); extended-length prefixes (`\\?\`); alternate data streams (`file.py:stream`); reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`) in any path segment; symlinks, junctions, and other reparse points resolving outside the destination root; and target paths exceeding the platform maximum unless written through a supported long-path form. Containment is decided on the *resolved real* path, and the destination root itself is resolved once and cached for the operation. Names are compared case-insensitively on Windows so a case-variant of an existing file is treated as a conflict. Known residual risk: TOCTOU substitution of a reparse point between resolution and write is mitigated (resolve immediately before each write, verify the parent chain, abort on any mismatch) but is **not** claimed as prevented.
- NFR-4: **No undeclared dependencies.** Every import of a third-party package in shipped code is declared in `pyproject.toml` or is a sub-component's guarded optional dependency. A check verifies this.
- NFR-5: **Python 3.12 compatibility.** The suite passes on Python 3.12. Until the suite is executed on Python 3.12, compatibility remains explicitly unverified regardless of the `requires-python` floor.
- NFR-6: **No drift.** The cookbook data, component tests, and CLI listings are all derived from manifests; a check fails if any of them disagrees with the canonical source.
- NFR-7: **Readable errors.** All user-facing errors name the component, path, or setting at fault and the corrective action; expected failures never surface raw tracebacks.
- NFR-8: **No fabricated completeness.** Every claimed feature has real implementation, documentation, or an explicitly recorded status. No empty placeholder files are created to make the tree look complete.
- NFR-9: **Local-first, provider opt-in.** The library, CLI, and cookbook require no network access, hosted backend, account, or telemetry by default. A component calls an external provider only when the human explicitly configures provider, model, and credentials; no provider, model, or endpoint has a default value. Selecting such a component is a deliberate act recorded in the destination project's configuration.
- NFR-10: **Secrets hygiene.** API keys are represented as `SecretStr`, never appear in logs, dumps, or error details, and are never written into the cookbook or generated data.
- NFR-11: **Reproducibility.** Given the committed `uv.lock`, a fresh `uv sync` reproduces an environment in which the full suite passes.
- NFR-12: **Bounded execution.** Exactly one task is active at a time. A task cannot reach `done` while any of its mandatory gate proofs is pending, failed, missing, or stale against the current source hash. Dependent tasks cannot activate before all their dependencies are `done`.
- NFR-13: **Deferred-surface discipline.** Components belonging to v2-deferred categories may be declared in manifests and listed by `ragkit list` with status `v2-deferred`, but must ship no implementation file, no example, and no test, and `ragkit add` must refuse them with a message naming the deferral.

## Constraints

- Python 3.12+; `uv` for environment and dependency management; VS Code as the editor.
- Local Windows host; the cookbook website is opened from disk, not served.
- Package name `rag-code-library`, import name `rag_code_library`, CLI name `ragkit`.
- Runtime core limited to `pydantic` and `pydantic-settings`; heavier libraries live in sub-components chosen at `ragkit add` time. New manifest dependency fields are required, not defaulted.
- Framework-neutral component contracts: no component may require LangChain, LlamaIndex, or any orchestration framework.
- License: as declared in `LICENSE` (MIT).
- Git is the only VCS; no commits, pushes, dependency installations, `uv.lock` regeneration, CI modification, or other consequential host change occurs without explicit, task-scoped, expiring human authorization recorded through Genesis.
- Substantive implementation-code ownership is whatever the human last recorded; the Genesis decision record is authoritative and supersedes prose in any other document. Current rule: **the agent authors production-grade implementation code; the human reviews and requires independent verification** (DECISION-0410c7dc, superseding DECISION-ba2975b0).
- Task dependencies are expressed only through Genesis `--depends` edges and are enforced at runtime; stage boundaries are never inferred from task ids or prose ordering. `genesis plan check` must pass with zero uncovered requirements.
- Every changed file in a task must lie within that task's declared `--scope`; scope violations block completion.

## Non-goals

- Publishing to PyPI, or any public API-stability or semver commitment.
- A hosted cookbook, a web backend, authentication, or multi-tenancy.
- Observability (structured logging, tracing, token/latency reporting) — deferred to v2; the category is declared with `v2-deferred` status and ships no code.
- Integration adapters for third-party frameworks — deferred to v2; same treatment.
- LLM provider clients — deferred to v2. v1 generation is a prompt builder and a grounded-answer contract only, both network-free.
- Any agent, planner, or autonomous orchestration layer.
- Performance benchmarking or a published benchmark suite.
- Supporting languages other than Python.

## Acceptance criteria

- AC-1: `uv sync --frozen` succeeds and `uv run ragkit --help` exits 0, listing `list`, `info`, and `add`.
- AC-2: `python -m pytest -q` runs the full suite with exit 0, zero collection errors, zero import errors, and zero skips that are not explicitly justified in the report.
- AC-3: `uv run --python 3.12 python -c "import sys; print(sys.version)"` prints a 3.12.x version, and `uv run --python 3.12 python -m pytest -q` exits 0 with zero failures and zero collection errors; `python -m pytest -q` on the host interpreter does the same. Both proof artifacts record the interpreter string. Until the 3.12 command has actually run and produced a passing proof, Python 3.12 compatibility is reported as **unverified**.
- AC-4: Every v1 component in the inventory below has an implementation file whose body is real logic (not a comment stub), a README, an example, and a test that exercises at least one success path and one failure path.
- AC-5: `python scripts/validate_components.py` exits 0 over the full component set, and exits non-zero with a per-component reason when a manifest is deliberately corrupted in a test fixture.
- AC-6: `python scripts/build_cookbook.py --check` exits 0 when the committed `cookbook/data.js` matches regenerated output byte-for-byte (generation sorts components by id and emits no timestamps, so output is deterministic), and exits non-zero naming the differing component when any manifest is changed.
- AC-7: `uv run ragkit list` output contains every component id in the registry, and `uv run ragkit list --category chunking` contains only chunking components.
- AC-8: `uv run ragkit info <id>` for any registered component prints description, dependencies, example, and usage notes.
- AC-9: `uv run ragkit add <id> --dry-run` writes nothing (verified by hashing the destination tree before and after) and prints every file operation the real run would perform.
- AC-10: A real `uv run ragkit add <id>` into a fixture project produces byte-identical copies of source, docs, and tests, verified by `python scripts/verify_generated_code.py`.
- AC-11: `python -m pytest tests/unit/test_path_safety.py -q` runs over hostile Windows fixtures: destination containing `../`, an absolute path outside the project, a drive-letter path, a UNC path, and a symlink and a junction pointing outside the project each fail with a typed path-safety error and write nothing.
- AC-12: A second `ragkit add` of the same component fails with a conflict error naming the existing file; with `--force` it succeeds, and `--force` still fails for a path outside the destination root.
- AC-13: Hostile fixtures in `tests/fixtures/invalid_manifests/` (missing required field, wrong type, invalid category, bad id pattern, bad version, extra property, path-traversal destination) are each rejected with a message naming the offending field.
- AC-14: `python scripts/validate_dependencies.py` exits 0; it walks every shipped `.py` file, extracts third-party imports, and fails listing each import that is neither declared in `pyproject.toml` (core) nor listed as an optional package in its component manifest with a matching guarded-import pattern.
- AC-15: **Manual, scripted-assist.** A script compares the component count per section in `cookbook/data.js` against the registry and exits 0 on match. A human then opens `cookbook/index.html` from disk and records a pass/fail against a fixed checklist: sidebar groups expand and collapse; each section lists the expected components; each component view shows description, code example, dependencies, use cases, trade-offs; the copy-code button places the example on the clipboard; search and filter narrow the list correctly; breadcrumbs and navigation reach every section. This criterion is verified by recorded human inspection, never by a claim of automated proof.
- AC-16: `python scripts/check_documented_commands.py` extracts every fenced shell command from `README.md` and `docs/*.md`, executes each in a scratch destination, and exits 0 only if every command succeeds and its documented behavior matches. Commands that cannot run unattended are listed with a recorded manual-check reason.
- AC-17: `python scripts/check_no_secrets.py` exits 0; no `SecretStr` value appears in any log line, error message, generated cookbook file, or committed artifact.
- AC-18: `python scripts/validate_components.py` exits 0 over the full registry, and exits non-zero listing each invalid component when any one manifest is corrupted in a test fixture; every v1 component in the inventory below is present in the registry with a manifest satisfying the schema.
- AC-19: `python scripts/check_deferred_surface.py` exits 0; no file under the v2-deferred categories (`observability`, `integrations`) exists as implementation code at v1 release.
- AC-20: The authoritative decision record on implementation-code ownership matches the responsibility split stated in this specification; any task whose scope is substantive implementation code while the rule forbids agent authorship is marked `blocked` with that exact reason rather than completed or substituted.

## Canonical v1 component inventory

Protocols, defined once in the `core-types` component and reused by every area:

| Protocol | Contract |
|---|---|
| `ChunkingProtocol` | `chunk(text, *, size, overlap, **kw) -> list[Chunk]` |
| `EmbeddingProtocol` | `embed(texts) -> list[list[float]]`, `dimensions: int`, `model_id: str` |
| `VectorStoreProtocol` | `add(...)`, `query(vector, k, filter)`, `delete(ids)`, `count()` |
| `DocumentStoreProtocol` | `put(doc)`, `get(id)`, `list(filter)` |
| `RerankerProtocol` | `rerank(query, documents, top_k) -> list[ScoredDocument]` |
| `ContextBuilderProtocol` | `build(query, chunks, budget) -> Context` |
| `MetricProtocol` | `compute(dataset) -> MetricResult` |

| # | Component ID | Category | Protocol | Source | Optional deps |
|---|---|---|---|---|---|
| 1 | `core-types` | storage | — | `core_types.py` | — |
| 2 | `fixed-size-chunker` | chunking | Chunking | `fixed_size_chunker.py` | — |
| 3 | `recursive-chunker` | chunking | Chunking | `recursive_chunker.py` | — |
| 4 | `token-based-chunker` | chunking | Chunking | `token_chunker.py` | `tiktoken` |
| 5 | `markdown-aware-chunker` | chunking | Chunking | `markdown_chunker.py` | — |
| 6 | `structure-aware-chunker` | chunking | Chunking | `structure_aware_chunker.py` | — |
| 7 | `embedding-protocol` | embeddings | Embedding | `embedding_protocol.py` | — |
| 8 | `local-embeddings` | embeddings | Embedding | `local_embeddings.py` | `sentence-transformers` |
| 9 | `remote-embeddings` | embeddings | Embedding | `remote_embeddings.py` | `httpx` |
| 10 | `batch-embeddings` | embeddings | Embedding | `batch_embeddings.py` | — |
| 11 | `document-store` | storage | DocumentStore | `document_store.py` | — |
| 12 | `vector-store-protocol` | storage | VectorStore | `vector_store_protocol.py` | — |
| 13 | `storage-adapters` | storage | VectorStore | `adapters/` | `chromadb`, `faiss` |
| 14 | `dense-retrieval` | retrieval | VectorStore | `dense_retrieval.py` | — |
| 15 | `sparse-retrieval` | retrieval | — | `sparse_retrieval.py` | `rank-bm25` |
| 16 | `hybrid-retrieval` | retrieval | — | `hybrid_retrieval.py` | — |
| 17 | `metadata-filter` | retrieval | — | `metadata_filter.py` | — |
| 18 | `retrieval-result-normalizer` | retrieval | — | `retrieval_result_normalizer.py` | — |
| 19 | `cross-encoder-reranker` | reranking | Reranker | `cross_encoder_reranker.py` | `sentence-transformers` |
| 20 | `relevance-ranking` | reranking | Reranker | `relevance_ranking.py` | — |
| 21 | `token-budget` | context | ContextBuilder | `token_budget.py` | `tiktoken` |
| 22 | `context-selector` | context | ContextBuilder | `context_selector.py` | — |
| 23 | `citation-formatter` | context | ContextBuilder | `citation_formatter.py` | — |
| 24 | `prompt-builder` | generation | — | `prompt_builder.py` | — |
| 25 | `grounded-answer-contract` | generation | — | `grounded_answer.py` | — |
| 26 | `input-validation` | security | — | `input_validation.py` | — |
| 27 | `metadata-validation` | security | — | `metadata_validation.py` | — |
| 28 | `retrieved-content-safety` | security | — | `retrieved_content_safety.py` | — |
| 29 | `text-normalizer` | ingestion | — | `text_normalizer.py` | — |
| 30 | `pdf-extraction-pymupdf` | ingestion | — | `pdf_extraction/pymupdf.py` | `pymupdf` |
| 31 | `pdf-extraction-ocr` | ingestion | — | `pdf_extraction/ocr.py` | `pytesseract`, `pillow` |
| 32 | `retrieval-metrics` | evaluation | Metric | `retrieval_metrics.py` | — |
| 33 | `answer-metrics` | evaluation | Metric | `answer_metrics.py` | — |
| 34 | `evaluation-dataset` | evaluation | Metric | `evaluation_dataset.py` | — |
| 35 | `settings` | config | — | (already implemented) | — |

Every component additionally requires: `manifest.json` (schema-valid), `README.md` (purpose, API, example, dependencies, trade-offs, when-not-to-use), `example.py` runnable standalone, `tests/test_component.py` with success and failure paths.

Each component must also state its verification command; scoped `pytest` runs per area are the default.

**v2-deferred categories (declared, no code):** `observability` (0 components), `integrations` (0 components). **v2-deferred within v1 categories:** LLM provider clients under `generation`, document loaders under `ingestion` beyond PDF and text.

## Risks

- **Scope risk:** v1 spans 35 components across 11 areas. Mitigation: requirement-linked bounded tasks with executable gates, implemented and proven area by area rather than in bulk.
- **Honest-verification risk:** claiming untested compatibility or passing tests. Mitigation: NFR-2, per-interpreter runs, and Genesis proof bound to the current source hash.
- **Dependency drift risk:** components silently requiring heavy libraries. Mitigation: FR-3, FR-28 to FR-30, NFR-4, and AC-14.
- **Safety risk:** `ragkit add` writing outside the destination project. Mitigation: single path-safety module, NFR-3, hostile-path fixtures, AC-11.
- **Metadata drift risk:** CLI, tests, and website disagreeing about a component. Mitigation: single canonical manifest source, NFR-6, AC-5, AC-6.
- **Windows-specific risk:** path and symlink behavior differing from POSIX. Mitigation: path-safety tests exercised on the actual host OS, per FR-34.
- **Tooling risk:** the Genesis CLI on `PATH` may lag the kit source, silently dropping unsupported flags. Mitigation: verify CLI behavior before relying on a flag; treat a silently accepted unknown flag as a defect.

## Open questions

- None blocking. All eight review questions (U1 ownership, U2 core types, U3 manifest migration, U4 generation scope, U5 deferred categories, U6 `.gitignore`, U7 `uv.lock` authorization, U8 task granularity) are resolved and recorded as decisions.
- Carried forward as assumptions to confirm during planning: default embedding/vector-store providers for the shipped examples (kept provider-neutral; no default model is chosen for the human), and whether `structure_aware` chunking should target a specific document format beyond Markdown.
