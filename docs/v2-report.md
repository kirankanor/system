# V2 completion report

What V2 set out to do, what actually exists, and what does not. Every one of
the twenty steps in `v2.md` is classified below as **complete**, **partial**,
**missing** or **blocked**, each with file and line evidence.

Classifications follow `v2.md` §2 exactly:

- **complete** — implementation *and* acceptance tests satisfy the requirements.
- **partial** — implementation exists but behaviour, edge cases or tests are missing.
- **missing** — no suitable implementation exists.
- **blocked** — a required external service, unresolved interface or product decision prevents verification.

NFR-18 applies: no step is called complete because a class, file or placeholder
exists. Where a step is complete, the evidence names the behaviour a test
exercises, not just the file that contains it. Where it is not, the gap is
stated rather than argued away.

Verified on `.venv` CPython 3.12.14.

## Summary

| # | Step (`v2.md` §1) | Classification |
|---|---|---|
| 1 | Document Discovery | complete |
| 2 | File Validation | complete |
| 3 | Document Identification | complete |
| 4 | Document Loading | **partial** |
| 5 | Text Extraction | **partial** |
| 6 | Text Normalization | **partial** |
| 7 | Metadata Enrichment | **partial** |
| 8 | Chunking | complete |
| 9 | Chunk Validation | complete |
| 10 | Chunk Deduplication | **partial** |
| 11 | Embedding | complete |
| 12 | Indexing | **partial** |
| 13 | Index Verification | complete |
| 14 | Failure Handling | complete |
| 15 | Idempotency | complete |
| 16 | Versioning | **partial** |
| 17 | Observability | **partial** |
| 18 | Testing | complete |
| 19 | Security | **partial** |
| 20 | Lifecycle | **partial** |

**Tally: 10 complete, 10 partial, 0 missing, 0 blocked.**

Nothing is classified *blocked* as a whole step, because every step has a
working and tested implementation. Two steps do contain a blocked *component*
— PDF extraction and the two uninstalled vector backends — and those blockers
are named in place rather than used to downgrade the whole step.

Steps 16, 17 and 20 were explicitly deferred by `DECISION-1aedada9` ("spine
first") and are reported as partial on their actual merits, not as done.

---

## The twenty steps

### 1. Document Discovery — complete

Evidence:

- `components/ingestion/document_discovery/document_discovery.py` L293 — `Discovery` walks a source root, recursive or not, honouring configured extensions and exclusion patterns, and emits one structured record per candidate.
- `components/ingestion/document_discovery/document_discovery.py` L131 — `EntryState` defines `added`, `modified`, `deleted`, `unchanged` and `unknown`, so a scan is a delta against the previous one rather than a fresh list.
- `components/ingestion/document_discovery/document_discovery.py` L171 — `DiscoveryReport` carries skip counts and reasons: unreadable directories, symlink or junction escape, duplicate paths, empty folders.
- `components/ingestion/document_discovery/tests/test_component.py` — 43 test functions covering deterministic ordering, non-recursive mode, exclusion patterns, Unicode paths, inaccessible directories, and delta reporting including `deleted`.

Gaps named honestly:

- No cloud, S3, GCS or HTTP discovery adapters exist. `v2.md` L53 forbids
  claiming them, and this project is local-only by decision (no hosted
  backend), so they are deliberately absent rather than incomplete.

### 2. File Validation — complete

Evidence:

- `components/ingestion/file_type_validator/file_type_validator.py` L278 — `FileTypeValidator.validate` decides from magic bytes first and consults the extension only as supporting evidence, returning `FileAssessment` (L209) that records which evidence won.
- `components/ingestion/file_type_validator/file_type_validator.py` L62 — `FileKind` distinguishes 17 kinds including `docx`, `xlsx`, `pptx`, `rtf`, archives and images, so an unsupported kind is *refused by name* instead of being opened and mis-parsed.
- `components/ingestion/file_type_validator/file_type_validator.py` L378 — `validate` raises distinct `UnsupportedType`, `FileTooLarge` and `EmptyFile` exceptions, all subclasses of `FileTypeError`, so callers can branch per reason.
- `components/ingestion/file_type_validator/tests/test_component.py` — 23 test functions covering extension/content disagreement, oversized files, empty files and corrupt headers.

Gaps: none for the required capability. The formats `FileKind` recognises
but that have no parser are listed under step 4 and in *Not implemented*.

### 3. Document Identification — complete

Evidence:

- `components/ingestion/document_identity_resolver/document_identity_resolver.py` L139 — `DocumentIdentityResolver` derives identity from content, not from path or filename.
- `components/ingestion/document_identity_resolver/document_identity_resolver.py` L100 — `ContentIdentity` holds the sha256 of the bytes, so a moved or renamed file resolves to the same document.
- `components/ingestion/document_identity_resolver/document_identity_resolver.py` L181 — `resolve` applies the configured rename policy and reports duplicates, preserving the original source path as provenance instead of overwriting it.
- `components/ingestion/document_identity_resolver/tests/test_component.py` — 25 test functions covering rename detection, duplicate content with different paths, and stable ids across rescans.

Gaps: none.

### 4. Document Loading — partial

Evidence that exists:

- `components/ingestion/document_loader/document_loader.py` L429 — `DocumentLoader` dispatches on `detect_format` (L245), which names the evidence it used (`Evidence` L84: bytes, magic number, or declared extension).
- `components/ingestion/document_loader/document_loader.py` L285, L299, L355 — `TextAdapter`, `MarkdownAdapter` and `HtmlAdapter` are the default adapter set (installed at L448–L450) and use only the standard library, so plain text, Markdown and HTML all parse with no optional dependency.
- `components/ingestion/document_loader/document_loader.py` L133, L154 — `Page` and `LoadedDocument` carry page numbers and `source_locator`, so every extracted character remains attributable to an offset in the source.
- `components/ingestion/document_loader/tests/test_component.py` — 45 test functions covering format detection evidence, unreadable and empty documents, adapter injection, and page/offset integrity.

Why it is partial:

- **PDF is not verified end to end.** The seam exists
  (`PdfAdapter` `components/ingestion/document_loader/document_loader.py` L373) and two real extractors are written
  — `components/ingestion/pdf_extraction_pymupdf/pymupdf.py` L150 `PdfExtractor`
  and `components/ingestion/pdf_extraction_ocr/ocr.py` L162 `OcrExtractor` —
  but neither optional dependency is installed, and installing one would
  modify a dependency file, which is forbidden. Both raise a guarded
  `MissingDependency` (`components/ingestion/pdf_extraction_pymupdf/pymupdf.py` L96, `components/ingestion/pdf_extraction_ocr/ocr.py` L106) with the message FR-3
  requires. The PDF path is therefore *blocked by a product decision* inside
  an otherwise partial step.
- **DOCX, XLSX, PPTX, RTF have no parser at all.** They are classified by
  `FileKind` so they are refused cleanly, and never ingested.
- CSV and JSON are detected as `text` (`components/ingestion/document_loader/document_loader.py` L288) and handed
  to `TextAdapter` verbatim. They are not parsed structurally.

### 5. Text Extraction — partial

Evidence that exists:

- `components/ingestion/document_loader/document_loader.py` L133 — `Page` maps every character back to a page number and a `source_locator`, satisfying the page-map requirement.
- `components/chunking/markdown_aware/markdown_chunker.py` L266 — `_attach_headings` carries section headings into chunks, so heading context survives chunking.
- `tests/integration/pipeline_stages.py` — the `extract` stage joins loader output into the document state and the e2e suite asserts offsets round-trip through it.

Why it is partial:

- Tables and lists are not represented structurally anywhere. `HtmlAdapter`
  (`components/ingestion/document_loader/document_loader.py` L355) reduces HTML to flat text with the stdlib
  parser; there is no table/list/section intermediate representation.
- No OCR output path is verified, for the same dependency reason as step 4.
- Partial-failure recording is per document, not per page: a document that
  fails extraction is reported as failed, but a single bad page inside an
  otherwise good document is not isolated and reported separately.

### 6. Text Normalization — partial

Evidence that exists:

- `components/ingestion/text_normalizer/text_normalizer.py` L172 — `TextNormalizer` applies Unicode normalization, line-ending and whitespace folding, returning `NormalizationResult` (L134) so every change is reported rather than silent.
- `components/ingestion/text_normalizer/text_normalizer.py` L308 — `normalize` is idempotent: normalizing twice equals normalizing once, asserted directly.
- `DECISION-90af5d8d` — U+00AD, U+FEFF and U+200B are removed as the `invisible` rule; U+200D and U+00A0 are deliberately left alone.
- `components/ingestion/text_normalizer/tests/test_component.py` — 46 test functions covering idempotence, transformation reporting, and the exact invisible-character set.

Why it is partial:

- Repeated running headers and footers are **not** collapsed. `v2.md` L161
  lists this as a required capability and the normalizer contains no
  header/footer logic at all.

### 7. Metadata Enrichment — partial

Evidence that exists:

- `components/ingestion/metadata_enricher/metadata_enricher.py` L193 — `MetadataEnricher` merges with precedence `extracted < source < system` (`DECISION-a7b54f00`), so a document cannot outrank the system.
- `components/ingestion/metadata_enricher/metadata_enricher.py` L82 — `SYSTEM_PREFIX = "system_"` reserves the namespace the component computes: `system_source_path`, `system_ingested_at`, `system_content_sha256`.
- `components/ingestion/metadata_enricher/metadata_enricher.py` L219 — `enrich` records an overrule in `omitted` only when the values actually differ, so the audit trail is not padded with noise.
- `components/ingestion/metadata_enricher/tests/test_component.py` — 31 test functions covering precedence, overrule recording, and reserved-key protection.

Why it is partial:

- The protection is narrower than the namespace. `MetadataEnricher` defends
  only the `system_` keys *it computes*. A document that asserts a
  reserved-prefixed key the component has no value for — `system_document_id`
  is the concrete case — survives into merged metadata as an ordinary source
  key. Nothing downstream is misled, because document identity comes from the
  content hash in the seed rather than from metadata, but the `system_` prefix
  is not by itself a guarantee. This is asserted as current behaviour in
  `tests/integration/test_ingestion_pipeline.py` so that tightening the
  component breaks that test visibly.

### 8. Chunking — complete

Evidence:

- `components/chunking/fixed_size/fixed_size_chunker.py` L129, `components/chunking/recursive/recursive_chunker.py` L230, `components/chunking/markdown_aware/markdown_chunker.py` L315, `components/chunking/token_based/token_chunker.py` L180, `components/chunking/structure_aware/structure_aware_chunker.py` L315 — five independent strategies, each a single standard-library-only file with its own `chunk` entry point.
- Chunk ids, offsets and provenance are produced by each strategy and asserted
  to round-trip: the e2e suite indexes a citation offset and then slices the
  source with it, and the slice returns exactly what the index claims.
- `components/chunking/fixed_size/tests/test_component.py` 19, `components/chunking/recursive/tests/test_component.py` 21, `components/chunking/markdown_aware/tests/test_component.py` 22, `components/chunking/token_based/tests/test_component.py` 23, `components/chunking/structure_aware/tests/test_component.py` 26 — 111 test functions across the five strategies.
- `tests/unit/test_chunking.py` — 38 further test functions across strategies.

Gaps named honestly: semantic chunking is not implemented; `v2.md` L171
qualifies it "where implemented", and it is not. Token-aware chunking *is*
implemented (`token_based`), so that half of step 8 is present.

### 9. Chunk Validation — complete

Evidence:

- `components/chunking/chunk_validator/chunk_validator.py` L242 — `ChunkValidator` checks empty and whitespace-only text, minimum and maximum size, required metadata, chunk-to-document references, and id/location sanity, returning `ValidationOutcome` (L199).
- `components/chunking/chunk_validator/chunk_validator.py` L94 — `Severity` separates warning from fatal, and `FatalPolicy` at `components/chunking/chunk_validator/chunk_validator.py` L131 lets the caller choose reject, repair or quarantine, so severity is never conflated with disposition.
- `components/chunking/chunk_validator/tests/test_component.py` — 65 test functions covering each rule, both severities, and each disposition policy.
- Regression: whitespace-only chunks reaching the validator are dropped and counted rather than failing the document — the defect found and fixed during V2-T09.

Gaps: none.

### 10. Chunk Deduplication — partial

Evidence that exists:

- `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L240 — `Deduplicator.dedupe` removes exact duplicates and returns `DedupOutcome` (L201), which reports what was removed rather than silently shrinking the list.
- `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L484 — `EmbeddingReuseGate.check` allows embedding reuse only across a compatible configuration; incompatible configurations can never share an embedding, which is AC-24.
- `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L397, L461 — `IndexFingerprint` and `fingerprint_of` produce the opaque identity that makes that compatibility decision checkable.
- `components/chunking/chunk_deduplicator/tests/test_component.py` — 76 test functions covering exact-duplicate removal, reuse across compatible configs, refusal across incompatible ones, and provenance preservation.

Why it is partial:

- **Near-duplicate (similarity) detection does not exist.** This is a
  deliberate scope exclusion recorded as `FR-43` and stated in the component
  itself at `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L28–L29 and L125. It is reported here as
  absent, not as done.
- **Deduplication runs per document, not corpus-wide.** The orchestration
  layer has exactly two phases — corpus stages, then per-document stages —
  with no phase after the documents, so a corpus-wide dedup stage cannot be
  expressed. Scoped to a single document it is correct; the corpus-wide
  guarantee the component is capable of is not delivered by the pipeline.

### 11. Embedding — complete

Evidence:

- `components/embeddings/embedding_protocol/embedding_protocol.py` L87 — `EmbeddingProtocol` is the single interface every embedder satisfies; `components/storage/core_types/core_types.py` L270 re-declares it for copied-in use.
- `components/embeddings/local_embeddings/local_embeddings.py` L93 — `LocalEmbedder` runs offline with no network call, so the default path needs no credentials.
- `components/embeddings/remote_embeddings/remote_embeddings.py` L94 — `RemoteEmbedder` calls a caller-supplied HTTP endpoint with a bearer key read from the environment, and the key is never placed in a log line or exception text.
- `components/embeddings/batch_embeddings/batch_embeddings.py` L59 — `BatchEmbedder` batches calls and preserves order.
- Model identity and vector width are recorded through `IndexFingerprint` (`components/chunking/chunk_deduplicator/chunk_deduplicator.py` L397), and a width mismatch is a validation error rather than a silent downcast.
- Tests: `components/embeddings/embedding_protocol/tests/test_component.py` 16, `local_embeddings` 27, `remote_embeddings` 33, `batch_embeddings` 26 — 102 test functions, plus `tests/unit/test_embeddings.py` 47.

Gaps named honestly: no provider-specific SDK integration (OpenAI, Anthropic,
Voyage and others). `RemoteEmbedder` is endpoint-agnostic by design, and no
live-API test exists — the project verifies against fakes and never performs
network calls.

### 12. Indexing — partial

Evidence that exists:

- `components/storage/index_writer/index_writer.py` L488 — `IndexWriter` performs idempotent upserts and batch writes, returning `BatchReport` (L320) that distinguishes per-item success from failure instead of all-or-nothing.
- `components/storage/index_writer/index_writer.py` L124 — `DestructiveOperationRefused` makes an unguarded destructive call fail loudly.
- `components/storage/index_writer/index_writer.py` L219 — `record_from` is the single mapping from an `EmbeddedChunk` to a stored record, so payload shape cannot drift between writer and reader.
- `components/storage/index_verifier/index_verifier.py` L378 — the verifier's `verify` is the counterparty that proves what the writer wrote can be read back.
- `components/storage/index_writer/tests/test_component.py` — 62 test functions covering idempotency, partial batch failure, destructive refusal, and dimension/schema mismatch.

Why it is partial:

- **Qdrant and pgvector have no adapter.** They are simply absent.
- **Chroma and Faiss adapters exist but are unverifiable here.**
  `components/storage/adapters/storage_adapters.py` L248 `ChromaVectorStore`
  and L422 `FaissVectorStore` are written, but `chromadb` and `faiss` are not
  installed and installing them would modify a dependency file. They are
  tested only against `FakeChromaClient`
  (`components/storage/adapters/example.py` L114), which proves protocol
  conformance, not a real round trip. The component says so itself at
  `components/storage/adapters/storage_adapters.py` L22.
- **No persistent index backend is exercised with a third-party engine.** The
  end-to-end pipeline writes through `PersistentVectorStore` and
  `PersistentDocumentStore` defined in `tests/integration/pipeline_stages.py`,
  which are file-backed stores built for this repository, not a production
  engine.

### 13. Index Verification — complete

Evidence:

- `components/storage/index_verifier/index_verifier.py` L255 — `IndexVerifier` checks stored count against expected count, verifies every expected chunk is present, reads vectors back through `store.get`, and checks vector width.
- `components/storage/index_verifier/index_verifier.py` L397 — fingerprint comparison detects vectors produced by a different embedding or chunking configuration.
- `components/storage/index_verifier/index_verifier.py` L139, L167 — `Finding` records each observation with a `VerificationVerdict` (L167) separating warning from fatal, so severity is never collapsed into a boolean.
- `components/storage/index_verifier/index_verifier.py` L214 — `raise_if_failed` is the explicit fatal boundary; a caller that ignores findings gets no exception, and one that wants one gets exactly it.
- `components/storage/index_verifier/tests/test_component.py` — 53 test functions covering count mismatch, missing chunks, readback failure, width mismatch, fingerprint mismatch, and retry-with-wait polling.

Regression: the count check compares `store.count()` to the length of the
*expected set*, which over a shared accumulating index would report sibling
documents as surplus. The e2e suite verifies the whole index per document as
a result — the defect found and fixed during V2-T09.

Gaps: none for the required capability.

### 14. Failure Handling — complete

Evidence:

- `components/config/retry.py` L50 — `RetryPolicy` with `max_attempts = 1 + max_retries` (FR-57), living in configuration rather than inside any component.
- `src/rag_code_library/pipeline.py` L136 — `FailurePolicy` offers `stop`, `skip` and `quarantine`.
- `src/rag_code_library/pipeline.py` L666 — `Pipeline.run` always returns a `RunSummary` (L425) for document failures; `FailurePolicy.stop` stops early but the summary is still produced, so a stopped run is never indistinguishable from a crash.
- `src/rag_code_library/pipeline.py` L294 — `CheckpointStore` persists progress so an interrupted run is resumable rather than restarted.
- `KNOWLEDGE-fb9e08cf` — `config.is_retryable` returns `False` for exceptions it does not own, so the orchestration layer must pass `retry_if` explicitly; this is a binding constraint on the retry path, not a default.
- Tests: `components/config/tests/test_config.py` and `tests/unit/test_pipeline.py` (78 test functions) cover each policy, resume, and summary completeness.

Gaps: none.

### 15. Idempotency — complete

Evidence:

- `src/rag_code_library/pipeline.py` L711 — `run_document` derives identity from content, so an unchanged document is skipped rather than re-embedded.
- `src/rag_code_library/pipeline.py` L294 — checkpoints record an honest `stages_completed` prefix; a full match skips the document, a strict in-order prefix resumes at the first incomplete stage, anything else reprocesses.
- `src/rag_code_library/pipeline.py` L419 — `resumed_from` is recorded on the document, so a resumed run is distinguishable from a fresh one in the summary.
- `src/rag_code_library/pipeline.py` L978 — `_write_lock` prevents two runs writing the same destination concurrently.
- `src/rag_code_library/pipeline.py` L61 — `PipelineSettings.fingerprint` (config.py) plus `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L397 `IndexFingerprint` mean a configuration change forces reprocessing instead of silently mixing incompatible vectors.
- End-to-end proof in `tests/integration/test_ingestion_pipeline.py`: re-ingesting unchanged sources writes nothing new; editing one document reprocesses only that document; interrupting and re-running converges on the same index as an uninterrupted run.

Gaps: none.

### 16. Versioning — partial

Evidence that exists:

- `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L397 — `IndexFingerprint` covers embedding model identity and vector width together with chunking strategy and its configuration, which is the part of step 16 that prevents silently mixing incompatible vectors.
- `components/config/settings.py` L116 — `Settings.index_fingerprint` is the documented hook: store it with your vectors and compare on startup.
- `components/config/models.py` L89 — `EmbeddedChunk.index_fingerprint` records which configuration produced each stored vector, so stale data is detectable per chunk rather than only globally.
- `components/config/tests/test_config.py` L141 — `test_fingerprint_tracks_index_identity_only` asserts the fingerprint is deterministic and changes only for index-relevant settings.

Why it is partial:

- Only embedding and chunking identity are versioned. There is **no parser
  implementation version, no normalization configuration version, no metadata
  schema version, no index schema version and no pipeline configuration
  version**. `src/rag_code_library/config.py` L72 `fingerprint` is a
  caller-supplied opaque string, not a computed one.
- No controlled rebuild, no rollback, no migration procedure and no
  migration documentation exist.

### 17. Observability — partial

Evidence that exists:

- `src/rag_code_library/pipeline.py` L425 — `RunSummary` reports per-stage status and duration plus per-document outcome, counts for discovered, succeeded, skipped, failed and rejected, and chunk totals.
- `src/rag_code_library/pipeline.py` L751 — a `correlation_id` is attached so all events belonging to one run can be grouped, satisfying the correlation requirement.
- `src/rag_code_library/cli.py` L353 — `_cmd_ingest` writes the summary to `run-summary.json` at the destination.
- `scripts/check_no_secrets.py` — enforces that no API key or token reaches a log line, exception message or test fixture.

Why it is partial:

- There are **no structured logs, no counters and no latency histograms**.
  `components/observability/` is an empty directory by `DECISION-544dc702`,
  kept empty so that `scripts/check_deferred_surface.py` and AC-19 pass
  unmodified. A JSON summary file is not a logging layer.
- Failure rate is reported per run but never accumulated across runs.

### 18. Testing — complete

Evidence:

- `tests/integration/test_ingestion_pipeline.py` L172 — `TestThirteenStages.test_ragkit_ingest_runs_every_stage_in_order` drives the real `ragkit ingest` CLI end to end; that module collects 55 tests covering AC-21, AC-22, AC-23, AC-28, AC-29 and AC-30.
- `tests/unit/test_pipeline.py` L130 — `test_a_stage_output_is_visible_to_the_next_stage` opens the orchestration suite, 78 test functions.
- `tests/unit/test_storage.py` L223 — `test_nested_filter_values_are_rejected_not_dropped` is one of 81 storage test functions, which include the protocol contract checks.
- `tests/unit/test_ingestion.py` L218 — `test_unicode_normalization_is_idempotent` is one of 64 ingestion test functions.
- `tests/unit/test_path_safety.py` L76 — `test_relative_traversal_rejected` is one of 22 hostile-path cases; `tests/unit/test_security_components.py` L132 `test_hostile_input_is_rejected` is one of 66 security test functions.
- `components/` — 44 manifests, each carrying its own tests directory and
  test module, 1524 test functions in total.
- Full suite: **3516 passed, 10 skipped**, every skip reported with its reason (NFR-2) — the skips are environmental (two refused symlinks on Windows, one `pymupdf` absence, and the remainder likewise stated in place).
- Layers present: unit, storage protocol contract, chunking, ingestion,
  idempotency, failure and recovery, security and hostile input, integration
  and end-to-end, plus regression tests for every defect found during V2.

Gaps named honestly: no live-API integration tests (the project never makes
network calls) and no PDF round trip, both for the reasons above. Those two
absences are why steps 4 and 12 are partial rather than complete.

### 19. Security — partial

Evidence that exists:

- `src/rag_code_library/security/path_safety.py` L316 — `SafeDestination` is the single validated route for every filesystem write reachable from the CLI; `INVARIANT-fd89977b` makes it non-optional.
- `src/rag_code_library/security/path_safety.py` L65 — `UnsafePathError` reports traversal, project-root escape and symlink escape distinctly rather than as one generic failure.
- `src/rag_code_library/security/path_safety.py` L212 — `_check_reparse_points` handles Windows junctions and reparse points, which a plain `resolve()` comparison would miss.
- `components/ingestion/file_type_validator/file_type_validator.py` L378 — magic-byte validation before any content is parsed, with size and emptiness limits.
- `components/ingestion/metadata_enricher/metadata_enricher.py` L82 — reserved namespace protection so untrusted metadata cannot override system fields (see the step 7 caveat).
- `components/security/input_validation/input_validation.py` L170 — `InputValidator`; `components/security/metadata_validation/metadata_validation.py` L143 — `MetadataSchema`.
- `tests/unit/test_path_safety.py` — 22 test functions including hostile traversal and junction cases; `tests/unit/test_security_components.py` — 66.

Why it is partial:

- **No bounded resource limits for parsing.** File size is capped, but parse
  *time* and *memory* are not, so a decompression-bomb-shaped input is
  rejected by size only if it is large on disk.
- **Prompt-injection detection is not implemented** and is not claimed by any
  component.
- Multi-tenant and authorization boundaries are absent; this is a single-user
  local tool by decision, so they are out of scope rather than missing
  requirements.

### 20. Lifecycle — partial

Evidence that exists:

- `components/ingestion/document_discovery/document_discovery.py` L131 — delta states `added`/`modified`/`deleted`/`unchanged` give the ingestion lifecycle its trigger.
- `tests/integration/pipeline_stages.py` — the `identify` stage reconciles the index against the current document set: stale chunks are deleted *and* the stale document record is dropped, so a deleted source leaves no orphan behind.
- `src/rag_code_library/pipeline.py` L294 — checkpoints make an interrupted lifecycle resumable, and `tests/integration/test_ingestion_pipeline.py` asserts that resume converges on the same index as an uninterrupted run.
- `src/rag_code_library/pipeline.py` L425 — `RunSummary` records `chunks_removed` and `documents_removed`, giving an auditable count of destructive actions.

Why it is partial:

- **The deletion policy lives in the test assembly, not in `src/`.**
  Reconciliation is implemented in `tests/integration/pipeline_stages.py`
  because V2-T08's two-phase orchestration has no post-document phase to run a
  corpus-wide reconciliation stage in. It is proven end to end, but a user
  copying components out of this library does not get it from `src/`.
- No tombstones, no reference counting for chunks shared between documents,
  no scheduled index rebuild, and no retention policy.
- The reconciliation sweep keys on the current document-id set and therefore
  assumes one index per source directory; an index shared by several sources
  would have the others reconciled away.

---

## Not implemented

### File formats

| Format | State |
|---|---|
| Plain text (`.txt`) | implemented — `TextAdapter`, `components/ingestion/document_loader/document_loader.py` L285 |
| Markdown (`.md`, `.markdown`) | implemented — `MarkdownAdapter`, `components/ingestion/document_loader/document_loader.py` L299 |
| HTML (`.html`, `.htm`) | implemented, reduced to flat text — `HtmlAdapter`, `components/ingestion/document_loader/document_loader.py` L355 |
| PDF | **not verified end to end** — extractors written (`components/ingestion/pdf_extraction_pymupdf/pymupdf.py` L150, `components/ingestion/pdf_extraction_ocr/ocr.py` L162) but their dependencies cannot be installed; guarded `MissingDependency` is raised instead |
| DOCX | **not implemented** — no parser; refused by name (`FileKind.DOCX`, `components/ingestion/file_type_validator/file_type_validator.py` L62) |
| XLSX | **not implemented** — no parser; refused by name |
| PPTX | **not implemented** — no parser; refused by name |
| RTF | **not implemented** — no parser; refused by name |
| CSV | **not implemented as a format** — detected as `text` and read raw |
| JSON | **not implemented as a format** — detected as `text` and read raw |
| ZIP, GZIP, BZ2 archives | **not implemented** — classified and refused, never unpacked |
| PNG, JPEG, GIF, TIFF | **not implemented** — classified and refused; no standalone image OCR path |
| Scanned-document OCR | **not verified** — `OcrExtractor` exists but `tesseract` is unavailable |

### Services and storage backends

| Backend | State |
|---|---|
| File-backed stores used end to end | implemented — `PersistentVectorStore` / `PersistentDocumentStore`, both defined in `tests/integration/pipeline_stages.py` |
| In-memory document store | implemented — `components/storage/document_store/document_store.py` L191 |
| Chroma | **unverifiable here** — `components/storage/adapters/storage_adapters.py` L248; `chromadb` not installed, tested only against `FakeChromaClient` (`components/storage/adapters/example.py` L114) |
| Faiss | **unverifiable here** — `components/storage/adapters/storage_adapters.py` L422; `faiss` not installed |
| Qdrant | **not implemented** — no adapter |
| pgvector / PostgreSQL | **not implemented** — no adapter |
| Remote embedding endpoint | implemented and generic — `components/embeddings/remote_embeddings/remote_embeddings.py` L94; no provider SDK (OpenAI, Anthropic, Voyage, Cohere and others are all absent) |
| Cloud object storage (S3, GCS, Azure) discovery | **not implemented** — no adapter |
| HTTP/URL discovery | **not implemented** — no adapter |
| Hosted backend of any kind | **out of scope by decision** — this project is local-only |

### Verification checks not performed

- **Near-duplicate similarity check** — out of scope under `FR-43`, stated in
  `components/chunking/chunk_deduplicator/chunk_deduplicator.py` L28–L29 and L125; no similarity threshold is ever
  evaluated.
- **Real Chroma / Faiss round trip** — protocol conformance only, via fakes.
- **Qdrant / pgvector round trip** — no adapter exists to test.
- **PDF extraction round trip** — no PDF is ever parsed in this repository.
- **Live remote-embedding call** — `RemoteEmbedder` is exercised against a
  stub; no network request is made.
- **Bounded parse-time / parse-memory check** — no such limit is enforced.
- **Prompt-injection detection check** — not implemented, not claimed.
- **Index rebuild, rollback and migration check** — no migration path exists
  to test.
- **Cross-run failure-rate and latency check** — `RunSummary` covers one run;
  nothing aggregates across runs.

---

## What was verified, and how

- `python -m pytest -q` — **3516 passed, 10 skipped** on CPython 3.12.14
  (`.venv`), every skip reported with its reason.
- `scripts/validate_components.py` — 44 manifests valid.
- `scripts/validate_dependencies.py`, `scripts/check_no_secrets.py`,
  `scripts/check_deferred_surface.py` — exit 0.
- `scripts/check_documented_commands.py` — 52 passed, 3 manual, 0 failed.
- This report intentionally contains **no shell command blocks**: it is a
  report, not a how-to, and the documented-commands script named above
  executes every fenced command it finds.

The three steps deferred by `DECISION-1aedada9` — 16 Versioning, 17
Observability and 20 Lifecycle — are reported above on their actual merits.
None is claimed complete.
