# RAG Code Library — Master Project Plan

**Document type:** Product requirements, architecture, implementation specification, and acceptance criteria  
**Project status:** Planned / under development  
**Primary environment:** Windows, VS Code, Python, `uv`  
**Frontend:** Plain HTML, CSS, JavaScript  
**Primary usage:** Local, personal engineering reference and code-generation tool  
**Initial focus:** Document ingestion and PDF extraction  
**Core rule:** Canonical component source code is copied into client projects. Client applications must not be forced to import this library at runtime.

---

# 1. Project Vision

Build a local RAG Code Library that contains reusable, independently understandable, tested, and copy-ready components for building Retrieval-Augmented Generation (RAG) systems.

The library must help me:

1. Browse RAG components through a simple local website.
2. Understand what each component does and when to use it.
3. Compare implementation strategies and their trade-offs.
4. Copy a function or component manually from the website.
5. Use a CLI to copy a component and its required companion files into a client project.
6. Understand dependencies, compatibility, limitations, and security considerations.
7. Test the copied implementation without depending on the library at runtime.
8. Extend the library without duplicating or manually synchronizing implementations.

> **My note:** This is not just a collection of Python snippets. It is a maintained library of source-code building blocks, documentation, tests, and a tool for safely reusing those building blocks.

## 1.1 Primary design decision

The library must support two usage modes.

### Mode A — Manual copy

I open the local cookbook website, browse to a component, read its documentation, inspect its source code, and copy the code into my client project.

### Mode B — CLI generation

I run a command such as:

`raglib add ingestion.pdf_extraction.pymupdf --target ./my-rag-app`

The CLI resolves the component's required files, previews the proposed changes, checks for conflicts, and copies the approved files into the target project.

The generated client code must run independently of the library package at runtime, unless a component explicitly declares a different delivery mode and that exception is documented.

> **My note:** The website and CLI must use the same canonical source. Never maintain one implementation for the website and another for the CLI.

---

# 2. Goals and Non-Goals

## 2.1 Goals

- Build a reusable component library for the full RAG pipeline.
- Support multiple implementation strategies for the same problem.
- Keep each component's purpose and boundaries explicit.
- Provide complete usage instructions and copy-ready examples.
- Declare dependencies and optional dependencies accurately.
- Preserve source provenance and useful metadata.
- Test components independently and in realistic integrations.
- Generate the website catalogue from canonical component data.
- Generate client code safely.
- Avoid forcing a specific application architecture on client projects.
- Keep the initial system local and simple.
- Expand based on validated components rather than creating empty directories for every possible feature.

## 2.2 Non-goals for the initial version

- A hosted SaaS product.
- A multi-user cloud marketplace.
- A mandatory runtime SDK for client applications.
- Automatic installation of dependencies without permission.
- Automatic rewriting of client dependency files without permission.
- Implementing every planned RAG feature before the first end-to-end workflow works.
- Claiming production readiness merely because code executes or unit tests pass.

---

# 3. Non-Negotiable Architecture Principles

## 3.1 Single source of truth

The `components/` directory contains canonical implementations.

Each published component must have a defined source bundle, manifest, documentation, dependencies, tests, and examples where applicable.

The cookbook's generated data and the CLI's file bundle resolution must derive from these canonical definitions.

Never hand-edit generated website data.

## 3.2 Client independence

The library CLI copies component source into the target project.

The client should not need to install or import the RAG Code Library package at runtime.

Components may use external packages such as PyMuPDF, Pydantic, or an embedding provider. Those dependencies must be declared.

## 3.3 No forced client architecture

The library must not force client projects to use:

- FastAPI
- Streamlit
- A particular database
- A particular vector database
- A specific folder layout
- A particular dependency-injection framework
- A specific logging or configuration library

A component should have a minimal, explicit interface and should be adaptable to different application structures.

## 3.4 Safe code generation

The CLI must:

- Preview proposed files before writing.
- Detect target-file collisions.
- Refuse silent overwrites.
- Reject path traversal.
- Restrict writes to the intended target directory.
- Report dependencies and potential conflicts.
- Record component identity and version.
- Avoid deleting or modifying unrelated client files.
- Provide clear errors and recovery instructions.

## 3.5 Conservative dependency management

The CLI must report required packages and relevant version constraints.

It must not silently install packages, upgrade dependencies, or edit `pyproject.toml` or `requirements.txt`.

If dependency-file updates are added later, they must be explicit, previewable, and reversible where practical.

## 3.6 Evidence-based readiness

A component is not production-ready merely because its happy path works.

Readiness must be supported by tests, realistic fixtures, documented compatibility, security review, error handling, and integration evidence.

---

# 4. Target Repository Structure

Use the following as the intended structure. Add directories when their implementations become real; do not create large numbers of empty placeholder components just to make the tree look complete.

```text
rag-code-library/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── .gitignore
├── .pre-commit-config.yaml
│
├── src/
│   └── rag_code_library/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── errors.py
│       │
│       ├── registry/
│       │   ├── loader.py
│       │   ├── manifest.py
│       │   ├── validator.py
│       │   └── resolver.py
│       │
│       ├── generator/
│       │   ├── planner.py
│       │   ├── copier.py
│       │   ├── dependency_manager.py
│       │   ├── conflict_detector.py
│       │   └── provenance.py
│       │
│       ├── project/
│       │   ├── detector.py
│       │   ├── package_detector.py
│       │   └── path_resolver.py
│       │
│       └── security/
│           ├── path_safety.py
│           └── manifest_validation.py
│
├── components/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranking/
│   ├── context/
│   ├── evaluation/
│   ├── security/
│   └── storage/
│
├── cookbook/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── data.js
│   └── README.md
│
├── schemas/
│   └── component-manifest.schema.json
│
├── scripts/
│   ├── build_cookbook.py
│   ├── validate_components.py
│   └── verify_generated_code.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── examples/
│   ├── monolithic_client/
│   ├── layered_client/
│   └── microservices_client/
│
└── docs/
    ├── architecture.md
    ├── component_authoring.md
    ├── cli_reference.md
    ├── security.md
    ├── versioning.md
    └── troubleshooting.md
```

> **My note:** The target structure is a guide. Build the registry, generator, and website around a small working component before expanding every category.

---

# 5. Canonical Component Contract

Every published component must follow a documented contract.

## 5.1 Required component information

Each component must declare:

- Stable component ID.
- Component name and short description.
- Version.
- Category and subcategory.
- Problem solved.
- When to use it.
- When not to use it.
- Entry-point files.
- Required companion files.
- Python version compatibility.
- Required dependencies.
- Optional dependencies.
- Configuration options.
- Input and output contracts.
- Expected errors.
- Security considerations.
- Limitations.
- Test command.
- Compatibility status.
- Documentation and examples.
- Source files to copy.
- Files that must not be copied.

## 5.2 Component manifest

Use a structured manifest such as YAML or JSON.

Illustrative example:

```yaml
id: ingestion.pdf_extraction.pymupdf
name: PyMuPDF PDF Extraction
version: 0.1.0
category: ingestion
description: Extract text and metadata from digital PDFs.

entry_points:
  - extract_text.py

files:
  - extract_text.py
  - models.py
  - errors.py

dependencies:
  python: ">=3.11"
  packages:
    - "PyMuPDF"

optional_dependencies: []

tests:
  command: "pytest tests/"

delivery:
  mode: copy_source

status: experimental
```

This schema is illustrative. Define the actual manifest schema centrally and validate every manifest against it.

> **My note:** Do not let the website, CLI, and tests interpret component metadata differently. One validated manifest should drive the relevant workflows.

## 5.3 Component maturity states

Use explicit maturity labels:

- `experimental`: API and behavior may change.
- `tested`: declared unit and integration tests pass.
- `validated`: representative fixtures and target environments have been tested.
- `production_ready`: the documented readiness criteria have been met for a stated scope.

Do not automatically promote components between states.

---

# 6. Component Directory Convention

Use a consistent structure where it adds value:

```text
component_name/
├── implementation.py
├── manifest.yaml
├── README.md
├── examples/
│   └── basic_usage.py
└── tests/
    ├── test_implementation.py
    └── fixtures/
```

Shared dependencies should be explicit in the manifest or bundled source definition.

Do not duplicate common models or error types in every component without considering the consequences.

If a component depends on a shared helper, the CLI must either copy that helper as a companion file or generate an equivalent self-contained implementation.

> **My note:** A component is not truly copy-ready if its code silently depends on a helper that the user does not receive.

---

# 7. Ingestion Pipeline Scope

The ingestion pipeline must transform source files into normalized, traceable document records that are suitable for chunking.

It should support the following logical stages:

1. Document discovery
2. File validation
3. Document identification
4. Document loading
5. Text and structure extraction
6. Normalization
7. Metadata enrichment
8. Chunking
9. Chunk validation
10. Chunk deduplication
11. Embedding
12. Indexing
13. Index verification
14. Failure handling
15. Idempotency
16. Versioning
17. Observability
18. Testing
19. Security
20. Lifecycle management

Important distinctions:

- This is a logical pipeline, not a requirement to implement one giant function.
- Security, observability, versioning, and testing are cross-cutting concerns.
- Chunking and embedding should remain separate from format-specific parsing.
- Indexing and storage should remain separate from extraction.
- Updates and deletions need reconciliation and lifecycle behavior, not merely initial ingestion.

---

# 8. Ingestion Component Architecture

Target structure:

```text
components/
└── ingestion/
    ├── document_loader/
    ├── pdf_extraction/
    │   ├── pymupdf/
    │   ├── docling/
    │   └── ocr/
    ├── docx_extraction/
    ├── html_extraction/
    ├── markdown_extraction/
    ├── spreadsheet_extraction/
    ├── structured_data_extraction/
    ├── text_normalizer/
    └── extraction_quality/
```

## 8.1 `document_loader/`

Responsibility: coordinate loading and parser selection.

Expected capabilities:

- Validate source paths and supported input types.
- Detect format using more than just the filename extension where practical.
- Select the correct parser.
- Apply parser configuration.
- Convert parser output to a common result model.
- Surface expected errors and warnings.
- Preserve source identifiers and provenance.
- Avoid embedding, chunking, or indexing responsibilities.

Suggested modules:

- `load_document.py`
- `detect_file_type.py`
- `validate_file.py`
- `parser_registry.py`
- `models.py`

## 8.2 `pdf_extraction/`

Responsibility: extract PDF content using interchangeable strategies.

The component must preserve page numbers and identify which parser produced the content.

### `pdf_extraction/pymupdf/`

Use for ordinary digital PDFs with a usable text layer.

Expected capabilities:

- Page-by-page text extraction.
- PDF metadata extraction.
- Text-block extraction where needed.
- Link extraction where useful.
- Empty-page warnings.
- Clear errors for missing, malformed, or password-protected PDFs.

Main dependency: PyMuPDF.

Limitations to document:

- Multi-column reading order may be imperfect.
- Tables may not preserve rows and columns.
- Image-only pages require OCR.
- Extracted text does not represent every visual relationship.

### `pdf_extraction/docling/`

Use for complex layouts, technical reports, multi-column documents, tables, figures, and structured conversion.

Expected capabilities:

- Convert a PDF into a structured representation.
- Preserve headings, paragraphs, lists, and tables where supported.
- Map output into the library's common document model.
- Preserve page or element provenance when available.
- Record conversion warnings and limitations.

Docling must remain optional because it can require more dependencies and computational resources.

Do not assume page-level provenance works correctly without tests against the pinned version.

### `pdf_extraction/ocr/`

Use for scanned or image-based pages.

Expected capabilities:

- Detect pages that have insufficient native text.
- Render PDF pages at a configurable resolution.
- Run a configured OCR engine.
- Preserve page numbers and available OCR confidence information.
- Support configurable language data.
- Report empty or suspicious OCR output.
- Avoid reprocessing pages unnecessarily when results can be safely reused.

Possible engine: Tesseract, with its executable and language data installed separately.

OCR must not be applied indiscriminately to every PDF.

### PDF strategy selection

- `pymupdf`: ordinary digital PDFs.
- `docling`: structure-sensitive or layout-complex PDFs.
- `ocr`: scans and image-based pages.
- `auto`: native extraction first, then OCR for pages that meet a documented fallback rule.

The `auto` strategy should not claim to detect every extraction error. Text length is only a heuristic; it cannot reliably detect all incorrect reading orders, missing tables, or semantically wrong text.

## 8.3 `docx_extraction/`

Use for Microsoft Word documents.

Expected capabilities:

- Paragraph and heading extraction.
- Tables with row and cell relationships.
- Lists and styles where practical.
- Headers and footers when relevant.
- Document properties.
- Source and structural provenance.

Possible dependency: `python-docx`.

Do not collect all paragraphs and all tables separately if doing so destroys their original order. Document ordering and relationships must be considered.

## 8.4 `html_extraction/`

Use for saved HTML and web content.

Expected capabilities:

- Parse static HTML.
- Extract title and useful metadata.
- Preserve headings, paragraphs, lists, tables, and links.
- Exclude irrelevant navigation and boilerplate where appropriate.
- Support main-content extraction when needed.
- Use a browser renderer only when required for JavaScript-rendered content.

Possible tools: Beautiful Soup, lxml, a readability-style parser, or a browser renderer.

Keep web fetching and crawling separate from HTML parsing. Any fetching component must enforce network restrictions, URL policy, redirect checks, and protection against server-side request forgery.

## 8.5 `markdown_extraction/`

Use for Markdown documentation, README files, notes, and knowledge bases.

Expected capabilities:

- Preserve heading hierarchy.
- Preserve paragraph boundaries.
- Preserve code blocks and language labels.
- Preserve lists, tables, links, and relevant metadata.
- Handle supported Markdown dialects and optional front matter explicitly.

Use a Markdown parser instead of merely removing Markdown punctuation.

## 8.6 `spreadsheet_extraction/`

Use for CSV and Excel data.

Expected capabilities:

- CSV delimiter and encoding handling.
- Excel workbook and worksheet extraction.
- Sheet names and row/column context.
- Header detection and multi-row-header handling where required.
- Table serialization into retrieval-friendly records.
- Configurable treatment of formulas and cached values.
- Large-file limits.

Possible tools:

- Python `csv` for simple CSV files.
- `openpyxl` for `.xlsx` workbooks.
- pandas for tabular transformations.

Preserve the relationships between headers and values. Avoid flattening a spreadsheet into ambiguous isolated numbers.

## 8.7 `structured_data_extraction/`

Use for JSON, XML, and similar structured formats.

Expected capabilities:

- Native parsing.
- Optional schema validation.
- Preservation of keys, attributes, nesting, and repeated records.
- Controlled flattening into retrieval-friendly text.
- Source-path or record-level provenance.
- Safe parser configuration for untrusted inputs.

Structured data should not always be converted into a large unstructured text blob. Some use cases are better served by direct structured queries.

## 8.8 `text_normalizer/`

Responsibility: standardize extracted content without destroying meaning.

Expected capabilities:

- Conservative Unicode normalization.
- Whitespace normalization.
- Line-ending normalization.
- Explicitly defined cleanup rules.
- Optional handling of verified line-break hyphenation.
- Preservation of headings, paragraph boundaries, code indentation, and table structure.

Do not blindly alter:

- Legal numbering.
- Dates and numeric values.
- Code indentation.
- Table relationships.
- Meaningful line breaks.
- Original source content.

Keep enough original extraction data to debug normalization problems.

## 8.9 `extraction_quality/`

Responsibility: identify potentially bad extraction before it reaches chunking and indexing.

Expected checks:

- Empty document detection.
- Empty or suspicious pages.
- Unexpectedly low text density.
- Replacement characters and suspicious character sequences.
- Expected page or sheet coverage.
- Truncation indicators where detectable.
- Known heading or content checks for test fixtures.
- Structural warnings for tables and reading order when measurable.

Possible statuses:

- `pass`
- `warning`
- `fail`
- `needs_review`

Quality scores must be defined and evaluated. A score is not proof of correctness.

High-consequence documents may require manual review when automated checks cannot establish reliable extraction.

---

# 9. Common Document Output Model

All extraction strategies should return a consistent model.

Conceptual fields:

```text
ExtractedDocument
├── document_id
├── source
├── content
├── elements
├── metadata
├── provenance
├── extraction_warnings
└── quality_report
```

Each element may include:

- Element type.
- Text or structured value.
- Page number where available.
- Heading or section context.
- Table, row, or cell information where applicable.
- Source coordinates where available.
- Parser identity.
- Element-level warnings.

Requirements:

- Use typed models or well-defined schemas.
- Make page numbering conventions explicit.
- Preserve the source file identity.
- Distinguish missing metadata from invented metadata.
- Avoid making every format a flat string.
- Do not include full document text in logs by default.
- Define how this model maps to the next chunking stage.

A common contract must not erase format-specific information. Preserve structured elements when they matter.

---

# 10. Chunking Components

Target category:

```text
components/chunking/
├── fixed_size/
├── recursive/
├── token_based/
├── markdown_aware/
├── structure_aware/
├── chunk_validator/
└── chunk_deduplicator/
```

Expected strategies:

- Fixed-size chunking for simple baseline cases.
- Recursive chunking for general text.
- Token-based chunking when model token budgets matter.
- Markdown-aware chunking for technical documentation.
- Structure-aware chunking for headings, tables, and document elements.

Every chunk should preserve relevant provenance:

- Source document ID.
- Page or source location.
- Heading or section context.
- Chunk ID.
- Chunking strategy and version.
- Original element references where available.

Tests should verify size constraints, boundary behavior, metadata preservation, empty input handling, and deterministic behavior.

Do not assume one chunking strategy is optimal for all corpora.

---

# 11. Embedding Components

Target category:

```text
components/embeddings/
├── embedding_protocol/
├── local_embeddings/
├── remote_embeddings/
└── batch_embeddings/
```

Expected capabilities:

- Common embedding interface.
- Local model adapter.
- Remote provider adapter.
- Batch processing.
- Explicit dimensions and model identity.
- Input validation.
- Rate-limit and transient-error handling where applicable.
- Configurable batching and concurrency.
- Clear handling of provider failures.
- Reproducibility metadata.

Embedding models must be compatible with the target index dimensions and retrieval configuration.

Never silently switch models when an embedding provider fails. A model change may invalidate existing vectors.

---

# 12. Retrieval Components

Target category:

```text
components/retrieval/
├── dense/
├── sparse/
├── hybrid/
├── metadata_filter/
└── result_normalizer/
```

Expected strategies:

- Dense semantic retrieval.
- Sparse keyword retrieval, such as BM25.
- Hybrid retrieval combining dense and sparse results.
- Metadata filtering.
- Result normalization.

Expected concerns:

- Top-k and candidate-pool configuration.
- Score interpretation.
- Duplicate result handling.
- Stable result identifiers.
- Metadata and tenant isolation.
- Filter semantics.
- Empty-result behavior.
- Latency and failure handling.

Hybrid retrieval must define its fusion method rather than combining incomparable scores without justification.

---

# 13. Reranking Components

Target category:

```text
components/reranking/
├── cross_encoder/
└── relevance_ranking/
```

Expected capabilities:

- Rerank a bounded candidate set.
- Preserve source metadata and provenance.
- Define input limits and batch size.
- Handle model errors and timeouts.
- Record model identity and configuration.
- Evaluate whether reranking improves retrieval on the target dataset.

Reranking is optional. Do not force every client project to use it.

---

# 14. Context Construction Components

Target category:

```text
components/context/
├── context_selector/
├── token_budget/
└── citation_formatter/
```

Expected capabilities:

- Select relevant retrieval results.
- Enforce a configurable token budget.
- Avoid unnecessary duplicate context.
- Preserve source citations.
- Format context consistently for the generation layer.
- Handle cases where no useful results are found.
- Make truncation and selection behavior explicit.

The context builder should not be responsible for deciding whether the generated answer is factually correct. That belongs to evaluation and generation-layer controls.

---

# 15. Evaluation Components

Target category:

```text
components/evaluation/
├── retrieval_metrics/
├── answer_metrics/
└── evaluation_dataset/
```

Expected capabilities:

- A versioned evaluation dataset.
- Retrieval metrics such as Recall@k, Precision@k, MRR, and nDCG where appropriate.
- Answer relevance and faithfulness evaluation.
- Context precision and context recall where suitable.
- Regression testing across component changes.
- Clear separation between human-labelled ground truth and model-generated judgments.
- Reproducible configuration and recorded model versions.

Evaluation must cover more than one example or one happy-path query.

Do not treat a single metric as a complete measure of RAG quality. Establish baselines and evaluate trade-offs.

---

# 16. Security Components

Target category:

```text
components/security/
├── input_validation/
├── metadata_validation/
└── retrieved_content/
```

Expected capabilities:

- Input and configuration validation.
- Safe handling of untrusted document content.
- Metadata validation.
- Clear separation between retrieved content and trusted instructions.
- Protection against path traversal and unsafe file operations.
- Tenant and access-control metadata handling where applicable.
- Safe error reporting and sensitive-data-aware logging.

Important boundaries:

- A parser should not execute document content.
- Retrieved text must not automatically be treated as trusted instructions.
- Prompt-injection handling is a system-level concern, not a guarantee provided by one cleaning function.
- Security-sensitive components require adversarial tests.
- Parsing untrusted files in a production service requires process and resource isolation.

---

# 17. Storage Components

Target category:

```text
components/storage/
├── document_store/
├── vector_store_protocol/
├── adapters/
├── index_writer/
└── index_verifier/
```

Expected capabilities:

- Clear storage interfaces.
- Optional database and vector-store adapters.
- Stable document and chunk identifiers.
- Idempotent write behavior where supported.
- Index verification.
- Version and deletion reconciliation.
- Metadata filtering.
- Explicit error handling and transaction limitations.

Do not force one vector database on every client.

The storage interface should describe the capabilities it requires. Each adapter must document which capabilities it actually supports.

---

# 18. Cross-Cutting Pipeline Capabilities

The following capabilities may be shared services, interfaces, or explicit pipeline contracts. Do not force them into a single linear function.

## 18.1 Idempotency

Reprocessing an unchanged document should not create uncontrolled duplicate records.

Define stable source identities and content hashes. Specify what happens when the content changes.

## 18.2 Versioning

Track relevant versions such as:

- Component version.
- Parser version.
- Normalization configuration.
- Chunking strategy and version.
- Embedding model identity.
- Index schema version.

Version changes may require reprocessing or re-embedding. Document these implications.

## 18.3 Failure handling

Define:

- Which errors are retryable.
- Retry limits and backoff.
- Timeouts.
- Partial-success behavior.
- Failure reporting.
- Quarantine or dead-letter handling where appropriate.
- Recovery and reprocessing procedures.

Do not retry permanent validation failures indefinitely.

## 18.4 Observability

Provide integration points for:

- Structured logs.
- Stage duration.
- Success and failure counts.
- Parser and strategy selection.
- Quality warnings.
- Retry counts.
- Resource usage where practical.

Do not log sensitive document contents by default.

## 18.5 Lifecycle management

Define behavior for:

- New documents.
- Updated documents.
- Deleted documents.
- Failed reprocessing.
- Reindexing after model or schema changes.
- Stale chunks and stale embeddings.
- Retention and cleanup policies.

Lifecycle behavior must be explicit because updating a document can invalidate old chunks and vectors.

---

# 19. CLI Requirements

The CLI is a source-code generation tool.

Proposed commands:

```text
raglib list

raglib show ingestion.pdf_extraction.pymupdf

raglib plan ingestion.pdf_extraction.pymupdf \
  --target ./my-rag-app

raglib add ingestion.pdf_extraction.pymupdf \
  --target ./my-rag-app

raglib validate

raglib doctor --target ./my-rag-app
```

Adapt command syntax to the final CLI framework and platform.

## 19.1 `list`

List component IDs, categories, summaries, and maturity status.

## 19.2 `show`

Display:

- Purpose.
- Use cases.
- Limitations.
- Files included.
- Dependencies.
- Compatibility.
- Example usage.
- Tests and status.

## 19.3 `plan`

Resolve the component bundle without writing files.

Show:

- Files to create.
- Files already present.
- Conflicts.
- Required and optional dependencies.
- Compatibility concerns.
- Proposed destination paths.
- Any user actions required.

## 19.4 `add`

Copy files only after validation and conflict checks.

Default behavior must refuse silent overwrites.

If overwrite or update functionality is added, show the exact affected files and require explicit approval.

## 19.5 `validate`

Validate component manifests, required files, dependency declarations, and test configuration.

## 19.6 `doctor`

Inspect the target project and report relevant environment and dependency information without making unrequested modifications.

---

# 20. Code Generation and Provenance

The CLI must resolve a component into a complete bundle.

The bundle may include:

- Main source files.
- Companion models and errors.
- Configuration examples.
- Required helper functions.
- Optional tests.
- A generated provenance record.

A component bundle must not accidentally omit a required dependency on another internal helper.

The CLI should:

1. Resolve the component ID.
2. Validate the manifest.
3. Resolve companion components or files.
4. Calculate destination paths.
5. Check paths remain within the target root.
6. Detect collisions.
7. Report dependencies.
8. Display a preview.
9. Require approval when needed.
10. Copy files safely.
11. Record provenance.
12. Report exactly what changed.

Generated code should remain understandable and editable by the client developer.

Do not introduce unnecessary wrappers or hidden runtime dependencies.

---

# 21. Local Cookbook Website

Technology:

- HTML
- CSS
- JavaScript

No frontend framework is required for v1.

## 21.1 Layout

Include:

- Top navigation bar.
- Grouped, collapsible left sidebar.
- Breadcrumb navigation.
- Overview/home page.
- Category and section pages.
- Component detail view.
- Code display.
- Copy button.
- Usage notes.
- Dependencies and limitations.
- Links between related components.

## 21.2 Component detail page

Each component page should show:

1. Name and summary.
2. Problem solved.
3. When to use it.
4. When not to use it.
5. Implementation strategy.
6. Canonical source code.
7. Dependencies.
8. Input and output examples.
9. Usage instructions.
10. Configuration options.
11. Trade-offs.
12. Limitations.
13. Security considerations.
14. Tests and maturity status.
15. Related components.

## 21.3 Generated website data

`cookbook/data.js` is generated.

The build script should derive catalogue information and source code from canonical component files and manifests.

The website must not contain a manually maintained duplicate of the canonical implementation.

The build should be deterministic and should report invalid manifests or missing source files.

## 21.4 Local operation

Support opening the site locally where practical.

If browser restrictions prevent a feature under `file://`, provide a documented local-server option.

Test the actual copy button behavior in the browser environment being used.

Do not add syntax highlighting, search indexing, or other dependencies unless they materially improve the initial version.

---

# 22. Dependency Management

For each component, distinguish:

- Standard-library dependencies.
- Required third-party packages.
- Optional packages.
- Operating-system requirements.
- External executables.
- Model downloads or data files.
- Provider credentials, if applicable.

Examples:

- PyMuPDF is a Python package.
- Docling may require a larger dependency and model footprint.
- Tesseract OCR requires an operating-system executable and appropriate language data in addition to its Python wrapper.

Do not describe a component as fully installed just because its Python package is installed.

Pin or constrain dependencies deliberately. Validate compatibility before changing versions.

Use `uv` for the project's Python environment and dependency workflow. Generate a real lockfile through the intended `uv` workflow; never use a placeholder file as a lockfile.

---

# 23. Testing Strategy

## 23.1 Unit tests

Test individual functions and deterministic behavior.

Examples:

- Manifest validation.
- File type routing.
- Normalization.
- Quality rules.
- Path resolution.
- Conflict detection.
- Dependency reporting.

## 23.2 Component integration tests

Test each parser or adapter against actual fixture files.

Examples:

- PDF page extraction.
- OCR output.
- Docling conversion.
- DOCX table extraction.
- Markdown heading extraction.
- Spreadsheet row mapping.

## 23.3 Generator integration tests

Verify:

- Correct files are copied.
- Required companion files are included.
- Dry run writes nothing.
- Existing files are protected.
- Unsafe paths are rejected.
- Provenance is generated correctly.
- Dependency requirements are reported accurately.

## 23.4 End-to-end tests

Test the workflow:

1. Select a component.
2. Inspect its plan.
3. Generate it into a clean client project.
4. Install declared dependencies explicitly.
5. Run the generated example or test.
6. Verify output and provenance.
7. Confirm unrelated files were not modified.

## 23.5 Regression tests

Maintain a small, versioned set of representative documents and queries.

When a parser or strategy changes, compare results against known expectations.

For extraction, exact full-text equality may be too strict in some cases. Define appropriate checks for page count, expected phrases, structure, source mapping, and quality warnings.

---

# 24. Security and Reliability Requirements

The project must consider:

- Path traversal.
- Malicious filenames.
- Corrupt and malformed files.
- Excessively large documents.
- Archive expansion limits where applicable.
- Parser timeouts and crashes.
- Memory and CPU limits.
- Untrusted document instructions.
- Sensitive data in logs.
- Unsafe XML parsing.
- External URL fetching and SSRF if web crawling is added.
- Tenant and access-control boundaries when components are used in multi-tenant applications.

For production ingestion services, process untrusted files in a constrained worker with appropriate resource limits.

The library should expose clear interfaces and document safe deployment requirements, but copying a parser function does not itself provide sandboxing.

---

# 25. Documentation Requirements

The repository README must explain:

- What the project is.
- What it is not.
- Installation and development setup.
- How to run tests.
- How to build the cookbook.
- How to browse components.
- How to use the CLI.
- How to add a component.
- How to generate code into a client.
- How to handle dependencies and conflicts.
- Known limitations.

The architecture documentation must explain:

- Canonical component source.
- Manifest schema.
- Bundle resolution.
- Website data generation.
- CLI planning and copying.
- Provenance.
- Testing boundaries.
- Security assumptions.

Each component README must explain its own use cases, trade-offs, inputs, outputs, dependencies, examples, and limitations.

---

# 26. Continuous Integration and Repository Hygiene

Automate the following checks:

- Unit tests.
- Integration tests where practical.
- Manifest schema validation.
- Required-file validation.
- Generated cookbook freshness.
- Import or syntax checks for supported Python versions.
- Formatting and linting.
- Security checks appropriate to the project.
- Repository cleanliness checks.

Do not commit:

- Virtual environments.
- `__pycache__`.
- Unintended generated test fixtures.
- Secrets or credentials.
- Local machine-specific paths.
- Temporary artifacts.
- Placeholder lockfiles.

Commit only when explicitly intended, with meaningful commit messages.

---

# 27. Example Client Projects

Maintain small examples that prove the copied code does not require one fixed architecture.

## 27.1 Simple client

A small application with a direct function call and minimal structure.

## 27.2 Layered client

A project with separate domain, application, and infrastructure layers.

## 27.3 Service-oriented client

A project where a copied component is used behind a service or API boundary.

Examples should remain small and focused. They are compatibility demonstrations, not mandatory templates for all users.

---

# 28. Implementation Order

Work in dependency order, not by creating every folder first.

## Stage 1 — Foundation

- Set up the repository and `uv`.
- Configure package metadata and tests.
- Define errors and basic development commands.
- Write the initial README.
- Establish formatting and linting.

**Exit condition:** a fresh environment can install the project and run its tests.

## Stage 2 — Component contract

- Define the manifest schema.
- Define component IDs and versions.
- Define bundle files and entry points.
- Define dependency metadata.
- Define validation rules.

**Exit condition:** valid manifests pass and invalid manifests fail clearly.

## Stage 3 — PDF extraction vertical slice

- Implement PyMuPDF extraction.
- Define a shared result model.
- Preserve page provenance.
- Handle expected failures.
- Add representative fixtures and tests.
- Document limitations.

**Exit condition:** the basic digital-PDF path is tested and reproducible.

## Stage 4 — OCR and Docling

- Add optional OCR.
- Add scanned-page handling.
- Add optional Docling conversion.
- Validate element/page provenance.
- Test optional dependency failures.
- Document resource requirements.

**Exit condition:** each strategy has an explicit contract and tested limitations.

## Stage 5 — Normalization and quality

- Add conservative normalization.
- Add empty-page and suspicious-text checks.
- Add page-coverage diagnostics.
- Define warning/failure behavior.

**Exit condition:** known bad extraction cases are surfaced rather than silently accepted.

## Stage 6 — Registry and CLI planning

- Load and validate manifests.
- Resolve component bundles.
- Detect target paths.
- Report dependencies and conflicts.
- Implement dry-run planning.

**Exit condition:** the CLI can produce an accurate plan without changing the target project.

## Stage 7 — Safe generation

- Implement copying.
- Reject unsafe paths.
- Refuse silent overwrites.
- Record provenance.
- Test failure recovery.

**Exit condition:** generated files are correct and unrelated client files remain untouched.

## Stage 8 — Cookbook website

- Implement the overview and navigation.
- Implement component detail pages.
- Generate `data.js`.
- Implement copy behavior.
- Add deterministic build checks.

**Exit condition:** the website reflects canonical source and works locally.

## Stage 9 — End-to-end validation

- Generate PDF extraction into an example client.
- Run the generated code.
- Verify dependencies and provenance.
- Validate the website/CLI consistency.

**Exit condition:** the full browse-or-generate workflow works reproducibly.

## Stage 10 — Expand components

Add DOCX, HTML, Markdown, spreadsheet, and structured-data extraction.

Then expand into chunking, embeddings, retrieval, reranking, context, evaluation, security, and storage.

**Exit condition:** each added component follows the same manifest, documentation, testing, and generation contract.

## Stage 11 — Release readiness

- Add CI.
- Document compatibility.
- Review security assumptions.
- Verify generated data.
- Run tests from a clean environment.
- Record known limitations.

**Exit condition:** release claims are supported by actual verification evidence.

---

# 29. Definition of Done

The project is ready for its intended initial release when:

- A fresh checkout installs and runs the documented tests.
- The manifest schema is validated.
- The PDF extraction component works against the declared test corpus.
- Parser failures are surfaced clearly.
- Provenance and quality diagnostics are available.
- The cookbook is generated from canonical component data.
- The CLI can preview all proposed file changes.
- The CLI can copy a component into a clean target project.
- Existing files are not silently overwritten.
- Unsafe paths are rejected.
- Dependencies are reported accurately.
- Example clients can run the copied code.
- CI checks tests and generated-data consistency.
- Documentation explains limitations and compatibility.

Production readiness must be scoped to the tested environments, formats, dependencies, and use cases. Do not imply universal reliability.

---

# 30. Immediate Action List

Do these next, in order:

1. Inspect the existing PDF extraction ZIP and verify its actual contents.
2. Treat it as a prototype until all parser integrations have been tested.
3. Define the shared document/result model and page provenance.
4. Build a representative PDF fixture corpus.
5. Make the PyMuPDF path reliable first.
6. Add quality diagnostics and explicit error handling.
7. Add OCR fallback and validate its behavior.
8. Add Docling and verify structured extraction/provenance.
9. Define the component manifest and validation rules.
10. Integrate PDF extraction into the canonical `components/` structure.
11. Build the CLI's planning and conflict-detection workflow.
12. Implement safe source copying.
13. Generate cookbook data from canonical component source.
14. Add end-to-end tests for browsing, planning, copying, and running the generated code.
15. Only then expand the catalogue.

---

# Final Guiding Principle

Build a small number of complete, trustworthy components before building a large catalogue.

A component is complete when a developer can understand when to use it, inspect its source, identify its dependencies, copy it into a client project, run its tests, trace its output to the source, and recover safely from common failures.

**The first milestone is not “many RAG components.” It is one complete PDF extraction component, correctly documented, tested, catalogued, and safely copyable into an independent client project.**