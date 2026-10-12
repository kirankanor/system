# Project: Production-Ready RAG Code Library

I want to build a production-ready **RAG Code Library** using Python, `uv`, and VS Code.

## 1. What I want to build

I want a reusable library of production-oriented RAG components. Each component should contain copy-ready Python code, clear documentation, usage examples, tests, and information about when to use it and its trade-offs.

I also want a local cookbook website to browse components and a Python CLI called `ragkit` to list, inspect, and add components to projects.

## 2. Main components

Organize the library into these areas:

1. **Loaders** — PDF, DOCX, TXT, Markdown, HTML, and other supported document formats.
2. **Parsing and cleaning** — extract text, metadata, tables, and structured content.
3. **Chunking** — fixed-size, recursive, token-based, Markdown-header, semantic, and other strategies.
4. **Embeddings** — local and API-based embedding providers.
5. **Storage and indexing** — vector databases and index management.
6. **Retrieval** — dense, sparse/BM25, hybrid retrieval, metadata filtering, and query transformation.
7. **Reranking** — cross-encoder and other supported reranking strategies.
8. **Context building** — deduplication, token budgeting, ordering, and citation preservation.
9. **Generation** — LLM integration, prompt construction, grounded answers, and source citations.
10. **Evaluation** — datasets, metrics, retrieval evaluation, generation evaluation, and regression tests.
11. **Ingestion pipelines** — reusable end-to-end document ingestion workflows.
12. **Security** — input validation, path safety, tenant isolation patterns, secrets handling, and protection against document prompt injection.
13. **Observability** — structured logging, tracing, latency, token usage, and error reporting.
14. **Configuration** — validated settings and environment-based configuration.
15. **Integrations** — reusable adapters for supported frameworks, model providers, and vector stores.

Design these as modular components. Do not force every project to use every component or provider.

## 3. Python package and CLI

Create a Python package with a CLI named `ragkit`.

Required commands:
- `ragkit list` — list available components.
- `ragkit info <id>` — show component details, requirements, examples, and usage notes.
- `ragkit add <id>` — safely add a selected component to a destination project.

Use canonical component manifests and schemas so the CLI, tests, documentation, and cookbook website use consistent component metadata.

Validate manifests and destinations. Prevent unsafe paths, accidental overwrites, and path traversal. Explain conflicts clearly.

## 4. Local cookbook website

Build a local website using plain HTML, CSS, and JavaScript.

The website should have:
- A top navigation bar.
- A grouped, collapsible left sidebar.
- A home page explaining the library.
- Sections organized by the RAG pipeline.
- A dedicated page or view for each section.
- Component descriptions, code examples, dependencies, use cases, and trade-offs.
- A copy-code button.
- Search and filtering.
- Clear navigation and breadcrumbs.
- A light, clean design inspired by tutorial websites.

The website is for personal local use initially. It should work without requiring a hosted backend.

Generate the website's component data from canonical source files or manifests so documentation and metadata do not drift. Do not manually maintain duplicate component definitions.

## 5. Technical requirements

- Python 3.12 or newer.
- VS Code.
- `uv` for Python environment and dependency management.
- A modular Python package with clear interfaces.
- Pytest for automated testing.
- Explicit dependency declarations and reproducible-environment support.
- Useful type hints, docstrings, error handling, and readable code.
- Documentation and examples for each component.
- Unit tests, integration tests where appropriate, regression tests, and fixtures.
- Safe packaging and installation.
- A README with setup, commands, examples, architecture, and limitations.

Keep dependencies justified and avoid unnecessary complexity.

## 6. Quality requirements

Treat this as a production-oriented engineering project, not a collection of tutorial snippets.

Verify:
- Correctness and edge cases.
- Manifest and schema validation.
- File and path safety.
- Error handling and useful error messages.
- Dependency and configuration correctness.
- Test collection without import errors.
- Full test-suite results.
- Packaging and CLI behavior.
- Cookbook generation consistency.
- Documentation accuracy.
- Evidence for every claimed acceptance criterion.

Never claim that a test passed unless it actually ran. Distinguish failed tests, collection errors, skipped tests, and blocked checks.

Python 3.12 compatibility must remain unverified until tests are actually executed using Python 3.12.

## 7. My ownership rule for AI-assisted development

**I will write all substantive application implementation code myself.**

This includes the actual RAG algorithms and workflows, loaders, chunking logic, retrieval, reranking, generation, evaluation, CLI implementations, component generators, security logic, and other core application behavior.

AI may help me plan the architecture, create project structure, write supporting documentation, create tests and fixtures, define schemas and manifests, configure packaging, review my code, execute tests, diagnose errors, and verify requirements.

AI must not generate or delegate the generation of my substantive implementation code. It must not hide implementation logic in tests, fixtures, configuration, or generated artifacts.

When implementation code is missing, explain exactly what I need to write, where it belongs, what interface it must expose, and how I can verify it. Do not invent a substitute implementation.

## 8. Development workflow

Start from a clean project and work through the entire plan systematically.

1. Define requirements and acceptance criteria.
2. Design the architecture and package structure.
3. Create the project scaffold and supporting configuration.
4. Define component contracts, manifests, and schemas.
5. Prepare tests and fixtures.
6. Implement supporting tooling and the cookbook generation workflow.
7. Integrate the implementation code I write.
8. Run the relevant tests and verification checks.
9. Diagnose failures and fix issues within the permitted scope.
10. Complete documentation, packaging, and release-readiness checks.

Maintain a clear task list, dependencies, test evidence, blockers, and progress records. Use a project-management tool if useful, but do not make the project dependent on Genesis or any specific planning framework.

I want the workflow to proceed autonomously through all authorized tasks without repeatedly asking me for approval. If a task requires my implementation code, record the exact blocker and continue with genuinely independent work.

Do not commit, push, publish, install dependencies, or make consequential external changes without the appropriate authorization.

## 9. Expected deliverables

I want a complete, organized repository containing:
- Python package.
- `ragkit` CLI.
- Reusable RAG component library.
- Canonical manifests and validation schemas.
- Local cookbook website.
- Cookbook generation tooling.
- Tests and fixtures.
- Documentation and examples.
- Architecture and design decisions.
- Setup and usage instructions.
- Verification results and a list of outstanding limitations.

Do not create empty placeholder files merely to make the project look complete. Every claimed feature must have meaningful implementation, documentation, or explicitly recorded status.

## 10. First task

Before writing anything, inspect the current directory and confirm what exists after the previous Genesis setup was deleted.

Then propose a clean project structure, a complete implementation plan, task dependencies, and acceptance criteria for the entire project.

Show me the plan and identify which files are supporting artifacts versus substantive implementation code. Once the plan is established, execute authorized supporting work autonomously and request my code only where necessary.

My goal is to build a reusable, well-tested, maintainable RAG engineering library that I can use in real projects and eventually extend into a production-grade RAG toolkit.