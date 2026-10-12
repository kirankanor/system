# Discovery audit — repository state (2026-10-09)

Read-only audit performed during Genesis discovery phase. Evidence: line counts, direct file reads, Genesis adoption discovery report, and executed test commands. Classification legend: **implemented** / **placeholder** (single comment line) / **generated** / **docs-planning** / **infra**.

## 1. Classification by area

| Area | Classification | Evidence |
|---|---|---|
| `components/config/` | **implemented** (only real Python) | 7 files, 72–128 lines each + 267-line test; imports `pydantic`, `pydantic_settings`; **41/41 tests pass** (`python -m pytest components/config -q`, exit 0) |
| `components/` (other 30 components) | **placeholder** | every `.py` (impl, `example.py`, `tests/test_component.py`) and every `README.md` is exactly 1 comment line |
| `src/rag_code_library/` (21 files) | **placeholder** | all 1 comment line; no `ragkit` CLI, registry, generator, or security logic exists |
| `scripts/` (3 files) | **placeholder** | all 1 comment line (`build_cookbook.py`, `validate_components.py`, `verify_generated_code.py`) |
| `tests/unit/`, `tests/integration/` (7 files) | **placeholder** | all 1 comment line; `tests/fixtures/` contains only placeholder READMEs, no fixture data |
| `cookbook/` | **implemented static site** + **generated data** | `index.html` (12 lines, complete), `script.js` (14 dense lines, full search/nav/tabs/copy app), `style.css` (1 line, complete CSS); `data.js` self-labels "Generated placeholder data" |
| `schemas/component-manifest.schema.json` | **implemented** | 68-line valid JSON Schema (draft 2020-12), required fields + id pattern + category enum |
| component `manifest.json` × 30 | **implemented (data)** | real entries, e.g. `storage-adapters` with `destination: src/{project_package}/rag/...` |
| `pyproject.toml` | **placeholder** | 1 comment line: no metadata, no deps, no build backend, no `ragkit` entry point |
| `uv.lock` | **placeholder (generated)** | 1 comment line, self-declared invalid |
| `.github/workflows/*.yml` (2) | **placeholder** | 1 comment line each |
| `docs/` (6), `examples/` (3), component READMEs, root `README.md` | **placeholder** | 1 line each (only exception: `components/config/README.md`, 47 lines) |
| `MASTER_PLAN.md` | **docs-planning, human-supplied** | 1,087+ lines; did not exist at session start; Genesis `events.jsonl` records only `AGENTS.md`/`CLAUDE.md` as agent-connect outputs — **not** Genesis-generated. Contains full target architecture, 11-stage implementation order, DoD |
| `AGENTS.md`, `CLAUDE.md` | **infra, Genesis-generated** | `agent connect --write`, 22 lines each, `genesis:start` markers |
| `.genesis/` | **infra, Genesis-generated** | `project.json`, `KICKOFF.md`, index, dashboard; phase `discovery/ready` |
| `code/` | **empty inbox** | 0 files — no human-supplied implementation pending |

## 2. Verification of `SCAFFOLD_NOTES.md` claims

| Claim | Verdict |
|---|---|
| "Most Python files are intentionally comment-only placeholders" | **Confirmed** — 124 of 131 `.py` files are exactly 1 comment line; the 7 non-trivial files are all `components/config` |
| "cookbook … includes searchable component navigation and copy controls" | **Confirmed in source** (`script.js` implements search, nav, tabs, clipboard copy) — not executed in this audit (no browser run) |
| "cookbook/data.js is placeholder data … build script not implemented yet" | **Confirmed** — file self-labels as placeholder; `scripts/build_cookbook.py` is a 1-line placeholder |
| "uv.lock is a placeholder; run `uv lock` after completing pyproject.toml" | **Confirmed** — 1 comment line |
| "not an operational CLI" | **Confirmed** — `src/` has zero implementation; `ragkit` entry point does not exist |
| "Empty planned component directories contain no fake implementation files" | **Confirmed** — no fake code; placeholders are explicit comment stubs |

## 3. Executed checks (actual results)

| Check | Command | Result |
|---|---|---|
| Declared test command (full) | `python -m pytest` | **FAILED — exit 2.** Collected 41 items, **28 collection errors**: duplicate test module basename `tests/test_component.py` (28 components) without package markers → `import file mismatch`. Zero tests executed. |
| Scoped real suite | `python -m pytest components/config -q` | **PASSED — exit 0.** 41 passed in 0.96s |
| Dependency presence | `python -c "import pydantic, pydantic_settings"` | **PASSED** — pydantic 2.13.4, pydantic_settings 2.14.2 installed in environment (but **not declared** anywhere) |
| Environment | `python --version`, `git --version`, `node --version` | Python 3.14.5, Git 2.53.0, Node 24.11.1 |
| Genesis read-only adoption | `genesis adopt .` | exit 0, 221 files, test cmd recorded `python -m pytest` (*unverified* at the time) |

Not run (not authorized/not applicable): lint, type checkers (none configured — no implementation to lint), cookbook browser test, `uv lock` (pyproject incomplete by design).

## 4. Findings

- **F-1 (blocker):** Full test collection is broken (AC-1 violated at baseline). Root cause: identical `test_component.py` basenames in 28 non-package `tests/` dirs. Fix must be human-supplied (pytest config in `pyproject.toml`, `__init__.py` markers, or unique names) — Q-5.
- **F-2 (blocker):** `pyproject.toml` is a placeholder: package cannot install, `ragkit` entry point missing, dependencies (`pydantic`, `pydantic-settings`, `pytest`) undeclared, pytest configuration absent (NFR-1/FR-1/NFR-4 violated).
- **F-3:** `uv.lock` invalid; both CI workflows are comment stubs; root README and all `docs/` are placeholders.
- **F-4:** Working tree had no Git repo at session start; now initialized (commit `1e541d4`, clean).
- **F-5:** CLI naming discrepancy `ragkit` vs `raglib` (SPEC Q-1).
- **F-6:** Security modules (`src/rag_code_library/security/path_safety.py`, `manifest_validation.py`) are placeholders — NFR-3 has **no** implementation behind it yet.
- **F-7:** Cookbook source is functional but never validated against the live manifest set in this audit; `data.js` is generated placeholder content (AC-6 at risk until build script exists).

## 5. Operational limitations (current)

- No installable package, no CLI, no registry/generator logic, no runnable scripts, no CI, no docs, empty `code/` inbox.
- Only verified working capability: `components/config` (41 passing tests) and the static cookbook source files (unexecuted in this audit).
