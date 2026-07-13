# QA Report-Back Schema (MANDATORY)

Every chapter agent produces exactly one `qa_reports/CHXX_report.md` in this format. The rigid
structure is what lets results **roll up uniformly** across chapters so the reviewer inventory becomes a
*living coverage checklist* (`COVERAGE.md`) instead of a static spec. Do not improvise **frontmatter
fields or table columns**.

A chapter report has four required parts: **(A)** YAML frontmatter, **(B)** Table 1 — inventory
coverage, **(C)** Table 2 — new taxonomy hits, **(D)** the `needs-author-decision` escalation queue.
Prose sections are expected and encouraged around them: a one-paragraph summary, a short **Pedagogy
changes** list (Standard-depth edits that are neither inventory nor taxonomy items), and a
**Verification log** (the exact harness / sync / execute commands + their output). See `REPORT_TEMPLATE.md`.

---

## A. YAML frontmatter

```yaml
---
chapter: ch08                 # ch01..ch12 | appendixC
agent_model: claude-opus-4-8  # model that ran the pass
run_date: 2026-07-08          # ISO date
env_tier: advanced            # core | advanced | conda-ch9 | ch12
exec_tier: full               # full | smoke   (from the roster)
verification:
  static_all_cells_parse: pass    # pass | fail   (tools/validate_notebooks.py)
  imports_names_resolve: pass     # pass | fail
  execution: full                 # full | smoke | deferred
  execution_result: pass          # pass | fail | partial
  notebook_regenerated: true      # did `jupytext --sync` refresh the .ipynb?
  needs_full_gpu_run: false       # true when smoke-tier subsetted heavy cells
inventory_summary:
  total: 8                        # rows in Table 1
  fixed: 3
  already_ok: 3
  not_found: 1
  needs_author_decision: 1
new_taxonomy_hits: 2              # rows in Table 2
chapter_done: false              # see "Definition of done" below
---
```

**Field rules**
- `env_tier` / `exec_tier` come from the chapter roster in the plan; do not change them unilaterally.
- `execution: deferred` is only valid for `exec_tier: smoke` chapters where heavy cells were not run;
  it **requires** `needs_full_gpu_run: true`.
- The five `inventory_summary` counts must sum to `total`. Every inventory item lands in exactly one of
  the four terminal statuses.
- `chapter_done` is `true` **iff** all of: `static_all_cells_parse: pass`, `imports_names_resolve: pass`,
  execution passed at the chapter's tier, `notebook_regenerated: true`, **and** every Table-1 row is
  terminal with no *unresolved* `needs-author-decision`. (An open author question keeps a chapter
  `chapter_done: false` until Wave 2 resolves it.)

---

## B. Table 1 — Inventory coverage (one row per inventory item)

Slice the chapter's items from `code_listing_feedback_for_qa.md`. One row per `(listing, tag)` item.

| listing | tag | status | note |
|---------|-----|--------|------|
| 8.1 | BLOCKER | already-ok | `cude`/`2,0048` typos absent; `smiles` used consistently in current script |
| 8.2 | BUG | fixed | built `(smiles, activity, mol)` tuples so invalid-SMILES filtering keeps label alignment — `<sha>` |
| activity target | BUG | needs-author-decision | is `activities.standard_value` pIC50 or raw IC50(nM)? → escalation Q1 |
| 8.4 | CONSISTENCY | fixed | code `>= 6.3` vs text `> 6.3`; changed code to `> 6.3` to match prose — `<sha>` |

**Status decision tree (apply consistently):**
- **`fixed`** — the defect was present in the current `.py`; the agent changed code to resolve it.
  Note must say *what* changed and cite the commit.
- **`already-ok`** — the described defect is **not present** in this snapshot; current code is correct.
  (Use this for marker/`[CA]` items when the notebook is clean — and add "flag manuscript listing for
  production" to the note, since the artifact may still live in the book's AsciiDoc.)
- **`not-found`** — the referenced listing/identifier can't be located and can't be mapped to a current
  equivalent (renamed/removed/renumbered). Note must record *what was searched*.
- **`needs-author-decision`** — located, but the resolution depends on **data semantics**,
  **prose/figure/caption reconciliation**, or a **scientific/pedagogical judgment** the agent must not
  make alone. Every such row references a numbered question in section D.

**Two special cases that map onto the four (do not invent a fifth status):**
- **Optional / declined `ENHANCEMENT`** (a nicety you choose not to implement, e.g. "add count-based
  Morgan fingerprints") → **`already-ok`** with a note beginning "optional, deferred:". These must
  **never** be `needs-author-decision` and must **never** block `chapter_done`.
- **Manuscript-only artifact** (an item whose generating code is not in the notebook/repo — see the
  brief's cross-cutting findings) → **`needs-author-decision`** asking "add the code to the notebook, or
  is this manuscript-only?"; record the repo-wide search in the note.

**Commit citations:** notes cite the short sha of the chapter commit; in a no-commit pilot run, write
`(uncommitted)` instead.

---

## C. Table 2 — New taxonomy hits (found by the proactive sweep, NOT in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| `calculate_enrichment_factors` L957 | mutable default arg | BUG | default `ef_percentages=[...]` → `None`, set inside — `<sha>` |
| `smiles_to_fingerprint` bare `except:` L1119 | broad exception | BUG | → `except Exception as e:` with a message — `<sha>` |

Taxonomy classes (use these labels): `marker/non-parsing`, `import/undefined`, `mutable-default`,
`broad-except`, `rdkit-none-guard`, `pandas-indexing`, `gpu-memory`, `nondeterminism`, `deprecated-api`,
`shape/offbyone/path`, `headless-execution`.

A finding you deliberately did **not** fix (an env-nondeterministic output, a change too risky to make,
a hardcoded value the pipeline actually tunes) belongs here too, with action **"Observation, not fixed"**
+ the reason. It does not block `chapter_done`.

---

## D. `needs-author-decision` escalation queue

Numbered, each a crisp question + exactly what evidence is needed to resolve it. These pool into
`COVERAGE.md` for the author (Wave 2).

```
Q1 (Ch8 activity target): Is `activities.standard_value` already pIC50, or raw IC50 in nM?
   Evidence needed: the loaded ChEMBL frame's `standard_type` / `standard_units`; if raw nM,
   apply pIC50 = 9 - log10(IC50_nM). Blocks: label correctness for the whole chapter.
```

---

## E. Rollup mechanics

- The orchestrator appends each report's frontmatter summary to `qa_reports/COVERAGE.md` and flips that
  chapter's per-item statuses from `pending` to the reported terminal status.
- A chapter shows **done** in `COVERAGE.md` only when `chapter_done: true`.
- Open `needs-author-decision` questions are collected into the COVERAGE "Author queue" until resolved.
