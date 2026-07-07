# AGENTS.md — Modernization Playbook

This repo is being modernized chapter-by-chapter for the Manning 3P review of
*Build AI Drug Discovery Pipelines*. **Each agent run modernizes exactly one
chapter**, warm-started from its `qa/CH0X_task.md` spec and gated by the `QA`
GitHub Actions workflow. This file is the master playbook: standards, the bug
taxonomy checklist, the definition of done, and the exact edit mechanics.

## Golden rules

1. **One chapter per run.** Read `qa/CH0X_task.md` for your chapter first. It
   lists the audit findings (with file references), the pedagogy gaps, the cells
   to add, and the acceptance criteria.
2. **Edit notebooks in place** (`.ipynb`) with the NotebookEdit tool / nbformat.
   There are **no `.py` mirrors** for CH01–CH11 — do not create them and do not
   introduce jupytext pairing. For grepping, use `jupyter nbconvert --to script`
   on demand and throw the output away.
3. **Keep committed outputs.** Readers and Colab want rendered outputs, so
   execute the notebook and commit it *with* outputs (don't strip them).
4. **Review your diff with nbdime** (`uv run nbdime diff <old> <new>`) before
   committing, so the change is legible despite in-place editing.
5. **Merge only when `QA` is green** (see "Definition of done").

## Environment

- Python **3.12 everywhere**. Single source of truth: root `pyproject.toml`.
- `make env` → `uv venv --python 3.12 && uv sync --extra advanced --extra dev`.
- Tiers map to extras: base=`core` (Ch 1-4), `advanced` (Ch 5-8 + AppC),
  `full` (Ch 10-11). **Only Chapter 9** needs conda (`ml4dd2025.yml`).
- `numpy` stays `<2` until the deferred numpy-2.x validation pass — do not bump.

## Standards every chapter must follow

### Use `bookutils`, never re-implement
`bookutils.py` is the one shared module. In every chapter:

```python
import bookutils
bookutils.set_seed()        # seed=42 everywhere: random/numpy/torch/cudnn/PYTHONHASHSEED
bookutils.setup_style()     # one house palette + figsize + DPI policy
device = bookutils.get_device()          # DL chapters
bookutils.setup_rdkit_drawing()          # cheminformatics chapters
bookutils.save_figure(fig, "name", "chNN")   # consistent PNG + PDF
```

Delete ad-hoc seed blocks, `setup_visualization_style()` variants, and
per-chapter device/figure helpers — they are replaced by the calls above.
`bookutils` also re-exports `save/load/list_molecular_dataframe`.

### Portable-markdown headers
Titles use `# 📚 Chapter N: …` (portable markdown). **Drop deprecated
`<font color>` HTML** — GitHub won't render it.

### Cell order
Follow `docs/CHAPTER_TEMPLATE.md` exactly. Mandatory sections: "This chapter
covers" (objectives), Chapter Summary, References, and **post-figure
interpretation cells** ("what this shows / what to look for").

## Bug taxonomy checklist (apply to every chapter)

Work the chapter's `qa/CH0X_task.md` backlog, then sweep for these classes:

1. **Callout leakage** — none exist today; the CI guard keeps it that way.
2. **RDKit `None`** — guard every `MolFromSmiles(...)`; filter `None` before
   drawing or calling `MolToSmiles`.
3. **Non-determinism** — replace every ad-hoc seeder with `bookutils.set_seed`.
4. **GPU/device** — build tensors on the input's device; never assume CPU.
5. **Broad `except:`** — narrow to the real exception; never silently drop data.
6. **Mutable default args** — use a `None` sentinel.
7. **Deprecated APIs** — e.g. RDKit `GetMorganFingerprint` →
   `rdFingerprintGenerator`; pandas `inplace=True` on slices → reassign.
8. **torch API** — `torch.load` needs `map_location` (+ `weights_only=True` for
   pure state_dicts on torch≥2.6).

## Definition of done (per chapter)

- [ ] Uses `bookutils` (`set_seed`, `setup_style`, device/figure helpers).
- [ ] Portable-markdown headers; no `<font>` HTML.
- [ ] Objectives + Summary + References present; interpretation cell after each
      figure.
- [ ] All backlog items in `qa/CH0X_task.md` fixed; taxonomy sweep clean.
- [ ] Executes end-to-end; seeded outputs reproduce across two runs.
- [ ] Committed **with** outputs; reviewed with `nbdime`.
- [ ] `QA` workflow green: callout guard + ruff + black + CH12 unit tests.
      As a chapter gains a reduced-config fixture, add it to the blocking
      `nbmake` matrix in `.github/workflows/qa.yml`.

## Handy commands

```bash
make lint                 # callout guard + ruff + black (check)
make callouts             # just the callout guard, on all notebooks
make test                 # CH12 unit tests
make execute-ch NN=09     # nbmake-execute a chapter
uv run python qa/modernize_headers.py CHNN_FLYNN_ML4DD.ipynb  # portable headers
uv run nbdime diff a.ipynb b.ipynb
```

## Rollout order (post-pilot)
CH11 (RDKit None-guards) → CH10 → CH08 + Appendix C (DL) → CH02–CH07
(cheminformatics) → CH12 polish. CH01 (golden template) and CH09 (hard case)
are the piloted references; measure new chapters against CH01.
