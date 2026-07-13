---
chapter: chXX
agent_model: claude-opus-4-8
run_date: YYYY-MM-DD
env_tier: core            # core | advanced | conda-ch9 | ch12
exec_tier: full           # full | smoke
verification:
  static_all_cells_parse: pass      # pass | fail
  imports_names_resolve: pass       # pass | fail
  execution: full                   # full | smoke | deferred
  execution_result: pass            # pass | fail | partial
  notebook_regenerated: true
  needs_full_gpu_run: false
inventory_summary:
  total: 0
  fixed: 0
  already_ok: 0
  not_found: 0
  needs_author_decision: 0
new_taxonomy_hits: 0
chapter_done: false
---

# Chapter XX — QA Report

_One-paragraph summary: overall state found, what was fixed, what needs the author._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| X.Y | BLOCKER | fixed / already-ok / not-found / needs-author-decision | evidence; what changed & why; `<sha>` |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| ... | ... | BUG / ROBUSTNESS | ... `<sha>` |

## Author-decision queue

```
Q1 (<area>): <crisp question>.
   Evidence needed: <what to inspect in data/notebook>. Blocks: <impact>.
```

## Verification log

- `python tools/validate_notebooks.py CHXX_FLYNN_ML4DD.ipynb` → _paste result_
- `jupytext --sync CHXX_FLYNN_ML4DD.py` → _ok?_
- execution: `jupyter nbconvert --to notebook --execute --inplace CHXX_FLYNN_ML4DD.ipynb` → _ok / smoke notes_
