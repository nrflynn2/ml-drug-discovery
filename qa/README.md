# qa/ — Per-chapter modernization task specs

Each `CH0X_task.md` (and `APPENDIXC_task.md`) is the **warm-start prompt** handed
to the coding agent that modernizes that chapter. It pre-loads the chapter's
audit findings, pedagogy gaps, the cells to add, and acceptance criteria so the
agent starts warm, not cold. Read `../AGENTS.md` first for the shared standards.

- **Pilots (done this phase):** `CH01` (golden template), `CH09` (hard case).
- **CH12 package fixes** landed this phase with unit tests; `CH12_task.md`
  covers the remaining notebook polish.
- **Rollout order:** CH11 → CH10 → CH08 + AppendixC → CH02–CH07 → CH12 polish.

> Line numbers in these specs are approximate — they drift as notebooks are
> edited. Re-confirm each location when you open the chapter (`jupyter nbconvert
> --to script CHNN_FLYNN_ML4DD.ipynb` then grep the throwaway `.py`).

`qa/check_callouts.py` is the callout-leakage guard used by CI, pre-commit, and
`make callouts`.
