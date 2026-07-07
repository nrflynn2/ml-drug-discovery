# CH06 — Case Study: Small Molecule Binding to an RNA Target

## Code / taxonomy
- **pandas (#13):** around lines 1709/1710/1727, `inplace=True` on slices —
  reassign instead (`df = df.assign(...)` / `df[col] = ...`). Re-confirm lines.
- Replace ad-hoc seeding with `bookutils.set_seed()`; style via `bookutils`.
- Sweep RDKit None guards and bare excepts.

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Objectives, Chapter Summary, References; post-figure interpretation cells;
  `bookutils.save_figure(..., "ch06")`.

## Acceptance
- `make execute-ch NN=06` clean; outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
