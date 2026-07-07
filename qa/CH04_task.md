# CH04 — Solubility Deep Dive with Linear Models

## Code / taxonomy
- Replace ad-hoc seeding with `bookutils.set_seed()`; style via
  `bookutils.setup_style()`.
- Taxonomy sweep: RDKit None guards, bare excepts, pandas `inplace` on slices.

## Pedagogy (priority: objectives gap)
- **CH04 has no learning objectives** — add a "This chapter covers" cell.
- Portable-markdown title; drop `<font color>` HTML.
- Chapter Summary, References; post-figure interpretation cells;
  `bookutils.save_figure(..., "ch04")`.

## Acceptance
- `make execute-ch NN=04` clean; outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
