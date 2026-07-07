# CH03 — Ligand-based Screening: Machine Learning

## Code / taxonomy
- **RDKit None (#9):** around line 285 a `MolFromSmiles` list may contain `None`
  that is drawn downstream — filter `None` before drawing (re-confirm line).
- **Broad `except:` (#8):** narrow the bare except around line 1058.
- Replace ad-hoc seeding with `bookutils.set_seed()`; style via `bookutils`.

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Objectives, Chapter Summary, References; post-figure interpretation cells;
  `bookutils.save_figure(..., "ch03")`.

## Acceptance
- `make execute-ch NN=03` clean; outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
