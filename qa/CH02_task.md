# CH02 — Ligand-based Screening: Filtering & Similarity Searching

## Code / taxonomy
- **Broad `except:` (#8):** narrow the bare except around line 425 (re-confirm).
- Replace ad-hoc seeding with `bookutils.set_seed()`; style via
  `bookutils.setup_style()` + `bookutils.setup_rdkit_drawing()`.
- Sweep RDKit `None` guards (guard `MolFromSmiles`, filter before drawing).

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Ensure objectives, Chapter Summary, References; add post-figure interpretation
  cells; `bookutils.save_figure(..., "ch02")`.

## Acceptance
- `make execute-ch NN=02` clean; outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
