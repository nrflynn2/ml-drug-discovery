# CH07 — Unsupervised Learning: Repurposing, Curating & Screening

## Code / taxonomy
- Replace ad-hoc seeding with `bookutils.set_seed()` (clustering/UMAP
  reproducibility matters here); style via `bookutils.setup_style()`.
- Taxonomy sweep: RDKit None guards, bare excepts, pandas `inplace` on slices.

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Objectives, Chapter Summary, References; post-figure interpretation cells
  (especially for UMAP/cluster plots — "what to look for"); 
  `bookutils.save_figure(..., "ch07")`.

## Acceptance
- `make execute-ch NN=07` clean; outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
