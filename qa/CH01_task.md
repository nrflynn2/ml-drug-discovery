# CH01 — The Drug Discovery Process  (GOLDEN TEMPLATE PILOT)

Status: **piloted this phase.** CH01 is the reference implementation every other
chapter is measured against. Keep it exemplary.

**Done this phase:** portable-markdown headers (via `qa/modernize_headers.py`;
14 headers de-HTML'd). **Remaining (needs a deps-complete env to execute so
outputs regenerate cleanly):** the `bookutils` code changes, interpretation
cells, and the reproducibility check below — do these in one execution pass so
committed outputs stay consistent with the code.

## Role
Golden template: demonstrates the full `docs/CHAPTER_TEMPLATE.md` cell order and
`bookutils` usage end-to-end.

## Code / taxonomy
- Replace the ad-hoc seeding — the imports cell has a stray `np.random.seed(41)`
  (should be 42) — with a single `bookutils.set_seed()` call.
- Replace the local `setup_visualization_style()` cell with
  `bookutils.setup_style()`; keep the chapter-specific RDKit `drawOptions`
  (`rdkit_drawing_options`) cell — it is used downstream and is legitimately
  chapter-local.
- Swap any inline style setup for `bookutils.setup_style()`; use
  `bookutils.setup_rdkit_drawing()` for molecule rendering.
- Save every figure via `bookutils.save_figure(fig, "<name>", "ch01")`.
- Taxonomy sweep (RDKit None guards, bare excepts) — confirm clean.

## Pedagogy
- Portable-markdown title `# 📚 Chapter 1: …`; remove any `<font color>` HTML.
- Keep the existing objectives ("This chapter covers").
- Add a post-figure **interpretation cell** after each figure.
- Polish the Chapter Summary; ensure a References section exists.
- Standardize the environment cell to `bookutils.setup_environment("ch01", ...)`.

## Acceptance
- `make execute-ch NN=01` runs clean; seeded outputs reproduce across two runs.
- `make lint` green (callout guard + ruff + black).
- Reviewed with `nbdime`; committed with outputs.
