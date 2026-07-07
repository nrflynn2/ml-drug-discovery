# CH11 — Graph Neural Networks for Drug-Target Affinity

Recommended **first post-pilot chapter**: its bugs are confirmed cheap
one-liners and it has clear pedagogy gaps.

## Environment
- Installs under **uv/pip** (`uv sync --extra full`) — no conda needed.

## Code / taxonomy
- **RDKit None (#3):** three unguarded `MolToSmiles(MolFromSmiles(...))` calls
  around lines 1270 / 1338 / 2252 — add a `None` guard at each. Re-confirm.
- Replace ad-hoc seeding with `bookutils.set_seed()`; `device =
  bookutils.get_device()`; style via `bookutils`.

## Pedagogy (priority: objectives + summary gaps)
- **Objectives are an empty stub** — write real "This chapter covers" bullets.
- **No Chapter Summary** — add one.
- Portable-markdown title; drop `<font color>` HTML; References; post-figure
  interpretation cells; runtime/GPU callout; `bookutils.save_figure(..., "ch11")`.

## Acceptance
- `make execute-ch NN=11` clean (CPU-safe subset); outputs reproduce across two
  runs. `make lint` green; reviewed with `nbdime`; committed with outputs.
