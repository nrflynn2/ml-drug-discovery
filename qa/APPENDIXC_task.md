# Appendix C — Knowledge Distillation for Hierarchical Molecular Generation

Groups with the DL chapters (CH08) in the rollout.

## Environment
- Installs under `uv sync --extra advanced`.

## Code / taxonomy
- **Non-determinism (#5):** Appendix C has a weak seed stack — replace with
  `bookutils.set_seed()` (adds cudnn / `manual_seed_all` / `random` /
  `PYTHONHASHSEED`).
- `device = bookutils.get_device()`; style via `bookutils.setup_style()`.
- Sweep RDKit None guards and bare excepts.

## Pedagogy (priority: missing summary)
- **No Chapter Summary** — add one.
- Portable-markdown title; drop `<font color>` HTML; objectives; References;
  post-figure interpretation cells; `bookutils.save_figure(..., "appendix_c")`.

## Acceptance
- Executes clean (CPU-safe subset); outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
