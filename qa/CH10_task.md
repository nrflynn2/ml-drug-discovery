# CH10 — Generative Models for De Novo Design

## Environment
- Installs under **uv/pip** (`uv sync --extra full`) — no conda needed. Correct
  any lingering "requires conda" wording in-chapter.

## Code / taxonomy
- **Broad `except:` (#8):** around line 3493 a bare except **silently drops
  SMILES** — this is the most impactful except in the book; narrow it and log or
  keep the rejects. Also narrow the bare except around line 5016. Re-confirm.
- Replace ad-hoc seeding with `bookutils.set_seed()`; `device =
  bookutils.get_device()`; style via `bookutils`.
- Sweep RDKit None guards (generated SMILES frequently parse to `None`).

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Objectives, Chapter Summary, References; post-figure interpretation cells;
  runtime/GPU callout; `bookutils.save_figure(..., "ch10")`.

## Acceptance
- `make execute-ch NN=10` clean (CPU-safe subset); outputs reproduce across two
  runs. `make lint` green; reviewed with `nbdime`; committed with outputs.
