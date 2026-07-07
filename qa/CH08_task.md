# CH08 — Introduction to Deep Learning

## Code / taxonomy
- **Mutable default args (#7):** around lines 957/1021, list default arguments —
  switch to a `None` sentinel (`def f(x=None): x = x or []`). Re-confirm lines.
- **Broad `except:` (#8):** narrow the bare except around line 1119.
- Replace ad-hoc seeding with `bookutils.set_seed()`; `device =
  bookutils.get_device()`; style via `bookutils.setup_style()`.

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Objectives, Chapter Summary, References; post-figure interpretation cells;
  a runtime/GPU expectation callout; `bookutils.save_figure(..., "ch08")`.

## Acceptance
- `make execute-ch NN=08` clean (CPU-safe subset); outputs reproduce across two
  runs. `make lint` green; reviewed with `nbdime`; committed with outputs.
