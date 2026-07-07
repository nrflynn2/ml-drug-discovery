# CH05 — Classification: Cytochrome P450 Inhibition

## Code / taxonomy
- **Deprecated RDKit API (#12):** around line 1048 replace legacy
  `GetMorganFingerprint` with `rdFingerprintGenerator` (re-confirm line).
- **Broad `except:` (#8):** narrow the bare except around line 509.
- Replace ad-hoc seeding with `bookutils.set_seed()`; style via `bookutils`.

## Pedagogy
- Portable-markdown title; drop `<font color>` HTML.
- Objectives, Chapter Summary, References; post-figure interpretation cells;
  `bookutils.save_figure(..., "ch05")`.

## Acceptance
- `make execute-ch NN=05` clean; outputs reproduce across two runs.
- `make lint` green; reviewed with `nbdime`; committed with outputs.
