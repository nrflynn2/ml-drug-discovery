# CH09 — Structure-based Drug Design with Active Learning  (HARD-CASE PILOT)

Status: **piloted this phase.** Validates the conda-separate environment path
end-to-end and carries real code issues.

**Done this phase:** portable-markdown headers (via `qa/modernize_headers.py`;
17 headers de-HTML'd). **Remaining (needs the conda env to execute so outputs
regenerate cleanly):** the code fixes and pedagogy cells below — do these in one
execution pass so committed outputs stay consistent with the code.

## Environment (the reason CH09 is the hard pilot)
- The **only** chapter needing conda: `openmm`, `pdbfixer`, `vina` (+
  `mdtraj`/`prolif`/`meeko`/`openbabel`). Supported path: `conda env create -f
  ml4dd2025.yml`.
- **Excluded from CI** (`nbmake` skips CH09; no conda/docking on the runner).
- Colab: verify the **condacolab + `LD_LIBRARY_PATH` patch** flow still works.

## Code / taxonomy
- **OOM / GPU (#11):** the acquisition loop moves the *whole* candidate pool to
  the device each iteration (around lines 968/1035/1155/1190/1243 — re-confirm).
  Batch the acquisition tensor instead of `.to(device)` on the full pool.
- **Non-determinism (#5):** replace the weak seeder with `bookutils.set_seed()`.
- **Broad `except:` (#8):** narrow the bare excepts (around lines 1345/2111).
- Taxonomy sweep for RDKit None guards.

## Pedagogy
- Portable-markdown headers; drop `<font color>` HTML.
- Add objectives ("This chapter covers"), a Chapter Summary, References, and
  post-figure interpretation cells.
- Environment cell → `bookutils.setup_environment("ch09", tier="conda")`.

## Acceptance
- In the conda env: non-docking cells run clean; metrics identical across two
  runs (determinism); no OOM with the batched pool; spot-run one docking cell.
- Colab condacolab flow verified.
- `make callouts` + ruff/black green. Reviewed with `nbdime`; committed with
  outputs.
