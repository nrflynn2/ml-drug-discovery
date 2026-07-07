# CH12 — Transformer Architectures  (CAPSTONE PACKAGE)

CH12 is the **intentional template exception**: a real `src/` + `scripts/`
Python package with unit tests, not a jupytext chapter.

## Landed this phase (package fixes + tests)
- **#1** `scripts/train.py` — `auc_score` NameError when `num_classes != 2`
  (initialize to `nan`).
- **#2** `src/data.py::mask_sequences` — build helper tensors on
  `input_ids.device` (GPU device-mismatch fix).
- **#4** `src/model.py::expand_mask` — return the 4-D mask instead of `None`
  (attention was silently left unmasked).
- **#6** `scripts/evaluate.py`, `scripts/tune.py` — `torch.load` now passes
  `map_location` (+ `weights_only=True` for the pure state_dict load).
- **#5** `src/utils.py::set_seeds` — strengthened (cudnn / `manual_seed_all` /
  `PYTHONHASHSEED`, default seed 42).
- **#10** `src/data.py::BCRDataset.__getitem__` — `.iloc` (non-reset index).
- Unit tests in `CH12_FLYNN_ML4DD/tests/` (run via `make test`), gated in CI.

## Remaining (future polish)
- Notebooks under `CH12_FLYNN_ML4DD/notebooks/`: portable-markdown headers,
  interpretation cells, Summary/References — align to the template spirit
  without dismantling the package structure.
- Consider regenerating the frozen `CH12_FLYNN_ML4DD/requirements.txt` from the
  3.12 environment (currently a legacy pip freeze).

## Acceptance
- `make test` green; ruff + black clean on `src/`, `scripts/`, `tests/`.
