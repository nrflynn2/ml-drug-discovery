---
chapter: ch05
agent_model: claude-opus-4-8
run_date: 2026-07-13
env_tier: advanced
exec_tier: full
verification:
  static_all_cells_parse: pass
  imports_names_resolve: pass
  execution: full
  execution_result: pass
  notebook_regenerated: true
  needs_full_gpu_run: false
inventory_summary:
  total: 4
  fixed: 2
  already_ok: 1
  not_found: 0
  needs_author_decision: 1
new_taxonomy_hits: 7
chapter_done: false
---

# Chapter 05 — QA Report

_CH05 (CYP inhibition classification) passed the **static** harness at baseline (50 code cells, 0 errors)
but **could not actually execute end-to-end** under the installed stack — and neither blocker was visible
to a parse-only check. Two independent APIs had been removed out from under the chapter: (1) RDKit 2025.9
deleted the `dotsPerAngstrom` draw option, so `setup_rdkit_drawing()` raised
`AttributeError: Cannot set unknown attribute 'dotsPerAngstrom'` in the **third code cell** (the same defect
the author already fixed in CH07 at `8178f5e`, never back-ported to CH05); and (2) scikit-learn 1.9 removed
`cv="prefit"` outright — it is no longer merely deprecated but raises `InvalidParameterError` — which killed
the entire §5.2 calibration section. Both are fixed (`dotsPerAngstrom` dropped; `cv="prefit"` →
`FrozenEstimator`, the sanctioned ≥1.6 replacement and numerically equivalent, so the committed figures are
preserved). The good news the calibration flagged as the chapter's biggest risk — **scikit-multilearn — is a
non-issue**: `IterativeStratification` and `iterative_train_test_split` both import and run clean under
scikit-learn 1.9 / numpy 2.2, so no pin and no reimplementation were needed. The 5.4 US/DS label bug was real
and is fixed, and the `radius` keyword is already consistent. The one item I did **not** resolve is the §5.2
calibration methodology (Q1): the fix is real, but my numbers show it would **flip which calibrator the
chapter crowns best** (isotonic wins on the fit set, Platt wins on held-out data) — a pedagogical claim I must
not overturn alone, so `chapter_done: false`._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| 5.2 (calibration) | BUG | needs-author-decision | Confirmed live: `apply_calibration` fits the Platt/isotonic calibrators on `(valid_fingerprints, valid_df_processed.Inhibitor)` and the two reliability-diagram cells then evaluate those same calibrators on `valid_fingerprints` — fit-and-score on one set. I did **not** silently "fix" it, because measuring it shows the fix changes the chapter's conclusion, not just its numbers. Fitting on valid and scoring on the truly held-out **test** set (the reviewer's own recommendation): uncalibrated Brier .1654 / RMSCE .0882 → **Platt** .1570 / **.0302** → **isotonic** .1591 / **.0471**. So calibration still helps (narrative safe), but **Platt beats isotonic on held-out data while isotonic beats Platt on the fit set** (valid RMSCE: isotonic .0303 < Platt .0318). The buggy eval doesn't just inflate scores — it inverts the Platt-vs-isotonic ranking, which is exactly the kind of claim §5.2's prose/figure is likely built on. → Q1. (uncommitted) |
| 5.2 (calibration) | ROBUSTNESS | fixed | Not merely "version-dependent" — **hard-removed**. Under sklearn 1.9 `CalibratedClassifierCV(model, cv="prefit")` raises `InvalidParameterError: The 'cv' parameter ... Got 'prefit' instead`, so the whole §5.2 section was dead. Migrated to the sanctioned ≥1.6 replacement: `CalibratedClassifierCV(FrozenEstimator(model), method=...)` (added `from sklearn.frozen import FrozenEstimator`). Semantics are identical (calibrator trains on the passed set; base model is not refit), so the committed reliability figures are reproduced, not perturbed. Verified both `method='sigmoid'` and `method='isotonic'`. This fix is independent of Q1 — it restores execution without taking a position on the eval-set question. (uncommitted) |
| 5.4 (resampling) | BUG | fixed | Real, and exactly as described. The *models* are built correctly (`dt_downsampled` ← all positives + equal negatives sampled **without** replacement = majority reduced; `dt_upsampled` ← positives resampled **with** replacement + all negatives = minority grown), but the PR-AUC print labeled the **downsampled** model `"OS DT"` ("oversampled" — a synonym for *up*sampling), directly contradicting the `"US DT"` line above it. Relabeled to `"DS DT"`, so the pair now reads US = upsampled / DS = downsampled, matching both the model variables and the section's own "1. Downsampling … 2. Upsampling" markdown. Label-only change; no metric moves. (uncommitted) |
| fingerprint helper `radius`/`r` | CONSISTENCY | already-ok | No inconsistency exists. There is exactly one helper, `compute_fingerprint(mol, radius=2, nBits=2048)`, which forwards to `GetMorganGenerator(radius=radius, fpSize=nBits)`. `grep` finds **no** `r=` keyword anywhere in the chapter; the two call sites pass positionally/by default (`compute_fingerprint(x)` and `compute_fingerprint(m, 2, 2048)`), so neither can disagree with the signature. The Summary prose ("Morgan fingerprints with radius 2 and 2048 bits") also agrees. Nothing to change. (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| `setup_rdkit_drawing()` — `dopts.dotsPerAngstrom = 100` | deprecated-api | **BLOCKER** | **The single most important find.** RDKit 2025.9 removed the option: setting it raises `AttributeError: Cannot set unknown attribute 'dotsPerAngstrom'`. This is the *third code cell*, so a fresh-kernel run died before loading any data — yet the static harness passed, because the attribute only fails at runtime. Removed the line (the remaining draw options are unaffected). The author already made this exact fix in CH07 (`8178f5e`, "Remove deprecated dotsperangstrom") but it was never back-ported here. **Worth grepping the other chapters.** (uncommitted) |
| `reliability_diagram()` — bare `except:` around `ax.errorbar(...)` | broad-except | BUG | The taxonomy hit the brief predicted. Bare `except:` also swallows `KeyboardInterrupt`/`SystemExit` and hid *why* the error bars failed. → `except Exception as e:` and surfaced the cause in the fallback message (`f"Warning: Error bars could not be plotted ({e})"`). (uncommitted) |
| Import cell — `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)` | nondeterminism | ENHANCEMENT (minor) | → `RANDOM_SEED = set_seed(42)` (shared util seeds Python + NumPy + `PYTHONHASHSEED`). Same value 42, and it still seeds the global NumPy RNG that `handle_class_imbalance`'s `np.random.choice`/`permutation` rely on, so the resampling results are unchanged and every downstream `random_state=RANDOM_SEED` keeps working. (uncommitted) |
| 5× `display(...)` (df/molecule previews) with no import | import/undefined | ROBUSTNESS | `display` resolved only as an IPython-injected builtin (fine in-kernel, `NameError` as a plain script). Added `from IPython.display import display`, matching the CH02/CH03/CH04 treatment. (uncommitted) |
| `adversarial_validation()` — `train['label'] = 0` / `test['label'] = 1`; unseeded `train_test_split` + `RandomForestClassifier()` | pandas-indexing / nondeterminism | ROBUSTNESS | The function **mutated its caller's DataFrames**, permanently adding a `label` column to `train_valid_df_processed` and `test_df_processed` (benign today only because nothing reads them afterwards — a latent trap if a cell is ever added below). Now operates on `.copy()`s. Also seeded `train_test_split(..., random_state=RANDOM_SEED)` and `RandomForestClassifier(random_state=RANDOM_SEED)`, which were the chapter's only genuinely unseeded estimators — the adversarial-validation PR AUC previously changed on every run. Dropped a dead `pred = ...` line. (uncommitted) |
| Multilabel section — `LogisticRegression(..., n_jobs=-1)` | deprecated-api | ROBUSTNESS | Surfaced only by *running* the notebook (5× `FutureWarning`, invisible to the harness): `'n_jobs' has no effect since 1.8 and will be removed in 1.10`. So this was already a silent no-op **and** a scheduled hard failure on the next sklearn bump — the same trap as `cv="prefit"`, one release earlier in its lifecycle. Rather than just delete it, I preserved the author's evident intent to parallelize by moving `n_jobs=-1` to `MultiOutputClassifier(base_lr, n_jobs=-1)`, where it genuinely does fit the 5 CYP labels in parallel. Re-ran: warning gone, and every metric is byte-identical (Average LR PR AUC 0.788), confirming the parameter had been doing nothing. (uncommitted) |
| Multilabel loader, cache-miss `else:` branch — `merge(df, on='Inchi')` | shape/offbyone/path | ROBUSTNESS | **Observation, not fixed.** This branch is dead whenever `artifacts/ch05/cyp_inhibitor_df.pkl` exists (it does, and it is what the chapter ships), so I could not execute or verify any change to it — but it is broken in **three** compounding ways and a token fix would not make it run: (a) it merges `on='Inchi'` while the column it created is named `'InchiKey'` → `KeyError`; (b) that `'InchiKey'` column is actually populated with `MolToInchi(m)`, i.e. a full **InChI string, not an InChI key**; and (c) even with the key corrected, the per-isoform frames all carry a `Molecule` column, so after the first merge it becomes `Molecule_x`/`Molecule_y` and the final `[['Molecule', ...]]` selection would `KeyError` again. Fixing this properly is a rewrite of the fallback path (and a naming decision) that belongs to the author; deliberately left alone rather than fabricate an unverifiable fix. Does not block `chapter_done` — the shipped path is the cached one. (uncommitted) |

## Author-decision queue

```
Q1 (Ch5 §5.2 calibration — evaluate the calibrators on which set?): The calibration BUG is real:
   apply_calibration() fits Platt/isotonic on the validation set and the reliability-diagram cells
   score them on that same validation set, so the reported calibration quality is optimistic.
   The reviewer's prescribed fix (fit on valid = calibration set, score on the held-out test set) is
   a ~4-line change I have deliberately NOT made, because it would change a chapter CLAIM, not just
   the digits on a figure. Measured (LR + Morgan 2048, sklearn 1.9):

                 Brier(valid)  Brier(test)  RMSCE(valid)  RMSCE(test)
     uncalibrated    0.1616       0.1654       0.0865        0.0882
     Platt           0.1558       0.1570       0.0318        0.0302
     isotonic        0.1508       0.1591       0.0303        0.0471

   Two readings: (i) GOOD — calibration genuinely works; both methods beat uncalibrated on held-out
   test, so §5.2's core message survives the fix. (ii) THE CATCH — on the fit set isotonic looks best
   (RMSCE .0303 vs Platt .0318), but on held-out data that INVERTS and Platt wins decisively
   (.0302 vs .0471): isotonic is overfitting the 1,232-row calibration set, which is textbook
   isotonic behavior on small calibration sets. So if §5.2's prose/figure concludes anything like
   "isotonic gives the best-calibrated probabilities," the honest fix REVERSES that conclusion.
   Decision needed: (a) adopt the held-out-test evaluation and update the §5.2 figure + any
   isotonic-vs-Platt claim (my recommendation — the overfitting story is a *better* lesson and is
   exactly the "questioning probabilistic output assumptions" point the section is named for); or
   (b) keep evaluating on the calibration set and add an explicit caveat that the numbers are
   optimistic; or (c) split valid into calibrate/score halves (removes the double-dip but halves an
   already-small calibration set).
   Evidence needed: the §5.2 prose + the caption/claim attached to figures/ch05/calibration_comparison.png.
   Blocks: the 5.2 BUG row; keeps chapter_done=false. (Code currently RUNS and reproduces the
   committed figures — the cv="prefit" removal was fixed independently — so nothing is broken while
   this is open.)
```

## Pedagogy changes (Standard depth)

- **Standardized setup cell:** added `check_env(["numpy","pandas","scipy","sklearn","rdkit","matplotlib","seaborn"])`
  (chapter-tailored — no torch/xgboost, which this chapter never imports) + `RANDOM_SEED = set_seed(42)`
  replacing the raw `np.random.seed`, plus a one-line CPU/runtime note.
- **Learning objectives / takeaways:** already present as the "This chapter covers" bullets and the
  "Summary" section — kept as the pre-existing equivalents, not duplicated.
- **`preview_df` + asserts:** added `preview_df(train_df_processed, "train (processed + Morgan fingerprints)")`
  after the SMILES→mol→fingerprint transform (the chapter's one major DataFrame transform), guarded by two
  light asserts: `train_fingerprints.shape == (len(train_df_processed), 2048)` and that `Inhibitor ⊆ {0,1}`.
- **Seed consistency:** `base_rf` used a hardcoded `random_state=42` and `base_lr` had none; both now use
  `RANDOM_SEED` so the chapter references one seed constant throughout.
- **Colab cells:** tagged the 5 "Colab users only" cells `skip-execution`. Two of them were outright fatal
  locally (`condacolab.install()` / `condacolab.check()` are *uncommented*, and one cell is
  `os.kill(os.getpid(), 9)` — a kernel suicide). Checked the skip-tag fallout per the brief: `os` is also
  imported in the main import cell, and `CHAPTER` is used only inside the skipped cell, so nothing downstream
  goes undefined; `figures/ch05/` is committed, so the skipped `makedirs` calls aren't needed for `savefig`.
- Preserved the visual identity (`#A20025` headers, emoji section banners); no restructuring.

## Verification log

- `uv run python tools/validate_notebooks.py CH05_FLYNN_ML4DD.ipynb` → `✓ 50 code cells OK — 0 errors, 0 warnings`
  (same 50 at baseline; pedagogy lines landed inside existing cells). **Note the harness passed at baseline too
  — it is a parse-only check and cannot see the two runtime API removals that made the chapter unrunnable.**
- `uv run jupytext --sync CH05_FLYNN_ML4DD.py` → ok (benign "Notebook is not trusted" warning only); `.ipynb`
  regenerated from the edited `.py`, with `tags=["skip-execution"]` on the 5 Colab cells.
- execution: `uv run python tools/execute_notebook.py CH05_FLYNN_ML4DD.ipynb --timeout 1800` → **pass** —
  `OK: executed 45 cells, skipped 5 ('skip-execution'), 0 errors`. Post-run JSON inspection: 50 code cells,
  45 with an `execution_count`, 5 skip-tagged, **0 error outputs**. (Before the fixes this run died in the
  *third* cell on `dotsPerAngstrom`, and again in §5.2 on `cv="prefit"`.)
- Compatibility probes run before touching code: **`skmultilearn` is fine** —
  `IterativeStratification` *and* `iterative_train_test_split` both import **and run** clean under
  scikit-learn 1.9 / numpy 2.2 (the multilabel stratified split executes end-to-end in the full run), so **no
  pin and no `MultiOutputClassifier`/`OneVsRestClassifier` reimplementation was needed** and no results moved;
  `MolDraw2DCairo` + `SimilarityMaps.GetSimilarityMapForModel` work headless;
  `CalibratedClassifierCV(FrozenEstimator(m))` verified for both sigmoid and isotonic.
- Sanity of outputs: `check_env` → Python 3.12.13 (CPU only) / numpy 2.2.6 / pandas 2.3.3 / sklearn 1.9.0 /
  rdkit 2025.09.6. Data: train 8,629 / valid 1,232 / test 2,467 compounds, 0 invalid molecules;
  `preview_df` → `8,629 rows x 5 cols`. **§5.4 resampling now reads correctly: `US DT` 0.714 (upsampled) /
  `DS DT` 0.721 (downsampled)** alongside LR 0.813, DT 0.728, Weighted DT 0.740 — and the overfitting point
  the prose makes still lands (DT *train* PR AUC ≈ 1.000 vs 0.728 validation). RF: validation 0.834, OOB 0.799,
  **test 0.827**; applicability domain in-/out-of-domain 0.947 / 0.727 at 34.5% coverage; adversarial
  validation 51.6% (≈ chance, i.e. train/test are not trivially separable — the intended result).
  Multilabel (skmultilearn split): Average LR PR AUC 0.788, Average RF PR AUC 0.801. All `figures/ch05/*`
  refreshed. Remaining warnings are benign (`UndefinedMetricWarning` for a zero-prediction label in the
  multilabel `classification_report`); the `n_jobs` `FutureWarning` is gone.
- `data/` and `artifacts/` execution churn restored with `git checkout -- data artifacts`.

## Observations (non-blocking, for the author)

- **`dotsPerAngstrom` — now fully cleared repo-wide (verified).** CH07 was fixed in `8178f5e` but the fix was
  never back-ported to CH05, which silently kept the chapter unrunnable. I ran
  `grep -rn dotsPerAngstrom` across every `.py`/`.ipynb`: after this pass there are **zero** remaining
  occurrences, so CH05 was the last one. No further action needed — noting it only because the
  CH07→CH05 gap shows a rdkit-deprecation fix can be applied to one chapter and missed elsewhere.
- **`cv="prefit"` is contained to CH05 (verified).** `grep -rn prefit` across every `.py`/`.ipynb` returns
  no other occurrence, so no sibling chapter needs the `FrozenEstimator` migration. Flagging the *class* of
  failure rather than the instance: sklearn 1.9 **removed** this parameter rather than deprecating it, and it
  fails only at runtime — so the parse-only harness reports a green chapter that cannot actually run. Any
  future sklearn bump wants an execution pass, not just `validate_notebooks.py`.
- **The `else:` (cache-miss) branch of the multilabel loader is untested dead code** (see Table 2). Readers who
  delete `artifacts/ch05/cyp_inhibitor_df.pkl` to "start clean" will hit a `KeyError`, not a rebuild. Consider
  either repairing it or replacing it with a clear "download the artifact" message.
- **`apply_calibration` computes three Brier scores and discards them** (`brier_orig`/`brier_platt`/
  `brier_isotonic` are assigned but never returned or printed). If the intent was to show the calibration
  improvement numerically, they should be returned/printed — this is also the natural place to surface the
  Q1 numbers.
