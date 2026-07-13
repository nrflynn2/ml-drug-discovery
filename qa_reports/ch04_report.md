---
chapter: ch04
agent_model: claude-opus-4-8
run_date: 2026-07-07
env_tier: core
exec_tier: full
verification:
  static_all_cells_parse: pass
  imports_names_resolve: pass
  execution: full
  execution_result: pass
  notebook_regenerated: true
  needs_full_gpu_run: false
inventory_summary:
  total: 13
  fixed: 4
  already_ok: 7
  not_found: 0
  needs_author_decision: 2
new_taxonomy_hits: 7
chapter_done: false
---

# Chapter 04 — QA Report

_CH04 (linear models for solubility) was a mostly-recently-fixed snapshot: the harness passed at
baseline (48 code cells, 0 errors) and every listing BLOCKER the inventory flags is already resolved
in this `.py` — `Chem` is imported and `check_molecule_fragments` guards `None` (4.2); the ESOL
predicted variable was refactored to `y_pred_original`/`y_pred_refit` so `esol_val_pred` no longer
exists (4.6); `features.tolist()` is gone, the model takes prepared arrays (4.7 BLOCKER); feature
selection uses a safe list comprehension over `get_support()`, not `features[mask]` (4.8); and
`SGDRegressor` is imported, `engineered_features` is gone, y-scrambling already uses
`np.random.permutation` on numpy arrays, and the print already says "Average RMSE" (4.9). **The most
important work was in the proactive sweep, not the inventory:** on a fresh top-to-bottom run the final
section was broken two ways — the bias-variance cells silently clobbered the solubility descriptor
arrays `X_train/X_test/y_train/y_test` with 1-D synthetic polynomial data, and the learning-curve cell
referenced an undefined `ransac_regressor` — so Figures 4.20/4.21 either crashed (NameError / 1-D
array) or would have been drawn on the wrong data. Both are fixed — as is a third execution-blocker
(`from tqdm.notebook import trange` hard-crashed the y-scrambling cell under headless nbconvert; switched
to the graceful `tqdm.auto`). I also fixed the 4.3 div-by-zero
guard, added explicit MSE/MAE prints (4.4), upgraded the Ridge alpha search to `loguniform` (4.7
ENHANCEMENT), modernized a removed-in-SciPy-2.0 import, guarded a `MolFromSmiles→None` path, and applied
the Standard-depth pedagogy pass (`check_env`/`set_seed`, `preview_df`, an assert; the objectives box
and Summary already existed). Two items are genuine author calls — the logP applicability-domain bound
and the "2048 features" text/code mismatch (Q1, Q2) — which keep `chapter_done: false`._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| Applicability-domain table | BUG | needs-author-decision | The AD "table" is `train_df.describe()` (the book's bounding-box method = joint per-descriptor min/max). Computed on the fragment-filtered AqSolDB train set, `Crippen.MolLogP` genuinely spans **min = -17.41, max = +26.25** — i.e. the reviewer's "~17.4 to 26.2" is the range **[-17.4, +26.2]** with the minus sign dropped. This is **not** a code sign/unit/wrong-column bug: the min is correctly negative, and the extreme values are real outliers in a diverse set (MolWt up to 2,285 Da, up to 30 rings). All other bounds verified plausible-for-the-data. I added an explanatory note on how the table is generated + the outlier caveat, but whether to correct a manuscript-table sign or trim outliers is an author call → Q1. (uncommitted) |
| Descriptor dimensionality | CONSISTENCY | needs-author-decision | Code uses **11 RDKit physicochemical descriptors** (MolWt, HDonors, TPSA, logP, MolMR, RotBonds, Rings, LabuteASA, BalabanJ, BertzCT, AromaticProportion) and **no Morgan fingerprints** — "2048" appears nowhere in the notebook (`grep` → none). The §4.2.1 text saying "2048 features" is prose I must not rewrite → Q2. (uncommitted) |
| Polynomial order | CONSISTENCY | fixed | The synthetic **target** is 7th-order everywhere in code (`f = -0.5·i⁷ - 0.3·i³ + 1.2·i² + 0.2·i - 1`, labeled "7th Order Target" in all 4 plot references) — consistent. The real code defect: the **degree-20** overfit curve in Fig 4.18 was labeled `"10th Order Fit"` while its own title says "20th Order Model Overfits". Corrected the label to `"20th Order Fit"`. (Degree-10 is a separate, correctly-labeled model in Fig 4.19.) Flag manuscript listing for production: reconcile any 7th/10th prose to the now-consistent code (target 7th; fits 2nd/10th/20th). (uncommitted) |
| 4.2 | BLOCKER | already-ok | `from rdkit import Chem` is imported in the consolidated import cell; `check_molecule_fragments` uses `len(Chem.GetMolFrags(x)) if x is not None else np.nan`, so `Chem` resolves and `None` molecules are guarded. No NameError, no crash on invalid mols. (uncommitted) |
| 4.3 | ROBUSTNESS | fixed | `AromaticProportion` guarded only `mol is None`, still dividing by `mol.GetNumAtoms()`. Added `or mol.GetNumAtoms() == 0` so a zero-atom parse returns `np.nan` (consistent with the existing None branch; NaN rows are dropped downstream) instead of raising `ZeroDivisionError`. (uncommitted) |
| 4.4 | CONSISTENCY | fixed | `build_simple_linear_model` computed all four metrics and annotated them on the plot, but only logged RMSE and R². Added explicit `logger.info` prints for **MSE** and **MAE** so all of MSE/RMSE/MAE/R² appear in text output as the prose discusses. (uncommitted) |
| 4.6 | BLOCKER | already-ok | `esol_val_pred`/`esol_val_pred_original` do not exist (`grep` → none). The refactored `evaluate_esol_benchmark` builds `y_pred_original`/`y_pred_refit` and uses those exact defined names in `mean_squared_error(y_true, y_pred_original)`. No undefined variable. (uncommitted) |
| 4.7 | BLOCKER | already-ok | `features`/`features.tolist()` are absent. `tune_ridge_regression` receives already-prepared, scaled arrays (`X_train, X_val, y_train, y_val`); nothing indexes a `features` object. No NameError. (uncommitted) |
| 4.7 | ENHANCEMENT | fixed | Replaced `from scipy.stats import uniform as sp_rand` + `{'ridge__alpha': sp_rand()}` (samples α in [0,1) only) with `loguniform(1e-6, 1e2)`, so the RandomizedSearchCV explores both weak and strong L2 regularization on a log scale. (uncommitted) |
| 4.8 | BUG | already-ok | `sequential_feature_selection` does not do `features[mask]`; it uses `selected_indices = model[2].get_support()` then a list comprehension `[feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]`, which is safe whether `feature_names` is a list or array. The `np.array(features)` conversion is unnecessary here. (uncommitted) |
| 4.9 | BLOCKER | already-ok | `SGDRegressor` is imported (top cell `from sklearn.linear_model import (...)` and again locally at the SVR cell). `engineered_features` does not exist (`grep` → none); the code uses `available_features`/`feature_cols`. No NameError. (uncommitted) |
| 4.9 | BUG | already-ok | Y-scrambling already uses numpy arrays with no index to misalign: `X = combined_df[available_features].values`, `y = combined_df["Y"].values`, `y_scrambled = np.random.permutation(y)` — exactly the inventory's recommended form. (uncommitted) |
| 4.9 | CONSISTENCY | already-ok | The print already reads `Average RMSE over {n} y-scrambled iterations` and reports `np.mean` of per-iteration `np.sqrt(mean_squared_error(...))` (RMSE); the plot label likewise says "Avg Scrambled RMSE". No "Average MSE" mislabel. (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| Bias-variance cells (Fig 4.18/4.19) `X_train, X_test, y_train, y_test = train_test_split(X_target, ...)` | shape/offbyone/path (variable shadowing) | BUG | The synthetic 1-D polynomial demo reused the names `X_train/X_test/y_train/y_test`, **overwriting the 2-D solubility descriptor arrays** that the *later* validation-curve (Fig 4.21) and learning-curve (Fig 4.20) cells consume. On a fresh top-to-bottom run those cells got 1-D data → `ValueError: Expected 2D array`. Renamed the demo's variables to `X_poly_train/X_poly_test/y_poly_train/y_poly_test`; the descriptor `X_train/y_train` now survive intact. (uncommitted) |
| Learning-curve cell (Fig 4.20) `enumerate([ransac_regressor, sgd_regressor])` | import/undefined | BUG | `ransac_regressor` is a **local** inside `train_ransac_regressor`; at module level the fitted pipeline is `ransac_model`. Fresh-kernel run → `NameError` (the committed outputs came from an out-of-order session). Changed to `ransac_model` (the exact returned object). (uncommitted) |
| Import cell `from scipy.ndimage.filters import gaussian_filter1d` | deprecated-api | ROBUSTNESS | `scipy.ndimage.filters` is deprecated and "will be removed in SciPy 2.0.0" (DeprecationWarning under the installed scipy 1.18). Changed to `from scipy.ndimage import gaussian_filter1d`. (uncommitted) |
| `evaluate_esol_benchmark` — `delaney_df['SMILES'].apply(lambda x: MolToInchiKey(MolFromSmiles(x)) ...)` | rdkit-none-guard | ROBUSTNESS | `MolFromSmiles` returns `None` for an unparseable SMILES and `MolToInchiKey(None)` raises. Replaced the lambda with a `_smiles_to_inchikey` helper that guards both `pd.isna(smi)` and `mol is None`. (uncommitted) |
| Import cell — `display(...)` used 4× with no import | import/undefined | ROBUSTNESS | `display` only resolves as an IPython builtin; added `from IPython.display import display` so each listing also runs verbatim as a plain script (matches the CH02 treatment). (uncommitted) |
| Configure-settings cell — `np.random.seed(RANDOM_SEED)` | nondeterminism | ENHANCEMENT (minor) | Replaced with the standardized setup: `check_env([...])` + `RANDOM_SEED = set_seed(42)` (shared util seeds Python/NumPy/PYTHONHASHSEED and returns the seed, so all downstream `random_state=RANDOM_SEED` keep working). (uncommitted) |
| `compare_with_baselines` — `from tqdm.notebook import trange` | import/undefined | BUG | `tqdm.notebook` hard-raises `ImportError: IProgress not found` under headless nbconvert (ipywidgets not installed), which crashed the whole y-scrambling cell mid-execution. Switched to `from tqdm.auto import trange` (what the top of the notebook already uses), which falls back to a text progress bar. This was the one cell that blocked clean top-to-bottom execution. (uncommitted) |

## Author-decision queue

```
Q1 (Ch4 applicability-domain / logP bound): The AD table is `train_df.describe()`; the flagged
   logP range is REAL, not a code bug. Over the fragment-filtered AqSolDB training set,
   Crippen.MolLogP min = -17.41 and max = +26.25 — the reviewer's "17.4 to 26.2" is [-17.4, +26.2]
   with the minus sign dropped. These extremes come from genuine non-drug-like outliers in AqSolDB
   (MolWt up to ~2,285 Da, up to 30 rings), and every other descriptor bound is likewise the true
   data min/max. Decision needed: (a) does the manuscript's printed AD table drop the minus sign on
   the lower logP bound (should read -17.4, not 17.4)? and/or (b) keep the wide bounds as a teaching
   point about the bounding-box AD's fragility, or add outlier trimming / a distance-based domain?
   Evidence needed: the describe() logP row (provided above) + the manuscript's printed AD table.
   Blocks: the AD-table inventory row; keeps chapter_done=false until resolved. (Code is correct as-is;
   I added an explanatory note but made no code change.)

Q2 (Ch4 descriptor dimensionality / "2048 features"): The notebook's model consumes 11 RDKit
   physicochemical descriptors (MolWt, HDonors, TPSA, logP, MolMR, RotBonds, Rings, LabuteASA,
   BalabanJ, BertzCT, AromaticProportion) and NO Morgan fingerprints — "2048" appears nowhere in the
   code. Section 4.2.1 text says the feature vector has 2048 features. Decision needed: correct the
   prose to describe 11 physicochemical descriptors (not 2048-bit fingerprints), OR — if fingerprints
   are intended — add the fingerprint featurization to the notebook. This is a prose/code reconciliation
   I must not resolve by rewriting the text.
   Evidence needed: §4.2.1 manuscript wording + author intent (descriptors vs. fingerprints vs. both).
   Blocks: the descriptor-dimensionality inventory row; keeps chapter_done=false until resolved.
```

## Pedagogy changes (Standard depth)

- **Standardized setup cell:** replaced the raw `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)` with
  `check_env(["numpy","pandas","scipy","sklearn","rdkit","matplotlib","seaborn"])` (chapter-tailored —
  no torch/xgboost) + `RANDOM_SEED = set_seed(42)` + a one-line CPU/runtime note. `RANDOM_SEED` stays
  defined, so every downstream `random_state=RANDOM_SEED` is unchanged.
- **Learning-objectives box / Summary:** already present ("You'll learn how to:" and the "Summary"
  section) — left as the pre-existing equivalents, not duplicated.
- **`preview_df` + assert:** added `preview_df(train_df, ...)` after the single-fragment filter and after
  descriptor computation, plus one `assert (train_df['fragment_count'] == 1).all()` sanity check.
- **`display` import:** added `from IPython.display import display` (see Table 2).

## Verification log

- `uv run python tools/validate_notebooks.py CH04_FLYNN_ML4DD.ipynb` → `✓ 48 code cells OK — 0 errors, 0 warnings` (unchanged from baseline; pedagogy cells net out to the same count).
- `uv run jupytext --sync CH04_FLYNN_ML4DD.py` → ok (benign "Notebook is not trusted" warning only); `.ipynb` regenerated from the edited `.py` (outputs preserved across the final note-only re-sync).
- execution: `uv run python tools/execute_notebook.py CH04_FLYNN_ML4DD.ipynb --timeout 1800` → **pass** — `OK: executed 43 cells, skipped 5 ('skip-execution'), 0 errors`. Wall time ~48 min (the RANSAC-vs-SGD learning-curve cell alone is ~29 min; needed the first run's failure at the `tqdm.notebook` cell fixed before the full run could complete). 0 error outputs in the notebook JSON.
- Sanity of outputs: `check_env` → Python 3.12.13, numpy 2.2.6, pandas 2.3.3, scipy 1.18.0, sklearn 1.9.0, rdkit 2025.09.6, matplotlib 3.11.0, seaborn 0.13.2 (no torch/xgboost — chapter-tailored list). AqSolDB train 6,987→6,139 after single-fragment filter, valid 998→921, test 1,997 (0 invalid molecules). `preview_df` prints `6,139 rows x 5 cols` → `6,139 rows x 16 cols`. Simple logP model logs all four metrics (MSE 2.2906 / RMSE 1.5135 / MAE 1.1379 / R² 0.5170 — the 4.4 fix). `Prepared data with 11 features` (the Q2 point). **Ridge best alpha = 4.566** — a value >1 the old `uniform(0,1)` search could never reach, confirming the `loguniform` enhancement. Final SGD(SVR) test RMSE 1.4497 / MAE 0.9990 / R² 0.5919; ESOL test RMSE 1.3787; **y-scrambling Average RMSE 2.4773 ≫ 1.4497** (permutation test behaves correctly). All 24 `figures/ch04/*` refreshed (incl. `bias_variance_tradeoff`, `learning_curves`, `sgd_validation_curves`). `data/` and `artifacts/` restored via `git checkout`.

## Observations (non-blocking, for the author)

- **Learning-curve runtime:** the Fig 4.20 cell (`LearningCurveDisplay` for RANSAC + SGD with
  `ShuffleSplit(n_splits=50)` × `train_sizes=np.linspace(0.02, 1.0, 50)` = 2,500 fits/estimator) takes
  ~29 min on its own — the dominant cost of the notebook. Not changed (it would alter the committed
  figure), but the author may want to drop `n_splits`/`train_sizes` to ~20 each (~6× faster, visually
  near-identical) for reader ergonomics.
- **Y-scrambling scaling (minor):** the scrambling model is fit on unscaled `combined_df[available_features]`
  but predicts on the externally-scaled `X_test`, so its internal `StandardScaler` sees a scale mismatch.
  The qualitative result is unaffected (scrambled RMSE 2.48 ≫ model RMSE 1.45), so I left it; if tightened,
  fit on the same scaled `X` used for the real model.
- **`ConvergenceWarning`s:** the SVR `GridSearchCV` emits many benign SGD "maximum iterations reached"
  warnings for aggressive hyperparameter combos — expected during the search, not failures.
