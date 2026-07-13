---
chapter: ch03
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
  fixed: 1
  already_ok: 11
  not_found: 0
  needs_author_decision: 1
new_taxonomy_hits: 5
chapter_done: false
---

# Chapter 03 — QA Report

_CH03 is a recently-fixed snapshot in good shape: the harness passed at baseline (39 code cells, 0
errors) and every "very likely already fixed" BLOCKER the calibration flagged is indeed resolved here —
`RANDOM_SEED = 42` is defined (3.7), `PolynomialFeatures(... interaction_only=True)` has no stray comma
(3.17), the cache dir is `gs_linear_sgd` and results are read via `search.best_params_` (3.18), and the
model path has no trailing space (3.20). 11 of the 13 inventory items are already-ok; I fixed the one
genuine live inventory defect (3.15: the "Logistic Regressor" comment mislabeled the default hinge loss)
and escalated the standardization text↔code item as a manuscript-prose question (Q1). The proactive
sweep found 5 taxonomy hits — 4 fixed and 1 flagged: the known bare `except:` in `draw_fragment_from_bit`, the
non-standard `np.random.seed(42)` (→ `set_seed`), and — surfaced by skip-tagging the Colab cells — two
import gaps (`os`, used at `os.path.exists` in the executed path, was imported only inside the now-skipped
Colab cells; `display` was used 9× as an IPython builtin); plus one flagged-not-fixed latent fragility
(the bit-visualization hardcodes 2048-bit fingerprints while the grid model's `nBits` is tuned).
Applied the Standard-depth pedagogy pass
(standardized `check_env()` + `RANDOM_SEED = set_seed(42)` setup, `preview_df()` after standardization,
three light asserts) and tagged the 5 Colab-only setup cells `skip-execution` so the notebook
full-executes locally. **Execution passed 0-errors (34 cells run, 5 Colab cells skipped).** `chapter_done`
is **false** only because two author questions are open (Q1 standardization prose, Q2 the incomplete
final `malaria_box_hits = [...]` cross-chapter demo) — no code defect remains._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| 3.3 | BUG | already-ok | The deprecated `.append()` is absent — the current code uses `pd.concat([herg_blockers["pIC50"], simulated_error], ignore_index=True)`. (uncommitted) |
| 3.5 | ROBUSTNESS | already-ok | `process_smiles` already guards `mol = Chem.MolFromSmiles(smi); if mol is None: return None` before `Cleanup(mol)`. The pipeline transformer `SmilesToMols._process_smiles` guards it too. (uncommitted) |
| 3.6 | BLOCKER | already-ok | `GetMorganGenerator` (from `rdkit.Chem.rdFingerprintGenerator`) and `np` are both imported in the consolidated top-of-notebook import cell, so `compute_fingerprint` resolves them. Flag manuscript listing for production: show the relevant imports beside the printed listing so it prints as runnable-verbatim. (uncommitted) |
| 3.7 | BUG | already-ok | `split_data` already uses the reviewer's boolean-mask approach (`df[split_col].str.contains("Train")` / `"Test"`) with `.copy().reset_index(drop=True)` + a seeded shuffle — no `df.index[...]`→`.iloc[...]` misuse. Verified against the data: `Random Split` ∈ {"Training II","Test II"} with 0 NaNs, so the masks partition all 587 rows (392 train / 195 test). (uncommitted) |
| 3.7 | BLOCKER | already-ok | `RANDOM_SEED` is defined before first use. I additionally wired it through the shared helper (`RANDOM_SEED = set_seed(42)`) — same value 42, keeps every `random_state=RANDOM_SEED` working (see Table 2). (uncommitted) |
| 3.10 | CONSISTENCY | already-ok | Baseline is explicit and version-independent: `DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)`. (uncommitted) |
| 3.14 | BLOCKER (indentation) | already-ok | `FingerprintFeaturizer.transform` is correctly indented: the nested `compute_fp` closure sits at 8 spaces inside the 4-space method body, and `return np.vstack(...)` closes it. Parses clean (harness). (uncommitted) |
| 3.14 | BLOCKER (imports) | already-ok | `Chem`, `Cleanup`, `LargestFragmentChooser`, `Uncharger`, `TautomerEnumerator`, `GetMorganGenerator`, and `np` all resolve from the consolidated import cell. Flag manuscript listing for production: surface these imports beside the printed listing. (uncommitted) |
| 3.15 | CONSISTENCY | fixed | The inline comment claimed "(Logistic Regressor)" but the estimator is `SGDClassifier(...)` with no `loss=`, i.e. the default **hinge** loss (a linear SVM). Every committed output and the downstream L1-vs-L2 weight narrative were generated with hinge, so I corrected the **comment** (the reviewer's sanctioned option) rather than change the loss: `# Estimator: SGDClassifier (defaults to hinge loss -> linear SVM; pass loss="log_loss" for logistic regression)`. Zero output/prose churn. (Note: the printed, non-executed `demonstration_code` string shows `predict_proba`, which hinge SGD does not support — a latent detail for the author; folded into Q2's framing.) (uncommitted) |
| 3.17 | BLOCKER | already-ok | `PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)` — no double comma. Parses clean. (uncommitted) |
| 3.18 | BLOCKER | already-ok | No `gs.linear_sgd` identifier exists; the cache dir string is `cachedir = "gs_linear_sgd"` and results are read via `search.best_params_` / `search.best_score_`. (uncommitted) |
| 3.20 | BLOCKER | already-ok | Every model path is `"artifacts/ch03/herg_blockers_cls_model.pkl"` with no trailing space (`joblib.dump`, the demo string, and `save_load_model_demo`'s default). (uncommitted) |
| Text↔code (standardization) | CONSISTENCY | needs-author-decision | The **notebook** is already internally consistent: its markdown lists exactly the four code steps (Cleanup, LargestFragmentChooser, Uncharger, canonical TautomerEnumerator) and does *not* mention metal disconnection or stereochemistry assignment. The mismatch is with the **book prose** (reportedly says the pipeline disconnects metals + assigns stereo), which is outside the editable notebook. Do not rewrite prose → Q1. (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| `draw_fragment_from_bit` — bare `except:` around `Draw.DrawMorganBit(...)` | broad-except | BUG | Replaced bare `except:` with `except Exception as e:` and `raise ValueError(f"Featurization of mol doesn't have bit {bit_number} set") from e` — narrows the catch (no longer swallows `KeyboardInterrupt`/`SystemExit`) and chains the original error for debuggability. (uncommitted) |
| imports cell — `np.random.seed(42)` | nondeterminism | ENHANCEMENT (minor) | Replaced `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)` with `RANDOM_SEED = set_seed(42)` (shared util seeds Python + NumPy + `PYTHONHASHSEED`, and torch if present). Keeps the `RANDOM_SEED` name used by ~10 estimators; same seed value (42) so outputs are unchanged. (uncommitted) |
| `save_load_model_demo` — `os.path.exists(model_path)` | import/undefined | BUG | `os` was imported only inside the Colab setup cells; once those are tagged `skip-execution` (the standard local-run model), the executed path hits `os.path.exists` with `os` undefined → `NameError`. Added `import os` to the consolidated import cell. (uncommitted) |
| 9× `display(...)` calls (dataframe/molecule previews) | import/undefined | ROBUSTNESS | `display` was used as an IPython auto-injected builtin (works in-kernel, undefined in a plain interpreter). Added `from IPython.display import display` so every listing runs verbatim outside an interactive kernel (matches the CH02 fix). (uncommitted) |
| `draw_fragment_from_bit` / `get_examples_for_bit` — hardcoded `fpSize=2048` + 2048-bit `train_fingerprints` vs. the grid model's tuned `nBits` | shape/offbyone | ROBUSTNESS | **Observation, not fixed (pre-existing fragility).** The important-bit fragment viz hardcodes 2048-bit fingerprints, but `GridSearchCV` tunes `nBits ∈ {1024, 2048}`. In this environment the near-tied grid (0.8528 @ 1024 vs 0.8516 @ 2048) selected `nBits=1024`, so `best_model_gs_weights` (1024-dim) is drawn against 2048-bit fingerprints — the fragments no longer correspond to the bits the model actually weighted. Executes fine (0 errors); the committed run happened to pick 2048 (self-consistent). Recommend the author pin the visualization to the tuned model's `nBits` (or resolve the grid tie). Not fixed to avoid invasively rewriting the author's visualization cells. (uncommitted) |

## Author-decision queue

```
Q1 (Ch3 standardization — text vs code): The book prose reportedly states the SMILES-standardization
   pipeline "disconnects metals and assigns stereochemistry," but the code applies only Cleanup,
   LargestFragmentChooser, Uncharger, and canonical TautomerEnumerator — and the NOTEBOOK markdown
   already lists exactly those four steps (no metals/stereo), so the notebook is self-consistent.
   Decision needed: (a) update the book prose to match the four-step code (nothing to change in the
   notebook), or (b) genuinely add MetalDisconnector + AssignStereochemistry (rdkit rdMolStandardize)
   to BOTH process_smiles/SmilesToMols AND the notebook markdown.
   Evidence needed: the book's standardization paragraph. Blocks: notebook-vs-manuscript consistency
   (keeps chapter_done=false until resolved).

Q2 (Ch3 final cell — incomplete cross-chapter demo): The last code cell is
   `malaria_box_hits = [...]  # Load the saved hits from chapter 2` then
   `predictions = loaded_model.predict(malaria_box_hits)`. `[...]` is a literal Ellipsis placeholder,
   not data. The cell does NOT crash (the pipeline's SmilesToMols try/except swallows the bad input and
   predicts on a zero fingerprint) but the prediction is meaningless, and it prints a handled
   "Error processing SMILES Ellipsis" line. Chapter 2 committed `artifacts/ch02/specs_hits_to_malaria_box.csv`
   (1000 rows, columns `PUBCHEM_SUBSTANCE_ID, smiles, max_dice_sim`) — presumably the intended input.
   Decision needed: (a) wire the cell to `pd.read_csv("artifacts/ch02/specs_hits_to_malaria_box.csv")["smiles"]`
   — this introduces a cross-chapter runtime dependency (the ch2 artifact must exist); (b) replace the
   placeholder with a couple of illustrative SMILES; or (c) remove the cell. I did not invent the
   loading code. Blocks: a meaningful final "apply the saved model" demonstration.
```

## Execution note

CH03 is CPU-only and light (587-compound hERG dataset). The two slowest cells are the `GridSearchCV`
(2 penalties × 3 alphas × 2 radii × 2 nBits = 24 combos × 5 folds) and the `RandomizedSearchCV`
(20 iters × 5 folds) whose pipeline applies `PolynomialFeatures(degree=2, interaction_only=True)` to
Morgan fingerprints; both use `make_pipeline(..., memory=<cachedir>)` so the transformer steps are
cached across fits. Neither needs skip-tagging — they run in a couple of minutes. The two joblib memory
cache dirs the notebook creates in the repo root (`gs_linear_sgd/`, `rs_nonlinear_sgd/`) are execution
churn and were removed after the run (not committed).

## Pedagogy changes (Standard depth)

- **Standardized setup:** replaced the raw `np.random.seed(42)` with `RANDOM_SEED = set_seed(42)` and
  added `check_env(["numpy","pandas","scipy","sklearn","rdkit","matplotlib","seaborn"])` + a one-line
  CPU/runtime note in the consolidated import cell. Added `CHAPTER = "ch03"`.
- **Learning objectives / takeaways:** already present as the "This chapter covers" bullets (top) and
  the "Summary" section (bottom) — kept as-is per "pre-existing equivalents count."
- **preview_df:** added `preview_df(herg_blockers, "hERG blockers + standardized mol column")` after the
  standardization + invalid-drop step (the major DataFrame transform).
- **Light asserts:** `herg_blockers["mol"].notnull().all()` after standardization;
  `fingerprints.shape == (len(herg_blockers), 2048)` after featurization;
  `len(train_set) + len(test_set) == len(herg_blockers)` after the split.
- Preserved the visual identity (`#A20025` colored headers, emoji section banners); no restructuring.

## Verification log

- `uv run python tools/validate_notebooks.py CH03_FLYNN_ML4DD.ipynb` → `✓ 39 code cells OK — 0 errors, 0 warnings` (baseline before edits was also 39; the +3 pedagogy lines landed inside existing cells rather than adding cells). Re-run post-execution: still 39 cells / 0 errors.
- `uv run jupytext --sync CH03_FLYNN_ML4DD.py` → ok (benign "Notebook is not trusted" warning only); final sync reports `Unchanged` for both `.py` and `.ipynb`. The 5 Colab cells carry `tags=["skip-execution"]` in the regenerated `.ipynb`.
- execution: `uv run python tools/execute_notebook.py CH03_FLYNN_ML4DD.ipynb --timeout 1800` → **pass**: `OK: executed 34 cells, skipped 5 ('skip-execution'), 0 errors`. Post-run inspection: 39 code cells, 34 with an execution_count, 5 skip-tagged, **0 error outputs**. (The final `malaria_box_hits = [...]` cell ran without raising — its bad input is swallowed by the pipeline's try/except and printed as stdout, not an error output; see Q2.)
- Sanity of outputs: `check_env` → Python 3.12.13 (CPU only) / numpy 2.2.6 / pandas 2.3.3 / scipy 1.18.0 / sklearn 1.9.0 / rdkit 2025.09.6 / matplotlib 3.11.0 / seaborn 0.13.2. **Grid search** best = `{penalty=l1, alpha=0.01, radius=2, nBits=1024}`, f1 = 0.8528 (22 non-zero weights — L1 sparsity); **randomized search** best = `{penalty=l2, alpha≈0.469, radius=2, nBits=256}`, f1 = 0.8511 (reproduces the committed values exactly). Both preserve the **L1 (grid) / L2 (randomized)** penalties the weight-comparison prose describes — validating the comment-only fix for 3.15. Final held-out test: F1 (macro) = 0.7065. The grid's `nBits` flipped 2048→1024 vs. the committed run (a <0.2% near-tie; see the Table 2 bit-viz observation). All `figures/ch03/*` refreshed; `data/` + `artifacts/ch03/` execution churn restored with `git checkout`; the `gs_linear_sgd/` + `rs_nonlinear_sgd/` joblib caches removed.

_Note: no branch/PR opened and nothing committed, per the no-commit pilot scope; hence `(uncommitted)`
in lieu of `<sha>` citations. `CH03_FLYNN_ML4DD.py` is git-untracked (as at session start); the paired
`.ipynb` is modified in the working tree. `data/`/`artifacts/` execution churn restored with `git checkout`._
