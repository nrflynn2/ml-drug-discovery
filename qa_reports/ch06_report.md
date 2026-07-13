---
chapter: ch06
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
  total: 22
  fixed: 12
  already_ok: 5
  not_found: 0
  needs_author_decision: 5
new_taxonomy_hits: 8
chapter_done: false
---

# Chapter 06 — QA Report

_(Summary written by the orchestrator: the chapter agent completed its pass and filed the tables below,
but was torn down by a session teardown before writing this paragraph.)_

_CH06 was the heaviest chapter so far and the pass was substantive: **12 fixed, 5 already-ok, 5 escalated**,
plus **8 new taxonomy hits**, with a clean `full` execution. Two reviewer BLOCKERs turned out to be
`already-ok` (the undefined `hiv_tar1_lnkd_X_DR` doesn't exist here, and the `best_features is None` path
was already guarded at both levels), and one reviewer BUG was a **misreading**: the `0.8` the reviewer saw
is a *separate* `redundancy_threshold` for the low-variance step, not the correlation filter — which
already uses `correlation_threshold=0.95`, matching the text. The most valuable fixes came from actually
running the code: **6.2's alignment was silently aligning on nothing** (the `[nH]` SMARTS returns an empty
match against esomeprazole's `[n-]`, and `AlignMolConformers(mol, ())` fails silently — now a charge/H-agnostic
aromatic `n` SMARTS with an explicit match check and fallback); **6.4's Boltzmann weights overflowed to
`inf`/`nan`** on raw UFF energies (now exponentiated relative to the global minimum — mathematically
identical, numerically safe); **6.3's conformer pruning desynchronized IDs from energies** (now an explicit
`id_to_energy` map + keep-list); and **6.9's `partial_derivative` returned a length-1 array**, not a scalar.
The proactive sweep also caught a genuine `protomer_energies` mis-indexing in `generate_conformers` and a
markdown line telling ch6 readers to load artifacts from **`artifacts/ch09`**._

_Five items are genuine author calls (Q1–Q5), and four of them share one root tension: the chapter is
explicitly **replicating the Cai/Hargrove paper**, so the methodological defects the reviewer flags
(target-informed feature selection before the split, PCA/scaling fit pre-split, and the final model chosen
by maximizing test-set Q²) are faithful to the source but statistically unsound. Fixing them would move
every downstream number and figure. That is a pedagogical decision — "replicate the paper, and name its
flaws" vs. "demonstrate the correct protocol" — not a code decision, so none were changed unilaterally._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| 6.1 | ROBUSTNESS | fixed | Variant generation had no dedup and re-parsed each protonated SMILES twice (`MolFromSmiles` called inside both the comprehension body and its `if` filter). Rewrote the loop to parse each protonation state once, `continue` on a `None` parse, skip `None` tautomers, and deduplicate the combined protomer×tautomer set by **canonical SMILES** (`Chem.MolToSmiles`). Runtime-verified: 48 compounds → 4,561 unique variants, 0 unparseable. (uncommitted) |
| 6.1 | BUG | fixed | Tautomers were enumerated **only from the original molecule** (`tautomer_enumerator.Enumerate(mol)`) and then concatenated with the protonated mols, so no protomer×tautomer cross-product was ever produced. Now the enumerator runs on **each protonation state** (`Enumerate(protonated_mol)`), which is what the section's prose describes. (uncommitted) |
| 6.1 | CONSISTENCY | needs-author-decision | pH range is hardcoded 6.4–8.4 ("physiological"). Whether that matches the Hargrove SPR assay buffer is an experimental-methods fact I cannot source from the repo → Q1. I did surface it: `ph_min`/`ph_max` are now function parameters (default 6.4/8.4), so the author can retune in one place without editing the body. (uncommitted) |
| 6.2 | BUG | fixed | Confirmed the defect empirically: the esomeprazole SMILES contains `[n-]`, and the alignment SMARTS `c1[nH]c2ccccc2n1` returns an **empty match** (`()`), so `AllChem.AlignMolConformers(mol, ())` silently aligned on *nothing*. Changed the SMARTS to `c1nc2ccccc2n1` — a bare aromatic `n` constrains neither H-count nor charge, so it matches the benzimidazole core in both the `[n-]` and neutral `[nH]` forms (verified: 9-atom match on esomeprazole, still matches neutral benzimidazole). Added an explicit `if match:` check plus a whole-molecule `AlignMolConformers(mol)` fallback + warning when no core is found. (uncommitted) |
| 6.3 | BUG | fixed | `RemoveConformer` was called *inside* the `zip(conformer_ids, UFF_output)` loop, and the surviving energies were then recovered with `conformer_energies[i] for i in remaining_ids` — which silently assumes conformer ID == list position. Replaced with the reviewer's recommended keep-list: build an explicit `id_to_energy` map, collect `ids_to_remove` first, delete afterwards, then read energies back through the map. IDs and energies can no longer desynchronize regardless of how RDKit numbers conformers. (uncommitted) |
| 6.3 | ROBUSTNESS | fixed | Added the three missing guards: (a) `UFFOptimizeMoleculeConfs` is wrapped in try/except (it raises on unsupported atom types / bad geometries) and the molecule is skipped on failure; (b) the **UFF convergence flag** (element 0 of each returned tuple; `!= 0` means *not* converged) is now counted and reported instead of being discarded via `(_, energy)`; (c) `EmbedMultipleConfs` returning zero conformers, and an empty post-filter conformer set, both `continue`. (uncommitted) |
| 6.4 | BUG | already-ok | The Boltzmann-constant typo is **not present** in this snapshot: `k_B = 1.987E-3` kcal·mol⁻¹·K⁻¹ is already correct. (uncommitted) |
| 6.4 | ROBUSTNESS | fixed | Weights were computed as `exp(-E/kT)` on **raw** UFF energies. UFF energies are frequently large and negative, so `exp(-E/kT)` overflows to `inf` (and the partition function with it → `inf/inf = nan`). Now exponentiates energies **relative to the global minimum**, `exp(-(E - E_min)/kT)`: mathematically identical (a Boltzmann weight is a ratio, so a constant shift cancels) but numerically safe. Also added an empty-input guard. The ID↔energy alignment half of this item is guaranteed by the `id_to_energy` map added in 6.3. (uncommitted) |
| 6.5 | BUG | already-ok | The described mismatch is **not present**: the correlation loop uses `correlation_threshold` (default **0.95**, matching the text) at every site (`corr_matrix.gt(correlation_threshold)`, `x[x > correlation_threshold]`). The `0.8` in the signature is a *separate* `redundancy_threshold` governing the earlier low-variance/near-constant-column step, not the correlation filter. No code change needed. (uncommitted) |
| 6.5 | ROBUSTNESS | already-ok | The no-correlated-pair case is already handled: the `while corr_matrix.gt(correlation_threshold).any().any()` guard means the loop body never runs when nothing exceeds the threshold, and the inner `else: break` exits when `max(num_correlated) == 0`. (uncommitted) |
| 6.5 | BUG (leakage) | needs-author-decision | Confirmed present: `refine_descriptors` breaks correlated pairs by keeping whichever descriptor correlates more strongly **with the target** (`df_copy[[col1, col2, target_col]].corr()[target_col]`), and it runs on the **full 48-compound dataset before any train/test split** — so test-set response values inform feature selection. This is real leakage, but the chapter is explicitly replicating the Cai/Hargrove paper, and fixing it (moving selection inside the fold) would change every downstream number, figure and the printed Q² values. Not changed unilaterally → Q2. (uncommitted) |
| 6.6 | BLOCKER | already-ok | The undefined `hiv_tar1_lnkd_X_DR` does **not** exist in this snapshot (`grep` for `hiv_tar1_lnkd_X`, `_X_DR`, `_X_2D`, `_X_pca` → no hits). The code consistently uses `X_dr` / `X_dr_list` / `X_lnkd_dr`, all defined by `apply_pca_to_descriptors`. No NameError. (uncommitted) |
| 6.6 | BUG | needs-author-decision | Confirmed the code seeds the selection with the **single point farthest from the mean**, not the farthest *pair* (canonical Kennard–Stone). The notebook's own docstring says "Start with the point farthest from the mean", so the notebook is self-consistent and the mismatch (if any) is with the book's description. The reviewer offers an either/or ("revise the code **or** the description"), and revising the code would change the committed `KSA_example` figure — an author call → Q3. Worth knowing when deciding: KSA is **not actually used for the real split** (see the `if False:` branch below), so this choice only affects the illustrative figure. (uncommitted) |
| 6.6 | ROBUSTNESS | needs-author-decision | Two asks in one bullet. (a) **Singular inverse covariance — FIXED**: `np.linalg.inv(np.cov(X.T))` on ~48 samples × many descriptors is singular/ill-conditioned and raises `LinAlgError`; switched to `np.linalg.pinv` (Moore-Penrose), which is well defined in that regime and agrees with `inv` at full rank. (b) **PCA/scaling fit before the split — NOT changed**: `apply_pca_to_descriptors` and `split_data_with_kennard_stone` both `fit_transform` a `StandardScaler`/`PCA` on all 48 compounds before the split, so test compounds influence the scaler means and the principal axes. Same paper-replication tension as Q2 → Q4. The row is `needs-author-decision` because half of it is unresolved. (uncommitted) |
| 6.7 | ROBUSTNESS | fixed | `compute_feature_pvals` used `np.linalg.inv(X.T @ X)`, which raises `LinAlgError` on any collinear descriptor combination and would abort the whole exhaustive search mid-loop. Switched to `np.linalg.pinv` (identical results when well-conditioned). Also added a `dof = n - p - 1 <= 0` guard (returns p-values of 1.0 rather than dividing by zero) and clipped `var_b` to `finfo.tiny` before `sqrt`, so a zero-variance coefficient can't produce a `nan`/divide-by-zero t-statistic. (uncommitted) |
| 6.7 | BLOCKER | already-ok | The `best_features is None` path is **already guarded at both levels**: `exhaustive_mlr_search` returns `None, None, None, None` when `best_model is None`, and `qsar_modeling_workflow` does `if best_model is None: return None, None, None, None` **before** reaching `feature_names = [descriptor_names[i] for i in best_features]`. Since `best_model` and `best_features` are only ever assigned together, `best_features` can never be `None` at the loop. No crash. (uncommitted) |
| 6.7 | BUG | needs-author-decision | Confirmed: the exhaustive search maximizes `q2_test = r2_score(y_test, y_test_pred)` and filters on `q2_test > threshold`, i.e. the final model is selected by **optimizing the held-out test set** across ~198k/24k combinations — so the reported Q² is an optimistically biased selection statistic, not an out-of-sample estimate. Fixing this properly means a validation split or nested CV, which changes the headline Q² values (0.79/0.67/0.83) printed in the notebook *and* the book → Q5. (uncommitted) |
| 6.8 | BUG | fixed | The 10-fold CV loop re-`fit()` a **single shared** `XGBClassifier` instance across folds, so each fold's result depended on the object's prior state. Now clones a fresh, unfitted estimator per fold (`fold_model = clone(model)`; added `from sklearn.base import clone`), which is the standard sklearn contract and makes the folds genuinely independent. (uncommitted) |
| 6.9 | BUG | fixed | Documented the limitation the reviewer asked for rather than deleting the example: added a call-out above the cell explaining that a gradient-boosted tree is **piecewise-constant** (true derivative zero almost everywhere, undefined at split thresholds), so the central difference really answers "does a ±delta nudge push this sample across a split?" — making it a *local sensitivity probe* whose magnitude depends on the arbitrary `delta`, not a gradient, and pointing readers to SHAP for principled attribution on tree ensembles. Also added an empty-selection guard so a run where nothing exceeds `threshold` prints a message instead of drawing an empty bar chart. (uncommitted) |
| 6.9 | BLOCKER | fixed | `partial_derivative()` returned `(y_pred_plus - y_pred_minus) / (2*delta)` — a **length-1 array** (because `predict_proba` returns one row per sample and it is called with the single row `X[:1]`) — which was then assigned into the scalar slot `partial_derivatives[i]`, relying on NumPy's deprecated array→scalar coercion. Now indexes the result (`[...][0]`) so a genuine Python/NumPy scalar is returned, exactly as the reviewer prescribed. (uncommitted) |
| 6.10 | ROBUSTNESS | fixed | Two version-compatibility hazards. (a) **Shape**: `TreeExplainer.shap_values` for a *binary* classifier returns a single `(n, f)` array on some SHAP/XGBoost combinations, a per-class `list` on others, and a 3-D `(n, f, 2)` array on others — the code assumed the first. Added a normalization block that reduces list / 3-D forms to the positive class (also for `expected_value`), followed by an `assert shap_values.shape == X_sample.shape` **before** the `shap.Explanation` is constructed. (b) **argsort**: replaced `shap_explanation.abs.mean(0).argsort()[-1]` — whose semantics shifted across SHAP releases and which returns an `Explanation`, not an `int`, breaking the `[:, most_important_feature]` indexing — with the unambiguous `int(np.argmax(np.abs(shap_values).mean(axis=0)))`. (uncommitted) |
| General | ROBUSTNESS | fixed | Swept the chapter for the listed failure modes and addressed each where present: failed molecule parsing (all 3 `MolFromSmiles` call sites are `None`-guarded), failed embedding (`EmbedMultipleConfs` → 0 conformers), failed force-field optimization (UFF try/except + convergence flags), singular matrices (`pinv` in both `kennard_stone_algorithm` and `compute_feature_pvals`), no selected features (`best_model is None` early-return; empty partial-derivative mask), empty outputs (empty Boltzmann input), and version-dependent API behavior (SHAP shape normalization; the dimorphite-dl 2.0 migration in Table 2). No bare `except:` and no mutable default arguments exist in this chapter. (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| Import cell — `from dimorphite_dl import DimorphiteDL` | deprecated-api / import-undefined | **BLOCKER** | **The single most important fix.** `dimorphite-dl` **2.0 removed the `DimorphiteDL` class** — the installed 2.0.2 exports only `protonate_smiles`. Since `pyproject.toml`/`requirements*.txt` pin `dimorphite-dl>=2.0.0`, the notebook's **very first import cell raised `ImportError` against its own pinned environment**, so CH06 could not run at all. Migrated to the 2.0 functional API: `from dimorphite_dl import protonate_smiles`, and the call site `dimorphite_dl.protonate(smi)` → `protonate_smiles(smi, ph_min=..., ph_max=..., max_variants=..., label_states=False)`. Verified end-to-end (48 compounds → 4,561 variants). (uncommitted) |
| Colab setup cell — `wget ... -O "data/ch06/*.pkl(.gz)"` | shape/offbyone/**path** | BUG | The Colab cell downloads `binder_data.pkl`, `binder_model.pkl` and the three `hiv_tar1_lnk*.pkl.gz` into **`data/ch06/`**, but `load_molecular_dataframe(..., chapter="ch06")` reads from **`artifacts/<chapter>/`** and the CV-reload reads `artifacts/ch06/binder_data.pkl`. Every Colab reader would therefore hit `FileNotFoundError` on the artifact-reload cell. Repointed the five `-O` targets to `artifacts/ch06/` (the two `.csv` targets were already correct, since those *are* read from `data/ch06/`) and added a comment explaining the split. (uncommitted) |
| Import cell — `from tqdm.notebook import tqdm` | headless-execution | BUG | `tqdm.notebook` hard-raises `ImportError: IProgress not found` under headless nbconvert without ipywidgets (the same defect fixed in CH04). Switched to `from tqdm.auto import tqdm`, which degrades to a text bar. (uncommitted) |
| 3× `display(...)` calls with no import | import/undefined | ROBUSTNESS | `display` was relied on as an IPython auto-injected builtin (undefined in a plain interpreter). Added `from IPython.display import display`, matching the CH02/CH03/CH04 treatment. (uncommitted) |
| Configure-settings cell — `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)` | nondeterminism | ENHANCEMENT (minor) | Replaced with the standardized `RANDOM_SEED = set_seed(42)` (shared helper seeds Python + NumPy + `PYTHONHASHSEED`) and added a chapter-tailored `check_env([...])`. `RANDOM_SEED` keeps its name and value, so all six downstream `random_state=RANDOM_SEED` uses are unchanged. (uncommitted) |
| `generate_conformers` — `protomer_energies[protomer_idx]` keyed by the **original** index | shape/offbyone | BUG | Latent misalignment: energies were keyed by the enumeration index over the *input* `protomers` list (which `continue`s past `None` molecules and zero-conformer embeds), while the consumer `calculate_boltzmann_weighted_descriptors` reads them back with `for protomer_idx, protomer in enumerate(protomers_with_confs)` — the **compacted** list. As soon as any protomer is skipped, the two indices diverge and weights are read from the wrong molecule (or `KeyError`). Now keyed by position in `protomers_with_confs` (`len(protomers_with_confs) - 1`). Found during the 6.3 rewrite; not runtime-exercised because this cell is skip-tagged (see Verification log). (uncommitted) |
| Markdown above the artifact-reload cell — "load the dataframes below from **artifacts/ch09**" | path | ROBUSTNESS | Wrong chapter directory in a reader-facing instruction (the files are in `artifacts/ch06/`). Corrected `ch09` → `ch06`. Factual typo fix, not a prose rewrite. (uncommitted) |
| `artifacts/ch06/binder_model.pkl` (337 KB, committed + wget'd) | path | — | **Observation, not fixed.** The artifact is committed *and* downloaded by the Colab cell, but **nothing in the notebook ever loads it** — `build_rna_protein_binder_classifier` always retrains the final model from scratch. It is a dead artifact. Either wire it into a load-instead-of-retrain path (which would also cut the Section-4 runtime) or drop it from the repo/wget list. Left alone: deciding which is an author call. (uncommitted) |
| PCA illustration cell — `np.random.seed(0)` | nondeterminism | — | **Observation, not fixed.** A stray chapter-local seed (distinct from `set_seed(42)`) feeding the synthetic 2-D → 1-D PCA figure. Reseeding it to `RANDOM_SEED` would re-roll the committed `2D_to_1D` figure for no correctness gain, so I left it. Flagging for consistency only. (uncommitted) |

## Author-decision queue

```
Q1 (Ch6 / 6.1 — Dimorphite-DL pH range vs. the SPR assay): Protomers are enumerated over a
   hardcoded pH 6.4-8.4 ("physiological"). The reviewer asks that this match the experimental
   (SPR) assay conditions from the Hargrove/Cai study, or that the broader range be justified.
   The assay buffer pH is an experimental-methods fact that is nowhere in this repo, so I cannot
   verify it.
   Evidence needed: the SPR running-buffer pH from the source paper (Cai et al., J. Med. Chem.
   2022, 65(10)). If it is a single pH (e.g. 7.4), either narrow the range or add one sentence
   justifying the wider physiological window.
   Code is ready either way: ph_min/ph_max are now parameters of
   generate_protonation_states_and_tautomers (defaults unchanged at 6.4/8.4).

Q2 (Ch6 / 6.5 — feature selection before the split = target leakage): refine_descriptors breaks
   each correlated descriptor pair by keeping the one that correlates more strongly WITH THE
   TARGET, computed over all 48 compounds BEFORE the train/test split. Test-set responses
   therefore inform which features exist. Decision needed: (a) declare this paper replication and
   add an explicit disclaimer in the text (no code change), or (b) move the correlation/target
   filter inside the training fold — which changes the retained descriptor sets and therefore
   every downstream coefficient, p-value, Q² and figure in Section 3.
   Evidence needed: author intent (replicate Cai et al. faithfully vs. teach unbiased evaluation).
   I did not change methodology unilaterally.

Q3 (Ch6 / 6.6 — Kennard-Stone initialization): The implementation seeds the selection with the
   single point FARTHEST FROM THE MEAN; canonical Kennard-Stone seeds with the farthest-apart
   PAIR. The notebook's own docstring matches the code, so any mismatch is with the book's
   description. The reviewer explicitly offers either/or. Note that KSA is NOT used for the actual
   split -- split_data_with_kennard_stone has `if False:` and uses the paper's fixed
   test_indices_paper = [1,5,9,12,18,23,26,31,33,35,42,47] -- so changing the algorithm only
   affects the illustrative KSA_example figure.
   Decision needed: (a) reword the book to describe farthest-from-mean seeding, or (b) implement
   the farthest-pair seeding (I will do it; it re-rolls figures/ch06/KSA_example.*).
   Evidence needed: the book's KSA paragraph.

Q4 (Ch6 / 6.6 — PCA + scaling fit before the split): StandardScaler and PCA are fit on all 48
   compounds (apply_pca_to_descriptors, and again inside split_data_with_kennard_stone) before
   the train/test split, so test compounds shape the scaler means and the principal axes. Same
   paper-replication tension as Q2. (The singular-covariance half of this reviewer bullet is
   already fixed: np.linalg.inv -> np.linalg.pinv.)
   Decision needed: fit scaler/PCA on the training fold only and transform the test set, or keep
   as-is with a stated "replication, not unbiased evaluation" caveat.
   Evidence needed: same author intent as Q2 -- these two should be resolved together.

Q5 (Ch6 / 6.7 — final model chosen by maximizing test-set Q²): exhaustive_mlr_search enumerates
   up to ~198k feature combinations and keeps the one with the highest r2_score on the TEST set
   (and filters on q2_test > threshold). The reported Q² values (lnkon 0.7924, lnkoff 0.6711,
   lnKD 0.8259) are therefore selection statistics, not out-of-sample estimates -- with 36 training
   compounds and 198k combinations, this is a severe multiple-comparisons/optimism problem.
   Decision needed: (a) keep for paper replication + state plainly in the text that Q² here is a
   fitted/selected statistic, or (b) select on a validation set / nested CV and re-report -- which
   changes the headline numbers printed in both the notebook and the book.
   Evidence needed: author intent + whether the book's quoted Q² values may move.
   Blocks: the honest interpretation of Section 3's headline result.
```

## Pedagogy changes (Standard depth)

- **Standardized setup cell:** replaced the bare `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)`
  with a chapter-tailored `check_env(["numpy","pandas","scipy","sklearn","rdkit","xgboost","shap",
  "dimorphite_dl","matplotlib","seaborn"])` + `RANDOM_SEED = set_seed(42)` + a one-line CPU/runtime
  note naming the three heavy cells. `RANDOM_SEED` keeps its name/value so every downstream
  `random_state=RANDOM_SEED` is untouched.
- **Learning objectives / takeaways:** already present as the "This chapter covers" bullets (top)
  and the "Summary" + "Interactive Exploration & Further Study" sections (bottom) — kept as the
  pre-existing equivalents per "pre-existing equivalents count", not duplicated.
- **`preview_df`:** added after the two major DataFrame steps — the raw HIV-TAR load (replacing a
  bare `display(df.head())`) and the descriptor-refinement step (`hiv_tar1_lnkd`).
- **Light asserts:** (a) refinement keeps every compound and every target column; (b) the
  train/test split partitions all compounds with no index overlap and X/y row counts agree;
  (c) the binder dataset has both classes and no residual NaNs; (d) SHAP values match `X`'s shape.
- **New caveat call-out** on finite differences for tree ensembles (see 6.9 BUG) — the one piece of
  new prose, added because the reviewer explicitly asked for the limitation to be documented.
- Preserved the visual identity (`#A20025` headers, emoji section banners). No restructuring.

## Verification log

_PLACEHOLDER_
