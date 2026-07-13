# Code-Listing Feedback for the QA Pass — Companion Notebooks

**Purpose.** This document extracts *only the code-actionable* feedback from the reviewer files so a QA-pass coding agent can locate and fix the corresponding notebook listings. Conceptual, prose, biomedical-terminology, citation, and figure-aesthetic feedback has been left out **except** where it directly implies a code change (e.g., a text↔code mismatch, a version-fragile assertion in code, or a figure/table that is generated programmatically).

**Sources.** Per-chapter feedback files (`ch1`–`ch13`, `appC`, `appD`) plus the collated `04_reviewer14_evidence.md`. Where both sources cover the same listing, they have been merged; the reviewer-14 file was used to confirm listing numbers and exact code fragments.

**How to read an entry.** Each item is anchored to a listing number (or a named code location) and tagged by type so the agent can triage:

| Tag | Meaning |
|-----|---------|
| `BLOCKER` | Will not execute as printed — syntax error, undefined name, missing import, wrong/inconsistent variable name, stray manuscript marker. Fix first. |
| `BUG` | Executes but produces wrong or misleading results — logic error, wrong target/label, bad indexing, data leakage, deprecated API. |
| `ROBUSTNESS` | Add guards / edge-case handling — `None` checks, division-by-zero, empty inputs, device mismatch, library-version sensitivity. |
| `CONSISTENCY` | Code must be reconciled with the surrounding text, a figure, a caption, or made explicitly version-specific. |
| `ENHANCEMENT` | Optional reviewer-requested code addition, not a defect. |

> **Note on line-continuation artifacts.** Several listings contain a backslash immediately followed by an inline comment (e.g. `= \    #A`). A backslash cannot be followed by a comment, so these are hard syntax errors. They are tagged `BLOCKER` throughout and the fix is always to use parentheses or to inline the expression.

> **Note on example output in code cells.** A recurring issue is REPL/example output (lines beginning `>>> …`) pasted inside executable cells. These must be moved out of the executable block or rendered as output, not code.

---

## Chapter 1 — The Drug Discovery Process

*Most Chapter 1 feedback is conceptual. The code-actionable items:*

### Code presentation (Section 1.2)
- `BLOCKER` — Example output such as `>>> Number of atoms: 14` must not sit inside an executable code cell; it produces a syntax error. Separate code from output cleanly across all Section 1.2 snippets.
- `ENHANCEMENT` — Add explanatory comments to the Section 1.2 code examples and a one-line statement of what each snippet demonstrates / what the reader should take away.

### Listing 1.2 (RDKit + ECFP6 + PCA demo)
- `CONSISTENCY` — The listing is out of date; update it to reflect the current code block.
- `CONSISTENCY` — Make explicit in the code/comments (and text) where the data set resides / is loaded from (data path or source).

### Figure 1.2 (drugs-per-\$bn R&D plot — generated programmatically)
This figure is produced by code (from 2023 public data), so the fixes are code changes:
- `BUG` — There is an apparent LaTeX/label error in the "New drugs per bn\$ R&D" annotation; fix it.
- `ENHANCEMENT` — Regenerate an improved version: label X and Y axes, add a descriptive title, increase font sizes, move the annotation to the top-right (currently hidden behind the graph), and extend the data past 2010 if a more recent public source is available.

---

## Chapter 2 — Ligand-based Screening: Filtering & Similarity Searching

### General (whole chapter)
- `ROBUSTNESS` — Run every listing verbatim in a clean environment. Ensure imports appear near the relevant listing, use consistent variable names across snippets, and do not rely on variables defined only in earlier hidden cells unless that dependency is stated.
- `ENHANCEMENT` — After major DataFrame transformations, print a small preview and show column names before/after (raw SDF → parsed → descriptors → Ro5 columns → filter flags → fingerprints → similarity scores → top hits).

### Listing 2.1
- `BLOCKER` — Fix typo `LoadSF()` → `LoadSDF()`.
- `ROBUSTNESS` — `PandasTools.LoadSDF(...)[["PUBCHEM_SUBSTANCE_ID", "smiles"]]` can fail if the SDF property names differ or if `smilesName="smiles"` is not set. Verify the compressed-file path and the property names.

### Listing 2.2
- `BLOCKER` — `RO5_PROPS` is used but never defined. Add:
  ```python
  RO5_PROPS = ["ExactMolWt", "MolLogP", "NumHDonors", "NumHAcceptors"]
  ```

### Listing 2.3
- `BUG` — Replace the bare `except:` (it silently hides descriptor failures) with:
  ```python
  except Exception as e:
      print(f"Descriptor calculation failed for molecule {i}: {e}")
  ```

### Listing 2.4
- `BLOCKER` — `df['ro5_compliant'] = \    #A` is a syntax error (backslash before comment). Use parentheses:
  ```python
  df["ro5_compliant"] = (
      df["ro5_violations"] <= 1
  )
  ```

### Listing 2.5
- `BLOCKER` — Add missing import: `from rdkit.Chem import FilterCatalog`.
- `CONSISTENCY` — The function only prints "compounds before"; the manuscript output also shows the failing count, the after count, and the percentage retained/removed. Add the missing print statements so code output matches the text.

### Listing 2.6
- `BLOCKER` — Same backslash/comment issue: `alerts_df["ROMol"] = \    #A`. Replace with:
  ```python
  alerts_df["ROMol"] = alerts_df["smarts"].apply(MolFromSmarts)
  ```

### Listing 2.7
- `BUG` / `ROBUSTNESS` — Avoid mutating the input DataFrame directly (`df_copy = df.copy()` first) to prevent `SettingWithCopyWarning`, and guard against `None` molecules:
  ```python
  if mol is not None and mol.HasSubstructMatch(pattern):
      ...
  ```

### Listing 2.9
- `BLOCKER` — Fix the indentation error around `mol_img = Draw.MolsToGridImage(` (it is over-indented inside the nested function).
- `BLOCKER` — Add missing imports / definitions:
  ```python
  from rdkit.Chem import Draw
  from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator
  from IPython.display import display
  ```
  Ensure `rdkit_drawing_options` is defined before use.

### Listing 2.10
- `BLOCKER` — Variable mismatch: `query_index = 236` then `...iloc[query_idx]`. Use one name consistently (`query_idx`).
- `ROBUSTNESS` — Reading `.xls` files may require the correct Excel engine (e.g. `xlrd`) on modern pandas.

### Listing 2.12
- `BUG` — Indexing bug: `top_matches` are positional indices from `enumerate(library_fps)`, so `specs_filtered.filter(items=top_matches, axis=0)` may select the wrong rows on a non-contiguous index. Use positional indexing:
  ```python
  specs_hits_to_malaria_box = specs_filtered.iloc[top_matches]
  ```
- `ROBUSTNESS` — Guard against popping more hits than exist:
  ```python
  for i in range(min(budget, len(heap))):
      ...
  ```

### Version-fragile assertion
- `CONSISTENCY` — The claim that RDKit computes exactly 217 descriptors should be tied to the version: "In the RDKit version used here, `Descriptors._descList` reports 217 descriptors."

### Optional
- `ENHANCEMENT` — Add count-based Morgan fingerprints alongside the binary ones (RDKit's Morgan count vector) if you extend the fingerprint section.

---

## Chapter 3 — Ligand-based Screening: Machine Learning

### Listing 3.3
- `BUG` — Replace deprecated pandas `.append()`:
  ```python
  pd.concat([herg_blockers["pIC50"], simulated_error], ignore_index=True)
  ```

### Listing 3.5
- `ROBUSTNESS` — `MolFromSmiles` can return `None`; guard before `Cleanup`:
  ```python
  mol = Chem.MolFromSmiles(smi)
  if mol is None:
      return None
  mol = Cleanup(mol)
  ```

### Listing 3.6
- `BLOCKER` — Missing imports; the listing uses `GetMorganGenerator` and `np` without importing them.

### Listing 3.7
- `BUG` — Builds label values from `df.index[...]` then passes them to `.iloc[...]`; only safe with a clean `RangeIndex`. Reset the index (`df = df.reset_index(drop=True)`) or use boolean masks. A reviewer's suggested generalized splitter:
  ```python
  def split_data(df, split_col="Random Split"):
      train = df[df[split_col].str.contains("Train", na=False)]
      test  = df[df[split_col].str.contains("Test",  na=False)]
      return (
          train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True),
          test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True),
      )
  ```
- `BLOCKER` — `RANDOM_SEED` is used but not defined. Add `RANDOM_SEED = 42` before first use.

### Listing 3.10
- `CONSISTENCY` — Make the dummy baseline explicit and version-independent: `DummyClassifier(strategy="most_frequent")`.

### Listing 3.14
- `BLOCKER` — Indentation error makes it non-executable:
  ```python
  def transform(self, X, y=None):
        from rdkit import DataStructs
          def compute_fp(mol):
  ```
  Correct the indentation.
- `BLOCKER` — Missing imports; the listing uses `Chem`, `Cleanup`, `LargestFragmentChooser`, `Uncharger`, `TautomerEnumerator`, `GetMorganGenerator`, `np`.

### Listing 3.15 (SGDClassifier)
- `CONSISTENCY` — The comment labels it "Logistic regression," but `SGDClassifier` defaults to hinge loss (linear SVM-like). Either set `SGDClassifier(loss="log_loss")` or fix the comment to describe the default.

### Listing 3.17
- `BLOCKER` — Syntax error (extra comma):
  ```python
  PolynomialFeatures(degree=2, include_bias=False, , interaction_only=True)
  ```
  Fix:
  ```python
  PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
  ```

### Listing 3.18
- `BLOCKER` — Typo: `gs.linear_sgd.best_params_` → `gs_linear_sgd.best_params_`.

### Listing 3.20
- `BLOCKER` — Filename has a trailing space: `joblib.load("herg_blockers_cls_model.pkl ")` → `"herg_blockers_cls_model.pkl"`.

### Text↔code mismatch (standardization)
- `CONSISTENCY` — The text says the pipeline disconnects metals and assigns stereochemistry, but the code only applies `Cleanup`, `LargestFragmentChooser`, `Uncharger`, and canonical tautomer generation. Reconcile code and text.

---

## Chapter 4 — Solubility Deep Dive with Linear Models

### Applicability-domain table (generated)
- `BUG` — The table reports a logP range of ~17.4 to 26.2, implausibly high for drug-like molecules. Check for a sign error, formatting error, wrong descriptor column, wrong unit/transformation, or data-entry issue; verify all other bounds; add a note on how the table is generated.

### Descriptor dimensionality
- `CONSISTENCY` — Section 4.2.1 text says the feature vector has 2048 features, but the code appears to use ~11 RDKit descriptors (no fingerprints). Reconcile: state whether the model uses descriptors, Morgan fingerprints, or a combination, and fix the text if 2048-bit fingerprints aren't actually used.

### Polynomial order
- `CONSISTENCY` — Section 4.4.1 describes the polynomial target as 7th order in one place and 10th order later. Make the order consistent across text, **code**, figures, and captions.

### Listing 4.2
- `BLOCKER` — `Chem` is used (`len(Chem.GetMolFrags(x))`) but not imported. Add `from rdkit import Chem` and guard invalid molecules:
  ```python
  def num_fragments(mol):
      if mol is None:
          return None
      return len(Chem.GetMolFrags(mol))
  ```

### Listing 4.3
- `ROBUSTNESS` — `AromaticProportion` can divide by zero for an invalid or zero-atom molecule:
  ```python
  def aromatic_proportion(mol):
      if mol is None or mol.GetNumAtoms() == 0:
          return 0.0
      aromatic_atoms = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
      return len(aromatic_atoms) / mol.GetNumAtoms()
  ```

### Listing 4.4
- `CONSISTENCY` — The prose discusses RMSE, MSE, MAE, and R², but the listing only plots. Add explicit metric calculations:
  ```python
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import numpy as np
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)
  print(f"MSE: {mse:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}")
  ```

### Listing 4.6
- `BLOCKER` — `esol_val_pred` is undefined; the variable is `esol_val_pred_original`. Use the defined name in the `mean_squared_error(...)` call.

### Listing 4.7
- `BLOCKER` — `features` is used (`train_df[features.tolist()]`) but not defined in the shown code; define it explicitly.
- `ENHANCEMENT` — `sp_rand()` samples `alpha` in 0–1, omitting larger regularization. Prefer a log-uniform search:
  ```python
  from scipy.stats import loguniform
  param_distributions = {"alpha": loguniform(1e-6, 1e2)}
  ```

### Listing 4.8
- `BUG` — `features[model[2].get_support()]` fails if `features` is a Python list. Convert first:
  ```python
  selected_features = np.array(features)[model[2].get_support()]
  ```

### Listing 4.9
- `BLOCKER` — `SGDRegressor` is used but not imported (`from sklearn.linear_model import SGDRegressor`); `engineered_features` is also undefined.
- `BUG` — Y-scrambling is fragile because `X` keeps its index while `y_scrambled` gets a new one:
  ```python
  y_scrambled = y.sample(frac=1).to_numpy()   # or np.random.permutation(y.to_numpy())
  sgd_regressor.fit(X, y_scrambled)
  ```
- `CONSISTENCY` — The print says "Average MSE" but reports average RMSE. Fix the label (or report MSE).

---

## Chapter 5 — CYP Inhibition Classification

### Listing 5.2 (calibration)
- `BUG` — Calibration is fit on the validation set and then evaluated on the same validation set, giving optimistic results. Use a separate calibration set and held-out test set, or nested cross-validation.
- `ROBUSTNESS` — `cv="prefit"` in the scikit-learn calibration API may be version-dependent; pin/state the tested scikit-learn version or update to the current recommended prefit-calibration workflow.

### Listing 5.4 (resampling)
- `BUG` — The printed labels for upsampling and downsampling appear swapped: "US" should denote upsampling and "DS"/"downsampled" should denote downsampling/undersampling. Correct the labels.

### General code review
- `CONSISTENCY` — Confirm whether the fingerprint helper expects `radius` or `r`, and use the correct keyword consistently across the chapter's snippets.

---

## Chapter 6 — RNA-targeted Small-Molecule Discovery (TAR)

### Listing 6.1 (protonation/tautomer state enumeration)
- `ROBUSTNESS` — May create duplicate or invalid states. Add `None` checks, canonicalization, deduplication, and error handling for failed molecule generation.
- `BUG` — Enumerate tautomers from *each* protonation state, not only from the original molecule.
- `CONSISTENCY` — Ensure the Dimorphite-DL pH range matches the experimental (SPR) assay conditions, or explicitly justify a broader range.

### Listing 6.2 (esomeprazole substructure alignment)
- `BUG` — The SMARTS uses `[nH]` while the esomeprazole SMILES contains `[n-]`, so the match may fail. Add a check that the SMARTS match succeeds before alignment; raise a clear error or use an alternative alignment strategy if not.

### Listing 6.3 (conformer generation / minimization)
- `BUG` — Do not remove conformers while iterating over conformer IDs (this desynchronizes IDs and energies). Use a keep-list or remove in reverse order.
- `ROBUSTNESS` — Check UFF convergence flags from `UFFOptimizeMoleculeConfs`; handle `EmbedMultipleConfs` returning zero conformers; handle UFF failures from unsupported atoms or bad geometries.

### Listing 6.4 (Boltzmann weighting)
- `BUG` — Fix the Boltzmann-constant typo: use ≈ `1.987e-3` kcal·mol⁻¹·K⁻¹.
- `ROBUSTNESS` — Subtract relative energies before exponentiation to avoid overflow/underflow; verify conformer IDs still match energy indices after any filtering.

### Listing 6.5 (descriptor-correlation filtering)
- `BUG` — The loop uses `threshold=0.8` while the text states `correlation_threshold=0.95`. Make the code match the text.
- `ROBUSTNESS` — Handle the case where no correlated pair is found.
- `BUG` (leakage) — Using response-variable correlation for feature selection before the train/test split biases results; move this step inside the training fold, or state clearly that the goal is paper replication rather than unbiased evaluation.

### Listing 6.6 (Kennard-Stone)
- `BLOCKER` — Undefined variable `hiv_tar1_lnkd_X_DR` (likely should be `hiv_tar1_lnkd_X_2D` or `hiv_tar1_lnkd_X_pca`).
- `BUG` — The code selects the single point farthest from the mean rather than selecting the farthest *pair* first. Revise the code or the description to match.
- `ROBUSTNESS` — The inverse covariance matrix with only ~48 samples and many descriptors may be singular/unstable; guard accordingly. If PCA is fit before splitting, note that for unbiased evaluation scaling/PCA should be fit on the training set only.

### Listing 6.7 (linear-model selection / p-values)
- `ROBUSTNESS` — `np.linalg.inv(X.T @ X)` fails with collinear descriptors; use a pseudo-inverse or `statsmodels` OLS.
- `BLOCKER` — Guard the case where no model passes the filters (if `best_features` stays `None`, the later loop fails).
- `BUG` — Do not choose the final model by exhaustively optimizing test-set Q²; use a validation set or nested CV.

### Listing 6.8 (XGBoost comparison)
- `BUG` — Clone a fresh model inside each cross-validation fold rather than reusing the same model object.

### Listing 6.9 (interpretability via finite differences)
- `BUG` — Finite differences on tree ensembles are questionable (piecewise-constant → zero derivative except near split thresholds). If keeping the example, document the limitation.
- `BLOCKER` — Fix the array-to-scalar assignment by indexing the returned prediction, e.g. `[0]`.

### Listing 6.10 (SHAP)
- `ROBUSTNESS` — Check SHAP/XGBoost version compatibility; verify SHAP value shapes before constructing `shap.Explanation`; confirm the correct `argsort` usage for the installed SHAP version.

### General
- `ROBUSTNESS` — Add defensive checks throughout for empty outputs, failed molecule parsing, failed embedding, failed force-field optimization, singular matrices, no selected features, and version-dependent API behavior.

---

## Chapter 7 — Unsupervised Learning (Dimensionality Reduction, Clustering, KDE)

### Listing 7.1 (fingerprinting + activity labels)
- `BLOCKER` — Replace undefined `fp_size` with the defined `n_bits`; fix the malformed parentheses around the fingerprint conversion.
- `BUG` — Use `DataStructs.ConvertToNumpyArray()` when converting RDKit bit vectors to NumPy arrays.
- `BUG` — Check the active/inactive label logic: if the threshold is `IC50 < 5`, that should usually mark **active**, not inactive.
- `ROBUSTNESS` — Handle missing/NaN IC50 values explicitly rather than letting them fall into an implicit class.

### Listing 7.2 (SOM parameter grid)
- `BLOCKER` — Add the missing closing brace for `som_param_grid`; define/import `RANDOM_SEED`; define `n_nodes` and `m_nodes` before use.
- `CONSISTENCY` — If the text says the SOM is toroidal, ensure the implementation actually uses toroidal wrapping (otherwise revise the text).

### Listing 7.3 (UMAP / repurposing)
- `BLOCKER` — Remove invalid backslash-plus-comment syntax; replace the Unicode ellipsis `…` with valid Python or a clearly-marked placeholder comment; define `drh_compounds` and `retrospective_hits`.

### Listing 7.5 (reaction expansion)
- `BLOCKER` — The code defines `reaction_smirks` but uses `reaction_smarts`; unify the name.
- `ROBUSTNESS` — Prevent possible infinite loops in `while "[*]" in core_smi`; avoid mutating `core_smi` across independent expansion attempts unless intentional; validate the reaction atom mapping (use of `:0`).

### Listing 7.6 (BRICS fragments)
- `BLOCKER` — Creates `frag_mols` but later uses undefined `fragMols`; use one name.
- `BUG` — Replace the bare `except` with explicit exception handling so sanitization/parsing/chemistry failures are visible.

### Listing 7.7 (clustering fingerprints)
- `BLOCKER` — Add missing imports such as `DataStructs` and `AgglomerativeClustering`.
- `CONSISTENCY` — Clarify that the example uses RDKit path fingerprints (not Morgan), if that is what the code actually does.

### Listing 7.9 (cluster validation metrics)
- `ROBUSTNESS` — Add guards for degenerate cases (one cluster or all-singleton clusters) where metrics like silhouette score are undefined.

### Listing 7.10 (diversity calculation)
- `BLOCKER` — Add the missing closing parenthesis.
- `BUG` — Avoid double-counting redundant molecular pairs; exclude diagonal self-pairs (self-similarity biases the result).

### Listing 7.12 (pharmacophore distances)
- `BLOCKER` — Fix the function-name mismatch between definition and call; remove invalid backslash-plus-comment syntax.
- `ROBUSTNESS` — Handle the case where `distances_for_all_pairs` is empty.

### Listing 7.13 (KDE scoring)
- `ROBUSTNESS` — Check for empty or very small pharmacophore-pair distributions before fitting KDE.
- `BUG` — Normalize scoring across molecules with different numbers of pharmacophore pairs to avoid bias toward larger / feature-rich molecules.

### General
- `CONSISTENCY` — Ensure every listing runs from a clean notebook with imports/setup included; mark any abbreviated code explicitly as "schematic"/"partial" so it isn't mistaken for runnable code.

---

## Chapter 8 — Introduction to Deep Learning (EGFR)

### Listing 8.1
- `BLOCKER` — Variable-name mismatch: defines `Smiles` but later calls `scaffold_split(smiles, activities)`. Use a consistent name (`smiles`).
- `BLOCKER` — `criterion = nn.MSELoss()` is defined only inside `train_model(...)`, but `evaluate_model(...)` uses `criterion` as if global. Define it before both calls.
- `ROBUSTNESS` — Saving to `artifacts/ch08/kinase_binder_model.pth` fails if the directory is missing:
  ```python
  from pathlib import Path
  Path("artifacts/ch08").mkdir(parents=True, exist_ok=True)
  ```
- `ROBUSTNESS` — `ReduceLROnPlateau(..., verbose=True)` may be version-sensitive; verify against PyTorch 2.4.1 or state the tested version.
- `BLOCKER` — Typos: `cude` → `cuda`; input-layer size `2,0048` → `2048`.

### Listing 8.2
- `BUG` — Filters invalid SMILES but not the corresponding activity values, which misaligns molecules and labels. Build a cleaned list of `(smiles, activity, mol)` tuples so filtering preserves alignment.
- `ROBUSTNESS` — `Chem.MolFromSmiles(s)` returns `None` for invalid SMILES; check for `None` before scaffold/fingerprint calls.
- `BUG` — `activities[i]` can be label-based on a non-default index; use `activities.iloc[i]` (or convert to a NumPy array after cleaning).

### Listing 8.3
- `BUG` — Prefer RDKit's supported conversion over `np.array(mfpgen.GetFingerprint(mol))`:
  ```python
  arr = np.zeros((2048,), dtype=np.int8)
  DataStructs.ConvertToNumpyArray(fp, arr)
  ```
- `ROBUSTNESS` — Activity values may be strings/missing; add `activities = pd.to_numeric(activities, errors="coerce")` and drop NaNs before tensor creation.

### Activity target definition
- `BUG` — The code uses `activities.standard_value`, but pIC50 is not normally stored as `standard_value`. If it is raw IC50, the target is wrong unless converted: `pIC50 = -log10(IC50 in molar)`; for nM values, `pIC50 = 9 - log10(IC50 in nM)`. Verify units and endpoint filtering in code.

### Listing 8.4 (enrichment factor)
- `ROBUSTNESS` — Prevent division by zero when `n_actives == 0`, `n_compounds == 0`, or the top slice is empty.
- `BUG` — Avoid zero compounds at small cutoffs: `n_compounds = max(1, math.ceil(n_total * percentage))`.
- `CONSISTENCY` — The text says active is pIC50 > 6.3; the code uses `>= 6.3`. Make text and code match.

### Scaffold split
- `ENHANCEMENT` — Accumulate scaffold groups until the target train/test molecule ratio is reached, and report resulting split sizes, scaffold count, and activity distribution per split.

---

## Chapter 9 — Structure-based Drug Design with Active Learning

### Listing 9.1 (ligand extraction)
- `BUG` — `not protein and not water` may also select ions, cofactors, buffers, or multiple hetero ligands. Select the intended ligand explicitly by residue name, chain, or residue ID.
- `BLOCKER` — Typo `get_protein_ligand_idex` → `get_protein_ligand_idxs`.

### Listing 9.2 (docking box)
- `BLOCKER` — `Point` and `Box` are placeholders; define or import them so the listing is standalone.
- `CONSISTENCY` — Clarify whether padding means 5 Å total or 5 Å per side; if per side, the box dimension must add `2 * padding`.

### Listing 9.3 (docking)
- `BLOCKER` — Add missing imports for `MoleculePreparation` and `PDBQTWriterLegacy`.
- `ROBUSTNESS` — Check Meeko preparation succeeded before accessing `molsetup_list[0]`; handle invalid ligands (failed protonation, unsupported chemistry, bad geometry, failed PDBQT conversion).
- `BUG` (perf) — Do not recompute Vina maps for every ligand when the receptor and box are fixed; reuse maps.
- `CONSISTENCY` — Clarify the meaning of `cpu=0`, or use a fixed CPU count for reproducibility.

### Listing 9.4
- `BLOCKER` — Undefined variables/functions: `load_molecules`, `Preprocessor`, `fixed_receptor_file`, `receptor_pdbqt`, `ligand_pdbqt`, possibly `output_dir`. Define or reference them.
- `ROBUSTNESS` — Add checks for failed RDKit parsing, failed embedding, failed MMFF optimization; ensure `smiles_pool` and `X_pool` remain aligned after invalid molecules are skipped.

### Listing 9.5
- `BLOCKER` — Remove the duplicate unreachable `return`.
- `ROBUSTNESS` — `predict()` should force the fingerprint input to be 2D so a single molecule doesn't fail.

### Listing 9.6
- `ROBUSTNESS` — Prevent division by zero when the training set is empty; use batch-wise device transfer instead of moving the whole training set to GPU at once.

### Listing 9.7
- `ROBUSTNESS` — Check `MaxMinPicker.LazyBitVectorPick` compatibility with the fingerprint format; guard `n_samples > len(X_pool)`.

### Listing 9.8
- `BLOCKER` — `uncertainty_sampling()` computes `best_indices` but never returns it. Return it.
- `BUG` — Respect `batch_size` for uncertainty, PI, and EI acquisition (currently ignored); after Monte-Carlo dropout (`model.train()`), restore evaluation mode.
- `ROBUSTNESS` — Avoid loading the entire pool into GPU memory during each dropout pass.
- `CONSISTENCY` — State the minimization convention for greedy/PI/EI acquisition (lower Vina score is better).

### Listing 9.9
- `ROBUSTNESS` — Check whether `Chem.MolFromSmiles()` returns `None` (`Chem.AddHs(None)` crashes); check 3D embedding and MMFF status.
- `BUG` (perf) — Avoid creating a new `VinaDocking` object per molecule when receptor/box are fixed; ensure `deepdock_oracle()` does not depend on an implicit global `reference_df`.

### Listing 9.10
- `BUG` — Replace the mutable default argument `top_reference_smiles=[]` with `None`.
- `BUG` — Convert oracle outputs and `y_train` to NumPy arrays where needed; do not pass stale pool-relative indices to the oracle after shrinking the pool; prevent duplicate selected indices; handle a requested batch larger than the remaining pool.
- `CONSISTENCY` — Fix the repeated print that says "initial samples" on every iteration; make TensorBoard step numbering consistent.

### Listing 9.12
- `BLOCKER` — Define `CHAPTER` and `exp_dir`; ensure the artifacts directory exists before writing files.

### Figure 9.5 / results table (generated)
- `CONSISTENCY` — The table reports a best score near −20.66 while the Figure 9.5 caption says ≈ −10.5 kcal/mol. Reconcile the numbers produced by the code with the caption.

---

## Chapter 10 — Generative Models for De Novo Design (SMILES VAE-CYC)

### Listing 10.1
- `BLOCKER` — `build_vocab`, `_apply_substitutions`, `decode`, `sos_idx`, `eos_idx`, `unk_idx`, `pad_idx` are referenced later but not shown; include or explicitly reference their definitions.
- `ROBUSTNESS` — Character substitutions mapping `Cl`, `Br`, `[nH]`, `[H]` to placeholders `Q/W/X/Y` are fragile; assert those placeholder characters never occur in the SMILES vocabulary (or use a proper SMILES tokenizer).

### Listing 10.2
- `ROBUSTNESS` — Handle the empty-list case before `max(len(seq) for seq in encoded_sequences)`.
- `BUG` — Warn (or guard) when `max_length` truncates sequences and potentially removes `<eos>`.

### Listing 10.3
- `BUG` — Replace mutable default argument `hidden_dims=[512, 256]` with `hidden_dims=None`.

### Listings 10.3 / 10.4
- `BLOCKER` — Ensure `encode`, `decode`, `_pad_or_truncate`, `_initialize_weights` are fully shown or clearly referenced.
- `CONSISTENCY` — Compression-ratio arithmetic is wrong: a 100-token molecule with 128-dim embeddings is 12,800 values, not 128,000 (compression to 64 is 200:1). Fix the starting value in text/code.
- `BUG` — `F.cross_entropy` expects class indices, not one-hot targets; adjust the target format.

### Listing 10.5
- `ROBUSTNESS` — `.view()` can fail on non-contiguous tensors; use `.reshape()` or `.contiguous().view()`.

### Listing 10.6
- `BUG` — After `nn.init.normal_`, reset the padding embedding row to zeros if `padding_idx=0`.

---

## Chapter 11 — Graph Neural Networks for Drug-Target Affinity

### Listing 11.1 (molecular graph construction)
- `BUG` — `nx.Graph(edges)` omits isolated atoms; explicitly add all atoms as nodes before adding edges.
- `ENHANCEMENT` — Include bond features (bond order, aromaticity, conjugation, ring membership, stereochemistry) instead of only binary edges.
- `BUG` — Do not normalize one-hot atom features by their sum; this corrupts the categorical encoding and won't match GraphDTA-style preprocessing.
- `BUG` — Replace `np.matrix` with arrays (e.g. `np.eye(...)`).
- `CONSISTENCY` — Validate the stated 78-dimensional atom feature vector against the actual atom-symbol list / feature definitions.

### Listing 11.2 (atom features)
- `ROBUSTNESS` — `one_of_k_encoding` may fail for out-of-range degrees; use an unknown-token version or validate values first.
- `ROBUSTNESS` — Check RDKit compatibility around implicit-valence APIs (some have changed/been deprecated).

### Listing 11.3 (protein graph construction)
- `ROBUSTNESS` — Add a check that the contact-map shape matches the protein sequence length.
- `CONSISTENCY` — Make the self-loop logic explicit (if identity is added before thresholding, say so).
- `BLOCKER` — Define/show `CONFIG` before use.
- `BUG` — Return `edge_index` in PyG format `[2, num_edges]`, not `[num_edges, 2]`.

### Listing 11.4 (residue features)
- `ROBUSTNESS` — Handle unknown/uncommon amino acids (`X`, `U`, `O`, `B`, `Z`, gaps/missing residues).
- `CONSISTENCY` — Reconcile the feature-dimension conflict: text mentions 21-dim one-hot and 21-dim PSSM, but the alphabet appears to have 20 symbols. Clarify whether an unknown/gap token is included.

### PSSM calculation
- `BUG` — The denominator should count only accepted aligned sequences, not all lines read from the file.
- `BUG` — Handle gaps: if gaps are skipped, the denominator should reflect non-gap observations for that position.
- `CONSISTENCY` — The text describes background frequencies (b_a) but the code uses a uniform pseudocount; make them consistent.
- `BUG` — `dic_normalize()` sets `dic['X'] = (max_value + min_value) / 2.0` after normalization, which is not the normalized midpoint (should usually be `0.5`).
- `BUG` — `dic_normalize()` mutates dictionaries in place, causing repeated-normalization bugs; return a new dictionary instead.

### Listings 11.7 & 11.8 (model definition / forward)
- `BLOCKER` — Fatal variable-name bug in 11.7: the function argument is `mol_data` but the body uses `data_mol`.
- `BLOCKER` — Inconsistent layer names: earlier layers are `mol_fc_g1` / `mol_fc_g2`, but the code uses `mol_fc1_g1` / `mol_fc2_g2`.
- `BLOCKER` — 11.8 calls `forward_molecule` and `forward_protein`, but 11.7 defines `forward_mol`. Make the function names match exactly.

### General
- `ROBUSTNESS` — Perform a full code review and run all listings end-to-end in a clean environment.

---

## Chapter 12 — Transformers for Protein Structure Prediction

### Listings 12.1–12.3 (transformer / ESM-2 classifier)
- `BLOCKER` — Undefined classes/methods: `PositionalEncoding`, `Encoder`, and `mean_pooling` (called in 12.3 but not shown in the class). Include full definitions or clearly mark the listing as abbreviated. Also specify vocabulary construction and special-token handling.
- `ROBUSTNESS` — Move batch tensors to the model device (the listings assume they are already there).
- `ROBUSTNESS` — Set label dtype explicitly: `torch.tensor(labels, dtype=torch.long, device=device)` (CrossEntropyLoss needs `torch.long`).
- `ROBUSTNESS` — Use `model.eval()` and `torch.no_grad()` during validation/testing; add optional gradient clipping.
- `ROBUSTNESS` — Check attention-mask shape and dtype; avoid empty-group errors in the short/long sequence accuracy analysis; align the boolean `uncertain_mask` with `df_test` order/length.

### Attention-mask semantics
- `CONSISTENCY` — State whether masks use `1/0`, `True/False`, or additive `0/-inf` values, and specify the expected shape, e.g. `(batch, seq_len)` or `(batch, heads, query_len, key_len)`.

### Special tokens / vocabulary
- `CONSISTENCY` — `vocab_size=25` may not match amino acids plus padding/mask/unknown/CLS/EOS/BOS; define the vocabulary explicitly. Ensure pooling does not average special tokens ([CLS], [EOS], [MASK]) unless intended.

### MLM loss
- `BUG` — The MLM loss must ignore unmasked positions; set labels for unmasked tokens to `ignore_index=-100` with `nn.CrossEntropyLoss`. Never mask padding or structural special tokens unless deliberately designed.

### Parameter count
- `CONSISTENCY` — The stated 273,794 trainable parameters depends on the final `Encoder`/embedding/layer-norm/bias implementation; recompute after the code is fixed.

---

## Chapter 13 — Multimodal AI Systems

*Chapter 13 is a survey chapter; it contains essentially no runnable notebook listings.* The only code/markup-adjacent item:

- `CONSISTENCY` — Render protein-design search-space notation as `20^N` and `20^430` (not "20N" / "20430"). This is manuscript/markdown formatting rather than a notebook fix.

All other Chapter 13 feedback is claim-softening, numerical-accuracy (e.g., Rentosertib placebo FVC −62.3 mL vs −20.3 mL), and citation work — out of scope for the coding agent.

---

## Appendix C — Knowledge Distillation (Hierarchical Molecular Generation)

### Listing C.1
- `BLOCKER` — Remove the stray `[CA]` manuscript marker, which makes the listing non-executable:
  ```python
  self.clusters, self.atom_cls = \
  [CA]self.find_clusters()
  ```
- `ROBUSTNESS` — Guard against empty cluster lists before indexing `clusters[0]`:
  ```python
  if clusters and 0 not in clusters[0]:
      ...
  ```
- `CONSISTENCY` — The code moves the cluster containing atom 0 to the front; explain in text why (canonicalization / root selection / decoder requirement), since it introduces order dependence.

### Listing C.2
- `BLOCKER` — `roots` is used (`self.embed_root(hmess, tensors, roots)`) but not defined; include its construction or explain that it is produced elsewhere.
- `BLOCKER` — Constants `atom_size`, `bond_size`, `MAX_POS` are placeholders; define them in the listing or immediately before it (or mark the listing as excerpted pseudocode).

### Listing C.3
- `CONSISTENCY` — `z_log_var = -torch.abs(...)` forces log-variance ≤ 0 (variance ≤ 1), which is non-standard VAE practice. Justify the choice in text (stabilize training / avoid diffuse latent / mitigate posterior collapse).

### Listing C.4
- `CONSISTENCY` — Matching only the shared prefix `:min_dim` discards teacher information when the teacher latent dimension is larger. Add a learned projection layer (e.g. `projected_teacher = teacher_proj(teacher_latent)`), or explicitly label truncation as a tutorial simplification.
- `ROBUSTNESS` — `return torch.tensor(0.0)` can create a CPU tensor while the model is on GPU. Use a device-aware tensor:
  ```python
  torch.tensor(0.0, device=student_features.device)   # or student_features.new_tensor(0.0)
  ```
- `CONSISTENCY` — The text discusses motif-level and atom-level (hierarchical) representations, but the listing matches only `root_vecs`. Expand the implementation (root/motif/atom losses) or revise the text to say only root-level matching is demonstrated.

### Listing C.5
- `BLOCKER` — Remove the `[CA]` marker in the warmup calculation:
  ```python
  self.config.WARMUP_EPOCHS \
  [CA]* steps_per_epoch
  ```
- `ROBUSTNESS` — Handle `WARMUP_EPOCHS == 0` to avoid division by zero in learning-rate scheduling.

### Listing C.6
- `ROBUSTNESS` — Check that the dataloader is non-empty before dividing epoch losses by `steps_per_epoch`.
- `ROBUSTNESS` — Gradient clipping (`clip_grad_norm_`) controls magnitude but does not repair NaN/inf gradients; add a finite-value check, e.g. `if not torch.isfinite(loss): continue`.

### Listing C.7
- `BUG` — Zeroing every 1-D parameter (`if param.dim() == 1: nn.init.constant_(param, 0)`) also zeros biases and normalization scale parameters, which is usually undesirable. Prefer name-based checks:
  ```python
  if "bias" in name:
      nn.init.zeros_(param)
  elif "norm.weight" in name:
      nn.init.ones_(param)
  ```
  If the current behavior is intentional, document why.

### Offline vs online distillation
- `CONSISTENCY` — The text implies teacher and student train together, but the implementation freezes the teacher (`teacher.eval()` + `torch.no_grad()`) — i.e. offline KD. State explicitly which approach is implemented.

---

## Appendix D — Technical Deep Dive into Protein Structure Prediction

*No executable code listings.* The only actionable item for a documentation-capable agent:

- `CONSISTENCY` — Fix missing/broken references and placeholder citation markers ("MULTIPLE REFERENCES LOST"). This is bibliography/reference repair, not notebook code.

---

## Appendices with no notebook code to fix

For completeness (these appear in the collated reviewer file but contain no code listings to audit):

- **Appendix A (Glossary)** — definitions only; the reviewer explicitly notes "no executable source code is present." All fixes are terminology corrections (e.g., koff/kon/KD, ETKDG, therapeutic index), not notebook code.
- **Appendix B (Chemical Data Repositories)** — prose/statistics corrections (e.g., PubChem count typo), no code listings.
- **Appendix E (Extended Technical Material)** — prose commentary on gradient descent, RANSAC, SVR, PCA, etc.; no listing-level code defects flagged.
- **Appendix G (Chapter Exercises)** — exercise design feedback; no code defects.

---

## Quick index of execution-blocking (`BLOCKER`) items

Fix these first — the notebooks will not run as printed until they are resolved:

- **Ch1:** `>>>` output inside executable cells (Section 1.2).
- **Ch2:** 2.1 (`LoadSF`→`LoadSDF`), 2.2 (`RO5_PROPS`), 2.4 (`\ #A`), 2.5 (import), 2.6 (`\ #A`), 2.9 (indentation + imports), 2.10 (`query_index`/`query_idx`).
- **Ch3:** 3.6 (imports), 3.7 (`RANDOM_SEED`), 3.14 (indentation + imports), 3.17 (extra comma), 3.18 (`gs.linear_sgd`), 3.20 (trailing space).
- **Ch4:** 4.2 (`Chem` import), 4.6 (`esol_val_pred`), 4.7 (`features`), 4.9 (`SGDRegressor` import + `engineered_features`).
- **Ch6:** 6.6 (`hiv_tar1_lnkd_X_DR`), 6.7 (guard `best_features` None), 6.9 (array→scalar `[0]`).
- **Ch7:** 7.1 (`fp_size`/parens), 7.2 (brace + `RANDOM_SEED`/`n_nodes`/`m_nodes`), 7.3 (`\`+comment, ellipsis, undefined vars), 7.5 (`reaction_smirks`/`reaction_smarts`), 7.6 (`frag_mols`/`fragMols`), 7.7 (imports), 7.10 (closing paren), 7.12 (name mismatch + `\`).
- **Ch8:** 8.1 (`Smiles`/`smiles`, `criterion` scope, `cude`, `2,0048`).
- **Ch9:** 9.1 (`get_protein_ligand_idex`), 9.2 (`Point`/`Box`), 9.3 (imports), 9.4 (undefined names), 9.5 (unreachable return), 9.8 (missing return), 9.12 (`CHAPTER`/`exp_dir`).
- **Ch10:** 10.1 (undefined vocab helpers/indices), 10.3/10.4 (undefined methods).
- **Ch11:** 11.3 (`CONFIG`), 11.7 (`mol_data`/`data_mol`, layer names), 11.8 (`forward_molecule`/`forward_protein` vs `forward_mol`).
- **Ch12:** 12.1–12.3 (`PositionalEncoding`, `Encoder`, `mean_pooling`).
- **App C:** C.1 (`[CA]`), C.2 (`roots` + constants), C.5 (`[CA]`).
