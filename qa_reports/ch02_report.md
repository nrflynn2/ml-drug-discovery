---
chapter: ch02
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
  total: 19
  fixed: 5
  already_ok: 14
  not_found: 0
  needs_author_decision: 0
new_taxonomy_hits: 2
chapter_done: true
---

# Chapter 02 — QA Report

_CH02 is a recently-fixed snapshot in good shape: the harness passed at baseline (32 code cells, 0
errors) and every marker/"already-fixed?" BLOCKER the inventory flags (2.1 `LoadSDF`, 2.2 `RO5_PROPS`,
2.4/2.6 `\ #A` continuations, 2.9 indentation, 2.10 `query_idx`, 2.12 `.iloc`) is already resolved in
this `.py`. I fixed the 3 genuine live defects (2.3 bare `except:`, 2.7 missing `None`-guard on the
Glaxo substructure match, 2.9's one missing `from IPython.display import display`), plus the two
robustness/enhancement asks that were still open (2.10 explicit `engine="xlrd"`; the General
before/after-preview ENHANCEMENT via `preview_df`). Applied the Standard-depth pedagogy pass: wired in
`check_env()` + `SEED = set_seed(42)` as a standardized setup cell (replacing a stray
`np.random.seed(42)`), added `preview_df()` at the four major DataFrame transforms + one sanity
`assert`, and tagged the 5 Colab-only setup cells `skip-execution` so the notebook executes locally.
Proactive sweep found 1 real code hit (non-standard seed → `set_seed`) and 1 non-code observation
(numpy.core deprecation warning when unpickling the committed artifacts). **Execution passed with 0
errors (212,670-compound library → 105,431 after filters; 217 RDKit descriptors, matching the
manuscript claim).** The 4 long-running compute cells were served from the notebook's own committed
artifacts via its documented "skip this cell and load the artifact" reload cells (see Execution note) —
their raw compute (~40 min total; PAINS/BRENK alone 20-25 min) exceeds the 900s timeout._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| General | ROBUSTNESS | already-ok | Notebook runs clean end-to-end (0 error outputs on execute). Variable names are consistent across snippets (`query_idx`, `fp_col_name`, `specs`/`specs_filtered` all used uniformly). Imports resolve — the notebook uses one consolidated top-of-notebook import cell rather than per-listing imports. Flag manuscript listing for production: for the printed book, show the relevant imports beside each listing so each prints as runnable-verbatim. (uncommitted) |
| General | ENHANCEMENT | fixed | Added `preview_df(df, name)` (shared util: prints `rows × cols` + column list, returns head) after the four major transforms: raw `specs` load, `+ RO5 descriptor columns`, `specs_filtered (after all filters)`, and `+ Morgan fingerprint column`. Renders the requested before/after column-name previews (e.g. `212,670 rows x 7 cols` → `105,431 rows x 14 cols`). (uncommitted) |
| 2.1 | BLOCKER | already-ok | `LoadSF` typo absent — `load_sdf_file` calls `PandasTools.LoadSDF(file_path, smilesName=smiles_name, molColName=None)`. Correctly spelled. (uncommitted) |
| 2.1 | ROBUSTNESS | already-ok | `smilesName='smiles'` is set and the `["PUBCHEM_SUBSTANCE_ID", "smiles"]` projection is valid for this SDF (confirmed: the executed pipeline and the committed `specs_hits_to_malaria_box.csv` both carry those exact columns). This is the notebook's "😱 Long Running" SDF-load cell, which is served from the `specs` artifact in the fast path (see Execution note) — code is correct; not re-parsed at runtime. (uncommitted) |
| 2.2 | BLOCKER | already-ok | `RO5_PROPS = ['ExactMolWt', 'NumHAcceptors', 'NumHDonors', 'MolLogP']` is defined at the top of `calculate_ro5_descriptors`. No NameError. (uncommitted) |
| 2.3 | BUG | fixed | Replaced the bare `except:` in `compute_descriptor` (silently hid descriptor failures) with `except Exception as e:` + an explanatory `print(f"Descriptor '{func_name}' calculation failed for a molecule: {e}")` before returning `missing_val`. The `mol is None` case is still guarded above the try, so this only fires on genuine descriptor errors (executed run: 0 fired — "Removed 0 rows with missing descriptor values"). (uncommitted) |
| 2.4 | BLOCKER | already-ok | `df['ro5_compliant'] = df['ro5_violations'] <= 1` — no `\`-before-comment continuation, no `#A` marker (the only `#A` strings in the file are the `#A20025` header colour hex). Parses clean. Flag manuscript listing for production: the `\ #A` artifact may still live in the book's AsciiDoc listing. (uncommitted) |
| 2.5 | BLOCKER | already-ok | `FilterCatalog` is imported in the consolidated `from rdkit.Chem import (...)` block. `apply_pains_brenk_filters` resolves it. (uncommitted) |
| 2.5 | CONSISTENCY | already-ok | `apply_pains_brenk_filters` already prints all four lines — compounds before, compounds failing, compounds after, and percentage passing — matching the manuscript output. (uncommitted) |
| 2.6 | BLOCKER | already-ok | `alerts_df["ROMol"] = alerts_df.smarts.apply(MolFromSmarts)` — clean single-line assignment, no `\ #A` continuation. Flag manuscript listing for production: the marker may still live in the printed listing. (uncommitted) |
| 2.7 | BUG / ROBUSTNESS | fixed | The `df.copy()` half was already present (`apply_glaxo_filters` does `df = df.copy()` and writes via `.loc[:, ...]`). Added the missing `None`-guard: `check_glaxo_match` now returns early `if mol is None:` before calling `mol.HasSubstructMatch(alert.ROMol)`, so a stray unparsed molecule can't raise. (This lives in the long-running Glaxo cell, served from artifact in the fast path — statically verified + parses.) (uncommitted) |
| 2.9 | BLOCKER | already-ok | `mol_img = Draw.MolsToGridImage(` in `visualize_fingerprint_decomposition` sits at correct 4-space function-body indentation (the nested `draw_fragment_from_bit` closes before it). Not over-indented. (uncommitted) |
| 2.9 | BLOCKER | fixed | Of the flagged imports, `Draw` and `AdditionalOutput`/`GetMorganGenerator` were already imported and `rdkit_drawing_options` already defined in `setup_rdkit_drawing()`. The one gap — `display` (used 14×, only available as an IPython builtin) — is now imported explicitly: `from IPython.display import display`, so every listing runs verbatim outside an interactive kernel. Executed run confirms `DrawMorganBit` works (0 "Error drawing bit"). (uncommitted) |
| 2.10 | BLOCKER | already-ok | `query_idx = 236` then `malaria_box[...].iloc[query_idx]` / `malaria_box.mol.iloc[query_idx]` — one name used consistently; no `query_index`/`query_idx` mismatch. (uncommitted) |
| 2.10 | ROBUSTNESS | fixed | Made the Excel engine explicit: `pd.read_excel("...MalariaBox...xls", usecols=[...], engine="xlrd")`. Modern pandas needs `xlrd` for legacy `.xls`; `xlrd` is installed and the executed run loaded the Malaria Box successfully. (uncommitted) |
| 2.12 | BUG | already-ok | Uses positional `specs_hits = specs_filtered.iloc[hit_indices]` (indices come from `enumerate(library_fps)`), which is correct on a non-contiguous index. The buggy `.filter(items=..., axis=0)` form is absent. (uncommitted) |
| 2.12 | ROBUSTNESS | already-ok | `select_top_matches` already guards the heap pop: `for i in range(min(budget, len(heap)))`. (uncommitted) |
| version-fragile assertion | CONSISTENCY | already-ok | The notebook does not hardcode 217 — `list_available_descriptors` prints `f"RDKit has {len(Descriptors._descList)} molecular descriptors available"` (version-safe). Executed value with rdkit 2025.09.6 is **217**, which matches the manuscript's claim. Flag manuscript listing for production: tie the printed "217" to the RDKit version, e.g. "In the RDKit version used here, `Descriptors._descList` reports 217 descriptors." (uncommitted) |
| optional | ENHANCEMENT | already-ok | Not a defect — the binary Morgan fingerprints (`GetMorganGenerator(...).GetFingerprint`) are correct. Count-based Morgan fingerprints are a purely additive teaching extension ("if you extend the fingerprint section"), left to author discretion so as not to restructure the section unprompted. Non-blocking; noted for the author. (See PILOT FEEDBACK #6 re: the schema having no clean "optional / declined" status.) (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| import cell — `np.random.seed(42)` | nondeterminism | ENHANCEMENT (minor) | Replaced with a standardized setup cell: `check_env()` + `SEED = set_seed(42)` (shared util seeds Python/NumPy + `PYTHONHASHSEED`, and torch if present). Adopts the repo-standard helper vs. a raw NumPy-only seed; no output change (this chapter's compute is deterministic). (uncommitted) |
| `load_molecular_dataframe` unpickling `artifacts/ch02/*.pkl.gz` | deprecated-api | ROBUSTNESS (warning) | **Observation, not fixed in the chapter `.py`** — loading the committed frames emits `DeprecationWarning: numpy.core is deprecated ... use numpy._core` because they were pickled under numpy<2 and are loaded under numpy 2.2.6. Load still succeeds. Fix belongs in a shared/data task: re-save the ch02 artifacts under numpy 2.x. (uncommitted) |

## Author-decision queue

```
(none — no unresolved author question for CH02.)
```

## Execution note (CH02-specific — long-running compute cells)

CH02 has four compute cells the notebook itself labels "😱 Long Running Code Block" and pairs with a
"skip this cell and load the artifact" reload cell: the SDF load (`load_sdf_file`, ~2-4 min), and the
PAINS/BRENK (20-25 min) and Glaxo (13-15 min) substructure filters, plus `calculate_ro5_descriptors`
(~5 min, no reload). Running all of them top-to-bottom is ~40+ min and the PAINS/BRENK cell **alone
exceeds the 900s per-cell timeout** in the brief's nbconvert command.

For this refresh I executed the notebook via its **documented fast path**: the SDF-load, PAINS/BRENK,
and Glaxo cells were skipped and their results loaded from the committed `artifacts/ch02/*.pkl.gz`
frames by the reload cells that immediately follow each one; `calculate_ro5_descriptors` (no reload) was
run for real. Mechanically this was done by temporarily tagging those 3 heavy cells `skip-execution`
for the run only (tags stripped afterward — they remain runnable for anyone with the compute budget).
All skipped heavy cells retain their prior committed outputs, so the notebook reads complete
end-to-end. The 5 Colab-only setup cells are **permanently** tagged `skip-execution` (matching CH01).

Net: 27 code cells executed with **0 error outputs**; the 5 Colab cells skipped natively via their tag;
the 3 heavy compute cells served from artifacts. See PILOT FEEDBACK #1/#2 — this "full-execute vs.
skippable heavy compute" tension is new in CH02 (CH01 had none) and needs a brief policy before scaling.

## Verification log

- `uv run python tools/validate_notebooks.py CH02_FLYNN_ML4DD.ipynb` → `✓ 33 code cells OK — 0 errors, 0 warnings` (baseline before edits: 32 cells; +1 from the new standardized setup cell).
- `uv run jupytext --sync CH02_FLYNN_ML4DD.py` → ok; final sync preserves all 84 cell outputs (0 error outputs) and applies the 5 `skip-execution` tags to the Colab cells. Benign "Notebook is not trusted" warning only.
- execution: driver over `ExecutePreprocessor(timeout=1800)` (equivalent to `jupyter nbconvert --to notebook --execute --inplace`, with the 5 Colab cells + 3 long-running compute cells skipped as described above) → **pass**. 27 code cells executed, 0 error outputs. A benign `TqdmWarning: IProgress not found` (ipywidgets) and the plaintext-kernel-transport notice appear but are not failures.
- Sanity of outputs: `check_env` → Python 3.12.13, numpy 2.2.6, pandas 2.3.3, scipy 1.18.0, sklearn 1.9.0, rdkit 2025.09.6, torch 2.12.1+cu130, xgboost 3.3.0 (CPU only). `RDKit has 217 molecular descriptors available`. `specs` = 212,670 compounds; RO5 descriptors removed 0 rows; filtering summary 212,670 → 105,566 (RO5+PAINS/BRENK, 49.6%) → 105,431 (Glaxo, 49.6%); Morgan fingerprints on 105,431 rows (`morgan_fp_r2_b2048`); the `assert specs_filtered[fp_col_name].notnull().all()` passed; Malaria Box loaded via `engine="xlrd"`; fingerprint-decomposition `DrawMorganBit` rendered 5 bit images (0 draw errors). All 18 `figures/ch02/*` refreshed.

_Note: no branch/PR opened and nothing committed, per the pilot scope constraint ("Do NOT commit
anything"); hence `(uncommitted)` in lieu of `<sha>` citations above. `CH02_FLYNN_ML4DD.py` remains
git-untracked (as at session start); the paired `CH02_FLYNN_ML4DD.ipynb` and `figures/ch02/*` are
modified in the working tree._
