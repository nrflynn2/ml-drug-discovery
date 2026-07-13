---
chapter: ch07
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
  total: 21
  fixed: 11
  already_ok: 9
  not_found: 0
  needs_author_decision: 1
new_taxonomy_hits: 8
chapter_done: false
---

# Chapter 07 — QA Report

_As the calibration predicted, every parsing-level BLOCKER in the inventory is already resolved in this
snapshot (the harness passed at baseline: 60 code cells, 0 errors), so the `fp_size` parens, the SOM
brace, the `\`+ellipsis, `reaction_smirks`/`reaction_smarts`, `frag_mols`/`fragMols`, the 7.7 imports,
the 7.10 paren and the 7.12 name mismatch are all `already-ok`. **The real work was in the live defects
underneath them.** The headline fix is 7.1's activity labels, which were **fully inverted**: `f_avg_IC50`
is the COVID Moonshot fluorescence IC50 in *micromolar* (0.003–198 µM in this file; the companion
`f_avg_pIC50` column is entirely empty), so a *lower* value means a *more potent* compound — yet the code
labeled `IC50 >= 5 µM` as **Active**. The 782 most potent compounds were labeled "Inactive" and the 1,144
weakest were labeled "Active", which corrupted every downstream analysis that keys on `label == 1` (the
SOM node labeling, the whole drug-repurposing hit-recovery evaluation, and the UMAP `train_labels`). I
also had to recompute the labels **after** the load, because the committed `processed_CH07_activity_data.csv`
cache carries the old inverted labels and is loaded in preference to the raw file — fixing only the
function would have left the bug live. Beyond that: 7.10's redundancy metric subtracted `n` for
"self-comparisons" against a **zero** diagonal, which under-counted redundant pairs and could even go
negative — this was *manufacturing* the "close to 0 redundancy" result the notebook's own prose cites
(see Q2); and 7.13's score took a `max` over each pair type, biasing toward feature-rich molecules, so it
is now a mean-of-log-densities that is genuinely size-invariant. The proactive sweep found **two stacked
execution blockers** in the SOM starburst cell that a fresh top-to-bottom run hits immediately:
`matplotlib.cm.get_cmap` (removed in Matplotlib 3.9) and — revealed only once that was fixed — a NumPy-2
`repr` change that emitted `"rgb(np.float64(111.1), ...)"` and made Plotly reject the colorscale. A third,
`target` used one line *before* it is defined, was a guaranteed `NameError` on any fresh kernel. One
author question is open (Q1, the toroidal-SOM claim), so `chapter_done: false`._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| 7.1 | BLOCKER | fixed | Two halves. The **parens** around the fingerprint conversion are fine (harness parses clean) — `already-ok`, flag manuscript listing for production. The **`fp_size`** half was a live bug, just not the one described: the caller passed `generate_morgan_fingerprint(x, radius=2, n_bits=2048)` while the signature declared `fp_size=2048`, so the raw-data path raised `TypeError: unexpected keyword argument 'n_bits'`. It was masked only because the committed processed-CSV cache short-circuits that path — delete the cache and the chapter breaks. Renamed the parameter to `n_bits` (matching the caller and the sibling `generate_fingerprints_for_clustering`). (uncommitted) |
| 7.1 | BUG | fixed | Adopted `DataStructs.ConvertToNumpyArray()` into a preallocated `np.zeros((n_bits,), dtype=np.int8)`. **Honest note:** the prior `np.array(fp_gen.GetFingerprint(mol))` was *not* producing wrong numbers — I verified it is bit-for-bit identical to `ConvertToNumpyArray` on this RDKit (both give the same 45 set bits). So this is an idiom/robustness fix (explicit dtype + the documented RDKit API), not a correctness rescue. (uncommitted) |
| 7.1 | BUG | fixed | **The most important fix in the chapter — the labels were inverted.** `f_avg_IC50` is a raw IC50 *concentration* in µM (range 0.0027–198, median 13.2; the `f_avg_pIC50` column is empty, so there is no log-scale column in play). Lower IC50 = more potent. The code did `np.where(f_avg_IC50 >= 5, 1 /*Active*/, np.where(f_avg_IC50 < 5, 2 /*Inactive*/, 3))` — backwards. Verified against the data: cached `label==1` ("Active") = 1,144 compounds **all with IC50 ≥ 5 µM**, and `label==2` ("Inactive") = 782 compounds **all with IC50 < 5 µM**. Extracted an `assign_activity_labels()` helper with a named `IC50_ACTIVE_THRESHOLD_UM = 5`, so Active = IC50 < 5 µM (n=782, max 4.98) and Inactive = IC50 ≥ 5 µM (n=1,144, min 5.00). **Critically, the labels are also recomputed after the load block**: the committed `data/ch07/processed/processed_CH07_activity_data.csv` has the old inverted labels baked into a `label` column and is loaded in preference to the raw CSV, so fixing the function alone would have left the bug fully live at runtime. (uncommitted) |
| 7.1 | ROBUSTNESS | fixed | NaN IC50 did land in class 3, but only as an *implicit* fall-through (NaN fails both `>= 5` and `< 5`, so it dropped to the `else`). Made it explicit and first: `np.where(ic50.isna(), LABEL_UNKNOWN, ...)`. 136 compounds have no IC50 and are now deliberately, not accidentally, `Unknown`. Also replaced the magic `1/2/3` with `LABEL_ACTIVE/LABEL_INACTIVE/LABEL_UNKNOWN`. (uncommitted) |
| 7.2 | BLOCKER | already-ok | All three sub-items are clean: `som_param_grid` has its closing brace; `RANDOM_SEED` is defined well before use (I rewired it to `set_seed(42)` — see Table 2); and `n_nodes, m_nodes = 16, 16` is defined before the gridspec uses it. Harness parses clean. Flag manuscript listing for production — these artifacts may still live in the book's AsciiDoc. (uncommitted) |
| 7.2 | CONSISTENCY | needs-author-decision | The code passes `'topology': 'rectangular'` to MiniSom, and MiniSom's `topology` only selects `rectangular` vs `hexagonal` **node layout** — it has no toroidal/wrap-around option at all, so the implementation is definitively *not* toroidal. The word "toroidal" appears nowhere in the notebook (`grep` → 0 hits), so this is purely a book-prose claim I cannot see or rewrite → **Q1**. (uncommitted) |
| 7.3 | BLOCKER | already-ok | Nothing to fix: no backslash-plus-comment continuations (`grep -P '\\\s*#'` → none), no Unicode ellipsis `…` (→ none), and both names resolve — `drh_compounds` is assigned inside `identify_repurposing_candidates`, and `retrospective_hits` is defined as a literal `set([...])` before it is passed in. Flag manuscript listing for production. (uncommitted) |
| 7.5 | BLOCKER | already-ok | No name mismatch: the code both defines **and** uses `reaction_smirks` (`rxn = AllChem.ReactionFromSmarts(reaction_smirks)`). `reaction_smarts` does not appear anywhere (`grep` → none). (uncommitted) |
| 7.5 | ROBUSTNESS | already-ok | All three concerns are already handled. (a) The loop is `while "[*]" in working_smi and iterations < max_iterations` with `max_iterations = 10` — bounded, no infinite loop. (b) `core_smi` is **not** mutated across attempts: each expansion starts from `working_smi = core_smi` and Python strings are immutable, so rebinding `working_smi` cannot touch the core. (c) I empirically validated the `:0` atom mapping — `ReactionFromSmarts("[*:0][#0].[#0][*:1]>>[*:0][*:1]")` builds fine and correctly fires, consuming exactly one attachment point per application (`[*]c1ccccc1[*]` + `[*]CC` → `*c1ccccc1CC`). Map index 0 is accepted by RDKit here and the reaction does what the chapter intends. (uncommitted) |
| 7.6 | BLOCKER | already-ok | One name throughout: `frag_mols` is built, appended to, and passed to `BRICS.BRICSBuild(frag_mols, maxDepth=1)`. `fragMols` does not exist (`grep` → none). (uncommitted) |
| 7.6 | BUG | fixed | **There is no bare `except:` anywhere in this chapter** (`grep "except:"` → 0 hits), so that half is already-ok. But the item's *intent* — "so sanitization/parsing/chemistry failures are visible" — was live: the BRICS build loop caught `Exception as e` and then silently `continue`d, discarding `e`. Narrowed it to `except (Chem.MolSanitizeException, ValueError, RuntimeError)` and made failures visible **without flooding the output**: BRICSBuild proposes thousands of invalid recombinations, so per-failure prints would bury the notebook. Instead the loop tallies failures by exception type in a `Counter` and prints one summary line (`Skipped N unbuildable recombinations (AtomValenceException x…, KekulizeException x…)`). (uncommitted) |
| 7.7 | BLOCKER | already-ok | Both names resolve from the consolidated top-of-notebook import cell: `DataStructs` (via `from rdkit.Chem import (... DataStructs ...)`) and `AgglomerativeClustering` (via `from sklearn.cluster import AgglomerativeClustering`). Flag manuscript listing for production: print the relevant imports beside the listing so it reads as runnable-verbatim. (uncommitted) |
| 7.7 | CONSISTENCY | fixed | Confirmed the reviewer's suspicion — the clustering example really does use **RDKit path (topological) fingerprints** (`rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)`), *not* Morgan, even though the chapter's first half uses Morgan for the SOM/UMAP work. Worse, the signature was actively misleading: `generate_fingerprints_for_clustering(molecules, radius=2, n_bits=2048)` advertised **Morgan-style parameters that the body completely ignored**. Changed the signature to the parameter it actually honors (`max_path=5`) and stated the fingerprint type explicitly in the docstring. Flag for the author: make sure the book prose for §7.2 says *path/topological*, not Morgan. (uncommitted) |
| 7.9 | ROBUSTNESS | fixed | `evaluate_clustering` relied on a blanket `try/except Exception` to absorb the degenerate cases. Added explicit up-front guards — silhouette / Calinski-Harabasz / Davies-Bouldin are only *defined* for `2 <= n_labels <= n_samples - 1`, and **both** extremes are genuinely reachable here (a loose Butina cutoff collapses everything into one cluster; a tight one makes every molecule a singleton). The function now detects each case, prints which degeneracy occurred, and returns `None` metrics instead of raising; the residual `except` is narrowed to `ValueError`. (uncommitted) |
| 7.10 | BLOCKER | already-ok | No missing parenthesis — the diversity block parses clean (harness). Flag manuscript listing for production. (uncommitted) |
| 7.10 | BUG | fixed | `calculate_diversity` was **already correct** (its matrix has a zero diagonal, it subtracts `np.trace`, and its `n*(n-1)` denominator exactly matches the double-counted numerator). The live bug was next door in **`calculate_redundancy`**: it computed `(similarity_matrix > threshold).sum() - n  # Exclude self-comparisons`, but that matrix's diagonal is **zero** (self-similarity is never computed — the loop is `for j in range(i+1, n)`), so no self-pair was ever counted and subtracting `n` simply deleted `n` real redundant pairs. It under-counts and can even go **negative**. Verified on a toy set where 2 of 6 pairs are redundant: true ratio 0.333, old formula → **0.000**, fixed formula → 0.333. Removed the `- n`. Note this bug was *manufacturing* the near-zero redundancy the notebook's prose reports → **Q2**. (uncommitted) |
| 7.12 | BLOCKER | already-ok | No mismatch: `calculate_pairwise_distances` and `calculate_pairwise_distances_within_molecule` are each defined and called under exactly those names. No backslash-plus-comment syntax anywhere (`grep` → none). (uncommitted) |
| 7.12 | ROBUSTNESS | fixed | The empty case was a real latent crash. `calculate_pairwise_distances` legitimately returns `np.array([])` when a pharmacophore type is absent, and the very next line did `f"... range [{distances.min():.2f}, {distances.max():.2f}]"` — `.min()` on an empty array raises `ValueError: zero-size array to reduction operation minimum`. Guarded the summary print so an absent pair reports "no distances found" instead of taking down the cell. (uncommitted) |
| 7.13 | ROBUSTNESS | fixed | `fit_kernel_density` only guarded `len(distances) == 0`, but it runs a 5-fold `GridSearchCV`, which raises whenever there are fewer samples than folds. Changed the guard to `len(distances) < cv_folds` so a 1–4-point distribution returns `None` with an explanatory message rather than exploding inside cross-validation. (The call site's separate `> 10` check remains.) (uncommitted) |
| 7.13 | BUG | fixed | Confirmed the size bias. `score_molecule_with_kdes` took `np.max(probabilities)` **within** each pharmacophore pair type — and a max over *more* samples is systematically larger, so a feature-rich molecule scored higher simply for having more distances to draw from. Replaced the per-type `max` with the **mean log-density** across that type's distances (which is invariant to how many distances there are), then averaged across the pair types present. The score is now a mean log-density *per pharmacophore pair*, comparable across molecules of different size. Also changed the "unscorable molecule" return from `0.0` to `None`: on a log-density scale `0.0` is an excellent score, so a featureless molecule was previously sorting to the **top** of the ranking (it now sorts last as NaN). (uncommitted) |
| General | CONSISTENCY | already-ok | The notebook runs clean top-to-bottom from a fresh kernel (0 error outputs) using one consolidated import cell, and I found no abbreviated/schematic code masquerading as runnable — every listing executes. Flag manuscript listing for production: because imports are consolidated at the top rather than repeated per listing, the printed listings should show the imports they need in order to read as runnable-verbatim. (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| `plotStarburstMap` — `matplotlib.cm.get_cmap('bone_r')` | deprecated-api | BUG (execution blocker) | `matplotlib.cm.get_cmap` was **removed in Matplotlib 3.9**; on the installed 3.11.0 it raises `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'`, killing the SOM starburst cell on any fresh run. Switched to the supported registry lookup `matplotlib.colormaps['bone_r']`. Also dropped the dead `boner_rgb = []` local. (uncommitted) |
| `matplotlib_cmap_to_plotly` — `'rgb'+str((C[0], C[1], C[2]))` | deprecated-api | BUG (execution blocker) | **Only surfaced once the `get_cmap` crash above was fixed — it was hiding behind it.** `C` holds `np.float64`s, and `str()` of a tuple uses each element's `repr`; NumPy 2 changed that repr, so the colorscale string came out as `"rgb(np.float64(111.12), np.float64(122.25), np.float64(143.0))"` and Plotly rejected it (`ValueError: Invalid value ... for the 'colorscale' property`). Reproduced on numpy 2.2.6, then cast each channel to a plain Python int: `f'rgb({r}, {g}, {b})'` → `"rgb(111, 122, 143)"`. (uncommitted) |
| Figure 7.4 cell — `labels_map = som.labels_map(fingerprints, [label_names[t] for t in target])` **before** `target = data['label'].values` | headless-execution | BUG (execution blocker) | `target` was consumed on the line *above* the line that defines it — a guaranteed `NameError` on a fresh top-to-bottom kernel. The committed outputs could only have come from an out-of-order interactive session where `target` happened to survive from an earlier cell. Swapped the two lines. (uncommitted) |
| Configure-settings cell — `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)` | nondeterminism | BUG | Seeding only NumPy left the chapter **genuinely irreproducible**: the combinatorial library builder draws with Python's `random.randint`/`random.sample`, and Python's `random` was never seeded — so the generated library, every cluster derived from it, and the whole diversity/redundancy comparison differed on every run. Replaced with the standardized `RANDOM_SEED = set_seed(42)` (shared util seeds Python `random` + NumPy + `PYTHONHASHSEED`), which is what actually makes this chapter reproducible. (uncommitted) |
| `train_and_evaluate_som` — `train_test_split(fingerprints, test_size=0.2)` | nondeterminism | ROBUSTNESS | No `random_state`, so the SOM hyperparameter search scored each configuration on a different split. Added `random_state=RANDOM_SEED`. (Function is currently only invoked from the commented-out grid search, so this is latent, but it is the cell a reader uncomments.) (uncommitted) |
| `save_figure` — `fig.savefig(f'figures/{CHAPTER}/...')` | shape/offbyone/path | ROBUSTNESS | `figures/ch07/` is created **only** in the Colab-only setup cell — the very cell local readers are told to skip (and which I now tag `skip-execution`). On a fresh clone the first `save_figure` would fail. Added `os.makedirs(f'figures/{CHAPTER}', exist_ok=True)` inside `save_figure` so the notebook is self-sufficient. (uncommitted) |
| `display(example_pcore_df)` used with no import | import/undefined | ROBUSTNESS | `display` only resolves as an IPython-injected builtin; added `from IPython.display import display` so the listing also runs verbatim outside an interactive kernel (matches the CH02/CH03/CH04 treatment). Also hoisted the two function-local `import random` statements to the top import cell. (uncommitted) |
| Long-running cells with no committed artifact (BRICS/combinatorial library build, Taylor-Butina cutoff sweep, FRESCO scoring) | headless-execution | ROBUSTNESS | **Observation, not fixed.** Unlike CH02, this chapter has **no `artifacts/ch07/` directory at all**, so its "😱 Long Running Code Block" cells (the notebook advertises ~15 min and ~12 min) have no reload cell to fall back on. I therefore ran them for real to meet the `full` bar. Recommend the author add committed artifacts + `skip-execution` tags (or subsample) so readers are not forced through the full library build. Profiling for that decision: `silhouette_score` is cheap at this scale (7 s at n=8,000; 31 s at n=16,000) — the cost is dominated by the library construction and the O(n²) Tanimoto sweep, not the metrics. (uncommitted) |

## Author-decision queue

```
Q1 (Ch7 §7.1 — "toroidal" SOM claim): The inventory says the text may describe the SOM as toroidal.
   The code trains MiniSom with topology='rectangular', and MiniSom's `topology` argument only chooses
   the node LAYOUT (rectangular vs hexagonal) — the library has no toroidal / wrap-around neighborhood
   option at all. So the implementation is definitively NOT toroidal, and it cannot be made toroidal by
   flipping a flag. The word "toroidal" does not appear anywhere in the notebook (grep -i → 0 hits), so
   there is nothing for me to change in the editable artifact and I will not rewrite book prose.
   Decision needed: (a) correct the book text to describe a non-wrapping rectangular grid (my
   recommendation — it matches the code and the committed figures), or (b) if a genuinely toroidal SOM
   is pedagogically required, that means swapping the SOM implementation (MiniSom cannot do it), which
   would change every SOM figure in the chapter.
   Evidence needed: the manuscript sentence describing the SOM topology.
   Blocks: the 7.2 CONSISTENCY row; keeps chapter_done=false until resolved.

Q2 (Ch7 §7.2 — the "redundancy ≈ 0" claim now that the metric is fixed): The notebook's own markdown
   asserts "Both methods result in a diversity of around 0.8 Tanimoto distance ... and close to 0 for
   the redundancy measure," and the comparison narrative says Taylor-Butina had "slightly more diversity
   and less redundancy." That near-zero redundancy was partly an ARTIFACT of the bug I fixed in 7.10:
   `calculate_redundancy` subtracted n for self-comparisons against a zero diagonal, deleting n real
   redundant pairs and pushing the ratio toward (or below) zero. With the corrected metric the redundancy
   numbers change, and the executed values are now in the notebook's refreshed outputs.
   Decision needed: re-read the §7.2 comparison paragraph against the corrected numbers and adjust any
   claim that depends on the old (buggy) near-zero redundancy. I did not touch the prose.
   Evidence needed: the author's read of the refreshed comparison table vs. the manuscript paragraph.
   Blocks: nothing structurally — the code is now correct — but the prose may now overstate the result.
```

## Pedagogy changes (Standard depth)

- **Standardized setup cell:** replaced the raw `RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)` with a
  chapter-tailored `check_env(["numpy","pandas","scipy","sklearn","rdkit","matplotlib","seaborn","umap","statsmodels"])`
  (no torch/xgboost — this is a CPU-only chapter) plus `RANDOM_SEED = set_seed(42)` and a one-line
  runtime/hardware note. `RANDOM_SEED` keeps its name, so all ~20 downstream `random_seed=RANDOM_SEED` /
  `random_state=RANDOM_SEED` uses are unchanged — but Python's `random` is now seeded too, which is what
  actually makes the combinatorial library reproducible (see Table 2).
- **Learning objectives / Key takeaways:** already present as the "This chapter covers" bullets at the top
  and the "Summary" section at the bottom — kept as the pre-existing equivalents rather than duplicated.
- **`preview_df`:** added after the two major DataFrame transforms — the COVID Moonshot activity frame
  (post-load, post-labeling) and the extracted Mpro fragment pharmacophore table.
- **Light asserts:** labels are a subset of `{Active, Inactive, Unknown}`; `len(fingerprints) == len(data)`
  (guards the cache/dataframe alignment the whole SOM section silently depends on); and the pharmacophore
  frame is non-empty before the KDE work.
- **Named constants:** `IC50_ACTIVE_THRESHOLD_UM = 5` and `LABEL_ACTIVE/LABEL_INACTIVE/LABEL_UNKNOWN`
  replace bare `5` / `1` / `2` / `3` magic values, so the (previously inverted) potency rule now reads
  correctly at a glance.
- Preserved the visual identity (`#A20025` headers, emoji section banners). No restructuring.
- Tagged the 5 Colab-only setup cells `skip-execution` (`!pip`/`!wget`/`condacolab`/`os.kill(os.getpid(), 9)`).
  Checked the standard follow-on trap: `os` is imported only inside those cells **and** in the main import
  cell, so the executed path still resolves it.

## Verification log

_PENDING_
