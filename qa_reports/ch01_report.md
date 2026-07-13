---
chapter: ch01
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
  total: 6
  fixed: 4
  already_ok: 2
  not_found: 0
  needs_author_decision: 0
new_taxonomy_hits: 4
chapter_done: true
---

# Chapter 01 — QA Report

> **Update (Q1 resolved — author elected to add the code):** Figure 1.2's generating code was added to
> the notebook (§3 "Introducing Drug Discovery", `plot_eroom_law`) with a sourced, documented dataset
> (`data/ch01/eroom_law.csv` + `eroom_law_SOURCES.md`): the Eroom's-Law efficiency trend from Scannell
> 2012 / Ringel 2020, overlaid with real FDA new-drug approvals per year (Our World in Data). The plot
> fixes the annotation label, adds labeled log-scale axes + title + larger fonts, moves the annotation
> to the top-right, and extends past 2010 to 2024. Both Figure 1.2 rows below are now **`fixed`** →
> `chapter_done: true`.

_The CH01 script was in good shape: the harness passed at baseline (26 code cells, 0 errors) and the
Section 1.2 `>>>`-in-code BLOCKER was already absent. I fixed 2 inventory items (added a data-source
note and one-line "Takeaway" comments to the Section 1.2 demos), and confirmed 2 more as already-ok
manuscript-only artifacts. The proactive sweep surfaced 4 real code hits (non-standard seed, pandas
chained-assignment, a deprecated seaborn `.fig` attr + stray blank figure, and hardcoded figure paths),
all fixed. I also applied the Standard-depth pedagogy pass (wired in `set_seed(42)`/`check_env()`,
added `preview_df()` + light asserts after the two major DataFrame transforms) and tagged the 5
Colab-only setup cells `skip-execution` so the notebook full-executes locally. **The two Figure 1.2
items are blocked: that "drugs-per-$bn R&D" plot's generating code is NOT in the notebook or anywhere
in the repo — see Q1.** That open author question is why `chapter_done: false`. Full top-to-bottom
execution passed (23 cells run, 5 Colab cells skipped, 0 errors)._

## Table 1 — Inventory coverage

| listing | tag | status | note |
|---------|-----|--------|------|
| Section 1.2 code presentation | BLOCKER | already-ok | No `>>>` REPL prompts sit in any executable Section 1.2 cell (`grep '>>>'` → none). `display_molecule_properties` emits results via `print(f"Number of atoms: ...")`, and the caffeine cell shows properties then renders the mol — code and output are already cleanly separated. Notebook parses clean (harness) and full-executes. Flag manuscript listing for production: the `>>> Number of atoms: 14` output-in-code artifact likely still lives in the book's AsciiDoc listing and should be split there. (uncommitted) |
| Section 1.2 code presentation | ENHANCEMENT | fixed | Every demo already carries a full docstring; the gap was a per-snippet takeaway. Added concise `# Takeaway:` comments to the four Section 1.2 demonstrations — caffeine (core SMILES→Mol→properties→draw workflow), USAN load (assembling a labeled table), PCA (unsupervised variance compression), logistic regression (supervised decision boundary in PCA space). (uncommitted) |
| Listing 1.2 (RDKit+ECFP6+PCA) | CONSISTENCY (out of date) | already-ok | The notebook IS the current source of truth: it parses clean and executes end-to-end with the modern APIs (`GetMorganGenerator`, `PandasTools`, sklearn `PCA`/`LogisticRegression`). Nothing to change in the `.py`. Flag manuscript listing for production: re-sync the printed Listing 1.2 in the book to the current notebook code. (uncommitted) |
| Listing 1.2 (RDKit+ECFP6+PCA) | CONSISTENCY (data location) | fixed | Added an explicit data-source comment above `usan_stems`: one CSV per USAN stem under `data/ch01/fda_approved_drugs/<stem>.csv`, ChEMBL exports (`;`-delimited), committed in-repo so no download is needed locally. `load_usan_stem_data` already parameterizes `data_path=f"data/{CHAPTER}/fda_approved_drugs"`. (uncommitted) |
| Figure 1.2 (drugs-per-$bn R&D plot) | BUG | fixed | Added the missing generating code (`plot_eroom_law`, §3) and set a correct "New drugs per \$bn R&D" annotation/axis label; added labeled log-scale axes, a descriptive title, and larger fonts. Data sourced to `data/ch01/eroom_law.csv` (Scannell 2012 / Ringel 2020 efficiency trend + real FDA approvals via OWID; see `eroom_law_SOURCES.md`). (uncommitted) |
| Figure 1.2 (drugs-per-$bn R&D plot) | ENHANCEMENT | fixed | Regenerated per the reviewer asks: annotation moved to the top-right (no longer hidden behind the curve), series extended past 2010 to 2024 (Ringel post-2010 rebound + real FDA approval overlay), larger fonts, saved to `figures/ch01/eroom_law.{png,svg}`. (uncommitted) |

## Table 2 — New taxonomy hits (not in the inventory)

| location (cell / func / line) | taxonomy class | severity | action taken |
|-------------------------------|----------------|----------|--------------|
| setup/constants cell — `np.random.seed(41)` | nondeterminism | BUG (minor) | Replaced with `SEED = set_seed(42)` (utils helper: seeds Python/NumPy/torch + `PYTHONHASHSEED`). Adopts the repo-standard seed 42 vs. the stray 41. No output change: PCA and LogisticRegression already pass `random_state=42` and nothing else draws from `np.random`. (uncommitted) |
| `add_molecular_data` — `df = df[~df['ROMol'].isnull()]` then `df['ECFP6'] = ...` | pandas-indexing | ROBUSTNESS | Added `.copy()` to the filtered slice so the subsequent fingerprint-column assignment writes to an owned frame (avoids the SettingWithCopy / chained-assignment pitfall; matters under pandas Copy-on-Write). (uncommitted) |
| `plot_pca_results` — `return pairplot.fig` (+ stray `plt.figure(figsize=(12,10))`) | deprecated-api | ROBUSTNESS | Changed `.fig` → `.figure` (seaborn 0.13 deprecates the `.fig` alias; the same function already used `.figure` three times — inconsistent). Also removed the stray `plt.figure(...)` before `sns.pairplot`, which created an unused blank figure (pairplot builds its own). (uncommitted) |
| PCA/decision-boundary save cells — `plt.savefig('figures/ch01/...')` | shape/offbyone/path | ENHANCEMENT | Switched hardcoded `'figures/ch01/...'` to `f'figures/{CHAPTER}/...'` (consistent with the rest of the notebook) and changed the PCA save from global `plt.savefig` to `pca_fig.savefig(...)` so it targets the returned Figure rather than relying on pyplot's current-figure state across cells (matches the existing `cillin_fig`/`olol_fig` pattern). (uncommitted) |

_Also handled (workflow, not a taxonomy class): the 5 Colab-only setup cells (`os.makedirs`+`!wget`,
`!pip`, `condacolab.install()`, `condacolab.check()`+`!mamba`, and `os.kill(os.getpid(), 9)`) were
tagged `tags=["skip-execution"]`. nbclient 0.11 honors this tag, so `jupyter nbconvert --execute`
skips them locally (the `os.kill` cell would otherwise kill the kernel) while they remain intact for
Colab users. See PILOT FEEDBACK._

## Author-decision queue

```
Q1 (Ch1 Figure 1.2 — "drugs-per-$bn R&D" / Eroom's-law plot): the code that generates this figure is
   not present in CH01_FLYNN_ML4DD.py/.ipynb or anywhere else in the repo (verified by full-repo grep;
   the notebook only produces the caffeine image, USAN grid, PCA pairplot, and decision boundaries).
   The inventory frames the fixes (label/LaTeX BUG + regenerate ENHANCEMENT) as code changes, but there
   is no code to change in the editable artifact.
   Decision needed: (a) Should the figure's generating code be ADDED to the CH01 notebook (then QA can
   fix the "New drugs per bn$ R&D" annotation label, add axis labels/title, enlarge fonts, move the
   annotation to top-right, and extend the series past 2010)? Or (b) is this figure manuscript-only
   (generated in the book build), so the fixes belong there and this item should be closed as
   out-of-scope for the notebook?
   Evidence needed: the original figure-generation script + its data source; and, if extending past
   2010, the author's chosen public source (the inventory says "from 2023 public data" yet also asks to
   "extend past 2010", which is internally inconsistent and needs the author to pin down the dataset).
   Blocks: both Figure 1.2 inventory rows; keeps chapter_done=false until resolved.
```

## Verification log

- `uv run python tools/validate_notebooks.py CH01_FLYNN_ML4DD.ipynb` → `✓ 28 code cells OK — 0 errors, 0 warnings` (baseline before edits: 26 cells OK; +2 from the new preview/assert cells).
- `uv run jupytext --sync CH01_FLYNN_ML4DD.py` → ok; final re-sync reports `Unchanged` for both `.py` and `.ipynb` (code in sync; the only jupytext message is a benign "Notebook is not trusted" warning).
- execution: `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 CH01_FLYNN_ML4DD.ipynb` → **pass** in ~56s. 23 code cells executed, 5 Colab cells skipped (execution_count None), 0 error outputs. nbconvert prints a benign `Notebook JSON is invalid: 'id' was unexpected` schema warning (cell `id` is valid in nbformat 4.5+; `nbformat.read` loads it fine) and a plaintext-kernel-transport warning — neither is an execution failure.
- Sanity of outputs: `check_env` → Python 3.12.13, numpy 2.2.6, pandas 2.3.3, seaborn 0.13.2, sklearn 1.9.0, rdkit 2025.09.6; 343 valid molecules (0 invalid); PCA PC1–PC4 variance printed; `preview_df` shows 343×6 then 343×10; -cillin accuracy 1.0000, -olol accuracy 0.9563. All four figures in `figures/ch01/` refreshed.

_Note: no branch/PR opened and nothing committed, per the pilot scope constraint ("Do NOT commit
anything"); hence `(uncommitted)` in lieu of `<sha>` citations above._
