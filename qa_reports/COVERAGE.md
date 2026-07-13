# QA Coverage Checklist (living rollup)

> ### 👉 Looking for what's waiting on *you*? → **[`AUTHOR_QUEUE.md`](AUTHOR_QUEUE.md)**
> Single consolidated list of every open action + decision (this file is the per-chapter detail).

This is the **living** version of the reviewer inventory. Each chapter agent files
`qa_reports/CHXX_report.md` (per `SCHEMA.md`); the orchestrator flips the rows below from `pending` to
the reported terminal status and marks the chapter **done** when `chapter_done: true`.

**Status legend:** `pending` · `fixed` · `already-ok` · `not-found` · `author` (needs-author-decision).
Rows are at **listing granularity**; per-`(listing, tag)` detail lives in each chapter report's Table 1.

## Master status

| Ch | Env tier | Exec | Listings | Chapter status |
|----|----------|------|----------|----------------|
| 1  | core | full | 3 | ✅ done (`chapter_done: true`) — [ch01_report](ch01_report.md) |
| 2  | core | full | 13 | ✅ done (`chapter_done: true`) — [ch02_report](ch02_report.md) |
| 3  | core | full | 11 | 🔶 code clean; author-blocked (Q1 standardization, Q2 final cell) — [ch03_report](ch03_report.md) |
| 4  | core | full | 10 | 🔶 code clean; author-blocked (Ch4·Q1 logP sign, Ch4·Q2 2048-vs-11); slow ~48min — [ch04_report](ch04_report.md) |
| 5  | advanced | full | 3 | 🔶 code clean (was UNRUNNABLE — fixed); author-blocked (Ch5·Q1 calibration) — [ch05_report](ch05_report.md) |
| 6  | advanced | full | 11 | 🔶 code clean; author-blocked (Ch6·Q1–Q5, mostly paper-replication methodology) — [ch06_report](ch06_report.md) |
| 7  | advanced | full | 11 | ☐ not started |
| 8  | advanced | full | 6 | ☐ not started |
| 9  | conda-ch9 | smoke | 12 | ☐ not started |
| 10 | advanced | full/smoke | 6 | ☐ not started |
| 11 | advanced | smoke | 7 | ☐ not started |
| 12 | ch12 | smoke | 5 | ☐ not started |
| C  | advanced | full/smoke | 8 | ☐ not started |

Wave-0 foundation (all ✅): validation harness (514 cells, 0 errors) · `set_seed`/`preview_df`/
`check_env` · schema+templates+coverage · env modernization (pyproject extras; mordred dropped) ·
README/INSTALL (uv + Py3.12) · justfile + `tools/bootstrap_env.sh` · Ch9 conda island + CH12 pins ·
**`uv.lock` generated + advanced env installed & validated on WSL2** (Py3.12; numpy 2.2.6, pandas 2.3.3,
torch 2.12.1, rdkit 2025.9; numba 0.66) · **jupytext pairing established for all 15 notebooks**.
Caps applied: `numpy<2.3` (numba ceiling via shap/umap), `pandas<3.0` (behavioral stability), `numba>=0.60`.

---

## Chapter 1 — Drug Discovery Process
→ **Piloted + Q1 resolved** ([ch01_report.md](ch01_report.md)): 4 fixed · 2 already-ok · 0 author · +4 taxonomy fixes. **`chapter_done: true`**.
| listing | tags | item | status |
|---------|------|------|--------|
| §1.2 presentation | BLOCKER, ENHANCEMENT | `>>>` out of cells; explanatory comments | already-ok (no `>>>`) + fixed (takeaways) |
| Listing 1.2 | CONSISTENCY | update to current code; state data source | already-ok (current) + fixed (data-source note) |
| Figure 1.2 | BUG, ENHANCEMENT | add code; fix label; regenerate; extend past 2010 | **fixed** (`plot_eroom_law` + `eroom_law.csv`) |

## Chapter 2 — Ligand-based Screening (filtering/similarity)
→ **Piloted** ([ch02_report.md](ch02_report.md)): 5 fixed · 14 already-ok · 0 author · +2 taxonomy fixes. **`chapter_done: true`**.
| listing | tags | item | status |
|---------|------|------|--------|
| General | ROBUSTNESS, ENHANCEMENT | clean-run all; before/after previews | fixed (preview_df) |
| 2.1 | BLOCKER, ROBUSTNESS | `LoadSF`→`LoadSDF`; SDF prop names | already-ok |
| 2.2 | BLOCKER | `RO5_PROPS` defined | already-ok |
| 2.3 | BUG | bare `except:` → explicit | **fixed** |
| 2.4 | BLOCKER | `\ #A` continuation (marker) | already-ok (clean; flag manuscript) |
| 2.5 | BLOCKER, CONSISTENCY | import `FilterCatalog`; before/after prints | already-ok |
| 2.6 | BLOCKER | `\ #A` continuation (marker) | already-ok (clean; flag manuscript) |
| 2.7 | BUG, ROBUSTNESS | `df.copy()` + None guard | **fixed** (None guard) |
| 2.9 | BLOCKER | indentation + `display` import | **fixed** (display import) |
| 2.10 | BLOCKER, ROBUSTNESS | `query_idx`; xls engine | **fixed** (xlrd engine) |
| 2.12 | BUG, ROBUSTNESS | positional `.iloc`; guard heap pop | already-ok |
| version-assertion | CONSISTENCY | tie "217 descriptors" to RDKit version | already-ok (217 confirmed) |
| optional | ENHANCEMENT | count-based Morgan fingerprints | already-ok (optional, deferred) |

## Chapter 3 — Ligand-based Screening (ML)
→ **Done + author-blocked** ([ch03_report.md](ch03_report.md)): 1 fixed · 11 already-ok · 1 needs-author · +5 taxonomy (4 fixed; incl. `import os` load-bearing + bare-except). `chapter_done: false` (Q1, Q2).
| listing | tags | item | status |
|---------|------|------|--------|
| 3.3 | BUG | pandas `.append` → `pd.concat` | already-ok |
| 3.5 | ROBUSTNESS | `MolFromSmiles` None guard before `Cleanup` | already-ok |
| 3.6 | BLOCKER | missing `GetMorganGenerator`/`np` imports | already-ok |
| 3.7 | BUG, BLOCKER | reset index / boolean masks; `RANDOM_SEED` | already-ok |
| 3.10 | CONSISTENCY | `DummyClassifier(strategy="most_frequent")` | already-ok |
| 3.14 | BLOCKER | indentation + `Chem`/`Cleanup`/… imports | already-ok |
| 3.15 | CONSISTENCY | SGD "logistic" comment vs hinge default | **fixed** (comment corrected to hinge) |
| 3.17 | BLOCKER | extra comma in `PolynomialFeatures` | already-ok |
| 3.18 | BLOCKER | `gs.linear_sgd`→`gs_linear_sgd` | already-ok |
| 3.20 | BLOCKER | trailing space in pkl filename | already-ok |
| standardization | CONSISTENCY | text (metals/stereo) vs code (Cleanup/…) | **needs-author (Q1)** |

## Chapter 4 — Solubility / Linear Models
→ **Done + author-blocked** ([ch04_report.md](ch04_report.md)): 4 fixed · 7 already-ok · 2 needs-author · +7 taxonomy (incl. a real execution-order bug: bias-variance cells clobbered `X_train`/`y_train`; `ransac_regressor`→`ransac_model`; `tqdm.notebook`→`tqdm.auto`). `chapter_done: false` (Ch4·Q1, Q2). **Slow chapter** — learning-curve cell ~29 min.
| listing | tags | item | status |
|---------|------|------|--------|
| AD-table | BUG | logP range ~17.4–26.2 | **needs-author (Ch4·Q1)** — NOT a bug: real range is `[-17.4, +26.2]`, reviewer dropped the minus sign |
| descriptor-dim | CONSISTENCY | text "2048 features" vs ~11 descriptors | **needs-author (Ch4·Q2)** — code uses 11 descriptors, no fingerprints |
| poly-order | CONSISTENCY | 7th vs 10th order | **fixed** (target 7th consistent; corrected mislabeled "10th"→"20th Order Fit") |
| 4.2 | BLOCKER | `from rdkit import Chem` + None guard | already-ok |
| 4.3 | ROBUSTNESS | `AromaticProportion` div-by-zero guard | **fixed** |
| 4.4 | CONSISTENCY | add explicit MSE/RMSE/MAE/R² prints | **fixed** (added MSE/MAE) |
| 4.6 | BLOCKER | `esol_val_pred` undefined | already-ok (refactored to `y_pred_original`) |
| 4.7 | BLOCKER, ENHANCEMENT | define `features`; `loguniform` alpha search | **fixed** (loguniform); BLOCKER already-ok |
| 4.8 | BUG | `np.array(features)[...get_support()]` | already-ok (safe list-comp over `get_support()`) |
| 4.9 | BLOCKER, BUG, CONSISTENCY | `SGDRegressor` import; y-scramble; "Average MSE" | already-ok (all three already correct) |

## Chapter 5 — CYP Inhibition Classification
→ **Done + author-blocked** ([ch05_report.md](ch05_report.md)): 2 fixed · 1 already-ok · 1 needs-author · **+7 taxonomy**. `chapter_done: false` (Ch5·Q1). **⚠️ CH05 was entirely unrunnable** — killed on its 3rd cell by `dotsPerAngstrom` (removed in RDKit 2025.9) and all of §5.2 dead from `cv="prefit"` (removed in sklearn 1.9). The harness was green throughout. Both fixed; execution now passes (45 cells, 0 errors).
| listing | tags | item | status |
|---------|------|------|--------|
| 5.2 | BUG | calibration fit + eval on the same val set | **needs-author (Ch5·Q1)** — the bug *inverts* the Platt-vs-isotonic ranking |
| 5.2 | ROBUSTNESS | `cv="prefit"` version-sensitivity | **fixed** — it was *removed* in sklearn 1.9 → `CalibratedClassifierCV(FrozenEstimator(m))` |
| 5.4 | BUG | upsample/downsample labels swapped | **fixed** — downsampled model was printed as `"OS DT"`; relabeled `DS` |
| general | CONSISTENCY | `radius` vs `r` keyword | already-ok (one helper, `radius=`) |
| — | — | **scikit-multilearn risk: did NOT materialize** | already-ok — imports + runs clean on sklearn 1.9 / numpy 2.2; no pin needed (ch5 is its only consumer) |

## Chapter 6 — RNA-targeted (TAR)
→ **Done + author-blocked** ([ch06_report.md](ch06_report.md)): 12 fixed · 5 already-ok · **5 needs-author** · +8 taxonomy. `chapter_done: false`. Execution `full` → **pass**.
| listing | tags | item | status |
|---------|------|------|--------|
| 6.1 | ROBUSTNESS, BUG | dedup/None states; tautomers per protonation state | **fixed** (48 cmpds → 4,561 unique variants) |
| 6.1 | CONSISTENCY | Dimorphite pH range vs SPR assay | **needs-author (Ch6·Q1)** — pH now a parameter |
| 6.2 | BUG | SMARTS `[nH]` vs `[n-]` | **fixed** — was silently aligning on an **empty match** |
| 6.3 | BUG, ROBUSTNESS | conformer removal desyncs IDs↔energies; UFF/embed guards | **fixed** (`id_to_energy` map + keep-list) |
| 6.4 | BUG | Boltzmann const `1.987e-3` | already-ok (already correct) |
| 6.4 | ROBUSTNESS | subtract relative energies | **fixed** — raw UFF energies overflowed to `inf`/`nan` |
| 6.5 | BUG | threshold 0.8 vs 0.95 | already-ok — **reviewer misread**: `0.8` is a *separate* redundancy threshold |
| 6.5 | ROBUSTNESS | no-correlated-pair case | already-ok |
| 6.5 | BUG (leakage) | target-informed selection before split | **needs-author (Ch6·Q2)** — paper replication |
| 6.6 | BLOCKER | undefined `hiv_tar1_lnkd_X_DR` | already-ok (doesn't exist) |
| 6.6 | BUG | Kennard-Stone farthest-pair vs farthest-from-mean | **needs-author (Ch6·Q3)** — KSA isn't even used for the real split |
| 6.6 | ROBUSTNESS | singular inverse covariance; PCA pre-split | **fixed** (`pinv`) + **needs-author (Ch6·Q4)** (pre-split PCA) |
| 6.7 | ROBUSTNESS | `np.linalg.inv` collinearity | **fixed** (`pinv`) |
| 6.7 | BLOCKER | guard `best_features is None` | already-ok (guarded at both levels) |
| 6.7 | BUG | final model chosen by maximizing test-set Q² | **needs-author (Ch6·Q5)** |
| 6.8 | BUG | clone fresh model per CV fold | **fixed** |
| 6.9 | BUG, BLOCKER | finite-diff caveat; array→scalar | **fixed** (documented + `partial_derivative` returned a length-1 array) |
| 6.10 | ROBUSTNESS | SHAP/XGBoost shapes + version | **fixed** |
| general | ROBUSTNESS | defensive checks throughout | **fixed** |

## Chapter 7 — Unsupervised Learning
| listing | tags | item | status |
|---------|------|------|--------|
| 7.1 | BLOCKER, BUG, ROBUSTNESS | `fp_size`→`n_bits`/parens; `ConvertToNumpyArray`; active/inactive logic; NaN IC50 | pending |
| 7.2 | BLOCKER, CONSISTENCY | brace + `RANDOM_SEED`/`n_nodes`/`m_nodes`; toroidal SOM | pending |
| 7.3 | BLOCKER | backslash+ellipsis; define `drh_compounds`/`retrospective_hits` | pending |
| 7.5 | BLOCKER, ROBUSTNESS | `reaction_smirks`/`reaction_smarts`; infinite-loop guard | pending |
| 7.6 | BLOCKER, BUG | `frag_mols`/`fragMols`; bare `except` | pending |
| 7.7 | BLOCKER, CONSISTENCY | `DataStructs`/`AgglomerativeClustering` imports; path-vs-Morgan | pending |
| 7.9 | ROBUSTNESS | degenerate-cluster metric guards | pending |
| 7.10 | BLOCKER, BUG | closing paren; exclude self-pairs | pending |
| 7.12 | BLOCKER, ROBUSTNESS | name mismatch + backslash; empty distances | pending |
| 7.13 | ROBUSTNESS, BUG | empty KDE; normalize per-molecule scoring | pending |
| general | CONSISTENCY | mark schematic vs runnable (**docstring `>>>` = OK, don't touch**) | pending |

## Chapter 8 — Intro to Deep Learning (EGFR)
| listing | tags | item | status |
|---------|------|------|--------|
| 8.1 | BLOCKER, ROBUSTNESS | `Smiles`/`smiles`; `criterion` scope; mkdir artifacts; `ReduceLROnPlateau` ver; `cude`/`2,0048` (already-ok?) | pending |
| 8.2 | BUG, ROBUSTNESS | `(smiles,activity,mol)` alignment; None guard; `.iloc` | pending |
| 8.3 | BUG, ROBUSTNESS | `ConvertToNumpyArray`; `to_numeric` + dropna | pending |
| activity-target | BUG | `standard_value` = pIC50 or raw IC50? (author) | pending |
| 8.4 | ROBUSTNESS, BUG, CONSISTENCY | EF div-by-zero; `max(1,ceil(..))`; `>6.3` vs `>=6.3` | pending |
| scaffold-split | ENHANCEMENT | accumulate to target ratio; report split stats | pending |

## Chapter 9 — Structure-based / Active Learning
| listing | tags | item | status |
|---------|------|------|--------|
| 9.1 | BUG, BLOCKER | explicit ligand selection; `idex`→`idxs` | pending |
| 9.2 | BLOCKER, CONSISTENCY | define `Point`/`Box`; padding per-side (`2*padding`) | pending |
| 9.3 | BLOCKER, ROBUSTNESS, BUG, CONSISTENCY | Meeko imports; prep guards; reuse Vina maps; `cpu=0` | pending |
| 9.4 | BLOCKER, ROBUSTNESS | undefined names; parse/embed/mmff alignment | pending |
| 9.5 | BLOCKER, ROBUSTNESS | unreachable return; force 2D fingerprint | pending |
| 9.6 | ROBUSTNESS | empty-train div0; batch-wise device transfer | pending |
| 9.7 | ROBUSTNESS | `LazyBitVectorPick` compat; `n_samples>len(pool)` | pending |
| 9.8 | BLOCKER, BUG, ROBUSTNESS, CONSISTENCY | return `best_indices`; respect `batch_size` + restore eval; pool GPU; minimization convention | pending |
| 9.9 | ROBUSTNESS, BUG | None/embed/mmff; VinaDocking reuse + `reference_df` global | pending |
| 9.10 | BUG, CONSISTENCY | mutable default `[]`; numpy/stale-index/dup/batch; "initial samples" print | pending |
| 9.12 | BLOCKER | define `CHAPTER`/`exp_dir`; ensure artifacts dir | pending |
| Figure-9.5 | CONSISTENCY | table −20.66 vs caption ≈ −10.5 (author) | pending |

## Chapter 10 — Generative (SMILES VAE-CYC)
| listing | tags | item | status |
|---------|------|------|--------|
| 10.1 | BLOCKER, ROBUSTNESS | vocab helpers/indices; placeholder-char asserts | pending |
| 10.2 | ROBUSTNESS, BUG | empty-list before `max(len ...)`; truncation warns/`<eos>` | pending |
| 10.3 | BUG | mutable default `hidden_dims=[512,256]`→None | pending |
| 10.3/10.4 | BLOCKER, CONSISTENCY, BUG | show encode/decode/…; compression ratio (author); `cross_entropy` targets | pending |
| 10.5 | ROBUSTNESS | `.view`→`.reshape`/`.contiguous()` | pending |
| 10.6 | BUG | reset padding-idx embedding row to zero | pending |

## Chapter 11 — GNNs for DTA
| listing | tags | item | status |
|---------|------|------|--------|
| 11.1 | BUG, ENHANCEMENT, CONSISTENCY | isolated atoms; bond features; no one-hot normalize; `np.matrix`→arrays; 78-dim (author) | pending |
| 11.2 | ROBUSTNESS | `one_of_k_encoding` unknown token; implicit-valence API | pending |
| 11.3 | ROBUSTNESS, CONSISTENCY, BLOCKER, BUG | contact-map shape; self-loop; `CONFIG`; `edge_index` `[2,E]` | pending |
| 11.4 | ROBUSTNESS, CONSISTENCY | unknown AAs; 21-dim vs 20-symbol (author) | pending |
| PSSM | BUG, CONSISTENCY | denominator counts; gaps; background freq; `dic['X']` midpoint; in-place mutate | pending |
| 11.7/11.8 | BLOCKER | `mol_data`/`data_mol`; layer names; `forward_mol`/`forward_molecule` | pending |
| general | ROBUSTNESS | full review + clean-run | pending |

## Chapter 12 — Transformers (protein) — package: notebooks + src/scripts
| listing | tags | item | status |
|---------|------|------|--------|
| 12.1–12.3 | BLOCKER, ROBUSTNESS | classes imported from `src` (author: inline vs note); device; label dtype; eval/no_grad/clip; mask shapes | pending |
| attention-mask | CONSISTENCY | state 1/0 vs bool vs additive; shape (author) | pending |
| special-tokens | CONSISTENCY | `vocab_size=25` vs tokenizer; pooling excludes specials (author) | pending |
| MLM-loss | BUG | `ignore_index=-100` for unmasked positions | pending |
| parameter-count | CONSISTENCY | recompute 273,794 after fixes (author) | pending |
| (package) | — | fix `requirements.txt` torch trio; `pyproject` py3.8→3.12 | pending |

## Appendix C — Knowledge Distillation
| listing | tags | item | status |
|---------|------|------|--------|
| C.1 | BLOCKER, ROBUSTNESS, CONSISTENCY | `[CA]` marker (confirm clean); empty-cluster guard; explain move-to-front | pending |
| C.2 | BLOCKER | define `roots`; `atom_size`/`bond_size`/`MAX_POS` | pending |
| C.3 | CONSISTENCY | justify `z_log_var ≤ 0` (author) | pending |
| C.4 | CONSISTENCY, ROBUSTNESS | teacher projection vs truncation (author); device-aware tensor; hierarchical vs root-only | pending |
| C.5 | BLOCKER, ROBUSTNESS | `[CA]` marker (confirm clean); `WARMUP_EPOCHS==0` div0 | pending |
| C.6 | ROBUSTNESS | empty-dataloader; NaN/inf grad `isfinite` check | pending |
| C.7 | BUG | zeroing all 1-D params → name-based bias/norm checks | pending |
| offline/online | CONSISTENCY | state offline KD (teacher frozen) (author) | pending |

---

## Author-decision queue (pooled; resolve in Wave 2)

Seeded from the plan's escalation catalog; agents append numbered questions here as they surface.

**Resolved:**
- ✅ **Ch1 · Q1 — Figure 1.2 "drugs-per-$bn R&D" plot:** author chose (a) *add the code*. Done — new
  `plot_eroom_law` in CH01 §3 + sourced `data/ch01/eroom_law.csv` (Scannell 2012 / Ringel 2020 trend +
  real FDA approvals via OWID, extended to 2024). CH01 `chapter_done: true`.

**Live (open — surfaced by chapter agents):**
- **Ch3 · Q1 — SMILES standardization, text vs code:** prose reportedly says the pipeline disconnects metals + assigns stereochemistry, but the code only does Cleanup / LargestFragmentChooser / Uncharger / canonical tautomer. Reconcile the prose to the code (or add the steps). Prose-only.
- **Ch3 · Q2 — incomplete final cell:** the last code cell is an incomplete cross-chapter demo — complete it, remove it, or mark it clearly as a teaser.
- **Ch4 · Q1 — logP AD bound (NOT a code bug):** the AD table is `train_df.describe()`; `MolLogP` really spans `[-17.4, +26.2]` — the reviewer's "17.4 to 26.2" dropped the minus sign. Decide: (a) correct the manuscript table's sign, and (b) keep vs trim the real non-drug-like outliers (MolWt→2,285 Da).
- **Ch4 · Q2 — "2048 features" vs code:** the model uses 11 RDKit descriptors and no fingerprints ("2048" is nowhere in the code). Fix the §4.2.1 prose (or add fingerprints if 2048 was intended).
- **Ch4 · (fixed, FYI):** poly-order — target is 7th everywhere; a mislabeled "10th Order Fit" was corrected to "20th"; reconcile any 7th/10th manuscript prose to the code.
- **Ch6 · Q1–Q5 — four of the five share ONE root tension: the chapter deliberately replicates the Cai/Hargrove paper.** The reviewer's methodology flags are all *faithful to the source but statistically unsound*, and fixing them moves every downstream number and figure. This is one pedagogical decision, not five code decisions — **"replicate the paper and name its flaws" vs. "demonstrate the correct protocol."** Specifically: **Q2** target-informed feature selection before the split (real leakage), **Q4** PCA/scaling fit pre-split, **Q5** final model chosen by maximizing test-set Q². Plus **Q1** (Dimorphite pH 6.4–8.4 vs the actual SPR assay buffer — an experimental fact only you can source; it's now a parameter) and **Q3** (Kennard-Stone seeds from the farthest *point*, not the farthest *pair* — note KSA isn't even used for the real split, so this only affects an illustrative figure).
- **Ch5 · Q1 — §5.2 calibration evaluated on its own fit set (the fix changes the chapter's *conclusion*):** calibrators are fit on validation and scored on that same validation set. Measured on held-out test data, calibration still works (both beat uncalibrated) — but **the buggy eval inverts the Platt-vs-isotonic ranking**: isotonic wins on the fit set (RMSCE 0.0303) while **Platt wins decisively held-out (0.0302 vs isotonic's 0.0471)** — isotonic is overfitting the 1,232-row calibration set. If §5.2's prose/figure concludes "isotonic calibrates best," the honest fix **reverses that claim**. Recommendation: adopt the held-out evaluation — the overfitting story is a better lesson and is exactly the "question your probabilistic outputs" point the section is named for. *Chapter runs and reproduces its committed figures while this is open.*

**Pre-seeded (from the plan's escalation catalog):**
- **Ch5:** US/DS label direction; calibration methodology (fix vs documented simplification).
- **Ch6:** Dimorphite pH vs SPR assay; feature-selection-before-split; test-Q² over-optimization.
- **Ch8:** `standard_value` = pIC50 or raw IC50(nM); active threshold `>6.3` vs `>=6.3`.
- **Ch9:** results −20.66 vs Figure 9.5 caption ≈ −10.5 kcal/mol.
- **Ch10:** compression-ratio arithmetic (12,800 vs 128,000).
- **Ch11:** 78-dim atom vector; 21-dim vs 20-symbol alphabet; PSSM background vs uniform pseudocount.
- **Ch12:** classes inline vs imported-from-`src`; `vocab_size=25`; recompute 273,794 params; mask semantics.
- **App C:** offline vs online KD; `z_log_var ≤ 0`; root-only vs hierarchical matching.
- **All:** any `[CA]`/marker item → notebook clean ⇒ `already-ok` + flag manuscript listing for production.
