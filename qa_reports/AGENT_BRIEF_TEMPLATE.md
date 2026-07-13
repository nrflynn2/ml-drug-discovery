# Chapter Agent Brief — TEMPLATE (paste-to-launch)

> Fill the `{{PLACEHOLDERS}}` and hand this whole brief to one coding agent (Opus 4.8 or peer). It is
> the complete, self-contained context bundle for a **single chapter**. Deliberately scoped: the agent
> gets the global setup spec + the taxonomy + **only this chapter's** inventory slice + this chapter's
> script — never the whole inventory.

---

## 0. Fill these in before launching

| Placeholder | Value |
|-------------|-------|
| `{{CHAPTER}}` | e.g. `ch08` |
| `{{SCRIPT}}` | e.g. `CH08_FLYNN_ML4DD.py` (paired to `CH08_FLYNN_ML4DD.ipynb`) — for ch12: the whole `CH12_FLYNN_ML4DD/` package |
| `{{ENV_TIER}}` | `core` \| `advanced` \| `conda-ch9` \| `ch12` |
| `{{EXEC_TIER}}` | `full` \| `smoke` |
| `{{INVENTORY_SLICE}}` | the chapter's section from `code_listing_feedback_for_qa.md` (paste the `## Chapter N` block + its lines from the BLOCKER quick-index) |

---

## 1. Mission

You are running the QA + modernization pass for **{{CHAPTER}}** of the book *Build AI Drug Discovery
Pipelines*. Fix real defects, proactively sweep for new ones, apply the pedagogy standard, verify to the
bar, and file a report in the mandated schema. **Edit the `.py`; the `.ipynb` is regenerated from it.**

## 2. Read first — context you are given

1. **Global setup spec:** `README.md` + `INSTALL.md` (uv + Python 3.12; conda island for ch9) and
   `pyproject.toml` extras. Bootstrap with `tools/bootstrap_env.sh` or the commands in the `justfile`.
2. **Shared helpers:** `utils.py` — `set_seed(42)`, `preview_df(df, name)`, `check_env()`. Use them in
   the standardized setup cell and previews (do not reinvent).
3. **This chapter's inventory slice:** `qa_reports/inventory_slices/{{CHAPTER}}.md` (the *only* inventory
   you get; sliced from `code_listing_feedback_for_qa.md`). This is the **floor** of what to address.
4. **This chapter's script:** `{{SCRIPT}}`.
5. **The report schema:** `qa_reports/SCHEMA.md` and `qa_reports/REPORT_TEMPLATE.md`.

## 3. Cross-cutting findings (calibrated from the actual repo — internalize these)

- **The scripts are a recently-fixed snapshot.** Many inventory `BLOCKER`s are **already resolved**.
  Your job is **verify-then-fix**, not blind application. Expect lots of `already-ok`.
- **Manuscript markers (`\`+comment, `#A/#B`, `[CA]/[CB]`) are NOT in the notebooks** (verified: 512
  code cells parse clean). If an inventory item is about a marker, confirm the notebook cell is clean →
  status `already-ok` + note "flag manuscript listing for production" (the fix belongs in the book build,
  not the notebook).
- **"Undefined" can mean "imported."** (e.g. ch12's `PositionalEncoding`/`Encoder`/`mean_pooling` live
  in `src/model.py`.) Trace imports before declaring something undefined.
- **False positives to leave alone:** docstring `>>>` examples are valid Python — do not "fix" them.
- **Inventory items can target artifacts that aren't in the notebook.** Some listings/figures/tables are
  produced in the *manuscript build*, not the companion notebook (e.g. CH01's "drugs-per-$bn R&D" plot
  has no generating code anywhere in the repo). If an item's code isn't in the artifact, grep the whole
  repo to confirm, then escalate as `needs-author-decision` ("add the code to the notebook, or is this
  manuscript-only?") and record what you searched. Never invent code to satisfy the item.

## 4. Proactive taxonomy sweep (scan the whole chapter, independent of the inventory)

`marker/non-parsing` · `import/undefined` · `mutable-default` (incl. non-empty like `x=[...]`) ·
`broad-except` (bare `except:`) · `rdkit-none-guard` (`MolFromSmiles` → None) · `pandas-indexing`
(`.loc` vs `.iloc`, chained assignment) · `gpu-memory` (whole pool → device) · `nondeterminism`
(seed via `set_seed`) · `deprecated-api` (rdkit/pandas/sklearn/torch) · `shape/offbyone/path` · `headless-execution`
(use `tqdm.auto`, not `tqdm.notebook`; a name imported ONLY in a Colab cell but used downstream after
you skip-tag it — e.g. `os`; cells that reuse/clobber earlier variables so a fresh-kernel top-to-bottom
run differs from the committed interactive outputs). Record every new hit in Table 2.

> ### ⚠️ `deprecated-api` is mostly **RUNTIME-ONLY** — the harness cannot see it
> Our modernization (RDKit 2025.9, scikit-learn 1.9, pandas 2.3, torch 2.12) **removed** APIs that still
> *parse* perfectly. So `validate_notebooks.py` goes **green on a chapter that is completely unrunnable.**
> Ch5 proved it: it died on its **third cell** (`dopts.dotsPerAngstrom`, removed in RDKit 2025.9), and all
> of §5.2 was dead (`cv="prefit"` — **removed**, not deprecated, in sklearn 1.9; it now *raises*
> `InvalidParameterError`). Known removals to watch for:
> - `dopts.dotsPerAngstrom` (RDKit 2025.9) — now purged repo-wide; don't reintroduce.
> - `cv="prefit"` (sklearn ≥1.6/1.9) → `CalibratedClassifierCV(FrozenEstimator(fitted_model), method=…)`.
> - `normalize=` on linear models (removed sklearn 1.2) → explicit `StandardScaler` in a Pipeline.
> - `LogisticRegression(n_jobs=…)` — silent no-op, removed in sklearn 1.10.
> - pandas `.append` (removed 2.0) → `pd.concat`.
>
> **The only reliable detector is a fresh-kernel, top-to-bottom execution.** Treat a green harness as
> *necessary but never sufficient*. When a fix changes an API, re-run and confirm the committed numbers
> still reproduce (or say explicitly that they moved, and why).

## 5. Operating protocol (SOP)

1. **Bootstrap** the `{{ENV_TIER}}` env (already installed at `./.venv` — prefix tools with `uv run`;
   uv is at `~/.local/bin/uv`). Confirm pairing (`uv run jupytext --sync {{SCRIPT}}`) and baseline the
   harness (`uv run python tools/validate_notebooks.py <your .ipynb>`).
2. **Verify-pass** each inventory item — locate by **listing label + surrounding code, never line
   number** (scripts were renumbered). Assign a terminal status.
3. **Fix** confirmed defects in the `.py`. Minimal, surgical, matching surrounding style. Do **not**
   rewrite prose/figures/captions — escalate those as `needs-author-decision`.
4. **Sweep** the taxonomy; fix new hits; log them in Table 2.
5. **Pedagogy (Standard depth) — apply to EVERY chapter** so the book is consistent: a Learning-
   Objectives box, one standardized setup cell (`check_env(<this chapter's packages>)` +
   `SEED = set_seed(42)` + a one-line runtime/hardware note), `preview_df()` after major DataFrame
   transforms, light `assert`s, and a Key-Takeaways close. **Pre-existing equivalents count** ("This
   chapter covers" ≈ objectives, "Summary" ≈ takeaways) — enhance them, don't duplicate. Call
   `check_env()` with a **chapter-tailored** package list (don't print torch/xgboost in a core chapter).
   Replace chapter-local seed logic (e.g. CH11 defines its own `set_seed`; a stray `seed(41)` in CH01)
   with the shared helper. Preserve the visual identity (colored headers, emoji banners). No restructuring.
6. **Verify to the bar** (`{{EXEC_TIER}}`) in a **fresh kernel, top-to-bottom** (committed outputs may
   have come from an out-of-order interactive session — watch for a cell that reuses/clobbers earlier
   variables, e.g. a demo that overwrites `X_train`/`y_train` that a later cell then consumes): static-all
   (harness) must pass; `uv run jupytext --sync`; then run the output refresh with the shared executor,
   which honors `skip-execution` cell tags: `uv run python tools/execute_notebook.py <your .ipynb> [--timeout N]`.
   - Add `# %% tags=["skip-execution"]` to (a) every "Colab users only" cell (`!pip`/`!wget`/
     `condacolab`/`os.kill`) and (b) any intentionally long-running cell the notebook **already pairs
     with a committed-artifact reload cell** (e.g. CH02's ~20-min PAINS/BRENK). "Full execution" = run
     everything else top-to-bottom clean.
   - **After skip-tagging Colab cells, check for a name imported/defined ONLY there but used downstream**
     (e.g. `os`) — move that import into a normal cell, or the executed path will `NameError`.
   - **A legitimately slow cell with NO artifact reload** (e.g. CH04's ~30-min learning-curve compute):
     run it for the `full` bar, but flag a speedup/"add a committed artifact + skip-tag" opportunity as a
     non-blocking observation. Do **not** merely raise the timeout for a cell that should be skip-tagged.
   - `smoke` (ch9/11/12) → additionally subset heavy train/dock cells (1 epoch / small sample) and set
     `needs_full_gpu_run: true`.
7. **Report** to `qa_reports/{{CHAPTER}}_report.md` per the schema, then **commit your chapter on a
   branch** (`.py` + regenerated `.ipynb` + report) and cite the short sha in each note. *(In a no-commit
   pilot, write `(uncommitted)` instead.)* **Do not stage `data/` or `artifacts/` changes** — execution
   can re-write committed inputs (a re-sorted CSV, a decompressed `Specs.sdf`); the real outputs live in
   the `.ipynb`, so `git checkout -- data artifacts` before finishing.

## 6. Definition of done

`chapter_done: true` only when: harness passes, imports/names resolve, execution passed at your tier,
the `.ipynb` was regenerated, and every inventory item is terminal with no *unresolved* author question.
Otherwise `chapter_done: false` and your Author-decision queue explains what's outstanding.

## 7. Guardrails

- Never fabricate a fix to hit a status — `already-ok`/`not-found`/`needs-author-decision` are valid,
  expected outcomes. Report faithfully.
- Don't touch other chapters, shared config, or `utils.py` beyond what your chapter needs (flag shared
  changes instead).
- Keep diffs reviewable: the author reviews the `.py`, so favor clear, minimal edits.
- **Repo hygiene:** never leave/commit `data/` or `artifacts/` churn from execution, the decompressed
  `Specs.sdf` (CH02), or other large regenerated inputs — restore them (`git checkout`). Flag the
  `utils.py` `.pkl.gz`-is-actually-raw-pickle naming for the author but do **not** change it (that would
  invalidate committed artifacts).
- **Input vs output artifacts:** committed *input* data (CSVs/SDFs) → always restore. A committed
  *output* a chapter regenerates (e.g. a tuned-model `.pkl`) may legitimately differ across environments
  (CH03's grid search flipped `nBits` 2048→1024 on a near-tie) — restoring it leaves the committed
  artifact out of sync with the fresh `.ipynb` outputs. Restore it to keep the diff clean, and flag the
  env-nondeterministic output (and any code that hardcodes a value the pipeline actually tunes) as an
  observation for the author rather than committing the flip.
