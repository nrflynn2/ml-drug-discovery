# Workstream P — Reader PR integration (thomas-to-bcheme #24–28)

**Verdict: ACCEPT AND MERGE.** These are high-quality contributions and should land with the
contributor's authorship intact.

> **You run the git/gh ops** (agents hold no GitHub write auth). Everything below is a runbook.

---

## 1. Review — what each PR actually does

Reviewed at the **code-cell** level (notebook JSON/output noise stripped). One coherent theme:
**hardware-agnostic device selection + reproducibility**, via the modern `torch.accelerator` API
(unifies CUDA / Apple MPS / Intel XPU / CPU). Verified working on our torch 2.12.

| PR | File | Code changes | Assessment |
|----|------|--------------|------------|
| **#24** | `CH08_FLYNN_ML4DD.ipynb` | ~4 cells: `set_random_seeds(seed, device)`; `torch.accelerator` device; cudnn flags gated to CUDA only; **`ReduceLROnPlateau(verbose=)` deprecation → manual LR logging** | ✅ Strong. **Also fixes a Ch8 inventory item** (the `verbose` deprecation we had flagged). |
| **#25** | `CH09_FLYNN_ML4DD.ipynb` | ~9 cells: device detection (incl. a class `self.device`); **gzip support**; suppresses Open Babel's very noisy console warnings (broken PDB CONECT records) | ✅ Strong. The Open Babel noise suppression is a real reader-experience win. |
| **#26** | `CH10_FLYNN_ML4DD.ipynb` | ~8 cells: `setup_device()` — detects CUDA/XPU/MPS, prints device name + memory, returns `torch.device`; seeding | ✅ Strong. |
| **#27** | `CH11_FLYNN_ML4DD.ipynb` | ~38 changed: `CONFIG['device']` via `torch.accelerator`; `device_information()` with per-backend branches (cuda/mps/xpu/cpu) | ✅ Strong; the largest and most careful. |
| **#28** | `CH12_FLYNN_ML4DD/src/utils.py` | 2 lines: `get_device()` → `torch.accelerator` | ✅ Trivial and correct. |

**Why I rate these highly:** the comments show genuine understanding rather than cargo-culting — e.g.
*"`torch.accelerator.current_accelerator()` returns torch object whereas `.type` returns string"*,
*"auto-tuner built in unique to nvidia cuda"*, *"For CPU and other devices, the global flags and manual
seeds are sufficient for reproducibility."* They are focused, single-file, and all report
`mergeable: clean`.

**Do NOT re-implement these changes locally** — that would rob the contributor of attribution. Merge his
commits.

---

## 2. The one thing that will silently destroy his work

His PRs edit the **`.ipynb`**. Our QA workflow treats the **paired `.py` as the source of truth** and
regenerates the `.ipynb` from it. So if a chapter agent runs on ch8–12 *before* his changes are pulled
into the `.py`, `jupytext --sync` will **overwrite his notebook edits from the stale `.py`.**

**⇒ After merging each PR you MUST sync the notebook back into the `.py` before any ch8–12 agent runs.**

---

## 3. Runbook

```bash
# 0. Land our Wave-0 + ch1-7 QA work first (it is large, verified, and currently uncommitted).
git checkout -b modernization/qa-pass
git add -A && git commit -m "QA + modernization: Wave-0 foundation, ch1-7 passes, security floors"
git push -u origin modernization/qa-pass

# 1. Merge the reader PRs into main (each is mergeable=clean; his authorship is preserved).
gh pr merge 24 --merge     # CH08
gh pr merge 25 --merge     # CH09
gh pr merge 26 --merge     # CH10
gh pr merge 27 --merge     # CH11
gh pr merge 28 --merge     # CH12 src/utils.py

# 2. Bring them into the modernization branch.
git checkout modernization/qa-pass
git merge origin/main       # notebook conflicts, if any, are our pairing metadata vs his code cells

# 3. CRITICAL — push his .ipynb changes into the paired .py review surface.
for nb in CH08 CH09 CH10 CH11; do
  uv run jupytext --sync ${nb}_FLYNN_ML4DD.ipynb     # .ipynb -> .py (his edits now in the .py)
done
uv run python tools/validate_notebooks.py             # must stay green

# 4. Only now run the ch8-12 chapter agents; they build ON his merged work.
```

**Conflict note:** our Wave-0 added jupytext pairing metadata to every `.ipynb`, and his PRs change code
cells in the same files. These live in different regions of the notebook JSON, so they usually auto-merge;
if git balks, take **his** code cells and **our** `metadata.jupytext` block.

---

## 4. How the ch8–12 agents must treat his work

Add to each ch8–12 agent brief:

- **Build on, don't redo.** Device selection and seeding in ch8–12 are already handled by merged PRs
  #24–27. Verify them; do not reimplement.
- **Consolidation is allowed but must not delete capability.** A shared `utils.get_device()` now exists
  (it adopts his `torch.accelerator` pattern and credits him in the docstring). Where a chapter merely
  duplicates that logic, it may call the helper — but his richer `setup_device()` / `device_information()`
  (which report device name + memory) are *features*, not duplication. Keep them.
- **Attribution stays in git history** via his merge commits; do not squash them away.

---

## 5. Status

- ✅ `utils.get_device()` added (adopts #28's `torch.accelerator` pattern; credits thomas-to-bcheme).
- ✅ All 5 PRs reviewed at code level; all recommended for merge.
- ◻ Merges + pushes — **author action** (steps 1–3 above).
- ◻ ch8–12 agents — gated on step 3.
