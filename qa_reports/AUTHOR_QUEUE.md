# 📋 Author Queue — everything waiting on you

**One place for all open items.** Generated from the filed chapter reports + workstream docs.
Last updated: 2026-07-13 · Chapters done: 1–7 (Ch7 output-refresh in flight) · Gated: 8–12, App C.

---

## A. Actions to run (4) — these unblock the rest of the project

| # | Action | Why it matters | Where |
|---|--------|----------------|-------|
| **A1** | **Merge reader PRs #24–28** (`gh pr merge …`) | **Blocks ch8–12.** All 5 reviewed and recommended. ⚠️ You *must* `jupytext --sync` the `.ipynb`→`.py` after merging, or our workflow silently overwrites his edits. | [`PR_INTEGRATION.md`](PR_INTEGRATION.md) §3 |
| **A2** | **Push the branch** | Closes the **82 Dependabot alerts** — the fix is already in the tree (4 floors bumped, incl. the critical `ray`). | [`SECURITY.md`](SECURITY.md) |
| **A3** | **Decide on `ray`** | 🔴 Critical advisory with **no upstream patch**. It's an *optional* extra used only by `scripts/tune.py`. **Rec:** keep it optional + document. Alt: drop the `tuning` extra. | [`SECURITY.md`](SECURITY.md) |
| **A4** | **Close/repurpose PR #23** | Your own 45-file modernization PR; you chose "local work supersedes." *Optional later:* port its CI (`qa.yml`) + CH12 tests onto our base. | — |

---

## B. Decisions (12 questions) — grouped, because they collapse

### ⭐ B1. THE BIG ONE — resolves 4 questions at once
**"Replicate the paper faithfully (and name its flaws)" vs. "demonstrate the correct ML protocol."**

Ch6 explicitly replicates the Cai/Hargrove paper, so several reviewer-flagged defects are *faithful to the
source but statistically unsound*. Fixing them moves every downstream number and figure. This is one
pedagogical call, not four code fixes:

| Q | Issue |
|---|-------|
| **Ch6·Q2** | Feature selection uses target correlation on all 48 compounds **before the split** → real leakage |
| **Ch6·Q4** | `StandardScaler` + PCA fit **pre-split** → test compounds inform the axes |
| **Ch6·Q5** | Final model chosen by **maximizing test-set Q²** |
| **Ch5·Q1** | (Same family) §5.2 calibrators fit *and scored* on the same validation set |

> **Ch5·Q1 is the one that actually changes a conclusion.** Measured on held-out data, **Platt wins
> (RMSCE 0.0302) and isotonic loses (0.0471)** — the reverse of what the buggy eval shows. If §5.2 says
> *"isotonic calibrates best,"* the honest fix **reverses that claim**. My recommendation: adopt the
> held-out evaluation — the overfitting story is a *better* lesson and is precisely the "question your
> probabilistic outputs" point the section is named for.

**A single ruling here ("keep replication, add a caveat box" or "fix the protocol") closes all four.**

### B2. Prose / manuscript reconciliation (3) — quick, no code risk
| Q | Issue | Note |
|---|-------|------|
| **Ch3·Q1** | Text says the pipeline disconnects metals + assigns stereochemistry; code only does Cleanup/LargestFragment/Uncharger/tautomer | Fix the prose (or add the steps) |
| **Ch4·Q1** | logP AD bound "17.4–26.2" | **Not a bug** — the real range is `[-17.4, +26.2]`; the reviewer **dropped the minus sign**. Fix the manuscript table; decide whether to trim the genuine non-drug-like outliers (MolWt→2,285 Da) |
| **Ch4·Q2** | Text says "2048 features"; code uses **11 RDKit descriptors, no fingerprints** ("2048" appears nowhere) | Fix the prose (or add fingerprints if 2048 was intended) |

### B3. Factual / scientific (1)
| Q | Issue |
|---|-------|
| **Ch6·Q1** | Dimorphite-DL pH range is hardcoded 6.4–8.4 ("physiological"). Does that match the Hargrove **SPR assay buffer**? Only you can source this. (It's now a function parameter, so it's a one-line retune.) |

### B4. Small / low-stakes (4)
| Q | Issue |
|---|-------|
| **Ch3·Q2** | Final code cell is an incomplete cross-chapter demo — complete, remove, or mark as a teaser |
| **Ch6·Q3** | Kennard–Stone seeds from the farthest *point*, not the farthest *pair*. Note: **KSA isn't even used for the real split**, so this only affects an illustrative figure |
| **Ch7·Q1** | Is the SOM actually **toroidal** as the text claims? |
| **Ch7·Q2** | The "redundancy ≈ 0" claim now that the diversity metric is fixed (self-pairs excluded) |

---

## C. Status board

| Ch | Status | Open |
|----|--------|------|
| 1 | ✅ done | — |
| 2 | ✅ done | — |
| 3 | 🔶 code clean | Q1, Q2 |
| 4 | 🔶 code clean | Q1, Q2 |
| 5 | 🔶 code clean *(was entirely unrunnable — fixed)* | Q1 |
| 6 | 🔶 code clean | Q1–Q5 |
| 7 | 🔶 code clean *(output refresh in flight)* | Q1, Q2 |
| 8–12, C | ⛔ **gated on A1** (PR merges) | — |

**Nothing is committed.** All QA work sits in the working tree awaiting A1/A2.

---

## D. Suggested order

1. **A1 + A2** (merge + push) — unblocks ch8–12 *and* closes all 82 security alerts.
2. **B1** — one ruling clears 4 questions and is the only one that changes a chapter's conclusion.
3. **B2** — three quick manuscript fixes.
4. **A3, B3, B4** — mop-up.
