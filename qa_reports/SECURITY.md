# Security remediation — Dependabot alerts (Workstream S) — **RESOLVED**

**Input:** the author's export of all **82 open Dependabot alerts** (`/home/noahr/dependabot_alerts.json`).

## Headline

**All 82 alerts were raised against a single file: `CH12_FLYNN_ML4DD/requirements.txt`** — the old
200-line *frozen `==` pin-list*, which the modernization had already deleted and replaced with a lean,
floor-based file. Nothing else in the repo was flagged (not the root `pyproject.toml`, not `uv.lock`,
not the conda env).

| | |
|---|---|
| Total alerts | **82** (32 unique packages) |
| Severity | 4 critical · 27 high · 38 medium · 13 low |
| Source manifest | `CH12_FLYNN_ML4DD/requirements.txt` (100%) |

## What the audit found (and the fix)

Replacing the frozen pin-list cleared the great majority outright. But the audit caught something a
"we bumped everything" assumption would have missed: **four packages were still declared with `>=`
floors low enough to *admit* a vulnerable release** — including the critical one. Those floors are now
raised to at-or-above the first patched version:

| package | old floor | new floor | severity | where |
|---------|-----------|-----------|----------|-------|
| **ray** | `>=2.11.0` | **`>=2.54.0`** | 🔴 **critical** | CH12 `requirements.txt` + `pyproject.toml` (`tuning` extra) |
| **pyarrow** | `>=16.0.0` | **`>=23.0.1`** | high | CH12 `requirements.txt` + `pyproject.toml` |
| **jupyterlab** | `>=4.3.0` | **`>=4.5.9`** | high | CH12 + **root** `pyproject.toml` |
| **torch** | `>=2.4.0` | **`>=2.10.0`** | medium | CH12 + **root** `pyproject.toml` |

The root `pyproject.toml` was **not** flagged (its `uv.lock` pins current versions), but its `jupyterlab`
and `torch` floors permitted the same vulnerable releases, so they were bumped too — hygiene, and
`torch>=2.10` also guarantees the `torch.accelerator` API that `utils.get_device()` relies on.

The remaining 26 flagged packages (aiohttp, pillow, requests, urllib3, jinja2, tornado, certifi,
nbconvert, jupyter-server, soupsieve, msgpack, setuptools, …) are **transitive only** — they have no
declared floor, so pip/uv resolve them to current, patched versions. The committed `uv.lock` already
ships patched releases for all of them (pillow 12.3, aiohttp 3.14.1, requests 2.34.2, urllib3 2.7.0,
jinja2 3.1.6, certifi 2026.6.17, jupyter-server 2.20.0 = the patched version for the one critical
jupyter-server advisory).

**Verification:** re-running the mapping against the bumped manifests gives
**`alerts whose CH12 floor still admits a vulnerable version: 0`**.

## Residual: 4 packages with NO upstream fix published

Dependabot lists no `first_patched_version` for these, so there is nothing to bump to. They are *not*
resolvable by version pinning:

| package | severity | our version | note |
|---------|----------|-------------|------|
| **ray** | 🔴 critical | (optional extra) | **Needs an author decision — see below** |
| mistune | medium | 3.3.2 (latest) | transitive via `nbconvert`; no patch exists |
| biopython | medium | latest | used by the ch12 mutation-scoring notebook; no patch exists |
| torch | low | 2.12.1 (latest) | no patch exists |

### ⚠️ Author decision — `ray` (critical, unpatched)
`ray` is used **only** by `CH12_FLYNN_ML4DD/scripts/tune.py` (hyperparameter tuning) and is already an
**optional** extra (`[tuning]`), so it is *not* installed by the notebook path readers use. Options:
1. **Keep as-is** (recommended): it stays optional + the floor is raised to `>=2.54.0`, and we document
   that the residual advisory has no upstream patch.
2. **Drop it**: remove the `tuning` extra / `scripts/tune.py` if the book doesn't lean on Ray Tune.

## Action to actually close the alerts

The fix is already in the working tree. Dependabot re-scans on push:

```bash
# after the modernization branch lands:
git push        # Dependabot re-scans CH12_FLYNN_ML4DD/requirements.txt
```
Expect the alert count to drop from 82 to ~the handful of unpatched advisories above.

## Note
`uv lock` warns that two transitive pins are yanked (`apsw 3.53.3.0`, `grpcio 1.82.0` — the latter a Ray
dependency). Not security alerts, but worth a re-lock once upstream settles.
