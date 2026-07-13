# Data sources — Figure 1.2 (Eroom's Law)

`eroom_law.csv` — pharmaceutical R&D productivity, 1950–2024.

## Columns
- `year`
- `drugs_per_bn_rd` — a **reconstruction** of the Eroom's-Law efficiency metric (new drugs approved per
  inflation-adjusted US\$ billion of R&D). This is *not* a raw measurement — the underlying long-run
  R&D-spend series is proprietary/aggregated — it reproduces the published trend:
  - **1950–2010:** Scannell, Blanckley, Boldon & Warrington (2012), "Diagnosing the decline in
    pharmaceutical R&D efficiency," *Nature Reviews Drug Discovery* 11, 191–200. The metric halves
    roughly every 9 years (~100× decline over 1950–2010); anchored here at ~30 in 1950 (ratio 101.6×).
  - **2011–2024:** Ringel, Scannell, Baedeker & Schulze (2020), "Breaking Eroom's Law," *Nature Reviews
    Drug Discovery* 19, 833–834 — a modest post-2010 rebound, modeled as a gentle recovery.
- `fda_new_drug_approvals` — **real data**: FDA "all new drug approvals" per year, from Our World in
  Data (compiled from FDA/CDER),
  <https://ourworldindata.org/grapher/new-drugs-approved-in-the-united-states> (downloaded 2026-07).
  Blank where OWID has no entry for that year.

## Note
The efficiency series is illustrative of the *published* Eroom's-Law trend, not a primary measurement;
treat `drugs_per_bn_rd` as a teaching reconstruction with the citations above. `fda_new_drug_approvals`
is the real, verifiable output series.
