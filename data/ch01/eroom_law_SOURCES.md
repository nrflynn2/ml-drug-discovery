# Data sources — Figure 1.2 (Eroom's Law)

`eroom_law.csv` — pharmaceutical R&D efficiency, 1950–2025 (76 rows).

The series is a ratio of two independently sourced quantities, following the construction in
Scannell, Blanckley, Boldon & Warrington (2012), "Diagnosing the decline in pharmaceutical R&D
efficiency," *Nature Reviews Drug Discovery* 11, 191–200. It is **computed from the two source
series, not fitted to the published trend** — that it reproduces Scannell's headline result is a
check on the reconstruction, not an input to it.

Rebuild the CSV and the figure from scratch with `python data/ch01/eroom_law_build.py`, which carries
every source value as a literal.

## Numerator: FDA new drug approvals

New molecular entities plus new therapeutic biologics approved by the US FDA.

- **1950–1992:** FDA History Office, "Summary of NDA Approvals & Receipts, 1938 to the present"
  (New Molecular Entities column). Underlying sources are the FDA Annual Reports (1950–1976), the
  CDER "Offices of Drug Evaluation: Statistical Report" typescript (1951–1989), and de Haen's
  "Compilation of New Drugs" (1940–1950). **Data through 1976 are fiscal years** (1 Jul – 30 Jun);
  calendar years thereafter.
- **1993–2025:** FDA CDER, "Novel Drug Approvals" (NMEs approved under an NDA plus new therapeutic
  biologics under a BLA).

## Denominator: pharmaceutical R&D spending

Total R&D of PhRMA member companies, from the *PhRMA Annual Membership Survey*, deflated by the BLS
CPI-U (annual average, 1982–84 = 100) to constant 2024 dollars. The survey reports directly only
from 1983, so earlier years are filled in three ways, each flagged per row in `rd_provenance`:

| Span | Treatment |
|---|---|
| 1983–2024 | Reported by the survey |
| 1970–1982 | Log-linear interpolation between published anchors (\$0.62 bn in 1970, \$2.0 bn in 1980, 1983 reported) |
| 1950–1969 | **Reconstructed** log-linearly from a \$50 m 1950 anchor. No comparable source exists for this span. |
| 2025 | **Provisional**, carried forward from 2024 pending publication of the survey |

The \$50 m 1950 anchor implies R&D intensity of roughly 4% of a mid-century US drug market of about
\$1.2 bn, rising to the documented 12.4% by 1970. Its sensitivity is carried in the CSV: moving the
anchor between \$35 m and \$85 m shifts the fitted halving time from 9.3 to 11.0 years.

## Columns

| Column | Meaning |
|---|---|
| `year` | 1950–2025 |
| `fda_new_drug_approvals` | Approval count for the year (the numerator) |
| `approvals_source` | Which of the two FDA series the count came from |
| `rd_nominal_musd` | PhRMA member R&D, nominal US\$ millions |
| `rd_provenance` | How that year's R&D value was obtained (reported / interpolated / reconstructed / provisional) |
| `cpi_u_annual_avg_1982_84_100` | BLS CPI-U annual average, 1982–84 = 100 |
| `rd_real_2024_busd` | R&D deflated to constant 2024 US\$ billions (the denominator) |
| `approvals_per_busd_rd` | **The Eroom's Law metric.** Annual approvals divided by real R&D |
| `approvals_5yr_mean` | 5-year centred mean of the approval count |
| `approvals_per_busd_rd_5yr` | 5-year centred mean of the metric (the heavy crimson line in the figure) |
| `rd_real_5yr_mean_busd` | 5-year centred mean of real R&D |
| `rd_real_index_1950` | Real R&D re-indexed to 1950 = 1 (lower panel) |
| `approvals_index_1950` | Smoothed approvals re-indexed to 1950 = 1 (lower panel) |
| `eroom_trend_1950_2010_fitted` | Least-squares log-linear fit to 1950–2010, extended across the full span |
| `sens_rd_nominal_musd_anchor35m` | Nominal R&D under a \$35 m 1950 anchor (blank from 1970) |
| `sens_rd_nominal_musd_anchor85m` | Nominal R&D under an \$85 m 1950 anchor (blank from 1970) |
| `sens_approvals_per_busd_rd_anchor35m` | The metric under the \$35 m anchor (blank from 1970) |
| `sens_approvals_per_busd_rd_anchor85m` | The metric under the \$85 m anchor (blank from 1970) |
| `evidence_tier` | A, B, or C (below) |

### Evidence tiers

- **A — both series reported** (55 rows, 1970–2024): numerator and denominator each trace to a
  published source.
- **B — reconstructed denominator** (20 rows, 1950–1969): approvals are reported, R&D is
  reconstructed from the 1950 anchor. Drawn dashed in the figure, with the sensitivity band.
- **C — provisional denominator** (1 row, 2025): R&D carried forward from 2024. Marked with an open
  circle in the figure.

## What the data show

From 1950 to 2010 the metric falls by a factor of about 66, halving roughly every 9.9 years, with an
R² of 0.95 against the smoothed series. This closely reproduces Scannell's published halving of
"roughly every 9 years" and ~80-fold fall, from independently assembled data. The decomposition in
the lower panel shows this is a denominator story: real R&D grew about 160-fold while output grew
only about 2.5-fold.

After 2010 the decline stops. The metric flattens, and by 2025 sits about 3.9× above where the
1950–2010 trend line would have put it — the break documented by Ringel, Scannell, Baedeker &
Schulze (2020), "Breaking Eroom's Law," *Nature Reviews Drug Discovery* 19, 833–834, and attributed
there to oncology and orphan indications, expedited approval pathways, and roughly flat real R&D
budgets.

Two inflections are visible along the way: the 1962 Kefauver–Harris Amendments, which first required
proof of efficacy and reset the cost of an approval, and the mid-1990s bump from PDUFA clearing the
review backlog alongside the wave of HIV antiretrovirals.

## Caveats

1. **The height is an index, not a cost per drug.** PhRMA members are only 75–85% of industry R&D,
   and a shrinking share of originators. The slope is meaningful; the absolute level is not a
   cost-per-approval figure.
2. **An approval counts the same whether it is a breakthrough or a me-too.** The numerator measures
   regulatory output, not therapeutic value.
3. **CBER products are excluded from 1993 onward** — vaccines, cell and gene therapies. This
   understates recent output rather than flattering it.
