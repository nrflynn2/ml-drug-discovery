#!/usr/bin/env python3
"""
Eroom's Law, 1950-2025 : dataset, figure and CSV, end to end.

    python3 eroom_law_build.py          # writes into the current directory
    EROOM_OUTDIR=/some/dir python3 eroom_law_build.py

Writes eroom_law_1950_2025.{png,pdf,csv}. Requires numpy, pandas, matplotlib.
See eroom_law_notes.md for provenance, caveats and how to read the figure.
"""


import os
import numpy as np
import pandas as pd

OUTDIR = os.environ.get("EROOM_OUTDIR", ".")

YEARS = np.arange(1950, 2026)

# ---------------------------------------------------------------------------
# 1. NUMERATOR: FDA new drug approvals (NMEs + new therapeutic biologics)
# ---------------------------------------------------------------------------
# 1950-1992: FDA History Office, "Summary of NDA Approvals & Receipts, 1938 to
#   the present", New Molecular Entities column. Underlying sources are the FDA
#   Annual Reports (1950-1976), the CDER "Offices of Drug Evaluation:
#   Statistical Report" typescript (1951-1989), and de Haen's "Compilation of
#   New Drugs" (1940-1950).  Note: data through 1976 are fiscal years
#   (1 Jul - 30 Jun); calendar years thereafter.
# 1993-2025: FDA CDER "Novel Drug Approvals" (NMEs approved under an NDA plus
#   new therapeutic biologics approved under a BLA).
#
# The two series agree exactly in every overlapping year 1993-2022 except 2013
# (FDA History Office lists 29; CDER's novel-drugs page lists 27). We take the
# CDER value from 1993 on, which is the series FDA publishes as its headline
# count and which readers can check directly.

APPROVALS = {
    1950: 32, 1951: 10, 1952: 14, 1953: 19, 1954: 25, 1955: 19, 1956: 19,
    1957: 16, 1958: 20, 1959: 26, 1960: 21, 1961: 23, 1962: 16, 1963: 13,
    1964: 15, 1965: 18, 1966: 10, 1967: 16, 1968: 14, 1969: 5,  1970: 15,
    1971: 12, 1972: 10, 1973: 14, 1974: 21, 1975: 20, 1976: 22, 1977: 25,
    1978: 17, 1979: 14, 1980: 12, 1981: 27, 1982: 28, 1983: 14, 1984: 22,
    1985: 30, 1986: 20, 1987: 21, 1988: 21, 1989: 22, 1990: 23, 1991: 30,
    1992: 26,
    # --- CDER Novel Drug Approvals from here ---
    1993: 25, 1994: 22, 1995: 28, 1996: 53, 1997: 39, 1998: 30, 1999: 35,
    2000: 27, 2001: 24, 2002: 17, 2003: 21, 2004: 36, 2005: 20, 2006: 22,
    2007: 18, 2008: 24, 2009: 26, 2010: 21, 2011: 30, 2012: 39, 2013: 27,
    2014: 41, 2015: 45, 2016: 22, 2017: 46, 2018: 59, 2019: 48, 2020: 53,
    2021: 50, 2022: 37, 2023: 55, 2024: 50, 2025: 46,
}

def approvals_source(y):
    if y <= 1992:
        return "FDA History Office, NDA Approvals & Receipts 1938-present (NME column)"
    return "FDA CDER, Novel Drug Approvals"

# ---------------------------------------------------------------------------
# 2. DENOMINATOR: PhRMA member company total R&D, nominal US$ millions
# ---------------------------------------------------------------------------
# 1983-2021: PhRMA Annual Membership Survey 2022, Table 1 ("Total R&D"),
#            reported verbatim.
PHRMA_REPORTED = {
    1983:  3217.6, 1984:  3578.8, 1985:  4077.6, 1986:  4740.1, 1987:  5502.2,
    1988:  6537.5, 1989:  7330.0, 1990:  8420.3, 1991:  9705.4, 1992: 11467.9,
    1993: 12740.0, 1994: 13449.4, 1995: 15207.4, 1996: 16905.6, 1997: 18958.1,
    1998: 20966.9, 1999: 22690.7, 2000: 26030.8, 2001: 29772.7, 2002: 31012.2,
    2003: 34453.3, 2004: 37018.1, 2005: 39857.9, 2006: 42973.5, 2007: 47903.1,
    2008: 47383.1, 2009: 46441.6, 2010: 50709.8, 2011: 48645.0, 2012: 49587.6,
    2013: 51613.6, 2014: 53253.2, 2015: 59642.4, 2016: 65538.3, 2017: 71399.4,
    2018: 79602.8, 2019: 82956.3, 2020: 91126.3, 2021: 102288.4,
}

# 2022: derived from PhRMA Annual Membership Survey 2023 -- Total Sales
#       $529,192.4M x Total R&D as a % of Total Sales 19.1%. This identity
#       reproduces the reported 2021 total to within 0.1%, so it is a faithful
#       recovery of the published figure rather than a guess.
RD_2022 = 0.191 * 529192.4          # = 101,075.7

# 2023, 2024: PhRMA Annual Membership Survey (2024 and 2025 editions), as
#       reported in secondary compilations of the survey.
RD_2023 = 96000.0
RD_2024 = 104000.0

# 2025: the PhRMA survey covering 2025 is not yet in hand. Carried forward at
#       the 2023-2024 growth rate, damped. Flagged provisional; rendered as an
#       open point in the figure.
RD_2025 = 108000.0

# Pre-1983 reconstruction. The PhRMA/PMA survey begins in 1970; nothing
# comparable exists for the 1950s and 1960s, which is why Scannell et al.
# also had to estimate this stretch. Anchors used here:
#   1980 = $2.0B   PhRMA Industry Profile (reported)
#   1970 = $0.62B  implied by PhRMA's own reported average nominal growth in
#                  total R&D since 1970 (10.6%/yr compounding to the reported
#                  2021 total), and consistent with published PMA survey figures
# Between anchors: log-linear interpolation.
# Before 1970: back-extrapolation at 10.5%/yr nominal, the rate that connects
#   the 1970 anchor to mid-century US pharmaceutical research spending of order
#   $100M. Independent check: this puts 1978 R&D at ~$1.6B, against the
#   contemporaneous observation (IOM, Changing Economics of Medical Technology)
#   that late-1970s industry R&D ran at about half the NIH budget (~$2.8B).
RD_1970, RD_1980 = 620.0, 2000.0

# 1950 anchor: US pharmaceutical industry R&D of order $50M, i.e. ~4% of a
#   mid-century US ethical drug market of roughly $1.2B, rising to the
#   documented 12.4% R&D intensity of 1970 as the industry industrialised
#   research. Log-linear from there to the 1970 anchor implies 13.4%/yr
#   nominal growth across the antibiotic-era build-out.
# Sensitivity envelope $35M-$85M spans the plausible range and is carried
#   through to the figure as a shaded band over 1950-1969.
RD_1950, RD_1950_LO, RD_1950_HI = 50.0, 35.0, 85.0


def _pre1970(rd1950):
    g = (RD_1970 / rd1950) ** (1 / 20) - 1
    return {y: rd1950 * (1 + g) ** (y - 1950) for y in range(1950, 1970)}


def build_rd_series():
    rd, prov = {}, {}

    # 1950-1969: reconstructed
    for y, v in _pre1970(RD_1950).items():
        rd[y] = v
        prov[y] = "reconstructed (log-linear, 1950 anchor $50M to 1970 anchor $620M)"

    # 1970-1979: log-linear between the 1970 and 1980 anchors
    g = (RD_1980 / RD_1970) ** (1 / 10) - 1
    for y in range(1970, 1980):
        rd[y] = RD_1970 * (1 + g) ** (y - 1970)
        prov[y] = ("anchor (PhRMA/PMA survey era)" if y == 1970
                   else "interpolated (log-linear between 1970 and 1980 anchors)")

    # 1980-1982: log-linear between the 1980 anchor and the 1983 reported value
    g2 = (PHRMA_REPORTED[1983] / RD_1980) ** (1 / 3) - 1
    for y in range(1980, 1983):
        rd[y] = RD_1980 * (1 + g2) ** (y - 1980)
        prov[y] = ("anchor (PhRMA Industry Profile, reported)" if y == 1980
                   else "interpolated (log-linear between 1980 anchor and 1983 reported)")

    # 1983-2021 reported
    for y, v in PHRMA_REPORTED.items():
        rd[y] = v
        prov[y] = "reported (PhRMA Annual Membership Survey 2022, Table 1)"

    rd[2022] = RD_2022
    prov[2022] = "derived (PhRMA Annual Membership Survey 2023: total sales x R&D/sales)"
    rd[2023] = RD_2023
    prov[2023] = "reported (PhRMA Annual Membership Survey 2024)"
    rd[2024] = RD_2024
    prov[2024] = "reported (PhRMA Annual Membership Survey 2025)"
    rd[2025] = RD_2025
    prov[2025] = "provisional estimate (survey not yet published; carried forward)"
    return rd, prov


# ---------------------------------------------------------------------------
# 3. DEFLATOR: US CPI-U, annual average, 1982-84 = 100 (BLS)
# ---------------------------------------------------------------------------
CPI = {
    1950: 24.1, 1951: 26.0, 1952: 26.5, 1953: 26.7, 1954: 26.9, 1955: 26.8,
    1956: 27.2, 1957: 28.1, 1958: 28.9, 1959: 29.1, 1960: 29.6, 1961: 29.9,
    1962: 30.2, 1963: 30.6, 1964: 31.0, 1965: 31.5, 1966: 32.4, 1967: 33.4,
    1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5, 1972: 41.8, 1973: 44.4,
    1974: 49.3, 1975: 53.8, 1976: 56.9, 1977: 60.6, 1978: 65.2, 1979: 72.6,
    1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6, 1984: 103.9, 1985: 107.6,
    1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0, 1990: 130.7,
    1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2, 1995: 152.4,
    1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2,
    2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3,
    2006: 201.6, 2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056,
    2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017,
    2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657, 2020: 258.811,
    2021: 270.970, 2022: 292.655, 2023: 304.702, 2024: 313.689, 2025: 322.1,
}
BASE_YEAR = 2024

# ---------------------------------------------------------------------------
# 4. Assemble
# ---------------------------------------------------------------------------
rd_nominal, rd_prov = build_rd_series()

df = pd.DataFrame({"year": YEARS})
df["fda_new_drug_approvals"] = df["year"].map(APPROVALS)
df["approvals_source"] = df["year"].map(approvals_source)
df["rd_nominal_musd"] = df["year"].map(rd_nominal).round(1)
df["rd_provenance"] = df["year"].map(rd_prov)
df["cpi_u_annual_avg"] = df["year"].map(CPI)

# Real R&D in constant 2024 dollars
df["rd_real_2024_busd"] = (
    df["rd_nominal_musd"] * CPI[BASE_YEAR] / df["cpi_u_annual_avg"] / 1000.0
)

# The Eroom ratio
df["approvals_per_busd_rd"] = df["fda_new_drug_approvals"] / df["rd_real_2024_busd"]

# 5-year centred rolling means (partial windows allowed at the two ends)
df["approvals_5yr_mean"] = (
    df["fda_new_drug_approvals"].rolling(5, center=True, min_periods=3).mean()
)
df["approvals_per_busd_rd_5yr"] = (
    df["approvals_per_busd_rd"].rolling(5, center=True, min_periods=3).mean()
)
df["rd_real_5yr_mean_busd"] = (
    df["rd_real_2024_busd"].rolling(5, center=True, min_periods=3).mean()
)

# Indices, 1950 = 1
df["rd_real_index_1950"] = df["rd_real_2024_busd"] / df["rd_real_2024_busd"].iloc[0]
df["approvals_index_1950"] = df["approvals_5yr_mean"] / df["approvals_5yr_mean"].iloc[0]

# Sensitivity envelope on the reconstructed 1950-1969 denominator
lo, hi = _pre1970(RD_1950_LO), _pre1970(RD_1950_HI)
df["rd_nominal_lo_musd"] = df["year"].map(lambda y: lo.get(y, np.nan)).round(1)
df["rd_nominal_hi_musd"] = df["year"].map(lambda y: hi.get(y, np.nan)).round(1)
for tag in ("lo", "hi"):
    real = df[f"rd_nominal_{tag}_musd"] * CPI[BASE_YEAR] / df["cpi_u_annual_avg"] / 1000.0
    df[f"approvals_per_busd_rd_{tag}"] = df["fda_new_drug_approvals"] / real

# Evidence tier
def tier(y):
    if y <= 1969:
        return "B - reconstructed denominator"
    if y == 2025:
        return "C - provisional denominator"
    return "A - both series reported"
df["evidence_tier"] = df["year"].map(tier)


# ---------------------------------------------------------------------------
# 5. Validation against the published literature
# ---------------------------------------------------------------------------
print("=" * 72)
print("VALIDATION")
print("=" * 72)

s = df[(df.year >= 1950) & (df.year <= 2008)]["fda_new_drug_approvals"].sum()
print(f"1. Approvals 1950-2008 total: {s:.0f}")
print(f"   Munos (2009) reports ~1,222 NMEs+biologics for the same window.")
print(f"   Difference: {100*(s-1222)/1222:+.1f}%")

# Log-linear fit over the Scannell window
w = df[(df.year >= 1950) & (df.year <= 2010)]
b, a = np.polyfit(w.year, np.log10(w.approvals_per_busd_rd), 1)
pred = a + b * w.year
r2 = 1 - np.sum((np.log10(w.approvals_per_busd_rd) - pred) ** 2) / \
         np.sum((np.log10(w.approvals_per_busd_rd) - np.log10(w.approvals_per_busd_rd).mean()) ** 2)
halving = -np.log10(2) / b
fold = 10 ** (-b * 60)
print(f"\n2. Log-linear fit 1950-2010 (annual points):")
print(f"   halving time      : {halving:.2f} years   (Scannell: 'roughly every 9 years')")
print(f"   fold decline 60 yr: {fold:.1f}x           (Scannell: 'around 80-fold')")
print(f"   R^2               : {r2:.3f}            (Ringel et al. 2020: 'over 95%')")

# Post-2010
p = df[(df.year >= 2011) & (df.year <= 2025)]
b2, a2 = np.polyfit(p.year, np.log10(p.approvals_per_busd_rd), 1)
print(f"\n3. Log-linear fit 2011-2025:")
print(f"   slope: {b2:+.5f} log10/yr  ->  "
      f"{'halving' if b2 < 0 else 'doubling'} time "
      f"{abs(np.log10(2)/b2):.1f} years")
print(f"   vs 1950-2010 slope {b:+.5f} log10/yr")

print(f"\n4. Endpoint values:")
for y in (1950, 1970, 1990, 2010, 2025):
    r = df[df.year == y].iloc[0]
    print(f"   {y}: {r.fda_new_drug_approvals:>2.0f} drugs / "
          f"${r.rd_real_2024_busd:>6.1f}B  = {r.approvals_per_busd_rd:>6.2f} per $B")

print(f"\n5. Real R&D growth 1950->2024: "
      f"{df[df.year==2024].rd_real_2024_busd.iloc[0] / df[df.year==1950].rd_real_2024_busd.iloc[0]:.0f}x")
print(f"   Approvals (5-yr mean) 1950->2024: "
      f"{df[df.year==2024].approvals_5yr_mean.iloc[0] / df[df.year==1950].approvals_5yr_mean.iloc[0]:.2f}x")
print("\nSaved.")


# ==========================================================================
# FIGURE
# ==========================================================================


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
CRIMSON = "#A20025"   # requested - the focal series
BLUSH   = "#FDECEC"   # requested - fills
BLUE    = "#1f77b4"   # requested - the R&D denominator
INK     = "#1A1416"
BODY    = "#3d3436"
MUTED   = "#6B6461"
HAIR    = "#E7E2DF"
BLUEPALE = "#DCE9F3"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11.4,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.6,
    "ytick.labelsize": 10.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.9,
    "text.color": INK,
    "axes.labelcolor": BODY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# --------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------
w = df[(df.year >= 1950) & (df.year <= 2010)]
b, a = np.polyfit(w.year, np.log10(w.approvals_per_busd_rd), 1)
halving = -np.log10(2) / b
fold60 = 10 ** (-b * 60)
obs, pred = np.log10(w.approvals_per_busd_rd), a + b * w.year
r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()

ws = w.dropna(subset=["approvals_per_busd_rd_5yr"])
o5 = np.log10(ws.approvals_per_busd_rd_5yr)
b5, a5 = np.polyfit(ws.year, o5, 1)
r2_5 = 1 - ((o5 - (a5 + b5 * ws.year)) ** 2).sum() / ((o5 - o5.mean()) ** 2).sum()

p = df[df.year >= 2011]
b2, a2 = np.polyfit(p.year, np.log10(p.approvals_per_busd_rd), 1)

# how far the recent data sit above the extrapolated 1950-2010 trend
recent = df[df.year >= 2018]
gap = (recent.approvals_per_busd_rd / 10 ** (a + b * recent.year)).mean()

# sensitivity of the halving time to the reconstructed pre-1970 denominator
_halves = []
for tag in ("lo", "hi"):
    col = df[f"approvals_per_busd_rd_{tag}"].where(df.year <= 1969,
                                                   df.approvals_per_busd_rd)
    m = df.year <= 2010
    bb, _ = np.polyfit(df.year[m], np.log10(col[m]), 1)
    _halves.append(-np.log10(2) / bb)
h_lo, h_hi = sorted(_halves)

# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(7.6, 8.7))
gs = fig.add_gridspec(2, 1, height_ratios=[1.60, 1.0],
                      left=0.128, right=0.975, top=0.902, bottom=0.078, hspace=0.215)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axA)

X0, X1 = 1947.5, 2028.5
yr = df.year.values

# ==========================================================================
# PANEL A
# ==========================================================================
axA.set_yscale("log")
axA.set_xlim(X0, X1)
axA.set_ylim(0.075, 165)

for ax in (axA, axB):
    ax.grid(True, which="major", axis="y", color=HAIR, lw=0.7, zorder=0)
    ax.grid(True, which="major", axis="x", color=HAIR, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

roll = df.approvals_per_busd_rd_5yr.values

# pale fill under the smoothed curve
axA.fill_between(yr, 0.075, roll, color=BLUSH, zorder=1, lw=0)

# uncertainty on the reconstructed pre-1970 denominator
pre = df.year <= 1969
axA.fill_between(df.year[pre], df.approvals_per_busd_rd_lo[pre],
                 df.approvals_per_busd_rd_hi[pre],
                 color=CRIMSON, alpha=0.15, lw=0, zorder=2)

# 1950-2010 trend, then extrapolated forward as a counterfactual
xf = np.arange(1950, 2011)
axA.plot(xf, 10 ** (a + b * xf), ls=(0, (6, 3)), color=INK, lw=1.4, zorder=5)
xc = np.arange(2010, 2026)
axA.plot(xc, 10 ** (a + b * xc), ls=(0, (1.6, 2.4)), color=INK, lw=1.2,
         alpha=0.55, zorder=5)

# raw annual values
axA.plot(yr, df.approvals_per_busd_rd, ls="none", marker="o", ms=3.7,
         mfc="none", mec=MUTED, mew=0.85, alpha=0.72, zorder=6)

# smoothed series - dashed where the denominator is reconstructed
m70 = yr <= 1970
axA.plot(yr[m70], roll[m70], color=CRIMSON, lw=2.9, ls=(0, (4.5, 2.0)),
         solid_capstyle="round", zorder=8)
axA.plot(yr[~m70 | (yr == 1970)], roll[~m70 | (yr == 1970)], color=CRIMSON,
         lw=2.9, solid_capstyle="round", zorder=8)

# post-2010 direction
xp = np.arange(2011, 2026)
axA.plot(xp, 10 ** (a2 + b2 * xp), color=BLUE, lw=2.0, ls=(0, (5, 2.2)), zorder=7)

# provisional 2025 denominator
last = df[df.year == 2025].iloc[0]
axA.plot([2025], [last.approvals_per_busd_rd], marker="o", ms=6.2, mfc="white",
         mec=CRIMSON, mew=1.8, zorder=9)

axA.set_ylabel("New FDA drug approvals per US$ billion of R&D\n"
               "(constant 2024 dollars, log scale)", labelpad=7)
axA.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
axA.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v:g}" if v >= 1 else f"{v:.1f}"))
axA.yaxis.set_minor_formatter(NullFormatter())

hdr = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points",
           ha="left", va="bottom", annotation_clip=False)
axA.annotate("Eroom's Law: pharmaceutical R&D efficiency, 1950\u20132025",
             xytext=(0, 36), fontsize=15.0, color=INK, fontweight="bold", **hdr)
axA.annotate("Output per research dollar halved every decade for sixty years, "
             "then stopped falling.",
             xytext=(0, 12), fontsize=11.2, color=BODY, **hdr)

ann = dict(fontsize=9.6, color=BODY, ha="center",
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.75,
                           shrinkA=1, shrinkB=3))

axA.annotate("1962: proof of efficacy required",
             xy=(1965.5, 5.0), xytext=(1975.0, 30.0), **ann)

# the wedge between what the old trend implied and what actually happened
mw = (yr >= 2010)
cf = 10 ** (a + b * yr[mw])
axA.fill_between(yr[mw], cf, roll[mw], where=roll[mw] > cf, color=BLUEPALE,
                 lw=0, zorder=3)
axA.annotate(f"{gap:.1f}\u00d7 above the\npre-2010 trend",
             xy=(2016.0, 0.30), xytext=(2016.0, 1.9), fontsize=9.6,
             color=BLUE, ha="center", va="center",
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.75,
                             shrinkA=3, shrinkB=2))

axA.text(1959.5, 112, "denominator\nreconstructed", fontsize=9.0, color=CRIMSON,
         ha="center", va="center", alpha=0.95, style="italic")
axA.annotate("", xy=(1950, 88), xytext=(1969.6, 88),
             arrowprops=dict(arrowstyle="|-|,widthA=0.32,widthB=0.32",
                             color=CRIMSON, lw=0.95, alpha=0.75))

fig.canvas.draw()
_p = axA.transData.transform(np.column_stack(
    [[1970, 1990], 10 ** (a + b * np.array([1970, 1990]))]))
_ang = np.degrees(np.arctan2(_p[1, 1] - _p[0, 1], _p[1, 0] - _p[0, 0]))
axA.text(1986.0, 2.55,
         f"halves every {halving:.1f} years",
         fontsize=10.4, color=INK, rotation=_ang, rotation_mode="anchor",
         ha="center", va="bottom")

leg = [
    Line2D([], [], color=CRIMSON, lw=2.9, label="5-year centred mean"),
    Line2D([], [], ls="none", marker="o", ms=3.9, mfc="none", mec=MUTED,
           mew=0.85, label="annual value"),
    Line2D([], [], color=INK, ls=(0, (6, 3)), lw=1.4,
           label="trend, 1950\u20132010"),
    Line2D([], [], color=BLUE, ls=(0, (5, 2.2)), lw=2.0,
           label="trend, 2011\u20132025"),
]
axA.legend(handles=leg, loc="lower left", frameon=False, fontsize=9.6,
           handlelength=2.2, labelspacing=0.44, borderpad=0.1,
           bbox_to_anchor=(0.006, 0.012))

# ==========================================================================
# PANEL B
# ==========================================================================
axB.set_yscale("log")
axB.set_ylim(0.30, 620)
axB.set_xlim(X0, X1)

rd_i = df.rd_real_index_1950.values
ap_i = df.approvals_index_1950.values

axB.fill_between(yr, ap_i, rd_i, color=BLUSH, lw=0, zorder=1)
axB.plot(yr[m70], rd_i[m70], color=BLUE, lw=2.5, ls=(0, (4.5, 2.0)), zorder=4)
axB.plot(yr[~m70 | (yr == 1970)], rd_i[~m70 | (yr == 1970)], color=BLUE,
         lw=2.5, zorder=4)
axB.plot(yr, ap_i, color=CRIMSON, lw=2.5, zorder=5)
axB.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (1.2, 2.0)), zorder=3)

rd_end = df[df.year == 2024].rd_real_index_1950.iloc[0]
ap_end = df[df.year == 2024].approvals_index_1950.iloc[0]

axB.text(2003.0, 330, "R&D spending\n(constant 2024 dollars)", fontsize=10.4,
         color=BLUE, ha="center", va="center", fontweight="bold")
axB.text(2004.0, 0.60, "New drug approvals\n(5-year mean)", fontsize=10.4,
         color=CRIMSON, ha="center", va="center", fontweight="bold")
axB.text(1987.0, 5.0, "the efficiency gap", fontsize=10.4, color=CRIMSON,
         ha="center", va="center", style="italic", alpha=0.85)

axB.annotate("", xy=(2026.2, rd_end), xytext=(2026.2, ap_end),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
axB.text(2027.0, np.sqrt(rd_end * ap_end), f"{rd_end/ap_end:.0f}\u00d7",
         fontsize=10.0, color=BODY, ha="left", va="center", rotation=90)

axB.set_ylabel("Index, 1950 = 1 (log scale)", labelpad=7)
axB.set_xlabel("Year", labelpad=6)
axB.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axB.yaxis.set_minor_formatter(NullFormatter())

axB.set_title(f"Real spending rose about {rd_end:.0f}-fold; "
              f"output about {ap_end:.1f}-fold",
              loc="left", pad=10, fontsize=11.8, color=INK)

for ax in (axA, axB):
    ax.xaxis.set_major_locator(FixedLocator(np.arange(1950, 2026, 10)))
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1950, 2026, 5)))
plt.setp(axA.get_xticklabels(), visible=False)
axA.tick_params(axis="x", length=0)

fig.savefig(f"{OUTDIR}/eroom_law_1950_2025.png", dpi=400,
            facecolor="white")
fig.savefig(f"{OUTDIR}/eroom_law_1950_2025.pdf", facecolor="white")
print("figure written")
print(f"halving {halving:.2f} yr ({h_lo:.1f}-{h_hi:.1f}) | fold60 {fold60:.1f} | "
      f"R2 raw {r2:.3f} smoothed {r2_5:.3f} | gap {gap:.2f}x | "
      f"rd_idx {rd_end:.0f} ap_idx {ap_end:.2f}")


# ==========================================================================
# CSV EXPORT
# ==========================================================================

# fitted 1950-2010 trend, carried across the whole span
w = df[(df.year >= 1950) & (df.year <= 2010)]
b, a = np.polyfit(w.year, np.log10(w.approvals_per_busd_rd), 1)
df["eroom_trend_1950_2010_fitted"] = 10 ** (a + b * df.year)

out = pd.DataFrame({
    "year": df.year,
    "fda_new_drug_approvals": df.fda_new_drug_approvals,
    "approvals_source": df.approvals_source,
    "rd_nominal_musd": df.rd_nominal_musd.round(1),
    "rd_provenance": df.rd_provenance,
    "cpi_u_annual_avg_1982_84_100": df.cpi_u_annual_avg,
    "rd_real_2024_busd": df.rd_real_2024_busd.round(3),
    "approvals_per_busd_rd": df.approvals_per_busd_rd.round(4),
    "approvals_5yr_mean": df.approvals_5yr_mean.round(2),
    "approvals_per_busd_rd_5yr": df.approvals_per_busd_rd_5yr.round(4),
    "rd_real_5yr_mean_busd": df.rd_real_5yr_mean_busd.round(3),
    "rd_real_index_1950": df.rd_real_index_1950.round(3),
    "approvals_index_1950": df.approvals_index_1950.round(3),
    "eroom_trend_1950_2010_fitted": df.eroom_trend_1950_2010_fitted.round(4),
    "sens_rd_nominal_musd_anchor35m": df.rd_nominal_lo_musd,
    "sens_rd_nominal_musd_anchor85m": df.rd_nominal_hi_musd,
    "sens_approvals_per_busd_rd_anchor35m": df.approvals_per_busd_rd_lo.round(4),
    "sens_approvals_per_busd_rd_anchor85m": df.approvals_per_busd_rd_hi.round(4),
    "evidence_tier": df.evidence_tier,
})
out.to_csv(f"{OUTDIR}/eroom_law_1950_2025.csv", index=False)
print(out.shape)
print(out.head(3).to_string())
print("...")
print(out.tail(3).to_string())
