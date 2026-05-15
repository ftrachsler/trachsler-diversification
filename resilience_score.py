"""
Trachsler Diversification Study
================================
Resilience Score Computation — Chapter 6
Computes the Trachsler Resilience Score for each sector based on
pairwise delta-rho values across all major crisis episodes.
Produces:
  13_resilience_scores.png  — bar chart of resilience scores ranked
  14_resilience_heatmap.png — heatmap of pairwise delta-rho values
  output/resilience_scores.csv
Author: Fabio Trachsler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OUTPUT_DIR = "plots"
DATA_DIR   = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("output",   exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# ─────────────────────────────────────────────
# CRISIS PERIODS (same as analysis.py)
# ─────────────────────────────────────────────

CRISES = [
    ("2000-03-01", "2002-10-31", "Dotcom Bust"),
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-12-31", "Rate Hike"),
]

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("Loading sector data ...")
sectors = pd.read_parquet(os.path.join(DATA_DIR, "sectors.parquet"))
log_ret = np.log(sectors / sectors.shift(1)).dropna()
n_sectors = len(log_ret.columns)
sector_names = list(log_ret.columns)

# ─────────────────────────────────────────────
# 2. COMPUTE PAIRWISE CORRELATIONS
#    per crisis and for the full non-crisis baseline
# ─────────────────────────────────────────────

print("Computing pairwise correlations ...")

# Build a mask for crisis days
crisis_mask = pd.Series(False, index=log_ret.index)
for start, end, _ in CRISES:
    crisis_mask.loc[start:end] = True

# Non-crisis baseline: full sample minus all crisis periods
baseline_ret = log_ret[~crisis_mask]
corr_normal  = baseline_ret.corr()

# Per-crisis correlations (averaged across all crises)
crisis_corrs = []
for start, end, label in CRISES:
    c = log_ret.loc[start:end].corr()
    if c.isnull().all().all():
        print(f"  Skipping {label} (no data)")
        continue
    crisis_corrs.append(c)
    print(f"  Computed correlation for {label}")

# Average correlation across all available crises
corr_crisis = pd.concat(crisis_corrs).groupby(level=0).mean()

# Pairwise delta-rho matrix
delta_rho = corr_crisis - corr_normal

# ─────────────────────────────────────────────
# 3. COMPUTE RESILIENCE SCORE
# ─────────────────────────────────────────────

print("\nComputing Trachsler Resilience Scores ...")

resilience = {}
for sector in sector_names:
    # Average (1 - delta_rho_ij) over all other sectors j
    others = [s for s in sector_names if s != sector]
    score  = np.mean([1 - delta_rho.loc[sector, j] for j in others])
    resilience[sector] = score

df_resilience = pd.DataFrame.from_dict(
    resilience, orient="index", columns=["Resilience Score"]
).sort_values("Resilience Score", ascending=False)
df_resilience["Rank"] = range(1, len(df_resilience) + 1)

print("\n  Trachsler Resilience Scores (ranked):")
print(df_resilience.round(4).to_string())

# ─────────────────────────────────────────────
# 4. PLOT 13 — BAR CHART OF RESILIENCE SCORES
# ─────────────────────────────────────────────

print("\nPlotting resilience score bar chart ...")

fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#2ecc71" if v >= df_resilience["Resilience Score"].mean()
          else "#e74c3c"
          for v in df_resilience["Resilience Score"]]

mean_val  = df_resilience["Resilience Score"].mean()
max_val   = df_resilience["Resilience Score"].max()
label_x   = max_val + 0.012   # fixed x position for all labels

bars = ax.barh(df_resilience.index[::-1],
               df_resilience["Resilience Score"][::-1],
               color=colors[::-1], alpha=0.85, edgecolor="white")

# All value labels at the same x position — straight vertical line
for bar, val in zip(bars, df_resilience["Resilience Score"][::-1]):
    ax.text(label_x, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
            color="black")

# Mean line only within the plot area (ymin/ymax in axes fraction)
ax.axvline(mean_val, color="navy", linewidth=1.5, linestyle=":",
           ymin=0.0, ymax=1.0)

ax.set_title("Trachsler Resilience Score by Sector\n"
             "(higher = more crisis-robust diversifier)",
             fontsize=12)
ax.set_xlabel("Resilience Score $R_i$")
ax.set_xlim(0, label_x + 0.06)

# Mean label bottom right below x-axis
ax.annotate(f"Mean = {mean_val:.4f}",
            xy=(mean_val, 0), xycoords=("data", "axes fraction"),
            xytext=(6, -28), textcoords="offset points",
            color="navy", fontsize=9, ha="left", va="top")

plt.tight_layout()

path13 = os.path.join(OUTPUT_DIR, "13_resilience_scores.png")
plt.savefig(path13)
plt.close()
print(f"  [plot] saved → {path13}")

# ─────────────────────────────────────────────
# 5. PLOT 14 — HEATMAP OF PAIRWISE DELTA-RHO
# ─────────────────────────────────────────────

print("Plotting delta-rho heatmap ...")

fig, ax = plt.subplots(figsize=(10, 8))

mask = np.triu(np.ones_like(delta_rho, dtype=bool), k=1)  # show lower triangle + diag

sns.heatmap(
    delta_rho,
    annot=True, fmt=".2f",
    cmap="RdYlGn_r",
    center=0,
    vmin=-0.3, vmax=0.5,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "$\\Delta\\rho$ (crisis minus normal)"},
)

ax.set_title("Pairwise $\\Delta\\rho$ Across All Major Crises\n"
             "(positive = correlation rose in crisis, diversification deteriorated)",
             fontsize=11)
plt.tight_layout()

path14 = os.path.join(OUTPUT_DIR, "14_resilience_heatmap.png")
plt.savefig(path14)
plt.close()
print(f"  [plot] saved → {path14}")

# ─────────────────────────────────────────────
# 6. SAVE CSV
# ─────────────────────────────────────────────

csv_path = os.path.join("output", "resilience_scores.csv")
df_resilience.to_csv(csv_path)
print(f"\n  [csv] saved → {csv_path}")

print("\n=== Resilience Score computation complete ===")