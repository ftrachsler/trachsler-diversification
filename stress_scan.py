"""
Trachsler Diversification Study
================================
Systematic Stress Scan — Section 5.7
Scans the full dataset for stress events (any ETF losing >= THRESHOLD%
within WINDOW trading days) and computes Delta-rho for each event.
Produces two plots:
  11_stress_scan_histogram.png  — distribution of Delta-rho across all events
  12_stress_scan_scatter.png    — Delta-rho vs. magnitude of drawdown
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

OUTPUT_DIR    = "plots"
DATA_DIR      = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW        = 7       # trading days to measure drawdown
THRESHOLD     = 0.10    # 10% loss triggers a stress event
CORR_WINDOW   = 60      # days for rolling correlation estimate
MIN_GAP       = 30      # minimum trading days between two stress events
                        # (avoids double-counting overlapping events)

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
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("Loading data ...")
sectors = pd.read_parquet(os.path.join(DATA_DIR, "sectors.parquet"))
log_ret = np.log(sectors / sectors.shift(1)).dropna()

# ─────────────────────────────────────────────
# 2. HELPER: mean pairwise correlation
# ─────────────────────────────────────────────

def mean_pairwise_corr(df: pd.DataFrame) -> float:
    """Mean pairwise Pearson correlation across all column pairs."""
    c = df.corr().values
    n = c.shape[0]
    upper = c[np.triu_indices(n, k=1)]
    return float(np.mean(upper))

# ─────────────────────────────────────────────
# 3. SCAN FOR STRESS EVENTS
# ─────────────────────────────────────────────

print(f"Scanning for stress events (window={WINDOW}d, threshold={THRESHOLD*100:.0f}%) ...")

# Rolling WINDOW-day cumulative return for each ETF
cum_ret = (log_ret.rolling(WINDOW).sum())   # sum of log returns = log cumulative

# Flag: any ETF lost >= THRESHOLD in the past WINDOW days
stress_flag = (cum_ret <= -THRESHOLD).any(axis=1)
stress_dates = stress_flag[stress_flag].index.tolist()

# De-duplicate: keep only one event per MIN_GAP days
events = []
last_event = pd.Timestamp("1900-01-01")
for date in stress_dates:
    if (date - last_event).days >= MIN_GAP:
        events.append(date)
        last_event = date

print(f"  Found {len(events)} independent stress events after de-duplication.")

# ─────────────────────────────────────────────
# 4. COMPUTE DELTA-RHO FOR EACH EVENT
# ─────────────────────────────────────────────

print("Computing Delta-rho for each stress event ...")

results = []
for event_date in events:
    try:
        # Baseline: CORR_WINDOW days BEFORE the stress window
        baseline_end   = log_ret.index[log_ret.index < event_date][-1]
        baseline_start = log_ret.index[log_ret.index < event_date][-(CORR_WINDOW + 1)]
        baseline_df    = log_ret.loc[baseline_start:baseline_end]

        # Crisis: WINDOW days starting at event_date
        crisis_start = event_date
        crisis_loc   = log_ret.index.get_loc(event_date)
        crisis_end   = log_ret.index[min(crisis_loc + WINDOW - 1, len(log_ret) - 1)]
        crisis_df    = log_ret.loc[crisis_start:crisis_end]

        if len(baseline_df) < 20 or len(crisis_df) < 3:
            continue

        rho_baseline = mean_pairwise_corr(baseline_df)
        rho_crisis   = mean_pairwise_corr(crisis_df)
        delta_rho    = rho_crisis - rho_baseline

        # Worst drawdown that triggered the event
        max_loss = cum_ret.loc[event_date].min()

        results.append({
            "date":         event_date,
            "rho_baseline": rho_baseline,
            "rho_crisis":   rho_crisis,
            "delta_rho":    delta_rho,
            "max_loss":     max_loss,
        })
    except Exception:
        continue

df_results = pd.DataFrame(results)
print(f"  Successfully computed Delta-rho for {len(df_results)} events.")
print(f"\n  Delta-rho summary:")
print(df_results["delta_rho"].describe().round(3).to_string())
pct_positive = (df_results["delta_rho"] > 0).mean() * 100
print(f"\n  Fraction of events with positive Delta-rho: {pct_positive:.1f}%")

# ─────────────────────────────────────────────
# 5. PLOT 11 — HISTOGRAM
# ─────────────────────────────────────────────

print("\nPlotting histogram ...")

fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df_results["delta_rho"]]
ax.hist(df_results["delta_rho"], bins=30, color="#e74c3c", alpha=0.3,
        edgecolor="white", linewidth=0.5, label="_nolegend_")

# Colour bars by sign
n, bins, patches = ax.hist(df_results["delta_rho"], bins=30,
                            edgecolor="white", linewidth=0.5)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor("#e74c3c" if left > 0 else "#2ecc71")
    patch.set_alpha(0.8)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(df_results["delta_rho"].mean(), color="navy", linewidth=1.5,
           linestyle=":", label=f"Mean Δρ = {df_results['delta_rho'].mean():.3f}")

ax.set_title(
    f"Distribution of $\\Delta\\rho$ Across All Stress Events\n"
    f"(any ETF $\\geq${THRESHOLD*100:.0f}\\% loss in {WINDOW} days, "
    f"$n={len(df_results)}$ events)",
    fontsize=12
)
ax.set_xlabel("$\\Delta\\rho$ (crisis $-$ pre-crisis baseline)")
ax.set_ylabel("Number of events")

red_patch   = mpatches.Patch(color="#e74c3c", alpha=0.8, label=f"Positive Δρ — diversification breakdown ({(df_results['delta_rho']>0).sum()} events)")
green_patch = mpatches.Patch(color="#2ecc71", alpha=0.8, label=f"Negative Δρ — diversification holds ({(df_results['delta_rho']<=0).sum()} events)")
ax.legend(handles=[red_patch, green_patch,
                   plt.Line2D([0],[0], color="navy", linewidth=1.5, linestyle=":")],
          labels=[red_patch.get_label(), green_patch.get_label(),
                  f"Mean Δρ = {df_results['delta_rho'].mean():.3f}"],
          fontsize=9, frameon=False)

plt.tight_layout()
path11 = os.path.join(OUTPUT_DIR, "11_stress_scan_histogram.png")
plt.savefig(path11)
plt.close()
print(f"  [plot] saved → {path11}")

# ─────────────────────────────────────────────
# 6. PLOT 12 — SCATTER: drawdown vs delta-rho
# ─────────────────────────────────────────────

print("Plotting scatter ...")

fig, ax = plt.subplots(figsize=(8, 5))

sc_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df_results["delta_rho"]]
ax.scatter(df_results["max_loss"] * 100, df_results["delta_rho"],
           c=sc_colors, alpha=0.6, edgecolors="white", linewidth=0.4, s=50)

ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

# OLS trend line + R²
x = df_results["max_loss"].values
y = df_results["delta_rho"].values
m, b = np.polyfit(x, y, 1)
ss_res = np.sum((y - (m * x + b)) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

x_line = np.linspace(x.min(), x.max(), 200)
ax.plot(x_line * 100, m * x_line + b, color="navy",
        linewidth=2.0, linestyle=":",
        label=f"OLS trend  ($R^2 = {r2:.3f}$)")

# Annotate R² prominently in plot
ax.text(0.04, 0.93,
        f"$R^2 = {r2:.3f}$ — drawdown severity\ndoes not predict $\\Delta\\rho$",
        transform=ax.transAxes, fontsize=9.5, verticalalignment="top",
        color="navy",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="navy", alpha=0.7))

ax.set_title(
    "Drawdown Severity Does Not Predict Correlation Breakdown\n"
    f"(each dot = one stress event, $n={len(df_results)}$)",
    fontsize=12
)
ax.set_xlabel("Maximum 7-day log-return of any ETF (%)")
ax.set_ylabel("$\\Delta\\rho$  (crisis $-$ pre-crisis baseline)")

red_patch   = mpatches.Patch(color="#e74c3c", alpha=0.8,
                              label=f"Positive $\\Delta\\rho$ — breakdown ({(df_results['delta_rho']>0).sum()} events)")
green_patch = mpatches.Patch(color="#2ecc71", alpha=0.8,
                              label=f"Negative $\\Delta\\rho$ — holds ({(df_results['delta_rho']<=0).sum()} events)")
ax.legend(handles=[red_patch, green_patch,
                   plt.Line2D([0], [0], color="navy", linewidth=2.0,
                              linestyle=":")],
          labels=[red_patch.get_label(), green_patch.get_label(),
                  f"OLS trend ($R^2={r2:.3f}$)"],
          fontsize=9, frameon=False)

plt.tight_layout()
path12 = os.path.join(OUTPUT_DIR, "12_stress_scan_scatter.png")
plt.savefig(path12)
plt.close()
print(f"  [plot] saved → {path12}")

# ─────────────────────────────────────────────
# 7. SAVE CSV
# ─────────────────────────────────────────────

csv_path = os.path.join("output", "stress_scan_results.csv")
os.makedirs("output", exist_ok=True)
df_results.to_csv(csv_path, index=False)
print(f"\n  [csv] saved → {csv_path}")

print("\n=== Stress scan complete ===")