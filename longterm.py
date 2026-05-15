"""
Trachsler Diversification Study
================================
Long-Term Comparison — Exploratory (not in paper)
Compares the Resilience Portfolio (top-5 sectors) against a naive
equal-weight portfolio over the full 1999-2025 sample.

Note: This is purely exploratory. The Resilience Score is based on
crisis-period correlations and is not designed to maximise long-term
returns. Results here are informational only.

Produces:
  longterm_cumret.png     — cumulative return over full period
  longterm_annual.png     — annual returns comparison
  output/longterm_results.csv

Author: Fabio Trachsler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

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

TOP_K_SECTORS = [
    "Consumer Stap",
    "Real Estate",
    "Healthcare",
    "Financials",
    "Industrials",
]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading data ...")
sectors  = pd.read_parquet(os.path.join(DATA_DIR, "sectors.parquet"))
log_ret  = np.log(sectors / sectors.shift(1)).dropna()
all_sectors = list(log_ret.columns)

# ─────────────────────────────────────────────
# COMPUTE PORTFOLIO RETURNS
# ─────────────────────────────────────────────

res_daily   = log_ret[TOP_K_SECTORS].mean(axis=1)
naive_daily = log_ret[all_sectors].mean(axis=1)

res_cum   = res_daily.cumsum()
naive_cum = naive_daily.cumsum()

# Annualised return and volatility
n_years = (log_ret.index[-1] - log_ret.index[0]).days / 365.25

res_ann_ret  = res_daily.sum()   / n_years
naive_ann_ret= naive_daily.sum() / n_years
res_ann_vol  = res_daily.std()   * np.sqrt(252)
naive_ann_vol= naive_daily.std() * np.sqrt(252)
res_sharpe   = res_ann_ret   / res_ann_vol
naive_sharpe = naive_ann_ret / naive_ann_vol

# Max drawdown
def mdd(cum_series):
    roll_max = cum_series.cummax()
    return float((cum_series - roll_max).min())

res_mdd   = mdd(res_cum)
naive_mdd = mdd(naive_cum)

print(f"\n  Full period: {log_ret.index[0].date()} to {log_ret.index[-1].date()}")
print(f"\n  Resilience Portfolio (top {len(TOP_K_SECTORS)}):")
print(f"    Annualised return : {res_ann_ret*100:.2f}%")
print(f"    Annualised vol    : {res_ann_vol*100:.2f}%")
print(f"    Sharpe ratio      : {res_sharpe:.3f}")
print(f"    Max drawdown      : {res_mdd*100:.1f}%")
print(f"    Total log-return  : {res_cum.iloc[-1]*100:.1f}%")

print(f"\n  Naive Portfolio (all {len(all_sectors)}):")
print(f"    Annualised return : {naive_ann_ret*100:.2f}%")
print(f"    Annualised vol    : {naive_ann_vol*100:.2f}%")
print(f"    Sharpe ratio      : {naive_sharpe:.3f}")
print(f"    Max drawdown      : {naive_mdd*100:.1f}%")
print(f"    Total log-return  : {naive_cum.iloc[-1]*100:.1f}%")

# Annual returns
res_annual   = res_daily.resample("YE").sum() * 100
naive_annual = naive_daily.resample("YE").sum() * 100

# ─────────────────────────────────────────────
# PLOT 1 — CUMULATIVE RETURN
# ─────────────────────────────────────────────

print("\nPlotting cumulative returns ...")
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(res_cum.index,   res_cum.values   * 100,
        color="#2ecc71", linewidth=2.0, label=f"Resilience (top {len(TOP_K_SECTORS)})")
ax.plot(naive_cum.index, naive_cum.values * 100,
        color="#e74c3c", linewidth=2.0, linestyle="--",
        label=f"Naive (all {len(all_sectors)})")
ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.set_title("Cumulative Log-Return: Resilience vs. Naive Portfolio\n"
             "Full Sample 1999–2025 (exploratory, not in paper)",
             fontsize=12)
ax.set_ylabel("Cumulative log-return (%)")
ax.legend(fontsize=10, frameon=False)

# Add summary stats as text box
stats_text = (
    f"Resilience:  ann. return {res_ann_ret*100:.1f}%  |  "
    f"vol {res_ann_vol*100:.1f}%  |  Sharpe {res_sharpe:.2f}  |  "
    f"MDD {res_mdd*100:.1f}%\n"
    f"Naive:           ann. return {naive_ann_ret*100:.1f}%  |  "
    f"vol {naive_ann_vol*100:.1f}%  |  Sharpe {naive_sharpe:.2f}  |  "
    f"MDD {naive_mdd*100:.1f}%"
)
ax.text(0.01, 0.04, stats_text, transform=ax.transAxes,
        fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.8))

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "longterm_cumret.png")
plt.savefig(path1)
plt.close()
print(f"  [plot] saved -> {path1}")

# ─────────────────────────────────────────────
# PLOT 2 — ANNUAL RETURNS
# ─────────────────────────────────────────────

print("Plotting annual returns ...")
years = res_annual.index.year
x     = np.arange(len(years))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, res_annual.values,   width,
       label=f"Resilience (top {len(TOP_K_SECTORS)})",
       color="#2ecc71", alpha=0.85)
ax.bar(x + width/2, naive_annual.values, width,
       label=f"Naive (all {len(all_sectors)})",
       color="#e74c3c", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Annual log-return (%)")
ax.set_title("Annual Returns: Resilience vs. Naive Portfolio\n"
             "Full Sample 1999–2025 (exploratory)",
             fontsize=12)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "longterm_annual.png")
plt.savefig(path2)
plt.close()
print(f"  [plot] saved -> {path2}")

# ─────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────

df_out = pd.DataFrame({
    "Metric": ["Ann. Return (%)", "Ann. Vol (%)", "Sharpe", "Max DD (%)",
               "Total Log-Return (%)"],
    "Resilience": [res_ann_ret*100, res_ann_vol*100, res_sharpe,
                   res_mdd*100, res_cum.iloc[-1]*100],
    "Naive":      [naive_ann_ret*100, naive_ann_vol*100, naive_sharpe,
                   naive_mdd*100, naive_cum.iloc[-1]*100],
})
df_out.to_csv(os.path.join("output", "longterm_results.csv"), index=False)
print(f"\n  [csv] saved -> output/longterm_results.csv")

print("\n=== Long-term comparison complete ===")
