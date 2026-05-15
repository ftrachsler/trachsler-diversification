"""
Trachsler Diversification Study
================================
Backtest — Chapter 6
Compares the Resilience Portfolio (top-k sectors by Trachsler Resilience Score)
against a naive equal-weight portfolio across major AND minor crisis periods.

Important note (disclosed in paper):
  The Resilience Score is computed on the same major crises used for backtesting.
  This is an in-sample validation for major crises, not a true out-of-sample test.
  The minor crisis backtest is partially out-of-sample since minor crises
  were not used to compute the score.

Produces:
  15_backtest_returns.png         — cumulative returns during major crises
  16_backtest_summary.png         — summary bar charts major crises
  17_backtest_mini_returns.png    — cumulative returns during minor crises
  18_backtest_mini_summary.png    — summary bar charts minor crises
  output/backtest_results.csv
  output/backtest_mini_results.csv

Author: Fabio Trachsler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

K = 5

plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "serif",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

# ─────────────────────────────────────────────
# CRISIS PERIODS
# ─────────────────────────────────────────────

MAJOR_CRISES = [
    ("2000-03-01", "2002-10-31", "Dotcom Bust"),
    ("2008-09-01", "2009-03-31", "Global Financial Crisis"),
    ("2020-02-15", "2020-04-30", "COVID Crash"),
    ("2022-01-01", "2022-12-31", "Rate Hike Shock"),
]

MINI_CRISES = [
    ("2010-04-01", "2010-07-31", "Euro Debt Crisis"),
    ("2013-05-01", "2013-08-31", "Taper Tantrum"),
    ("2015-08-01", "2015-09-30", "Flash Crash"),
    ("2019-09-01", "2019-10-31", "Repo Market Crisis"),
    ("2025-04-01", "2025-05-31", "Trump Tariff Shock"),
]

# ─────────────────────────────────────────────
# RESILIENCE SCORE RANKING (from resilience_score.py)
# Top-5: Consumer Stap, Real Estate, Healthcare, Financials, Industrials
# ─────────────────────────────────────────────

TOP_K_SECTORS = [
    "Consumer Stap",
    "Real Estate",
    "Healthcare",
    "Financials",
    "Industrials",
]

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("Loading sector data ...")
sectors_main = pd.read_parquet(os.path.join(DATA_DIR, "sectors.parquet"))
sectors_mini = pd.read_parquet(os.path.join(DATA_DIR, "sectors_mini.parquet"))

log_ret_main = np.log(sectors_main / sectors_main.shift(1)).dropna()
log_ret_mini = np.log(sectors_mini / sectors_mini.shift(1)).dropna()

all_sectors = list(log_ret_main.columns)
print(f"  Available sectors: {all_sectors}")
print(f"  Top-{K} resilience sectors: {TOP_K_SECTORS}")

# ─────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def portfolio_cumret(log_ret_df, sectors):
    return log_ret_df[sectors].mean(axis=1).cumsum()

def max_drawdown(log_ret_df, sectors):
    port     = log_ret_df[sectors].mean(axis=1).cumsum()
    roll_max = port.cummax()
    return float((port - roll_max).min())

def mean_pairwise_corr(log_ret_df, sectors):
    c = log_ret_df[sectors].corr().values
    n = c.shape[0]
    return float(np.mean(c[np.triu_indices(n, k=1)]))

def total_return(log_ret_df, sectors):
    return float(log_ret_df[sectors].mean(axis=1).sum())

def run_backtest(crises, log_ret):
    results        = []
    crisis_cumrets = {}
    for start, end, label in crises:
        crisis_ret = log_ret.loc[start:end]
        if len(crisis_ret) < 5:
            print(f"  Skipping {label} (insufficient data)")
            continue

        res_ret   = total_return(crisis_ret, TOP_K_SECTORS)
        res_mdd   = max_drawdown(crisis_ret, TOP_K_SECTORS)
        res_corr  = mean_pairwise_corr(crisis_ret, TOP_K_SECTORS)
        res_cum   = portfolio_cumret(crisis_ret, TOP_K_SECTORS)

        naive_ret  = total_return(crisis_ret, all_sectors)
        naive_mdd  = max_drawdown(crisis_ret, all_sectors)
        naive_corr = mean_pairwise_corr(crisis_ret, all_sectors)
        naive_cum  = portfolio_cumret(crisis_ret, all_sectors)

        results.append({
            "Crisis":           label,
            "Res_Return":       res_ret,
            "Naive_Return":     naive_ret,
            "Res_MaxDD":        res_mdd,
            "Naive_MaxDD":      naive_mdd,
            "Res_Corr":         res_corr,
            "Naive_Corr":       naive_corr,
            "Return_Advantage": res_ret - naive_ret,
            "MDD_Advantage":    naive_mdd - res_mdd,
            "Corr_Advantage":   naive_corr - res_corr,
        })
        crisis_cumrets[label] = {"resilience": res_cum, "naive": naive_cum}

        print(f"\n  {label}:")
        print(f"    Resilience — return: {res_ret*100:.1f}%  "
              f"MDD: {res_mdd*100:.1f}%  corr: {res_corr:.3f}")
        print(f"    Naive      — return: {naive_ret*100:.1f}%  "
              f"MDD: {naive_mdd*100:.1f}%  corr: {naive_corr:.3f}")
        print(f"    Advantage  — return: {(res_ret-naive_ret)*100:+.1f}pp  "
              f"MDD: {(naive_mdd-res_mdd)*100:+.1f}pp  "
              f"corr: {(naive_corr-res_corr):+.3f}")

    return pd.DataFrame(results), crisis_cumrets

# ─────────────────────────────────────────────
# 3. PLOTTING FUNCTIONS
# ─────────────────────────────────────────────

def plot_cumrets(crisis_cumrets, title, path):
    n    = len(crisis_cumrets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (label, series) in zip(axes, crisis_cumrets.items()):
        ax.plot(series["resilience"].values * 100,
                color="#2ecc71", linewidth=2.0, label=f"Resilience (top {K})")
        ax.plot(series["naive"].values * 100,
                color="#e74c3c", linewidth=2.0, linestyle="--",
                label="Naive (all 10)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Trading days")
        ax.set_ylabel("Cumulative log-return (%)")
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved -> {path}")

def plot_summary(df, title, path):
    crises_labels = df["Crisis"].tolist()
    x     = np.arange(len(crises_labels))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, col_res, col_naive, ttl in zip(
        axes,
        ["Res_Return", "Res_MaxDD", "Res_Corr"],
        ["Naive_Return", "Naive_MaxDD", "Naive_Corr"],
        ["Total Return (%)",
         "Maximum Drawdown (%)\n(less negative = better)",
         "Mean Pairwise Correlation\nduring Crisis"],
    ):
        scale = 100 if "Return" in col_res or "MaxDD" in col_res else 1
        ax.bar(x - width/2, df[col_res]   * scale, width,
               label=f"Resilience (top {K})", color="#2ecc71", alpha=0.85)
        ax.bar(x + width/2, df[col_naive] * scale, width,
               label="Naive (all 10)",         color="#e74c3c", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(ttl)
        ax.set_xticks(x)
        ax.set_xticklabels(crises_labels, rotation=15, ha="right", fontsize=8)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved -> {path}")

# ─────────────────────────────────────────────
# 4. MAJOR CRISES
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("MAJOR CRISES BACKTEST")
print("="*50)
df_major, cumrets_major = run_backtest(MAJOR_CRISES, log_ret_main)

plot_cumrets(
    cumrets_major,
    f"Cumulative Returns: Resilience Portfolio (top {K}) vs. Naive\nMajor Crisis Periods",
    os.path.join(OUTPUT_DIR, "15_backtest_returns.png")
)
plot_summary(
    df_major,
    f"Resilience Portfolio (top {K}) vs. Naive — Major Crisis Summary",
    os.path.join(OUTPUT_DIR, "16_backtest_summary.png")
)
df_major.to_csv(os.path.join("output", "backtest_results.csv"), index=False)

# ─────────────────────────────────────────────
# 5. MINOR CRISES
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("MINOR CRISES BACKTEST")
print("="*50)
df_mini, cumrets_mini = run_backtest(MINI_CRISES, log_ret_mini)

plot_cumrets(
    cumrets_mini,
    f"Cumulative Returns: Resilience Portfolio (top {K}) vs. Naive\nMinor Crisis Periods",
    os.path.join(OUTPUT_DIR, "17_backtest_mini_returns.png")
)
plot_summary(
    df_mini,
    f"Resilience Portfolio (top {K}) vs. Naive — Minor Crisis Summary",
    os.path.join(OUTPUT_DIR, "18_backtest_mini_summary.png")
)
df_mini.to_csv(os.path.join("output", "backtest_mini_results.csv"), index=False)

print("\n=== Backtest complete ===")