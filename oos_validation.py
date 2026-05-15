"""
Trachsler Diversification Study
================================
Out-of-Sample Validation — Chapter 6
Tests whether the Resilience Portfolio (top-5 sectors by Trachsler
Resilience Score) outperforms the naive equal-weight portfolio across
stress events that were NOT used to compute the score.

Method:
  1. Take the 59 stress events from stress_scan.py
  2. Remove any event that overlaps with the 9 pre-defined crisis periods
  3. Use the remaining events as a clean out-of-sample test set
  4. For each OOS event: compute portfolio returns and drawdown
  5. Run paired t-tests to assess statistical significance

Produces:
  19_oos_delta_returns.png    — distribution of return differences (OOS)
  20_oos_ttest_summary.png    — t-test results summary
  output/oos_results.csv

Author: Fabio Trachsler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
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

K             = 5
WINDOW        = 7     # stress event window (days)
THRESHOLD     = 0.10  # 10% loss threshold
MIN_GAP       = 30    # min days between stress events
CORR_WINDOW   = 60    # baseline window

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
# ALL PREDEFINED CRISIS PERIODS
# (major + minor — used to define in-sample)
# ─────────────────────────────────────────────

ALL_CRISIS_PERIODS = [
    ("2000-03-01", "2002-10-31"),  # Dotcom Bust
    ("2008-09-01", "2009-03-31"),  # GFC
    ("2020-02-15", "2020-04-30"),  # COVID
    ("2022-01-01", "2022-12-31"),  # Rate Hike
    ("2010-04-01", "2010-07-31"),  # Euro Debt
    ("2013-05-01", "2013-08-31"),  # Taper Tantrum
    ("2015-08-01", "2015-09-30"),  # Flash Crash
    ("2019-09-01", "2019-10-31"),  # Repo Market
    ("2025-04-01", "2025-05-31"),  # Trump Tariffs
]

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

print("Loading data ...")
sectors  = pd.read_parquet(os.path.join(DATA_DIR, "sectors_mini.parquet"))
log_ret  = np.log(sectors / sectors.shift(1)).dropna()
all_sectors = list(log_ret.columns)

print(f"  Sample: {log_ret.index[0].date()} to {log_ret.index[-1].date()}")
print(f"  Top-{K} sectors: {TOP_K_SECTORS}")

# ─────────────────────────────────────────────
# 2. BUILD IN-SAMPLE MASK
# ─────────────────────────────────────────────

in_sample_mask = pd.Series(False, index=log_ret.index)
for start, end in ALL_CRISIS_PERIODS:
    in_sample_mask.loc[start:end] = True

print(f"\n  In-sample days: {in_sample_mask.sum()}")
print(f"  Out-of-sample days available: {(~in_sample_mask).sum()}")

# ─────────────────────────────────────────────
# 3. IDENTIFY STRESS EVENTS (same as stress_scan.py)
# ─────────────────────────────────────────────

print("\nScanning for stress events ...")
cum_ret      = np.log(sectors / sectors.shift(1)).rolling(WINDOW).sum()
stress_flag  = (cum_ret <= -THRESHOLD).any(axis=1)
stress_dates = stress_flag[stress_flag].index.tolist()

# De-duplicate
all_events = []
last_event = pd.Timestamp("1900-01-01")
for date in stress_dates:
    if (date - last_event).days >= MIN_GAP:
        all_events.append(date)
        last_event = date

print(f"  Total stress events found: {len(all_events)}")

# ─────────────────────────────────────────────
# 4. FILTER TO PURE OUT-OF-SAMPLE EVENTS
# ─────────────────────────────────────────────

def is_in_sample(date):
    """Check if a date falls within any predefined crisis window."""
    for start, end in ALL_CRISIS_PERIODS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return True
    return False

oos_events = [d for d in all_events if not is_in_sample(d)]
print(f"  Out-of-sample stress events (after removing crisis overlaps): "
      f"{len(oos_events)}")

# ─────────────────────────────────────────────
# 5. COMPUTE PORTFOLIO METRICS FOR EACH OOS EVENT
# ─────────────────────────────────────────────

print("\nComputing portfolio metrics for each OOS event ...")

def portfolio_return(log_ret_df, sectors, start, window):
    """Total return of equal-weight portfolio over event window."""
    end_loc = log_ret_df.index.get_loc(start)
    end_idx = log_ret_df.index[min(end_loc + window - 1,
                                   len(log_ret_df) - 1)]
    window_ret = log_ret_df.loc[start:end_idx]
    return float(window_ret[sectors].mean(axis=1).sum())

def portfolio_mdd(log_ret_df, sectors, start, window):
    """Maximum drawdown of equal-weight portfolio over event window."""
    end_loc = log_ret_df.index.get_loc(start)
    end_idx = log_ret_df.index[min(end_loc + window - 1,
                                   len(log_ret_df) - 1)]
    window_ret  = log_ret_df.loc[start:end_idx]
    cum         = window_ret[sectors].mean(axis=1).cumsum()
    roll_max    = cum.cummax()
    return float((cum - roll_max).min())

results = []
for event_date in oos_events:
    try:
        res_ret   = portfolio_return(log_ret, TOP_K_SECTORS, event_date, WINDOW)
        naive_ret = portfolio_return(log_ret, all_sectors,   event_date, WINDOW)
        res_mdd   = portfolio_mdd(log_ret, TOP_K_SECTORS, event_date, WINDOW)
        naive_mdd = portfolio_mdd(log_ret, all_sectors,   event_date, WINDOW)

        results.append({
            "date":             event_date,
            "res_return":       res_ret,
            "naive_return":     naive_ret,
            "return_diff":      res_ret - naive_ret,
            "res_mdd":          res_mdd,
            "naive_mdd":        naive_mdd,
            "mdd_diff":         naive_mdd - res_mdd,  # positive = resilience better
        })
    except Exception as e:
        print(f"  Skipping {event_date}: {e}")
        continue

df_oos = pd.DataFrame(results)
print(f"  Successfully computed metrics for {len(df_oos)} OOS events.")

# ─────────────────────────────────────────────
# 6. SUMMARY STATISTICS
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("OUT-OF-SAMPLE RESULTS SUMMARY")
print("="*50)

mean_ret_diff = df_oos["return_diff"].mean()
mean_mdd_diff = df_oos["mdd_diff"].mean()
pct_ret_wins  = (df_oos["return_diff"] > 0).mean() * 100
pct_mdd_wins  = (df_oos["mdd_diff"]   > 0).mean() * 100

print(f"\n  Return advantage (Resilience minus Naive):")
print(f"    Mean:        {mean_ret_diff*100:+.2f}pp")
print(f"    Std:         {df_oos['return_diff'].std()*100:.2f}pp")
print(f"    Win rate:    {pct_ret_wins:.1f}% of events")

print(f"\n  Drawdown advantage (Naive MDD minus Resilience MDD):")
print(f"    Mean:        {mean_mdd_diff*100:+.2f}pp")
print(f"    Std:         {df_oos['mdd_diff'].std()*100:.2f}pp")
print(f"    Win rate:    {pct_mdd_wins:.1f}% of events")

# ─────────────────────────────────────────────
# 7. PAIRED T-TESTS
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("PAIRED T-TESTS")
print("="*50)

# H0: mean return difference = 0
t_ret, p_ret = stats.ttest_1samp(df_oos["return_diff"], 0)
# H0: mean MDD advantage = 0
t_mdd, p_mdd = stats.ttest_1samp(df_oos["mdd_diff"], 0)

print(f"\n  Return difference (H0: mean = 0):")
print(f"    t-statistic: {t_ret:.3f}")
print(f"    p-value:     {p_ret:.4f}")
print(f"    Significant at 5%: {'YES' if p_ret < 0.05 else 'NO'}")
print(f"    Significant at 10%: {'YES' if p_ret < 0.10 else 'NO'}")

print(f"\n  MDD advantage (H0: mean = 0):")
print(f"    t-statistic: {t_mdd:.3f}")
print(f"    p-value:     {p_mdd:.4f}")
print(f"    Significant at 5%: {'YES' if p_mdd < 0.05 else 'NO'}")
print(f"    Significant at 10%: {'YES' if p_mdd < 0.10 else 'NO'}")

# Also test: is mean sector delta-rho > 0 during stress events?
# (robustness check from stress_scan)
stress_csv = os.path.join("output", "stress_scan_results.csv")
if os.path.exists(stress_csv):
    df_scan = pd.read_csv(stress_csv)
    t_drho, p_drho = stats.ttest_1samp(df_scan["delta_rho"], 0)
    print(f"\n  Delta-rho > 0 during stress events (H0: mean = 0):")
    print(f"    n events:    {len(df_scan)}")
    print(f"    mean Δρ:     {df_scan['delta_rho'].mean():.4f}")
    print(f"    t-statistic: {t_drho:.3f}")
    print(f"    p-value:     {p_drho:.6f}")
    print(f"    Significant at 1%: {'YES' if p_drho < 0.01 else 'NO'}")

# ─────────────────────────────────────────────
# 8. PLOT 19 — RETURN DIFFERENCE DISTRIBUTION
# ─────────────────────────────────────────────

print("\nPlotting return difference distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Return difference histogram
ax = axes[0]
colors = ["#2ecc71" if v > 0 else "#e74c3c"
          for v in df_oos["return_diff"]]
n, bins, patches = ax.hist(df_oos["return_diff"] * 100, bins=20,
                            edgecolor="white", linewidth=0.5)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor("#2ecc71" if left > 0 else "#e74c3c")
    patch.set_alpha(0.8)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(mean_ret_diff * 100, color="navy", linewidth=1.5,
           linestyle=":", label=f"Mean = {mean_ret_diff*100:+.2f}pp")
ax.set_title(f"Return Advantage: Resilience minus Naive\n"
             f"({len(df_oos)} OOS stress events, "
             f"win rate = {pct_ret_wins:.0f}%)",
             fontsize=11)
ax.set_xlabel("Return difference (pp)")
ax.set_ylabel("Number of events")
ax.legend(fontsize=9, frameon=False)

green_patch = mpatches.Patch(color="#2ecc71", alpha=0.8,
                              label=f"Resilience wins ({(df_oos['return_diff']>0).sum()} events)")
red_patch   = mpatches.Patch(color="#e74c3c", alpha=0.8,
                              label=f"Naive wins ({(df_oos['return_diff']<=0).sum()} events)")
ax.legend(handles=[green_patch, red_patch,
                   plt.Line2D([0],[0], color="navy", linewidth=1.5,
                              linestyle=":")],
          labels=[green_patch.get_label(), red_patch.get_label(),
                  f"Mean = {mean_ret_diff*100:+.2f}pp  "
                  f"(p = {p_ret:.3f})"],
          fontsize=8, frameon=False)

# Right: MDD advantage histogram
ax = axes[1]
n, bins, patches = ax.hist(df_oos["mdd_diff"] * 100, bins=20,
                            edgecolor="white", linewidth=0.5)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor("#2ecc71" if left > 0 else "#e74c3c")
    patch.set_alpha(0.8)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(mean_mdd_diff * 100, color="navy", linewidth=1.5,
           linestyle=":", label=f"Mean = {mean_mdd_diff*100:+.2f}pp")
ax.set_title(f"Drawdown Advantage: Naive MDD minus Resilience MDD\n"
             f"({len(df_oos)} OOS stress events, "
             f"win rate = {pct_mdd_wins:.0f}%)",
             fontsize=11)
ax.set_xlabel("MDD advantage (pp, positive = resilience better)")
ax.set_ylabel("Number of events")

green_patch2 = mpatches.Patch(color="#2ecc71", alpha=0.8,
                               label=f"Resilience wins ({(df_oos['mdd_diff']>0).sum()} events)")
red_patch2   = mpatches.Patch(color="#e74c3c", alpha=0.8,
                               label=f"Naive wins ({(df_oos['mdd_diff']<=0).sum()} events)")
ax.legend(handles=[green_patch2, red_patch2,
                   plt.Line2D([0],[0], color="navy", linewidth=1.5,
                              linestyle=":")],
          labels=[green_patch2.get_label(), red_patch2.get_label(),
                  f"Mean = {mean_mdd_diff*100:+.2f}pp  "
                  f"(p = {p_mdd:.3f})"],
          fontsize=8, frameon=False)

plt.suptitle("Out-of-Sample Validation: Resilience Portfolio vs. Naive Equal-Weight\n"
             "(stress events excluded from score computation)",
             fontsize=11)
plt.tight_layout()
path19 = os.path.join(OUTPUT_DIR, "19_oos_validation.png")
plt.savefig(path19, bbox_inches="tight")
plt.close()
print(f"  [plot] saved -> {path19}")

# ─────────────────────────────────────────────
# 9. PLOT 20 — T-TEST SUMMARY
# ─────────────────────────────────────────────

print("Plotting t-test summary ...")

fig, ax = plt.subplots(figsize=(8, 5))

tests   = ["Return advantage\n(Resilience vs. Naive)",
           "Drawdown advantage\n(Resilience vs. Naive)"]
t_stats = [t_ret, t_mdd]
p_vals  = [p_ret, p_mdd]
colors  = ["#2ecc71" if p < 0.05 else
           "#f39c12" if p < 0.10 else "#e74c3c"
           for p in p_vals]

bars = ax.barh(tests, [abs(t) for t in t_stats],
               color=colors, alpha=0.85, edgecolor="white")

for bar, t, p in zip(bars, t_stats, p_vals):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "n.s."
    ax.text(bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"t = {t:.2f},  p = {p:.3f}  {sig}",
            va="center", ha="left", fontsize=10)

ax.set_xlabel("|t-statistic|")
ax.set_title("Paired t-Tests: Resilience vs. Naive Portfolio\n"
             f"Out-of-Sample ({len(df_oos)} stress events)\n"
             "* p<0.10   ** p<0.05   *** p<0.01",
             fontsize=11)
ax.set_xlim(0, max([abs(t) for t in t_stats]) * 1.6)

green_p = mpatches.Patch(color="#2ecc71", alpha=0.85, label="p < 0.05 (significant)")
orange_p= mpatches.Patch(color="#f39c12", alpha=0.85, label="p < 0.10 (marginal)")
red_p   = mpatches.Patch(color="#e74c3c", alpha=0.85, label="p >= 0.10 (not significant)")
ax.legend(handles=[green_p, orange_p, red_p], fontsize=9, frameon=False)

plt.tight_layout()
path20 = os.path.join(OUTPUT_DIR, "20_oos_ttest.png")
plt.savefig(path20, bbox_inches="tight")
plt.close()
print(f"  [plot] saved -> {path20}")

# ─────────────────────────────────────────────
# 10. SAVE CSV
# ─────────────────────────────────────────────

df_oos.to_csv(os.path.join("output", "oos_results.csv"), index=False)
print(f"\n  [csv] saved -> output/oos_results.csv")

print("\n=== Out-of-sample validation complete ===")
