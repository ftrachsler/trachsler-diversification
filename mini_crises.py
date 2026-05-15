"""
Trachsler Diversification Study — Mini Crisis Analysis
=======================================================
Focused analysis of smaller, shorter crisis episodes.
Compares sector vs. international correlation breakdown
for each event individually.

Crises covered:
  1. Euro Debt Crisis       (May 2010)
  2. Taper Tantrum          (May–Jun 2013)
  3. Flash Crash            (Aug 2015, China devaluation)
  4. Repo Market Crisis     (Sep 2019)
  5. Trump Tariff Shock     (Apr 2025)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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
os.makedirs(DATA_DIR,   exist_ok=True)

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
# MINI CRISES — each has a pre/during/post window
# ─────────────────────────────────────────────
# Format: (name, crisis_start, crisis_end, color)
# Pre-window = 60 trading days before crisis_start
# Post-window = 60 trading days after crisis_end
MINI_CRISES = [
    ("Euro Debt Crisis",    "2010-04-15", "2010-07-01",  "#7c3aed"),
    ("Taper Tantrum",       "2013-05-22", "2013-08-30",  "#0891b2"),
    ("Flash Crash",         "2015-08-10", "2015-09-30",  "#dc2626"),
    ("Repo Market Crisis",  "2019-09-16", "2019-10-15",  "#d97706"),
    ("Trump Tariff Shock",  "2025-04-02", "2025-05-15",  "#16a34a"),
]

# Download window: start well before first crisis, end after last
DOWNLOAD_START = "2009-01-01"
DOWNLOAD_END   = "2025-12-31"

# ─────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────
SECTORS = {
    "Technology":    "XLK",
    "Financials":    "XLF",
    "Healthcare":    "XLV",
    "Energy":        "XLE",
    "Industrials":   "XLI",
    "Consumer Disc": "XLY",
    "Consumer Stap": "XLP",
    "Utilities":     "XLU",
    "Materials":     "XLB",
    "Real Estate":   "IYR",
}

INTERNATIONAL = {
    "USA (S&P500)":  "SPY",
    "Europe":        "IEV",
    "Japan":         "EWJ",
    "Emerging Mkts": "EEM",
    "UK":            "EWU",
    "Germany":       "EWG",
    "Brazil":        "EWZ",
    "Pacific ex-JP": "EPP",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download(tickers: dict, start: str, end: str, cache: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{cache}.parquet")
    if os.path.exists(path):
        print(f"  [cache] {cache}")
        return pd.read_parquet(path)
    print(f"  [download] {cache} ...")
    raw = yf.download(list(tickers.values()), start=start,
                      end=end, auto_adjust=True, progress=False)["Close"]
    raw.columns = list(tickers.keys())
    raw = raw.dropna(how="all")
    raw.to_parquet(path)
    return raw


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def mean_pairwise_corr(ret: pd.DataFrame, min_obs: int = 10) -> float:
    """Mean upper-triangle correlation, ignoring NaN columns."""
    valid = ret.columns[ret.notna().sum() >= min_obs]
    sub = ret[valid].dropna()
    if len(sub) < min_obs or len(valid) < 2:
        return np.nan
    c = sub.corr().values
    return float(np.nanmean(c[np.triu_indices(len(valid), k=1)]))


def window_returns(ret: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return ret.loc[start:end]


def get_pre_crisis_normal(ret: pd.DataFrame, crisis_start: str,
                          pre_days: int = 60) -> pd.DataFrame:
    """60 trading days before the crisis as 'normal' baseline."""
    cs = pd.Timestamp(crisis_start)
    idx = ret.index.searchsorted(cs)
    start_idx = max(0, idx - pre_days)
    return ret.iloc[start_idx:idx]


# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def compute_crisis_table(ret_sec: pd.DataFrame,
                          ret_intl: pd.DataFrame) -> pd.DataFrame:
    """
    For each mini crisis compute:
      rho_pre_sector, rho_crisis_sector, delta_sector
      rho_pre_intl,   rho_crisis_intl,   delta_intl
    Using 60-day pre-crisis window as the 'normal' baseline.
    """
    rows = []
    for name, cs, ce, color in MINI_CRISES:
        pre_sec  = get_pre_crisis_normal(ret_sec,  cs)
        pre_intl = get_pre_crisis_normal(ret_intl, cs)
        cr_sec   = window_returns(ret_sec,  cs, ce)
        cr_intl  = window_returns(ret_intl, cs, ce)

        rho_pre_s  = mean_pairwise_corr(pre_sec)
        rho_cr_s   = mean_pairwise_corr(cr_sec)
        rho_pre_i  = mean_pairwise_corr(pre_intl)
        rho_cr_i   = mean_pairwise_corr(cr_intl)

        rows.append({
            "Crisis":         name,
            "Period":         f"{cs[:7]} – {ce[:7]}",
            "Color":          color,
            # Sector
            "rho_pre_sector": round(rho_pre_s,  4),
            "rho_cr_sector":  round(rho_cr_s,   4),
            "delta_sector":   round(rho_cr_s  - rho_pre_s,  4),
            # International
            "rho_pre_intl":   round(rho_pre_i,  4),
            "rho_cr_intl":    round(rho_cr_i,   4),
            "delta_intl":     round(rho_cr_i  - rho_pre_i,  4),
        })
    return pd.DataFrame(rows)


def rolling_corr_around_crisis(ret: pd.DataFrame, crisis_start: str,
                                crisis_end: str, window: int = 20,
                                buffer_days: int = 60) -> pd.Series:
    """
    Rolling mean pairwise correlation in a tight window (20 days)
    around a crisis. Returns a Series indexed by date.
    """
    cs  = pd.Timestamp(crisis_start)
    ce  = pd.Timestamp(crisis_end)
    idx_s = ret.index.searchsorted(cs)
    idx_e = ret.index.searchsorted(ce)
    start_idx = max(0, idx_s - buffer_days)
    end_idx   = min(len(ret), idx_e + buffer_days)
    sub = ret.iloc[start_idx:end_idx]

    n = len(sub.columns)
    results = []
    for i in range(window, len(sub)):
        w = sub.iloc[i - window: i]
        valid = w.columns[w.notna().sum() >= window // 2]
        w2 = w[valid].dropna()
        if len(w2) < window // 2 or len(valid) < 2:
            results.append((sub.index[i], np.nan))
            continue
        c = w2.corr().values
        results.append((sub.index[i], np.nanmean(c[np.triu_indices(len(valid), k=1)])))

    idx, vals = zip(*results)
    return pd.Series(vals, index=idx)


# ─────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────

def plot_delta_comparison(df: pd.DataFrame):
    """
    Main overview: grouped bar chart comparing
    delta_sector vs delta_intl for each crisis.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(df))
    w = 0.35

    bars_s = ax.bar(x - w/2, df["delta_sector"], width=w,
                    color=[c for c in df["Color"]],
                    label="Sector Δρ", zorder=3, alpha=0.9)
    bars_i = ax.bar(x + w/2, df["delta_intl"], width=w,
                    color=[c for c in df["Color"]],
                    label="International Δρ", zorder=3, alpha=0.45,
                    edgecolor=[c for c in df["Color"]], linewidth=1.5)

    # Value labels
    for bar in bars_s:
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.005 if v >= 0 else -0.012),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold")

    for bar in bars_i:
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.005 if v >= 0 else -0.012),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold", color="#374151")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Crisis"], fontsize=9)
    ax.set_ylabel("Δρ  (crisis − pre-crisis baseline)")
    ax.set_title("Mini-Crisis Correlation Breakdown: Sector vs. International",
                 fontweight="bold")
    ax.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)

    # Legend
    solid = mpatches.Patch(color="gray", alpha=0.9, label="Sector Δρ (solid)")
    faded = mpatches.Patch(facecolor="gray", alpha=0.45,
                           edgecolor="gray", label="International Δρ (faded)")
    ax.legend(handles=[solid, faded], fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_mini_crisis_delta_comparison.png")
    fig.savefig(path)
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_rolling_per_crisis(ret_sec: pd.DataFrame,
                             ret_intl: pd.DataFrame):
    """
    5 subplots (one per crisis): 20-day rolling correlation
    for sector (blue) and international (orange) around each event.
    Crisis window shaded in red.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    fig.suptitle(
        "20-Day Rolling Correlation Around Each Mini-Crisis\n"
        "(Sector = blue, International = orange, crisis window = shaded)",
        fontweight="bold", fontsize=12
    )

    for ax, (name, cs, ce, color) in zip(axes, MINI_CRISES):
        roll_s = rolling_corr_around_crisis(ret_sec,  cs, ce, window=20)
        roll_i = rolling_corr_around_crisis(ret_intl, cs, ce, window=20)

        ax.plot(roll_s.index, roll_s.values, color="#2563eb", lw=1.5,
                label="Sector")
        ax.plot(roll_i.index, roll_i.values, color="#f97316", lw=1.5,
                ls="--", label="Intl")
        ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                   alpha=0.2, color=color)
        ax.axvline(pd.Timestamp(cs), color=color, lw=0.8, ls=":")
        ax.axvline(pd.Timestamp(ce), color=color, lw=0.8, ls=":")

        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Mean ρ (20-day rolling)")
            ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_mini_crisis_rolling_per_event.png")
    fig.savefig(path)
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_summary_table(df: pd.DataFrame):
    """
    Clean visual table showing all numbers side by side.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    col_labels = ["Crisis", "Period",
                  "ρ pre\n(Sector)", "ρ crisis\n(Sector)", "Δρ Sector",
                  "ρ pre\n(Intl)", "ρ crisis\n(Intl)", "Δρ Intl"]

    table_data = []
    for _, row in df.iterrows():
        def fmt(v):
            if pd.isna(v): return "—"
            return f"{v:+.3f}" if "delta" in str(v) else f"{v:.3f}"

        table_data.append([
            row["Crisis"], row["Period"],
            f"{row['rho_pre_sector']:.3f}",
            f"{row['rho_cr_sector']:.3f}",
            f"{row['delta_sector']:+.3f}",
            f"{row['rho_pre_intl']:.3f}" if not pd.isna(row['rho_pre_intl']) else "—",
            f"{row['rho_cr_intl']:.3f}"  if not pd.isna(row['rho_cr_intl'])  else "—",
            f"{row['delta_intl']:+.3f}"  if not pd.isna(row['delta_intl'])   else "—",
        ])

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Color header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1e3a5f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Color delta cells: red if positive (correlation rose), green if negative
    delta_cols = [4, 7]  # Δρ Sector, Δρ Intl
    for i, row in enumerate(table_data):
        for j in delta_cols:
            val_str = row[j]
            if val_str == "—": continue
            val = float(val_str)
            tbl[i+1, j].set_facecolor("#fecaca" if val > 0.05 else
                                       "#dcfce7" if val < 0 else "#fef9c3")

    # Alternate row colors
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            if j not in delta_cols:
                tbl[i+1, j].set_facecolor("#f8fafc" if i % 2 == 0 else "white")

    ax.set_title("Mini-Crisis Summary: Correlation Levels Before and During Each Event",
                 fontweight="bold", pad=15, fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_mini_crisis_summary_table.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  TRACHSLER — MINI CRISIS ANALYSIS")
    print("="*55 + "\n")

    # ── 1. Data ───────────────────────────────────────────────
    print("[1/4] Loading data ...")
    sec_prices  = download(SECTORS,       DOWNLOAD_START, DOWNLOAD_END, "sectors_mini")
    intl_prices = download(INTERNATIONAL, DOWNLOAD_START, DOWNLOAD_END, "international_mini")

    sec_ret  = log_returns(sec_prices)
    intl_ret = log_returns(intl_prices)

    print(f"      Sectors:       {sec_prices.shape[1]} ETFs  "
          f"({sec_prices.index[0].date()} – {sec_prices.index[-1].date()})")
    print(f"      International: {intl_prices.shape[1]} indices\n")

    # ── 2. Compute delta table ────────────────────────────────
    print("[2/4] Computing Δρ for each mini crisis ...")
    df = compute_crisis_table(sec_ret, intl_ret)

    print("\n  Results:")
    print(df[["Crisis", "Period", "delta_sector", "delta_intl"]].to_string(index=False))
    print()

    # Save CSV
    df.drop(columns=["Color"]).to_csv(
        os.path.join("output", "mini_crisis_stats.csv"), index=False)

    # ── 3. Plots ──────────────────────────────────────────────
    print("[3/4] Generating plots ...")
    plot_delta_comparison(df)
    plot_rolling_per_crisis(sec_ret, intl_ret)
    plot_summary_table(df)

    # ── 4. Print interpretation ───────────────────────────────
    print("\n[4/4] Key findings:")
    for _, row in df.iterrows():
        ds = row["delta_sector"]
        di = row["delta_intl"]
        sec_dir  = "ROSE" if ds > 0.02 else ("FELL" if ds < -0.02 else "stable")
        intl_dir = "ROSE" if di > 0.02 else ("FELL" if di < -0.02 else "stable") if not pd.isna(di) else "N/A"
        print(f"  {row['Crisis']:<25} Sector Δρ={ds:+.3f} ({sec_dir})"
              f"   Intl Δρ={di:+.3f} ({intl_dir})" if not pd.isna(di)
              else f"  {row['Crisis']:<25} Sector Δρ={ds:+.3f} ({sec_dir})   Intl Δρ=N/A")

    print("\n" + "="*55)
    print("  DONE — plots 08, 09, 10 saved to /plots/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
