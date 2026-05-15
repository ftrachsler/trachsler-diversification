"""
Trachsler Diversification Study
================================
Empirical analysis of diversification effectiveness across sectors
and international markets, with focus on crisis periods.

Paper: "Crisis-Robust Sector Selection: A Stress-Conditional
        Diversification Model for Equity Portfolios"
Author: Fabio Trachsler
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
os.makedirs(DATA_DIR,   exist_ok=True)

# Plot style
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
# 1. SECTOR ETFs  (US, SPDR Sector ETFs)
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
    "Real Estate":   "IYR",   # iShares US Real Estate, available since 2000
}

# ─────────────────────────────────────────────
# 2. INTERNATIONAL INDICES
# ─────────────────────────────────────────────
INTERNATIONAL = {
    "USA (S&P500)":   "SPY",   # since 1993
    "Europe":         "IEV",   # iShares Europe, since 2001 (replaces VGK which starts 2005)
    "Japan":          "EWJ",   # since 1996
    "Emerging Mkts":  "EEM",   # since 2003
    "UK":             "EWU",   # since 1996
    "Germany":        "EWG",   # since 1996
    "Brazil":         "EWZ",   # since 2000
    "Pacific ex-JP":  "EPP",   # since 2001 (replaces MCHI which starts 2011)
}

# ─────────────────────────────────────────────
# 3. CRISIS PERIODS  (start, end, label)
# ─────────────────────────────────────────────
CRISES = [
    ("2000-03-01", "2002-10-31", "Dotcom\nBust"),
    ("2008-09-01", "2009-03-31", "Global\nFinancial\nCrisis"),
    ("2020-02-15", "2020-04-30", "COVID\nCrash"),
    ("2022-01-01", "2022-12-31", "Rate\nHike\nShock"),
]

START_DATE = "1999-01-01"
END_DATE   = "2025-12-31"


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def download_data(tickers: dict, start: str, end: str, cache_name: str) -> pd.DataFrame:
    """Download adjusted close prices, use local cache if available."""
    cache_path = os.path.join(DATA_DIR, f"{cache_name}.parquet")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading {cache_name} from disk …")
        return pd.read_parquet(cache_path)

    print(f"  [download] Fetching {cache_name} …")
    raw = yf.download(
        list(tickers.values()),
        start=start, end=end,
        auto_adjust=True, progress=False
    )["Close"]
    raw.columns = list(tickers.keys())
    raw = raw.dropna(how="all")
    raw.to_parquet(cache_path)
    return raw


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def rolling_corr_mean(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """Mean pairwise correlation in a rolling window."""
    n = len(returns.columns)
    results = []
    for end_idx in range(window, len(returns)):
        window_data = returns.iloc[end_idx - window: end_idx]
        corr = window_data.corr().values
        upper = corr[np.triu_indices(n, k=1)]
        results.append((returns.index[end_idx], upper.mean()))
    idx, vals = zip(*results)
    return pd.Series(vals, index=idx)


def crisis_vs_normal(returns: pd.DataFrame, crises: list):
    """
    For each crisis: compute mean pairwise correlation inside
    vs. outside the crisis window.
    Per window, only columns with sufficient data are used so that
    early crises (Dotcom, GFC) work even if some ETFs did not exist yet.
    """
    def mean_corr(sub: pd.DataFrame, min_obs: int = 20) -> float:
        valid_cols = sub.columns[sub.notna().sum() >= min_obs]
        sub2 = sub[valid_cols].dropna()
        if len(sub2) < min_obs or len(valid_cols) < 2:
            return np.nan
        c = sub2.corr().values
        upper = c[np.triu_indices(len(valid_cols), k=1)]
        return float(np.nanmean(upper))

    all_crisis_mask = pd.Series(False, index=returns.index)
    for s, e, label in crises:
        all_crisis_mask |= (returns.index >= s) & (returns.index <= e)

    rho_normal = mean_corr(returns.loc[~all_crisis_mask])

    rows = []
    for s, e, label in crises:
        crisis_mask = (returns.index >= s) & (returns.index <= e)
        rho_crisis  = mean_corr(returns.loc[crisis_mask])
        delta       = (rho_crisis - rho_normal) if not np.isnan(rho_crisis) else np.nan
        rows.append({
            "Crisis":   label.replace("\n", " "),
            "rho_normal": round(rho_normal, 4),
            "rho_crisis": round(rho_crisis, 4) if not np.isnan(rho_crisis) else np.nan,
            "delta_rho":  round(delta, 4) if not np.isnan(delta) else np.nan,
        })
    return pd.DataFrame(rows)

def plot_rolling_correlation(series: pd.Series, crises: list,
                              title: str, filename: str):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series.index, series.values, color="#2563eb", lw=1.5,
            label="Mean pairwise correlation (252-day rolling)")

    colors = ["#fee2e2", "#fef9c3", "#dcfce7", "#ede9fe"]
    for (s, e, label), col in zip(crises, colors):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.45, color=col, label=label.replace("\n", " "))
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ax.text(mid, series.max() * 0.97, label,
                ha="center", va="top", fontsize=7.5, color="#374151")

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Mean Pairwise Correlation ρ")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, 1)
    ax.axhline(series.mean(), color="gray", lw=0.8, ls="--",
               label=f"Overall mean ρ = {series.mean():.2f}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_correlation_heatmap(returns: pd.DataFrame, title: str, filename: str):
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn_r", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_crisis_delta(df: pd.DataFrame, title: str, filename: str):
    # Always show crises in chronological order
    crisis_order = ["Dotcom Bust", "Global Financial Crisis", "COVID Crash", "Rate Hike Shock"]
    df = df.set_index("Crisis").reindex(crisis_order).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))

    x_positions = np.arange(len(df))
    labels = df["Crisis"].tolist()

    for i, (_, row) in enumerate(df.iterrows()):
        val = row["delta_rho"]
        if pd.isna(val):
            # Grey hatched bar for missing data
            ax.bar(i, 0.008, color="#e5e7eb", width=0.5, zorder=3, edgecolor="#9ca3af")
            ax.text(i, 0.012, "No data", ha="center", va="bottom",
                    fontsize=8, color="#6b7280", style="italic")
        else:
            color = "#22c55e" if val < 0 else "#ef4444"
            ax.bar(i, val, color=color, width=0.5, zorder=3)
            label = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
            if val >= 0:
                ax.text(i, val + 0.005, label,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            else:
                ax.text(i, val - 0.008, label,
                        ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δρ  (crisis − normal)")
    ax.set_title(title, fontweight="bold")

    valid_vals = df["delta_rho"].dropna()
    ymin = min(valid_vals.min() - 0.05, -0.05) if len(valid_vals) else -0.05
    ymax = valid_vals.max() + 0.08 if len(valid_vals) else 0.1
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", lw=0.5, alpha=0.5, zorder=0)

    # Annotation for Dotcom if negative
    dotcom_row = df[df["Crisis"] == "Dotcom Bust"]
    if not dotcom_row.empty and not pd.isna(dotcom_row["delta_rho"].values[0]):
        dotcom_val = dotcom_row["delta_rho"].values[0]
        dotcom_idx = list(df["Crisis"]).index("Dotcom Bust")
        if dotcom_val < 0:
            ax.annotate(
                "Tech crashed alone -\ndiversification worked",
                xy=(dotcom_idx, dotcom_val),
                xytext=(dotcom_idx + 0.7, dotcom_val - 0.03),
                fontsize=7.5, color="#166534", style="italic",
                arrowprops=dict(arrowstyle="->", color="#166534", lw=0.8)
            )

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_decade_heatmaps(returns: pd.DataFrame, filename: str):
    """Side-by-side correlation heatmaps per decade."""
    decades = {
        "2001–2010": ("2001-01-01", "2010-12-31"),
        "2011–2020": ("2011-01-01", "2020-12-31"),
        "2021–2025": ("2021-01-01", "2025-12-31"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, (s, e)) in zip(axes, decades.items()):
        sub = returns.loc[s:e]
        corr = sub.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    vmin=0, vmax=1, linewidths=0.4, ax=ax,
                    cbar=False, annot_kws={"size": 7})
        ax.set_title(label, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.suptitle("Sector Correlation by Decade — Are markets more correlated today?",
                 fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  TRACHSLER DIVERSIFICATION STUDY — starting analysis")
    print("="*55 + "\n")

    # ── 1. Download data ──────────────────────────────────────
    print("[1/5] Downloading / loading price data …")
    sector_prices = download_data(SECTORS,        START_DATE, END_DATE, "sectors")
    intl_prices   = download_data(INTERNATIONAL,  START_DATE, END_DATE, "international")

    sector_ret = log_returns(sector_prices)
    intl_ret   = log_returns(intl_prices)

    print(f"      Sectors:       {sector_prices.shape[1]} ETFs, "
          f"{len(sector_prices)} days ({sector_prices.index[0].date()} – "
          f"{sector_prices.index[-1].date()})")
    print(f"      International: {intl_prices.shape[1]} indices\n")

    # ── 2. Rolling correlation ────────────────────────────────
    print("[2/5] Computing rolling correlations …")
    sector_roll = rolling_corr_mean(sector_ret, window=252)
    intl_roll   = rolling_corr_mean(intl_ret,   window=252)

    plot_rolling_correlation(
        sector_roll, CRISES,
        "Mean Pairwise Sector Correlation (252-day rolling)",
        "01_sector_rolling_corr.png"
    )
    plot_rolling_correlation(
        intl_roll, CRISES,
        "Mean Pairwise International Correlation (252-day rolling)",
        "02_intl_rolling_corr.png"
    )

    # ── 3. Decade heatmaps ───────────────────────────────────
    print("[3/5] Building decade heatmaps …")
    plot_decade_heatmaps(sector_ret, "03_sector_decade_heatmaps.png")

    # Full-period heatmaps
    plot_correlation_heatmap(sector_ret, "Sector Correlations (Full Period)",
                             "04_sector_heatmap_full.png")
    plot_correlation_heatmap(intl_ret,   "International Correlations (Full Period)",
                             "05_intl_heatmap_full.png")

    # ── 4. Crisis vs. Normal analysis ────────────────────────
    print("[4/5] Crisis vs. normal correlation analysis …")
    sector_crisis_df = crisis_vs_normal(sector_ret, CRISES)
    intl_crisis_df   = crisis_vs_normal(intl_ret,   CRISES)

    print("\n  SECTOR Δρ (crisis minus normal):")
    print(sector_crisis_df.to_string(index=False))
    print("\n  INTERNATIONAL Δρ (crisis minus normal):")
    print(intl_crisis_df.to_string(index=False))

    plot_crisis_delta(sector_crisis_df,
                      "Sector Correlation Spike in Crisis Periods (Δρ)",
                      "06_sector_crisis_delta.png")
    plot_crisis_delta(intl_crisis_df,
                      "International Correlation Spike in Crisis Periods (Δρ)",
                      "07_intl_crisis_delta.png")

    # ── 5. Summary statistics ─────────────────────────────────
    print("\n[5/5] Saving summary statistics …")
    def avg_corr(df):
        c = df.dropna().corr().values
        n = len(df.dropna().columns)
        return np.nanmean(c[np.triu_indices(n, k=1)])

    summary = pd.DataFrame({
        "Metric": [
            "Avg sector rho (full period)",
            "Avg sector rho (2001-2010)",
            "Avg sector rho (2011-2020)",
            "Avg sector rho (2021-2025)",
            "Avg intl rho (full period)",
            "Avg intl rho (2003-2010)",
            "Avg intl rho (2011-2020)",
            "Avg intl rho (2021-2025)",
        ],
        "Value": [
            avg_corr(sector_ret),
            avg_corr(sector_ret.loc["2001":"2010"]),
            avg_corr(sector_ret.loc["2011":"2020"]),
            avg_corr(sector_ret.loc["2021":"2025"]),
            avg_corr(intl_ret),
            avg_corr(intl_ret.loc["2003":"2010"]),  # VGK/MCHI not available before 2003
            avg_corr(intl_ret.loc["2011":"2020"]),
            avg_corr(intl_ret.loc["2021":"2025"]),
        ]
    })
    summary["Value"] = summary["Value"].round(4)
    summary.to_csv(os.path.join("output", "summary_stats.csv"), index=False)
    print(summary.to_string(index=False))

    print("\n" + "="*55)
    print("  DONE — all plots saved to /plots/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()