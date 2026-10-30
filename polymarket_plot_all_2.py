"""
Polymarket EDA - plot the 16 kept tables

Reads plot_data/<prefix>__<table>.parquet (written by the last cell of 414_midterm_2.ipynb), draws one figure
per table from the SPECS registry below, and writes:
    figures/<prefix>__<table>.png     one PNG per chart (title, data window, chart)
    figures/guide.md                  every chart with its data window, grouped by window
    polymarket_eda_figures.pdf        every chart in one multi-page PDF

Title style: Title Case, with verbs, articles, conjunctions and prepositions in lowercase.
Every chart has a "Data:" line under its title that says which dates it covers. For the charts whose window is
decided while the notebook runs (calibration, daily overview) the
line is read from the table itself, so it always matches what was really plotted.

Needs only matplotlib, pandas, numpy and duckdb. A chart that fails is reported (with a traceback) and skipped.
Edit any chart in SPECS (title=, labels). Set ONLY = [...] to redo some.
"""
import glob
import math
import os
import re
import textwrap
import traceback

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

PLOT_DIR = "plot_data"
FIG_DIR = "figures"
PDF_PATH = "polymarket_eda_figures.pdf"
SHOW = True      # display each figure inline (notebooks); False = only write files
DPI = 150
ONLY = None      # e.g. ["catalog__volume_pareto", "cross__calibration_book"]

# Date windows shown under the chart titles. Keep in sync with the notebook constants
# (SAMPLE_START/SAMPLE_END, DAILY_START/DAILY_END, SAMPLE_DATE / SAMPLE_HOUR_OF_DAY, RES_START/RES_END).
CATALOG_ASOF = "Oct 27, 2026"                  # when the catalog was pulled
W_TRADES = "Oct 14 - Oct 20, 2026 (UTC)"       # trade window loaded locally in the notebook
W_DAILY = "Oct 1 - Oct 26, 2026 (UTC), one point per day"
W_HOUR_START = "order-book snapshot at 14:00 UTC on Oct 20, 2026"
MIN_MARKETS_PER_BUCKET = 20                    # calibration buckets with fewer markets are drawn faded

os.makedirs(FIG_DIR, exist_ok=True)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 100, "legend.fontsize": 9})

# ----------------------------------------------------------------------------
# Load every exported table
# ----------------------------------------------------------------------------
T: dict[str, pd.DataFrame] = {}
for f in sorted(glob.glob(f"{PLOT_DIR}/*.parquet")):
    key = os.path.splitext(os.path.basename(f))[0]
    T[key] = duckdb.sql(f"SELECT * FROM read_parquet('{f.replace(chr(92), '/')}')").df()
print(f"loaded {len(T)} tables from {PLOT_DIR}/")


def lab(v) -> str:
    """
    Drops the leading numeric sort prefix and replaces underscores with spaces (i.e. '3_6-24h' -> '6-24h', 'after_close' -> 'after close').
    
    Parameters:
        v (str): The input string to be formatted.

    Returns:
        str: The formatted string with the numeric sort prefix removed and underscores replaced by spaces.

    """

    return re.sub(r"^\d+_", "", str(v)).replace("_", " ")


def _plain_number(v, _pos=None) -> str:
    """
    Formats a number as a plain string with thousands separators for large numbers and scientific notation for small numbers.

    Parameters:
        v (float): The number to be formatted.
        _pos (int, optional): Position of the tick (used by matplotlib's FuncFormatter).

    Returns:
        str: The formatted number as a string.
    """

    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def plain_linear_ticks(ax):
    """
    Format the y-axis ticks with thousands separators for large numbers such that they are easier to read (i.e. thousands separators instead of '1e6' offsets).
       
    Parameters:
        ax (matplotlib.axes.Axes): The axes object to format.
       
    Returns:
        None
    """
    if ax.get_yscale() == "linear":
        lo, hi = ax.get_ylim()
        if max(abs(lo), abs(hi)) >= 10000:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))


def plain_log_ticks(ax):
    """
    Log-scaled axes show plain numbers (0.01, 1, 100) instead of powers of ten (10^-2, 10^0, 10^2).

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to format.

    Returns:
        None
    """
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_number))
        ax.yaxis.set_minor_formatter(NullFormatter())


def decorate(ax, s):
    """
    Decorates the axes with various plot settings specified in the `s` dictionary.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to decorate.
        s (dict): A dictionary containing plot settings such as 'logx', 'xlabel', 'ylabel', 'xlim', 'ylim', and 'hline'.

    Returns:
        None
    """

    if s.get("logx"):
        ax.set_xscale("log")
    if "xlabel" in s:
        ax.set_xlabel(s["xlabel"])
    if "ylabel" in s:
        ax.set_ylabel(s["ylabel"])
    if "xlim" in s:
        ax.set_xlim(*s["xlim"])
    if "ylim" in s:
        ax.set_ylim(*s["ylim"])
    if "hline" in s:
        ax.axhline(s["hline"], color="grey", ls="--", lw=1)


def new_fig(s, default=(9, 5)):
    """
    Creates a new figure with the specified size.

    Parameters:
        s (dict): A dictionary containing plot settings, including 'figsize'.
        default (tuple): The default figure size to use if 'figsize' is not specified in `s`.

    Returns:
        tuple: A tuple containing the new figure and axes objects.
    """

    return plt.subplots(figsize=s.get("figsize", default))


def as_list(v):
    """
    Ensures the input is a list.

    Parameters:
        v (any): The input value to convert to a list.

    Returns:
        list: A list containing the input value.
    """

    return v if isinstance(v, list) else [v]


def series_name(s, c):
    """
    Returns the series name for a given column `c` from the `s` dictionary.

    Parameters:
        s (dict): A dictionary containing 'series_labels'.
        c (str): The column name to get the series name for.

    Returns:
        str: The series name for the column `c`.
    """

    return s.get("series_labels", {}).get(c, c)


# ----------------------------------------------------------------------------
# Renderers (only the ones the 16 kept charts use)
# ----------------------------------------------------------------------------
def r_line(df, s):
    """
    Renders a line plot based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'x', 'y', 'xdiv', 'marker', and 'diag'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the line plot.
    """

    fig, ax = new_fig(s)
    x, ys = s["x"], as_list(s["y"])
    xdiv = s.get("xdiv", 1)
    d = df.sort_values(x)
    xv = d[x] if xdiv == 1 else d[x] / xdiv
    for y in ys:
        ax.plot(xv, d[y], label=series_name(s, y), marker=s.get("marker", "o"), ms=3, lw=1.5)
    if len(ys) > 1:
        ax.legend()
    if s.get("diag"):
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, label="everyone equal")
        ax.legend()
    decorate(ax, s)
    return fig


def r_hist(df, s):
    """
    Renders a pre-binned histogram based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'x', 'y', and 'log'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the histogram.
    """

    fig, ax = new_fig(s)
    d = df.dropna(subset=[s["x"]]).sort_values(s["x"])
    x = d[s["x"]].astype(float).to_numpy()
    base, step = s["log"]
    left = base ** x
    width = base ** (x + step) - left
    ax.set_xscale("log")
    ax.bar(left, d[s["y"]].astype(float).to_numpy(), width=width, align="edge", alpha=0.9,
           label=series_name(s, s["y"]), edgecolor="white", linewidth=0.3)
    decorate(ax, s)
    return fig


def r_bar(df, s):
    """
    Renders a bar chart based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'x', 'y', 'ys', 'hue', 'yerr', 'normalize', and 'annotate'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the bar chart.
    """

    fig, ax = new_fig(s)
    x = s["x"]
    d = df.dropna(subset=[x]).copy()
    if "ys" in s:
        piv = d.set_index(x)[s["ys"]]
    elif s.get("hue"):
        piv = d.pivot_table(index=x, columns=s["hue"], values=s["y"], aggfunc="sum")
    else:
        piv = d.groupby(x)[[s["y"]]].sum()
    piv = piv.sort_index()
    err = None
    if s.get("yerr") and s.get("hue"):
        sd_col, n_col = s["yerr"]
        sd = d.pivot_table(index=x, columns=s["hue"], values=sd_col, aggfunc="mean").sort_index()
        nn = d.pivot_table(index=x, columns=s["hue"], values=n_col, aggfunc="sum").sort_index()
        err = 1.96 * sd / np.sqrt(nn)          # 95% confidence interval of the average
    if s.get("normalize"):
        piv = piv / piv.sum()
    n = piv.shape[1]
    w = 0.8 / n
    xs = np.arange(len(piv))
    for i, c in enumerate(piv.columns):
        ax.bar(xs + (i - (n - 1) / 2) * w, piv[c].to_numpy(dtype=float), w,
               label=s.get("series_labels", {}).get(c, lab(c)),
               yerr=None if err is None else err[c].to_numpy(dtype=float), capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab(i) for i in piv.index])
    if n > 1:
        ax.legend(title=s.get("hue_name"))
    if s.get("annotate") and n == 1:
        counts = d.groupby(x)[s["annotate"]].sum().reindex(piv.index)
        for xi, (val, cnt) in enumerate(zip(piv.iloc[:, 0], counts)):
            ax.annotate(f"n={int(cnt):,}", (xi, val), ha="center", va="bottom", fontsize=8)
    if s.get("normalize"):
        ax.set_ylabel("Share of Total")
    decorate(ax, s)
    return fig


LABEL_FNS = {"title": lambda v: v if len(v) <= 70 else v[:67] + "..."}


def r_hbar(df, s):
    """
    Renders a horizontal bar chart based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'y', 'label', 'top', 'figsize', 'xlabel', and 'label_fn'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the horizontal bar chart.
    """

    d = df.sort_values(s["y"], ascending=False).head(s.get("top", 25))
    fn = LABEL_FNS.get(s.get("label_fn"), lambda v: v)
    labels = [fn(str(v)) for v in d[s["label"]]]
    fig, ax = plt.subplots(figsize=s.get("figsize", (10, 0.34 * len(d) + 2.0)))
    ax.barh(range(len(d)), d[s["y"]].astype(float).to_numpy())
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.set_xlabel(s.get("xlabel", s["y"]))
    return fig


def r_stacked(df, s):
    """
    Renders a stacked bar chart based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'index', 'columns', 'values', 'legend_title', 'rotate', and 'figsize'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the stacked bar chart.
    """
    piv = df.pivot_table(index=s["index"], columns=s["columns"], values=s["values"],
                         aggfunc="sum", fill_value=0).sort_index()
    piv.index = [lab(i) for i in piv.index]
    fig, ax = new_fig(s)
    piv.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.legend(title=s.get("legend_title", s["columns"]), bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=s.get("rotate", 45))
    ax.set_ylabel(s["values"])
    decorate(ax, s)
    return fig


def r_heatmap(df, s):
    """
    Renders a heatmap based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'index', 'columns', 'values', 'ylabels', 'colorbar_label', 'xlabel', and 'ylabel'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the heatmap.
    """
    piv = df.pivot_table(index=s["index"], columns=s["columns"], values=s["values"], aggfunc="sum").sort_index()
    fig, ax = new_fig(s, (10, 4.5))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ylabs = s.get("ylabels")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([ylabs[int(i)] for i in piv.index] if ylabs else list(piv.index))
    fig.colorbar(im, ax=ax, label=s.get("colorbar_label", s["values"]))
    ax.grid(False)
    if "xlabel" in s:
        ax.set_xlabel(s["xlabel"])
    if "ylabel" in s:
        ax.set_ylabel(s["ylabel"])
    return fig


def r_panels(df, s):
    """
    Renders a multi-panel line plot based on the provided DataFrame and plot settings.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'x', 'ys', 'ncols', 'panel_titles', 'datetime_x', and 'figsize'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the multi-panel line plot.
    """

    ys = s["ys"]
    ncols = s.get("ncols", 1)
    nrows = math.ceil(len(ys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                             figsize=s.get("figsize", (5 * ncols + (4 if ncols == 1 else 0), 2.7 * nrows + 1.2)))
    d = df.copy()
    if s.get("datetime_x"):
        d[s["x"]] = pd.to_datetime(d[s["x"]])
    d = d.sort_values(s["x"])
    for ax, y in zip(axes.ravel(), ys):
        ax.plot(d[s["x"]], d[y], marker="o", ms=3)
        ax.set_title(s.get("panel_titles", {}).get(y, y), fontsize=10)
    for ax in axes.ravel()[len(ys):]:
        ax.axis("off")
    fig.autofmt_xdate()
    return fig


def money_short(v) -> str:
    """
    Formats a number as a short monetary string with one significant digit.

    Parameters:
        v (float): The number to format.

    Returns:
        str: The formatted monetary string (e.g., '$1K', '$3K', '$300K', '$3M').
    """

    m = 10 ** math.floor(math.log10(v))
    d = round(v / m) * m
    if d >= 1e6:
        return f"\\${d / 1e6:g}M"      # backslash-escaped so two $ in one label are not read as math text
    if d >= 1e3:
        return f"\\${d / 1e3:g}K"
    return f"\\${d:g}"


def r_liquidity(df, s):
    """
    Renders a two-panel plot showing the cost of trading (gap) and the extra cost of a $100 / $1,000 purchase.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'figsize'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the two-panel plot.
    """

    d = df.sort_values("volume_lo").reset_index(drop=True)
    pos = np.arange(len(d))
    names = [f"{money_short(lo)}-{money_short(hi)}\n({int(n):,})" for lo, hi, n in
             zip(d["volume_lo"], d["volume_hi"], d["books"])]
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=s.get("figsize", (10, 7.6)))
    a1.plot(pos, d["median_gap_cents"], marker="o", color="C0", lw=2)
    a1.set_title("The Gap between the Price to buy and the Price to sell", fontsize=10.5)
    a1.set_ylabel("Cents per Share")
    a1.set_ylim(bottom=0)
    for col, label, colr in (("median_extra_100_cents", "Buying $100 Worth", "C1"),
                             ("median_extra_1000_cents", "Buying $1,000 Worth", "C3")):
        a2.plot(pos, d[col], marker="o", color=colr, lw=2, label=label)
    a2.set_title("Extra Cost per Share when buying YES Shares at the Best Price on Offer", fontsize=10.5)
    a2.set_ylabel("Extra Cents per Share")
    a2.set_ylim(bottom=0)
    # groups where most books are too thin to fill the order have no line: shade them instead of leaving a mystery gap
    handles, _ = a2.get_legend_handles_labels()
    for col, shade, text in (("median_extra_1000_cents", 0.10, "Most Books too thin for a $1,000 Order"),
                             ("median_extra_100_cents", 0.22, "Most Books too thin for a $100 Order")):
        gaps = [p for p, v in zip(pos, d[col]) if pd.isna(v)]
        if gaps:
            for p in gaps:
                a2.axvspan(p - 0.5, p + 0.5, color="grey", alpha=shade, lw=0)
            handles.append(Patch(color="grey", alpha=shade + 0.1, label=text))
    a2.legend(handles=handles, loc="upper right", fontsize=8.5)
    a2.set_xticks(pos)
    a2.set_xticklabels(names, fontsize=8.5)
    a2.set_xlabel("Total Trading Volume of the Market (Number of Markets in Brackets)")
    return fig


def r_calib_buckets(df, s):
    """
    Renders a calibration plot with one panel per time horizon and ten price buckets.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        s (dict): A dictionary containing plot settings such as 'figsize'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the calibration plot.

    Notes:
        - Bar represents how often YES actually won.
        - Diamond represents what the price said.
    """

    hs = sorted(df["horizon_h"].unique(), reverse=True)
    fig, axes = plt.subplots(len(hs), 1, sharex=True, figsize=s.get("figsize", (10, 3.3 * len(hs) + 1.0)))
    axes = np.atleast_1d(axes)
    edges = np.round(np.arange(0, 1.0, 0.1), 1)
    labels = [f"{int(round(e * 100))}-{int(round(e * 100)) + 10}%" for e in edges]
    for ax, h in zip(axes, hs):
        g = df[df["horizon_h"] == h].set_index("price_bucket_lo").reindex(edges)
        n = g["markets"].fillna(0).astype(int).to_numpy()
        win = (g["win_rate"].astype(float) * 100).to_numpy()
        said = (g["avg_price"].astype(float) * 100).to_numpy()
        few = (n < MIN_MARKETS_PER_BUCKET)
        x = np.arange(len(edges))
        ax.bar(x[~few], np.nan_to_num(win[~few]), color="C0", width=0.7)
        ax.bar(x[few], np.nan_to_num(win[few]), color="C0", alpha=0.3, width=0.7, hatch="//", edgecolor="white")
        ax.scatter(x[n > 0], said[n > 0], marker="D", s=46, color="black", zorder=3)
        for xi, ni, wi, si in zip(x, n, win, said):
            if ni > 0:
                ax.text(xi, min(max(wi, si) + 4, 108), f"{ni:,}", ha="center", va="bottom", fontsize=8.5)
            else:
                ax.text(xi, 3, "none", ha="center", va="bottom", fontsize=8, color="grey")
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("YES Won (%)")
        ax.set_title(f"{h} Hour{'' if h == 1 else 's'} before the Deadline", fontsize=10.5, loc="left")
    axes[-1].set_xticks(np.arange(len(edges)))
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("What the Market's Price said: the Chance of YES (Number of Markets above Each Bar)")
    axes[0].legend(handles=[Patch(color="C0", label="How often YES actually won"),
                            Line2D([], [], marker="D", ls="", color="black", label="What the price said"),
                            Patch(facecolor="C0", alpha=0.3, hatch="//", edgecolor="white",
                                  label=f"Fewer than {MIN_MARKETS_PER_BUCKET} Markets: too few to trust")],
                   loc="upper left", fontsize=8.5, ncol=3, bbox_to_anchor=(0, 1.0), frameon=True)
    return fig


RENDER = {"line": r_line, "hist": r_hist, "bar": r_bar, "hbar": r_hbar, "stacked": r_stacked,
          "heatmap": r_heatmap, "panels": r_panels, "liquidity": r_liquidity, "calib_buckets": r_calib_buckets}

# ----------------------------------------------------------------------------
# Registry: one entry per kept table (16). Titles and axis labels are in Title Case with verbs,
# articles, conjunctions and prepositions in lowercase. Legend entries and tick labels are
# left as they come. Edit freely.
# Vocabulary: a *market* is one yes/no question people trade on; a YES (or NO) *share* pays
# $1 if that answer turns out right, so its price between $0 and $1 is roughly the crowd's
# probability for it.
# ----------------------------------------------------------------------------
SPECS: dict[str, dict] = {}


def add(key, kind, title, **kw):
    SPECS[key] = {"kind": kind, "title": title, **kw}


# --- catalog ----------------------------------------------------------------
add("catalog__volume_pareto", "line", "Distribution of Trading Volume across Top X% Markets",
    x="market_percentile", y="cum_volume_share", xdiv=0.01, marker="", logx=True, ylim=(0, 1),
    xlabel="Top X% of Markets", ylabel="Share of Trading Volume")
add("catalog__yes_rate_by_neg_risk", "bar", "How Often the Answer turns out to be YES",
    x="neg_risk", y="yes_rate", annotate="resolved_markets", ylim=(0, 1),
    xlabel="Only One Answer can win?", ylabel="Share of Markets where YES won")
add("catalog__duration_hist", "hist", "How Long Markets are planned to run",
    x="log10_days_bin", y="markets", log=(10, 0.25),
    xlabel="Planned Lifespan in Days (Each Gridline = 10x)", ylabel="Number of Markets")
add("catalog__markets_opened_per_month", "line", "How Many New Markets are created Each Month",
    x="month", y="markets_opened", xlabel="Month", ylabel="New Markets Created")

# --- wallet trades ----------------------------------------------------------
add("wallets__wallet_lorenz", "line", "How Unevenly Trading is spread across Traders",
    x="wallet_percentile", y="cum_dollar_share", xdiv=100, diag=True, marker="", xlim=(0, 1), ylim=(0, 1),
    xlabel="Share of Traders, from Smallest to Biggest", ylabel="Share of All Dollars Traded")
add("wallets__dow_hour_heatmap", "heatmap", "When do People trade? Weekday and Hour",
    index="day_of_week_sun0", columns="hour_utc", values="dollars",
    ylabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="Hour of Day (UTC)", ylabel="Day of Week", colorbar_label="Average Dollars Traded in the Hour")
add("wallets__daily_overview", "panels", "Daily Trading Activity",
    x="date", ys=["dollars", "active_wallets", "active_markets", "avg_trade_usd"], ncols=2, datetime_x=True,
    panel_titles={"dollars": "Dollars Traded", "active_wallets": "Distinct Traders (Wallets)",
                  "active_markets": "Markets Traded", "avg_trade_usd": "Average Trade Size ($)"})
add("wallets__time_to_close", "bar", "Distribution of Trades and Dollars Spent leading to a Market's End",
    x="time_before_close", ys=["trades", "dollars"], normalize=True,
    series_labels={"trades": "share of trades", "dollars": "share of dollars"},
    xlabel="Time Left before the Market settles")
add("wallets__burner_buckets", "bar", "One-Market Traders versus Traders Who spread across Markets",
    x="markets_traded", ys=["wallets", "dollars"], normalize=True,
    series_labels={"wallets": "share of wallets", "dollars": "share of dollars"},
    xlabel="Number of Markets the Wallet traded")
add("wallets__top_markets", "hbar", "The 20 Most Traded Markets",
    label="title", label_fn="title", y="dollars", top=20, xlabel="Dollars Traded")
add("wallets__topic_share_by_day", "stacked", "What do People trade? (Rough Topics)",
    index="day", columns="topic", values="dollars", legend_title="Topic",
    xlabel="Day (UTC)", ylabel="Dollars Traded")

# --- cross-source -----------------------------------------------------------
add("cross__edge_by_time_to_close", "bar", "Do Last-Minute Traders do Better?",
    x="time_before_close", y="avg_edge_per_share", hue="side", hue_name="", yerr=("sd_edge_per_share", "trades"),
    hline=0, xlabel="Time Left before the Market settles", ylabel="Average Profit per Share ($)")
add("cross__calibration_book", "calib_buckets", "When the Market says a Price, How Often does YES win?",
    x="price_bucket_lo", y="win_rate", hue="horizon_h", size="markets")
add("cross__edge_by_trade_size", "bar", "Do Bigger Trades make More Money?",
    x="size_bucket", y="return_per_dollar", hue="side", hue_name="", hline=0,
    xlabel="Size of the Trade", ylabel="Profit per Dollar Traded")
add("cross__edge_by_wallet_class", "bar", "Which Kinds of Wallets earn Better Returns?",
    x="wallet_class", y="return_per_dollar", annotate="trades", hline=0,
    xlabel="Number of Markets the Wallet traded", ylabel="Profit per Dollar Traded")

# --- order book -------------------------------------------------------------
add("orderbook__liquidity_by_volume", "liquidity", "Do Bigger Markets cost Less to trade?")


# ----------------------------------------------------------------------------
# Data window of each chart (the charts do NOT all cover the same dates)
# ----------------------------------------------------------------------------
def tag(when, *keys):
    for k in keys:
        SPECS[k]["when"] = when


tag(f"all markets in the catalog, as of {CATALOG_ASOF}",
    "catalog__markets_opened_per_month", "catalog__duration_hist", "catalog__volume_pareto")
tag(f"all settled markets in the catalog, through {CATALOG_ASOF}", "catalog__yes_rate_by_neg_risk")
tag(W_DAILY, "wallets__daily_overview")
tag(f"all trades, {W_TRADES}", "wallets__wallet_lorenz", "wallets__burner_buckets", "wallets__top_markets",
    "wallets__topic_share_by_day")
# these two only count trades in markets that have settled (the join to the resolutions table)
tag(f"trades {W_TRADES}, in markets settled by {CATALOG_ASOF}", "wallets__time_to_close",
    "cross__edge_by_trade_size", "cross__edge_by_wallet_class")
tag(f"{W_HOUR_START}; YES side of each market's order book; markets grouped by total trading volume (catalog)",
    "orderbook__liquidity_by_volume")

tag(f"all trades, {W_TRADES}, averaged per day", "wallets__dow_hour_heatmap")
tag(f"trades {W_TRADES}, in markets settled by {CATALOG_ASOF}", "cross__edge_by_time_to_close")
# If a table was saved with window columns (a longer-window run), that window replaces the default text above.
SPECS["wallets__dow_hour_heatmap"]["when_tpl"] = "all trades, {w} (UTC), averaged per day{note}"
SPECS["cross__edge_by_time_to_close"]["when_tpl"] = "trades {w} (UTC), in markets settled by " + CATALOG_ASOF + "{note}"

# Calibration: the notebook saves the settlement days it actually had snapshots for (window_start, window_end,
# days_loaded, days_in_window) with the table; that window replaces the default text below automatically.
SPECS["cross__calibration_book"]["when"] = "markets settled Oct 1 - Oct 20, 2026 (UTC), using the order-book snapshot 1, 6 and 24 hours before each settlement"
SPECS["cross__calibration_book"]["when_tpl"] = ("markets settled {w} (UTC), using the order-book snapshot 1, 6 and 24 hours "
                                                "before each settlement{note}")
SPECS["cross__calibration_book"]["missing_what"] = "order-book snapshots"

_untagged = [k for k, s in SPECS.items() if "when" not in s]
if _untagged:
    print("WARNING: charts with no data window:", _untagged)


def fmt_window(start, end) -> str:
    """
    Formats a date range as a human-readable string.

    Parameters:
        start (str or datetime): The start date of the range.
        end (str or datetime): The end date of the range.

    Returns:
        str: A formatted string representing the date range.
    """

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s.year == e.year:
        return f"{s:%b} {s.day} - {e:%b} {e.day}, {e.year}"
    return f"{s:%b} {s.day}, {s.year} - {e:%b} {e.day}, {e.year}"


def window_from_table(df, spec):
    """
    Generates the 'Data:' text from the window saved with the table, if it has one.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the table data.
        spec (dict): The specification dictionary containing 'when_tpl' and 'missing_what'.

    Returns:
        str or None: The formatted 'Data:' text, or None if no valid window is found.
    """

    need = {"window_start", "window_end", "days_loaded", "days_in_window"}
    if "when_tpl" not in spec or not need <= set(df.columns) or df.empty:
        return None
    r = df.iloc[0]
    w = fmt_window(r["window_start"], r["window_end"])
    note = ""
    if int(r["days_loaded"]) < int(r["days_in_window"]):
        note = f" ({int(r['days_loaded'])} of {int(r['days_in_window'])} days had {spec.get('missing_what', 'trade files')})"
    return spec["when_tpl"].format(w=w, note=note)


def daily_window(df):
    """
    Generates a daily overview of the data, highlighting the days that really have numbers (a missing file leaves a gap, not a zero).

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the daily data.

    Returns:
        str or None: A formatted string representing the daily overview, or None if no valid data is found.
    """

    ok = df.dropna(subset=["dollars"])
    if ok.empty:
        return None
    w = fmt_window(ok["date"].min(), ok["date"].max())
    gaps = len(df) - len(ok)
    note = f" ({gaps} day{'s' if gaps != 1 else ''} without data)" if gaps else ""
    return f"{w} (UTC), one point per day{note}"


# ----------------------------------------------------------------------------
# Render everything
# ----------------------------------------------------------------------------
def resolve(key):
    """
    Resolves the table for a given spec: the exact file name, else the same table name under another prefix.

    Parameters:
        key (str): The key representing the spec.

    Returns:
        pandas.DataFrame or None: The resolved DataFrame, or None if no match is found.
    """

    if key in T:
        return T[key]
    name = key.split("__", 1)[1]
    hits = [k for k in T if k.split("__", 1)[-1] == name]
    return T[hits[0]] if len(hits) == 1 else None


def add_header(fig, spec, key):
    """
    Adds a title, a 'Data:' line with the date window, and the chart to the figure. Nothing is added under the chart.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to which the header will be added.
        spec (dict): The specification dictionary containing 'title' and 'when'.
        key (str): The key representing the spec, used for the footer.
    """
    
    for ax in fig.axes:
        plain_log_ticks(ax)
        plain_linear_ticks(ax)
    w, h = fig.get_size_inches()
    title = textwrap.fill(spec["title"], width=max(30, int(w * 8.2)))
    # fig.text (not suptitle): tight_layout would reserve extra space for a suptitle on top of the rect below
    fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=14, fontweight="bold")
    top = 0.995 - (title.count("\n") + 1) * 0.27 / h
    when = spec.get("when")
    if when:
        sub = textwrap.fill(f"Data: {when}", width=max(40, int(w * 12.5)))
        fig.text(0.5, top - 0.005, sub, ha="center", va="top", fontsize=10, color="#555555")
        top -= 0.005 + (sub.count("\n") + 1) * 0.2 / h
    fig.text(0.99, 0.002, key, ha="right", va="bottom", fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.015, 1, top - 0.01))


rendered, skipped, failed = [], [], []
pdf = None
try:
    pdf = PdfPages(PDF_PATH)
except OSError as e:  # e.g. the PDF is open in a viewer: still write the PNGs
    print(f"WARNING: cannot write {PDF_PATH} ({e}); PNGs will still be written")

for key, spec in SPECS.items():
    if ONLY and key not in ONLY:
        continue
    df_src = resolve(key)
    if df_src is None:
        skipped.append(key)
        continue
    try:
        win = daily_window(df_src) if key == "wallets__daily_overview" else window_from_table(df_src, spec)
        if win:
            spec["when"] = win
        fig = RENDER[spec["kind"]](df_src.copy(), spec)
        add_header(fig, spec, key)
        fig.savefig(f"{FIG_DIR}/{key}.png", dpi=DPI, bbox_inches="tight")
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        rendered.append(key)
        if SHOW:
            plt.show()
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        failed.append((key, repr(e)[:160]))
        print(f"--- {key} failed ---")
        traceback.print_exc()
        plt.close("all")
if pdf is not None:
    pdf.close()

# Guide: every rendered chart with its data window (handy for slides or a report)
with open(f"{FIG_DIR}/guide.md", "w", encoding="utf-8") as g:
    g.write("# Polymarket EDA: Chart Guide\n\n")
    g.write("A *market* is one yes/no question people trade on. A YES (or NO) *share* pays $1 if that answer "
            "turns out right, so its price between $0 and $1 is roughly the crowd's probability for it.\n\n")
    g.write("## Data Windows at a Glance\n\nThe charts do not all cover the same dates. Grouped by window:\n\n")
    windows: dict[str, list[str]] = {}
    for key in rendered:
        windows.setdefault(SPECS[key].get("when", "not specified"), []).append(SPECS[key]["title"])
    for w_, titles in windows.items():
        g.write(f"- **{w_}**: " + "; ".join(titles) + "\n")
    g.write("\n")
    for key in rendered:
        g.write(f"## {SPECS[key]['title']}\n\n**Data:** {SPECS[key].get('when', 'not specified')}\n\n"
                f"![{SPECS[key]['title']}]({key}.png)\n\n")

print(f"\nrendered {len(rendered)} of {len(SPECS)} | skipped (parquet not exported yet) {len(skipped)} | failed {len(failed)}")
if skipped:
    print("skipped:", skipped)
for k, err in failed:
    print("FAILED", k, "->", err)
extra = sorted(set(T) - set(SPECS))
if extra:
    print("Parquet files with no spec (ignored):", extra)
print(f"PNGs in {FIG_DIR}/, guide in {FIG_DIR}/guide.md, all charts in {PDF_PATH}")
