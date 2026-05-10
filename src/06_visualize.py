"""
06_visualize.py  –  Phase 5: Charts & Output
=============================================
Reads the evaluation outputs produced by 05_evaluate.py and generates
a comprehensive suite of publication-quality charts.

Charts produced (saved to output/charts/)
-----------------------------------------
01_support_confidence_scatter.png  – bubble chart coloured by lift
02_lift_distribution.png           – histogram + KDE of lift values
03_top_rules_heatmap.png           – heatmap of top-rule metrics
04_quality_pie.png                 – quality-tier breakdown
05_itemset_frequency_bar.png       – most frequent itemsets by support
06_confidence_lift_violin.png      – violin plots per quality tier
07_rule_network.png                – network graph of antecedent → consequent
08_metric_correlation.png          – pairplot of numeric metrics
"""

import os
import sys
import logging
import warnings
import re

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── optional heavy deps (graceful degradation) ───────────────────────────────
try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False

try:
    import networkx as nx
    _NETWORKX = True
except ImportError:
    _NETWORKX = False

# ── project root ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import TOP_N_PATTERNS, ANOMALY_THRESHOLD

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(ROOT, "output")
CHARTS_DIR  = os.path.join(OUTPUT_DIR, "charts")
EVAL_CSV    = os.path.join(OUTPUT_DIR, "evaluation_report.csv")
ITEMS_CSV   = os.path.join(OUTPUT_DIR, "frequent_itemsets.csv")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0f1117",
    "panel":     "#1a1d2e",
    "accent1":   "#6c63ff",
    "accent2":   "#ff6584",
    "accent3":   "#43e8d8",
    "accent4":   "#f9a825",
    "text":      "#e0e0e0",
    "subtext":   "#9e9e9e",
    "grid":      "#2a2d3e",
    "excellent": "#43e8d8",
    "good":      "#6c63ff",
    "acceptable":"#f9a825",
    "weak":      "#ff6584",
}

QUALITY_COLORS = {
    "⭐ Excellent": PALETTE["excellent"],
    "✔ Good":      PALETTE["good"],
    "~ Acceptable":PALETTE["acceptable"],
    "✗ Weak":      PALETTE["weak"],
}

CMAP_LIFT = LinearSegmentedColormap.from_list(
    "lift_cmap",
    ["#1a1d2e", "#6c63ff", "#43e8d8", "#f9a825"],
)


# ════════════════════════════════════════════════════════════════════════════
# global matplotlib style
# ════════════════════════════════════════════════════════════════════════════

def _apply_global_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":    PALETTE["bg"],
        "axes.facecolor":      PALETTE["panel"],
        "axes.edgecolor":      PALETTE["grid"],
        "axes.labelcolor":     PALETTE["text"],
        "axes.titlecolor":     PALETTE["text"],
        "axes.titlesize":      14,
        "axes.labelsize":      11,
        "axes.grid":           True,
        "grid.color":          PALETTE["grid"],
        "grid.linewidth":      0.6,
        "xtick.color":         PALETTE["subtext"],
        "ytick.color":         PALETTE["subtext"],
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.facecolor":    PALETTE["panel"],
        "legend.edgecolor":    PALETTE["grid"],
        "legend.labelcolor":   PALETTE["text"],
        "legend.fontsize":     9,
        "text.color":          PALETTE["text"],
        "figure.titlesize":    16,
        "figure.titleweight":  "bold",
    })


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Saved → '%s'", name)


def _title_box(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, pad=12, color=PALETTE["text"],
                 fontsize=13, fontweight="bold")


# ════════════════════════════════════════════════════════════════════════════
# data helpers
# ════════════════════════════════════════════════════════════════════════════

def _load(path: str, name: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        log.warning("%s not found – skipping charts that need it.", name)
        return None
    df = pd.read_csv(path)
    log.info("Loaded %d rows from '%s'", len(df), name)
    return df


def _short_label(value, max_len: int = 40) -> str:
    """Shorten a frozenset string to a readable label."""
    s = str(value)
    s = re.sub(r"frozenset\(\{?", "", s)
    s = re.sub(r"\}?\)", "", s)
    s = s.replace("'", "").strip()
    return s[:max_len] + "…" if len(s) > max_len else s


# ════════════════════════════════════════════════════════════════════════════
# individual chart functions
# ════════════════════════════════════════════════════════════════════════════

def chart_01_scatter(df: pd.DataFrame) -> None:
    """Support vs Confidence bubble chart, bubble size & colour = lift."""
    if not {"support", "confidence", "lift"}.issubset(df.columns):
        log.warning("chart_01: missing columns – skipped.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    lift_norm = (df["lift"] - df["lift"].min()) / (df["lift"].max() - df["lift"].min() + 1e-9)
    sizes = 30 + lift_norm * 250

    sc = ax.scatter(
        df["support"], df["confidence"],
        s=sizes, c=df["lift"],
        cmap=CMAP_LIFT, alpha=0.85, edgecolors=PALETTE["grid"], linewidths=0.5,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Lift", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"])

    # anomaly threshold line
    ax.axhline(y=0.5, color=PALETTE["accent2"], lw=1.2, ls="--",
               label=f"Min confidence = 0.5")
    ax.axvline(x=df["support"].quantile(0.5), color=PALETTE["accent3"],
               lw=1.0, ls=":", label="Median support")

    _title_box(ax, "Support vs Confidence", "bubble size & colour = lift value")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.legend()
    _save(fig, "01_support_confidence_scatter.png")


def chart_02_lift_histogram(df: pd.DataFrame) -> None:
    """Histogram + KDE of lift distribution."""
    if "lift" not in df.columns:
        log.warning("chart_02: 'lift' column missing – skipped.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    lift_vals = df["lift"].dropna()

    n_bins = min(40, max(10, len(lift_vals) // 5))
    ax.hist(lift_vals, bins=n_bins, color=PALETTE["accent1"],
            edgecolor=PALETTE["bg"], alpha=0.80, zorder=2)

    if _SEABORN:
        ax2 = ax.twinx()
        sns.kdeplot(lift_vals, ax=ax2, color=PALETTE["accent3"], lw=2.5)
        ax2.set_ylabel("Density", color=PALETTE["accent3"])
        ax2.tick_params(colors=PALETTE["subtext"])
        ax2.spines[:].set_color(PALETTE["grid"])

    ax.axvline(ANOMALY_THRESHOLD, color=PALETTE["accent2"],
               lw=1.5, ls="--",
               label=f"Anomaly threshold ({ANOMALY_THRESHOLD})")
    ax.axvline(lift_vals.mean(), color=PALETTE["accent4"],
               lw=1.5, ls="-.", label=f"Mean lift ({lift_vals.mean():.2f})")

    _title_box(ax, "Lift Distribution", "histogram of all association rules")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, "02_lift_distribution.png")


def chart_03_top_rules_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of metrics for the top-N rules."""
    metrics = [c for c in ["support", "confidence", "lift", "leverage", "conviction"]
               if c in df.columns]
    if not metrics:
        log.warning("chart_03: no numeric metric columns – skipped.")
        return

    top = df.head(TOP_N_PATTERNS).copy()
    if "antecedents" in top.columns:
        top["rule"] = (
            top["antecedents"].apply(_short_label) + "  →  " +
            top["consequents"].apply(_short_label)
        )
    else:
        top["rule"] = top.index.astype(str)

    heat_data = top.set_index("rule")[metrics]
    # normalise per column for visual clarity
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)

    fig_h = max(6, len(top) * 0.42)
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.6), fig_h))

    cmap = "YlOrRd" if not _SEABORN else "magma"
    im = ax.imshow(heat_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(heat_norm.index, fontsize=8)

    # annotate cells with original values
    for i in range(len(top)):
        for j, metric in enumerate(metrics):
            val = heat_data.iloc[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "–"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalised value", color=PALETTE["text"])
    cbar.ax.tick_params(colors=PALETTE["subtext"])

    _title_box(ax, f"Top-{TOP_N_PATTERNS} Rules – Metric Heatmap",
               "cells show raw values; colour = normalised intensity")
    _save(fig, "03_top_rules_heatmap.png")


def chart_04_quality_pie(df: pd.DataFrame) -> None:
    """Donut chart of quality-tier distribution."""
    if "quality" not in df.columns:
        log.warning("chart_04: 'quality' column missing – skipped.")
        return

    counts = df["quality"].value_counts()
    colors = [QUALITY_COLORS.get(k, PALETTE["accent1"]) for k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
    )
    for t in texts:
        t.set_color(PALETTE["text"])
        t.set_fontsize(10)
    for a in autotexts:
        a.set_color(PALETTE["bg"])
        a.set_fontsize(9)
        a.set_fontweight("bold")

    # donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc=PALETTE["panel"])
    ax.add_patch(centre_circle)
    ax.text(0, 0, f"{len(df)}\nrules", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["text"])

    _title_box(ax, "Rule Quality Distribution", "by quality tier")
    _save(fig, "04_quality_pie.png")


def chart_05_itemset_frequency(items_df: pd.DataFrame) -> None:
    """Horizontal bar chart of the most frequent itemsets."""
    if items_df is None:
        return
    if "support" not in items_df.columns:
        log.warning("chart_05: 'support' column missing – skipped.")
        return

    # derive label
    label_col = next(
        (c for c in ["itemsets", "items", "itemset"] if c in items_df.columns),
        None,
    )
    top = items_df.nlargest(20, "support").copy()
    if label_col:
        top["label"] = top[label_col].apply(_short_label)
    else:
        top["label"] = top.index.astype(str)

    top = top.sort_values("support")
    colors = [
        PALETTE["accent1"] if i < len(top) * 0.33
        else PALETTE["accent3"] if i < len(top) * 0.66
        else PALETTE["accent4"]
        for i in range(len(top))
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["label"], top["support"], color=colors,
                   edgecolor=PALETTE["bg"], height=0.7)
    for bar, val in zip(bars, top["support"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=PALETTE["subtext"])

    _title_box(ax, "Top Frequent Itemsets", "by support value")
    ax.set_xlabel("Support")
    ax.set_ylabel("Itemset")
    _save(fig, "05_itemset_frequency_bar.png")


def chart_06_violin(df: pd.DataFrame) -> None:
    """Violin plots of confidence & lift per quality tier."""
    if not _SEABORN:
        log.warning("chart_06: seaborn not installed – skipped.")
        return
    if "quality" not in df.columns:
        log.warning("chart_06: 'quality' column missing – skipped.")
        return

    metrics_avail = [m for m in ["confidence", "lift"] if m in df.columns]
    if not metrics_avail:
        return

    fig, axes = plt.subplots(1, len(metrics_avail),
                             figsize=(6 * len(metrics_avail), 6))
    if len(metrics_avail) == 1:
        axes = [axes]

    order = [k for k in QUALITY_COLORS if k in df["quality"].unique()]
    pal   = {k: QUALITY_COLORS[k] for k in order}

    for ax, metric in zip(axes, metrics_avail):
        sns.violinplot(
            data=df, x="quality", y=metric,
            order=order, palette=pal,
            inner="quartile", ax=ax,
            linewidth=1.2,
        )
        ax.set_facecolor(PALETTE["panel"])
        _title_box(ax, f"{metric.capitalize()} by Quality Tier")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    _title_box(axes[0], "Metric Distribution by Quality Tier")
    _save(fig, "06_confidence_lift_violin.png")


def chart_07_network(df: pd.DataFrame) -> None:
    """Network graph: antecedent items → consequent items."""
    if not _NETWORKX:
        log.warning("chart_07: networkx not installed – skipped.")
        return
    if not {"antecedents", "consequents", "lift"}.issubset(df.columns):
        log.warning("chart_07: required columns missing – skipped.")
        return

    top = df.nlargest(min(30, len(df)), "lift")

    G = nx.DiGraph()
    for _, row in top.iterrows():
        ant   = _short_label(row["antecedents"], 30)
        cons  = _short_label(row["consequents"], 30)
        lft   = float(row.get("lift", 1))
        conf  = float(row.get("confidence", 0.5))
        G.add_edge(ant, cons, weight=lft, confidence=conf)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor(PALETTE["bg"])

    pos = nx.spring_layout(G, k=2.5, seed=42)

    # edge widths & colours from lift
    edges      = G.edges(data=True)
    weights    = [d["weight"] for _, _, d in edges]
    max_w      = max(weights) if weights else 1
    edge_widths = [1.0 + 3.0 * w / max_w for w in weights]
    edge_colors = [CMAP_LIFT(w / max_w) for w in weights]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths, edge_color=edge_colors,
        arrows=True, arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        alpha=0.75,
    )

    # node sizes from degree
    degrees   = dict(G.degree())
    node_sizes = [300 + degrees[n] * 120 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=PALETTE["accent1"],
        alpha=0.9,
        linewidths=1.5,
        edgecolors=PALETTE["accent3"],
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7, font_color=PALETTE["text"],
    )

    _title_box(ax, "Association Rule Network",
               f"top-{len(top)} rules by lift  |  edge width = lift")
    ax.axis("off")

    # legend via colour bar proxy
    sm = plt.cm.ScalarMappable(cmap=CMAP_LIFT,
                               norm=plt.Normalize(min(weights), max(weights)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Lift", color=PALETTE["text"])
    cbar.ax.tick_params(colors=PALETTE["subtext"])

    _save(fig, "07_rule_network.png")


def chart_08_metric_correlation(df: pd.DataFrame) -> None:
    """Pairplot / correlation heatmap of numeric metrics."""
    numeric_cols = [c for c in ["support", "confidence", "lift",
                                "leverage", "conviction", "zhang_metric"]
                    if c in df.columns]
    if len(numeric_cols) < 2:
        log.warning("chart_08: need ≥2 numeric columns – skipped.")
        return

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(len(numeric_cols) * 1.5 + 2,
                                    len(numeric_cols) * 1.5 + 1))
    cmap_c = LinearSegmentedColormap.from_list(
        "corr_cmap", ["#ff6584", "#1a1d2e", "#43e8d8"]
    )
    im = ax.imshow(corr.values, cmap=cmap_c, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Pearson r", color=PALETTE["text"])
    cbar.ax.tick_params(colors=PALETTE["subtext"])

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=10)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else PALETTE["subtext"])

    _title_box(ax, "Metric Correlation Matrix",
               "Pearson correlation between quality metrics")
    _save(fig, "08_metric_correlation.png")


# ════════════════════════════════════════════════════════════════════════════
# summary dashboard (all charts on one figure)
# ════════════════════════════════════════════════════════════════════════════

def chart_dashboard(df: pd.DataFrame) -> None:
    """
    Compose a single dashboard PNG with 4 key sub-charts.
    (scatter, lift histogram, quality pie, top-rules bar)
    """
    if not {"support", "confidence", "lift"}.issubset(df.columns):
        return

    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    fig.suptitle("System Failure Pattern Analysis  –  Dashboard",
                 fontsize=18, fontweight="bold", color=PALETTE["text"], y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── scatter ──
    ax1 = fig.add_subplot(gs[0, :2])
    lift_norm = (df["lift"] - df["lift"].min()) / (df["lift"].max() - df["lift"].min() + 1e-9)
    sc = ax1.scatter(df["support"], df["confidence"],
                     s=20 + lift_norm * 180, c=df["lift"],
                     cmap=CMAP_LIFT, alpha=0.80,
                     edgecolors=PALETTE["grid"], linewidths=0.4)
    fig.colorbar(sc, ax=ax1, label="Lift")
    _title_box(ax1, "Support vs Confidence")
    ax1.set_xlabel("Support"); ax1.set_ylabel("Confidence")

    # ── quality pie ──
    ax2 = fig.add_subplot(gs[0, 2])
    if "quality" in df.columns:
        counts = df["quality"].value_counts()
        colors = [QUALITY_COLORS.get(k, PALETTE["accent1"]) for k in counts.index]
        ax2.pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.0f%%", startangle=140,
                wedgeprops={"edgecolor": PALETTE["bg"]},
                textprops={"color": PALETTE["text"], "fontsize": 8})
        circle = plt.Circle((0, 0), 0.5, fc=PALETTE["panel"])
        ax2.add_patch(circle)
        _title_box(ax2, "Quality Tiers")

    # ── lift histogram ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.hist(df["lift"].dropna(), bins=30, color=PALETTE["accent1"],
             edgecolor=PALETTE["bg"], alpha=0.85)
    ax3.axvline(ANOMALY_THRESHOLD, color=PALETTE["accent2"], lw=1.5, ls="--",
                label=f"Anomaly threshold={ANOMALY_THRESHOLD}")
    ax3.axvline(df["lift"].mean(), color=PALETTE["accent4"], lw=1.5, ls="-.",
                label=f"Mean={df['lift'].mean():.2f}")
    ax3.legend(fontsize=8)
    _title_box(ax3, "Lift Distribution")
    ax3.set_xlabel("Lift"); ax3.set_ylabel("Count")

    # ── top-10 rules confidence bar ──
    ax4 = fig.add_subplot(gs[1, 2])
    top10 = df.head(10)
    if "antecedents" in top10.columns:
        labels = [_short_label(r, 22) for r in top10["antecedents"]]
    else:
        labels = [str(i) for i in top10.index]
    bars = ax4.barh(range(len(top10)), top10["confidence"],
                    color=PALETTE["accent3"], edgecolor=PALETTE["bg"])
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.invert_yaxis()
    _title_box(ax4, "Top-10 Rules (confidence)")
    ax4.set_xlabel("Confidence")

    _save(fig, "00_dashboard.png")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def visualize(eval_path: str = EVAL_CSV, items_path: str = ITEMS_CSV) -> None:
    log.info("═" * 60)
    log.info("Phase 5 – Visualisation")
    log.info("═" * 60)

    _apply_global_style()

    df    = _load(eval_path, "evaluation_report.csv")
    items = _load(items_path, "frequent_itemsets.csv")

    if df is None:
        log.error("Evaluation report not found.  Run 05_evaluate.py first.")
        return

    # generate all charts
    chart_dashboard(df)
    chart_01_scatter(df)
    chart_02_lift_histogram(df)
    chart_03_top_rules_heatmap(df)
    chart_04_quality_pie(df)
    chart_05_itemset_frequency(items)
    chart_06_violin(df)
    chart_07_network(df)
    chart_08_metric_correlation(df)

    log.info("All charts saved to: '%s'", CHARTS_DIR)


if __name__ == "__main__":
    visualize()
