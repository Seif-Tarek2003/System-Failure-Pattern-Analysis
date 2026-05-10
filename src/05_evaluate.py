"""
05_evaluate.py  –  Phase 4: Confidence & Lift Analysis
=======================================================
Loads association rules produced by 03_apriori.py / 04_fpgrowth.py,
evaluates every rule with plain-English interpretation of Confidence
and Lift, ranks patterns, flags anomalies, and saves a detailed report.

Core Concepts (plain English)
------------------------------
SUPPORT    – How often do events A and B both appear in log sessions?
             e.g. support=0.15 → 15% of all sessions contain both.

CONFIDENCE – Given that event A occurred, how often does B also occur?
             e.g. confidence=0.80 → 80% of the time when A fires, B follows.
             Think: "reliability of the rule."

LIFT       – Does A actually *cause* or predict B, or is it just coincidence?
             lift > 1.0 → A and B co-occur more than by chance (real pattern)
             lift = 1.0 → purely random co-occurrence (no useful rule)
             lift < 1.0 → A actually suppresses B

LEVERAGE   – Extra sessions where A∩B occurred compared to independence.
             Positive leverage confirms the rule is non-trivial.

CONVICTION – How often the rule would be wrong if A and B were independent.
             Higher = more reliable prediction (∞ means 100% confidence).

Outputs
-------
output/evaluation_report.csv   – full ranked rule table with interpretations
output/top_rules.csv           – top-N rules by composite score
output/anomaly_rules.csv       – high-lift anomaly rules
"""

import os
import sys
import logging
import warnings
import textwrap

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── resolve project root so the script works from any CWD ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import (
    REPORTS_DIR,
    MIN_CONFIDENCE,
    MIN_LIFT,
    TOP_N_PATTERNS,
    ANOMALY_THRESHOLD,
)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(ROOT, "output")
RULES_CSV   = os.path.join(OUTPUT_DIR, "association_rules.csv")
EVAL_CSV    = os.path.join(OUTPUT_DIR, "evaluation_report.csv")
TOP_CSV     = os.path.join(OUTPUT_DIR, "top_rules.csv")
ANOMALY_CSV = os.path.join(OUTPUT_DIR, "anomaly_rules.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# concept explanation banner (printed once at startup)
# ════════════════════════════════════════════════════════════════════════════

_CONCEPT_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 4 — ASSOCIATION RULE EVALUATION                      ║
║           System Failure Pattern Analysis                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  WHAT ARE WE DOING?                                                  ║
║  We treat each log time-window as a "transaction" (like a shopping   ║
║  basket). Each error / event in that window is an "item". We then    ║
║  look for rules of the form:                                         ║
║       IF {{EventA, EventB}} occur  ->  THEN {{EventC}} also occurs      ║
║  and measure how strong / reliable each rule is.                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  KEY METRICS (plain English)                                         ║
║                                                                      ║
║  SUPPORT    – How common is this combination across all sessions?    ║
║               0.05 = appears in 5% of all log windows.              ║
║                                                                      ║
║  CONFIDENCE – If the LEFT side fires, how often does RIGHT follow?  ║
║               0.80 = 80% of the time — very reliable rule.          ║
║               Threshold used: {conf:.2f}                               ║
║                                                                      ║
║  LIFT       – Is this pattern real, or just random coincidence?     ║
║               > 1.0  → events co-occur MORE than expected by chance  ║
║               = 1.0  → purely random, no real association            ║
║               < 1.0  → events actually avoid each other             ║
║               Threshold used: {lift:.2f}                               ║
║                                                                      ║
║  LEVERAGE   – Extra co-occurrences beyond statistical independence.  ║
║               Positive = the rule is non-trivial and informative.    ║
║                                                                      ║
║  CONVICTION – Penalty for wrong predictions. Higher = more reliable. ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(conf=MIN_CONFIDENCE, lift=MIN_LIFT)


# ════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_rules(path: str) -> pd.DataFrame:
    """Load the association-rules CSV; raise clearly if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  Rules file not found: {path}\n"
            "  Run 03_apriori.py or 04_fpgrowth.py first to generate rules."
        )
    df = pd.read_csv(path)
    log.info("Loaded %d rules from '%s'", len(df), path)
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names so the script is robust against minor
    differences between the Apriori and FP-Growth output schemas.
    """
    rename_map = {}
    for col in df.columns:
        low = col.lower().replace(" ", "_")
        if low in {"antecedents", "antecedent"}:
            rename_map[col] = "antecedents"
        elif low in {"consequents", "consequent"}:
            rename_map[col] = "consequents"
        elif low in {"support", "antecedent_support", "consequent_support",
                     "confidence", "lift", "leverage", "conviction"}:
            rename_map[col] = low
    return df.rename(columns=rename_map)


def _compute_extra_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leverage, conviction, and Zhang's metric if not already present."""

    # leverage: P(A∩B) − P(A)·P(B)
    if "leverage" not in df.columns:
        if {"antecedent_support", "consequent_support", "support"}.issubset(df.columns):
            df["leverage"] = (
                df["support"]
                - df["antecedent_support"] * df["consequent_support"]
            )
        else:
            df["leverage"] = np.nan

    # conviction: (1 − P(B)) / (1 − confidence)
    if "conviction" not in df.columns:
        if "confidence" in df.columns and "consequent_support" in df.columns:
            denom = 1 - df["confidence"].replace(1.0, np.nan)
            df["conviction"] = (1 - df["consequent_support"]) / denom
        else:
            df["conviction"] = np.nan

    # Zhang's metric
    if "zhang_metric" not in df.columns:
        if {"support", "antecedent_support", "consequent_support"}.issubset(df.columns):
            pab = df["support"]
            pa  = df["antecedent_support"]
            pb  = df["consequent_support"]
            num = pab - pa * pb
            den = np.maximum(pab * (1 - pb), pb * (pa - pab))
            df["zhang_metric"] = num / den.replace(0, np.nan)
        else:
            df["zhang_metric"] = np.nan

    return df


# ── quality tier label ───────────────────────────────────────────────────────

def _quality_label(row: pd.Series) -> str:
    lift = row.get("lift", 1)
    conf = row.get("confidence", 0)
    lev  = row.get("leverage", 0) or 0

    if lift >= 3.0 and conf >= 0.75 and lev > 0:
        return "⭐ Excellent"
    elif lift >= 2.0 and conf >= 0.60:
        return "✔ Good"
    elif lift >= MIN_LIFT and conf >= MIN_CONFIDENCE:
        return "~ Acceptable"
    else:
        return "✗ Weak"


# ── plain-English interpretation of a single rule ────────────────────────────

def _interpret_confidence(conf: float) -> str:
    """Return a one-line plain-English verdict on the confidence value."""
    if conf >= 0.90:
        return f"{conf:.0%} confidence → Near-certain prediction. Almost every time the LHS events occur, the RHS failure follows."
    elif conf >= 0.75:
        return f"{conf:.0%} confidence → Strong prediction. This rule is reliable in 3 out of 4 matching sessions."
    elif conf >= 0.60:
        return f"{conf:.0%} confidence → Moderate prediction. Fires correctly more often than not."
    elif conf >= 0.50:
        return f"{conf:.0%} confidence → Marginal prediction. Just above the minimum threshold — use with caution."
    else:
        return f"{conf:.0%} confidence → Weak prediction. Below the minimum threshold of {MIN_CONFIDENCE:.0%}."


def _interpret_lift(lift: float) -> str:
    """Return a one-line plain-English verdict on the lift value."""
    if lift >= 5.0:
        return f"Lift = {lift:.2f} → Extremely strong association — events co-occur {lift:.1f}× more than pure chance."
    elif lift >= 3.0:
        return f"Lift = {lift:.2f} → Very strong real pattern — highly unlikely to be random."
    elif lift >= 2.0:
        return f"Lift = {lift:.2f} → Strong positive association — events clearly attract each other."
    elif lift >= 1.2:
        return f"Lift = {lift:.2f} → Moderate positive association — above the minimum useful threshold."
    elif lift > 1.0:
        return f"Lift = {lift:.2f} → Slight positive association — barely above random."
    elif abs(lift - 1.0) < 0.05:
        return f"Lift = {lift:.2f} → Essentially random co-occurrence. This rule is not useful."
    else:
        return f"Lift = {lift:.2f} → Negative association — the LHS events actually suppress the RHS."


def _interpret_leverage(lev) -> str:
    if lev is None or (isinstance(lev, float) and np.isnan(lev)):
        return ""
    if lev > 0.05:
        return f"Leverage = {lev:.4f} → Strongly non-trivial. Many extra co-occurrences beyond independence."
    elif lev > 0.01:
        return f"Leverage = {lev:.4f} → Moderately non-trivial rule."
    elif lev > 0:
        return f"Leverage = {lev:.4f} → Slight positive leverage. Pattern is real but subtle."
    else:
        return f"Leverage = {lev:.4f} → Non-positive leverage. Rule may be trivial or spurious."


def _short(value, n: int = 55) -> str:
    """Clean up a frozenset string for readable display."""
    import re
    s = str(value)
    s = re.sub(r"frozenset\(\{?", "", s)
    s = re.sub(r"\}?\)", "", s)
    s = s.replace("'", "").strip()
    return (s[:n] + "…") if len(s) > n else s


def _itemset_size(value) -> int:
    if isinstance(value, str):
        import re
        s = re.sub(r"frozenset\(\{?|\}?\)", "", value)
        return len([i for i in s.split(",") if i.strip().strip("{}' ")])
    return 1


# ════════════════════════════════════════════════════════════════════════════
# per-rule print  (the educational core)
# ════════════════════════════════════════════════════════════════════════════

def _print_rule_detail(idx: int, row: pd.Series) -> None:
    """Print a full plain-English breakdown of one association rule."""
    W = 70
    ant  = _short(row.get("antecedents", "?"))
    cons = _short(row.get("consequents", "?"))
    conf = float(row.get("confidence", 0))
    lift = float(row.get("lift", 1))
    sup  = float(row.get("support", 0))
    lev  = row.get("leverage", None)
    qual = row.get("quality", "")

    print(f"\n  ┌{'─' * W}┐")
    print(f"  │  Rule #{idx + 1:<4}  {qual:<18}{'':>{W - 30}}│")
    print(f"  │  IF   [ {ant} ]{'':>{max(1, W - 10 - len(ant))}}│")
    print(f"  │  THEN [ {cons} ]{'':>{max(1, W - 10 - len(cons))}}│")
    print(f"  ├{'─' * W}┤")
    print(f"  │  Support    : {sup:.4f}  ({sup*100:.1f}% of all log sessions){'':>{max(1, W - 50)}}│")
    print(f"  │  Confidence : {conf:.4f}{'':>{W - 18}}│")
    # wrap interpretation lines
    for line in textwrap.wrap(_interpret_confidence(conf), W - 5):
        print(f"  │    {line:<{W - 4}}│")
    print(f"  │  Lift       : {lift:.4f}{'':>{W - 18}}│")
    for line in textwrap.wrap(_interpret_lift(lift), W - 5):
        print(f"  │    {line:<{W - 4}}│")
    lev_str = _interpret_leverage(lev)
    if lev_str:
        for line in textwrap.wrap(lev_str, W - 5):
            print(f"  │    {line:<{W - 4}}│")
    print(f"  └{'─' * W}┘")


# ════════════════════════════════════════════════════════════════════════════
# summary report
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(df: pd.DataFrame) -> None:
    D = "═" * 70

    print(f"\n{D}")
    print("  ASSOCIATION RULE QUALITY SUMMARY")
    print(D)
    print(f"  Total rules evaluated  : {len(df):,}")

    for metric, label in [("confidence", "Confidence"), ("lift", "Lift"),
                           ("support", "Support"), ("leverage", "Leverage")]:
        if metric in df.columns and not df[metric].isna().all():
            col = df[metric].dropna()
            print(f"  {label:<13}:  "
                  f"min={col.min():.4f}  mean={col.mean():.4f}  max={col.max():.4f}")

    if "quality" in df.columns:
        print(f"\n  Quality breakdown:")
        for label, count in df["quality"].value_counts().items():
            bar = "█" * int(count / len(df) * 30)
            print(f"    {label:<16} {count:>5}  ({count/len(df)*100:5.1f}%)  {bar}")

    print(D)
    print("\n  METRIC INTERPRETATION GUIDE")
    print("  " + "─" * 68)
    print("  Confidence ≥ 0.90  →  Near-certain rule  (use as a hard alarm)")
    print("  Confidence ≥ 0.75  →  Strong rule         (highly actionable)")
    print("  Confidence ≥ 0.60  →  Moderate rule       (monitor closely)")
    print("  Lift ≥ 3.0         →  Very strong pattern (real causal signal)")
    print("  Lift ≥ 2.0         →  Strong pattern      (non-random)")
    print("  Lift ≥ 1.2         →  Acceptable pattern  (above random)")
    print("  Lift ≈ 1.0         →  Random — discard this rule")
    print("  " + "─" * 68)
    print()


def _print_top_rules(df: pd.DataFrame, n: int = 10) -> None:
    print("═" * 70)
    print(f"  TOP {n} RULES  (ranked by lift × confidence)")
    print("═" * 70)
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        _print_rule_detail(i, row)
    print()


# ════════════════════════════════════════════════════════════════════════════
# main evaluation pipeline
# ════════════════════════════════════════════════════════════════════════════

def evaluate(rules_path: str = RULES_CSV) -> pd.DataFrame:

    # ── 0. concept banner ────────────────────────────────────────────────────
    print(_CONCEPT_BANNER)

    log.info("Loading and evaluating association rules …")

    # ── 1. load ──────────────────────────────────────────────────────────────
    df = _load_rules(rules_path)
    df = _ensure_columns(df)

    # ── 2. compute extra metrics ──────────────────────────────────────────────
    df = _compute_extra_metrics(df)

    # ── 3. derived columns ────────────────────────────────────────────────────
    df["antecedent_len"] = df["antecedents"].apply(_itemset_size)
    df["consequent_len"] = df["consequents"].apply(_itemset_size)
    df["rule_len"]       = df["antecedent_len"] + df["consequent_len"]
    df["quality"]        = df.apply(_quality_label, axis=1)

    # ── 4. filter by thresholds ───────────────────────────────────────────────
    before = len(df)
    filtered = df[
        (df["confidence"] >= MIN_CONFIDENCE) &
        (df["lift"]       >= MIN_LIFT)
    ].copy()

    if filtered.empty:
        log.warning(
            "No rules met conf≥%.2f & lift≥%.2f — showing all %d rules.",
            MIN_CONFIDENCE, MIN_LIFT, before,
        )
        filtered = df.copy()
    else:
        log.info(
            "Filter applied: %d → %d rules  (conf≥%.2f, lift≥%.2f)",
            before, len(filtered), MIN_CONFIDENCE, MIN_LIFT,
        )

    df = filtered

    # ── 5. rank by composite score ────────────────────────────────────────────
    df["score"] = df["lift"] * df["confidence"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # ── 6. print educational summary ─────────────────────────────────────────
    _print_summary(df)

    # ── 7. print per-rule interpretations (top N) ─────────────────────────────
    _print_top_rules(df, n=min(TOP_N_PATTERNS, len(df)))

    # ── 8. anomaly detection ──────────────────────────────────────────────────
    anomalies = df[df["lift"] >= ANOMALY_THRESHOLD].copy()
    if not anomalies.empty:
        print("═" * 70)
        print(f"  ⚠  ANOMALY RULES  (lift ≥ {ANOMALY_THRESHOLD})")
        print(f"     These rules show unusually strong co-occurrence.")
        print(f"     They may represent critical failure cascades.\n")
        for i, (_, row) in enumerate(anomalies.head(5).iterrows()):
            _print_rule_detail(i, row)
        print("═" * 70 + "\n")

    # ── 9. save outputs ───────────────────────────────────────────────────────
    df.to_csv(EVAL_CSV, index=False)
    log.info("Full evaluation report  → '%s'", EVAL_CSV)

    df.head(TOP_N_PATTERNS).to_csv(TOP_CSV, index=False)
    log.info("Top-%d rules            → '%s'", TOP_N_PATTERNS, TOP_CSV)

    anomalies.to_csv(ANOMALY_CSV, index=False)
    log.info(
        "%d anomaly rules (lift≥%.1f) → '%s'",
        len(anomalies), ANOMALY_THRESHOLD, ANOMALY_CSV,
    )

    # ── 10. final verdict ─────────────────────────────────────────────────────
    best = df.iloc[0]
    print("  ✅ EVALUATION COMPLETE")
    print(f"  Best rule found:  {_short(best.get('antecedents','?'))}  →  "
          f"{_short(best.get('consequents','?'))}")
    print(f"  Score: {best['score']:.4f}  |  "
          f"Confidence: {best.get('confidence',0):.2%}  |  "
          f"Lift: {best.get('lift',0):.2f}")
    print()

    return df


# ════════════════════════════════════════════════════════════════════════════
# entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    evaluate()
