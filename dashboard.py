"""
System Failure Pattern Analysis — Dynamic Streamlit Dashboard
============================================================
Full pipeline flow
──────────────────
  User changes slider(s)
        ↓
  "⚠️ Params changed" banner appears  +  sidebar "Run Mining" turns primary
        ↓
  User clicks either  ⚡ Run Mining  (sidebar)
                 or   🔄 Rebuild Report  (main-page banner)
        ↓
  rebuild_reports() is called
      ├─ patch_config_py()   → writes new values to config.py on disk
      └─ run_mining_pipeline()
             ├─ Apriori runs   (via run_mining.py subprocess)
             └─ FP-Growth runs
        ↓
  New CSVs saved to  reports/<timestamp>_*.csv
  New summary.txt    reports/<timestamp>_summary.txt
        ↓
  st.cache_data.clear()   (bust all @cache_data caches)
  st.session_state.selected_run = None   (snap to latest)
  st.rerun()              (Streamlit re-executes top-to-bottom)
        ↓
  New charts appear with updated data

Features vs original
─────────────────────
  • Edit ALL mining parameters live in the sidebar
  • Changes are written directly to config.py on disk
  • Click "⚡ Run Mining" (sidebar) OR "🔄 Rebuild Report" (main page)
      → pipeline executes, new report generated
  • rebuild_reports() orchestrates: config patch → subprocess → save CSVs
  • run_mining_pipeline() now passes ALL params explicitly as CLI args
      (belt-and-suspenders: config.py patched AND CLI args both provided)
  • Every run is cached in reports/ (timestamped, never deleted)
  • Browse any historical run via the Report History selector
  • Report History tab shows a cross-run comparison timeline
"""

from __future__ import annotations

import glob
import pathlib
import re
import subprocess
import sys
from datetime import datetime

import importlib.util
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="System Failure Pattern Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ───────────────────────────────────────────────────────────
CLR = dict(
    bg="#0f1117", card="#1a1d27", border="#2a2d3e",
    accent="#6c63ff", accent2="#ff6584", accent3="#43e8d8",
    warning="#fbbf24", danger="#ef4444", success="#22c55e",
    text="#e2e8f0", muted="#94a3b8",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
    background-color: {CLR['bg']}; color: {CLR['text']};
  }}
  [data-testid="stSidebar"] {{
    background-color: {CLR['card']};
    border-right: 1px solid {CLR['border']};
  }}
  .metric-card {{
    background: {CLR['card']}; border: 1px solid {CLR['border']};
    border-radius: 12px; padding: 18px 22px; text-align: center;
    transition: transform 0.2s;
  }}
  .metric-card .value {{ font-size: 2.2rem; font-weight: 700; }}
  .metric-card .label {{ font-size: 0.82rem; color: {CLR['muted']}; margin-top: 4px; }}
  .section-title {{
    font-size: 1.05rem; font-weight: 600; color: {CLR['text']};
    border-left: 3px solid {CLR['accent']}; padding-left: 10px;
    margin: 16px 0 10px;
  }}
  .run-badge {{
    display: inline-block; padding: 2px 9px; border-radius: 10px;
    font-size: 0.72rem; font-weight: 600; margin: 2px;
  }}
  .run-badge-ap  {{ background: rgba(108,99,255,0.15); color: {CLR['accent']}; border: 1px solid {CLR['accent']}; }}
  .run-badge-fp  {{ background: rgba(67,232,216,0.15); color: {CLR['accent3']}; border: 1px solid {CLR['accent3']}; }}
  .dirty-banner {{
    background: rgba(251,191,36,0.10); border: 1px solid {CLR['warning']};
    border-radius: 10px; padding: 12px 18px; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }}
  div[data-testid="stPlotlyChart"] {{ border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

# ── Path resolution ──────────────────────────────────────────────────────────
_here       = pathlib.Path(__file__).resolve().parent
_sub        = _here / "System-Failure-Pattern-Analysis-feature-algorithms"
BASE        = _sub if (_sub / "data").exists() else _here
CONFIG_PATH = BASE / "config.py"
DATA_CSV    = BASE / "data" / "OpenStack_2k.log_structured.csv"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC CONFIG LOADER — reads config.py fresh on every Streamlit rerun
# ═══════════════════════════════════════════════════════════════════════════════
def _load_config_defaults() -> dict:
    try:
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location("_dyn_cfg", CONFIG_PATH)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {
            "MIN_SUPPORT":            getattr(mod, "MIN_SUPPORT",            0.15),
            "MIN_CONFIDENCE":         getattr(mod, "MIN_CONFIDENCE",         0.50),
            "MIN_LIFT":               getattr(mod, "MIN_LIFT",               1.05),
            "MAX_ITEMSET_LEN":        getattr(mod, "MAX_ITEMSET_LEN",        3),
            "TIME_WINDOW_SECONDS":    getattr(mod, "TIME_WINDOW_SECONDS",    5),
            "SESSION_WINDOW_SECONDS": getattr(mod, "SESSION_WINDOW_SECONDS", 300),
            "MIN_EVENTS_PER_WINDOW":  getattr(mod, "MIN_EVENTS_PER_WINDOW",  2),
            "TOP_N_PATTERNS":         getattr(mod, "TOP_N_PATTERNS",         20),
            "ANOMALY_THRESHOLD":      getattr(mod, "ANOMALY_THRESHOLD",      2.5),
            "FAILURE_LEVELS":         getattr(mod, "FAILURE_LEVELS",         {"ERROR","CRITICAL","TRACE","WARNING"}),
            "CRITICAL_LEVELS":        getattr(mod, "CRITICAL_LEVELS",        {"ERROR","CRITICAL"}),
            "CRITICAL_COMPONENTS":    getattr(mod, "CRITICAL_COMPONENTS",    set()),
            "SEVERITY_WEIGHTS":       getattr(mod, "SEVERITY_WEIGHTS",       {}),
            "MAX_FREQUENT_ITEMSETS":  getattr(mod, "MAX_FREQUENT_ITEMSETS",  50_000),
        }
    except Exception:
        return {
            "MIN_SUPPORT": 0.15, "MIN_CONFIDENCE": 0.50, "MIN_LIFT": 1.05,
            "MAX_ITEMSET_LEN": 3, "TIME_WINDOW_SECONDS": 5,
            "SESSION_WINDOW_SECONDS": 300, "MIN_EVENTS_PER_WINDOW": 2,
            "TOP_N_PATTERNS": 20, "ANOMALY_THRESHOLD": 2.5,
            "FAILURE_LEVELS": {"ERROR","CRITICAL","TRACE","WARNING"},
            "CRITICAL_LEVELS": {"ERROR","CRITICAL"}, "CRITICAL_COMPONENTS": set(),
            "SEVERITY_WEIGHTS": {}, "MAX_FREQUENT_ITEMSETS": 50_000,
        }

CFG = _load_config_defaults()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def patch_config_py(updates: dict) -> tuple[bool, str]:
    """
    Safely patch numeric values in config.py using line-level regex.
    Preserves comments and all other content.
    Returns (success, message).
    """
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        applied = []
        for key, val in updates.items():
            pattern = rf'^({re.escape(key)}\s*=\s*)([0-9][0-9_]*(?:\.[0-9]+)?)'
            new_str = f"{val:.6g}" if isinstance(val, float) else str(int(val))
            new_text, n = re.subn(
                pattern, rf'\g<1>{new_str}', text,
                count=1, flags=re.MULTILINE
            )
            if n:
                text = new_text
                applied.append(key)
        CONFIG_PATH.write_text(text, encoding="utf-8")
        return True, "Updated: " + ", ".join(applied) if applied else "Nothing changed"
    except Exception as exc:
        return False, str(exc)


def run_mining_pipeline(algo: str, strategy: str, params: dict | None = None) -> tuple[int, str]:
    """
    Invoke run_mining.py as a subprocess.

    Passes ALL threshold params BOTH ways:
      1. config.py is already patched on disk  (run_mining.py imports it at startup)
      2. CLI args --support / --confidence / --lift / --max-len are passed explicitly
         (belt-and-suspenders — handles any import-time caching edge-cases)

    Windows Unicode fix:
      Sets PYTHONIOENCODING=utf-8 in the subprocess environment so that
      mining_base.print_report()'s box-drawing characters (─ ≥ →) do not
      crash the process on cp1252 consoles.  run_mining.py also calls
      sys.stdout.reconfigure(encoding="utf-8") for direct CLI invocations.

    Returns (returncode, combined_stdout_stderr).
    """
    import os as _os
    algo_arg = {"Apriori": "apriori", "FP-Growth": "fpgrowth", "Both": "both"}.get(algo, "both")
    p = params or {}

    # Resolve each param: prefer explicit dict, fall back to current CFG
    sup  = str(p.get("MIN_SUPPORT",            CFG["MIN_SUPPORT"]))
    conf = str(p.get("MIN_CONFIDENCE",         CFG["MIN_CONFIDENCE"]))
    lift = str(p.get("MIN_LIFT",               CFG["MIN_LIFT"]))
    mlen = str(int(p.get("MAX_ITEMSET_LEN",    CFG["MAX_ITEMSET_LEN"])))

    cmd = [
        sys.executable, str(BASE / "run_mining.py"), str(DATA_CSV),
        "--algorithm",  algo_arg,
        "--strategy",   strategy,
        "--support",    sup,
        "--confidence", conf,
        "--lift",       lift,
        "--max-len",    mlen,
        "--no-compare",
    ]

    # Force UTF-8 I/O in the child process — prevents UnicodeEncodeError on
    # Windows cp1252 consoles when mining_base prints box-drawing characters.
    env = _os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"          # Python 3.7+ alternative / redundancy

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(BASE), env=env, encoding="utf-8", errors="replace",
        )
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        return result.returncode, combined.strip()
    except subprocess.TimeoutExpired:
        return -1, "Mining timed out (>300 s). Try raising min_support."
    except Exception as exc:
        return -1, str(exc)


def rebuild_reports(algo: str, strategy: str, params: dict) -> tuple[bool, str]:
    """
    Orchestrate the FULL rebuild pipeline:

        patch_config_py(params)          ← write new values to config.py
              ↓
        run_mining_pipeline(...)         ← Apriori / FP-Growth subprocess
              ↓
        New CSVs + summary.txt saved     ← handled inside run_mining.py
              ↓
        return (True, output)            ← caller does cache_clear + rerun

    The caller is responsible for calling  st.cache_data.clear()  and
    st.rerun()  after a successful return — that causes Streamlit to
    re-execute the whole script and display the new charts.

    Parameters
    ----------
    algo     : "Apriori" | "FP-Growth" | "Both"
    strategy : "time_window" | "session"
    params   : dict with keys MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT,
               MAX_ITEMSET_LEN, TIME_WINDOW_SECONDS, SESSION_WINDOW_SECONDS,
               MIN_EVENTS_PER_WINDOW  (numeric values)

    Returns
    -------
    (success: bool, message: str)
    """
    # ── Step 1: patch config.py ──────────────────────────────────────────────
    ok, cfg_msg = patch_config_py(params)
    if not ok:
        return False, f"Config patch failed: {cfg_msg}"

    # ── Step 2: run mining subprocess ────────────────────────────────────────
    rc, pipeline_out = run_mining_pipeline(algo, strategy, params)

    if rc == 0:
        return True, pipeline_out
    return False, pipeline_out


# ── Report / run discovery ───────────────────────────────────────────────────

def get_available_runs() -> list[dict]:
    """Scan reports/ and return metadata dicts sorted newest-first."""
    runs = []
    for f in REPORTS_DIR.glob("*_summary.txt"):
        m = re.match(r"(\d{8}_\d{6})_summary\.txt$", f.name)
        if not m:
            continue
        stamp = m.group(1)
        has_ap = (REPORTS_DIR / f"{stamp}_apriori_rules.csv").exists()
        has_fp = (REPORTS_DIR / f"{stamp}_fpgrowth_rules.csv").exists()
        try:
            dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            label = dt.strftime("%b %d, %Y  %H:%M:%S")
        except ValueError:
            label = stamp
        runs.append({
            "stamp": stamp, "label": label,
            "has_apriori": has_ap, "has_fpgrowth": has_fp,
        })
    return sorted(runs, key=lambda r: r["stamp"], reverse=True)


def parse_run_meta_from_summary(text: str) -> dict:
    """Extract key stats from a summary.txt for the history table."""
    meta = {"n_transactions": None, "n_items": None,
            "ap_elapsed": None, "fp_elapsed": None,
            "ap_rules": None, "fp_rules": None,
            "ap_itemsets": None, "fp_itemsets": None}
    txn_m = re.search(r"TransactionDB\s*:\s*(\d+)\s*transactions\s*/\s*(\d+)\s*unique items", text)
    if txn_m:
        meta["n_transactions"] = int(txn_m.group(1))
        meta["n_items"]        = int(txn_m.group(2))
    for algo_key, prefix in [("ap", "APRIORI"), ("fp", "FPGROWTH")]:
        block = re.search(rf'\[{prefix}\](.*?)(?=\[|\Z)', text, re.DOTALL)
        if block:
            blk = block.group(1)
            e = re.search(r"Elapsed\s*:\s*([\d.]+)s", blk)
            if e: meta[f"{algo_key}_elapsed"] = float(e.group(1))
            fi = re.search(r"Frequent itemsets\s*:\s*(\d+)", blk)
            if fi: meta[f"{algo_key}_itemsets"] = int(fi.group(1))
            ru = re.search(r"Rules generated\s*:\s*(\d+)", blk)
            if ru: meta[f"{algo_key}_rules"] = int(ru.group(1))
    return meta


# ── Data loaders (cached) ────────────────────────────────────────────────────

@st.cache_data
def load_log_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["Time_dt"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
    )
    df["Hour"]   = df["Time_dt"].dt.hour
    df["Minute"] = df["Time_dt"].dt.minute
    return df


@st.cache_data
def load_run_data(stamp: str) -> tuple:
    """Load all CSV files for a specific report run. Empty DF if file missing."""
    def _try(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    def _read(path):
        return path.read_text(encoding="utf-8") if path.exists() else ""
    r = REPORTS_DIR
    return (
        _try(r / f"{stamp}_apriori_frequent_itemsets.csv"),
        _try(r / f"{stamp}_apriori_rules.csv"),
        _try(r / f"{stamp}_fpgrowth_frequent_itemsets.csv"),
        _try(r / f"{stamp}_fpgrowth_rules.csv"),
        _read(r / f"{stamp}_summary.txt"),
    )


@st.cache_data
def load_all_summaries(stamps: tuple) -> list[dict]:
    """Load and parse summary.txt for every run (for history table)."""
    rows = []
    for stamp in stamps:
        summary = (REPORTS_DIR / f"{stamp}_summary.txt").read_text(encoding="utf-8") \
                  if (REPORTS_DIR / f"{stamp}_summary.txt").exists() else ""
        m = parse_run_meta_from_summary(summary)
        m["stamp"] = stamp
        try:
            m["label"] = datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%b %d  %H:%M:%S")
        except ValueError:
            m["label"] = stamp
        rows.append(m)
    return rows


# ── Small helpers ─────────────────────────────────────────────────────────────

def _prep_fi(fi_df: pd.DataFrame) -> pd.DataFrame:
    if fi_df.empty:
        return fi_df
    fi = fi_df.copy()

    def _item_type(s):
        if s.startswith("EVT"):   return "Event"
        if s.startswith("COMP"):  return "Component"
        if s.startswith("FLAG"):  return "Flag"
        if s.startswith("HOUR"):  return "Hour"
        if s.startswith("LEVEL"): return "Level"
        return "Other"

    fi["type"] = fi["itemsets"].apply(_item_type)
    fi["size"] = fi["itemsets"].apply(lambda x: len(str(x).split("|")))
    return fi


def _filter_rules(df: pd.DataFrame, lift: float, conf: float, sup: float) -> pd.DataFrame:
    if df.empty or not {"lift","confidence","support"}.issubset(df.columns):
        return pd.DataFrame()
    return df[(df["lift"] >= lift) & (df["confidence"] >= conf) & (df["support"] >= sup)]


def _derive_n_transactions(fi_df: pd.DataFrame) -> int:
    if fi_df.empty or "support" not in fi_df.columns:
        return 0
    for n in range(50, 2000):
        if ((fi_df["support"] * n).round() - fi_df["support"] * n).abs().max() < 0.015:
            return n
    return max(1, round(1 / fi_df["support"].min()))


def _unique_items(fi_df: pd.DataFrame) -> int:
    if fi_df.empty:
        return 0
    s = set()
    for row in fi_df["itemsets"]:
        for item in str(row).split("|"):
            s.add(item.strip())
    return len(s)


def _node_color(label: str) -> str:
    if label.startswith("EVT"):  return CLR['accent']
    if label.startswith("COMP"): return CLR['accent3']
    if label.startswith("FLAG"): return CLR['danger']
    if label.startswith("HOUR"): return CLR['accent2']
    return CLR['warning']


def _plotly_base(fig, height=None):
    kw = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font_color=CLR['text'],
              xaxis=dict(gridcolor=CLR['border']),
              yaxis=dict(gridcolor=CLR['border']),
              legend=dict(font=dict(color=CLR['text'])),
              margin=dict(t=12, b=12, l=12, r=12))
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for _k, _v in [("mining_status","idle"), ("mining_output",""), ("selected_run",None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Control Panel")

    # ── Section 1: Mining Parameters ─────────────────────────────────────
    with st.expander("⚙️ Mining Parameters", expanded=True):
        algo_choice = st.radio(
            "Algorithm", ["Apriori", "FP-Growth", "Both"], index=2,
            horizontal=True, key="algo_choice",
        )
        strategy_choice = st.radio(
            "Strategy", ["time_window", "session"], index=0,
            horizontal=True, key="strategy_choice",
        )

        st.markdown("**Thresholds**")
        new_support = st.slider(
            "Min Support", 0.01, 0.50, float(CFG["MIN_SUPPORT"]), 0.01,
            help="Fraction of transactions an itemset must appear in. "
                 "Lower = more patterns found, slower run.",
        )
        new_confidence = st.slider(
            "Min Confidence", 0.30, 1.00, float(CFG["MIN_CONFIDENCE"]), 0.05,
            help="P(consequent | antecedent) threshold for rule generation.",
        )
        new_lift = st.slider(
            "Min Lift", 1.00, 6.00, float(CFG["MIN_LIFT"]), 0.05,
            help="Lift > 1 means items co-occur more than by chance.",
        )
        new_max_len = st.slider(
            "Max Itemset Length", 2, 5, int(CFG["MAX_ITEMSET_LEN"]), 1,
            help="Max number of items in a frequent itemset.",
        )

        st.markdown("**Transaction Building**")
        new_time_window = st.slider(
            "Time Window (s)", 5, 300, int(CFG["TIME_WINDOW_SECONDS"]), 5,
            help="Width of each time-window transaction in seconds.",
        )
        new_session_window = st.slider(
            "Session Window (s)", 60, 900, int(CFG["SESSION_WINDOW_SECONDS"]), 30,
            help="Max idle gap before a new session starts.",
        )
        new_min_events = st.number_input(
            "Min Events / Window", 1, 20, int(CFG["MIN_EVENTS_PER_WINDOW"]), 1,
            help="Minimum log entries required for a window to become a transaction.",
        )

        # Collect all proposed param changes into one dict
        _proposed = {
            "MIN_SUPPORT":            new_support,
            "MIN_CONFIDENCE":         new_confidence,
            "MIN_LIFT":               new_lift,
            "MAX_ITEMSET_LEN":        new_max_len,
            "TIME_WINDOW_SECONDS":    new_time_window,
            "SESSION_WINDOW_SECONDS": new_session_window,
            "MIN_EVENTS_PER_WINDOW":  new_min_events,
        }

        # Detect which keys differ from what's currently in config.py
        _changed_keys = [
            k for k, v in _proposed.items()
            if abs(float(v) - float(CFG.get(k, v))) > 1e-9
        ]

        if _changed_keys:
            st.warning(
               f"⚠️ Unsaved changes: `{'`  `'.join(_changed_keys)}`\n\n"
    "Click **⚡ Run Mining** to apply & regenerate reports."
            )

        btn_run = st.button(
            "⚡ Run Mining", use_container_width=True, type="primary",
            help="Writes params to config.py, then runs the full pipeline.",
        )

    # ── Section 2: Report History Selector ───────────────────────────────
    st.markdown("---")
    st.markdown("### 📂 Report History")

    available_runs = get_available_runs()

    if not available_runs:
        st.info("No reports yet. Set parameters above and click **Run Mining**.")
        _selected_stamp = None
    else:
        _run_labels = [
            ("🟢 Latest  " if i == 0 else "  ") + r["label"]
            for i, r in enumerate(available_runs)
        ]
        _run_stamps = [r["stamp"] for r in available_runs]

        _cur_idx = 0
        if st.session_state.selected_run in _run_stamps:
            _cur_idx = _run_stamps.index(st.session_state.selected_run)

        _sel_idx = st.selectbox(
            f"{len(available_runs)} run(s) cached",
            range(len(_run_labels)),
            format_func=lambda i: _run_labels[i],
            index=_cur_idx,
        )
        st.session_state.selected_run = _run_stamps[_sel_idx]
        _selected_stamp = _run_stamps[_sel_idx]

        _ri = available_runs[_sel_idx]
        badge_html = ""
        if _ri["has_apriori"]:  badge_html += "<span class='run-badge run-badge-ap'>Apriori</span>"
        if _ri["has_fpgrowth"]: badge_html += "<span class='run-badge run-badge-fp'>FP-Growth</span>"
        st.markdown(badge_html, unsafe_allow_html=True)

    # ── Section 3: Display Filters ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔭 Display Filters")
    st.caption("These only filter what's shown — they don't re-run mining.")
    disp_algo = st.radio(
        "View Algorithm", ["Apriori", "FP-Growth", "Both"], index=2, key="disp_algo"
    )
    disp_lift = st.slider("Min Lift",       1.0, 6.0, float(CFG["MIN_LIFT"]),       0.05, key="disp_lift")
    disp_conf = st.slider("Min Confidence", 0.3, 1.0, float(CFG["MIN_CONFIDENCE"]), 0.05, key="disp_conf")
    disp_sup  = st.slider("Min Support",    0.0, 1.0, float(CFG["MIN_SUPPORT"]),    0.01, key="disp_sup")
    disp_topn = st.slider("Top-N Rules",    5,   50,  int(CFG["TOP_N_PATTERNS"]),         key="disp_topn")

    # ── Section 4: Live Config Summary ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 config.py (current)")
    for _k, _v in [
        ("MIN_SUPPORT",         CFG["MIN_SUPPORT"]),
        ("MIN_CONFIDENCE",      CFG["MIN_CONFIDENCE"]),
        ("MIN_LIFT",            CFG["MIN_LIFT"]),
        ("MAX_ITEMSET_LEN",     CFG["MAX_ITEMSET_LEN"]),
        ("TIME_WINDOW_SECONDS", CFG["TIME_WINDOW_SECONDS"]),
    ]:
        _marker = "🟡 " if _k in _changed_keys else ""
        st.markdown(f"{_marker}`{_k}` = `{_v}`")


# ═══════════════════════════════════════════════════════════════════════════════
# DIRTY-STATE BANNER  (main page — visible without opening sidebar)
# ═══════════════════════════════════════════════════════════════════════════════
# When the user changes any slider, a prominent "Rebuild Report" button
# appears at the top of the main page so they don't have to hunt for it.


# ═══════════════════════════════════════════════════════════════════════════════
# MINING EXECUTION
# Triggered by EITHER the sidebar "⚡ Run Mining" button
#              OR the main-page "🔄 Rebuild Report" button.
#
# Full flow:
#   rebuild_reports(algo, strategy, _proposed)
#       └─ patch_config_py(_proposed)       → config.py updated on disk
#       └─ run_mining_pipeline(algo, …)     → subprocess runs run_mining.py
#              ├─ Apriori mines transactions
#              └─ FP-Growth mines transactions
#              └─ CSVs + summary.txt saved to reports/
#   st.cache_data.clear()                   → bust @cache_data caches
#   st.session_state.selected_run = None    → snap to latest run
#   st.rerun()                              → Streamlit re-executes top-to-bottom
#                                             → new charts loaded from new CSVs
# ═══════════════════════════════════════════════════════════════════════════════
if btn_run:
    with st.spinner(
        f"⚡ Running **{algo_choice}** mining  ·  strategy={strategy_choice}  "
        "·  This may take up to a minute…"
    ):
        success, output = rebuild_reports(algo_choice, strategy_choice, _proposed)

    st.session_state.mining_output = output

    if success:
        st.session_state.mining_status = "success"
        st.session_state.selected_run  = None   # snap to latest run
        st.cache_data.clear()                   # bust ALL @cache_data caches
        st.rerun()                              # re-execute → new charts appear
    else:
        st.session_state.mining_status = "error"


# ── Status banners ────────────────────────────────────────────────────────────
if st.session_state.mining_status == "success":
    st.success("✅ Mining completed — dashboard now shows the latest report.")
    st.session_state.mining_status = "idle"    # show banner only once
elif st.session_state.mining_status == "error":
    st.error("❌ Mining failed. Check the output below.")
    if st.session_state.mining_output:
        with st.expander("🔍 Mining output / error log"):
            st.code(st.session_state.mining_output)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
log = load_log_df()

# Apply config-driven columns to log
_FL  = CFG["FAILURE_LEVELS"]
_CL  = CFG["CRITICAL_LEVELS"]
_CC  = CFG["CRITICAL_COMPONENTS"]
_SW  = CFG["SEVERITY_WEIGHTS"]
log["is_failure"]       = log["Level"].isin(_FL)
log["is_critical"]      = log["Level"].isin(_CL)
log["severity_score"]   = log["Level"].map(_SW).fillna(0)
log["ShortComp"]        = log["Component"].str.split(".").str[-1]
log["is_critical_comp"] = log["Component"].isin(_CC)

# Guard: no reports at all
available_runs = get_available_runs()   # refresh after potential rerun
if not available_runs:
    st.markdown("""
    <div style='text-align:center; padding:80px; color:#94a3b8;'>
      <h2>🚀 No mining reports yet</h2>
      <p>Configure parameters in the sidebar and click <b>⚡ Run Mining</b> to generate your first report.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# Load selected run's data
_sel_stamp = st.session_state.selected_run or available_runs[0]["stamp"]
ap_fi_raw, ap_rules_raw, fp_fi_raw, fp_rules_raw, run_summary = load_run_data(_sel_stamp)

ap_fi    = _prep_fi(ap_fi_raw)
fp_fi    = _prep_fi(fp_fi_raw)
ap_rules = ap_rules_raw.copy() if not ap_rules_raw.empty else pd.DataFrame()
fp_rules = fp_rules_raw.copy() if not fp_rules_raw.empty else pd.DataFrame()

# Parse elapsed from summary
run_meta = parse_run_meta_from_summary(run_summary)
AP_ELAPSED = run_meta.get("ap_elapsed")
FP_ELAPSED = run_meta.get("fp_elapsed")
N_TRANSACTIONS = (run_meta.get("n_transactions")
                  or _derive_n_transactions(ap_fi if not ap_fi.empty else fp_fi))
N_UNIQUE_ITEMS = (run_meta.get("n_items")
                  or _unique_items(ap_fi if not ap_fi.empty else fp_fi))
fmt_e = lambda v: f"{v:.4f} s" if v is not None else "—"

# Apply display filters
ap_f = _filter_rules(ap_rules, disp_lift, disp_conf, disp_sup)
fp_f = _filter_rules(fp_rules, disp_lift, disp_conf, disp_sup)

if disp_algo == "Apriori":
    active_rules = ap_f
elif disp_algo == "FP-Growth":
    active_rules = fp_f
else:
    parts = []
    if not ap_f.empty: parts.append(ap_f.assign(algo="Apriori"))
    if not fp_f.empty: parts.append(fp_f.assign(algo="FP-Growth"))
    active_rules = pd.concat(parts) if parts else pd.DataFrame()

_fi = (ap_fi if disp_algo == "Apriori" else
       fp_fi if disp_algo == "FP-Growth" else
       pd.concat([ap_fi, fp_fi]).drop_duplicates() if not ap_fi.empty else fp_fi)

_n_rules  = len(active_rules)
_max_lift = f"{active_rules['lift'].max():.2f}"        if _n_rules > 0 and "lift" in active_rules.columns else "—"
_avg_conf = f"{active_rules['confidence'].mean():.2f}" if _n_rules > 0 and "confidence" in active_rules.columns else "—"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='text-align:center; font-size:2rem; font-weight:700;
           background:linear-gradient(90deg,#6c63ff,#43e8d8);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           margin-bottom:4px;'>
  🛡️ System Failure Pattern Analysis
</h1>
<p style='text-align:center; color:#94a3b8; font-size:0.88rem; margin-bottom:2px;'>
  Apriori &amp; FP-Growth mining · OpenStack failure logs
</p>
<p style='text-align:center; font-size:0.78rem; margin-bottom:18px;'>
  <span style='color:{CLR["muted"]};'>Viewing run:</span>
  <code style='color:{CLR["accent3"]};'>{_sel_stamp}</code>
  &nbsp;·&nbsp;
  <span style='color:{CLR["muted"]};'>{len(available_runs)} report(s) cached</span>
</p>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
for col, val, lbl, clr in [
    (k1, len(log),                    "Log Entries",     CLR["accent"]),
    (k2, int(log["is_failure"].sum()), "Failure Events", CLR["danger"]),
    (k3, N_TRANSACTIONS,              "Transactions",    CLR["accent3"]),
    (k4, _n_rules,                    "Filtered Rules",  CLR["warning"]),
    (k5, _max_lift,                   "Max Lift",        CLR["success"]),
    (k6, _avg_conf,                   "Avg Confidence",  CLR["accent2"]),
]:
    with col:
        dv = f"{val:,}" if isinstance(val, int) else val
        st.markdown(f"""
        <div class='metric-card'>
          <div class='value' style='color:{clr}'>{dv}</div>
          <div class='label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Log Overview",
    "📦 Frequent Itemsets",
    "🔗 Association Rules",
    "🕸️ Rule Network",
    "⚖️ Algorithm Comparison",
    "📚 Report History",
])


# ─── TAB 1 — LOG OVERVIEW ────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>Log Level Distribution</div>", unsafe_allow_html=True)
        lv = log["Level"].value_counts().reset_index()
        lv.columns = ["Level", "Count"]
        fig = px.pie(lv, names="Level", values="Count",
                     color_discrete_sequence=[CLR["accent"], CLR["warning"], CLR["danger"], CLR["accent3"]],
                     hole=0.55)
        fig.update_traces(textfont_color=CLR["text"])
        st.plotly_chart(_plotly_base(fig), use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Events per Minute (Timeline)</div>", unsafe_allow_html=True)
        timeline = log.groupby("Minute").agg(
            total=("LineId","count"), failures=("is_failure","sum")
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timeline["Minute"], y=timeline["total"], mode="lines",
                                  name="All Events", line=dict(color=CLR["accent"], width=2),
                                  fill="tozeroy", fillcolor="rgba(108,99,255,0.12)"))
        fig2.add_trace(go.Scatter(x=timeline["Minute"], y=timeline["failures"], mode="lines",
                                  name="Failures", line=dict(color=CLR["warning"], width=2, dash="dot")))
        fig2.update_layout(xaxis_title="Minute of Day", yaxis_title="Count")
        st.plotly_chart(_plotly_base(fig2), use_container_width=True)

    st.markdown("<div class='section-title'>Top Components by Log Volume</div>", unsafe_allow_html=True)
    comp = log["Component"].value_counts().head(15).reset_index()
    comp.columns = ["Component", "Count"]
    comp["ShortComp"] = comp["Component"].str.split(".").str[-1]
    fail_comp = log[log["is_failure"]].groupby("Component").size().rename("Failures")
    comp = comp.merge(fail_comp, on="Component", how="left").fillna(0)
    comp["Failures"] = comp["Failures"].astype(int)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(y=comp["ShortComp"], x=comp["Count"],    name="All",      orientation="h", marker_color=CLR["accent"]))
    fig3.add_trace(go.Bar(y=comp["ShortComp"], x=comp["Failures"], name="Failures", orientation="h", marker_color=CLR["danger"]))
    fig3.update_layout(barmode="overlay", xaxis_title="Count", height=380)
    st.plotly_chart(_plotly_base(fig3), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='section-title'>Top Event IDs</div>", unsafe_allow_html=True)
        evts = log["EventId"].value_counts().head(20).reset_index()
        evts.columns = ["EventId", "Count"]
        fig4 = px.bar(evts, x="EventId", y="Count", color="Count",
                      color_continuous_scale="Plasma", labels={"Count": "Occurrences"})
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_plotly_base(fig4), use_container_width=True)

    with c4:
        st.markdown("<div class='section-title'>Severity Score by Component × Level</div>", unsafe_allow_html=True)
        sev_pivot = (log.groupby(["ShortComp","Level"])["severity_score"]
                     .sum().unstack(fill_value=0).head(12))
        fig5 = px.imshow(sev_pivot, color_continuous_scale="Reds", aspect="auto",
                         labels=dict(x="Level", y="Component", color="Severity"))
        fig5.update_layout(
            coloraxis_colorbar=dict(tickfont=dict(color=CLR["text"]),
                                    title=dict(font=dict(color=CLR["text"]))),
            height=380)
        st.plotly_chart(_plotly_base(fig5), use_container_width=True)


# ─── TAB 2 — FREQUENT ITEMSETS ───────────────────────────────────────────────
with tab2:
    if ap_fi.empty and fp_fi.empty:
        st.info("No frequent itemset data for this run.")
    else:
        _ref_fi = ap_fi if not ap_fi.empty else fp_fi

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-title'>Itemset Type Distribution</div>", unsafe_allow_html=True)
            tc = _ref_fi.groupby("type")["support"].count().reset_index()
            tc.columns = ["Type","Count"]
            fig = px.pie(tc, names="Type", values="Count", hole=0.5,
                         color_discrete_sequence=[CLR["accent"],CLR["accent3"],CLR["warning"],CLR["danger"],CLR["accent2"]])
            fig.update_traces(textfont_color=CLR["text"])
            st.plotly_chart(_plotly_base(fig), use_container_width=True)

        with c2:
            st.markdown("<div class='section-title'>Itemset Size Distribution</div>", unsafe_allow_html=True)
            sd = _ref_fi["size"].value_counts().reset_index().sort_values("size")
            sd.columns = ["Size","Count"]
            fig = px.bar(sd, x="Size", y="Count", color="Count", color_continuous_scale="Viridis",
                         labels={"Size": f"Itemset Size (max={CFG['MAX_ITEMSET_LEN']})"})
            fig.add_vline(x=CFG["MAX_ITEMSET_LEN"], line_dash="dash", line_color=CLR["warning"],
                          annotation_text=f"max={CFG['MAX_ITEMSET_LEN']}", annotation_font_color=CLR["warning"])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_plotly_base(fig), use_container_width=True)

        st.markdown("<div class='section-title'>Top 25 Frequent Itemsets by Support</div>", unsafe_allow_html=True)
        top_fi = _ref_fi.sort_values("support", ascending=False).head(25)
        fig = px.bar(top_fi, x="support", y="itemsets", orientation="h", color="type",
                     color_discrete_map={"Event":CLR["accent"],"Component":CLR["accent3"],
                                         "Flag":CLR["warning"],"Hour":CLR["accent2"],"Level":CLR["danger"]},
                     labels={"support":"Support","itemsets":"Itemset"})
        fig.add_vline(x=CFG["MIN_SUPPORT"], line_dash="dash", line_color=CLR["success"],
                      annotation_text=f"min_sup={CFG['MIN_SUPPORT']}", annotation_font_color=CLR["success"])
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        st.plotly_chart(_plotly_base(fig), use_container_width=True)

        st.markdown("<div class='section-title'>Support Distribution by Item Type</div>", unsafe_allow_html=True)
        fig = px.violin(_ref_fi, x="type", y="support", color="type", box=True, points="all",
                        color_discrete_map={"Event":CLR["accent"],"Component":CLR["accent3"],
                                            "Flag":CLR["warning"],"Hour":CLR["accent2"],"Level":CLR["danger"]})
        fig.update_layout(showlegend=False)
        st.plotly_chart(_plotly_base(fig), use_container_width=True)


# ─── TAB 3 — ASSOCIATION RULES ────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>Support vs. Confidence (colour = Lift)</div>", unsafe_allow_html=True)
    if active_rules.empty:
        st.info("No rules match the current display filters. Try lowering thresholds in the sidebar.")
    else:
        disp = active_rules.head(400).copy()
        fig = px.scatter(disp, x="support", y="confidence",
                         color="lift", size=disp["lift"].clip(lower=1),
                         color_continuous_scale="Plasma",
                         hover_data=["antecedents","consequents","lift","support","confidence"])
        fig.add_hline(y=disp_conf, line_dash="dot", line_color=CLR["accent2"],
                      annotation_text=f"min_conf={disp_conf}", annotation_font_color=CLR["accent2"])
        fig.add_vline(x=disp_sup, line_dash="dot", line_color=CLR["success"],
                      annotation_text=f"min_sup={disp_sup}", annotation_font_color=CLR["success"])
        fig.update_layout(coloraxis_colorbar=dict(
            title=dict(text="Lift", font=dict(color=CLR["text"])),
            tickfont=dict(color=CLR["text"])))
        st.plotly_chart(_plotly_base(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Top Rules by Lift</div>", unsafe_allow_html=True)
        if not active_rules.empty:
            top_rules = active_rules.nlargest(disp_topn, "lift").copy()
            top_rules["rule"] = (top_rules["antecedents"].str[:28] + " → " + top_rules["consequents"].str[:28])
            fig = px.bar(top_rules, x="lift", y="rule", orientation="h",
                         color="confidence", color_continuous_scale="Teal")
            fig.add_vline(x=disp_lift, line_dash="dash", line_color=CLR["warning"],
                          annotation_text=f"min_lift={disp_lift}", annotation_font_color=CLR["warning"])
            fig.update_layout(height=max(280, disp_topn * 28),
                               yaxis=dict(autorange="reversed"),
                               coloraxis_colorbar=dict(title=dict(text="Conf", font=dict(color=CLR["text"])),
                                                       tickfont=dict(color=CLR["text"])))
            st.plotly_chart(_plotly_base(fig), use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Lift Distribution Histogram</div>", unsafe_allow_html=True)
        ref_rules = ap_rules if not ap_rules.empty else fp_rules
        if not ref_rules.empty and "lift" in ref_rules.columns:
            fig = px.histogram(ref_rules, x="lift", nbins=40,
                               color_discrete_sequence=[CLR["accent"]],
                               labels={"lift":"Lift","count":"# Rules"})
            fig.add_vline(x=disp_lift, line_dash="dash", line_color=CLR["warning"],
                          annotation_text=f"Filter: {disp_lift}", annotation_font_color=CLR["warning"])
            st.plotly_chart(_plotly_base(fig), use_container_width=True)

    # Rule metrics heatmap
    if not ap_f.empty:
        st.markdown("<div class='section-title'>Rule Metrics Heatmap (Top-40 by Lift)</div>", unsafe_allow_html=True)
        hr = ap_f.nlargest(40, "lift")[["antecedents","consequents","support","confidence","lift","leverage","conviction"]].copy()
        hr["rule"] = hr["antecedents"].str[:20] + "→" + hr["consequents"].str[:20]
        seen, unique_labels = {}, []
        for lbl in hr["rule"]:
            seen[lbl] = seen.get(lbl, 0) + 1
            unique_labels.append(lbl if seen[lbl] == 1 else f"{lbl}({seen[lbl]})")
        hr["rule"] = unique_labels
        heat = hr.set_index("rule")[["support","confidence","lift","leverage","conviction"]]
        fig = px.imshow(heat.T, color_continuous_scale="Plasma", aspect="auto",
                        labels=dict(x="Rule", y="Metric", color="Value"))
        fig.update_layout(height=260, xaxis=dict(tickfont=dict(size=8)),
                          coloraxis_colorbar=dict(tickfont=dict(color=CLR["text"]),
                                                  title=dict(font=dict(color=CLR["text"]))),
                          margin=dict(t=12, b=80, l=12, r=12))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color=CLR["text"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Filtered Rules Table</div>", unsafe_allow_html=True)
    if not active_rules.empty:
        show_cols = [c for c in ["antecedents","consequents","support","confidence","lift","leverage","conviction"]
                     if c in active_rules.columns]
        st.dataframe(
            active_rules[show_cols].nlargest(50,"lift").reset_index(drop=True).round(4),
            use_container_width=True, height=300
        )


# ─── TAB 4 — RULE NETWORK ─────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>Association Rule Network Graph</div>", unsafe_allow_html=True)
    st.caption("Nodes = items/events  ·  Edges = association rules  ·  Node size = degree")
    n_net = st.slider("Rules to visualise", 10, 100, 40, key="net_slider")
    net_rules = active_rules.nlargest(n_net, "lift") if not active_rules.empty else pd.DataFrame()

    if net_rules.empty:
        st.info("No rules with current filters.")
    else:
        G = nx.DiGraph()
        for _, row in net_rules.iterrows():
            src = str(row["antecedents"])[:30]
            dst = str(row["consequents"])[:30]
            G.add_edge(src, dst, lift=row["lift"])

        if G.nodes:
            pos = nx.spring_layout(G, seed=42, k=2.5)
            ex, ey = [], []
            for u, v in G.edges():
                x0,y0 = pos[u]; x1,y1 = pos[v]
                ex += [x0,x1,None]; ey += [y0,y1,None]

            edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                                    line=dict(width=1.2, color="rgba(108,99,255,0.4)"),
                                    hoverinfo="none")
            deg = [G.degree(n) for n in G.nodes]
            node_trace = go.Scatter(
                x=[pos[n][0] for n in G.nodes], y=[pos[n][1] for n in G.nodes],
                mode="markers+text",
                marker=dict(size=[8+d*4 for d in deg],
                            color=[_node_color(n) for n in G.nodes],
                            line=dict(width=1, color=CLR["border"])),
                text=list(G.nodes), textposition="top center",
                textfont=dict(size=9, color=CLR["text"]),
                hovertemplate="<b>%{text}</b><br>Degree: %{marker.size}<extra></extra>",
            )
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                showlegend=False, height=600,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=CLR["text"],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(t=12, b=12, l=12, r=12),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='display:flex;gap:18px;font-size:0.8rem;color:#94a3b8;margin-top:-8px;'>
              <span>🟣 Event (EVT)</span><span>🩵 Component (COMP)</span>
              <span>🔴 Flag (FLAG)</span><span>🟡 Hour (HOUR)</span>
            </div>""", unsafe_allow_html=True)


# ─── TAB 5 — ALGORITHM COMPARISON ────────────────────────────────────────────
with tab5:
    ap_e  = fmt_e(AP_ELAPSED)
    fp_e  = fmt_e(FP_ELAPSED)
    speedup_txt = ""
    if AP_ELAPSED and FP_ELAPSED and FP_ELAPSED > 0:
        speedup_txt = f" ({AP_ELAPSED/FP_ELAPSED:.1f}× slower than FP-Growth)"

    st.markdown(f"""
    <div style='background:{CLR["card"]};border:1px solid {CLR["border"]};
                border-left:4px solid {CLR["accent"]};border-radius:10px;
                padding:16px 20px;margin-bottom:16px;'>
      <b style='color:{CLR["accent"]};font-size:1rem;'>Why do the results look identical?</b><br>
      <span style='color:{CLR["muted"]};font-size:0.87rem;'>
        Both algorithms are <b>exact</b> — same min_support → same itemsets → same rules.
        They differ only in <b>how</b> they compute:<br>
        · <b>Apriori</b> breadth-first, k database scans{speedup_txt}<br>
        · <b>FP-Growth</b> compresses into an FP-tree, 2 scans only
      </span>
    </div>""", unsafe_allow_html=True)

    max_k = ap_fi["size"].max() if not ap_fi.empty else CFG["MAX_ITEMSET_LEN"]

    prop_data = {
        "Property":   ["Execution time","Frequent itemsets","Rules generated","Filtered rules",
                        "DB scans","Candidate generation","Memory model","Best for"],
        "Apriori":    [ap_e, f"{len(ap_fi):,}", f"{len(ap_rules):,}", f"{len(ap_f):,}",
                       f"k passes (k≤{max_k})","Explicit breadth-first","Candidate list","Small datasets"],
        "FP-Growth":  [fp_e, f"{len(fp_fi):,}", f"{len(fp_rules):,}", f"{len(fp_f):,}",
                       "2 passes","None (pattern growth)","FP-tree compressed","Large/dense datasets"],
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Algorithm Properties</div>", unsafe_allow_html=True)
        df_prop = pd.DataFrame(prop_data)
        fig = go.Figure(go.Table(
            columnwidth=[2.0, 2.2, 2.2],
            header=dict(values=["<b>Property</b>","<b>Apriori</b>","<b>FP-Growth</b>"],
                        fill_color=CLR["card"], font=dict(color=CLR["text"], size=13),
                        align="left", line_color=CLR["border"], height=34),
            cells=dict(values=[df_prop[c] for c in df_prop.columns],
                       fill_color=[[CLR["bg"]]*len(df_prop)]*3,
                       font=dict(color=[CLR["muted"],CLR["accent"],CLR["accent3"]], size=12),
                       align="left", line_color=CLR["border"], height=28),
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0,b=0,l=0,r=0), height=310)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Execution Time</div>", unsafe_allow_html=True)
        if AP_ELAPSED is not None and FP_ELAPSED is not None:
            fig = go.Figure(go.Bar(
                x=["Apriori","FP-Growth"], y=[AP_ELAPSED, FP_ELAPSED],
                marker_color=[CLR["accent"], CLR["accent3"]],
                text=[ap_e, fp_e], textposition="outside", width=0.4,
            ))
            if FP_ELAPSED > 0:
                fig.add_annotation(x=0, y=AP_ELAPSED, text=f"{AP_ELAPSED/FP_ELAPSED:.1f}× slower",
                                   showarrow=False, font=dict(color=CLR["warning"],size=11), yshift=18)
            fig.update_layout(yaxis=dict(title="Seconds", gridcolor=CLR["border"],
                                         range=[0, max(AP_ELAPSED, FP_ELAPSED)*1.45]),
                               xaxis=dict(gridcolor=CLR["border"]), margin=dict(t=30,b=12,l=12,r=12))
            st.plotly_chart(_plotly_base(fig), use_container_width=True)
        else:
            st.info("Run the pipeline to see timing data.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='section-title'>Database Scans Required</div>", unsafe_allow_html=True)
        k_range = list(range(1, max_k + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=k_range, mode="lines+markers",
                                 name="Apriori", line=dict(color=CLR["accent"], width=2)))
        fig.add_trace(go.Scatter(x=k_range, y=[2]*len(k_range), mode="lines+markers",
                                 name="FP-Growth", line=dict(color=CLR["accent3"], width=2, dash="dash")))
        fig.update_layout(xaxis=dict(title="Max itemset size (k)", dtick=1),
                          yaxis=dict(title="# DB scans"), margin=dict(t=12,b=12,l=12,r=12))
        st.plotly_chart(_plotly_base(fig), use_container_width=True)

    with c4:
        st.markdown("<div class='section-title'>Rule Overlap</div>", unsafe_allow_html=True)
        if not ap_rules.empty and not fp_rules.empty:
            ap_set = set(zip(ap_rules["antecedents"], ap_rules["consequents"]))
            fp_set = set(zip(fp_rules["antecedents"], fp_rules["consequents"]))
            only_ap = len(ap_set - fp_set)
            only_fp = len(fp_set - ap_set)
            shared  = len(ap_set & fp_set)
            fig = go.Figure(go.Bar(
                x=["Only Apriori","Shared","Only FP-Growth"], y=[only_ap, shared, only_fp],
                marker_color=[CLR["accent"],CLR["success"],CLR["accent3"]],
                text=[only_ap,shared,only_fp], textposition="auto",
            ))
            fig.update_layout(yaxis=dict(title="# Rules"))
            st.plotly_chart(_plotly_base(fig), use_container_width=True)
        else:
            st.info("Both algorithms needed for overlap analysis.")


# ─── TAB 6 — REPORT HISTORY ───────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='section-title'>All Cached Runs</div>", unsafe_allow_html=True)

    all_stamps = tuple(r["stamp"] for r in available_runs)
    all_meta   = load_all_summaries(all_stamps)

    if not all_meta:
        st.info("No historical runs found.")
    else:
        hist_rows = []
        for m in all_meta:
            hist_rows.append({
                "Run": m.get("label","?"),
                "Stamp": m.get("stamp","?"),
                "Transactions": m.get("n_transactions","?"),
                "Unique Items": m.get("n_items","?"),
                "AP Itemsets": m.get("ap_itemsets","?"),
                "AP Rules": m.get("ap_rules","?"),
                "AP Time (s)": m.get("ap_elapsed"),
                "FP Itemsets": m.get("fp_itemsets","?"),
                "FP Rules": m.get("fp_rules","?"),
                "FP Time (s)": m.get("fp_elapsed"),
            })
        hist_df = pd.DataFrame(hist_rows)
        st.dataframe(hist_df.set_index("Stamp"), use_container_width=True)

        if len(all_meta) > 1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Rules Generated per Run</div>", unsafe_allow_html=True)
                run_labels = [m["label"] for m in all_meta]
                fig = go.Figure()
                ap_r = [m.get("ap_rules") or 0 for m in all_meta]
                fp_r = [m.get("fp_rules") or 0 for m in all_meta]
                fig.add_trace(go.Bar(name="Apriori",   x=run_labels, y=ap_r, marker_color=CLR["accent"]))
                fig.add_trace(go.Bar(name="FP-Growth", x=run_labels, y=fp_r, marker_color=CLR["accent3"]))
                fig.update_layout(barmode="group", yaxis_title="# Rules", xaxis_tickangle=-30, height=320)
                st.plotly_chart(_plotly_base(fig), use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Execution Time per Run</div>", unsafe_allow_html=True)
                fig = go.Figure()
                ap_t = [m.get("ap_elapsed") for m in all_meta]
                fp_t = [m.get("fp_elapsed") for m in all_meta]
                if any(v is not None for v in ap_t):
                    fig.add_trace(go.Scatter(x=run_labels, y=ap_t, mode="lines+markers",
                                             name="Apriori", line=dict(color=CLR["accent"],width=2)))
                if any(v is not None for v in fp_t):
                    fig.add_trace(go.Scatter(x=run_labels, y=fp_t, mode="lines+markers",
                                             name="FP-Growth", line=dict(color=CLR["accent3"],width=2,dash="dash")))
                fig.update_layout(yaxis_title="Elapsed (s)", xaxis_tickangle=-30, height=320)
                st.plotly_chart(_plotly_base(fig), use_container_width=True)

            st.markdown("<div class='section-title'>Transactions per Run</div>", unsafe_allow_html=True)
            txn_vals = [m.get("n_transactions") or 0 for m in all_meta]
            fig = go.Figure(go.Bar(
                x=run_labels, y=txn_vals,
                marker_color=[CLR["success"] if m["stamp"] == _sel_stamp else CLR["muted"]
                               for m in all_meta],
                text=txn_vals, textposition="outside",
            ))
            fig.update_layout(yaxis_title="# Transactions", xaxis_tickangle=-30, height=280)
            st.plotly_chart(_plotly_base(fig), use_container_width=True)

        st.markdown("<div class='section-title'>Raw Summary Text</div>", unsafe_allow_html=True)
        st.text_area("Selected run summary", run_summary, height=250, label_visibility="collapsed")


# ── Footer ────────────────────────────────────────────────────────────────────
log_date = str(log["Date"].iloc[0]) if "Date" in log.columns else "N/A"
st.markdown(f"""
<hr style='border-color:{CLR["border"]};margin-top:28px;'>
<p style='text-align:center;color:{CLR["muted"]};font-size:0.76rem;'>
  System Failure Pattern Analysis · OpenStack 2k Dataset · {log_date}
  · Run <code style="color:{CLR["accent3"]}">{_sel_stamp}</code>
  · {N_TRANSACTIONS} transactions / {N_UNIQUE_ITEMS} items
  · min_sup={CFG["MIN_SUPPORT"]} · min_conf={CFG["MIN_CONFIDENCE"]} · min_lift={CFG["MIN_LIFT"]}
  · {len(available_runs)} cached report(s)
</p>
""", unsafe_allow_html=True)