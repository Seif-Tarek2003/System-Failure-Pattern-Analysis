"""
run_mining.py
=============
Orchestrator: runs Apriori and/or FP-Growth and (optionally) prints a
side-by-side comparison of the two results.

This is the main entry point for running the full pipeline.

Usage — as a module
-------------------
    from run_mining import run_all
    from log_parser import OpenStackLogParser
    from transaction_converter import build_transaction_db

    entries = OpenStackLogParser().parse_file("openstack.log")
    txn_db  = build_transaction_db(entries, strategy="time_window")
    results = run_all(txn_db)          # {"apriori": ..., "fpgrowth": ...}

Usage — CLI
-----------
    # Run both algorithms
    python run_mining.py openstack.log

    # Run only one
    python run_mining.py openstack.log --algorithm apriori
    python run_mining.py openstack.log --algorithm fpgrowth

    # Session-based transactions, lower support
    python run_mining.py openstack.log --strategy session --support 0.01
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import sys, os

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────────
# mining_base.print_report() uses Unicode box-drawing chars (─ U+2500),
# arrows (→ U+2192), and ≥ (U+2265).  On Windows the default console
# codec is cp1252 which cannot encode any of these, causing:
#   UnicodeEncodeError: 'charmap' codec can't encode character …
#
# Fix: reconfigure stdout/stderr to UTF-8 with errors='replace' so that
# any un-encodable character is silently swapped for '?' instead of
# crashing the process.  This is safe on all platforms.
#
# Two-pronged approach:
#   1. sys.stdout.reconfigure()  — works for direct `python run_mining.py` calls
#   2. PYTHONIOENCODING env-var  — set by dashboard.py when spawning this as a
#      subprocess (so the reconfigure below is belt-and-suspenders for CLI use)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, MAX_ITEMSET_LEN
from mining_base import BaseMiner, MiningResult
from apriori_miner import AprioriMiner
from fpgrowth_miner import FPGrowthMiner
from transaction_converter import TransactionDB

logger = logging.getLogger(__name__)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all(
    txn_db:         TransactionDB,
    algorithm:      str   = "both",      # "apriori" | "fpgrowth" | "both"
    min_support:    float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift:       float = MIN_LIFT,
    max_len:        int   = MAX_ITEMSET_LEN,
    print_results:  bool  = True,
    save_results:   bool  = True,
    compare:        bool  = True,        # side-by-side diff when running both
) -> Dict[str, MiningResult]:
    """
    Run one or both mining algorithms on a TransactionDB.

    Parameters
    ----------
    txn_db         : TransactionDB from build_transaction_db()
    algorithm      : "apriori" | "fpgrowth" | "both"
    min_support    : minimum support threshold
    min_confidence : minimum confidence threshold
    min_lift       : minimum lift filter
    max_len        : maximum frequent itemset length
    print_results  : print per-algorithm report to stdout
    save_results   : write CSVs + summary TXT to reports/
    compare        : print a comparison table when both algorithms run

    Returns
    -------
    dict  {"apriori": MiningResult, "fpgrowth": MiningResult}
          (keys present only for algorithms that were run)
    """
    common = dict(
        min_support    = min_support,
        min_confidence = min_confidence,
        min_lift       = min_lift,
        max_len        = max_len,
    )

    results: Dict[str, MiningResult] = {}

    if algorithm in ("apriori", "both"):
        miner = AprioriMiner(txn_db, **common)
        results["apriori"] = miner.run()

    if algorithm in ("fpgrowth", "both"):
        miner = FPGrowthMiner(txn_db, **common)
        results["fpgrowth"] = miner.run()

    # ── print reports ─────────────────────────────────────────────────────────
    if print_results:
        # reuse the shared print_report from BaseMiner (either subclass works)
        AprioriMiner.print_report(results)

    # ── comparison ────────────────────────────────────────────────────────────
    if compare and len(results) == 2:
        _print_comparison(results)

    # ── save reports ─────────────────────────────────────────────────────────
    if save_results:
        # Use any miner instance for the save method (state-independent)
        _saver = AprioriMiner(txn_db, **common)
        _saver.save_report(results)

    return results


# ── Comparison helper ─────────────────────────────────────────────────────────

def _print_comparison(results: Dict[str, MiningResult]) -> None:
    """Print a concise side-by-side comparison of Apriori vs FP-Growth."""
    ap  = results.get("apriori")
    fp  = results.get("fpgrowth")
    if ap is None or fp is None:
        return

    sep = "-" * 60
    print(f"\n{sep}")
    print("  COMPARISON: Apriori  vs  FP-Growth")
    print(sep)

    rows = [
        ("Frequent itemsets",  ap.n_frequent_itemsets,     fp.n_frequent_itemsets),
        ("Association rules",  ap.n_rules,                 fp.n_rules),
        ("Failure rules",      len(ap.failure_rules()),    len(fp.failure_rules())),
        ("Elapsed (s)",        f"{ap.elapsed_seconds:.3f}", f"{fp.elapsed_seconds:.3f}"),
    ]

    col_w = 22
    print(f"  {'Metric':<{col_w}}  {'Apriori':>12}  {'FP-Growth':>12}")
    print(f"  {'-'*col_w}  {'-'*12}  {'-'*12}")
    for label, a_val, f_val in rows:
        print(f"  {label:<{col_w}}  {str(a_val):>12}  {str(f_val):>12}")

    # Speed-up
    if ap.elapsed_seconds > 0:
        speedup = ap.elapsed_seconds / max(fp.elapsed_seconds, 1e-9)
        print(f"\n  FP-Growth speed-up over Apriori: {speedup:.1f}x")

    # Rule overlap
    if ap.n_rules > 0 and fp.n_rules > 0:
        ap_sigs = {
            (frozenset(r["antecedents"]), frozenset(r["consequents"]))
            for _, r in ap.rules.iterrows()
        }
        fp_sigs = {
            (frozenset(r["antecedents"]), frozenset(r["consequents"]))
            for _, r in fp.rules.iterrows()
        }
        overlap = len(ap_sigs & fp_sigs)
        union   = len(ap_sigs | fp_sigs)
        print(f"  Rule overlap (Jaccard): {overlap}/{union}  "
              f"({overlap/union*100:.1f}%)")

    print(f"{sep}\n")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from log_parser import OpenStackLogParser
    from transaction_converter import build_transaction_db

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Run Apriori and/or FP-Growth on OpenStack logs."
    )
    ap.add_argument("log_file",
                    help="Path to .log / .log.gz / .csv file")
    ap.add_argument("--strategy",   default="time_window",
                    choices=["time_window", "session"],
                    help="Transaction-building strategy  (default: time_window)")
    ap.add_argument("--algorithm",  default="both",
                    choices=["apriori", "fpgrowth", "both"],
                    help="Which algorithm(s) to run  (default: both)")
    ap.add_argument("--support",    type=float, default=MIN_SUPPORT,
                    help=f"Min support       (default: {MIN_SUPPORT})")
    ap.add_argument("--confidence", type=float, default=MIN_CONFIDENCE,
                    help=f"Min confidence    (default: {MIN_CONFIDENCE})")
    ap.add_argument("--lift",       type=float, default=MIN_LIFT,
                    help=f"Min lift          (default: {MIN_LIFT})")
    ap.add_argument("--max-len",    type=int,   default=MAX_ITEMSET_LEN,
                    help=f"Max itemset length  (default: {MAX_ITEMSET_LEN})")
    ap.add_argument("--no-save",    action="store_true",
                    help="Skip writing report files")
    ap.add_argument("--no-compare", action="store_true",
                    help="Skip the side-by-side comparison table")
    args = ap.parse_args()

    entries = OpenStackLogParser().parse_file(args.log_file)
    txn_db  = build_transaction_db(entries, strategy=args.strategy)
    print(f"\nTransactionDB stats: {txn_db.stats()}\n")

    run_all(
        txn_db         = txn_db,
        algorithm      = args.algorithm,
        min_support    = args.support,
        min_confidence = args.confidence,
        min_lift       = args.lift,
        max_len        = args.max_len,
        save_results   = not args.no_save,
        compare        = not args.no_compare,
    )