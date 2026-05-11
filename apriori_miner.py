"""
apriori_miner.py
================
Apriori-based frequent-pattern mining for OpenStack log transactions.

Inherits encoding, rule generation, and reporting from BaseMiner
(mining_base.py) and only overrides the frequent-itemset step.

Usage — as a module
-------------------
    from apriori_miner import AprioriMiner
    from log_parser import OpenStackLogParser
    from transaction_converter import build_transaction_db

    entries = OpenStackLogParser().parse_file("openstack.log")
    txn_db  = build_transaction_db(entries, strategy="time_window")

    miner  = AprioriMiner(txn_db)
    result = miner.run()

    AprioriMiner.print_report({"apriori": result})
    miner.save_report({"apriori": result})

Usage — CLI
-----------
    python apriori_miner.py openstack.log
    python apriori_miner.py openstack.log --strategy session --support 0.03
"""

from __future__ import annotations

import logging
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from mlxtend.frequent_patterns import apriori

from config import MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, MAX_ITEMSET_LEN
from mining_base import BaseMiner, MiningResult
from transaction_converter import TransactionDB

logger = logging.getLogger(__name__)


# ── Apriori miner ─────────────────────────────────────────────────────────────

class AprioriMiner(BaseMiner):
    """
    Frequent-pattern miner using the Apriori algorithm (mlxtend).

    Apriori uses a breadth-first, candidate-generation approach.
    It is simpler conceptually but slower than FP-Growth on large,
    dense datasets because it scans the database once per itemset level.

    Best suited for:
      • Small-to-medium transaction databases
      • Datasets where you want an exact, well-understood baseline
      • Debugging / interpretability scenarios

    Parameters  (all inherited from BaseMiner)
    ----------
    txn_db         : TransactionDB from build_transaction_db()
    min_support    : minimum support  (default from config.MIN_SUPPORT)
    min_confidence : minimum confidence for rules  (config.MIN_CONFIDENCE)
    min_lift       : minimum lift filter  (config.MIN_LIFT)
    max_len        : maximum itemset length  (config.MAX_ITEMSET_LEN)
    """

    algorithm_name = "apriori"

    def _run_frequent_itemsets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run mlxtend's apriori() on the one-hot DataFrame.

        Returns
        -------
        pd.DataFrame with columns [support, itemsets]
        """
        logger.debug(
            "apriori: support=%.4f  max_len=%d  shape=%s",
            self.min_support, self.max_len, df.shape,
        )
        freq_items = apriori(
            df,
            min_support  = self.min_support,
            max_len      = self.max_len,
            use_colnames = True,
            low_memory   = True,    # safer for varying dataset sizes
        )
        logger.info(
            "Apriori: found %d frequent itemsets", len(freq_items)
        )
        return freq_items


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_apriori(
    txn_db:         TransactionDB,
    min_support:    float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift:       float = MIN_LIFT,
    max_len:        int   = MAX_ITEMSET_LEN,
    print_results:  bool  = True,
    save_results:   bool  = True,
) -> MiningResult:
    """
    One-call wrapper: instantiate → run → (optionally) report & save.

    Returns the MiningResult for further programmatic use.
    """
    miner  = AprioriMiner(txn_db, min_support, min_confidence, min_lift, max_len)
    result = miner.run()

    if print_results:
        AprioriMiner.print_report({"apriori": result})
    if save_results:
        miner.save_report({"apriori": result})

    return result


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from log_parser import OpenStackLogParser
    from transaction_converter import build_transaction_db

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Run Apriori frequent-pattern mining on OpenStack logs."
    )
    ap.add_argument("log_file",
                    help="Path to .log / .log.gz / .csv file")
    ap.add_argument("--strategy",   default="time_window",
                    choices=["time_window", "session"],
                    help="Transaction-building strategy (default: time_window)")
    ap.add_argument("--support",    type=float, default=MIN_SUPPORT,
                    help=f"Min support  (default: {MIN_SUPPORT})")
    ap.add_argument("--confidence", type=float, default=MIN_CONFIDENCE,
                    help=f"Min confidence  (default: {MIN_CONFIDENCE})")
    ap.add_argument("--lift",       type=float, default=MIN_LIFT,
                    help=f"Min lift  (default: {MIN_LIFT})")
    ap.add_argument("--max-len",    type=int,   default=MAX_ITEMSET_LEN,
                    help=f"Max itemset length  (default: {MAX_ITEMSET_LEN})")
    ap.add_argument("--no-save",    action="store_true",
                    help="Skip writing report files")
    args = ap.parse_args()

    entries = OpenStackLogParser().parse_file(args.log_file)
    txn_db  = build_transaction_db(entries, strategy=args.strategy)
    print(f"\nTransactionDB stats: {txn_db.stats()}\n")

    run_apriori(
        txn_db         = txn_db,
        min_support    = args.support,
        min_confidence = args.confidence,
        min_lift       = args.lift,
        max_len        = args.max_len,
        save_results   = not args.no_save,
    )