"""
fpgrowth_miner.py
=================
FP-Growth frequent-pattern mining for OpenStack log transactions.

Inherits encoding, rule generation, and reporting from BaseMiner
(mining_base.py) and only overrides the frequent-itemset step.

Usage — as a module
-------------------
    from fpgrowth_miner import FPGrowthMiner
    from log_parser import OpenStackLogParser
    from transaction_converter import build_transaction_db

    entries = OpenStackLogParser().parse_file("openstack.log")
    txn_db  = build_transaction_db(entries, strategy="time_window")

    miner  = FPGrowthMiner(txn_db)
    result = miner.run()

    FPGrowthMiner.print_report({"fpgrowth": result})
    miner.save_report({"fpgrowth": result})

Usage — CLI
-----------
    python fpgrowth_miner.py openstack.log
    python fpgrowth_miner.py openstack.log --strategy session --support 0.01
"""

from __future__ import annotations

import logging
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

from config import MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, MAX_ITEMSET_LEN
from mining_base import BaseMiner, MiningResult
from transaction_converter import TransactionDB

logger = logging.getLogger(__name__)


# ── FP-Growth miner ───────────────────────────────────────────────────────────

class FPGrowthMiner(BaseMiner):
    """
    Frequent-pattern miner using the FP-Growth algorithm (mlxtend).

    FP-Growth compresses the transaction database into a prefix tree
    (FP-tree) and mines patterns without generating explicit candidates.
    This makes it significantly faster and more memory-efficient than
    Apriori, especially on large or dense datasets.

    Best suited for:
      • Large transaction databases (thousands of windows / sessions)
      • Dense item co-occurrence matrices
      • Low minimum-support thresholds where Apriori would be too slow

    Parameters  (all inherited from BaseMiner)
    ----------
    txn_db         : TransactionDB from build_transaction_db()
    min_support    : minimum support  (default from config.MIN_SUPPORT)
    min_confidence : minimum confidence for rules  (config.MIN_CONFIDENCE)
    min_lift       : minimum lift filter  (config.MIN_LIFT)
    max_len        : maximum itemset length  (config.MAX_ITEMSET_LEN)
    """

    algorithm_name = "fpgrowth"

    def _run_frequent_itemsets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run mlxtend's fpgrowth() on the one-hot DataFrame.

        Returns
        -------
        pd.DataFrame with columns [support, itemsets]
        """
        logger.debug(
            "fpgrowth: support=%.4f  max_len=%d  shape=%s",
            self.min_support, self.max_len, df.shape,
        )
        freq_items = fpgrowth(
            df,
            min_support  = self.min_support,
            max_len      = self.max_len,
            use_colnames = True,
        )
        logger.info(
            "FP-Growth: found %d frequent itemsets", len(freq_items)
        )
        return freq_items


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_fpgrowth(
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
    miner  = FPGrowthMiner(txn_db, min_support, min_confidence, min_lift, max_len)
    result = miner.run()

    if print_results:
        FPGrowthMiner.print_report({"fpgrowth": result})
    if save_results:
        miner.save_report({"fpgrowth": result})

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
        description="Run FP-Growth frequent-pattern mining on OpenStack logs."
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

    run_fpgrowth(
        txn_db         = txn_db,
        min_support    = args.support,
        min_confidence = args.confidence,
        min_lift       = args.lift,
        max_len        = args.max_len,
        save_results   = not args.no_save,
    )
