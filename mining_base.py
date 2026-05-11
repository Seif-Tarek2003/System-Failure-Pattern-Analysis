"""
mining_base.py
==============
Shared foundation for all pattern-mining algorithms.

Provides:
  • MiningResult  — dataclass holding frequent itemsets + rules
  • BaseMiner     — abstract base with OHE encoding, rule generation,
                    print_report(), and save_report()

Both AprioriMiner (apriori_miner.py) and FPGrowthMiner (fpgrowth_miner.py)
inherit from BaseMiner and only override _run_frequent_itemsets().
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Dependency version guard ──────────────────────────────────────────────────
# Pinned to requirements.txt.  Checked once at import time; fails fast with a
# clear message rather than a cryptic runtime error deep inside mlxtend/pandas.

_REQUIRED_VERSIONS = {
    "pandas":       "3.0.2",
    "numpy":        "2.4.4",
    "mlxtend":      "0.24.0",
    "sklearn":      "1.8.0",   # scikit-learn exposes as sklearn
    "scipy":        "1.17.1",
    "matplotlib":   "3.10.9",
}

def _check_versions(strict: bool = False) -> None:
    """
    Verify installed package versions match requirements.txt.

    Parameters
    ----------
    strict : if True, raise ImportError on any mismatch;
             if False (default), emit a WARNING and continue.
    """
    import importlib.metadata as _meta

    mismatches = []
    for pkg, required in _REQUIRED_VERSIONS.items():
        # scikit-learn is installed as 'scikit-learn' but imported as 'sklearn'
        dist_name = "scikit-learn" if pkg == "sklearn" else pkg
        try:
            installed = _meta.version(dist_name)
        except _meta.PackageNotFoundError:
            mismatches.append(f"  {dist_name}: NOT INSTALLED (need {required})")
            continue
        if installed != required:
            mismatches.append(
                f"  {dist_name}: installed={installed}  required={required}"
            )

    if mismatches:
        msg = (
            "Dependency version mismatch(es) detected:\n"
            + "\n".join(mismatches)
            + "\nRun:  pip install -r requirements.txt"
        )
        if strict:
            raise ImportError(msg)
        else:
            logging.getLogger(__name__).warning(msg)

_check_versions(strict=False)   # set strict=True to hard-fail on mismatch
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

from config import (
    MIN_SUPPORT,
    MIN_CONFIDENCE,
    MIN_LIFT,
    MAX_ITEMSET_LEN,
    MAX_FREQUENT_ITEMSETS,
    TOP_N_PATTERNS,
    REPORTS_DIR,
)
from transaction_converter import TransactionDB

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MiningResult:
    """
    Output of one algorithm run.

    Attributes
    ----------
    algorithm       : "apriori" | "fpgrowth"
    frequent_items  : DataFrame  [support, itemsets]
    rules           : DataFrame  [antecedents, consequents, support,
                                  confidence, lift, leverage, conviction]
    elapsed_seconds : wall-clock time for the mining step
    n_transactions  : number of transactions in the encoded matrix
    min_support / min_confidence / min_lift : thresholds used
    """

    algorithm:       str
    frequent_items:  pd.DataFrame
    rules:           pd.DataFrame
    elapsed_seconds: float = 0.0
    n_transactions:  int   = 0
    min_support:     float = MIN_SUPPORT
    min_confidence:  float = MIN_CONFIDENCE
    min_lift:        float = MIN_LIFT

    # ── accessors ────────────────────────────────────────────────────────────

    @property
    def n_frequent_itemsets(self) -> int:
        return len(self.frequent_items)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def top_rules(self, n: int = TOP_N_PATTERNS) -> pd.DataFrame:
        """Top-N rules ranked by lift then confidence."""
        if self.rules.empty:
            return self.rules
        return (
            self.rules
            .sort_values(["lift", "confidence"], ascending=False)
            .head(n)
        )

    def failure_rules(self) -> pd.DataFrame:
        """Rules whose consequent contains FLAG:FAILURE or FLAG:CRITICAL."""
        if self.rules.empty:
            return self.rules
        mask = self.rules["consequents"].apply(
            lambda cs: any("FLAG:" in c for c in cs)
        )
        return self.rules[mask].sort_values("lift", ascending=False)


# ── Abstract base miner ───────────────────────────────────────────────────────

class BaseMiner(ABC):
    """
    Shared logic for Apriori and FP-Growth miners.

    Subclasses must implement _run_frequent_itemsets(df) and set
    the class attribute `algorithm_name`.

    Parameters
    ----------
    txn_db         : TransactionDB from build_transaction_db()
    min_support    : minimum support threshold  (0–1)
    min_confidence : minimum confidence for rule generation (0–1)
    min_lift       : minimum lift filter applied after rule generation
    max_len        : maximum itemset length
    """

    algorithm_name: str = ""   # overridden by subclasses

    def __init__(
        self,
        txn_db:         TransactionDB,
        min_support:    float = MIN_SUPPORT,
        min_confidence: float = MIN_CONFIDENCE,
        min_lift:       float = MIN_LIFT,
        max_len:        int   = MAX_ITEMSET_LEN,
    ):
        self.txn_db         = txn_db
        self.min_support    = min_support
        self.min_confidence = min_confidence
        self.min_lift       = min_lift
        self.max_len        = max_len
        self._ohe_df: Optional[pd.DataFrame] = None

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> MiningResult:
        """Encode → mine → generate rules → return MiningResult."""
        df = self._encode()

        # Safety floor: require at least 3 transactions to be "frequent".
        # With dense log data even a 2-transaction floor still produces millions
        # of candidate itemsets that crash association_rules().
        n_txns = len(df)
        floor = 3.0 / n_txns if n_txns > 0 else self.min_support
        if self.min_support < floor:
            logger.warning(
                "%s: min_support=%.4f is below the 3-transaction floor "
                "(%.4f for %d txns) — raising automatically.",
                self.algorithm_name, self.min_support, floor, n_txns,
            )
            self.min_support = floor

        logger.info(
            "Running %s  (support=%.3f  confidence=%.3f  lift≥%.2f  max_len=%d) …",
            self.algorithm_name, self.min_support,
            self.min_confidence, self.min_lift, self.max_len,
        )

        t0 = time.perf_counter()
        freq_items = self._run_frequent_itemsets(df)
        rules      = self._generate_rules(freq_items)
        elapsed    = time.perf_counter() - t0

        logger.info(
            "%s finished: %d frequent itemsets  %d rules  (%.2fs)",
            self.algorithm_name, len(freq_items), len(rules), elapsed,
        )
        return MiningResult(
            algorithm       = self.algorithm_name,
            frequent_items  = freq_items,
            rules           = rules,
            elapsed_seconds = elapsed,
            n_transactions  = len(df),
            min_support     = self.min_support,
            min_confidence  = self.min_confidence,
            min_lift        = self.min_lift,
        )

    # ── abstract hook ─────────────────────────────────────────────────────────

    @abstractmethod
    def _run_frequent_itemsets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given the one-hot DataFrame, return a frequent-itemsets DataFrame
        with columns [support, itemsets].
        Implemented by AprioriMiner and FPGrowthMiner.
        """

    # ── shared internals ─────────────────────────────────────────────────────

    def _encode(self) -> pd.DataFrame:
        """One-hot encode the TransactionDB (result cached after first call)."""
        if self._ohe_df is not None:
            return self._ohe_df

        raw = [list(t.items) for t in self.txn_db.transactions]
        if not raw:
            raise ValueError("TransactionDB is empty — nothing to mine.")

        te = TransactionEncoder()
        matrix = te.fit_transform(raw)
        self._ohe_df = pd.DataFrame(matrix, columns=te.columns_)
        logger.info(
            "One-hot matrix: %d transactions × %d items",
            *self._ohe_df.shape,
        )
        return self._ohe_df

    def _generate_rules(self, freq_items: pd.DataFrame) -> pd.DataFrame:
        """Build association rules from frequent itemsets, then filter by lift."""
        _empty = pd.DataFrame(
            columns=["antecedents", "consequents",
                     "support", "confidence", "lift",
                     "leverage", "conviction"]
        )
        if freq_items.empty:
            logger.warning("%s: no frequent itemsets — skipping rule generation.",
                           self.algorithm_name)
            return _empty

        # Guard: association_rules() exhausts memory on millions of itemsets.
        # This happens on dense log data even at moderate support thresholds.
        # Bail out early with a clear message rather than crashing silently.
        if len(freq_items) > MAX_FREQUENT_ITEMSETS:
            logger.error(
                "%s: %d frequent itemsets exceeds MAX_FREQUENT_ITEMSETS=%d. "
                "Raise min_support or lower MAX_ITEMSET_LEN in config.py, "
                "then re-run. Skipping rule generation.",
                self.algorithm_name, len(freq_items), MAX_FREQUENT_ITEMSETS,
            )
            return _empty

        # Suppress mlxtend's internal RuntimeWarning:
        # "invalid value encountered in divide" on cert_metric when confidence=1.0
        # causes a 0-division in (certainty_num / certainty_denom). The result is
        # correctly handled by np.where, so the warning is a false alarm.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            rules = association_rules(
                freq_items,
                metric        = "confidence",
                min_threshold = self.min_confidence,
                num_itemsets  = len(freq_items),
            )
        rules = rules[rules["lift"] >= self.min_lift].reset_index(drop=True)
        return rules

    # ── reporting (shared, static-style) ─────────────────────────────────────

    @staticmethod
    def print_report(
        results: Dict[str, MiningResult],
        top_n:   int = TOP_N_PATTERNS,
    ) -> None:
        """Pretty-print a summary of one or more MiningResults to stdout."""
        sep = "─" * 72
        for algo, res in results.items():
            print(f"\n{sep}")
            print(
                f"  {algo.upper()}  |  transactions={res.n_transactions}"
                f"  support={res.min_support}  confidence={res.min_confidence}"
                f"  lift≥{res.min_lift}  ({res.elapsed_seconds:.2f}s)"
            )
            print(sep)
            print(f"  Frequent itemsets : {res.n_frequent_itemsets}")
            print(f"  Association rules : {res.n_rules}")

            top = res.top_rules(top_n)
            if top.empty:
                print("  (no rules generated)")
            else:
                print(f"\n  ── Top {min(top_n, len(top))} rules by lift ──")
                for _, row in top.iterrows():
                    ant = ", ".join(sorted(row["antecedents"]))
                    con = ", ".join(sorted(row["consequents"]))
                    print(
                        f"    [{row['support']:.3f} sup | "
                        f"{row['confidence']:.3f} conf | "
                        f"{row['lift']:.2f} lift]  {ant}  →  {con}"
                    )

            fail = res.failure_rules()
            if not fail.empty:
                print(f"\n  ── Failure / Critical rules ({len(fail)}) ──")
                for _, row in fail.head(top_n).iterrows():
                    ant = ", ".join(sorted(row["antecedents"]))
                    con = ", ".join(sorted(row["consequents"]))
                    print(f"    [lift={row['lift']:.2f}]  {ant}  →  {con}")
        print(f"\n{sep}\n")

    def save_report(
        self,
        results:    Dict[str, MiningResult],
        output_dir: str | Path = REPORTS_DIR,
    ) -> None:
        """
        Write per-algorithm CSVs and a combined summary TXT.

        Files produced (all prefixed with a datetime stamp):
          <stamp>_<algo>_frequent_itemsets.csv
          <stamp>_<algo>_rules.csv
          <stamp>_summary.txt
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary: List[str] = [
            f"Pattern Mining Report — {stamp}",
            f"TransactionDB : {len(self.txn_db)} transactions  "
            f"/ {len(self.txn_db.vocabulary)} unique items",
            "",
        ]

        for algo, res in results.items():
            # frequent itemsets CSV
            if not res.frequent_items.empty:
                fi = res.frequent_items.copy()
                fi["itemsets"] = fi["itemsets"].apply(lambda s: "|".join(sorted(s)))
                fi_path = out / f"{stamp}_{algo}_frequent_itemsets.csv"
                fi.to_csv(fi_path, index=False)
                logger.info("Saved %s", fi_path)

            # rules CSV
            if not res.rules.empty:
                ru = res.rules.copy()
                ru["antecedents"] = ru["antecedents"].apply(lambda s: "|".join(sorted(s)))
                ru["consequents"] = ru["consequents"].apply(lambda s: "|".join(sorted(s)))
                ru_path = out / f"{stamp}_{algo}_rules.csv"
                ru.to_csv(ru_path, index=False)
                logger.info("Saved %s", ru_path)

            # summary block
            summary += [
                f"[{algo.upper()}]",
                f"  Elapsed           : {res.elapsed_seconds:.3f}s",
                f"  Frequent itemsets : {res.n_frequent_itemsets}",
                f"  Rules generated   : {res.n_rules}",
                "",
            ]
            top = res.top_rules(TOP_N_PATTERNS)
            if not top.empty:
                summary.append(f"  Top {len(top)} rules by lift:")
                for _, row in top.iterrows():
                    ant = "|".join(sorted(row["antecedents"]))
                    con = "|".join(sorted(row["consequents"]))
                    summary.append(
                        f"    sup={row['support']:.3f}  conf={row['confidence']:.3f}"
                        f"  lift={row['lift']:.2f}  {ant} → {con}"
                    )
            summary.append("")

        txt_path = out / f"{stamp}_summary.txt"
        txt_path.write_text("\n".join(summary), encoding="utf-8")
        logger.info("Summary → %s", txt_path)
        print(f"\nReports written to: {out.resolve()}")