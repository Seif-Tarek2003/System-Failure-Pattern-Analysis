from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TIME_WINDOW_SECONDS,
    SESSION_WINDOW_SECONDS,
    MIN_EVENTS_PER_WINDOW,
    FAILURE_LEVELS,
)
from log_parser import LogEntry

logger = logging.getLogger(__name__)

# Item construction helpers 

def _make_items(entry: LogEntry) -> List[str]:
    """
    Convert one log entry into a set of descriptive items.

    Item types produced:
      • LEVEL:<level>           e.g.  LEVEL:ERROR
      • COMP:<component_short>  e.g.  COMP:compute.manager
      • EVT:<event_id>          e.g.  EVT:E12   (if available)
      • KW:<keyword>            e.g.  KW:timeout  (important tokens)
      • HOUR:<hh>               e.g.  HOUR:03
    """
    items: List[str] = []

    # Level item
    items.append(f"LEVEL:{entry.level}")

    # Component item (short form keeps cardinality manageable)
    comp = entry.component_short.replace(".", "_")
    items.append(f"COMP:{comp}")

    # Event template ID (very informative when available)
    if entry.event_id:
        items.append(f"EVT:{entry.event_id}")

    # Keyword items from content
    failure_keywords = {
        "timeout", "failed", "failure", "error", "aborted",
        "refused", "unreachable", "lost", "killed", "exception",
        "deadlock", "overflow", "segfault", "panic", "unavailable",
        "retrying", "connection", "build", "launch", "spawn",
        "migrate", "evacuate", "resize", "reboot", "delete",
    }
    for tok in entry.tokens:
        if tok in failure_keywords:
            items.append(f"KW:{tok}")

    # Hour-of-day item (captures time-based patterns)
    if entry.hour is not None:
        items.append(f"HOUR:{entry.hour:02d}")

    # Failure flag item
    if entry.is_failure:
        items.append("FLAG:FAILURE")
    if entry.is_critical:
        items.append("FLAG:CRITICAL")

    return list(set(items))  # deduplicate within one entry

# Data container

@dataclass
class Transaction:
    tid:         int
    items:       FrozenSet[str]
    start_time:  Optional[datetime] = None
    end_time:    Optional[datetime] = None
    has_failure: bool = False
    entry_count: int  = 0
    window_label: str = ""

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class TransactionDB:
    transactions: List[Transaction] = field(default_factory=list)
    vocabulary:   List[str]         = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transactions)

    def item_list(self) -> List[FrozenSet[str]]:
        return [t.items for t in self.transactions]

    def failure_transactions(self) -> List[Transaction]:
        return [t for t in self.transactions if t.has_failure]

    def stats(self) -> dict:
        if not self.transactions:
            return {}
        total   = len(self.transactions)
        failure = len(self.failure_transactions())
        sizes   = [len(t) for t in self.transactions]
        return {
            "total_transactions":   total,
            "failure_transactions": failure,
            "failure_rate":         failure / total,
            "vocabulary_size":      len(self.vocabulary),
            "avg_items_per_txn":    sum(sizes) / total,
            "max_items_per_txn":    max(sizes),
            "min_items_per_txn":    min(sizes),
        }
    
# Strategy 1: Time-Window

class TimeWindowConverter:
    """
    Partition logs into fixed-width time windows.
    Each window becomes one transaction containing the union of items
    from all log entries that fall within it.
    """

    def __init__(self, window_seconds: int = TIME_WINDOW_SECONDS):
        self.window_seconds = window_seconds

    def convert(self, entries: List[LogEntry]) -> TransactionDB:
        if not entries:
            return TransactionDB()

        # Sort by timestamp; push entries with None timestamps to the back
        sorted_entries = sorted(
            entries,
            key=lambda e: e.timestamp or datetime.max
        )

        transactions: List[Transaction] = []
        vocab: set[str] = set()

        # Group by window
        windows: Dict[int, List[LogEntry]] = defaultdict(list)
        base_ts: Optional[datetime] = None

        for entry in sorted_entries:
            if entry.timestamp is None:
                windows[-1].append(entry)
                continue
            if base_ts is None:
                base_ts = entry.timestamp
            delta = (entry.timestamp - base_ts).total_seconds()
            bucket = int(delta // self.window_seconds)
            windows[bucket].append(entry)

        tid = 0
        for bucket, group in sorted(windows.items()):
            if len(group) < MIN_EVENTS_PER_WINDOW:
                continue

            items: set[str] = set()
            has_failure = False
            timestamps  = []

            for entry in group:
                entry_items = _make_items(entry)
                items.update(entry_items)
                vocab.update(entry_items)
                if entry.is_failure:
                    has_failure = True
                if entry.timestamp:
                    timestamps.append(entry.timestamp)

            start = min(timestamps) if timestamps else None
            end   = max(timestamps) if timestamps else None
            label = (
                f"W{bucket:04d}_{start.strftime('%H%M%S') if start else '------'}"
                f"–{end.strftime('%H%M%S') if end else '------'}"
            )

            transactions.append(Transaction(
                tid=tid,
                items=frozenset(items),
                start_time=start,
                end_time=end,
                has_failure=has_failure,
                entry_count=len(group),
                window_label=label,
            ))
            tid += 1

        logger.info(
            "TimeWindowConverter: %d windows → %d transactions (window=%ds)",
            len(windows), len(transactions), self.window_seconds,
        )
        return TransactionDB(
            transactions=transactions,
            vocabulary=sorted(vocab),
        )

# Strategy 2: Session-Based

class SessionConverter:
    """
    Group log entries by request / session ID extracted from content,
    or fall back to PID-based grouping with a time-out boundary.
    """

    # Patterns for OpenStack request IDs
    _REQ_RE = re.compile(r"req-([0-9a-f\-]{36})")
    _UUID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")

    def __init__(self, timeout_seconds: int = SESSION_WINDOW_SECONDS):
        self.timeout_seconds = timeout_seconds

    def convert(self, entries: List[LogEntry]) -> TransactionDB:
        if not entries:
            return TransactionDB()

        # Assign session key to each entry
        sessions: Dict[str, List[LogEntry]] = defaultdict(list)
        for entry in entries:
            key = self._session_key(entry)
            sessions[key].append(entry)

        # Split sessions that exceed timeout into sub-sessions
        split_sessions: List[List[LogEntry]] = []
        for key, group in sessions.items():
            split_sessions.extend(self._split_by_timeout(group))

        # Build transactions
        transactions: List[Transaction] = []
        vocab: set[str] = set()

        for tid, group in enumerate(split_sessions):
            if len(group) < MIN_EVENTS_PER_WINDOW:
                continue

            items: set[str] = set()
            has_failure = False
            timestamps  = []

            for entry in group:
                entry_items = _make_items(entry)
                items.update(entry_items)
                vocab.update(entry_items)
                if entry.is_failure:
                    has_failure = True
                if entry.timestamp:
                    timestamps.append(entry.timestamp)

            start = min(timestamps) if timestamps else None
            end   = max(timestamps) if timestamps else None

            transactions.append(Transaction(
                tid=tid,
                items=frozenset(items),
                start_time=start,
                end_time=end,
                has_failure=has_failure,
                entry_count=len(group),
                window_label=f"SES{tid:04d}",
            ))

        logger.info(
            "SessionConverter: %d sessions → %d transactions",
            len(split_sessions), len(transactions),
        )
        return TransactionDB(
            transactions=transactions,
            vocabulary=sorted(vocab),
        )

    def _session_key(self, entry: LogEntry) -> str:
        """Extract request ID, then fall back to PID, then 'global'."""
        m = self._REQ_RE.search(entry.content)
        if m:
            return f"req-{m.group(1)}"
        m = self._UUID_RE.search(entry.content)
        if m:
            return f"uuid-{m.group(1)}"
        if entry.pid:
            return f"pid-{entry.pid}"
        return "global"

    def _split_by_timeout(
        self, group: List[LogEntry]
    ) -> List[List[LogEntry]]:
        """Split a session group at time gaps > timeout."""
        sorted_group = sorted(
            group,
            key=lambda e: e.timestamp or datetime.min,
        )
        if not sorted_group:
            return []

        sub_sessions: List[List[LogEntry]] = []
        current: List[LogEntry] = [sorted_group[0]]

        for prev, curr in zip(sorted_group, sorted_group[1:]):
            if (
                prev.timestamp and curr.timestamp
                and (curr.timestamp - prev.timestamp).total_seconds()
                > self.timeout_seconds
            ):
                sub_sessions.append(current)
                current = [curr]
            else:
                current.append(curr)

        if current:
            sub_sessions.append(current)
        return sub_sessions
    
# Factory 

def build_transaction_db(
    entries: List[LogEntry],
    strategy: str = "time_window",
    **kwargs,
) -> TransactionDB:
    """
    Convenience factory.

    Parameters
    ----------
    entries  : parsed log entries
    strategy : "time_window" | "session"
    kwargs   : forwarded to the chosen converter
    """
    if strategy == "time_window":
        return TimeWindowConverter(**kwargs).convert(entries)
    elif strategy == "session":
        return SessionConverter(**kwargs).convert(entries)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'time_window' or 'session'.")