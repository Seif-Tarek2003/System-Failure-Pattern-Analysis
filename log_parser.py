from __future__ import annotations

import re
import csv
import gzip
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterator
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OPENSTACK_LOG_PATTERN, FAILURE_LEVELS, CRITICAL_COMPONENTS

logger = logging.getLogger(__name__)


# Data Model

@dataclass
class LogEntry:
    line_id:       int
    timestamp:     Optional[datetime]
    level:         str                  # INFO / WARNING / ERROR / CRITICAL / DEBUG
    component:     str                  # e.g. nova.compute.manager
    content:       str                  # raw message text
    event_id:      Optional[str] = None # template ID (from structured CSV)
    event_template: Optional[str] = None
    pid:           Optional[int] = None
    is_failure:    bool = False
    is_critical:   bool = False
    severity_score: int = 0
    tokens:        List[str] = field(default_factory=list)

    # computed helpers
    @property
    def hour(self) -> Optional[int]:
        return self.timestamp.hour if self.timestamp else None

    @property
    def minute(self) -> Optional[int]:
        return self.timestamp.minute if self.timestamp else None

    @property
    def component_short(self) -> str:
        """Last two segments of component name."""
        parts = self.component.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else self.component

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "??:??:??"
        return f"[{ts}] {self.level:8s} {self.component_short:30s} | {self.content[:60]}"


# Parser

class OpenStackLogParser:
    """
    Supports two input formats:

    1. Raw log  — produced directly by OpenStack services
       Pattern: <timestamp> <pid> <level> <component> <content>

    2. Structured CSV — loghub pre-processed format
       Columns: LineId, Month, Date, Time, Pid, Component, ADDR,
                Content, EventId, EventTemplate
    """

    _RAW_RE = re.compile(OPENSTACK_LOG_PATTERN)

    # Severity mapping
    _SEVERITY = {
        "DEBUG":    0,
        "INFO":     1,
        "WARNING":  3,
        "ERROR":    7,
        "CRITICAL": 10,
        "TRACE":    2,
        "AUDIT":    1,
    }

    # public API

    def parse_file(self, path: str | Path) -> List[LogEntry]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".gz":
            # peek at inner extension
            inner = Path(path.stem).suffix.lower()
            opener = gzip.open
        else:
            inner  = suffix
            opener = open

        logger.info("Parsing %s …", path.name)

        if inner == ".csv":
            return self._parse_structured_csv(path, opener)
        else:
            return self._parse_raw_log(path, opener)

    # raw log

    def _parse_raw_log(self, path: Path, opener) -> List[LogEntry]:
        entries: List[LogEntry] = []
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line_id, raw_line in enumerate(fh, start=1):
                entry = self._parse_raw_line(raw_line.strip(), line_id)
                if entry:
                    entries.append(entry)
        logger.info("  → %d entries parsed from raw log", len(entries))
        return entries

    def _parse_raw_line(self, line: str, line_id: int) -> Optional[LogEntry]:
        if not line:
            return None
        m = self._RAW_RE.match(line)
        if not m:
            # Try to salvage partial lines (common in OpenStack)
            return self._fallback_parse(line, line_id)

        ts = self._parse_timestamp(m.group("timestamp"))
        level = m.group("level").upper()
        component = m.group("component")
        content = m.group("content").strip()
        try:
            pid = int(m.group("pid"))
        except (ValueError, IndexError):
            pid = None

        return self._build_entry(
            line_id=line_id,
            timestamp=ts,
            level=level,
            component=component,
            content=content,
            pid=pid,
        )

    def _fallback_parse(self, line: str, line_id: int) -> Optional[LogEntry]:
        """Best-effort parse for non-standard lines."""
        level = "INFO"
        for lvl in ("CRITICAL", "ERROR", "WARNING", "DEBUG", "TRACE"):
            if lvl in line.upper():
                level = lvl
                break
        return self._build_entry(
            line_id=line_id,
            timestamp=None,
            level=level,
            component="unknown",
            content=line[:300],
        )

    # structured CSV

    def _parse_structured_csv(self, path: Path, opener) -> List[LogEntry]:
        entries: List[LogEntry] = []
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry = self._parse_csv_row(row)
                if entry:
                    entries.append(entry)
        logger.info("  → %d entries parsed from CSV", len(entries))
        return entries

    def _parse_csv_row(self, row: dict) -> Optional[LogEntry]:
        try:
            line_id  = int(row.get("LineId", 0))
            date_str = row.get("Date", "")
            time_str = row.get("Time", "00:00:00")
            # Date column is already YYYY-MM-DD; Time is HH:MM:SS[.ffffff]
            ts = self._parse_timestamp(f"{date_str} {time_str}")

            level     = (row.get("Level", row.get("Levelname", "INFO")) or "INFO").upper()
            component = row.get("Component", "unknown")
            content   = row.get("Content", "")
            pid_raw   = row.get("Pid", "")
            pid       = int(pid_raw) if pid_raw and pid_raw.isdigit() else None

            entry = self._build_entry(
                line_id=line_id,
                timestamp=ts,
                level=level,
                component=component,
                content=content,
                pid=pid,
                event_id=row.get("EventId"),
                event_template=row.get("EventTemplate"),
            )
            return entry
        except Exception as exc:
            logger.debug("Skipping row due to: %s", exc)
            return None

    # builder

    def _build_entry(
        self,
        line_id: int,
        timestamp: Optional[datetime],
        level: str,
        component: str,
        content: str,
        pid: Optional[int] = None,
        event_id: Optional[str] = None,
        event_template: Optional[str] = None,
    ) -> LogEntry:
        severity = self._SEVERITY.get(level, 1)
        is_failure  = level in FAILURE_LEVELS
        is_critical = level in {"ERROR", "CRITICAL"}
        tokens = self._tokenise(content)

        return LogEntry(
            line_id=line_id,
            timestamp=timestamp,
            level=level,
            component=component,
            content=content,
            pid=pid,
            event_id=event_id,
            event_template=event_template,
            is_failure=is_failure,
            is_critical=is_critical,
            severity_score=severity,
            tokens=tokens,
        )

    # utilities

    @staticmethod
    def _parse_timestamp(s: str) -> Optional[datetime]:
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%b %d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lowercase alpha tokens, min length 3, no pure numbers."""
        return [
            t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
            if not t.isdigit()
        ]

    @staticmethod
    def _month_num(name: str) -> int:
        months = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        return months.get(name[:3].capitalize(), 1)

    # batch helper

    def parse_directory(self, directory: str | Path, pattern: str = "*.log") -> List[LogEntry]:
        """Parse all log files in a directory."""
        directory = Path(directory)
        all_entries: List[LogEntry] = []
        for log_file in sorted(directory.glob(pattern)):
            all_entries.extend(self.parse_file(log_file))
        return all_entries

    # statistics 
    @staticmethod
    def summarise(entries: List[LogEntry]) -> dict:
        from collections import Counter
        levels  = Counter(e.level for e in entries)
        comps   = Counter(e.component_short for e in entries)
        return {
            "total": len(entries),
            "failures": sum(1 for e in entries if e.is_failure),
            "critical": sum(1 for e in entries if e.is_critical),
            "level_counts": dict(levels.most_common(10)),
            "top_components": dict(comps.most_common(10)),
            "time_range": (
                min((e.timestamp for e in entries if e.timestamp), default=None),
                max((e.timestamp for e in entries if e.timestamp), default=None),
            ),
        }