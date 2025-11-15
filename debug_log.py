"""Lightweight in-memory logging so debug info can be surfaced in Streamlit UI."""

from __future__ import annotations

from datetime import datetime
from typing import List

_MAX_ENTRIES = 500
_log_entries: List[str] = []


def log(message: str) -> None:
    """Record a debug message with a UTC timestamp and mirror to stdout."""
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    _log_entries.append(entry)
    if len(_log_entries) > _MAX_ENTRIES:
        del _log_entries[: len(_log_entries) - _MAX_ENTRIES]


def get_logs() -> List[str]:
    """Return a copy of accumulated log entries."""
    return list(_log_entries)


def clear() -> None:
    """Reset the log buffer for a new Streamlit run."""
    _log_entries.clear()
