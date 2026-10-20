"""Logging setup using only the standard library."""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call repeatedly (e.g. Streamlit reruns)."""
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=numeric, format=_FORMAT)
    else:
        root.setLevel(numeric)
