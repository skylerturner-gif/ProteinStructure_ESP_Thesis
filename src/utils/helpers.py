"""
src/utils/helpers.py

Shared utility functions: logging setup, timing, and directory creation.

Logging design:
    - All log output goes to file only — no terminal output during processing
    - A single print() notification is emitted when a protein completes or fails
    - get_logger()          — returns a file-only logger for src/ modules
    - get_pipeline_logger() — returns the global pipeline logger (writes to
                              the log_file path from config)
    - notify()              — prints a single terminal notification line

Timing:
    - timer()               — context manager that measures elapsed seconds
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


# ── Formatters ────────────────────────────────────────────────────────────────

_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s"
)


# ── Logger factory ────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Return a named logger that writes to file only — no console output.

    If log_file is provided, a FileHandler is added for that path.
    If log_file is None, the logger has no handlers and is effectively
    silent until a parent logger with handlers is configured.

    Call once per module:
        log = get_logger(__name__)

    Args:
        name:     logger name, typically __name__
        log_file: optional path to a log file

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured, avoid duplicate handlers

    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent output bubbling to root logger

    if log_file is not None:
        _add_file_handler(logger, log_file)

    return logger


def get_pipeline_logger(log_file: Path) -> logging.Logger:
    """
    Return the global pipeline logger that writes to the main log file.
    Used by scripts and the full pipeline to log high-level events.

    Args:
        log_file: path to the global pipeline log file

    Returns:
        Configured logging.Logger instance named 'pipeline'
    """
    logger = logging.getLogger("pipeline")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    _add_file_handler(logger, log_file)
    return logger


def _add_file_handler(logger: logging.Logger, log_file: Path) -> None:
    """Add a FileHandler to a logger, creating parent directories as needed."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(_FILE_FORMATTER)
    logger.addHandler(fh)


# ── Terminal notification ─────────────────────────────────────────────────────

def notify(protein_id: str, status: str, detail: str = "") -> None:
    """
    Print a single terminal notification line for a completed protein.
    This is the only output that appears in the terminal during a pipeline run.

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        status:     "complete", "failed", or "skipped"
        detail:     optional short detail string (e.g. "apbs failed")

    Example output:
        ✓  AF-Q16613-F1                        complete
        ✗  AF-Q16613-F1                        failed  —  apbs failed
    """
    icon = "✓" if status == "complete" else "✗" if status == "failed" else "–"
    line = f"{icon}  {protein_id:<35}  {status}"
    if detail:
        line += f"  —  {detail}"
    print(line)


# ── Timing ───────────────────────────────────────────────────────────────────

class _Timer:
    """Holds the elapsed time after a timer context exits."""
    def __init__(self):
        self.seconds: float = 0.0

    @property
    def rounded(self) -> float:
        """Elapsed time rounded to 3 decimal places."""
        return round(self.seconds, 3)


@contextmanager
def timer() -> Generator[_Timer, None, None]:
    """
    Context manager that measures elapsed wall-clock time in seconds.
    The elapsed time is available on the returned object after the block exits.

    Usage:
        with timer() as t:
            do_something()
        elapsed = t.seconds        # float
        elapsed_rounded = t.rounded  # rounded to 3 decimal places

    Example — writing to metadata:
        with timer() as t:
            process_pdb2pqr(protein_id, data_root)
        update_metadata(protein_id, data_root, {"time_pdb2pqr_sec": t.rounded})
    """
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start


# ── Directory creation ────────────────────────────────────────────────────────

def ensure_dirs(*dirs) -> None:
    """
    Create one or more directories (and any missing parents).
    Accepts Path objects or strings. Safe to call if they already exist.

    Usage:
        ensure_dirs(RAW_DIR, PQR_DIR, ESP_DIR)
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)