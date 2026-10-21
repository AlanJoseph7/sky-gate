"""
logger.py — SkyGate Structured Logging

Single source of truth for all logging configuration.
Every module imports `get_logger(__name__)` — no bare print() calls anywhere.

Log levels used in SkyGate
───────────────────────────
  DEBUG    Internal state, model internals, feature values
  INFO     Normal operational events (fetch cycle, SSE connect, retrain)
  WARNING  Recoverable issues (worker restart, missing model file, API 429)
  ERROR    Failed operations that reduced capability (pipeline step failed)
  CRITICAL Non-recoverable issues (DB init failed, secret missing)

JSON output (production)
────────────────────────
Set SKYGATE_LOG_FORMAT=json in .env for structured JSON output suitable
for ingestion by Elasticsearch / Datadog / Loki.

  {"time":"14:31:02","level":"INFO","logger":"server","msg":"...","val":"..."}

Plain text output (default, development)
────────────────────────────────────────
  14:31:02  INFO   server          SkyGate server ready  http://localhost:5000
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────────────────
# Log format selection (env-driven)
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FORMAT  = os.getenv("SKYGATE_LOG_FORMAT", "text").lower()   # "text" | "json"
_LOG_LEVEL   = os.getenv("SKYGATE_LOG_LEVEL",  "INFO").upper()
_LOG_FILE    = os.getenv("SKYGATE_LOG_FILE",   "")               # optional file path


class _PlainFormatter(logging.Formatter):
    """
    Human-readable single-line format for development:
      HH:MM:SS  LEVEL  logger_name  message  [extra]
    """
    FMT = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
    DATEFMT = "%H:%M:%S"

    COLOURS = {
        "DEBUG":    "\033[34m",   # blue
        "INFO":     "\033[36m",   # cyan
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{self.RESET}" if colour else record.levelname
        record.name = record.name.split(".")[-1]   # show only leaf module name
        return super().format(record)

    def formatTime(self, record, datefmt=None):  # noqa: N802
        return datetime.now(timezone.utc).strftime(datefmt or self.DATEFMT)


class _JsonFormatter(logging.Formatter):
    """
    Newline-delimited JSON for log aggregators.
    Each log line is a self-contained JSON object.
    """
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time":    datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name.split(".")[-1],
            "msg":     record.getMessage(),
        }
        # Attach any extra fields passed via logger.info("...", extra={"val": "..."})
        for key in ("val", "event", "icao24", "callsign", "score"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _build_handler(stream=sys.stdout) -> logging.Handler:
    handler = logging.StreamHandler(stream)
    if _LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_PlainFormatter(
            fmt=_PlainFormatter.FMT,
            datefmt=_PlainFormatter.DATEFMT,
        ))
    return handler


def _configure_root() -> None:
    """Called once at import time to configure the root logger."""
    root = logging.getLogger()
    if root.handlers:
        return   # already configured (e.g., uvicorn called basicConfig first)

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    root.addHandler(_build_handler(sys.stdout))

    # Optional file sink
    if _LOG_FILE:
        try:
            fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
            fh.setFormatter(_JsonFormatter())     # always JSON in files
            root.addHandler(fh)
        except OSError as e:
            root.warning("Could not open log file %s: %s", _LOG_FILE, e)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access", "tensorflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger.

    Usage (in every SkyGate module):
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Fetching ADS-B batch", extra={"val": f"flights={n}"})
    """
    return logging.getLogger(name)
