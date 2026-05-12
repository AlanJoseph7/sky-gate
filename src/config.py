"""
config.py — SkyGate Centralised Configuration

Single source of truth for every tuneable setting. All external
configuration comes from environment variables with sensible defaults.

Usage:
    from config import cfg

    print(cfg.CORS_ORIGINS)       # list[str]
    print(cfg.RATE_LIMIT_LOGIN)   # "10/minute"
    print(cfg.JWT_SECRET_KEY)     # str
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger("skygate")


def _csv_list(raw: str) -> List[str]:
    """Split a comma-separated env var into a trimmed list."""
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class SkyGateConfig:
    """Immutable application configuration — populated once at import time."""

    # ── JWT / Auth ───────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = field(default_factory=lambda: os.environ.get(
        "SKYGATE_SECRET_KEY", ""
    ))
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.environ.get("SKYGATE_JWT_EXPIRE_MINUTES", "1440"))

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = field(default_factory=lambda: _csv_list(
        os.environ.get(
            "SKYGATE_CORS_ORIGINS",
            "http://localhost:5000,http://127.0.0.1:5000",
        )
    ))

    # ── Rate Limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_LOGIN: str = os.environ.get("SKYGATE_RATE_LIMIT_LOGIN", "10/minute")
    RATE_LIMIT_SIGNUP: str = os.environ.get("SKYGATE_RATE_LIMIT_SIGNUP", "5/minute")

    # ── OpenSky Credentials ──────────────────────────────────────────────────
    OPENSKY_USERNAME: str = os.environ.get("OPENSKY_USERNAME", "")
    OPENSKY_PASSWORD: str = os.environ.get("OPENSKY_PASSWORD", "")

    # ── adsb.lol API ─────────────────────────────────────────────────────────
    ADSBLOL_API_BASE: str = os.environ.get("ADSBLOL_API_BASE", "https://api.adsb.lol")
    ADSBLOL_API_PATH: str = os.environ.get("ADSBLOL_API_PATH", "/v2/point/20.5/82.5/250")
    ADSBLOL_API_KEY: str = os.environ.get("ADSBLOL_API_KEY", "")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    FETCH_INTERVAL_SEC: int = int(os.environ.get("SKYGATE_FETCH_INTERVAL", "30"))
    MAX_LOG_ENTRIES: int = int(os.environ.get("SKYGATE_MAX_LOG", "200"))

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = os.environ.get("SKYGATE_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("SKYGATE_PORT", "5000"))

    def __post_init__(self) -> None:
        # Generate a random fallback if no secret key was provided.
        if not self.JWT_SECRET_KEY:
            import warnings
            object.__setattr__(self, "JWT_SECRET_KEY", secrets.token_hex(32))
            warnings.warn(
                "SKYGATE_SECRET_KEY not set — using a random key. "
                "JWTs will be invalidated on server restart. "
                "Set SKYGATE_SECRET_KEY in your environment for production.",
                stacklevel=2,
            )


# ── Module-level singleton ───────────────────────────────────────────────────
cfg = SkyGateConfig()
