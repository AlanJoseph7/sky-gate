"""
pipeline.py — SkyGate Detection Pipeline (wired to real detectors)

Called by server.py on every background fetch cycle.
Accepts a raw DataFrame from fetch_live_data() and returns a list
of anomaly dicts ready for the frontend.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── Real SkyGate modules ─────────────────────────────────────────────────────
from features         import compute_features_live
from rules            import apply_rules
from isolation_forest import run_isolation_forest
from lstm_model       import run_lstm_autoencoder
from detect           import combine_detections

# ── Minimum flights per cycle to bother running ML ───────────────────────────
MIN_FLIGHTS = 5

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(flights: pd.DataFrame) -> list[dict]:
    """
    Run the full three-layer SkyGate detection pipeline on a live batch.

    Parameters
    ----------
    flights : pd.DataFrame
        Raw output of fetch_live_data() — columns: icao, timestamp,
        latitude, longitude, altitude, velocity, heading, label(=0)

    Returns
    -------
    list[dict]  — one dict per anomalous flight, each containing:
        icao24, callsign, anomaly, layer, score, severity, timestamp,
        latitude, longitude, altitude, velocity, heading,
        rule_anomaly, if_anomaly, lstm_anomaly, final_score
    """
    if flights is None or len(flights) < MIN_FLIGHTS:
        return []

    df = flights.copy()

    # ── Ensure label column exists (live data has no ground truth) ────────────
    df["label"] = 0

    # ── Step 1: Feature engineering (uses persistent history buffer) ─────────
    try:
        df = compute_features_live(df)
    except RuntimeError as e:
        # All rows dropped — too early in session, no history yet
        print(f"  [Pipeline] Feature engineering skipped: {e}")
        return []
    except Exception as e:
        print(f"  [Pipeline] Feature engineering failed: {e}")
        traceback.print_exc()
        return []

    if len(df) == 0:
        return []

    # ── Step 2: Rule-based detection ─────────────────────────────────────────
    try:
        df = apply_rules(df)
    except Exception as e:
        print(f"  [Pipeline] Rules failed: {e}")
        df["rule_anomaly"] = 0

    # ── Step 3: Isolation Forest ──────────────────────────────────────────────
    try:
        df = run_isolation_forest(df)
    except Exception as e:
        print(f"  [Pipeline] Isolation Forest failed: {e}")
        df["if_anomaly"] = 0
        df["if_score"]   = 0.0

    # ── Step 4: Autoencoder ───────────────────────────────────────────────────
    try:
        df = run_lstm_autoencoder(df)
    except Exception as e:
        print(f"  [Pipeline] Autoencoder failed: {e}")
        df["lstm_anomaly"] = 0
        df["lstm_error"]   = 0.0

    # ── Step 5: Ensemble combination ─────────────────────────────────────────
    try:
        df = combine_detections(df)
    except Exception as e:
        print(f"  [Pipeline] Ensemble failed: {e}")
        # Fall back to majority vote
        votes = (
            df.get("rule_anomaly", pd.Series(0, index=df.index)) +
            df.get("if_anomaly",   pd.Series(0, index=df.index)) +
            df.get("lstm_anomaly", pd.Series(0, index=df.index))
        )
        df["final_anomaly"] = (votes >= 2).astype(int)
        df["final_score"]   = votes / 3.0

    # ── Step 6: Build anomaly dicts for the frontend ──────────────────────────
    flagged = df[df["final_anomaly"] == 1].copy()

    anomalies = []
    ts_now = datetime.now(timezone.utc).strftime("%H:%M:%S")

    for _, row in flagged.iterrows():
        anomalies.append(_build_anomaly_dict(row, ts_now))

    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_anomaly_dict(row: pd.Series, ts: str) -> dict:
    """Convert a flagged DataFrame row to the frontend-expected dict format."""

    rule  = int(row.get("rule_anomaly", 0))
    iso   = int(row.get("if_anomaly",   0))
    ae    = int(row.get("lstm_anomaly", 0))
    score = float(row.get("final_score", 0.0))

    # Determine which layer fired first (for display)
    if rule:
        layer   = "Rule-Based"
        anomaly = _rule_anomaly_type(row)
    elif iso:
        layer   = "Iso Forest"
        anomaly = "Statistical Outlier"
    else:
        layer   = "Autoencoder"
        anomaly = "Recon Error"

    # Severity from score
    if score >= 0.75:
        severity = "critical"
    elif score >= 0.45:
        severity = "warn"
    else:
        severity = "info"

    return {
        "icao24":       str(row.get("icao", "??????")).upper(),
        "callsign":     str(row.get("callsign", "N/A")).strip() or "N/A",
        "anomaly":      anomaly,
        "layer":        layer,
        "score":        round(score, 3),
        "severity":     severity,
        "timestamp":    ts,
        "latitude":     _safe_float(row.get("latitude")),
        "longitude":    _safe_float(row.get("longitude")),
        "altitude":     _safe_float(row.get("altitude")),
        "velocity":     _safe_float(row.get("velocity")),
        "heading":      _safe_float(row.get("heading")),
        "rule_anomaly": rule,
        "if_anomaly":   iso,
        "lstm_anomaly": ae,
        "final_score":  round(score, 3),
    }


def _rule_anomaly_type(row: pd.Series) -> str:
    """Identify which rule triggered for a human-readable label."""
    from rules import (
        MAX_CLIMB_RATE, MAX_SPEED, MAX_DISTANCE, MAX_HEADING_CHANGE
    )
    if abs(row.get("climb_rate",      0) or 0) > MAX_CLIMB_RATE:
        return "Altitude Spike"
    if     row.get("velocity",         0) or 0  > MAX_SPEED:
        return "Impossible Speed"
    if     row.get("distance_km",      0) or 0  > MAX_DISTANCE:
        return "Position Teleport"
    if     row.get("heading_change",   0) or 0  > MAX_HEADING_CHANGE:
        return "Heading Violation"
    return "Rule Trigger"


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return round(v, 4) if not (v != v) else default   # NaN check
    except (TypeError, ValueError):
        return default