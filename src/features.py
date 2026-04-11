import os
import pandas as pd
import numpy as np

from utils import get_data_path, ensure_data_dirs

# -----------------------------
# REQUIRED COLUMNS
# -----------------------------
REQUIRED_COLUMNS = ["icao", "timestamp", "latitude", "longitude", "altitude", "velocity", "heading"]

# Path where last-seen positions are persisted across live cycles
HISTORY_PATH = "data/live_aircraft_history.csv"

# -----------------------------
# FEATURE CLAMP LIMITS
# Applied AFTER computation to prevent extreme values from
# breaking MinMaxScaler in the LSTM pipeline.
# These are physically motivated upper bounds:
#   climb_rate     : ±100 ft/s (extreme military climb rate)
#   acceleration   : ±50 knots/s (physically impossible above this)
#   speed_consistency: 0–1 km/s (above this = teleport noise)
#   heading_change : 0–180 deg (already bounded by wrap logic)
#   distance_km    : 0–500 km  (per 5s step, above = teleport)
# -----------------------------
CLAMP_LIMITS = {
    "climb_rate"        : (-100,  100),
    "acceleration"      : (-50,   50),
    "speed_consistency" : (0,     0.35),
    "distance_km"       : (0,     500),
    "heading_change"    : (0,     180),
}

# -----------------------------
# HAVERSINE DISTANCE
# -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -----------------------------
# SAFE DIVISION (avoids inf)
# -----------------------------
def safe_divide(numerator, denominator, fill=0.0):
    """
    Division by zero (duplicate timestamps) produces inf silently.
    Replace with fill value (default 0) instead.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0, fill, numerator / denominator)
    return result

# -----------------------------
# CORE FEATURE ENGINEERING
# (shared between synthetic and live modes)
# -----------------------------
def _engineer_features(df):
    """
    Computes all derived features from a sorted DataFrame that
    already has at least 2 rows per aircraft (prev row = history).

    Returns the DataFrame with features added and intermediate
    columns dropped. Does NOT call dropna() — callers handle that.
    """
    # Time difference (seconds)
    df["delta_time"] = (
        df.groupby("icao")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # Previous values
    df["prev_lat"]     = df.groupby("icao")["latitude"].shift(1)
    df["prev_lon"]     = df.groupby("icao")["longitude"].shift(1)
    df["prev_alt"]     = df.groupby("icao")["altitude"].shift(1)
    df["prev_vel"]     = df.groupby("icao")["velocity"].shift(1)
    df["prev_heading"] = df.groupby("icao")["heading"].shift(1)

    # Distance
    df["distance_km"] = haversine_distance(
        df["prev_lat"], df["prev_lon"],
        df["latitude"], df["longitude"]
    )

    # Climb rate (ft/s)
    df["delta_altitude"] = df["altitude"] - df["prev_alt"]
    df["climb_rate"]     = safe_divide(df["delta_altitude"], df["delta_time"])

    # Acceleration (knots/s)
    df["delta_velocity"] = df["velocity"] - df["prev_vel"]
    df["acceleration"]   = safe_divide(df["delta_velocity"], df["delta_time"])

    # Heading change (vectorised, 0–180°)
    raw_change = (df["heading"] - df["prev_heading"]).abs()
    df["heading_change"] = np.where(
        raw_change.isna(),
        np.nan,
        np.minimum(raw_change, 360 - raw_change)
    )

    # Speed consistency (km/s)
    df["speed_consistency"] = safe_divide(df["distance_km"], df["delta_time"])

    # Rolling velocity std
    df["velocity_std"] = (
        df.groupby("icao")["velocity"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(0, drop=True)
    )

    # Drop intermediate columns
    drop_cols = [
        "prev_lat", "prev_lon", "prev_alt", "prev_vel", "prev_heading",
        "delta_altitude", "delta_velocity", "delta_time",
        "altitude_variation", "speed_diff",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Replace inf before clamping
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clamp extreme values
    for col, (low, high) in CLAMP_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)

    return df


# ══════════════════════════════════════════════════════════
# LIVE MODE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
def compute_features_live(df):
    """
    Live-mode feature engineering using a persistent history buffer.

    Problem: each live fetch is a single snapshot (1 row per aircraft),
    so .diff() and .shift(1) produce NaN for every row, and dropna()
    kills the entire batch.

    Solution:
      1. Load the last-seen position of each aircraft from disk
      2. Prepend those history rows to the current batch
      3. Compute features normally — history row provides the
         "previous" values needed for deltas
      4. Strip out the history rows, keeping only the new batch rows
      5. Drop rows that are still NaN (aircraft seen for the first
         time this session — unavoidable, they have no history yet)
      6. Save the current batch as the new history for next cycle

    On the very first cycle (no history file yet), aircraft seen
    for the first time are dropped. From cycle 2 onward, all
    returning aircraft produce valid features.
    """
    df = df.copy()

    # Validate input
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Input rows: {len(df)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["icao", "timestamp"]).reset_index(drop=True)

    # Mark new rows so we can strip history rows after feature computation
    df["_is_new"] = True

    # ── Load history ──────────────────────────────────────────────
    history = _load_history()

    if history is not None and len(history) > 0:
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history["_is_new"]   = False

        # Only keep history rows for aircraft present in current batch
        # (no point carrying irrelevant aircraft)
        history = history[history["icao"].isin(df["icao"].unique())]

        combined = pd.concat([history, df], ignore_index=True)
        combined = combined.sort_values(by=["icao", "timestamp"]).reset_index(drop=True)

        matched = history["icao"].isin(df["icao"].unique()).sum()
        print(f"  [Features] History loaded: {matched} aircraft matched from previous cycle.")
    else:
        combined = df.copy()
        print("  [Features] No history yet — first cycle will have higher drop rate.")

    # ── Compute features on combined df ───────────────────────────
    combined = _engineer_features(combined)

    # ── Keep only the new rows ────────────────────────────────────
    result = combined[combined["_is_new"] == True].copy()
    result = result.drop(columns=["_is_new"], errors="ignore")

    # ── Diagnose NaN per column BEFORE any dropping ──────────────
    # Prints exactly which column is causing row loss.
    nan_summary = result.isna().sum()
    nan_cols    = nan_summary[nan_summary > 0]
    if len(nan_cols) > 0:
        print(f"  [Features] NaN counts in result ({len(result)} rows):")
        for col, count in nan_cols.items():
            print(f"             {col:<25} {count}/{len(result)} rows NaN")

    # ── Hard-drop: only rows missing the 4 critical numeric cols ──
    # latitude / longitude / altitude / velocity are physically essential.
    # We deliberately EXCLUDE icao, timestamp, heading from this check —
    # edge cases (NaT, empty string) in those columns should not silently
    # discard an otherwise valid flight row.
    critical_cols = [c for c in ["latitude", "longitude", "altitude", "velocity"]
                     if c in result.columns]
    rows_before  = len(result)
    result       = result.dropna(subset=critical_cols)
    hard_dropped = rows_before - len(result)

    # ── Fill ALL remaining NaN with neutral zeros ─────────────────
    # Covers: derived features for first-time aircraft (no history),
    # heading/timestamp edge cases, and any residual NaN from the pipeline.
    nan_before = result.isna().sum().sum()
    result     = result.fillna(0)

    rows_after = len(result)

    if hard_dropped > 0:
        print(f"  [Features] Hard-dropped {hard_dropped} rows (missing raw positional data).")
    if nan_before > 0:
        print(f"  [Features] Filled {nan_before} NaN derived-feature cells with 0 "
              f"(first-time aircraft — no prior history).")
    print(f"Rows dropped during cleanup : {hard_dropped}")
    print(f"Rows remaining              : {rows_after}")

    # ── Save current batch as history for next cycle ──────────────
    _save_history(df.drop(columns=["_is_new"], errors="ignore"))

    if rows_after == 0:
        raise RuntimeError(
            "All rows dropped after cleanup — all 180 rows were missing "
            "raw positional data (altitude/velocity/heading/lat/lon). "
            "Check OpenSky fetch output."
        )

    _print_feature_diagnostics(result)

    return result


# ══════════════════════════════════════════════════════════
# SYNTHETIC MODE FEATURE ENGINEERING  (unchanged)
# ══════════════════════════════════════════════════════════
def compute_features(df):
    """
    Synthetic-mode feature engineering.
    Expects multi-row time series per aircraft (as generated by generator.py).
    """
    df = df.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Input rows: {len(df)}")

    df = df.sort_values(by=["icao", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = _engineer_features(df)

    rows_before = len(df)
    df = df.dropna()
    rows_after  = len(df)

    print(f"Rows dropped during cleanup : {rows_before - rows_after}")
    print(f"Rows remaining              : {rows_after}")

    if rows_after == 0:
        raise RuntimeError(
            "All rows dropped after cleanup — check your input data for NaNs or invalid values."
        )

    _print_feature_diagnostics(df)

    return df


# ══════════════════════════════════════════════════════════
# HISTORY HELPERS
# ══════════════════════════════════════════════════════════
def _save_history(df):
    """
    Saves the latest row per aircraft to the history file.
    Called at the end of every live cycle so the next cycle
    has previous positions to diff against.
    """
    os.makedirs("data", exist_ok=True)

    # Keep only the most recent row per aircraft
    latest = (
        df.sort_values("timestamp")
        .groupby("icao")
        .tail(1)
        .reset_index(drop=True)
    )

    # Only persist the columns needed for delta computation
    keep_cols = [c for c in REQUIRED_COLUMNS + ["label"] if c in latest.columns]
    latest[keep_cols].to_csv(HISTORY_PATH, index=False)


def _load_history():
    """Loads the history file. Returns None if it doesn't exist yet."""
    if not os.path.exists(HISTORY_PATH):
        return None
    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# DIAGNOSTICS HELPER
# ══════════════════════════════════════════════════════════
def _print_feature_diagnostics(df):
    FEATURE_COLUMNS = [
        "altitude", "velocity", "heading", "distance_km",
        "climb_rate", "acceleration", "heading_change", "speed_consistency"
    ]

    available = [c for c in FEATURE_COLUMNS if c in df.columns]

    print("\n=== FEATURE RANGES (post-cleanup) ===")
    stats = df[available].describe().loc[["min", "mean", "max"]].T
    print(stats.to_string())

    inf_count = np.isinf(df[available].values).sum()
    nan_count = df[available].isna().sum().sum()
    print(f"\nInf values remaining : {inf_count}")
    print(f"NaN values remaining : {nan_count}")

    if inf_count > 0 or nan_count > 0:
        print("WARNING: inf/NaN values still present — LSTM scaling will be broken.")
    else:
        print("✅ All feature values are finite and clean.")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()

    df = pd.read_csv(get_data_path("adsb_raw.csv"))

    df_features = compute_features(df)

    df_features.to_csv(get_data_path("adsb_features.csv"), index=False)

    print("\nFeature dataset saved at data/adsb_features.csv")