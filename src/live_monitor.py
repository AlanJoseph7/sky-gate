import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import pandas as pd
import numpy as np
from datetime import datetime

from utils import get_data_path, ensure_data_dirs, FEATURE_COLUMNS
from data_fetcher import fetch_live_data, OPENSKY_USERNAME, OPENSKY_PASSWORD
from features import compute_features
from rules import apply_rules
from isolation_forest import run_isolation_forest
from lstm_model import run_lstm_autoencoder
from detect import combine_detections

# -----------------------------
# LIVE MONITOR CONFIG
# -----------------------------
FETCH_INTERVAL_SEC = 30     # seconds between each fetch
MAX_CYCLES         = None   # None = run forever, int = stop after N cycles
ALERT_THRESHOLD    = 2      # flag aircraft detected by this many detectors

# -----------------------------
# ALERT COLOURS (terminal)
# -----------------------------
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# -----------------------------
# PRINT HELPERS
# -----------------------------
def print_header():
    print(f"\n{BOLD}{CYAN}")
    print("=" * 60)
    print("   ✈️  SKYGATE — LIVE ADS-B ANOMALY MONITOR")
    print("   Region : Indian Airspace")
    print("   Mode   : Real-Time Detection")
    print("=" * 60)
    print(f"{RESET}")

def print_cycle_header(cycle, timestamp):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  CYCLE {cycle} — {timestamp}{RESET}")
    print(f"{'─' * 60}")

def print_alert(icao, detectors_fired, row):
    print(f"\n{RED}{BOLD}  🚨 ANOMALY DETECTED{RESET}")
    print(f"{RED}  ├─ ICAO       : {icao.upper()}{RESET}")
    print(f"{RED}  ├─ Detectors  : {', '.join(detectors_fired)}{RESET}")
    print(f"{RED}  ├─ Altitude   : {row.get('altitude', 'N/A'):.0f} ft{RESET}")
    print(f"{RED}  ├─ Velocity   : {row.get('velocity', 'N/A'):.0f} knots{RESET}")
    print(f"{RED}  ├─ Heading    : {row.get('heading', 'N/A'):.1f}°{RESET}")
    print(f"{RED}  └─ Position   : ({row.get('latitude', 'N/A'):.4f}°N, {row.get('longitude', 'N/A'):.4f}°E){RESET}")

def print_summary(total, flagged, cycle_time):
    print(f"\n{GREEN}  ✅ Cycle complete in {cycle_time:.1f}s{RESET}")
    print(f"  ├─ Flights analysed : {total}")
    print(f"  ├─ Anomalies flagged: {flagged}")
    rate = (flagged / total * 100) if total > 0 else 0
    print(f"  └─ Anomaly rate     : {rate:.1f}%")

# -----------------------------
# RUN ONE DETECTION CYCLE
# -----------------------------
def run_detection_cycle(cycle_num):
    """
    Fetches one snapshot of live ADS-B data, runs it through
    the full SkyGate pipeline, and prints any anomalies detected.
    """
    cycle_start = time.time()
    timestamp   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print_cycle_header(cycle_num, timestamp)

    # -----------------------------
    # STEP 1 — FETCH LIVE DATA
    # -----------------------------
    print(f"\n  📡 Fetching live ADS-B data...")
    df_raw = fetch_live_data(
        username=OPENSKY_USERNAME,
        password=OPENSKY_PASSWORD
    )

    if df_raw.empty:
        print(f"  ⚠️  No data received — skipping cycle")
        return 0, 0

    # -----------------------------
    # STEP 2 — FEATURE ENGINEERING
    # Need at least 2 rows per aircraft for delta features
    # Filter out single-point aircraft
    # -----------------------------
    print(f"\n  ⚙️  Computing features...")
    try:
        # Real data has no label column — add dummy for pipeline compatibility
        df_raw["label"] = 0

        # Only keep aircraft with enough rows for feature computation
        counts  = df_raw.groupby("icao").size()
        valid   = counts[counts >= 2].index
        df_raw  = df_raw[df_raw["icao"].isin(valid)]

        if df_raw.empty:
            print("  ⚠️  Not enough multi-point aircraft for feature computation")
            return 0, 0

        df_feat = compute_features(df_raw)

    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")
        return 0, 0

    if df_feat.empty:
        print("  ⚠️  No features computed")
        return 0, 0

    # -----------------------------
    # STEP 3 — RULE-BASED DETECTION
    # -----------------------------
    print(f"\n  📏 Running rule-based detection...")
    try:
        df_rules = apply_rules(df_feat)
    except Exception as e:
        print(f"  ❌ Rule detection failed: {e}")
        df_rules = df_feat.copy()
        df_rules["rule_anomaly"] = 0

    # -----------------------------
    # STEP 4 — ISOLATION FOREST
    # For real data: train on this batch (all assumed normal)
    # In production: load a pre-trained model instead
    # -----------------------------
    print(f"\n  🌲 Running Isolation Forest...")
    try:
        df_if = run_isolation_forest(df_rules)
    except Exception as e:
        print(f"  ❌ Isolation Forest failed: {e}")
        df_if = df_rules.copy()
        df_if["if_anomaly"] = 0
        df_if["if_score"]   = 0.0

    # -----------------------------
    # STEP 5 — AUTOENCODER
    # For real data: train on this batch (all assumed normal)
    # In production: load a pre-trained model instead
    # -----------------------------
    print(f"\n  🧠 Running Autoencoder...")
    try:
        df_lstm = run_lstm_autoencoder(df_if)
    except Exception as e:
        print(f"  ❌ Autoencoder failed: {e}")
        df_if["lstm_anomaly"] = 0
        df_if["lstm_error"]   = 0.0
        df_lstm = df_if.copy()

    # -----------------------------
    # STEP 6 — ENSEMBLE
    # -----------------------------
    print(f"\n  🔀 Running ensemble detection...")
    try:
        df_final = combine_detections(df_lstm)
    except Exception as e:
        print(f"  ❌ Ensemble failed: {e}")
        df_lstm["final_anomaly"] = df_lstm.get("rule_anomaly", 0)
        df_final = df_lstm.copy()

    # -----------------------------
    # STEP 7 — ALERTS
    # Print flagged aircraft with detector breakdown
    # -----------------------------
    flagged_df = df_final[df_final["final_anomaly"] == 1]
    total      = len(df_final)
    flagged    = len(flagged_df)

    if flagged == 0:
        print(f"\n{GREEN}  ✅ No anomalies detected this cycle{RESET}")
    else:
        print(f"\n{RED}{BOLD}  🚨 {flagged} anomalous flight(s) detected!{RESET}")

        # Print each flagged aircraft
        for icao, group in flagged_df.groupby("icao"):
            row = group.iloc[-1]  # most recent state for this aircraft

            detectors_fired = []
            if row.get("rule_anomaly", 0) == 1:
                detectors_fired.append("Rules")
            if row.get("if_anomaly", 0) == 1:
                detectors_fired.append("IsolationForest")
            if row.get("lstm_anomaly", 0) == 1:
                detectors_fired.append("Autoencoder")

            print_alert(icao, detectors_fired, row)

    # -----------------------------
    # STEP 8 — SAVE CYCLE RESULTS
    # Append to running log file
    # -----------------------------
    log_path = get_data_path("adsb_live_log.csv")
    write_header = not pd.io.common.file_exists(log_path)
    df_final.to_csv(log_path, mode="a", header=write_header, index=False)

    cycle_time = time.time() - cycle_start
    print_summary(total, flagged, cycle_time)

    return total, flagged

# -----------------------------
# MAIN MONITORING LOOP
# -----------------------------
def run_live_monitor(
    interval=FETCH_INTERVAL_SEC,
    max_cycles=MAX_CYCLES
):
    ensure_data_dirs()
    print_header()

    # Credential check
    if not OPENSKY_USERNAME or not OPENSKY_PASSWORD:
        print(f"{YELLOW}")
        print("  ⚠️  WARNING: OpenSky credentials not set!")
        print("  Edit data_fetcher.py and fill in:")
        print("    OPENSKY_USERNAME = 'your_username'")
        print("    OPENSKY_PASSWORD = 'your_password'")
        print(f"  Anonymous access is limited to 400 requests/day.{RESET}\n")

    print(f"  Fetch interval : {interval}s")
    print(f"  Max cycles     : {'∞' if max_cycles is None else max_cycles}")
    print(f"  Alert threshold: {ALERT_THRESHOLD} detectors")
    print(f"\n  Press Ctrl+C to stop monitoring\n")

    cycle        = 0
    total_flights = 0
    total_flagged = 0
    start_time   = time.time()

    try:
        while True:
            cycle += 1

            flights, flagged = run_detection_cycle(cycle)
            total_flights   += flights
            total_flagged   += flagged

            if max_cycles and cycle >= max_cycles:
                print(f"\n  Max cycles ({max_cycles}) reached — stopping.")
                break

            # Wait for next cycle
            print(f"\n  ⏳ Next fetch in {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n{BOLD}  Monitor stopped by user.{RESET}")

    finally:
        # Final session summary
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"{BOLD}  SESSION SUMMARY{RESET}")
        print(f"{'=' * 60}")
        print(f"  Duration        : {elapsed/60:.1f} minutes")
        print(f"  Cycles run      : {cycle}")
        print(f"  Flights analysed: {total_flights}")
        print(f"  Anomalies flagged: {total_flagged}")
        if total_flights > 0:
            print(f"  Overall rate    : {total_flagged/total_flights*100:.1f}%")
        print(f"  Log saved to    : data/adsb_live_log.csv")
        print(f"{'=' * 60}\n")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_live_monitor(
        interval=FETCH_INTERVAL_SEC,
        max_cycles=MAX_CYCLES
    )