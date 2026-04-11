import sys
import time
import traceback
import pandas as pd
import numpy as np

from utils import get_data_path, ensure_data_dirs

# -----------------------------
# MODE SWITCH
# -----------------------------
# "synthetic" — runs full training pipeline on generated data
# "live"      — fetches real ADS-B data from OpenSky and runs
#               detection every 30 seconds, retraining models
#               in the background every RETRAIN_INTERVAL seconds
# -----------------------------
MODE = "live"   # ← change to "synthetic" for training run

# How often to fetch new data in live mode (seconds)
LIVE_INTERVAL = 30

# Minimum flights needed to run detection in live mode
MIN_FLIGHTS = 10


# -----------------------------
# PIPELINE STEP RUNNER
# -----------------------------
def run_step(step_number, step_name, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  STEP {step_number}: {step_name}")
    print(f"{'='*60}")

    start = time.time()

    try:
        result  = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n  ✅ {step_name} completed in {elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ {step_name} FAILED after {elapsed:.1f}s")
        print(f"\n  Error: {e}")
        traceback.print_exc()
        print(f"\nPipeline stopped at Step {step_number}. Fix the error above and rerun.")
        sys.exit(1)


# -----------------------------
# SYNTHETIC PIPELINE STEPS
# -----------------------------
def step_generate():
    from generator import generate_dataset
    df = generate_dataset()
    df.to_csv(get_data_path("adsb_raw.csv"), index=False)
    print(f"  Rows generated : {len(df)}")
    print(f"  Anomaly rows   : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    return df

def step_features():
    from features import compute_features
    df = pd.read_csv(get_data_path("adsb_raw.csv"))
    print(f"  Input rows  : {len(df)}")
    df_features = compute_features(df)
    df_features.to_csv(get_data_path("adsb_features.csv"), index=False)
    print(f"  Output rows : {len(df_features)}")
    return df_features

def step_rules():
    from rules import apply_rules
    df = pd.read_csv(get_data_path("adsb_features.csv"))
    df_rules = apply_rules(df)
    df_rules.to_csv(get_data_path("adsb_rules.csv"), index=False)
    return df_rules

def step_isolation_forest():
    from isolation_forest import run_isolation_forest
    df = pd.read_csv(get_data_path("adsb_rules.csv"))
    df_if = run_isolation_forest(df)
    df_if.to_csv(get_data_path("adsb_if.csv"), index=False)
    return df_if

def step_lstm():
    from lstm_model import run_lstm_autoencoder
    df = pd.read_csv(get_data_path("adsb_if.csv"))
    df_lstm = run_lstm_autoencoder(df)
    df_lstm.to_csv(get_data_path("adsb_lstm.csv"), index=False)
    return df_lstm

def step_detect():
    from detect import combine_detections
    df = pd.read_csv(get_data_path("adsb_lstm.csv"))
    df_final = combine_detections(df)
    df_final.to_csv(get_data_path("adsb_final.csv"), index=False)
    return df_final

def step_evaluate():
    from evaluation import evaluate_all
    df = pd.read_csv(get_data_path("adsb_final.csv"))
    evaluate_all(df)
    return df

def step_visualize():
    from visualize import (
        plot_trajectory,
        plot_lstm_error,
        plot_error_distribution,
        plot_detector_comparison
    )
    df = pd.read_csv(get_data_path("adsb_final.csv"))
    plot_trajectory(df)
    plot_lstm_error(df)
    plot_error_distribution(df)
    plot_detector_comparison(df)
    return df


# -----------------------------
# SYNTHETIC PIPELINE SUMMARY
# -----------------------------
def print_summary(df_final):
    from sklearn.metrics import f1_score, precision_score, recall_score

    print(f"\n{'='*60}")
    print("  SKYGATE — PIPELINE COMPLETE")
    print(f"{'='*60}")

    detectors = {
        "Rule-Based"       : "rule_anomaly",
        "Isolation Forest" : "if_anomaly",
        "Autoencoder"      : "lstm_anomaly",
        "Final Ensemble"   : "final_anomaly",
    }

    y_true = df_final["label"]

    print(f"\n  {'Detector':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*52}")

    for name, col in detectors.items():
        if col in df_final.columns:
            y_pred = df_final[col]
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred,    zero_division=0)
            f = f1_score(y_true, y_pred,         zero_division=0)
            print(f"  {name:<22} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print(f"\n  Plots saved to : data/plots/")
    print(f"  Final CSV      : data/adsb_final.csv")
    print(f"\n{'='*60}\n")


# -----------------------------
# LIVE DETECTION — SINGLE CYCLE
# -----------------------------
def run_live_detection_cycle(cycle_num, models):
    """
    Runs one detection cycle:
    1. Fetch live ADS-B data from OpenSky  (auto-logged to live_log.csv)
    2. Engineer features
    3. Apply rules (loads thresholds from models/thresholds.json if present)
    4. Run Isolation Forest (loads retrained model if available)
    5. Run Autoencoder     (loads fine-tuned model if available)
    6. Combine detections  (uses saved weights from synthetic training)
    7. Print alerts
    """
    from data_fetcher import fetch_live_data
    from features import compute_features_live
    from rules import apply_rules
    from isolation_forest import run_isolation_forest
    from lstm_model import run_lstm_autoencoder
    from detect import combine_detections

    print(f"\n{'='*60}")
    print(f"  CYCLE {cycle_num} — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")

    # Step 1 — Fetch  (save_live_batch is called automatically inside fetch_live_data)
    df = fetch_live_data()
    if df is None or len(df) < MIN_FLIGHTS:
        print(f"  [SKIP] Not enough flights ({len(df) if df is not None else 0} < {MIN_FLIGHTS})")
        return

    # Step 2 — Features (live version uses persistent history buffer)
    try:
        df = compute_features_live(df)
    except Exception as e:
        print(f"  [SKIP] Feature engineering failed: {e}")
        return

    if len(df) == 0:
        print("  [SKIP] No rows remaining after feature engineering.")
        return

    # Step 3 — Rules
    df = apply_rules(df)

    # Step 4 — Isolation Forest (uses retrained model from disk if available)
    df = run_isolation_forest(df, contamination=models["if_contamination"])

    # Step 5 — Autoencoder (uses fine-tuned model from disk if available)
    df = run_lstm_autoencoder(df)

    # Step 6 — Ensemble (uses saved weights from synthetic training)
    df["label"] = 0   # placeholder — real data has no ground truth
    df = combine_detections(df)

    # Step 7 — Save and alert
    df.to_csv(get_data_path("adsb_live_latest.csv"), index=False)
    print_live_alerts(df, cycle_num)


# -----------------------------
# LIVE ALERTS PRINTER
# -----------------------------
def print_live_alerts(df, cycle_num):
    """
    Prints a summary of flagged aircraft for this cycle.
    Highlights aircraft flagged by multiple detectors.
    """
    flagged = df[df["final_anomaly"] == 1].copy()
    total   = len(df)

    print(f"\n  Total flights monitored : {total}")
    print(f"  Anomalies flagged       : {len(flagged)}")

    if len(flagged) == 0:
        print("  ✅ No anomalies detected this cycle.")
        return

    print(f"\n  {'ICAO':<10} {'Lat':>8} {'Lon':>8} {'Alt(ft)':>10} "
          f"{'Vel(kts)':>10} {'Rule':>6} {'IF':>4} {'AE':>4} {'Score':>7}")
    print(f"  {'-'*72}")

    for _, row in flagged.iterrows():
        votes = int(row.get("rule_anomaly", 0)) + \
                int(row.get("if_anomaly",   0)) + \
                int(row.get("lstm_anomaly", 0))

        marker = "⚠️ " if votes >= 2 else "  "

        print(
            f"  {marker}{str(row['icao']):<8} "
            f"{row['latitude']:>8.3f} "
            f"{row['longitude']:>8.3f} "
            f"{row['altitude']:>10.0f} "
            f"{row['velocity']:>10.1f} "
            f"{int(row.get('rule_anomaly', 0)):>6} "
            f"{int(row.get('if_anomaly',   0)):>4} "
            f"{int(row.get('lstm_anomaly', 0)):>4} "
            f"{row.get('final_score', 0.0):>7.3f}"
        )

    multi = flagged[
        (flagged.get("rule_anomaly", 0) +
         flagged.get("if_anomaly",   0) +
         flagged.get("lstm_anomaly", 0)) >= 2
    ] if all(c in flagged.columns for c in ["rule_anomaly", "if_anomaly", "lstm_anomaly"]) else pd.DataFrame()

    if len(multi) > 0:
        print(f"\n  ⚠️  {len(multi)} aircraft flagged by 2+ detectors — HIGH CONFIDENCE anomaly")


# -----------------------------
# LIVE MONITORING LOOP
# -----------------------------
def run_live_mode():
    """
    Continuously fetches live ADS-B data from OpenSky Network
    and runs the full detection pipeline every LIVE_INTERVAL seconds.

    A background thread retrains all three model layers every hour
    using the accumulated live_log.csv data.

    Pre-requisite: run synthetic mode first to produce initial models
    and save ensemble weights to models/ensemble_weights.json.
    """
    print("\n" + "="*60)
    print("  SKYGATE — LIVE MONITORING MODE")
    print("  Region  : Indian Airspace")
    print(f"  Interval: every {LIVE_INTERVAL}s  |  Retrain: every 60 min")
    print("  Press Ctrl+C to stop")
    print("="*60)

    # ── Start background retraining scheduler ──────────────────
    from trainer import start_training_scheduler
    start_training_scheduler()   # daemon thread — stops with main process
    # ───────────────────────────────────────────────────────────

    models = {
        "if_contamination": 0.05
    }

    cycle = 1
    try:
        while True:
            run_live_detection_cycle(cycle, models)
            cycle += 1
            print(f"\n  Next fetch in {LIVE_INTERVAL}s... (Ctrl+C to stop)")
            time.sleep(LIVE_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  Monitoring stopped after {cycle - 1} cycles.")
        print("  Latest results saved to data/adsb_live_latest.csv")
        print("  Live log saved to      data/live_log.csv")


# -----------------------------
# SYNTHETIC TRAINING PIPELINE
# -----------------------------
def run_synthetic_mode():
    print("\n" + "="*60)
    print("  SKYGATE — SYNTHETIC TRAINING PIPELINE")
    print("="*60)
    print("  Running all steps sequentially...")

    pipeline_start = time.time()
    ensure_data_dirs()

    run_step(1, "Synthetic Data Generation",  step_generate)
    run_step(2, "Feature Engineering",        step_features)
    run_step(3, "Rule-Based Detection",       step_rules)
    run_step(4, "Isolation Forest",           step_isolation_forest)
    run_step(5, "Autoencoder",                step_lstm)
    run_step(6, "Ensemble Detection",         step_detect)   # ← saves weights here
    run_step(7, "Evaluation",                 step_evaluate)
    run_step(8, "Visualisation",              step_visualize)

    df_final = pd.read_csv(get_data_path("adsb_final.csv"))
    print_summary(df_final)

    total_time = time.time() - pipeline_start
    print(f"  Total pipeline time: {total_time:.1f}s\n")
    print("  ✅ Ensemble weights saved to models/ensemble_weights.json")
    print("     Switch MODE to 'live' to start real-time monitoring.\n")


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
def main():
    if MODE == "live":
        run_live_mode()
    elif MODE == "synthetic":
        run_synthetic_mode()
    else:
        print(f"Unknown MODE: '{MODE}'. Set MODE to 'synthetic' or 'live'.")
        sys.exit(1)


if __name__ == "__main__":
    main()