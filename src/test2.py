import sys
import time
import traceback
import pandas as pd
import numpy as np
from utils import get_data_path, ensure_data_dirs

MODE = "live" 
LIVE_INTERVAL = 30
MIN_FLIGHTS = 10

def run_step(step_number, step_name, fn, *args, **kwargs):
    print(f"\nSTEP {step_number}: {step_name}")
    start = time.time()
    try:
        result  = fn(*args, **kwargs)
        print(f"✅ {step_name} done in {time.time() - start:.1f}s")
        return result
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

# Synthetic steps (remains exactly as you wrote)
def step_generate():
    from generator import generate_dataset
    df = generate_dataset()
    df.to_csv(get_data_path("adsb_raw.csv"), index=False)
    return df

def step_features():
    from features import compute_features
    df = pd.read_csv(get_data_path("adsb_raw.csv"))
    df_features = compute_features(df)
    df_features.to_csv(get_data_path("adsb_features.csv"), index=False)
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

def run_live_detection_cycle(cycle_num, models):
    from data_fetcher import fetch_live_data
    from features import compute_features_live
    from rules import apply_rules
    from isolation_forest import run_isolation_forest
    from lstm_model import run_lstm_autoencoder
    from detect import combine_detections

    print(f"\nCYCLE {cycle_num} — {pd.Timestamp.now()}")
    df = fetch_live_data()
    if df is None or len(df) < MIN_FLIGHTS: return
    
    df = compute_features_live(df)
    df = apply_rules(df)
    df = run_isolation_forest(df, contamination=models["if_contamination"])
    df = run_lstm_autoencoder(df)
    df["label"] = 0
    df = combine_detections(df)
    df.to_csv(get_data_path("adsb_live_latest.csv"), index=False)

def run_live_mode():
    from trainer import start_training_scheduler
    start_training_scheduler()
    models = {"if_contamination": 0.05}
    cycle = 1
    try:
        while True:
            run_live_detection_cycle(cycle, models)
            cycle += 1
            time.sleep(LIVE_INTERVAL)
    except KeyboardInterrupt: pass

def run_synthetic_mode():
    ensure_data_dirs()
    run_step(1, "Generation", step_generate)
    run_step(2, "Features", step_features)
    run_step(3, "Rules", step_rules)
    run_step(4, "Isolation Forest", step_isolation_forest)
    run_step(5, "Autoencoder", step_lstm)
    run_step(6, "Ensemble", step_detect)

if __name__ == "__main__":
    if MODE == "live": run_live_mode()
    else: run_synthetic_mode()