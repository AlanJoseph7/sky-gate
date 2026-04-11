"""
trainer.py — SkyGate Incremental Retraining Module

Runs in the background during live monitoring and periodically
updates all three detection layers using accumulated live data.

Schedule (default):
  - Every RETRAIN_INTERVAL seconds (3600s = 1 hour)
  - Requires at least MIN_ROWS_TO_RETRAIN rows in live_log.csv

Layers updated:
  1. Rule thresholds   → recomputed from live data percentiles
  2. Isolation Forest  → fully retrained on sliding window
  3. Autoencoder       → fine-tuned for a few epochs (low LR, Keras)
"""

import os
import json
import threading
import time
import joblib
import numpy as np
import pandas as pd

from utils import load_keras_model_compatible

# -----------------------------
# CONFIG
# -----------------------------
RETRAIN_INTERVAL    = 3600        # seconds between retraining runs
MIN_ROWS_TO_RETRAIN = 500         # don't retrain on too little data
SLIDING_WINDOW      = 10_000      # use only the most recent N rows
LIVE_LOG_PATH       = "data/live_log.csv"
THRESHOLDS_PATH     = "models/thresholds.json"
IF_MODEL_PATH       = "models/iso_forest.pkl"
IF_SCALER_PATH      = "models/if_scaler.pkl"
AE_MODEL_PATH       = "models/autoencoder.keras"
AE_SCALER_PATH      = "models/ae_scaler.pkl"
AE_THRESHOLD_PATH   = "models/ae_threshold.json"

FEATURE_COLS = ["altitude", "velocity", "heading", "latitude", "longitude"]


# ══════════════════════════════════════════════════════════
# 1.  RULE THRESHOLD UPDATER
# ══════════════════════════════════════════════════════════
def update_rule_thresholds(log_path=LIVE_LOG_PATH):
    """
    Recomputes rule-based thresholds from live data percentiles
    and saves them to models/thresholds.json.
    """
    df = _load_log(log_path)
    if df is None:
        return

    thresholds = {
        "max_altitude":      float(df["altitude"].quantile(0.995)),
        "max_velocity":      float(df["velocity"].quantile(0.995)),
        "max_vertical_rate": float(df["vertical_rate"].abs().quantile(0.995))
        if "vertical_rate" in df.columns else 6000.0,
    }

    os.makedirs("models", exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"  [Trainer] Rule thresholds updated → {thresholds}")


# ══════════════════════════════════════════════════════════
# 2.  ISOLATION FOREST RETRAINER
# ══════════════════════════════════════════════════════════
def retrain_isolation_forest(log_path=LIVE_LOG_PATH,
                             model_path=IF_MODEL_PATH,
                             scaler_path=IF_SCALER_PATH):
    """
    Fully retrains the Isolation Forest on recent live data using
    the same StandardScaler approach as the synthetic training run.
    Saves both the model and the scaler.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df = _load_log(log_path)
    if df is None:
        return

    features = _extract_features(df)
    if features is None:
        return

    try:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled)

        # Save to temp first — avoids corrupt model on disk if interrupted
        for obj, path in [(model, model_path), (scaler, scaler_path)]:
            tmp = path + ".tmp"
            joblib.dump(obj, tmp)
            os.replace(tmp, path)

        print(f"  [Trainer] Isolation Forest retrained on {len(features)} rows")
        print(f"            Model  → {model_path}")
        print(f"            Scaler → {scaler_path}")

    except Exception as e:
        print(f"  [Trainer] Isolation Forest retraining failed: {e}")


# ══════════════════════════════════════════════════════════
# 3.  AUTOENCODER FINE-TUNER  (Keras)
# ══════════════════════════════════════════════════════════
def finetune_autoencoder(log_path=LIVE_LOG_PATH,
                         model_path=AE_MODEL_PATH,
                         scaler_path=AE_SCALER_PATH,
                         threshold_path=AE_THRESHOLD_PATH):
    """
    Fine-tunes the existing Keras autoencoder on recent live data.

    Strategy:
    - Low learning rate (1e-4) — adapts to real-world patterns
      without overwriting synthetic training (avoids catastrophic
      forgetting)
    - Only 5 epochs — quick adaptation pass, not a full retrain
    - Recomputes the reconstruction threshold from the new data
    - Saves model, scaler, and threshold only if fine-tuning succeeds
    """
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    missing = [p for p in [model_path, scaler_path] if not os.path.exists(p)]
    if missing:
        print(f"  [Trainer] Autoencoder fine-tune skipped — missing: {missing}")
        print("            Run synthetic mode first to create initial models.")
        return

    df = _load_log(log_path)
    if df is None:
        return

    features = _extract_features(df)
    if features is None:
        return

    try:
        import tf_keras.backend as K

        # Load existing model and scaler
        model  = load_keras_model_compatible(model_path)
        scaler = joblib.load(scaler_path)

        # Lower learning rate for fine-tuning
        K.set_value(model.optimizer.learning_rate, 1e-4)

        # Scale with existing scaler — don't refit, preserves original range
        X = scaler.transform(features).clip(0.0, 1.0)

        print(f"  [Trainer] Fine-tuning autoencoder on {len(X)} rows...")
        history = model.fit(
            X, X,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            verbose=0
        )
        final_loss = history.history["loss"][-1]
        print(f"  [Trainer] AE fine-tune complete — final loss: {final_loss:.6f}")

        # Recompute threshold on the updated model's error distribution
        reconstructions = model.predict(X, verbose=0)
        errors          = np.mean(np.square(X - reconstructions), axis=1)
        new_threshold   = float(np.percentile(errors, 85))

        # Atomic save — temp swap avoids corrupt file on interruption
        tmp_model = model_path + ".tmp"
        model.save(tmp_model)
        os.replace(tmp_model, model_path)

        with open(threshold_path, "w") as f:
            json.dump({"threshold": new_threshold}, f, indent=2)

        print(f"  [Trainer] AE model     saved → {model_path}")
        print(f"  [Trainer] AE threshold saved → {threshold_path}  "
              f"(value={new_threshold:.6f})")

    except Exception as e:
        print(f"  [Trainer] Autoencoder fine-tuning failed: {e}")


# ══════════════════════════════════════════════════════════
# 4.  FULL RETRAINING RUN
# ══════════════════════════════════════════════════════════
def run_retraining():
    """Runs all three retraining steps in sequence."""
    print(f"\n{'─'*60}")
    print(f"  [Trainer] Starting scheduled retraining — "
          f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'─'*60}")

    update_rule_thresholds()
    retrain_isolation_forest()
    finetune_autoencoder()

    print(f"  [Trainer] Retraining complete.\n")


# ══════════════════════════════════════════════════════════
# 5.  BACKGROUND SCHEDULER
# ══════════════════════════════════════════════════════════
def start_training_scheduler(interval=RETRAIN_INTERVAL):
    """
    Launches the retraining loop in a daemon background thread.
    Automatically stops when the main process exits.
    """
    def _loop():
        print(f"  [Trainer] Scheduler started — retraining every "
              f"{interval//60} minutes.")
        while True:
            time.sleep(interval)
            run_retraining()

    t = threading.Thread(target=_loop, daemon=True, name="SkyGate-Trainer")
    t.start()
    return t


# ══════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════
def _load_log(log_path):
    """
    Loads the live log CSV, applies the sliding window,
    and validates that enough rows exist to retrain.
    """
    if not os.path.exists(log_path):
        print(f"  [Trainer] No live log found at {log_path} — skipping.")
        return None

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"  [Trainer] Failed to read log: {e}")
        return None

    if len(df) < MIN_ROWS_TO_RETRAIN:
        print(f"  [Trainer] Only {len(df)} rows in log "
              f"(need {MIN_ROWS_TO_RETRAIN}) — skipping retraining.")
        return None

    df = df.tail(SLIDING_WINDOW).reset_index(drop=True)
    print(f"  [Trainer] Loaded {len(df)} rows from live log (sliding window).")
    return df


def _extract_features(df):
    """
    Extracts and validates feature columns. Returns clean DataFrame or None.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 3:
        print(f"  [Trainer] Not enough feature columns: {available}")
        return None

    features = df[available].dropna()
    if len(features) < MIN_ROWS_TO_RETRAIN:
        print(f"  [Trainer] Too many NaN rows — only {len(features)} clean rows.")
        return None

    return features


# ══════════════════════════════════════════════════════════
# CLI — manual trigger for testing
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running manual retraining cycle...")
    run_retraining()