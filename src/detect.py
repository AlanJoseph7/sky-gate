import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Where ensemble weights + threshold are persisted after synthetic training
WEIGHTS_PATH = "models/ensemble_weights.json"

# -----------------------------
# FINAL DETECTION LOGIC
# -----------------------------
def combine_detections(df):
    df = df.copy()

    # Ensure detector columns exist
    required_cols = ["rule_anomaly", "if_anomaly", "lstm_anomaly", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # ──────────────────────────────────────────────────────────
    # LIVE MODE DETECTION
    # In live mode label is all zeros — F1 scores would be 0 for
    # every detector, causing equal-weight fallback which ignores
    # what was learned during synthetic training.
    # Instead, load the weights + threshold saved from the last
    # synthetic training run.
    # ──────────────────────────────────────────────────────────
    is_live_mode = (df["label"].sum() == 0)

    if is_live_mode:
        w_rule, w_if, w_lstm, best_thresh = _load_weights()
        print(f"  [detect] Live mode — loaded saved weights: "
              f"Rule={w_rule:.3f} | IF={w_if:.3f} | LSTM={w_lstm:.3f} | "
              f"Threshold={best_thresh:.2f}")

    else:
        # ──────────────────────────────────────────────────────
        # SYNTHETIC MODE — compute weights from F1, then save
        # ──────────────────────────────────────────────────────
        f1_rule = f1_score(df["label"], df["rule_anomaly"], zero_division=0)
        f1_if   = f1_score(df["label"], df["if_anomaly"],   zero_division=0)
        f1_lstm = f1_score(df["label"], df["lstm_anomaly"], zero_division=0)

        print(f"Rule F1 : {f1_rule:.4f}")
        print(f"IF F1   : {f1_if:.4f}")
        print(f"LSTM F1 : {f1_lstm:.4f}")

        total = f1_rule + f1_if + f1_lstm
        if total == 0:
            print("WARNING: All detectors have F1=0. Using equal weights.")
            w_rule = w_if = w_lstm = 1 / 3
        else:
            w_rule = f1_rule / total
            w_if   = f1_if   / total
            w_lstm = f1_lstm / total

        print(f"\nDerived weights → Rule: {w_rule:.3f} | IF: {w_if:.3f} | LSTM: {w_lstm:.3f}")

        # Compute best threshold on synthetic data
        score_tmp = (
            w_rule * df["rule_anomaly"] +
            w_if   * df["if_anomaly"]   +
            w_lstm * df["lstm_anomaly"]
        )
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_thresh = max(
            thresholds,
            key=lambda t: f1_score(
                df["label"],
                (score_tmp >= t).astype(int),
                zero_division=0
            )
        )
        print(f"Best threshold (max F1): {best_thresh:.2f}")

        # Persist for live mode
        _save_weights(w_rule, w_if, w_lstm, best_thresh)

    # -----------------------------
    # WEIGHTED SCORE (both modes)
    # -----------------------------
    df["final_score"] = (
        w_rule * df["rule_anomaly"] +
        w_if   * df["if_anomaly"]   +
        w_lstm * df["lstm_anomaly"]
    )

    df["final_anomaly"] = (df["final_score"] >= best_thresh).astype(int)

    return df


# -----------------------------
# WEIGHT PERSISTENCE HELPERS
# -----------------------------
def _save_weights(w_rule, w_if, w_lstm, threshold):
    """Saves ensemble weights and decision threshold to disk."""
    os.makedirs("models", exist_ok=True)
    payload = {
        "w_rule":     round(float(w_rule),     6),
        "w_if":       round(float(w_if),       6),
        "w_lstm":     round(float(w_lstm),     6),
        "threshold":  round(float(threshold),  4),
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [detect] Ensemble weights saved to {WEIGHTS_PATH}")


def _load_weights():
    """
    Loads saved ensemble weights and threshold.
    Falls back to equal weights (0.333) and threshold 0.4
    if no file exists yet (e.g. first run before synthetic training).
    """
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r") as f:
            p = json.load(f)
        return p["w_rule"], p["w_if"], p["w_lstm"], p["threshold"]
    else:
        print("  [detect] WARNING: No saved weights found. "
              "Run synthetic mode first. Using equal fallback weights.")
        return 1/3, 1/3, 1/3, 0.4


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    from utils import get_data_path, ensure_data_dirs
    ensure_data_dirs()
    df = pd.read_csv(get_data_path("adsb_lstm.csv")) 

    df_final = combine_detections(df)

    df_final.to_csv("../data/adsb_final.csv", index=False)
    print("\nFinal anomaly results saved at data/adsb_final.csv")

    # -----------------------------
    # FULL EVALUATION
    # -----------------------------
    print("\n--- SUMMARY ---")
    print(f"True anomalies    : {df['label'].sum()}")
    print(f"Rule detected     : {df['rule_anomaly'].sum()}")
    print(f"IF detected       : {df['if_anomaly'].sum()}")
    print(f"LSTM detected     : {df['lstm_anomaly'].sum()}")
    print(f"Final detected    : {df_final['final_anomaly'].sum()}")

    print("\n--- FINAL ENSEMBLE EVALUATION ---")
    y_true = df_final["label"]
    y_pred = df_final["final_anomaly"]

    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")