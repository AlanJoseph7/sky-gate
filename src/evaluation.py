import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils import get_data_path

# -----------------------------
# EVALUATE A SINGLE DETECTOR
# -----------------------------
def evaluate_detector(y_true, y_pred, name="Detector"):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred,    zero_division=0)
    f1        = f1_score(y_true, y_pred,         zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    # Labelled confusion matrix — much easier to read
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"]
    )

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(cm_df.to_string())

    return {"name": name, "precision": precision, "recall": recall, "f1": f1}


# -----------------------------
# EVALUATE ALL DETECTORS
# -----------------------------
def evaluate_all(df):
    y_true = df["label"]

    # Print class distribution for context
    total     = len(y_true)
    n_anomaly = y_true.sum()
    print(f"\n--- CLASS DISTRIBUTION ---")
    print(f"  Total samples : {total}")
    print(f"  Normal        : {total - n_anomaly} ({(total - n_anomaly)/total*100:.1f}%)")
    print(f"  Anomaly       : {n_anomaly} ({n_anomaly/total*100:.1f}%)")

    # Detectors to evaluate — only include columns that exist
    detectors = {
        "Rule-Based"     : "rule_anomaly",
        "Isolation Forest": "if_anomaly",
        "LSTM Autoencoder": "lstm_anomaly",
        "Final Ensemble" : "final_anomaly",
    }

    results = []
    for name, col in detectors.items():
        if col in df.columns:
            results.append(evaluate_detector(y_true, df[col], name))
        else:
            print(f"\n[SKIPPED] Column '{col}' not found — run the full pipeline first.")

    # -----------------------------
    # COMPARISON TABLE
    # -----------------------------
    if results:
        print(f"\n{'='*40}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*40}")
        summary_df = pd.DataFrame(results).set_index("name")
        summary_df = summary_df.sort_values("f1", ascending=False)
        print(summary_df.to_string(float_format="{:.4f}".format))

    return results


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    # Use adsb_final.csv since it contains all detector columns
    # Falls back gracefully if only adsb_lstm.csv is available
    try:
        df = pd.read_csv(get_data_path("adsb_final.csv"))
        print("Loaded: adsb_final.csv")
    except FileNotFoundError:
        df = pd.read_csv(get_data_path("adsb_lstm.csv"))
        print("Loaded: adsb_lstm.csv (adsb_final.csv not found — run detect.py first)")

    evaluate_all(df)