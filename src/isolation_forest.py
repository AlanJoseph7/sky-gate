import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

from utils import get_data_path, ensure_data_dirs, FEATURE_COLUMNS, RANDOM_SEED

# -----------------------------
# MODEL PATHS
# -----------------------------
IF_MODEL_PATH  = "models/iso_forest.pkl"
IF_SCALER_PATH = "models/if_scaler.pkl"


# -----------------------------
# REMOVE ANOMALY NEIGHBOURS
# -----------------------------
def get_clean_normal_indices(df):
    """
    Exclude rows adjacent to anomalies from training,
    same approach used in the autoencoder to keep both models consistent.
    """
    remove_indices = set()
    anomaly_indices = df[df["label"] == 1].index

    for idx in anomaly_indices:
        for i in range(idx - 2, idx + 3):
            if i in df.index:
                remove_indices.add(i)

    clean_indices = df[
        (df["label"] == 0) & (~df.index.isin(remove_indices))
    ].index

    return clean_indices


# -----------------------------
# TRAIN + PREDICT
# -----------------------------
def run_isolation_forest(df, contamination=None):
    df = df.copy()

    is_live_mode = (df["label"].sum() == 0)

    if is_live_mode:
        # ── LIVE MODE: load saved model + scaler ──────────────────
        if not os.path.exists(IF_MODEL_PATH) or not os.path.exists(IF_SCALER_PATH):
            print("  [IF] WARNING: No saved model found. Run synthetic mode first.")
            print("               Falling back to untrained predictions (all normal).")
            df["if_score"]   = 0.0
            df["if_anomaly"] = 0
            return df

        model  = joblib.load(IF_MODEL_PATH)
        scaler = joblib.load(IF_SCALER_PATH)
        print(f"  [IF] Live mode — loaded model from {IF_MODEL_PATH}")

        X_all_scaled     = scaler.transform(df[FEATURE_COLUMNS])
        df["if_score"]   = model.decision_function(X_all_scaled)
        raw_predictions  = model.predict(X_all_scaled)
        df["if_anomaly"] = (raw_predictions == -1).astype(int)

        return df

    # ── SYNTHETIC MODE: train, evaluate, then save ────────────────

    # Derive contamination from actual label rate
    if contamination is None:
        contamination = round(float(df["label"].mean()), 4)
        contamination = np.clip(contamination, 0.01, 0.5)
        print(f"Auto contamination rate: {contamination:.4f}")

    # -----------------------------
    # CLEAN TRAINING DATA
    # -----------------------------
    clean_indices = get_clean_normal_indices(df)
    train_df      = df.loc[clean_indices]

    print(f"Training rows (clean normal): {len(train_df)}")
    print(f"Evaluation rows (all)       : {len(df)}")

    X_train = train_df[FEATURE_COLUMNS]
    X_all   = df[FEATURE_COLUMNS]

    # -----------------------------
    # SCALING — fit on clean normal only
    # -----------------------------
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled   = scaler.transform(X_all)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train_scaled)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    df["if_score"]   = model.decision_function(X_all_scaled)
    raw_predictions  = model.predict(X_all_scaled)
    df["if_anomaly"] = (raw_predictions == -1).astype(int)

    # -----------------------------
    # SAVE MODEL + SCALER
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  IF_MODEL_PATH)
    joblib.dump(scaler, IF_SCALER_PATH)
    print(f"  [IF] Model  saved → {IF_MODEL_PATH}")
    print(f"  [IF] Scaler saved → {IF_SCALER_PATH}")

    return df


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()

    df = pd.read_csv(get_data_path("adsb_rules.csv"))
    df_if = run_isolation_forest(df)
    df_if.to_csv(get_data_path("adsb_if.csv"), index=False)
    print("Isolation Forest results saved at data/adsb_if.csv")

    y_true = df_if["label"]
    y_pred = df_if["if_anomaly"]

    print("\n--- ISOLATION FOREST EVALUATION ---")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred,    zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred,        zero_division=0):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"]
    )
    print(f"\nConfusion Matrix:\n{cm_df.to_string()}")