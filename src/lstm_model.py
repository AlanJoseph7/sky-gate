import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tf_keras.models import Model
from tf_keras.layers import Input, Dense, Dropout
from tf_keras.callbacks import EarlyStopping

from utils import get_data_path, ensure_data_dirs, FEATURE_COLUMNS, RANDOM_SEED, load_keras_model_compatible

# -----------------------------
# CONFIG
# -----------------------------
EPOCHS     = 100
BATCH_SIZE = 32

# -----------------------------
# MODEL PATHS
# -----------------------------
AE_MODEL_PATH     = "models/autoencoder.keras"
AE_SCALER_PATH    = "models/ae_scaler.pkl"
AE_THRESHOLD_PATH = "models/ae_threshold.json"


# -----------------------------
# REMOVE ANOMALY NEIGHBORS
# -----------------------------
def clean_normal_data(df):
    df = df.copy()

    remove_indices  = set()
    anomaly_indices = df[df["label"] == 1].index

    for idx in anomaly_indices:
        for i in range(idx - 2, idx + 3):
            if i in df.index:
                remove_indices.add(i)

    clean_df = df.drop(index=remove_indices, errors="ignore")
    clean_df = clean_df[clean_df["label"] == 0]

    return clean_df, clean_df.index


# -----------------------------
# CLIP OUTLIERS
# -----------------------------
def clip_outliers(df, feature_cols, clean_indices, multiplier=4.0):
    """
    Clips each feature to [mean - multiplier*std, mean + multiplier*std]
    computed on clean normal data only, so anomaly extremes don't
    collapse the MinMaxScaler range for normal data.
    """
    df       = df.copy()
    clean_df = df.loc[clean_indices, feature_cols]

    print("\n--- OUTLIER CLIPPING ---")
    for col in feature_cols:
        mean       = clean_df[col].mean()
        std        = clean_df[col].std()
        lower      = mean - multiplier * std
        upper      = mean + multiplier * std
        before_min = df[col].min()
        before_max = df[col].max()
        df[col]    = df[col].clip(lower, upper)
        after_min  = df[col].min()
        after_max  = df[col].max()
        if round(before_max, 2) != round(after_max, 2) or round(before_min, 2) != round(after_min, 2):
            print(f"  {col:<22} [{before_min:.2f}, {before_max:.2f}] → [{after_min:.2f}, {after_max:.2f}]")

    return df


# -----------------------------
# BUILD FEEDFORWARD AUTOENCODER
# -----------------------------
def build_model(n_features):
    inputs = Input(shape=(n_features,))

    # Encoder
    x = Dense(32, activation="relu")(inputs)
    x = Dropout(0.1)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(8,  activation="relu")(x)
    x = Dropout(0.1)(x)
    encoded = Dense(2, activation="relu")(x)   # bottleneck

    # Decoder
    x = Dense(8,  activation="relu")(encoded)
    x = Dropout(0.1)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    decoded = Dense(n_features, activation="sigmoid")(x)

    model = Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")

    return model


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run_lstm_autoencoder(df):
    """
    Function name kept as run_lstm_autoencoder for pipeline compatibility.
    Internally uses a feedforward autoencoder.

    Synthetic mode: trains from scratch, saves model/scaler/threshold.
    Live mode     : loads saved model/scaler/threshold, skips training.
    """
    df = df.copy()
    df = df.sort_values(by=["icao", "timestamp"]).reset_index(drop=True)

    is_live_mode = (df["label"].sum() == 0)

    # ── LIVE MODE ─────────────────────────────────────────────────
    if is_live_mode:
        return _run_inference(df)

    # ── SYNTHETIC MODE ────────────────────────────────────────────
    return _run_training(df)


# ══════════════════════════════════════════════════════════
# INFERENCE (live mode)
# ══════════════════════════════════════════════════════════
def _run_inference(df):
    """Loads saved model, scaler, and threshold and runs inference only."""
    missing = [
        p for p in [AE_MODEL_PATH, AE_SCALER_PATH, AE_THRESHOLD_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        print(f"  [AE] WARNING: Missing saved files: {missing}")
        print("               Run synthetic mode first. Defaulting to all-normal.")
        df["lstm_error"]   = 0.0
        df["lstm_anomaly"] = 0
        return df

    # Load artefacts
    model  = load_keras_model_compatible(AE_MODEL_PATH)
    scaler = joblib.load(AE_SCALER_PATH)
    with open(AE_THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    print(f"  [AE] Live mode — loaded model from {AE_MODEL_PATH} | threshold={threshold:.6f}")

    # Pre-flight NaN/Inf cleanup
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    X_all = scaler.transform(df[FEATURE_COLUMNS]).clip(0.0, 1.0)

    reconstructions       = model.predict(X_all, verbose=0)
    errors                = np.mean(np.square(X_all - reconstructions), axis=1)
    df["lstm_error"]      = errors
    df["lstm_anomaly"]    = (errors > threshold).astype(int)

    flagged = df["lstm_anomaly"].sum()
    print(f"  [AE] {flagged}/{len(df)} flights flagged (threshold={threshold:.6f})")

    return df


# ══════════════════════════════════════════════════════════
# TRAINING (synthetic mode)
# ══════════════════════════════════════════════════════════
def _run_training(df):
    """Full training run. Saves model, scaler, and threshold to disk."""

    # -----------------------------
    # PRE-FLIGHT CHECKS
    # -----------------------------
    print("\n--- PRE-FLIGHT CHECKS ---")

    nan_count = df[FEATURE_COLUMNS].isna().sum().sum()
    inf_count = np.isinf(df[FEATURE_COLUMNS].values).sum()

    print(f"  NaN count in features : {nan_count}")
    print(f"  Inf count in features : {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("  Cleaning NaN/Inf values before proceeding...")
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    print(f"  Total rows            : {len(df)}")
    print(f"  Normal rows           : {(df['label'] == 0).sum()}")
    print(f"  Anomaly rows          : {(df['label'] == 1).sum()}")
    print(f"  Aircraft count        : {df['icao'].nunique()}")

    # -----------------------------
    # CLEAN NORMAL DATA
    # -----------------------------
    _, clean_indices = clean_normal_data(df)
    print(f"\n  Clean normal rows (excl. anomaly neighbours): {len(clean_indices)}")

    # -----------------------------
    # CLIP OUTLIERS
    # -----------------------------
    df = clip_outliers(df, FEATURE_COLUMNS, clean_indices, multiplier=4.0)

    # -----------------------------
    # SCALING — fit on clean normal rows only
    # -----------------------------
    clean_normal_df            = df.loc[clean_indices]
    scaler                     = MinMaxScaler()
    scaler.fit(clean_normal_df[FEATURE_COLUMNS])

    df_scaled                  = df.copy()
    df_scaled[FEATURE_COLUMNS] = scaler.transform(
        df[FEATURE_COLUMNS]
    ).clip(0.0, 1.0)

    scaled_min = df_scaled[FEATURE_COLUMNS].min().min()
    scaled_max = df_scaled[FEATURE_COLUMNS].max().max()
    print(f"\n  Scaled feature range  : [{scaled_min:.4f}, {scaled_max:.4f}]")

    X_train = df_scaled.loc[clean_indices, FEATURE_COLUMNS].values
    X_all   = df_scaled[FEATURE_COLUMNS].values

    print(f"  Training rows        : {X_train.shape}")
    print(f"  Evaluation rows      : {X_all.shape}")

    # -----------------------------
    # ADD NOISE
    # -----------------------------
    noise_factor  = 0.08
    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)

    # -----------------------------
    # MODEL
    # -----------------------------
    np.random.seed(RANDOM_SEED)

    model = build_model(len(FEATURE_COLUMNS))
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    print("\n--- TRAINING ---")
    model.fit(
        X_train_noisy,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # -----------------------------
    # RECONSTRUCTION ERROR + THRESHOLD
    # -----------------------------
    reconstructions = model.predict(X_all)
    errors          = np.mean(np.square(X_all - reconstructions), axis=1)

    df_result               = df.copy()
    df_result["lstm_error"] = errors

    normal_errors = df_result.loc[df_result["label"] == 0, "lstm_error"].values
    threshold     = float(np.percentile(normal_errors, 85))

    df_result["lstm_anomaly"] = (df_result["lstm_error"] > threshold).astype(int)

    # -----------------------------
    # SAVE MODEL, SCALER, THRESHOLD
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    model.save(AE_MODEL_PATH)
    joblib.dump(scaler, AE_SCALER_PATH)
    with open(AE_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    print(f"\n  [AE] Model     saved → {AE_MODEL_PATH}")
    print(f"  [AE] Scaler    saved → {AE_SCALER_PATH}")
    print(f"  [AE] Threshold saved → {AE_THRESHOLD_PATH}  (value={threshold:.6f})")

    # -----------------------------
    # DIAGNOSTICS
    # -----------------------------
    print(f"\n--- AUTOENCODER DIAGNOSTICS ---")
    print(f"  Threshold (85th pct of normal errors) : {threshold:.6f}")
    print(f"  Mean error — normal                   : {normal_errors.mean():.6f}")

    anomaly_errors = df_result.loc[df_result["label"] == 1, "lstm_error"].values
    if len(anomaly_errors):
        print(f"  Mean error — anomaly                  : {anomaly_errors.mean():.6f}")
        separation = (
            anomaly_errors.mean() / normal_errors.mean()
            if normal_errors.mean() > 0 else 0
        )
        print(f"  Anomaly/Normal error ratio            : {separation:.2f}x  (want > 2.0x)")

    # -----------------------------
    # EVALUATION
    # -----------------------------
    y_true = df_result["label"].values
    y_pred = df_result["lstm_anomaly"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred,    zero_division=0)
    f1        = f1_score(y_true, y_pred,         zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"]
    )

    print(f"\n--- AUTOENCODER EVALUATION ---")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(cm_df.to_string())

    return df_result


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()
    df      = pd.read_csv(get_data_path("adsb_if.csv"))
    df_lstm = run_lstm_autoencoder(df)
    df_lstm.to_csv(get_data_path("adsb_lstm.csv"), index=False)
    print("\nAutoencoder results saved at data/adsb_lstm.csv")