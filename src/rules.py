import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils import get_data_path, ensure_data_dirs

# -----------------------------
# THRESHOLDS (TUNABLE)
# FIX #1: units corrected to match feature engineering output
#   - velocity    : knots  (generator uses knots, features.py preserves this)
#   - climb_rate  : ft/s   (altitude in feet ÷ delta_time in seconds)
#   - distance_km : km     (haversine output, correct)
#   - heading_change: degrees (0–180 after wrap, correct)
# -----------------------------
MAX_CLIMB_RATE     = 50    # ft/s  (~3000 fpm, aggressive but possible)
MAX_SPEED          = 630   # knots (above any commercial aircraft limit)
MAX_DISTANCE       = 50    # km per timestep (teleport detection)
MAX_HEADING_CHANGE = 45    # degrees per timestep

# -----------------------------
# RULE-BASED DETECTION
# -----------------------------
def apply_rules(df):
    df = df.copy()

    # FIX #5: explicitly fill NaN before comparisons so behaviour is transparent
    climb_rate     = df["climb_rate"].abs().fillna(0)
    velocity       = df["velocity"].fillna(0)
    distance_km    = df["distance_km"].fillna(0)
    heading_change = df["heading_change"].fillna(0)

    # Individual rule flags (kept for diagnostics printout but not saved to CSV)
    rule_climb    = climb_rate     > MAX_CLIMB_RATE
    rule_speed    = velocity       > MAX_SPEED
    rule_distance = distance_km   > MAX_DISTANCE
    rule_heading  = heading_change > MAX_HEADING_CHANGE

    # Combined anomaly flag
    df["rule_anomaly"] = (
        rule_climb    |
        rule_speed    |
        rule_distance |
        rule_heading
    ).astype(int)

    # FIX #7: per-rule summary (diagnostic only, not saved to CSV)
    print("\n--- RULE TRIGGER SUMMARY ---")
    print(f"  rule_climb    (>{MAX_CLIMB_RATE} ft/s)   : {rule_climb.sum():>5} rows")
    print(f"  rule_speed    (>{MAX_SPEED} knots)        : {rule_speed.sum():>5} rows")
    print(f"  rule_distance (>{MAX_DISTANCE} km/step)   : {rule_distance.sum():>5} rows")
    print(f"  rule_heading  (>{MAX_HEADING_CHANGE} deg) : {rule_heading.sum():>5} rows")
    print(f"  Total flagged                             : {df['rule_anomaly'].sum():>5} rows")

    # FIX #4: do NOT save intermediate rule flag columns to CSV
    return df

# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()

    # FIX #2: use get_data_path instead of hardcoded relative path
    df = pd.read_csv(get_data_path("adsb_features.csv"))

    df_rules = apply_rules(df)

    df_rules.to_csv(get_data_path("adsb_rules.csv"), index=False)
    print("\nRule-based anomalies saved at data/adsb_rules.csv")

    # FIX #3: print evaluation metrics
    y_true = df_rules["label"]
    y_pred = df_rules["rule_anomaly"]

    print("\n--- RULE-BASED EVALUATION ---")
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