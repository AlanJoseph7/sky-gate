import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from utils import get_data_path, ensure_data_dirs

# -----------------------------
# CONFIG
# -----------------------------
NUM_AIRCRAFT       = 20
POINTS_PER_FLIGHT  = 300
ANOMALY_RATE       = 0.05
RANDOM_SEED        = 42

# Physical bounds
MAX_ALTITUDE_FT    = 43000
MIN_ALTITUDE_FT    = 1000
MAX_VELOCITY_KNOTS = 650
MIN_VELOCITY_KNOTS = 150
LAT_BOUNDS         = (10, 30)
LON_BOUNDS         = (70, 90)

# -----------------------------
# HELPERS
# -----------------------------
def generate_icao(rng):
    return hex(rng.randint(0x100000, 0xFFFFFF))[2:]

def haversine_step(lat, lon, distance_km, bearing_deg):
    R = 6371

    bearing = np.radians(bearing_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(distance_km / R) +
        np.cos(lat1) * np.sin(distance_km / R) * np.cos(bearing)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(distance_km / R) * np.cos(lat1),
        np.cos(distance_km / R) - np.sin(lat1) * np.sin(lat2)
    )

    return np.degrees(lat2), np.degrees(lon2)

# -----------------------------
# NORMAL FLIGHT GENERATION
# -----------------------------
def generate_normal_flight(rng, start_time):
    data = []

    icao     = generate_icao(rng)
    lat      = rng.uniform(*LAT_BOUNDS)
    lon      = rng.uniform(*LON_BOUNDS)
    altitude = rng.uniform(10000, 35000)
    velocity = rng.uniform(300, 600)
    heading  = rng.uniform(0, 360)

    timestamp = start_time

    for _ in range(POINTS_PER_FLIGHT):
        altitude = np.clip(altitude + rng.uniform(-50, 50),  MIN_ALTITUDE_FT, MAX_ALTITUDE_FT)
        velocity = np.clip(velocity + rng.uniform(-5, 5),    MIN_VELOCITY_KNOTS, MAX_VELOCITY_KNOTS)
        heading  = (heading + rng.uniform(-2, 2)) % 360

        lat, lon = haversine_step(lat, lon, velocity / 3600, heading)

        data.append([
            timestamp, icao, lat, lon,
            altitude, velocity, heading, 0
        ])

        timestamp += timedelta(seconds=5)

    return data

# -----------------------------
# ANOMALY INJECTION
# -----------------------------
def inject_anomalies(df, rng):
    """
    Anomaly types split into two categories:

    SINGLE-POINT anomalies (detectable by rules + Isolation Forest):
        - altitude_jump  : sudden large altitude change at one point
        - teleport       : sudden lat/lon jump at one point
        - speed_spike    : sudden velocity spike at one point

    SEQUENCE anomalies (detectable by LSTM — gradual drift over 10 steps):
        - gradual_speed_drift   : velocity increases with noise over 10 steps
        - altitude_oscillation  : variable up/down altitude pattern over 10 steps
        - heading_oscillation   : variable zigzag heading over 10 steps

    FIX: sequence anomalies now have RANDOM magnitude per step so they
    are not perfectly predictable. A perfectly regular pattern (e.g.
    exactly +2000ft every other step) is actually easier for the LSTM
    to reconstruct than noisy normal data, producing ratio < 1.0x.
    Adding per-step randomness makes anomalous sequences genuinely
    harder to reconstruct than normal ones.
    """
    df = df.copy().reset_index(drop=True)

    num_anomalies = int(ANOMALY_RATE * len(df))
    print(f"  Targeting {num_anomalies} anomalies ({ANOMALY_RATE*100:.0f}% of {len(df)} rows)")

    # Spaced sampling — no two anomaly starts within 15 rows of each other
    # (15 > SEQUENCE_LENGTH so sequence anomalies don't overlap)
    candidate_indices = list(range(15, len(df) - 15))
    spaced = []
    last   = -30

    for idx in sorted(candidate_indices):
        if idx - last >= 15:
            spaced.append(idx)
            last = idx

    anomaly_indices = rng.sample(spaced, min(num_anomalies, len(spaced)))

    anomaly_types = [
        "altitude_jump",          # single-point — rules/IF
        "teleport",               # single-point — rules/IF
        "speed_spike",            # single-point — rules/IF
        "gradual_speed_drift",    # sequence     — LSTM
        "altitude_oscillation",   # sequence     — LSTM
        "heading_oscillation",    # sequence     — LSTM
    ]

    type_counts = {t: 0 for t in anomaly_types}

    for idx in anomaly_indices:
        anomaly_type = rng.choice(anomaly_types)
        type_counts[anomaly_type] += 1

        # ---------------------------
        # SINGLE-POINT ANOMALIES
        # ---------------------------
        if anomaly_type == "altitude_jump":
            new_alt = df.loc[idx, "altitude"] + rng.uniform(5000, 10000)
            df.loc[idx, "altitude"] = min(new_alt, MAX_ALTITUDE_FT)
            df.loc[idx, "label"]    = 1

        elif anomaly_type == "teleport":
            df.loc[idx, "latitude"]  = np.clip(
                df.loc[idx, "latitude"]  + rng.uniform(3, 8), *LAT_BOUNDS
            )
            df.loc[idx, "longitude"] = np.clip(
                df.loc[idx, "longitude"] + rng.uniform(3, 8), *LON_BOUNDS
            )
            df.loc[idx, "label"] = 1

        elif anomaly_type == "speed_spike":
            new_vel = df.loc[idx, "velocity"] + rng.uniform(200, 400)
            df.loc[idx, "velocity"] = min(new_vel, MAX_VELOCITY_KNOTS)
            df.loc[idx, "label"]    = 1

        # ---------------------------
        # SEQUENCE ANOMALIES (10 steps)
        # FIX: variable magnitude per step via rng so patterns are
        # not perfectly regular and are harder to reconstruct
        # ---------------------------
        elif anomaly_type == "gradual_speed_drift":
            # Velocity accelerates ~8% per step WITH per-step noise
            # Normal: ±5 knot variation per step
            # Anomaly: compounding 6-10% increase per step — clearly abnormal
            for j in range(idx, min(idx + 10, len(df))):
                factor = 1 + (0.08 + rng.uniform(-0.02, 0.02)) * (j - idx)
                df.loc[j, "velocity"] = min(
                    df.loc[j, "velocity"] * factor,
                    MAX_VELOCITY_KNOTS
                )
                df.loc[j, "label"] = 1

        elif anomaly_type == "altitude_oscillation":
            # Rapid oscillation with variable magnitude per step
            # Normal: ±50ft variation per step
            # Anomaly: ±1500-2500ft oscillation — clearly abnormal
            for j in range(idx, min(idx + 10, len(df))):
                direction = 1 if (j - idx) % 2 == 0 else -1
                magnitude = rng.uniform(1500, 2500)
                df.loc[j, "altitude"] = np.clip(
                    df.loc[j, "altitude"] + direction * magnitude,
                    MIN_ALTITUDE_FT, MAX_ALTITUDE_FT
                )
                df.loc[j, "label"] = 1

        elif anomaly_type == "heading_oscillation":
            # Zigzag heading with variable magnitude per step
            # Normal: ±2° heading change per step
            # Anomaly: ±20-45° zigzag — clearly abnormal
            for j in range(idx, min(idx + 10, len(df))):
                direction = 1 if (j - idx) % 2 == 0 else -1
                magnitude = rng.uniform(20, 45)
                df.loc[j, "heading"] = (
                    df.loc[j, "heading"] + direction * magnitude
                ) % 360
                df.loc[j, "label"] = 1

    # Print breakdown
    print(f"\n  Anomaly type breakdown:")
    for atype, count in type_counts.items():
        category = "sequence " if atype in [
            "gradual_speed_drift", "altitude_oscillation", "heading_oscillation"
        ] else "single-pt"
        print(f"    [{category}] {atype:<25} : {count}")

    return df

# -----------------------------
# MAIN GENERATION FUNCTION
# -----------------------------
def generate_dataset(seed=RANDOM_SEED):
    rng = random.Random(seed)
    np.random.seed(seed)

    all_data  = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(NUM_AIRCRAFT):
        start_time = base_time + timedelta(minutes=i * 10)
        flight     = generate_normal_flight(rng, start_time)
        all_data.extend(flight)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "icao", "latitude", "longitude",
        "altitude", "velocity", "heading", "label"
    ])

    df = inject_anomalies(df, rng)

    print(f"\nTotal rows      : {len(df)}")
    print(f"Normal rows     : {(df['label'] == 0).sum()}")
    print(f"Anomaly rows    : {(df['label'] == 1).sum()}")
    print(f"Anomaly rate    : {df['label'].mean()*100:.1f}%")

    return df

# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()

    df = generate_dataset()

    df.to_csv(get_data_path("adsb_raw.csv"), index=False)

    print("\nDataset generated at data/adsb_raw.csv")