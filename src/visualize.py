import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import get_data_path, ensure_data_dirs

# -----------------------------
# HELPER — SAVE FIGURE
# -----------------------------
def save_fig(filename):
    """Save to data/plots/ and close figure cleanly."""
    path = get_data_path(f"plots/{filename}")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# -----------------------------
# 1. TRAJECTORY PLOT
# FIX #2: uses final_anomaly (falls back to lstm_anomaly if not present)
# FIX #3: explicit colours — red anomalies, steel-blue normal
# FIX #1: saves to disk instead of plt.show()
# -----------------------------
def plot_trajectory(df):
    anomaly_col = "final_anomaly" if "final_anomaly" in df.columns else "lstm_anomaly"

    fig, ax = plt.subplots(figsize=(12, 7))

    for icao in df["icao"].unique():
        aircraft_df = df[df["icao"] == icao].sort_values("timestamp")

        normal  = aircraft_df[aircraft_df[anomaly_col] == 0]
        anomaly = aircraft_df[aircraft_df[anomaly_col] == 1]

        ax.plot(
            normal["longitude"], normal["latitude"],
            color="steelblue", linewidth=0.8, alpha=0.6
        )
        if len(anomaly):
            ax.scatter(
                anomaly["longitude"], anomaly["latitude"],
                color="red", s=40, zorder=5, label="_nolegend_"
            )

    normal_patch  = mpatches.Patch(color="steelblue", label="Normal")
    anomaly_patch = mpatches.Patch(color="red",       label="Anomaly")
    ax.legend(handles=[normal_patch, anomaly_patch])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"ADS-B Flight Trajectory — Anomalies from `{anomaly_col}`")
    ax.grid(True, alpha=0.3)

    save_fig("trajectory.png")


# -----------------------------
# 2. LSTM ERROR PLOT
# FIX #4: uses reset integer index to avoid index gap illusion
# FIX #3: explicit colours
# FIX #1: saves to disk
# -----------------------------
def plot_lstm_error(df):
    if "lstm_error" not in df.columns:
        print("  [SKIP] No lstm_error column found — run lstm_autoencoder.py first.")
        return

    plot_df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        plot_df.index, plot_df["lstm_error"],
        color="steelblue", linewidth=0.7, label="Reconstruction Error"
    )

    anomaly_mask = plot_df["lstm_anomaly"] == 1
    ax.scatter(
        plot_df.index[anomaly_mask],
        plot_df.loc[anomaly_mask, "lstm_error"],
        color="red", s=20, zorder=5, label="Flagged Anomaly"
    )

    # Draw threshold line if computable
    if "label" in plot_df.columns:
        normal_errors = plot_df.loc[plot_df["label"] == 0, "lstm_error"]
        threshold = np.percentile(normal_errors, 95)
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1, label=f"Threshold (95th pct = {threshold:.4f})")

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("MSE Reconstruction Error")
    ax.set_title("LSTM Reconstruction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("lstm_error.png")


# -----------------------------
# 3. ERROR DISTRIBUTION (FIX #5 — NEW)
# Overlaid histogram of normal vs anomaly reconstruction errors.
# This is the most diagnostic plot for an autoencoder.
# -----------------------------
def plot_error_distribution(df):
    if "lstm_error" not in df.columns or "label" not in df.columns:
        print("  [SKIP] Need lstm_error and label columns.")
        return

    normal_errors  = df.loc[df["label"] == 0, "lstm_error"]
    anomaly_errors = df.loc[df["label"] == 1, "lstm_error"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(normal_errors,  bins=60, alpha=0.6, color="steelblue", label="Normal",  density=True)
    ax.hist(anomaly_errors, bins=60, alpha=0.6, color="red",       label="Anomaly", density=True)

    threshold = np.percentile(normal_errors, 95)
    ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5,
               label=f"Threshold (95th pct = {threshold:.4f})")

    ax.set_xlabel("MSE Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution — Normal vs Anomaly")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("error_distribution.png")


# -----------------------------
# 4. DETECTOR COMPARISON BAR CHART (FIX #6 — NEW)
# Shows how many anomalies each detector flagged vs true count.
# -----------------------------
def plot_detector_comparison(df):
    detectors = {
        "True Labels"     : "label",
        "Rule-Based"      : "rule_anomaly",
        "Isolation Forest": "if_anomaly",
        "LSTM"            : "lstm_anomaly",
        "Final Ensemble"  : "final_anomaly",
    }

    available = {name: col for name, col in detectors.items() if col in df.columns}
    if len(available) < 2:
        print("  [SKIP] Not enough detector columns found.")
        return

    counts = {name: df[col].sum() for name, col in available.items()}

    colours = ["black"] + ["steelblue"] * (len(counts) - 2) + ["green"]
    colours = colours[:len(counts)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=colours, alpha=0.8, edgecolor="white")

    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Rows Flagged as Anomaly")
    ax.set_title("Anomaly Count per Detector")
    ax.grid(axis="y", alpha=0.3)

    save_fig("detector_comparison.png")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()

    # Load best available file
    try:
        df = pd.read_csv(get_data_path("adsb_final.csv"))
        print("Loaded: adsb_final.csv")
    except FileNotFoundError:
        df = pd.read_csv(get_data_path("adsb_lstm.csv"))
        print("Loaded: adsb_lstm.csv (run detect.py first for full ensemble plots)")

    print("\nGenerating plots...")

    plot_trajectory(df)
    plot_lstm_error(df)
    plot_error_distribution(df)
    plot_detector_comparison(df)

    print("\nAll plots saved to data/plots/")