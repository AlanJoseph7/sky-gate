import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from datetime import datetime

from utils import get_data_path, ensure_data_dirs

# -----------------------------
# OPENSKY CREDENTIALS
# -----------------------------
OPENSKY_USERNAME = "alanjosephkurian"
OPENSKY_PASSWORD = "Appa@2525"        

# -----------------------------
# INDIAN AIRSPACE BOUNDING BOX
# -----------------------------
INDIA_BOUNDS = {
    "lamin":  6.0,    # southernmost point (Kanyakumari)
    "lomin": 68.0,    # westernmost point  (Gujarat)
    "lamax": 35.0,    # northernmost point (Kashmir)
    "lomax": 97.0,    # easternmost point  (Arunachal Pradesh)
}

# -----------------------------
# ADSB.LOL API CONFIG
# -----------------------------
ADSBLOL_API_BASE = os.environ.get("ADSBLOL_API_BASE", "https://api.adsb.lol")
ADSBLOL_API_PATH = os.environ.get("ADSBLOL_API_PATH", "/v2/point/20.5/82.5/250")
ADSBLOL_API_KEY  = os.environ.get("ADSBLOL_API_KEY", "")

# -----------------------------
# OPENSKY STATE VECTOR COLUMNS
# -----------------------------
OPENSKY_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source"
]

# -----------------------------
# FETCH LIVE DATA
# -----------------------------
def fetch_live_data(*args, **kwargs):
    """
    Fetches current ADS-B state vectors from adsb.lol for Indian airspace.

    Returns a cleaned DataFrame matching the pipeline schema,
    or None if the fetch fails.

    This adapter normalizes the adsb.lol response into the schema used
    by the rest of the SkyGate pipeline.

    Also auto-saves each successful fetch to data/live_log.csv
    for use in incremental model retraining.
    """
    if ADSBLOL_API_PATH.startswith("http"):
        url = ADSBLOL_API_PATH
    else:
        url = f"{ADSBLOL_API_BASE.rstrip('/')}/{ADSBLOL_API_PATH.lstrip('/')}"

    headers = {"Accept": "application/json"}
    if ADSBLOL_API_KEY:
        headers["Authorization"] = f"Bearer {ADSBLOL_API_KEY}"

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            flights = None

            if isinstance(data, dict):
                flights = data.get("ac") or data.get("aircraft") or data.get("states") or data.get("data") or data.get("results")
            elif isinstance(data, list):
                flights = data

            if not flights:
                print("  [adsb.lol] No flights in response.")
                return None

            df = parse_adsb_lol_response(flights)

            if df is None or df.empty:
                print("  [adsb.lol] No valid airborne flights found.")
                return None

            # Filter to Indian airspace only
            df = df[
                (df["latitude"] >= INDIA_BOUNDS["lamin"]) &
                (df["latitude"] <= INDIA_BOUNDS["lamax"]) &
                (df["longitude"] >= INDIA_BOUNDS["lomin"]) &
                (df["longitude"] <= INDIA_BOUNDS["lomax"])
            ].copy()

            if len(df) == 0:
                print("  [adsb.lol] No flights within Indian airspace bounds.")
                return None

            # Filter out ground flights when possible
            if "on_ground" in df.columns:
                df = df[df["on_ground"] == False].copy()

            if len(df) == 0:
                print("  [adsb.lol] No airborne flights found.")
                return None

            result = format_for_pipeline(df, data.get("now") or data.get("time") or data.get("timestamp"))

            if result is not None and len(result) > 0:
                save_live_batch(result)

            return result

        elif response.status_code == 401:
            print("  [adsb.lol] Authentication failed — check ADSBLOL_API_KEY.")
            return None

        elif response.status_code == 429:
            print("  [adsb.lol] Rate limit hit — slow down requests.")
            return None

        else:
            print(f"  [adsb.lol] API error: HTTP {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("  [adsb.lol] Connection error — check your internet.")
        return None

    except requests.exceptions.Timeout:
        print("  [adsb.lol] Request timed out.")
        return None

    except Exception as e:
        print(f"  [adsb.lol] Unexpected error: {e}")
        return None


def parse_adsb_lol_response(states):
    """Normalize adsb.lol response payload into OpenSky-like state vector rows."""
    rows = []

    def pick(src, keys, default=None):
        for key in keys:
            if isinstance(src, dict) and key in src and src[key] is not None:
                return src[key]
        return default

    for item in states:
        if not item:
            continue

        # If each entry is a list rather than a dict, skip it
        if not isinstance(item, dict):
            continue

        icao = pick(item, ["icao", "icao24", "hex", "hexident"])
        if not icao:
            continue

        callsign = pick(item, ["callsign", "flight", "callsign_raw", "call"]) or ""
        latitude = pick(item, ["lat", "latitude", "position_lat", "lat_deg"])
        longitude = pick(item, ["lon", "longitude", "position_lon", "lon_deg"])
        altitude = pick(item, ["baro_altitude", "alt_baro", "geometric_altitude", "alt", "altitude_feet"])
        velocity = pick(item, ["velocity", "speed", "groundspeed", "gs", "spd"])
        heading = pick(item, ["track", "heading", "true_track", "true_heading", "hdg"])
        on_ground = pick(item, ["on_ground", "ground", "onground", "is_grounded"], False)

        rows.append({
            "icao24": str(icao).upper().strip(),
            "callsign": str(callsign).strip().upper(),
            "longitude": longitude,
            "latitude": latitude,
            "baro_altitude": altitude,
            "on_ground": bool(on_ground),
            "velocity": velocity,
            "true_track": heading,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# FORMAT FOR PIPELINE
# -----------------------------
def format_for_pipeline(df, fetch_time=None):
    """
    Converts OpenSky state vectors to the schema expected
    by features.py and the rest of the SkyGate pipeline.
    """
    out = pd.DataFrame()

    # Timestamp (adsb.lol sends milliseconds; handle both sec and ms)
    if fetch_time:
        try:
            # Try as milliseconds first
            if int(fetch_time) > 1e10:
                out["timestamp"] = pd.to_datetime(fetch_time, unit="ms")
            else:
                out["timestamp"] = pd.to_datetime(fetch_time, unit="s")
        except Exception:
            out["timestamp"] = datetime.utcnow()
    else:
        out["timestamp"] = datetime.utcnow()

    # ICAO identifier
    out["icao"] = df["icao24"].str.upper().str.strip()

    # Callsign (for adsbdb enrichment); OpenSky may omit or pad with spaces
    cs = df["callsign"].fillna("").astype(str).str.strip().str.upper()
    out["callsign"] = cs.replace({"NAN": "", "NONE": ""})

    # Position
    out["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Altitude: metres → feet
    out["altitude"] = pd.to_numeric(
        df["baro_altitude"], errors="coerce"
    ) * 3.28084

    # Velocity: m/s → knots
    out["velocity"] = pd.to_numeric(
        df["velocity"], errors="coerce"
    ) * 1.94384

    # Heading: degrees (already correct)
    out["heading"] = pd.to_numeric(df["true_track"], errors="coerce")

    # Real data has no ground truth labels
    out["label"] = 0

    # Drop rows with missing critical fields
    out = out.dropna(
        subset=["latitude", "longitude", "altitude", "velocity", "heading"]
    )

    # Clip to physical bounds
    out["altitude"] = out["altitude"].clip(100, 50000)
    out["velocity"] = out["velocity"].clip(0, 700)
    out["heading"]  = out["heading"].clip(0, 360)

    print(f"  [adsb.lol] {len(out)} airborne flights over Indian airspace.")

    return out


# -----------------------------
# SAVE LIVE BATCH FOR RETRAINING
# -----------------------------
def save_live_batch(df: pd.DataFrame, path="data/live_log.csv"):
    """
    Appends a live fetch batch to the rolling log file used
    for incremental model retraining.

    Skips anomaly-flagged rows if a 'final_anomaly' column is
    present, so the models don't learn to reconstruct attacks.
    """
    os.makedirs("data", exist_ok=True)

    clean = df.copy()

    # Exclude flagged anomalies from the training log
    if "final_anomaly" in clean.columns:
        before = len(clean)
        clean = clean[clean["final_anomaly"] == 0]
        excluded = before - len(clean)
        if excluded > 0:
            print(f"  [Logger] Excluded {excluded} flagged rows from training log.")

    if len(clean) == 0:
        return

    clean["fetch_time"] = datetime.utcnow().isoformat()

    file_exists = os.path.exists(path)
    clean.to_csv(path, mode="a", header=not file_exists, index=False)
    print(f"  [Logger] Appended {len(clean)} rows to {path}")


# -----------------------------
# RUN SCRIPT — TEST FETCH
# -----------------------------
if __name__ == "__main__":
    ensure_data_dirs()

    print("Testing adsb.lol connection...")
    df = fetch_live_data()

    if df is not None:
        print("\nSample data:")
        print(df[["icao", "latitude", "longitude",
                   "altitude", "velocity", "heading"]].head(10).to_string())
        df.to_csv(get_data_path("adsb_live_sample.csv"), index=False)
        print(f"\nSaved to data/adsb_live_sample.csv")
    else:
        print("Fetch failed — check ADSBLOL_API_KEY and connection.")