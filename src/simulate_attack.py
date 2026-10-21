"""
simulate_attack.py - SkyGate Live Attack & Anomaly Simulation Tool

Simulates realistic ADS-B spoofing attacks against the SkyGate AI/ML pipeline
and triggers real-time SIEM alerts and automated quarantine.

Supported attack profiles:
  • teleport   - Ghost Track / Position Teleportation (450 km impossible displacement)
  • altitude   - Altimeter Tampering / Vertical Spurious Spike (>80,000 ft/min)
  • speed      - Supersonic Velocity Spike (>1,100 knots)
  • drift      - Gradual Heading & Velocity Drift (Reconstruction Error)

Usage:
  python src/simulate_attack.py
  python src/simulate_attack.py --attack altitude --callsign CLIMB99
  python src/simulate_attack.py --attack speed --callsign MACH2
  python src/simulate_attack.py --attack drift --callsign DRIFT01
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to sys.path
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import requests


def simulate_local(attack_type: str, callsign: str, icao: str):
    """Execute attack simulation directly through local pipeline."""
    import pandas as pd
    from pipeline import run_pipeline

    now = datetime.now(timezone.utc)
    prev = now - timedelta(seconds=10)

    base_flights = [
        {'icao': '800001', 'callsign': 'IGO101', 'latitude': 18.5, 'longitude': 73.8, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 90.0, 'timestamp': prev, 'label': 0},
        {'icao': '800001', 'callsign': 'IGO101', 'latitude': 18.52, 'longitude': 73.82, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 90.0, 'timestamp': now, 'label': 0},
        {'icao': '800002', 'callsign': 'AIC202', 'latitude': 19.1, 'longitude': 74.2, 'altitude': 32000.0, 'velocity': 480.0, 'heading': 110.0, 'timestamp': prev, 'label': 0},
        {'icao': '800002', 'callsign': 'AIC202', 'latitude': 19.12, 'longitude': 74.22, 'altitude': 32000.0, 'velocity': 480.0, 'heading': 110.0, 'timestamp': now, 'label': 0},
        {'icao': '800003', 'callsign': 'AXB303', 'latitude': 17.8, 'longitude': 75.0, 'altitude': 28000.0, 'velocity': 420.0, 'heading': 85.0, 'timestamp': prev, 'label': 0},
        {'icao': '800003', 'callsign': 'AXB303', 'latitude': 17.82, 'longitude': 75.02, 'altitude': 28000.0, 'velocity': 420.0, 'heading': 85.0, 'timestamp': now, 'label': 0},
        {'icao': '800004', 'callsign': 'SEJ404', 'latitude': 20.0, 'longitude': 76.5, 'altitude': 36000.0, 'velocity': 460.0, 'heading': 95.0, 'timestamp': prev, 'label': 0},
        {'icao': '800004', 'callsign': 'SEJ404', 'latitude': 20.02, 'longitude': 76.52, 'altitude': 36000.0, 'velocity': 460.0, 'heading': 95.0, 'timestamp': now, 'label': 0},
    ]

    if attack_type == "altitude":
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 33000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 19.52, 'longitude': 78.02, 'altitude': 54000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': now, 'label': 0}
    elif attack_type == "speed":
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 19.65, 'longitude': 78.15, 'altitude': 35000.0, 'velocity': 1150.0, 'heading': 45.0, 'timestamp': now, 'label': 0}
    elif attack_type == "drift":
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 19.58, 'longitude': 78.08, 'altitude': 35000.0, 'velocity': 680.0, 'heading': 175.0, 'timestamp': now, 'label': 0}
    else:  # teleport
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 23.5, 'longitude': 82.0, 'altitude': 48000.0, 'velocity': 980.0, 'heading': 45.0, 'timestamp': now, 'label': 0}

    df = pd.DataFrame(base_flights + [p1, p2])
    anomalies = run_pipeline(df)
    sim_anomalies = [a for a in anomalies if a.get("icao24") == icao] or anomalies
    return sim_anomalies


def print_attack_report(results: list, attack_type: str, via_server: bool = False):
    """Format and print an aesthetic security alert report."""
    print("\n" + "="*70)
    print("   SKYGATE AVIATION CYBERSECURITY - ATTACK INJECTION VERDICT")
    print("="*70)
    print(f"  Mode           : {'Live Server API + Web Dashboard' if via_server else 'Direct Pipeline Local'}")
    print(f"  Attack Profile : {attack_type.upper()}")
    print(f"  Detections     : {len(results)} target(s) flagged")
    print("-" * 70)

    for i, a in enumerate(results, 1):
        quarantined = a.get("quarantine", False)
        status_tag = "[QUARANTINED]" if quarantined else "[FLAGGED]"
        print(f"\n  [{i}] {status_tag} {a.get('callsign', 'UNKNOWN')} ({a.get('icao24', 'N/A')})")
        print(f"      Threat Classification : {a.get('threat_class', 'Unknown')}")
        print(f"      Primary Anomaly Type  : {a.get('anomaly', 'N/A')}")
        print(f"      Detection Layer       : {a.get('layer', 'Ensemble')}")
        print(f"      Ensemble Risk Score   : {a.get('score', 0.0):.3f} / 1.000")
        print(f"      Severity Level        : {a.get('severity', 'info').upper()}")
        print(f"      MLAT Triangulation    : {'PASS (Verified)' if a.get('mlat_verified') else 'FAIL (Unverified Position)'}")
        print(f"      PKI Digital Signature : {'VALID' if a.get('valid_signature') else 'INVALID / SPOOFED'}")
        print(f"      Modelled RSSI Signal  : {a.get('rssi_dbm', 'N/A')} dBm")
        if quarantined:
            print("      Action Taken          : >> ISOLATED FROM MAIN ATC AIR TRAFFIC RADAR <<")

    print("\n" + "="*70)
    if via_server:
        print("  [SUCCESS] Alert broadcasted to connected web clients via SSE stream.")
        print("  View live in dashboard at:")
        print("    -> http://localhost:5000/siem.html     (SIEM Incident Monitor)")
        print("    -> http://localhost:5000/anomalies.html (Anomalies Telemetry)")
        print("    -> http://localhost:5000/monitor.html   (Airspace Radar)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SkyGate Attack & Anomaly Simulator")
    parser.add_argument("--attack", choices=["teleport", "altitude", "speed", "drift"], default="teleport",
                        help="Type of ADS-B attack profile to simulate (default: teleport)")
    parser.add_argument("--callsign", default="GHOST99", help="Callsign for spoofed aircraft")
    parser.add_argument("--icao", default="DEAD99", help="ICAO 24-bit hex address for spoofed aircraft")
    parser.add_argument("--url", default="http://localhost:5000", help="SkyGate server URL")
    parser.add_argument("--user", default="tester", help="Username for authentication")
    parser.add_argument("--password", default="password123", help="Password for authentication")
    parser.add_argument("--local", action="store_true", help="Run locally without querying HTTP server")

    args = parser.parse_args()

    # Try hitting the live server first unless --local is passed
    if not args.local:
        try:
            # 1. Login
            auth_res = requests.post(
                f"{args.url}/api/auth/login",
                data={"username": args.user, "password": args.password},
                timeout=10
            )
            if auth_res.status_code == 200:
                token = auth_res.json().get("access_token")
                headers = {"Authorization": f"Bearer {token}"}

                # 2. Trigger simulate-attack endpoint
                sim_res = requests.post(
                    f"{args.url}/api/simulate-attack",
                    json={
                        "attack_type": args.attack,
                        "callsign": args.callsign,
                        "icao24": args.icao,
                    },
                    headers=headers,
                    timeout=30
                )
                if sim_res.status_code == 200:
                    data = sim_res.json()
                    results = data.get("results", [])
                    print_attack_report(results, args.attack, via_server=True)
                    return
                else:
                    print(f"  [Server Warning] API returned HTTP {sim_res.status_code}: {sim_res.text}")
            else:
                print(f"  [Auth Warning] Login returned HTTP {auth_res.status_code}: {auth_res.text}")
        except Exception as e:
            print(f"  [Server Info] Could not reach server API ({e}) - using local pipeline.")

    print("  [Simulator] Running simulation through local detection pipeline...")
    results = simulate_local(args.attack, args.callsign.upper(), args.icao.upper())
    print_attack_report(results, args.attack, via_server=False)


if __name__ == "__main__":
    main()
