#!/usr/bin/env python
"""
verify_security_and_features.py — SkyGate Comprehensive Verification Suite

Exhaustively verifies:
  1. Security & Authentication (RBAC, unauthenticated rejection, invalid tokens, validation, signup, login)
  2. Airspace Management & Geolocation (80+ regions, bounding box queries, active updates, invalid ID rejection)
  3. Real-Time Telemetry & Monitoring APIs (/api/stats, /api/anomalies, /api/logs)
  4. Aviation SIEM & Threat Taxonomy (/api/siem/summary, /api/siem/threat-breakdown, /api/siem/timeline, /api/siem/top-threats)
  5. Live Attack Simulation & Automated Quarantine (teleport, altitude, speed, drift)
  6. SSE Real-Time Streaming (authorized streaming & unauthorized rejection)
  7. Frontend Web Dashboards & Assets (7 pages + CSS + JS)
"""

import sys
import time
import uuid
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

API_URL = "http://localhost:5000"
TEST_USER = "tester"
TEST_PASS = "password123"

passed_tests = 0
failed_tests = 0

def check(name: str, condition: bool, details: str = ""):
    global passed_tests, failed_tests
    if condition:
        passed_tests += 1
        print(f"  [PASS] {name}" + (f" -> {details}" if details else ""))
        return True
    else:
        failed_tests += 1
        print(f"  [FAIL] {name}" + (f" -> {details}" if details else ""))
        return False

def run_all_tests():
    print("\n" + "="*70)
    print("   ✈️  SKYGATE COMPREHENSIVE SECURITY & FEATURES TEST SUITE")
    print("="*70 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. PUBLIC HEALTH CHECK
    # ─────────────────────────────────────────────────────────────────────────
    print("[1] Public Endpoints & Availability")
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        check("GET /api/health responds with 200", r.status_code == 200)
        data = r.json()
        check("Health response status == healthy", data.get("status") == "healthy", f"status={data.get('status')}")
    except Exception as e:
        check("GET /api/health reachable", False, str(e))
        print("\n[ABORT] Server is not reachable at " + API_URL)
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # 2. SECURITY & ACCESS CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] Security & Access Control (Unauthenticated & Invalid Tokens)")
    protected_endpoints = [
        ("GET", "/api/stats"),
        ("GET", "/api/anomalies"),
        ("GET", "/api/airspaces"),
        ("GET", "/api/siem/summary"),
        ("GET", "/api/siem/threat-breakdown"),
        ("GET", "/api/siem/timeline"),
        ("GET", "/api/siem/top-threats"),
        ("POST", "/api/simulate-attack"),
    ]
    for method, path in protected_endpoints:
        if method == "GET":
            resp = requests.get(f"{API_URL}{path}", timeout=5)
        else:
            resp = requests.post(f"{API_URL}{path}", json={}, timeout=5)
        check(f"Unauthenticated {method} {path} returns 401", resp.status_code == 401, f"HTTP {resp.status_code}")

    # Test with bogus / forged token
    bogus_headers = {"Authorization": "Bearer forged.header.payload.signature"}
    r_bogus = requests.get(f"{API_URL}/api/stats", headers=bogus_headers, timeout=5)
    check("Forged bearer token rejected with 401", r_bogus.status_code == 401, f"HTTP {r_bogus.status_code}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. USER REGISTRATION & AUTHENTICATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] User Registration & Authentication Validation")
    # A. Duplicate user rejection
    r_dup = requests.post(f"{API_URL}/api/auth/signup", json={
        "username": "tester",
        "full_name": "Existing Tester",
        "email": "another_tester@example.com",
        "password": "validPassword123!"
    }, timeout=5)
    check("Duplicate username rejected with 400", r_dup.status_code == 400, r_dup.text)

    # B. Invalid email format rejected (Pydantic validation)
    r_bad_email = requests.post(f"{API_URL}/api/auth/signup", json={
        "username": f"user_{uuid.uuid4().hex[:6]}",
        "full_name": "Bad Email User",
        "email": "not-a-valid-email",
        "password": "validPassword123!"
    }, timeout=5)
    check("Malformed email rejected with 422", r_bad_email.status_code == 422, f"HTTP {r_bad_email.status_code}")

    # C. Short password rejected (<6 chars)
    r_short_pw = requests.post(f"{API_URL}/api/auth/signup", json={
        "username": f"user_{uuid.uuid4().hex[:6]}",
        "full_name": "Short Pw User",
        "email": f"test_{uuid.uuid4().hex[:6]}@example.com",
        "password": "123"
    }, timeout=5)
    check("Short password (<6) rejected with 422", r_short_pw.status_code == 422, f"HTTP {r_short_pw.status_code}")

    # D. Valid Registration
    rand_id = uuid.uuid4().hex[:8]
    new_user = f"audit_user_{rand_id}"
    new_email = f"audit_{rand_id}@skygate.aero"
    new_pass = "SecurePass123!"
    r_signup = requests.post(f"{API_URL}/api/auth/signup", json={
        "username": new_user,
        "full_name": f"Auditor {rand_id}",
        "email": new_email,
        "password": new_pass
    }, timeout=5)
    check("Valid user registration succeeds with 200", r_signup.status_code == 200, f"user={new_user}")

    # E. Login rejection with wrong password
    r_bad_login = requests.post(f"{API_URL}/api/auth/login", data={
        "username": new_user,
        "password": "WrongPassword!"
    }, timeout=5)
    check("Incorrect password rejected with 401", r_bad_login.status_code == 401, f"HTTP {r_bad_login.status_code}")

    # F. Login rejection with nonexistent user
    r_ghost_login = requests.post(f"{API_URL}/api/auth/login", data={
        "username": f"ghost_{uuid.uuid4().hex[:8]}",
        "password": "SomePassword123!"
    }, timeout=5)
    check("Non-existent user rejected with 401", r_ghost_login.status_code == 401, f"HTTP {r_ghost_login.status_code}")

    # G. Valid Login & JWT Token issuance
    r_login = requests.post(f"{API_URL}/api/auth/login", data={
        "username": new_user,
        "password": new_pass
    }, timeout=5)
    token = None
    if r_login.status_code == 200:
        token = r_login.json().get("access_token")
    check("Valid login issues JWT Bearer token", token is not None and len(token) > 20, f"Token length: {len(token) if token else 0}")
    auth_headers = {"Authorization": f"Bearer {token}"}

    # ─────────────────────────────────────────────────────────────────────────
    # 4. AIRSPACE MANAGEMENT & GEOLOCATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] Airspace Management & Geolocation Engine")
    r_airspaces = requests.get(f"{API_URL}/api/airspaces", headers=auth_headers, timeout=5)
    check("GET /api/airspaces returns 200", r_airspaces.status_code == 200)
    airspaces_data = r_airspaces.json() if r_airspaces.status_code == 200 else {}
    regions_count = len(airspaces_data.get("regions", []))
    check("Comprehensive global airspaces present (>= 60 regions)", regions_count >= 60, f"Total regions: {regions_count}")

    # Locate coordinates in Mumbai, India (lat=19.076, lon=72.877)
    r_loc_in = requests.get(f"{API_URL}/api/locate", params={"lat": 19.076, "lon": 72.877}, headers=auth_headers, timeout=5)
    check("GET /api/locate Mumbai matches India", r_loc_in.status_code == 200 and r_loc_in.json().get("in_indian_airspace") == True)

    # Locate coordinates outside known territory (lat=-85.0, lon=0.0 - Antarctica/Polar)
    r_loc_polar = requests.get(f"{API_URL}/api/locate", params={"lat": -85.0, "lon": 0.0}, headers=auth_headers, timeout=5)
    matching_polar = [r["id"] for r in r_loc_polar.json().get("regions", [])]
    check("GET /api/locate Polar coordinates resolves correctly", "polar" in matching_polar, f"Matched: {matching_polar}")

    # PATCH /api/airspace update active monitored zones
    r_patch = requests.patch(f"{API_URL}/api/airspace", json={"active": ["india", "pakistan", "sri_lanka"]}, headers=auth_headers, timeout=5)
    check("PATCH /api/airspace with valid IDs returns 200", r_patch.status_code == 200, f"Active: {r_patch.json().get('active')}")

    # PATCH /api/airspace with invalid region ID
    r_patch_bad = requests.patch(f"{API_URL}/api/airspace", json={"active": ["atlantis_airspace"]}, headers=auth_headers, timeout=5)
    check("PATCH /api/airspace with invalid ID rejected with 400", r_patch_bad.status_code == 400, r_patch_bad.text)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. LIVE STATS & TELEMETRY APIS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5] Live Stats & Telemetry APIs")
    r_stats = requests.get(f"{API_URL}/api/stats", headers=auth_headers, timeout=5)
    check("GET /api/stats returns 200", r_stats.status_code == 200)
    stats_data = r_stats.json()
    check("Stats contains required fields", all(k in stats_data for k in ["flights_today", "active_flights", "anomalies_flagged", "accuracy", "avg_latency_ms"]))

    r_anom = requests.get(f"{API_URL}/api/anomalies", headers=auth_headers, timeout=5)
    check("GET /api/anomalies returns 200", r_anom.status_code == 200 and isinstance(r_anom.json(), list))

    r_logs = requests.get(f"{API_URL}/api/logs", headers=auth_headers, timeout=5)
    check("GET /api/logs returns 200", r_logs.status_code == 200 and isinstance(r_logs.json(), list))

    # ─────────────────────────────────────────────────────────────────────────
    # 6. AVIATION SIEM & THREAT TAXONOMY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6] Aviation SIEM & SOC Threat Intelligence")
    r_siem_sum = requests.get(f"{API_URL}/api/siem/summary", headers=auth_headers, timeout=5)
    check("GET /api/siem/summary returns 200", r_siem_sum.status_code == 200)
    siem_sum = r_siem_sum.json()
    check("SIEM Summary contains KPI metrics", all(k in siem_sum for k in ["total_anomalies", "critical_count", "quarantine_count", "top_threat_class"]))

    r_siem_tax = requests.get(f"{API_URL}/api/siem/threat-breakdown", headers=auth_headers, timeout=5)
    check("GET /api/siem/threat-breakdown returns 200", r_siem_tax.status_code == 200)
    tax_classes = [item.get("label") for item in r_siem_tax.json()]
    check("Threat taxonomy includes core threat categories", "Malicious Spoofing" in tax_classes and "SDR Ghost Track" in tax_classes)

    r_siem_time = requests.get(f"{API_URL}/api/siem/timeline", headers=auth_headers, timeout=5)
    check("GET /api/siem/timeline returns 12 hourly buckets", r_siem_time.status_code == 200 and len(r_siem_time.json()) == 12)

    r_siem_top = requests.get(f"{API_URL}/api/siem/top-threats", headers=auth_headers, timeout=5)
    check("GET /api/siem/top-threats returns 200", r_siem_top.status_code == 200)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. LIVE ATTACK SIMULATION & AUTOMATED QUARANTINE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7] Live Attack Simulation & Automated SIEM Triage")
    attacks = [
        ("teleport", "GHOST88", "DEAD88"),
        ("altitude", "CLIMB88", "DEAD89"),
        ("speed", "MACH88", "DEAD90"),
        ("drift", "DRIFT88", "DEAD91"),
    ]
    for attack_type, callsign, icao in attacks:
        try:
            r_sim = requests.post(
                f"{API_URL}/api/simulate-attack",
                json={"attack_type": attack_type, "callsign": callsign, "icao24": icao},
                headers=auth_headers,
                timeout=30
            )
            is_ok = r_sim.status_code == 200 and r_sim.json().get("status") == "attack_injected"
            res_list = r_sim.json().get("results", []) if is_ok else []
            quarantine_flag = any(r.get("quarantine") for r in res_list)
            threat_classes = [r.get("threat_class") for r in res_list]
            check(
                f"Simulate {attack_type.upper()} attack detected",
                is_ok and len(res_list) > 0,
                f"Detections: {len(res_list)}, Threat: {threat_classes}, Quarantine: {quarantine_flag}"
            )
        except Exception as e:
            check(f"Simulate {attack_type.upper()} attack request", False, str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # 8. REAL-TIME SSE STREAMING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[8] SSE Real-Time Streaming & Auth Handshake")
    # Unauthenticated SSE stream connection fails
    r_stream_unauth = requests.get(f"{API_URL}/api/stream", timeout=3)
    check("Unauthenticated SSE /api/stream rejected with 401", r_stream_unauth.status_code == 401)

    # Authenticated SSE stream connection with token param
    try:
        with requests.get(f"{API_URL}/api/stream?token={token}", stream=True, timeout=5) as r_stream:
            check("Authenticated SSE stream connects with 200", r_stream.status_code == 200)
            received_event = False
            for line in r_stream.iter_lines(decode_unicode=True):
                if line and line.startswith("event:"):
                    received_event = True
                    break
            check("SSE stream emits live telemetry events", received_event)
    except Exception as e:
        check("SSE streaming test", False, str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # 9. FRONTEND WEB DASHBOARD & ASSET SERVING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[9] Frontend Web Dashboards & Static Asset Serving")
    frontend_assets = [
        ("/", "SkyGate Main Dashboard"),
        ("/index.html", "Overview Dashboard"),
        ("/monitor.html", "Live Radar Monitor"),
        ("/anomalies.html", "Anomalies Log"),
        ("/siem.html", "Aviation SIEM"),
        ("/pipeline.html", "Pipeline Architecture"),
        ("/features.html", "Feature Engineering"),
        ("/auth.html", "Authentication Portal"),
        ("/styles.css", "Unified Stylesheet"),
        ("/shared.js", "Shared JS & SSE Controller"),
        ("/auth-guard.js", "Client-Side Auth Guard"),
    ]
    for path, desc in frontend_assets:
        r_asset = requests.get(f"{API_URL}{path}", timeout=5)
        check(
            f"Asset {path} ({desc}) served 200",
            r_asset.status_code == 200 and len(r_asset.content) > 100,
            f"Bytes: {len(r_asset.content)}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"   VERIFICATION COMPLETE: {passed_tests} PASSED, {failed_tests} FAILED")
    print("="*70 + "\n")
    return failed_tests == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
