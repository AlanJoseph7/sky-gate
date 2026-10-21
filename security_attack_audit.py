#!/usr/bin/env python
"""
security_attack_audit.py — SkyGate Automated Security & Attack Resilience Audit

Executes targeted penetration probes against SkyGate attack surfaces:
  1. SQL Injection (SQLi) Probes
  2. Authentication Bypass & JWT Tampering (Algorithm 'none', forged signatures)
  3. XSS & Payload Sanitization (Script tags, HTML event handlers in telemetry)
  4. Path Traversal & SSRF Probes on Flight Details
  5. Coordinate & Type Fuzzing (Extreme / Invalid coordinates)
  6. Rate Limiting & Brute Force Defense
  7. HTTP Security Headers Verification (Clickjacking, MIME sniffing, XSS filter)
"""

import sys
import time
import requests
from jose import jwt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_URL = "http://localhost:5000"

total_probes = 0
resilient_count = 0
vulnerable_count = 0

def probe_result(name: str, secure: bool, reason: str = ""):
    global total_probes, resilient_count, vulnerable_count
    total_probes += 1
    if secure:
        resilient_count += 1
        print(f"  [DEFENDED] {name}" + (f" -> {reason}" if reason else ""))
    else:
        vulnerable_count += 1
        print(f"  [VULNERABLE] {name}" + (f" -> {reason}" if reason else ""))

def run_attack_audit():
    print("\n" + "="*70)
    print("   🛡️  SKYGATE COMPREHENSIVE CYBERSECURITY PENETRATION AUDIT")
    print("="*70 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SQL INJECTION (SQLi) ATTACK PROBES
    # ─────────────────────────────────────────────────────────────────────────
    print("[1] Testing SQL Injection (SQLi) Resistance")
    sqli_payloads = [
        "admin' OR '1'='1",
        "' OR 1=1 --",
        "admin' UNION SELECT id, username, 'hacked' FROM users--",
        "'; DROP TABLE users; --",
    ]
    for payload in sqli_payloads:
        r = requests.post(
            f"{API_URL}/api/auth/login",
            data={"username": payload, "password": "password123"},
            timeout=5
        )
        # Must be rejected with 401 or 422, never 200 or 500 (server crash)
        probe_result(
            f"SQLi attempt in login username: '{payload[:25]}...'",
            r.status_code == 401,
            f"Blocked with HTTP {r.status_code}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. JWT TAMPERING & ALGORITHM CONFUSION PROBES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] Testing JWT Tampering & Algorithm Confusion Resistance")
    # A. Algorithm 'none' attack (manually base64url encoded)
    import base64, json
    h = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    p = base64.urlsafe_b64encode(json.dumps({"sub": "admin"}).encode()).decode().rstrip("=")
    forged_none_token = f"{h}.{p}."
    r_none = requests.get(
        f"{API_URL}/api/stats",
        headers={"Authorization": f"Bearer {forged_none_token}"},
        timeout=5
    )
    probe_result("JWT 'none' algorithm bypass attack", r_none.status_code == 401, f"Rejected with HTTP {r_none.status_code}")

    # B. Forged signature with wrong secret key
    forged_secret_token = jwt.encode({"sub": "admin"}, "wrong_secret_key_12345", algorithm="HS256")
    r_forged = requests.get(
        f"{API_URL}/api/stats",
        headers={"Authorization": f"Bearer {forged_secret_token}"},
        timeout=5
    )
    probe_result("Forged HMAC signature attack", r_forged.status_code == 401, f"Rejected with HTTP {r_forged.status_code}")

    # C. Missing 'sub' claim in JWT
    import os
    from datetime import datetime, timezone, timedelta
    test_key = os.getenv("SKYGATE_SECRET_KEY", "test_audit_secret_key_minimum_32_characters_long")
    missing_sub_token = jwt.encode(
        {"exp": datetime.now(timezone.utc) + timedelta(minutes=10)},
        test_key,
        algorithm="HS256"
    )
    r_no_sub = requests.get(
        f"{API_URL}/api/stats",
        headers={"Authorization": f"Bearer {missing_sub_token}"},
        timeout=5
    )
    probe_result("Token missing identity subject ('sub') claim", r_no_sub.status_code == 401, f"Rejected with HTTP {r_no_sub.status_code}")

    # Get valid token for subsequent authenticated probes
    r_login = requests.post(f"{API_URL}/api/auth/login", data={"username": "tester", "password": "password123"}, timeout=5)
    valid_token = r_login.json().get("access_token")
    auth_headers = {"Authorization": f"Bearer {valid_token}"}

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PATH TRAVERSAL & SSRF PROBES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] Testing Path Traversal & SSRF Defense")
    traversal_probes = [
        "../../etc/passwd",
        "..%2f..%2fadmin",
        "http://attacker.com",
        "<script>",
        "DEAD99;--",
    ]
    for probe in traversal_probes:
        r_trav = requests.get(f"{API_URL}/api/flight-detail/{probe}", headers=auth_headers, timeout=5)
        # Must reject invalid ICAO with 400 Bad Request or 404, never 500 or 200 SSRF
        probe_result(
            f"ICAO24 path traversal / SSRF attempt: '{probe}'",
            r_trav.status_code in [400, 404, 422],
            f"Blocked with HTTP {r_trav.status_code}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. INPUT SANITIZATION & MALICIOUS TELEMETRY PROBES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] Testing Input Sanitization & Malicious Telemetry Resistance")
    xss_callsigns = [
        "<script>alert(1)</script>",
        "');alert(1);//",
        "<img src=x onerror=alert(1)>",
    ]
    for callsign in xss_callsigns:
        r_xss = requests.post(
            f"{API_URL}/api/simulate-attack",
            json={"attack_type": "teleport", "callsign": callsign, "icao24": "DEAD99"},
            headers=auth_headers,
            timeout=5
        )
        # Should either reject with 422 (Pydantic regex validation)
        probe_result(
            f"XSS payload injection in callsign: '{callsign[:20]}'",
            r_xss.status_code in [400, 422],
            f"Blocked by schema regex with HTTP {r_xss.status_code}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. COORDINATE & RANGE FUZZING PROBES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5] Testing Coordinate & Range Boundary Fuzzing")
    fuzz_coords = [
        (99999.0, 73.8),     # Impossible latitude
        (-120.0, 73.8),     # Below -90 lat
        (18.5, 999999.0),   # Impossible longitude
        (18.5, -999999.0),  # Below -180 lon
        ("NaN", 73.8),      # Non-numeric string
    ]
    for lat, lon in fuzz_coords:
        r_fuzz = requests.get(f"{API_URL}/api/locate", params={"lat": lat, "lon": lon}, headers=auth_headers, timeout=5)
        probe_result(
            f"Out-of-bounds / invalid coordinate probe: lat={lat}, lon={lon}",
            r_fuzz.status_code == 422,
            f"Blocked by Query range validator with HTTP {r_fuzz.status_code}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. HTTP SECURITY HEADERS VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6] Testing HTTP Security Headers & Browser Hardening")
    r_head = requests.get(f"{API_URL}/api/health", timeout=5)
    headers = r_head.headers

    probe_result("X-Content-Type-Options: nosniff present", headers.get("X-Content-Type-Options") == "nosniff")
    probe_result("X-Frame-Options: DENY present (Clickjacking protection)", headers.get("X-Frame-Options") == "DENY")
    probe_result("X-XSS-Protection: 1; mode=block present", headers.get("X-XSS-Protection") == "1; mode=block")
    probe_result("Referrer-Policy: strict-origin-when-cross-origin present", headers.get("Referrer-Policy") == "strict-origin-when-cross-origin")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. RATE LIMITING & BRUTE FORCE RESISTANCE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7] Testing Rate Limiting & Brute Force Defense")
    triggered_limit = False
    for i in range(15):
        r_rl = requests.post(
            f"{API_URL}/api/auth/login",
            data={"username": f"brute_force_{i}", "password": "wrong_password"},
            timeout=5
        )
        if r_rl.status_code == 429:
            triggered_limit = True
            break
        time.sleep(0.05)

    probe_result("Rate Limiting on login triggers HTTP 429 Too Many Requests", triggered_limit, "Throttled brute force attempt")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"   AUDIT SUMMARY: {resilient_count}/{total_probes} ATTACK PROBES SUCCESSFULLY DEFENDED")
    print(f"   VULNERABILITIES FOUND: {vulnerable_count}")
    print("="*70 + "\n")
    return vulnerable_count == 0

if __name__ == "__main__":
    success = run_attack_audit()
    sys.exit(0 if success else 1)
