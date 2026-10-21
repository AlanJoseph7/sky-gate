#!/usr/bin/env python
"""Test script to verify all fixed endpoints."""

import sys
import time
import requests
import json

# Configuration
API_URL = "http://localhost:5000"
TEST_USER = "tester"
TEST_PASS = "password123"

def test_endpoints():
    """Test all endpoints."""
    print("[TEST] Starting endpoint verification...")
    print(f"API: {API_URL}\n")
    
    # 0. Health check
    print("[0] Testing GET /api/health (public)...")
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        if r.status_code == 200:
            print(f"    OK: {r.json()}\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False

    # 1. Login
    print("[1] Testing /api/auth/login...")
    try:
        r = requests.post(
            f"{API_URL}/api/auth/login",
            data={"username": TEST_USER, "password": TEST_PASS},
            timeout=5
        )
        if r.status_code == 200:
            token = r.json().get("access_token")
            print(f"    OK: Got bearer token\n")
        else:
            print(f"    FAILED: {r.status_code} - {r.text}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Test /api/airspaces
    print("[2] Testing GET /api/airspaces...")
    try:
        r = requests.get(f"{API_URL}/api/airspaces", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: {len(data.get('regions', []))} regions found")
            print(f"    Sample Regions: {[r['name'] for r in data.get('regions', [])[:5]]}\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False
    
    # 3. Test /api/locate
    print("[3] Testing GET /api/locate?lat=18.5&lon=73.8...")
    try:
        r = requests.get(
            f"{API_URL}/api/locate",
            params={"lat": 18.5, "lon": 73.8},
            headers=headers,
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: in_indian_airspace={data.get('in_indian_airspace')}")
            print(f"    Regions: {[r['name'] for r in data.get('regions', [])]}\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False
    
    # 4. Test PATCH /api/airspace with valid region IDs
    print("[4] Testing PATCH /api/airspace...")
    try:
        r = requests.patch(
            f"{API_URL}/api/airspace",
            json={"active": ["india", "sri_lanka"]},
            headers=headers,
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: Updated active regions to {data.get('active')}\n")
        else:
            print(f"    FAILED: {r.status_code} - {r.text}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False
    
    # 5. Test /api/stats
    print("[5] Testing GET /api/stats...")
    try:
        r = requests.get(f"{API_URL}/api/stats", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: Flights today={data.get('flights_today')}, Anomalies={data.get('anomalies_flagged')}, Accuracy={data.get('accuracy')}%\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False

    # 6. Test /api/anomalies
    print("[6] Testing GET /api/anomalies...")
    try:
        r = requests.get(f"{API_URL}/api/anomalies", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: Returned {len(data)} anomalies\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False

    # 7. Test SIEM Endpoints
    print("[7] Testing GET /api/siem/summary...")
    try:
        r = requests.get(f"{API_URL}/api/siem/summary", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: total_anomalies={data.get('total_anomalies')}, critical={data.get('critical_count')}, top_threat={data.get('top_threat_class')}\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False

    print("[8] Testing GET /api/siem/threat-breakdown...")
    try:
        r = requests.get(f"{API_URL}/api/siem/threat-breakdown", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"    OK: {len(data)} threat taxonomy classes present\n")
        else:
            print(f"    FAILED: {r.status_code}\n")
            return False
    except Exception as e:
        print(f"    ERROR: {e}\n")
        return False
    
    print("[PASS] All endpoints working correctly!")
    return True

if __name__ == "__main__":
    time.sleep(1)  # Give server time to start
    success = test_endpoints()
    sys.exit(0 if success else 1)

