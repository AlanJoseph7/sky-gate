"""
flight_lookup.py — SkyGate Flight Enrichment

Looks up airline name, flight number, and origin/destination airport
for a given ICAO callsign using the free adsbdb.com API.

Results are cached in memory for 1 hour so repeated lookups for the
same flight within a monitoring session cost zero network calls.

API used: https://api.adsbdb.com/v0/callsign/{callsign}
  - No API key required
  - Rate limit: generous for our 30-second polling interval
  - Returns: airline, origin airport, destination airport
"""

import time
import requests

# ── Config ────────────────────────────────────────────────────────────────────
ADSBDB_URL  = "https://api.adsbdb.com/v0/callsign/{callsign}"
CACHE_TTL   = 3600      # seconds — 1 hour per entry
REQUEST_TIMEOUT = 4     # seconds — don't block the worker thread


# ── In-memory cache: callsign (str) → enriched dict ──────────────────────────
_cache: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def lookup_flight(callsign: str) -> dict:
    """
    Return enriched flight info for a callsign.

    Always returns a dict — empty strings for unknown fields so callers
    never need to guard against missing keys.

    Keys returned:
        airline          str   "Air India"
        flight_number    str   "AI302"   (IATA) or "AIC302" (ICAO fallback)
        origin           str   "DEL"
        origin_city      str   "New Delhi"
        destination      str   "BOM"
        destination_city str   "Mumbai"
    """
    if not callsign:
        return _empty()

    cs = callsign.strip().upper()
    if cs in ("", "N/A", "UNKNOWN"):
        return _empty()

    # ── Cache hit ─────────────────────────────────────────────────────────────
    cached = _cache.get(cs)
    if cached and (time.time() - cached.get("_ts", 0)) < CACHE_TTL:
        return {k: v for k, v in cached.items() if not k.startswith("_")}

    # ── Network lookup ────────────────────────────────────────────────────────
    result = _fetch(cs)
    result["_ts"] = time.time()
    _cache[cs]    = result

    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    return clean


def cache_size() -> int:
    return len(_cache)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(callsign: str) -> dict:
    """Hit the adsbdb API. Returns _empty() on any failure."""
    try:
        resp = requests.get(
            ADSBDB_URL.format(callsign=callsign),
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "SkyGate-AnomalyDetector/1.0"}
        )

        if resp.status_code != 200:
            return _empty()

        data  = resp.json()
        route = (data.get("response") or {}).get("flightroute") or {}

        if not route:
            return _empty()

        airline  = route.get("airline") or {}
        origin   = route.get("origin")  or {}
        dest     = route.get("destination") or {}

        # Prefer IATA flight number (e.g. "AI302"), fall back to ICAO callsign
        flight_num = (
            route.get("callsign_iata")
            or route.get("callsign_icao")
            or callsign
        )

        return {
            "airline":          airline.get("name", ""),
            "flight_number":    flight_num,
            "origin":           origin.get("iata_code", ""),
            "origin_city":      origin.get("municipality", "")
                                or origin.get("name", ""),
            "destination":      dest.get("iata_code", ""),
            "destination_city": dest.get("municipality", "")
                                or dest.get("name", ""),
        }

    except requests.exceptions.Timeout:
        print(f"  [Lookup] Timeout for callsign {callsign}")
        return _empty()

    except Exception as e:
        print(f"  [Lookup] Error for {callsign}: {e}")
        return _empty()


def _empty() -> dict:
    return {
        "airline":          "",
        "flight_number":    "",
        "origin":           "",
        "origin_city":      "",
        "destination":      "",
        "destination_city": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_callsigns = ["AIC302", "IGO2153", "SEJ3782", "UAL123", "UNKNOWN"]

    for cs in test_callsigns:
        info = lookup_flight(cs)
        route = (
            f"{info['origin']} → {info['destination']}"
            if info["origin"] else "route unknown"
        )
        airline = info["airline"] or "unknown airline"
        print(f"  {cs:<10}  {airline:<30}  {info['flight_number']:<8}  {route}")