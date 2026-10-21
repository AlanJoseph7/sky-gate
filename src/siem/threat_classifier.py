"""
threat_classifier.py — SkyGate SIEM Threat Enrichment

Enriches each detected anomaly with three verification dimensions and
derives a deterministic threat classification from them.

Verification model
──────────────────
Real ADS-B security systems use three independent signal-quality checks:
  • MLAT (Multilateration) — cross-triangulates position from ≥4 receivers.
    If MLAT disagrees with the broadcast position, the signal is suspect.
  • Digital signature (PKI) — future ADS-B standard (ADS-B v3 / ACAS Xu)
    will mandate a broadcast signature. Currently modelled from score.
  • RSSI (Received Signal Strength Indicator) — a spoofed signal injected
    locally is typically much stronger or weaker than expected for an
    aircraft at the reported altitude/range.

None of these can be computed from a passive ADS-B feed alone. The values
below are *deterministically derived* from the anomaly score and type —
they are an analytic model, not sensor measurements. They are clearly
labelled as such in the API response (field: "verification_source").

Threat taxonomy (from detection layer + verification outcome)
──────────────────────────────────────────────────────────────
Score ≥ 0.75 + unverified MLAT + invalid signature + position anomaly
  → "Malicious Spoofing"   (highest confidence, quarantine)
Score ≥ 0.75 + unverified MLAT + invalid signature + other
  → "SDR Ghost Track"      (injected false target)
Score ≥ 0.45 + unverified MLAT + valid signature
  → "GPS Jamming / Glitch" or "Altimeter Tampering"
Score ≥ 0.08 + all checks pass
  → "Kinematic Violation (Verified)"  (real aircraft, bad manoeuvre)
Score < 0.08
  → "Nominal"
"""


def enrich_and_classify(anomaly_dict: dict) -> dict:
    """
    Deterministically enriches an anomaly dict with MLAT/PKI/RSSI
    verification flags and derives a threat classification.

    All verification values are analytically modelled from the anomaly
    score — not sourced from real sensor hardware.

    Parameters
    ----------
    anomaly_dict : dict
        Output of pipeline._build_anomaly_dict().

    Returns
    -------
    dict — same dict, mutated in-place with extra keys:
        mlat_verified, valid_signature, rssi_dbm,
        threat_class, quarantine, verification_source
    """
    score        = float(anomaly_dict.get("score", 0.0))
    severity     = anomaly_dict.get("severity", "info")
    anomaly_desc = anomaly_dict.get("anomaly", "")
    layer        = anomaly_dict.get("layer", "")

    # ── 1. Deterministic verification flags ──────────────────────────────────
    #
    # MLAT confidence degrades as score increases.
    # Threshold chosen so that score > 0.45 ("warn") reliably fails MLAT,
    # and score > 0.75 ("critical") always fails.
    #
    mlat_verified   = score < 0.45                   # True = position cross-checks OK
    valid_signature = score < 0.75                   # True = broadcast integrity OK

    # RSSI model: legitimate aircraft at cruise → signal strength roughly
    # proportional to range; anomalous signals tend toward extremes.
    # Map score [0, 1] → RSSI [-70, -40] for normal, [-90, -70] for anomalous.
    if score < 0.08:
        # Normal range — moderate, expected signal
        rssi_dbm = round(-70.0 + score * 375.0, 1)   # -70 → ~-40 dBm
    else:
        # Anomalous — weak or noisy (possible SDR ghost / jammer noise floor)
        rssi_dbm = round(-90.0 + (1.0 - score) * 25.0, 1)  # -90 → ~-65 dBm

    anomaly_dict["mlat_verified"]       = mlat_verified
    anomaly_dict["valid_signature"]     = valid_signature
    anomaly_dict["rssi_dbm"]            = rssi_dbm
    anomaly_dict["verification_source"] = "analytic_model"  # ← explicit label

    # ── 2. Threat classification decision tree ───────────────────────────────
    is_position_anomaly = any(
        kw in anomaly_desc
        for kw in ("Teleport", "Impossible Speed", "Position", "Speed")
    )
    is_altitude_anomaly = any(
        kw in anomaly_desc
        for kw in ("Altitude", "Altimeter", "Climb")
    )

    if score < 0.08:
        threat_class = "Nominal"

    elif not mlat_verified and not valid_signature:
        # Both checks failed — highest confidence threat
        if is_position_anomaly:
            threat_class = "Malicious Spoofing"
        else:
            threat_class = "SDR Ghost Track"

    elif not mlat_verified:
        # MLAT failed but signature plausible — signal distortion / jamming
        if is_altitude_anomaly:
            threat_class = "Altimeter Tampering"
        else:
            threat_class = "GPS Jamming / Glitch"

    elif not valid_signature:
        # Signature failed but MLAT passes — replay or replay-variant
        threat_class = "Broadcast Integrity Failure"

    else:
        # All checks pass — real aircraft doing something kinematically odd
        threat_class = "Kinematic Violation (Verified)"

    anomaly_dict["threat_class"] = threat_class

    # ── 3. Quarantine flag ───────────────────────────────────────────────────
    # Only auto-quarantine when critical severity AND MLAT fails.
    # A verified kinematic violation is a compliance issue, not a quarantine.
    is_critical = severity == "critical"
    anomaly_dict["quarantine"] = is_critical and not mlat_verified

    return anomaly_dict
