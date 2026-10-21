"""
schema.py — SkyGate Unified Event Schema

Every event that flows through the SIEM is normalised into a SkyGateEvent.
This is the single source of truth for field names across all detection layers.

Inspired by Elastic ECS (Elastic Common Schema) — the industry standard for
log normalisation used by Elastic SIEM, Wazuh, and Chronicle.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Allowed values for key fields (like an enum — prevents typos in severity)
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_LEVELS  = ("critical", "high", "medium", "low", "info")
EVENT_CATEGORIES = ("detection", "auth", "system", "correlation")
SOURCES          = (
    "rule_engine", "isolation_forest", "autoencoder",
    "ensemble", "auth", "server"
)


# ─────────────────────────────────────────────────────────────────────────────
# The Unified Event Schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkyGateEvent:
    """
    A normalised security event in the SkyGate SIEM.

    Every anomaly, login, or system log is converted to this structure
    before being stored, correlated, or displayed in the dashboard.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    # Every event gets a unique ID + timestamp. UUID4 is random — no collision.
    # Timestamp is always UTC so events from different timezones compare correctly.
    event_id:        str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Classification ────────────────────────────────────────────────────────
    # category: what TYPE of event is this? (detection, auth log, system message)
    # event_type: specific action (e.g. 'anomaly_detected', 'user_login')
    # severity: how serious? maps to the SEVERITY_LEVELS tuple above
    # source: which detector/system produced this?
    category:        str = "detection"
    event_type:      str = "anomaly_detected"
    severity:        str = "info"
    source:          str = "ensemble"

    # ── Subject (the "who" / "what") ─────────────────────────────────────────
    # For detection events: the aircraft
    # For auth events: icao24 is None, we use a separate user field
    icao24:          Optional[str]   = field(default=None)
    callsign:        Optional[str]   = field(default=None)
    latitude:        Optional[float] = field(default=None)
    longitude:       Optional[float] = field(default=None)
    altitude:        Optional[float] = field(default=None)
    velocity:        Optional[float] = field(default=None)
    heading:         Optional[float] = field(default=None)

    # ── Detection payload ─────────────────────────────────────────────────────
    # Which anomaly type was detected? Which layer found it? How confident?
    # rule_hit / if_hit / ae_hit = booleans for each detector that flagged this
    anomaly_type:    Optional[str]   = field(default=None)
    detection_layer: Optional[str]   = field(default=None)
    score:           float           = 0.0
    rule_hit:        bool            = False
    if_hit:          bool            = False
    ae_hit:          bool            = False

    # ── Threat Intelligence ───────────────────────────────────────────────────
    # MITRE ATT&CK fields: tactic (the goal) and technique (the method)
    # Example: tactic="Deception", technique="T0856"
    # ioc_match: True if this ICAO24 is on a known-bad watchlist
    mitre_tactic:    Optional[str]   = field(default=None)
    mitre_technique: Optional[str]   = field(default=None)
    ioc_match:       bool            = False

    # ── Correlation linkage ───────────────────────────────────────────────────
    # If the correlation engine groups this event with others, it sets
    # correlated=True and stamps the group's correlation_id
    # alert_id links this event to the alert it triggered (if any)
    correlated:      bool            = False
    correlation_id:  Optional[str]   = field(default=None)
    alert_id:        Optional[str]   = field(default=None)

    # ── Auth fields (used only for category='auth' events) ───────────────────
    # Keeping auth fields on the same schema means one query can search
    # both flight anomalies and user logins — powerful for forensics
    user:            Optional[str]   = field(default=None)
    action:          Optional[str]   = field(default=None)  # 'login'|'logout'|'failed_login'
    ip_address:      Optional[str]   = field(default=None)

    # ── Raw data ──────────────────────────────────────────────────────────────
    # Always store the original data. You may need it for forensics later.
    # In Splunk, this is called the "raw event". In ECS, it's stored in `event.original`
    raw:             Optional[dict]  = field(default=None)

    # ── Convenience method ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Convert to dict for JSON responses and DB storage."""
        return {
            "event_id":        self.event_id,
            "timestamp":       self.timestamp,
            "category":        self.category,
            "event_type":      self.event_type,
            "severity":        self.severity,
            "source":          self.source,
            "icao24":          self.icao24,
            "callsign":        self.callsign,
            "latitude":        self.latitude,
            "longitude":       self.longitude,
            "altitude":        self.altitude,
            "velocity":        self.velocity,
            "heading":         self.heading,
            "anomaly_type":    self.anomaly_type,
            "detection_layer": self.detection_layer,
            "score":           self.score,
            "rule_hit":        self.rule_hit,
            "if_hit":          self.if_hit,
            "ae_hit":          self.ae_hit,
            "mitre_tactic":    self.mitre_tactic,
            "mitre_technique": self.mitre_technique,
            "ioc_match":       self.ioc_match,
            "correlated":      self.correlated,
            "correlation_id":  self.correlation_id,
            "alert_id":        self.alert_id,
            "user":            self.user,
            "action":          self.action,
            "ip_address":      self.ip_address,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK Mapping for ADS-B Anomaly Types
#
# MITRE ATT&CK is an industry framework cataloguing adversary tactics
# and techniques. Tagging events with MITRE IDs is standard practice in
# professional SIEMs — it tells analysts exactly what the attacker is doing
# and links to documentation on how to respond.
#
# Note: MITRE ATT&CK ICS (Industrial Control Systems) is the closest
# framework to ADS-B attacks. T08xx techniques are from MITRE ATT&CK ICS.
# ─────────────────────────────────────────────────────────────────────────────

MITRE_MAP = {
    "Position Teleport":  {"tactic": "Deception",              "technique": "T0856"},
    "Altitude Spike":     {"tactic": "Impair Process Control",  "technique": "T0855"},
    "Impossible Speed":   {"tactic": "Deception",              "technique": "T0856"},
    "Heading Violation":  {"tactic": "Defence Evasion",         "technique": "T0849"},
    "Statistical Outlier":{"tactic": "Defence Evasion",         "technique": "T0849"},
    "Recon Error":        {"tactic": "Evasion",                "technique": "T0836"},
    "Rule Trigger":       {"tactic": "Deception",              "technique": "T0856"},
    "Unstable Signal":    {"tactic": "Impact",                 "technique": "T0829"},
}

# ─────────────────────────────────────────────────────────────────────────────
# IOC (Indicator of Compromise) Watchlist
#
# In a real SIEM, this would be pulled from a threat intelligence feed
# (e.g. STIX/TAXII, VirusTotal, Mandiant). For SkyGate, we seed it with
# a small static list. You can add ICAO24 codes here if you observe suspicious
# aircraft repeatedly.
# ─────────────────────────────────────────────────────────────────────────────

IOC_WATCHLIST: set[str] = {
    # Add known suspicious ICAO24 transponder IDs here (lowercase hex)
    # Example: "4b1800",
}


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation Function
#
# This is the core of the normalisation layer. It takes the raw output dict
# from pipeline.py and maps it to a SkyGateEvent.
#
# This is equivalent to a Logstash filter in the ELK stack, or a
# "parsing rule" in Splunk. Every SIEM has one of these.
# ─────────────────────────────────────────────────────────────────────────────

def normalise_pipeline_output(anomaly: dict) -> SkyGateEvent:
    """
    Convert a raw pipeline anomaly dict (from pipeline.py) into a
    normalised SkyGateEvent.

    Parameters
    ----------
    anomaly : dict
        One entry from the list returned by run_pipeline().
        Expected keys: icao24, callsign, anomaly, layer, score,
                       severity, timestamp, latitude, longitude,
                       altitude, velocity, heading,
                       rule_anomaly, if_anomaly, lstm_anomaly

    Returns
    -------
    SkyGateEvent
    """
    anomaly_type = anomaly.get("anomaly") or anomaly.get("type") or "Unknown"
    icao24_raw   = (anomaly.get("icao24") or "").lower().strip()

    # Look up MITRE tags for this anomaly type
    mitre = MITRE_MAP.get(anomaly_type, {})

    # Map pipeline severity strings to our SIEM severity levels
    # (pipeline uses 'critical'/'warn'/'info'; we want 'high' as second tier)
    sev_map = {"critical": "critical", "warn": "high", "info": "medium"}
    severity = sev_map.get(anomaly.get("severity", "info"), "medium")

    return SkyGateEvent(
        # Classification
        category        = "detection",
        event_type      = "anomaly_detected",
        severity        = severity,
        source          = _map_layer_to_source(anomaly.get("layer", "")),

        # Subject
        icao24          = icao24_raw or None,
        callsign        = anomaly.get("callsign") or None,
        latitude        = anomaly.get("latitude"),
        longitude       = anomaly.get("longitude"),
        altitude        = anomaly.get("altitude"),
        velocity        = anomaly.get("velocity"),
        heading         = anomaly.get("heading"),

        # Detection
        anomaly_type    = anomaly_type,
        detection_layer = anomaly.get("layer"),
        score           = float(anomaly.get("score", 0.0)),
        rule_hit        = bool(anomaly.get("rule_anomaly", 0)),
        if_hit          = bool(anomaly.get("if_anomaly", 0)),
        ae_hit          = bool(anomaly.get("lstm_anomaly", 0)),

        # Threat intelligence
        mitre_tactic    = mitre.get("tactic"),
        mitre_technique = mitre.get("technique"),
        ioc_match       = icao24_raw in IOC_WATCHLIST,

        # Raw payload
        raw             = anomaly,
    )


def normalise_auth_event(
    action: str,
    username: str,
    success: bool,
    ip_address: str = "unknown"
) -> SkyGateEvent:
    """
    Normalise an authentication event (login/signup/failed login) into a
    SkyGateEvent so it appears in the unified SIEM event stream alongside
    flight anomalies.

    This is important: real attackers often compromise user accounts before
    or after a physical attack. Correlating auth events with flight anomalies
    reveals attack chains that neither log alone would expose.
    """
    if action == "login" and success:
        severity   = "info"
        event_type = "user_login"
    elif action == "login" and not success:
        severity   = "high"
        event_type = "failed_login"
    elif action == "signup":
        severity   = "info"
        event_type = "user_signup"
    else:
        severity   = "info"
        event_type = action

    return SkyGateEvent(
        category   = "auth",
        event_type = event_type,
        severity   = severity,
        source     = "auth",
        user       = username,
        action     = action,
        ip_address = ip_address,
    )


def normalise_system_event(level: str, msg: str, val: str = "") -> SkyGateEvent:
    """
    Normalise a server/system log entry into a SkyGateEvent.

    Mapping server logs into the SIEM means analysts can see system health
    events on the same timeline as security events — essential for
    distinguishing genuine attacks from system errors.
    """
    sev_map = {"info": "info", "ok": "info", "warn": "medium",
               "alert": "high", "error": "high"}
    return SkyGateEvent(
        category   = "system",
        event_type = "system_log",
        severity   = sev_map.get(level.lower(), "info"),
        source     = "server",
        raw        = {"level": level, "msg": msg, "val": val},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_layer_to_source(layer: str) -> str:
    """Map pipeline layer display name to SIEM source identifier."""
    mapping = {
        "Rule-Based":  "rule_engine",
        "Iso Forest":  "isolation_forest",
        "Autoencoder": "autoencoder",
    }
    return mapping.get(layer, "ensemble")
