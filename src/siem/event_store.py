"""
event_store.py — SkyGate SIEM Database Layer

Defines four SQLAlchemy models (database tables):
  - Event       : every normalised security event (append-only log)
  - Alert       : fired when a correlation rule matches
  - Incident    : analyst-managed investigation grouping alerts
  - IncidentNote: timeline entries (analyst comments) on an incident

We store everything in a separate siem.db SQLite file next to users.db.
Keeping SIEM data separate from auth data is a best practice — different
access requirements, different retention policies.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text, DateTime, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ─────────────────────────────────────────────────────────────────────────────
# Database path — stored next to users.db in the data/ directory
# ─────────────────────────────────────────────────────────────────────────────

# Walk up from this file (src/siem/) → src/ → SkyGate/ → data/
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIEM_DB_PATH = _BASE_DIR / "data" / "siem.db"
DATABASE_URL  = f"sqlite:///{SIEM_DB_PATH}"

# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy setup
# declarative_base() creates a base class all our models will inherit from.
# This is how SQLAlchemy knows which classes to treat as database tables.
# ─────────────────────────────────────────────────────────────────────────────

Base    = declarative_base()
engine  = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: events
#
# This is the core of the SIEM — an append-only log of every normalised event.
# "Append-only" is critical: we NEVER update or delete events. This is
# standard in SOC environments — the log is your evidence chain.
# Deleting or altering events would be a forensics violation.
# ─────────────────────────────────────────────────────────────────────────────

class Event(Base):
    __tablename__ = "events"

    # Primary key — UUID from schema.py
    event_id        = Column(String,  primary_key=True)

    # When did this event happen? Always stored as ISO 8601 UTC string
    timestamp       = Column(String,  nullable=False, index=True)

    # Classification — maps directly to SkyGateEvent fields
    category        = Column(String,  nullable=False, index=True)   # 'detection'|'auth'|'system'
    event_type      = Column(String,  nullable=False)               # 'anomaly_detected'|'user_login'|...
    severity        = Column(String,  nullable=False, index=True)   # 'critical'|'high'|'medium'|'low'|'info'
    source          = Column(String,  nullable=False)               # which detector produced this

    # Subject — the aircraft (or None for auth/system events)
    icao24          = Column(String,  nullable=True,  index=True)
    callsign        = Column(String,  nullable=True)
    latitude        = Column(Float,   nullable=True)
    longitude       = Column(Float,   nullable=True)
    altitude        = Column(Float,   nullable=True)
    velocity        = Column(Float,   nullable=True)
    heading         = Column(Float,   nullable=True)

    # Detection
    anomaly_type    = Column(String,  nullable=True)
    detection_layer = Column(String,  nullable=True)
    score           = Column(Float,   nullable=True, default=0.0)
    rule_hit        = Column(Boolean, nullable=False, default=False)
    if_hit          = Column(Boolean, nullable=False, default=False)
    ae_hit          = Column(Boolean, nullable=False, default=False)

    # Threat intelligence
    mitre_tactic    = Column(String,  nullable=True)
    mitre_technique = Column(String,  nullable=True)
    ioc_match       = Column(Boolean, nullable=False, default=False)

    # Correlation linkage — filled in by the correlation engine
    correlated      = Column(Boolean, nullable=False, default=False, index=True)
    correlation_id  = Column(String,  nullable=True)
    alert_id        = Column(String,  nullable=True)

    # Auth fields — only populated for category='auth' events
    user            = Column(String,  nullable=True)
    action          = Column(String,  nullable=True)
    ip_address      = Column(String,  nullable=True)

    # Raw JSON — stored as a text string, parsed back to dict when read
    # This is the "raw event" — your forensic source of truth
    raw_json        = Column(Text,    nullable=True)

    def to_dict(self) -> dict:
        """Return this DB row as a plain dict (for API responses)."""
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
# TABLE 2: alerts
#
# An alert is created by the correlation engine when a rule fires.
# Think of this as the SIEM's "inbox" — every new alert appears here,
# and analysts work through them (triage, investigate, close or mark false+).
#
# Alert statuses:
#   new           → just created, nobody has looked at it yet
#   acknowledged  → analyst has seen it and claimed it
#   investigating → analyst is actively working it
#   resolved      → confirmed real threat, response completed
#   false_positive → was not a real threat, suppressed
# ─────────────────────────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"

    alert_id    = Column(String,   primary_key=True,
                         default=lambda: str(uuid.uuid4()))

    # Display info
    title       = Column(String,   nullable=False)
    description = Column(Text,     nullable=True)
    severity    = Column(String,   nullable=False, index=True)

    # Lifecycle status — see comment above for allowed values
    status      = Column(String,   nullable=False, default="new", index=True)

    # Which correlation rule fired this alert?
    rule_name   = Column(String,   nullable=False)

    # JSON list of event_ids that triggered this alert
    # e.g. '["uuid1", "uuid2"]'
    event_ids   = Column(Text,     nullable=False, default="[]")

    # Comma-separated ICAO24 codes involved (for quick display)
    icao24s     = Column(String,   nullable=True)

    # MITRE context (copied from the triggering events)
    mitre_tactic    = Column(String, nullable=True)
    mitre_technique = Column(String, nullable=True)

    # Assignment
    assigned_to = Column(String,   nullable=True)

    # Timestamps
    created_at  = Column(DateTime, nullable=False,
                         default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime, nullable=True,
                         onupdate=lambda: datetime.now(timezone.utc))

    # MTTD = Mean Time To Detect (seconds from first event to alert creation)
    # This is a key SOC KPI — how fast did we catch it?
    mttd_seconds = Column(Float,    nullable=True)

    def get_event_ids(self) -> list:
        """Parse the JSON-stored list of event IDs."""
        try:
            return json.loads(self.event_ids or "[]")
        except Exception:
            return []

    def to_dict(self) -> dict:
        return {
            "alert_id":         self.alert_id,
            "title":            self.title,
            "description":      self.description,
            "severity":         self.severity,
            "status":           self.status,
            "rule_name":        self.rule_name,
            "event_ids":        self.get_event_ids(),
            "icao24s":          self.icao24s,
            "mitre_tactic":     self.mitre_tactic,
            "mitre_technique":  self.mitre_technique,
            "assigned_to":      self.assigned_to,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "updated_at":       self.updated_at.isoformat() if self.updated_at else None,
            "mttd_seconds":     self.mttd_seconds,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: incidents
#
# An incident groups one or more alerts into a single investigation.
# Example: 3 alerts about the same aircraft over 10 minutes get merged
# into one incident: "Sustained spoofing — flight VT-ABC".
#
# This is the same concept as a "case" in TheHive or an "incident" in
# PagerDuty and ServiceNow. Analysts track their entire investigation here.
#
# Incident statuses:
#   open        → created, not yet assigned
#   in_progress → assigned and being investigated
#   resolved    → root cause found, threat neutralised
#   closed      → fully documented and signed off
# ─────────────────────────────────────────────────────────────────────────────

class Incident(Base):
    __tablename__ = "incidents"

    incident_id  = Column(String,   primary_key=True,
                          default=lambda: str(uuid.uuid4()))

    title        = Column(String,   nullable=False)
    description  = Column(Text,     nullable=True)
    severity     = Column(String,   nullable=False, index=True)
    status       = Column(String,   nullable=False, default="open", index=True)

    # Who is investigating this?
    assigned_to  = Column(String,   nullable=True)

    # JSON list of alert IDs grouped into this incident
    alert_ids    = Column(Text,     nullable=False, default="[]")

    # MITRE ATT&CK context
    mitre_tactics = Column(String,  nullable=True)  # comma-separated
    
    # Playbook reference (e.g. 'PB-001-SPOOFING')
    # Playbooks are documented response procedures — standard in enterprise SOCs
    playbook_ref  = Column(String,  nullable=True)

    # Timestamps
    created_at   = Column(DateTime, nullable=False,
                          default=lambda: datetime.now(timezone.utc))
    resolved_at  = Column(DateTime, nullable=True)

    # MTTR = Mean Time To Respond (seconds from first alert to incident resolved)
    # This is the other key SOC KPI alongside MTTD
    mttr_seconds = Column(Float,    nullable=True)

    def get_alert_ids(self) -> list:
        try:
            return json.loads(self.alert_ids or "[]")
        except Exception:
            return []

    def to_dict(self) -> dict:
        return {
            "incident_id":   self.incident_id,
            "title":         self.title,
            "description":   self.description,
            "severity":      self.severity,
            "status":        self.status,
            "assigned_to":   self.assigned_to,
            "alert_ids":     self.get_alert_ids(),
            "mitre_tactics": self.mitre_tactics,
            "playbook_ref":  self.playbook_ref,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
            "resolved_at":   self.resolved_at.isoformat() if self.resolved_at else None,
            "mttr_seconds":  self.mttr_seconds,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4: incident_notes
#
# Notes are the analyst's investigation timeline — every action taken,
# every piece of evidence found, every decision made gets logged here.
#
# In a real SOC, this is called the "case timeline" or "war room log".
# It's critical for post-incident reviews and compliance audits.
# Every note is append-only — analysts cannot edit or delete past notes.
# ─────────────────────────────────────────────────────────────────────────────

class IncidentNote(Base):
    __tablename__ = "incident_notes"

    note_id     = Column(String,   primary_key=True,
                         default=lambda: str(uuid.uuid4()))

    # Foreign key — which incident does this note belong to?
    # (We keep it as a plain String rather than a SQLAlchemy ForeignKey
    #  to keep the schema simple — good enough for this scale)
    incident_id = Column(String,   nullable=False, index=True)

    # Who wrote this note?
    analyst     = Column(String,   nullable=False)

    # The note body — free-form text, analyst writes whatever is relevant
    text        = Column(Text,     nullable=False)

    # Optional: attach evidence (e.g. a JSON dump of the raw event)
    evidence    = Column(Text,     nullable=True)

    # When was this note written?
    created_at  = Column(DateTime, nullable=False,
                         default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "note_id":     self.note_id,
            "incident_id": self.incident_id,
            "analyst":     self.analyst,
            "text":        self.text,
            "evidence":    self.evidence,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Database initialisation
#
# Call init_siem_db() once at server startup (we'll wire this into server.py).
# Base.metadata.create_all() reads all the class definitions above and creates
# the corresponding tables in siem.db if they don't already exist.
# If the tables already exist, it does nothing — safe to call every startup.
# ─────────────────────────────────────────────────────────────────────────────

def init_siem_db():
    """Create all SIEM tables in siem.db if they don't exist."""
    SIEM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print(f"  [SIEM] Database ready at {SIEM_DB_PATH}")


def get_siem_db():
    """
    Dependency-injection helper for FastAPI routes.
    Yields a database session and ensures it is closed after the request,
    even if an exception occurs.

    Usage in a FastAPI route:
        @app.get("/api/siem/events")
        def get_events(db: Session = Depends(get_siem_db)):
            ...
    """
    db = Session()
    try:
        yield db
    finally:
        db.close()
