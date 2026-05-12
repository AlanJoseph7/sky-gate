import bcrypt
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from config import cfg

log = logging.getLogger(__name__)

# -----------------------------
# Configuration  (delegated to config.py)
# -----------------------------
SECRET_KEY = cfg.JWT_SECRET_KEY
ALGORITHM = cfg.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = cfg.JWT_EXPIRE_MINUTES

# -----------------------------
# Password Hashing
# -----------------------------
def _pre_hash(password: str) -> str:
    """Collapses any length password into a 32-byte SHA-256 hex string."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a password against a hash using direct bcrypt library."""
    try:
        # We pre-hash with SHA-256 so Bcrypt always sees a fixed-length string.
        # This prevents the 72-byte limit crash entirely.
        password_bytes = _pre_hash(plain_password).encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Generates a bcrypt hash for a password (pre-hashed with SHA-256)."""
    # Hash the password first to ensure it's under Bcrypt's 72-byte limit.
    password_bytes = _pre_hash(password).encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

# -----------------------------
# JWT Handling
# -----------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload if payload.get("sub") else None
    except JWTError:
        return None
