import bcrypt
import os
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt

# -----------------------------
# Configuration
# -----------------------------
# JWT secret MUST be set via environment variable in production.
# A random fallback is generated for local development only.
_DEFAULT_SECRET = secrets.token_hex(32)
SECRET_KEY = os.environ.get("SKYGATE_SECRET_KEY", _DEFAULT_SECRET)
if "SKYGATE_SECRET_KEY" not in os.environ:
    import warnings
    warnings.warn(
        "SKYGATE_SECRET_KEY not set — using a random key. "
        "JWTs will be invalidated on server restart. "
        "Set SKYGATE_SECRET_KEY in your environment for production.",
        stacklevel=2,
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440 # 24 hours

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
