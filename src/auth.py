import bcrypt
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt

# -----------------------------
# Configuration
# -----------------------------
SECRET_KEY = os.getenv("SKYGATE_SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload if payload.get("sub") else None
    except JWTError:
        return None
