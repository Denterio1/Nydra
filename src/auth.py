"""
auth.py — Nydra Authentication & Identity Hub
==============================================
JWT management and password hashing via SecurityVault.
"""

from __future__ import annotations
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

from src.security_vault import get_vault

vault = get_vault()


def _require_secret(name: str) -> str:
    """Load a signing secret from the environment. No predictable fallbacks."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add a long random value to your .env file "
            f"(e.g. python -c \"import secrets; print(secrets.token_urlsafe(48))\")."
        )
    if len(value) < 32:
        raise RuntimeError(f"{name} must be at least 32 characters.")
    return value


SECRET_KEY = _require_secret("SECRET_KEY")
REFRESH_SECRET = _require_secret("REFRESH_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_EXPIRE_DAYS", "7"))


def hash_password(password: str) -> str:
    """Hash a password using SecurityVault (PBKDF2)."""
    return vault.hash(password, context="auth")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its vault-generated hash."""
    if not hashed_password:
        return False
    return vault.verify_hash(plain_password, hashed_password, context="auth")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a secure JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    import secrets
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh", "jti": secrets.token_urlsafe(16)})
    return jwt.encode(to_encode, REFRESH_SECRET, algorithm=ALGORITHM)


def verify_token(token: str, is_refresh: bool = False) -> dict | None:
    """Verify and decode a JWT token."""
    try:
        key = REFRESH_SECRET if is_refresh else SECRET_KEY
        return jwt.decode(token, key, algorithms=[ALGORITHM])
    except JWTError:
        return None


def generate_user_id() -> str:
    """Generate a unique ID for a new user."""
    return str(uuid.uuid4())


def get_user_stats(email: str) -> dict[str, Any]:
    """
    Get usage statistics for a user.
    Note: In a full implementation, this queries the SQL database.
    """
    return {
        "name": email.split("@")[0].title() if "@" in email else "User",
        "plan": "professional",
        "total_sessions": 0,
        "files_analysed": 0,
        "avg_ml_score": 0,
        "last_seen": datetime.now().isoformat()[:19].replace("T", " "),
        "member_since": datetime.now().isoformat()[:10],
    }
