"""Security helpers: password hashing, JWT tokens, simple rate limiting.

Uses passlib/bcrypt when available; falls back to PBKDF2-HMAC-SHA256 via hashlib
so the platform runs without extra installs. Never store plaintext passwords.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from .config import settings

try:
    from passlib.context import CryptContext  # type: ignore
    _pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    _HAS_PASSLIB = True
except Exception:
    _HAS_PASSLIB = False


def hash_password(password: str) -> str:
    if _HAS_PASSLIB:
        return _pwd.hash(password)
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"pbkdf2${salt}${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    if _HAS_PASSLIB:
        try:
            return _pwd.verify(password, hashed)
        except Exception:
            return False
    try:
        algo, salt, hexdk = hashed.split("$", 2)
        if algo != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
        return hmac.compare_digest(dk.hex(), hexdk)
    except Exception:
        return False


def create_access_token(user_id: str, username: str, expires_hours: Optional[int] = None) -> str:
    """Create a signed JWT (HS256). Payload includes sub, name, exp."""
    import json
    import base64
    exp = datetime.now(timezone.utc) + timedelta(hours=expires_hours or settings.JWT_EXPIRE_HOURS)
    payload = {"sub": user_id, "name": username, "exp": int(exp.timestamp())}
    header = {"alg": "HS256", "typ": "JWT"}
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d, separators=(",", ":")).encode()).rstrip(b"=").decode()
    h = b64(header)
    p = b64(payload)
    sig = hmac.new(settings.SECRET_KEY.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    s = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


def decode_access_token(token: str) -> Optional[dict]:
    import json
    import base64
    try:
        h, p, s = token.split(".")
        sig = hmac.new(settings.SECRET_KEY.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        expected = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        if not hmac.compare_digest(expected, s):
            return None
        def unb64(seg: str) -> dict:
            pad = "=" * (-len(seg) % 4)
            return json.loads(base64.urlsafe_b64decode(seg + pad))
        payload = unb64(p)
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# Simple in-memory rate limiter (per-process; fine for single worker dev)
_rate_buckets: dict[str, list[float]] = {}


def rate_limit_ok(key: str, limit_per_min: Optional[int] = None) -> bool:
    limit = limit_per_min or settings.RATE_LIMIT_PER_MIN
    now = time.time()
    bucket = _rate_buckets.get(key, [])
    bucket = [t for t in bucket if now - t < 60]
    if len(bucket) >= limit:
        _rate_buckets[key] = bucket
        return False
    bucket.append(now)
    _rate_buckets[key] = bucket
    return True


def allowed_file_type(filename: str) -> Optional[str]:
    """Return normalized type if allowed, else None. Never executes files."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext in ("txt", "json", "jsonl", "csv", "md"):
        return ext
    return None
