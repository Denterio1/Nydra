"""
security.py — Security & UX Module for Nydra

Features:
    - Rate limiting per user
    - File validation & sanitization
    - Session management
    - Privacy controls
    - Input sanitization
    - Audit logging
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE_MB    = 10000
ALLOWED_EXTENSIONS  = {".csv", ".xlsx", ".xls", ".json", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_REQUESTS_PER_MIN= 30
SESSION_TIMEOUT_MIN = 60
MAX_COLUMNS         = 500
MAX_ROWS            = 50_000_000


# ══════════════════════════════════════════════════════════════════════════════
# Rate Limiter
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Token bucket rate limiter per user."""

    def __init__(self, max_requests: int = MAX_REQUESTS_PER_MIN, window: int = 60):
        self.max_requests = max_requests
        self.window       = window
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed.

        Returns:
            (allowed, remaining_requests)
        """
        now      = time.time()
        cutoff   = now - self.window
        requests = self._requests[user_id]

        # Remove old requests
        self._requests[user_id] = [r for r in requests if r > cutoff]

        if len(self._requests[user_id]) >= self.max_requests:
            wait = int(self.window - (now - self._requests[user_id][0]))
            return False, wait

        self._requests[user_id].append(now)
        remaining = self.max_requests - len(self._requests[user_id])
        return True, remaining

    def reset(self, user_id: str) -> None:
        self._requests.pop(user_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# File Validator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    is_valid:  bool
    errors:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)
    file_hash: str = ""


class FileValidator:
    """Validate uploaded files for security and integrity."""

    def validate(self, file_name: str, file_size: int, file_bytes: bytes | None = None) -> ValidationResult:
        errors   = []
        warnings = []

        # Extension check
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"File type '{ext}' not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

        # Size check
        size_mb = file_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            errors.append(f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB")
        elif size_mb > 100:
            warnings.append(f"Large file ({size_mb:.1f}MB) — processing may be slow.")

        # Filename sanitization
        if not self._is_safe_filename(file_name):
            errors.append("Filename contains unsafe characters.")

        # File hash for integrity
        file_hash = ""
        if file_bytes:
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

            # Magic bytes check
            if ext == ".csv" and file_bytes[:3] == b'\xff\xfe\x00':
                warnings.append("File appears to be UTF-16 encoded. May cause issues.")

        return ValidationResult(
            is_valid  = len(errors) == 0,
            errors    = errors,
            warnings  = warnings,
            file_hash = file_hash,
        )

    def _is_safe_filename(self, filename: str) -> bool:
        dangerous = ["../", "..\\", "<", ">", ":", '"', "|", "?", "*", "\x00"]
        return not any(d in filename for d in dangerous)

    def validate_dataframe(self, df_shape: tuple[int, int]) -> ValidationResult:
        errors   = []
        warnings = []
        rows, cols = df_shape

        if rows == 0:
            errors.append("File contains no data rows.")
        if cols == 0:
            errors.append("File contains no columns.")
        if cols > MAX_COLUMNS:
            warnings.append(f"{cols} columns detected — very wide dataset, some features may be slow.")
        if rows > MAX_ROWS:
            warnings.append(f"{rows:,} rows — will sample to {MAX_ROWS:,} for performance.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


# ══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserSession:
    user_id:    str
    email:      str
    created_at: datetime = field(default_factory=datetime.now)
    last_active:datetime = field(default_factory=datetime.now)
    actions:    list[str] = field(default_factory=list)
    files:      list[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        return datetime.now() - self.last_active > timedelta(minutes=SESSION_TIMEOUT_MIN)

    def update_activity(self, action: str = "", file: str = "") -> None:
        self.last_active = datetime.now()
        if action:
            self.actions.append(f"{datetime.now().isoformat()[:19]}: {action}")
        if file and file not in self.files:
            self.files.append(file)

    @property
    def duration_minutes(self) -> int:
        return int((datetime.now() - self.created_at).total_seconds() / 60)


class SessionManager:
    """Manage user sessions in memory."""

    def __init__(self):
        self._sessions: dict[str, UserSession] = {}

    def create_or_get(self, email: str) -> UserSession:
        user_id = hashlib.md5(email.encode()).hexdigest()

        if user_id in self._sessions:
            session = self._sessions[user_id]
            if session.is_expired():
                session = UserSession(user_id=user_id, email=email)
                self._sessions[user_id] = session
            return session

        session = UserSession(user_id=user_id, email=email)
        self._sessions[user_id] = session
        return session

    def get(self, email: str) -> UserSession | None:
        user_id = hashlib.md5(email.encode()).hexdigest()
        return self._sessions.get(user_id)

    def cleanup_expired(self) -> int:
        expired = [uid for uid, s in self._sessions.items() if s.is_expired()]
        for uid in expired:
            del self._sessions[uid]
        return len(expired)


# ══════════════════════════════════════════════════════════════════════════════
# Input Sanitizer
# ══════════════════════════════════════════════════════════════════════════════

class InputSanitizer:
    """Sanitize user inputs to prevent injection attacks."""

    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
    MAX_NAME_LEN  = 100
    MAX_EMAIL_LEN = 254

    def sanitize_email(self, email: str) -> tuple[bool, str]:
        """
        Validate and sanitize email address.

        Returns:
            (is_valid, sanitized_email)
        """
        email = email.strip().lower()
        if len(email) > self.MAX_EMAIL_LEN:
            return False, ""
        if not self.EMAIL_PATTERN.match(email):
            return False, ""
        return True, email

    def sanitize_name(self, name: str) -> str:
        name = name.strip()[:self.MAX_NAME_LEN]
        name = re.sub(r'[<>"\'/\\]', '', name)
        return name

    def sanitize_column_name(self, col: str) -> str:
        return re.sub(r'[^\w\s\-]', '', str(col))[:100]


# ══════════════════════════════════════════════════════════════════════════════
# Audit Logger
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    timestamp:  str
    user_email: str
    action:     str
    file_name:  str
    success:    bool
    details:    str = ""


class AuditLogger:
    """Log security-relevant events."""

    def __init__(self, max_events: int = 1000):
        self._events: list[AuditEvent] = []
        self._max    = max_events

    def log(self, email: str, action: str, file_name: str = "", success: bool = True, details: str = "") -> None:
        event = AuditEvent(
            timestamp  = datetime.now().isoformat()[:19],
            user_email = email,
            action     = action,
            file_name  = file_name,
            success    = success,
            details    = details,
        )
        self._events.append(event)
        if len(self._events) > self._max:
            self._events = self._events[-self._max:]

    def get_recent(self, n: int = 50) -> list[AuditEvent]:
        return self._events[-n:]

    def get_user_events(self, email: str) -> list[AuditEvent]:
        return [e for e in self._events if e.user_email == email]

    def get_failed_events(self) -> list[AuditEvent]:
        return [e for e in self._events if not e.success]


# ══════════════════════════════════════════════════════════════════════════════
# Privacy Manager
# ══════════════════════════════════════════════════════════════════════════════

class PrivacyManager:
    """Manage data privacy and anonymization."""

    PII_PATTERNS = {
        "email":   re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'),
        "phone":   re.compile(r'\b(\+?[\d\s\-\(\)]{10,15})\b'),
        "ssn":     re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit":  re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
        "ip":      re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    }

    def detect_pii(self, df: Any) -> dict[str, list[str]]:
        """
        Detect potential PII columns in a dataframe.

        Returns:
            {"pii_type": ["column1", "column2", ...]}
        """
        import pandas as pd
        findings: dict[str, list[str]] = {}

        for col in df.columns:
            import pandas as pd_api
            if not pd.api.types.is_string_dtype(df[col]):
                continue
            sample = df[col].dropna().astype(str).head(20)
            for pii_type, pattern in self.PII_PATTERNS.items():
                matches = sample.apply(lambda x: bool(pattern.search(x))).sum()
                if matches >= 2:
                    findings.setdefault(pii_type, []).append(col)

        return findings

    def anonymize_column(self, series: Any, method: str = "hash") -> Any:
        """
        Anonymize a column.

        Methods:
            hash     — replace with SHA256 hash
            mask     — replace with ***
            drop     — return None series
            fake_id  — replace with sequential ID
        """
        import pandas as pd
        if method == "hash":
            return series.apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8] if pd.notna(x) else x)
        elif method == "mask":
            return series.apply(lambda x: "***" if pd.notna(x) else x)
        elif method == "drop":
            return pd.Series([None] * len(series), index=series.index)
        elif method == "fake_id":
            return pd.Series([f"ID_{i:06d}" for i in range(len(series))], index=series.index)
        return series

    def privacy_report(self, df: Any) -> dict[str, Any]:
        """Generate a privacy assessment report."""
        pii_findings = self.detect_pii(df)
        risk_level   = "high" if len(pii_findings) >= 3 else "medium" if pii_findings else "low"

        recommendations = []
        if "email" in pii_findings:
            recommendations.append(f"Hash or remove email columns: {pii_findings['email']}")
        if "phone" in pii_findings:
            recommendations.append(f"Mask phone columns: {pii_findings['phone']}")
        if "ssn" in pii_findings:
            recommendations.append(f"⚠️ SSN detected in: {pii_findings['ssn']} — remove immediately!")
        if "credit" in pii_findings:
            recommendations.append(f"⚠️ Credit card data in: {pii_findings['credit']} — remove immediately!")

        return {
            "risk_level":      risk_level,
            "pii_found":       pii_findings,
            "pii_types":       list(pii_findings.keys()),
            "affected_cols":   sum(len(v) for v in pii_findings.values()),
            "recommendations": recommendations,
            "safe_for_ml":     risk_level == "low",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Security Manager — Main Interface
# ══════════════════════════════════════════════════════════════════════════════

class SecurityManager:
    """
    Main security interface — orchestrates all security components.

    Usage:
        security = SecurityManager()
        allowed, remaining = security.rate_limiter.is_allowed(email)
        valid = security.file_validator.validate(name, size)
        session = security.session_manager.create_or_get(email)
        privacy = security.privacy_manager.privacy_report(df)
    """

    def __init__(self):
        self.rate_limiter    = RateLimiter()
        self.file_validator  = FileValidator()
        self.session_manager = SessionManager()
        self.input_sanitizer = InputSanitizer()
        self.audit_logger    = AuditLogger()
        self.privacy_manager = PrivacyManager()

    def validate_user_input(self, email: str, name: str = "") -> tuple[bool, str, str]:
        """
        Validate and sanitize user inputs.

        Returns:
            (is_valid, clean_email, clean_name)
        """
        is_valid, clean_email = self.input_sanitizer.sanitize_email(email)
        clean_name = self.input_sanitizer.sanitize_name(name)
        return is_valid, clean_email, clean_name

    def check_request(self, email: str, action: str, file_name: str = "") -> tuple[bool, str]:
        """
        Full security check for a request.

        Returns:
            (allowed, reason)
        """
        allowed, remaining = self.rate_limiter.is_allowed(email)
        if not allowed:
            self.audit_logger.log(email, action, file_name, success=False, details="Rate limited")
            return False, f"Too many requests. Please wait {remaining} seconds."

        self.audit_logger.log(email, action, file_name, success=True)
        session = self.session_manager.create_or_get(email)
        session.update_activity(action, file_name)
        return True, f"{remaining} requests remaining this minute."


# ── Singleton instance ────────────────────────────────────────────────────────
security = SecurityManager()
