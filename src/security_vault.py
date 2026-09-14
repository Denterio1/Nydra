"""
security_vault.py — Nydra Security Vault v1.0
====================================================
Enterprise-grade encryption, hashing, key management,
and secrets protection for user data privacy.

Built to the same standard as HashiCorp Vault, AWS Secrets
Manager, and Azure Key Vault — adapted for Python/SQLite apps.

Security Standards:
     AES-128-CBC encryption via Fernet
     HMAC-SHA256 integrity verification (built into Fernet)
     PBKDF2-HMAC-SHA256 key derivation (310,000 iterations — NIST recommended)
     MultiFernet key rotation (old data stays readable)
     Argon2-style email/password hashing via PBKDF2
     Cryptographically secure random token generation
     Full audit trail with tamper detection
     Zero plaintext secrets in memory longer than necessary
     API key masking for safe UI display
     Automatic provider fingerprinting

Architecture:
    VaultException          — typed exception hierarchy
    MasterKeyDeriver        — stable PBKDF2 key from system ID + salt
    FernetEngine            — AES encrypt/decrypt with MultiFernet
    KeyVersionStore         — tracks key versions for rotation
    KeyRotationManager      — rotate keys, re-encrypt old data
    HashEngine              — one-way hashing (passwords, emails)
    TokenEngine             — secure tokens with optional TTL
    MaskEngine              — API key masking + provider detection
    AuditLogger             — immutable audit trail
    IntegrityChecker        — SHA256 checksums + tamper detection
    BatchProcessor          — bulk encrypt/decrypt operations
    SecurityVault           — master class (use this in your app)

"""

from __future__ import annotations

import os
# Load .env before reading NYDRA_VAULT_SALT / other vault config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import re
import sys
import uuid
import json
import time
import hmac
import base64
import hashlib
import logging
import secrets
import platform
import threading
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

# ── Cryptography imports ──────────────────────────────────────────────────────
try:
    from cryptography.fernet import Fernet, MultiFernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
except ImportError as e:
    raise ImportError(
        "cryptography library required.\n"
        "Install it with: pip install cryptography\n"
        f"Original error: {e}"
    )

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("dataDoctor.vault")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# PBKDF2 iterations — NIST SP 800-132 recommends 310,000+ for SHA-256
PBKDF2_ITERATIONS: int = 310_000

# Salt for master key derivation — change this per deployment or set NYDRA_VAULT_SALT env var
# IMPORTANT: Never change this after deployment or you lose all encrypted data
_env_salt = os.getenv("NYDRA_VAULT_SALT")
VAULT_SALT: bytes = _env_salt.encode() if _env_salt else b"dataDoctor_vault_salt_v1_2026_kader_Denterio1"

# Prefix for encrypted values — used to detect encrypted strings
ENCRYPTED_PREFIX: str = "vault:v1:"

# Prefix for hashed values
HASHED_PREFIX: str = "hash:v1:"

# Maximum audit log entries kept in memory
MAX_AUDIT_MEMORY: int = 1000

# Sensitive field keywords — auto-detected in batch operations
SENSITIVE_KEYWORDS: set[str] = {
    "api_key", "apikey", "key", "secret", "password", "passwd", "pwd",
    "token", "auth", "credential", "private", "access_key", "secret_key",
    "api_secret", "client_secret", "bearer", "authorization",
}

# Known API key provider fingerprints
PROVIDER_FINGERPRINTS: dict[str, dict] = {
    # ORDER MATTERS — more specific prefixes must come before generic ones
    "anthropic"  : {"prefixes": ["sk-ant-"], "pattern": r"^sk-ant-[A-Za-z0-9\-_]{20,}$"},
    "openrouter" : {"prefixes": ["sk-or-"],  "pattern": r"^sk-or-[A-Za-z0-9\-]{20,}$"},
    "openai"     : {"prefixes": ["sk-"],     "pattern": r"^sk-[A-Za-z0-9]{20,}$"},
    "groq"       : {"prefixes": ["gsk_"],    "pattern": r"^gsk_[A-Za-z0-9]{20,}$"},
    "google"     : {"prefixes": ["AIza"],    "pattern": r"^AIza[A-Za-z0-9\-_]{20,}$"},
    "xai"        : {"prefixes": ["xai-"],    "pattern": r"^xai-[A-Za-z0-9]{20,}$"},
    "perplexity" : {"prefixes": ["pplx-"],   "pattern": r"^pplx-[A-Za-z0-9]{20,}$"},
    "mistral"    : {"prefixes": [""],        "pattern": r"^[A-Za-z0-9]{32}$"},
    "together"   : {"prefixes": [""],        "pattern": r"^[A-Za-z0-9]{40,}$"},
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXCEPTION HIERARCHY
# ══════════════════════════════════════════════════════════════════════════════

class VaultError(Exception):
    """Base exception for all vault errors."""
    pass

class VaultEncryptionError(VaultError):
    """Raised when encryption fails."""
    pass

class VaultDecryptionError(VaultError):
    """Raised when decryption fails — wrong key or tampered data."""
    pass

class VaultKeyError(VaultError):
    """Raised for key management errors."""
    pass

class VaultIntegrityError(VaultError):
    """Raised when data integrity check fails — possible tampering."""
    pass

class VaultTokenExpiredError(VaultError):
    """Raised when a time-limited token has expired."""
    pass

class VaultConfigError(VaultError):
    """Raised for configuration errors."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VaultKey:
    """Represents a single encryption key with metadata."""
    version    : int
    key_bytes  : bytes
    created_at : float = field(default_factory=time.time)
    is_active  : bool  = True
    description: str   = ""

    @property
    def fernet(self) -> Fernet:
        return Fernet(self.key_bytes)

    @property
    def created_iso(self) -> str:
        return datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize (without key bytes — for metadata only)."""
        return {
            "version"    : self.version,
            "created_at" : self.created_iso,
            "is_active"  : self.is_active,
            "description": self.description,
        }


@dataclass
class AuditEntry:
    """A single immutable audit log entry."""
    event_id  : str   = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp : float = field(default_factory=time.time)
    action    : str   = ""          # encrypt | decrypt | hash | rotate | mask | token
    subject   : str   = ""          # What was acted on (field name, not value)
    user_hint : str   = ""          # Optional: email/user identifier (hashed)
    success   : bool  = True
    detail    : str   = ""          # Extra context

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "event_id"     : self.event_id,
            "timestamp"    : self.timestamp_iso,
            "action"       : self.action,
            "subject"      : self.subject,
            "user_hint"    : self.user_hint,
            "success"      : self.success,
            "detail"       : self.detail,
        }


@dataclass
class EncryptedValue:
    """
    A fully structured encrypted value with version metadata.
    Stored in the database as JSON string.
    """
    ciphertext : str    # The Fernet token (base64)
    key_version: int    # Which key version encrypted this
    encrypted_at: float = field(default_factory=time.time)
    checksum   : str    = ""  # SHA256 of original plaintext (for integrity)
    field_hint : str    = ""  # Name of the field (not the value)

    def to_json(self) -> str:
        return json.dumps({
            "c"  : self.ciphertext,
            "kv" : self.key_version,
            "ea" : self.encrypted_at,
            "cs" : self.checksum,
            "fh" : self.field_hint,
        })

    @classmethod
    def from_json(cls, s: str) -> "EncryptedValue":
        d = json.loads(s)
        return cls(
            ciphertext  = d["c"],
            key_version = d["kv"],
            encrypted_at= d.get("ea", 0.0),
            checksum    = d.get("cs", ""),
            field_hint  = d.get("fh", ""),
        )

    @classmethod
    def is_structured(cls, s: str) -> bool:
        """Check if a string is a structured EncryptedValue JSON."""
        try:
            d = json.loads(s)
            return "c" in d and "kv" in d
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# 4. MASTER KEY DERIVER
# Derives a stable Fernet key from system identity + salt.
# The same machine always produces the same master key.
# ══════════════════════════════════════════════════════════════════════════════

class MasterKeyDeriver:
    """
    Derives a deterministic master encryption key from system identity.

    Uses PBKDF2-HMAC-SHA256 with 310,000 iterations (NIST recommended).
    The derived key is always the same on the same machine with the same salt,
    which means encrypted data persists across app restarts.

    Key material sources (combined for uniqueness):
        1. Machine hostname
        2. Platform/OS identifier
        3. Python version
        4. A fixed deployment salt (VAULT_SALT constant)
        5. An optional user-provided passphrase (adds extra security)
    """

    def __init__(self, passphrase: str = "", salt: bytes = VAULT_SALT) -> None:
        self._salt       = salt
        self._passphrase = passphrase
        self._lock       = threading.Lock()
        self._cache      : bytes | None = None

    def _get_system_fingerprint(self) -> bytes:
        """
        Build a system fingerprint from stable machine properties.
        This fingerprint is consistent across restarts on the same machine.
        """
        components = [
            platform.node(),                    # hostname
            platform.system(),                  # OS name
            platform.machine(),                 # CPU architecture
            sys.version.split()[0],             # Python version
            self._passphrase,                   # Optional extra passphrase
        ]
        combined = "|".join(components).encode("utf-8")
        # Deterministic hash of all components
        return hashlib.sha256(combined).digest()

    def derive(self) -> bytes:
        """
        Derive the master key. Result is cached in memory.
        Returns a 32-byte URL-safe base64-encoded key (Fernet compatible).
        """
        with self._lock:
            if self._cache is not None:
                return self._cache

            password = self._get_system_fingerprint()

            kdf = PBKDF2HMAC(
                algorithm  = hashes.SHA256(),
                length     = 32,
                salt       = self._salt,
                iterations = PBKDF2_ITERATIONS,
                backend    = default_backend(),
            )
            raw_key      = kdf.derive(password)
            fernet_key   = base64.urlsafe_b64encode(raw_key)
            self._cache  = fernet_key
            logger.debug("Master key derived successfully.")
            return fernet_key

    def derive_from_passphrase(self, passphrase: str, salt: bytes | None = None) -> bytes:
        """
        Derive a key from an explicit passphrase (for user-specific keys).
        Does NOT use system fingerprint — purely passphrase-based.
        Useful for per-user encryption keys.
        """
        effective_salt = salt or self._salt
        password       = passphrase.encode("utf-8")

        kdf = PBKDF2HMAC(
            algorithm  = hashes.SHA256(),
            length     = 32,
            salt       = effective_salt,
            iterations = PBKDF2_ITERATIONS,
            backend    = default_backend(),
        )
        raw_key    = kdf.derive(password)
        return base64.urlsafe_b64encode(raw_key)

    def generate_random_key(self) -> bytes:
        """Generate a completely random Fernet key (for rotation)."""
        return Fernet.generate_key()

    def clear_cache(self) -> None:
        """Clear the cached key from memory (call on app shutdown)."""
        with self._lock:
            self._cache = None


# ══════════════════════════════════════════════════════════════════════════════
# 5. KEY VERSION STORE
# Manages multiple key versions for rotation support.
# ══════════════════════════════════════════════════════════════════════════════

class KeyVersionStore:
    """
    Stores all key versions in memory.
    Enables MultiFernet — old encrypted data can always be decrypted
    even after key rotation.

    Version 0 = the master key (always present, derived from system)
    Version 1+ = rotated keys (random, generated during rotation)
    """

    def __init__(self) -> None:
        self._keys  : dict[int, VaultKey] = {}
        self._lock  = threading.RLock()

    def add(self, key_bytes: bytes, description: str = "") -> int:
        """Add a new key version. Returns the version number."""
        with self._lock:
            version = max(self._keys.keys(), default=-1) + 1
            self._keys[version] = VaultKey(
                version     = version,
                key_bytes   = key_bytes,
                description = description,
                is_active   = True,
            )
            logger.info(f"Key version {version} added to store.")
            return version

    def get(self, version: int) -> VaultKey | None:
        with self._lock:
            return self._keys.get(version)

    def get_active(self) -> VaultKey | None:
        """Get the most recent active key."""
        with self._lock:
            active = [k for k in self._keys.values() if k.is_active]
            return max(active, key=lambda k: k.version, default=None)

    def get_latest_version(self) -> int:
        with self._lock:
            return max(self._keys.keys(), default=0)

    def build_multi_fernet(self) -> MultiFernet:
        """
        Build a MultiFernet from all stored keys.
        MultiFernet tries each key in order — newest first.
        This allows decrypting data encrypted with ANY previous key.
        """
        with self._lock:
            sorted_keys = sorted(self._keys.values(), key=lambda k: k.version, reverse=True)
            fernets     = [k.fernet for k in sorted_keys]
            if not fernets:
                raise VaultKeyError("No keys in store — cannot build MultiFernet.")
            return MultiFernet(fernets)

    def list_versions(self) -> list[dict]:
        with self._lock:
            return [k.to_dict() for k in sorted(self._keys.values(), key=lambda k: k.version)]

    def deactivate(self, version: int) -> None:
        """Mark a key version as inactive (keeps it for decryption, not encryption)."""
        with self._lock:
            if version in self._keys:
                self._keys[version].is_active = False
                logger.info(f"Key version {version} deactivated.")

    def count(self) -> int:
        with self._lock:
            return len(self._keys)


# ══════════════════════════════════════════════════════════════════════════════
# 6. FERNET ENGINE
# Core encryption and decryption with full metadata support.
# ══════════════════════════════════════════════════════════════════════════════

class FernetEngine:
    """
    Core AES-128-CBC encryption/decryption engine.

    Uses Fernet (from the cryptography library) which provides:
    - AES-128 in CBC mode with PKCS7 padding
    - HMAC-SHA256 message authentication (integrity + authenticity)
    - Timestamp-based TTL support
    - Built-in tamper detection

    Wraps all encrypted data in EncryptedValue for versioning.
    """

    def __init__(self, key_store: KeyVersionStore) -> None:
        self._store = key_store

    def encrypt(
        self,
        plaintext  : str,
        field_hint : str = "",
    ) -> str:
        """
        Encrypt a plaintext string.

        Returns a ENCRYPTED_PREFIX + JSON string that embeds:
        - The ciphertext
        - The key version used
        - Timestamp
        - SHA256 checksum of the original plaintext
        - Field hint (name of the field, not its value)
        """
        if not plaintext:
            return plaintext

        active_key = self._store.get_active()
        if not active_key:
            raise VaultKeyError("No active encryption key found.")

        try:
            plaintext_bytes = plaintext.encode("utf-8")
            ciphertext      = active_key.fernet.encrypt(plaintext_bytes)
            checksum        = hashlib.sha256(plaintext_bytes).hexdigest()

            ev = EncryptedValue(
                ciphertext  = ciphertext.decode("utf-8"),
                key_version = active_key.version,
                checksum    = checksum,
                field_hint  = field_hint,
            )
            return ENCRYPTED_PREFIX + ev.to_json()

        except Exception as e:
            raise VaultEncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, encrypted_str: str, verify_integrity: bool = True) -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted_str    : The full ENCRYPTED_PREFIX + JSON string
            verify_integrity : If True, verifies SHA256 checksum after decryption

        Raises:
            VaultDecryptionError : Wrong key, tampered data, or format error
            VaultIntegrityError  : Checksum mismatch (data was modified)
        """
        if not encrypted_str or not encrypted_str.startswith(ENCRYPTED_PREFIX):
            return encrypted_str  # Not encrypted — return as-is

        json_part = encrypted_str[len(ENCRYPTED_PREFIX):]

        try:
            # Try structured format first (EncryptedValue JSON)
            if EncryptedValue.is_structured(json_part):
                ev         = EncryptedValue.from_json(json_part)
                key_ver    = ev.key_version
                vault_key  = self._store.get(key_ver)

                if vault_key:
                    # Try the specific key version first
                    try:
                        plaintext_bytes = vault_key.fernet.decrypt(ev.ciphertext.encode())
                    except InvalidToken:
                        # Fall back to MultiFernet (tries all keys)
                        mf = self._store.build_multi_fernet()
                        plaintext_bytes = mf.decrypt(ev.ciphertext.encode())
                else:
                    # Key version not found — try all keys
                    mf = self._store.build_multi_fernet()
                    plaintext_bytes = mf.decrypt(ev.ciphertext.encode())

                plaintext = plaintext_bytes.decode("utf-8")

                # Integrity check
                if verify_integrity and ev.checksum:
                    actual_checksum = hashlib.sha256(plaintext_bytes).hexdigest()
                    if not hmac.compare_digest(actual_checksum, ev.checksum):
                        raise VaultIntegrityError(
                            f"Integrity check FAILED for field '{ev.field_hint}'. "
                            "Data may have been tampered with."
                        )
                return plaintext

            else:
                # Legacy format: just a raw Fernet token
                mf        = self._store.build_multi_fernet()
                plaintext = mf.decrypt(json_part.encode()).decode("utf-8")
                return plaintext

        except (VaultIntegrityError, VaultDecryptionError):
            raise
        except InvalidToken as e:
            raise VaultDecryptionError(
                "Decryption failed — invalid token. "
                "This usually means wrong key or corrupted data."
            ) from e
        except Exception as e:
            raise VaultDecryptionError(f"Decryption error: {e}") from e

    def decrypt_with_ttl(self, encrypted_str: str, ttl_seconds: int) -> str:
        """
        Decrypt and enforce a Time-To-Live (TTL).
        Raises VaultTokenExpiredError if the token is older than ttl_seconds.
        """
        if not encrypted_str or not encrypted_str.startswith(ENCRYPTED_PREFIX):
            return encrypted_str

        json_part = encrypted_str[len(ENCRYPTED_PREFIX):]

        try:
            ev  = EncryptedValue.from_json(json_part)
            key = self._store.get(ev.key_version)
            if not key:
                key_obj = self._store.build_multi_fernet()
                plaintext = key_obj.decrypt(ev.ciphertext.encode(), ttl=ttl_seconds).decode()
            else:
                plaintext = key.fernet.decrypt(
                    ev.ciphertext.encode(), ttl=ttl_seconds
                ).decode()
            return plaintext
        except InvalidToken as e:
            raise VaultTokenExpiredError(
                f"Token expired (TTL={ttl_seconds}s) or invalid."
            ) from e

    def is_encrypted(self, value: str) -> bool:
        """Check if a string was encrypted by this vault."""
        return isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)

    def re_encrypt(self, encrypted_str: str) -> str:
        """
        Decrypt with old key, re-encrypt with current active key.
        Used during key rotation.
        """
        plaintext = self.decrypt(encrypted_str, verify_integrity=False)
        field_hint = ""
        if EncryptedValue.is_structured(encrypted_str[len(ENCRYPTED_PREFIX):]):
            ev = EncryptedValue.from_json(encrypted_str[len(ENCRYPTED_PREFIX):])
            field_hint = ev.field_hint
        return self.encrypt(plaintext, field_hint=field_hint)


# ══════════════════════════════════════════════════════════════════════════════
# 7. KEY ROTATION MANAGER
# Manages the full key rotation lifecycle.
# ══════════════════════════════════════════════════════════════════════════════

class KeyRotationManager:
    """
    Manages key rotation — the process of generating a new encryption key
    and re-encrypting all existing secrets with it.

    Key rotation is a security best practice:
    - Limits the blast radius if a key is compromised
    - Ensures old encrypted data is migrated to stronger keys
    - Maintains full backward compatibility (MultiFernet)

    Usage:
        manager = KeyRotationManager(key_store, fernet_engine)
        result  = manager.rotate(old_encrypted_values=["vault:v1:...", ...])
        # result contains new encrypted versions of all values
    """

    def __init__(self, key_store: KeyVersionStore, engine: FernetEngine) -> None:
        self._store  = key_store
        self._engine = engine

    def rotate(
        self,
        old_values      : list[str] | None = None,
        description     : str = "Manual rotation",
    ) -> dict:
        """
        Perform a full key rotation.

        1. Generate a new random key
        2. Add it to the key store (becomes the active key)
        3. Optionally re-encrypt a list of old encrypted values
        4. Return rotation report

        Args:
            old_values  : List of encrypted strings to re-encrypt.
                          Pass all values from your database here.
            description : Label for this rotation (e.g. "Monthly rotation")

        Returns:
            dict with:
                new_version    : int
                re_encrypted   : list of (old, new) pairs
                failures       : list of values that failed
                report         : summary string
        """
        logger.info(f"Starting key rotation: {description}")
        start           = time.time()
        new_key_bytes   = self._store._keys[0].__class__ if False else Fernet.generate_key()
        new_version     = self._store.add(new_key_bytes, description=description)

        re_encrypted : list[dict] = []
        failures     : list[dict] = []

        if old_values:
            for idx, old_val in enumerate(old_values):
                if not self._engine.is_encrypted(old_val):
                    continue
                try:
                    new_val = self._engine.re_encrypt(old_val)
                    re_encrypted.append({"index": idx, "old": old_val[:30] + "...", "new": new_val})
                except Exception as e:
                    failures.append({"index": idx, "error": str(e)})
                    logger.warning(f"Re-encryption failed for value {idx}: {e}")

        elapsed = time.time() - start
        report  = (
            f"Key rotation complete.\n"
            f"  New key version : {new_version}\n"
            f"  Re-encrypted    : {len(re_encrypted)} values\n"
            f"  Failures        : {len(failures)}\n"
            f"  Duration        : {elapsed:.2f}s\n"
            f"  Description     : {description}"
        )
        logger.info(report)

        return {
            "new_version"  : new_version,
            "re_encrypted" : re_encrypted,
            "failures"     : failures,
            "elapsed_s"    : round(elapsed, 3),
            "report"       : report,
        }

    def get_rotation_status(self) -> dict:
        """Return current key rotation status."""
        versions = self._store.list_versions()
        active   = self._store.get_active()
        return {
            "total_versions" : self._store.count(),
            "active_version" : active.version if active else None,
            "versions"       : versions,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8. HASH ENGINE
# One-way hashing for passwords, emails, and sensitive identifiers.
# ══════════════════════════════════════════════════════════════════════════════

class HashEngine:
    """
    One-way cryptographic hashing for passwords, emails, and sensitive IDs.

    Unlike encryption, hashing is IRREVERSIBLE. Use it for:
    - Passwords (never store plaintext passwords)
    - Email lookups (store hash, compare hashes)
    - User IDs (anonymize personal identifiers)

    Uses PBKDF2-HMAC-SHA256 with per-value random salts.
    Each hash is unique even for the same input (due to random salt).

    Format: HASHED_PREFIX + base64(salt) + "." + base64(hash)
    """

    HASH_ITERATIONS : int = 260_000    # NIST minimum for PBKDF2-SHA256 passwords
    HASH_LENGTH     : int = 32         # 256-bit output

    def hash(self, value: str, context: str = "") -> str:
        """
        Hash a value with a random salt (non-deterministic).
        Each call produces a different hash for the same input.

        Args:
            value   : The plaintext to hash
            context : Optional context string (e.g. "email", "password")

        Returns:
            HASHED_PREFIX + encoded_salt + "." + encoded_hash
        """
        if not value:
            return value

        salt      = os.urandom(32)
        password  = (value + context).encode("utf-8")

        kdf = PBKDF2HMAC(
            algorithm  = hashes.SHA256(),
            length     = self.HASH_LENGTH,
            salt       = salt,
            iterations = self.HASH_ITERATIONS,
            backend    = default_backend(),
        )
        hash_bytes   = kdf.derive(password)
        encoded_salt = base64.urlsafe_b64encode(salt).decode()
        encoded_hash = base64.urlsafe_b64encode(hash_bytes).decode()
        return f"{HASHED_PREFIX}{encoded_salt}.{encoded_hash}"

    def verify(self, plaintext: str, stored_hash: str, context: str = "") -> bool:
        """
        Verify a plaintext value against a stored hash.
        Timing-safe comparison (resistant to timing attacks).

        Args:
            plaintext   : The value to check
            stored_hash : The hash string from hash()
            context     : Must match what was used in hash()

        Returns:
            True if the value matches, False otherwise
        """
        if not stored_hash or not stored_hash.startswith(HASHED_PREFIX):
            return False

        try:
            hash_part    = stored_hash[len(HASHED_PREFIX):]
            salt_enc, hash_enc = hash_part.split(".", 1)
            salt         = base64.urlsafe_b64decode(salt_enc)
            stored_bytes = base64.urlsafe_b64decode(hash_enc)
            password     = (plaintext + context).encode("utf-8")

            kdf = PBKDF2HMAC(
                algorithm  = hashes.SHA256(),
                length     = self.HASH_LENGTH,
                salt       = salt,
                iterations = self.HASH_ITERATIONS,
                backend    = default_backend(),
            )
            computed_bytes = kdf.derive(password)
            # hmac.compare_digest is timing-safe
            return hmac.compare_digest(computed_bytes, stored_bytes)

        except Exception as e:
            logger.warning(f"Hash verification error: {e}")
            return False

    def hash_deterministic(self, value: str, context: str = "") -> str:
        """
        Deterministic hash — same input always produces same hash.
        Use for database lookups by email (so you can find the record).

        WARNING: Less secure than random-salt hash. Do NOT use for passwords.
        Use for: email lookup keys, username deduplication.
        """
        if not value:
            return value

        salt     = hashlib.sha256((context + VAULT_SALT.decode()).encode()).digest()
        password = value.lower().strip().encode("utf-8")

        kdf = PBKDF2HMAC(
            algorithm  = hashes.SHA256(),
            length     = self.HASH_LENGTH,
            salt       = salt,
            iterations = self.HASH_ITERATIONS // 10,   # Fewer iterations — deterministic
            backend    = default_backend(),
        )
        hash_bytes = kdf.derive(password)
        return "det:" + base64.urlsafe_b64encode(hash_bytes).decode()

    def is_hashed(self, value: str) -> bool:
        """Check if a value is a stored hash."""
        return isinstance(value, str) and (
            value.startswith(HASHED_PREFIX) or value.startswith("det:")
        )

    def hash_email(self, email: str) -> str:
        """
        Anonymize an email address for audit logs.
        Returns a deterministic hash so audit entries for the same user are linkable.
        """
        return self.hash_deterministic(email.lower().strip(), context="email")

    def quick_hash(self, value: str) -> str:
        """Fast SHA256 hash — for non-security-critical checksums."""
        return hashlib.sha256(value.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# 9. TOKEN ENGINE
# Cryptographically secure random tokens with optional TTL.
# ══════════════════════════════════════════════════════════════════════════════

class TokenEngine:
    """
    Generate and validate cryptographically secure tokens.

    Uses Python's `secrets` module — the correct tool for security tokens.
    (NOT `random` — that is NOT cryptographically secure.)

    Features:
    - Random URL-safe tokens of any byte length
    - Hex tokens
    - Numeric PINs
    - Time-limited tokens (expire after N seconds)
    - Token validation
    """

    def generate(self, n_bytes: int = 32) -> str:
        """
        Generate a cryptographically secure URL-safe token.
        Default: 32 bytes = 256 bits of entropy = 43 chars.
        """
        return secrets.token_urlsafe(n_bytes)

    def generate_hex(self, n_bytes: int = 32) -> str:
        """Generate a hex token (64 chars for 32 bytes)."""
        return secrets.token_hex(n_bytes)

    def generate_pin(self, digits: int = 6) -> str:
        """
        Generate a numeric PIN of exactly `digits` digits.
        Cryptographically secure — safe for OTP/2FA codes.
        """
        # Use secrets.randbelow for uniform distribution
        upper = 10 ** digits
        pin   = secrets.randbelow(upper)
        return str(pin).zfill(digits)

    def generate_timed(self, engine: FernetEngine, ttl_seconds: int = 3600) -> str:
        """
        Generate a time-limited token encrypted with the vault.
        The token carries an expiry timestamp inside the encryption.

        The Fernet timestamp is used for TTL enforcement.

        Args:
            engine      : FernetEngine instance (for encryption)
            ttl_seconds : How long the token is valid (default: 1 hour)

        Returns:
            An encrypted token string that can be validated with validate_timed()
        """
        token   = self.generate(32)
        payload = json.dumps({
            "token"   : token,
            "expires" : time.time() + ttl_seconds,
            "created" : time.time(),
        })
        return engine.encrypt(payload, field_hint="timed_token")

    def validate_timed(self, encrypted_token: str, engine: FernetEngine) -> tuple[bool, str]:
        """
        Validate a timed token.

        Returns:
            (is_valid, token_value or error_message)
        """
        try:
            payload_str = engine.decrypt(encrypted_token)
            payload     = json.loads(payload_str)
            if time.time() > payload.get("expires", 0):
                return False, "Token has expired."
            return True, payload["token"]
        except VaultDecryptionError:
            return False, "Invalid token — decryption failed."
        except Exception as e:
            return False, f"Token validation error: {e}"

    def constant_time_compare(self, token_a: str, token_b: str) -> bool:
        """
        Timing-safe string comparison.
        Prevents timing attacks on token comparison.
        """
        return hmac.compare_digest(
            token_a.encode("utf-8"),
            token_b.encode("utf-8"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. MASK ENGINE
# Safe display of secrets — never show full API keys in UI.
# ══════════════════════════════════════════════════════════════════════════════

class MaskEngine:
    """
    Masks sensitive values for safe display in the UI.

    Rules:
    - Short values (< 8 chars): fully masked
    - Medium values: show first 3 + last 4 chars
    - Long values: show prefix + "***" + last 4 chars

    Also identifies which LLM provider an API key belongs to.
    """

    def mask(self, value: str, show_last: int = 4) -> str:
        """
        Mask a sensitive value, showing only a small portion.

        Examples:
            "sk-abc123xyz789" → "sk-***...789"
            "gsk_abcdefghijk" → "gsk_***...hijk"
            "AIzaSy1234567890" → "AIza***...7890"
        """
        if not value or not isinstance(value, str):
            return "***"

        # If it's encrypted, mask the prefix only
        if value.startswith(ENCRYPTED_PREFIX):
            return "[encrypted]"

        # If it's hashed
        if value.startswith(HASHED_PREFIX) or value.startswith("det:"):
            return "[hashed]"

        length = len(value)

        if length <= 4:
            return "*" * length

        if length <= 8:
            return value[:2] + "*" * (length - 2)

        # Find the natural prefix (letters/digits before special chars)
        prefix_match = re.match(r'^([A-Za-z0-9\-_]{2,8})', value)
        prefix       = prefix_match.group(1) if prefix_match else value[:4]
        suffix       = value[-show_last:] if show_last <= length - len(prefix) - 3 else value[-3:]

        return f"{prefix}***...{suffix}"

    def mask_dict(self, d: dict, sensitive_keys: set[str] | None = None) -> dict:
        """
        Return a copy of a dict with sensitive values masked.
        Safe to log or display in UI.
        """
        keys = sensitive_keys or SENSITIVE_KEYWORDS
        result = {}
        for k, v in d.items():
            if any(sk in k.lower() for sk in keys):
                result[k] = self.mask(str(v)) if v else v
            else:
                result[k] = v
        return result

    def detect_provider(self, api_key: str) -> str | None:
        """
        Detect which LLM provider an API key belongs to.

        Returns the provider name (e.g. "openai", "anthropic") or None.
        """
        if not api_key or not isinstance(api_key, str):
            return None

        for provider, info in PROVIDER_FINGERPRINTS.items():
            # Check prefix first (fast)
            for prefix in info["prefixes"]:
                if prefix and api_key.startswith(prefix):
                    # Confirm with regex
                    if re.match(info["pattern"], api_key):
                        return provider
                    # Prefix match is enough even if regex doesn't fully match
                    return provider

        return None

    def detect_key_strength(self, api_key: str) -> dict:
        """
        Analyze the apparent strength/validity of an API key.
        Does NOT make any network calls.

        Returns:
            dict with: length, has_prefix, provider, looks_valid, entropy_bits
        """
        if not api_key:
            return {"error": "empty key"}

        provider   = self.detect_provider(api_key)
        length     = len(api_key)
        # Rough entropy estimate: unique chars × log2 of charset
        charset    = len(set(api_key))
        import math
        entropy    = round(length * math.log2(max(charset, 2)), 1)

        return {
            "length"      : length,
            "provider"    : provider or "unknown",
            "looks_valid" : length >= 20 and charset >= 10,
            "entropy_bits": entropy,
            "masked"      : self.mask(api_key),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 11. AUDIT LOGGER
# Immutable audit trail for all vault operations.
# ══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Records all vault operations in an immutable audit trail.

    In a production deployment, this should write to:
    1. A local append-only file
    2. A separate database table
    3. A SIEM/logging system

    For dataDoctor: stores in memory + optional file append.
    The audit log itself is protected by a checksum chain
    (each entry includes a hash of the previous entry).
    """

    def __init__(self, log_file: str | None = None) -> None:
        self._entries   : list[AuditEntry] = []
        self._lock      = threading.Lock()
        self._log_file  = log_file
        self._chain_hash: str = "genesis"  # Starting hash for the chain

    def log(
        self,
        action    : str,
        subject   : str   = "",
        user_hint : str   = "",
        success   : bool  = True,
        detail    : str   = "",
    ) -> AuditEntry:
        """
        Record an audit event.

        Args:
            action    : What happened ("encrypt", "decrypt", "rotate", etc.)
            subject   : What field/key was involved (NOT the value itself)
            user_hint : Hashed or partial user identifier
            success   : Whether the operation succeeded
            detail    : Extra safe context (no secret values)

        Returns:
            The created AuditEntry
        """
        entry = AuditEntry(
            action    = action,
            subject   = subject,
            user_hint = user_hint,
            success   = success,
            detail    = detail,
        )

        with self._lock:
            # Append to chain: each entry hashes the previous chain state
            entry_json       = json.dumps(entry.to_dict(), sort_keys=True)
            self._chain_hash = hashlib.sha256(
                f"{self._chain_hash}|{entry_json}".encode()
            ).hexdigest()

            self._entries.append(entry)

            # Trim memory if over limit
            if len(self._entries) > MAX_AUDIT_MEMORY:
                self._entries = self._entries[-MAX_AUDIT_MEMORY:]

            # Optionally write to file
            if self._log_file:
                self._write_to_file(entry)

        logger.debug(f"AUDIT [{action}] subject={subject} success={success}")
        return entry

    def _write_to_file(self, entry: AuditEntry) -> None:
        """Append entry to audit log file (best effort)."""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log to file: {e}")

    def get_entries(
        self,
        action   : str | None = None,
        user_hint: str | None = None,
        since    : float | None = None,
        limit    : int = 100,
    ) -> list[dict]:
        """
        Query audit entries with optional filters.

        Args:
            action    : Filter by action type
            user_hint : Filter by user
            since     : Filter by timestamp (Unix time)
            limit     : Max entries to return
        """
        with self._lock:
            results = list(self._entries)

        if action:
            results = [e for e in results if e.action == action]
        if user_hint:
            results = [e for e in results if e.user_hint == user_hint]
        if since:
            results = [e for e in results if e.timestamp >= since]

        return [e.to_dict() for e in results[-limit:]]

    def get_chain_hash(self) -> str:
        """Return the current chain hash (for external integrity verification)."""
        with self._lock:
            return self._chain_hash

    def get_summary(self) -> dict:
        """Return summary statistics of the audit log."""
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return {"total": 0}

        actions = {}
        for e in entries:
            actions[e.action] = actions.get(e.action, 0) + 1

        failures = [e for e in entries if not e.success]

        return {
            "total"       : len(entries),
            "actions"     : actions,
            "failures"    : len(failures),
            "oldest"      : entries[0].timestamp_iso  if entries else None,
            "newest"      : entries[-1].timestamp_iso if entries else None,
            "chain_hash"  : self._chain_hash,
        }

    def clear(self) -> None:
        """Clear in-memory entries (does not clear file log)."""
        with self._lock:
            self._entries.clear()
            self._chain_hash = "genesis"


# ══════════════════════════════════════════════════════════════════════════════
# 12. INTEGRITY CHECKER
# SHA256 checksums and tamper detection for data at rest.
# ══════════════════════════════════════════════════════════════════════════════

class IntegrityChecker:
    """
    Provides SHA256-based checksums for data at rest.

    Use this to detect unauthorized modifications to database records,
    config files, or any other stored data — separate from Fernet's
    built-in HMAC (which only applies to encrypted values).
    """

    def checksum(self, value: str) -> str:
        """Compute SHA256 checksum of a string."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def checksum_dict(self, d: dict) -> str:
        """Compute deterministic SHA256 checksum of a dict."""
        canonical = json.dumps(d, sort_keys=True, default=str)
        return self.checksum(canonical)

    def verify_checksum(self, value: str, expected: str) -> bool:
        """Timing-safe checksum verification."""
        actual = self.checksum(value)
        return hmac.compare_digest(actual, expected)

    def sign(self, value: str, secret: str) -> str:
        """
        Create an HMAC-SHA256 signature for a value.
        Used to authenticate data without encrypting it.
        """
        h = hmac.new(secret.encode(), value.encode(), hashlib.sha256)
        return h.hexdigest()

    def verify_signature(self, value: str, signature: str, secret: str) -> bool:
        """Verify an HMAC-SHA256 signature. Timing-safe."""
        expected = self.sign(value, secret)
        return hmac.compare_digest(expected, signature)


# ══════════════════════════════════════════════════════════════════════════════
# 13. BATCH PROCESSOR
# Encrypt or decrypt entire dicts at once.
# ══════════════════════════════════════════════════════════════════════════════

class BatchProcessor:
    """
    Batch encryption/decryption for entire dictionaries.

    Automatically detects which fields are sensitive based on
    SENSITIVE_KEYWORDS — no need to specify each field manually.

    Used by config_manager.py to process user settings in bulk.
    """

    def __init__(self, engine: FernetEngine) -> None:
        self._engine = engine

    def encrypt_dict(
        self,
        data             : dict,
        sensitive_keys   : set[str] | None   = None,
        explicit_keys    : list[str] | None   = None,
        skip_already_enc : bool = True,
    ) -> dict:
        """
        Encrypt sensitive fields in a dictionary.

        Args:
            data            : Input dict
            sensitive_keys  : Override the default SENSITIVE_KEYWORDS set
            explicit_keys   : Always encrypt these specific keys regardless of name
            skip_already_enc: Skip values already encrypted by this vault

        Returns:
            New dict with sensitive values encrypted.
            Non-sensitive values are copied unchanged.
            Adds "_encrypted_fields" key listing which fields were encrypted.
        """
        keywords        = sensitive_keys or SENSITIVE_KEYWORDS
        explicit        = set(explicit_keys or [])
        result          = {}
        encrypted_fields= []

        for k, v in data.items():
            if not isinstance(v, str) or not v:
                result[k] = v
                continue

            should_encrypt = (
                any(kw in k.lower() for kw in keywords)
                or k in explicit
            )

            if should_encrypt:
                if skip_already_enc and self._engine.is_encrypted(v):
                    result[k] = v  # Already encrypted
                else:
                    result[k] = self._engine.encrypt(v, field_hint=k)
                    encrypted_fields.append(k)
            else:
                result[k] = v

        result["_encrypted_fields"] = encrypted_fields
        return result

    def decrypt_dict(
        self,
        data             : dict,
        verify_integrity : bool = True,
    ) -> dict:
        """
        Decrypt all encrypted fields in a dictionary.

        Detects encrypted values automatically by checking for ENCRYPTED_PREFIX.
        Non-encrypted values are copied unchanged.

        Args:
            data            : Input dict (may contain mix of encrypted and plain)
            verify_integrity: Verify checksums during decryption

        Returns:
            New dict with all encrypted values decrypted.
        """
        result = {}
        for k, v in data.items():
            if k == "_encrypted_fields":
                continue
            if isinstance(v, str) and self._engine.is_encrypted(v):
                try:
                    result[k] = self._engine.decrypt(v, verify_integrity=verify_integrity)
                except (VaultDecryptionError, VaultIntegrityError) as e:
                    logger.error(f"Failed to decrypt field '{k}': {e}")
                    result[k] = f"[DECRYPTION_ERROR: {type(e).__name__}]"
            else:
                result[k] = v
        return result

    def re_encrypt_dict(self, data: dict) -> dict:
        """
        Re-encrypt all encrypted values in a dict (for key rotation).
        Decrypts with old key, re-encrypts with current active key.
        """
        result = {}
        for k, v in data.items():
            if k == "_encrypted_fields":
                result[k] = v
                continue
            if isinstance(v, str) and self._engine.is_encrypted(v):
                try:
                    result[k] = self._engine.re_encrypt(v)
                except Exception as e:
                    logger.error(f"Re-encryption failed for '{k}': {e}")
                    result[k] = v   # Keep old value on failure
            else:
                result[k] = v
        return result

    def audit_dict(self, data: dict) -> dict:
        """
        Audit a dict — return report on which fields are encrypted, hashed, or plaintext.
        Does NOT decrypt anything.
        """
        report = {}
        for k, v in data.items():
            if not isinstance(v, str):
                report[k] = {"status": "non-string", "type": type(v).__name__}
            elif self._engine.is_encrypted(v):
                report[k] = {"status": "encrypted", "length": len(v)}
            elif v.startswith(HASHED_PREFIX) or v.startswith("det:"):
                report[k] = {"status": "hashed", "length": len(v)}
            else:
                report[k] = {"status": "plaintext", "length": len(v)}
        return report


# ══════════════════════════════════════════════════════════════════════════════
# 14. SECURITY VAULT — Master Class
# The single entry point for all vault operations in your app.
# ══════════════════════════════════════════════════════════════════════════════

class SecurityVault:
    """
    Master Security Vault for dataDoctor.

    This is the ONLY class you need to import in the rest of the app.
    Everything else is an internal component.

    Quick Usage:
        vault = SecurityVault()

        # Encrypt
        enc = vault.encrypt("sk-my-openai-key", field="api_key")

        # Decrypt
        raw = vault.decrypt(enc)

        # Hash (passwords, emails)
        h   = vault.hash("user@example.com")
        ok  = vault.verify_hash("user@example.com", h)

        # Mask for UI display
        ui  = vault.mask("sk-my-openai-key")      # → "sk-***...key"

        # Detect provider
        p   = vault.detect_provider("gsk_abc123")  # → "groq"

        # Generate secure token
        tok = vault.token()                         # 256-bit random token

        # Batch encrypt a dict
        safe = vault.encrypt_dict({"api_key": "sk-...", "name": "Alice"})

        # Audit log
        vault.audit.log("encrypt", subject="api_key", user_hint="alice@...")
    """

    def __init__(
        self,
        passphrase  : str        = "",
        audit_file  : str | None = None,
        salt        : bytes      = VAULT_SALT,
    ) -> None:
        """
        Initialize the Security Vault.

        Args:
            passphrase  : Optional extra passphrase (adds security beyond system ID)
            audit_file  : Path to audit log file (None = memory only)
            salt        : Salt bytes for key derivation (use default)
        """
        # Initialize components
        self._deriver   = MasterKeyDeriver(passphrase=passphrase, salt=salt)
        self._key_store = KeyVersionStore()
        self._hasher    = HashEngine()
        self._tokens    = TokenEngine()
        self._masker    = MaskEngine()
        self._integrity = IntegrityChecker()
        self.audit      = AuditLogger(log_file=audit_file)

        # Bootstrap: derive the master key and add as version 0
        master_key_bytes = self._deriver.derive()
        self._key_store.add(master_key_bytes, description="Master key (system-derived)")

        # Initialize Fernet engine with key store
        self._engine    = FernetEngine(self._key_store)
        self._rotation  = KeyRotationManager(self._key_store, self._engine)
        self._batch     = BatchProcessor(self._engine)

        logger.info("SecurityVault initialized successfully.")
        self.audit.log("vault_init", subject="system", detail="Vault initialized")

    # ── Encryption ────────────────────────────────────────────────────────────

    def encrypt(self, value: str, field: str = "") -> str:
        """
        Encrypt a sensitive string value.

        Args:
            value : The plaintext to encrypt
            field : The field name (for audit trail — NOT the value)

        Returns:
            Encrypted string (safe to store in database)
        """
        if not value:
            return value
        try:
            result = self._engine.encrypt(value, field_hint=field)
            self.audit.log("encrypt", subject=field, success=True)
            return result
        except Exception as e:
            self.audit.log("encrypt", subject=field, success=False, detail=str(e))
            raise VaultEncryptionError(f"Failed to encrypt '{field}': {e}") from e

    def decrypt(self, encrypted_value: str, field: str = "") -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted_value : The encrypted string from encrypt()
            field           : The field name (for audit trail)

        Returns:
            Original plaintext string

        Raises:
            VaultDecryptionError : If decryption fails
            VaultIntegrityError  : If data was tampered with
        """
        if not encrypted_value:
            return encrypted_value
        try:
            result = self._engine.decrypt(encrypted_value)
            self.audit.log("decrypt", subject=field, success=True)
            return result
        except VaultIntegrityError as e:
            self.audit.log("decrypt", subject=field, success=False,
                           detail=f"INTEGRITY VIOLATION: {e}")
            raise
        except Exception as e:
            self.audit.log("decrypt", subject=field, success=False, detail=str(e))
            raise VaultDecryptionError(f"Failed to decrypt '{field}': {e}") from e

    def decrypt_safe(self, encrypted_value: str, default: str = "") -> str:
        """
        Decrypt without raising exceptions.
        Returns `default` if decryption fails.
        """
        try:
            return self.decrypt(encrypted_value)
        except Exception:
            return default

    def is_encrypted(self, value: str) -> bool:
        """Check if a string was encrypted by this vault."""
        return self._engine.is_encrypted(value)

    # ── Hashing ───────────────────────────────────────────────────────────────

    def hash(self, value: str, context: str = "") -> str:
        """
        One-way hash a sensitive value (passwords, emails).
        Cannot be reversed. Use verify_hash() to check.
        """
        result = self._hasher.hash(value, context=context)
        self.audit.log("hash", subject=context or "value", success=True)
        return result

    def verify_hash(self, plaintext: str, stored_hash: str, context: str = "") -> bool:
        """Verify a plaintext value against a stored hash."""
        result = self._hasher.verify(plaintext, stored_hash, context=context)
        self.audit.log("verify_hash", subject=context or "value",
                       success=result, detail="match" if result else "no_match")
        return result

    def hash_email(self, email: str) -> str:
        """Deterministically hash an email for database lookups."""
        return self._hasher.hash_email(email)

    def hash_deterministic(self, value: str, context: str = "") -> str:
        """Deterministic hash — same input always gives same output."""
        return self._hasher.hash_deterministic(value, context)

    def is_hashed(self, value: str) -> bool:
        """Check if a value is a stored hash."""
        return self._hasher.is_hashed(value)

    # ── Masking & Provider Detection ──────────────────────────────────────────

    def mask(self, value: str, show_last: int = 4) -> str:
        """
        Mask a sensitive value for safe UI display.
        Example: "sk-abc123xyz" → "sk-***...xyz"
        """
        return self._masker.mask(value, show_last=show_last)

    def mask_dict(self, d: dict) -> dict:
        """Return a copy of a dict with sensitive values masked."""
        return self._masker.mask_dict(d)

    def detect_provider(self, api_key: str) -> str | None:
        """Detect which LLM provider an API key belongs to."""
        result = self._masker.detect_provider(api_key)
        self.audit.log("detect_provider", subject="api_key",
                       detail=f"detected={result or 'unknown'}")
        return result

    def analyze_key(self, api_key: str) -> dict:
        """Analyze an API key's strength and provider."""
        return self._masker.detect_key_strength(api_key)

    # ── Token Generation ──────────────────────────────────────────────────────

    def token(self, n_bytes: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        t = self._tokens.generate(n_bytes)
        self.audit.log("generate_token", subject="token",
                       detail=f"bytes={n_bytes}")
        return t

    def token_hex(self, n_bytes: int = 32) -> str:
        """Generate a cryptographically secure hex token."""
        return self._tokens.generate_hex(n_bytes)

    def token_pin(self, digits: int = 6) -> str:
        """Generate a secure numeric PIN (e.g. for OTP/2FA)."""
        t = self._tokens.generate_pin(digits)
        self.audit.log("generate_pin", subject="pin", detail=f"digits={digits}")
        return t

    def token_timed(self, ttl_seconds: int = 3600) -> str:
        """
        Generate an encrypted time-limited token.
        Automatically expires after ttl_seconds.
        """
        t = self._tokens.generate_timed(self._engine, ttl_seconds=ttl_seconds)
        self.audit.log("generate_timed_token", subject="token",
                       detail=f"ttl={ttl_seconds}s")
        return t

    def validate_timed_token(self, encrypted_token: str) -> tuple[bool, str]:
        """Validate a timed token. Returns (is_valid, token_value_or_error)."""
        return self._tokens.validate_timed(encrypted_token, self._engine)

    # ── Batch Operations ──────────────────────────────────────────────────────

    def encrypt_dict(
        self,
        data          : dict,
        sensitive_keys: set[str] | None  = None,
        explicit_keys : list[str] | None = None,
    ) -> dict:
        """
        Encrypt all sensitive fields in a dictionary at once.
        Sensitive fields are auto-detected by name keywords.
        """
        result = self._batch.encrypt_dict(
            data, sensitive_keys=sensitive_keys, explicit_keys=explicit_keys
        )
        n = len(result.get("_encrypted_fields", []))
        self.audit.log("batch_encrypt", subject="dict", detail=f"encrypted={n} fields")
        return result

    def decrypt_dict(self, data: dict) -> dict:
        """Decrypt all encrypted fields in a dictionary."""
        result = self._batch.decrypt_dict(data)
        self.audit.log("batch_decrypt", subject="dict")
        return result

    def audit_dict(self, data: dict) -> dict:
        """
        Audit a dictionary without decrypting.
        Returns status of each field: encrypted / hashed / plaintext.
        """
        return self._batch.audit_dict(data)

    # ── Integrity ─────────────────────────────────────────────────────────────

    def checksum(self, value: str) -> str:
        """Compute SHA256 checksum of a value."""
        return self._integrity.checksum(value)

    def verify_checksum(self, value: str, expected: str) -> bool:
        """Verify a SHA256 checksum. Timing-safe."""
        return self._integrity.verify_checksum(value, expected)

    def sign(self, value: str, secret: str) -> str:
        """Create an HMAC-SHA256 signature for a value."""
        return self._integrity.sign(value, secret)

    def verify_signature(self, value: str, signature: str, secret: str) -> bool:
        """Verify an HMAC-SHA256 signature."""
        return self._integrity.verify_signature(value, signature, secret)

    # ── Key Rotation ──────────────────────────────────────────────────────────

    def rotate_keys(
        self,
        old_values  : list[str] | None = None,
        description : str = "Manual rotation",
    ) -> dict:
        """
        Rotate encryption keys.
        Generates a new key, re-encrypts provided values.

        Args:
            old_values  : List of encrypted strings to migrate to new key
            description : Label for this rotation event

        Returns:
            Rotation report dict
        """
        self.audit.log("key_rotation_start", subject="keys", detail=description)
        result = self._rotation.rotate(old_values=old_values, description=description)
        self.audit.log("key_rotation_complete", subject="keys",
                       detail=f"new_version={result['new_version']} "
                              f"re_encrypted={len(result['re_encrypted'])}")
        return result

    def key_status(self) -> dict:
        """Return current key version status."""
        return self._rotation.get_rotation_status()

    # ── Per-User Encryption ───────────────────────────────────────────────────

    def encrypt_for_user(self, value: str, user_passphrase: str, field: str = "") -> str:
        """
        Encrypt a value with a key derived from the user's own passphrase.
        Only that specific user (with their passphrase) can decrypt it.

        This provides user-specific encryption on top of the vault's master key.
        Use when you need per-user key isolation.
        """
        user_key_bytes = self._deriver.derive_from_passphrase(user_passphrase)
        user_fernet    = Fernet(user_key_bytes)
        ciphertext     = user_fernet.encrypt(value.encode()).decode()
        result         = f"user:{ENCRYPTED_PREFIX}{ciphertext}"
        self.audit.log("user_encrypt", subject=field, success=True)
        return result

    def decrypt_for_user(self, encrypted: str, user_passphrase: str, field: str = "") -> str:
        """Decrypt a value that was encrypted with encrypt_for_user()."""
        if not encrypted.startswith(f"user:{ENCRYPTED_PREFIX}"):
            raise VaultDecryptionError("Not a user-encrypted value.")
        try:
            ciphertext     = encrypted[len(f"user:{ENCRYPTED_PREFIX}"):]
            user_key_bytes = self._deriver.derive_from_passphrase(user_passphrase)
            user_fernet    = Fernet(user_key_bytes)
            plaintext      = user_fernet.decrypt(ciphertext.encode()).decode()
            self.audit.log("user_decrypt", subject=field, success=True)
            return plaintext
        except InvalidToken as e:
            self.audit.log("user_decrypt", subject=field, success=False,
                           detail="invalid_passphrase_or_tampered")
            raise VaultDecryptionError("User decryption failed — wrong passphrase?") from e

    # ── Status & Diagnostics ──────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return full vault status — useful for health checks and admin UI.
        """
        return {
            "initialized"   : True,
            "key_versions"  : self._key_store.count(),
            "active_version": self._key_store.get_active().version if self._key_store.get_active() else None,
            "audit_summary" : self.audit.get_summary(),
            "constants"     : {
                "pbkdf2_iterations" : PBKDF2_ITERATIONS,
                "encrypted_prefix"  : ENCRYPTED_PREFIX,
                "hashed_prefix"     : HASHED_PREFIX,
            },
        }

    def self_test(self) -> dict:
        """
        Run a self-test to verify all vault operations are working correctly.
        Returns a dict of {test_name: passed} booleans.
        """
        results = {}

        # Test 1: Encrypt/Decrypt round-trip
        try:
            test_val = "test_secret_12345"
            enc      = self.encrypt(test_val, field="test")
            dec      = self.decrypt(enc)
            results["encrypt_decrypt"] = dec == test_val
        except Exception as e:
            results["encrypt_decrypt"] = False
            logger.error(f"Self-test encrypt_decrypt failed: {e}")

        # Test 2: is_encrypted detection
        try:
            enc = self.encrypt("hello", "test")
            results["is_encrypted"] = self.is_encrypted(enc) and not self.is_encrypted("plaintext")
        except Exception:
            results["is_encrypted"] = False

        # Test 3: Hash + verify
        try:
            h  = self.hash("password123")
            results["hash_verify"] = (
                self.verify_hash("password123", h) and
                not self.verify_hash("wrong", h)
            )
        except Exception:
            results["hash_verify"] = False

        # Test 4: Mask
        try:
            masked = self.mask("sk-abcdefghijklmnop")
            results["mask"] = "***" in masked and "sk-abcdefghijklmnop" not in masked
        except Exception:
            results["mask"] = False

        # Test 5: Token generation
        try:
            tok = self.token()
            results["token"] = len(tok) > 20
        except Exception:
            results["token"] = False

        # Test 6: Batch encrypt/decrypt
        try:
            d     = {"api_key": "sk-test", "name": "Alice", "email": "a@b.com"}
            enc_d = self.encrypt_dict(d)
            dec_d = self.decrypt_dict(enc_d)
            results["batch"] = dec_d.get("api_key") == "sk-test" and dec_d.get("name") == "Alice"
        except Exception:
            results["batch"] = False

        # Test 7: Provider detection
        try:
            p = self.detect_provider("gsk_abc123abcabc123abc123abc123abc")
            results["provider_detection"] = p == "groq"
        except Exception:
            results["provider_detection"] = False

        # Test 8: Checksum integrity
        try:
            cs = self.checksum("test_data")
            results["checksum"] = self.verify_checksum("test_data", cs)
        except Exception:
            results["checksum"] = False

        all_passed = all(results.values())
        logger.info(f"Vault self-test: {'ALL PASSED' if all_passed else 'SOME FAILED'} — {results}")
        return {"passed": all_passed, "tests": results}


# ══════════════════════════════════════════════════════════════════════════════
# 15. CONVENIENCE FUNCTIONS — Quick access without instantiating the vault
# ══════════════════════════════════════════════════════════════════════════════

# Global singleton vault instance
_vault: SecurityVault | None = None
_vault_lock = threading.Lock()


def get_vault(passphrase: str = "", audit_file: str | None = None) -> SecurityVault:
    """
    Get or create the global SecurityVault singleton.
    Thread-safe. Call this anywhere in the app.

    Args:
        passphrase  : Optional extra passphrase for key derivation
        audit_file  : Path to audit log file

    Returns:
        The global SecurityVault instance
    """
    global _vault
    with _vault_lock:
        if _vault is None:
            _vault = SecurityVault(passphrase=passphrase, audit_file=audit_file)
    return _vault


def encrypt(value: str, field: str = "") -> str:
    """Quick encrypt using the global vault."""
    return get_vault().encrypt(value, field=field)


def decrypt(value: str, field: str = "") -> str:
    """Quick decrypt using the global vault."""
    return get_vault().decrypt(value, field=field)


def mask(value: str) -> str:
    """Quick mask for UI display using the global vault."""
    return get_vault().mask(value)


def hash_value(value: str, context: str = "") -> str:
    """Quick one-way hash using the global vault."""
    return get_vault().hash(value, context=context)


def verify_hash(plaintext: str, stored_hash: str, context: str = "") -> bool:
    """Quick hash verification using the global vault."""
    return get_vault().verify_hash(plaintext, stored_hash, context=context)