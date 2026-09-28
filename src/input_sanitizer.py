"""
Nydra - src/input_sanitizer.py
Input sanitization for the security hardening layer (module 2 of 4).

Scope (deliberately narrow - only what Nydra is actually exposed to):

    validate_url          SSRF guard for user-supplied URLs (custom LLM base
                          URL, /config/llm/models base_url). Resolves DNS and
                          checks every resulting IP, handles legacy IPv4
                          forms (decimal/octal/hex), IPv4-mapped IPv6, NAT64,
                          6to4, cloud metadata endpoints. Fails closed.
    sanitize_filename     Upload filenames: traversal, null bytes, Windows
                          reserved names, extension allowlist, length.
    sanitize_text         Chat/settings text: unicode normalization, control
                          chars, trojan-source bidi overrides, length limits.
                          Preserves legitimate Arabic/Persian shaping chars.
    neutralize_csv_cell / neutralize_dataframe
                          CSV/Excel formula injection on exported data.

Not covered here on purpose:
    * SQL injection - SQLAlchemy already parameterizes queries.
    * Prompt injection - can only be flagged, not stripped -> attack_detector.

Design rules:
    * Never raise on bad input; return a SanitizeResult (callers decide).
      Use `result.unwrap()` when an exception is more convenient.
    * Reasons shown to clients never echo internal IPs (DNS-recon oracle);
      details go in `result.meta` for server-side logging only.
    * Stdlib only, so it is importable and testable anywhere.

Known limitation (documented, not solvable here): validate_url checks the
DNS answer at validation time; an HTTP client re-resolves later
(DNS rebinding / TOCTOU). Mitigations for callers: keep
follow_redirects=False, and optionally connect to `meta["resolved_ips"][0]`.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit

__all__ = [
    "SanitizeResult",
    "InputRejectedError",
    "validate_url",
    "sanitize_filename",
    "sanitize_text",
    "neutralize_csv_cell",
    "neutralize_dataframe",
    "ENV_ALLOW_PRIVATE_URLS",
]

ENV_ALLOW_PRIVATE_URLS = "NYDRA_ALLOW_PRIVATE_LLM_URLS"


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────
class InputRejectedError(ValueError):
    """Raised by SanitizeResult.unwrap() when the input was rejected."""

    def __init__(self, result: "SanitizeResult") -> None:
        super().__init__("; ".join(result.reasons) or "input rejected")
        self.result = result


@dataclass
class SanitizeResult:
    value: Any = None
    changed: bool = False
    rejected: bool = False
    reasons: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.rejected

    def unwrap(self) -> Any:
        if self.rejected:
            raise InputRejectedError(self)
        return self.value


def _reject(reason: str, **meta: Any) -> SanitizeResult:
    return SanitizeResult(value=None, rejected=True, reasons=[reason], meta=dict(meta))


# ─────────────────────────────────────────────────────────────────────────────
# FILENAMES
# ─────────────────────────────────────────────────────────────────────────────
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
    f"LPT{i}" for i in range(1, 10)
}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_FILENAME_BAD_RE = re.compile(r'[<>:"|?*]')
_PATH_SEP_RE = re.compile(r"[\\/]")


def sanitize_filename(
    name: Any,
    allowed_extensions: Optional[Iterable[str]] = None,
    max_length: int = 150,
) -> SanitizeResult:
    """Return a safe basename, or a rejection. Never returns path separators."""
    if not isinstance(name, str):
        return _reject("filename must be a string")
    if "\x00" in name:
        return _reject("filename contains a null byte")

    reasons: List[str] = []
    cleaned = unicodedata.normalize("NFKC", name)  # folds full-width '／' etc.

    if _PATH_SEP_RE.search(cleaned):
        cleaned = _PATH_SEP_RE.split(cleaned)[-1]
        reasons.append("path components removed")

    stripped = _CONTROL_RE.sub("", cleaned)
    if stripped != cleaned:
        reasons.append("control characters removed")
    cleaned = stripped

    replaced = _FILENAME_BAD_RE.sub("_", cleaned)
    if replaced != cleaned:
        reasons.append("unsafe characters replaced")
    cleaned = replaced

    trimmed = cleaned.strip(" .")  # also drops leading dots -> no hidden files
    if trimmed != cleaned:
        reasons.append("leading/trailing dots or spaces removed")
    cleaned = trimmed

    if not cleaned:
        return _reject("filename is empty after sanitizing")

    if cleaned.split(".", 1)[0].strip().upper() in _WINDOWS_RESERVED:
        return _reject("filename uses a reserved system name")

    if allowed_extensions is not None:
        allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        ext = cleaned.rsplit(".", 1)[-1].lower() if "." in cleaned else ""
        if ext not in allowed:
            return _reject("file type not allowed")

    if len(cleaned) > max_length:
        stem, dot, ext = cleaned.rpartition(".")
        if dot and len(ext) + 1 < max_length:
            cleaned = stem[: max_length - len(ext) - 1].rstrip(" .") + "." + ext
        else:
            cleaned = cleaned[:max_length].rstrip(" .")
        reasons.append("filename truncated")
        if not cleaned or cleaned.startswith("."):
            return _reject("filename is empty after sanitizing")

    return SanitizeResult(value=cleaned, changed=cleaned != name, reasons=reasons)


# ─────────────────────────────────────────────────────────────────────────────
# FREE TEXT
# ─────────────────────────────────────────────────────────────────────────────
# Stripped: zero-width space, word joiner, BOM, and the bidi
# embedding/override/isolate controls used in "trojan source" spoofing.
# Deliberately KEPT: U+200C/U+200D (ZWNJ/ZWJ) and U+200E/U+200F (LRM/RLM) -
# they are required for correct Arabic/Persian/Indic text.
_INVISIBLE_STRIP = {
    ord(c): None
    for c in "\u200b\u2060\ufeff\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
}
_CTRL_KEEP_NEWLINES_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_CTRL_ALL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def sanitize_text(
    text: Any,
    *,
    max_length: int = 10_000,
    allow_newlines: bool = True,
    allow_empty: bool = False,
    on_too_long: str = "reject",
    normalization: Optional[str] = "NFC",
) -> SanitizeResult:
    """Clean user text. `on_too_long` is 'reject' (default) or 'truncate'."""
    if not isinstance(text, str):
        return _reject("text must be a string")

    reasons: List[str] = []
    out = text

    try:
        out.encode("utf-8")
    except UnicodeEncodeError:
        out = out.encode("utf-8", "ignore").decode("utf-8")
        reasons.append("invalid unicode removed")

    if normalization:
        normalized = unicodedata.normalize(normalization, out)
        if normalized != out:
            reasons.append(f"unicode normalized ({normalization})")
            out = normalized

    if "\r\n" in out:
        out = out.replace("\r\n", "\n")
        reasons.append("line endings normalized")

    if allow_newlines:
        cleaned = _CTRL_KEEP_NEWLINES_RE.sub("", out)
    else:
        cleaned = _CTRL_ALL_RE.sub("", re.sub(r"[\r\n\t]+", " ", out))
    if cleaned != out:
        reasons.append("control characters removed")
    out = cleaned

    cleaned = out.translate(_INVISIBLE_STRIP)
    if cleaned != out:
        reasons.append("invisible or bidi-override characters removed")
    out = cleaned

    if len(out) > max_length:
        if on_too_long == "truncate":
            out = out[:max_length]
            reasons.append("text truncated")
        else:
            return _reject(f"text exceeds {max_length} characters")

    if not allow_empty and not out.strip():
        return _reject("text is empty")

    return SanitizeResult(value=out, changed=out != text, reasons=reasons)


# ─────────────────────────────────────────────────────────────────────────────
# URL / SSRF
# ─────────────────────────────────────────────────────────────────────────────
_METADATA_IPS = {
    ipaddress.ip_address(a)
    for a in (
        "169.254.169.254",  # AWS / GCP / Azure / OpenStack
        "169.254.170.2",    # AWS ECS task metadata
        "100.100.100.200",  # Alibaba Cloud
        "168.63.129.16",    # Azure wireserver
        "fd00:ec2::254",    # AWS IPv6 metadata
    )
}
_ALWAYS_BLOCKED_HOSTS = {
    "metadata",
    "metadata.google.internal",
    "instance-data",
    "instance-data.ec2.internal",
}
_LOCAL_HOSTS = {"localhost", "ip6-localhost", "ip6-loopback", "broadcasthost"}
_LOCAL_SUFFIXES = (
    ".localhost", ".local", ".internal", ".localdomain", ".lan", ".home.arpa", ".intranet",
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPAT = ipaddress.ip_network("::/96")
# Whitespace, control chars and backslashes are rejected BEFORE parsing:
# urlsplit silently strips tab/CR/LF, and clients disagree about '\'.
_URL_FORBIDDEN_RE = re.compile(r"[\s\x00-\x1f\x7f\\]")


def _env_allows_private() -> bool:
    return os.getenv(ENV_ALLOW_PRIVATE_URLS, "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_ip_literal(host: str) -> Optional[Any]:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:  # legacy IPv4 forms the OS resolver also accepts: 2130706433, 0x7f.1, 0177.0.0.1, 127.1
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def _unwrap_ip(ip: Any) -> Any:
    """Reduce IPv6 forms that embed an IPv4 address to that IPv4 address."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip in _NAT64 or ip in _IPV4_COMPAT:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if ip.sixtofour is not None:
            return ip.sixtofour
    return ip


def _ip_problem(ip: Any, allow_private: bool) -> Optional[str]:
    ip = _unwrap_ip(ip)
    if ip in _METADATA_IPS:
        return "cloud metadata address"
    if ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        return "multicast, reserved or unspecified address"
    if allow_private:
        return None
    if not ip.is_global:
        return "non-public address"
    return None


def _default_resolver(host: str) -> List[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def validate_url(
    url: Any,
    *,
    allowed_schemes: Sequence[str] = ("https", "http"),
    allow_private: Optional[bool] = None,
    resolver: Optional[Callable[[str], Iterable[str]]] = None,
    max_length: int = 2048,
) -> SanitizeResult:
    """
    SSRF-safe URL validation. Blocks private/loopback/link-local/CGNAT/
    reserved ranges by default; `allow_private=True` (or env
    NYDRA_ALLOW_PRIVATE_LLM_URLS=1) permits them for local models such as
    Ollama, but cloud metadata endpoints stay blocked either way.
    An explicit `allow_private` argument always beats the env variable.
    """
    if not isinstance(url, str):
        return _reject("URL must be a string")
    original = url
    url = url.strip()
    if not url:
        return _reject("URL is empty")
    if len(url) > max_length:
        return _reject("URL is too long")
    if _URL_FORBIDDEN_RE.search(url):
        return _reject("URL contains whitespace, control characters or backslashes")

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return _reject("URL is malformed")

    scheme = parts.scheme.lower()
    if scheme not in {s.lower() for s in allowed_schemes}:
        return _reject("URL scheme not allowed")
    if parts.username is not None or parts.password is not None:
        return _reject("URL must not contain credentials")

    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        return _reject("URL has no host")

    if allow_private is None:
        allow_private = _env_allows_private()

    if host in _ALWAYS_BLOCKED_HOSTS:
        return _reject("host is not allowed", host=host)
    if not allow_private and (host in _LOCAL_HOSTS or host.endswith(_LOCAL_SUFFIXES)):
        return _reject("private or local hosts are not allowed", host=host)

    literal = _parse_ip_literal(host)
    if literal is not None:
        ips = [literal]
    else:
        try:
            raw = list((resolver or _default_resolver)(host))
        except Exception:  # fail closed: any resolver problem rejects
            return _reject("host could not be resolved", host=host)
        if not raw:
            return _reject("host could not be resolved", host=host)
        ips = []
        for item in raw:
            try:
                ips.append(ipaddress.ip_address(str(item).split("%", 1)[0]))
            except ValueError:
                return _reject("host resolved to an invalid address", host=host)

    for ip in ips:
        problem = _ip_problem(ip, allow_private)
        if problem:
            # Generic text for the client; the address stays in meta for logs.
            return _reject(
                "private or non-public destinations are not allowed",
                host=host, blocked_ip=str(ip), problem=problem,
            )

    reasons = ["surrounding whitespace removed"] if url != original else []
    return SanitizeResult(
        value=url,
        changed=url != original,
        reasons=reasons,
        meta={"host": host, "port": port, "resolved_ips": [str(i) for i in ips]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV / EXCEL FORMULA INJECTION
# ─────────────────────────────────────────────────────────────────────────────
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def neutralize_csv_cell(value: Any) -> Any:
    """Prefix a single quote so spreadsheets treat risky strings as text."""
    if not isinstance(value, str) or not value:
        return value
    if value[0] in _FORMULA_PREFIXES:
        if value[0] in "+-" and _looks_numeric(value):
            return value  # legitimate signed number
        return "'" + value
    return value


def neutralize_dataframe(df: Any, *, headers: bool = True):
    """Return (sanitized_copy, n_cells_changed). The input is never mutated."""
    import pandas as pd  # lazy: keep the module stdlib-only at import time

    out = df.copy()
    changed = 0
    for col in list(out.columns):
        series = out[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            old = series.tolist()
            new = [neutralize_csv_cell(v) for v in old]
            changed += sum(1 for a, b in zip(old, new) if isinstance(a, str) and a != b)
            out[col] = new
    if headers:
        renamed = {c: neutralize_csv_cell(c) for c in out.columns}
        changed += sum(1 for c, n in renamed.items() if isinstance(c, str) and c != n)
        out = out.rename(columns=renamed)
    return out, changed