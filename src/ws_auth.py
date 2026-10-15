"""First-message WebSocket authentication.

The client connects WITHOUT a token in the URL, then sends one message:

    {"type": "auth", "token": "<access_token>"}

The server accepts the socket, waits a short time for that message, validates
the token, and returns the decoded payload. On any failure the socket is
closed with code 4001 and None is returned.

This keeps JWTs out of URLs (browser history, proxy logs, referrers).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

WS_AUTH_TIMEOUT_S = 5.0
WS_CLOSE_UNAUTHORIZED = 4001
MAX_AUTH_MESSAGE_CHARS = 8192


async def _close(websocket: Any, reason: str) -> None:
    try:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason=reason)
    except Exception:
        pass  # already closed / client gone


async def ws_authenticate(
    websocket: Any,
    decode: Callable[[str], dict],
    timeout: float = WS_AUTH_TIMEOUT_S,
) -> Optional[dict]:
    """Accept `websocket`, read the auth message, return the payload or None.

    `decode` is the token decoder (e.g. decode_access_token); any exception it
    raises counts as an invalid token. The payload must contain a `sub`.
    """
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
    except asyncio.TimeoutError:
        await _close(websocket, "Auth timeout")
        return None
    except Exception:
        return None  # client disconnected before authenticating

    if not isinstance(raw, str) or len(raw) > MAX_AUTH_MESSAGE_CHARS:
        await _close(websocket, "Invalid auth message")
        return None

    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        await _close(websocket, "Invalid auth message")
        return None

    if not isinstance(msg, dict) or msg.get("type") != "auth":
        await _close(websocket, "Missing token")
        return None

    token = msg.get("token")
    if not isinstance(token, str) or not token:
        await _close(websocket, "Missing token")
        return None

    try:
        payload = decode(token)
    except Exception:
        await _close(websocket, "Invalid token")
        return None

    if not isinstance(payload, dict) or not payload.get("sub"):
        await _close(websocket, "Invalid token")
        return None

    return payload