"""Tests for src.ws_auth (first-message WebSocket auth). No real sockets."""
import asyncio
import json

import pytest

from src.ws_auth import (
    MAX_AUTH_MESSAGE_CHARS,
    WS_CLOSE_UNAUTHORIZED,
    ws_authenticate,
)

HANG = object()


class FakeWS:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.accepted = 0
        self.closed = []  # (code, reason)

    async def accept(self):
        self.accepted += 1

    async def receive_text(self):
        item = self.incoming.pop(0)
        if item is HANG:
            await asyncio.sleep(10)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self, code=1000, reason=""):
        self.closed.append((code, reason))


def good_decode(token):
    if token == "good":
        return {"sub": "user-1"}
    raise ValueError("bad token")


def run(ws, decode=good_decode, timeout=0.5):
    return asyncio.run(ws_authenticate(ws, decode, timeout=timeout))


def auth_msg(token="good", **extra):
    return json.dumps({"type": "auth", "token": token, **extra})


def assert_rejected(ws, result):
    assert result is None
    assert len(ws.closed) == 1
    assert ws.closed[0][0] == WS_CLOSE_UNAUTHORIZED


def test_valid_token_returns_payload():
    ws = FakeWS([auth_msg()])
    assert run(ws) == {"sub": "user-1"}
    assert ws.closed == []


def test_accept_called_exactly_once():
    ws = FakeWS([auth_msg()])
    run(ws)
    assert ws.accepted == 1


def test_invalid_token_closes_4001():
    ws = FakeWS([auth_msg("nope")])
    assert_rejected(ws, run(ws))


def test_missing_token_field():
    ws = FakeWS([json.dumps({"type": "auth"})])
    assert_rejected(ws, run(ws))


@pytest.mark.parametrize("tok", ["", None, 123, ["good"], {"a": 1}])
def test_non_string_or_empty_token(tok):
    ws = FakeWS([json.dumps({"type": "auth", "token": tok})])
    assert_rejected(ws, run(ws))


def test_wrong_message_type():
    ws = FakeWS([json.dumps({"type": "ping", "token": "good"})])
    assert_rejected(ws, run(ws))


def test_first_message_is_chat_payload_not_auth():
    ws = FakeWS([json.dumps({"message": "hi", "job_id": "x"})])
    assert_rejected(ws, run(ws))


@pytest.mark.parametrize("raw", ["not json", "", "[1,2]", "42", "null", '"str"'])
def test_garbage_first_message(raw):
    ws = FakeWS([raw])
    assert_rejected(ws, run(ws))


def test_oversized_message_rejected_before_decode():
    called = []

    def decode(t):
        called.append(t)
        return {"sub": "u"}

    big = auth_msg("x" * (MAX_AUTH_MESSAGE_CHARS + 10))
    ws = FakeWS([big])
    assert_rejected(ws, run(ws, decode=decode))
    assert called == []


def test_timeout_closes_4001():
    ws = FakeWS([HANG])
    assert_rejected(ws, run(ws, timeout=0.05))


def test_disconnect_before_auth_returns_none_without_close():
    ws = FakeWS([RuntimeError("disconnected")])
    assert run(ws) is None
    assert ws.closed == []


def test_payload_without_sub_rejected():
    ws = FakeWS([auth_msg()])
    assert_rejected(ws, run(ws, decode=lambda t: {"role": "x"}))


def test_non_dict_payload_rejected():
    ws = FakeWS([auth_msg()])
    assert_rejected(ws, run(ws, decode=lambda t: "user-1"))


def test_decoder_raising_any_exception_is_rejection():
    def boom(t):
        raise KeyError("x")

    ws = FakeWS([auth_msg()])
    assert_rejected(ws, run(ws, decode=boom))


def test_close_failure_does_not_raise():
    class BadClose(FakeWS):
        async def close(self, code=1000, reason=""):
            raise RuntimeError("already closed")

    ws = BadClose([auth_msg("nope")])
    assert run(ws) is None


def test_token_is_not_logged_or_returned_in_close_reason():
    ws = FakeWS([auth_msg("secret-token-123")])
    run(ws)
    assert "secret-token-123" not in ws.closed[0][1]