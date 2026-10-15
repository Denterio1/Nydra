"""Route-level tests: WebSocket auth is first-message only, never ?token=.

Positive path (valid auth -> job result) is covered by the terminal-job test
in test_api.py. Chat success needs an LLM, so only rejections are tested here.
"""
import pytest
from starlette.websockets import WebSocketDisconnect


def expect_close(ws, code):
    with pytest.raises(WebSocketDisconnect) as info:
        ws.receive_text()
    assert info.value.code == code


# ── /api/v1/ws/chat ──────────────────────────────────────────────────────────
def test_chat_query_token_alone_does_not_authenticate(registered_user):
    client, _, token, _ = registered_user
    with client.websocket_connect(f"/api/v1/ws/chat?token={token}") as ws:
        ws.send_json({"message": "hi"})  # first message is not an auth message
        expect_close(ws, 4001)


def test_chat_wrong_token_closed(registered_user):
    client = registered_user[0]
    with client.websocket_connect("/api/v1/ws/chat") as ws:
        ws.send_json({"type": "auth", "token": "not-a-real-token"})
        expect_close(ws, 4001)


def test_chat_auth_without_token_closed(registered_user):
    client = registered_user[0]
    with client.websocket_connect("/api/v1/ws/chat") as ws:
        ws.send_json({"type": "auth"})
        expect_close(ws, 4001)


def test_chat_non_json_first_message_closed(registered_user):
    client = registered_user[0]
    with client.websocket_connect("/api/v1/ws/chat") as ws:
        ws.send_text("hello")
        expect_close(ws, 4001)


# ── /api/v1/ws/{job_id} ──────────────────────────────────────────────────────
def test_job_query_token_alone_does_not_authenticate(registered_user):
    client, _, token, _ = registered_user
    with client.websocket_connect(f"/api/v1/ws/some-job?token={token}") as ws:
        ws.send_json({"type": "ping"})
        expect_close(ws, 4001)


def test_job_wrong_token_closed(registered_user):
    client = registered_user[0]
    with client.websocket_connect("/api/v1/ws/some-job") as ws:
        ws.send_json({"type": "auth", "token": "not-a-real-token"})
        expect_close(ws, 4001)


def test_job_valid_auth_unknown_job_closes_4004(registered_user):
    client, _, token, _ = registered_user
    with client.websocket_connect("/api/v1/ws/does-not-exist") as ws:
        ws.send_json({"type": "auth", "token": token})
        expect_close(ws, 4004)