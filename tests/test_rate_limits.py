"""1.4: FailureCounter unit tests + route-level limits (login, refresh, google)."""
import pytest
from fastapi import HTTPException

from src.login_guard import FailureCounter

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
GOOGLE = "/api/v1/auth/google/exchange"


# ---------- FailureCounter (unit, fake clock) ----------

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_blocks_after_max_failures():
    c = FailureCounter(3, 60, clock=Clock())
    for _ in range(2):
        c.add("a")
    assert not c.is_blocked("a")
    c.add("a")
    assert c.is_blocked("a")


def test_keys_are_independent():
    c = FailureCounter(2, 60, clock=Clock())
    c.add("a")
    c.add("a")
    assert c.is_blocked("a")
    assert not c.is_blocked("b")


def test_window_expires():
    clk = Clock()
    c = FailureCounter(2, 60, clock=clk)
    c.add("a")
    c.add("a")
    assert c.is_blocked("a")
    clk.t += 61
    assert not c.is_blocked("a")


def test_clear_resets():
    c = FailureCounter(2, 60, clock=Clock())
    c.add("a")
    c.add("a")
    c.clear("a")
    assert not c.is_blocked("a")
    c.clear("never-seen")  # must not raise


def test_stale_keys_are_pruned():
    clk = Clock()
    c = FailureCounter(5, 60, clock=clk, prune_above=10)
    for i in range(50):
        c.add(f"k{i}")
    clk.t += 120
    c.add("fresh")  # triggers prune (len above threshold)
    assert len(c) == 1


# ---------- routes ----------

def _register(client, username, password, email=None):
    return client.post(REGISTER, json={
        "username": username, "email": email or f"{username}@example.com", "password": password,
    })


def _login(client, who, password):
    return client.post(LOGIN, json={"username": who, "password": password})


def test_account_locks_after_8_failures_even_for_correct_password(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "locky", strong_password).status_code == 201
    for _ in range(8):
        assert _login(client, "locky", "WrongPass999").status_code == 401
    r = _login(client, "locky", strong_password)
    assert r.status_code == 429, r.text


def test_success_clears_failure_count(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "resetty", strong_password).status_code == 201
    for _ in range(5):
        assert _login(client, "resetty", "WrongPass999").status_code == 401
    assert _login(client, "resetty", strong_password).status_code == 200
    for _ in range(5):  # would be a 429 if the 5 earlier failures still counted
        assert _login(client, "resetty", "WrongPass999").status_code == 401


def test_lockout_is_case_insensitive(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "casey", strong_password).status_code == 201
    for who in ["casey", "CASEY", "Casey", "cAsEy"] * 2:
        assert _login(client, who, "WrongPass999").status_code == 401
    assert _login(client, "casey", strong_password).status_code == 429


def test_unknown_username_is_also_counted(api_client, strong_password):
    """Same behaviour for real and fake accounts: no account-existence leak."""
    client, _ = api_client
    for _ in range(8):
        assert _login(client, "ghost_user", "WrongPass999").status_code == 401
    assert _login(client, "ghost_user", "WrongPass999").status_code == 429


def test_login_ip_limit_after_30_attempts(api_client):
    client, _ = api_client
    codes = [_login(client, f"nobody{i}", "WrongPass999").status_code for i in range(31)]
    assert 429 not in codes[:30]
    assert codes[30] == 429


def test_refresh_is_rate_limited(api_client):
    client, _ = api_client
    codes = [client.post(REFRESH, json={"refresh_token": "x" * 40}).status_code for _ in range(31)]
    assert 429 not in codes[:30]
    assert codes[30] == 429


def test_google_exchange_is_rate_limited(api_client, monkeypatch):
    client, api_module = api_client

    async def fake_exchange(code):
        raise HTTPException(status_code=400, detail="bad code")

    monkeypatch.setattr(api_module, "_exchange_google_code", fake_exchange)
    codes = [client.post(GOOGLE, json={"code": "abc"}).status_code for _ in range(21)]
    assert 429 not in codes[:20]
    assert codes[20] == 429