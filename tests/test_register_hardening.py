"""Register/login hardening: case-insensitive identity + hostile input.

Uses the shared fixtures from conftest.py (api_client, strong_password).
Each test keeps register/login calls few so auth_limiter is never the cause
of a failure; 422 validation errors never reach the handler.
"""
import pytest

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"


def _register(client, username, email, password):
    return client.post(REGISTER, json={"username": username, "email": email, "password": password})


def _login(client, who, password):
    return client.post(LOGIN, json={"username": who, "password": password})


# ---------- case-insensitive identity ----------

def test_username_case_variant_is_duplicate(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "Alice", "alice@example.com", strong_password).status_code == 201
    r = _register(client, "alice", "other@example.com", strong_password)
    assert r.status_code == 409


def test_email_case_variant_is_duplicate(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "bob_one", "Bob@Example.com", strong_password).status_code == 201
    r = _register(client, "bob_two", "bob@example.com", strong_password)
    assert r.status_code == 409


def test_email_is_stored_lowercase(api_client, strong_password):
    client, _ = api_client
    r = _register(client, "carol", "CaRoL@Example.com", strong_password)
    assert r.status_code == 201
    assert r.json()["email"] == "carol@example.com"


def test_username_keeps_typed_case_for_display(api_client, strong_password):
    client, _ = api_client
    r = _register(client, "DaveSmith", "dave@example.com", strong_password)
    assert r.status_code == 201
    assert r.json()["username"] == "DaveSmith"


def test_login_with_different_username_case(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "Erin", "erin@example.com", strong_password).status_code == 201
    r = _login(client, "ERIN", strong_password)
    assert r.status_code == 200, r.text
    assert "access_token" in r.json()


def test_login_with_email_in_different_case(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "frank", "frank@example.com", strong_password).status_code == 201
    r = _login(client, "FRANK@EXAMPLE.COM", strong_password)
    assert r.status_code == 200, r.text


def test_wrong_password_still_rejected_with_different_case(api_client, strong_password):
    client, _ = api_client
    assert _register(client, "grace", "grace@example.com", strong_password).status_code == 201
    r = _login(client, "GRACE", "WrongPass999")
    assert r.status_code == 401


# ---------- hostile usernames ----------

@pytest.mark.parametrize("bad", [
    "ali\nce",
    "ali\r\nce",
    "alice\x00",
    "<b>alice</b>",
    "<script>alert(1)</script>",
    "ali ce",
    "ali;ce",
    "ali'ce",
    "al",                # too short
    "a" * 51,            # too long
    "x" * 10_000,
])
def test_hostile_username_rejected(api_client, strong_password, bad):
    client, _ = api_client
    r = _register(client, bad, "hostile@example.com", strong_password)
    assert r.status_code == 422, f"{bad!r} -> {r.status_code}"


def test_hostile_email_rejected(api_client, strong_password):
    client, _ = api_client
    for bad in ("not-an-email", "a@b", "<x>@example.com", "a" * 400 + "@example.com"):
        r = _register(client, "hostile_mail", bad, strong_password)
        assert r.status_code == 422, f"{bad[:30]!r} -> {r.status_code}"


# ---------- oversized passwords / login fields ----------

def test_register_rejects_10kb_password(api_client):
    client, _ = api_client
    r = _register(client, "bigpass", "bigpass@example.com", "A1" + "x" * 10_000)
    assert r.status_code == 422


def test_register_accepts_128_char_password(api_client):
    client, _ = api_client
    pw = "A1" + "x" * 126  # exactly 128
    assert _register(client, "maxpass", "maxpass@example.com", pw).status_code == 201


def test_login_rejects_10kb_password(api_client):
    client, _ = api_client
    r = _login(client, "someone", "x" * 10_000)
    assert r.status_code == 422


def test_login_rejects_oversized_username_field(api_client, strong_password):
    client, _ = api_client
    r = _login(client, "u" * 300, strong_password)
    assert r.status_code == 422


def test_login_accepts_128_char_password_field(api_client, strong_password):
    """The cap must not lock anyone out: a 128-char password registers and logs in."""
    client, _ = api_client
    pw = "A1" + "x" * 126
    assert _register(client, "edgepass", "edgepass@example.com", pw).status_code == 201
    r = _login(client, "edgepass", pw)
    assert r.status_code == 200, r.text