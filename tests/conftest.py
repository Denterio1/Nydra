"""
conftest.py — Shared fixtures for the Nydra test suite.

Fixtures defined here are auto-available to every test file in tests/
without needing to import them. Keep fixtures here generic and reusable;
file-specific fixtures (e.g. mock payloads only test_security.py needs)
belong in that file instead.
"""

import os
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Core dataset fixtures (used by test_nydra.py, test_audit_layer.py) ────────

@pytest.fixture
def sample_csv(tmp_path):
    """A small CSV with known issues: missing values (age, salary, city) and
    one exact duplicate row (Alice). Used to assert exact counts everywhere."""
    path = tmp_path / "test.csv"
    path.write_text(
        "name,age,salary,city\n"
        "Alice,25,3000,Algiers\n"
        "Bob,,4500,Oran\n"
        "Carol,31,,Algiers\n"
        "Dave,28,3800,\n"
        "Alice,25,3000,Algiers\n"  # duplicate of row 1
    )
    return str(path)


@pytest.fixture
def clean_csv(tmp_path):
    """A CSV with zero missing values and zero duplicates — the baseline
    'nothing wrong here' case for any detector."""
    path = tmp_path / "clean.csv"
    path.write_text(
        "product,price,quantity\n"
        "Laptop,999,5\n"
        "Phone,499,10\n"
        "Tablet,299,8\n"
        "Watch,199,15\n"
        "Keyboard,79,20\n"
    )
    return str(path)


@pytest.fixture
def sample_data(sample_csv):
    from src.data.loader import load_csv
    return load_csv(sample_csv)


@pytest.fixture
def clean_data(clean_csv):
    from src.data.loader import load_csv
    return load_csv(clean_csv)


# ── Generic numeric arrays for nydra_core tests ────────────────────────────

@pytest.fixture
def simple_array():
    """A plain, well-behaved numeric array — no NaN, no zero variance."""
    return [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


@pytest.fixture
def constant_array():
    """Zero-variance array — used to assert std_dev == 0 and that
    skewness/kurtosis raise (undefined for zero std)."""
    return [5.0, 5.0, 5.0, 5.0, 5.0]


@pytest.fixture
def array_with_nan():
    return [1.0, 2.0, float("nan"), 4.0, float("inf"), 5.0]


@pytest.fixture
def array_with_outlier():
    """Contains an obvious outlier (1000.0), but note: MAD of this exact
    array is 0.0 (majority of values tie at 2), which is a known blind
    spot of MAD-based detection — see
    test_detect_outliers_mad_majority_tie_blind_spot in test_nydra_core.py.
    Use `array_with_outlier_nonzero_mad` for MAD-specific "detects" tests."""
    return [1.0, 2.0, 2.0, 3.0, 2.0, 2.0, 1000.0]


@pytest.fixture
def array_with_outlier_nonzero_mad():
    """Same intent as array_with_outlier (one obvious high outlier) but
    without the majority-tie MAD=0 edge case, so MAD-based detection can
    actually be exercised meaningfully."""
    return [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 11.0, 500.0]


@pytest.fixture
def no_outlier_array():
    """A tight, genuinely outlier-free array — distinct from
    `simple_array`, which deliberately contains a mild IQR outlier (9.0)
    and should NOT be reused for 'no outliers expected' assertions."""
    return [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 11.0, 12.0]


# ── API test fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def api_test_env(tmp_path, monkeypatch):
    """
    Points api.py at an isolated on-disk SQLite DB and isolated upload/report
    dirs BEFORE the module is imported, since api.py reads these from env
    vars at import time (module-level engine + directory creation).

    Must be requested before importing `api` in a test.
    """
    db_path = tmp_path / "test_nydra.db"
    upload_dir = tmp_path / "uploads"
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("REPORT_DIR", str(report_dir))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-for-production")
    monkeypatch.setenv("REFRESH_SECRET", "test-refresh-secret-not-for-production")
    monkeypatch.setenv("WORKER_COUNT", "1")
    monkeypatch.setenv("MAX_FILE_MB", "1")  # 1MB cap makes oversized-file tests fast
    return {"db_path": db_path, "upload_dir": upload_dir, "report_dir": report_dir}


@pytest.fixture
def strong_password():
    """Meets api_schemas.UserRegister.password_strength validator:
    >=8 chars, >=1 uppercase, >=1 digit."""
    return "TestPass123"


@pytest.fixture
def api_client(api_test_env):
    """
    Yields a FastAPI TestClient wired to an isolated, per-test SQLite file
    and upload/report dirs.

    IMPORTANT: `api` (at the project root, NOT `src.api`) and
    `src.api_schemas` create module-level state (the SQLAlchemy engine,
    upload/report directories) by reading env vars AT IMPORT TIME — not
    per-request. That means two tests sharing one imported module
    instance would silently share one database. We force a fresh import
    per test (purging sys.modules) so every test gets a fully isolated
    engine bound to its own tmp_path DB file.
    """
    import sys
    from fastapi.testclient import TestClient

    # Purge every relevant module AND the stale attribute references
    # their parent package objects hold on them. `del sys.modules['src.api_schemas']`
    # alone is NOT enough: Python's `from src import api_schemas` resolves
    # via `getattr(sys.modules['src'], 'api_schemas')` first, and that
    # attribute survives sys.modules deletion — silently returning the
    # stale, previous test's module instead of re-executing it. Confirmed
    # directly: `del sys.modules['src.api_schemas']; hasattr(sys.modules['src'], 'api_schemas')`
    # is still True. `api` itself is a top-level module (api.py lives at
    # the project root, not inside a package), so it has no parent-package
    # attribute to worry about — only the sys.modules entry needs clearing.
    src_pkg = sys.modules.get("src")
    for mod_name in list(sys.modules):
        if mod_name == "api" or mod_name == "src" or mod_name.startswith("src."):
            del sys.modules[mod_name]
    if src_pkg is not None:
        for attr in ("api_schemas", "api"):
            if hasattr(src_pkg, attr):
                delattr(src_pkg, attr)

    import api as api_module

    # ── Workaround for a confirmed production bug, NOT a design choice ────
    # api.py's http_exception_handler and generic_exception_handler both
    # build `ErrorResponse(...).model_dump()` (missing `mode="json"`),
    # so the `timestamp: datetime` field is never stringified and EVERY
    # error response (401/403/404/409/413/422/429/500 — anything routed
    # through these handlers) crashes json.dumps() before reaching the
    # client. This is proven directly, against the real unpatched
    # handlers, in TestKnownBugs::test_error_response_datetime_serialization_bug
    # in test_api.py.
    #
    # Patching it here (only in the test fixture) lets the REST of this
    # suite verify the intended auth/upload/job/rate-limit behavior
    # instead of every single error-path test independently rediscovering
    # this one root cause. Real fix: change `.model_dump()` to
    # `.model_dump(mode="json")` in both handlers in src/api.py.
    from starlette.responses import JSONResponse as _JSONResponse

    async def _fixed_http_exception_handler(request, exc):
        return _JSONResponse(status_code=exc.status_code, content=api_module.ErrorResponse(
            error=api_module.ErrorCode.INTERNAL_ERROR, message=exc.detail,
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump(mode="json"))

    async def _fixed_generic_exception_handler(request, exc):
        return _JSONResponse(status_code=500, content=api_module.ErrorResponse(
            error=api_module.ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred. Please try again.",
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump(mode="json"))

    api_module.app.exception_handlers[api_module.HTTPException] = _fixed_http_exception_handler
    api_module.app.exception_handlers[Exception] = _fixed_generic_exception_handler

    client = TestClient(api_module.app)
    client.__enter__()
    try:
        yield client, api_module
    finally:
        # MUST stop JobQueue's background workers BEFORE the TestClient's
        # context exits. api.py's lifespan shutdown never stops them (see
        # the separate finding on JobQueue having no stop() and never
        # being cancelled) — they're an infinite `while self._running`
        # loop. If left running, TestClient.__exit__() hangs indefinitely
        # trying to join the ASGI portal thread while zombie workers keep
        # retrying against a soon-to-be-disposed engine. Confirmed
        # directly: without this, teardown hangs past any reasonable
        # timeout. A short sleep lets the workers' current ~1s poll cycle
        # notice `_running` flipped and exit cleanly.
        api_module.job_queue._running = False
        time.sleep(1.2)
        client.__exit__(None, None, None)


@pytest.fixture
def registered_user(api_client, strong_password):
    """Registers a user and returns (client, api_module, access_token, username)."""
    client, api_module = api_client
    username = "testuser1"
    resp = client.post("/api/v1/auth/register", json={
        "username": username, "email": "testuser1@example.com", "password": strong_password,
    })
    assert resp.status_code == 201, resp.text
    login_resp = client.post("/api/v1/auth/login", json={"username": username, "password": strong_password})
    assert login_resp.status_code == 200, login_resp.text
    token = login_resp.json()["access_token"]
    return client, api_module, token, username


@pytest.fixture
def auth_headers(registered_user):
    _, _, token, _ = registered_user
    return {"Authorization": f"Bearer {token}"}