"""Route tests for GET /api/v1/jobs/{job_id}/cleaned (repaired CSV download)."""
import pytest

# reuse the second-user fixture from the vision route tests
try:
    from test_vision_routes import second_headers  # noqa: F401
except ImportError:  # tests is a package
    from tests.test_vision_routes import second_headers  # noqa: F401


@pytest.fixture
def env(registered_user, tmp_path, monkeypatch):
    client, api_module, token, _username = registered_user
    monkeypatch.setattr(api_module, "REPORT_DIR", tmp_path)

    async def _fake_agent(*a, **k):      # keep the background job from doing real work
        return {}

    monkeypatch.setattr(api_module, "_run_agent", _fake_agent)
    headers = {"Authorization": f"Bearer {token}"}
    return client, headers, tmp_path


def _make_job(client, headers):
    up = client.post("/api/v1/upload", headers=headers,
                     files={"file": ("d.csv", b"a,b\n1,2\n3,4\n", "text/csv")})
    assert up.status_code in (200, 201), up.text
    file_id = up.json()["file"]["file_id"]
    jr = client.post("/api/v1/jobs", headers=headers, json={"file_id": file_id, "goal": "inspect"})
    assert jr.status_code in (200, 201, 202), jr.text
    return jr.json()["job_id"]


def test_requires_auth(env):
    client, headers, _ = env
    job_id = _make_job(client, headers)
    assert client.get(f"/api/v1/jobs/{job_id}/cleaned").status_code == 401


def test_unknown_job_404(env):
    client, headers, _ = env
    r = client.get("/api/v1/jobs/does-not-exist/cleaned", headers=headers)
    assert r.status_code == 404
    assert r.json()  # error body present


def test_own_job_without_file_404(env):
    client, headers, _ = env
    job_id = _make_job(client, headers)
    r = client.get(f"/api/v1/jobs/{job_id}/cleaned", headers=headers)
    assert r.status_code == 404
    assert "cleaned" in r.text.lower()


def test_download_returns_the_csv(env):
    client, headers, out = env
    job_id = _make_job(client, headers)
    payload = "\ufeffa,b\n1,2\n".encode("utf-8")
    (out / f"{job_id}_cleaned.csv").write_bytes(payload)
    r = client.get(f"/api/v1/jobs/{job_id}/cleaned", headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "cleaned_data.csv" in r.headers.get("content-disposition", "")
    assert r.content == payload


def test_other_user_cannot_download(env, second_headers):
    client, headers, out = env
    job_id = _make_job(client, headers)
    (out / f"{job_id}_cleaned.csv").write_bytes(b"secret\n")
    r = client.get(f"/api/v1/jobs/{job_id}/cleaned", headers=second_headers)
    assert r.status_code == 404
    assert b"secret" not in r.content


def test_file_without_db_job_is_not_served(env):
    client, headers, out = env
    (out / "ghost_cleaned.csv").write_bytes(b"ghost\n")
    r = client.get("/api/v1/jobs/ghost/cleaned", headers=headers)
    assert r.status_code == 404
    assert b"ghost" not in r.content


@pytest.mark.parametrize("bad", ["..%2F..%2Fetc%2Fpasswd", "..%5C..%5Cx", "%2e%2e"])
def test_traversal_ids_are_rejected(env, bad):
    client, headers, _ = env
    r = client.get(f"/api/v1/jobs/{bad}/cleaned", headers=headers)
    assert r.status_code == 404