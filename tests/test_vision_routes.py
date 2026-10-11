"""Route tests for /jobs/{id}/clean-images and /jobs/{id}/image-report.

The heavy workers are monkeypatched, so no real image pipeline runs.
A patched worker never updates the state dict, so a started job stays "running".
"""
import io
import zipfile

import pytest

FEATURES = [
    # (url suffix, state dict name, worker name)
    ("clean-images", "_CLEAN_JOBS", "_run_clean_images"),
    ("image-report", "_REPORT_JOBS", "_run_image_report"),
]


def _zip_bytes(tag="a"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"cats/{tag}.txt", f"x-{tag}")
    return buf.getvalue()


def _upload(client, headers, name, data):
    r = client.post("/api/v1/upload", files={"file": (name, io.BytesIO(data), "application/octet-stream")},
                    headers=headers)
    assert r.status_code in (200, 201), r.text
    return r.json()["file"]["file_id"]


def _make_job(client, headers, file_id, goal="images"):
    r = client.post("/api/v1/jobs", json={"file_id": file_id, "goal": goal}, headers=headers)
    assert r.status_code in (200, 201, 202), r.text
    return r.json()["job_id"]


@pytest.fixture
def env(registered_user, auth_headers, monkeypatch):
    client, api_module, _token, _name = registered_user

    async def _noop(*a, **k):
        return {}

    # keep the job queue from running the real image pipeline
    monkeypatch.setattr(api_module, "_run_images", _noop)
    for _suffix, state, worker in FEATURES:
        monkeypatch.setattr(api_module, worker, _noop)
        getattr(api_module, state).clear()
    yield client, api_module, auth_headers
    for _suffix, state, _worker in FEATURES:
        getattr(api_module, state).clear()


@pytest.fixture
def second_headers(env, strong_password):
    client = env[0]
    r = client.post("/api/v1/auth/register", json={
        "username": "testuser2", "email": "testuser2@example.com", "password": strong_password})
    assert r.status_code == 201, r.text
    login = client.post("/api/v1/auth/login", json={"username": "testuser2", "password": strong_password})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.parametrize("suffix,state,worker", FEATURES)
class TestStartAndStatus:
    def test_requires_auth(self, env, suffix, state, worker):
        client, _m, _h = env
        assert client.post(f"/api/v1/jobs/whatever/{suffix}").status_code == 401
        assert client.get(f"/api/v1/jobs/whatever/{suffix}").status_code == 401

    def test_unknown_job_is_404(self, env, suffix, state, worker):
        client, _m, h = env
        assert client.post(f"/api/v1/jobs/nope/{suffix}", headers=h).status_code == 404
        assert client.get(f"/api/v1/jobs/nope/{suffix}", headers=h).status_code == 404

    def test_other_users_job_is_404(self, env, second_headers, suffix, state, worker):
        client, _m, h = env
        job = _make_job(client, h, _upload(client, h, "d.zip", _zip_bytes("own")))
        assert client.post(f"/api/v1/jobs/{job}/{suffix}", headers=second_headers).status_code == 404
        assert client.get(f"/api/v1/jobs/{job}/{suffix}", headers=second_headers).status_code == 404

    def test_non_images_goal_is_400(self, env, suffix, state, worker):
        client, _m, h = env
        fid = _upload(client, h, "d.csv", b"a,b\n1,2\n3,4\n")
        job = _make_job(client, h, fid, goal="inspect")
        assert client.post(f"/api/v1/jobs/{job}/{suffix}", headers=h).status_code == 400

    def test_non_archive_upload_is_400(self, env, suffix, state, worker):
        client, _m, h = env
        fid = _upload(client, h, "d.csv", b"a,b\n1,2\n3,4\n")
        job = _make_job(client, h, fid, goal="images")
        assert client.post(f"/api/v1/jobs/{job}/{suffix}", headers=h).status_code == 400

    def test_status_is_none_before_start(self, env, suffix, state, worker):
        client, _m, h = env
        job = _make_job(client, h, _upload(client, h, "d.zip", _zip_bytes("n")))
        r = client.get(f"/api/v1/jobs/{job}/{suffix}", headers=h)
        assert r.status_code == 200
        assert r.json()["status"] == "none"

    def test_start_returns_202_and_second_start_returns_same_state(self, env, suffix, state, worker):
        client, m, h = env
        job = _make_job(client, h, _upload(client, h, "d.zip", _zip_bytes("s")))
        r1 = client.post(f"/api/v1/jobs/{job}/{suffix}", headers=h)
        assert r1.status_code == 202, r1.text
        assert r1.json()["status"] == "running"
        assert getattr(m, state)[job]["status"] == "running"
        r2 = client.post(f"/api/v1/jobs/{job}/{suffix}", headers=h)
        assert r2.status_code == 202
        assert r2.json()["status"] == "running"
        assert client.get(f"/api/v1/jobs/{job}/{suffix}", headers=h).json()["status"] == "running"

    def test_second_job_of_same_user_while_running_is_409(self, env, suffix, state, worker):
        client, _m, h = env
        job1 = _make_job(client, h, _upload(client, h, "a.zip", _zip_bytes("one")))
        job2 = _make_job(client, h, _upload(client, h, "b.zip", _zip_bytes("two")))
        assert job1 != job2
        assert client.post(f"/api/v1/jobs/{job1}/{suffix}", headers=h).status_code == 202
        assert client.post(f"/api/v1/jobs/{job2}/{suffix}", headers=h).status_code == 409


class TestDownloads:
    def test_report_bad_format_is_404(self, env):
        client, _m, h = env
        job = _make_job(client, h, _upload(client, h, "d.zip", _zip_bytes("f")))
        assert client.get(f"/api/v1/jobs/{job}/image-report/exe", headers=h).status_code == 404

    @pytest.mark.parametrize("fmt", ["html", "json", "md"])
    def test_report_download_before_build_is_404(self, env, fmt):
        client, _m, h = env
        job = _make_job(client, h, _upload(client, h, "d.zip", _zip_bytes("g")))
        assert client.get(f"/api/v1/jobs/{job}/image-report/{fmt}", headers=h).status_code == 404

    def test_report_download_requires_auth(self, env):
        client, _m, _h = env
        assert client.get("/api/v1/jobs/x/image-report/html").status_code == 401

    def test_cleaned_zip_before_build_is_404(self, env):
        client, _m, h = env
        job = _make_job(client, h, _upload(client, h, "d.zip", _zip_bytes("c")))
        assert client.get(f"/api/v1/jobs/{job}/cleaned-images", headers=h).status_code == 404