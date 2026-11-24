"""Tests for GET /api/v1/files (list my files + storage usage)."""
import io


def _upload(client, headers, name, content):
    resp = client.post(
        "/api/v1/upload",
        files={"file": (name, io.BytesIO(content), "text/csv")},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["file"]


def _second_user_headers(client, strong_password):
    resp = client.post("/api/v1/auth/register", json={
        "username": "testuser2", "email": "testuser2@example.com", "password": strong_password,
    })
    assert resp.status_code == 201, resp.text
    login = client.post("/api/v1/auth/login", json={"username": "testuser2", "password": strong_password})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_list_requires_login(registered_user):
    client, _, _, _ = registered_user
    assert client.get("/api/v1/files").status_code in (401, 403)


def test_new_user_has_empty_list(registered_user, auth_headers):
    client, _, _, _ = registered_user
    resp = client.get("/api/v1/files", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["files"] == []
    assert body["used_bytes"] == 0
    assert isinstance(body["quota_mb"], int)


def test_own_files_listed_without_server_path(registered_user, auth_headers):
    client, _, _, _ = registered_user
    a = _upload(client, auth_headers, "a.csv", b"a,b\n1,2\n3,4\n")
    b = _upload(client, auth_headers, "b.csv", b"x,y,z\n9,8,7\n6,5,4\n3,2,1\n")
    resp = client.get("/api/v1/files", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ids = {f["file_id"] for f in body["files"]}
    assert ids == {a["file_id"], b["file_id"]}
    # the browser must never see where the file lives on the server
    assert "storage_path" not in resp.text
    for f in body["files"]:
        assert set(f) == {"file_id", "filename", "file_type", "size_bytes", "uploaded_at"}
    assert body["used_bytes"] == a["size_bytes"] + b["size_bytes"]


def test_other_users_files_are_not_visible(registered_user, auth_headers, strong_password):
    client, _, _, _ = registered_user
    mine = _upload(client, auth_headers, "mine.csv", b"a,b\n1,2\n")
    other_headers = _second_user_headers(client, strong_password)
    theirs = _upload(client, other_headers, "theirs.csv", b"c,d\n5,6\n7,8\n")

    mine_list = client.get("/api/v1/files", headers=auth_headers).json()
    theirs_list = client.get("/api/v1/files", headers=other_headers).json()
    assert {f["file_id"] for f in mine_list["files"]} == {mine["file_id"]}
    assert {f["file_id"] for f in theirs_list["files"]} == {theirs["file_id"]}


def test_deleted_file_disappears_from_list(registered_user, auth_headers):
    client, _, _, _ = registered_user
    f = _upload(client, auth_headers, "gone.csv", b"a,b\n1,2\n")
    assert client.delete(f"/api/v1/files/{f['file_id']}", headers=auth_headers).status_code == 204
    body = client.get("/api/v1/files", headers=auth_headers).json()
    assert body["files"] == []
    assert body["used_bytes"] == 0