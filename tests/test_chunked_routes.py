import hashlib
import io
import os
from pathlib import Path

MB = 1024 * 1024


def _init(client, headers, size, chunk=MB, name="data.zip"):
    return client.post("/api/v1/uploads/init",
                       json={"filename": name, "total_size": size, "chunk_size": chunk},
                       headers=headers)


def _put(client, headers, uid, i, data):
    return client.put(f"/api/v1/uploads/{uid}/chunks/{i}", content=data, headers=headers)


def test_chunked_roundtrip_out_of_order(auth_headers, registered_user):
    client, _, _, _ = registered_user
    data = os.urandom(int(2.5 * MB))
    r = _init(client, auth_headers, len(data))
    assert r.status_code == 201, r.text
    uid = r.json()["upload_id"]
    assert r.json()["n_chunks"] == 3
    parts = [data[:MB], data[MB:2 * MB], data[2 * MB:]]
    for i in (2, 0, 1):
        assert _put(client, auth_headers, uid, i, parts[i]).status_code == 200
    st = client.get(f"/api/v1/uploads/{uid}", headers=auth_headers).json()
    assert st["complete"] is True and st["received"] == [0, 1, 2]
    r = client.post(f"/api/v1/uploads/{uid}/complete", headers=auth_headers)
    assert r.status_code == 201, r.text
    f = r.json()["file"]
    assert f["file_type"] == "zip"
    assert f["size_bytes"] == len(data)
    assert f["checksum"] == hashlib.sha256(data).hexdigest()
    assert Path(f["storage_path"]).read_bytes() == data
    assert client.get(f"/api/v1/uploads/{uid}", headers=auth_headers).status_code == 404


def test_resume_shows_received_chunks(auth_headers, registered_user):
    client, _, _, _ = registered_user
    uid = _init(client, auth_headers, 2 * MB).json()["upload_id"]
    _put(client, auth_headers, uid, 1, b"x" * MB)
    st = client.get(f"/api/v1/uploads/{uid}", headers=auth_headers).json()
    assert st["received"] == [1] and st["complete"] is False


def test_complete_with_missing_chunk_is_409(auth_headers, registered_user):
    client, _, _, _ = registered_user
    uid = _init(client, auth_headers, 2 * MB).json()["upload_id"]
    _put(client, auth_headers, uid, 0, b"x" * MB)
    assert client.post(f"/api/v1/uploads/{uid}/complete", headers=auth_headers).status_code == 409


def test_wrong_chunk_size_is_400(auth_headers, registered_user):
    client, _, _, _ = registered_user
    uid = _init(client, auth_headers, 2 * MB).json()["upload_id"]
    assert _put(client, auth_headers, uid, 0, b"x" * (MB - 1)).status_code == 400
    assert _put(client, auth_headers, uid, 9, b"x" * MB).status_code == 400


def test_bad_extension_and_sizes(auth_headers, registered_user):
    client, _, _, _ = registered_user
    assert _init(client, auth_headers, 2 * MB, name="evil.exe").status_code == 400
    assert _init(client, auth_headers, 0).status_code == 400
    assert _init(client, auth_headers, 2 * MB, chunk=10).status_code == 400
    assert _init(client, auth_headers, 100 * 1024 * MB).status_code == 413


def test_traversal_filename_is_sanitized(auth_headers, registered_user):
    client, _, _, _ = registered_user
    r = _init(client, auth_headers, MB, name="../../evil.zip")
    assert r.status_code == 201, r.text
    assert ".." not in r.json()["filename"] and "/" not in r.json()["filename"]


def test_abort_removes_upload(auth_headers, registered_user):
    client, _, _, _ = registered_user
    uid = _init(client, auth_headers, MB).json()["upload_id"]
    assert client.delete(f"/api/v1/uploads/{uid}", headers=auth_headers).status_code == 204
    assert client.get(f"/api/v1/uploads/{uid}", headers=auth_headers).status_code == 404


def test_requires_auth(api_client):
    client = api_client[0] if isinstance(api_client, tuple) else api_client
    r = client.post("/api/v1/uploads/init",
                    json={"filename": "a.zip", "total_size": MB, "chunk_size": MB})
    assert r.status_code in (401, 403)


def test_unknown_upload_id_is_404(auth_headers, registered_user):
    client, _, _, _ = registered_user
    assert client.get("/api/v1/uploads/" + "a" * 32, headers=auth_headers).status_code == 404
    assert client.get("/api/v1/uploads/not-an-id", headers=auth_headers).status_code == 404
