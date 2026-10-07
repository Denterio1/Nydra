import hashlib
import os
import time

import pytest

from src.chunked_upload import ChunkLimits, ChunkStore, ChunkUploadError

LIM = ChunkLimits(min_chunk_bytes=1, max_chunk_bytes=10_000, max_chunks=50,
                  max_total_bytes=100_000, max_active_per_user=2, stale_after_s=100)


@pytest.fixture
def store(tmp_path):
    return ChunkStore(tmp_path / "chunks", LIM, allowed_extensions=["zip", "png"])


def start(store, user="u1", size=2500, chunk=1000, name="data.zip"):
    return store.init(user, name, size, chunk)["upload_id"]


def err(fn, *a, **k):
    with pytest.raises(ChunkUploadError) as e:
        fn(*a, **k)
    return e.value


def test_roundtrip_out_of_order(store, tmp_path):
    data = os.urandom(2500)
    uid = start(store)
    parts = [data[:1000], data[1000:2000], data[2000:]]
    for i in (2, 0, 1):
        store.put_chunk(uid, "u1", i, parts[i])
    dest = tmp_path / "out" / "final.zip"
    sha, size = store.assemble(uid, "u1", dest)
    assert size == 2500
    assert sha == hashlib.sha256(data).hexdigest()
    assert dest.read_bytes() == data
    assert err(store.status, uid, "u1").status == 404


def test_resume_status(store):
    uid = start(store)
    store.put_chunk(uid, "u1", 1, b"x" * 1000)
    st = store.status(uid, "u1")
    assert st["received"] == [1] and not st["complete"]


def test_duplicate_chunk_is_idempotent(store):
    uid = start(store)
    store.put_chunk(uid, "u1", 0, b"a" * 1000)
    assert store.put_chunk(uid, "u1", 0, b"b" * 1000) == 1


def test_wrong_size_and_index(store):
    uid = start(store)
    assert err(store.put_chunk, uid, "u1", 0, b"a" * 999).status == 400
    assert err(store.put_chunk, uid, "u1", 2, b"a" * 1000).status == 400
    assert err(store.put_chunk, uid, "u1", 3, b"a" * 500).status == 400
    assert err(store.put_chunk, uid, "u1", -1, b"a").status == 400


def test_missing_chunk_blocks_assemble(store, tmp_path):
    uid = start(store)
    store.put_chunk(uid, "u1", 0, b"a" * 1000)
    assert err(store.assemble, uid, "u1", tmp_path / "x.zip").status == 409
    assert store.status(uid, "u1")["received"] == [0]


def test_other_user_gets_404(store, tmp_path):
    uid = start(store)
    assert err(store.put_chunk, uid, "evil", 0, b"a" * 1000).status == 404
    assert err(store.status, uid, "evil").status == 404
    assert err(store.assemble, uid, "evil", tmp_path / "x").status == 404
    assert err(store.abort, uid, "evil").status == 404


def test_bad_upload_ids(store):
    for bad in ("../x", "..", "", "A" * 32, "g" * 32, None, 123):
        assert err(store.status, bad, "u1").status == 404


def test_init_limits(store):
    assert err(store.init, "u1", "a.zip", 200_000, 1000).status == 413
    assert err(store.init, "u1", "a.zip", 5000, 20_000).status == 400
    assert err(store.init, "u1", "a.zip", 5000, 0).status == 400
    assert err(store.init, "u1", "a.zip", 0, 100).status == 400
    assert err(store.init, "u1", "a.zip", True, 100).status == 400
    assert err(store.init, "u1", "a.zip", 90_000, 100).status == 400
    assert err(store.init, "u1", "a.exe", 1000, 100).status == 400


def test_active_session_limit(store):
    start(store)
    start(store)
    assert err(store.init, "u1", "a.zip", 1000, 100).status == 429
    start(store, user="u2")


def test_disk_headroom(store, monkeypatch):
    class U:
        free = 100
    monkeypatch.setattr("src.chunked_upload.shutil.disk_usage", lambda p: U)
    assert err(store.init, "u1", "a.zip", 1000, 100).status == 507


def test_abort_frees_slot(store):
    uid = start(store)
    start(store)
    store.abort(uid, "u1")
    start(store)


def test_cleanup_stale(store):
    old = start(store)
    new = start(store, user="u2")
    d = store.root / old
    past = time.time() - 1000
    os.utime(d, (past, past))
    assert store.cleanup_stale() == 1
    assert err(store.status, old, "u1").status == 404
    assert store.status(new, "u2")["n_chunks"] == 3


def test_single_chunk_file(store, tmp_path):
    uid = store.init("u1", "a.png", 300, 1000)["upload_id"]
    store.put_chunk(uid, "u1", 0, b"z" * 300)
    sha, size = store.assemble(uid, "u1", tmp_path / "a.png")
    assert size == 300
