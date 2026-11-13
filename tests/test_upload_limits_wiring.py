import pytest

from src.chunked_upload import ChunkStore, ChunkLimits, ChunkUploadError
from src.upload_limits import max_any_bytes

MB = 1024 ** 2
CHUNK = 16 * MB

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_TABLE_GB", "2")
    monkeypatch.setenv("MAX_OTHER_MB", "100")
    import shutil
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        shutil, "disk_usage", lambda p: Usage(10 ** 15, 0, 10 ** 15)
    )
    return ChunkStore(
        tmp_path / "_chunks",
        ChunkLimits(max_total_bytes=max_any_bytes()),
        allowed_extensions={"csv", "tsv", "parquet", "json", "xlsx", "txt", "pdf"},
    )


def test_big_table_is_accepted(store):
    info = store.init("u1", "big.csv", 1500 * MB, CHUNK)
    assert info


def test_big_json_is_refused_with_the_right_limit(store):
    with pytest.raises(ChunkUploadError) as exc:
        store.init("u1", "big.json", 300 * MB, CHUNK)
    assert "100 MB" in str(exc.value)


def test_small_json_is_accepted(store):
    assert store.init("u1", "small.json", 50 * MB, CHUNK)


def test_table_above_table_limit_is_refused(store):
    with pytest.raises(ChunkUploadError):
        store.init("u1", "huge.csv", 3 * 1024 * MB, CHUNK)    