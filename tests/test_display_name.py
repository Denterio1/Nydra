"""Tests for api._display_name: original upload filename from the job row,
falling back to the stored file name. Uses a fake db session; no database needed."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def api():
    import api as api_module  # noqa: PLC0415
    return api_module


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeDB:
    def __init__(self, value=None, exc=None):
        self.value = value
        self.exc = exc
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        if self.exc:
            raise self.exc
        return FakeResult(self.value)


STORED = Path("uploads/12_abcdef0123456789.zip")


def call(api, db, tracker, file_path=STORED):
    return asyncio.run(api._display_name(db, tracker, file_path))


def test_returns_original_name_from_row(api):
    db = FakeDB("my holiday photos.zip")
    assert call(api, db, SimpleNamespace(job_id="j1")) == "my holiday photos.zip"
    assert len(db.statements) == 1


def test_falls_back_when_row_missing(api):
    assert call(api, FakeDB(None), SimpleNamespace(job_id="j1")) == STORED.name


def test_falls_back_when_row_empty_string(api):
    assert call(api, FakeDB(""), SimpleNamespace(job_id="j1")) == STORED.name


def test_no_id_attribute_skips_the_query(api):
    db = FakeDB("should not be used")
    assert call(api, db, SimpleNamespace()) == STORED.name
    assert db.statements == []


def test_empty_job_id_skips_the_query(api):
    db = FakeDB("should not be used")
    assert call(api, db, SimpleNamespace(job_id="", id=None)) == STORED.name
    assert db.statements == []


def test_uses_id_when_job_id_missing(api):
    db = FakeDB("from id.zip")
    assert call(api, db, SimpleNamespace(id="j2")) == "from id.zip"
    assert len(db.statements) == 1


def test_uses_private_job_id_as_last_resort(api):
    db = FakeDB("from private.zip")
    assert call(api, db, SimpleNamespace(_job_id="j3")) == "from private.zip"


def test_only_first_found_id_is_queried(api):
    # documents current behaviour: after the first truthy id the loop breaks,
    # even if the row is missing, so the second attribute is never tried
    db = FakeDB(None)
    out = call(api, db, SimpleNamespace(job_id="a", id="b"))
    assert out == STORED.name
    assert len(db.statements) == 1


def test_db_error_falls_back_and_does_not_raise(api):
    db = FakeDB(exc=RuntimeError("db is down"))
    assert call(api, db, SimpleNamespace(job_id="j1")) == STORED.name


def test_non_string_row_is_converted(api):
    assert call(api, FakeDB(Path("a.zip")), SimpleNamespace(job_id="j1")) == str(Path("a.zip"))


def test_query_selects_filename_for_the_right_job(api):
    db = FakeDB("x.zip")
    call(api, db, SimpleNamespace(job_id="job-123"))
    stmt = db.statements[0]
    compiled = stmt.compile()
    assert "filename" in str(compiled).lower()
    assert "job-123" in compiled.params.values()