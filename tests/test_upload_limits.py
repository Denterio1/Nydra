import pytest

from src import upload_limits as ul

MB = 1024 ** 2
GB = 1024 ** 3


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("MAX_TABLE_GB", "MAX_CHUNKED_GB", "MAX_OTHER_MB"):
        monkeypatch.delenv(k, raising=False)


def test_defaults():
    assert ul.limit_bytes_for("csv") == 10 * GB
    assert ul.limit_bytes_for("pdf") == 500 * MB


def test_tables_use_max_table_gb(monkeypatch):
    monkeypatch.setenv("MAX_TABLE_GB", "50")
    for ext in ("csv", "tsv", "parquet", "CSV", ".parquet"):
        assert ul.limit_bytes_for(ext) == 50 * GB


def test_table_falls_back_to_max_chunked_gb(monkeypatch):
    monkeypatch.setenv("MAX_CHUNKED_GB", "2")
    assert ul.limit_bytes_for("csv") == 2 * GB


def test_max_table_gb_wins_over_chunked(monkeypatch):
    monkeypatch.setenv("MAX_CHUNKED_GB", "2")
    monkeypatch.setenv("MAX_TABLE_GB", "8")
    assert ul.limit_bytes_for("tsv") == 8 * GB


def test_other_types_use_max_other_mb(monkeypatch):
    monkeypatch.setenv("MAX_OTHER_MB", "100")
    monkeypatch.setenv("MAX_TABLE_GB", "50")
    for ext in ("xlsx", "json", "txt", "pdf", "docx", "zip", "png", ""):
        assert ul.limit_bytes_for(ext) == 100 * MB


@pytest.mark.parametrize("bad", ["abc", "", "   ", "0", "-5", "nan", "inf"])
def test_invalid_values_fall_back(monkeypatch, bad):
    monkeypatch.setenv("MAX_TABLE_GB", bad)
    monkeypatch.setenv("MAX_OTHER_MB", bad)
    assert ul.limit_bytes_for("csv") == 10 * GB
    assert ul.limit_bytes_for("json") == 500 * MB


def test_max_any_is_the_larger_limit(monkeypatch):
    monkeypatch.setenv("MAX_TABLE_GB", "2")
    monkeypatch.setenv("MAX_OTHER_MB", "100")
    assert ul.max_any_bytes() == 2 * GB
    monkeypatch.setenv("MAX_OTHER_MB", "9000")
    assert ul.max_any_bytes() == 9000 * MB


def test_ext_of():
    assert ul.ext_of("Data.CSV") == "csv"
    assert ul.ext_of("a.b.parquet") == "parquet"
    assert ul.ext_of("noext") == ""
    assert ul.ext_of(None) == ""


def test_message_names_the_right_limit(monkeypatch):
    monkeypatch.setenv("MAX_TABLE_GB", "2")
    monkeypatch.setenv("MAX_OTHER_MB", "100")
    assert "2048 MB" in ul.too_large_message("csv")
    assert "100 MB" in ul.too_large_message("json")