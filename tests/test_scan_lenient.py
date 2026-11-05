"""A CSV whose last row ends in CRLF while the rest end in LF (common after hand edits)."""
import duckdb
import pytest

from src.scan_engine import scan_table

ROWS = 300


def _write_mixed(path):
    lines = ["id,customer,amount,qty,category"]
    for i in range(ROWS):
        lines.append(f"{i},cust_{i % 50},{(i * 7) % 1000 / 10},{i % 9},{'abcd'[i % 4]}")
    data = "\n".join(lines) + "\n"
    data += "999999,cust_x,1.5,1,a\r\n"
    path.write_bytes(data.encode("utf-8"))


def test_mixed_line_endings_are_scanned_not_rejected(tmp_path):
    csv = tmp_path / "mixed.csv"
    _write_mixed(csv)

    # control: prove this file really trips the strict parser, otherwise the test proves nothing
    con = duckdb.connect()
    try:
        con.execute(f"SELECT count(*) FROM read_csv('{csv.as_posix()}')").fetchone()
        strict_ok = True
    except Exception:
        strict_ok = False
    if strict_ok:
        pytest.skip("this DuckDB version reads the mixed file strictly; nothing to retry")

    res = scan_table(csv, tmp_path / "out", max_rows=100)
    assert res is not None
    assert res["total_rows"] == ROWS + 1
    assert res["analyzed_rows"] == 100
    assert res["sampled"] is True