"""_sanitize_for_json must make results savable as JSON (dates, numpy, tuples...)."""
import ast, datetime as dt, json
from pathlib import Path

import numpy as np
import pandas as pd


def _load():
    src = Path("api.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sanitize_for_json")
    ns = {}
    exec(compile(ast.Module([fn], []), "api.py", "exec"), ns)
    return ns["_sanitize_for_json"]


def test_dates_and_numpy_become_json_safe():
    f = _load()
    data = {
        "d": dt.date(2024, 5, 1),
        "ts": pd.Timestamp("2024-05-01 10:00"),
        "nat": pd.NaT,
        "i": np.int64(7),
        "x": np.float64(1.5),
        "t": (1, dt.date(2020, 1, 2)),
        "s": {3},
        "arr": np.array([1, 2]),
        "nested": [{"when": dt.datetime(2024, 1, 1, 8, 30)}],
        dt.date(2024, 1, 1): "date key",
    }
    out = f(data)
    json.dumps(out)  # must not raise
    assert out["d"] == "2024-05-01"
    assert out["nat"] is None
    assert out["i"] == 7 and out["t"][1] == "2020-01-02"
    assert out["2024-01-01"] == "date key"


def test_dataframe_still_summarised():
    f = _load()
    out = f({"df": pd.DataFrame({"a": [1, 2]})})
    assert out["df"] == {"_type": "DataFrame", "rows": 2, "columns": ["a"]}