"""The `dna` goal: enum value, plan, dispatcher output."""
import json

import pandas as pd


def _csv(tmp_path):
    df = pd.DataFrame({
        "a": range(60),
        "b": [i % 7 for i in range(60)],
        "c": ["x", "y", "z"] * 20,
    })
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    return p


def test_enum_has_dna():
    from src.api_schemas import JobGoal
    assert JobGoal("dna") is JobGoal.DNA


def test_dispatch_dna_returns_json_safe_dna(tmp_path):
    import api
    out = api._dispatch_step(None, "DNA", api.JobGoal.DNA, _csv(tmp_path), {}, {})
    assert isinstance(out.get("dna"), dict) and out["dna"]
    json.dumps(out)  # must be JSON-serializable


def test_dispatch_dna_is_deterministic(tmp_path):
    import api
    p = _csv(tmp_path)
    a = api._dispatch_step(None, "DNA", api.JobGoal.DNA, p, {}, {})
    b = api._dispatch_step(None, "DNA", api.JobGoal.DNA, p, {}, {})
    a["dna"].pop("created_at", None)
    b["dna"].pop("created_at", None)
    assert a == b

def test_dispatch_dna_unreadable_type_returns_empty(tmp_path):
    import api
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert api._dispatch_step(None, "DNA", api.JobGoal.DNA, p, {}, {}) == {}