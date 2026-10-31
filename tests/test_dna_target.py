"""target_column for the dna goal: schema, valid / unknown / missing target."""
import json

import pandas as pd


def _csv(tmp_path):
    n = 80
    df = pd.DataFrame({
        "x": [i * 1.5 for i in range(n)],
        "z": [(i * 7) % 11 for i in range(n)],
        "label": ["a" if i % 2 else "b" for i in range(n)],
    })
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return p


def _run(api_mod, path, req):
    out = api_mod._dispatch_step(None, "DNA", api_mod.JobGoal.DNA, path, req, {})
    dna = dict(out["dna"])
    dna.pop("created_at", None)
    return out, json.dumps(dna, sort_keys=True, default=str)


def test_schema_default_and_value():
    from src.api_schemas import DnaRequest
    assert DnaRequest(file_id="f").target_column is None
    assert DnaRequest(file_id="f", target_column="label").target_column == "label"


def test_valid_target_changes_dna_and_is_reported(tmp_path):
    import api as api_mod
    p = _csv(tmp_path)
    out_none, j_none = _run(api_mod, p, {})
    out_t, j_t = _run(api_mod, p, {"target_column": "label"})
    assert out_none["dna_target"] is None
    assert out_t["dna_target"] == "label"
    assert j_t != j_none


def test_unknown_target_falls_back_to_no_target(tmp_path):
    import api as api_mod
    p = _csv(tmp_path)
    out_none, j_none = _run(api_mod, p, {})
    out_bad, j_bad = _run(api_mod, p, {"target_column": "nope"})
    assert out_bad["dna_target"] is None
    assert j_bad == j_none


def test_non_string_target_ignored(tmp_path):
    import api as api_mod
    out, _ = _run(api_mod, _csv(tmp_path), {"target_column": 5})
    assert out["dna_target"] is None