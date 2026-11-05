"""Uniqueness bar and the duplicates issue must use the exact scan numbers for big tables."""


def _result(scan=None):
    r = {
        "analyze": {
            "shape": {"rows": 25000, "columns": 5},
            "missing_values": {"qty": 2514},
            "duplicate_rows": 8,
        },
        "detect_outliers": {"n_outliers": 0, "outlier_rate": 0.0},
    }
    if scan is not None:
        r["scan"] = scan
    return r


def _find_dim(obj, name):
    """Walk the result and return the first dict whose 'name' equals name."""
    if isinstance(obj, dict):
        if obj.get("name") == name and "score" in obj:
            return obj
        for v in obj.values():
            hit = _find_dim(v, name)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_dim(v, name)
            if hit:
                return hit
    return None


def test_scan_numbers_drive_uniqueness_and_issue():
    import api
    out = api._enrich_result_for_ui(_result({"duplicate_rows": 40000, "total_rows": 2000000}))
    assert "issues" in out, "enrich did not run (input shape not accepted)"
    dim = _find_dim(out, "Uniqueness")
    assert dim is not None and dim["score"] == 98.0
    texts = " ".join(str(i.get("description", "")) for i in out["issues"])
    assert "40000 duplicate rows (2.0%)" in texts


def test_without_scan_behaviour_is_unchanged():
    import api
    out = api._enrich_result_for_ui(_result())
    dim = _find_dim(out, "Uniqueness")
    assert dim is not None and dim["score"] == 100.0
    texts = " ".join(str(i.get("description", "")) for i in out["issues"])
    assert "8 duplicate rows (0.0%)" in texts