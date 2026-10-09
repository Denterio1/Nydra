from pathlib import Path

from PIL import Image

from src import vision_ui


def _make_ds(root, spec):
    k = 0
    for cls, n in spec.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            k += 1
            Image.new("RGB", (8, 8), (k * 7 % 256, (k * 13) % 256, k % 256)).save(d / ("%d.png" % i))
    (root / "dogs").mkdir(exist_ok=True)
    (root / "dogs" / "bad.png").write_bytes(b"not an image")


def _fake(seen, score=90.0):
    def fake(root, name, max_images):
        seen["root"] = Path(root)
        n = sum(1 for _ in Path(root).rglob("*.png"))
        seen["n"] = n
        return {"status": "ok", "mode": "dataset",
                "counts": {"found": n, "analyzed": n, "readable": n, "unreadable": 0, "classes": 2},
                "readiness": {"score": score, "sub_scores": {"balance": 100}}, "balance": {}, "duplicates": {}}
    return fake


def test_big_dataset_cheap_pass(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_ds(root, {"cats": 20, "dogs": 10})
    seen = {}
    monkeypatch.setattr(vision_ui, "_build_dataset_report_inner", _fake(seen))
    rep = vision_ui.build_dataset_report(root, "t", max_images=10)
    assert rep["status"] == "ok" and rep["truncated"] is True
    assert rep["counts"]["found"] == 31 and rep["counts"]["readable"] == 30
    assert rep["counts"]["unreadable"] == 1 and rep["counts"]["classes"] == 2
    assert seen["n"] <= 10
    assert rep["balance"]["class_counts"] == {"cats": 20, "dogs": 10}
    assert rep["unreadable_files"] and rep["duplicates"]["exact_flagged"] == 0
    assert not seen["root"].exists()  # temp sample folder is removed
    assert rep["readiness"]["score"] == 90.0  # ratio 2 -> no penalty


def test_imbalance_penalty_and_exact_duplicates(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_ds(root, {"cats": 40, "dogs": 5})
    (root / "cats" / "copy.png").write_bytes((root / "cats" / "0.png").read_bytes())
    seen = {}
    monkeypatch.setattr(vision_ui, "_build_dataset_report_inner", _fake(seen))
    rep = vision_ui.build_dataset_report(root, "t", max_images=10)
    assert rep["duplicates"]["exact_flagged"] == 1
    assert rep["readiness"]["balance_penalty"] > 0 and rep["readiness"]["score"] < 90.0


def test_small_dataset_goes_straight_to_inner(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_ds(root, {"cats": 3, "dogs": 2})
    seen = {}
    monkeypatch.setattr(vision_ui, "_build_dataset_report_inner", _fake(seen))
    rep = vision_ui.build_dataset_report(root, "t", max_images=50)
    assert seen["root"] == root and "truncated" not in rep