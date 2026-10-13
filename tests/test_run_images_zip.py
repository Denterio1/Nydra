"""Tests for api._run_images_zip: score bands, issues builder, verdict downgrade,
extract/tar choice, cleanup and error paths. Heavy work is faked; no real images."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def api():
    import api as api_module  # noqa: PLC0415
    return api_module


class FakeTracker:
    def __init__(self, n=2):
        self.steps = list(range(n))
        self.started = 0
        self.finished = 0

    async def start_step(self, db):
        self.started += 1

    async def finish_step(self, db):
        self.finished += 1


def make_report(score=85, **over):
    rep = {
        "status": "ok",
        "counts": {"found": 100, "analyzed": 100, "readable": 100, "unreadable": 0, "classes": 2},
        "readiness": {"score": score, "priority_issues": [], "sub_scores": {"balance": 90.0}},
        "quality": {},
        "duplicates": {},
        "balance": {},
        "outliers": {},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(rep.get(k), dict):
            rep[k].update(v)
        else:
            rep[k] = v
    return rep


def run(api, tmp_path, monkeypatch, report=None, extract_exc=None, build_exc=None,
        filename="ds.zip", display_name=None):
    """Run _run_images_zip with fakes. Returns (result, calls, dest_path)."""
    import src.tar_safe as tar_safe  # noqa: PLC0415
    import src.vision_ui as vision_ui  # noqa: PLC0415
    import src.zip_safe as zip_safe  # noqa: PLC0415

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    calls = {"extract": None, "build": None}
    file_path = tmp_path / filename
    dest = tmp_path / "_zip" / file_path.stem

    def make_extract(kind):
        def _extract(path, dest_dir, limits=None):
            calls["extract"] = kind
            if extract_exc:
                raise extract_exc
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            (Path(dest_dir) / "marker.txt").write_text("x")
            return SimpleNamespace(dataset_root=dest_dir, extracted=7, total_bytes=123, skipped=2)
        return _extract

    def fake_build(root, name, max_images):
        calls["build"] = (str(root), name, max_images)
        if build_exc:
            raise build_exc
        return copy.deepcopy(report if report is not None else make_report())

    monkeypatch.setattr(zip_safe, "extract_image_zip", make_extract("zip"))
    monkeypatch.setattr(tar_safe, "extract_image_tar", make_extract("tar"))
    monkeypatch.setattr(vision_ui, "build_dataset_report", fake_build)

    tracker = FakeTracker()
    result = asyncio.run(api._run_images_zip(file_path, tracker, None, display_name))
    calls["tracker"] = tracker
    return result, calls, dest


def ids(result):
    return {i["issue_id"]: i for i in result["issues"]}


# ---------------------------------------------------------------- score bands
@pytest.mark.parametrize("score,verdict,grade", [
    (95, "ready", "A"), (90, "ready", "A"), (85, "ready", "B"), (80, "ready", "B"),
    (79, "needs_work", "C"), (65, "needs_work", "C"), (64, "needs_work", "D"),
    (50, "needs_work", "D"), (49, "not_ready", "F"), (0, "not_ready", "F"),
])
def test_score_bands(api, tmp_path, monkeypatch, score, verdict, grade):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(score))
    q = res["quality"]
    assert (q["verdict"], q["grade"]) == (verdict, grade)
    assert q["overall_score"] == score
    assert res["ml_readiness"]["score"] == score


def test_nan_score_becomes_zero(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(float("nan")))
    assert res["quality"]["overall_score"] == 0
    assert res["quality"]["verdict"] == "not_ready"


def test_missing_score_becomes_zero(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"] = {}
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    assert res["quality"]["overall_score"] == 0


# ------------------------------------------------------------ priority issues
def test_priority_issues_severity_and_ids(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"]["priority_issues"] = ["a", "b", "c", "d", "e"]
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    got = [(i["issue_id"], i["severity"]) for i in res["issues"] if i["issue_id"].startswith("ds-") and "-q-" not in i["issue_id"]]
    assert got == [("ds-0", "high"), ("ds-1", "high"), ("ds-2", "high"), ("ds-3", "medium"), ("ds-4", "medium")]


def test_priority_issues_capped_at_20(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"]["priority_issues"] = ["issue %d" % n for n in range(30)]
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    plain = [i for i in res["issues"] if "-q-" not in i["issue_id"]]
    assert len(plain) == 20


def test_priority_issue_dict_forms(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"]["priority_issues"] = [{"issue": "From issue key"}, {"title": "From title key"}]
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    assert res["issues"][0]["title"] == "From issue key"
    assert res["issues"][1]["title"] == "From title key"


def test_priority_issue_title_truncated(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"]["priority_issues"] = ["x" * 500]
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    assert len(res["issues"][0]["title"]) == 200
    assert len(res["issues"][0]["description"]) == 500


# --------------------------------------------------------------- quality flags
def test_flag_issue_title_and_severity_high(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(quality={"noisy": 30}))
    it = ids(res)["ds-q-noisy"]
    assert it["title"] == "30 noisy images (30% of analyzed)"
    assert it["severity"] == "high"
    assert it["auto_fixable"] is True


@pytest.mark.parametrize("count,sev", [(5, "low"), (10, "medium"), (29, "medium"), (30, "high")])
def test_flag_severity_thresholds(api, tmp_path, monkeypatch, count, sev):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(quality={"blurry": count}))
    assert ids(res)["ds-q-blurry"]["severity"] == sev


def test_zero_flags_make_no_issue(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(quality={"noisy": 0, "dark": 0}))
    assert "ds-q-noisy" not in ids(res)
    assert "ds-q-dark" not in ids(res)


def test_all_seven_flags_are_covered(api, tmp_path, monkeypatch):
    q = {k: 1 for k in ("noisy", "blurry", "dark", "overexposed", "low_res", "artifacts", "rejected")}
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(quality=q))
    got = ids(res)
    for k in q:
        assert "ds-q-" + k in got


# ------------------------------------------------------------------ unreadable
def test_unreadable_high_when_5_percent_or_more(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(counts={"unreadable": 6}))
    assert ids(res)["ds-q-unreadable"]["severity"] == "high"


def test_unreadable_medium_when_small_share(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(counts={"unreadable": 2}))
    it = ids(res)["ds-q-unreadable"]
    assert it["severity"] == "medium"
    assert it["title"] == "2 unreadable or corrupted files"


# ------------------------------------------------------------------ duplicates
def test_duplicates_count_is_exact_plus_pairs(api, tmp_path, monkeypatch):
    dup = {"exact_flagged": 2, "pairs": [{"a": "x", "b": "y"}]}
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(duplicates=dup))
    assert ids(res)["ds-q-dups"]["title"] == "3 duplicate or near-duplicate images"


def test_no_duplicates_no_issue(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report())
    assert "ds-q-dups" not in ids(res)


# --------------------------------------------------------------------- balance
def test_balance_issue_only_when_flag_set(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(balance={"severely_imbalanced": True, "imbalance_ratio": 25.0}))
    it = ids(res)["ds-q-balance"]
    assert it["severity"] == "high"
    assert "25.0" in it["title"]
    res2, _, _ = run(api, tmp_path, monkeypatch, make_report(balance={"severely_imbalanced": False}))
    assert "ds-q-balance" not in ids(res2)


# -------------------------------------------------------------------- outliers
@pytest.mark.parametrize("rate,present", [(0.02, False), (0.03, True), (0.2, True)])
def test_outlier_threshold(api, tmp_path, monkeypatch, rate, present):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(outliers={"rate": rate}))
    assert ("ds-q-outliers" in ids(res)) is present


def test_outlier_title_percent(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(outliers={"rate": 0.04}))
    assert ids(res)["ds-q-outliers"]["title"] == "4% of images look like outliers"


# ------------------------------------------------------------ verdict downgrade
def test_downgrade_when_one_flaw_over_half(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(90, quality={"noisy": 60}))
    assert res["quality"]["verdict"] == "needs_work"
    assert res["quality"]["grade"] == "C"
    assert res["quality"]["overall_score"] == 90  # score itself unchanged by design


def test_no_downgrade_at_exactly_half(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(90, quality={"noisy": 50}))
    assert res["quality"]["verdict"] == "ready"
    assert res["quality"]["grade"] == "A"


def test_downgrade_uses_worst_flag_not_sum(api, tmp_path, monkeypatch):
    q = {"noisy": 30, "blurry": 30, "dark": 30}
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(90, quality=q))
    assert res["quality"]["verdict"] == "ready"


def test_downgrade_leaves_not_ready_alone(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(30, quality={"noisy": 90}))
    assert res["quality"]["verdict"] == "not_ready"
    assert res["quality"]["grade"] == "F"


def test_severe_imbalance_flag_downgrades_ready(api, tmp_path, monkeypatch):
    rep = make_report(85, balance={"severely_imbalanced": True, "imbalance_ratio": 25.0})
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    assert res["quality"]["verdict"] == "needs_work"
    assert res["quality"]["grade"] == "C"
    assert res["quality"]["overall_score"] == 85  # score unchanged by design
    assert "ds-q-balance" in ids(res)


def test_severe_imbalance_in_priority_issues_downgrades_ready(api, tmp_path, monkeypatch):
    rep = make_report(88)
    rep["readiness"]["priority_issues"] = ["Severe class imbalance"]
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    assert res["quality"]["verdict"] == "needs_work"
    assert res["quality"]["grade"] == "C"


def test_no_imbalance_no_downgrade(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, make_report(88))
    assert res["quality"]["verdict"] == "ready"


# ----------------------------------------------------------------- result shape
def test_dimensions_only_numeric_sub_scores(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"]["sub_scores"] = {"balance": 80.0, "note": "text", "quality": 70}
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    names = [d["name"] for d in res["quality"]["dimensions"]]
    assert names == ["balance", "quality"]


def test_critical_and_checklist_slices(api, tmp_path, monkeypatch):
    rep = make_report()
    rep["readiness"]["priority_issues"] = ["i%d" % n for n in range(12)]
    res, _, _ = run(api, tmp_path, monkeypatch, rep)
    assert len(res["quality"]["critical_issues"]) == 3
    assert len(res["quality"]["fix_checklist"]) == 8


def test_extract_info_added_to_report(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch)
    assert res["vision_report"]["extract"] == {"extracted": 7, "total_bytes": 123, "skipped": 2}


def test_all_steps_started_and_finished(api, tmp_path, monkeypatch):
    _, calls, _ = run(api, tmp_path, monkeypatch)
    assert calls["tracker"].started == 2
    assert calls["tracker"].finished == 2


# ------------------------------------------------------- routing / display name
def test_display_name_and_cap_passed_to_builder(api, tmp_path, monkeypatch):
    _, calls, _ = run(api, tmp_path, monkeypatch, display_name="my photos.zip")
    assert calls["build"][1] == "my photos.zip"
    assert calls["build"][2] == 150


def test_falls_back_to_stored_name(api, tmp_path, monkeypatch):
    _, calls, _ = run(api, tmp_path, monkeypatch, filename="stored.zip")
    assert calls["build"][1] == "stored.zip"


@pytest.mark.parametrize("filename,kind", [
    ("a.zip", "zip"), ("a.ZIP", "zip"), ("a.tar", "tar"), ("a.tgz", "tar"), ("a.gz", "tar"),
])
def test_extractor_choice(api, tmp_path, monkeypatch, filename, kind):
    _, calls, _ = run(api, tmp_path, monkeypatch, filename=filename)
    assert calls["extract"] == kind


# ------------------------------------------------------------ errors + cleanup
def test_unsafe_zip_becomes_error_report(api, tmp_path, monkeypatch):
    from src.zip_safe import UnsafeZipError  # noqa: PLC0415
    res, _, _ = run(api, tmp_path, monkeypatch, extract_exc=UnsafeZipError("The zip contains path traversal entries."))
    vr = res["vision_report"]
    assert vr["status"] == "error"
    assert vr["message"] == "The zip contains path traversal entries."
    assert "issues" not in res and "quality" not in res


def test_unexpected_error_hides_details(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, build_exc=RuntimeError("secret path /etc/x"))
    vr = res["vision_report"]
    assert vr["status"] == "error"
    assert vr["message"] == "Dataset analysis failed: RuntimeError"
    assert "secret" not in vr["message"]


def test_error_report_from_builder_skips_issue_building(api, tmp_path, monkeypatch):
    res, _, _ = run(api, tmp_path, monkeypatch, report={"status": "error", "message": "boom"})
    assert res["vision_report"]["status"] == "error"
    assert "issues" not in res


def test_extracted_folder_removed_after_success(api, tmp_path, monkeypatch):
    _, _, dest = run(api, tmp_path, monkeypatch)
    assert not dest.exists()


def test_extracted_folder_removed_when_build_fails(api, tmp_path, monkeypatch):
    _, _, dest = run(api, tmp_path, monkeypatch, build_exc=RuntimeError("x"))
    assert not dest.exists()