"""Tests for src.vision_report_adapter (synthetic frames, no image libraries)."""
import numpy as np
import pandas as pd
import pytest

from src.vision_report_adapter import prepare_report_frames


def _frames():
    loader = pd.DataFrame({
        "file_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "broken.jpg", "dup.jpg"],
        "width": [800, 800, 800, 800, 0, 100],
        "height": [600, 600, 600, 600, 0, 100],
        "is_valid": [True, True, True, True, False, True],
        "is_corrupted": [False] * 6,
        "is_duplicate": [False, False, False, False, False, True],
    })
    quality = pd.DataFrame({
        "file_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "broken.jpg", "dup.jpg"],
        "overall_score": [90.0, 70.0, 50.0, 30.0, np.nan, 80.0],
    })
    return loader, quality


def test_scores_are_renamed_not_zero():
    loader, quality = _frames()
    _, q2 = prepare_report_frames(loader, quality)
    assert list(q2["quality_score"][:4]) == [90.0, 70.0, 50.0, 30.0]
    assert np.isnan(q2.loc[4, "quality_score"])


def test_severity_bands():
    loader, quality = _frames()
    _, q2 = prepare_report_frames(loader, quality)
    sev = dict(zip(q2["file_path"], q2["issue_severity"]))
    assert pd.isna(sev["a.jpg"])  # no issue (None or NaN depending on pandas)
    assert sev["b.jpg"] == "low"
    assert sev["c.jpg"] == "medium"
    assert sev["d.jpg"] == "high"
    assert sev["broken.jpg"] == "critical"
    assert sev["dup.jpg"] == "low"  # duplicate with a good score


def test_corrupted_flag_set_for_unreadable_only():
    loader, quality = _frames()
    l2, _ = prepare_report_frames(loader, quality)
    flags = dict(zip(l2["file_path"], l2["is_corrupted"]))
    assert flags["broken.jpg"] is True or flags["broken.jpg"] == True  # noqa: E712
    assert not any(v for k, v in flags.items() if k != "broken.jpg")


def test_low_res_and_low_quality_flags():
    loader, quality = _frames()
    _, q2 = prepare_report_frames(loader, quality)
    lr = dict(zip(q2["file_path"], q2["is_low_res"]))
    lq = dict(zip(q2["file_path"], q2["is_low_quality"]))
    assert lr["dup.jpg"] and not lr["a.jpg"]
    assert not lr["broken.jpg"]  # unreadable is not "low-res"
    assert lq["c.jpg"] and lq["d.jpg"]
    assert not lq["b.jpg"] and not lq["broken.jpg"]


def test_inputs_not_mutated():
    loader, quality = _frames()
    loader_before, quality_before = loader.copy(), quality.copy()
    prepare_report_frames(loader, quality)
    pd.testing.assert_frame_equal(loader, loader_before)
    pd.testing.assert_frame_equal(quality, quality_before)


def test_row_count_and_order_preserved():
    loader, quality = _frames()
    quality = quality.iloc[::-1]  # different order
    l2, q2 = prepare_report_frames(loader, quality)
    assert len(l2) == len(q2) == len(loader)
    assert list(q2["file_path"]) == list(loader["file_path"])
    assert q2.loc[0, "quality_score"] == 90.0


def test_works_without_quality_frame():
    loader, _ = _frames()
    l2, q2 = prepare_report_frames(loader, None)
    assert q2["quality_score"].isna().all()
    # without scores we cannot call anything unreadable except the invalid row
    assert list(l2["is_corrupted"]) == [False, False, False, False, True, False]


def test_missing_optional_columns_tolerated():
    loader = pd.DataFrame({"file_path": ["a.jpg", "b.jpg"]})
    quality = pd.DataFrame({"file_path": ["a.jpg", "b.jpg"], "overall_score": [80.0, 40.0]})
    l2, q2 = prepare_report_frames(loader, quality)
    assert not q2["is_low_res"].any()
    assert q2.loc[1, "issue_severity"] == "high"


def test_requires_file_path():
    with pytest.raises(ValueError):
        prepare_report_frames(pd.DataFrame({"x": [1]}), None)