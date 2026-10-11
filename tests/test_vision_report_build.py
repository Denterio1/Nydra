"""Tests for src.vision_report_build with fake pipeline modules (no image libraries)."""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.vision_report_build import REPORT_FILES, ReportBuildError, build_report_files


@pytest.fixture
def fakes(monkeypatch):
    spec = {"classes": {"cats": 3, "dogs": 2}, "break_generate": False}
    seen = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    class ZipLimits:
        def __init__(self, **kw):
            self.kw = kw

    def extract(archive, dest, limits):
        dest = Path(dest)
        n = 0
        for cls, count in spec["classes"].items():
            for i in range(count):
                f = dest / cls / f"{i}.jpg"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"x" * 10)
                n += 1
        return SimpleNamespace(dataset_root=dest, extracted=n)

    def stratified_sample(df, max_images, label_col="label", seed=0):
        per = max(1, max_images // df[label_col].nunique())
        return df.groupby(label_col).head(per).reset_index(drop=True)

    def load_images(path, verbose=False):
        rows = [{"file_path": str(p), "label": p.parent.name, "width": 640, "height": 480,
                 "is_valid": True, "is_duplicate": False, "is_corrupted": False}
                for p in sorted(Path(path).rglob("*.jpg"))]
        seen["labels"] = {r["label"] for r in rows}
        return pd.DataFrame(rows)

    def analyze(df, n_workers=2, verbose=False):
        return pd.DataFrame({"file_path": df["file_path"], "overall_score": 80.0}), None

    def generate_report(df, output_dir, formats, quality_df=None):
        seen["quality_cols"] = list(quality_df.columns)
        if spec["break_generate"]:
            return {}
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        for fmt in formats:
            p = out / f"image_report_stamp.{fmt}"
            p.write_text(json.dumps({"summary": {"avg_quality_score": 80.0, "dataset_grade": "B"}})
                         if fmt == "json" else "x", encoding="utf-8")
            paths[fmt] = p
        return paths

    mod("src.zip_safe", ZipLimits=ZipLimits, extract_image_zip=extract)
    mod("src.tar_safe", extract_image_tar=extract)
    mod("src.sampling", stratified_sample=stratified_sample)
    mod("src.vision_nlp.image_loader", load_images=load_images)
    mod("src.vision_nlp.image_quality", analyze_dataset_quality=analyze)
    mod("src.vision_nlp.image_report", generate_report=generate_report)
    return SimpleNamespace(spec=spec, seen=seen)


def _run(tmp_path, **kw):
    work, out = tmp_path / "work", tmp_path / "out"
    summary = build_report_files(tmp_path / "a.zip", work, out, **kw)
    return summary, work, out


def test_small_archive_not_sampled(tmp_path, fakes):
    summary, work, out = _run(tmp_path)
    assert summary["images_found"] == summary["images_analyzed"] == 5
    assert summary["sampled"] is False
    assert summary["avg_score"] == 80.0 and summary["grade"] == "B"
    assert sorted(p.name for p in out.iterdir()) == sorted(REPORT_FILES.values())


def test_adapter_ran_before_report(tmp_path, fakes):
    _run(tmp_path)
    assert "quality_score" in fakes.seen["quality_cols"]
    assert "issue_severity" in fakes.seen["quality_cols"]


def test_large_archive_sampled_keeps_class_folders(tmp_path, fakes):
    fakes.spec["classes"] = {"cats": 10, "dogs": 10}
    summary, work, out = _run(tmp_path, max_images=6)
    assert summary["sampled"] is True
    assert summary["images_found"] == 20 and summary["images_analyzed"] == 6
    assert fakes.seen["labels"] == {"cats", "dogs"}
    assert "6 of 20" in summary["note"]


def test_json_gets_sample_block(tmp_path, fakes):
    _, _, out = _run(tmp_path)
    data = json.loads((out / REPORT_FILES["json"]).read_text(encoding="utf-8"))
    assert data["sample"]["found"] == 5 and data["sample"]["sampled"] is False
    assert data["summary"]["dataset_grade"] == "B"


def test_work_folder_removed_on_success(tmp_path, fakes):
    _, work, _ = _run(tmp_path)
    assert not work.exists()


def test_failure_leaves_nothing_behind(tmp_path, fakes):
    fakes.spec["break_generate"] = True
    work, out = tmp_path / "work", tmp_path / "out"
    with pytest.raises(ReportBuildError):
        build_report_files(tmp_path / "a.zip", work, out)
    assert not work.exists()
    assert not out.exists()
    assert not (tmp_path / "out.partial").exists()


def test_failure_keeps_previous_report(tmp_path, fakes):
    _, _, out = _run(tmp_path)
    fakes.spec["break_generate"] = True
    with pytest.raises(ReportBuildError):
        build_report_files(tmp_path / "a.zip", tmp_path / "work", out)
    assert (out / REPORT_FILES["html"]).exists()


def test_existing_report_is_replaced(tmp_path, fakes):
    _, _, out = _run(tmp_path)
    (out / "stale.txt").write_text("old", encoding="utf-8")
    _run(tmp_path)
    assert not (out / "stale.txt").exists()