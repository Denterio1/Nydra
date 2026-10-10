import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src import vision_clean
from src.vision_clean import CleanLimitError, clean_archive_to_zip


def _install_fakes(monkeypatch):
    pkg = types.ModuleType("src.vision_nlp")
    pkg.__path__ = []
    loader = types.ModuleType("src.vision_nlp.image_loader")
    quality = types.ModuleType("src.vision_nlp.image_quality")
    cleaner = types.ModuleType("src.vision_nlp.image_cleaner")

    def load_images(path, verbose=False):
        files = sorted(str(p) for p in Path(path).rglob("*.jpg"))
        return pd.DataFrame({"file_path": files, "label": [Path(f).parent.name for f in files]})

    def analyze_dataset_quality(df, n_workers=2, verbose=False):
        return df, None

    def clean_dataset(df, output_dir, input_dir=".", resize=None):
        out = Path(output_dir)
        keep = df[~df["file_path"].str.contains("dup")].copy()
        paths, ok, fixes, errs = [], [], [], []
        for fp in keep["file_path"]:
            dst = out / fp
            dst.parent.mkdir(parents=True, exist_ok=True)
            if "broken" in fp:
                paths.append(""); ok.append(False); fixes.append(""); errs.append("unreadable")
            else:
                dst.write_bytes(Path(fp).read_bytes())
                paths.append(str(dst)); ok.append(True); fixes.append("brightness, contrast"); errs.append("")
        keep["cleaned_path"], keep["cleaning_success"] = paths, ok
        keep["fixes_applied"], keep["cleaning_error"] = fixes, errs
        (out / "before_after.csv").write_text("a,b\n")
        return keep

    loader.load_images, quality.analyze_dataset_quality, cleaner.clean_dataset = load_images, analyze_dataset_quality, clean_dataset
    for name, mod in (("src.vision_nlp", pkg), ("src.vision_nlp.image_loader", loader),
                      ("src.vision_nlp.image_quality", quality), ("src.vision_nlp.image_cleaner", cleaner)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr("src.vision_nlp", pkg, raising=False)


def _make_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        for n in ("ds/cats/a.jpg", "ds/cats/b.jpg", "ds/cats/a_dup.jpg", "ds/dogs/c.jpg", "ds/dogs/broken.jpg"):
            z.writestr(n, b"\xff\xd8" + n.encode())
        z.writestr("ds/notes.txt", "hi")


def test_cleaned_zip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _install_fakes(monkeypatch)
    _make_zip(tmp_path / "in.zip")
    s = clean_archive_to_zip("in.zip", "work", "reports/out.zip")
    assert s["images_in"] == 5 and s["cleaned"] == 3 and s["failed"] == 1 and s["removed"] == 1
    assert s["fixes"] == {"brightness": 3, "contrast": 3}
    names = sorted(zipfile.ZipFile("reports/out.zip").namelist())
    assert names == ["_reports/before_after.csv", "cats/a.jpg", "cats/b.jpg", "dogs/c.jpg"]
    assert not Path("work").exists() and not Path("reports/out.zip.partial").exists()


def test_limit_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _install_fakes(monkeypatch)
    _make_zip(tmp_path / "in.zip")
    with pytest.raises(CleanLimitError):
        clean_archive_to_zip("in.zip", "work", "out.zip", max_images=2)
    assert not Path("work").exists() and not Path("out.zip").exists()