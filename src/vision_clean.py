"""Clean every image of an uploaded archive (zip / tar) and return a zip of the cleaned dataset."""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

CLEAN_MAX_IMAGES = 3000
_REPORT_FILES = ("before_after.csv", "cleaning_report.json", "cleaning_report.md")


class CleanLimitError(Exception):
    """Client-safe message."""


def _rel_arg(p: Path) -> str:
    # image_cleaner mirrors input paths under its output dir; relative paths are what we tested with
    try:
        return os.path.relpath(p)
    except ValueError:  # different drive on Windows
        return str(p)


def clean_archive_to_zip(archive, work_dir, out_zip, max_images: int = CLEAN_MAX_IMAGES, limits=None) -> dict:
    t = time.time()
    archive, work_dir, out_zip = Path(archive), Path(work_dir), Path(out_zip)
    from src.zip_safe import ZipLimits, extract_image_zip
    from src.tar_safe import extract_image_tar

    limits = limits or ZipLimits(max_entries=100_000, max_files=50_000,
                                 max_file_bytes=100 * 1024 * 1024, max_total_bytes=10 * 1024 ** 3)
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        extract = extract_image_zip if archive.suffix.lower() == ".zip" else extract_image_tar
        ex = extract(archive, work_dir / "in", limits)
        if ex.extracted > max_images:
            raise CleanLimitError("Cleaning is limited to %d images for now (this archive has %d)."
                                  % (max_images, ex.extracted))
        root_arg = _rel_arg(ex.dataset_root)
        out_arg = _rel_arg(work_dir / "out")

        with contextlib.redirect_stdout(io.StringIO()):
            from src.vision_nlp.image_loader import load_images
            from src.vision_nlp.image_quality import analyze_dataset_quality
            from src.vision_nlp.image_cleaner import clean_dataset
            df = load_images(root_arg, verbose=False)
            dfq, _ = analyze_dataset_quality(df, n_workers=2, verbose=False)
            res = clean_dataset(dfq, out_arg, input_dir=".", resize=None)

        ok = res["cleaning_success"].astype(bool)
        n_ok = int(ok.sum())
        if n_ok == 0:
            raise CleanLimitError("No images could be cleaned.")

        fixes = Counter()
        for s in res.loc[ok, "fixes_applied"].fillna("").astype(str):
            for f in (x.strip() for x in s.split(",")):
                if f:
                    fixes[f] += 1

        out_zip.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = out_zip.with_name(out_zip.name + ".partial")
        written = 0
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as zf:
            for fp, cp in zip(res.loc[ok, "file_path"], res.loc[ok, "cleaned_path"]):
                cp = Path(str(cp))
                if not cp.is_file():
                    continue
                try:
                    arc = Path(os.path.normpath(str(fp))).relative_to(Path(os.path.normpath(root_arg))).as_posix()
                except ValueError:
                    arc = Path(str(fp)).name
                zf.write(cp, arc)
                written += 1
            for name in _REPORT_FILES:
                rp = Path(out_arg) / name
                if rp.is_file():
                    zf.write(rp, "_reports/" + name, compress_type=zipfile.ZIP_DEFLATED)
        if written == 0:
            tmp_zip.unlink(missing_ok=True)
            raise CleanLimitError("No cleaned images were produced.")
        os.replace(tmp_zip, out_zip)

        return {"images_in": int(len(dfq)), "cleaned": written,
                "failed": int((~ok).sum()), "removed": int(len(dfq) - len(res)),
                "fixes": dict(fixes), "zip_mb": round(out_zip.stat().st_size / 1048576, 2),
                "seconds": round(time.time() - t, 1)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)