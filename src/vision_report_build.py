"""Build the Vision Lab dataset report files (HTML / JSON / Markdown).

Flow: extract the archive safely -> stratified sample (max REPORT_MAX_IMAGES)
-> load + quality score -> adapter (column names image_report.py reads)
-> ``generate_report`` -> three fixed-name files in ``out_dir``.

Heavy modules are imported inside the function, so importing this file is cheap.
The work folder is always removed. ``out_dir`` is replaced only on success.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path

REPORT_MAX_IMAGES = 150
REPORT_FILES = {"html": "image_report.html", "json": "image_report.json", "md": "image_report.md"}


class ReportBuildError(Exception):
    """Client-safe message."""


def _rel_arg(p: Path) -> str:
    # same convention as vision_clean: relative paths are what the loader was tested with
    try:
        return os.path.relpath(p)
    except ValueError:  # different drive on Windows
        return str(p)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sample_folder(root: Path, dest: Path, max_images: int):
    """Return (folder_to_analyze, files_found, files_used).

    At or below the cap the original folder is used. Above it, a stratified sample
    is hardlinked (copy fallback) into ``dest`` keeping the class folder layout,
    so the loader derives the same labels.
    """
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    n_found = len(files)
    if n_found <= max_images:
        return root, n_found, n_found

    import pandas as pd
    from src.sampling import stratified_sample

    df = pd.DataFrame({
        "path": [str(p) for p in files],
        "label": [p.parent.name if p.parent != root else "(unlabeled)" for p in files],
    })
    picked = stratified_sample(df, max_images, label_col="label", seed=0)
    for s in picked["path"]:
        src = Path(s)
        _link_or_copy(src, dest / src.relative_to(root))
    return dest, n_found, len(picked)


def build_report_files(archive, work_dir, out_dir, max_images: int = REPORT_MAX_IMAGES, limits=None) -> dict:
    t = time.time()
    archive, work_dir, out_dir = Path(archive), Path(work_dir), Path(out_dir)

    from src.zip_safe import ZipLimits, extract_image_zip
    from src.tar_safe import extract_image_tar

    limits = limits or ZipLimits(max_entries=100_000, max_files=50_000,
                                 max_file_bytes=100 * 1024 * 1024, max_total_bytes=10 * 1024 ** 3)
    shutil.rmtree(work_dir, ignore_errors=True)
    partial = out_dir.with_name(out_dir.name + ".partial")
    try:
        extract = extract_image_zip if archive.suffix.lower() == ".zip" else extract_image_tar
        ex = extract(archive, work_dir / "in", limits)
        folder, n_found, n_used = _sample_folder(Path(ex.dataset_root), work_dir / "sample", max_images)

        gen_dir = work_dir / "report"
        with contextlib.redirect_stdout(io.StringIO()):
            from src.vision_nlp.image_loader import load_images
            from src.vision_nlp.image_quality import analyze_dataset_quality
            from src.vision_nlp.image_report import generate_report
            from src.vision_report_adapter import prepare_report_frames

            df = load_images(_rel_arg(folder), verbose=False)
            if len(df) == 0:
                raise ReportBuildError("No images could be read from this archive.")
            dfq, _ = analyze_dataset_quality(df, n_workers=2, verbose=False)
            df2, q2 = prepare_report_frames(df, dfq)
            paths = generate_report(df2, output_dir=str(gen_dir), formats=("html", "json", "md"),
                                    quality_df=q2)

        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        for fmt, name in REPORT_FILES.items():
            produced = paths.get(fmt)
            if not produced or not Path(produced).exists():
                raise ReportBuildError("The report could not be generated (%s file missing)." % fmt)
            shutil.copy2(produced, partial / name)

        sampled = n_found > n_used
        note = ("Analyzed %d of %d images (stratified by class)." % (n_used, n_found)) if sampled \
            else "Analyzed all %d images." % n_found
        jpath = partial / REPORT_FILES["json"]
        data = json.loads(jpath.read_text(encoding="utf-8"))
        data["sample"] = {"found": n_found, "analyzed": n_used, "sampled": sampled, "note": note}
        jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(partial, out_dir)

        summ = data.get("summary", {}) or {}
        return {
            "images_found": n_found,
            "images_analyzed": n_used,
            "sampled": sampled,
            "note": note,
            "avg_score": summ.get("avg_quality_score"),
            "grade": summ.get("dataset_grade"),
            "formats": list(REPORT_FILES),
            "seconds": round(time.time() - t, 1),
        }
    finally:
        shutil.rmtree(partial, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)