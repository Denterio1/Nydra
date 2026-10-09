"""Safe extraction of images from .tar / .tar.gz / .tgz archives (same rules as zip_safe)."""
from __future__ import annotations

import re
import shutil
import tarfile
from pathlib import Path
from typing import Optional, Union

from src.zip_safe import ExtractResult, UnsafeZipError, ZipLimits, _find_dataset_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic"}
_BAD = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul"} | {"com%d" % i for i in range(1, 10)} | {"lpt%d" % i for i in range(1, 10)}


def _clean_part(part: str) -> Optional[str]:
    part = _BAD.sub("_", part).strip(" .")
    if not part:
        return None
    if part.split(".")[0].lower() in _RESERVED:
        part = "_" + part
    return part[:120]


def _extract(tar_path: Path, stage: Path, lim: ZipLimits):
    archive_size = max(tar_path.stat().st_size, 1)
    stage_real = stage.resolve()
    skipped: dict = {}
    extracted = total = entries = 0

    def skip(k):
        skipped[k] = skipped.get(k, 0) + 1

    with tarfile.open(tar_path, mode="r|*") as tf:
        for m in tf:
            entries += 1
            if entries > lim.max_entries:
                raise UnsafeZipError("The archive has too many entries.")
            raw = m.name.replace("\\", "/")
            if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
                raise UnsafeZipError("The archive contains absolute paths.")
            parts = [p for p in raw.split("/") if p not in ("", ".")]
            if any(p == ".." for p in parts):
                raise UnsafeZipError("The archive contains path traversal entries.")
            if not m.isfile():
                if not m.isdir():
                    skip("links_or_special")
                continue
            if any(p.startswith(".") or p == "__MACOSX" for p in parts):
                skip("hidden")
                continue
            if Path(parts[-1]).suffix.lower() not in IMAGE_EXTS:
                skip("non_image")
                continue
            if m.size > lim.max_file_bytes:
                skip("too_large")
                continue
            if extracted + 1 > lim.max_files:
                raise UnsafeZipError("The archive has too many images.")
            clean = [c for c in (_clean_part(p) for p in parts) if c]
            if not clean:
                skip("bad_name")
                continue
            target = stage.joinpath(*clean)
            n = 0
            while target.exists():
                n += 1
                target = target.with_name("%s_%d%s" % (Path(clean[-1]).stem, n, Path(clean[-1]).suffix))
            if stage_real not in target.resolve().parents:
                raise UnsafeZipError("The archive contains path traversal entries.")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                skip("corrupt")
                continue
            written = 0
            too_big = False
            with open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > lim.max_file_bytes:
                        too_big = True
                        break
                    out.write(chunk)
            if too_big:
                target.unlink(missing_ok=True)
                skip("too_large")
                continue
            total += written
            if total > lim.max_total_bytes:
                raise UnsafeZipError("The archive is too large when extracted.")
            if total > lim.ratio_min_size and total > lim.max_ratio * archive_size:
                raise UnsafeZipError("The archive expands too much (possible archive bomb).")
            extracted += 1
    if extracted == 0:
        raise UnsafeZipError("No images found in the archive.")
    return extracted, total, skipped


def extract_image_tar(tar_path: Union[str, Path], dest_dir: Union[str, Path],
                      limits: Optional[ZipLimits] = None) -> ExtractResult:
    """Extract only the images from a .tar/.tar.gz/.tgz into dest_dir (replaced if it exists)."""
    lim = limits or ZipLimits()
    tar_path, dest_dir = Path(tar_path), Path(dest_dir)
    stage = dest_dir.with_name(dest_dir.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        extracted, total, skipped = _extract(tar_path, stage, lim)
    except tarfile.TarError:
        shutil.rmtree(stage, ignore_errors=True)
        raise UnsafeZipError("Invalid or unreadable archive.")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    shutil.rmtree(dest_dir, ignore_errors=True)
    stage.replace(dest_dir)
    return ExtractResult(dataset_root=_find_dataset_root(dest_dir), extracted=extracted,
                         total_bytes=total, skipped=skipped)