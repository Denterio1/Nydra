"""Safe extraction of image datasets from user-uploaded zips.

Rules: traversal/absolute paths reject the whole zip; symlinks, hidden files,
encrypted entries and non-images are skipped; limits are enforced on bytes
actually written (headers can lie). Extraction happens in "<dest>.partial" and
is swapped into place only on success.
"""
from __future__ import annotations

import re
import shutil
import stat
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

IMAGE_EXTENSIONS = frozenset(
    {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif", "heic"}
)

_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_DRIVE = re.compile(r"^[A-Za-z]:")
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_CHUNK = 1024 * 1024


class UnsafeZipError(Exception):
    """Invalid, hostile, or over-limit zip. The message is safe to show clients."""


@dataclass(frozen=True)
class ZipLimits:
    max_entries: int = 20_000
    max_files: int = 2_000
    max_file_bytes: int = 25 * 1024 * 1024
    max_total_bytes: int = 500 * 1024 * 1024
    max_ratio: float = 100.0
    ratio_min_size: int = 1024 * 1024


@dataclass
class ExtractResult:
    dataset_root: Path
    extracted: int
    total_bytes: int
    skipped: Dict[str, int] = field(default_factory=dict)


def _clean_part(part: str) -> Optional[str]:
    part = _BAD_CHARS.sub("_", part).strip(" .")
    if not part:
        return None
    if part.split(".")[0].upper() in _RESERVED:
        part = "_" + part
    if len(part) > 120:
        stem, dot, ext = part.rpartition(".")
        part = (stem[:100] + dot + ext) if dot and len(ext) <= 8 else part[:120]
    return part


def _unique_target(stage_real: Path, clean: list) -> Path:
    target = stage_real.joinpath(*clean)
    n = 1
    while target.exists():
        stem, dot, ext = clean[-1].rpartition(".")
        target = stage_real.joinpath(*clean[:-1], f"{stem}_{n}{dot}{ext}")
        n += 1
    if stage_real not in target.resolve().parents:
        raise UnsafeZipError("The zip contains path traversal entries.")
    return target


def _find_dataset_root(root: Path) -> Path:
    """Descend through a single wrapper folder so labels are the class folders."""
    cur = root
    while True:
        children = list(cur.iterdir())
        if (
            len(children) == 1
            and children[0].is_dir()
            and any(c.is_dir() for c in children[0].iterdir())
        ):
            cur = children[0]
        else:
            return cur


def _extract(zip_path: Path, stage: Path, lim: ZipLimits):
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise UnsafeZipError("The file is not a valid zip archive.") from None

    skipped: Dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    extracted = 0
    total = 0
    stage_real = stage.resolve()

    with zf:
        infos = zf.infolist()
        if len(infos) > lim.max_entries:
            raise UnsafeZipError("The zip contains too many entries.")

        for info in infos:
            if info.is_dir():
                continue
            raw = info.filename.replace("\\", "/")
            if raw.startswith("/") or _DRIVE.match(raw):
                raise UnsafeZipError("The zip contains absolute paths.")
            parts = [p for p in raw.split("/") if p not in ("", ".")]
            if not parts:
                continue
            if ".." in parts:
                raise UnsafeZipError("The zip contains path traversal entries.")
            if parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
                skip("hidden")
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                skip("symlink")
                continue
            if info.flag_bits & 0x1:
                skip("encrypted")
                continue
            name = parts[-1]
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in IMAGE_EXTENSIONS:
                skip("not_image")
                continue
            if info.file_size > lim.max_file_bytes:
                skip("too_large")
                continue
            if (
                info.file_size > lim.ratio_min_size
                and info.compress_size > 0
                and info.file_size / info.compress_size > lim.max_ratio
            ):
                raise UnsafeZipError("The zip looks like a compression bomb.")
            clean = [_clean_part(p) for p in parts]
            if any(c is None for c in clean):
                skip("bad_name")
                continue
            if extracted >= lim.max_files:
                raise UnsafeZipError(f"The zip has more than {lim.max_files} images.")

            target = _unique_target(stage_real, clean)
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            try:
                with zf.open(info) as src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(_CHUNK)
                        if not chunk:
                            break
                        written += len(chunk)
                        total += len(chunk)
                        if written > lim.max_file_bytes or total > lim.max_total_bytes:
                            raise UnsafeZipError("The zip is too large once extracted.")
                        out.write(chunk)
            except (zipfile.BadZipFile, zlib.error):
                target.unlink(missing_ok=True)
                skip("corrupt")
                continue
            extracted += 1

    if extracted == 0:
        raise UnsafeZipError("No images were found in the zip.")
    return extracted, total, skipped


def extract_image_zip(
    zip_path: Union[str, Path],
    dest_dir: Union[str, Path],
    limits: Optional[ZipLimits] = None,
) -> ExtractResult:
    """Extract only the images from a zip into dest_dir (replaced if it exists)."""
    lim = limits or ZipLimits()
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    stage = dest_dir.with_name(dest_dir.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        extracted, total, skipped = _extract(zip_path, stage, lim)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    shutil.rmtree(dest_dir, ignore_errors=True)
    stage.replace(dest_dir)
    return ExtractResult(
        dataset_root=_find_dataset_root(dest_dir),
        extracted=extracted,
        total_bytes=total,
        skipped=skipped,
    )
