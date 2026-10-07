"""Resumable chunked uploads. Filesystem only: no DB, no API."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_READ = 1024 * 1024


class ChunkUploadError(Exception):
    """Client-safe message plus a suggested HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class ChunkLimits:
    max_total_bytes: int = 10 * 1024 ** 3
    min_chunk_bytes: int = 1024 * 1024
    max_chunk_bytes: int = 64 * 1024 * 1024
    max_chunks: int = 5000
    max_active_per_user: int = 3
    disk_headroom: float = 2.2
    stale_after_s: int = 24 * 3600


class ChunkStore:
    def __init__(self, root, limits: Optional[ChunkLimits] = None, allowed_extensions=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits or ChunkLimits()
        self.allowed = ({e.lower().lstrip(".") for e in allowed_extensions}
                        if allowed_extensions else None)

    # ---- helpers -------------------------------------------------------
    def _dir(self, upload_id) -> Path:
        if not isinstance(upload_id, str) or not _ID_RE.match(upload_id):
            raise ChunkUploadError("Upload not found.", 404)
        return self.root / upload_id

    def _load(self, upload_id, user_id):
        d = self._dir(upload_id)
        try:
            meta = json.loads((d / "meta.json").read_text("utf-8"))
        except (OSError, ValueError):
            raise ChunkUploadError("Upload not found.", 404)
        if meta.get("user_id") != user_id:
            raise ChunkUploadError("Upload not found.", 404)
        return d, meta

    @staticmethod
    def _received(d: Path) -> List[int]:
        out = []
        try:
            for p in d.iterdir():
                if p.suffix == ".part":
                    try:
                        out.append(int(p.stem))
                    except ValueError:
                        pass
        except OSError:
            pass
        return sorted(out)

    def _active_count(self, user_id) -> int:
        n = 0
        for d in self.root.iterdir():
            if d.is_dir() and _ID_RE.match(d.name):
                try:
                    if json.loads((d / "meta.json").read_text("utf-8")).get("user_id") == user_id:
                        n += 1
                except (OSError, ValueError):
                    pass
        return n

    # ---- API -----------------------------------------------------------
    def init(self, user_id, filename, total_size, chunk_size) -> dict:
        L = self.limits
        for v in (total_size, chunk_size):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ChunkUploadError("Invalid size.", 400)
        if total_size > L.max_total_bytes:
            raise ChunkUploadError("File too large. Maximum is %d MB." % (L.max_total_bytes // 1024 ** 2), 413)
        if not (L.min_chunk_bytes <= chunk_size <= L.max_chunk_bytes):
            raise ChunkUploadError("Invalid chunk size.", 400)
        n_chunks = -(-total_size // chunk_size)
        if n_chunks > L.max_chunks:
            raise ChunkUploadError("Chunk size too small for this file.", 400)
        ext = Path(str(filename)).suffix.lower().lstrip(".")
        if self.allowed is not None and ext not in self.allowed:
            raise ChunkUploadError("File type not allowed.", 400)
        if self._active_count(user_id) >= L.max_active_per_user:
            raise ChunkUploadError("Too many unfinished uploads. Finish or cancel one first.", 429)
        if shutil.disk_usage(self.root).free < total_size * L.disk_headroom:
            raise ChunkUploadError("Not enough free disk space on the server.", 507)
        uid = uuid.uuid4().hex
        d = self.root / uid
        d.mkdir()
        meta = {"upload_id": uid, "user_id": user_id, "filename": str(filename),
                "total_size": total_size, "chunk_size": chunk_size,
                "n_chunks": n_chunks, "created_at": time.time()}
        (d / "meta.json").write_text(json.dumps(meta), "utf-8")
        return meta

    def put_chunk(self, upload_id, user_id, index, data: bytes) -> int:
        d, meta = self._load(upload_id, user_id)
        n = meta["n_chunks"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n:
            raise ChunkUploadError("Chunk index out of range.", 400)
        expected = (meta["chunk_size"] if index < n - 1
                    else meta["total_size"] - meta["chunk_size"] * (n - 1))
        if len(data) != expected:
            raise ChunkUploadError("Chunk %d has the wrong size." % index, 400)
        part = d / ("%06d.part" % index)
        tmp = d / ("%06d.tmp-%s" % (index, uuid.uuid4().hex[:8]))
        try:
            tmp.write_bytes(data)
            os.replace(tmp, part)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise ChunkUploadError("Could not store the chunk.", 409)
        return len(self._received(d))

    def status(self, upload_id, user_id) -> dict:
        d, meta = self._load(upload_id, user_id)
        got = self._received(d)
        return {"upload_id": upload_id, "filename": meta["filename"],
                "total_size": meta["total_size"], "chunk_size": meta["chunk_size"],
                "n_chunks": meta["n_chunks"], "received": got,
                "complete": len(got) == meta["n_chunks"]}

    def assemble(self, upload_id, user_id, dest):
        """Blocking (run it in a thread). Returns (sha256_hex, size)."""
        d, meta = self._load(upload_id, user_id)
        n = meta["n_chunks"]
        got = set(self._received(d))
        missing = [i for i in range(n) if i not in got]
        if missing:
            raise ChunkUploadError("Missing %d chunk(s)." % len(missing), 409)
        lock = self.root / (upload_id + ".assembling")
        try:
            os.rename(d, lock)
        except OSError:
            raise ChunkUploadError("Upload is already being finalized.", 409)
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".partial")
        h = hashlib.sha256()
        total = 0
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as out:
                for i in range(n):
                    with open(lock / ("%06d.part" % i), "rb") as src:
                        while True:
                            b = src.read(_READ)
                            if not b:
                                break
                            h.update(b)
                            out.write(b)
                            total += len(b)
            if total != meta["total_size"]:
                raise ChunkUploadError("Assembled size does not match.", 400)
            os.replace(tmp, dest)
        except BaseException as e:
            try:
                tmp.unlink()
            except OSError:
                pass
            try:
                os.rename(lock, d)
            except OSError:
                pass
            if isinstance(e, OSError):
                raise ChunkUploadError("Could not assemble the file.", 500) from e
            raise
        shutil.rmtree(lock, ignore_errors=True)
        return h.hexdigest(), total

    def abort(self, upload_id, user_id) -> None:
        d, _ = self._load(upload_id, user_id)
        shutil.rmtree(d, ignore_errors=True)

    def cleanup_stale(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        removed = 0
        for d in list(self.root.iterdir()):
            if not d.is_dir():
                continue
            if not (_ID_RE.match(d.name) or d.name.endswith(".assembling")):
                continue
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if age > self.limits.stale_after_s:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed
