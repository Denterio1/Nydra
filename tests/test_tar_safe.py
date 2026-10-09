import io
import tarfile

import pytest

from src.tar_safe import extract_image_tar
from src.zip_safe import UnsafeZipError, ZipLimits


def _make(path, entries, mode="w:gz"):
    with tarfile.open(path, mode) as tf:
        for name, data in entries:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


def test_basic_tgz_and_wrapper(tmp_path):
    a = _make(tmp_path / "a.tgz", [("ds/cats/1.jpg", b"x" * 50), ("ds/dogs/2.png", b"y" * 50), ("ds/readme.txt", b"hi")])
    r = extract_image_tar(a, tmp_path / "out")
    assert r.extracted == 2 and r.skipped.get("non_image") == 1
    assert sorted(p.name for p in r.dataset_root.iterdir()) == ["cats", "dogs"]
    assert not (tmp_path / "out.partial").exists()


def test_plain_tar(tmp_path):
    a = _make(tmp_path / "a.tar", [("cats/1.jpg", b"x"), ("dogs/2.jpg", b"y")], mode="w")
    assert extract_image_tar(a, tmp_path / "o").extracted == 2


@pytest.mark.parametrize("name", ["../evil.jpg", "a/../../evil.jpg", "/abs/evil.jpg", "C:/evil.jpg", "..\\evil.jpg"])
def test_rejects_traversal(tmp_path, name):
    a = _make(tmp_path / "a.tgz", [("ok/1.jpg", b"x"), (name, b"x")])
    with pytest.raises(UnsafeZipError):
        extract_image_tar(a, tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()
    assert not (tmp_path / "evil.jpg").exists()


def test_symlink_and_hidden_skipped(tmp_path):
    p = tmp_path / "a.tgz"
    with tarfile.open(p, "w:gz") as tf:
        ti = tarfile.TarInfo("cats/link.jpg"); ti.type = tarfile.SYMTYPE; ti.linkname = "/etc/passwd"
        tf.addfile(ti)
        for n in ("cats/.hidden.jpg", "__MACOSX/x.jpg", "cats/1.jpg"):
            t = tarfile.TarInfo(n); t.size = 1
            tf.addfile(t, io.BytesIO(b"x"))
    r = extract_image_tar(p, tmp_path / "o")
    assert r.extracted == 1 and r.skipped["links_or_special"] == 1 and r.skipped["hidden"] == 2


def test_limits(tmp_path):
    a = _make(tmp_path / "a.tgz", [("c/%d.jpg" % i, b"x") for i in range(5)])
    with pytest.raises(UnsafeZipError):
        extract_image_tar(a, tmp_path / "o", ZipLimits(max_files=3))
    with pytest.raises(UnsafeZipError):
        extract_image_tar(a, tmp_path / "o", ZipLimits(max_entries=2))
    b = _make(tmp_path / "b.tgz", [("c/1.jpg", b"x" * 100)])
    with pytest.raises(UnsafeZipError):  # the only image is over the per-file limit -> nothing left
        extract_image_tar(b, tmp_path / "o2", ZipLimits(max_file_bytes=10))


def test_no_images_and_bad_archive(tmp_path):
    a = _make(tmp_path / "a.tgz", [("notes.txt", b"hi")])
    with pytest.raises(UnsafeZipError):
        extract_image_tar(a, tmp_path / "o")
    bad = tmp_path / "bad.tgz"
    bad.write_bytes(b"this is not an archive" * 10)
    with pytest.raises(UnsafeZipError):
        extract_image_tar(bad, tmp_path / "o2")


def test_bomb_ratio(tmp_path):
    a = _make(tmp_path / "a.tgz", [("c/1.jpg", b"\0" * (3 * 1024 * 1024))])
    with pytest.raises(UnsafeZipError):
        extract_image_tar(a, tmp_path / "o", ZipLimits(max_ratio=10.0))