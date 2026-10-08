import stat
import zipfile

import pytest

from src.zip_safe import ExtractResult, UnsafeZipError, ZipLimits, extract_image_zip


def make_zip(path, entries, compress=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compress) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def labels(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def test_extracts_class_folders(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"cats/a.jpg": b"x", "cats/b.png": b"x", "dogs/c.jpg": b"x"})
    dest = tmp_path / "out"
    r = extract_image_zip(z, dest)
    assert isinstance(r, ExtractResult)
    assert r.extracted == 3
    assert r.dataset_root == dest
    assert labels(r.dataset_root) == ["cats", "dogs"]


def test_single_wrapper_folder_is_flattened(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"dataset/cats/a.jpg": b"x", "dataset/dogs/b.jpg": b"x"})
    r = extract_image_zip(z, tmp_path / "out")
    assert r.dataset_root.name == "dataset"
    assert labels(r.dataset_root) == ["cats", "dogs"]


def test_single_class_zip_is_not_over_flattened(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"cats/a.jpg": b"x"})
    dest = tmp_path / "out"
    r = extract_image_zip(z, dest)
    assert r.dataset_root == dest
    assert labels(r.dataset_root) == ["cats"]


@pytest.mark.parametrize(
    "name",
    ["../evil.jpg", "a/../../evil.jpg", "/abs/evil.jpg", "C:/evil.jpg", "..\\evil.jpg", "a\\..\\..\\evil.jpg"],
)
def test_hostile_paths_reject_whole_zip(tmp_path, name):
    z = make_zip(tmp_path / "z.zip", {"cats/ok.jpg": b"x", name: b"x"})
    dest = tmp_path / "out"
    with pytest.raises(UnsafeZipError):
        extract_image_zip(z, dest)
    assert not dest.exists()
    assert not (tmp_path / "out.partial").exists()
    assert not list(tmp_path.glob("**/evil.jpg"))


def test_skips_non_images_and_hidden(tmp_path):
    z = make_zip(
        tmp_path / "z.zip",
        {
            "cats/a.jpg": b"x",
            "readme.txt": b"x",
            "cats/notes.pdf": b"x",
            "__MACOSX/cats/._a.jpg": b"x",
            "cats/.DS_Store": b"x",
        },
    )
    r = extract_image_zip(z, tmp_path / "out")
    assert r.extracted == 1
    assert r.skipped == {"not_image": 2, "hidden": 2}


def test_symlink_entries_are_skipped(tmp_path):
    zp = tmp_path / "z.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("cats/a.jpg", b"x")
        info = zipfile.ZipInfo("cats/link.jpg")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "/etc/passwd")
    r = extract_image_zip(zp, tmp_path / "out")
    assert r.extracted == 1
    assert r.skipped.get("symlink") == 1


def test_too_many_files(tmp_path):
    z = make_zip(tmp_path / "z.zip", {f"cats/{i}.jpg": b"x" for i in range(4)})
    dest = tmp_path / "out"
    with pytest.raises(UnsafeZipError):
        extract_image_zip(z, dest, ZipLimits(max_files=3))
    assert not dest.exists()


def test_too_many_entries(tmp_path):
    z = make_zip(tmp_path / "z.zip", {f"cats/{i}.jpg": b"x" for i in range(3)})
    with pytest.raises(UnsafeZipError):
        extract_image_zip(z, tmp_path / "out", ZipLimits(max_entries=2))


def test_total_bytes_limit(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"cats/a.jpg": b"12345678", "cats/b.jpg": b"12345678"})
    with pytest.raises(UnsafeZipError):
        extract_image_zip(z, tmp_path / "out", ZipLimits(max_total_bytes=10))


def test_oversized_single_file_is_skipped(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"cats/a.jpg": b"123", "cats/big.jpg": b"1234567890"})
    r = extract_image_zip(z, tmp_path / "out", ZipLimits(max_file_bytes=5))
    assert r.extracted == 1
    assert r.skipped.get("too_large") == 1


def test_compression_bomb_rejected(tmp_path):
    z = make_zip(
        tmp_path / "z.zip",
        {"cats/bomb.jpg": b"\0" * 3_000_000, "cats/ok.jpg": b"x"},
        compress=zipfile.ZIP_DEFLATED,
    )
    dest = tmp_path / "out"
    with pytest.raises(UnsafeZipError):
        extract_image_zip(z, dest)
    assert not dest.exists()


def test_not_a_zip(tmp_path):
    p = tmp_path / "fake.zip"
    p.write_text("definitely not a zip")
    with pytest.raises(UnsafeZipError):
        extract_image_zip(p, tmp_path / "out")


def test_zip_without_images(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"readme.txt": b"x"})
    dest = tmp_path / "out"
    with pytest.raises(UnsafeZipError):
        extract_image_zip(z, dest)
    assert not dest.exists()


def test_windows_reserved_names_are_renamed(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"cats/CON.jpg": b"x"})
    r = extract_image_zip(z, tmp_path / "out")
    assert (r.dataset_root / "cats" / "_CON.jpg").exists()


def test_names_that_collide_after_cleaning_are_kept_apart(tmp_path):
    z = make_zip(tmp_path / "z.zip", {"cats/a?.jpg": b"1", "cats/a*.jpg": b"2"})
    r = extract_image_zip(z, tmp_path / "out")
    assert r.extracted == 2
    assert len(list((r.dataset_root / "cats").iterdir())) == 2


def test_reextract_replaces_old_contents(tmp_path):
    dest = tmp_path / "out"
    extract_image_zip(make_zip(tmp_path / "a.zip", {"cats/a.jpg": b"x"}), dest)
    extract_image_zip(make_zip(tmp_path / "b.zip", {"dogs/b.jpg": b"x"}), dest)
    assert not (dest / "cats").exists()
    assert (dest / "dogs" / "b.jpg").exists()
