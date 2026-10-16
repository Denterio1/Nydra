"""1.3: the direct upload path (POST /api/v1/upload) sanitizes client filenames."""
import pytest

UPLOAD = "/api/v1/upload"
CSV = b"a,b\n1,2\n3,4\n"


def _upload(registered_user, filename, content=CSV):
    client, _, token, _ = registered_user
    return client.post(
        UPLOAD,
        files={"file": (filename, content, "text/csv")},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_normal_name_is_unchanged(registered_user):
    r = _upload(registered_user, "sales_2026.csv")
    assert r.status_code in (200, 201), r.text
    assert r.json()["file"]["filename"] == "sales_2026.csv"


@pytest.mark.parametrize("sent", [
    "..\\..\\evil.csv",
    "../../evil.csv",
    "C:\\Windows\\evil.csv",
    "/etc/evil.csv",
])
def test_path_components_are_removed(registered_user, sent):
    r = _upload(registered_user, sent)
    assert r.status_code in (200, 201), r.text
    assert r.json()["file"]["filename"] == "evil.csv"


def test_unsafe_characters_are_replaced(registered_user):
    r = _upload(registered_user, "a<b>c.csv")
    assert r.status_code in (200, 201), r.text
    name = r.json()["file"]["filename"]
    assert "<" not in name and ">" not in name
    assert name.endswith(".csv")


def test_reserved_windows_name_rejected(registered_user):
    r = _upload(registered_user, "CON.csv")
    assert r.status_code == 400


def test_very_long_name_is_truncated_keeping_extension(registered_user):
    r = _upload(registered_user, "x" * 300 + ".csv")
    assert r.status_code in (200, 201), r.text
    name = r.json()["file"]["filename"]
    assert len(name) <= 150
    assert name.endswith(".csv")


def test_unsupported_extension_still_rejected(registered_user):
    r = _upload(registered_user, "payload.exe")
    assert r.status_code == 422


def test_name_with_only_dots_and_spaces_rejected_or_unsupported(registered_user):
    r = _upload(registered_user, " .. ")
    assert r.status_code in (400, 422)