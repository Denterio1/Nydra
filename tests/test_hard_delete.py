import uuid
from src.hard_delete import safe_unlink, remove_job_artifacts


def test_safe_unlink_removes_file_inside_root(tmp_path):
    f = tmp_path / "a.csv"; f.write_text("x")
    assert safe_unlink(f, tmp_path) is True and not f.exists()


def test_safe_unlink_refuses_outside_root(tmp_path):
    root = tmp_path / "up"; root.mkdir()
    out = tmp_path / "other.txt"; out.write_text("x")
    assert safe_unlink(out, root) is False and out.exists()


def test_safe_unlink_refuses_traversal(tmp_path):
    root = tmp_path / "up"; root.mkdir()
    out = tmp_path / "other.txt"; out.write_text("x")
    assert safe_unlink(root / ".." / "other.txt", root) is False and out.exists()


def test_safe_unlink_missing_file_and_directory(tmp_path):
    assert safe_unlink(tmp_path / "nope", tmp_path) is False
    d = tmp_path / "d"; d.mkdir()
    assert safe_unlink(d, tmp_path) is False and d.exists()


def test_remove_job_artifacts_removes_only_that_job(tmp_path):
    rep, up = tmp_path / "reports", tmp_path / "uploads"
    jid, other = str(uuid.uuid4()), str(uuid.uuid4())
    (rep / f"{jid}_image_report").mkdir(parents=True)
    (rep / f"{jid}_image_report" / "image_report.html").write_text("x")
    (rep / f"{jid}_cleaned.csv").write_text("x")
    (rep / f"{other}_cleaned.csv").write_text("x")
    (up / "_clean" / jid).mkdir(parents=True)
    (up / "_report" / jid).mkdir(parents=True)
    assert remove_job_artifacts(jid, rep, up) == 4
    assert not list(rep.glob(f"{jid}_*"))
    assert (rep / f"{other}_cleaned.csv").exists()
    assert not (up / "_clean" / jid).exists()


def test_remove_job_artifacts_rejects_bad_ids(tmp_path):
    (tmp_path / "keep_1.txt").write_text("x")
    for bad in ("", "*", "../x", "a/b", "a\\b"):
        assert remove_job_artifacts(bad, tmp_path, tmp_path) == 0
    assert (tmp_path / "keep_1.txt").exists()


def test_remove_job_artifacts_nothing_to_remove(tmp_path):
    assert remove_job_artifacts(str(uuid.uuid4()), tmp_path / "r", tmp_path / "u") == 0


def test_job_data_route_requires_auth(registered_user):
    client, api_module, token, username = registered_user
    r = client.delete(f"/api/v1/jobs/{uuid.uuid4()}/data")
    assert r.status_code in (401, 403)


def test_job_data_route_unknown_job_404(registered_user):
    client, api_module, token, username = registered_user
    r = client.delete(f"/api/v1/jobs/{uuid.uuid4()}/data",
                      headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404