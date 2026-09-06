"""
debug_job.py — verifies the StepTracker/JobStep datetime-crash fix (#1)

Flow: register -> login -> upload a small CSV -> create an "inspect" job
      -> poll until DONE/FAILED -> fetch result.

Success criteria:
  - No traceback mentioning model_dump() / datetime serialization
  - Job status progresses to "done" (not stuck at "pending"/"running" forever,
    not "failed")
  - /jobs/{job_id}/result returns 200 with actual content

Run from the project root (same place you ran debug_refresh.py):
    python debug_job.py
"""

import io
import sys
import time
import uuid

from fastapi.testclient import TestClient

from api import app  # noqa: E402  (adjust if your app object lives elsewhere)

# Use a fresh username every run so we never hit the 409-conflict noise
RUN_ID = uuid.uuid4().hex[:8]
USERNAME = f"debugjob_{RUN_ID}"
EMAIL = f"debugjob_{RUN_ID}@example.com"
PASSWORD = "DebugJob!12345"


def die(msg: str):
    print(f"\nFAILED: {msg}")
    sys.exit(1)


def main(client: TestClient):
    print("=== Register ===")
    r = client.post(
        "/api/v1/auth/register",
        json={"username": USERNAME, "email": EMAIL, "password": PASSWORD},
    )
    print(r.status_code, r.text[:300])
    if r.status_code not in (200, 201):
        die(f"register failed: {r.status_code} {r.text}")

    print("\n=== Login ===")
    r = client.post(
        "/api/v1/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
    )
    print(r.status_code, r.text[:300])
    if r.status_code != 200:
        die(f"login failed: {r.status_code} {r.text}")
    access_token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    print("\n=== Upload ===")
    csv_bytes = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n"
    files = {"file": ("debug_job_sample.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/api/v1/upload", headers=headers, files=files)
    print(r.status_code, r.text[:500])
    if r.status_code != 201:
        die(f"upload failed: {r.status_code} {r.text}")
    file_id = r.json()["file"]["file_id"]
    print("file_id:", file_id)

    print("\n=== Create job (goal=inspect) ===")
    r = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={"file_id": file_id, "goal": "inspect"},
    )
    print(r.status_code, r.text[:800])
    if r.status_code != 202:
        die(f"job creation failed (this is where the model_dump()/datetime "
            f"crash would show up): {r.status_code} {r.text}")
    job_id = r.json()["job_id"]
    print("job_id:", job_id)

    print("\n=== Poll job status ===")
    terminal_states = {"done", "failed", "cancelled"}
    final_status = None
    for i in range(30):  # up to ~30s
        r = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        if r.status_code != 200:
            die(f"poll failed: {r.status_code} {r.text}")
        body = r.json()
        status = body["status"]
        progress = body.get("progress")
        current_step = body.get("current_step")
        print(f"  [{i:02d}] status={status} progress={progress} step={current_step}")
        if status in terminal_states:
            final_status = status
            final_body = body
            break
        time.sleep(1)
    else:
        die("job never reached a terminal state after ~30s (possibly stuck)")

    print("\nFull final job status payload:")
    print(final_body)

    if final_status != "done":
        die(f"job ended in status={final_status!r}, error={final_body.get('error')!r}")

    print("\n=== Fetch job result ===")
    r = client.get(f"/api/v1/jobs/{job_id}/result", headers=headers)
    print(r.status_code, r.text[:1000])
    if r.status_code != 200:
        die(f"result fetch failed: {r.status_code} {r.text}")

    print("\nSUCCESS: job reached 'done' and result was fetched with no crash.")


if __name__ == "__main__":
    # IMPORTANT: TestClient must be used as a context manager, or FastAPI's
    # lifespan startup/shutdown events (which start/stop the JobQueue worker
    # pool) never fire, and every job will sit at "pending" forever.
    with TestClient(app) as client:
        main(client)