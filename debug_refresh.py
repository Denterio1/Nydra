import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./debug_test.db"
os.environ["UPLOAD_DIR"] = "./debug_uploads"
os.environ["REPORT_DIR"] = "./debug_reports"
os.environ["SECRET_KEY"] = "debug-secret-key-for-testing-only"
os.environ["REFRESH_SECRET"] = "debug-refresh-secret-for-testing-only"
os.environ["WORKER_COUNT"] = "1"

import secrets
print("=== Direct secrets.token_urlsafe(64) sanity check ===")
t1 = secrets.token_urlsafe(64)
t2 = secrets.token_urlsafe(64)
print("t1:", t1)
print("t2:", t2)
print("Identical?", t1 == t2)
print()

from fastapi.testclient import TestClient
import api

with TestClient(api.app) as client:
    print("=== Register ===")
    r = client.post("/api/v1/auth/register", json={
        "username": "debuguser", "email": "debug@example.com", "password": "DebugPass123"
    })
    print(r.status_code, r.text[:200])

    print("=== Login #1 ===")
    r1 = client.post("/api/v1/auth/login", json={"username": "debuguser", "password": "DebugPass123"})
    print(r1.status_code)
    body1 = r1.json()
    print("refresh_token #1:", body1.get("refresh_token"))

    print("=== Login #2 ===")
    r2 = client.post("/api/v1/auth/login", json={"username": "debuguser", "password": "DebugPass123"})
    print(r2.status_code, r2.text[:500])
    if r2.status_code == 200:
        body2 = r2.json()
        print("refresh_token #2:", body2.get("refresh_token"))
        print("Same as #1?", body1.get("refresh_token") == body2.get("refresh_token"))

    api.job_queue._running = False