import pytest
from fastapi.testclient import TestClient
from api import app
import time

client = TestClient(app)

def test_auth_endpoints():
    unique_suffix = int(time.time())
    username = f"user_{unique_suffix}"
    email = f"user_{unique_suffix}@nydra.ai"

    # Test register
    reg_res = client.post("/api/v1/auth/register", json={
        "username": username,
        "email": email,
        "password": "SecurePassword123!"
    })
    print("Register status:", reg_res.status_code, reg_res.text)
    assert reg_res.status_code in [200, 201]

    # Test login with json
    login_res = client.post("/api/v1/auth/login", json={
        "username": username,
        "password": "SecurePassword123!"
    })
    print("Login status:", login_res.status_code, login_res.text)
    assert login_res.status_code == 200
    data = login_res.json()
    assert "access_token" in data
    assert "refresh_token" in data
