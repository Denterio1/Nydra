import pytest
from datetime import timedelta
from src.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    generate_user_id,
    get_user_stats
)

def test_password_hashing():
    pwd = "SecurePassword123$"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

def test_jwt_tokens():
    user_id = generate_user_id()
    data = {"sub": "user@example.com", "user_id": user_id}
    
    access_token = create_access_token(data, expires_delta=timedelta(minutes=15))
    refresh_token = create_refresh_token(data)
    
    assert access_token is not None
    assert refresh_token is not None
    
    # Verify access token
    payload = verify_token(access_token)
    assert payload is not None
    assert payload["sub"] == "user@example.com"
    assert payload["user_id"] == user_id
    
    # Verify refresh token
    refresh_payload = verify_token(refresh_token, is_refresh=True)
    assert refresh_payload is not None
    assert refresh_payload["sub"] == "user@example.com"
    assert refresh_payload.get("type") == "refresh"

def test_user_stats():
    stats = get_user_stats("test@example.com")
    assert isinstance(stats, dict)
    assert stats["name"] == "Test"
    assert stats["plan"] == "professional"
