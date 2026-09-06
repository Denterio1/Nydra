import pytest
import sys

def test_api_syntax():
    with open("api.py", "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    compiled = compile(code, "api.py", "exec")
    assert compiled is not None
