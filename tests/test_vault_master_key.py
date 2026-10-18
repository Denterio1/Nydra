"""NYDRA_VAULT_KEY override for the vault master key."""
import platform
import sys


def _derive(monkeypatch, node="host-a", version="3.11.9 (x)"):
    monkeypatch.setattr(platform, "node", lambda: node)
    monkeypatch.setattr(sys, "version", version)
    from src.security_vault import MasterKeyDeriver
    return MasterKeyDeriver().derive()


def test_env_key_survives_hostname_and_python_change(monkeypatch):
    monkeypatch.setenv("NYDRA_VAULT_KEY", "a-long-random-secret")
    k1 = _derive(monkeypatch, "container-1", "3.11.9 (x)")
    k2 = _derive(monkeypatch, "container-2", "3.11.10 (y)")
    assert k1 == k2


def test_different_env_keys_give_different_keys(monkeypatch):
    monkeypatch.setenv("NYDRA_VAULT_KEY", "secret-one")
    k1 = _derive(monkeypatch)
    monkeypatch.setenv("NYDRA_VAULT_KEY", "secret-two")
    k2 = _derive(monkeypatch)
    assert k1 != k2


def test_without_env_key_old_behaviour_kept(monkeypatch):
    monkeypatch.delenv("NYDRA_VAULT_KEY", raising=False)
    k1 = _derive(monkeypatch, "host-a")
    k2 = _derive(monkeypatch, "host-b")
    assert k1 != k2
    assert k1 == _derive(monkeypatch, "host-a")


def test_empty_env_key_falls_back(monkeypatch):
    monkeypatch.delenv("NYDRA_VAULT_KEY", raising=False)
    base = _derive(monkeypatch)
    monkeypatch.setenv("NYDRA_VAULT_KEY", "")
    assert _derive(monkeypatch) == base


def test_env_key_differs_from_fingerprint_key(monkeypatch):
    monkeypatch.delenv("NYDRA_VAULT_KEY", raising=False)
    base = _derive(monkeypatch)
    monkeypatch.setenv("NYDRA_VAULT_KEY", "something")
    assert _derive(monkeypatch) != base