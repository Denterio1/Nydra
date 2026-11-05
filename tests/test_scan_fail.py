"""A failed big-table scan must abort the job, not fall back to a full pandas load."""
import asyncio

import numpy as np
import pandas as pd
import pytest


class _Step:
    def __init__(self, name):
        self.name = name


class _Tracker:
    def __init__(self, names):
        self.job_id = "jobfail"
        self.steps = [_Step(n) for n in names]

    async def start_step(self, db):
        pass

    async def finish_step(self, db, warnings=None):
        pass

    async def emit_progress(self, progress, message):
        pass


class _FakeDoctor:
    pass


def test_scan_error_fails_job_and_never_reads_full_file(tmp_path, monkeypatch):
    import api
    from src import scan_engine
    from src.api_schemas import JobGoal
    import src.core.agent as agent_mod

    csv = tmp_path / "big.csv"
    pd.DataFrame({"a": np.arange(500), "b": np.arange(500) * 2}).to_csv(csv, index=False)

    monkeypatch.setenv("SCAN_MIN_MB", "0")
    monkeypatch.setattr(agent_mod, "Nydra", _FakeDoctor)

    async def _no_warning(*a, **k):
        pass

    monkeypatch.setattr(api.ws_manager, "send_warning", _no_warning)

    msg = "The file could not be parsed as a table (test)."

    def _boom(*a, **k):
        raise scan_engine.ScanError(msg)

    monkeypatch.setattr(scan_engine, "scan_table", _boom)

    calls = []
    real_read_csv = pd.read_csv

    def _spy(*a, **k):
        calls.append(a)
        return real_read_csv(*a, **k)

    monkeypatch.setattr(pd, "read_csv", _spy)

    tracker = _Tracker(["Loading", "DNA"])
    with pytest.raises(Exception) as ei:
        asyncio.run(api._run_agent(JobGoal.DNA, csv, {}, tracker, None))

    assert type(ei.value).__name__ == "ScanError"
    assert msg in str(ei.value)
    assert calls == [], "full pandas read happened after a failed scan"