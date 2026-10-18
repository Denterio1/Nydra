import asyncio
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import Boolean, Column, DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src import quotas

Base = declarative_base()


class F(Base):
    __tablename__ = "f"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    size_bytes = Column(Integer)
    checksum = Column(String)
    is_deleted = Column(Boolean, default=False)


class J(Base):
    __tablename__ = "j"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    error = Column(String, nullable=True)
    finished_at = Column(DateTime, nullable=True)


def run(coro_fn):
    async def main():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(Base.metadata.create_all)
        sm = async_sessionmaker(eng, expire_on_commit=False)
        try:
            return await coro_fn(sm)
        finally:
            await eng.dispose()
    return asyncio.run(main())


MB = 1024 * 1024


def test_storage_counts_identical_checksum_once_and_skips_deleted():
    async def go(sm):
        async with sm() as db:
            db.add_all([
                F(user_id="u", size_bytes=10 * MB, checksum="a"),
                F(user_id="u", size_bytes=10 * MB, checksum="a"),
                F(user_id="u", size_bytes=5 * MB, checksum="b"),
                F(user_id="u", size_bytes=99 * MB, checksum="c", is_deleted=True),
                F(user_id="other", size_bytes=50 * MB, checksum="d"),
            ])
            await db.commit()
            return await quotas.user_storage_bytes(db, F, "u")
    assert run(go) == 15 * MB


def test_storage_enforce_over_and_under(monkeypatch):
    monkeypatch.setattr(quotas, "MAX_USER_STORAGE_MB", 20)
    async def go(sm):
        async with sm() as db:
            db.add(F(user_id="u", size_bytes=15 * MB, checksum="a"))
            await db.commit()
            await quotas.enforce_storage(db, F, "u", 5 * MB)  # exactly at limit: ok
            with pytest.raises(HTTPException) as e:
                await quotas.enforce_storage(db, F, "u", 5 * MB + 1)
            return e.value.status_code
    assert run(go) == 413


def test_storage_limit_zero_disables(monkeypatch):
    monkeypatch.setattr(quotas, "MAX_USER_STORAGE_MB", 0)
    async def go(sm):
        async with sm() as db:
            await quotas.enforce_storage(db, F, "u", 10 ** 12)
    run(go)


def test_job_cap_per_user(monkeypatch):
    monkeypatch.setattr(quotas, "MAX_ACTIVE_JOBS_PER_USER", 2)
    async def go(sm):
        async with sm() as db:
            db.add_all([J(user_id="u", status="pending"), J(user_id="u", status="running"),
                        J(user_id="u", status="done"), J(user_id="v", status="running")])
            await db.commit()
            with pytest.raises(HTTPException) as e:
                await quotas.enforce_job_caps(db, J, "u", 0, 1800)
            await quotas.enforce_job_caps(db, J, "v", 0, 1800)  # other user unaffected
            return e.value.status_code
    assert run(go) == 429


def test_stuck_old_job_does_not_block(monkeypatch):
    monkeypatch.setattr(quotas, "MAX_ACTIVE_JOBS_PER_USER", 1)
    async def go(sm):
        async with sm() as db:
            db.add(J(user_id="u", status="running", created_at=datetime.utcnow() - timedelta(hours=5)))
            await db.commit()
            await quotas.enforce_job_caps(db, J, "u", 0, 1800)
    run(go)


def test_queue_depth_cap(monkeypatch):
    monkeypatch.setattr(quotas, "MAX_QUEUE_DEPTH", 3)
    async def go(sm):
        async with sm() as db:
            await quotas.enforce_job_caps(db, J, "u", 2, 1800)
            with pytest.raises(HTTPException) as e:
                await quotas.enforce_job_caps(db, J, "u", 3, 1800)
            return e.value.status_code
    assert run(go) == 503


def test_reset_stale_jobs():
    async def go(sm):
        async with sm() as db:
            db.add_all([J(user_id="u", status="pending"), J(user_id="u", status="running"),
                        J(user_id="u", status="done")])
            await db.commit()
        n = await quotas.reset_stale_jobs(sm, J)
        async with sm() as db:
            rows = (await db.execute(select(J).order_by(J.id))).scalars().all()
        return n, [r.status for r in rows], rows[0].error, rows[0].finished_at is not None
    n, st, err, fin = run(go)
    assert n == 2 and st == ["failed", "failed", "done"]
    assert err == "Interrupted by server restart" and fin


def test_declared_size():
    class R:
        def __init__(s, h): s.headers = h
    assert quotas.declared_size(R({"content-length": "123"})) == 123
    assert quotas.declared_size(R({})) == 0
    assert quotas.declared_size(R({"content-length": "x"})) == 0
    assert quotas.declared_size(object()) == 0