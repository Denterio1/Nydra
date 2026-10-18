"""Per-user quotas and job caps (launch plan task 1.4b).

Models and session factories are passed in, so this module never imports api.py.
Limits are read at call time from module globals (env-configurable).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import func, select, update


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


MAX_ACTIVE_JOBS_PER_USER = _env_int("MAX_ACTIVE_JOBS_PER_USER", 2)
MAX_QUEUE_DEPTH = _env_int("MAX_QUEUE_DEPTH", 20)
MAX_USER_STORAGE_MB = _env_int("MAX_USER_STORAGE_MB", 2048)
ACTIVE_STATUSES = ("pending", "running")
STALE_GRACE_S = 600


def declared_size(request) -> int:
    """Content-Length header as int; 0 when missing or invalid."""
    try:
        return max(0, int(request.headers.get("content-length", 0)))
    except (TypeError, ValueError, AttributeError):
        return 0


async def user_storage_bytes(db, file_model, user_id: str) -> int:
    """Stored bytes of a user's non-deleted files; identical files (same checksum) count once."""
    sub = (
        select(file_model.checksum, func.max(file_model.size_bytes).label("sz"))
        .where(file_model.user_id == user_id, file_model.is_deleted == False)  # noqa: E712
        .group_by(file_model.checksum)
        .subquery()
    )
    res = await db.execute(select(func.coalesce(func.sum(sub.c.sz), 0)))
    return int(res.scalar_one() or 0)


async def enforce_storage(db, file_model, user_id: str, incoming_bytes: int) -> None:
    limit = MAX_USER_STORAGE_MB * 1024 * 1024
    if limit <= 0:
        return
    used = await user_storage_bytes(db, file_model, user_id)
    if used + max(0, incoming_bytes) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Storage quota exceeded ({MAX_USER_STORAGE_MB} MB per account). Delete old files and try again.",
        )


async def active_job_count(db, job_model, user_id: str, max_age_s: int) -> int:
    """Pending/running jobs created recently; older rows are treated as stuck and ignored."""
    since = datetime.utcnow() - timedelta(seconds=max_age_s + STALE_GRACE_S)
    res = await db.execute(
        select(func.count()).select_from(job_model).where(
            job_model.user_id == user_id,
            job_model.status.in_(ACTIVE_STATUSES),
            job_model.created_at >= since,
        )
    )
    return int(res.scalar_one() or 0)


async def enforce_job_caps(db, job_model, user_id: str, queue_depth: int, max_age_s: int) -> None:
    if MAX_QUEUE_DEPTH > 0 and queue_depth >= MAX_QUEUE_DEPTH:
        raise HTTPException(status_code=503, detail="Server is busy. Please try again in a few minutes.")
    if MAX_ACTIVE_JOBS_PER_USER > 0:
        n = await active_job_count(db, job_model, user_id, max_age_s)
        if n >= MAX_ACTIVE_JOBS_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=f"You already have {n} active jobs (limit {MAX_ACTIVE_JOBS_PER_USER}). Wait for one to finish.",
            )


async def reset_stale_jobs(session_factory, job_model) -> int:
    """At startup: jobs left pending/running by a previous process can never finish; fail them."""
    async with session_factory() as db:
        res = await db.execute(
            update(job_model)
            .where(job_model.status.in_(ACTIVE_STATUSES))
            .values(status="failed", error="Interrupted by server restart", finished_at=datetime.utcnow())
        )
        await db.commit()
        return int(res.rowcount or 0)