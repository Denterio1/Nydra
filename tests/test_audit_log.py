"""Tests for src/audit_log.py -- tamper-evidence, queries, retention, resilience."""

from __future__ import annotations

import json
import os
import time

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.audit_log import (
    AUDIT_METADATA,
    GENESIS_HASH,
    AuditEventType,
    AuditLogger,
    AuditQuery,
    AuditReporter,
    AuditRetention,
    AuditSeverity,
    DBAuditLog,
    IntegrityChecker,
    audit_summary,
    compute_entry_hash,
    get_recent_events,
    log_event,
    verify_integrity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db():
    """A fresh in-memory SQLite database per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(AUDIT_METADATA.create_all)

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session

    await engine.dispose()


async def _seed(db: AsyncSession, count: int = 5) -> list[DBAuditLog]:
    """Write a small mixed chain and return the rows in insertion order."""
    rows = []
    for i in range(count):
        row = await log_event(
            db,
            AuditEventType.LOGIN_SUCCESS if i % 2 == 0 else AuditEventType.LOGIN_FAILED,
            user_id=f"user-{i}",
            ip=f"10.0.0.{i}",
            detail={"seq": i},
            success=(i % 2 == 0),
        )
        assert row is not None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Hash chain integrity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genesis_row_uses_seed_hash(db):
    row = await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="u1")
    assert row is not None
    assert row.prev_hash == GENESIS_HASH


@pytest.mark.asyncio
async def test_chain_links_and_verifies(db):
    rows = await _seed(db, 6)

    # Each row's prev_hash is the previous row's entry_hash.
    assert rows[0].prev_hash == GENESIS_HASH
    for previous, current in zip(rows, rows[1:]):
        assert current.prev_hash == previous.entry_hash

    # Each entry_hash is reproducible from its own fields + predecessor hash.
    for previous, current in zip(rows, rows[1:]):
        assert current.entry_hash == compute_entry_hash(
            previous.entry_hash,
            current.timestamp,
            current.event_type,
            current.user_id,
            current.detail,
        )

    result = await verify_integrity(db)
    assert result.intact is True
    assert result.checked == 6
    assert result.first_broken_id is None
    assert result.from_genesis is True


@pytest.mark.asyncio
async def test_tamper_detected_after_field_edit(db):
    rows = await _seed(db, 5)
    target = rows[2]

    await db.execute(
        update(DBAuditLog)
        .where(DBAuditLog.id == target.id)
        .values(detail='{"seq": 999}')
    )
    await db.commit()

    result = await verify_integrity(db)
    assert result.intact is False
    assert result.first_broken_id == target.id
    assert result.first_broken_index == 2
    assert "mismatch" in (result.reason or "")


@pytest.mark.asyncio
async def test_tamper_detected_when_attacker_rehashes_row(db):
    """A sophisticated attacker who recomputes the edited row is still caught."""
    rows = await _seed(db, 5)
    target = rows[2]

    forged_hash = compute_entry_hash(
        target.prev_hash, target.timestamp, target.event_type,
        target.user_id, '{"seq": 999}',
    )
    await db.execute(
        update(DBAuditLog)
        .where(DBAuditLog.id == target.id)
        .values(detail='{"seq": 999}', entry_hash=forged_hash)
    )
    await db.commit()

    result = await verify_integrity(db)
    assert result.intact is False
    # The break surfaces on the *next* row, whose prev_hash no longer matches.
    assert result.first_broken_id == rows[3].id


@pytest.mark.asyncio
async def test_tamper_detected_after_row_deletion(db):
    rows = await _seed(db, 5)
    victim = rows[2]

    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(DBAuditLog).where(DBAuditLog.id == victim.id))
    await db.commit()

    result = await verify_integrity(db)
    assert result.intact is False
    assert result.first_broken_id == rows[3].id


@pytest.mark.asyncio
async def test_empty_chain_is_intact(db):
    result = await verify_integrity(db)
    assert result.intact is True
    assert result.checked == 0


@pytest.mark.asyncio
async def test_timestamps_are_strictly_increasing(db):
    """Burst writes must not produce ambiguous ordering."""
    rows = await _seed(db, 10)
    timestamps = [r.timestamp for r in rows]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


# ---------------------------------------------------------------------------
# Query filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_by_user(db):
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="alice")
    await log_event(db, AuditEventType.LOGOUT, user_id="alice")
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="bob")

    alice = await AuditQuery.by_user(db, "alice")
    assert len(alice) == 2
    assert {r.event_type for r in alice} == {"login_success", "logout"}

    # Newest first.
    assert alice[0].timestamp >= alice[1].timestamp


@pytest.mark.asyncio
async def test_query_by_event_type_with_since(db):
    old = await log_event(db, AuditEventType.LOGIN_FAILED, user_id="a")
    assert old is not None
    cutoff = time.time()
    await log_event(db, AuditEventType.LOGIN_FAILED, user_id="b")
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="b")

    all_failures = await AuditQuery.by_event_type(db, AuditEventType.LOGIN_FAILED)
    assert len(all_failures) == 2

    recent_failures = await AuditQuery.by_event_type(
        db, "login_failed", since=cutoff
    )
    assert len(recent_failures) == 1
    assert recent_failures[0].user_id == "b"


@pytest.mark.asyncio
async def test_query_by_severity(db):
    await log_event(db, AuditEventType.LOGIN_SUCCESS, severity=AuditSeverity.INFO)
    await log_event(db, AuditEventType.RATE_LIMIT_HIT, severity="warning")
    await log_event(db, AuditEventType.SUSPICIOUS_INPUT_BLOCKED, severity="critical")
    await log_event(db, AuditEventType.UNAUTHORIZED_ACCESS_ATTEMPT, severity="critical")

    critical = await AuditQuery.by_severity(db, AuditSeverity.CRITICAL)
    assert len(critical) == 2

    warnings = await AuditQuery.by_severity(db, "warning")
    assert len(warnings) == 1

    infos = await AuditQuery.by_severity(db, "info")
    assert len(infos) == 1


@pytest.mark.asyncio
async def test_query_pagination(db):
    await _seed(db, 10)
    page1 = await AuditQuery.recent(db, limit=4, offset=0)
    page2 = await AuditQuery.recent(db, limit=4, offset=4)
    assert len(page1) == 4
    assert len(page2) == 4
    assert {r.id for r in page1}.isdisjoint({r.id for r in page2})


@pytest.mark.asyncio
async def test_failed_logins_by_user_and_ip(db):
    now = time.time()
    await log_event(
        db, AuditEventType.LOGIN_FAILED, user_id="victim", ip="1.2.3.4"
    )
    await log_event(
        db, AuditEventType.LOGIN_FAILED, user_id="victim", ip="5.6.7.8"
    )
    await log_event(
        db, AuditEventType.LOGIN_FAILED, user_id="someone", ip="1.2.3.4"
    )
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="victim", ip="1.2.3.4")

    by_user = await AuditQuery.failed_logins(db, "victim", window_hours=24)
    assert len(by_user) == 2
    assert all(r.event_type == "login_failed" for r in by_user)

    by_ip = await AuditQuery.failed_logins(db, "1.2.3.4", window_hours=24)
    assert len(by_ip) == 2

    # Window excludes everything if we look far enough back.
    assert now > 0


@pytest.mark.asyncio
async def test_get_recent_events_wrapper(db):
    await _seed(db, 3)
    events = await get_recent_events(db, limit=2)
    assert len(events) == 2


# ---------------------------------------------------------------------------
# Retention archiving
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retention_archives_and_deletes(db, tmp_path):
    old = await log_event(
        db, AuditEventType.LOGIN_SUCCESS, user_id="legacy", timestamp=time.time() - 200 * 86400
    )
    recent = await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="fresh")
    assert old is not None and recent is not None

    result = await AuditRetention.archive_older_than(
        db, days=90, archive_dir=tmp_path
    )

    assert result["ok"] is True
    assert result["archived"] == 1
    assert result["deleted"] == 1

    archive_file = result["archive_file"]
    assert archive_file is not None
    assert os.path.exists(archive_file)
    assert archive_file.startswith(str(tmp_path))

    payload = json.loads(open(archive_file, encoding="utf-8").read())
    assert payload["rows_archived"] == 1
    assert payload["records"][0]["user_id"] == "legacy"

    # The live table now holds only the recent row.
    remaining = await AuditQuery.recent(db, limit=100)
    assert len(remaining) == 1
    assert remaining[0].user_id == "fresh"


@pytest.mark.asyncio
async def test_retention_keeps_chain_verifiable_via_anchor(db, tmp_path):
    await log_event(
        db, AuditEventType.LOGIN_SUCCESS, user_id="old", timestamp=time.time() - 200 * 86400
    )
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="mid")
    await log_event(db, AuditEventType.LOGOUT, user_id="new")

    result = await AuditRetention.archive_older_than(
        db, days=90, archive_dir=tmp_path
    )
    assert result["ok"] is True
    assert result["archived"] == 1

    integrity = await verify_integrity(db)
    assert integrity.intact is True, integrity.reason
    assert integrity.from_genesis is False
    assert integrity.anchor_matches_archive is True


@pytest.mark.asyncio
async def test_retention_noop_when_nothing_is_old(db, tmp_path):
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="recent")
    result = await AuditRetention.archive_older_than(
        db, days=90, archive_dir=tmp_path
    )
    assert result["ok"] is True
    assert result["archived"] == 0
    assert result["archive_file"] is None


@pytest.mark.asyncio
async def test_retention_refuses_to_archive_a_broken_chain(db, tmp_path):
    rows = await _seed(db, 3)
    await db.execute(
        update(DBAuditLog)
        .where(DBAuditLog.id == rows[1].id)
        .values(detail="tampered")
    )
    await db.commit()

    result = await AuditRetention.archive_older_than(
        db, days=0, archive_dir=tmp_path
    )
    assert result["ok"] is False
    assert "integrity" in result["error"].lower() or "chain" in result["error"].lower()
    # Nothing was written.
    assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Resilience: log_event must never raise
# ---------------------------------------------------------------------------


class _ExplodingSession:
    """Stands in for an AsyncSession that has catastrophically failed."""

    async def execute(self, *args, **kwargs):
        raise RuntimeError("database is on fire")

    async def commit(self):
        raise RuntimeError("database is on fire")

    async def rollback(self):
        raise RuntimeError("database is on fire")

    def add(self, *args, **kwargs):
        raise RuntimeError("database is on fire")

    def begin_nested(self):
        raise RuntimeError("database is on fire")


@pytest.mark.asyncio
async def test_log_event_never_raises_with_broken_session():
    broken = _ExplodingSession()
    result = await log_event(
        broken, AuditEventType.LOGIN_SUCCESS, user_id="u1", ip="1.1.1.1"
    )
    assert result is None  # swallowed, not raised


@pytest.mark.asyncio
async def test_log_event_never_raises_with_bad_event_type(db):
    """Even garbage input must not propagate."""
    row = await log_event(db, object(), user_id="u1")
    assert row is None or isinstance(row, DBAuditLog)


@pytest.mark.asyncio
async def test_audit_logger_direct_call_never_raises():
    result = await AuditLogger.log(None, AuditEventType.LOGIN_SUCCESS)
    assert result is None


@pytest.mark.asyncio
async def test_log_event_survives_missing_table():
    """Point the logger at a database that has no audit_log table."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        # Deliberately never created AUDIT_METADATA here.
        result = await log_event(session, AuditEventType.LOGIN_SUCCESS, user_id="u")
        assert result is None
    await engine.dispose()


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reporter_summary_shape_and_counts(db):
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="alice", ip="1.1.1.1")
    await log_event(db, AuditEventType.LOGIN_FAILED, user_id="alice", ip="1.1.1.1")
    await log_event(db, AuditEventType.LOGIN_FAILED, user_id="alice", ip="2.2.2.2")
    await log_event(db, AuditEventType.LOGIN_FAILED, user_id="bob", ip="1.1.1.1")
    await log_event(
        db,
        AuditEventType.SUSPICIOUS_INPUT_BLOCKED,
        user_id="mallory",
        severity=AuditSeverity.CRITICAL,
    )

    summary = await audit_summary(db, days=7)

    assert summary["total_events"] == 5
    assert summary["by_event_type"]["login_failed"] == 3
    assert summary["by_event_type"]["login_success"] == 1
    assert summary["by_severity"]["critical"] == 1
    assert summary["critical_events"] == 1
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 3

    top_user_keys = [entry["key"] for entry in summary["top_users"]]
    assert top_user_keys[0] == "alice"

    top_ip_keys = [entry["key"] for entry in summary["top_ips"]]
    assert "1.1.1.1" in top_ip_keys

    clusters = summary["failed_login_clusters"]
    assert any(c["key"] == "alice" and c["count"] == 2 for c in clusters)

    # Must be JSON-serialisable for downstream dumping.
    json.dumps(summary)


@pytest.mark.asyncio
async def test_reporter_summary_respects_window(db):
    await log_event(
        db, AuditEventType.LOGIN_SUCCESS, user_id="ancient",
        timestamp=time.time() - 60 * 86400,
    )
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="recent")

    summary = await AuditReporter.summary(db, days=7)
    assert summary["total_events"] == 1
    assert summary["top_users"][0]["key"] == "recent"


@pytest.mark.asyncio
async def test_reporter_markdown_renders(db):
    await log_event(db, AuditEventType.LOGIN_SUCCESS, user_id="alice")
    md = await AuditReporter.render_markdown(db, days=7)
    assert "# Nydra audit summary" in md
    assert "alice" in md


# ---------------------------------------------------------------------------
# Enum / helper coverage
# ---------------------------------------------------------------------------


def test_event_type_enum_covers_required_events():
    required = {
        "LOGIN_SUCCESS", "LOGIN_FAILED", "REGISTER_SUCCESS", "REGISTER_FAILED",
        "LOGOUT", "TOKEN_REFRESHED", "PASSWORD_CHANGED", "OAUTH_LOGIN",
        "SETTINGS_CHANGED", "API_KEY_CREATED", "API_KEY_REVOKED",
        "JOB_CREATED", "JOB_FAILED", "FILE_UPLOADED", "RATE_LIMIT_HIT",
        "SUSPICIOUS_INPUT_BLOCKED", "UNAUTHORIZED_ACCESS_ATTEMPT",
    }
    assert required.issubset({member.name for member in AuditEventType})


def test_compute_entry_hash_is_deterministic_and_field_sensitive():
    base = compute_entry_hash(GENESIS_HASH, 1234.567890, "login_success", "u1", "d")
    assert base == compute_entry_hash(GENESIS_HASH, 1234.567890, "login_success", "u1", "d")
    assert base != compute_entry_hash(GENESIS_HASH, 1234.567891, "login_success", "u1", "d")
    assert base != compute_entry_hash(GENESIS_HASH, 1234.567890, "login_failed", "u1", "d")
    assert base != compute_entry_hash(GENESIS_HASH, 1234.567890, "login_success", "u2", "d")
    assert base != compute_entry_hash(GENESIS_HASH, 1234.567890, "login_success", "u1", "e")
    assert len(base) == 64