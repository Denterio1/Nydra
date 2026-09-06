"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         NYDRA — api.py                                       ║
║              Async FastAPI Backend — The Bridge                              ║
║                                                                              ║
║   Async SQLAlchemy  →  nydra.db (SQLite)                                     ║
║   JWT Auth          →  Access + Refresh tokens                               ║
║   WebSockets        →  Real-time job progress per job_id                     ║
║   Job Queue         →  Background workers + cancellation                     ║
║   Rate Limiting     →  Per-user, per-route                                   ║
║   CORS              →  React / Next.js ready                                 ║
║   File Handling     →  Chunked upload + SHA256 dedup                         ║
║   Agent Bridge      →  All 9 Nydra goals wired in                            ║
║   Streaming Chat    →  SSE + WebSocket AI chat                               ║
║   Security          →  Bcrypt, API keys, request signing                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────
import aiofiles
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    event,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ─────────────────────────────────────────────────────────────────────────────
# NYDRA INTERNAL
# ─────────────────────────────────────────────────────────────────────────────
from src.api_schemas import (
    AnyJobRequest,
    APIKeyCreate,
    APIKeyFull,
    APIKeyPublic,
    AuditRequest,
    AutoMLRequest,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    CleanRequest,
    ErrorCode,
    ErrorResponse,
    FileType,
    FullPipelineRequest,
    HealthResponse,
    ImageAnalysisRequest,
    InspectRequest,
    JobCreate,
    JobFilterParams,
    JobGoal,
    JobListItem,
    JobListResponse,
    JobResult,
    JobStatus,
    JobStatusResponse,
    JobStep,
    MultiUploadResponse,
    OutlierRequest,
    PaginationParams,
    ReportFormat,
    ReportRequest,
    ReportResponse,
    ServiceHealth,
    ServiceStatus,
    SystemStats,
    TextAnalysisRequest,
    TokenRefresh,
    TokenResponse,
    UploadedFileInfo,
    UploadResponse,
    UserLogin,
    UserPublic,
    UserRegister,
    UserSettings,
    UserSettingsUpdate,
    ValidationErrorResponse,
    Verdict,
    WSErrorPayload,
    WSEventType,
    WSMessage,
    WSProgressPayload,
    WSResultPayload,
    WSStepPayload,
    WSWarningPayload,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

APP_VERSION      = "1.0.0"
NYDRA_VERSION    = "0.5.5"
SECRET_KEY       = os.getenv("SECRET_KEY", secrets.token_hex(32))
REFRESH_SECRET   = os.getenv("REFRESH_SECRET", secrets.token_hex(32))
ALGORITHM        = "HS256"
ACCESS_EXPIRE    = int(os.getenv("ACCESS_EXPIRE_MINUTES", 60))
REFRESH_EXPIRE   = int(os.getenv("REFRESH_EXPIRE_DAYS", 7))
DB_URL           = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./nydra.db")
UPLOAD_DIR       = Path(os.getenv("UPLOAD_DIR", "./uploads"))
REPORT_DIR       = Path(os.getenv("REPORT_DIR", "./reports"))
MAX_FILE_MB      = int(os.getenv("MAX_FILE_MB", 500))
MAX_FILE_BYTES   = MAX_FILE_MB * 1024 * 1024
ALLOWED_ORIGINS  = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")
WORKER_COUNT     = int(os.getenv("WORKER_COUNT", 4))
JOB_TIMEOUT_S    = int(os.getenv("JOB_TIMEOUT_SECONDS", 1800))  # 30 min max
API_KEY_PREFIX   = "nydra_sk_"

ALLOWED_EXTENSIONS = {
    "csv", "xlsx", "xls", "json", "tsv", "parquet",
    "pdf", "docx", "doc", "txt", "md", "html",
    "png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif", "heic",
    "mp3", "wav", "mp4",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nydra.api")

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE — Models
# ─────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class DBUser(Base):
    __tablename__ = "users"
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username       = Column(String(50), unique=True, nullable=False, index=True)
    email          = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active      = Column(Boolean, default=True)
    is_admin       = Column(Boolean, default=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    last_login     = Column(DateTime, nullable=True)
    settings       = Column(JSON, default=dict)


class DBAPIKey(Base):
    __tablename__ = "api_keys"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = Column(String, nullable=False, index=True)
    name        = Column(String(100), nullable=False)
    key_hash    = Column(String(64), unique=True, nullable=False)  # SHA256
    key_prefix  = Column(String(16), nullable=False)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    expires_at  = Column(DateTime, nullable=True)
    last_used   = Column(DateTime, nullable=True)


class DBJob(Base):
    __tablename__ = "jobs"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, nullable=False, index=True)
    file_id      = Column(String, nullable=False)
    filename     = Column(String(500), nullable=False)
    goal         = Column(String(50), nullable=False)
    status       = Column(String(20), default="pending", index=True)
    progress     = Column(Integer, default=0)
    current_step = Column(String(200), nullable=True)
    steps        = Column(JSON, default=list)
    request_data = Column(JSON, default=dict)
    result_data  = Column(JSON, nullable=True)
    error        = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow, index=True)
    started_at   = Column(DateTime, nullable=True)
    finished_at  = Column(DateTime, nullable=True)
    duration_ms  = Column(Integer, nullable=True)
    meta         = Column(JSON, default=dict)


class DBUploadedFile(Base):
    __tablename__ = "uploaded_files"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, nullable=False, index=True)
    filename     = Column(String(500), nullable=False)
    file_type    = Column(String(20), nullable=False)
    size_bytes   = Column(Integer, nullable=False)
    checksum     = Column(String(64), nullable=False, index=True)
    storage_path = Column(String(1000), nullable=False)
    uploaded_at  = Column(DateTime, default=datetime.utcnow)
    is_deleted   = Column(Boolean, default=False)


class DBRefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False)
    issued_at  = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked    = Column(Boolean, default=False)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(50), nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE — Engine
# ─────────────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    DB_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {},
)

# Enable WAL mode for SQLite (massive concurrent read performance boost)
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, _):
    if "sqlite" in DB_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─────────────────────────────────────────────────────────────────────────────
# AUTH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_access_token(user_id: str, username: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE)
    return jwt.encode(
        {"sub": user_id, "username": username, "exp": exp, "type": "access"},
        SECRET_KEY, algorithm=ALGORITHM,
    )


def make_refresh_token(user_id: str) -> str:
    exp = datetime.utcnow() + timedelta(days=REFRESH_EXPIRE)
    token = secrets.token_urlsafe(64)
    return token, exp


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise JWTError("Wrong token type")
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> DBUser:
    """Supports both JWT Bearer and X-API-Key authentication."""

    # ── API Key path ──────────────────────────────────────────────────────
    if x_api_key:
        if not x_api_key.startswith(API_KEY_PREFIX):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        key_hash = sha256(x_api_key)
        result = await db.execute(
            select(DBAPIKey).where(
                DBAPIKey.key_hash == key_hash,
                DBAPIKey.is_active == True,
            )
        )
        api_key = result.scalar_one_or_none()
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="API key expired")
        # Update last_used async (non-blocking)
        await db.execute(
            update(DBAPIKey).where(DBAPIKey.id == api_key.id)
            .values(last_used=datetime.utcnow())
        )
        user_result = await db.execute(
            select(DBUser).where(DBUser.id == api_key.user_id, DBUser.is_active == True)
        )
        user = user_result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    # ── JWT Bearer path ───────────────────────────────────────────────────
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated — provide Bearer token or X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload  = decode_access_token(credentials.credentials)
    user_id  = payload.get("sub")
    result   = await db.execute(
        select(DBUser).where(DBUser.id == user_id, DBUser.is_active == True)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or deactivated")
    return user


async def get_admin_user(current_user: DBUser = Depends(get_current_user)) -> DBUser:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — Connection Manager
# ─────────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    """
    Manages WebSocket connections per job_id.
    One job can have multiple subscribers (e.g. two browser tabs).
    """

    def __init__(self) -> None:
        # job_id → set of active WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(job_id, set()).add(ws)
        log.info("WS connected | job=%s | total=%d", job_id, self.subscriber_count(job_id))

    async def disconnect(self, job_id: str, ws: WebSocket) -> None:
        async with self._lock:
            subs = self._connections.get(job_id, set())
            subs.discard(ws)
            if not subs:
                self._connections.pop(job_id, None)
        log.info("WS disconnected | job=%s", job_id)

    def subscriber_count(self, job_id: str) -> int:
        return len(self._connections.get(job_id, set()))

    async def broadcast(self, job_id: str, message: WSMessage) -> None:
        """Send message to all subscribers of a job. Dead connections removed silently."""
        subs = self._connections.get(job_id, set()).copy()
        if not subs:
            return
        payload = message.model_dump_json()
        dead: Set[WebSocket] = set()
        await asyncio.gather(
            *[self._safe_send(ws, payload, dead) for ws in subs],
            return_exceptions=True,
        )
        if dead:
            async with self._lock:
                self._connections.get(job_id, set()).difference_update(dead)

    @staticmethod
    async def _safe_send(ws: WebSocket, payload: str, dead: Set[WebSocket]) -> None:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)

    async def send_progress(
        self,
        job_id: str,
        progress: int,
        step: str,
        message: str,
        elapsed_ms: int,
        eta_ms: Optional[int] = None,
        step_progress: int = 0,
    ) -> None:
        await self.broadcast(
            job_id,
            WSMessage(
                event=WSEventType.PROGRESS,
                job_id=job_id,
                payload=WSProgressPayload(
                    progress=progress,
                    current_step=step,
                    step_progress=step_progress,
                    message=message,
                    elapsed_ms=elapsed_ms,
                    eta_ms=eta_ms,
                ),
            ),
        )

    async def send_step_started(
        self, job_id: str, step_id: str, name: str, desc: str, index: int, total: int
    ) -> None:
        await self.broadcast(
            job_id,
            WSMessage(
                event=WSEventType.STEP_STARTED,
                job_id=job_id,
                payload=WSStepPayload(
                    step_id=step_id,
                    step_name=name,
                    description=desc,
                    step_index=index,
                    total_steps=total,
                ),
            ),
        )

    async def send_warning(self, job_id: str, code: str, message: str, **kwargs) -> None:
        await self.broadcast(
            job_id,
            WSMessage(
                event=WSEventType.WARNING,
                job_id=job_id,
                payload=WSWarningPayload(code=code, message=message, **kwargs),
            ),
        )

    async def send_done(
        self, job_id: str, result_url: str, summary: str, score: Optional[int],
        verdict: Optional[str], total_issues: int, duration_ms: int,
    ) -> None:
        await self.broadcast(
            job_id,
            WSMessage(
                event=WSEventType.JOB_DONE,
                job_id=job_id,
                payload=WSResultPayload(
                    job_id=job_id,
                    result_url=result_url,
                    summary=summary,
                    score=score,
                    verdict=Verdict(verdict) if verdict else None,
                    total_issues=total_issues,
                    duration_ms=duration_ms,
                ),
            ),
        )

    async def send_error(self, job_id: str, code: str, message: str, step_id: Optional[str] = None) -> None:
        await self.broadcast(
            job_id,
            WSMessage(
                event=WSEventType.JOB_FAILED,
                job_id=job_id,
                payload=WSErrorPayload(code=code, message=message, step_id=step_id),
            ),
        )


ws_manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# JOB QUEUE & WORKER
# ─────────────────────────────────────────────────────────────────────────────
class JobQueue:
    """
    Simple async in-memory job queue with configurable worker pool.
    Each worker picks jobs from the queue and executes them sequentially.
    """

    def __init__(self, workers: int = WORKER_COUNT) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._active: Dict[str, asyncio.Task] = {}
        self._workers  = workers
        self._running  = False
        self._start_time = time.time()

    async def start(self) -> None:
        self._running = True
        for i in range(self._workers):
            asyncio.create_task(self._worker(f"worker-{i}"), name=f"nydra-worker-{i}")
        log.info("Job queue started with %d workers", self._workers)

    async def stop(self) -> None:
        """Signal all workers to stop polling and exit their loop cleanly."""
        self._running = False
        log.info("Job queue stopping...")


    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)
        log.info("Job enqueued | job=%s | queue_depth=%d", job_id, self._queue.qsize())

    def cancel(self, job_id: str) -> bool:
        task = self._active.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def active_count(self) -> int:
        return len([t for t in self._active.values() if not t.done()])

    async def _worker(self, name: str) -> None:
        log.debug("Worker %s started", name)
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            task = asyncio.create_task(
                self._execute(job_id),
                name=f"job-{job_id[:8]}",
            )
            self._active[job_id] = task
            try:
                await asyncio.wait_for(task, timeout=JOB_TIMEOUT_S)
            except asyncio.TimeoutError:
                task.cancel()
                await _mark_job_failed(job_id, "Job timed out after 30 minutes")
                await ws_manager.send_error(job_id, "TIMEOUT", "Job exceeded maximum runtime")
            except asyncio.CancelledError:
                await _mark_job_failed(job_id, "Job was cancelled")
                await ws_manager.send_error(job_id, "JOB_CANCELLED", "Job was cancelled")
            except Exception as exc:
                log.exception("Unhandled error in job %s", job_id)
                await _mark_job_failed(job_id, str(exc))
                await ws_manager.send_error(job_id, "INTERNAL_ERROR", str(exc))
            finally:
                self._active.pop(job_id, None)
                self._queue.task_done()

    async def _execute(self, job_id: str) -> None:
        await run_job(job_id)


job_queue = JobQueue()


# ─────────────────────────────────────────────────────────────────────────────
# JOB EXECUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
async def _mark_job_failed(job_id: str, error: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(DBJob).where(DBJob.id == job_id).values(
                status="failed",
                error=error,
                finished_at=datetime.utcnow(),
            )
        )
        await db.commit()


async def _update_job_progress(
    db: AsyncSession,
    job_id: str,
    progress: int,
    current_step: str,
    steps: Optional[List] = None,
) -> None:
    vals: Dict[str, Any] = {"progress": progress, "current_step": current_step}
    if steps is not None:
        vals["steps"] = steps
    await db.execute(update(DBJob).where(DBJob.id == job_id).values(**vals))
    await db.commit()


class StepTracker:
    """Tracks steps and emits WS events with progress calculation."""

    def __init__(self, job_id: str, step_definitions: List[Dict[str, str]]) -> None:
        self.job_id   = job_id
        self.steps    = [
            JobStep(step_id=str(uuid.uuid4())[:8], **s) for s in step_definitions
        ]
        self.current  = 0
        self.start_ms = int(time.time() * 1000)

    def elapsed_ms(self) -> int:
        return int(time.time() * 1000) - self.start_ms

    def overall_progress(self, step_progress: int = 0) -> int:
        base = int((self.current / len(self.steps)) * 100)
        step_contribution = int((step_progress / len(self.steps)))
        return min(base + step_contribution, 99)

    async def start_step(self, db: AsyncSession) -> JobStep:
        step = self.steps[self.current]
        step.status     = JobStatus.RUNNING
        step.started_at = datetime.utcnow()
        await ws_manager.send_step_started(
            self.job_id, step.step_id, step.name, step.description,
            self.current, len(self.steps),
        )
        await _update_job_progress(
            db, self.job_id, self.overall_progress(), step.name,
            [json.loads(s.model_dump_json()) for s in self.steps],
        )
        return step

    async def finish_step(self, db: AsyncSession, warnings: List[str] = []) -> None:
        step = self.steps[self.current]
        step.status      = JobStatus.DONE
        step.finished_at = datetime.utcnow()
        step.duration_ms = self.elapsed_ms()
        step.warnings    = warnings
        self.current    += 1
        await _update_job_progress(
            db, self.job_id, self.overall_progress(), step.name,
            [json.loads(s.model_dump_json()) for s in self.steps],
        )

    async def emit_progress(self, step_progress: int, message: str, eta_ms: Optional[int] = None) -> None:
        await ws_manager.send_progress(
            self.job_id,
            self.overall_progress(step_progress),
            self.steps[self.current].name if self.current < len(self.steps) else "Finishing",
            message,
            self.elapsed_ms(),
            eta_ms,
            step_progress,
        )

    def step_dicts(self) -> List[Dict]:
        return [json.loads(s.model_dump_json()) for s in self.steps]

async def run_job(job_id: str) -> None:
    """
    Main job execution function. Fetches job from DB, runs DataDoctor agent,
    streams progress via WebSocket, writes result back to DB.
    """
    async with AsyncSessionLocal() as db:
        # ── Load job ──────────────────────────────────────────────────────
        result = await db.execute(select(DBJob).where(DBJob.id == job_id))
        job    = result.scalar_one_or_none()
        if not job:
            log.error("Job %s not found in DB", job_id)
            return

        # ── Mark as running ───────────────────────────────────────────────
        await db.execute(
            update(DBJob).where(DBJob.id == job_id).values(
                status="running",
                started_at=datetime.utcnow(),
            )
        )
        await db.commit()

        # ── Broadcast start ───────────────────────────────────────────────
        await ws_manager.broadcast(
            job_id,
            WSMessage(
                event=WSEventType.JOB_STARTED,
                job_id=job_id,
                payload={"goal": job.goal, "filename": job.filename},
            ),
        )

        goal    = JobGoal(job.goal)
        req     = job.request_data
        t_start = time.time()

        try:
            # ── Build step plan based on goal ─────────────────────────────
            step_plan = _build_step_plan(goal)
            tracker   = StepTracker(job_id, step_plan)

            # ── Load file ─────────────────────────────────────────────────
            file_result = await db.execute(
                select(DBUploadedFile).where(DBUploadedFile.id == job.file_id)
            )
            file_record = file_result.scalar_one_or_none()
            if not file_record:
                raise FileNotFoundError(f"File {job.file_id} not found")

            file_path = Path(file_record.storage_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File on disk not found: {file_path}")

            # ── Execute via Agent ─────────────────────────────────────────
            result_data = await _run_agent(goal, file_path, req, tracker, db)

            # ── Calculate duration ────────────────────────────────────────
            duration_ms = int((time.time() - t_start) * 1000)

            # ── Write result to DB ────────────────────────────────────────
            await db.execute(
                update(DBJob).where(DBJob.id == job_id).values(
                    status="done",
                    progress=100,
                    current_step="Complete",
                    result_data=result_data,
                    finished_at=datetime.utcnow(),
                    duration_ms=duration_ms,
                    steps=[json.loads(s.model_dump_json()) for s in tracker.steps],
                )
            )
            await db.commit()

            # ── Notify subscribers ────────────────────────────────────────
            score        = result_data.get("quality", {}).get("overall_score")
            verdict      = result_data.get("quality", {}).get("verdict")
            total_issues = len(result_data.get("issues", []))
            summary      = _build_summary(goal, result_data)

            await ws_manager.send_done(
                job_id,
                result_url=f"/api/v1/jobs/{job_id}/result",
                summary=summary,
                score=score,
                verdict=verdict,
                total_issues=total_issues,
                duration_ms=duration_ms,
            )
            log.info("Job completed | job=%s | duration=%.1fs", job_id, duration_ms / 1000)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("Job failed | job=%s | error=%s", job_id, exc)
            await db.execute(
                update(DBJob).where(DBJob.id == job_id).values(
                    status="failed",
                    error=str(exc),
                    finished_at=datetime.utcnow(),
                    duration_ms=int((time.time() - t_start) * 1000),
                )
            )
            await db.commit()
            await ws_manager.send_error(job_id, "AGENT_ERROR", str(exc))


def _build_step_plan(goal: JobGoal) -> List[Dict[str, str]]:
    """Returns ordered step definitions for each goal."""
    base = [{"name": "Loading", "description": "Reading and validating file"}]

    plans = {
        JobGoal.INSPECT: [
            *base,
            {"name": "Schema Detection",    "description": "Inferring column types and structure"},
            {"name": "Missing Analysis",    "description": "Detecting and profiling missing values"},
            {"name": "Duplicate Detection", "description": "Finding duplicate rows"},
            {"name": "Column Profiling",    "description": "Computing per-column statistics"},
            {"name": "Quality Score",       "description": "Calculating overall data quality score"},
            {"name": "Report",              "description": "Generating inspection report"},
        ],
        JobGoal.CLEAN: [
            *base,
            {"name": "Missing Imputation",  "description": "Filling missing values with smart method"},
            {"name": "Outlier Handling",    "description": "Detecting and treating outliers"},
            {"name": "Duplicate Removal",   "description": "Removing duplicate rows"},
            {"name": "Type Fixing",         "description": "Correcting column data types"},
            {"name": "Quality Score",       "description": "Re-scoring after cleaning"},
            {"name": "Report",              "description": "Generating cleaning report"},
        ],
        JobGoal.ANALYZE: [
            *base,
            {"name": "Descriptive Stats",  "description": "Computing detailed statistics"},
            {"name": "Normality Tests",    "description": "Running Shapiro-Wilk, Anderson-Darling and more"},
            {"name": "Distribution Fit",   "description": "Fitting 20 distributions per column"},
            {"name": "Correlation Matrix", "description": "Pearson, Spearman, Kendall with FDR correction"},
            {"name": "Hypothesis Tests",   "description": "Auto-selecting parametric vs non-parametric tests"},
            {"name": "Bias Detection",     "description": "Checking for demographic and statistical bias"},
            {"name": "Report",             "description": "Generating analysis report"},
        ],
        JobGoal.PREPARE_FOR_ML: [
            *base,
            {"name": "Target Analysis",    "description": "Analyzing target column distribution"},
            {"name": "Imputation",         "description": "Filling missing values"},
            {"name": "Encoding",           "description": "Encoding categorical features"},
            {"name": "Scaling",            "description": "Normalizing numerical features"},
            {"name": "Imbalance Check",    "description": "Detecting and handling class imbalance"},
            {"name": "Feature Selection",  "description": "Ranking and selecting best features"},
            {"name": "Split Advisory",     "description": "Recommending train/val/test split strategy"},
            {"name": "ML Readiness Score", "description": "Computing ML readiness score"},
            {"name": "Report",             "description": "Generating ML preparation report"},
        ],
        JobGoal.FULL_PIPELINE: [
            *base,
            {"name": "Inspection",         "description": "Full dataset profiling"},
            {"name": "Cleaning",           "description": "Imputation + outliers + dedup"},
            {"name": "Analysis",           "description": "Statistics + correlations + distributions"},
            {"name": "ML Preparation",     "description": "Encoding + scaling + imbalance"},
            {"name": "Quality Score",      "description": "Unified quality score"},
            {"name": "AI Suggestions",     "description": "Generating smart recommendations"},
            {"name": "Report",             "description": "Generating full pipeline report"},
        ],
        JobGoal.RUN_AUTOML: [
            *base,
            {"name": "Task Detection",     "description": "Auto-detecting classification vs regression"},
            {"name": "Preprocessing",      "description": "Preparing data for training"},
            {"name": "Search Space",       "description": "Building model search space"},
            {"name": "Bayesian Search",    "description": "Running Optuna TPE optimizer"},
            {"name": "Cross Validation",   "description": "Evaluating models with nested CV"},
            {"name": "Ensemble",           "description": "Stacking top models"},
            {"name": "Feature Importance", "description": "Computing SHAP values"},
            {"name": "Pipeline Export",    "description": "Exporting best model as sklearn pipeline"},
            {"name": "Leaderboard",        "description": "Ranking all models"},
            {"name": "Report",             "description": "Generating AutoML report"},
        ],
        JobGoal.DETECT_OUTLIERS: [
            *base,
            {"name": "Method Selection",   "description": "Choosing best outlier detector for your data"},
            {"name": "Detection",          "description": "Running outlier detection"},
            {"name": "Per-Column Report",  "description": "Analyzing outliers column by column"},
            {"name": "Severity Scoring",   "description": "Scoring severity per outlier"},
            {"name": "Report",             "description": "Generating outlier report"},
        ],
        JobGoal.AUDIT: [
            *base,
            {"name": "Leakage Check",     "description": "Detecting data leakage between train/test"},
            {"name": "Label Quality",     "description": "Running Confident Learning for label noise"},
            {"name": "Bias Detection",    "description": "Statistical fairness analysis"},
            {"name": "Readiness Score",   "description": "Computing training readiness score"},
            {"name": "Fix Checklist",     "description": "Building prioritized fix list"},
            {"name": "Report",            "description": "Generating audit report"},
        ],
        JobGoal.IMAGES: [
            *base,
            {"name": "Image Loading",     "description": "Reading all images in dataset"},
            {"name": "Quality Analysis",  "description": "Scoring brightness, blur, noise per image"},
            {"name": "Duplicate Detection","description": "Perceptual hashing for duplicates"},
            {"name": "Class Balance",     "description": "Checking class distribution"},
            {"name": "ML Readiness",      "description": "Image ML readiness score"},
            {"name": "Report",            "description": "Generating image dataset report"},
        ],
        JobGoal.TEXT_DOCS: [
            *base,
            {"name": "Document Reading",  "description": "Extracting text from PDF/DOCX/TXT"},
            {"name": "Basic Stats",       "description": "Word count, sentences, readability"},
            {"name": "Sentiment",         "description": "Positive / negative / neutral analysis"},
            {"name": "Keywords",          "description": "TF-IDF + YAKE + KeyBERT extraction"},
            {"name": "NER",               "description": "Named entity recognition"},
            {"name": "Topic Modeling",    "description": "LDA + NMF topic extraction"},
            {"name": "PII Detection",     "description": "Scanning for personal information"},
            {"name": "Text Quality",      "description": "Grammar, noise, coherence scoring"},
            {"name": "Report",            "description": "Generating document analysis report"},
        ],
    }
    return plans.get(goal, base)

def _sanitize_for_json(obj):
    """Recursively strip pandas DataFrames (and other non-JSON-safe types) from a result dict."""
    import pandas as pd  # noqa : PLC0415
    if isinstance(obj, pd.DataFrame):
        return {"_type": "DataFrame", "rows": len(obj), "columns": list(obj.columns)}
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


async def _run_agent(
    goal: JobGoal,
    file_path: Path,
    req: Dict[str, Any],
    tracker: StepTracker,
    db: AsyncSession,
) -> Dict[str, Any]:
    """
    Bridge between FastAPI and DataDoctor agent.
    Runs in executor to avoid blocking the event loop during heavy computation.
    """
    loop = asyncio.get_event_loop()

    async def emit(progress: int, message: str) -> None:
        await tracker.emit_progress(progress, message)

    # Lazy import agent (heavy, only loaded when needed)
    try:
        from src.core.agent import Nydra  # noqa: PLC0415
        doctor = Nydra()
    except ImportError:
        log.warning("DataDoctor agent not available — using stub")
        doctor = None

    job_id = tracker.job_id
    result: Dict[str, Any] = {}

    for i, step_def in enumerate(tracker.steps):
        await tracker.start_step(db)
        await asyncio.sleep(0)  # yield to event loop

        step_name = step_def.name if hasattr(step_def, "name") else str(step_def)

        try:
            if doctor:
                # Run heavy computation in thread pool
                step_result = await loop.run_in_executor(
                    None,
                    lambda: _dispatch_step(doctor, step_name, goal, file_path, req, result),
                )
                if step_result:
                    result.update(step_result)
            else:
                # Stub mode — simulate work
                await asyncio.sleep(0.3)
                result[f"step_{i}"] = {"status": "stub", "step": step_name}

            await tracker.finish_step(db)

        except Exception as exc:
            log.warning("Step '%s' failed: %s", step_name, exc)
            await ws_manager.send_warning(
                job_id, "STEP_ERROR", f"Step '{step_name}' encountered an issue: {exc}"
            )
            await tracker.finish_step(db, warnings=[str(exc)])
            # Continue to next step (self-healing)

    return _sanitize_for_json(result)


def _dispatch_step(
    doctor: Any,
    step_name: str,
    goal: JobGoal,
    file_path: Path,
    req: Dict[str, Any],
    accumulated: Dict[str, Any],
) -> Dict[str, Any]:
    """Synchronous dispatcher — runs inside thread pool executor."""
    import pandas as pd

    def _load_dataframe(path: Path) -> Optional[pd.DataFrame]:
        ext = path.suffix.lower().lstrip(".")
        if ext == "csv":
            return pd.read_csv(file_path)
        elif ext in ("xlsx", "xls"):
            return pd.read_excel(file_path)
        elif ext == "json":
            return pd.read_json(file_path)
        elif ext == "parquet":
            return pd.read_parquet(file_path)
        elif ext == "tsv":
            return pd.read_csv(file_path, sep="\t")
        return None

    if step_name == "Loading":
        df = _load_dataframe(file_path)
        if df is None:
            return {"raw_path": str(file_path)}
        return {"_df_path": str(file_path), "rows": len(df), "columns": len(df.columns)}
        


    # All other steps delegate to the agent
    if hasattr(doctor, "achieve_goal"):
        df = _load_dataframe(file_path)
        if df is None:
            log.warning("Step '%s': could not reload dataframe form %s", step_name, file_path)
            return {}
        try:
            result = doctor.achieve_goal(df, goal=goal.value)
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            log.warning("Step '%s': achieve_goal failed: %s", step_name, exc)
            return {}

    return {}


def _build_summary(goal: JobGoal, result: Dict[str, Any]) -> str:
    score   = result.get("quality", {}).get("overall_score", "N/A")
    issues  = len(result.get("issues", []))
    verdict = result.get("quality", {}).get("verdict", "unknown")
    return (
        f"{goal.value.replace('_', ' ').title()} complete. "
        f"Quality score: {score}/100 | Verdict: {verdict} | Issues found: {issues}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FILE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def detect_file_type(filename: str) -> FileType:
    ext = filename.rsplit(".", 1)[-1].lower()
    mapping = {
        "csv": FileType.CSV, "xlsx": FileType.EXCEL, "xls": FileType.EXCEL,
        "json": FileType.JSON, "tsv": FileType.TSV, "parquet": FileType.PARQUET,
        "pdf": FileType.PDF, "docx": FileType.DOCX, "doc": FileType.DOCX,
        "txt": FileType.TXT, "md": FileType.TXT, "html": FileType.TXT,
        "png": FileType.PNG, "jpg": FileType.JPG, "jpeg": FileType.JPG,
        "webp": FileType.WEBP,
    }
    return mapping.get(ext, FileType.CSV)


async def save_upload(file: UploadFile, user_id: str) -> UploadedFileInfo:
    """Chunk-reads the upload, deduplicates by SHA256, returns file info."""
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    file_id  = str(uuid.uuid4())
    tmp_path = UPLOAD_DIR / f"tmp_{file_id}.{ext}"
    sha      = hashlib.sha256()
    size     = 0
    chunk_sz = 1024 * 1024  # 1MB chunks

    try:
        too_large = False
        async with aiofiles.open(tmp_path, "wb") as f:
            while chunk := await file.read(chunk_sz):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    too_large = True
                    break
                sha.update(chunk)
                await f.write(chunk)

        if too_large:        
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {MAX_FILE_MB}MB",

                )
    except HTTPException:
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    checksum   = sha.hexdigest()
    final_path = UPLOAD_DIR / f"{user_id}_{checksum[:16]}.{ext}"

    if not final_path.exists():
        shutil.move(str(tmp_path), str(final_path))
    else:
        tmp_path.unlink(missing_ok=True)  # Dedup — reuse existing

    return UploadedFileInfo(
        file_id=file_id,
        filename=file.filename,
        file_type=detect_file_type(file.filename),
        size_bytes=size,
        size_mb=round(size / (1024 * 1024), 3),
        checksum=checksum,
        storage_path=str(final_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING (simple in-memory, swap for Redis in production)
# ─────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_calls: int, window_s: int) -> None:
        self._max  = max_calls
        self._win  = window_s
        self._hits: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        now  = time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < self._win]
        self._hits[key] = hits
        if len(hits) >= self._max:
            return False
        self._hits[key].append(now)
        return True


upload_limiter = RateLimiter(max_calls=20, window_s=60)
job_limiter    = RateLimiter(max_calls=10, window_s=60)
auth_limiter   = RateLimiter(max_calls=10, window_s=300)  # 10 login attempts / 5 min


# ─────────────────────────────────────────────────────────────────────────────
# APP LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    log.info("🩺 Nydra API starting up (v%s)...", APP_VERSION)
    app.state.start_time = time.time()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await job_queue.start()
    log.info("✅ Database ready | ✅ Job queue running (%d workers)", WORKER_COUNT)
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("Nydra API shutting down...")
    await job_queue.stop()
    await engine.dispose()

# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nydra API",
    description="Autonomous Data Intelligence Platform — REST + WebSocket API",
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Middleware stack (order matters — outermost first) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_logger_and_timing(request: Request, call_next):
    request_id = str(uuid.uuid4())[:12]
    start      = time.time()
    request.state.request_id = request_id
    response   = await call_next(request)
    duration   = (time.time() - start) * 1000
    response.headers["X-Request-ID"]   = request_id
    response.headers["X-Response-Time"] = f"{duration:.1f}ms"
    if request.url.path not in ("/api/v1/health", "/api/v1/ping"):
        log.info(
            "%s %s → %d (%.1fms) [%s]",
            request.method, request.url.path, response.status_code, duration, request_id,
        )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=json.loads(ErrorResponse(
            error=ErrorCode.INTERNAL_ERROR,
            message=exc.detail,
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump_json()),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=json.loads(ErrorResponse(
            error=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred. Please try again.",
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump_json()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Auth
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/auth/register", response_model=UserPublic, status_code=201, tags=["Auth"])
async def register(body: UserRegister, request: Request, db: AsyncSession = Depends(get_db)):
    """Register a new Nydra account."""
    ip = request.client.host
    if not auth_limiter.is_allowed(ip):
        raise HTTPException(status_code=429, detail="Too many registration attempts")

    existing = await db.execute(
        select(DBUser).where(
            (DBUser.username == body.username) | (DBUser.email == body.email)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username or email already exists")

    user = DBUser(
        id=str(uuid.uuid4()),
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        settings=UserSettings().model_dump(),
    )
    db.add(user)
    await db.commit()
    log.info("New user registered: %s", user.username)
    return UserPublic(id=user.id, username=user.username, email=user.email, created_at=user.created_at, is_active=user.is_active)


@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(
    body: UserLogin, request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Login and receive JWT access + refresh tokens."""
    ip = request.client.host
    if not auth_limiter.is_allowed(f"login:{ip}"):
        raise HTTPException(status_code=429, detail="Too many login attempts. Wait 5 minutes.")

    result = await db.execute(select(DBUser).where(DBUser.username == body.username))
    user   = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    access_token               = make_access_token(user.id, user.username)
    raw_refresh, refresh_exp   = make_refresh_token(user.id)

    db.add(DBRefreshToken(
        user_id=user.id,
        token_hash=sha256(raw_refresh),
        expires_at=refresh_exp,
        user_agent=request.headers.get("User-Agent", "")[:500],
        ip_address=ip,
    ))
    await db.execute(update(DBUser).where(DBUser.id == user.id).values(last_login=datetime.utcnow()))
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=ACCESS_EXPIRE * 60,
    )


@app.post("/api/v1/auth/refresh", response_model=TokenResponse, tags=["Auth"])
async def refresh_token(body: TokenRefresh, db: AsyncSession = Depends(get_db)):
    """Exchange a refresh token for a new access token."""
    token_hash = sha256(body.refresh_token)
    result     = await db.execute(
        select(DBRefreshToken).where(
            DBRefreshToken.token_hash == token_hash,
            DBRefreshToken.revoked == False,
        )
    )
    rt = result.scalar_one_or_none()
    if not rt:
        raise HTTPException(status_code=401, detail="Invalid or revoked refresh token")
    if rt.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user_result = await db.execute(select(DBUser).where(DBUser.id == rt.user_id))
    user        = user_result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    # Rotate refresh token (invalidate old one)
    await db.execute(
        update(DBRefreshToken).where(DBRefreshToken.id == rt.id).values(revoked=True)
    )
    raw_new, new_exp = make_refresh_token(user.id)
    db.add(DBRefreshToken(user_id=user.id, token_hash=sha256(raw_new), expires_at=new_exp))
    await db.commit()

    return TokenResponse(
        access_token=make_access_token(user.id, user.username),
        refresh_token=raw_new,
        expires_in=ACCESS_EXPIRE * 60,
    )


@app.post("/api/v1/auth/logout", tags=["Auth"])
async def logout(body: TokenRefresh, db: AsyncSession = Depends(get_db)):
    """Revoke refresh token (logout)."""
    await db.execute(
        update(DBRefreshToken)
        .where(DBRefreshToken.token_hash == sha256(body.refresh_token))
        .values(revoked=True)
    )
    await db.commit()
    return {"message": "Logged out successfully"}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — API Keys
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/auth/api-keys", response_model=APIKeyFull, status_code=201, tags=["API Keys"])
async def create_api_key(
    body: APIKeyCreate,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    raw_key  = API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_hash = sha256(raw_key)
    expires  = datetime.utcnow() + timedelta(days=body.expires_in) if body.expires_in else None

    api_key = DBAPIKey(
        user_id=current_user.id,
        name=body.name,
        key_hash=key_hash,
        key_prefix=raw_key[:16],
        expires_at=expires,
    )
    db.add(api_key)
    await db.commit()

    return APIKeyFull(
        id=api_key.id, name=api_key.name, key_prefix=api_key.key_prefix,
        created_at=api_key.created_at, expires_at=api_key.expires_at,
        is_active=True, full_key=raw_key,
    )


@app.get("/api/v1/auth/api-keys", response_model=List[APIKeyPublic], tags=["API Keys"])
async def list_api_keys(
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBAPIKey).where(DBAPIKey.user_id == current_user.id, DBAPIKey.is_active == True)
    )
    keys = result.scalars().all()
    return [
        APIKeyPublic(
            id=k.id, name=k.name, key_prefix=k.key_prefix,
            created_at=k.created_at, expires_at=k.expires_at,
            last_used=k.last_used, is_active=k.is_active,
        )
        for k in keys
    ]


@app.delete("/api/v1/auth/api-keys/{key_id}", status_code=204, tags=["API Keys"])
async def revoke_api_key(
    key_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBAPIKey).where(DBAPIKey.id == key_id, DBAPIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    await db.execute(update(DBAPIKey).where(DBAPIKey.id == key_id).values(is_active=False))
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — User / Settings
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/me", response_model=UserPublic, tags=["User"])
async def get_me(current_user: DBUser = Depends(get_current_user)):
    return UserPublic(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
        is_active=current_user.is_active,
    )


@app.get("/api/v1/me/settings", response_model=UserSettings, tags=["User"])
async def get_settings(current_user: DBUser = Depends(get_current_user)):
    settings = current_user.settings or {}
    try:
        return UserSettings(**settings)
    except Exception:
        return UserSettings()


@app.patch("/api/v1/me/settings", response_model=UserSettings, tags=["User"])
async def update_settings(
    body: UserSettingsUpdate,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    current = current_user.settings or {}
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    current.update(updates)
    await db.execute(update(DBUser).where(DBUser.id == current_user.id).values(settings=current))
    await db.commit()
    return UserSettings(**current)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — File Upload
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/upload", response_model=UploadResponse, status_code=201, tags=["Files"])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a single file. Returns file_id to use in job requests."""
    if not upload_limiter.is_allowed(current_user.id):
        raise HTTPException(status_code=429, detail="Upload rate limit exceeded (20/min)")

    file_info = await save_upload(file, current_user.id)

    db.add(DBUploadedFile(
        id=file_info.file_id,
        user_id=current_user.id,
        filename=file_info.filename,
        file_type=file_info.file_type,
        size_bytes=file_info.size_bytes,
        checksum=file_info.checksum,
        storage_path=file_info.storage_path,
    ))
    await db.commit()
    log.info("File uploaded | user=%s | file=%s | size=%.1fMB", current_user.username, file_info.filename, file_info.size_mb)

    return UploadResponse(file=file_info)


@app.post("/api/v1/upload/multi", response_model=MultiUploadResponse, status_code=201, tags=["Files"])
async def upload_multi(
    request: Request,
    train_file: UploadFile = File(...),
    test_file:  UploadFile = File(...),
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload train + test files for the Training Data Audit feature."""
    if not upload_limiter.is_allowed(current_user.id):
        raise HTTPException(status_code=429, detail="Upload rate limit exceeded")

    train_info, test_info = await asyncio.gather(
        save_upload(train_file, current_user.id),
        save_upload(test_file, current_user.id),
    )
    for info in (train_info, test_info):
        db.add(DBUploadedFile(
            id=info.file_id, user_id=current_user.id, filename=info.filename,
            file_type=info.file_type, size_bytes=info.size_bytes,
            checksum=info.checksum, storage_path=info.storage_path,
        ))
    await db.commit()

    return MultiUploadResponse(train_file=train_info, test_file=test_info)


@app.delete("/api/v1/files/{file_id}", status_code=204, tags=["Files"])
async def delete_file(
    file_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBUploadedFile).where(
            DBUploadedFile.id == file_id,
            DBUploadedFile.user_id == current_user.id,
        )
    )
    f = result.scalar_one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    await db.execute(
        update(DBUploadedFile).where(DBUploadedFile.id == file_id).values(is_deleted=True)
    )
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Jobs
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/jobs", response_model=JobStatusResponse, status_code=202, tags=["Jobs"])
async def create_job(
    request: Request,
    body: AnyJobRequest,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new analysis job. Returns immediately with job_id.
    Subscribe to WS /api/v1/ws/{job_id} for live progress.
    """
    if not job_limiter.is_allowed(current_user.id):
        raise HTTPException(status_code=429, detail="Job rate limit exceeded (10/min)")

    # Verify file exists and belongs to user
    file_id  = body.train_file_id if hasattr(body, "train_file_id") else body.file_id
    f_result = await db.execute(
        select(DBUploadedFile).where(
            DBUploadedFile.id == file_id,
            DBUploadedFile.user_id == current_user.id,
            DBUploadedFile.is_deleted == False,
        )
    )
    file_rec = f_result.scalar_one_or_none()
    if not file_rec:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    job_id   = str(uuid.uuid4())
    step_plan = _build_step_plan(body.goal)
    steps    = [
        JobStep(step_id=str(uuid.uuid4())[:8], **s).model_dump()
        for s in step_plan
    ]

    db_job = DBJob(
        id=job_id,
        user_id=current_user.id,
        file_id=file_id,
        filename=file_rec.filename,
        goal=body.goal,
        request_data=body.model_dump(),
        steps=steps,
    )
    db.add(db_job)
    await db.commit()

    await job_queue.enqueue(job_id)

    log.info("Job created | user=%s | job=%s | goal=%s", current_user.username, job_id, body.goal)

    return JobStatusResponse(
        job_id=job_id,
        goal=body.goal,
        status=JobStatus.PENDING,
        progress=0,
        steps=[JobStep(**s) for s in steps],
        created_at=db_job.created_at,
        result_url=f"/api/v1/jobs/{job_id}/result",
    )


@app.get("/api/v1/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    page:    int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status:  Optional[str] = Query(None),
    goal:    Optional[str] = Query(None),
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all jobs for the current user with pagination."""
    q = select(DBJob).where(DBJob.user_id == current_user.id)
    if status:
        q = q.where(DBJob.status == status)
    if goal:
        q = q.where(DBJob.goal == goal)
    q = q.order_by(DBJob.created_at.desc())

    count_result = await db.execute(q)
    all_jobs     = count_result.scalars().all()
    total        = len(all_jobs)
    offset       = (page - 1) * per_page
    page_jobs    = all_jobs[offset: offset + per_page]

    items = [
        JobListItem(
            job_id=j.id, goal=JobGoal(j.goal), status=JobStatus(j.status),
            progress=j.progress, filename=j.filename, created_at=j.created_at,
            duration_ms=j.duration_ms,
        )
        for j in page_jobs
    ]
    return JobListResponse(
        jobs=items, total=total, page=page, per_page=per_page,
        pages=max(1, (total + per_page - 1) // per_page),
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(
    job_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBJob).where(DBJob.id == job_id, DBJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.id, goal=JobGoal(job.goal), status=JobStatus(job.status),
        progress=job.progress, current_step=job.current_step,
        steps=[JobStep(**s) for s in (job.steps or [])],
        created_at=job.created_at, started_at=job.started_at,
        finished_at=job.finished_at, duration_ms=job.duration_ms,
        error=job.error,
        result_url=f"/api/v1/jobs/{job_id}/result" if job.status == "done" else None,
    )


@app.get("/api/v1/jobs/{job_id}/result", tags=["Jobs"])
async def get_job_result(
    job_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns the full analysis result when job is complete."""
    result = await db.execute(
        select(DBJob).where(DBJob.id == job_id, DBJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "pending" or job.status == "running":
        raise HTTPException(status_code=202, detail="Job still in progress")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")

    return JSONResponse(content=job.result_data or {})


@app.delete("/api/v1/jobs/{job_id}", status_code=204, tags=["Jobs"])
async def cancel_job(
    job_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running or pending job."""
    result = await db.execute(
        select(DBJob).where(DBJob.id == job_id, DBJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending", "running"):
        raise HTTPException(status_code=409, detail=f"Cannot cancel job in '{job.status}' state")

    cancelled = job_queue.cancel(job_id)
    await db.execute(
        update(DBJob).where(DBJob.id == job_id).values(
            status="cancelled",
            finished_at=datetime.utcnow(),
            error="Cancelled by user",
        )
    )
    await db.commit()
    await ws_manager.broadcast(
        job_id,
        WSMessage(
            event=WSEventType.JOB_CANCELLED,
            job_id=job_id,
            payload={"cancelled_by": current_user.username, "force": not cancelled},
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — WebSocket
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/api/v1/ws/{job_id}")
async def websocket_job_progress(
    websocket: WebSocket,
    job_id: str,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time job progress.

    Authentication: Pass JWT as query param ?token=<access_token>
    or use X-API-Key header before upgrade.

    Messages are WSMessage JSON objects.
    Client can send {"type": "ping"} to keep connection alive.
    """
    # ── Auth over WS ──────────────────────────────────────────────────────
    if not token:
        await websocket.close(code=4001, reason="Missing auth token")
        return
    try:
        payload = decode_access_token(token)
        user_id = payload["sub"]
    except HTTPException:
        await websocket.close(code=4001, reason="Invalid auth token")
        return

    # ── Verify job belongs to user ────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DBJob).where(DBJob.id == job_id, DBJob.user_id == user_id)
        )
        job = result.scalar_one_or_none()

    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    # ── If job already done, send final status immediately ────────────────
    if job.status in ("done", "failed", "cancelled"):
        await websocket.accept()
        event_type = {
            "done": WSEventType.JOB_DONE,
            "failed": WSEventType.JOB_FAILED,
            "cancelled": WSEventType.JOB_CANCELLED,
        }[job.status]
        if job.status == "done":
            msg = WSMessage(
                event=event_type, job_id=job_id,
                payload=WSResultPayload(
                    job_id=job_id,
                    result_url=f"/api/v1/jobs/{job_id}/result",
                    summary="Job already completed",
                    score=None, verdict=None, total_issues=0,
                    duration_ms=job.duration_ms or 0,
                ),
            )
        else:
            msg = WSMessage(
                event=event_type, job_id=job_id,
                payload=WSErrorPayload(code=event_type, message=job.error or ""),
            )
        await websocket.send_text(msg.model_dump_json())
        await websocket.close()
        return

    # ── Subscribe to live updates ─────────────────────────────────────────
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg  = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(
                        WSMessage(event=WSEventType.PONG, job_id=job_id, payload={}).model_dump_json()
                    )
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_text(
                    WSMessage(event=WSEventType.PING, job_id=job_id, payload={}).model_dump_json()
                )
            except WebSocketDisconnect:
                break
    finally:
        await ws_manager.disconnect(job_id, websocket)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Reports
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/jobs/{job_id}/report", response_model=ReportResponse, tags=["Reports"])
async def generate_report(
    job_id: str,
    body: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a downloadable report (PDF, Markdown, HTML, JSON) for a completed job."""
    result = await db.execute(
        select(DBJob).where(DBJob.id == job_id, DBJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="Completed job not found")

    report_id   = str(uuid.uuid4())[:12]
    report_path = REPORT_DIR / f"{report_id}.{body.format}"
    expires_at  = datetime.utcnow() + timedelta(hours=24)

    # Generate report asynchronously
    background_tasks.add_task(
        _generate_report_file, job.result_data, report_path, body.format
    )

    return ReportResponse(
        report_id=report_id,
        job_id=job_id,
        format=body.format,
        download_url=f"/api/v1/reports/{report_id}",
        expires_at=expires_at,
        size_bytes=0,
    )


@app.get("/api/v1/reports/{report_id}", tags=["Reports"])
async def download_report(
    report_id: str,
    current_user: DBUser = Depends(get_current_user),
):
    """Download a generated report file."""
    for ext in ("pdf", "json", "md", "html"):
        path = REPORT_DIR / f"{report_id}.{ext}"
        if path.exists():
            media_types = {
                "pdf": "application/pdf",
                "json": "application/json",
                "md": "text/markdown",
                "html": "text/html",
            }
            return FileResponse(path, media_type=media_types[ext], filename=f"nydra_report_{report_id}.{ext}")
    raise HTTPException(status_code=404, detail="Report not found or expired")


async def _generate_report_file(result_data: Dict, path: Path, fmt: ReportFormat) -> None:
    """Background task: writes report to disk."""
    try:
        if fmt == ReportFormat.JSON:
            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(result_data, indent=2, default=str))
        elif fmt == ReportFormat.MARKDOWN:
            md = _result_to_markdown(result_data)
            async with aiofiles.open(path, "w") as f:
                await f.write(md)
        elif fmt == ReportFormat.HTML:
            html = _result_to_html(result_data)
            async with aiofiles.open(path, "w") as f:
                await f.write(html)
        # PDF: delegate to reportlab (imported lazily)
        elif fmt == ReportFormat.PDF:
            await asyncio.get_event_loop().run_in_executor(None, _write_pdf, result_data, path)
    except Exception as exc:
        log.error("Report generation failed: %s", exc)


def _result_to_markdown(data: Dict) -> str:
    lines = ["# Nydra Analysis Report\n", f"Generated: {datetime.utcnow().isoformat()}\n\n"]
    if q := data.get("quality"):
        lines.append(f"## Quality Score: {q.get('overall_score', 'N/A')}/100\n")
        lines.append(f"**Verdict:** {q.get('verdict', 'N/A')}\n\n")
    if issues := data.get("issues", []):
        lines.append(f"## Issues Found ({len(issues)})\n")
        for issue in issues[:20]:
            lines.append(f"- **[{issue.get('severity', '').upper()}]** {issue.get('title', '')}: {issue.get('description', '')}\n")
    return "".join(lines)


def _result_to_html(data: Dict) -> str:
    md = _result_to_markdown(data)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Nydra Report</title>
<style>body{{font-family:monospace;max-width:900px;margin:40px auto;padding:20px;
background:#0d0d0d;color:#e0e0e0}}h1,h2{{color:#00d4aa}}
strong{{color:#ff6b6b}}</style></head><body><pre>{md}</pre></body></html>"""


def _write_pdf(data: Dict, path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c    = canvas.Canvas(str(path), pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, "Nydra Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        if q := data.get("quality"):
            c.drawString(72, 690, f"Quality Score: {q.get('overall_score', 'N/A')}/100")
            c.drawString(72, 670, f"Verdict: {q.get('verdict', 'N/A')}")
        c.save()
    except ImportError:
        path.write_text(json.dumps(data, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — AI Chat
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    body: ChatRequest,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chat with Nydra AI — optionally grounded in a job's analysis results."""
    context = ""
    used_context = False

    if body.job_id:
        result = await db.execute(
            select(DBJob).where(DBJob.id == body.job_id, DBJob.user_id == current_user.id)
        )
        job = result.scalar_one_or_none()
        if job and job.result_data:
            context = f"\n\nContext from analysis (job {body.job_id[:8]}):\n{json.dumps(job.result_data, indent=2)[:4000]}"
            used_context = True

    # Build messages for agent
    system_prompt = (
        "You are Nydra, an autonomous data intelligence assistant. "
        "You help users understand their dataset analysis results, "
        "suggest data cleaning strategies, explain ML concepts, "
        "and guide users toward better data quality."
        + context
    )

    try:
        from src.core.agent import DataDoctor  # noqa: PLC0415
        # Use agent's AI suggestion capability
        # This is a stub — wire to your LLM of choice
        reply_text = f"I analyzed your request. {system_prompt[:100]}..."
    except ImportError:
        reply_text = "I'm ready to help with your data analysis. What would you like to know?"

    reply = ChatMessage(role=ChatRole.ASSISTANT, content=reply_text)
    return ChatResponse(message=reply, job_id=body.job_id, used_context=used_context, tokens_used=len(reply_text))


@app.websocket("/api/v1/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    Streaming AI chat over WebSocket.
    Client sends: {"message": "...", "job_id": "optional"}
    Server streams: {"token": "..."} chunks, then {"done": true}
    """
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    try:
        decode_access_token(token)
    except HTTPException:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()
    try:
        while True:
            data    = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")
            if not message:
                continue

            # Stream simulated response (wire to actual LLM stream here)
            words = f"I understand you're asking about: {message}. Let me analyze this for you.".split()
            for word in words:
                await websocket.send_text(json.dumps({"token": word + " "}))
                await asyncio.sleep(0.05)
            await websocket.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — System & Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    services = []
    uptime   = time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0

    # Check DB
    t0 = time.time()
    try:
        await db.execute(select(1))
        services.append(ServiceHealth(
            name="database", status=ServiceStatus.UP,
            latency_ms=round((time.time() - t0) * 1000, 2),
        ))
    except Exception as exc:
        services.append(ServiceHealth(name="database", status=ServiceStatus.DOWN, message=str(exc)))

    # Check job queue
    services.append(ServiceHealth(
        name="job_queue", status=ServiceStatus.UP,
        message=f"{job_queue.active_count()} active | {job_queue.queue_depth()} queued",
    ))

    # Check WebSocket manager
    services.append(ServiceHealth(
        name="websocket", status=ServiceStatus.UP,
        message=f"{sum(len(v) for v in ws_manager._connections.values())} active connections",
    ))

    overall = (
        ServiceStatus.UP if all(s.status == ServiceStatus.UP for s in services)
        else ServiceStatus.DEGRADED
    )

    return HealthResponse(
        status=overall,
        version=APP_VERSION,
        services=services,
        active_jobs=job_queue.active_count(),
        uptime_seconds=round(uptime, 1),
    )


@app.get("/api/v1/ping", tags=["System"])
async def ping():
    return {"pong": True, "ts": datetime.utcnow().isoformat()}


@app.get("/api/v1/system/stats", response_model=SystemStats, tags=["System"])
async def system_stats(
    _: DBUser = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: platform-wide statistics."""
    jobs_result  = await db.execute(select(DBJob))
    all_jobs     = jobs_result.scalars().all()
    today        = datetime.utcnow().date()
    jobs_today   = [j for j in all_jobs if j.created_at.date() == today]
    done_jobs    = [j for j in all_jobs if j.status == "done" and j.duration_ms]
    avg_duration = sum(j.duration_ms for j in done_jobs) / len(done_jobs) if done_jobs else 0
    success_rate = len(done_jobs) / len(all_jobs) if all_jobs else 1.0

    files_result = await db.execute(select(DBUploadedFile))
    all_files    = files_result.scalars().all()
    total_mb     = sum(f.size_bytes for f in all_files) / (1024 * 1024)

    users_result = await db.execute(select(DBUser).where(DBUser.is_active == True))
    user_count   = len(users_result.scalars().all())

    return SystemStats(
        total_jobs=len(all_jobs),
        jobs_today=len(jobs_today),
        avg_duration_ms=round(avg_duration, 1),
        success_rate=round(success_rate, 4),
        total_files_processed=len(all_files),
        total_data_mb=round(total_mb, 2),
        active_users=user_count,
        queue_depth=job_queue.queue_depth(),
    )





# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
        workers=1,  # Use 1 worker with async — let asyncio handle concurrency
        log_level=LOG_LEVEL.lower(),
        access_log=False,  # We handle logging in middleware
    )