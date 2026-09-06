# Changelog

## v1.0.0 — 2026-09-05

First stable public release since v0.5.0. Major rebuild across the board:
project renamed dataDoctor → Nydra, full backend stabilization, and a
verified 196-test suite.

### Added
- FastAPI backend (`api.py`) with JWT auth (access + refresh, `jti`-based
  uniqueness), API key dual-auth, Google OAuth
- WebSocket-based job progress system with async `JobQueue`
- Rebuilt `agent.py` orchestrator — 9 goals, pipeline planning, caching,
  self-healing, undo/redo session state, plugin system
- Advanced data layer: `advanced_imputer`, `advanced_outlier` (11 detectors),
  `advanced_automl` (15+ models, Optuna CASH), `advanced_stats`,
  `quality_score`
- Vision & text audit orchestrators (`src/vision_nlp/`): data leakage,
  label quality, bias detection, training readiness, image/text audits
- Rust/Python hybrid stats engine (`nydra_core.py`) with pure-Python fallback
- Full pytest suite: 196 tests across core pipeline, audit layer, stats
  engine, and API

### Fixed
- `StepTracker`/`JobStep` datetime serialization crash (jobs failed
  immediately on creation)
- Exception handler datetime crash on error responses
- `is_active` incorrectly defaulting to `False` on registration
- `JobQueue` had no shutdown path, causing unclean server shutdown
- Windows-specific `PermissionError` on upload cleanup
  (unlink-while-file-open)
- `achieve_goal()` silently receiving `None` instead of the loaded
  DataFrame, causing empty analysis results
- Wrong agent class imported in `_run_agent()`, silently falling back to
  stub mode for all analysis
- Raw DataFrames not JSON-serializable in job results
- `_step_outliers()` crashing with `NameError` (missing severity
  classification helper) and `KeyError` (reading recommendations from the
  wrong object)
- `detect_outliers_mad` silently missed outliers in majority-tie data
  (>50% identical values) — now falls back to IQR, which isn't fooled by
  this data shape
- Deprecated `@app.on_event("startup")` merged into the `lifespan` handler

### Known limitations
- Login requires `username`, not email — email-as-login-identifier is not
  yet supported (separate from Google OAuth, which does use email)
- Some pre-existing deprecation warnings remain (Pydantic `json_encoders`,
  pandas `select_dtypes`) — tracked for a future cleanup pass