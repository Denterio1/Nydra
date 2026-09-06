"""
╔══════════════════════════════════════════════════════════════════════╗
║                    NYDRA — API Schemas v1.0                         ║
║              Complete Type Contract: Python ↔ React                 ║
║                                                                      ║
║  Every request, response, WebSocket message, and result             ║
║  is fully typed here. This is the single source of truth.           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────

class NydraBase(BaseModel):
    """Base model — all Nydra schemas inherit from this."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_default=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class JobGoal(str, Enum):
    INSPECT         = "inspect"
    CLEAN           = "clean"
    ANALYZE         = "analyze"
    PREPARE_FOR_ML  = "prepare_for_ml"
    FULL_PIPELINE   = "full_pipeline"
    RUN_AUTOML      = "run_automl"
    DETECT_OUTLIERS = "detect_outliers"
    AUDIT           = "audit"
    IMAGES          = "images"
    TEXT_DOCS       = "text_docs"


class FileType(str, Enum):
    CSV     = "csv"
    EXCEL   = "excel"
    JSON    = "json"
    TSV     = "tsv"
    PARQUET = "parquet"
    PDF     = "pdf"
    DOCX    = "docx"
    TXT     = "txt"
    PNG     = "png"
    JPG     = "jpg"
    WEBP    = "webp"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


class OutlierMethod(str, Enum):
    SMART              = "smart"
    ISOLATION_FOREST   = "isolation_forest"
    LOF                = "lof"
    DBSCAN             = "dbscan"
    ONE_CLASS_SVM      = "one_class_svm"
    ZSCORE             = "zscore"
    IQR                = "iqr"
    MAHALANOBIS        = "mahalanobis"
    ELLIPTIC_ENVELOPE  = "elliptic_envelope"
    GRADIENT_BOOSTING  = "gradient_boosting"
    ENSEMBLE           = "ensemble"


class ImputationMethod(str, Enum):
    SMART             = "smart"
    MEAN              = "mean"
    MEDIAN            = "median"
    MODE              = "mode"
    KNN               = "knn"
    MICE              = "mice"
    MISS_FOREST       = "miss_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE          = "ensemble"
    FFILL             = "ffill"
    BFILL             = "bfill"
    INTERPOLATE       = "interpolate"


class AutoMLTask(str, Enum):
    CLASSIFICATION  = "classification"
    REGRESSION      = "regression"
    MULTICLASS      = "multiclass"
    AUTO            = "auto"


class ReportFormat(str, Enum):
    JSON     = "json"
    MARKDOWN = "markdown"
    HTML     = "html"
    PDF      = "pdf"


class WSEventType(str, Enum):
    JOB_STARTED    = "job_started"
    STEP_STARTED   = "step_started"
    STEP_DONE      = "step_done"
    PROGRESS       = "progress"
    WARNING        = "warning"
    ERROR          = "error"
    JOB_DONE       = "job_done"
    JOB_FAILED     = "job_failed"
    JOB_CANCELLED  = "job_cancelled"
    PING           = "ping"
    PONG           = "pong"


class ColumnType(str, Enum):
    NUMERIC    = "numeric"
    CATEGORICAL = "categorical"
    DATETIME   = "datetime"
    TEXT       = "text"
    BOOLEAN    = "boolean"
    UNKNOWN    = "unknown"


class Verdict(str, Enum):
    READY      = "ready"
    NEEDS_WORK = "needs_work"
    NOT_READY  = "not_ready"


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

class UserRegister(NydraBase):
    username: str   = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    email:    EmailStr
    password: str   = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(NydraBase):
    """Login with username or email in the `username` field (kept for API compat)."""
    username: str = Field(..., min_length=3, description="Username or email address")
    password: str = Field(..., min_length=1)


class OAuthCodeExchange(NydraBase):
    """Exchange a one-time OAuth code for JWT tokens."""
    code: str = Field(..., min_length=16)


class UserPublic(NydraBase):
    id:         str
    username:   str
    email:      EmailStr
    created_at: datetime
    is_active:  bool = True


class TokenResponse(NydraBase):
    access_token:  str
    refresh_token: str
    token_type:    Literal["bearer"] = "bearer"
    expires_in:    int = Field(3600, description="Seconds until access token expires")


class TokenRefresh(NydraBase):
    refresh_token: str


class APIKeyCreate(NydraBase):
    name:       str = Field(..., min_length=1, max_length=100)
    expires_in: Optional[int] = Field(None, description="Days until expiry. None = never expires")


class APIKeyPublic(NydraBase):
    id:         str
    name:       str
    key_prefix: str = Field(..., description="First 8 chars of key, e.g. 'nydra_sk'")
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used:  Optional[datetime] = None
    is_active:  bool = True


class APIKeyFull(APIKeyPublic):
    """Returned ONCE on creation — never again."""
    full_key: str = Field(..., description="Full API key — store it now, won't be shown again")


# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

class UserSettings(NydraBase):
    # Core Preferences
    default_goal:              JobGoal         = JobGoal.FULL_PIPELINE
    default_outlier_method:    OutlierMethod   = OutlierMethod.SMART
    default_imputation_method: ImputationMethod = ImputationMethod.SMART
    default_automl_trials:     int             = Field(50, ge=10, le=500)
    default_report_format:     ReportFormat    = ReportFormat.JSON
    
    # AI / LLM Config
    llm_provider:              str             = Field("groq", description="e.g. 'groq', 'openai', 'anthropic'")
    llm_model:                 str             = Field("llama-3.3-70b-versatile")
    llm_api_key:               Optional[str]   = Field(None, description="Sensitive: will be vault-encrypted")
    llm_temperature:           float           = Field(0.2, ge=0.0, le=1.0)
    llm_max_tokens:            int             = Field(1500, ge=500, le=4000)
    
    # System Preferences
    max_file_size_mb:          int             = Field(10000, ge=1, le=10000)
    enable_pii_detection:      bool            = True
    enable_bias_detection:     bool            = True
    theme:                     Literal["dark", "light", "system"] = "dark"
    language:                  str             = Field("en", pattern=r"^[a-z]{2}$")
    notifications_enabled:     bool            = True


class UserSettingsUpdate(NydraBase):
    """Partial update — all fields optional."""
    default_goal:              Optional[JobGoal]           = None
    default_outlier_method:    Optional[OutlierMethod]     = None
    default_imputation_method: Optional[ImputationMethod]  = None
    default_automl_trials:     Optional[int]               = Field(None, ge=10, le=500)
    default_report_format:     Optional[ReportFormat]      = None
    
    # AI / LLM Config
    llm_provider:              Optional[str]               = None
    llm_model:                 Optional[str]               = None
    llm_api_key:               Optional[str]               = None
    llm_temperature:           Optional[float]             = Field(None, ge=0.0, le=1.0)
    llm_max_tokens:            Optional[int]               = Field(None, ge=500, le=4000)

    enable_pii_detection:      Optional[bool]              = None
    enable_bias_detection:     Optional[bool]              = None
    theme:                     Optional[Literal["dark", "light", "system"]] = None
    notifications_enabled:     Optional[bool]              = None


# ─────────────────────────────────────────────────────────────────────────────
# LLM REGISTRY (Discovery)
# ─────────────────────────────────────────────────────────────────────────────

class LLMProviderInfo(NydraBase):
    name:         str
    format:       str
    requires_key: bool
    key_hint:     str
    free_tier:    bool
    models:       List[str]
    recommended:  str
    notes:        str


class LLMRegistryResponse(NydraBase):
    providers:    Dict[str, LLMProviderInfo]
    all_keys:     List[str]


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

class UploadedFileInfo(NydraBase):
    file_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename:   str
    file_type:  FileType
    size_bytes: int
    size_mb:    float
    checksum:   str = Field(..., description="SHA256 of file content")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    storage_path: str = Field(..., description="Server-side path to file")


class UploadResponse(NydraBase):
    file:    UploadedFileInfo
    message: str = "File uploaded successfully"


class MultiUploadResponse(NydraBase):
    """For audit feature that needs train + test files."""
    train_file: UploadedFileInfo
    test_file:  UploadedFileInfo
    message:    str = "Files uploaded successfully"


# ─────────────────────────────────────────────────────────────────────────────
# JOB REQUESTS
# ─────────────────────────────────────────────────────────────────────────────

class BaseJobRequest(NydraBase):
    file_id:       str
    goal:          JobGoal
    report_format: ReportFormat = ReportFormat.JSON
    priority:      Literal[1, 2, 3] = Field(2, description="1=low, 2=normal, 3=high")
    notify_on_done: bool = False


class InspectRequest(BaseJobRequest):
    goal: Literal[JobGoal.INSPECT] = JobGoal.INSPECT
    sample_size: Optional[int] = Field(None, description="Rows to sample. None = full file")


class CleanRequest(BaseJobRequest):
    goal:               Literal[JobGoal.CLEAN] = JobGoal.CLEAN
    imputation_method:  ImputationMethod = ImputationMethod.SMART
    outlier_method:     OutlierMethod    = OutlierMethod.SMART
    remove_duplicates:  bool             = True
    fix_dtypes:         bool             = True
    target_column:      Optional[str]    = None


class AnalyzeRequest(BaseJobRequest):
    goal:              Literal[JobGoal.ANALYZE] = JobGoal.ANALYZE
    run_stats:         bool = True
    run_correlations:  bool = True
    run_distributions: bool = True
    run_normality:     bool = True
    sensitive_columns: List[str] = Field(default_factory=list)


class PrepareMLRequest(BaseJobRequest):
    goal:              Literal[JobGoal.PREPARE_FOR_ML] = JobGoal.PREPARE_FOR_ML
    target_column:     str
    imputation_method: ImputationMethod = ImputationMethod.SMART
    outlier_method:    OutlierMethod    = OutlierMethod.SMART
    encode_categoricals: bool = True
    scale_features:      bool = True
    handle_imbalance:    bool = True


class FullPipelineRequest(BaseJobRequest):
    goal:              Literal[JobGoal.FULL_PIPELINE] = JobGoal.FULL_PIPELINE
    target_column:     Optional[str]    = None
    imputation_method: ImputationMethod = ImputationMethod.SMART
    outlier_method:    OutlierMethod    = OutlierMethod.SMART
    sensitive_columns: List[str]        = Field(default_factory=list)
    run_automl:        bool             = False
    automl_trials:     int              = Field(50, ge=10, le=500)


class AutoMLRequest(BaseJobRequest):
    goal:          Literal[JobGoal.RUN_AUTOML] = JobGoal.RUN_AUTOML
    target_column: str
    task:          AutoMLTask = AutoMLTask.AUTO
    n_trials:      int        = Field(50, ge=10, le=500)
    cv_folds:      int        = Field(5, ge=3, le=10)
    use_ensemble:  bool       = True
    time_budget:   Optional[int] = Field(None, description="Max seconds for search")
    models:        Optional[List[str]] = Field(
        None, description="Specific models to try. None = all available"
    )


class OutlierRequest(BaseJobRequest):
    goal:   Literal[JobGoal.DETECT_OUTLIERS] = JobGoal.DETECT_OUTLIERS
    method: OutlierMethod = OutlierMethod.SMART
    run_benchmark: bool   = False
    columns: Optional[List[str]] = Field(None, description="Specific columns. None = all numeric")


class AuditRequest(NydraBase):
    """Training data audit — requires two files."""
    train_file_id:    str
    test_file_id:     str
    goal:             Literal[JobGoal.AUDIT] = JobGoal.AUDIT
    target_column:    str
    sensitive_columns: List[str] = Field(default_factory=list)
    check_leakage:    bool = True
    check_labels:     bool = True
    check_bias:       bool = True
    report_format:    ReportFormat = ReportFormat.JSON
    priority:         Literal[1, 2, 3] = 2


class ImageAnalysisRequest(BaseJobRequest):
    goal:              Literal[JobGoal.IMAGES] = JobGoal.IMAGES
    check_quality:     bool = True
    detect_duplicates: bool = True
    auto_clean:        bool = False
    resize_to:         Optional[tuple[int, int]] = None


class TextAnalysisRequest(BaseJobRequest):
    goal:            Literal[JobGoal.TEXT_DOCS] = JobGoal.TEXT_DOCS
    run_sentiment:   bool = True
    run_topics:      bool = True
    run_ner:         bool = True
    run_keywords:    bool = True
    detect_pii:      bool = True
    language:        Optional[str] = None  # None = auto-detect


# Union type for routing
AnyJobRequest = Union[
    InspectRequest,
    CleanRequest,
    AnalyzeRequest,
    PrepareMLRequest,
    FullPipelineRequest,
    AutoMLRequest,
    OutlierRequest,
    AuditRequest,
    ImageAnalysisRequest,
    TextAnalysisRequest,
]


# ─────────────────────────────────────────────────────────────────────────────
# JOB STATE
# ─────────────────────────────────────────────────────────────────────────────

class JobStep(NydraBase):
    step_id:     str
    name:        str
    description: str
    status:      JobStatus = JobStatus.PENDING
    progress:    int       = Field(0, ge=0, le=100)
    started_at:  Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int]      = None
    error:       Optional[str]      = None
    warnings:    List[str]          = Field(default_factory=list)


class JobCreate(NydraBase):
    job_id:     str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id:    str
    goal:       JobGoal
    file_id:    str
    status:     JobStatus = JobStatus.PENDING
    created_at: datetime  = Field(default_factory=datetime.utcnow)
    steps:      List[JobStep] = Field(default_factory=list)
    meta:       Dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(NydraBase):
    job_id:      str
    goal:        JobGoal
    status:      JobStatus
    progress:    int = Field(0, ge=0, le=100, description="Overall progress 0-100")
    current_step: Optional[str] = None
    steps:       List[JobStep]
    created_at:  datetime
    started_at:  Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int]      = None
    error:       Optional[str]      = None
    result_url:  Optional[str]      = Field(None, description="URL to fetch results when done")


class JobListItem(NydraBase):
    job_id:     str
    goal:       JobGoal
    status:     JobStatus
    progress:   int
    filename:   str
    created_at: datetime
    duration_ms: Optional[int] = None


class JobListResponse(NydraBase):
    jobs:    List[JobListItem]
    total:   int
    page:    int
    per_page: int
    pages:   int


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET MESSAGES
# ─────────────────────────────────────────────────────────────────────────────

class WSProgressPayload(NydraBase):
    progress:     int = Field(..., ge=0, le=100)
    current_step: str
    step_progress: int = Field(0, ge=0, le=100)
    message:      str
    elapsed_ms:   int
    eta_ms:       Optional[int] = None


class WSStepPayload(NydraBase):
    step_id:     str
    step_name:   str
    description: str
    step_index:  int
    total_steps: int


class WSWarningPayload(NydraBase):
    code:    str
    message: str
    column:  Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class WSErrorPayload(NydraBase):
    code:      str
    message:   str
    traceback: Optional[str] = None
    step_id:   Optional[str] = None


class WSResultPayload(NydraBase):
    job_id:      str
    result_url:  str
    summary:     str
    score:       Optional[int] = Field(None, ge=0, le=100)
    verdict:     Optional[Verdict] = None
    total_issues: int = 0
    duration_ms: int


class WSMessage(NydraBase):
    """Single WebSocket envelope — all WS traffic uses this format."""
    event:     WSEventType
    job_id:    str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload:   Union[
        WSProgressPayload,
        WSStepPayload,
        WSWarningPayload,
        WSErrorPayload,
        WSResultPayload,
        Dict[str, Any],   # ping/pong and custom events
    ]

    @model_validator(mode="after")
    def validate_payload_type(self) -> "WSMessage":
        event_payload_map = {
            WSEventType.PROGRESS:      WSProgressPayload,
            WSEventType.STEP_STARTED:  WSStepPayload,
            WSEventType.STEP_DONE:     WSStepPayload,
            WSEventType.WARNING:       WSWarningPayload,
            WSEventType.ERROR:         WSErrorPayload,
            WSEventType.JOB_FAILED:    WSErrorPayload,
            WSEventType.JOB_DONE:      WSResultPayload,
        }
        expected = event_payload_map.get(self.event)
        if expected and not isinstance(self.payload, expected):
            raise ValueError(
                f"Event '{self.event}' expects payload type '{expected.__name__}'"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# DATASET OVERVIEW (returned after inspect)
# ─────────────────────────────────────────────────────────────────────────────

class ColumnStats(NydraBase):
    name:          str
    dtype:         str
    inferred_type: ColumnType
    count:         int
    missing:       int
    missing_pct:   float
    unique:        int
    unique_pct:    float
    # Numeric only
    mean:          Optional[float] = None
    std:           Optional[float] = None
    min:           Optional[float] = None
    p25:           Optional[float] = None
    median:        Optional[float] = None
    p75:           Optional[float] = None
    max:           Optional[float] = None
    skewness:      Optional[float] = None
    kurtosis:      Optional[float] = None
    outlier_count: Optional[int]   = None
    # Categorical only
    top_values:    Optional[List[Dict[str, Any]]] = None
    cardinality:   Optional[str] = Field(None, description="low / medium / high")


class DatasetOverview(NydraBase):
    rows:             int
    columns:          int
    file_size_mb:     float
    file_type:        FileType
    total_missing:    int
    missing_pct:      float
    total_duplicates: int
    duplicate_pct:    float
    numeric_columns:  int
    categorical_columns: int
    datetime_columns: int
    text_columns:     int
    memory_mb:        float
    column_stats:     List[ColumnStats]


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

class QualityDimension(NydraBase):
    name:        str
    score:       int = Field(..., ge=0, le=100)
    weight:      float
    weighted_score: float
    issues:      List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class QualityScoreResult(NydraBase):
    overall_score:  int = Field(..., ge=0, le=100)
    verdict:        Verdict
    grade:          Literal["A", "B", "C", "D", "F"]
    dimensions:     List[QualityDimension]
    critical_issues: List[str]
    high_issues:    List[str]
    fix_checklist:  List[str]
    estimated_model_impact: str = Field(
        ..., description="e.g. 'Low quality score may reduce model accuracy by ~15%'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE / FINDING
# ─────────────────────────────────────────────────────────────────────────────

class Issue(NydraBase):
    issue_id:   str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    code:       str = Field(..., description="e.g. 'HIGH_MISSING', 'LABEL_NOISE'")
    title:      str
    description: str
    severity:   Severity
    affected_columns: List[str] = Field(default_factory=list)
    affected_rows:    Optional[int] = None
    suggestion:  str
    auto_fixable: bool = False
    docs_url:    Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class OutlierColumnResult(NydraBase):
    column:        str
    method_used:   OutlierMethod
    outlier_count: int
    outlier_pct:   float
    outlier_indices: List[int] = Field(default_factory=list, max_length=1000)
    severity:      Severity
    suggestion:    str


class OutlierResult(NydraBase):
    method_used:      OutlierMethod
    total_outliers:   int
    outlier_pct:      float
    columns_analyzed: int
    per_column:       List[OutlierColumnResult]
    global_severity:  Severity
    recommendations:  List[str]
    benchmark:        Optional[Dict[str, Any]] = None  # if run_benchmark=True


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class ImputationColumnResult(NydraBase):
    column:          str
    method_used:     ImputationMethod
    missing_before:  int
    missing_after:   int
    missing_pct_before: float
    missing_pct_after:  float
    missing_type:    Literal["MCAR", "MAR", "MNAR", "unknown"]
    rmse:            Optional[float] = None
    mae:             Optional[float] = None


class ImputationResult(NydraBase):
    method_used:         ImputationMethod
    columns_imputed:     int
    total_filled:        int
    per_column:          List[ImputationColumnResult]
    overall_quality:     str
    recommendations:     List[str]


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class NormalityResult(NydraBase):
    column:     str
    is_normal:  bool
    confidence: float
    tests_run:  List[str]
    p_values:   Dict[str, float]
    verdict:    str
    suggested_transformation: Optional[str] = None


class CorrelationPair(NydraBase):
    col_a:       str
    col_b:       str
    pearson:     Optional[float] = None
    spearman:    Optional[float] = None
    kendall:     Optional[float] = None
    p_value:     Optional[float] = None
    significant: bool = False


class DistributionFit(NydraBase):
    column:           str
    best_fit:         str
    aic:              float
    bic:              float
    ks_statistic:     float
    ks_p_value:       float
    all_fits:         List[Dict[str, Any]] = Field(default_factory=list)


class StatsResult(NydraBase):
    normality:     List[NormalityResult]
    correlations:  List[CorrelationPair]
    distributions: List[DistributionFit]
    hypothesis_tests: List[Dict[str, Any]]
    time_series:   Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# AUTOML RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class ModelResult(NydraBase):
    rank:          int
    model_name:    str
    task:          AutoMLTask
    # Classification metrics
    accuracy:      Optional[float] = None
    auc_roc:       Optional[float] = None
    f1_macro:      Optional[float] = None
    mcc:           Optional[float] = None
    precision:     Optional[float] = None
    recall:        Optional[float] = None
    # Regression metrics
    r2:            Optional[float] = None
    rmse:          Optional[float] = None
    mae:           Optional[float] = None
    # Common
    cv_score:      float
    cv_std:        float
    train_time_s:  float
    best_params:   Dict[str, Any]
    overfitting:   bool = False
    underfitting:  bool = False


class FeatureImportanceItem(NydraBase):
    feature:    str
    importance: float
    rank:       int


class AutoMLResult(NydraBase):
    task:              AutoMLTask
    target_column:     str
    n_trials:          int
    best_model:        ModelResult
    leaderboard:       List[ModelResult]
    feature_importance: List[FeatureImportanceItem]
    ensemble_score:    Optional[float] = None
    cash_used:         bool = False
    warm_started:      bool = False
    pipeline_code:     Optional[str] = None
    recommendations:   List[str]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA AUDIT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class LeakageResult(NydraBase):
    leakage_type:     str
    detected:         bool
    severity:         Severity
    affected_features: List[str]
    description:      str
    fix:              str


class LabelQualityResult(NydraBase):
    noise_rate:           float
    inconsistent_count:   int
    ambiguous_count:      int
    label_distribution:   Dict[str, int]
    imbalance_ratio:      float
    issues:               List[Issue]


class BiasResult(NydraBase):
    demographic_bias:     bool
    representation_bias:  bool
    statistical_fairness: Dict[str, float]
    affected_features:    List[str]
    severity:             Severity
    recommendations:      List[str]


class AuditResult(NydraBase):
    overall_score:    int = Field(..., ge=0, le=100)
    verdict:          Verdict
    data_quality:     QualityScoreResult
    leakage:          List[LeakageResult]
    label_quality:    LabelQualityResult
    bias:             BiasResult
    critical_issues:  List[Issue]
    fix_checklist:    List[str]
    estimated_impact: str


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE ANALYSIS RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class ImageQualityScore(NydraBase):
    filename:        str
    overall_score:   int = Field(..., ge=0, le=100)
    brightness:      float
    contrast:        float
    sharpness:       float
    blur_score:      float
    noise_level:     float
    is_corrupted:    bool
    resolution:      tuple[int, int]
    file_size_kb:    float
    format:          str
    issues:          List[str]


class ImageDatasetResult(NydraBase):
    total_images:        int
    corrupted:           int
    duplicates:          int
    low_quality:         int
    ml_readiness_score:  int = Field(..., ge=0, le=100)
    class_distribution:  Optional[Dict[str, int]] = None
    size_distribution:   Dict[str, int]
    format_distribution: Dict[str, int]
    per_image:           List[ImageQualityScore]
    recommendations:     List[str]
    cleaning_applied:    bool = False


# ─────────────────────────────────────────────────────────────────────────────
# TEXT / DOCUMENT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class SentimentResult(NydraBase):
    label:      Literal["positive", "negative", "neutral"]
    score:      float = Field(..., ge=0.0, le=1.0)
    positive:   float
    negative:   float
    neutral:    float


class KeywordResult(NydraBase):
    keyword:   str
    score:     float
    method:    Literal["tfidf", "yake", "keybert"]


class NEREntity(NydraBase):
    text:       str
    label:      str  # PERSON, ORG, DATE, GPE, etc.
    start:      int
    end:        int
    confidence: float


class PIIFinding(NydraBase):
    pii_type:  str  # EMAIL, PHONE, NAME, ID, etc.
    value:     str  # Redacted: e.g. "j***@g***.com"
    count:     int
    severity:  Severity


class TextQualityResult(NydraBase):
    duplicate_sentences: int
    duplicate_pct:       float
    empty_sections:      int
    noise_score:         float
    coherence_score:     float
    encoding_issues:     int
    language_quality:    float
    overall_score:       int = Field(..., ge=0, le=100)
    pii_findings:        List[PIIFinding]


class DocumentResult(NydraBase):
    filename:        str
    file_type:       FileType
    language:        str
    word_count:      int
    sentence_count:  int
    paragraph_count: int
    char_count:      int
    page_count:      Optional[int] = None
    readability:     Dict[str, float]
    sentiment:       SentimentResult
    keywords:        List[KeywordResult]
    entities:        List[NEREntity]
    topics:          List[Dict[str, Any]]
    text_quality:    TextQualityResult
    summary:         str
    recommendations: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# FULL JOB RESULT (the complete response when job is done)
# ─────────────────────────────────────────────────────────────────────────────

class JobResult(NydraBase):
    job_id:      str
    goal:        JobGoal
    status:      Literal[JobStatus.DONE] = JobStatus.DONE
    created_at:  datetime
    finished_at: datetime
    duration_ms: int
    file_info:   UploadedFileInfo
    overview:    Optional[DatasetOverview]   = None
    quality:     Optional[QualityScoreResult] = None
    outliers:    Optional[OutlierResult]      = None
    imputation:  Optional[ImputationResult]   = None
    stats:       Optional[StatsResult]        = None
    automl:      Optional[AutoMLResult]       = None
    audit:       Optional[AuditResult]        = None
    images:      Optional[ImageDatasetResult] = None
    document:    Optional[DocumentResult]     = None
    issues:      List[Issue]                  = Field(default_factory=list)
    warnings:    List[str]                    = Field(default_factory=list)
    decision_log: List[Dict[str, Any]]        = Field(default_factory=list)
    agent_steps:  List[JobStep]               = Field(default_factory=list)

    @property
    def critical_issue_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)

    @property
    def summary_verdict(self) -> str:
        if self.quality:
            return self.quality.verdict
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# AI CHAT
# ─────────────────────────────────────────────────────────────────────────────

class ChatRole(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class ChatMessage(NydraBase):
    role:       ChatRole
    content:    str = Field(..., min_length=1, max_length=10_000)
    timestamp:  datetime = Field(default_factory=datetime.utcnow)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])


class ChatRequest(NydraBase):
    messages:   List[ChatMessage]
    job_id:     Optional[str] = Field(None, description="Attach context from a completed job")
    stream:     bool = True
    max_tokens: int  = Field(1000, ge=100, le=4000)


class ChatResponse(NydraBase):
    message:    ChatMessage
    job_id:     Optional[str] = None
    used_context: bool = False
    tokens_used:  int  = 0


# ─────────────────────────────────────────────────────────────────────────────
# PAGINATION & FILTERING
# ─────────────────────────────────────────────────────────────────────────────

class PaginationParams(NydraBase):
    page:     int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.per_page


class JobFilterParams(NydraBase):
    status:     Optional[JobStatus] = None
    goal:       Optional[JobGoal]   = None
    date_from:  Optional[datetime]  = None
    date_to:    Optional[datetime]  = None
    search:     Optional[str]       = Field(None, max_length=100)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM & HEALTH
# ─────────────────────────────────────────────────────────────────────────────

class ServiceStatus(str, Enum):
    UP       = "up"
    DEGRADED = "degraded"
    DOWN     = "down"


class ServiceHealth(NydraBase):
    name:    str
    status:  ServiceStatus
    latency_ms: Optional[float] = None
    message: Optional[str]      = None


class HealthResponse(NydraBase):
    status:    ServiceStatus
    version:   str = "0.5.5"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services:  List[ServiceHealth]
    active_jobs: int = 0
    uptime_seconds: float


class SystemStats(NydraBase):
    total_jobs:       int
    jobs_today:       int
    avg_duration_ms:  float
    success_rate:     float
    total_files_processed: int
    total_data_mb:    float
    active_users:     int
    queue_depth:      int


# ─────────────────────────────────────────────────────────────────────────────
# ERRORS
# ─────────────────────────────────────────────────────────────────────────────

class ErrorCode(str, Enum):
    # Auth
    INVALID_CREDENTIALS  = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED        = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMS   = "INSUFFICIENT_PERMS"
    # Files
    FILE_NOT_FOUND       = "FILE_NOT_FOUND"
    FILE_TOO_LARGE       = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT   = "UNSUPPORTED_FORMAT"
    CORRUPTED_FILE       = "CORRUPTED_FILE"
    # Jobs
    JOB_NOT_FOUND        = "JOB_NOT_FOUND"
    JOB_ALREADY_RUNNING  = "JOB_ALREADY_RUNNING"
    JOB_CANCELLED        = "JOB_CANCELLED"
    # Data
    EMPTY_DATASET        = "EMPTY_DATASET"
    TARGET_NOT_FOUND     = "TARGET_NOT_FOUND"
    INSUFFICIENT_DATA    = "INSUFFICIENT_DATA"
    # System
    AGENT_ERROR          = "AGENT_ERROR"
    TIMEOUT              = "TIMEOUT"
    RATE_LIMITED         = "RATE_LIMITED"
    INTERNAL_ERROR       = "INTERNAL_ERROR"


class ErrorDetail(NydraBase):
    field:   Optional[str] = None
    message: str
    code:    Optional[str] = None


class ErrorResponse(NydraBase):
    error:     ErrorCode
    message:   str
    details:   List[ErrorDetail] = Field(default_factory=list)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp:  datetime = Field(default_factory=datetime.utcnow)
    docs_url:   Optional[str] = None


class ValidationErrorResponse(NydraBase):
    error:   Literal["VALIDATION_ERROR"] = "VALIDATION_ERROR"
    message: str = "Request validation failed"
    errors:  List[ErrorDetail]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

class ReportRequest(NydraBase):
    job_id:  str
    format:  ReportFormat = ReportFormat.PDF
    include_charts:  bool = True
    include_raw_data: bool = False


class ReportResponse(NydraBase):
    report_id:   str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    job_id:      str
    format:      ReportFormat
    download_url: str
    expires_at:  datetime
    size_bytes:  int


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTS  (everything React needs to import)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enums
    "JobStatus", "JobGoal", "FileType", "Severity", "OutlierMethod",
    "ImputationMethod", "AutoMLTask", "ReportFormat", "WSEventType",
    "ColumnType", "Verdict", "ErrorCode", "ChatRole", "ServiceStatus",
    # Auth
    "UserRegister", "UserLogin", "OAuthCodeExchange", "UserPublic", "TokenResponse",
    "TokenRefresh", "APIKeyCreate", "APIKeyPublic", "APIKeyFull",
    # Settings
    "UserSettings", "UserSettingsUpdate",
    # Upload
    "UploadedFileInfo", "UploadResponse", "MultiUploadResponse",
    # Requests
    "InspectRequest", "CleanRequest", "AnalyzeRequest", "PrepareMLRequest",
    "FullPipelineRequest", "AutoMLRequest", "OutlierRequest", "AuditRequest",
    "ImageAnalysisRequest", "TextAnalysisRequest", "AnyJobRequest",
    # Jobs
    "JobStep", "JobCreate", "JobStatusResponse", "JobListItem", "JobListResponse",
    # WebSocket
    "WSProgressPayload", "WSStepPayload", "WSWarningPayload",
    "WSErrorPayload", "WSResultPayload", "WSMessage",
    # Results
    "ColumnStats", "DatasetOverview", "QualityDimension", "QualityScoreResult",
    "Issue", "OutlierColumnResult", "OutlierResult",
    "ImputationColumnResult", "ImputationResult",
    "NormalityResult", "CorrelationPair", "DistributionFit", "StatsResult",
    "ModelResult", "FeatureImportanceItem", "AutoMLResult",
    "LeakageResult", "LabelQualityResult", "BiasResult", "AuditResult",
    "ImageQualityScore", "ImageDatasetResult",
    "SentimentResult", "KeywordResult", "NEREntity", "PIIFinding",
    "TextQualityResult", "DocumentResult",
    "JobResult",
    # Chat
    "ChatMessage", "ChatRequest", "ChatResponse",
    # Pagination
    "PaginationParams", "JobFilterParams",
    # System
    "ServiceHealth", "HealthResponse", "SystemStats",
    # Errors
    "ErrorDetail", "ErrorResponse", "ValidationErrorResponse",
    # Reports
    "ReportRequest", "ReportResponse",
]