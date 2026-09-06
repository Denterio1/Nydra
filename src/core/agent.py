"""
agent.py — Nydra: Intelligent Data Orchestration Agent
============================================================
The central brain of nydra. Orchestrates all modules,
manages sessions, plans pipelines, self-heals on failures,
and provides a unified API for CLI, Web UI, and REST API.

Usage:
    from src.core.agent import Nydra

    doctor = Nydra()
    report = doctor.inspect("examples/sample_sales.csv")
    print(report["summary"])

    # Goal-oriented mode
    result = doctor.achieve_goal(df, goal="prepare_for_ml")

    # Full pipeline
    result = doctor.full_pipeline(df, target_column="churn")
"""

from __future__ import annotations

import time
import logging
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.smart_report import SmartReport
from src.data import analyzer
from src.data.quality_score import DataQualityScorer
from src.data.advanced_stats import AdvancedStats
# ── Core data modules ──────────────────────────────────────────────────────
from src.data.loader import load_csv
from src.data.analyzer import full_report
from src.data.cleaner import MissingStrategy, handle_missing, remove_duplicates
from src.data.ml_readiness import ml_readiness

# ── Advanced modules (new) ─────────────────────────────────────────────────
from src.data.advanced_imputer import SmartImputer, ImputationReporter, ImputationAudit
from src.data.advanced_outlier import (
    detect_outliers,
    OutlierAnalyzer,
    OutlierReport,
    SmartDetector,
    EnsembleOutlierDetector,
    OutlierReporter,
)
# ── Audit Layer (v0.6.0) ──────────────────────────────
from src.vision_nlp.data_leakage       import LeakageOrchestrator as DataLeakageDetector
from src.vision_nlp.label_quality      import LabelQualityAnalyzer as LabelQualityChecker
from src.vision_nlp.bias_detector      import BiasDetector
from src.vision_nlp.training_readiness import TrainingReadinessAnalyzer as TrainingReadinessScore
from src.vision_nlp.audit_orchestrator import AuditOrchestrator, run_image_audit as run_audit, run_image_audit as quick_audit
from src.vision_nlp.text_orchestrator  import TextOrchestrator, run_text_audit
from src.vision_nlp.training_orchestrator import TrainingOrchestrator, run_training_audit


def _severity_from_rate(rate: float) -> str:
    """Classify outlier severity from the consensus outlier rate."""
    if rate >= 0.10:
        return "high"
    elif rate >= 0.03:
        return "medium"
    elif rate > 0:
        return "low"
    return "none"

# ── Optional modules — graceful fallback ───────────────────────────────────
def _safe_import(module_path: str, names: List[str]) -> Dict[str, Any]:
    """Import names from module, return None for unavailable ones."""
    result = {}
    try:
        import importlib
        mod = importlib.import_module(module_path)
        for name in names:
            result[name] = getattr(mod, name, None)
    except ImportError:
        for name in names:
            result[name] = None
    return result

_preparator   = _safe_import("src.data.preparator",          ["prepare_for_ml"])
_drift        = _safe_import("src.data.drift",               ["detect_drift"])
_memory       = _safe_import("src.data.memory",              ["DataMemory"])
_ai           = _safe_import("src.data.ai_suggestions",      ["get_ai_suggestions"])
_feat_eng     = _safe_import("src.data.feature_engineer",    ["engineer_features"])
_target_det   = _safe_import("src.data.target_detector",     ["detect_target"])
_corr_net     = _safe_import("src.data.correlation_network", ["build_correlation_network"])
_automl       = _safe_import("src.data.advanced_automl",     ["AdvancedAutoML", "AutoMLConfig"])
_feat_imp     = _safe_import("src.data.feature_importance",  ["compute_feature_importance"])
_split        = _safe_import("src.data.split_advisor",       ["advise_split"])
_imbalance    = _safe_import("src.data.imbalance_detector",  ["detect_imbalance"])
_pipeline_exp = _safe_import("src.data.pipeline_export",     ["export_pipeline"])
_dna          = _safe_import("src.data.cognitive_dna",       ["CognitiveDNA"])
_dna_mem      = _safe_import("src.data.dna_memory",          ["DNAMemory"])
_relations    = _safe_import("src.data.relationships",       ["detect_relationships"])
_adv_stats    = _safe_import("src.data.advanced_stats",      ["AdvancedStats"])
_quality_sc   = _safe_import("src.data.quality_score",       ["DataQualityScore"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nydra.agent")


# ─────────────────────────────────────────────
#  Enums & Constants
# ─────────────────────────────────────────────

class AgentGoal(str, Enum):
    INSPECT          = "inspect"
    CLEAN            = "clean"
    ANALYZE          = "analyze"
    PREPARE_FOR_ML   = "prepare_for_ml"
    FULL_PIPELINE    = "full_pipeline"
    DETECT_OUTLIERS  = "detect_outliers"
    DETECT_DRIFT     = "detect_drift"
    GENERATE_REPORT  = "generate_report"
    COMPUTE_DNA      = "compute_dna"
    RUN_AUTOML       = "run_automl"


class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────

@dataclass
class PipelineStep:
    """A single step in the agent's execution plan."""
    name: str
    fn: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    fallback: Optional[Callable] = None
    required: bool = True


@dataclass
class DecisionLog:
    """Records why the agent made each decision."""
    step: str
    decision: str
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentSession:
    """Full session state — supports undo/redo."""
    session_id: str
    source: str
    original_df: Optional[pd.DataFrame] = None
    current_df: Optional[pd.DataFrame] = None
    steps_taken: List[str] = field(default_factory=list)
    decisions: List[DecisionLog] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    history: List[pd.DataFrame] = field(default_factory=list)  # for undo
    start_time: float = field(default_factory=time.time)

    def snapshot(self):
        """Save current state for undo."""
        if self.current_df is not None:
            self.history.append(self.current_df.copy())

    def undo(self) -> bool:
        """Roll back last transformation."""
        if len(self.history) >= 2:
            self.history.pop()
            self.current_df = self.history[-1].copy()
            return True
        return False

    def elapsed(self) -> float:
        return round(time.time() - self.start_time, 2)


# ─────────────────────────────────────────────
#  Pipeline Planner
# ─────────────────────────────────────────────

class PipelinePlanner:
    """
    Analyzes data characteristics and builds an optimal execution plan.
    Selects which modules to run and in what order based on:
    - Dataset size
    - Missing value rate
    - Data types
    - User goal
    """

    def __init__(self, agent: "Nydra"):
        self._agent = agent

    def plan(self, df: pd.DataFrame, goal: AgentGoal) -> List[str]:
        """Return ordered list of step names for this goal + data profile."""
        n_rows, n_cols = df.shape
        missing_rate = df.isnull().mean().mean()
        has_numeric = len(df.select_dtypes(include=[np.number]).columns) > 0
        is_large = n_rows > 50_000
        has_missing = missing_rate > 0.0

        plan: List[str] = []

        # Always start with analysis
        plan.append("analyze")

        # Conditional steps
        if has_missing:
            plan.append("impute")
        plan.append("remove_duplicates")
        plan.append("detect_outliers")

        if goal == AgentGoal.INSPECT:
            plan += ["ml_readiness", "relationships"]

        elif goal == AgentGoal.CLEAN:
            pass  # already covered above

        elif goal == AgentGoal.PREPARE_FOR_ML:
            plan += ["feature_engineering", "target_detection",
                     "ml_readiness", "split_advisor", "imbalance_check"]

        elif goal == AgentGoal.FULL_PIPELINE:
            plan += [
                "feature_engineering", "target_detection",
                "correlation_network", "ml_readiness",
                "advanced_stats", "quality_score",
                "split_advisor", "imbalance_check",
                "automl", "cognitive_dna",
            ]
            if not is_large:
                plan.append("learning_curves")

        elif goal == AgentGoal.RUN_AUTOML:
            plan += ["feature_engineering", "target_detection",
                     "ml_readiness", "automl"]

        elif goal == AgentGoal.COMPUTE_DNA:
            plan.append("cognitive_dna")

        # Always end with report
        plan.append("build_report")

        # Log the plan
        self._agent._log_decision(
            step="pipeline_planner",
            decision=f"Plan: {' → '.join(plan)}",
            reason=f"Goal={goal}, rows={n_rows}, missing={missing_rate:.2%}, large={is_large}"
        )
        return plan


# ─────────────────────────────────────────────
#  Cache Manager
# ─────────────────────────────────────────────

class CacheManager:
    """
    In-memory cache for expensive computations.
    Key = (session_id, step_name, data_hash).
    Avoids recomputing same result twice.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}

    def _key(self, session_id: str, step: str, df: pd.DataFrame) -> str:
        h = str(hash(pd.util.hash_pandas_object(df).sum()))
        return f"{session_id}_{step}_{h}"

    def get(self, session_id: str, step: str, df: pd.DataFrame) -> Optional[Any]:
        return self._store.get(self._key(session_id, step, df))

    def set(self, session_id: str, step: str, df: pd.DataFrame, value: Any):
        self._store[self._key(session_id, step, df)] = value

    def clear(self, session_id: Optional[str] = None):
        if session_id:
            self._store = {k: v for k, v in self._store.items()
                          if not k.startswith(session_id)}
        else:
            self._store.clear()


# ─────────────────────────────────────────────
#  Step Executor
# ─────────────────────────────────────────────

class StepExecutor:
    """
    Executes pipeline steps with:
    - Error isolation (one step fails → others continue)
    - Automatic fallback on failure
    - Timing per step
    - Progress callbacks
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self._callback = progress_callback

    def run(self, step: PipelineStep) -> PipelineStep:
        step.status = StepStatus.RUNNING
        if self._callback:
            self._callback(step.name, "running")

        t0 = time.time()
        try:
            step.result = step.fn(*step.args, **step.kwargs)
            step.status = StepStatus.DONE
        except Exception as e:
            step.error = str(e)
            logger.warning(f"[{step.name}] Failed: {e}")
            if step.fallback:
                try:
                    step.result = step.fallback(*step.args, **step.kwargs)
                    step.status = StepStatus.DONE
                    step.error = f"Used fallback (original error: {e})"
                except Exception as fe:
                    step.status = StepStatus.FAILED
                    step.error = f"Both primary and fallback failed: {e} | {fe}"
            else:
                step.status = StepStatus.FAILED if step.required else StepStatus.SKIPPED

        step.duration = round(time.time() - t0, 3)
        if self._callback:
            self._callback(step.name, step.status)
        return step


# ─────────────────────────────────────────────
#  Nydra — Main Agent Class
# ─────────────────────────────────────────────

class Nydra:
    """
    Intelligent data orchestration agent.

    Capabilities:
    - Auto-plans pipeline based on data + goal
    - Self-heals on module failures
    - Maintains session state with undo/redo
    - Caches expensive computations
    - Logs every decision with reasoning
    - Integrates all 20+ Nydra modules
    - Plugin-ready architecture

    Parameters
    ----------
    remove_dupes      : Auto-remove duplicate rows
    missing_strategy  : Default imputation strategy
    fill_value        : Used when missing_strategy == "fill"
    outlier_method    : Default outlier detection method
    scaler            : Feature scaling method
    verbose           : Log progress to console
    progress_callback : Called on each step (name, status)
    """

    VERSION = "0.5.0"

    def __init__(
        self,
        remove_dupes: bool = True,
        missing_strategy: MissingStrategy = "mean",
        fill_value: Any = None,
        outlier_method: str = "smart",
        outlier_contamination: float = 0.05,
        scaler: str = "robust",
        verbose: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        self.remove_dupes = remove_dupes
        self.missing_strategy = missing_strategy
        self.fill_value = fill_value
        self.outlier_method = outlier_method
        self.outlier_contamination = outlier_contamination
        self.scaler = scaler
        self.verbose = verbose

        self._session: Optional[AgentSession] = None
        self._cache = CacheManager()
        self._executor = StepExecutor(progress_callback)
        self._planner = PipelinePlanner(self)
        self._plugins: Dict[str, Callable] = {}
        self._decision_log: List[DecisionLog] = []
        
    def quality_score(self, df):
        return DataQualityScorer().score(df)
     
    def advanced_outliers(self, df: pd.DataFrame, contamination: float = 0.05) -> OutlierReport:
        analyzer = OutlierAnalyzer(contamination=contamination, verbose=False)
        return analyzer.analyze(df)  

    def generate_report(self, profile, df, dataset_name="Dataset",
                    output_dir="reports", formats=["pdf","docx","pptx"]):
        reporter = SmartReport(output_dir=output_dir)
        return reporter.generate_all(profile, df, dataset_name, formats=formats)
     
    def audit_training_data(self, train_df, test_df=None, target_col="target", sensitive_cols=None):
        """
        Runs comprehensive data leakage, bias, and label quality audit on training data.
        
        Returns:
        --------
        dict containing leakage results, bias findings, and label quality report.
        """
        return run_training_audit(train_df, test_df, target_col, sensitive_cols)

    def audit_text(self, text_or_df: Union[str, pd.DataFrame], text_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs comprehensive text analysis and quality audit.
        
        Parameters:
        -----------
        text_or_df : str or pd.DataFrame
            The raw text string or a DataFrame containing a text column.
        text_column : Optional[str]
            The column name to analyze if a DataFrame is provided.
            
        Returns:
        --------
        dict containing intelligence results and quality findings.
        """
        return run_text_audit(text_or_df, text_column)

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def inspect(self, filepath: str) -> Dict[str, Any]:
        """
        Full inspection pipeline on a file.
        Supports: CSV, Excel, JSON, TSV, Parquet.
        """
        self._log(f"📂 Loading: {filepath}")
        raw_data = load_csv(filepath)

        if isinstance(raw_data, dict) and "df" in raw_data:
            df = raw_data["df"]
        elif isinstance(raw_data, dict):
            df = pd.DataFrame(raw_data.get("data", raw_data))
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data
        else:
            df = pd.DataFrame(raw_data)

        self._start_session(filepath, df)
        return self._run_goal(df, AgentGoal.INSPECT, source=filepath)

    def inspect_df(self, df: pd.DataFrame, name: str = "dataframe") -> Dict[str, Any]:
        """Inspect a DataFrame directly (no file needed)."""
        self._start_session(name, df)
        return self._run_goal(df, AgentGoal.INSPECT, source=name)

    def clean(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Clean a DataFrame — remove dupes, handle missing, remove outliers."""
        self._start_session("clean", df)
        return self._run_goal(df, AgentGoal.CLEAN)

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Deep statistical analysis of a DataFrame."""
        self._start_session("analyze", df)
        return self._run_goal(df, AgentGoal.ANALYZE)

    def prepare_for_ml(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full ML preparation pipeline."""
        self._start_session("prepare_ml", df)
        return self._run_goal(df, AgentGoal.PREPARE_FOR_ML,
                              target_column=target_column)

    def full_pipeline(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run everything — the complete Nydra pipeline."""
        self._start_session("full_pipeline", df)
        return self._run_goal(df, AgentGoal.FULL_PIPELINE,
                              target_column=target_column)

    def run_automl(
        self, df: pd.DataFrame, target_column: str, n_trials: int = 50
    ) -> Dict[str, Any]:
        """Run advanced AutoML on the dataset."""
        self._start_session("automl", df)
        return self._run_goal(df, AgentGoal.RUN_AUTOML,
                              target_column=target_column, n_trials=n_trials)

    def detect_outliers_full(
        self, df: pd.DataFrame, method: str = "ensemble"
    ) -> Dict[str, Any]:
        """Full outlier detection + report."""
        # result is an OutlierReport object
        report = detect_outliers(df, strategy=method,
                                  contamination=self.outlier_contamination)
        reporter = OutlierReporter(report, df)
        
        # Determine clean df - consensus mask removes rows that most methods agree on
        mask = report.ensemble.consensus_mask
        clean_df = df[~mask].copy()

        return {
            "result": report,
            "report": reporter.generate(),
            "report_markdown": reporter.to_markdown(),
            "clean_df": clean_df.reset_index(drop=True),
        }

    def detect_drift(
        self, df_reference: pd.DataFrame, df_current: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare two datasets for data drift."""
        fn = _drift.get("detect_drift")
        if fn is None:
            return {"error": "drift module not available"}
        return fn(df_reference, df_current)

    def compute_dna(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute Cognitive Data DNA fingerprint."""
        cls = _dna.get("CognitiveDNA")
        if cls is None:
            return {"error": "cognitive_dna module not available"}
        dna = cls(df)
        return dna.compute()

    def get_ai_suggestions(
        self, df: pd.DataFrame, api_key: str, provider: str = "groq"
    ) -> str:
        """Get AI-powered suggestions for the dataset."""
        fn = _ai.get("get_ai_suggestions")
        if fn is None:
            return "AI suggestions module not available."
        analysis = full_report(df) if isinstance(df, pd.DataFrame) else {}
        return fn(analysis, api_key=api_key, provider=provider)

    def achieve_goal(
        self, df: pd.DataFrame, goal: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Goal-oriented mode — user states a goal, agent plans + executes.
        
        Goals: 'inspect' | 'clean' | 'analyze' | 'prepare_for_ml' |
               'full_pipeline' | 'detect_outliers' | 'run_automl' |
               'compute_dna' | 'generate_report'
        """
        try:
            agent_goal = AgentGoal(goal)
        except ValueError:
            return {"error": f"Unknown goal: {goal}. Available: {[g.value for g in AgentGoal]}"}

        self._start_session(goal, df)
        return self._run_goal(df, agent_goal, **kwargs)

    def undo(self) -> Optional[pd.DataFrame]:
        """Undo last transformation on current session."""
        if self._session and self._session.undo():
            self._log("↩️  Undo successful")
            return self._session.current_df
        return None

    def get_session_info(self) -> Dict[str, Any]:
        """Return current session metadata."""
        if not self._session:
            return {"error": "No active session"}
        return {
            "session_id": self._session.session_id,
            "source": self._session.source,
            "steps_taken": self._session.steps_taken,
            "elapsed_s": self._session.elapsed(),
            "decisions": [
                {"step": d.step, "decision": d.decision, "reason": d.reason}
                for d in self._session.decisions
            ],
        }

    def get_decision_log(self) -> List[Dict]:
        """Return full decision audit log."""
        return [
            {"step": d.step, "decision": d.decision, "reason": d.reason}
            for d in self._decision_log
        ]

    def register_plugin(self, name: str, fn: Callable):
        """Register a custom plugin step."""
        self._plugins[name] = fn
        self._log(f"🔌 Plugin registered: {name}")

    def benchmark_outlier_methods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Benchmark all outlier methods and return comparison table."""
        return benchmark_outliers(df, contamination=self.outlier_contamination)

    def impute_advanced(
        self, df: pd.DataFrame, strategy: str = "smart"
    ) -> Dict[str, Any]:
        """Run advanced imputation with full report."""
        imputer = SmartImputer()
        imputed_df = imputer.fit_transform(df)
        reporter = ImputationReporter(df, imputed_df)
        return {
            "imputed_df": imputed_df,
            "report": reporter.report(),
            "n_filled": int(df.isnull().sum().sum() - imputed_df.isnull().sum().sum()),
        }

    # ─────────────────────────────────────────
    #  Core Pipeline Runner
    # ─────────────────────────────────────────

    def _run_goal(
        self,
        df: pd.DataFrame,
        goal: AgentGoal,
        source: str = "",
        target_column: Optional[str] = None,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Core execution engine.
        Builds plan → executes steps → collects results → builds output.
        """
        session = self._session
        session.snapshot()

        # Get execution plan
        plan = self._planner.plan(df, goal)
        current_df = df.copy()
        output: Dict[str, Any] = {"goal": goal.value, "source": source}
        step_log: List[Dict] = []

        self._log(f"🎯 Goal: {goal.value} | Plan: {' → '.join(plan)}")

        for step_name in plan:
            self._log(f"  ▶ {step_name}...")
            t0 = time.time()
            result = self._execute_step(step_name, current_df, target_column, n_trials, output)
            duration = round(time.time() - t0, 3)

            if result is not None:
                output[step_name] = result
                # Update current_df if step returns cleaned data
                if isinstance(result, dict) and "clean_df" in result:
                    current_df = result["clean_df"]
                    session.snapshot()
                elif isinstance(result, pd.DataFrame):
                    current_df = result
                    session.snapshot()

            session.steps_taken.append(step_name)
            step_log.append({"step": step_name, "duration_s": duration,
                             "status": "done" if result is not None else "failed"})

        session.current_df = current_df
        output.update({
            "clean_df": current_df,
            "shape_original": df.shape,
            "shape_cleaned": current_df.shape,
            "rows_removed": df.shape[0] - current_df.shape[0],
            "step_log": step_log,
            "session_id": session.session_id,
            "elapsed_s": session.elapsed(),
            "decision_log": self.get_decision_log(),
        })

        # Build final human-readable summary
        output["summary"] = self._build_summary(df, current_df, output, source)

        # ── Compatibility Layer for Tests ──
        # Existing tests expect certain keys in the output dict
        if "analyze" in output:
            output["raw_analysis"] = output["analyze"]
        
        output["clean_data"] = {"df": current_df}
        
        cleaning_log = {}
        if "remove_duplicates" in output and isinstance(output["remove_duplicates"], dict):
            removed = output["remove_duplicates"].get("removed", 0)
            if removed > 0:
                cleaning_log["duplicates_removed"] = removed
        if "impute" in output and isinstance(output["impute"], dict):
            cleaning_log["missing_filled"] = output["impute"].get("filled", 0)
        
        output["cleaning_log"] = cleaning_log

        self._log(f"✅ Done in {session.elapsed()}s")
        return output

    # ─────────────────────────────────────────
    #  Step Dispatcher
    # ─────────────────────────────────────────

    def _execute_step(
        self,
        step_name: str,
        df: pd.DataFrame,
        target_column: Optional[str],
        n_trials: int,
        current_output: Dict,
    ) -> Any:
        """Dispatch to the correct module based on step name."""
        try:
            # ── Core steps ──
            if step_name == "analyze":
                return self._step_analyze(df)

            elif step_name == "impute":
                return self._step_impute(df)

            elif step_name == "remove_duplicates":
                return self._step_remove_dupes(df)

            elif step_name == "detect_outliers":
                return self._step_outliers(df)

            elif step_name == "ml_readiness":
                return self._step_ml_readiness(df)

            elif step_name == "relationships":
                return self._step_relationships(df)

            elif step_name == "feature_engineering":
                return self._step_feature_engineering(df)

            elif step_name == "target_detection":
                return self._step_target_detection(df, target_column)

            elif step_name == "correlation_network":
                return self._step_correlation_network(df)

            elif step_name == "advanced_stats":
                return self._step_advanced_stats(df)

            elif step_name == "quality_score":
                return self._step_quality_score(df, current_output)

            elif step_name == "split_advisor":
                return self._step_split_advisor(df, target_column)

            elif step_name == "imbalance_check":
                return self._step_imbalance(df, target_column)

            elif step_name == "automl":
                return self._step_automl(df, target_column, n_trials)

            elif step_name == "cognitive_dna":
                return self._step_cognitive_dna(df)

            elif step_name == "build_report":
                return self._step_build_report(df, current_output)

            # ── Plugin steps ──
            elif step_name in self._plugins:
                return self._plugins[step_name](df)

            else:
                self._log(f"  ⚠ Unknown step: {step_name}")
                return None

        except Exception as e:
            logger.warning(f"[{step_name}] Error: {e}\n{traceback.format_exc()}")
            self._log_decision(step_name, "FAILED", str(e))
            return {"error": str(e)}

    # ─────────────────────────────────────────
    #  Individual Steps
    # ─────────────────────────────────────────

    def _step_analyze(self, df: pd.DataFrame) -> Dict:
        """Full statistical analysis."""
        cached = self._cache.get(self._session.session_id, "analyze", df)
        if cached:
            self._log_decision("analyze", "Using cached result", "Same data hash")
            return cached
        result = full_report({"df": df})
        self._cache.set(self._session.session_id, "analyze", df, result)
        return result

    def _step_impute(self, df: pd.DataFrame) -> Dict:
        """Smart imputation — auto-selects best method."""
        missing_rate = df.isnull().mean().mean()
        if missing_rate == 0:
            self._log_decision("impute", "Skipped", "No missing values")
            return {"skipped": True, "reason": "No missing values"}

        method = "knn" if missing_rate < 0.2 else "mice"
        self._log_decision("impute", f"Using {method}", f"missing_rate={missing_rate:.2%}")

        try:
            imputer = SmartImputer()
            result = imputer.fit_transform(df)
            imputed = result.df_imputed
            filled = int(df.isnull().sum().sum() - imputed.isnull().sum().sum())
            return {"clean_df": imputed, "filled": filled, "method": method}
        except Exception:
            # Fallback to basic imputation
            self._log_decision("impute", "Fallback to median", "SmartImputer failed")
            data_dict = {"df": df}
            filled_data, log = handle_missing(data_dict, strategy=self.missing_strategy,
                                              fill_value=self.fill_value)
            return {"clean_df": filled_data["df"], "log": log, "method": "median_fallback"}

    def _step_remove_dupes(self, df: pd.DataFrame) -> Dict:
        """Remove duplicate rows."""
        if not self.remove_dupes:
            return {"skipped": True}
        n_before = len(df)
        data_dict = {"df": df}
        clean_data, removed = remove_duplicates(data_dict)
        self._log_decision("remove_duplicates", f"Removed {removed}", f"Before={n_before}")
        return {"clean_df": clean_data["df"], "removed": removed}

    def _step_outliers(self, df: pd.DataFrame) -> Dict:
        """Smart outlier detection."""
        try:
            result = detect_outliers(df, method=self.outlier_method,
                                      contamination=self.outlier_contamination)
            reporter = OutlierReporter(result, df)
            report = reporter.generate()
            self._log_decision(
                "detect_outliers",
                f"Found {result.n_outliers} outliers ({result.outlier_rate:.1%})",
                f"Method: {result.smart_choice}, Outliers: {result.n_outliers} ({result.outlier_rate:.1%})"
            )
            return {
                "n_outliers": result.n_outliers,
                "outlier_rate": result.outlier_rate,
                "severity": _severity_from_rate(result.outlier_rate),
                "method_used": result.smart_choice,  
                "outlier_indices": result.outlier_indices.tolist(),
                "recommendations": result.recommendations,
                "report": report,
            }
        except Exception as e:
            return {"error": str(e)}

    def _step_ml_readiness(self, df: pd.DataFrame) -> Dict:
        """Compute ML Readiness Score."""
        try:
            # ml_readiness expects (data_dict, outliers_dict)
            data_dict = {"df": df}
            # We use the simple detector for the readiness score
            simple_outliers = analyzer.detect_outliers(data_dict)
            return ml_readiness(data_dict, simple_outliers)
        except Exception as e:
            return {"error": str(e)}

    def _step_relationships(self, df: pd.DataFrame) -> Dict:
        """Detect column relationships."""
        fn = _relations.get("detect_relationships")
        if fn is None:
            return {"error": "relationships module not available"}
        try:
            return fn(df)
        except Exception as e:
            return {"error": str(e)}

    def _step_feature_engineering(self, df: pd.DataFrame) -> Dict:
        """Auto feature engineering."""
        fn = _feat_eng.get("engineer_features")
        if fn is None:
            return {"error": "feature_engineer module not available"}
        try:
            engineered = fn(df)
            new_features = len(engineered.columns) - len(df.columns)
            self._log_decision("feature_engineering", f"Added {new_features} features",
                                "Date/numeric/text transformations applied")
            return {"engineered_df": engineered, "new_features": new_features}
        except Exception as e:
            return {"error": str(e)}

    def _step_target_detection(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Dict:
        """Detect or confirm target column."""
        if target_column:
            self._log_decision("target_detection", f"Using provided: {target_column}", "User specified")
            return {"target": target_column, "method": "user_specified"}
        fn = _target_det.get("detect_target")
        if fn is None:
            return {"error": "target_detector module not available"}
        try:
            result = fn(df)
            self._log_decision("target_detection", f"Detected: {result.get('target')}",
                                result.get("reason", ""))
            return result
        except Exception as e:
            return {"error": str(e)}

    def _step_correlation_network(self, df: pd.DataFrame) -> Dict:
        """Build correlation network."""
        fn = _corr_net.get("build_correlation_network")
        if fn is None:
            return {"error": "correlation_network module not available"}
        try:
            return fn(df)
        except Exception as e:
            return {"error": str(e)}

    def _step_advanced_stats(self, df: pd.DataFrame) -> Dict:
        """Advanced statistics — skewness, kurtosis, normality, distribution fitting."""
        cls = _adv_stats.get("AdvancedStats")
        if cls is None:
            return {"error": "advanced_stats module not available — will be added in next update"}
        try:
            stats = cls(df)
            return stats.compute_all()
        except Exception as e:
            return {"error": str(e)}

    def _step_quality_score(self, df: pd.DataFrame, current_output: Dict) -> Dict:
        """Compute unified Data Quality Score."""
        try:
            scorer = DataQualityScorer()
            profile = scorer.score(df)
            return profile.to_dict()
        except Exception as e:
            return {"error": str(e)}

    def _step_split_advisor(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Dict:
        """Train/test split advisor."""
        fn = _split.get("advise_split")
        if fn is None:
            return {"error": "split_advisor module not available"}
        try:
            return fn(df, target_column=target_column)
        except Exception as e:
            return {"error": str(e)}

    def _step_imbalance(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Dict:
        """Class imbalance detection."""
        fn = _imbalance.get("detect_imbalance")
        if fn is None or not target_column:
            return {"skipped": True, "reason": "No target column or module unavailable"}
        try:
            return fn(df, target_column=target_column)
        except Exception as e:
            return {"error": str(e)}

    def _step_automl(
        self, df: pd.DataFrame, target_column: Optional[str], n_trials: int
    ) -> Dict:
        """Run advanced AutoML pipeline."""
        if not target_column:
            # Try to auto-detect
            fn = _target_det.get("detect_target")
            if fn:
                try:
                    det = fn(df)
                    target_column = det.get("target")
                except Exception:
                    pass

        if not target_column:
            return {"error": "No target column — cannot run AutoML"}

        cls = _automl.get("AdvancedAutoML")
        cfg_cls = _automl.get("AutoMLConfig")
        if cls is None:
            return {"error": "advanced_automl module not available"}
        try:
            config = cfg_cls(n_trials=n_trials, verbose=self.verbose) if cfg_cls else None
            aml = cls(config) if config else cls()
            results = aml.fit(df, target_column)
            self._log_decision(
                "automl",
                f"Best model: {results.get('best_model', {}).model_name if results.get('best_model') else 'N/A'}",
                f"n_trials={n_trials}, target={target_column}"
            )
            return results
        except Exception as e:
            return {"error": str(e)}

    def _step_cognitive_dna(self, df: pd.DataFrame) -> Dict:
        """Compute Cognitive Data DNA."""
        cls = _dna.get("CognitiveDNA")
        if cls is None:
            return {"error": "cognitive_dna module not available"}
        try:
            dna = cls(df)
            return dna.compute()
        except Exception as e:
            return {"error": str(e)}

    def _step_build_report(self, df: pd.DataFrame, current_output: Dict) -> Dict:
        """Build final structured report."""
        return {
            "total_modules_run": len([k for k, v in current_output.items()
                                      if isinstance(v, dict) and "error" not in v]),
            "modules_failed": len([k for k, v in current_output.items()
                                   if isinstance(v, dict) and "error" in v]),
            "session_id": self._session.session_id if self._session else None,
        }

    # ─────────────────────────────────────────
    #  Summary Builder
    # ─────────────────────────────────────────

    def _build_summary(
        self,
        original_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        output: Dict,
        source: str = "",
    ) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append(f"  Nydra v{self.VERSION} — Full Report")
        lines.append("=" * 65)
        if source:
            lines.append(f"  Source  : {source}")
        lines.append(f"  Original: {original_df.shape[0]} rows × {original_df.shape[1]} cols")
        lines.append(f"  Cleaned : {clean_df.shape[0]} rows × {clean_df.shape[1]} cols")
        lines.append(f"  Removed : {original_df.shape[0] - clean_df.shape[0]} rows")
        lines.append("")

        # Analysis
        analysis = output.get("analyze", {})
        if analysis and "missing_values" in analysis:
            missing_cols = {k: v for k, v in analysis["missing_values"].items() if v > 0}
            lines.append("── Data Quality ───────────────────────────────────────────")
            lines.append(f"  Missing columns  : {len(missing_cols)}")
            lines.append(f"  Duplicate rows   : {analysis.get('duplicate_rows', 0)}")

        # Outliers
        outlier_info = output.get("detect_outliers", {})
        if outlier_info and "n_outliers" in outlier_info:
            lines.append(f"  Outliers detected: {outlier_info['n_outliers']} "
                        f"({outlier_info.get('outlier_rate', 0):.1%}) "
                        f"— Severity: {outlier_info.get('severity', 'N/A')}")
        lines.append("")

        # ML Readiness
        ml_info = output.get("ml_readiness", {})
        if ml_info and "score" in ml_info:
            lines.append("── ML Readiness ───────────────────────────────────────────")
            lines.append(f"  Score: {ml_info['score']}/100")
        lines.append("")

        # AutoML
        automl_info = output.get("automl", {})
        if automl_info and "best_model" in automl_info and automl_info["best_model"]:
            best = automl_info["best_model"]
            lines.append("── AutoML Results ─────────────────────────────────────────")
            lines.append(f"  Best Model : {best.model_name}")
            lines.append(f"  CV Score   : {best.cv_mean:.4f} ± {best.cv_std:.4f}")
        lines.append("")

        # Steps
        lines.append("── Steps Executed ─────────────────────────────────────────")
        for step in output.get("step_log", []):
            status = "✅" if step["status"] == "done" else "❌"
            lines.append(f"  {status} {step['step']} ({step['duration_s']}s)")
        lines.append("")
        lines.append(f"  ⏱  Total time: {output.get('elapsed_s', 0)}s")
        lines.append("=" * 65)
        return "\n".join(lines)

    # ─────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────

    def _start_session(self, source: str, df: pd.DataFrame):
        import hashlib
        sid = hashlib.md5(f"{source}{time.time()}".encode()).hexdigest()[:10]
        self._session = AgentSession(
            session_id=sid,
            source=source,
            original_df=df.copy(),
            current_df=df.copy(),
        )
        self._decision_log.clear()

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def _log_decision(self, step: str, decision: str, reason: str):
        entry = DecisionLog(step=step, decision=decision, reason=reason)
        self._decision_log.append(entry)
        if self._session:
            self._session.decisions.append(entry)


# ─────────────────────────────────────────────
#  Convenience Functions
# ─────────────────────────────────────────────

def quick_inspect(filepath: str, verbose: bool = True) -> Dict[str, Any]:
    """One-line file inspection."""
    return Nydra(verbose=verbose).inspect(filepath)


def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """One-line DataFrame cleaning. Returns cleaned DataFrame."""
    result = Nydra(verbose=False).clean(df)
    return result.get("clean_df", df)


def quick_automl(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """One-line AutoML. Returns leaderboard + best model."""
    return Nydra().run_automl(df, target_column)


def full_analysis(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete Nydra pipeline in one call."""
    return Nydra(verbose=verbose).full_pipeline(df, target_column)
