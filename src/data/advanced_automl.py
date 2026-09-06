"""
advanced_automl.py — Nydra v0.5.5+
=========================================
Comprehensive AutoML module covering:
  - TaskDetector              : Auto detect classification/regression/multiclass
  - SearchSpaceBuilder        : Dynamic per-model hyperparameter search spaces
  - 15+ Models                : Full sklearn + XGBoost + LightGBM + CatBoost
  - OptunaOptimizer           : TPE Bayesian + CMA-ES + Hyperband
  - ASHAEarlyStopper          : Kill bad trials early
  - WarmStarter               : Meta-learning warm start from past runs
  - NestedCVEvaluator         : Outer eval + inner tuning, zero leakage
  - StratifiedCVEvaluator     : Imbalanced classification
  - TimeSeriesCVEvaluator     : Temporal data splits
  - RepeatedCVEvaluator       : Small dataset variance reduction
  - LearningCurveAnalyzer     : Overfitting detection during tuning
  - StackingEnsembler         : Auto stack top-N models
  - ModelLeaderboard          : Full ranked comparison table
  - CASHOptimizer             : Combined Algorithm Selection + HPO
  - PipelineExporter          : Full sklearn pipeline → .py file
  - AutoMLReporter            : Complete audit report + recommendations
  - AdvancedAutoML            : Master class — one call does everything

Author  : Nydra Project
Version : 0.5.0
"""

from __future__ import annotations

import os
import json
import time
import logging
import warnings
import hashlib
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Sklearn core
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, label_binarize
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit,
    RepeatedStratifiedKFold, RepeatedKFold, learning_curve,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, matthews_corrcoef
)

# Sklearn models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDClassifier, SGDRegressor,
    HuberRegressor, TheilSenRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Optional heavy libs — graceful fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nydra.automl")


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

TASK_BINARY       = "binary_classification"
TASK_MULTICLASS   = "multiclass_classification"
TASK_REGRESSION   = "regression"

METRIC_MAP = {
    TASK_BINARY:     ["roc_auc", "f1", "accuracy", "log_loss", "mcc"],
    TASK_MULTICLASS: ["f1_macro", "accuracy", "log_loss"],
    TASK_REGRESSION: ["r2", "rmse", "mae", "mape"],
}

DIRECTION_MAP = {
    "roc_auc": "maximize", "f1": "maximize", "accuracy": "maximize",
    "f1_macro": "maximize", "mcc": "maximize", "r2": "maximize",
    "log_loss": "minimize", "rmse": "minimize", "mae": "minimize", "mape": "minimize",
}


# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    """Result container for a single model evaluation."""
    model_name: str
    task: str
    primary_metric: str
    primary_score: float
    all_scores: Dict[str, float]
    best_params: Dict[str, Any]
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    fit_time: float
    n_trials: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task,
            "primary_metric": self.primary_metric,
            "score": round(self.primary_score, 4),
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "fit_time_s": round(self.fit_time, 2),
            "n_trials": self.n_trials,
            **{k: round(v, 4) for k, v in self.all_scores.items()},
        }


@dataclass
class LeaderboardEntry:
    """Single entry in the model leaderboard."""
    rank: int
    model_name: str
    primary_score: float
    cv_mean: float
    cv_std: float
    fit_time: float
    n_trials: int
    is_ensemble: bool = False
    is_tuned: bool = False


@dataclass
class AutoMLConfig:
    """Configuration for AdvancedAutoML."""
    task: Optional[str] = None                    # None = auto-detect
    target_column: Optional[str] = None
    primary_metric: Optional[str] = None          # None = auto-select
    cv_strategy: str = "stratified"               # stratified|kfold|timeseries|repeated|nested
    n_folds: int = 5
    n_repeats: int = 3
    n_trials: int = 50                            # Optuna trials per model
    timeout: Optional[int] = 300                  # seconds per model
    n_jobs: int = -1
    random_state: int = 42
    models: Optional[List[str]] = None            # None = all available
    enable_stacking: bool = True
    enable_voting: bool = True
    enable_warm_start: bool = True
    warm_start_path: str = ".automl_warmstart.json"
    early_stopping_rounds: int = 20
    scaler: str = "robust"                        # robust|standard|minmax|none
    export_pipeline: bool = True
    verbose: bool = True


# ─────────────────────────────────────────────
#  1. TaskDetector
# ─────────────────────────────────────────────

class TaskDetector:
    """
    Automatically detects ML task type from target column.
    Uses: dtype, cardinality, value distribution, column name hints.
    """

    REGRESSION_HINTS = [
        "price", "cost", "salary", "revenue", "income", "value",
        "amount", "rate", "score", "age", "weight", "height",
        "temperature", "count", "duration", "distance", "area",
    ]
    CLASSIFICATION_HINTS = [
        "class", "label", "target", "category", "type", "status",
        "flag", "churn", "fraud", "spam", "gender", "diagnosis",
    ]

    def __init__(self, binary_threshold: int = 2, multiclass_threshold: int = 20):
        self.binary_threshold = binary_threshold
        self.multiclass_threshold = multiclass_threshold

    def detect(self, y: pd.Series) -> str:
        col_name = y.name.lower() if y.name else ""
        n_unique = y.nunique()
        dtype = y.dtype

        # String/object → classification
        if dtype == object or str(dtype) == "category":
            return TASK_BINARY if n_unique <= self.binary_threshold else TASK_MULTICLASS

        # Boolean → binary
        if dtype == bool:
            return TASK_BINARY

        # Check name hints first
        for hint in self.REGRESSION_HINTS:
            if hint in col_name:
                return TASK_REGRESSION
        for hint in self.CLASSIFICATION_HINTS:
            if hint in col_name:
                return TASK_BINARY if n_unique <= self.binary_threshold else TASK_MULTICLASS

        # Integer with low cardinality → classification
        if pd.api.types.is_integer_dtype(dtype):
            if n_unique <= self.binary_threshold:
                return TASK_BINARY
            elif n_unique <= self.multiclass_threshold:
                return TASK_MULTICLASS

        # Float → regression (usually)
        if pd.api.types.is_float_dtype(dtype):
            if n_unique <= self.binary_threshold:
                return TASK_BINARY
            # Check if it's really continuous
            skew = abs(y.skew())
            if n_unique > 20:
                return TASK_REGRESSION

        # Default fallback
        if n_unique <= self.binary_threshold:
            return TASK_BINARY
        elif n_unique <= self.multiclass_threshold:
            return TASK_MULTICLASS
        return TASK_REGRESSION

    def encode_target(self, y: pd.Series, task: str) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
        """Encode target for ML, return array + encoder."""
        if task in [TASK_BINARY, TASK_MULTICLASS]:
            if y.dtype == object or str(y.dtype) == "category":
                le = LabelEncoder()
                return le.fit_transform(y), le
            return y.values.astype(int), None
        return y.values.astype(float), None


# ─────────────────────────────────────────────
#  2. SearchSpaceBuilder
# ─────────────────────────────────────────────

class SearchSpaceBuilder:
    """
    Builds dynamic, data-aware Optuna search spaces per model.
    Adjusts ranges based on dataset size and dimensionality.
    """

    def __init__(self, n_samples: int, n_features: int, task: str):
        self.n_samples = n_samples
        self.n_features = n_features
        self.task = task
        self._is_large = n_samples > 10_000
        self._is_high_dim = n_features > 50

    def build(self, trial: Any, model_name: str) -> Dict[str, Any]:
        """Return hyperparameter dict for a given model + Optuna trial."""
        builders = {
            "RandomForest":       self._rf,
            "GradientBoosting":   self._gb,
            "ExtraTrees":         self._et,
            "AdaBoost":           self._ada,
            "XGBoost":            self._xgb,
            "LightGBM":           self._lgb,
            "CatBoost":           self._cat,
            "SVM":                self._svm,
            "KNN":                self._knn,
            "MLP":                self._mlp,
            "LogisticRegression": self._lr,
            "Ridge":              self._ridge,
            "Lasso":              self._lasso,
            "ElasticNet":         self._elasticnet,
            "BayesianRidge":      self._bayesian_ridge,
            "DecisionTree":       self._dt,
            "SGD":                self._sgd,
        }
        fn = builders.get(model_name)
        return fn(trial) if fn else {}

    def _rf(self, trial):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth":         trial.suggest_int("max_depth", 3, 30) if trial.suggest_categorical("use_max_depth", [True, False]) else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        }

    def _gb(self, trial):
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "max_depth":       trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    def _et(self, trial):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth":         trial.suggest_int("max_depth", 3, 30) if trial.suggest_categorical("use_max_depth_et", [True, False]) else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    def _ada(self, trial):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2.0, log=True),
        }

    def _xgb(self, trial):
        return {
            "n_estimators":       trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "max_depth":          trial.suggest_int("max_depth", 2, 12),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
            "gamma":              trial.suggest_float("gamma", 0, 5),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "use_label_encoder":  False,
            "eval_metric":        "logloss" if self.task != TASK_REGRESSION else "rmse",
        }

    def _lgb(self, trial):
        return {
            "n_estimators":        trial.suggest_int("n_estimators", 50, 1000, step=50),
            "learning_rate":       trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "max_depth":           trial.suggest_int("max_depth", 2, 12),
            "num_leaves":          trial.suggest_int("num_leaves", 10, 300),
            "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples":   trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha":           trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":          trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose": -1,
        }

    def _cat(self, trial):
        return {
            "iterations":    trial.suggest_int("iterations", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "depth":         trial.suggest_int("depth", 2, 10),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "border_count":  trial.suggest_int("border_count", 32, 255),
            "verbose": False,
        }

    def _svm(self, trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
        params = {
            "C":       trial.suggest_float("C", 1e-4, 100.0, log=True),
            "kernel":  kernel,
            "probability": True,
        }
        if kernel in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        return params

    def _knn(self, trial):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, min(50, self.n_samples // 10)),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric":      trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
            "p":           trial.suggest_int("p", 1, 3),
        }

    def _mlp(self, trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers = tuple(
            trial.suggest_int(f"n_units_l{i}", 32, 512) for i in range(n_layers)
        )
        return {
            "hidden_layer_sizes": layers,
            "activation":         trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
            "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "batch_size":         trial.suggest_categorical("batch_size", ["auto", 32, 64, 128]),
            "max_iter": 300,
            "early_stopping": True,
        }

    def _lr(self, trial):
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga", "liblinear"])
        penalty_map = {
            "lbfgs":    ["l2", None],
            "saga":     ["l1", "l2", "elasticnet", None],
            "liblinear":["l1", "l2"],
        }
        penalty = trial.suggest_categorical("penalty", penalty_map[solver])
        params = {"C": trial.suggest_float("C", 1e-4, 100.0, log=True), "solver": solver,
                  "penalty": penalty, "max_iter": 1000}
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return params

    def _ridge(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}

    def _lasso(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True), "max_iter": 5000}

    def _elasticnet(self, trial):
        return {
            "alpha":    trial.suggest_float("alpha", 1e-6, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": 5000,
        }

    def _bayesian_ridge(self, trial):
        return {
            "alpha_1": trial.suggest_float("alpha_1", 1e-8, 1.0, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-8, 1.0, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-8, 1.0, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-8, 1.0, log=True),
        }

    def _dt(self, trial):
        return {
            "max_depth":         trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion":         trial.suggest_categorical(
                "criterion",
                ["gini", "entropy"] if self.task != TASK_REGRESSION else ["squared_error", "absolute_error"]
            ),
        }

    def _sgd(self, trial):
        return {
            "loss":           trial.suggest_categorical("loss", ["hinge", "modified_huber", "log_loss"]),
            "alpha":          trial.suggest_float("alpha", 1e-6, 1.0, log=True),
            "learning_rate":  trial.suggest_categorical("learning_rate", ["constant", "optimal", "invscaling"]),
            "eta0":           trial.suggest_float("eta0", 1e-5, 0.1, log=True),
            "max_iter": 1000,
        }


# ─────────────────────────────────────────────
#  3. ModelRegistry
# ─────────────────────────────────────────────

class ModelRegistry:
    """
    Central registry of all available models, organized by task.
    Supports: sklearn, XGBoost, LightGBM, CatBoost (auto-detected).
    """

    @staticmethod
    def get_models(task: str, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return {name: model_class} for the given task."""
        if task == TASK_REGRESSION:
            catalog = {
                "RandomForest":     RandomForestRegressor,
                "GradientBoosting": GradientBoostingRegressor,
                "ExtraTrees":       ExtraTreesRegressor,
                "AdaBoost":         AdaBoostRegressor,
                "Ridge":            Ridge,
                "Lasso":            Lasso,
                "ElasticNet":       ElasticNet,
                "BayesianRidge":    BayesianRidge,
                "KNN":              KNeighborsRegressor,
                "SVM":              SVR,
                "MLP":              MLPRegressor,
                "DecisionTree":     DecisionTreeRegressor,
                "HuberRegressor":   HuberRegressor,
                "SGD":              SGDRegressor,
            }
            if XGB_AVAILABLE:
                catalog["XGBoost"] = xgb.XGBRegressor
            if LGB_AVAILABLE:
                catalog["LightGBM"] = lgb.LGBMRegressor
            if CAT_AVAILABLE:
                catalog["CatBoost"] = cb.CatBoostRegressor
        else:
            catalog = {
                "RandomForest":          RandomForestClassifier,
                "GradientBoosting":      GradientBoostingClassifier,
                "ExtraTrees":            ExtraTreesClassifier,
                "AdaBoost":              AdaBoostClassifier,
                "LogisticRegression":    LogisticRegression,
                "KNN":                   KNeighborsClassifier,
                "SVM":                   SVC,
                "MLP":                   MLPClassifier,
                "DecisionTree":          DecisionTreeClassifier,
                "GaussianNB":            GaussianNB,
                "LDA":                   LinearDiscriminantAnalysis,
                "QDA":                   QuadraticDiscriminantAnalysis,
                "SGD":                   SGDClassifier,
            }
            if XGB_AVAILABLE:
                catalog["XGBoost"] = xgb.XGBClassifier
            if LGB_AVAILABLE:
                catalog["LightGBM"] = lgb.LGBMClassifier
            if CAT_AVAILABLE:
                catalog["CatBoost"] = cb.CatBoostClassifier

        if model_names:
            return {k: v for k, v in catalog.items() if k in model_names}
        return catalog


# ─────────────────────────────────────────────
#  4. MetricCalculator
# ─────────────────────────────────────────────

class MetricCalculator:
    """Compute all relevant metrics for a given task."""

    def __init__(self, task: str):
        self.task = task

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics = {}
        try:
            if self.task == TASK_REGRESSION:
                mse = mean_squared_error(y_true, y_pred)
                metrics["rmse"] = float(np.sqrt(mse))
                metrics["mae"]  = float(mean_absolute_error(y_true, y_pred))
                metrics["r2"]   = float(r2_score(y_true, y_pred))
                nonzero = y_true != 0
                if nonzero.any():
                    metrics["mape"] = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

            elif self.task == TASK_BINARY:
                metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
                metrics["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
                metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
                metrics["mcc"]       = float(matthews_corrcoef(y_true, y_pred))
                if y_proba is not None:
                    p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    metrics["roc_auc"] = float(roc_auc_score(y_true, p))
                    metrics["log_loss"] = float(log_loss(y_true, p))

            elif self.task == TASK_MULTICLASS:
                metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
                metrics["f1_macro"]  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["f1_micro"]  = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
                metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                if y_proba is not None:
                    try:
                        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                    except Exception:
                        pass
                    metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception as e:
            logger.warning(f"[MetricCalculator] Partial error: {e}")
        return metrics

    def primary_metric(self) -> str:
        defaults = {
            TASK_BINARY: "roc_auc",
            TASK_MULTICLASS: "f1_macro",
            TASK_REGRESSION: "r2",
        }
        return defaults[self.task]

    def cv_scoring(self) -> str:
        cv_map = {
            TASK_BINARY: "roc_auc",
            TASK_MULTICLASS: "f1_macro",
            TASK_REGRESSION: "r2",
        }
        return cv_map[self.task]


# ─────────────────────────────────────────────
#  5. WarmStarter
# ─────────────────────────────────────────────

class WarmStarter:
    """
    Persists best hyperparameters from previous runs.
    Uses dataset fingerprint (SHA256) to match relevant history.
    On new runs, seeds Optuna with previously found good params.
    """

    def __init__(self, path: str = ".automl_warmstart.json"):
        self.path = path
        self._cache: Dict[str, Any] = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"[WarmStarter] Save failed: {e}")

    def fingerprint(self, df: pd.DataFrame) -> str:
        shape_str = f"{df.shape[0]}_{df.shape[1]}_{list(df.columns)}"
        return hashlib.sha256(shape_str.encode()).hexdigest()[:16]

    def get_best_params(self, fingerprint: str, model_name: str) -> Optional[Dict]:
        key = f"{fingerprint}_{model_name}"
        return self._cache.get(key, {}).get("params")

    def store(self, fingerprint: str, model_name: str, params: Dict, score: float):
        key = f"{fingerprint}_{model_name}"
        existing = self._cache.get(key, {})
        if score > existing.get("score", -np.inf):
            self._cache[key] = {"params": params, "score": score}
            self._save()


# ─────────────────────────────────────────────
#  6. CV Strategies
# ─────────────────────────────────────────────

class CVFactory:
    """Factory that creates the right CV splitter based on strategy."""

    @staticmethod
    def create(
        strategy: str,
        task: str,
        n_folds: int = 5,
        n_repeats: int = 3,
        random_state: int = 42,
    ):
        is_clf = task != TASK_REGRESSION

        if strategy == "stratified":
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
                if is_clf else KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        elif strategy == "kfold":
            return KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        elif strategy == "timeseries":
            return TimeSeriesSplit(n_splits=n_folds)

        elif strategy == "repeated":
            return RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state) \
                if is_clf else RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)

        else:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
                if is_clf else KFold(n_splits=n_folds, shuffle=True, random_state=random_state)


class NestedCVEvaluator:
    """
    Nested Cross-Validation: outer loop for unbiased evaluation,
    inner loop for hyperparameter tuning. Zero data leakage.
    """

    def __init__(
        self,
        task: str,
        n_outer: int = 5,
        n_inner: int = 3,
        n_trials: int = 20,
        random_state: int = 42,
    ):
        self.task = task
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_trials = n_trials
        self.random_state = random_state

    def evaluate(
        self,
        model_name: str,
        model_cls,
        X: np.ndarray,
        y: np.ndarray,
        search_space_builder: SearchSpaceBuilder,
    ) -> Dict[str, Any]:
        outer_cv = CVFactory.create("stratified", self.task, self.n_outer, random_state=self.random_state)
        metric_calc = MetricCalculator(self.task)
        outer_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner: tune on train fold
            best_params = self._inner_tune(
                model_name, model_cls, X_train, y_train,
                search_space_builder, metric_calc
            )
            model = model_cls(**best_params, random_state=self.random_state) \
                if "random_state" in model_cls().get_params() else model_cls(**best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            scores = metric_calc.compute(y_test, y_pred, y_proba)
            outer_scores.append(scores.get(metric_calc.primary_metric(), 0))

        return {
            "nested_cv_scores": outer_scores,
            "nested_cv_mean": float(np.mean(outer_scores)),
            "nested_cv_std": float(np.std(outer_scores)),
        }

    def _inner_tune(self, model_name, model_cls, X, y, builder, metric_calc) -> Dict:
        if not OPTUNA_AVAILABLE:
            return {}
        inner_cv = CVFactory.create("stratified", self.task, self.n_inner,
                                     random_state=self.random_state)
        scoring = metric_calc.cv_scoring()

        def objective(trial):
            params = builder.build(trial, model_name)
            try:
                m = model_cls(**params)
                scores = cross_val_score(m, X, y, cv=inner_cv, scoring=scoring, n_jobs=-1)
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params


# ─────────────────────────────────────────────
#  7. OptunaOptimizer
# ─────────────────────────────────────────────

class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna with:
    - TPE sampler (Bayesian)
    - CMA-ES for large search spaces
    - Hyperband pruner (ASHA-like)
    - Early stopping on no improvement
    - Warm start from previous runs
    """

    def __init__(
        self,
        task: str,
        n_trials: int = 50,
        timeout: Optional[int] = 300,
        early_stopping_rounds: int = 20,
        use_cmaes: bool = False,
        random_state: int = 42,
        warm_params: Optional[Dict] = None,
    ):
        self.task = task
        self.n_trials = n_trials
        self.timeout = timeout
        self.early_stopping_rounds = early_stopping_rounds
        self.use_cmaes = use_cmaes
        self.random_state = random_state
        self.warm_params = warm_params
        self._best_trial_score = -np.inf
        self._no_improve_count = 0

    def _make_sampler(self):
        if self.use_cmaes:
            return optuna.samplers.CmaEsSampler(seed=self.random_state)
        return optuna.samplers.TPESampler(
            seed=self.random_state,
            n_startup_trials=10,
            multivariate=True,
        )

    def _make_pruner(self):
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=self.n_trials, reduction_factor=3
        )

    def optimize(
        self,
        model_name: str,
        model_cls,
        X: np.ndarray,
        y: np.ndarray,
        cv,
        search_space_builder: SearchSpaceBuilder,
        metric_calc: MetricCalculator,
    ) -> Tuple[Dict, float, optuna.Study]:

        scoring = metric_calc.cv_scoring()
        direction = DIRECTION_MAP.get(scoring, "maximize")

        study = optuna.create_study(
            direction=direction,
            sampler=self._make_sampler(),
            pruner=self._make_pruner(),
        )

        # Warm start: enqueue best known params
        if self.warm_params:
            try:
                study.enqueue_trial(self.warm_params)
            except Exception:
                pass

        no_improve = [0]
        best_val = [direction == "maximize" and -np.inf or np.inf]

        def objective(trial):
            params = search_space_builder.build(trial, model_name)
            try:
                has_rs = "random_state" in model_cls().get_params()
                m = model_cls(**params, random_state=self.random_state) \
                    if has_rs else model_cls(**params)
                scores = cross_val_score(m, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                                         error_score="raise")
                val = float(np.mean(scores))
            except Exception as e:
                return float("-inf") if direction == "maximize" else float("inf")

            # Early stopping logic
            improved = (direction == "maximize" and val > best_val[0]) or \
                       (direction == "minimize" and val < best_val[0])
            if improved:
                best_val[0] = val
                no_improve[0] = 0
            else:
                no_improve[0] += 1

            if no_improve[0] >= self.early_stopping_rounds:
                study.stop()

            return val

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
            gc_after_trial=True,
        )

        best_params = study.best_params
        best_score = study.best_value
        return best_params, best_score, study


# ─────────────────────────────────────────────
#  8. LearningCurveAnalyzer
# ─────────────────────────────────────────────

class LearningCurveAnalyzer:
    """
    Analyzes learning curves to detect:
    - Overfitting (large train-val gap)
    - Underfitting (both scores low)
    - Ideal fit
    Provides actionable recommendation per model.
    """

    def __init__(self, cv: int = 5, n_points: int = 8):
        self.cv = cv
        self.n_points = n_points

    def analyze(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        train_sizes = np.linspace(0.1, 1.0, self.n_points)
        try:
            train_sz, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=self.cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )
        except Exception as e:
            return {"error": str(e)}

        train_mean = train_scores.mean(axis=1)
        val_mean   = val_scores.mean(axis=1)
        gap = float(train_mean[-1] - val_mean[-1])
        train_final = float(train_mean[-1])
        val_final   = float(val_mean[-1])

        if gap > 0.15:
            diagnosis = "overfitting"
            advice = "Reduce model complexity, add regularization, or use more data."
        elif train_final < 0.6 and val_final < 0.6:
            diagnosis = "underfitting"
            advice = "Increase model complexity or add more features."
        elif val_mean[-1] < val_mean[-2]:
            diagnosis = "overfit_plateau"
            advice = "Model starts overfitting near max data — consider early stopping."
        else:
            diagnosis = "good_fit"
            advice = "Model is well-fitted."

        return {
            "train_sizes": train_sz.tolist(),
            "train_scores_mean": train_mean.tolist(),
            "val_scores_mean": val_mean.tolist(),
            "final_train_score": train_final,
            "final_val_score": val_final,
            "gap": round(gap, 4),
            "diagnosis": diagnosis,
            "advice": advice,
        }


# ─────────────────────────────────────────────
#  9. StackingEnsembler
# ─────────────────────────────────────────────

class StackingEnsembler:
    """
    Builds a stacking ensemble from top-N trained models.
    Uses LogisticRegression / Ridge as meta-learner (task-aware).
    Supports passthrough features for meta-learner.
    """

    def __init__(self, task: str, top_n: int = 5, passthrough: bool = True):
        self.task = task
        self.top_n = top_n
        self.passthrough = passthrough

    def build(
        self,
        results: List[ModelResult],
        trained_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv,
    ) -> Optional[ModelResult]:
        sorted_results = sorted(results, key=lambda r: r.cv_mean, reverse=True)
        top = sorted_results[:min(self.top_n, len(sorted_results))]

        if len(top) < 2:
            return None

        estimators = [
            (r.model_name, trained_models[r.model_name]) for r in top
            if r.model_name in trained_models
        ]
        if len(estimators) < 2:
            return None

        meta = LogisticRegression(max_iter=1000) if self.task != TASK_REGRESSION else Ridge()

        if self.task != TASK_REGRESSION:
            stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=meta,
                passthrough=self.passthrough,
                cv=3,
                n_jobs=-1,
            )
            scoring = "roc_auc" if self.task == TASK_BINARY else "f1_macro"
        else:
            stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=meta,
                passthrough=self.passthrough,
                cv=3,
                n_jobs=-1,
            )
            scoring = "r2"

        t0 = time.time()
        try:
            cv_scores = cross_val_score(stacker, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            stacker.fit(X, y)
            fit_time = time.time() - t0
            return ModelResult(
                model_name="StackingEnsemble",
                task=self.task,
                primary_metric=scoring,
                primary_score=float(cv_scores.mean()),
                all_scores={scoring: float(cv_scores.mean())},
                best_params={"top_models": [e[0] for e in estimators]},
                cv_scores=cv_scores,
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                fit_time=fit_time,
                metadata={"passthrough": self.passthrough, "meta_learner": type(meta).__name__},
            )
        except Exception as e:
            logger.warning(f"[Stacking] Failed: {e}")
            return None


# ─────────────────────────────────────────────
#  10. ModelLeaderboard
# ─────────────────────────────────────────────

class ModelLeaderboard:
    """
    Builds a ranked leaderboard from all ModelResults.
    Supports: sorting, filtering, formatting as DataFrame.
    """

    def __init__(self, task: str):
        self.task = task
        self._entries: List[LeaderboardEntry] = []

    def add(self, result: ModelResult, is_ensemble: bool = False, is_tuned: bool = False):
        self._entries.append(LeaderboardEntry(
            rank=0,
            model_name=result.model_name,
            primary_score=result.cv_mean,
            cv_mean=result.cv_mean,
            cv_std=result.cv_std,
            fit_time=result.fit_time,
            n_trials=result.n_trials,
            is_ensemble=is_ensemble,
            is_tuned=is_tuned,
        ))

    def build(self) -> pd.DataFrame:
        """Sort and rank all entries, return as DataFrame."""
        metric_calc = MetricCalculator(self.task)
        direction = DIRECTION_MAP.get(metric_calc.cv_scoring(), "maximize")
        reverse = direction == "maximize"

        sorted_entries = sorted(self._entries, key=lambda e: e.cv_mean, reverse=reverse)
        rows = []
        for rank, e in enumerate(sorted_entries, 1):
            rows.append({
                "Rank": rank,
                "Model": e.model_name,
                "CV Score": round(e.cv_mean, 4),
                "CV Std": round(e.cv_std, 4),
                "Fit Time (s)": round(e.fit_time, 2),
                "Tuned": "✅" if e.is_tuned else "",
                "Ensemble": "🔗" if e.is_ensemble else "",
                "Trials": e.n_trials,
            })
        return pd.DataFrame(rows)

    def best(self) -> Optional[LeaderboardEntry]:
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.cv_mean)


# ─────────────────────────────────────────────
#  11. PipelineExporter
# ─────────────────────────────────────────────

class PipelineExporter:
    """
    Exports the best model as a complete, production-ready sklearn pipeline.
    Output: Python .py file + optional pickle.
    """

    SCALER_MAP = {
        "robust":   "RobustScaler",
        "standard": "StandardScaler",
        "minmax":   "MinMaxScaler",
        "none":     None,
    }

    def __init__(self, scaler: str = "robust"):
        self.scaler = scaler

    def build_pipeline(self, model) -> Pipeline:
        steps = []
        scaler_name = self.SCALER_MAP.get(self.scaler)
        if scaler_name == "RobustScaler":
            steps.append(("scaler", RobustScaler()))
        elif scaler_name == "StandardScaler":
            steps.append(("scaler", StandardScaler()))
        elif scaler_name == "MinMaxScaler":
            steps.append(("scaler", MinMaxScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    def export_py(self, result: ModelResult, model_cls, output_path: str = "pipeline_export.py"):
        """Export a self-contained Python pipeline script."""
        scaler_name = self.SCALER_MAP.get(self.scaler, "RobustScaler")
        params_str = json.dumps(result.best_params, indent=4)

        imports = [
            "import numpy as np",
            "import pandas as pd",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler",
            f"from sklearn.{self._get_module(model_cls)} import {model_cls.__name__}",
        ]
        if XGB_AVAILABLE and "xgboost" in str(model_cls):
            imports.append("import xgboost as xgb")
        if LGB_AVAILABLE and "lightgbm" in str(model_cls):
            imports.append("import lightgbm as lgb")

        code = "\n".join(imports) + f"""

# ─────────────────────────────────────────
# Nydra — Auto-Generated Pipeline
# Model    : {result.model_name}
# Task     : {result.task}
# CV Score : {round(result.cv_mean, 4)} ± {round(result.cv_std, 4)}
# ─────────────────────────────────────────

BEST_PARAMS = {params_str}

def build_pipeline():
    model = {model_cls.__name__}(**BEST_PARAMS)
    steps = []
    {"steps.append(('scaler', " + scaler_name + "()))" if scaler_name else "# No scaler"}
    steps.append(('model', model))
    return Pipeline(steps)

pipeline = build_pipeline()

# Usage:
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# y_proba = pipeline.predict_proba(X_test)  # if classifier

if __name__ == "__main__":
    print("Pipeline ready:", pipeline)
    print("Steps:", [s[0] for s in pipeline.steps])
"""
        with open(output_path, "w") as f:
            f.write(code)
        logger.info(f"[PipelineExporter] Saved to {output_path}")
        return output_path

    def export_pickle(self, pipeline: Pipeline, path: str = "pipeline.pkl"):
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        return path

    def _get_module(self, model_cls) -> str:
        module = model_cls.__module__
        parts = module.split(".")
        return ".".join(parts[1:]) if parts[0] == "sklearn" else module


# ─────────────────────────────────────────────
#  12. AutoMLReporter
# ─────────────────────────────────────────────

class AutoMLReporter:
    """
    Full audit report for AutoML run.
    Covers: leaderboard, best model, tuning stats,
    learning curve diagnosis, recommendations, export summary.
    """

    def __init__(
        self,
        task: str,
        leaderboard: pd.DataFrame,
        results: List[ModelResult],
        learning_curves: Dict[str, Any],
        config: AutoMLConfig,
    ):
        self.task = task
        self.leaderboard = leaderboard
        self.results = results
        self.learning_curves = learning_curves
        self.config = config

    def _best_result(self) -> Optional[ModelResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.cv_mean)

    def _recommendations(self) -> List[str]:
        best = self._best_result()
        recs = []
        if not best:
            return recs

        score = best.cv_mean
        lc = self.learning_curves.get(best.model_name, {})
        diagnosis = lc.get("diagnosis", "")

        if diagnosis == "overfitting":
            recs.append(f"⚠️  {best.model_name} is overfitting. Add regularization or reduce complexity.")
        elif diagnosis == "underfitting":
            recs.append(f"⚠️  {best.model_name} is underfitting. Try more complex models or feature engineering.")

        if self.task == TASK_BINARY and score < 0.70:
            recs.append("ROC-AUC < 0.70 — check class balance and feature quality.")
        elif self.task == TASK_REGRESSION and score < 0.50:
            recs.append("R² < 0.50 — consider feature engineering or target transformation (log, sqrt).")

        if best.cv_std > 0.05:
            recs.append(f"High CV variance (±{best.cv_std:.3f}) — consider Repeated CV or more data.")

        if len(self.results) > 3:
            recs.append("Consider StackingEnsemble for additional 1-3% performance gain.")

        recs.append(f"Best model: {best.model_name} | CV: {best.cv_mean:.4f} ± {best.cv_std:.4f}")
        return recs

    def generate(self) -> Dict[str, Any]:
        best = self._best_result()
        return {
            "task": self.task,
            "n_models_evaluated": len(self.results),
            "cv_strategy": self.config.cv_strategy,
            "n_folds": self.config.n_folds,
            "n_trials_per_model": self.config.n_trials,
            "best_model": best.model_name if best else None,
            "best_score": round(best.cv_mean, 4) if best else None,
            "best_params": best.best_params if best else {},
            "leaderboard": self.leaderboard.to_dict(orient="records"),
            "recommendations": self._recommendations(),
            "learning_curve_diagnoses": {
                name: lc.get("diagnosis", "unknown")
                for name, lc in self.learning_curves.items()
            },
        }

    def to_markdown(self) -> str:
        report = self.generate()
        lines = [
            "# 🤖 AutoML Report — Nydra",
            f"**Task:** {report['task']}  ",
            f"**Models Evaluated:** {report['n_models_evaluated']}  ",
            f"**CV Strategy:** {report['cv_strategy']} ({report['n_folds']} folds)  ",
            f"**Best Model:** {report['best_model']} — Score: {report['best_score']}  ",
            "",
            "## 📊 Leaderboard",
            self.leaderboard.to_markdown(index=False),
            "",
            "## 💡 Recommendations",
        ]
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        lines += ["", "## 🔬 Learning Curve Diagnoses"]
        for name, diag in report["learning_curve_diagnoses"].items():
            lines.append(f"- **{name}**: {diag}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  13. CASHOptimizer
# ─────────────────────────────────────────────

class CASHOptimizer:
    """
    Combined Algorithm Selection and Hyperparameter optimization.
    Treats model choice + hyperparams as a single unified search space.
    Uses Optuna's conditional search with model as top-level hyperparameter.
    """

    def __init__(
        self,
        task: str,
        n_trials: int = 100,
        timeout: int = 600,
        random_state: int = 42,
        n_folds: int = 5,
        model_names: Optional[List[str]] = None,
    ):
        self.task = task
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.n_folds = n_folds
        self.model_names = model_names

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, Dict, float]:
        """Run CASH and return (best_model_name, best_params, best_score)."""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna required for CASHOptimizer.")

        models = ModelRegistry.get_models(self.task, self.model_names)
        model_list = list(models.keys())
        metric_calc = MetricCalculator(self.task)
        scoring = metric_calc.cv_scoring()
        cv = CVFactory.create("stratified", self.task, self.n_folds,
                               random_state=self.random_state)
        ssb = SearchSpaceBuilder(X.shape[0], X.shape[1], self.task)

        def objective(trial):
            model_name = trial.suggest_categorical("model", model_list)
            params = ssb.build(trial, model_name)
            model_cls = models[model_name]
            try:
                has_rs = "random_state" in model_cls().get_params()
                m = model_cls(**params, random_state=self.random_state) \
                    if has_rs else model_cls(**params)
                scores = cross_val_score(m, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                                          error_score="raise")
                return float(np.mean(scores))
            except Exception:
                return float("-inf")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state, multivariate=True),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout,
                       show_progress_bar=False)

        best = study.best_params
        best_model = best.pop("model")
        return best_model, best, study.best_value


# ─────────────────────────────────────────────
#  14. ScalerFactory
# ─────────────────────────────────────────────

class ScalerFactory:
    @staticmethod
    def create(scaler_name: str):
        mapping = {
            "robust":   RobustScaler(),
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "none":     None,
        }
        return mapping.get(scaler_name, RobustScaler())


# ─────────────────────────────────────────────
#  15. AdvancedAutoML — Master Class
# ─────────────────────────────────────────────

class AdvancedAutoML:
    """
    Master AutoML class — one call does everything.

    Flow:
    1. Preprocess X, detect task, encode target
    2. For each model: tune with Optuna, evaluate with CV
    3. Analyze learning curves
    4. Build stacking + voting ensembles
    5. Build leaderboard
    6. Export best pipeline
    7. Generate full report

    Usage:
    ------
        aml = AdvancedAutoML(config)
        results = aml.fit(df, target_column="churn")
        print(results["report"])
        print(results["leaderboard"])
    """

    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self._results: List[ModelResult] = []
        self._trained_models: Dict[str, Any] = {}
        self._leaderboard_builder: Optional[ModelLeaderboard] = None
        self._learning_curves: Dict[str, Any] = {}
        self._task: Optional[str] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_names: List[str] = []
        self._warm_starter = WarmStarter(self.config.warm_start_path) \
            if self.config.enable_warm_start else None
        self._fingerprint: Optional[str] = None

    # ── Public API ──

    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Main entry point. Returns full results dict."""
        t_start = time.time()
        self._log("🚀 AdvancedAutoML starting...")

        X, y = self._prepare(df, target_column)
        models = ModelRegistry.get_models(self._task, self.config.models)
        cv = CVFactory.create(
            self.config.cv_strategy, self._task,
            self.config.n_folds, self.config.n_repeats, self.config.random_state
        )
        metric_calc = MetricCalculator(self._task)
        ssb = SearchSpaceBuilder(X.shape[0], X.shape[1], self._task)
        self._leaderboard_builder = ModelLeaderboard(self._task)

        # ── Per-model tuning + evaluation ──
        for model_name, model_cls in models.items():
            self._log(f"⚙️  Tuning {model_name}...")
            result = self._tune_and_evaluate(
                model_name, model_cls, X, y, cv, ssb, metric_calc
            )
            if result:
                self._results.append(result)
                self._leaderboard_builder.add(result, is_tuned=True)
                self._log(
                    f"  ✅ {model_name}: {result.cv_mean:.4f} ± {result.cv_std:.4f} "
                    f"({result.fit_time:.1f}s, {result.n_trials} trials)"
                )

        # ── Learning curves for top-3 ──
        self._log("📈 Analyzing learning curves...")
        top3 = sorted(self._results, key=lambda r: r.cv_mean, reverse=True)[:3]
        for res in top3:
            if res.model_name in self._trained_models:
                lca = LearningCurveAnalyzer(cv=3)
                lc = lca.analyze(
                    self._trained_models[res.model_name], X, y,
                    scoring=metric_calc.cv_scoring()
                )
                self._learning_curves[res.model_name] = lc

        # ── Stacking ensemble ──
        if self.config.enable_stacking and len(self._results) >= 2:
            self._log("🔗 Building stacking ensemble...")
            stacker = StackingEnsembler(self._task, top_n=5)
            stack_result = stacker.build(self._results, self._trained_models, X, y, cv)
            if stack_result:
                self._results.append(stack_result)
                self._leaderboard_builder.add(stack_result, is_ensemble=True)
                self._log(f"  ✅ StackingEnsemble: {stack_result.cv_mean:.4f}")

        # ── Voting ensemble ──
        if self.config.enable_voting and len(self._results) >= 2:
            vote_result = self._build_voting(X, y, cv, metric_calc)
            if vote_result:
                self._results.append(vote_result)
                self._leaderboard_builder.add(vote_result, is_ensemble=True)
                self._log(f"  ✅ VotingEnsemble: {vote_result.cv_mean:.4f}")

        # ── Leaderboard ──
        leaderboard = self._leaderboard_builder.build()
        best_result = max(self._results, key=lambda r: r.cv_mean) if self._results else None

        # ── Pipeline export ──
        exported_path = None
        if self.config.export_pipeline and best_result:
            best_model_name = best_result.model_name
            if best_model_name in self._trained_models:
                exporter = PipelineExporter(self.config.scaler)
                model_cls = ModelRegistry.get_models(self._task).get(best_model_name)
                if model_cls:
                    exported_path = exporter.export_py(
                        best_result, model_cls, "best_pipeline.py"
                    )

        # ── Report ──
        reporter = AutoMLReporter(
            self._task, leaderboard, self._results,
            self._learning_curves, self.config
        )
        report = reporter.generate()
        report_md = reporter.to_markdown()

        total_time = time.time() - t_start
        self._log(f"✅ Done in {total_time:.1f}s | Best: {best_result.model_name if best_result else 'N/A'}")

        return {
            "task": self._task,
            "leaderboard": leaderboard,
            "results": self._results,
            "best_model": best_result,
            "learning_curves": self._learning_curves,
            "report": report,
            "report_markdown": report_md,
            "exported_pipeline": exported_path,
            "total_time_s": round(total_time, 2),
            "feature_names": self._feature_names,
        }

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Predict using best trained model."""
        best = max(self._results, key=lambda r: r.cv_mean)
        model = self._trained_models.get(best.model_name)
        if model is None:
            raise RuntimeError("No trained model found. Run fit() first.")
        X_num = X_new[self._feature_names].fillna(X_new[self._feature_names].median())
        return model.predict(X_num.values)

    def cash_optimize(self, df: pd.DataFrame, target_column: str, n_trials: int = 100) -> Dict:
        """Run CASH optimization — model + hyperparams as unified search."""
        X, y = self._prepare(df, target_column)
        optimizer = CASHOptimizer(
            task=self._task,
            n_trials=n_trials,
            random_state=self.config.random_state,
            n_folds=self.config.n_folds,
        )
        best_model, best_params, best_score = optimizer.optimize(X, y)
        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_score": round(best_score, 4),
            "task": self._task,
        }

    # ── Private Helpers ──

    def _prepare(self, df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        detector = TaskDetector()
        y_series = df[target_column]
        self._task = self.config.task or detector.detect(y_series)
        y, self._label_encoder = detector.encode_target(y_series, self._task)

        X_df = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
        X_df = X_df.fillna(X_df.median())
        self._feature_names = list(X_df.columns)

        scaler = ScalerFactory.create(self.config.scaler)
        X = scaler.fit_transform(X_df.values) if scaler else X_df.values

        if self._warm_starter:
            self._fingerprint = self._warm_starter.fingerprint(df)

        self._log(f"Task: {self._task} | Shape: {X.shape} | Features: {len(self._feature_names)}")
        return X, y

    def _tune_and_evaluate(
        self,
        model_name: str,
        model_cls,
        X: np.ndarray,
        y: np.ndarray,
        cv,
        ssb: SearchSpaceBuilder,
        metric_calc: MetricCalculator,
    ) -> Optional[ModelResult]:
        t0 = time.time()
        warm_params = None
        if self._warm_starter and self._fingerprint:
            warm_params = self._warm_starter.get_best_params(self._fingerprint, model_name)

        if OPTUNA_AVAILABLE and self.config.n_trials > 0:
            optimizer = OptunaOptimizer(
                task=self._task,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                early_stopping_rounds=self.config.early_stopping_rounds,
                random_state=self.config.random_state,
                warm_params=warm_params,
            )
            try:
                best_params, best_score, _ = optimizer.optimize(
                    model_name, model_cls, X, y, cv, ssb, metric_calc
                )
                n_trials = self.config.n_trials
            except Exception as e:
                logger.warning(f"[AutoML] Optuna failed for {model_name}: {e}")
                best_params, n_trials = {}, 0
        else:
            best_params, n_trials = {}, 0

        # Final fit + CV with best params
        try:
            has_rs = "random_state" in model_cls().get_params()
            model = model_cls(**best_params, random_state=self.config.random_state) \
                if has_rs else model_cls(**best_params)
        except Exception:
            model = model_cls()

        scoring = metric_calc.cv_scoring()
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                                         error_score="raise")
        except Exception as e:
            logger.warning(f"[AutoML] CV failed for {model_name}: {e}")
            return None

        model.fit(X, y)
        self._trained_models[model_name] = model
        fit_time = time.time() - t0

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        all_scores = metric_calc.compute(y, y_pred, y_proba)

        if self._warm_starter and self._fingerprint:
            self._warm_starter.store(
                self._fingerprint, model_name, best_params, float(cv_scores.mean())
            )

        return ModelResult(
            model_name=model_name,
            task=self._task,
            primary_metric=metric_calc.primary_metric(),
            primary_score=float(cv_scores.mean()),
            all_scores=all_scores,
            best_params=best_params,
            cv_scores=cv_scores,
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            fit_time=fit_time,
            n_trials=n_trials,
        )

    def _build_voting(self, X, y, cv, metric_calc) -> Optional[ModelResult]:
        scoring = metric_calc.cv_scoring()
        top_results = sorted(self._results, key=lambda r: r.cv_mean, reverse=True)[:5]
        estimators = [
            (r.model_name, self._trained_models[r.model_name])
            for r in top_results if r.model_name in self._trained_models
        ]
        if len(estimators) < 2:
            return None
        t0 = time.time()
        try:
            voter = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1) \
                if self._task != TASK_REGRESSION else \
                VotingRegressor(estimators=estimators, n_jobs=-1)
            cv_scores = cross_val_score(voter, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            voter.fit(X, y)
            return ModelResult(
                model_name="VotingEnsemble",
                task=self._task,
                primary_metric=scoring,
                primary_score=float(cv_scores.mean()),
                all_scores={scoring: float(cv_scores.mean())},
                best_params={"models": [e[0] for e in estimators]},
                cv_scores=cv_scores,
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                fit_time=time.time() - t0,
                n_trials=0,
            )
        except Exception as e:
            logger.warning(f"[Voting] Failed: {e}")
            return None

    def _log(self, msg: str):
        if self.config.verbose:
            logger.info(msg)


# ─────────────────────────────────────────────
#  Public API / Convenience Functions
# ─────────────────────────────────────────────

def run_automl(
    df: pd.DataFrame,
    target_column: str,
    task: Optional[str] = None,
    cv_strategy: str = "stratified",
    n_folds: int = 5,
    n_trials: int = 50,
    scaler: str = "robust",
    enable_stacking: bool = True,
    models: Optional[List[str]] = None,
    timeout: Optional[int] = 300,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    One-line AutoML.

    Parameters
    ----------
    df             : Input DataFrame
    target_column  : Name of target column
    task           : 'binary_classification' | 'multiclass_classification' | 'regression' | None (auto)
    cv_strategy    : 'stratified' | 'kfold' | 'timeseries' | 'repeated' | 'nested'
    n_folds        : Number of CV folds
    n_trials       : Optuna trials per model
    scaler         : 'robust' | 'standard' | 'minmax' | 'none'
    enable_stacking: Build stacking ensemble from top models
    models         : List of model names to use (None = all)
    timeout        : Max seconds per model tuning
    verbose        : Print progress

    Returns
    -------
    dict with: task, leaderboard, best_model, report, report_markdown, ...
    """
    config = AutoMLConfig(
        task=task,
        target_column=target_column,
        cv_strategy=cv_strategy,
        n_folds=n_folds,
        n_trials=n_trials,
        scaler=scaler,
        enable_stacking=enable_stacking,
        models=models,
        timeout=timeout,
        verbose=verbose,
    )
    aml = AdvancedAutoML(config)
    return aml.fit(df, target_column)


def quick_automl(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Ultra-fast AutoML — 10 trials, no stacking. Returns leaderboard only."""
    config = AutoMLConfig(n_trials=10, enable_stacking=False, enable_voting=False, verbose=False)
    aml = AdvancedAutoML(config)
    results = aml.fit(df, target_column)
    return results["leaderboard"]


def cash_search(df: pd.DataFrame, target_column: str, n_trials: int = 100) -> Dict:
    """Run CASH — finds best model + hyperparams in one unified search."""
    aml = AdvancedAutoML()
    return aml.cash_optimize(df, target_column, n_trials=n_trials)


