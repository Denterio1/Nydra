"""
pipeline_export.py — sklearn Pipeline Export.

Generates a complete, production-ready sklearn pipeline based on
dataset analysis. The pipeline handles:
    - Missing value imputation
    - Feature encoding (categorical)
    - Feature scaling (numeric)
    - Class imbalance handling (optional)
    - Model selection
    - Cross-validation

Output: ready-to-run Python script + joblib model saving.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any


def export_pipeline(
    data:         dict[str, Any],
    target_col:   str,
    task_type:    str  = "auto",
    model_name:   str  = "random_forest",
    handle_imbalance: bool = False,
    output_path:  str  = "pipeline.py",
) -> dict[str, Any]:
    """
    Generate a complete sklearn pipeline script.

    Args:
        data:             Structured data dict from loader.
        target_col:       Target column name.
        task_type:        "classification", "regression", or "auto".
        model_name:       "random_forest", "gradient_boosting", "logistic", "svm", "knn".
        handle_imbalance: Include SMOTE for imbalanced data.
        output_path:      Where to save the pipeline script.

    Returns:
        {
            "code":         str (complete pipeline Python script),
            "output_path":  str,
            "task_type":    str,
            "model":        str,
            "features":     list,
            "target":       str,
            "n_numeric":    int,
            "n_categorical":int,
        }
    """
    df = data["df"].copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── Detect task type ──────────────────────────────────────────────────────
    if task_type == "auto":
        n_unique  = y.nunique()
        task_type = "classification" if (y.dtype == object or n_unique <= 20) else "regression"

    # ── Categorise features ───────────────────────────────────────────────────
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if X[c].dtype == object]

    # ── Model selection ───────────────────────────────────────────────────────
    model_map = {
        "random_forest": {
            "classification": "RandomForestClassifier(n_estimators=100, random_state=42)",
            "regression":     "RandomForestRegressor(n_estimators=100, random_state=42)",
            "import_clf":     "from sklearn.ensemble import RandomForestClassifier",
            "import_reg":     "from sklearn.ensemble import RandomForestRegressor",
        },
        "gradient_boosting": {
            "classification": "GradientBoostingClassifier(n_estimators=100, random_state=42)",
            "regression":     "GradientBoostingRegressor(n_estimators=100, random_state=42)",
            "import_clf":     "from sklearn.ensemble import GradientBoostingClassifier",
            "import_reg":     "from sklearn.ensemble import GradientBoostingRegressor",
        },
        "logistic": {
            "classification": "LogisticRegression(max_iter=1000, random_state=42)",
            "regression":     "Ridge(alpha=1.0)",
            "import_clf":     "from sklearn.linear_model import LogisticRegression",
            "import_reg":     "from sklearn.linear_model import Ridge",
        },
        "svm": {
            "classification": "SVC(kernel='rbf', random_state=42, probability=True)",
            "regression":     "SVR(kernel='rbf')",
            "import_clf":     "from sklearn.svm import SVC",
            "import_reg":     "from sklearn.svm import SVR",
        },
        "knn": {
            "classification": "KNeighborsClassifier(n_neighbors=5)",
            "regression":     "KNeighborsRegressor(n_neighbors=5)",
            "import_clf":     "from sklearn.neighbors import KNeighborsClassifier",
            "import_reg":     "from sklearn.neighbors import KNeighborsRegressor",
        },
    }

    m        = model_map.get(model_name, model_map["random_forest"])
    model_str  = m[task_type]
    import_str = m["import_clf"] if task_type == "classification" else m["import_reg"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    if task_type == "classification":
        metric_import = "from sklearn.metrics import classification_report, accuracy_score, roc_auc_score"
        metric_code   = """
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")"""
    else:
        metric_import = "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error"
        metric_code   = """
    print(f"R²  Score : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE       : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE      : {mean_squared_error(y_test, y_pred, squared=False):.4f}")"""

    # ── Stratify ──────────────────────────────────────────────────────────────
    stratify_str = ", stratify=y_train" if task_type == "classification" else ""

    # ── SMOTE ────────────────────────────────────────────────────────────────
    smote_import = "from imblearn.pipeline import Pipeline  # replaces sklearn Pipeline" if handle_imbalance else ""
    smote_step   = "\n    ('smote', SMOTE(random_state=42))," if handle_imbalance else ""
    smote_import2= "from imblearn.over_sampling import SMOTE" if handle_imbalance else ""
    pipeline_import = "from imblearn.pipeline import Pipeline" if handle_imbalance else "from sklearn.pipeline import Pipeline"

    # ── CV ────────────────────────────────────────────────────────────────────
    cv_class = "StratifiedKFold" if task_type == "classification" else "KFold"
    cv_import = f"from sklearn.model_selection import {cv_class}, cross_val_score"

    # ── Generate code ─────────────────────────────────────────────────────────
    num_cols_repr = repr(num_cols)
    cat_cols_repr = repr(cat_cols)

    code = f'''"""
Auto-generated sklearn Pipeline by Nydra
Task     : {task_type.upper()}
Target   : {target_col}
Model    : {model_name.replace("_", " ").title()}
Features : {len(X.columns)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, {cv_class}, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
{pipeline_import}
{import_str}
{metric_import}
{"from imblearn.over_sampling import SMOTE" if handle_imbalance else ""}

# ── Load your data ────────────────────────────────────────────────────────────
df = pd.read_csv("your_data.csv")  # Change to your file

TARGET     = "{target_col}"
NUM_COLS   = {num_cols_repr}
CAT_COLS   = {cat_cols_repr}

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    {"stratify=y," if task_type == "classification" else ""}
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer",  SimpleImputer(strategy="most_frequent")),
    ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     NUM_COLS),
    ("cat", categorical_transformer, CAT_COLS),
])

# ── Full Pipeline ─────────────────────────────────────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),{smote_step}
    ("model",        {model_str}),
])

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training pipeline...")
pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("\\n── Test Set Results ──────────────────────────────────"){ metric_code}

# ── Cross-Validation ──────────────────────────────────────────────────────────
cv      = {cv_class}(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring={"accuracy" if task_type == "classification" else "r2"},
)
print(f"\\nCross-Validation: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

# ── Save Pipeline ─────────────────────────────────────────────────────────────
joblib.dump(pipeline, "nydra_pipeline.joblib")
print("\\n✓ Pipeline saved to nydra_pipeline.joblib")

# ── Load and Predict (example) ────────────────────────────────────────────────
# loaded_pipeline = joblib.load("nydra_pipeline.joblib")
# new_data = pd.DataFrame([dict(feature1=value1, feature2=value2)])
# prediction = loaded_pipeline.predict(new_data)
'''

    # ── Save to file ──────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    return {
        "code":          code,
        "output_path":   output_path,
        "task_type":     task_type,
        "model":         model_name,
        "features":      list(X.columns),
        "target":        target_col,
        "n_numeric":     len(num_cols),
        "n_categorical": len(cat_cols),
        "n_features":    len(X.columns),
    }
