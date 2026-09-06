"""
Auto-generated sklearn Pipeline by Nydra
Task     : CLASSIFICATION
Target   : category
Model    : Knn
Features : 7 (4 numeric, 0 categorical)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# ── Load your data ────────────────────────────────────────────────────────────
df = pd.read_csv("your_data.csv")  # Change to your file

TARGET     = "category"
NUM_COLS   = ['order_id', 'quantity', 'unit_price', 'total']
CAT_COLS   = []

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
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
    ("preprocessor", preprocessor),
    ("model",        KNeighborsClassifier(n_neighbors=5)),
])

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training pipeline...")
pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("\n── Test Set Results ──────────────────────────────────")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ── Cross-Validation ──────────────────────────────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipeline, X, y,
    cv='cv',
    scoring='accuracy',
)
print(f"\nCross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Save Pipeline ─────────────────────────────────────────────────────────────
joblib.dump(pipeline, "nydra_pipeline.joblib")
print("\n✓ Pipeline saved to nydra_pipeline.joblib")

# ── Load and Predict (example) ────────────────────────────────────────────────
# loaded_pipeline = joblib.load("nydra_pipeline.joblib")
# new_data = pd.DataFrame([dict(feature1=value1, feature2=value2)])
# prediction = loaded_pipeline.predict(new_data)

