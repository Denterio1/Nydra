"""
human_analyst.py - Safety-first analyst simulation for nydra.

This module provides a deterministic "human analyst" style layer that:
- reasons in explicit steps,
- highlights risk with confidence levels,
- proposes conservative actions by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AnalystFinding:
    severity: str
    title: str
    detail: str
    confidence: int


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _detect_potential_id_columns(df: pd.DataFrame) -> list[str]:
    suspects: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return suspects

    for col in df.columns:
        unique_ratio = _safe_div(df[col].nunique(dropna=False), n_rows)
        name = str(col).lower()
        if unique_ratio >= 0.98 and any(k in name for k in ("id", "uuid", "key", "token")):
            suspects.append(str(col))
    return suspects


def _detect_high_cardinality_text(df: pd.DataFrame) -> list[str]:
    risky: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return risky

    for col in df.columns:
        if df[col].dtype != object:
            continue
        nunique = df[col].nunique(dropna=True)
        ratio = _safe_div(float(nunique), float(n_rows))
        if ratio >= 0.4 and nunique > 30:
            risky.append(str(col))
    return risky


def _detect_possible_leakage(df: pd.DataFrame, id_cols: list[str]) -> list[str]:
    warnings: list[str] = []
    for col in id_cols:
        warnings.append(
            f"'{col}' looks like an identifier and can leak row identity into ML models."
        )

    # Heuristic: columns with names hinting post-outcome information.
    leak_terms = ("result", "outcome", "label", "target", "status_final", "approved")
    for col in df.columns:
        lower = str(col).lower()
        if any(term in lower for term in leak_terms):
            warnings.append(
                f"'{col}' may encode final outcomes. Validate if it is available at prediction time."
            )
    return warnings


def _action(severity: str, action: str, why: str) -> dict[str, str]:
    return {"severity": severity, "action": action, "why": why}


def run_human_analyst(
    data: dict[str, Any],
    analysis: dict[str, Any],
    outliers: dict[str, Any],
    ml: dict[str, Any] | None,
    rels: list[dict[str, Any]] | None,
    privacy_report: dict[str, Any] | None,
    risk_profile: str = "balanced",
) -> dict[str, Any]:
    """
    Produce a structured, explainable analyst report.

    risk_profile:
      - conservative: favors safer/no-regret actions.
      - balanced: practical default.
      - aggressive: accepts higher transformation risk.
    """
    df = data["df"]
    rows = int(analysis["shape"]["rows"])
    cols = int(analysis["shape"]["columns"])
    missing_total = int(sum(analysis["missing_values"].values()))
    dupes = int(analysis["duplicate_rows"])
    
    # Handle OutlierReport object or dict
    if hasattr(outliers, "method_results"):
        outlier_cols = len(set(r.column for r in outliers.method_results if r.column and r.outlier_count > 0))
    else:
        outlier_cols = len(outliers)
        
    rel_count = len(rels or [])
    ml_score = int(ml["score"]) if ml else None

    pii_risk = (privacy_report or {}).get("risk_level", "low")
    pii_cols = int((privacy_report or {}).get("affected_cols", 0))

    id_cols = _detect_potential_id_columns(df)
    high_card_text = _detect_high_cardinality_text(df)
    leakage_hints = _detect_possible_leakage(df, id_cols)

    findings: list[AnalystFinding] = []
    actions: list[dict[str, str]] = []
    reasoning_steps: list[str] = [
        "Step 1 - Profiled dataset shape, type mix, and base quality signals.",
        "Step 2 - Evaluated operational risks (privacy, leakage, unstable features).",
        "Step 3 - Ranked corrective actions by safety and expected impact.",
    ]

    if rows < 50:
        findings.append(AnalystFinding("high", "Very small dataset", f"Only {rows} rows detected; statistical conclusions are fragile.", 94))
        actions.append(_action("high", "Collect or merge more rows before training", "Small sample sizes inflate variance and overfitting risk."))
    elif rows < 300:
        findings.append(AnalystFinding("medium", "Limited sample size", f"{rows} rows is workable but weak for robust ML validation.", 86))

    if missing_total > 0:
        miss_pct = 100.0 * _safe_div(missing_total, max(rows * max(cols, 1), 1))
        sev = "high" if miss_pct > 15 else "medium"
        findings.append(AnalystFinding(sev, "Missing data detected", f"{missing_total} missing cells ({miss_pct:.2f}% of dataset).", 93))
        actions.append(_action(sev, "Use column-aware imputation with audit log", "Global fill strategies can distort distributions."))

    if dupes > 0:
        findings.append(AnalystFinding("medium", "Duplicate records present", f"{dupes} duplicate rows can bias metrics and leakage checks.", 96))
        actions.append(_action("medium", "Drop duplicates before train/test split", "Duplicate leakage inflates offline scores."))

    if outlier_cols > 0:
        sev = "high" if outlier_cols >= max(3, cols // 4) else "medium"
        findings.append(AnalystFinding(sev, "Outlier pressure", f"{outlier_cols} column(s) contain outliers.", 84))
        actions.append(_action(sev, "Winsorize or robust-scale only impacted columns", "Targeted handling is safer than blanket clipping."))

    if pii_risk in ("medium", "high"):
        sev = "high" if pii_risk == "high" else "medium"
        findings.append(AnalystFinding(sev, "Privacy risk detected", f"{pii_cols} potential PII column(s) flagged.", 89))
        actions.append(_action(sev, "Hash/mask PII columns before sharing/export", "Minimizes data exposure and compliance risk."))

    if id_cols:
        findings.append(AnalystFinding("high", "Potential identifier leakage", f"Likely ID columns: {', '.join(id_cols[:6])}.", 81))
        actions.append(_action("high", "Exclude identifier-like columns from modeling features", "Identifiers can leak identity and harm generalization."))

    if high_card_text:
        findings.append(AnalystFinding("medium", "High-cardinality text columns", f"{', '.join(high_card_text[:6])} may overfit with naive encoding.", 78))
        actions.append(_action("medium", "Use frequency or target-safe encoding with CV safeguards", "One-hot on high-cardinality fields can explode feature space."))

    if leakage_hints:
        findings.append(AnalystFinding("high", "Leakage candidates found", leakage_hints[0], 72))
        actions.append(_action("high", "Validate feature availability at prediction time", "Prevents impossible-in-production features."))

    if ml_score is not None:
        if ml_score < 60:
            findings.append(AnalystFinding("high", "Low ML readiness", f"Current ML readiness score is {ml_score}/100.", 90))
        elif ml_score < 80:
            findings.append(AnalystFinding("medium", "Moderate ML readiness", f"Current ML readiness score is {ml_score}/100.", 82))

    if rel_count > 0:
        findings.append(AnalystFinding("low", "Relationships discovered", f"{rel_count} strong column relationship(s) detected.", 75))

    if not findings:
        findings.append(AnalystFinding("low", "Healthy baseline", "No major quality or safety blockers were detected.", 70))

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (severity_rank.get(f.severity, 9), -f.confidence))

    if risk_profile == "conservative":
        preferred_mode = "Human-safe mode: prioritize explainability and minimal risky transforms."
    elif risk_profile == "aggressive":
        preferred_mode = "Fast-iteration mode: stronger transforms accepted for rapid modeling."
    else:
        preferred_mode = "Balanced mode: safety first, then performance optimization."

    high_count = sum(1 for f in findings if f.severity == "high")
    medium_count = sum(1 for f in findings if f.severity == "medium")
    overall = "high" if high_count >= 2 else "medium" if high_count == 1 or medium_count >= 2 else "low"

    executive = (
        f"Analyzed {rows:,} rows x {cols} columns. "
        f"Risk profile '{risk_profile}' selected. "
        f"Overall operational risk: {overall.upper()}."
    )

    return {
        "executive_summary": executive,
        "overall_risk": overall,
        "preferred_mode": preferred_mode,
        "reasoning_steps": reasoning_steps,
        "findings": [
            {
                "severity": f.severity,
                "title": f.title,
                "detail": f.detail,
                "confidence": f.confidence,
            }
            for f in findings
        ],
        "action_plan": actions[:10],
    }

