"""
image_report.py — Nydra v0.6.0
=====================================
Generates professional multi-format reports for the image analysis pipeline.
Combines output from: image_loader → image_quality → image_analyzer → image_cleaner

Exports: HTML (interactive Plotly) | JSON (machine-readable) | Markdown (GitHub/Notion)

"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Plotly is optional — used for HTML charts only
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "0.6.0"

SEVERITY_ORDER  = ["critical", "high", "medium", "low", "info"]
SEVERITY_EMOJI  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "ℹ️"}
SEVERITY_COLOR  = {"critical": "#e74c3c", "high": "#e67e22",
                   "medium": "#f1c40f", "low": "#2ecc71", "info": "#3498db"}

GRADE_THRESHOLDS = [(90, "A", "Excellent 🟢"), (75, "B", "Good 🟡"),
                    (60, "C", "Fair 🟠"),      (45, "D", "Poor 🔴"),
                    (0,  "F", "Critical ⛔")]

REQUIRED_COLS = {
    "file_path"     : "str",
    "quality_score" : "float",
}

METRIC_COLS = [
    "brightness_score", "contrast_score",
    "blur_score", "noise_score", "artifact_score",
]

ISSUE_COLS = [
    "is_duplicate", "is_corrupted", "is_low_res",
    "is_low_quality", "issue_severity",
]

CLEAN_COLS = [
    "cleaned", "fix_applied", "score_before", "score_after",
]


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — REPORT BUILDER
# Purpose : Collect, merge and validate all pipeline DataFrames.
#           Produces one unified "master_df" plus dataset-level summary dict.
# ─────────────────────────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Merges outputs from all image pipeline stages into a single master DataFrame.

    Usage
    -----
    builder = ReportBuilder(loader_df, quality_df, analyzer_df, cleaner_df)
    master  = builder.master_df
    summary = builder.summary
    """

    def __init__(
        self,
        loader_df   : pd.DataFrame,
        quality_df  : pd.DataFrame | None = None,
        analyzer_df : pd.DataFrame | None = None,
        cleaner_df  : pd.DataFrame | None = None,
    ) -> None:
        self.loader_df   = loader_df.copy()
        self.quality_df  = quality_df.copy()  if quality_df  is not None else None
        self.analyzer_df = analyzer_df.copy() if analyzer_df is not None else None
        self.cleaner_df  = cleaner_df.copy()  if cleaner_df  is not None else None

        self.master_df : pd.DataFrame = pd.DataFrame()
        self.summary   : dict[str, Any] = {}

        self._validate_inputs()
        self._merge()
        self._compute_summary()

    # ── validation ──────────────────────────────────────────────────────────

    def _validate_inputs(self) -> None:
        """Make sure loader_df has the minimum required columns."""
        if "file_path" not in self.loader_df.columns:
            raise ValueError(
                "loader_df must contain a 'file_path' column. "
                "Make sure you pass the output of image_loader."
            )

    # ── merge ────────────────────────────────────────────────────────────────

    def _merge(self) -> None:
        """Left-join all available DataFrames on 'file_path'."""
        df = self.loader_df.copy()

        for stage_df in (self.quality_df, self.analyzer_df, self.cleaner_df):
            if stage_df is not None and "file_path" in stage_df.columns:
                # Avoid duplicate columns — keep only new ones from stage_df
                new_cols = [c for c in stage_df.columns
                            if c not in df.columns or c == "file_path"]
                df = df.merge(stage_df[new_cols], on="file_path", how="left")

        # Ensure quality_score exists (default 0.0 if quality_df not provided)
        if "quality_score" not in df.columns:
            df["quality_score"] = 0.0

        self.master_df = df

    # ── summary ───────────────────────────────────────────────────────────────

    def _compute_summary(self) -> None:
        """Compute dataset-level aggregate statistics."""
        df = self.master_df
        n  = len(df)

        scores = df["quality_score"].dropna()
        avg    = float(scores.mean()) if len(scores) else 0.0

        # Grade distribution
        grade_counts: dict[str, int] = {}
        for thresh, grade, _ in GRADE_THRESHOLDS:
            mask = scores >= thresh
            grade_counts[grade] = int(mask.sum())
            scores = scores[~mask]          # remove already-bucketed rows

        # Issue counts (columns may not exist → default 0)
        def _count(col: str) -> int:
            return int(df[col].sum()) if col in df.columns else 0

        duplicates  = _count("is_duplicate")
        corrupted   = _count("is_corrupted")
        low_res     = _count("is_low_res")
        low_quality = _count("is_low_quality")

        top_issues: list[dict] = []
        for issue, count in [
            ("Duplicates",   duplicates),
            ("Corrupted",    corrupted),
            ("Low-Res",      low_res),
            ("Low-Quality",  low_quality),
        ]:
            if count > 0:
                top_issues.append({"issue": issue, "count": count,
                                   "pct": round(count / n * 100, 1) if n else 0})
        top_issues.sort(key=lambda x: x["count"], reverse=True)

        # Most common fix
        most_common_fix = "N/A"
        if "fix_applied" in df.columns:
            fix_series = df["fix_applied"].dropna()
            if len(fix_series):
                most_common_fix = fix_series.value_counts().index[0]

        self.summary = {
            "generated_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nydra_ver"   : VERSION,
            "total_images"     : n,
            "avg_quality_score": round(avg, 2),
            "dataset_grade"    : _score_to_grade(avg)[0],
            "grade_counts"     : grade_counts,
            "top_issues"       : top_issues,
            "total_issues"     : duplicates + corrupted + low_res + low_quality,
            "most_common_fix"  : most_common_fix,
            "cleaning_ran"     : self.cleaner_df is not None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — DATASET OVERVIEW SECTION
# Purpose : Top-level summary — counts, formats, resolution stats,
#           class distribution (subfolders), before/after comparison.
# ─────────────────────────────────────────────────────────────────────────────

class DatasetOverviewSection:
    """
    Generates the Dataset Overview portion of the report.

    What it covers
    ──────────────
    • Total image count + format breakdown
    • Resolution statistics (min / max / median W × H)
    • Class distribution when images are in labelled sub-folders
    • Before vs After quality score comparison (if cleaning ran)
    """

    def __init__(self, master_df: pd.DataFrame, summary: dict) -> None:
        self.df      = master_df
        self.summary = summary

    # ── public API ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "format_breakdown"  : self._format_breakdown(),
            "resolution_stats"  : self._resolution_stats(),
            "class_distribution": self._class_distribution(),
            "before_after"      : self._before_after(),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _format_breakdown(self) -> dict[str, int]:
        if "format" not in self.df.columns:
            # Try to infer from file_path extension
            if "file_path" in self.df.columns:
                exts = (
                    self.df["file_path"]
                    .apply(lambda p: Path(str(p)).suffix.lower().lstrip("."))
                    .replace("", "unknown")
                )
                return exts.value_counts().to_dict()
            return {}
        return self.df["format"].value_counts().to_dict()

    def _resolution_stats(self) -> dict:
        stats: dict[str, Any] = {}
        for dim in ("width", "height"):
            if dim in self.df.columns:
                col = self.df[dim].dropna()
                stats[dim] = {
                    "min"   : int(col.min()),
                    "max"   : int(col.max()),
                    "median": float(col.median()),
                    "mean"  : round(float(col.mean()), 1),
                }
        if "width" in self.df.columns and "height" in self.df.columns:
            mp = (self.df["width"] * self.df["height"]).dropna() / 1_000_000
            stats["megapixels"] = {
                "min" : round(float(mp.min()), 2),
                "max" : round(float(mp.max()), 2),
                "mean": round(float(mp.mean()), 2),
            }
        return stats

    def _class_distribution(self) -> dict[str, int]:
        """Detect class labels from 'label' column or parent folder name."""
        if "label" in self.df.columns:
            return self.df["label"].value_counts().to_dict()
        if "file_path" in self.df.columns:
            parents = self.df["file_path"].apply(
                lambda p: Path(str(p)).parent.name
            )
            unique = parents.unique()
            # Only treat as class labels if there are 2–50 distinct parent dirs
            if 2 <= len(unique) <= 50:
                return parents.value_counts().to_dict()
        return {}

    def _before_after(self) -> dict:
        if not self.summary.get("cleaning_ran"):
            return {}
        result: dict[str, Any] = {}
        if "score_before" in self.df.columns and "score_after" in self.df.columns:
            before = self.df["score_before"].dropna()
            after  = self.df["score_after"].dropna()
            result = {
                "avg_before": round(float(before.mean()), 2),
                "avg_after" : round(float(after.mean()), 2),
                "delta"     : round(float(after.mean()) - float(before.mean()), 2),
                "improved"  : int((after > before).sum()),
                "unchanged" : int((after == before).sum()),
                "worsened"  : int((after < before).sum()),
            }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — QUALITY ANALYSIS SECTION
# Purpose : Score histogram, per-metric breakdown, top-10 worst / best images.
# ─────────────────────────────────────────────────────────────────────────────

class QualityAnalysisSection:
    """
    Analyses and summarises per-image quality scores.

    Metrics covered  : brightness, contrast, blur, noise, artifacts
    Tables produced  : top-10 worst images | top-10 best images
    """

    def __init__(self, master_df: pd.DataFrame) -> None:
        self.df = master_df

    def to_dict(self) -> dict:
        return {
            "score_histogram"   : self._score_histogram(),
            "metric_breakdown"  : self._metric_breakdown(),
            "top_10_worst"      : self._top_n("worst"),
            "top_10_best"       : self._top_n("best"),
            "grade_distribution": self._grade_distribution(),
        }

    def _score_histogram(self) -> dict:
        """Returns bin edges + counts for a 10-bucket histogram."""
        scores = self.df["quality_score"].dropna().values
        if len(scores) == 0:
            return {}
        counts, edges = np.histogram(scores, bins=10, range=(0, 100))
        return {
            "bin_edges": [round(float(e), 1) for e in edges],
            "counts"   : [int(c) for c in counts],
        }

    def _metric_breakdown(self) -> dict[str, dict]:
        """Average score per quality metric."""
        result: dict[str, dict] = {}
        for col in METRIC_COLS:
            if col in self.df.columns:
                series = self.df[col].dropna()
                metric = col.replace("_score", "").replace("_", " ").title()
                result[metric] = {
                    "mean"  : round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2),
                    "min"   : round(float(series.min()), 2),
                    "max"   : round(float(series.max()), 2),
                }
        return result

    def _top_n(self, direction: str, n: int = 10) -> list[dict]:
        """Return top-n worst or best images as a list of dicts."""
        df = self.df[["file_path", "quality_score"]].dropna(subset=["quality_score"])
        ascending = direction == "worst"
        top = df.sort_values("quality_score", ascending=ascending).head(n)
        records: list[dict] = []
        for _, row in top.iterrows():
            score = float(row["quality_score"])
            records.append({
                "file"        : Path(str(row["file_path"])).name,
                "score"       : round(score, 2),
                "grade"       : _score_to_grade(score)[0],
            })
        return records

    def _grade_distribution(self) -> dict[str, int]:
        scores = self.df["quality_score"].dropna()
        result: dict[str, int] = {}
        for thresh, grade, _ in GRADE_THRESHOLDS:
            mask = scores >= thresh
            result[grade] = int(mask.sum())
            scores = scores[~mask]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — ISSUE ANALYSIS SECTION
# Purpose : Severity breakdown, issue type grouping, per-class breakdown.
# ─────────────────────────────────────────────────────────────────────────────

class IssueAnalysisSection:
    """
    Organises all detected issues by severity and type.

    Severities : Critical → High → Medium → Low
    Types      : duplicates | corrupted | low-res | low-quality
    """

    def __init__(self, master_df: pd.DataFrame) -> None:
        self.df = master_df

    def to_dict(self) -> dict:
        return {
            "by_severity"   : self._by_severity(),
            "by_type"       : self._by_type(),
            "per_class"     : self._per_class(),
            "issue_images"  : self._issue_image_list(),
        }

    def _by_severity(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if "issue_severity" not in self.df.columns:
            return result
        n = len(self.df)
        for sev in SEVERITY_ORDER:
            count = int((self.df["issue_severity"] == sev).sum())
            if count > 0:
                result[sev] = {
                    "count"  : count,
                    "pct"    : round(count / n * 100, 1),
                    "emoji"  : SEVERITY_EMOJI.get(sev, ""),
                }
        return result

    def _by_type(self) -> dict[str, dict]:
        n = len(self.df)
        type_map = {
            "Duplicates" : "is_duplicate",
            "Corrupted"  : "is_corrupted",
            "Low-Res"    : "is_low_res",
            "Low-Quality": "is_low_quality",
        }
        result: dict[str, dict] = {}
        for label, col in type_map.items():
            if col in self.df.columns:
                count = int(self.df[col].sum())
                if count > 0:
                    result[label] = {
                        "count": count,
                        "pct"  : round(count / n * 100, 1),
                    }
        return result

    def _per_class(self) -> dict[str, dict]:
        """Issue breakdown per class/label when label column exists."""
        label_col = None
        if "label" in self.df.columns:
            label_col = "label"
        elif "file_path" in self.df.columns:
            parents = self.df["file_path"].apply(lambda p: Path(str(p)).parent.name)
            unique  = parents.unique()
            if 2 <= len(unique) <= 50:
                self.df = self.df.copy()
                self.df["_class"] = parents
                label_col = "_class"

        if label_col is None:
            return {}

        result: dict[str, dict] = {}
        for cls, grp in self.df.groupby(label_col):
            issues = 0
            for col in ("is_duplicate", "is_corrupted", "is_low_res", "is_low_quality"):
                if col in grp.columns:
                    issues += int(grp[col].sum())
            avg_q = (
                round(float(grp["quality_score"].mean()), 2)
                if "quality_score" in grp.columns else None
            )
            result[str(cls)] = {
                "count"      : len(grp),
                "total_issues": issues,
                "avg_quality": avg_q,
            }
        return result

    def _issue_image_list(self, max_per_type: int = 20) -> dict[str, list[str]]:
        """For each issue type, list the file names of affected images."""
        type_map = {
            "Duplicates" : "is_duplicate",
            "Corrupted"  : "is_corrupted",
            "Low-Res"    : "is_low_res",
            "Low-Quality": "is_low_quality",
        }
        result: dict[str, list[str]] = {}
        for label, col in type_map.items():
            if col in self.df.columns and "file_path" in self.df.columns:
                paths = (
                    self.df.loc[self.df[col].astype(bool), "file_path"]
                    .head(max_per_type)
                    .apply(lambda p: Path(str(p)).name)
                    .tolist()
                )
                if paths:
                    result[label] = paths
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — CLEANING SUMMARY SECTION
# Purpose : Before vs after score comparison, fix frequencies, improvement stats.
# ─────────────────────────────────────────────────────────────────────────────

class CleaningSummarySection:
    """
    Summarises what the image_cleaner changed.

    Covers
    ──────
    • Before / after average quality score
    • Per fix-type frequency table
    • Count of improved / unchanged / worsened images
    """

    def __init__(self, master_df: pd.DataFrame) -> None:
        self.df = master_df

    def to_dict(self) -> dict:
        if not self._cleaning_ran():
            return {"ran": False}
        return {
            "ran"           : True,
            "score_delta"   : self._score_delta(),
            "fix_frequency" : self._fix_frequency(),
            "improvement"   : self._improvement_counts(),
        }

    def _cleaning_ran(self) -> bool:
        return "cleaned" in self.df.columns or "fix_applied" in self.df.columns

    def _score_delta(self) -> dict:
        if "score_before" not in self.df.columns or "score_after" not in self.df.columns:
            return {}
        before = self.df["score_before"].dropna()
        after  = self.df["score_after"].dropna()
        return {
            "avg_before"     : round(float(before.mean()), 2),
            "avg_after"      : round(float(after.mean()), 2),
            "delta"          : round(float(after.mean()) - float(before.mean()), 2),
            "pct_improvement": round(
                (float(after.mean()) - float(before.mean())) / max(float(before.mean()), 1) * 100,
                1
            ),
        }

    def _fix_frequency(self) -> list[dict]:
        if "fix_applied" not in self.df.columns:
            return []
        counts = self.df["fix_applied"].dropna().value_counts()
        total  = counts.sum()
        return [
            {
                "fix"  : str(fix),
                "count": int(cnt),
                "pct"  : round(int(cnt) / total * 100, 1),
            }
            for fix, cnt in counts.items()
        ]

    def _improvement_counts(self) -> dict:
        if "score_before" not in self.df.columns or "score_after" not in self.df.columns:
            return {}
        diff = self.df["score_after"] - self.df["score_before"]
        return {
            "improved" : int((diff > 0).sum()),
            "unchanged": int((diff == 0).sum()),
            "worsened" : int((diff < 0).sum()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — RECOMMENDATIONS SECTION
# Purpose : Priority-ordered fix list based on detected issues.
#           Each item = issue + severity + count + suggested action.
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationsSection:
    """
    Auto-generates an ordered list of actionable recommendations.

    Logic
    ─────
    Scans the master_df for known issue patterns, ranks by severity × count,
    and maps each issue to a concrete suggested action.
    """

    # (issue_col, severity, label, action_template)
    _RULES: list[tuple[str, str, str, str]] = [
        (
            "is_corrupted",
            "critical",
            "Remove corrupted images",
            "Delete {count} corrupted files — they cannot be used for training.",
        ),
        (
            "is_duplicate",
            "high",
            "De-duplicate dataset",
            "Remove {count} duplicate images to prevent data leakage and biased models.",
        ),
        (
            "is_low_res",
            "high",
            "Exclude or upsample low-resolution images",
            "{count} images are below the minimum resolution threshold. "
            "Upscale with bicubic/ESRGAN or exclude them.",
        ),
        (
            "is_low_quality",
            "medium",
            "Enhance low-quality images",
            "{count} images scored below the quality threshold. "
            "Apply brightness/contrast correction or CLAHE.",
        ),
    ]

    # Score-based recommendations
    _SCORE_RULES: list[tuple[float, str, str, str]] = [
        (
            40.0, "blur_score", "critical",
            "Apply sharpening filter or collect sharper images "
            "— mean blur score is critically low ({mean:.1f}/100).",
        ),
        (
            40.0, "brightness_score", "high",
            "Normalise brightness — mean score {mean:.1f}/100 indicates "
            "systematic over/under-exposure.",
        ),
        (
            40.0, "noise_score", "medium",
            "Apply denoising (Gaussian / NLMeans) — mean noise score is {mean:.1f}/100.",
        ),
    ]

    def __init__(self, master_df: pd.DataFrame, summary: dict) -> None:
        self.df      = master_df
        self.summary = summary

    def to_dict(self) -> dict:
        recs = self._issue_recs() + self._score_recs()
        # Sort: critical first, then by count desc
        order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        recs.sort(key=lambda r: (order.get(r["severity"], 99), -r.get("count", 0)))
        return {
            "recommendations": recs,
            "total"          : len(recs),
        }

    def _issue_recs(self) -> list[dict]:
        recs: list[dict] = []
        n = len(self.df)
        for col, sev, label, action_tpl in self._RULES:
            if col not in self.df.columns:
                continue
            count = int(self.df[col].sum())
            if count == 0:
                continue
            recs.append({
                "label"   : label,
                "severity": sev,
                "count"   : count,
                "pct"     : round(count / n * 100, 1),
                "action"  : action_tpl.format(count=count),
                "emoji"   : SEVERITY_EMOJI.get(sev, ""),
            })
        return recs

    def _score_recs(self) -> list[dict]:
        recs: list[dict] = []
        for threshold, col, sev, action_tpl in self._SCORE_RULES:
            if col not in self.df.columns:
                continue
            mean_val = float(self.df[col].dropna().mean())
            if mean_val < threshold:
                recs.append({
                    "label"   : f"Low {col.replace('_score','').replace('_',' ').title()} Score",
                    "severity": sev,
                    "count"   : int((self.df[col] < threshold).sum()),
                    "action"  : action_tpl.format(mean=mean_val),
                    "emoji"   : SEVERITY_EMOJI.get(sev, ""),
                })
        return recs


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — HTML EXPORTER
# Purpose : Renders a responsive, interactive HTML report.
#           Uses inline Plotly JSON — no external dependencies.
# ─────────────────────────────────────────────────────────────────────────────

class HTMLExporter:
    """
    Produces a self-contained HTML report with embedded Plotly charts.

    The output file is a single .html — open it in any browser, no server needed.
    Charts are embedded as JSON (no CDN calls after first load).
    Sections are collapsible via vanilla JS.
    """

    def __init__(
        self,
        summary   : dict,
        overview  : dict,
        quality   : dict,
        issues    : dict,
        cleaning  : dict,
        recs      : dict,
    ) -> None:
        self.summary  = summary
        self.overview = overview
        self.quality  = quality
        self.issues   = issues
        self.cleaning = cleaning
        self.recs     = recs

    def export(self, output_path: str | Path) -> Path:
        """Write HTML to *output_path*. Returns the resolved path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(), encoding="utf-8")
        return path

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> str:
        s      = self.summary
        grade  = s.get("dataset_grade", "?")
        avg_q  = s.get("avg_quality_score", 0)
        label  = _score_to_grade(avg_q)[1]
        color  = _grade_color(grade)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Nydra Image Report</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1e2130; --border: #2d3250;
    --text: #e8eaf0; --muted: #9099b0; --accent: #4f8ef7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font-family: 'Segoe UI', system-ui, sans-serif; }}
  .container {{ max-width: 1100px; margin: auto; padding: 32px 20px; }}
  h1 {{ font-size: 2rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.25rem; color: var(--accent); margin: 28px 0 12px; }}
  h3 {{ font-size: 1rem; color: var(--muted); margin-bottom: 8px; }}
  .badge {{
    display: inline-block; padding: 6px 18px; border-radius: 8px;
    font-size: 2.5rem; font-weight: 900; color: {color}; border: 3px solid {color};
  }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 18px;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .stat {{ text-align: center; padding: 12px; }}
  .stat-val {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .stat-lbl {{ font-size: 0.8rem; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #252840; text-align: left; padding: 8px 12px;
        color: var(--muted); font-weight: 600; }}
  td {{ padding: 8px 12px; border-top: 1px solid var(--border); }}
  tr:hover td {{ background: #252840; }}
  details summary {{ cursor: pointer; font-weight: 600; color: var(--accent); }}
  .sev-critical {{ color: #e74c3c; }} .sev-high {{ color: #e67e22; }}
  .sev-medium   {{ color: #f1c40f; }} .sev-low  {{ color: #2ecc71; }}
  .chart-container {{ margin-top: 16px; }}
  @media (max-width: 640px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <!-- HEADER -->
  <h1>🩺 Nydra — Image Report</h1>
  <p class="meta">
    Generated: {s.get('generated_at','')} &nbsp;|&nbsp;
    Nydra v{s.get('nydra_ver','')} &nbsp;|&nbsp;
    {s.get('total_images', 0):,} images analysed
  </p>

  <!-- SCORE CARD -->
  <div class="card">
    <div style="display:flex;align-items:center;gap:32px;flex-wrap:wrap;">
      <div class="badge">{grade}</div>
      <div>
        <h2 style="margin:0">{label}</h2>
        <p style="color:var(--muted)">
          Average Quality Score: <strong style="color:{color}">{avg_q}/100</strong>
        </p>
        <p style="color:var(--muted)">
          Total Issues Found: <strong>{s.get('total_issues', 0)}</strong>
        </p>
      </div>
    </div>
  </div>

  <!-- KEY STATS -->
  <div class="card grid-2">
    {self._stat_html('Total Images',    f"{s.get('total_images',0):,}")}
    {self._stat_html('Avg Quality',     f"{avg_q}/100")}
    {self._stat_html('Total Issues',    str(s.get('total_issues', 0)))}
    {self._stat_html('Most Common Fix', s.get('most_common_fix','N/A'))}
  </div>

  {self._overview_html()}
  {self._quality_html()}
  {self._issues_html()}
  {self._cleaning_html()}
  {self._recs_html()}

  <p class="meta" style="margin-top:40px;text-align:center;">
    Built with ❤️ by Nydra Team &nbsp;|&nbsp;
    <a href="https://github.com/Denterio1/Nydra"
       style="color:var(--accent)">GitHub</a>
  </p>
</div>
{self._plotly_script()}
</body>
</html>"""

    # ── section helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _stat_html(label: str, value: str) -> str:
        return (
            f'<div class="stat">'
            f'<div class="stat-val">{value}</div>'
            f'<div class="stat-lbl">{label}</div>'
            f'</div>'
        )

    def _overview_html(self) -> str:
        ov = self.overview
        fmt_rows = "".join(
            f"<tr><td>{fmt}</td><td><strong>{cnt}</strong></td></tr>"
            for fmt, cnt in ov.get("format_breakdown", {}).items()
        )
        ba = ov.get("before_after", {})
        ba_html = ""
        if ba:
            delta_color = "#2ecc71" if ba.get("delta", 0) >= 0 else "#e74c3c"
            ba_html = f"""
            <h3>Before vs After Cleaning</h3>
            <table>
              <tr><th>Metric</th><th>Value</th></tr>
              <tr><td>Avg Before</td><td>{ba.get('avg_before', '—')}</td></tr>
              <tr><td>Avg After</td><td>{ba.get('avg_after', '—')}</td></tr>
              <tr><td>Delta</td>
                  <td style="color:{delta_color}"><strong>
                    {'+' if ba.get('delta',0)>=0 else ''}{ba.get('delta','—')}
                  </strong></td></tr>
              <tr><td>Improved</td><td>{ba.get('improved','—')}</td></tr>
              <tr><td>Unchanged</td><td>{ba.get('unchanged','—')}</td></tr>
              <tr><td>Worsened</td><td>{ba.get('worsened','—')}</td></tr>
            </table>"""
        return f"""
  <details open>
    <summary><h2 style="display:inline">📁 Dataset Overview</h2></summary>
    <div class="card" style="margin-top:12px">
      <h3>Format Distribution</h3>
      <table>
        <tr><th>Format</th><th>Count</th></tr>
        {fmt_rows or '<tr><td colspan="2">No format data</td></tr>'}
      </table>
      {ba_html}
    </div>
  </details>"""

    def _quality_html(self) -> str:
        q     = self.quality
        worst = q.get("top_10_worst", [])
        best  = q.get("top_10_best",  [])

        def table_rows(items: list[dict]) -> str:
            return "".join(
                f"<tr><td>{i['file']}</td><td>{i['score']}</td>"
                f"<td>{i['grade']}</td></tr>"
                for i in items
            ) or '<tr><td colspan="3">No data</td></tr>'

        metric_rows = "".join(
            f"<tr><td>{m}</td><td>{v['mean']}</td>"
            f"<td>{v['min']} – {v['max']}</td></tr>"
            for m, v in q.get("metric_breakdown", {}).items()
        )

        chart_div = '<div id="quality-chart" class="chart-container"></div>' if PLOTLY_AVAILABLE else ""

        return f"""
  <details open>
    <summary><h2 style="display:inline">📊 Quality Analysis</h2></summary>
    <div class="card" style="margin-top:12px">
      {chart_div}
      <div class="grid-2" style="margin-top:20px">
        <div>
          <h3>🔴 10 Worst Images</h3>
          <table>
            <tr><th>File</th><th>Score</th><th>Grade</th></tr>
            {table_rows(worst)}
          </table>
        </div>
        <div>
          <h3>🟢 10 Best Images</h3>
          <table>
            <tr><th>File</th><th>Score</th><th>Grade</th></tr>
            {table_rows(best)}
          </table>
        </div>
      </div>
      <h3 style="margin-top:20px">Per-Metric Breakdown</h3>
      <table>
        <tr><th>Metric</th><th>Mean</th><th>Range</th></tr>
        {metric_rows or '<tr><td colspan="3">No metric data</td></tr>'}
      </table>
    </div>
  </details>"""

    def _issues_html(self) -> str:
        issues = self.issues
        sev_rows = "".join(
            f'<tr><td class="sev-{sev}">{info["emoji"]} {sev.title()}</td>'
            f'<td>{info["count"]}</td><td>{info["pct"]}%</td></tr>'
            for sev, info in issues.get("by_severity", {}).items()
        )
        type_rows = "".join(
            f"<tr><td>{lbl}</td><td>{info['count']}</td><td>{info['pct']}%</td></tr>"
            for lbl, info in issues.get("by_type", {}).items()
        )
        return f"""
  <details open>
    <summary><h2 style="display:inline">⚠️ Issue Analysis</h2></summary>
    <div class="card grid-2" style="margin-top:12px">
      <div>
        <h3>By Severity</h3>
        <table>
          <tr><th>Severity</th><th>Count</th><th>%</th></tr>
          {sev_rows or '<tr><td colspan="3">No issues found ✅</td></tr>'}
        </table>
      </div>
      <div>
        <h3>By Type</h3>
        <table>
          <tr><th>Type</th><th>Count</th><th>%</th></tr>
          {type_rows or '<tr><td colspan="3">No issues found ✅</td></tr>'}
        </table>
      </div>
    </div>
  </details>"""

    def _cleaning_html(self) -> str:
        cl = self.cleaning
        if not cl.get("ran"):
            return """
  <div class="card">
    <h2>🧹 Cleaning Summary</h2>
    <p style="color:var(--muted)">Cleaning was not run on this dataset.</p>
  </div>"""
        delta  = cl.get("score_delta", {})
        imp    = cl.get("improvement", {})
        fix_rows = "".join(
            f"<tr><td>{r['fix']}</td><td>{r['count']}</td><td>{r['pct']}%</td></tr>"
            for r in cl.get("fix_frequency", [])
        )
        d_color = "#2ecc71" if delta.get("delta", 0) >= 0 else "#e74c3c"
        return f"""
  <details open>
    <summary><h2 style="display:inline">🧹 Cleaning Summary</h2></summary>
    <div class="card grid-2" style="margin-top:12px">
      <div>
        <h3>Score Delta</h3>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Before</td><td>{delta.get('avg_before','—')}</td></tr>
          <tr><td>After</td><td>{delta.get('avg_after','—')}</td></tr>
          <tr><td>Δ</td>
              <td style="color:{d_color}"><strong>
                {'+' if delta.get('delta',0)>=0 else ''}{delta.get('delta','—')}
                ({delta.get('pct_improvement','—')}%)
              </strong></td></tr>
          <tr><td>Improved</td><td>{imp.get('improved','—')}</td></tr>
          <tr><td>Unchanged</td><td>{imp.get('unchanged','—')}</td></tr>
          <tr><td>Worsened</td><td>{imp.get('worsened','—')}</td></tr>
        </table>
      </div>
      <div>
        <h3>Fix Frequency</h3>
        <table>
          <tr><th>Fix</th><th>Count</th><th>%</th></tr>
          {fix_rows or '<tr><td colspan="3">No fixes recorded</td></tr>'}
        </table>
      </div>
    </div>
  </details>"""

    def _recs_html(self) -> str:
        recs = self.recs.get("recommendations", [])
        if not recs:
            return """
  <div class="card">
    <h2>💡 Recommendations</h2>
    <p style="color:#2ecc71">✅ No critical issues — dataset looks healthy!</p>
  </div>"""
        rows = "".join(
            f'<tr><td class="sev-{r["severity"]}">{r.get("emoji","")} {r["severity"].title()}</td>'
            f'<td><strong>{r["label"]}</strong><br>'
            f'<small style="color:var(--muted)">{r["action"]}</small></td>'
            f'<td>{r.get("count","—")}</td></tr>'
            for r in recs
        )
        return f"""
  <details open>
    <summary><h2 style="display:inline">💡 Recommendations ({len(recs)})</h2></summary>
    <div class="card" style="margin-top:12px">
      <table>
        <tr><th>Severity</th><th>Recommendation</th><th>Affected</th></tr>
        {rows}
      </table>
    </div>
  </details>"""

    def _plotly_script(self) -> str:
        """Embed quality score histogram as inline Plotly JSON."""
        if not PLOTLY_AVAILABLE:
            return ""
        hist = self.quality.get("score_histogram", {})
        if not hist:
            return ""
        try:
            edges  = hist["bin_edges"]
            counts = hist["counts"]
            labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(counts))]
            colors = [SEVERITY_COLOR["critical"] if v < 45 else
                      SEVERITY_COLOR["high"]     if v < 60 else
                      SEVERITY_COLOR["medium"]   if v < 75 else
                      SEVERITY_COLOR["low"]
                      for v in [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]]

            fig = go.Figure(go.Bar(
                x=labels, y=counts, marker_color=colors,
                text=counts, textposition="outside",
            ))
            fig.update_layout(
                title="Quality Score Distribution",
                xaxis_title="Score Range",
                yaxis_title="Image Count",
                paper_bgcolor="#1e2130",
                plot_bgcolor="#1e2130",
                font=dict(color="#e8eaf0"),
                margin=dict(t=40, b=40, l=40, r=20),
                height=300,
            )
            fig_json = fig.to_json()
            return f"""
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  var figData = {fig_json};
  Plotly.newPlot('quality-chart', figData.data, figData.layout, {{responsive: true}});
</script>"""
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — JSON EXPORTER
# Purpose : Full machine-readable JSON dump of all stats.
#           Can be consumed by app.py or external tools.
# ─────────────────────────────────────────────────────────────────────────────

class JSONExporter:
    """
    Dumps the complete report data as a structured JSON file.

    The JSON schema is stable — external tools can rely on it.
    All NaN / Inf values are replaced with None (JSON null).
    """

    def __init__(self, all_sections: dict) -> None:
        self.data = all_sections

    def export(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = _sanitize_for_json(self.data)
        path.write_text(
            json.dumps(cleaned, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 9 — MARKDOWN EXPORTER
# Purpose : Clean Markdown with tables, emoji status, per-section anchors.
#           Ready for GitHub README or Notion.
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownExporter:
    """
    Produces a GitHub-flavoured Markdown report.

    Features
    ────────
    • Emoji status indicators per grade / severity
    • Tables for issues, quality metrics, recommendations
    • Per-section anchors for deep-linking
    • Lightweight — no external libs required
    """

    def __init__(
        self,
        summary   : dict,
        overview  : dict,
        quality   : dict,
        issues    : dict,
        cleaning  : dict,
        recs      : dict,
    ) -> None:
        self.summary  = summary
        self.overview = overview
        self.quality  = quality
        self.issues   = issues
        self.cleaning = cleaning
        self.recs     = recs

    def export(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(), encoding="utf-8")
        return path

    def _render(self) -> str:
        s     = self.summary
        grade = s.get("dataset_grade", "?")
        avg_q = s.get("avg_quality_score", 0)
        label = _score_to_grade(avg_q)[1]

        parts = [
            f"# 🩺 Nydra — Image Report\n",
            f"> Generated: {s.get('generated_at','')}  \n"
            f"> Nydra v{s.get('nydra_ver','')}  \n"
            f"> {s.get('total_images',0):,} images analysed\n",
            "---\n",
            "## 🏆 Dataset Grade\n",
            f"**Grade : `{grade}` — {label}**  \n"
            f"Average Quality Score: **{avg_q}/100**  \n"
            f"Total Issues: **{s.get('total_issues', 0)}**\n",
            "---\n",
            self._overview_md(),
            self._quality_md(),
            self._issues_md(),
            self._cleaning_md(),
            self._recs_md(),
            "---\n",
            "*Built with ❤️ by [Nydra Team](https://github.com/Denterio1/Nydra) — Algeria 🇩🇿*\n",
        ]
        return "\n".join(parts)

    # ── sections ──────────────────────────────────────────────────────────────

    def _overview_md(self) -> str:
        ov  = self.overview
        fmts = ov.get("format_breakdown", {})
        lines = [
            "## 📁 Dataset Overview\n",
            "### Format Distribution\n",
            "| Format | Count |",
            "|--------|-------|",
        ]
        if fmts:
            lines += [f"| {f} | {c} |" for f, c in fmts.items()]
        else:
            lines.append("| — | — |")

        ba = ov.get("before_after", {})
        if ba:
            delta = ba.get("delta", 0)
            sign  = "+" if delta >= 0 else ""
            lines += [
                "\n### Before vs After Cleaning\n",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Avg Before | {ba.get('avg_before','—')} |",
                f"| Avg After  | {ba.get('avg_after','—')} |",
                f"| Δ          | **{sign}{delta}** |",
                f"| Improved   | {ba.get('improved','—')} |",
                f"| Unchanged  | {ba.get('unchanged','—')} |",
                f"| Worsened   | {ba.get('worsened','—')} |",
            ]
        return "\n".join(lines) + "\n\n---\n"

    def _quality_md(self) -> str:
        q     = self.quality
        worst = q.get("top_10_worst", [])
        best  = q.get("top_10_best",  [])
        metrics = q.get("metric_breakdown", {})

        def table(items: list[dict]) -> list[str]:
            rows = ["| File | Score | Grade |", "|------|-------|-------|"]
            for i in items:
                rows.append(f"| {i['file']} | {i['score']} | {i['grade']} |")
            return rows

        lines = [
            "## 📊 Quality Analysis\n",
            "### 🔴 10 Worst Images\n",
        ] + table(worst) + [
            "\n### 🟢 10 Best Images\n",
        ] + table(best) + [
            "\n### Per-Metric Breakdown\n",
            "| Metric | Mean | Min | Max |",
            "|--------|------|-----|-----|",
        ]
        for m, v in metrics.items():
            lines.append(f"| {m} | {v['mean']} | {v['min']} | {v['max']} |")
        return "\n".join(lines) + "\n\n---\n"

    def _issues_md(self) -> str:
        issues = self.issues
        lines  = ["## ⚠️ Issue Analysis\n"]

        by_sev = issues.get("by_severity", {})
        if by_sev:
            lines += [
                "### By Severity\n",
                "| Severity | Count | % |",
                "|----------|-------|---|",
            ]
            for sev, info in by_sev.items():
                lines.append(
                    f"| {info.get('emoji','')} {sev.title()} | {info['count']} | {info['pct']}% |"
                )

        by_type = issues.get("by_type", {})
        if by_type:
            lines += [
                "\n### By Type\n",
                "| Type | Count | % |",
                "|------|-------|---|",
            ]
            for t, info in by_type.items():
                lines.append(f"| {t} | {info['count']} | {info['pct']}% |")

        if not by_sev and not by_type:
            lines.append("✅ No issues found — dataset looks healthy!")
        return "\n".join(lines) + "\n\n---\n"

    def _cleaning_md(self) -> str:
        cl = self.cleaning
        if not cl.get("ran"):
            return "## 🧹 Cleaning Summary\n\nCleaning was not run.\n\n---\n"
        delta = cl.get("score_delta", {})
        imp   = cl.get("improvement", {})
        delta_val = delta.get("delta", 0)
        sign      = "+" if delta_val >= 0 else ""
        lines = [
            "## 🧹 Cleaning Summary\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Avg Before | {delta.get('avg_before','—')} |",
            f"| Avg After  | {delta.get('avg_after','—')} |",
            f"| Δ          | **{sign}{delta_val}** ({delta.get('pct_improvement','—')}%) |",
            f"| Improved   | {imp.get('improved','—')} |",
            f"| Unchanged  | {imp.get('unchanged','—')} |",
            f"| Worsened   | {imp.get('worsened','—')} |",
            "\n### Fix Frequency\n",
            "| Fix | Count | % |",
            "|-----|-------|---|",
        ]
        for r in cl.get("fix_frequency", []):
            lines.append(f"| {r['fix']} | {r['count']} | {r['pct']}% |")
        return "\n".join(lines) + "\n\n---\n"

    def _recs_md(self) -> str:
        recs = self.recs.get("recommendations", [])
        lines = ["## 💡 Recommendations\n"]
        if not recs:
            lines.append("✅ No recommendations — dataset is in great shape!\n")
            return "\n".join(lines) + "\n---\n"
        lines += [
            "| # | Severity | Recommendation | Affected |",
            "|---|----------|----------------|---------|",
        ]
        for i, r in enumerate(recs, 1):
            emoji = r.get("emoji", "")
            lines.append(
                f"| {i} | {emoji} {r['severity'].title()} | "
                f"**{r['label']}** — {r['action']} | {r.get('count','—')} |"
            )
        return "\n".join(lines) + "\n\n---\n"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 10 — ImageReport MASTER CLASS + CONVENIENCE FUNCTIONS
# Purpose : One class, one call — generates all formats.
# ─────────────────────────────────────────────────────────────────────────────

class ImageReport:
    """
    Master report generator for the Nydra image pipeline.

    Wires together all 9 previous classes into a single API.

    Usage
    ─────
    report = ImageReport(
        loader_df   = loader_result["df"],
        quality_df  = quality_result["df"],
        analyzer_df = analyzer_result["df"],
        cleaner_df  = cleaner_result["df"],   # optional
    )

    # Generate all formats
    paths = report.generate(output_dir="reports/", formats=["html", "json", "md"])

    # Quick markdown only
    path = report.quick_report(output_dir="reports/")
    """

    def __init__(
        self,
        loader_df   : pd.DataFrame,
        quality_df  : pd.DataFrame | None = None,
        analyzer_df : pd.DataFrame | None = None,
        cleaner_df  : pd.DataFrame | None = None,
    ) -> None:
        # ── Build master DataFrame + summary ──
        self._builder = ReportBuilder(
            loader_df   = loader_df,
            quality_df  = quality_df,
            analyzer_df = analyzer_df,
            cleaner_df  = cleaner_df,
        )
        df      = self._builder.master_df
        summary = self._builder.summary

        # ── Build each section ────────────────
        self._overview  = DatasetOverviewSection(df, summary).to_dict()
        self._quality   = QualityAnalysisSection(df).to_dict()
        self._issues    = IssueAnalysisSection(df).to_dict()
        self._cleaning  = CleaningSummarySection(df).to_dict()
        self._recs      = RecommendationsSection(df, summary).to_dict()
        self._summary   = summary

        # All data in one dict for JSON export
        self._all = {
            "summary"     : summary,
            "overview"    : self._overview,
            "quality"     : self._quality,
            "issues"      : self._issues,
            "cleaning"    : self._cleaning,
            "recommendations": self._recs,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        output_dir : str | Path = "reports",
        formats    : list[str] = ("html", "json", "md"),
        prefix     : str = "image_report",
    ) -> dict[str, Path]:
        """
        Generate report in one or more formats.

        Parameters
        ----------
        output_dir : where to write files
        formats    : any subset of ["html", "json", "md"]
        prefix     : filename prefix (default "image_report")

        Returns
        -------
        dict  mapping format → resolved Path
        """
        out  = Path(output_dir)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out / f"{prefix}_{ts}"
        paths: dict[str, Path] = {}

        if "html" in formats:
            paths["html"] = HTMLExporter(
                self._summary, self._overview, self._quality,
                self._issues, self._cleaning, self._recs,
            ).export(base.with_suffix(".html"))

        if "json" in formats:
            paths["json"] = JSONExporter(self._all).export(
                base.with_suffix(".json")
            )

        if "md" in formats:
            paths["md"] = MarkdownExporter(
                self._summary, self._overview, self._quality,
                self._issues, self._cleaning, self._recs,
            ).export(base.with_suffix(".md"))

        return paths

    def quick_report(
        self,
        output_dir : str | Path = "reports",
        prefix     : str = "quick_image_report",
    ) -> Path:
        """Generate Markdown only — fast, no external deps."""
        return self.generate(
            output_dir=output_dir,
            formats=["md"],
            prefix=prefix,
        )["md"]

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def master_df(self) -> pd.DataFrame:
        return self._builder.master_df

    @property
    def summary(self) -> dict:
        return self._summary

    @property
    def all_data(self) -> dict:
        return self._all


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS (module-level shortcuts)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    df         : pd.DataFrame,
    output_dir : str | Path = "reports",
    formats    : list[str] = ("html", "json", "md"),
    quality_df  : pd.DataFrame | None = None,
    analyzer_df : pd.DataFrame | None = None,
    cleaner_df  : pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    One-call report generation.

    Parameters
    ----------
    df          : loader DataFrame (must have 'file_path' column)
    output_dir  : output directory
    formats     : list of formats — ["html", "json", "md"]
    quality_df  : output of image_quality (optional)
    analyzer_df : output of image_analyzer (optional)
    cleaner_df  : output of image_cleaner (optional)

    Returns
    -------
    dict  {format: Path}  ← paths to generated files
    """
    report = ImageReport(
        loader_df   = df,
        quality_df  = quality_df,
        analyzer_df = analyzer_df,
        cleaner_df  = cleaner_df,
    )
    return report.generate(output_dir=output_dir, formats=formats)


def quick_report(
    df         : pd.DataFrame,
    output_dir : str | Path = "reports",
) -> Path:
    """
    Markdown-only fast report.

    Parameters
    ----------
    df         : loader DataFrame (must have 'file_path' column)
    output_dir : output directory

    Returns
    -------
    Path  to the generated .md file
    """
    report = ImageReport(loader_df=df)
    return report.quick_report(output_dir=output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_grade(score: float) -> tuple[str, str]:
    """Convert a numeric score to (grade_letter, label_string)."""
    for thresh, grade, label in GRADE_THRESHOLDS:
        if score >= thresh:
            return grade, label
    return "F", "Critical ⛔"


def _grade_color(grade: str) -> str:
    return {
        "A": "#2ecc71",
        "B": "#f1c40f",
        "C": "#e67e22",
        "D": "#e74c3c",
        "F": "#8e44ad",
    }.get(grade, "#9099b0")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN / Inf with None so json.dumps doesn't fail."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
