"""
cognitive_dna.py — DataDNA v0.5.0

Computes the "statistical identity" of a dataset across four dimensions:

1. Statistical  (entropy, moments, distribution shape, correlation structure)
2. Structural   (schema, missingness, outliers, duplicates, cardinality)
3. ML           (task type, target signal, class balance, redundancy, complexity)
4. Temporal     (date columns, trend, seasonality, gap regularity)

Public API
----------
compute_dna(df, target_col=None, source="dataframe") -> DataDNA
CognitiveDNA(df, target_col=None)  -> thin wrapper (CLI compatible)
DNASimilarityEngine().compare(a, b) -> SimilarityReport

Design notes
------------
* Deterministic: same data (any row order) -> same ``dna_hash``.
  Large frames are sampled with a fixed seed; the hash also includes the true
  row count.
* ``entropy_vector`` has exactly one value per column, in column order
  (the CLI zips it with ``df.columns``).
* The missing mechanism is estimated as NONE / MCAR / MAR. MNAR cannot be
  identified from observed data alone, so it is never reported.
* ``recommended_task`` is "unsupervised" and ``target_signal`` is "none"
  when no target column is given.
* Complexity and personality are documented heuristics, not ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DNA_VERSION = "0.5.0"
SAMPLE_ROWS = 50_000       # frames above this are sampled for the statistics
SAMPLE_SEED = 0
MAX_CORR_COLS = 200        # numeric columns used for the correlation matrix
MAX_MI_COLS = 12           # numeric columns used for the mutual-information matrix
MAX_ASSOC_FEATURES = 100   # features scored against the target
MAX_CATEGORY_LEVELS = 50   # categorical features above this are treated as ID-like


# ══════════════════════════════════════════════════════════════════════════════
# Data structures (field layout is relied on by dna_memory.py — do not change)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StatisticalDNA:
    entropy_vector:     Tuple[float, ...]
    moment_signatures:  Tuple[Tuple[float, ...], ...]
    distribution_types: Tuple[str, ...]
    correlation_hash:   str
    normality_scores:   Tuple[float, ...]
    mutual_info_matrix: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class StructuralDNA:
    schema_hash:          str
    missing_mechanism:    str  # "NONE", "MCAR", "MAR"
    missing_pattern_hash: str
    cardinality_profile:  Tuple[int, ...]
    outlier_density:      float
    duplicate_ratio:      float
    shape_signature:      Tuple[int, int]  # (rows, cols)


@dataclass(frozen=True)
class MLDNA:
    feature_redundancy:  float
    target_signal:       str  # "strong", "medium", "weak", "none"
    separability_score:  float
    class_balance_score: float
    recommended_task:    str  # "classification", "regression", "unsupervised"
    complexity_score:    float


@dataclass(frozen=True)
class TemporalDNA:
    has_temporal:      bool
    temporal_columns:  Tuple[str, ...]
    trend_detected:    bool
    seasonality_score: float
    time_gaps_uniform: bool


@dataclass
class DataDNA:
    statistical: StatisticalDNA
    structural:  StructuralDNA
    ml:          MLDNA
    temporal:    TemporalDNA
    dna_hash:    str
    created_at:  str
    source:      str
    version:     str = "0.4.0"
    personality: str = ""
    tags:        List[str] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        return self.dna_hash[:8]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary (used by the API)."""
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class SimilarityReport:
    overall_similarity: float
    dimension_scores:   Dict[str, float]
    key_differences:    List[str]
    recommendation:     str


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(x: Any) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "nan"
    return format(f, ".6g") if math.isfinite(f) else "nan"


def _sha(parts: List[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _floats(s: pd.Series) -> np.ndarray:
    """Series -> float array with NaN for missing and non-finite values."""
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    v = np.array(v, dtype=float)
    v[~np.isfinite(v)] = np.nan
    return v


def _unique_names(columns) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in columns:
        name = str(c)
        if name in seen:
            seen[name] += 1
            name = f"{name}#{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _ratio(a: float, b: float) -> float:
    a, b = abs(float(a)), abs(float(b))
    if a == 0 and b == 0:
        return 1.0
    return min(a, b) / max(a, b)


def _close(a: float, b: float, scale: float) -> float:
    return float(max(0.0, 1.0 - abs(float(a) - float(b)) / scale))


# ══════════════════════════════════════════════════════════════════════════════
# Statistical building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _entropy_bits(s: pd.Series, numeric: bool) -> float:
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    if numeric:
        v = _floats(s)
        v = v[np.isfinite(v)]
        if v.size == 0 or np.ptp(v) == 0:
            return 0.0
        bins = int(np.clip(math.sqrt(v.size), 2, 50))
        counts, _ = np.histogram(v, bins=bins)
    else:
        counts = s.astype(str).value_counts().to_numpy()
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _moments(v: np.ndarray) -> Tuple[float, float, float, float]:
    """(mean, std, skewness, excess kurtosis) of finite values."""
    if v.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    mean = float(v.mean())
    std = float(v.std())
    if std == 0:
        return (mean, 0.0, 0.0, 0.0)
    z = (v - mean) / std
    return (mean, std, float((z ** 3).mean()), float((z ** 4).mean() - 3.0))


def _normality(v: np.ndarray, skew: float, kurt: float) -> float:
    """Jarque-Bera p-value (chi-square with 2 dof has the closed form exp(-x/2))."""
    n = v.size
    if n < 8:
        return 0.0
    jb = n / 6.0 * (skew ** 2 + (kurt ** 2) / 4.0)
    return float(math.exp(-jb / 2.0))


def _distribution_type(v: np.ndarray, skew: float, kurt: float, p_norm: float) -> str:
    if v.size == 0:
        return "empty"
    uniq = np.unique(v)
    if uniq.size == 1:
        return "constant"
    if uniq.size == 2:
        return "binary"
    if np.all(np.mod(v, 1) == 0) and uniq.size <= 15:
        return "discrete"
    if p_norm > 0.05:
        return "normal"
    if abs(skew) < 0.3 and -1.5 < kurt < -0.9:
        return "uniform"
    if skew > 1.0:
        return "right_skewed"
    if skew < -1.0:
        return "left_skewed"
    if kurt > 3.0:
        return "heavy_tailed"
    return "other"


def _nmi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Normalised mutual information of two numeric arrays, in [0, 1]."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = joint / joint.sum()
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    nz = pxy > 0
    outer = np.outer(px, py)
    mi = float((pxy[nz] * np.log(pxy[nz] / outer[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    d = min(hx, hy)
    return float(np.clip(mi / d, 0.0, 1.0)) if d > 0 else 0.0


def _spearman_signed(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return 0.0
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _eta(values: np.ndarray, groups: np.ndarray) -> float:
    """Correlation ratio between a numeric variable and integer group codes."""
    m = np.isfinite(values) & (groups >= 0)
    v, g = values[m], groups[m]
    if v.size < 5 or np.ptp(v) == 0:
        return 0.0
    mean = v.mean()
    sst = float(((v - mean) ** 2).sum())
    counts = np.bincount(g)
    sums = np.bincount(g, weights=v)
    ok = counts > 0
    ssb = float((counts[ok] * ((sums[ok] / counts[ok]) - mean) ** 2).sum())
    return float(math.sqrt(max(0.0, min(1.0, ssb / sst))))


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramér's V between two integer code arrays."""
    m = (a >= 0) & (b >= 0)
    a, b = a[m], b[m]
    if a.size < 5:
        return 0.0
    r, k = int(a.max()) + 1, int(b.max()) + 1
    table = np.bincount(a * k + b, minlength=r * k).reshape(r, k).astype(float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    r, k = table.shape
    if r < 2 or k < 2:
        return 0.0
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(math.sqrt(phi2 / denom)) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Structural building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _missing_mechanism(work: pd.DataFrame, num_cols: List[str]) -> str:
    miss = work.isna()
    if not miss.to_numpy().any():
        return "NONE"
    rates = miss.mean()
    targets = [c for c in work.columns if 0 < rates[c] < 1][:30]
    others = num_cols[:30]
    for c in targets:
        ind = miss[c].to_numpy().astype(float)
        for o in others:
            if o == c:
                continue
            vals = _floats(work[o])
            ok = np.isfinite(vals)
            if ok.sum() < 20 or ind[ok].std() == 0 or vals[ok].std() == 0:
                continue
            if min(ind[ok].sum(), (1 - ind[ok]).sum()) < 5:
                continue
            r = float(np.corrcoef(ind[ok], vals[ok])[0, 1])
            n = int(ok.sum())
            t = abs(r) * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
            if abs(r) >= 0.15 and t > 3.3:
                return "MAR"
    return "MCAR"


def _missing_pattern_hash(work: pd.DataFrame) -> str:
    miss = work.isna()
    arr = miss.to_numpy()
    if not arr.any():
        return "none"
    n_patterns = int(np.unique(arr.astype(np.uint8), axis=0).shape[0])
    rates = [_fmt(round(float(x), 3)) for x in miss.mean().to_numpy()]
    return _sha(["missing", n_patterns] + rates)[:16]


def _outlier_density(work: pd.DataFrame, num_cols: List[str]) -> float:
    """Share of numeric cells outside the 1.5 x IQR fences."""
    total = 0
    out = 0
    for c in num_cols:
        v = _floats(work[c])
        v = v[np.isfinite(v)]
        if v.size < 8:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out += int(((v < lo) | (v > hi)).sum())
        total += v.size
    return round(out / total, 4) if total else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ML building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _infer_task(t: pd.Series) -> str:
    t = t.dropna()
    if t.empty:
        raise ValueError("Target column has no values")
    if not _is_numeric(t):
        return "classification"
    nun = int(t.nunique())
    if nun <= 2:
        return "classification"
    v = _floats(t)
    v = v[np.isfinite(v)]
    int_like = v.size > 0 and bool(np.all(np.mod(v, 1) == 0))
    return "classification" if (int_like and nun <= 10) else "regression"


def _is_id_like(s: pd.Series, n: int) -> bool:
    nun = s.nunique(dropna=True)
    if n == 0:
        return False
    if _is_numeric(s) and pd.api.types.is_integer_dtype(s):
        return nun / n > 0.98
    return False


def _target_associations(
    work: pd.DataFrame, kinds: Dict[str, str], target: str, task: str
) -> List[float]:
    t = work[target]
    t_codes, _ = pd.factorize(t)
    t_float = _floats(t) if task == "regression" else None
    n = len(work)
    scores: List[float] = []
    features = [c for c in work.columns if c != target][:MAX_ASSOC_FEATURES]
    for c in features:
        s = work[c]
        kind = kinds[c]
        if kind == "datetime" or _is_id_like(s, n):
            continue
        if kind == "numeric":
            f = _floats(s)
            if task == "regression":
                scores.append(abs(_spearman_signed(f, t_float)))
            else:
                scores.append(_eta(f, t_codes))
        else:
            nun = s.nunique(dropna=True)
            if nun < 2 or nun > MAX_CATEGORY_LEVELS or nun / n > 0.9:
                continue
            f_codes, _ = pd.factorize(s.astype(object))
            if task == "classification":
                scores.append(_cramers_v(f_codes, t_codes))
            else:
                scores.append(_eta(t_float, f_codes))
    return scores


def _signal_label(top: float) -> str:
    if top >= 0.5:
        return "strong"
    if top >= 0.25:
        return "medium"
    if top >= 0.10:
        return "weak"
    return "none"


# ══════════════════════════════════════════════════════════════════════════════
# Temporal building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


def _detect_temporal(work: pd.DataFrame, kinds: Dict[str, str]) -> Dict[str, pd.Series]:
    found: Dict[str, pd.Series] = {}
    for c in work.columns:
        s = work[c]
        if kinds[c] == "datetime":
            found[c] = s
            continue
        if kinds[c] != "categorical":
            continue
        sample = s.dropna().astype(str).head(200)
        if len(sample) < 5:
            continue
        if not all(any(ch in v for ch in "-/:") for v in sample):
            continue
        if _to_datetime(sample).notna().mean() >= 0.9:
            parsed = _to_datetime(s)
            if parsed.notna().mean() >= 0.9:
                found[c] = parsed
    return found


def _temporal_dna(
    work: pd.DataFrame, kinds: Dict[str, str], num_cols: List[str]
) -> TemporalDNA:
    found = _detect_temporal(work, kinds)
    if not found:
        return TemporalDNA(False, (), False, 0.0, True)

    first = next(iter(found))
    ts = found[first]
    ok = ts.notna().to_numpy()
    order = np.argsort(ts.to_numpy()[ok].astype("datetime64[ns]").astype(np.int64), kind="stable")

    # gap regularity
    uniq = np.unique(ts[ok].to_numpy().astype("datetime64[ns]").astype(np.int64))
    if uniq.size < 3:
        uniform = True
    else:
        diffs = np.diff(uniq).astype(float)
        mean = diffs.mean()
        uniform = bool(mean > 0 and diffs.std() / mean < 0.1)

    # trend and seasonality on up to 3 numeric series
    trend = False
    season = 0.0
    for c in [x for x in num_cols if x not in found][:3]:
        v = _floats(work[c])[ok][order]
        finite = np.isfinite(v)
        v = v[finite]
        if v.size < 10:
            continue
        idx = np.arange(v.size, dtype=float)
        if abs(_spearman_signed(idx, v)) > 0.3:
            trend = True
        if v.size >= 16 and np.ptp(v) > 0:
            slope, intercept = np.polyfit(idx, v, 1)
            resid = v - (slope * idx + intercept)
            denom = float((resid ** 2).sum())
            if denom > 0:
                best = 0.0
                for lag in range(2, min(v.size // 2, 60) + 1):
                    ac = float((resid[:-lag] * resid[lag:]).sum() / denom)
                    best = max(best, ac)
                season = max(season, best)
    return TemporalDNA(
        True, tuple(found.keys()), trend, round(float(np.clip(season, 0.0, 1.0)), 4), uniform
    )


# ══════════════════════════════════════════════════════════════════════════════
# compute_dna
# ══════════════════════════════════════════════════════════════════════════════

def _content_fingerprint(
    work: pd.DataFrame, kinds: Dict[str, str], n_rows: int, target: Optional[str]
) -> List[Any]:
    parts: List[Any] = ["rows", n_rows, "target", target or ""]
    for c in work.columns:
        s = work[c]
        parts.append(c)
        if kinds[c] == "numeric":
            v = _floats(s)
            v = v[np.isfinite(v)]
            if v.size:
                q = np.percentile(v, [0, 25, 50, 75, 100])
                parts += [v.size, _fmt(v.mean()), _fmt(v.std())] + [_fmt(x) for x in q]
            else:
                parts.append("empty")
        elif kinds[c] == "datetime":
            nn = s.dropna()
            parts += [len(nn), str(nn.min()), str(nn.max()), nn.nunique()]
        else:
            vc = s.dropna().astype(str).value_counts()
            top = sorted(vc.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            parts += [len(s.dropna()), vc.size] + [f"{k}={v}" for k, v in top]
    return parts


def _personality_and_tags(
    n_rows: int, n_cols: int, missing_frac: float, dup: float, outliers: float,
    ml: MLDNA, temporal: TemporalDNA, kinds: Dict[str, str],
) -> Tuple[str, List[str]]:
    tags: List[str] = []
    if missing_frac > 0:
        tags.append("has-missing")
    if dup > 0:
        tags.append("has-duplicates")
    if outliers > 0.02:
        tags.append("has-outliers")
    if temporal.has_temporal:
        tags.append("temporal")
    if n_cols >= 50:
        tags.append("wide")
    if n_rows < 100:
        tags.append("tiny")
    if n_rows >= 100_000:
        tags.append("large")
    if ml.recommended_task == "classification" and ml.class_balance_score < 0.7:
        tags.append("imbalanced")
    if ml.target_signal == "strong":
        tags.append("strong-signal")
    k = Counter(kinds.values())
    if k["categorical"] == 0 and k["numeric"] > 0:
        tags.append("numeric-only")
    elif k["numeric"] > 0 and k["categorical"] > 0:
        tags.append("mixed-types")
    elif k["categorical"] > k["numeric"]:
        tags.append("categorical-heavy")

    dirty = missing_frac > 0.10 or dup > 0.05 or outliers > 0.05
    if temporal.has_temporal:
        personality = "Time-Series Dataset"
    elif dirty:
        personality = "Messy Dataset"
    elif n_cols >= 50:
        personality = "High-Dimensional Dataset"
    elif n_rows < 100:
        personality = "Small Dataset"
    elif ml.target_signal == "strong":
        personality = "Predictive Dataset"
    else:
        personality = "Standard Dataset"
    return personality, tags


def compute_dna(
    df: pd.DataFrame, target_col: Optional[str] = None, source: str = "dataframe"
) -> DataDNA:
    """Compute the real DataDNA of a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("compute_dna expects a pandas DataFrame")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Cannot compute the DNA of an empty DataFrame")

    names = _unique_names(df.columns)
    work = df.set_axis(names, axis=1)
    target: Optional[str] = None
    if target_col is not None:
        if target_col in work.columns:
            target = target_col
        elif str(target_col) in work.columns:
            target = str(target_col)
        else:
            raise ValueError(f"Target column '{target_col}' not found")

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    if n_rows > SAMPLE_ROWS:
        work = work.sample(n=SAMPLE_ROWS, random_state=SAMPLE_SEED).sort_index()

    cols = list(work.columns)
    kinds: Dict[str, str] = {}
    for c in cols:
        s = work[c]
        kinds[c] = "numeric" if _is_numeric(s) else "datetime" if _is_datetime(s) else "categorical"
    num_cols = [c for c in cols if kinds[c] == "numeric"]

    # ── Statistical ──────────────────────────────────────────────────────────
    entropy = tuple(round(_entropy_bits(work[c], kinds[c] == "numeric"), 4) for c in cols)
    moments: List[Tuple[float, ...]] = []
    dist_types: List[str] = []
    normality: List[float] = []
    for c in num_cols:
        v = _floats(work[c])
        v = v[np.isfinite(v)]
        mean, std, skew, kurt = _moments(v)
        p = _normality(v, skew, kurt)
        moments.append(tuple(round(x, 6) for x in (mean, std, skew, kurt)))
        normality.append(round(p, 4))
        dist_types.append(_distribution_type(v, skew, kurt, p))

    corr_cols = num_cols[:MAX_CORR_COLS]
    corr = None
    if len(corr_cols) >= 2:
        numdf = pd.DataFrame({c: _floats(work[c]) for c in corr_cols})
        corr = np.nan_to_num(numdf.corr().to_numpy(), nan=0.0)
        iu = np.triu_indices(len(corr_cols), k=1)
        correlation_hash = _sha(
            ["corr", len(corr_cols)] + [_fmt(round(float(x), 1)) for x in corr[iu]]
        )[:16]
    else:
        correlation_hash = "none"

    mi_cols = num_cols[:MAX_MI_COLS]
    mi_matrix: List[Tuple[float, ...]] = []
    if len(mi_cols) >= 2:
        arrays = {c: _floats(work[c]) for c in mi_cols}
        size = len(mi_cols)
        mi = np.eye(size)
        for i in range(size):
            for j in range(i + 1, size):
                mi[i, j] = mi[j, i] = _nmi(arrays[mi_cols[i]], arrays[mi_cols[j]])
        mi_matrix = [tuple(round(float(x), 4) for x in row) for row in mi]

    statistical = StatisticalDNA(
        entropy_vector=entropy,
        moment_signatures=tuple(moments),
        distribution_types=tuple(dist_types),
        correlation_hash=correlation_hash,
        normality_scores=tuple(normality),
        mutual_info_matrix=tuple(mi_matrix),
    )

    # ── Structural ───────────────────────────────────────────────────────────
    schema_hash = _sha(
        ["schema"] + [f"{c}:{kinds[c]}:{str(work[c].dtype)}" for c in cols]
    )[:16]
    cardinality = tuple(int(work[c].nunique(dropna=True)) for c in cols)
    dup_ratio = round(float(work.duplicated().mean()), 4)
    outliers = _outlier_density(work, num_cols)
    miss_frac = float(work.isna().to_numpy().mean())
    structural = StructuralDNA(
        schema_hash=schema_hash,
        missing_mechanism=_missing_mechanism(work, num_cols),
        missing_pattern_hash=_missing_pattern_hash(work),
        cardinality_profile=cardinality,
        outlier_density=outliers,
        duplicate_ratio=dup_ratio,
        shape_signature=(n_rows, n_cols),
    )

    # ── ML ───────────────────────────────────────────────────────────────────
    feat_cols = [c for c in corr_cols if c != target]
    redundancy = 0.0
    if corr is not None and len(feat_cols) >= 2:
        idx = [corr_cols.index(c) for c in feat_cols]
        sub = np.abs(corr[np.ix_(idx, idx)])
        redundancy = float(sub[np.triu_indices(len(idx), k=1)].mean())

    if target is None:
        task, signal, separability, balance = "unsupervised", "none", 0.0, 1.0
    else:
        task = _infer_task(work[target])
        assoc = _target_associations(work, kinds, target, task)
        if assoc:
            top = sorted(assoc, reverse=True)
            signal = _signal_label(top[0])
            separability = float(np.mean(top[:3]))
        else:
            signal, separability = "none", 0.0
        if task == "classification":
            counts = work[target].dropna().value_counts().to_numpy()
            if counts.size < 2:
                balance = 0.0
            else:
                p = counts / counts.sum()
                balance = float(-(p * np.log(p)).sum() / math.log(counts.size))
        else:
            balance = 1.0

    high_card = float(np.mean([cd / max(1, n_rows) > 0.5 for cd in cardinality]))
    sep_term = (1.0 - separability) if target is not None else 0.5
    complexity = (
        0.25 * min(math.log10(n_cols + 1) / 2.0, 1.0)
        + 0.20 * min(miss_frac * 5.0, 1.0)
        + 0.20 * sep_term
        + 0.20 * (1.0 - balance)
        + 0.15 * high_card
    )
    ml = MLDNA(
        feature_redundancy=round(redundancy, 4),
        target_signal=signal,
        separability_score=round(separability, 4),
        class_balance_score=round(balance, 4),
        recommended_task=task,
        complexity_score=round(float(np.clip(complexity, 0.0, 1.0)), 4),
    )

    # ── Temporal ─────────────────────────────────────────────────────────────
    temporal = _temporal_dna(work, kinds, num_cols)

    personality, tags = _personality_and_tags(
        n_rows, n_cols, miss_frac, dup_ratio, outliers, ml, temporal, kinds
    )
    dna_hash = _sha(
        [DNA_VERSION, schema_hash] + _content_fingerprint(work, kinds, n_rows, target)
    )
    return DataDNA(
        statistical=statistical,
        structural=structural,
        ml=ml,
        temporal=temporal,
        dna_hash=dna_hash,
        created_at=datetime.now().isoformat(),
        source=source,
        version=DNA_VERSION,
        personality=personality,
        tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Similarity
# ══════════════════════════════════════════════════════════════════════════════

_SIGNAL_RANK = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
_WEIGHTS = {"statistical": 0.35, "structural": 0.30, "ml": 0.25, "temporal": 0.10}


def _resample_sorted(values, n: int = 16) -> np.ndarray:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return np.zeros(n)
    if arr.size == 1:
        return np.full(n, arr[0])
    return np.interp(np.linspace(0, arr.size - 1, n), np.arange(arr.size), arr)


def _vector_similarity(a, b, aligned: bool) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 and b.size == 0:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 0.0
    if not (aligned and a.size == b.size):
        a, b = _resample_sorted(a), _resample_sorted(b)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    return float(np.clip(1.0 - np.mean(np.abs(a - b) / denom), 0.0, 1.0))


class DNASimilarityEngine:
    """Compares two DataDNA objects and returns a SimilarityReport."""

    def _statistical(self, a: DataDNA, b: DataDNA, aligned: bool) -> float:
        sa, sb = a.statistical, b.statistical
        comps: List[float] = [_vector_similarity(sa.entropy_vector, sb.entropy_vector, aligned)]

        ma, mb = np.array(sa.moment_signatures, dtype=float), np.array(sb.moment_signatures, dtype=float)
        if ma.size and mb.size:
            if aligned and ma.shape == mb.shape:
                per_col = []
                for r1, r2 in zip(ma, mb):
                    d = abs(r1[0] - r2[0]) / (max(r1[1], r2[1]) + 1e-9)
                    per_col.append(
                        (math.exp(-d) + _ratio(r1[1], r2[1]) + _close(r1[2], r2[2], 2.0)) / 3.0
                    )
                comps.append(float(np.mean(per_col)))
            else:
                comps.append(
                    (_close(ma[:, 2].mean(), mb[:, 2].mean(), 2.0)
                     + _close(ma[:, 3].mean(), mb[:, 3].mean(), 6.0)) / 2.0
                )
        elif ma.size != mb.size:
            comps.append(0.0)

        if sa.correlation_hash == sb.correlation_hash and sa.correlation_hash != "none":
            comps.append(1.0)
        elif sa.mutual_info_matrix and sb.mutual_info_matrix:
            def off_diag(m):
                arr = np.array(m, dtype=float)
                return float(arr[np.triu_indices(len(arr), k=1)].mean())
            comps.append(_close(off_diag(sa.mutual_info_matrix), off_diag(sb.mutual_info_matrix), 0.5))

        ca, cb = Counter(sa.distribution_types), Counter(sb.distribution_types)
        if ca or cb:
            keys = set(ca) | set(cb)
            na, nb = max(1, sum(ca.values())), max(1, sum(cb.values()))
            comps.append(1.0 - 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys))
        return float(np.mean(comps))

    def _structural(self, a: DataDNA, b: DataDNA, aligned: bool) -> float:
        ta, tb = a.structural, b.structural
        rows_a, cols_a = ta.shape_signature
        rows_b, cols_b = tb.shape_signature
        schema = 1.0 if aligned else 0.5 * _ratio(cols_a, cols_b)
        card = _vector_similarity(
            np.log1p(ta.cardinality_profile), np.log1p(tb.cardinality_profile), aligned
        )
        mech = 1.0 if ta.missing_mechanism == tb.missing_mechanism else 0.4
        comps = [
            schema,
            math.sqrt(_ratio(rows_a, rows_b)),
            mech,
            _close(ta.outlier_density, tb.outlier_density, 0.1),
            _close(ta.duplicate_ratio, tb.duplicate_ratio, 0.1),
            card,
        ]
        return float(np.mean(comps))

    def _ml(self, a: DataDNA, b: DataDNA) -> float:
        ma, mb = a.ml, b.ml
        comps = [
            1.0 if ma.recommended_task == mb.recommended_task else 0.0,
            1.0 - abs(_SIGNAL_RANK[ma.target_signal] - _SIGNAL_RANK[mb.target_signal]) / 3.0,
            _close(ma.separability_score, mb.separability_score, 0.5),
            _close(ma.class_balance_score, mb.class_balance_score, 0.5),
            _close(ma.feature_redundancy, mb.feature_redundancy, 0.5),
            _close(ma.complexity_score, mb.complexity_score, 0.5),
        ]
        return float(np.mean(comps))

    def _temporal(self, a: DataDNA, b: DataDNA) -> float:
        ta, tb = a.temporal, b.temporal
        if not ta.has_temporal and not tb.has_temporal:
            return 1.0
        if ta.has_temporal != tb.has_temporal:
            return 0.0
        comps = [
            1.0,
            1.0 if ta.trend_detected == tb.trend_detected else 0.0,
            _close(ta.seasonality_score, tb.seasonality_score, 0.5),
            1.0 if ta.time_gaps_uniform == tb.time_gaps_uniform else 0.0,
        ]
        return float(np.mean(comps))

    def compare(self, dna1: DataDNA, dna2: DataDNA) -> SimilarityReport:
        if dna1.dna_hash == dna2.dna_hash:
            return SimilarityReport(
                1.0,
                {k: 1.0 for k in _WEIGHTS},
                [],
                "Datasets are identical",
            )
        aligned = dna1.structural.schema_hash == dna2.structural.schema_hash
        scores = {
            "statistical": self._statistical(dna1, dna2, aligned),
            "structural": self._structural(dna1, dna2, aligned),
            "ml": self._ml(dna1, dna2),
            "temporal": self._temporal(dna1, dna2),
        }
        overall = sum(scores[k] * w for k, w in _WEIGHTS.items())

        s1, s2 = dna1.structural, dna2.structural
        diffs: List[str] = []
        if not aligned:
            diffs.append("Schema has changed")
        r1, r2 = s1.shape_signature[0], s2.shape_signature[0]
        if _ratio(r1, r2) < 0.8:
            diffs.append(f"Row count changed: {r1} -> {r2}")
        if dna1.ml.recommended_task != dna2.ml.recommended_task:
            diffs.append(f"Task mismatch: {dna1.ml.recommended_task} vs {dna2.ml.recommended_task}")
        if dna1.ml.target_signal != dna2.ml.target_signal:
            diffs.append(f"Target signal changed: {dna1.ml.target_signal} -> {dna2.ml.target_signal}")
        if s1.missing_mechanism != s2.missing_mechanism:
            diffs.append(f"Missing mechanism changed: {s1.missing_mechanism} -> {s2.missing_mechanism}")
        if abs(s1.outlier_density - s2.outlier_density) > 0.05:
            diffs.append(f"Outlier density shifted: {s1.outlier_density:.1%} -> {s2.outlier_density:.1%}")
        if abs(s1.duplicate_ratio - s2.duplicate_ratio) > 0.05:
            diffs.append(f"Duplicate ratio shifted: {s1.duplicate_ratio:.1%} -> {s2.duplicate_ratio:.1%}")
        if dna1.temporal.has_temporal != dna2.temporal.has_temporal:
            diffs.append("Temporal structure changed")
        if scores["statistical"] < 0.6:
            diffs.append(f"Statistical profile differs strongly (similarity {scores['statistical']:.2f})")

        if overall >= 0.95:
            rec = "Datasets are nearly identical"
        elif overall >= 0.8:
            rec = "Datasets are highly similar"
        elif overall >= 0.6:
            rec = "Datasets are related but show drift"
        else:
            rec = "Datasets have drifted significantly"
        return SimilarityReport(
            overall_similarity=round(float(overall), 4),
            dimension_scores={k: round(float(v), 4) for k, v in scores.items()},
            key_differences=diffs,
            recommendation=rec,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI-compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveDNA:
    """
    Wrapper around ``compute_dna`` that exposes the attributes the CLI reads.

    ``self.data_dna`` is a real ``DataDNA`` and can be passed to ``DNAStore``.
    """

    def __init__(
        self, df: pd.DataFrame, target_col: Optional[str] = None, source: str = "dataframe"
    ):
        self.target_col = target_col
        self.data_dna = compute_dna(df, target_col=target_col, source=source)
        d = self.data_dna
        self.statistical = d.statistical
        self.structural = d.structural
        self.ml = d.ml
        self.temporal = d.temporal
        self.dna_hash = d.dna_hash
        self.created_at = d.created_at
        self.source = d.source
        self.personality = d.personality
        self.tags = d.tags

    @property
    def short_hash(self) -> str:
        return self.dna_hash[:8]

    def similarity(self, other: "CognitiveDNA | DataDNA") -> SimilarityReport:
        other_dna = other.data_dna if isinstance(other, CognitiveDNA) else other
        return DNASimilarityEngine().compare(self.data_dna, other_dna)

    def to_dict(self) -> Dict[str, Any]:
        return self.data_dna.to_dict()