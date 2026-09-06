"""
nydra_core — Python Fallback Bridge for Nydra Core
==================================================

This file acts as a seamless wrapper around the high-performance Rust core (`nydra_core` compiled binary).
If the Rust binary is not compiled or the compiler toolchain is missing, this file automatically
falls back to pure-Python/NumPy/SciPy implementations of the exact same statistical functions.

This ensures the application is 100% stable and runnable out-of-the-box on any machine,
while unlocking 10x performance gains automatically if the Rust extension is present.
"""

import math
import numpy as np
import scipy.stats as stats_scipy

# Try to import the compiled Rust extension
try:
    import _nydra_core_binary as rust_core
    HAS_RUST = True
except ImportError:
    rust_core = None
    HAS_RUST = False

# Export status for informational/telemetry purposes
__has_rust__ = HAS_RUST
__version__  = "0.1.0"
__doc__      = "nydra_core — Hybrid Rust/Python statistical compute engine for Nydra."

# =============================================================================
# INTERNAL PURE-PYTHON IMPLEMENTATIONS (FALLBACKS)
# =============================================================================

def _py_std_dev(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    return float(np.std(data))

def _py_std_dev_sample(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    if len(data) < 2: return 0.0
    return float(np.std(data, ddof=1))

def _py_mean(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    return float(np.mean(data))

def _py_median(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    return float(np.median(data))

def _py_variance(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    return float(np.var(data))

def _py_variance_sample(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    if len(data) < 2: return 0.0
    return float(np.var(data, ddof=1))

def _py_skewness(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    if float(np.std(data)) == 0: raise ValueError("Standard deviation is zero")
    return float(stats_scipy.skew(data, bias=False))

def _py_excess_kurtosis(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    if float(np.std(data)) == 0: raise ValueError("Standard deviation is zero")
    return float(stats_scipy.kurtosis(data, bias=False))

def _py_mad(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    med = np.median(data)
    return float(np.median(np.abs(data - med)))

def _py_mad_scaled(data):
    return _py_mad(data) * 1.4826

def _py_percentile(data, p):
    if len(data) == 0: raise ValueError("Input slice is empty")
    if not (0.0 <= p <= 100.0): raise ValueError(f"Percentile {p} out of range")
    return float(np.percentile(data, p, method='linear'))

def _py_quartiles(data):
    return (_py_percentile(data, 25.0), _py_percentile(data, 75.0))

def _py_kahan_sum(data):
    return float(math.fsum(data))

def _py_filter_nan(data):
    arr = np.array(data, dtype=np.float64)
    clean = arr[np.isfinite(arr)]
    removed = len(data) - len(clean)
    return (clean.tolist(), removed)

def _py_shannon_entropy(counts):
    if len(counts) == 0 or sum(counts) == 0: raise ValueError("Input slice is empty")
    total = sum(counts)
    probs = [c / total for c in counts if c > 0]
    entropy_bits = float(-sum(p * math.log2(p) for p in probs))
    entropy_nats = entropy_bits * math.log(2)
    n_unique = len(counts)
    max_bits = math.log2(n_unique) if n_unique > 1 else 1.0
    normalized = entropy_bits / max_bits if max_bits > 0 else 1.0
    return {
        "entropy_bits": entropy_bits, "entropy_nats": entropy_nats,
        "n_unique": n_unique, "n_total": int(total), "normalized": normalized
    }

def _py_entropy_from_strings(values):
    if len(values) == 0: raise ValueError("Input slice is empty")
    from collections import Counter
    counts = list(Counter(values).values())
    return _py_shannon_entropy(counts)

def _py_entropy_numeric(data, n_bins=10):
    if len(data) == 0: raise ValueError("Input slice is empty")
    if n_bins < 1: raise ValueError("Bins count must be >= 1")
    counts, _ = np.histogram(data, bins=n_bins)
    return _py_shannon_entropy(counts.tolist())

def _py_pearson_correlation(x, y):
    if len(x) != len(y): raise ValueError("Both slices must have the same length")
    if len(x) == 0: raise ValueError("Input slice is empty")
    r, p = stats_scipy.pearsonr(x, y)
    if np.isnan(r): raise ValueError("Standard deviation is zero")
    return {"method": "pearson", "coefficient": float(r), "p_value": float(p), "is_significant": bool(p < 0.05), "n": len(x)}

def _py_spearman_correlation(x, y):
    if len(x) != len(y): raise ValueError("Both slices must have the same length")
    if len(x) == 0: raise ValueError("Input slice is empty")
    r, p = stats_scipy.spearmanr(x, y)
    if np.isnan(r): raise ValueError("Standard deviation is zero")
    return {"method": "spearman", "coefficient": float(r), "p_value": float(p), "is_significant": bool(p < 0.05), "n": len(x)}

def _py_correlation_matrix(columns, col_names, method="pearson"):
    if len(columns) != len(col_names): raise ValueError("columns and col_names must have the same length")
    k = len(columns)
    matrix = [[1.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            if method == "spearman":
                res = _py_spearman_correlation(columns[i], columns[j])
            else:
                res = _py_pearson_correlation(columns[i], columns[j])
            matrix[i][j] = res["coefficient"]
            matrix[j][i] = res["coefficient"]
    return {"column_names": col_names, "matrix": matrix, "method": method}

def _py_full_descriptive(data):
    if len(data) == 0: raise ValueError("Input slice is empty")
    clean, removed = _py_filter_nan(data)
    if len(clean) == 0: raise ValueError("Input slice is empty")
    
    q1, q3 = _py_quartiles(clean)
    mean_val = _py_mean(clean)
    std_val = _py_std_dev(clean)
    
    # Safely compute skew/kurtosis
    try:
        skew = _py_skewness(clean)
        kurt = _py_excess_kurtosis(clean)
    except ValueError:
        skew = float('nan')
        kurt = float('nan')
        
    cv = (std_val / abs(mean_val)) * 100.0 if mean_val != 0 else float('nan')
    min_val = float(np.min(clean))
    max_val = float(np.max(clean))

    return {
        "count": len(clean), "mean": mean_val, "median": _py_median(clean),
        "std_dev": std_val, "std_dev_sample": _py_std_dev_sample(clean),
        "variance": _py_variance(clean), "variance_sample": _py_variance_sample(clean),
        "min": min_val, "max": max_val, "range": max_val - min_val,
        "q1": q1, "q3": q3, "iqr": q3 - q1, "mad": _py_mad(clean),
        "skewness": skew, "kurtosis": kurt, "cv": cv, "sum": _py_kahan_sum(clean),
        "nan_count": removed, "infinite_count": 0
    }

def _py_detect_outliers_iqr(data, k=1.5):
    if len(data) == 0: raise ValueError("Input slice is empty")
    q1, q3 = _py_quartiles(data)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    indices = [i for i, x in enumerate(data) if x < lower or x > upper]
    return {
        "method": f"IQR (k={k})", "outlier_indices": indices, "outlier_count": len(indices),
        "outlier_fraction": len(indices) / len(data), "lower_fence": float(lower), "upper_fence": float(upper)
    }

def _py_detect_outliers_mad(data, threshold=3.5):
    if len(data) == 0: raise ValueError("Input slice is empty")
    med = _py_median(data)
    mad_val = _py_mad(data)
    if mad_val == 0:
        # MAD is 0 when >50% of values are identical (majority-tie data).
        # Modified Z-score can't detect outliers in this case, so fall back
        # to IQR, which isn't fooled by majority-tie distributions.
        fallback = _py_detect_outliers_iqr(data, k=1.5)
        fallback["method"] = f"IQR (k=1.5) — MAD fallback, majority-tie data detected"
        return fallback

    scale = 0.6745 / mad_val

    scale = 0.6745 / mad_val
    lower = med - threshold / scale
    upper = med + threshold / scale
    indices = []
    for i, x in enumerate(data):
        if (0.6745 * (x - med) / mad_val) > threshold:
            indices.append(i)
    return {
        "method": f"Modified Z-Score (MAD, threshold={threshold})", "outlier_indices": indices, "outlier_count": len(indices),
        "outlier_fraction": len(indices) / len(data), "lower_fence": float(lower), "upper_fence": float(upper)
    }

def _py_rolling_mean(data, window):
    if len(data) < window: raise ValueError("Window size is larger than data length")
    return [float(x) for x in np.convolve(data, np.ones(window)/window, mode='valid')]

def _py_rolling_std(data, window):
    if len(data) < window: raise ValueError("Window size is larger than data length")
    res = []
    for i in range(len(data) - window + 1):
        res.append(_py_std_dev_sample(data[i:i+window]))
    return res

# =============================================================================
# PUBLIC EXPOSED CORE INTERFACE (HYBRID ROUTING)
# =============================================================================

def std_dev(data):
    return rust_core.std_dev(data) if HAS_RUST else _py_std_dev(data)

def std_dev_sample(data):
    return rust_core.std_dev_sample(data) if HAS_RUST else _py_std_dev_sample(data)

def mean(data):
    return rust_core.mean(data) if HAS_RUST else _py_mean(data)

def median(data):
    return rust_core.median(data) if HAS_RUST else _py_median(data)

def variance(data):
    return rust_core.variance(data) if HAS_RUST else _py_variance(data)

def variance_sample(data):
    return rust_core.variance_sample(data) if HAS_RUST else _py_variance_sample(data)

def skewness(data):
    return rust_core.skewness(data) if HAS_RUST else _py_skewness(data)

def excess_kurtosis(data):
    return rust_core.excess_kurtosis(data) if HAS_RUST else _py_excess_kurtosis(data)

def mad(data):
    return rust_core.mad(data) if HAS_RUST else _py_mad(data)

def mad_scaled(data):
    return rust_core.mad_scaled(data) if HAS_RUST else _py_mad_scaled(data)

def percentile(data, p):
    return rust_core.percentile(data, p) if HAS_RUST else _py_percentile(data, p)

def quartiles(data):
    return rust_core.quartiles(data) if HAS_RUST else _py_quartiles(data)

def kahan_sum(data):
    return rust_core.kahan_sum(data) if HAS_RUST else _py_kahan_sum(data)

def filter_nan(data):
    return rust_core.filter_nan(data) if HAS_RUST else _py_filter_nan(data)

def full_descriptive(data):
    return rust_core.full_descriptive(data) if HAS_RUST else _py_full_descriptive(data)

def detect_outliers_iqr(data, k=1.5):
    return rust_core.detect_outliers_iqr(data, k) if HAS_RUST else _py_detect_outliers_iqr(data, k)

def detect_outliers_mad(data, threshold=3.5):
    return rust_core.detect_outliers_mad(data, threshold) if HAS_RUST else _py_detect_outliers_mad(data, threshold)

def shannon_entropy(counts):
    return rust_core.shannon_entropy(counts) if HAS_RUST else _py_shannon_entropy(counts)

def entropy_from_strings(values):
    return rust_core.entropy_from_strings(values) if HAS_RUST else _py_entropy_from_strings(values)

def entropy_numeric(data, n_bins=10):
    return rust_core.entropy_numeric(data, n_bins) if HAS_RUST else _py_entropy_numeric(data, n_bins)

def pearson_correlation(x, y):
    return rust_core.pearson_correlation(x, y) if HAS_RUST else _py_pearson_correlation(x, y)

def spearman_correlation(x, y):
    return rust_core.spearman_correlation(x, y) if HAS_RUST else _py_spearman_correlation(x, y)

def correlation_matrix(columns, col_names, method="pearson"):
    return rust_core.correlation_matrix(columns, col_names, method) if HAS_RUST else _py_correlation_matrix(columns, col_names, method)

def rolling_mean(data, window):
    return rust_core.rolling_mean(data, window) if HAS_RUST else _py_rolling_mean(data, window)

def rolling_std(data, window):
    return rust_core.rolling_std(data, window) if HAS_RUST else _py_rolling_std(data, window)
