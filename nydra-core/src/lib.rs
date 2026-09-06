// =============================================================================
// src/lib.rs — Nydra Core: Python ↔ Rust Bridge
// =============================================================================
//
// PURPOSE:
//   This is the "bridge" between Python and Rust. It has ONE job:
//     1. Accept Python values (lists, dicts, strings)
//     2. Convert them to Rust types
//     3. Call the real computation in stats.rs
//     4. Convert the result back to Python types
//     5. Return to Python
//
// WHAT LIVES HERE vs stats.rs:
//   lib.rs    — Type conversion, error formatting, Python module registration
//   stats.rs  — Pure numerical algorithms (knows nothing about Python)
//
// HOW PyO3 WORKS:
//   #[pyfunction]  — Makes a Rust function callable from Python
//   #[pymodule]   — Defines the top-level `import nydra_core` module
//   Python<'_>    — Token that proves we hold the GIL (required by PyO3)
//   PyResult<T>   — Equivalent to Python's `T | raise Exception`
//
// ERROR HANDLING POLICY:
//   All `StatsError` variants are converted to Python's `ValueError`
//   with a human-readable message. Python callers use standard try/except.
//
//   Python example:
//     try:
//         result = nydra_core.std_dev([1.0, 2.0, float('nan')])
//     except ValueError as e:
//         print(f"Computation failed: {e}")
//
// THREAD SAFETY:
//   PyO3 automatically releases the GIL around `allow_threads` calls.
//   Rayon runs its thread pool outside the GIL, so our parallel functions
//   don't block other Python threads while computing.
//
// =============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// Import our computational module
mod stats;

use stats::{
    StatsError,
    DescriptiveStats,
    OutlierResult,
    CorrelationResult,
    CorrelationMatrix,
    EntropyResult,
};

// =============================================================================
// ERROR CONVERSION: StatsError → Python ValueError
// =============================================================================

/// Convert any `StatsError` into a Python `ValueError` with a clear message.
/// This is called automatically by the `?` operator inside `#[pyfunction]` bodies.
impl From<StatsError> for PyErr {
    fn from(e: StatsError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

// =============================================================================
// HELPER: EXTRACT & VALIDATE PYTHON LIST → Vec<f64>
// =============================================================================

/// Extract a Python list of numbers into a `Vec<f64>`.
///
/// Accepts: int, float, bool (True=1.0, False=0.0)
/// Rejects: strings, None, other non-numeric types → ValueError
fn extract_f64_vec(py: Python<'_>, list: &PyAny) -> PyResult<Vec<f64>> {
    list.extract::<Vec<f64>>().map_err(|_| {
        PyValueError::new_err(
            "Expected a list of numbers (int or float). \
             Got non-numeric values. Remove None/NaN before calling."
        )
    })
}

// =============================================================================
// MODULE DEFINITION
// =============================================================================

/// `nydra_core` — High-performance statistical functions for Nydra.
///
/// This module is compiled from Rust and loaded by Python as a native extension.
/// All functions in this module run outside the GIL using Rayon's thread pool,
/// providing linear speedups proportional to available CPU cores.
///
/// Quick start:
///     import nydra_core
///
///     data = [1.0, 2.0, 3.0, 4.0, 5.0]
///
///     # Basic stats
///     print(nydra_core.std_dev(data))                 # 1.4142...
///     print(nydra_core.mean(data))                    # 3.0
///
///     # Full report (returns a dict)
///     stats = nydra_core.full_descriptive(data)
///     print(stats["mean"], stats["skewness"])
///
///     # Outlier detection
///     outliers = nydra_core.detect_outliers_iqr(data)
///     print(outliers["outlier_indices"])
///
///     # Correlation
///     corr = nydra_core.pearson_correlation(data, data)
///     print(corr["coefficient"])                      # 1.0
///
#[pymodule]
fn _nydra_core_binary(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // Register all exposed functions in the module
    m.add_function(wrap_pyfunction!(py_std_dev,                 m)?)?;
    m.add_function(wrap_pyfunction!(py_std_dev_sample,          m)?)?;
    m.add_function(wrap_pyfunction!(py_mean,                    m)?)?;
    m.add_function(wrap_pyfunction!(py_median,                  m)?)?;
    m.add_function(wrap_pyfunction!(py_variance,                m)?)?;
    m.add_function(wrap_pyfunction!(py_variance_sample,         m)?)?;
    m.add_function(wrap_pyfunction!(py_skewness,                m)?)?;
    m.add_function(wrap_pyfunction!(py_excess_kurtosis,         m)?)?;
    m.add_function(wrap_pyfunction!(py_mad,                     m)?)?;
    m.add_function(wrap_pyfunction!(py_mad_scaled,              m)?)?;
    m.add_function(wrap_pyfunction!(py_percentile,              m)?)?;
    m.add_function(wrap_pyfunction!(py_quartiles,               m)?)?;
    m.add_function(wrap_pyfunction!(py_kahan_sum,               m)?)?;
    m.add_function(wrap_pyfunction!(py_filter_nan,              m)?)?;
    m.add_function(wrap_pyfunction!(py_full_descriptive,        m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_outliers_iqr,     m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_outliers_mad,     m)?)?;
    m.add_function(wrap_pyfunction!(py_shannon_entropy,         m)?)?;
    m.add_function(wrap_pyfunction!(py_entropy_numeric,         m)?)?;
    m.add_function(wrap_pyfunction!(py_entropy_from_strings,    m)?)?;
    m.add_function(wrap_pyfunction!(py_pearson_correlation,     m)?)?;
    m.add_function(wrap_pyfunction!(py_spearman_correlation,    m)?)?;
    m.add_function(wrap_pyfunction!(py_correlation_matrix,      m)?)?;
    m.add_function(wrap_pyfunction!(py_rolling_mean,            m)?)?;
    m.add_function(wrap_pyfunction!(py_rolling_std,             m)?)?;

    // Version metadata (accessible as nydra_core.__version__)
    m.add("__version__", "0.1.0")?;
    m.add("__author__",  "Nydra Contributors")?;
    m.add("__doc__",
        "nydra_core — High-performance Rust statistical engine for Nydra.\n\
         Built with PyO3 + Rayon. Provides parallel, numerically stable \n\
         implementations of common statistical functions."
    )?;

    Ok(())
}

// =============================================================================
// SECTION 1: BASIC SCALAR STATISTICS
// =============================================================================

/// Compute population standard deviation (denominator = n).
///
/// Args:
///     data: List of numeric values (int or float). Must be non-empty and finite.
///
/// Returns:
///     float: Population standard deviation.
///
/// Raises:
///     ValueError: If data is empty, contains NaN, or contains Inf.
///
/// Example:
///     >>> import nydra_core
///     >>> nydra_core.std_dev([2, 4, 4, 4, 5, 5, 7, 9])
///     2.0
#[pyfunction]
#[pyo3(name = "std_dev", signature = (data))]
fn py_std_dev(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    // allow_threads: release the GIL while Rayon runs in parallel
    py.allow_threads(|| stats::std_dev(&vec).map_err(PyErr::from))
}

/// Compute sample standard deviation (denominator = n-1, unbiased estimator).
///
/// Use this when `data` is a sample from a larger population.
/// Use `std_dev` when `data` is the entire population.
///
/// Example:
///     >>> nydra_core.std_dev_sample([2, 4, 4, 4, 5, 5, 7, 9])
///     2.138...
#[pyfunction]
#[pyo3(name = "std_dev_sample", signature = (data))]
fn py_std_dev_sample(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::std_dev_sample(&vec).map_err(PyErr::from))
}

/// Compute the arithmetic mean using Kahan compensated summation.
///
/// More accurate than Python's sum(data)/len(data) for large datasets.
///
/// Example:
///     >>> nydra_core.mean([1, 2, 3, 4, 5])
///     3.0
#[pyfunction]
#[pyo3(name = "mean", signature = (data))]
fn py_mean(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    if vec.is_empty() {
        return Err(PyValueError::new_err("Cannot compute mean of empty list"));
    }
    py.allow_threads(|| {
        stats::validate_finite(&vec).map_err(PyErr::from)?;
        Ok(stats::parallel_mean(&vec))
    })
}

/// Compute the median (50th percentile) using linear interpolation.
///
/// Matches numpy's default behavior: np.percentile(data, 50, interpolation='linear').
///
/// Example:
///     >>> nydra_core.median([1, 2, 3, 4, 5])
///     3.0
///     >>> nydra_core.median([1, 2, 3, 4])
///     2.5
#[pyfunction]
#[pyo3(name = "median", signature = (data))]
fn py_median(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::median(&vec).map_err(PyErr::from))
}

/// Compute population variance (denominator = n).
///
/// Uses Welford's single-pass algorithm for numerical stability.
///
/// Example:
///     >>> nydra_core.variance([2, 4, 4, 4, 5, 5, 7, 9])
///     4.0
#[pyfunction]
#[pyo3(name = "variance", signature = (data))]
fn py_variance(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| {
        let (_, pop_var, _, _) = stats::welford_parallel(&vec).map_err(PyErr::from)?;
        Ok(pop_var)
    })
}

/// Compute sample variance (denominator = n-1).
///
/// Example:
///     >>> nydra_core.variance_sample([2, 4, 4, 4, 5, 5, 7, 9])
///     4.571...
#[pyfunction]
#[pyo3(name = "variance_sample", signature = (data))]
fn py_variance_sample(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| {
        let (_, _, sample_var, _) = stats::welford_parallel(&vec).map_err(PyErr::from)?;
        Ok(sample_var)
    })
}

/// Compute Fisher's skewness (g1) — the standardized third central moment.
///
/// > 0 → right-skewed (long tail on the right)
/// < 0 → left-skewed  (long tail on the left)
/// ≈ 0 → approximately symmetric
///
/// Example:
///     >>> nydra_core.skewness([1, 2, 2, 3, 3, 3, 4, 4, 5])
///     0.0  # symmetric
#[pyfunction]
#[pyo3(name = "skewness", signature = (data))]
fn py_skewness(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::skewness(&vec).map_err(PyErr::from))
}

/// Compute excess kurtosis (g2) — standardized fourth central moment minus 3.
///
/// > 0 → heavy tails (leptokurtic — e.g., financial returns)
/// < 0 → light tails  (platykurtic — e.g., uniform distribution)
/// ≈ 0 → normal-like tail behavior
///
/// Example:
///     >>> nydra_core.excess_kurtosis([1, 2, 3, 4, 5])
///     -1.3
#[pyfunction]
#[pyo3(name = "excess_kurtosis", signature = (data))]
fn py_excess_kurtosis(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::excess_kurtosis(&vec).map_err(PyErr::from))
}

/// Compute Median Absolute Deviation (MAD) — robust alternative to std dev.
///
/// MAD = median(|xi - median(x)|)
///
/// Not affected by extreme outliers. For normally distributed data:
/// std ≈ 1.4826 * MAD.
///
/// Example:
///     >>> nydra_core.mad([1, 2, 3, 4, 5])
///     1.0
#[pyfunction]
#[pyo3(name = "mad", signature = (data))]
fn py_mad(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::mad(&vec).map_err(PyErr::from))
}

/// Compute MAD scaled to be a consistent estimator of std dev.
///
/// Returns MAD * 1.4826. This makes it directly comparable to std_dev
/// for normally distributed data.
///
/// Example:
///     >>> nydra_core.mad_scaled([1, 2, 3, 4, 5])
///     1.4826
#[pyfunction]
#[pyo3(name = "mad_scaled", signature = (data))]
fn py_mad_scaled(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::mad_scaled(&vec).map_err(PyErr::from))
}

/// Compute a percentile using linear interpolation (matches numpy's default).
///
/// Args:
///     data:       List of numeric values.
///     percentile: Value in [0.0, 100.0].
///
/// Returns:
///     float: The interpolated percentile value.
///
/// Example:
///     >>> nydra_core.percentile([1, 2, 3, 4, 5], 25.0)   # Q1
///     2.0
///     >>> nydra_core.percentile([1, 2, 3, 4, 5], 75.0)   # Q3
///     4.0
#[pyfunction]
#[pyo3(name = "percentile", signature = (data, p))]
fn py_percentile(py: Python<'_>, data: &PyAny, p: f64) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| {
        let mut buf = vec.clone();
        stats::percentile(&mut buf, p).map_err(PyErr::from)
    })
}

/// Compute Q1 (25th) and Q3 (75th) percentiles simultaneously.
///
/// Returns:
///     tuple[float, float]: (Q1, Q3)
///
/// Example:
///     >>> q1, q3 = nydra_core.quartiles([1, 2, 3, 4, 5])
///     >>> print(q1, q3)   # 2.0 4.0
#[pyfunction]
#[pyo3(name = "quartiles", signature = (data))]
fn py_quartiles(py: Python<'_>, data: &PyAny) -> PyResult<(f64, f64)> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::quartiles(&vec).map_err(PyErr::from))
}

/// Compute Kahan compensated sum (more accurate than Python's built-in sum).
///
/// Use when summing large lists of floating-point values.
///
/// Example:
///     >>> nydra_core.kahan_sum([1.0] * 1_000_000)
///     1000000.0  # exact, not 999999.99978...
#[pyfunction]
#[pyo3(name = "kahan_sum", signature = (data))]
fn py_kahan_sum(py: Python<'_>, data: &PyAny) -> PyResult<f64> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| Ok(stats::kahan_sum(&vec)))
}

/// Filter out NaN and Infinite values from a list.
///
/// Returns:
///     tuple[list[float], int]: (clean_data, removed_count)
///
/// Example:
///     >>> clean, n_removed = nydra_core.filter_nan([1, float('nan'), 3, float('inf'), 5])
///     >>> print(clean, n_removed)   # [1.0, 3.0, 5.0]  2
#[pyfunction]
#[pyo3(name = "filter_nan", signature = (data))]
fn py_filter_nan(py: Python<'_>, data: &PyAny) -> PyResult<(Vec<f64>, usize)> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| Ok(stats::filter_nan(&vec)))
}

// =============================================================================
// SECTION 2: FULL DESCRIPTIVE STATISTICS REPORT
// =============================================================================

/// Compute all descriptive statistics for a numeric column.
///
/// NaN and Infinite values are automatically excluded (counted in the output).
/// All statistics are computed in a minimal number of parallel passes.
///
/// Args:
///     data: List of numeric values (int or float). May contain NaN/Inf.
///
/// Returns:
///     dict with keys:
///       count, mean, median, std_dev, std_dev_sample, variance, variance_sample,
///       min, max, range, q1, q3, iqr, mad, skewness, kurtosis, cv, sum,
///       nan_count, infinite_count
///
/// Example:
///     >>> import nydra_core
///     >>> stats = nydra_core.full_descriptive([2, 4, 4, 4, 5, 5, 7, 9])
///     >>> print(stats["mean"], stats["std_dev"])   # 5.0  2.0
///     >>> print(stats["skewness"])                 # 0.0 (symmetric)
#[pyfunction]
#[pyo3(name = "full_descriptive", signature = (data))]
fn py_full_descriptive(py: Python<'_>, data: &PyAny) -> PyResult<PyObject> {
    let vec = extract_f64_vec(py, data)?;

    let result: DescriptiveStats = py.allow_threads(|| {
        stats::full_descriptive(&vec).map_err(PyErr::from)
    })?;

    // Build Python dict from the serialized result
    descriptive_to_py(py, &result)
}

/// Convert a DescriptiveStats struct into a Python dict.
fn descriptive_to_py(py: Python<'_>, s: &DescriptiveStats) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("count",           s.count)?;
    dict.set_item("mean",            s.mean)?;
    dict.set_item("median",          s.median)?;
    dict.set_item("std_dev",         s.std_dev)?;
    dict.set_item("std_dev_sample",  s.std_dev_sample)?;
    dict.set_item("variance",        s.variance)?;
    dict.set_item("variance_sample", s.variance_sample)?;
    dict.set_item("min",             s.min)?;
    dict.set_item("max",             s.max)?;
    dict.set_item("range",           s.range)?;
    dict.set_item("q1",              s.q1)?;
    dict.set_item("q3",              s.q3)?;
    dict.set_item("iqr",             s.iqr)?;
    dict.set_item("mad",             s.mad)?;
    dict.set_item("skewness",        s.skewness)?;
    dict.set_item("kurtosis",        s.kurtosis)?;
    dict.set_item("cv",              s.cv)?;
    dict.set_item("sum",             s.sum)?;
    dict.set_item("nan_count",       s.nan_count)?;
    dict.set_item("infinite_count",  s.infinite_count)?;
    Ok(dict.into())
}

// =============================================================================
// SECTION 3: OUTLIER DETECTION
// =============================================================================

/// Detect outliers using Tukey's IQR method.
///
/// Fences: lower = Q1 - k * IQR,  upper = Q3 + k * IQR
///
/// Args:
///     data: List of numeric values. Must be non-empty and finite.
///     k:    Fence multiplier (default 1.5 = Tukey's rule, use 3.0 for "far" outliers).
///
/// Returns:
///     dict with keys:
///       method, outlier_indices (list[int]), outlier_count (int),
///       outlier_fraction (float), lower_fence (float), upper_fence (float)
///
/// Example:
///     >>> result = nydra_core.detect_outliers_iqr([1, 2, 3, 4, 100], k=1.5)
///     >>> print(result["outlier_indices"])   # [4]
///     >>> print(result["outlier_count"])     # 1
#[pyfunction]
#[pyo3(name = "detect_outliers_iqr", signature = (data, k = 1.5))]
fn py_detect_outliers_iqr(py: Python<'_>, data: &PyAny, k: f64) -> PyResult<PyObject> {
    let vec = extract_f64_vec(py, data)?;
    let result: OutlierResult = py.allow_threads(|| {
        stats::detect_outliers_iqr(&vec, k).map_err(PyErr::from)
    })?;
    outlier_to_py(py, &result)
}

/// Detect outliers using the Modified Z-Score method (Iglewicz & Hoaglin, 1993).
///
/// Based on Median Absolute Deviation — more robust than standard Z-score
/// because the median and MAD are not influenced by extreme outliers.
///
/// Modified Z-score: Mi = 0.6745 * (xi - median) / MAD
///
/// Args:
///     data:      List of numeric values. Must be non-empty and finite.
///     threshold: Values with |Mi| > threshold are outliers (default 3.5).
///
/// Returns:
///     dict with same structure as detect_outliers_iqr.
///
/// Example:
///     >>> result = nydra_core.detect_outliers_mad([1, 2, 3, 4, 1000], threshold=3.5)
///     >>> print(result["outlier_indices"])   # [4]
#[pyfunction]
#[pyo3(name = "detect_outliers_mad", signature = (data, threshold = 3.5))]
fn py_detect_outliers_mad(py: Python<'_>, data: &PyAny, threshold: f64) -> PyResult<PyObject> {
    let vec = extract_f64_vec(py, data)?;
    let result: OutlierResult = py.allow_threads(|| {
        stats::detect_outliers_mad(&vec, threshold).map_err(PyErr::from)
    })?;
    outlier_to_py(py, &result)
}

/// Convert an OutlierResult struct into a Python dict.
fn outlier_to_py(py: Python<'_>, r: &OutlierResult) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("method",           &r.method)?;
    dict.set_item("outlier_indices",  r.outlier_indices.clone())?;
    dict.set_item("outlier_count",    r.outlier_count)?;
    dict.set_item("outlier_fraction", r.outlier_fraction)?;
    dict.set_item("lower_fence",      r.lower_fence)?;
    dict.set_item("upper_fence",      r.upper_fence)?;
    Ok(dict.into())
}

// =============================================================================
// SECTION 4: ENTROPY
// =============================================================================

/// Compute Shannon entropy from raw frequency counts.
///
/// H(X) = -Σ p(x) * log2(p(x))    (in bits)
///
/// Args:
///     counts: List of non-negative integers — frequency counts per category.
///
/// Returns:
///     dict with keys:
///       entropy_bits (float), entropy_nats (float),
///       n_unique (int), n_total (int), normalized (float in [0,1])
///
/// Example:
///     >>> # Uniform over 4 categories → max entropy = 2.0 bits
///     >>> nydra_core.shannon_entropy([1, 1, 1, 1])["entropy_bits"]
///     2.0
///     >>> # Perfectly predictable → 0 bits
///     >>> nydra_core.shannon_entropy([100])["entropy_bits"]
///     0.0
#[pyfunction]
#[pyo3(name = "shannon_entropy", signature = (counts))]
fn py_shannon_entropy(py: Python<'_>, counts: Vec<u64>) -> PyResult<PyObject> {
    let result: EntropyResult = py.allow_threads(|| {
        stats::shannon_entropy(&counts).map_err(PyErr::from)
    })?;
    entropy_to_py(py, &result)
}

/// Compute Shannon entropy from a list of string categories.
///
/// Automatically groups unique values and counts them.
///
/// Args:
///     values: List of strings (category labels).
///
/// Returns:
///     dict — same structure as shannon_entropy.
///
/// Example:
///     >>> nydra_core.entropy_from_strings(["a", "b", "a", "c", "b", "a"])
///     {"entropy_bits": 1.459..., "n_unique": 3, "normalized": 0.920...}
#[pyfunction]
#[pyo3(name = "entropy_from_strings", signature = (values))]
fn py_entropy_from_strings(py: Python<'_>, values: Vec<String>) -> PyResult<PyObject> {
    let result: EntropyResult = py.allow_threads(|| {
        stats::entropy_from_strings(&values).map_err(PyErr::from)
    })?;
    entropy_to_py(py, &result)
}

/// Compute Shannon entropy from a numeric column via equal-width binning.
///
/// Args:
///     data:   List of numeric values. Must be finite.
///     n_bins: Number of equal-width bins (default 10).
///
/// Returns:
///     dict — same structure as shannon_entropy.
///
/// Example:
///     >>> data = list(range(1000))
///     >>> nydra_core.entropy_numeric(data, n_bins=10)["normalized"]
///     ~1.0  # Uniform distribution → maximum entropy
#[pyfunction]
#[pyo3(name = "entropy_numeric", signature = (data, n_bins = 10))]
fn py_entropy_numeric(py: Python<'_>, data: &PyAny, n_bins: usize) -> PyResult<PyObject> {
    let vec = extract_f64_vec(py, data)?;
    let result: EntropyResult = py.allow_threads(|| {
        stats::entropy_numeric(&vec, n_bins).map_err(PyErr::from)
    })?;
    entropy_to_py(py, &result)
}

/// Convert an EntropyResult struct into a Python dict.
fn entropy_to_py(py: Python<'_>, e: &EntropyResult) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("entropy_bits", e.entropy_bits)?;
    dict.set_item("entropy_nats", e.entropy_nats)?;
    dict.set_item("n_unique",     e.n_unique)?;
    dict.set_item("n_total",      e.n_total)?;
    dict.set_item("normalized",   e.normalized)?;
    Ok(dict.into())
}

// =============================================================================
// SECTION 5: CORRELATION
// =============================================================================

/// Compute Pearson correlation between two numeric columns.
///
/// Measures linear relationship strength. Range: [-1.0, 1.0]
///   1.0  → perfect positive linear correlation
///   0.0  → no linear relationship
///  -1.0  → perfect negative linear correlation
///
/// Args:
///     x: First list of numeric values.
///     y: Second list of numeric values. Must have same length as x.
///
/// Returns:
///     dict with keys:
///       method ("pearson"), coefficient (float), p_value (float),
///       is_significant (bool), n (int)
///
/// Raises:
///     ValueError: If lengths differ, data is empty, or one variable has zero variance.
///
/// Example:
///     >>> nydra_core.pearson_correlation([1,2,3], [2,4,6])["coefficient"]
///     1.0
///     >>> nydra_core.pearson_correlation([1,2,3], [6,4,2])["coefficient"]
///     -1.0
#[pyfunction]
#[pyo3(name = "pearson_correlation", signature = (x, y))]
fn py_pearson_correlation(py: Python<'_>, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    let xv = extract_f64_vec(py, x)?;
    let yv = extract_f64_vec(py, y)?;
    let result: CorrelationResult = py.allow_threads(|| {
        stats::pearson_correlation(&xv, &yv).map_err(PyErr::from)
    })?;
    correlation_to_py(py, &result)
}

/// Compute Spearman rank correlation between two numeric columns.
///
/// Measures monotonic relationship (not limited to linear).
/// More robust than Pearson when data contains outliers or is not normally distributed.
///
/// Args:
///     x: First list of numeric values.
///     y: Second list of numeric values. Must have same length as x.
///
/// Returns:
///     dict — same structure as pearson_correlation (with method="spearman").
///
/// Example:
///     >>> # y = x² is monotonic but not linear — Spearman = 1.0, Pearson < 1.0
///     >>> x = [1, 2, 3, 4, 5]
///     >>> y = [1, 4, 9, 16, 25]
///     >>> nydra_core.spearman_correlation(x, y)["coefficient"]
///     1.0
#[pyfunction]
#[pyo3(name = "spearman_correlation", signature = (x, y))]
fn py_spearman_correlation(py: Python<'_>, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    let xv = extract_f64_vec(py, x)?;
    let yv = extract_f64_vec(py, y)?;
    let result: CorrelationResult = py.allow_threads(|| {
        stats::spearman_correlation(&xv, &yv).map_err(PyErr::from)
    })?;
    correlation_to_py(py, &result)
}

/// Compute the full correlation matrix for multiple columns.
///
/// The matrix is symmetric with 1s on the diagonal. Only the upper triangle
/// is computed; the lower triangle is filled by mirroring.
///
/// Args:
///     columns:     List of columns — each column is a list of floats.
///     col_names:   List of column name strings (same length as columns).
///     method:      "pearson" (default) or "spearman".
///
/// Returns:
///     dict with keys:
///       column_names (list[str]),
///       matrix (list[list[float]]) — row-major k×k matrix,
///       method (str)
///
/// Example:
///     >>> cols  = [[1,2,3], [2,4,6], [3,2,1]]
///     >>> names = ["a", "b", "c"]
///     >>> mat   = nydra_core.correlation_matrix(cols, names)
///     >>> mat["matrix"][0][1]   # corr(a, b) = 1.0
///     1.0
///     >>> mat["matrix"][0][2]   # corr(a, c) = -1.0
///     -1.0
#[pyfunction]
#[pyo3(name = "correlation_matrix", signature = (columns, col_names, method = "pearson"))]
fn py_correlation_matrix(
    py:       Python<'_>,
    columns:  Vec<Vec<f64>>,
    col_names: Vec<String>,
    method:   &str,
) -> PyResult<PyObject> {

    if columns.len() != col_names.len() {
        return Err(PyValueError::new_err(format!(
            "columns and col_names must have the same length — got {} and {}",
            columns.len(), col_names.len()
        )));
    }

    let method_owned = method.to_string();
    let result: CorrelationMatrix = py.allow_threads(|| {
        stats::correlation_matrix(&columns, &col_names, &method_owned).map_err(PyErr::from)
    })?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("column_names", result.column_names)?;
    dict.set_item("matrix",       result.matrix)?;
    dict.set_item("method",       result.method)?;
    Ok(dict.into())
}

/// Convert a CorrelationResult struct into a Python dict.
fn correlation_to_py(py: Python<'_>, r: &CorrelationResult) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("method",         &r.method)?;
    dict.set_item("coefficient",    r.coefficient)?;
    dict.set_item("p_value",        r.p_value)?;
    dict.set_item("is_significant", r.is_significant)?;
    dict.set_item("n",              r.n)?;
    Ok(dict.into())
}

// =============================================================================
// SECTION 6: ROLLING / WINDOWED STATISTICS
// =============================================================================

/// Compute rolling (moving) mean with a given window size.
///
/// Uses an O(n) sliding-window algorithm — much faster than computing
/// the mean from scratch for each window position.
///
/// Args:
///     data:   List of numeric values.
///     window: Window size (must be ≤ len(data)).
///
/// Returns:
///     list[float] of length (len(data) - window + 1).
///
/// Example:
///     >>> nydra_core.rolling_mean([1, 2, 3, 4, 5], 3)
///     [2.0, 3.0, 4.0]   # (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
#[pyfunction]
#[pyo3(name = "rolling_mean", signature = (data, window))]
fn py_rolling_mean(py: Python<'_>, data: &PyAny, window: usize) -> PyResult<Vec<f64>> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::rolling_mean(&vec, window).map_err(PyErr::from))
}

/// Compute rolling standard deviation with a given window size.
///
/// Args:
///     data:   List of numeric values.
///     window: Window size (must be ≤ len(data)).
///
/// Returns:
///     list[float] of length (len(data) - window + 1).
///
/// Example:
///     >>> nydra_core.rolling_std([1, 2, 3, 4, 5], 3)
///     [1.0, 1.0, 1.0]
#[pyfunction]
#[pyo3(name = "rolling_std", signature = (data, window))]
fn py_rolling_std(py: Python<'_>, data: &PyAny, window: usize) -> PyResult<Vec<f64>> {
    let vec = extract_f64_vec(py, data)?;
    py.allow_threads(|| stats::rolling_std(&vec, window).map_err(PyErr::from))
}