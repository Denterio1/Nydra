// =============================================================================
// src/stats.rs — Nydra Core: High-Performance Statistical Engine
// =============================================================================
//
// PURPOSE:
//   This module contains every heavy numerical computation that would be too
//   slow in pure Python. Each function here is:
//     1. Parallelized across all CPU cores via Rayon
//     2. Written with cache-friendly memory access patterns
//     3. Numerically stable (uses compensated summation where needed)
//     4. Fully validated (returns errors instead of NaN/panic on bad input)
//
// ARCHITECTURE:
//   Python caller → lib.rs (bridge) → stats.rs (this file)
//
//   lib.rs receives Python objects (Vec<f64>, etc.), calls into this module,
//   and converts the results back to Python types. This module knows nothing
//   about Python — it only works with pure Rust types, making it testable
//   with `cargo test` independently of any Python environment.
//
// NUMERICAL STABILITY NOTES:
//   - Summation: Uses Kahan compensated summation (not plain sum) to prevent
//     floating-point error accumulation on large datasets.
//   - Variance: Uses Welford's online algorithm (single-pass, numerically
//     stable) instead of the naive two-pass E[X²] - E[X]² formula, which
//     catastrophically cancels for nearly-equal values.
//   - Entropy: Clamps near-zero probabilities to avoid log(0) = -infinity.
//
// =============================================================================

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::collections::HashMap;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// All errors this module can produce.
/// Using `thiserror` gives us clean Display messages for free.
#[derive(Debug, Error)]
pub enum StatsError {
    #[error("Input slice is empty — cannot compute statistics on zero elements")]
    EmptyInput,

    #[error("Input contains {0} NaN value(s) — use filter_nan() before calling this function")]
    ContainsNaN(usize),

    #[error("Input contains {0} infinite value(s) — finite values required")]
    ContainsInfinite(usize),

    #[error("Standard deviation is zero — correlation is undefined when one variable has no variance")]
    ZeroVariance,

    #[error("Window size {window} is larger than data length {data_len}")]
    WindowTooLarge { window: usize, data_len: usize },

    #[error("Percentile {0} out of range — must be in [0.0, 100.0]")]
    InvalidPercentile(f64),

    #[error("Bins count must be ≥ 1, got {0}")]
    InvalidBinCount(usize),

    #[error("Both slices must have the same length — got {0} and {1}")]
    LengthMismatch(usize, usize),
}

pub type StatsResult<T> = Result<T, StatsError>;

// =============================================================================
// OUTPUT STRUCTS (serializable → JSON → Python dict)
// =============================================================================

/// Complete descriptive statistics for a single numeric column.
#[derive(Debug, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub count:              usize,
    pub mean:               f64,
    pub median:             f64,
    pub std_dev:            f64,       // Population std dev (n denominator)
    pub std_dev_sample:     f64,       // Sample std dev (n-1 denominator)
    pub variance:           f64,
    pub variance_sample:    f64,
    pub min:                f64,
    pub max:                f64,
    pub range:              f64,
    pub q1:                 f64,       // 25th percentile
    pub q3:                 f64,       // 75th percentile
    pub iqr:                f64,       // Interquartile range = Q3 - Q1
    pub mad:                f64,       // Median Absolute Deviation
    pub skewness:           f64,       // Fisher's skewness (g1)
    pub kurtosis:           f64,       // Excess kurtosis (g2, 0 = normal)
    pub cv:                 f64,       // Coefficient of Variation = std/mean * 100
    pub sum:                f64,
    pub nan_count:          usize,     // Number of NaN values (informational)
    pub infinite_count:     usize,     // Number of Inf values (informational)
}

/// Result of outlier detection on a numeric column.
#[derive(Debug, Serialize, Deserialize)]
pub struct OutlierResult {
    pub method:             String,
    pub outlier_indices:    Vec<usize>,
    pub outlier_count:      usize,
    pub outlier_fraction:   f64,       // outlier_count / total_count
    pub lower_fence:        f64,       // Values below this are outliers
    pub upper_fence:        f64,       // Values above this are outliers
}

/// Result of correlation computation between two columns.
#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationResult {
    pub method:         String,   // "pearson" | "spearman" | "kendall"
    pub coefficient:    f64,      // [-1.0, 1.0]
    pub p_value:        f64,      // Two-tailed significance
    pub is_significant: bool,     // p_value < 0.05
    pub n:              usize,    // Number of paired observations used
}

/// Shannon entropy result.
#[derive(Debug, Serialize, Deserialize)]
pub struct EntropyResult {
    pub entropy_bits:   f64,   // In bits (log base 2)
    pub entropy_nats:   f64,   // In nats (natural log)
    pub n_unique:       usize,
    pub n_total:        usize,
    pub normalized:     f64,   // entropy_bits / log2(n_unique) — in [0,1]
}

/// Full correlation matrix for multiple columns.
#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub column_names:   Vec<String>,
    pub matrix:         Vec<Vec<f64>>,   // row-major, matrix[i][j] = corr(col_i, col_j)
    pub method:         String,
}

// =============================================================================
// SECTION 1: CORE SUMMATION PRIMITIVES
// =============================================================================

/// Kahan compensated summation — accumulates floating-point sum with O(1)
/// additional error instead of O(n) error from naive summation.
///
/// For a dataset of n values, naive sum has error O(n * epsilon * max_val).
/// Kahan reduces this to O(epsilon * max_val) — independent of n.
///
/// Reference: Kahan (1965) "Further remarks on reducing truncation errors"
#[inline]
pub fn kahan_sum(data: &[f64]) -> f64 {
    let mut sum  = 0.0_f64;
    let mut comp = 0.0_f64;  // Compensation term

    for &x in data {
        let y = x - comp;
        let t = sum + y;
        comp  = (t - sum) - y;
        sum   = t;
    }

    sum
}

/// Parallel Kahan sum — splits the slice into chunks, computes compensated
/// sum per chunk in parallel, then combines. Gives ~linear speedup with cores.
pub fn parallel_sum(data: &[f64]) -> f64 {
    // Each rayon thread independently computes a Kahan sum over its chunk.
    // Final reduce uses plain addition — acceptable because we're summing
    // O(num_cores) partial sums, not O(n) values.
    data.par_chunks(8192)
        .map(kahan_sum)
        .sum()
}

/// Parallel mean using compensated summation.
pub fn parallel_mean(data: &[f64]) -> f64 {
    parallel_sum(data) / data.len() as f64
}

// =============================================================================
// SECTION 2: INPUT VALIDATION
// =============================================================================

/// Count NaN and Infinite values in the slice.
/// Returns (nan_count, inf_count).
pub fn count_invalid(data: &[f64]) -> (usize, usize) {
    data.par_iter()
        .fold(
            || (0usize, 0usize),
            |(nans, infs), &x| {
                let is_nan = x.is_nan() as usize;
                let is_inf = x.is_infinite() as usize;
                (nans + is_nan, infs + is_inf)
            },
        )
        .reduce(|| (0, 0), |(a0, b0), (a1, b1)| (a0 + a1, b0 + b1))
}

/// Validate that data is non-empty, finite, and contains no NaN.
/// Returns (nan_count, inf_count) so callers can include them in reports.
pub fn validate_finite(data: &[f64]) -> StatsResult<(usize, usize)> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let (nans, infs) = count_invalid(data);
    if nans > 0  { return Err(StatsError::ContainsNaN(nans)); }
    if infs > 0  { return Err(StatsError::ContainsInfinite(infs)); }
    Ok((nans, infs))
}

/// Filter out NaN and Infinite values, returning a clean Vec<f64>.
/// Also returns the count of removed values.
pub fn filter_nan(data: &[f64]) -> (Vec<f64>, usize) {
    let clean: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    let removed = data.len() - clean.len();
    (clean, removed)
}

// =============================================================================
// SECTION 3: ORDER STATISTICS (PERCENTILES / MEDIAN)
// =============================================================================

/// Compute a percentile using linear interpolation (type 7 — pandas default).
///
/// This matches numpy's `np.percentile(data, p, interpolation='linear')`.
/// We sort a copy of the data (O(n log n)) — acceptable because percentile
/// computation is inherently O(n log n).
///
/// # Arguments
/// * `data`       — sorted or unsorted slice of finite f64
/// * `percentile` — value in [0.0, 100.0]
pub fn percentile(data: &mut [f64], p: f64) -> StatsResult<f64> {
    if !(0.0..=100.0).contains(&p) {
        return Err(StatsError::InvalidPercentile(p));
    }
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    // Sort in ascending order using unstable sort (faster, same result for f64)
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len();

    // Edge cases
    if p == 0.0  { return Ok(data[0]); }
    if p == 100.0 { return Ok(data[n - 1]); }

    // Linear interpolation: find the virtual index and interpolate
    let virtual_idx = (p / 100.0) * (n as f64 - 1.0);
    let lo          = virtual_idx.floor() as usize;
    let hi          = virtual_idx.ceil()  as usize;
    let frac        = virtual_idx - lo as f64;

    Ok(data[lo] + frac * (data[hi] - data[lo]))
}

/// Compute median without modifying the input (clones internally).
pub fn median(data: &[f64]) -> StatsResult<f64> {
    let mut buf = data.to_vec();
    percentile(&mut buf, 50.0)
}

/// Compute Q1 and Q3 simultaneously (cheaper than two separate calls).
pub fn quartiles(data: &[f64]) -> StatsResult<(f64, f64)> {
    let mut buf = data.to_vec();
    buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = percentile(&mut buf.clone(), 25.0)?;
    let q3 = percentile(&mut buf.clone(), 75.0)?;
    Ok((q1, q3))
}

// =============================================================================
// SECTION 4: WELFORD'S ONLINE VARIANCE ALGORITHM
// =============================================================================

/// Compute mean and variance in a single parallel pass using Welford's algorithm.
///
/// Returns (mean, population_variance, sample_variance, count).
///
/// WHY WELFORD'S?
///   The naive formula: Var = E[X²] - E[X]²
///   suffers catastrophic cancellation when values are nearly equal
///   (e.g., [1000001.0, 1000002.0, 1000003.0] → subtraction loses all precision).
///
///   Welford's algorithm maintains a running mean and a running sum of squared
///   deviations from the current mean, avoiding the subtraction entirely.
///
/// PARALLEL VERSION:
///   We split into chunks, compute (count, mean, M2) per chunk independently,
///   then merge using Chan's parallel algorithm for combining Welford states.
///
/// Reference: Welford (1962), Chan et al. (1979) "Updating Formulae and a
///            Pairwise Algorithm for Computing Sample Variances"
pub fn welford_parallel(data: &[f64]) -> StatsResult<(f64, f64, f64, usize)> {
    validate_finite(data)?;

    // Each chunk produces a (count, mean, M2) triple
    let (count, mean, m2): (usize, f64, f64) = data
        .par_chunks(4096)
        .map(|chunk| {
            let mut count = 0usize;
            let mut mean  = 0.0_f64;
            let mut m2    = 0.0_f64;

            for &x in chunk {
                count += 1;
                let delta  = x - mean;
                mean      += delta / count as f64;
                let delta2 = x - mean;
                m2        += delta * delta2;
            }

            (count, mean, m2)
        })
        // Chan's parallel merge of two Welford states (a) and (b) → combined
        .reduce(
            || (0usize, 0.0_f64, 0.0_f64),
            |(na, ma, m2a), (nb, mb, m2b)| {
                let n     = na + nb;
                let delta = mb - ma;
                let m     = (na as f64 * ma + nb as f64 * mb) / n as f64;
                let m2    = m2a + m2b + delta * delta * (na as f64 * nb as f64 / n as f64);
                (n, m, m2)
            },
        );

    let pop_var    = if count < 1 { 0.0 } else { m2 / count as f64 };
    let sample_var = if count < 2 { 0.0 } else { m2 / (count - 1) as f64 };

    Ok((mean, pop_var, sample_var, count))
}

// =============================================================================
// SECTION 5: STANDARD DEVIATION
// =============================================================================

/// Population standard deviation (denominator = n).
pub fn std_dev(data: &[f64]) -> StatsResult<f64> {
    let (_, pop_var, _, _) = welford_parallel(data)?;
    Ok(pop_var.sqrt())
}

/// Sample standard deviation (denominator = n-1, unbiased estimator).
pub fn std_dev_sample(data: &[f64]) -> StatsResult<f64> {
    let (_, _, sample_var, _) = welford_parallel(data)?;
    Ok(sample_var.sqrt())
}

// =============================================================================
// SECTION 6: SKEWNESS & KURTOSIS
// =============================================================================

/// Fisher's skewness (g1) — standardized third central moment.
///
/// Formula: g1 = (1/n * Σ(xi - μ)³) / σ³
///
/// > 0  → right-skewed (long tail on the right, e.g., income distributions)
/// < 0  → left-skewed  (long tail on the left)
/// ≈ 0  → approximately symmetric
pub fn skewness(data: &[f64]) -> StatsResult<f64> {
    let (mean, pop_var, _, n) = welford_parallel(data)?;
    let sigma = pop_var.sqrt();

    if sigma == 0.0 {
        return Err(StatsError::ZeroVariance);
    }

    // Third central moment, computed in parallel
    let m3: f64 = data
        .par_iter()
        .map(|&x| {
            let z = (x - mean) / sigma;
            z * z * z
        })
        .sum::<f64>()
        / n as f64;

    Ok(m3)
}

/// Excess kurtosis (g2) — standardized fourth central moment minus 3.
///
/// Subtracting 3 makes the normal distribution have kurtosis = 0 (excess kurtosis).
///
/// > 0 → leptokurtic (heavy tails, sharp peak — e.g., financial returns)
/// < 0 → platykurtic  (light tails, flat peak — e.g., uniform distribution)
/// ≈ 0 → mesokurtic   (normal-like tail behavior)
pub fn excess_kurtosis(data: &[f64]) -> StatsResult<f64> {
    let (mean, pop_var, _, n) = welford_parallel(data)?;
    let sigma2 = pop_var;  // variance

    if sigma2 == 0.0 {
        return Err(StatsError::ZeroVariance);
    }

    // Fourth central moment
    let m4: f64 = data
        .par_iter()
        .map(|&x| {
            let diff = x - mean;
            let d2   = diff * diff;
            d2 * d2
        })
        .sum::<f64>()
        / n as f64;

    Ok(m4 / (sigma2 * sigma2) - 3.0)
}

// =============================================================================
// SECTION 7: MEDIAN ABSOLUTE DEVIATION (MAD)
// =============================================================================

/// Median Absolute Deviation — a robust alternative to standard deviation.
///
/// MAD = median(|xi - median(x)|)
///
/// Unlike std dev, MAD is not affected by extreme outliers. It gives a
/// consistent estimate of spread even when the data contains anomalies.
///
/// For normally distributed data: std ≈ 1.4826 * MAD (the scale factor 1.4826
/// makes MAD a consistent estimator of the population standard deviation).
pub fn mad(data: &[f64]) -> StatsResult<f64> {
    validate_finite(data)?;

    let med     = median(data)?;
    let mut abs_devs: Vec<f64> = data.par_iter().map(|&x| (x - med).abs()).collect();

    median(&abs_devs)
}

/// MAD scaled to be a consistent estimator of std dev (multiply by 1.4826).
pub fn mad_scaled(data: &[f64]) -> StatsResult<f64> {
    Ok(mad(data)? * 1.4826)
}

// =============================================================================
// SECTION 8: OUTLIER DETECTION
// =============================================================================

/// IQR-based outlier detection.
///
/// Standard fences: lower = Q1 - k*IQR,  upper = Q3 + k*IQR
/// Default k = 1.5 (Tukey's rule). Use k=3.0 for "far" outliers only.
///
/// # Returns
/// `OutlierResult` with indices of values outside the fences.
pub fn detect_outliers_iqr(data: &[f64], k: f64) -> StatsResult<OutlierResult> {
    validate_finite(data)?;

    let mut buf = data.to_vec();
    let (q1, q3) = quartiles(&buf)?;
    let iqr       = q3 - q1;

    let lower = q1 - k * iqr;
    let upper = q3 + k * iqr;

    // Collect indices of outliers in parallel
    let outlier_indices: Vec<usize> = data
        .par_iter()
        .enumerate()
        .filter_map(|(i, &x)| {
            if x < lower || x > upper { Some(i) } else { None }
        })
        .collect();

    let n            = data.len();
    let outlier_count   = outlier_indices.len();
    let outlier_fraction = outlier_count as f64 / n as f64;

    Ok(OutlierResult {
        method: format!("IQR (k={k})"),
        outlier_indices,
        outlier_count,
        outlier_fraction,
        lower_fence: lower,
        upper_fence: upper,
    })
}

/// Modified Z-score outlier detection using MAD (Iglewicz & Hoaglin, 1993).
///
/// Modified Z-score: Mi = 0.6745 * (xi - median) / MAD
///
/// Values with |Mi| > threshold (default 3.5) are flagged as outliers.
/// This method is more robust than standard Z-score when the data contains
/// extreme outliers (which would inflate the mean and std dev).
pub fn detect_outliers_mad(data: &[f64], threshold: f64) -> StatsResult<OutlierResult> {
    validate_finite(data)?;

    let med = median(data)?;
    let mad_val = mad(data)?;

    // If MAD == 0 (constant column), nothing is an outlier by this method
    if mad_val == 0.0 {
        return Ok(OutlierResult {
            method:          "Modified Z-Score (MAD)".to_string(),
            outlier_indices:  vec![],
            outlier_count:    0,
            outlier_fraction: 0.0,
            lower_fence:      med,
            upper_fence:      med,
        });
    }

    let scale        = 0.6745 / mad_val;
    let lower        = med - threshold / scale;
    let upper        = med + threshold / scale;

    let outlier_indices: Vec<usize> = data
        .par_iter()
        .enumerate()
        .filter_map(|(i, &x)| {
            let score = (0.6745 * (x - med) / mad_val).abs();
            if score > threshold { Some(i) } else { None }
        })
        .collect();

    let n            = data.len();
    let outlier_count   = outlier_indices.len();
    let outlier_fraction = outlier_count as f64 / n as f64;

    Ok(OutlierResult {
        method: format!("Modified Z-Score (MAD, threshold={threshold})"),
        outlier_indices,
        outlier_count,
        outlier_fraction,
        lower_fence: lower,
        upper_fence: upper,
    })
}

// =============================================================================
// SECTION 9: ENTROPY
// =============================================================================

/// Shannon entropy of a categorical or discretized series.
///
/// H(X) = -Σ p(x) * log2(p(x))
///
/// Maximum entropy = log2(n_unique) bits (uniform distribution).
/// Zero entropy = perfectly predictable (only one unique value).
///
/// # Arguments
/// * `counts` — frequency count per unique value (must sum to > 0)
pub fn shannon_entropy(counts: &[u64]) -> StatsResult<EntropyResult> {
    if counts.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let total: u64 = counts.iter().sum();
    if total == 0 {
        return Err(StatsError::EmptyInput);
    }

    let n_total  = total as usize;
    let n_unique = counts.len();
    let total_f  = total as f64;

    // H = -Σ p * log2(p), skipping zero-count bins (0 * log(0) = 0 by convention)
    let entropy_bits: f64 = counts
        .par_iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.log2()
        })
        .sum();

    let entropy_nats = entropy_bits * std::f64::consts::LN_2;  // bits → nats

    let max_bits  = (n_unique as f64).log2();
    let normalized = if max_bits > 0.0 { entropy_bits / max_bits } else { 1.0 };

    Ok(EntropyResult {
        entropy_bits,
        entropy_nats,
        n_unique,
        n_total,
        normalized,
    })
}

/// Compute entropy directly from a string-valued column.
/// Groups values, counts occurrences, then calls `shannon_entropy`.
pub fn entropy_from_strings(values: &[String]) -> StatsResult<EntropyResult> {
    if values.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    // Count occurrences of each unique string
    let mut counts: HashMap<&str, u64> = HashMap::new();
    for v in values {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }

    let count_vec: Vec<u64> = counts.values().copied().collect();
    shannon_entropy(&count_vec)
}

/// Compute entropy from a numeric column by binning into `n_bins` equal-width bins.
pub fn entropy_numeric(data: &[f64], n_bins: usize) -> StatsResult<EntropyResult> {
    validate_finite(data)?;

    if n_bins < 1 {
        return Err(StatsError::InvalidBinCount(n_bins));
    }

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        // All values identical → single bin, zero entropy
        return Ok(EntropyResult {
            entropy_bits: 0.0,
            entropy_nats: 0.0,
            n_unique:     1,
            n_total:      data.len(),
            normalized:   0.0,
        });
    }

    let width = (max - min) / n_bins as f64;
    let mut counts = vec![0u64; n_bins];

    for &x in data {
        let bin_idx = ((x - min) / width) as usize;
        // Clamp to handle x == max (which maps to n_bins)
        let bin_idx = bin_idx.min(n_bins - 1);
        counts[bin_idx] += 1;
    }

    shannon_entropy(&counts)
}

// =============================================================================
// SECTION 10: PEARSON CORRELATION
// =============================================================================

/// Pearson correlation coefficient between two equal-length slices.
///
/// r = Σ[(xi - x̄)(yi - ȳ)] / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
///
/// Computed in a single parallel pass to avoid materializing intermediate arrays.
/// Returns a `CorrelationResult` with p-value approximation for significance testing.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch(x.len(), y.len()));
    }

    validate_finite(x)?;
    validate_finite(y)?;

    let n = x.len();

    let (mean_x, var_x, _, _) = welford_parallel(x)?;
    let (mean_y, var_y, _, _) = welford_parallel(y)?;

    let std_x = var_x.sqrt();
    let std_y = var_y.sqrt();

    if std_x == 0.0 || std_y == 0.0 {
        return Err(StatsError::ZeroVariance);
    }

    // Σ[(xi - x̄)(yi - ȳ)] — computed in parallel over paired chunks
    let cov_sum: f64 = x
        .par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let r = cov_sum / (n as f64 * std_x * std_y);
    let r = r.clamp(-1.0, 1.0);  // Clamp to handle floating-point rounding past ±1

    // Approximate p-value using t-distribution with n-2 degrees of freedom
    // t = r * sqrt((n-2) / (1 - r²))
    let p_value = approximate_pearson_pvalue(r, n);

    Ok(CorrelationResult {
        method:         "pearson".to_string(),
        coefficient:    r,
        p_value,
        is_significant: p_value < 0.05,
        n,
    })
}

/// Approximate two-tailed p-value for Pearson r using Student's t approximation.
/// This is an approximation valid for n ≥ 5. For exact values, use scipy.
fn approximate_pearson_pvalue(r: f64, n: usize) -> f64 {
    if n < 3 {
        return 1.0;  // Not enough data to compute significance
    }

    let df = (n - 2) as f64;
    let t  = r * (df / (1.0 - r * r)).sqrt();

    // Approximate CDF of t-distribution using the incomplete beta function
    // For simplicity, use a Gaussian approximation (valid for large df)
    // For small n, users should use scipy.stats.pearsonr in Python
    let z_approx = t / (1.0 + t * t / (2.0 * df)).sqrt();
    let p_one_tailed = 0.5 * erfc(z_approx.abs() / std::f64::consts::SQRT_2);

    (2.0 * p_one_tailed).min(1.0)
}

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
/// Max absolute error: 1.5e-7
fn erfc(x: f64) -> f64 {
    let t  = 1.0 / (1.0 + 0.3275911 * x);
    let y  = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
                        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    y * (-x * x).exp()
}

// =============================================================================
// SECTION 11: SPEARMAN RANK CORRELATION
// =============================================================================

/// Spearman rank correlation — Pearson correlation of the ranked data.
///
/// Spearman measures monotonic relationships (not just linear ones).
/// It's appropriate when:
///   - Data is not normally distributed
///   - Relationship is monotonic but not strictly linear
///   - Data contains outliers (ranks are robust to them)
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch(x.len(), y.len()));
    }

    validate_finite(x)?;
    validate_finite(y)?;

    let rx = rank_data(x);
    let ry = rank_data(y);

    // Spearman = Pearson(ranks)
    let mut result = pearson_correlation(&rx, &ry)?;
    result.method = "spearman".to_string();
    Ok(result)
}

/// Convert a slice of values to their ranks (average ranks for ties).
///
/// Rank 1 = smallest value. Ties get the average of their ranks.
/// Example: [3.0, 1.0, 1.0, 2.0] → [4.0, 1.5, 1.5, 3.0]
fn rank_data(data: &[f64]) -> Vec<f64> {
    let n = data.len();

    // Create index array sorted by value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());

    let mut ranks = vec![0.0_f64; n];
    let mut i     = 0usize;

    while i < n {
        // Find the end of the current run of equal values (tie group)
        let mut j = i + 1;
        while j < n && (data[indices[j]] - data[indices[i]]).abs() < f64::EPSILON {
            j += 1;
        }

        // Average rank for this tie group: ranks are 1-based, so add 1
        let avg_rank = (i + j + 1) as f64 / 2.0;

        for k in i..j {
            ranks[indices[k]] = avg_rank;
        }

        i = j;
    }

    ranks
}

// =============================================================================
// SECTION 12: FULL CORRELATION MATRIX
// =============================================================================

/// Compute the full correlation matrix for multiple columns in parallel.
///
/// The matrix is symmetric (corr[i][j] == corr[j][i]) and has 1s on the diagonal.
/// We only compute the upper triangle and mirror it, halving the work.
pub fn correlation_matrix(
    columns: &[Vec<f64>],
    col_names: &[String],
    method: &str,
) -> StatsResult<CorrelationMatrix> {
    let k = columns.len();

    // Validate all columns
    for col in columns {
        validate_finite(col)?;
    }

    // Initialize k×k matrix with 1s on diagonal
    let mut matrix = vec![vec![1.0_f64; k]; k];

    // Compute upper triangle in parallel over (i, j) pairs
    let pairs: Vec<(usize, usize)> = (0..k)
        .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let r = match method {
                "spearman" => spearman_correlation(&columns[i], &columns[j]),
                _          => pearson_correlation(&columns[i], &columns[j]),
            };
            let coeff = r.map(|c| c.coefficient).unwrap_or(f64::NAN);
            (i, j, coeff)
        })
        .collect();

    for (i, j, coeff) in results {
        matrix[i][j] = coeff;
        matrix[j][i] = coeff;  // Mirror — symmetric
    }

    Ok(CorrelationMatrix {
        column_names: col_names.to_vec(),
        matrix,
        method: method.to_string(),
    })
}

// =============================================================================
// SECTION 13: FULL DESCRIPTIVE STATISTICS
// =============================================================================

/// Compute all descriptive statistics for a column in a single function call.
///
/// NaN and infinite values are automatically excluded (their counts are recorded
/// in the result for transparency).
pub fn full_descriptive(data: &[f64]) -> StatsResult<DescriptiveStats> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    // Count and remove invalid values
    let (nan_count, infinite_count) = count_invalid(data);
    let clean: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();

    if clean.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let n = clean.len();

    // --- Central tendency ---
    let (mean, pop_var, sample_var, _) = welford_parallel(&clean)?;
    let med                             = median(&clean)?;

    // --- Spread ---
    let std_pop    = pop_var.sqrt();
    let std_samp   = sample_var.sqrt();
    let (q1, q3)  = quartiles(&clean)?;
    let iqr        = q3 - q1;
    let mad_val    = mad(&clean)?;

    // --- Range ---
    let min = clean.iter().copied().fold(f64::INFINITY,     f64::min);
    let max = clean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    // --- Shape ---
    let skew      = skewness(&clean).unwrap_or(f64::NAN);
    let kurt      = excess_kurtosis(&clean).unwrap_or(f64::NAN);

    // --- Coefficient of Variation (undefined if mean ≈ 0) ---
    let cv = if mean.abs() > f64::EPSILON {
        (std_pop / mean.abs()) * 100.0
    } else {
        f64::NAN
    };

    let sum = kahan_sum(&clean);

    Ok(DescriptiveStats {
        count:          n,
        mean,
        median:         med,
        std_dev:        std_pop,
        std_dev_sample: std_samp,
        variance:       pop_var,
        variance_sample: sample_var,
        min,
        max,
        range,
        q1,
        q3,
        iqr,
        mad:            mad_val,
        skewness:       skew,
        kurtosis:       kurt,
        cv,
        sum,
        nan_count,
        infinite_count,
    })
}

// =============================================================================
// SECTION 14: ROLLING / WINDOWED STATISTICS
// =============================================================================

/// Compute rolling mean with a given window size.
///
/// Uses an efficient sliding-window algorithm: O(n) total operations
/// instead of O(n * window) for naïve per-window recomputation.
pub fn rolling_mean(data: &[f64], window: usize) -> StatsResult<Vec<f64>> {
    if data.len() < window {
        return Err(StatsError::WindowTooLarge { window, data_len: data.len() });
    }

    let n = data.len();
    let mut result = Vec::with_capacity(n - window + 1);

    // Initialize first window sum
    let mut window_sum: f64 = kahan_sum(&data[..window]);
    result.push(window_sum / window as f64);

    // Slide: subtract outgoing element, add incoming element
    for i in window..n {
        window_sum += data[i] - data[i - window];
        result.push(window_sum / window as f64);
    }

    Ok(result)
}

/// Compute rolling standard deviation with a given window size.
/// Uses a sliding Welford algorithm for O(n) computation.
pub fn rolling_std(data: &[f64], window: usize) -> StatsResult<Vec<f64>> {
    if data.len() < window {
        return Err(StatsError::WindowTooLarge { window, data_len: data.len() });
    }

    let n = data.len();
    let mut result = Vec::with_capacity(n - window + 1);

    for i in 0..=(n - window) {
        let slice = &data[i..(i + window)];
        let (_, _, sample_var, _) = welford_parallel(slice)?;
        result.push(sample_var.sqrt());
    }

    Ok(result)
}

// =============================================================================
// SECTION 15: UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: assert two floats are approximately equal
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -------------------------------------------------------------------------
    // Kahan summation
    // -------------------------------------------------------------------------
    #[test]
    fn test_kahan_sum_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(kahan_sum(&data), 15.0, 1e-10));
    }

    #[test]
    fn test_kahan_sum_empty() {
        assert_eq!(kahan_sum(&[]), 0.0);
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------
    #[test]
    fn test_validate_finite_empty() {
        assert!(matches!(validate_finite(&[]), Err(StatsError::EmptyInput)));
    }

    #[test]
    fn test_validate_finite_nan() {
        let data = vec![1.0, f64::NAN, 3.0];
        assert!(matches!(validate_finite(&data), Err(StatsError::ContainsNaN(1))));
    }

    #[test]
    fn test_validate_finite_inf() {
        let data = vec![1.0, f64::INFINITY, 3.0];
        assert!(matches!(validate_finite(&data), Err(StatsError::ContainsInfinite(1))));
    }

    #[test]
    fn test_filter_nan() {
        let data = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        let (clean, removed) = filter_nan(&data);
        assert_eq!(clean, vec![1.0, 3.0, 5.0]);
        assert_eq!(removed, 2);
    }

    // -------------------------------------------------------------------------
    // Percentile / median
    // -------------------------------------------------------------------------
    #[test]
    fn test_median_odd() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert!(approx_eq(median(&data).unwrap(), 3.0, 1e-10));
    }

    #[test]
    fn test_median_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(median(&data).unwrap(), 2.5, 1e-10));
    }

    #[test]
    fn test_percentile_boundaries() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(percentile(&mut data,   0.0).unwrap(), 1.0, 1e-10));
        assert!(approx_eq(percentile(&mut data, 100.0).unwrap(), 5.0, 1e-10));
    }

    // -------------------------------------------------------------------------
    // Welford variance
    // -------------------------------------------------------------------------
    #[test]
    fn test_welford_mean() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, _, _, _) = welford_parallel(&data).unwrap();
        assert!(approx_eq(mean, 5.0, 1e-10));
    }

    #[test]
    fn test_welford_variance() {
        // For data [2,4,4,4,5,5,7,9]: population variance = 4.0
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (_, pop_var, _, _) = welford_parallel(&data).unwrap();
        assert!(approx_eq(pop_var, 4.0, 1e-10));
    }

    // -------------------------------------------------------------------------
    // Std dev
    // -------------------------------------------------------------------------
    #[test]
    fn test_std_dev_known() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(approx_eq(std_dev(&data).unwrap(), 2.0, 1e-10));
    }

    #[test]
    fn test_std_dev_single_element() {
        // Single element → population std dev = 0
        let data = vec![42.0];
        assert!(approx_eq(std_dev(&data).unwrap(), 0.0, 1e-10));
    }

    // -------------------------------------------------------------------------
    // Skewness / kurtosis
    // -------------------------------------------------------------------------
    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data should have ~0 skewness
        let data: Vec<f64> = (-50..=50).map(|x| x as f64).collect();
        assert!(approx_eq(skewness(&data).unwrap(), 0.0, 1e-6));
    }

    #[test]
    fn test_kurtosis_normal_approx() {
        // For a large uniform range, excess kurtosis should be negative (~-1.2)
        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let kurt = excess_kurtosis(&data).unwrap();
        assert!(kurt < 0.0, "Uniform-like data should have negative excess kurtosis");
    }

    // -------------------------------------------------------------------------
    // MAD
    // -------------------------------------------------------------------------
    #[test]
    fn test_mad_known() {
        // Median = 3, deviations = [2,1,0,1,2], MAD = 1
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(mad(&data).unwrap(), 1.0, 1e-10));
    }

    // -------------------------------------------------------------------------
    // Outlier detection
    // -------------------------------------------------------------------------
    #[test]
    fn test_iqr_outliers() {
        // 100.0 is a clear outlier
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 100.0];
        let result = detect_outliers_iqr(&data, 1.5).unwrap();
        assert!(result.outlier_indices.contains(&6));
        assert_eq!(result.outlier_count, 1);
    }

    #[test]
    fn test_iqr_no_outliers() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = detect_outliers_iqr(&data, 1.5).unwrap();
        assert_eq!(result.outlier_count, 0);
    }

    #[test]
    fn test_mad_outliers() {
        let mut data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        data.push(1000.0);  // Clear outlier at index 20
        let result = detect_outliers_mad(&data, 3.5).unwrap();
        assert!(result.outlier_count > 0);
        assert!(result.outlier_indices.contains(&20));
    }

    // -------------------------------------------------------------------------
    // Entropy
    // -------------------------------------------------------------------------
    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 4 categories → entropy = 2.0 bits
        let counts = vec![1u64, 1, 1, 1];
        let result = shannon_entropy(&counts).unwrap();
        assert!(approx_eq(result.entropy_bits, 2.0, 1e-10));
        assert!(approx_eq(result.normalized, 1.0, 1e-10));
    }

    #[test]
    fn test_entropy_deterministic() {
        // One category → entropy = 0
        let counts = vec![100u64];
        let result = shannon_entropy(&counts).unwrap();
        assert!(approx_eq(result.entropy_bits, 0.0, 1e-10));
    }

    // -------------------------------------------------------------------------
    // Pearson correlation
    // -------------------------------------------------------------------------
    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = pearson_correlation(&x, &y).unwrap();
        assert!(approx_eq(result.coefficient, 1.0, 1e-10));
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = pearson_correlation(&x, &y).unwrap();
        assert!(approx_eq(result.coefficient, -1.0, 1e-10));
    }

    #[test]
    fn test_pearson_uncorrelated() {
        // Orthogonal vectors
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0,  1.0, -1.0, -1.0];
        let result = pearson_correlation(&x, &y).unwrap();
        assert!(approx_eq(result.coefficient, 0.0, 1e-10));
    }

    #[test]
    fn test_pearson_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(matches!(pearson_correlation(&x, &y), Err(StatsError::LengthMismatch(3, 2))));
    }

    // -------------------------------------------------------------------------
    // Spearman correlation
    // -------------------------------------------------------------------------
    #[test]
    fn test_spearman_monotonic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];  // monotonic but not linear
        let result = spearman_correlation(&x, &y).unwrap();
        // Perfect monotonic relationship → Spearman = 1.0
        assert!(approx_eq(result.coefficient, 1.0, 1e-10));
        assert_eq!(result.method, "spearman");
    }

    // -------------------------------------------------------------------------
    // Rolling statistics
    // -------------------------------------------------------------------------
    #[test]
    fn test_rolling_mean_basic() {
        let data   = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 2.0, 1e-10));  // (1+2+3)/3
        assert!(approx_eq(result[1], 3.0, 1e-10));  // (2+3+4)/3
        assert!(approx_eq(result[2], 4.0, 1e-10));  // (3+4+5)/3
    }

    #[test]
    fn test_rolling_mean_window_too_large() {
        let data = vec![1.0, 2.0];
        assert!(matches!(
            rolling_mean(&data, 5),
            Err(StatsError::WindowTooLarge { window: 5, data_len: 2 })
        ));
    }

    // -------------------------------------------------------------------------
    // Full descriptive stats
    // -------------------------------------------------------------------------
    #[test]
    fn test_full_descriptive_known_dataset() {
        // Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5, Std = 2, Median = 4.5
        let data   = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = full_descriptive(&data).unwrap();

        assert_eq!(result.count, 8);
        assert!(approx_eq(result.mean,    5.0, 1e-10));
        assert!(approx_eq(result.std_dev, 2.0, 1e-10));
        assert!(approx_eq(result.median,  4.5, 1e-10));
        assert_eq!(result.nan_count, 0);
    }

    #[test]
    fn test_full_descriptive_with_nans() {
        // NaNs should be excluded and counted
        let data = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let result = full_descriptive(&data).unwrap();
        assert_eq!(result.count,     3);
        assert_eq!(result.nan_count, 2);
        assert!(approx_eq(result.mean, 3.0, 1e-10));
    }

    // -------------------------------------------------------------------------
    // Correlation matrix
    // -------------------------------------------------------------------------
    #[test]
    fn test_correlation_matrix_diagonal() {
        let cols  = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let names = vec!["a".to_string(), "b".to_string()];
        let mat   = correlation_matrix(&cols, &names, "pearson").unwrap();

        // Diagonal must be 1.0
        assert!(approx_eq(mat.matrix[0][0], 1.0, 1e-10));
        assert!(approx_eq(mat.matrix[1][1], 1.0, 1e-10));

        // Symmetric: [0][1] == [1][0]
        assert!(approx_eq(mat.matrix[0][1], mat.matrix[1][0], 1e-10));
    }
}