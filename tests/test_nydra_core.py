"""
test_nydra_core.py — Tests for the nydra_core hybrid statistical engine.

nydra_core.py is a Python fallback bridge: every public function routes to
the compiled Rust extension if present (HAS_RUST=True), else to a pure
NumPy/SciPy implementation. These tests exercise the PUBLIC functions only,
so they are valid and meaningful whether or not the Rust binary is compiled
on the machine running CI.

ASSUMPTION TO CONFIRM: this file imports `from src import nydra_core as core`.
If the real project places this module elsewhere (e.g. a `nydra_core/`
package at the repo root rather than `src/nydra_core.py`), update the
import below — everything else in this file is import-path-agnostic.

Run with:
    pytest tests/test_nydra_core.py -v
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src import nydra_core as core


# ── Descriptive stats: happy path ───────────────────────────────────────────

class TestDescriptiveStats:
    def test_mean(self, simple_array):
        assert core.mean(simple_array) == pytest.approx(5.0)

    def test_median(self, simple_array):
        assert core.median(simple_array) == pytest.approx(4.5)

    def test_std_dev_population_vs_sample(self, simple_array):
        """Population std (ddof=0) must be strictly less than sample std
        (ddof=1) for n>1 — a classic off-by-one that's easy to get backwards."""
        pop = core.std_dev(simple_array)
        sample = core.std_dev_sample(simple_array)
        assert pop < sample

    def test_variance_is_std_dev_squared(self, simple_array):
        """Cross-check invariant: variance == std_dev ** 2, both population
        and sample flavors. Catches a std/variance mix-up bug directly."""
        assert core.variance(simple_array) == pytest.approx(core.std_dev(simple_array) ** 2)
        assert core.variance_sample(simple_array) == pytest.approx(core.std_dev_sample(simple_array) ** 2)

    def test_std_dev_sample_single_element_is_zero(self):
        """n=1 sample std is mathematically undefined (0/0); the
        implementation defines it as 0.0 rather than NaN or raising —
        lock that contract in explicitly."""
        assert core.std_dev_sample([42.0]) == 0.0
        assert core.variance_sample([42.0]) == 0.0

    def test_mad_scaled_matches_raw_mad_times_constant(self, simple_array):
        assert core.mad_scaled(simple_array) == pytest.approx(core.mad(simple_array) * 1.4826)

    def test_quartiles_matches_percentile_25_75(self, simple_array):
        q1, q3 = core.quartiles(simple_array)
        assert q1 == pytest.approx(core.percentile(simple_array, 25.0))
        assert q3 == pytest.approx(core.percentile(simple_array, 75.0))
        assert q1 <= q3

    def test_kahan_sum_matches_naive_sum_for_well_scaled_data(self, simple_array):
        assert core.kahan_sum(simple_array) == pytest.approx(sum(simple_array))

    def test_kahan_sum_more_accurate_than_naive_for_ill_conditioned_data(self):
        """The whole point of Kahan summation is catastrophic-cancellation
        resistance. This constructs a case where naive float sum visibly
        drifts and asserts Kahan sum is the more accurate one."""
        data = [1e16, 1.0, -1e16] * 1000
        naive = sum(data)
        kahan = core.kahan_sum(data)
        # True mathematical answer is exactly 1000.0
        assert abs(kahan - 1000.0) <= abs(naive - 1000.0)


# ── Empty / degenerate input handling ───────────────────────────────────────

class TestEmptyAndDegenerateInputs:
    @pytest.mark.parametrize("fn_name", [
        "std_dev", "std_dev_sample", "mean", "median", "variance",
        "variance_sample", "mad", "mad_scaled", "kahan_sum",
    ])
    def test_empty_input_raises_valueerror(self, fn_name):
        fn = getattr(core, fn_name)
        if fn_name == "kahan_sum":
            # kahan_sum has no explicit empty check in the fallback —
            # math.fsum([]) legitimately returns 0.0, not an error.
            assert fn([]) == 0.0
            return
        with pytest.raises(ValueError):
            fn([])

    def test_percentile_empty_raises(self):
        with pytest.raises(ValueError):
            core.percentile([], 50.0)

    def test_percentile_out_of_range_raises(self, simple_array):
        with pytest.raises(ValueError):
            core.percentile(simple_array, 150.0)
        with pytest.raises(ValueError):
            core.percentile(simple_array, -5.0)

    def test_skewness_zero_variance_raises(self, constant_array):
        """Skewness/kurtosis are mathematically undefined when std==0
        (division by zero); the implementation must raise, not return NaN
        silently or crash with an unrelated exception."""
        with pytest.raises(ValueError):
            core.skewness(constant_array)

    def test_kurtosis_zero_variance_raises(self, constant_array):
        with pytest.raises(ValueError):
            core.excess_kurtosis(constant_array)

    def test_std_dev_zero_variance_is_zero_not_error(self, constant_array):
        """Unlike skewness/kurtosis, std_dev of a constant array is a
        perfectly well-defined 0.0 — must NOT raise."""
        assert core.std_dev(constant_array) == 0.0

    def test_detect_outliers_mad_zero_mad_returns_no_outliers(self, constant_array):
        """When MAD==0 (identical values), the fallback special-cases this
        to avoid division by zero and returns zero outliers rather than
        raising or returning inf-based fences."""
        result = core.detect_outliers_mad(constant_array)
        assert result["outlier_count"] == 0
        assert result["outlier_indices"] == []

    def test_full_descriptive_empty_raises(self):
        with pytest.raises(ValueError):
            core.full_descriptive([])

    def test_full_descriptive_all_nan_raises(self):
        """Data that's entirely NaN/inf should raise after filtering,
        not silently return a descriptive dict of zeros."""
        with pytest.raises(ValueError):
            core.full_descriptive([float("nan"), float("inf"), float("-inf")])

    def test_detect_outliers_iqr_empty_raises(self):
        with pytest.raises(ValueError):
            core.detect_outliers_iqr([])

    def test_shannon_entropy_empty_raises(self):
        with pytest.raises(ValueError):
            core.shannon_entropy([])

    def test_shannon_entropy_all_zero_counts_raises(self):
        with pytest.raises(ValueError):
            core.shannon_entropy([0, 0, 0])

    def test_entropy_numeric_invalid_bins_raises(self, simple_array):
        with pytest.raises(ValueError):
            core.entropy_numeric(simple_array, n_bins=0)


# ── NaN / infinity handling ──────────────────────────────────────────────────

class TestNanAndInfHandling:
    def test_filter_nan_removes_nan_and_inf(self, array_with_nan):
        clean, removed = core.filter_nan(array_with_nan)
        assert removed == 2  # one nan, one inf
        assert all(math.isfinite(x) for x in clean)
        assert len(clean) == len(array_with_nan) - 2

    def test_filter_nan_no_nan_removes_nothing(self, simple_array):
        clean, removed = core.filter_nan(simple_array)
        assert removed == 0
        assert clean == simple_array

    def test_full_descriptive_reports_nan_count(self, array_with_nan):
        """full_descriptive must both compute over the cleaned data AND
        accurately report how many values it dropped — both halves of
        the contract, not just the numeric result."""
        result = core.full_descriptive(array_with_nan)
        assert result["nan_count"] == 2
        assert result["count"] == len(array_with_nan) - 2


# ── Correlation ──────────────────────────────────────────────────────────────

class TestCorrelation:
    def test_pearson_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = core.pearson_correlation(x, y)
        assert result["coefficient"] == pytest.approx(1.0)
        assert result["is_significant"] is True

    def test_pearson_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = core.pearson_correlation(x, y)
        assert result["coefficient"] == pytest.approx(-1.0)

    def test_pearson_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            core.pearson_correlation([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_pearson_empty_raises(self):
        with pytest.raises(ValueError):
            core.pearson_correlation([], [])

    def test_pearson_zero_variance_input_raises(self, constant_array):
        """Correlating a constant array against anything is undefined
        (0/0 in the correlation formula) — scipy returns NaN, and the
        wrapper must convert that into a raised ValueError rather than
        leaking a NaN into downstream reports."""
        with pytest.raises(ValueError):
            core.pearson_correlation(constant_array, list(range(len(constant_array))))

    def test_spearman_monotonic_nonlinear_relationship(self):
        """Spearman should catch a perfect monotonic-but-nonlinear
        relationship that Pearson would underrate — the whole reason
        both methods exist side by side."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]  # y = x^2, monotonic not linear
        spearman = core.spearman_correlation(x, y)
        pearson = core.pearson_correlation(x, y)
        assert spearman["coefficient"] == pytest.approx(1.0)
        assert pearson["coefficient"] < 1.0

    def test_correlation_matrix_diagonal_is_one(self):
        cols = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
        names = ["a", "b", "c"]
        result = core.correlation_matrix(cols, names, method="pearson")
        for i in range(3):
            assert result["matrix"][i][i] == pytest.approx(1.0)

    def test_correlation_matrix_is_symmetric(self):
        cols = [[1.0, 2.0, 3.0, 5.0], [4.0, 1.0, 2.0, 1.0], [1.0, 1.0, 2.0, 9.0]]
        names = ["a", "b", "c"]
        result = core.correlation_matrix(cols, names, method="pearson")
        m = result["matrix"]
        for i in range(3):
            for j in range(3):
                assert m[i][j] == pytest.approx(m[j][i])

    def test_correlation_matrix_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            core.correlation_matrix([[1.0, 2.0], [3.0, 4.0]], ["only_one_name"])


# ── Outlier detection ────────────────────────────────────────────────────────

class TestOutlierDetection:
    def test_iqr_detects_obvious_outlier(self, array_with_outlier):
        result = core.detect_outliers_iqr(array_with_outlier)
        assert result["outlier_count"] >= 1
        assert (len(array_with_outlier) - 1) in result["outlier_indices"]  # index of 1000.0

    def test_iqr_no_outliers_in_clean_data(self, no_outlier_array):
        result = core.detect_outliers_iqr(no_outlier_array)
        assert result["outlier_count"] == 0

    def test_mad_detects_obvious_outlier(self, array_with_outlier_nonzero_mad):
        result = core.detect_outliers_mad(array_with_outlier_nonzero_mad)
        assert result["outlier_count"] >= 1

    def test_detect_outliers_mad_majority_tie_fallback(self, array_with_outlier):
        """FIXED BEHAVIOR: when a majority of values tie (here, four 2.0's
        out of seven values), MAD == 0, which used to make detect_outliers_mad
        silently return ZERO outliers even with an obvious outlier present.
        It now falls back to IQR in this case, matching IQR's own result
        (see test_iqr_detects_obvious_outlier) instead of missing it."""
        mad_result = core.detect_outliers_mad(array_with_outlier)
        iqr_result = core.detect_outliers_iqr(array_with_outlier)
        assert core.mad(array_with_outlier) == 0.0
        assert mad_result["outlier_count"] == iqr_result["outlier_count"]
        assert mad_result["outlier_indices"] == iqr_result["outlier_indices"]
        assert "fallback" in mad_result["method"].lower()

    def test_iqr_larger_k_finds_fewer_or_equal_outliers(self, array_with_outlier):
        """A wider fence (bigger k) should never flag MORE outliers than
        a narrower one — monotonicity check on the threshold parameter."""
        narrow = core.detect_outliers_iqr(array_with_outlier, k=1.0)
        wide = core.detect_outliers_iqr(array_with_outlier, k=5.0)
        assert wide["outlier_count"] <= narrow["outlier_count"]


# ── Entropy ───────────────────────────────────────────────────────────────

class TestEntropy:
    def test_shannon_entropy_uniform_distribution_is_max(self):
        """Maximum entropy occurs for a uniform distribution — normalized
        entropy should be exactly 1.0 (within float tolerance)."""
        result = core.shannon_entropy([10, 10, 10, 10])
        assert result["normalized"] == pytest.approx(1.0, abs=1e-9)

    def test_shannon_entropy_single_value_is_zero(self):
        """All mass on one outcome = zero uncertainty = zero entropy."""
        result = core.shannon_entropy([100, 0, 0, 0])
        assert result["entropy_bits"] == pytest.approx(0.0, abs=1e-9)

    def test_entropy_from_strings_matches_manual_counts(self):
        values = ["a", "a", "b", "c", "c", "c"]
        result = core.entropy_from_strings(values)
        assert result["n_total"] == 6
        assert result["n_unique"] == 3

    def test_entropy_numeric_respects_bin_count(self):
        data = list(np.linspace(0, 100, 200))
        result = core.entropy_numeric(data, n_bins=5)
        assert result["n_unique"] == 5


# ── Rolling window functions ─────────────────────────────────────────────────

class TestRollingWindow:
    def test_rolling_mean_window_larger_than_data_raises(self):
        with pytest.raises(ValueError):
            core.rolling_mean([1.0, 2.0, 3.0], window=10)

    def test_rolling_std_window_larger_than_data_raises(self):
        with pytest.raises(ValueError):
            core.rolling_std([1.0, 2.0, 3.0], window=10)

    def test_rolling_mean_output_length(self):
        data = list(range(1, 11))  # 10 points
        result = core.rolling_mean([float(x) for x in data], window=3)
        assert len(result) == 10 - 3 + 1

    def test_rolling_mean_constant_data_returns_constant(self):
        data = [7.0] * 10
        result = core.rolling_mean(data, window=4)
        assert all(x == pytest.approx(7.0) for x in result)

    def test_rolling_window_equal_to_data_length_returns_single_value(self):
        data = [1.0, 2.0, 3.0, 4.0]
        result = core.rolling_mean(data, window=4)
        assert len(result) == 1
        assert result[0] == pytest.approx(2.5)


# ── Property-based invariant tests (Hypothesis) ─────────────────────────────

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32,
)
non_trivial_arrays = st.lists(finite_floats, min_size=2, max_size=50)


class TestPropertyInvariants:
    """These don't check exact values — they check mathematical
    invariants that must hold for ANY valid input. Property-based tests
    catch edge cases (extreme magnitudes, unusual distributions) that
    hand-written examples never think to try."""

    @given(data=non_trivial_arrays)
    @settings(max_examples=100, deadline=None)
    def test_std_dev_always_non_negative(self, data):
        assert core.std_dev(data) >= 0.0

    @given(data=non_trivial_arrays)
    @settings(max_examples=100, deadline=None)
    def test_variance_equals_std_dev_squared(self, data):
        assert core.variance(data) == pytest.approx(core.std_dev(data) ** 2, rel=1e-4, abs=1e-9)

    @given(data=non_trivial_arrays)
    @settings(max_examples=100, deadline=None)
    def test_min_le_median_le_max(self, data):
        assert min(data) <= core.median(data) <= max(data)

    @given(data=non_trivial_arrays)
    @settings(max_examples=100, deadline=None)
    def test_q1_le_q3(self, data):
        q1, q3 = core.quartiles(data)
        assert q1 <= q3

    @given(
        x=st.lists(finite_floats, min_size=3, max_size=30),
    )
    @settings(max_examples=50, deadline=None)
    def test_pearson_self_correlation_is_one(self, x):
        """Any array (with non-zero variance) correlated against itself
        must have a coefficient of exactly 1.0."""
        if len(set(x)) < 2:
            return  # zero-variance input is a separately-tested error case
        result = core.pearson_correlation(x, x)
        assert result["coefficient"] == pytest.approx(1.0, abs=1e-6)

    @given(data=st.lists(finite_floats, min_size=1, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_filter_nan_never_increases_length(self, data):
        clean, removed = core.filter_nan(data)
        assert len(clean) + removed == len(data)
        assert len(clean) <= len(data)