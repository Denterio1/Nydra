"""
advanced_stats.py — Nydra v0.5.0+
========================================
Comprehensive statistical analysis module covering:

  DescriptiveStats         : Full descriptive statistics beyond pandas
  NormalityTester          : Shapiro-Wilk, D'Agostino, Anderson-Darling,
                             KS, Jarque-Bera, Lilliefors
  DistributionFitter       : Fits 15+ distributions, finds best fit (AIC/BIC/KS)
  SkewnessKurtosisAnalyzer : Skewness, kurtosis, excess kurtosis, tail analysis
  CorrelationAnalyzer      : Pearson, Spearman, Kendall + significance tests
                             Partial correlations, Cramér's V, Point-biserial
  HypothesisTester         : t-test, Mann-Whitney, ANOVA, Kruskal-Wallis,
                             Levene, Bartlett, Chi-squared, Fisher
  OutlierStatsTester       : Grubbs, Dixon Q, Extreme Value (GEV)
  TimeSeriesStats          : Stationarity, autocorrelation, Granger causality
  TransformationAdvisor    : Recommends best transformation per column
  StatsReporter            : Full audit report + recommendations
  AdvancedStats            : Master class — compute_all() does everything

Author  : Nydra Project
Version : 0.5.0
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, anderson, kstest, jarque_bera,
    pearsonr, spearmanr, kendalltau, pointbiserialr,
    levene, bartlett, chi2_contingency, f_oneway,
    mannwhitneyu, kruskal, ttest_ind, ttest_1samp,
    fisher_exact, iqr, trim_mean, mstats,
    boxcox, yeojohnson,
)
from scipy.stats import rv_continuous

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nydra.stats")

# Optional statsmodels for advanced tests
try:
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson, jarque_bera as sm_jb
    from statsmodels.stats.diagnostic import lilliefors
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
    from statsmodels.stats.multicomp import multipletests
    STATSMODELS = True
except ImportError:
    STATSMODELS = False

# Significance levels
ALPHA_STRICT  = 0.01
ALPHA_STANDARD = 0.05
ALPHA_RELAXED  = 0.10


# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────

@dataclass
class NormalityResult:
    """Result of normality tests for a single column."""
    column: str
    n: int
    shapiro_stat: Optional[float] = None
    shapiro_p: Optional[float] = None
    dagostino_stat: Optional[float] = None
    dagostino_p: Optional[float] = None
    anderson_stat: Optional[float] = None
    anderson_critical: Optional[float] = None   # at 5% level
    ks_stat: Optional[float] = None
    ks_p: Optional[float] = None
    jb_stat: Optional[float] = None
    jb_p: Optional[float] = None
    lilliefors_stat: Optional[float] = None
    lilliefors_p: Optional[float] = None
    is_normal: bool = False
    confidence: str = "low"   # low | medium | high
    verdict: str = ""
    recommendation: str = ""

    def summary(self) -> Dict:
        return {
            "column": self.column,
            "n": self.n,
            "is_normal": self.is_normal,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "tests": {
                "shapiro_wilk": {"stat": self.shapiro_stat, "p": self.shapiro_p},
                "dagostino_pearson": {"stat": self.dagostino_stat, "p": self.dagostino_p},
                "anderson_darling": {"stat": self.anderson_stat, "critical_5pct": self.anderson_critical},
                "kolmogorov_smirnov": {"stat": self.ks_stat, "p": self.ks_p},
                "jarque_bera": {"stat": self.jb_stat, "p": self.jb_p},
                "lilliefors": {"stat": self.lilliefors_stat, "p": self.lilliefors_p},
            },
            "recommendation": self.recommendation,
        }


@dataclass
class DistributionFitResult:
    """Best-fit distribution for a column."""
    column: str
    best_distribution: str
    best_params: Tuple
    aic: float
    bic: float
    ks_stat: float
    ks_p: float
    all_fits: List[Dict] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "column": self.column,
            "best_fit": self.best_distribution,
            "params": self.best_params,
            "aic": round(self.aic, 4),
            "bic": round(self.bic, 4),
            "ks_stat": round(self.ks_stat, 4),
            "ks_p": round(self.ks_p, 4),
            "top_3": self.all_fits[:3],
        }


@dataclass
class CorrelationResult:
    """Correlation between two columns."""
    col_a: str
    col_b: str
    method: str
    coefficient: float
    p_value: float
    is_significant: bool
    effect_size: str   # negligible | small | medium | large

    def summary(self) -> Dict:
        return {
            "columns": f"{self.col_a} ↔ {self.col_b}",
            "method": self.method,
            "r": round(self.coefficient, 4),
            "p": round(self.p_value, 4),
            "significant": self.is_significant,
            "effect_size": self.effect_size,
        }


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def _clean_col(series: pd.Series) -> np.ndarray:
    """Drop NaN, return numpy array."""
    return series.dropna().values


def _effect_size_r(r: float) -> str:
    """Cohen's r interpretation."""
    r = abs(r)
    if r < 0.10: return "negligible"
    if r < 0.30: return "small"
    if r < 0.50: return "medium"
    return "large"


def _effect_size_cramers(v: float) -> str:
    if v < 0.10: return "negligible"
    if v < 0.30: return "small"
    if v < 0.50: return "medium"
    return "large"


def _aic(log_likelihood: float, k: int) -> float:
    return 2 * k - 2 * log_likelihood


def _bic(log_likelihood: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * log_likelihood


# ─────────────────────────────────────────────
#  1. DescriptiveStats
# ─────────────────────────────────────────────

class DescriptiveStats:
    """
    Comprehensive descriptive statistics beyond pandas.describe().
    Includes: trimmed mean, Winsorized mean, IQR, MAD, CV,
    confidence intervals, percentiles, L-moments.
    """

    def __init__(self, alpha: float = 0.05, trim: float = 0.1):
        self.alpha = alpha
        self.trim = trim

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute full descriptive stats for all numeric columns."""
        result = {}
        num_cols = df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            data = _clean_col(df[col])
            if len(data) < 3:
                continue
            result[col] = self._col_stats(col, data)

        result["__dataset__"] = self._dataset_stats(df)
        return result

    def _col_stats(self, col: str, data: np.ndarray) -> Dict:
        from src import nydra_core
        n = len(data)
        
        # Use nydra_core for full descriptive stats in a single pass (potentially parallel)
        core_stats = nydra_core.full_descriptive(data.tolist())
        
        mean = core_stats["mean"]
        median = core_stats["median"]
        std = core_stats["std_dev_sample"]
        var = core_stats["variance_sample"]
        sem = float(stats.sem(data))

        # Robust measures
        trimmed_mean = float(trim_mean(data, self.trim))
        wins_mean = float(stats.mstats.winsorize(data, limits=[0.05, 0.05]).mean())
        mad = core_stats["mad"]
        iqr_val = core_stats["iqr"]
        cv = core_stats["cv"]

        # Moments
        skewness = core_stats["skewness"]
        kurt = core_stats["kurtosis"]            # excess kurtosis
        kurt_type = self._kurtosis_type(kurt)

        # Confidence interval for mean
        ci_low, ci_high = stats.t.interval(
            1 - self.alpha, df=n - 1, loc=mean, scale=sem
        )

        # Percentiles
        percentiles = {
            f"p{p}": float(np.percentile(data, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        # Range stats
        data_range = float(data.max() - data.min())
        gini = self._gini(data)

        return {
            "n": n,
            "mean": round(mean, 6),
            "median": round(median, 6),
            "mode": float(stats.mode(data, keepdims=True).mode[0]),
            "std": round(std, 6),
            "variance": round(var, 6),
            "sem": round(sem, 6),
            "cv_%": round(cv, 4),
            "trimmed_mean_10pct": round(trimmed_mean, 6),
            "winsorized_mean": round(wins_mean, 6),
            "mad": round(mad, 6),
            "iqr": round(iqr_val, 6),
            "range": round(data_range, 6),
            "min": float(data.min()),
            "max": float(data.max()),
            "skewness": round(skewness, 4),
            "kurtosis_excess": round(kurt, 4),
            "kurtosis_type": kurt_type,
            "skewness_type": self._skewness_type(skewness),
            "ci_95_low": round(float(ci_low), 4),
            "ci_95_high": round(float(ci_high), 4),
            "gini_coefficient": round(gini, 4),
            **{f"percentile_{k}": round(v, 4) for k, v in percentiles.items()},
        }

    def _dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Dataset-level summary statistics."""
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(df.select_dtypes(include=["object", "category"]).columns),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_rate": round(float(df.isnull().mean().mean()), 4),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "duplicates": int(df.duplicated().sum()),
        }

    def _kurtosis_type(self, k: float) -> str:
        if k > 1.0:   return "leptokurtic (heavy tails)"
        if k < -1.0:  return "platykurtic (light tails)"
        return "mesokurtic (normal-like)"

    def _skewness_type(self, s: float) -> str:
        if s > 1.0:    return "highly_right_skewed"
        if s > 0.5:    return "moderately_right_skewed"
        if s < -1.0:   return "highly_left_skewed"
        if s < -0.5:   return "moderately_left_skewed"
        return "approximately_symmetric"

    def _gini(self, data: np.ndarray) -> float:
        """Gini coefficient — measure of inequality."""
        if len(data) == 0 or data.min() < 0:
            return np.nan
        sorted_data = np.sort(data)
        n = len(sorted_data)
        cumsum = np.cumsum(sorted_data)
        return float((2 * cumsum - sorted_data).sum() / (n * sorted_data.sum()) - (n + 1) / n)


# ─────────────────────────────────────────────
#  2. NormalityTester
# ─────────────────────────────────────────────

class NormalityTester:
    """
    Comprehensive normality testing using 6 tests.
    Combines results into a confidence-weighted verdict.

    Tests:
    - Shapiro-Wilk       : Best for n < 2000
    - D'Agostino-Pearson : Combines skew + kurtosis
    - Anderson-Darling   : Most sensitive to tails
    - Kolmogorov-Smirnov : Non-parametric, distribution-free
    - Jarque-Bera        : Econometrics standard
    - Lilliefors         : KS variant with estimated params
    """

    def __init__(self, alpha: float = 0.05, min_n: int = 8):
        self.alpha = alpha
        self.min_n = min_n

    def test_column(self, series: pd.Series) -> NormalityResult:
        data = _clean_col(series)
        n = len(data)
        result = NormalityResult(column=str(series.name), n=n)

        if n < self.min_n:
            result.verdict = f"Too few observations (n={n}, need ≥{self.min_n})"
            return result

        votes_normal = 0
        total_tests = 0

        # 1. Shapiro-Wilk (best for n < 5000)
        if n <= 5000:
            try:
                stat, p = shapiro(data)
                result.shapiro_stat = round(float(stat), 4)
                result.shapiro_p = round(float(p), 4)
                total_tests += 1
                if p > self.alpha:
                    votes_normal += 1
            except Exception:
                pass

        # 2. D'Agostino-Pearson
        try:
            stat, p = normaltest(data)
            result.dagostino_stat = round(float(stat), 4)
            result.dagostino_p = round(float(p), 4)
            total_tests += 1
            if p > self.alpha:
                votes_normal += 1
        except Exception:
            pass

        # 3. Anderson-Darling
        try:
            ad_result = anderson(data, dist="norm")
            result.anderson_stat = round(float(ad_result.statistic), 4)
            # Critical value at 5%
            idx_5pct = list(ad_result.significance_level).index(5.0) \
                if 5.0 in ad_result.significance_level else 2
            result.anderson_critical = round(float(ad_result.critical_values[idx_5pct]), 4)
            total_tests += 1
            if ad_result.statistic < ad_result.critical_values[idx_5pct]:
                votes_normal += 1
        except Exception:
            pass

        # 4. Kolmogorov-Smirnov (against fitted normal)
        try:
            mu, sigma = float(np.mean(data)), float(np.std(data))
            stat, p = kstest(data, "norm", args=(mu, sigma))
            result.ks_stat = round(float(stat), 4)
            result.ks_p = round(float(p), 4)
            total_tests += 1
            if p > self.alpha:
                votes_normal += 1
        except Exception:
            pass

        # 5. Jarque-Bera
        try:
            stat, p = jarque_bera(data)
            result.jb_stat = round(float(stat), 4)
            result.jb_p = round(float(p), 4)
            total_tests += 1
            if p > self.alpha:
                votes_normal += 1
        except Exception:
            pass

        # 6. Lilliefors
        if STATSMODELS:
            try:
                stat, p = lilliefors(data)
                result.lilliefors_stat = round(float(stat), 4)
                result.lilliefors_p = round(float(p), 4)
                total_tests += 1
                if p > self.alpha:
                    votes_normal += 1
            except Exception:
                pass

        # ── Verdict ──
        if total_tests == 0:
            result.verdict = "No tests could be run"
            return result

        vote_ratio = votes_normal / total_tests
        result.is_normal = vote_ratio >= 0.5

        if vote_ratio >= 0.80:
            result.confidence = "high"
        elif vote_ratio >= 0.50:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        result.verdict = (
            f"Normal (confidence: {result.confidence}, "
            f"{votes_normal}/{total_tests} tests pass)"
        ) if result.is_normal else (
            f"NOT normal (confidence: {result.confidence}, "
            f"only {votes_normal}/{total_tests} tests pass)"
        )

        result.recommendation = self._recommendation(result)
        return result

    def test_all(self, df: pd.DataFrame) -> Dict[str, NormalityResult]:
        results = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            results[col] = self.test_column(df[col])
        return results

    def summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.test_all(df)
        rows = []
        for col, r in results.items():
            rows.append({
                "Column": col,
                "N": r.n,
                "Normal": "✅" if r.is_normal else "❌",
                "Confidence": r.confidence,
                "Shapiro p": r.shapiro_p,
                "D'Agostino p": r.dagostino_p,
                "KS p": r.ks_p,
                "JB p": r.jb_p,
                "Verdict": r.verdict,
            })
        return pd.DataFrame(rows)

    def _recommendation(self, r: NormalityResult) -> str:
        if r.is_normal:
            return "Parametric tests (t-test, ANOVA, Pearson) are appropriate."
        return (
            "Use non-parametric alternatives: "
            "Mann-Whitney U instead of t-test, "
            "Spearman/Kendall instead of Pearson, "
            "Kruskal-Wallis instead of ANOVA. "
            "Consider log/sqrt/Box-Cox transformation."
        )


# ─────────────────────────────────────────────
#  3. DistributionFitter
# ─────────────────────────────────────────────

class DistributionFitter:
    """
    Fits 15+ theoretical distributions to each column.
    Ranks by AIC, BIC, and KS test.
    Identifies the best-fitting distribution.
    """

    DISTRIBUTIONS = [
        "norm", "lognorm", "expon", "gamma", "beta",
        "weibull_min", "weibull_max", "pareto", "t",
        "laplace", "logistic", "cauchy", "chi2",
        "rayleigh", "exponweib", "gumbel_r", "gumbel_l",
        "uniform", "triang", "burr",
    ]

    def __init__(self, top_n: int = 5, min_n: int = 20):
        self.top_n = top_n
        self.min_n = min_n

    def fit_column(self, series: pd.Series) -> DistributionFitResult:
        data = _clean_col(series)
        n = len(data)
        col = str(series.name)

        if n < self.min_n:
            return DistributionFitResult(
                column=col, best_distribution="insufficient_data",
                best_params=(), aic=np.nan, bic=np.nan,
                ks_stat=np.nan, ks_p=np.nan,
            )

        fits = []
        for dist_name in self.DISTRIBUTIONS:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                k = len(params)

                # Log-likelihood
                log_likelihood = float(np.sum(dist.logpdf(data, *params)))
                if not np.isfinite(log_likelihood):
                    continue

                aic = _aic(log_likelihood, k)
                bic = _bic(log_likelihood, k, n)

                # KS goodness-of-fit
                ks_stat, ks_p = kstest(data, dist_name, args=params)

                fits.append({
                    "distribution": dist_name,
                    "params": params,
                    "aic": round(aic, 4),
                    "bic": round(bic, 4),
                    "ks_stat": round(float(ks_stat), 4),
                    "ks_p": round(float(ks_p), 4),
                    "log_likelihood": round(log_likelihood, 4),
                })
            except Exception:
                continue

        if not fits:
            return DistributionFitResult(
                column=col, best_distribution="fit_failed",
                best_params=(), aic=np.nan, bic=np.nan,
                ks_stat=np.nan, ks_p=np.nan,
            )

        # Rank by AIC (lower = better)
        fits.sort(key=lambda x: x["aic"])
        best = fits[0]

        return DistributionFitResult(
            column=col,
            best_distribution=best["distribution"],
            best_params=best["params"],
            aic=best["aic"],
            bic=best["bic"],
            ks_stat=best["ks_stat"],
            ks_p=best["ks_p"],
            all_fits=[{
                "rank": i + 1,
                "distribution": f["distribution"],
                "aic": f["aic"],
                "bic": f["bic"],
                "ks_p": f["ks_p"],
            } for i, f in enumerate(fits[:self.top_n])],
        )

    def fit_all(self, df: pd.DataFrame) -> Dict[str, DistributionFitResult]:
        results = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            logger.info(f"[DistFitter] Fitting {col}...")
            results[col] = self.fit_column(df[col])
        return results

    def summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.fit_all(df)
        rows = []
        for col, r in results.items():
            rows.append({
                "Column": col,
                "Best Fit": r.best_distribution,
                "AIC": r.aic,
                "BIC": r.bic,
                "KS Stat": r.ks_stat,
                "KS p-value": r.ks_p,
                "Good Fit": "✅" if r.ks_p and r.ks_p > 0.05 else "❌",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  4. SkewnessKurtosisAnalyzer
# ─────────────────────────────────────────────

class SkewnessKurtosisAnalyzer:
    """
    Deep skewness and kurtosis analysis per column.
    Includes: excess kurtosis, tail heaviness,
    bimodality coefficient, transformation recommendations.
    """

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = _clean_col(df[col])
            if len(data) < 8:
                continue
            results[col] = self._analyze_col(col, data)
        return results

    def _analyze_col(self, col: str, data: np.ndarray) -> Dict:
        n = len(data)
        skew = float(stats.skew(data))
        kurt = float(stats.kurtosis(data))              # excess kurtosis
        kurt_fisher = float(stats.kurtosis(data, fisher=True))
        kurt_pearson = float(stats.kurtosis(data, fisher=False))  # Pearson = Fisher + 3

        # Skewness significance test
        skew_z = skew / np.sqrt(6 / n)
        skew_significant = abs(skew_z) > 1.96  # at 5% level

        # Kurtosis significance test
        kurt_z = kurt / np.sqrt(24 / n)
        kurt_significant = abs(kurt_z) > 1.96

        # Bimodality coefficient (BC > 0.555 suggests bimodality)
        bc = (skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))

        # Tail analysis
        z_scores = np.abs(stats.zscore(data))
        heavy_tail_ratio = float((z_scores > 3).mean())

        return {
            "n": n,
            "skewness": round(skew, 4),
            "skewness_type": self._skew_type(skew),
            "skewness_significant": skew_significant,
            "skewness_z_score": round(float(skew_z), 4),
            "kurtosis_excess": round(kurt, 4),
            "kurtosis_pearson": round(kurt_pearson, 4),
            "kurtosis_type": self._kurt_type(kurt),
            "kurtosis_significant": kurt_significant,
            "kurtosis_z_score": round(float(kurt_z), 4),
            "bimodality_coefficient": round(float(bc), 4),
            "possibly_bimodal": float(bc) > 0.555,
            "heavy_tail_ratio": round(heavy_tail_ratio, 4),
            "transformation_needed": abs(skew) > 0.5 or abs(kurt) > 2,
            "recommended_transformation": self._recommend_transform(skew, data),
        }

    def _skew_type(self, s: float) -> str:
        if s > 2.0:   return "extreme_right_skew"
        if s > 1.0:   return "high_right_skew"
        if s > 0.5:   return "moderate_right_skew"
        if s < -2.0:  return "extreme_left_skew"
        if s < -1.0:  return "high_left_skew"
        if s < -0.5:  return "moderate_left_skew"
        return "approximately_symmetric"

    def _kurt_type(self, k: float) -> str:
        if k > 3.0:    return "very_heavy_tails (extreme leptokurtic)"
        if k > 1.0:    return "heavy_tails (leptokurtic)"
        if k < -1.0:   return "light_tails (platykurtic)"
        return "normal_tails (mesokurtic)"

    def _recommend_transform(self, skew: float, data: np.ndarray) -> str:
        if abs(skew) < 0.5:
            return "none_needed"
        if skew > 0 and data.min() > 0:
            if skew > 2:
                return "log1p_transform"
            return "sqrt_or_log_transform"
        if skew < 0:
            return "reflect_then_log (negate + log)"
        return "box_cox_or_yeo_johnson"


# ─────────────────────────────────────────────
#  5. CorrelationAnalyzer
# ─────────────────────────────────────────────

class CorrelationAnalyzer:
    """
    Full correlation analysis:
    - Pearson, Spearman, Kendall (numeric pairs)
    - Point-biserial (numeric vs binary)
    - Cramér's V (categorical pairs)
    - Partial correlations (controlling for confounders)
    - Significance tests with p-values
    - Multiple testing correction (Bonferroni, FDR)
    """

    def __init__(self, alpha: float = 0.05, correction: str = "bonferroni"):
        self.alpha = alpha
        self.correction = correction  # 'bonferroni' | 'fdr_bh' | 'none'

    def compute_pairwise(
        self, df: pd.DataFrame, method: str = "all"
    ) -> pd.DataFrame:
        """
        Compute pairwise correlations for all numeric column pairs.
        Returns a DataFrame with coefficient, p-value, significance.
        """
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        pairs = [
            (num_cols[i], num_cols[j])
            for i in range(len(num_cols))
            for j in range(i + 1, len(num_cols))
        ]

        rows = []
        for col_a, col_b in pairs:
            data_a = _clean_col(df[col_a])
            data_b = _clean_col(df[col_b])
            min_n = min(len(data_a), len(data_b))
            data_a = data_a[:min_n]
            data_b = data_b[:min_n]

            if len(data_a) < 5:
                continue

            row = {"col_a": col_a, "col_b": col_b}

            if method in ("all", "pearson"):
                r, p = pearsonr(data_a, data_b)
                row.update({"pearson_r": round(float(r), 4), "pearson_p": round(float(p), 4)})

            if method in ("all", "spearman"):
                r, p = spearmanr(data_a, data_b)
                row.update({"spearman_r": round(float(r), 4), "spearman_p": round(float(p), 4)})

            if method in ("all", "kendall"):
                r, p = kendalltau(data_a, data_b)
                row.update({"kendall_tau": round(float(r), 4), "kendall_p": round(float(p), 4)})

            row["effect_size"] = _effect_size_r(row.get("pearson_r", 0))
            rows.append(row)

        result_df = pd.DataFrame(rows)

        # Multiple testing correction
        if not result_df.empty and self.correction != "none" and STATSMODELS:
            for p_col in ["pearson_p", "spearman_p", "kendall_p"]:
                if p_col in result_df.columns:
                    reject, p_corr, _, _ = multipletests(
                        result_df[p_col].fillna(1.0),
                        alpha=self.alpha,
                        method=self.correction,
                    )
                    result_df[f"{p_col}_corrected"] = p_corr.round(4)
                    result_df[f"{p_col.replace('_p', '')}_significant"] = reject

        return result_df

    def cramers_v(self, series_a: pd.Series, series_b: pd.Series) -> Dict:
        """Cramér's V for categorical-categorical correlation."""
        try:
            ct = pd.crosstab(series_a, series_b)
            chi2, p, dof, _ = chi2_contingency(ct)
            n = ct.values.sum()
            k = min(ct.shape) - 1
            v = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
            return {
                "cramers_v": round(v, 4),
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p), 4),
                "dof": int(dof),
                "significant": p < self.alpha,
                "effect_size": _effect_size_cramers(v),
            }
        except Exception as e:
            return {"error": str(e)}

    def point_biserial(self, binary_col: pd.Series, numeric_col: pd.Series) -> Dict:
        """Point-biserial correlation: binary vs numeric."""
        try:
            r, p = pointbiserialr(binary_col.dropna(), numeric_col.dropna())
            return {
                "r": round(float(r), 4),
                "p_value": round(float(p), 4),
                "significant": p < self.alpha,
                "effect_size": _effect_size_r(r),
            }
        except Exception as e:
            return {"error": str(e)}

    def partial_correlation(
        self, df: pd.DataFrame, col_a: str, col_b: str, control: str
    ) -> Dict:
        """
        Partial correlation between col_a and col_b,
        controlling for the 'control' variable.
        """
        try:
            data = df[[col_a, col_b, control]].dropna()
            # Residuals after regressing out control
            def residuals(y, x):
                x_ = sm.add_constant(x) if STATSMODELS else np.column_stack([np.ones(len(x)), x])
                if STATSMODELS:
                    model = sm.OLS(y, x_).fit()
                    return model.resid.values
                else:
                    beta = np.linalg.lstsq(x_, y, rcond=None)[0]
                    return y - x_ @ beta

            resid_a = residuals(data[col_a].values, data[control].values)
            resid_b = residuals(data[col_b].values, data[control].values)
            r, p = pearsonr(resid_a, resid_b)
            return {
                "partial_r": round(float(r), 4),
                "p_value": round(float(p), 4),
                "controlled_for": control,
                "significant": p < self.alpha,
            }
        except Exception as e:
            return {"error": str(e)}

    def correlation_matrix(
        self, df: pd.DataFrame, method: str = "pearson"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (correlation matrix, p-value matrix)."""
        num = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
        corr_matrix = num.corr(method=method)

        # P-value matrix
        cols = num.columns
        p_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), columns=cols, index=cols)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = num.iloc[:, i].values, num.iloc[:, j].values
                try:
                    if method == "pearson":
                        _, p = pearsonr(a, b)
                    elif method == "spearman":
                        _, p = spearmanr(a, b)
                    elif method == "kendall":
                        _, p = kendalltau(a, b)
                    else:
                        p = 1.0
                    p_matrix.iloc[i, j] = p
                    p_matrix.iloc[j, i] = p
                except Exception:
                    pass

        return corr_matrix, p_matrix


# ─────────────────────────────────────────────
#  6. HypothesisTester
# ─────────────────────────────────────────────

class HypothesisTester:
    """
    Comprehensive hypothesis testing suite.
    Auto-selects parametric or non-parametric based on normality.

    Covers:
    - One/two-sample t-test
    - Mann-Whitney U
    - One-way ANOVA
    - Kruskal-Wallis
    - Levene + Bartlett (variance equality)
    - Chi-squared independence
    - Fisher exact (small samples)
    - Welch's t-test (unequal variance)
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compare_groups(
        self, df: pd.DataFrame, numeric_col: str, group_col: str
    ) -> Dict:
        """
        Compare a numeric column across groups.
        Auto-selects: ANOVA vs Kruskal-Wallis.
        """
        groups = [
            _clean_col(grp[numeric_col])
            for _, grp in df.groupby(group_col)
            if len(grp[numeric_col].dropna()) >= 3
        ]
        if len(groups) < 2:
            return {"error": "Need at least 2 groups with n≥3"}

        n_groups = len(groups)

        # Check normality of each group
        all_normal = all(
            shapiro(g)[1] > self.alpha for g in groups if len(g) >= 8
        )

        # Check variance equality
        lev_stat, lev_p = levene(*groups)
        equal_variance = lev_p > self.alpha

        result = {
            "numeric_col": numeric_col,
            "group_col": group_col,
            "n_groups": n_groups,
            "group_sizes": [len(g) for g in groups],
            "all_normal": all_normal,
            "equal_variance": equal_variance,
            "levene_stat": round(float(lev_stat), 4),
            "levene_p": round(float(lev_p), 4),
        }

        if n_groups == 2:
            # Two groups
            g1, g2 = groups[0], groups[1]
            if all_normal and equal_variance:
                stat, p = ttest_ind(g1, g2)
                result.update({
                    "test": "independent_t_test",
                    "stat": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "significant": p < self.alpha,
                })
            elif all_normal and not equal_variance:
                stat, p = ttest_ind(g1, g2, equal_var=False)
                result.update({
                    "test": "welch_t_test",
                    "stat": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "significant": p < self.alpha,
                })
            else:
                stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
                result.update({
                    "test": "mann_whitney_u",
                    "stat": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "significant": p < self.alpha,
                })
            # Effect size (Cohen's d)
            result["effect_size_cohens_d"] = round(
                float(abs(np.mean(g1) - np.mean(g2)) /
                      np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2) / 2)), 4
            )
        else:
            # Multiple groups
            if all_normal and equal_variance:
                stat, p = f_oneway(*groups)
                result.update({
                    "test": "one_way_anova",
                    "stat": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "significant": p < self.alpha,
                })
                # Eta-squared
                grand_mean = np.concatenate(groups).mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                ss_total = sum(((g - grand_mean)**2).sum() for g in groups)
                result["eta_squared"] = round(float(ss_between / ss_total), 4)
            else:
                stat, p = kruskal(*groups)
                result.update({
                    "test": "kruskal_wallis",
                    "stat": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "significant": p < self.alpha,
                })

        result["interpretation"] = self._interpret(result)
        return result

    def chi_squared_test(
        self, df: pd.DataFrame, col_a: str, col_b: str
    ) -> Dict:
        """Chi-squared test of independence between two categorical columns."""
        ct = pd.crosstab(df[col_a], df[col_b])
        expected = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / ct.values.sum()
        use_fisher = ct.shape == (2, 2) and (expected < 5).any()

        if use_fisher:
            odds, p = fisher_exact(ct.values)
            return {
                "test": "fisher_exact",
                "odds_ratio": round(float(odds), 4),
                "p_value": round(float(p), 4),
                "significant": p < self.alpha,
            }
        else:
            chi2, p, dof, _ = chi2_contingency(ct)
            n = ct.values.sum()
            cramers = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
            return {
                "test": "chi_squared",
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p), 4),
                "dof": int(dof),
                "significant": p < self.alpha,
                "cramers_v": round(cramers, 4),
                "effect_size": _effect_size_cramers(cramers),
            }

    def one_sample_ttest(
        self, series: pd.Series, popmean: float
    ) -> Dict:
        """One-sample t-test against a population mean."""
        data = _clean_col(series)
        stat, p = ttest_1samp(data, popmean)
        return {
            "test": "one_sample_t_test",
            "sample_mean": round(float(data.mean()), 4),
            "popmean": popmean,
            "stat": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "significant": p < self.alpha,
        }

    def variance_equality(self, *groups: np.ndarray) -> Dict:
        """Test equality of variances: Levene + Bartlett."""
        result = {}
        lev_stat, lev_p = levene(*groups)
        result["levene"] = {
            "stat": round(float(lev_stat), 4),
            "p": round(float(lev_p), 4),
            "equal_variance": lev_p > self.alpha,
        }
        try:
            bart_stat, bart_p = bartlett(*groups)
            result["bartlett"] = {
                "stat": round(float(bart_stat), 4),
                "p": round(float(bart_p), 4),
                "equal_variance": bart_p > self.alpha,
            }
        except Exception:
            pass
        return result

    def _interpret(self, result: Dict) -> str:
        sig = result.get("significant", False)
        test = result.get("test", "test")
        p = result.get("p_value", 1.0)
        if sig:
            return (f"Significant difference found ({test}, p={p:.4f}). "
                    "Groups are statistically different.")
        return (f"No significant difference ({test}, p={p:.4f}). "
                "Groups are statistically similar.")


# ─────────────────────────────────────────────
#  7. OutlierStatsTester
# ─────────────────────────────────────────────

class OutlierStatsTester:
    """
    Statistical outlier tests (different from ML-based detection):
    - Grubbs test (single outlier)
    - Dixon Q test (small samples)
    - Extreme Value (GEV) analysis
    - GESD (Generalized ESD) for multiple outliers
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def grubbs_test(self, series: pd.Series) -> Dict:
        """
        Grubbs test for a single outlier.
        H0: No outlier in dataset.
        """
        data = _clean_col(series)
        n = len(data)
        if n < 6:
            return {"error": f"Grubbs requires n≥6, got n={n}"}

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        G = float(np.max(np.abs(data - mean)) / std)

        # Critical value
        t_crit = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

        outlier_idx = int(np.argmax(np.abs(data - mean)))
        outlier_val = float(data[outlier_idx])

        return {
            "test": "grubbs",
            "G_stat": round(G, 4),
            "G_critical": round(float(G_crit), 4),
            "outlier_detected": G > G_crit,
            "outlier_value": outlier_val,
            "outlier_index": outlier_idx,
            "alpha": self.alpha,
        }

    def dixon_q_test(self, series: pd.Series) -> Dict:
        """
        Dixon Q test for outliers in small samples (3 ≤ n ≤ 30).
        """
        data = np.sort(_clean_col(series))
        n = len(data)
        if not (3 <= n <= 30):
            return {"error": f"Dixon Q requires 3 ≤ n ≤ 30, got n={n}"}

        data_range = data[-1] - data[0]
        if data_range == 0:
            return {"error": "All values are identical"}

        # Q for lowest and highest
        Q_low  = float((data[1] - data[0]) / data_range)
        Q_high = float((data[-1] - data[-2]) / data_range)

        # Approximate critical values (alpha=0.05)
        Q_crit_table = {
            3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560,
            7: 0.507, 8: 0.468, 9: 0.437, 10: 0.412,
        }
        Q_crit = Q_crit_table.get(n, 0.3)

        return {
            "test": "dixon_q",
            "Q_high": round(Q_high, 4),
            "Q_low": round(Q_low, 4),
            "Q_critical": Q_crit,
            "outlier_high": Q_high > Q_crit,
            "outlier_low": Q_low > Q_crit,
            "possible_outlier_high": float(data[-1]),
            "possible_outlier_low": float(data[0]),
        }

    def gesd_test(self, series: pd.Series, max_outliers: int = 10) -> Dict:
        """
        Generalized ESD (Extreme Studentized Deviate) test.
        Detects up to max_outliers outliers simultaneously.
        """
        data = _clean_col(series).tolist()
        n = len(data)
        if n < 25:
            return {"error": f"GESD requires n≥25, got n={n}"}

        outlier_indices = []
        working_data = data.copy()
        R_stats = []

        for i in range(min(max_outliers, n // 3)):
            mean = np.mean(working_data)
            std = np.std(working_data, ddof=1)
            if std == 0:
                break
            deviations = [abs(x - mean) / std for x in working_data]
            R = max(deviations)
            R_stats.append(round(float(R), 4))

            # Critical value
            p = 1 - self.alpha / (2 * (n - i))
            t = stats.t.ppf(p, n - i - 2)
            lam = (n - i - 1) * t / np.sqrt((n - i - 2 + t**2) * (n - i))

            if R > lam:
                idx = deviations.index(max(deviations))
                outlier_indices.append(working_data[idx])
                working_data.pop(idx)
            else:
                break

        return {
            "test": "gesd",
            "n_outliers_detected": len(outlier_indices),
            "outlier_values": outlier_indices,
            "R_statistics": R_stats,
            "max_outliers_tested": max_outliers,
        }

    def run_all(self, series: pd.Series) -> Dict:
        """Run all available outlier stat tests."""
        data = _clean_col(series)
        n = len(data)
        results = {"column": str(series.name), "n": n}

        if n >= 6:
            results["grubbs"] = self.grubbs_test(series)
        if 3 <= n <= 30:
            results["dixon_q"] = self.dixon_q_test(series)
        if n >= 25:
            results["gesd"] = self.gesd_test(series)

        return results


# ─────────────────────────────────────────────
#  8. TimeSeriesStats
# ─────────────────────────────────────────────

class TimeSeriesStats:
    """
    Time series statistical analysis:
    - ADF test (stationarity)
    - KPSS test (stationarity)
    - Autocorrelation + Partial autocorrelation
    - Durbin-Watson (autocorrelation in residuals)
    - Granger causality
    - Trend detection
    """

    def __init__(self, alpha: float = 0.05, max_lags: int = 20):
        self.alpha = alpha
        self.max_lags = max_lags

    def test_stationarity(self, series: pd.Series) -> Dict:
        """ADF + KPSS stationarity tests."""
        data = _clean_col(series)
        result = {"column": str(series.name), "n": len(data)}

        if not STATSMODELS:
            return {**result, "error": "statsmodels required for stationarity tests"}

        # ADF test
        try:
            adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(data, autolag="AIC")
            result["adf"] = {
                "stat": round(float(adf_stat), 4),
                "p_value": round(float(adf_p), 4),
                "lags": int(adf_lags),
                "critical_values": {k: round(v, 4) for k, v in adf_crit.items()},
                "is_stationary": adf_p < self.alpha,
            }
        except Exception as e:
            result["adf"] = {"error": str(e)}

        # KPSS test
        try:
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(data, regression="c", nlags="auto")
            result["kpss"] = {
                "stat": round(float(kpss_stat), 4),
                "p_value": round(float(kpss_p), 4),
                "lags": int(kpss_lags),
                "is_stationary": kpss_p > self.alpha,  # H0 = stationary
            }
        except Exception as e:
            result["kpss"] = {"error": str(e)}

        # Combined verdict
        adf_stat_ok = result.get("adf", {}).get("is_stationary", False)
        kpss_stat_ok = result.get("kpss", {}).get("is_stationary", True)
        if adf_stat_ok and kpss_stat_ok:
            result["verdict"] = "stationary"
        elif not adf_stat_ok and not kpss_stat_ok:
            result["verdict"] = "non_stationary — differencing recommended"
        else:
            result["verdict"] = "inconclusive — possibly trend-stationary"

        return result

    def autocorrelation(self, series: pd.Series) -> Dict:
        """ACF and PACF analysis."""
        data = _clean_col(series)
        if not STATSMODELS or len(data) < 10:
            return {"error": "Need statsmodels and n≥10"}
        try:
            acf_vals = acf(data, nlags=min(self.max_lags, len(data) // 3), fft=True)
            pacf_vals = pacf(data, nlags=min(self.max_lags, len(data) // 3))
            conf_int = 1.96 / np.sqrt(len(data))
            sig_lags_acf = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > conf_int]
            sig_lags_pacf = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > conf_int]
            return {
                "acf": [round(float(v), 4) for v in acf_vals],
                "pacf": [round(float(v), 4) for v in pacf_vals],
                "confidence_interval": round(float(conf_int), 4),
                "significant_acf_lags": sig_lags_acf,
                "significant_pacf_lags": sig_lags_pacf,
                "has_autocorrelation": len(sig_lags_acf) > 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def granger_causality(
        self, df: pd.DataFrame, cause_col: str, effect_col: str, max_lag: int = 5
    ) -> Dict:
        """Granger causality test: does cause_col help predict effect_col?"""
        if not STATSMODELS:
            return {"error": "statsmodels required"}
        try:
            data = df[[effect_col, cause_col]].dropna()
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            sig_lags = []
            for lag, res in results.items():
                p = res[0]["ssr_ftest"][1]
                if p < self.alpha:
                    sig_lags.append({"lag": lag, "p_value": round(float(p), 4)})
            return {
                "cause": cause_col,
                "effect": effect_col,
                "granger_causes": len(sig_lags) > 0,
                "significant_lags": sig_lags,
                "interpretation": (
                    f"{cause_col} Granger-causes {effect_col} at lags {[s['lag'] for s in sig_lags]}"
                    if sig_lags else f"No evidence that {cause_col} Granger-causes {effect_col}"
                ),
            }
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────
#  9. TransformationAdvisor
# ─────────────────────────────────────────────

class TransformationAdvisor:
    """
    Recommends and evaluates data transformations per column.
    Tests: log, sqrt, Box-Cox, Yeo-Johnson, reciprocal, square.
    Scores each by resulting normality improvement.
    """

    TRANSFORMS = {
        "log1p":      lambda x: np.log1p(x),
        "sqrt":       lambda x: np.sqrt(np.abs(x)),
        "square":     lambda x: x ** 2,
        "reciprocal": lambda x: 1 / (x + 1e-9),
        "cbrt":       lambda x: np.cbrt(x),
    }

    def advise(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = _clean_col(df[col])
            if len(data) < 20:
                continue
            results[col] = self._advise_col(col, data)
        return results

    def _advise_col(self, col: str, data: np.ndarray) -> Dict:
        # Baseline normality
        baseline_p = self._normality_p(data)
        skew = float(stats.skew(data))

        candidates = {}

        # Standard transforms
        for name, fn in self.TRANSFORMS.items():
            try:
                transformed = fn(data)
                if not np.all(np.isfinite(transformed)):
                    continue
                p = self._normality_p(transformed)
                candidates[name] = {
                    "normality_p": round(float(p), 4),
                    "skewness_after": round(float(stats.skew(transformed)), 4),
                    "improvement": round(float(p - baseline_p), 4),
                }
            except Exception:
                continue

        # Box-Cox (requires positive data)
        if data.min() > 0:
            try:
                bc_data, lam = boxcox(data)
                p = self._normality_p(bc_data)
                candidates["box_cox"] = {
                    "normality_p": round(float(p), 4),
                    "lambda": round(float(lam), 4),
                    "skewness_after": round(float(stats.skew(bc_data)), 4),
                    "improvement": round(float(p - baseline_p), 4),
                }
            except Exception:
                pass

        # Yeo-Johnson (works on any data)
        try:
            yj_data, lam = yeojohnson(data)
            p = self._normality_p(yj_data)
            candidates["yeo_johnson"] = {
                "normality_p": round(float(p), 4),
                "lambda": round(float(lam), 4),
                "skewness_after": round(float(stats.skew(yj_data)), 4),
                "improvement": round(float(p - baseline_p), 4),
            }
        except Exception:
            pass

        # Find best
        best = max(candidates.items(), key=lambda x: x[1]["normality_p"],
                   default=(None, {}))

        return {
            "baseline_normality_p": round(float(baseline_p), 4),
            "baseline_skewness": round(skew, 4),
            "needs_transformation": baseline_p < 0.05 and abs(skew) > 0.5,
            "best_transform": best[0],
            "best_result": best[1],
            "all_candidates": candidates,
            "recommendation": self._recommend(skew, data.min() > 0, best[0]),
        }

    def _normality_p(self, data: np.ndarray) -> float:
        try:
            _, p = shapiro(data[:min(len(data), 5000)])
            return float(p)
        except Exception:
            return 0.0

    def _recommend(self, skew: float, all_positive: bool, best: Optional[str]) -> str:
        if best is None:
            return "No improvement found — data may be inherently non-normal."
        if best == "box_cox":
            return f"Apply Box-Cox transformation (requires positive values ✅)."
        if best == "yeo_johnson":
            return f"Apply Yeo-Johnson transformation (works on any values ✅)."
        return f"Apply {best} transformation for best normality improvement."


# ─────────────────────────────────────────────
#  10. StatsReporter
# ─────────────────────────────────────────────

class StatsReporter:
    """
    Generates a full statistical audit report.
    Combines all analyzers into one comprehensive output.
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def generate(self) -> Dict[str, Any]:
        return {
            "summary": self._summary(),
            "normality": self.results.get("normality", {}),
            "distributions": self.results.get("distributions", {}),
            "skewness_kurtosis": self.results.get("skewness_kurtosis", {}),
            "correlations": self.results.get("correlations"),
            "recommendations": self._recommendations(),
        }

    def _summary(self) -> Dict:
        norm = self.results.get("normality", {})
        normal_cols = [c for c, r in norm.items() if isinstance(r, dict) and r.get("is_normal")]
        non_normal = [c for c, r in norm.items() if isinstance(r, dict) and not r.get("is_normal")]
        return {
            "n_columns_analyzed": len(norm),
            "n_normal": len(normal_cols),
            "n_non_normal": len(non_normal),
            "normal_columns": normal_cols,
            "non_normal_columns": non_normal,
        }

    def _recommendations(self) -> List[str]:
        recs = []
        norm = self.results.get("normality", {})
        non_normal = [c for c, r in norm.items() if isinstance(r, dict) and not r.get("is_normal")]
        transforms = self.results.get("transformations", {})

        if non_normal:
            recs.append(f"⚠️  {len(non_normal)} non-normal column(s): {non_normal[:5]}")
            recs.append("Use non-parametric tests for these columns.")

        for col, t in transforms.items():
            if isinstance(t, dict) and t.get("needs_transformation"):
                best = t.get("best_transform")
                recs.append(f"  • {col}: Apply {best} transformation.")

        sk = self.results.get("skewness_kurtosis", {})
        bimodal = [c for c, r in sk.items() if isinstance(r, dict) and r.get("possibly_bimodal")]
        if bimodal:
            recs.append(f"⚠️  Possibly bimodal: {bimodal} — investigate subgroups.")

        if not recs:
            recs.append("✅ Data appears statistically well-behaved.")

        return recs

    def to_markdown(self) -> str:
        report = self.generate()
        summary = report["summary"]
        lines = [
            "# 📊 Advanced Statistics Report — Nydra",
            f"**Columns Analyzed:** {summary['n_columns_analyzed']}  ",
            f"**Normal:** {summary['n_normal']} | **Non-Normal:** {summary['n_non_normal']}  ",
            "",
            "## 💡 Recommendations",
        ]
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  11. AdvancedStats — Master Class
# ─────────────────────────────────────────────

class AdvancedStats:
    """
    Master class — runs all statistical analyses in one call.

    Usage:
    ------
        stats = AdvancedStats(df)
        results = stats.compute_all()
        print(results["normality"])
        print(results["report_markdown"])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float = 0.05,
        fit_distributions: bool = True,
        run_timeseries: bool = False,
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
    ):
        self.df = df
        self.alpha = alpha
        self.fit_distributions = fit_distributions
        self.run_timeseries = run_timeseries
        self.target_col = target_col
        self.group_col = group_col

    def compute_all(self) -> Dict[str, Any]:
        """Run all statistical modules. Returns full results dict."""
        results = {}

        logger.info("📊 [AdvancedStats] Running descriptive statistics...")
        desc = DescriptiveStats(alpha=self.alpha)
        results["descriptive"] = desc.compute(self.df)

        logger.info("🧪 [AdvancedStats] Testing normality...")
        norm_tester = NormalityTester(alpha=self.alpha)
        normality_raw = norm_tester.test_all(self.df)
        results["normality"] = {col: r.summary() for col, r in normality_raw.items()}
        results["normality_df"] = norm_tester.summary_df(self.df)

        logger.info("📐 [AdvancedStats] Analyzing skewness & kurtosis...")
        sk = SkewnessKurtosisAnalyzer()
        results["skewness_kurtosis"] = sk.analyze(self.df)

        if self.fit_distributions:
            logger.info("🔎 [AdvancedStats] Fitting distributions...")
            fitter = DistributionFitter()
            fits_raw = fitter.fit_all(self.df)
            results["distributions"] = {col: r.summary() for col, r in fits_raw.items()}
            results["distributions_df"] = fitter.summary_df(self.df)

        logger.info("🔗 [AdvancedStats] Computing correlations...")
        corr = CorrelationAnalyzer(alpha=self.alpha)
        results["correlations"] = corr.compute_pairwise(self.df)
        results["correlation_matrix"], results["p_value_matrix"] = \
            corr.correlation_matrix(self.df)

        logger.info("🔄 [AdvancedStats] Running transformation advisor...")
        advisor = TransformationAdvisor()
        results["transformations"] = advisor.advise(self.df)

        if self.group_col and self.group_col in self.df.columns:
            logger.info("📏 [AdvancedStats] Running hypothesis tests...")
            tester = HypothesisTester(alpha=self.alpha)
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            results["hypothesis_tests"] = {}
            for col in num_cols[:5]:  # Limit to top 5 for speed
                if col != self.group_col:
                    results["hypothesis_tests"][col] = tester.compare_groups(
                        self.df, col, self.group_col
                    )

        if self.run_timeseries:
            logger.info("📈 [AdvancedStats] Running time series tests...")
            ts = TimeSeriesStats(alpha=self.alpha)
            results["timeseries"] = {}
            for col in self.df.select_dtypes(include=[np.number]).columns:
                results["timeseries"][col] = ts.test_stationarity(self.df[col])

        # Generate final report
        reporter = StatsReporter(results)
        results["report"] = reporter.generate()
        results["report_markdown"] = reporter.to_markdown()

        logger.info("✅ [AdvancedStats] Complete.")
        return results

    def normality_only(self) -> pd.DataFrame:
        """Quick normality test only."""
        return NormalityTester(alpha=self.alpha).summary_df(self.df)

    def correlations_only(self, method: str = "all") -> pd.DataFrame:
        """Quick correlation matrix only."""
        return CorrelationAnalyzer(alpha=self.alpha).compute_pairwise(self.df, method=method)

    def distributions_only(self) -> pd.DataFrame:
        """Quick distribution fitting only."""
        return DistributionFitter().summary_df(self.df)


# ─────────────────────────────────────────────
#  Public Convenience Functions
# ─────────────────────────────────────────────

def quick_normality(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One-line normality test for all numeric columns."""
    return NormalityTester(alpha=alpha).summary_df(df)


def quick_correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """One-line pairwise correlations with significance."""
    return CorrelationAnalyzer().compute_pairwise(df, method=method)


def quick_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """One-line distribution fitting for all numeric columns."""
    return DistributionFitter().summary_df(df)


def quick_transformations(df: pd.DataFrame) -> Dict:
    """One-line transformation recommendations."""
    return TransformationAdvisor().advise(df)


def full_stats_report(
    df: pd.DataFrame,
    alpha: float = 0.05,
    fit_distributions: bool = True,
    output_format: str = "dict",
) -> Any:
    """
    Complete statistical analysis in one call.

    Parameters
    ----------
    df               : Input DataFrame
    alpha            : Significance level (default 0.05)
    fit_distributions: Fit 15+ distributions per column
    output_format    : 'dict' | 'markdown'
    """
    stats_engine = AdvancedStats(df, alpha=alpha, fit_distributions=fit_distributions)
    results = stats_engine.compute_all()
    if output_format == "markdown":
        return results["report_markdown"]
    return results
