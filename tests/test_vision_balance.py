import math

import pytest

from src.vision_ui import _apply_full_balance, _entropy_balance, _score_grade


def _ref_entropy(counts):
    """Independent copy of the analyzer formula (scipy entropy / log(n) * 100)."""
    scipy_stats = pytest.importorskip("scipy.stats")
    import numpy as np
    c = np.asarray(counts, dtype=float)
    return round(float(scipy_stats.entropy(c / c.sum()) / np.log(len(c)) * 100), 2)


@pytest.mark.parametrize("counts", [[200, 120, 40], [5, 1], [50, 50], [1000, 10, 10, 10], [7, 3, 3]])
def test_entropy_matches_analyzer_formula(counts):
    assert _entropy_balance(counts) == pytest.approx(_ref_entropy(counts), abs=0.01)


def test_entropy_edge_cases():
    assert _entropy_balance([]) == 100.0
    assert _entropy_balance([40]) == 100.0
    assert _entropy_balance([10, 10]) == 100.0
    assert _entropy_balance([10, 0]) == 100.0      # zero-count classes are ignored
    assert 0.0 < _entropy_balance([1000, 1]) < 15.0


def test_five_to_one_is_gentle_not_harsh():
    # the old min/max formula gave 20 here; the analyzer's gives about 65
    assert 60 < _entropy_balance([5, 1]) < 70


@pytest.mark.parametrize("score,grade", [(95, "A"), (90, "A"), (89.9, "B"), (75, "B"), (74.9, "C"),
                                         (60, "C"), (59.9, "D"), (45, "D"), (44.9, "F"), (0, "F")])
def test_grade_bands(score, grade):
    assert _score_grade(score) == grade


def test_apply_adjusts_by_weight_and_regrades():
    rd = {"score": 90.0, "grade": "A", "verdict": "Ready", "priority_issues": []}
    _apply_full_balance(rd, old_bal=100.0, new_bal=65.0, ratio=5.0)
    assert rd["score"] == pytest.approx(90.0 + 0.25 * (65.0 - 100.0), abs=0.01)   # 81.25
    assert rd["grade"] == "B"
    assert rd["verdict"] == "Ready"               # ratio 5.0 is not above the severe threshold
    assert rd["balance_penalty"] == pytest.approx(8.8, abs=0.1)


def test_apply_can_raise_the_score():
    rd = {"score": 70.0, "verdict": "Needs Work", "priority_issues": []}
    _apply_full_balance(rd, old_bal=40.0, new_bal=100.0, ratio=1.0)
    assert rd["score"] == pytest.approx(85.0)
    assert rd["balance_penalty"] == 0.0
    assert rd["verdict"] == "Ready"


def test_severe_ratio_adds_issue_and_blocks_ready():
    rd = {"score": 92.0, "verdict": "Ready", "priority_issues": ["Low image diversity"]}
    _apply_full_balance(rd, old_bal=100.0, new_bal=30.0, ratio=12.0)
    assert rd["priority_issues"][0] == "Severe class imbalance"
    assert rd["verdict"] == "Needs Work"
    assert rd["score"] == pytest.approx(74.5)     # 92 + 0.25 * (30 - 100)
    assert rd["grade"] == "C"


def test_severe_with_low_score_is_not_ready():
    rd = {"score": 55.0, "verdict": "Needs Work", "priority_issues": []}
    _apply_full_balance(rd, old_bal=100.0, new_bal=0.0, ratio=50.0)    # 55 - 25 = 30
    assert rd["score"] == pytest.approx(30.0)
    assert rd["verdict"] == "Not Ready"


def test_existing_imbalance_issue_is_not_duplicated():
    rd = {"score": 80.0, "priority_issues": ["Severe class imbalance"]}
    _apply_full_balance(rd, 100.0, 40.0, 9.0)
    assert rd["priority_issues"].count("Severe class imbalance") == 1


def test_shift_issue_blocks_ready():
    rd = {"score": 88.0, "priority_issues": ["Distribution shift between splits"]}
    _apply_full_balance(rd, 100.0, 100.0, 1.0)
    assert rd["verdict"] == "Needs Work"


def test_score_is_clamped():
    rd = {"score": 5.0, "priority_issues": []}
    _apply_full_balance(rd, 100.0, 0.0, 1.0)
    assert rd["score"] == 0.0
    rd = {"score": 99.0, "priority_issues": []}
    _apply_full_balance(rd, 0.0, 100.0, 1.0)
    assert rd["score"] == 100.0


@pytest.mark.parametrize("rd", [{}, {"score": None}, {"score": "x"}])
def test_missing_score_is_a_noop(rd):
    _apply_full_balance(rd, 100.0, 50.0, 9.0)
    assert rd.get("score") in (None, "x") or "score" not in rd
    assert rd.get("balance_penalty") == 0.0


def test_missing_old_balance_is_a_noop():
    rd = {"score": 80.0, "verdict": "Ready", "priority_issues": []}
    _apply_full_balance(rd, None, 50.0, 9.0)
    assert rd["score"] == 80.0 and rd["verdict"] == "Ready"
    assert rd["balance_penalty"] == 0.0