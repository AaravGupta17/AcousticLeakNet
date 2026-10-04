"""Fast tests of the pre-registered outcome-class logic in experiments/xrig_summary.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from xrig_summary import classify_direction, overall_class   # noqa: E402


def test_inverted_collapse():
    lab, _ = classify_direction(0.35, 0.50)
    assert lab.startswith("Collapse") and "inverted" in lab


def test_chance_collapse_boundaries():
    assert classify_direction(0.52, 0.5)[0] == "Collapse"
    assert classify_direction(0.60, 0.5)[0] == "Collapse"
    assert classify_direction(0.40, 0.5)[0] == "Collapse"
    assert classify_direction(0.61, 0.5)[0] == "Moderate/asymmetric"


def test_strong():
    lab, d = classify_direction(0.90, 0.80)
    assert lab == "Strong" and not d
    assert overall_class("Strong", "Strong") == "Strong"


def test_moderate_when_margin_small_or_below_08():
    assert classify_direction(0.85, 0.82)[0] == "Moderate/asymmetric"
    assert classify_direction(0.75, 0.50)[0] == "Moderate/asymmetric"
    assert overall_class("Strong", "Moderate/asymmetric") == "Moderate/asymmetric"


def test_collapse_dominates_overall():
    assert overall_class("Strong", "Collapse") == "Collapse"
    assert overall_class("Collapse (inverted ranking)", "Strong") == "Collapse"


def test_case_d_flag():
    assert classify_direction(0.70, 0.68)[1] is True
    assert classify_direction(0.70, 0.60)[1] is False
