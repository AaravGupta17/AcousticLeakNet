"""Synthetic-data tests for the helpers in experiments/xrig_shift.py (no real data)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import xrig_shift as S                                        # noqa: E402


def test_group_then_class_mean_and_contrast():
    # no-leak: a big group (value 0, 100 windows) and a small one (value 10, 2 windows)
    # leak: one group at 5. Group-then-class averaging -> no-leak mean 5, not ~0.2.
    vals = np.concatenate([np.zeros((100, 3)), np.full((2, 3), 10.0), np.full((4, 3), 5.0)])
    groups = np.array(["a"] * 100 + ["b"] * 2 + ["c"] * 4)
    y = np.array([0] * 102 + [1] * 4)
    gm = S.group_then_class_mean(vals, groups, y)
    assert gm[0][1] == 2 and gm[1][1] == 1
    assert np.allclose(gm[0][0], 5.0) and np.allclose(gm[1][0], 5.0)

    c = np.linspace(0, 1, 20)
    r = S.contrast_correlation(c, 2 * c + 1)
    assert abs(r["pearson"] - 1) < 1e-9 and abs(r["spearman"] - 1) < 1e-9
    r = S.contrast_correlation(c, -c ** 3)
    assert r["spearman"] < -0.99
