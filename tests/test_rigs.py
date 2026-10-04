"""Cross-rig benchmark: splits, threshold rule, same-stream pairing, and the
guarantee that the training script only ever loads its own rig. No real data."""

import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model_F"))
import rigs                                               # noqa: E402
import xrig_train                                         # noqa: E402


def _fake_rig(n_leak=10, n_none=4, per_group=6, streams=2, seed=0):
    rng = np.random.default_rng(seed)
    gs = [(f"g_leak{i}", 1) for i in range(n_leak)] + [(f"g_none{i}", 0) for i in range(n_none)]
    x, y, g, s, meta = [], [], [], [], []
    for name, lab in gs:
        for k in range(per_group):
            x.append(rng.standard_normal(2000))
            y.append(lab); g.append(name); s.append(f"{name}|ch{k % streams}"); meta.append("m")
    x = np.array(x, np.float32)
    return rigs.RigWindows(x, np.zeros(len(x), np.float32), np.array(y, np.int8), np.array(g),
                           np.array(s), np.array(meta))


def test_val_split_counts_and_whole_groups():
    rw = _fake_rig(10, 4)
    val = rigs.val_split(rw.group, rw.y)
    vg = set(rw.group[val])
    assert not (vg & set(rw.group[~val])), "a group was split between train and val"
    assert sum(g.startswith("g_leak") for g in vg) == math.ceil(0.2 * 10)
    assert sum(g.startswith("g_none") for g in vg) == math.ceil(0.2 * 4)


def test_val_split_minimum_one_and_deterministic():
    rw = _fake_rig(3, 2)
    a, b = rigs.val_split(rw.group, rw.y), rigs.val_split(rw.group[::-1], rw.y[::-1])
    assert len(set(rw.group[a])) == 2                       # 1 leak + 1 no-leak
    assert set(rw.group[a]) == set(rw.group[::-1][b])       # independent of row order


def test_val_split_rejects_mixed_group():
    with pytest.raises(AssertionError):
        rigs.val_split(np.array(["a", "a"]), np.array([0, 1]))


def test_fold_assign_covers_all_groups_and_classes():
    rw = _fake_rig(10, 4)
    f = rigs.fold_assign(rw.group, rw.y, k=3)
    assert set(f) == set(rw.group)
    for fold in range(3):
        mine = [g for g, v in f.items() if v == fold]
        assert any(g.startswith("g_leak") for g in mine) and any(g.startswith("g_none") for g in mine)
    assert sorted(np.bincount([v for g, v in f.items() if g.startswith("g_leak")])) == [3, 3, 4]


def test_youden_threshold_toy():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.7, 0.6, 0.8, 0.9])
    assert rigs.youden_threshold(y, s) == pytest.approx(0.6)   # ties with 0.7 at 5/6; smallest wins
    # perfectly separable: smallest threshold that attains the maximum
    assert rigs.youden_threshold([0, 0, 1, 1], [0.0, 1.0, 2.0, 3.0]) == 2.0
    # fully tied scores: the only candidate
    assert rigs.youden_threshold([0, 1], [5.0, 5.0]) == 5.0


def test_rigbank_partner_is_same_stream():
    rw = _fake_rig(4, 2, per_group=8, streams=2)
    bank = xrig_train.RigBank(rw)
    rng = np.random.default_rng(0)
    seen = set()
    for i in range(len(rw)):
        for _ in range(5):
            j = bank.partner(i, rng)
            assert rw.stream[j] == rw.stream[i]
            seen.add(j != i)
    assert True in seen                                    # actually draws other windows


def test_prepare_data_loads_only_the_training_rig(monkeypatch):
    calls = []

    def fake_load(name, cache=True):
        calls.append(name)
        return _fake_rig()

    monkeypatch.setattr(rigs, "load_rig", fake_load)
    d = xrig_train.prepare_data(Namespace(train_rig="sheffield", fold=-1))
    assert calls == ["sheffield"]
    assert not set(d["train_groups"]) & set(d["val_groups"])
    assert len(d["bank_va"].single) > 0 and d["test_groups"] == []


def test_within_rig_fold_removes_test_groups_before_val(monkeypatch):
    monkeypatch.setattr(rigs, "load_rig", lambda name, cache=True: _fake_rig())
    d = xrig_train.prepare_data(Namespace(train_rig="mendeley_acc", fold=1))
    assert d["test_groups"]
    used = set(d["train_groups"]) | set(d["val_groups"])
    assert not used & set(d["test_groups"])
    rw = _fake_rig()
    assert used | set(d["test_groups"]) == set(rw.group)


def test_train_script_never_references_synthetic_cache_or_other_rigs():
    src = (ROOT / "Model_F" / "xrig_train.py").read_text()
    assert "cache_f" not in src and "cache_" not in src.replace("cache_xrig", "")
    assert src.count("rigs.load_rig(") == 1
    assert "mendeley_hyd" not in src.split("TRAIN_RIGS =")[1].split("\n")[0]
