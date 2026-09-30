"""
Preprocessing validation for Model_F/snr_sweep_f.py (docs/MODEL_F_SNR_SWEEP_PREREG.md).

Uses only synthetic fixtures (no datasets/NetworkList, no checkpoints) --
mirrors tests/test_dataset_e.py and tests/test_model_f.py's own fixture
style. Confirms the sweep script's window-construction pipeline is genuinely
Model-F-compatible (z-score, 2 kHz band limit) and not accidentally using
Model C's fixed-scale convention, and that the controlled-factor design
(everything except injected SNR held fixed per window index) actually holds.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model_F"))
sys.path.insert(0, str(ROOT / "Model_E"))
sys.path.insert(0, str(ROOT))
import augment_f as A                                      # noqa: E402
import dataset_e as E                                       # noqa: E402
import train_f as TF                                         # noqa: E402
import snr_sweep_f as S                                       # noqa: E402
from test_dataset_e import _leak_cfg, _make                    # noqa: E402


@pytest.fixture
def bank(tmp_path):
    """Real bank.npz always stores band-limited windows (bank_f.py builds
    `single`/`pairs` from experiments/public_data.py's to_common(), which
    band-limits to 2 kHz) -- band-limit the fixture's fake data too, so a
    band-limit test against this fixture is actually meaningful."""
    rng = np.random.default_rng(0)
    n = 20
    groups = np.array([f"g{i // 2}" for i in range(n)])
    single = A.band_limit(rng.standard_normal((n, A.T))).astype(np.float16)
    pairs = A.band_limit(rng.standard_normal((3, 2, A.T))).astype(np.float16)
    np.savez(tmp_path / "bank.npz",
             single=single, s_y=(np.arange(n) % 2).astype(np.int8), s_group=groups,
             s_source=np.array(["hongkong"] * n), s_dataset=np.array(["hongkong"] * n),
             s_val=np.zeros(n, bool),
             pairs=pairs, p_group=np.array(["m"] * 3), p_val=np.zeros(3, bool))
    return TF.Bank(tmp_path / "bank.npz", val=False)


@pytest.fixture
def ds_leak():
    return _make(E.LeakDatasetE)


@pytest.fixture
def scenario():
    return _leak_cfg(10.0, 30.0)


def _power_at(x, hz):
    f = np.fft.rfftfreq(x.shape[-1], 1 / A.FS)
    return (np.abs(np.fft.rfft(x, axis=-1)) ** 2)[..., np.argmin(abs(f - hz))].mean()


# ── 1. windowing / labels ──────────────────────────────────────────────────

def test_window_shape_is_two_by_T(bank, ds_leak, scenario):
    x = S.make_window(0, True, 0.0, bank, ds_leak, [scenario])
    assert x.shape == (2, A.T)


def test_build_level_labels_are_exactly_balanced_and_fixed_index_order(bank, ds_leak, scenario):
    x1, y1 = S.build_level(0.0, bank, ds_leak, [scenario], n=40)
    x2, y2 = S.build_level(10.0, bank, ds_leak, [scenario], n=40)
    assert y1.mean() == 0.5
    np.testing.assert_array_equal(y1, y2)  # class assignment must not depend on SNR


# ── 2. controlled-factor design: nuisance variables independent of SNR ─────

def test_no_leak_windows_are_bit_identical_across_snr_levels(bank, ds_leak, scenario):
    """No-leak windows never touch snr_db at all -- the strongest possible
    version of 'nuisance variables held fixed' for the no-leak class."""
    a = S.make_window(25, False, -20.0, bank, ds_leak, [scenario])
    b = S.make_window(25, False, 10.0, bank, ds_leak, [scenario])
    np.testing.assert_array_equal(a, b)


def test_leak_window_background_is_independent_of_snr(monkeypatch, bank, ds_leak, scenario):
    """With the leak component monkeypatched to zero, a 'leak' window's
    construction must be identical across SNR levels -- proving background/
    interferers/EQ/pairing depend on the window index alone, never on SNR."""
    monkeypatch.setattr(S, "leak_window", lambda i, snr_db, ds, sc: np.zeros((2, A.T)))
    a = S.make_window(3, True, -20.0, bank, ds_leak, [scenario])
    b = S.make_window(3, True, 10.0, bank, ds_leak, [scenario])
    np.testing.assert_array_equal(a, b)


def test_leak_window_physics_jitter_is_independent_of_snr(ds_leak, scenario):
    """The leak's own random jitter (wave speed, attenuation, position noise
    inside dataset_e.py's _leak(), which uses global np.random) must depend
    on the window index alone, never on snr_db -- otherwise different SNR
    levels would secretly compare different leak *shapes*, not just
    different amplitudes."""
    lo = S.leak_window(7, -20.0, ds_leak, scenario)
    hi = S.leak_window(7, 10.0, ds_leak, scenario)
    ratio = np.sqrt(np.mean(hi.astype(np.float64) ** 2)) / np.sqrt(np.mean(lo.astype(np.float64) ** 2))
    # if position/wave-speed/attenuation jitter were identical, the ONLY
    # difference between -20dB and +10dB is the amplitude factor, exactly
    # 10**((10-(-20))/20) = 31.62x in RMS
    assert ratio == pytest.approx(10 ** ((10.0 - (-20.0)) / 20.0), rel=0.02)


# ── 3. SNR injection is quantitatively correct ─────────────────────────────

def test_snr_override_scales_leak_rms_exactly_as_specified(ds_leak, scenario):
    """leak_component's docstring contract: output is in background-RMS
    units, i.e. adding it to a unit-RMS background yields the requested SNR.
    Check the scaling law directly across the whole pre-registered grid."""
    ref_db = 0.0
    ref = S.leak_window(1, ref_db, ds_leak, scenario)
    ref_rms = np.sqrt(np.mean(ref.astype(np.float64) ** 2))
    for db in S.SNR_GRID_DB:
        x = S.leak_window(1, db, ds_leak, scenario)
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        expected = ref_rms * 10 ** ((db - ref_db) / 20.0)
        assert rms == pytest.approx(expected, rel=0.02), f"SNR {db} dB scaling off"


def test_snr_override_ignores_torricelli_amp(ds_leak):
    """torricelli_amp only determines the NATIVE (non-overridden) SNR draw
    (dataset_e.py's _leak(): `if self.snr_override_db is not None: snr_db =
    self.snr_override_db`, bypassing the torricelli_amp-based formula
    entirely). Two scenarios that differ ONLY in torricelli_amp must
    therefore give the same received RMS once an override is set -- distance
    and material are held fixed here since those legitimately change
    attenuation and are not part of this claim."""
    c1 = _leak_cfg(10.0, 30.0, torricelli_amp=1e-4)
    c2 = _leak_cfg(10.0, 30.0, torricelli_amp=5e-3)
    x1 = S.leak_window(0, 5.0, ds_leak, c1)
    x2 = S.leak_window(0, 5.0, ds_leak, c2)
    r1 = np.sqrt(np.mean(x1.astype(np.float64) ** 2))
    r2 = np.sqrt(np.mean(x2.astype(np.float64) ** 2))
    assert r1 == pytest.approx(r2, rel=0.02)


def test_snr_override_received_amplitude_depends_on_distance_and_material():
    """Documents a real, non-bug property the SNR-sweep protocol must
    account for: 'snr_override_db' sets the source SNR before propagation,
    so received amplitude still varies with distance/material attenuation
    -- confirmed here so the pre-registration doesn't mistakenly claim a
    flat SNR grid controls received loudness exactly."""
    ds = _make(E.LeakDatasetE)
    near = S.leak_window(0, 5.0, ds, _leak_cfg(5.0, 5.0, pipe_material="CI"))
    far = S.leak_window(0, 5.0, ds, _leak_cfg(60.0, 60.0, pipe_material="CI"))
    r_near = np.sqrt(np.mean(near.astype(np.float64) ** 2))
    r_far = np.sqrt(np.mean(far.astype(np.float64) ** 2))
    assert r_far < r_near  # attenuation over distance is real and not cancelled by the override


# ── 4. Model-F-specific preprocessing, not Model C's ───────────────────────

def test_output_is_band_limited_to_2khz(bank, ds_leak, scenario):
    x = S.make_window(0, True, 0.0, bank, ds_leak, [scenario])
    assert _power_at(x, 2400) < 1e-3 * _power_at(x, 500)


def test_output_is_joint_zscored_not_model_c_fixed_scale(bank, ds_leak, scenario):
    """Model F's finish() step: zero mean, ~unit RMS (clipped). Model C's
    fixed-scale convention has no such normalization (leak-free RMS ~ 0.1,
    mean can be non-zero from the DC term) -- these are easy to
    distinguish, and mixing them up is exactly the invalid-shortcut this
    script exists to avoid (see snr_sweep_f.py's module docstring)."""
    x = S.make_window(0, True, 0.0, bank, ds_leak, [scenario])
    assert abs(x.mean()) < 0.05
    assert 0.5 < x.std() < 1.5   # clipped joint z-score, not ~0.1 fixed-scale


def test_checkpoint_guard_rejects_non_model_f_checkpoints():
    """main() asserts cfg['input_norm'] == 'zscore' before scoring any
    checkpoint -- confirm that guard exists and would reject a Model C-style
    cfg (this is a static check of the source, not an import-time model
    load, since no checkpoint is touched by this test suite)."""
    src = (ROOT / "Model_F" / "snr_sweep_f.py").read_text(encoding="utf-8")
    assert 'ckpt["cfg"].get("input_norm") == "zscore"' in src


# ── 5. no label-correlated artifact in the nuisance (background) generation ─

def test_background_generation_code_path_does_not_depend_on_label(bank, ds_leak, scenario):
    """background()/interferers() are called identically regardless of
    is_leak in make_window's source (read directly, since the whole point
    is that the code path -- not just a particular random draw -- must not
    branch on label)."""
    src = (ROOT / "Model_F" / "snr_sweep_f.py").read_text(encoding="utf-8")
    fn = src.split("def make_window")[1].split("def build_level")[0]
    bg_line = [l for l in fn.splitlines() if "TF.background" in l][0]
    extra_line = [l for l in fn.splitlines() if "A.interferers" in l][0]
    assert "is_leak" not in bg_line and "is_leak" not in extra_line
