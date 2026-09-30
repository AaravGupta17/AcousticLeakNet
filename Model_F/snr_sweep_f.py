"""
snr_sweep_f.py -- Model F SNR robustness sweep
================================================================================
PRE-REGISTERED PROTOCOL: docs/MODEL_F_SNR_SWEEP_PREREG.md (read that first).
This script implements the protocol but has NOT been run as the registered
experiment as of the last commit that touches this file -- see the prereg
doc's own status line for whether that is still true.

Question: does Model F retain leak/no-leak discrimination, beyond an RMS
loudness baseline, as synthetic leak SNR is degraded -- using MODEL F's OWN
preprocessing (2 kHz band limit, random EQ, joint z-score:
Model_F/augment_f.py), not Model C's fixed-scale pipeline that
experiments/snr_sweep.py (E3) uses. E3 has never been run on a Model F
checkpoint, and pointing it at one directly would silently mis-evaluate the
model with the wrong input scaling -- this script exists because that
shortcut is invalid, not because E3 was insufficient.

No new training. Evaluation only, on the three frozen best_model_f_seed{0,1,2}.pt
checkpoints. Reuses, unmodified:
  - Model_E/dataset_e.py's LeakDatasetE.snr_override_db hook (pregen_f.py's own
    leak-generation path)
  - Model_F/augment_f.py's band_limit() / interferers() / finish() / joint_zscore()
  - Model_F/train_f.py's background() (real+synthetic mixture) and Bank class

Controlled-factor design
-------------------------
For a fixed window index i, EVERY stochastic ingredient except the injected
leak SNR is drawn from a seed derived from i ALONE, so it is identical across
every SNR level and every checkpoint:
  - background()/interferers()/random_eq()/channel-pairing: seeded via
    np.random.default_rng([EVAL_SEED, i]) (Generator API).
  - the leak component's own physics jitter (wave speed, attenuation,
    per-sample randomness inside dataset_e.py's _leak(), which uses the
    GLOBAL np.random state, not a passed Generator): seeded via
    np.random.seed(crc32(f"leak:{i}")) immediately before each call, which
    depends on i alone, not on snr_db. This mirrors the existing pattern in
    tests/test_model_f.py::test_leak_component_is_in_background_units_and_band_limited,
    which also reseeds global np.random immediately before calling
    pregen_f.leak_component().
  - Which underlying EPANET leak scenario a leak window uses is assigned by
    i % n_scenarios, fixed across SNR levels and checkpoints.

This isolates injected SNR as the one varying factor between conditions for
a given window index, and gives every checkpoint literally the same 2N
windows (only decoded through different weights) at every SNR level.

Data provenance and the checkpoint-selection-bias caveat
----------------------------------------------------------
Leak scenario configs are drawn from data/csv/val_sampled.csv -- the SAME
split train_f.py's own checkpoint-selection metric (auc_syn) is computed on,
at the NATIVE (non-overridden) SNR distribution. That means grid points
INSIDE Model F's native training SNR range (-10 to 12 dB; SNR_DB_MIN_E/MAX_E
in Model_E/dataset_e.py) are not fully independent of checkpoint selection,
even though the fixed-override construction differs from the native draw.
Grid points BELOW that floor (e.g. -15, -20 dB) are the ones checkpoint
selection had no way to reward, and are the cleanest test of genuine
robustness -- see the prereg doc for how this is used to define success.

    python Model_F/snr_sweep_f.py --dry-run          # build + sanity check only, no checkpoints
    python Model_F/snr_sweep_f.py                    # THE pre-registered run -- do not run without sign-off
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "Model_F"))
sys.path.insert(0, str(ROOT / "Model_E"))
import augment_f as A                                        # noqa: E402
import dataset_e as E                                         # noqa: E402
from pregen_f import leak_component                            # noqa: E402
import train_f as TF                                           # noqa: E402  (Bank, background)
from _common import PLOTS_DIR, load_model, predict_proba, record_run  # noqa: E402
from metrics import detection_report                            # noqa: E402

EVAL_SEED = 0
SNR_GRID_DB = [-20, -15, -10, -5, 0, 5, 10]     # matches experiments/snr_sweep.py's DEFAULT_SNR
SNR_FLOOR_NATIVE = E.SNR_DB_MIN_E                # -10 dB: below this, checkpoint selection
                                                  # never rewarded this amplitude regime
CKPTS = {s: f"best_model_f_seed{s}.pt" for s in (0, 1, 2)}
BANK = ROOT / "cache_f" / "bank.npz"
VAL_CSV = ROOT / "data" / "csv" / "val_sampled.csv"
N_PER_LEVEL = 2000        # matches experiments/snr_sweep.py's --n default
N_BOOT = 2000


def load_val_leak_scenarios():
    """Every leak_status==1 config from Model F's synthetic val split, in
    the same units train_f.py/pregen_f.py already use (pipe distance,
    material, wave speed, torricelli amplitude, ...)."""
    import os
    cwd = os.getcwd()
    os.chdir(ROOT / "model_C")        # index CSV paths are relative to model_C/, per pregen_f.py
    try:
        ds = E.LeakDatasetE(str(VAL_CSV), augment=False)
    finally:
        os.chdir(cwd)
    cfgs = [ds._cache[i] for i in ds._valid_idx if ds._cache[i]["leak_status"] == 1]
    assert cfgs, (
        "No leak scenarios cached from val_sampled.csv -- this almost always means "
        "datasets/NetworkList (the raw EPANET per-network CSVs) is not present locally. "
        "It is git-ignored, local-only data (AGENTS.md); this script cannot regenerate "
        "leak components at an overridden SNR without it, since dataset_e.LeakDatasetE "
        "reads each scenario's raw simulated pressure/flow trace from those files. "
        "This blocks *running* the pre-registered sweep, not building/validating it -- "
        "see tests/test_snr_sweep_f.py, which validates the pipeline with synthetic "
        "fixtures and needs no raw data.")
    return ds, cfgs


def leak_window(i: int, snr_db: float, ds_leak, scenario: dict) -> np.ndarray:
    """Deterministic leak component (background-RMS units, band-limited) at
    a fixed SNR, for window index i. Seeded from i alone -- never from
    snr_db -- so the same window's leak *shape* (position jitter, wave-speed
    jitter, attenuation jitter) is identical at every SNR level; only the
    amplitude implied by snr_db differs."""
    np.random.seed(zlib.crc32(f"leak:{i}".encode()) % (2 ** 31))
    ds_leak.snr_override_db = snr_db
    return leak_component(ds_leak, scenario)


def make_window(i: int, is_leak: bool, snr_db: float, bank: TF.Bank, ds_leak, scenarios) -> np.ndarray:
    """One (2, T) evaluation window. background/interferers/EQ/pairing are
    seeded from i alone (np.random.default_rng([EVAL_SEED, i])), identical
    across every SNR level and every checkpoint for this i."""
    rng = np.random.default_rng([EVAL_SEED, i])
    bg = TF.background(bank, rng)
    extra = A.interferers(rng)
    if is_leak:
        scenario = scenarios[i % len(scenarios)]
        extra = extra + leak_window(i, snr_db, ds_leak, scenario)
    x = bg + A.band_limit(extra)
    return A.finish(x, rng)


def build_level(snr_db: float, bank: TF.Bank, ds_leak, scenarios, n: int = N_PER_LEVEL):
    """n windows, class-balanced, index i in [0, n) -> i < n//2 is leak.
    Index-to-class assignment is fixed across every SNR level."""
    half = n // 2
    x = np.stack([make_window(i, i < half, snr_db, bank, ds_leak, scenarios) for i in range(n)])
    y = (np.arange(n) < half).astype(int)
    return x.astype(np.float32), y


def rms_score(x: np.ndarray) -> np.ndarray:
    """Loudness-only baseline: RMS over both channels, same convention as
    experiments/features.py:rms and E3's RMS baseline."""
    return np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=(1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build a handful of windows and print diagnostics; "
                         "touches no checkpoint, computes no AUROC")
    ap.add_argument("--n", type=int, default=N_PER_LEVEL)
    ap.add_argument("--snr", type=float, nargs="+", default=SNR_GRID_DB)
    args = ap.parse_args()

    bank = TF.Bank(BANK, val=True, exclude=())
    ds_leak, scenarios = load_val_leak_scenarios()
    print(f"Loaded {len(scenarios)} leak scenarios from {VAL_CSV.name}, "
          f"{len(bank.single)} real background windows, {len(bank.pairs)} Mendeley pairs")

    if args.dry_run:
        for lv in (0.0, -20.0):
            x, y = build_level(lv, bank, ds_leak, scenarios, n=8)
            print(f"  SNR {lv:+.0f} dB: shape {x.shape}, label balance {y.mean():.2f}, "
                  f"mean per-window std {x.std(axis=(1, 2)).mean():.3f} (should be ~1.0, "
                  f"joint_zscore output), rms leak-vs-noleak "
                  f"{rms_score(x[y==1]).mean():.4f} / {rms_score(x[y==0]).mean():.4f}")
        print("Dry run only -- no checkpoint loaded, no AUROC computed.")
        return

    results = {"eval_seed": EVAL_SEED, "n_per_level": args.n, "snr_grid_db": args.snr,
               "native_floor_db": SNR_FLOOR_NATIVE, "seeds": {}}
    # windows are rebuilt once per SNR level and reused for every checkpoint,
    # since construction does not depend on the checkpoint
    levels = {lv: build_level(lv, bank, ds_leak, scenarios, args.n) for lv in args.snr}

    y0 = levels[args.snr[0]][1]
    results["rms_baseline"] = {}
    for lv, (x, y) in levels.items():
        assert np.array_equal(y, y0), "class assignment must be identical across levels"
        # each window is an independently generated synthetic sample (not a
        # fragment of a shared recording), so a plain window-level bootstrap
        # is appropriate here -- unlike real Mendeley/HK/Dongguan data, which
        # this project always groups by recording/site.
        rep = detection_report(y, rms_score(x), groups=np.arange(len(y)),
                               threshold=float(np.median(rms_score(x))), n_boot=N_BOOT, seed=0)
        results["rms_baseline"][str(lv)] = rep

    for s, ckpt_name in CKPTS.items():
        model, ckpt = load_model(ckpt_name)
        assert ckpt["cfg"].get("input_norm") == "zscore", f"{ckpt_name} is not a Model F checkpoint"
        seed_entry = {"epoch": ckpt.get("epoch"), "levels": {}}
        for lv, (x, y) in levels.items():
            logit = predict_proba(model, x, logits=True)
            rep = detection_report(y, logit, groups=np.arange(len(y)), threshold=0.0,
                                   n_boot=N_BOOT, seed=0)
            seed_entry["levels"][str(lv)] = rep
            print(f"seed {s} | SNR {lv:+.0f} dB | model AUROC {rep['auroc']:.3f} "
                  f"{rep['ci95']['auroc']} | rms AUROC {results['rms_baseline'][str(lv)]['auroc']:.3f}")
        results["seeds"][str(s)] = seed_entry

    out_dir = ROOT / "results" / "snr_sweep_f"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "snr_sweep_f_results.json").write_text(json.dumps(results, indent=1))
    record_run("snr_sweep_f", vars(args), results,
               "see docs/MODEL_F_SNR_SWEEP_PREREG.md for the pre-registered interpretation")
    print(f"\nWrote {out_dir / 'snr_sweep_f_results.json'}")


if __name__ == "__main__":
    main()
