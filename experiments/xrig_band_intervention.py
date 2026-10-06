"""
xrig_band_intervention.py -- Phase 7 locked test of a physically motivated 3-band
representation, following the exploratory mechanism analysis (xrig_mechanism.py).
======================================================================================
METHODOLOGICAL STATUS: the three bands (400-600, 600-800, 1600-2000 Hz) were selected by
inspecting leak-AUROC and rig-AUROC on BOTH rigs (xrig_mechanism.py, part 1). This run is
therefore a HYPOTHESIS-DRIVEN EXPLORATORY INTERVENTION, not an independent confirmatory
test of those specific bands -- the test rig's own labels informed which bands were
chosen. What IS locked here: the classifier (StandardScaler + LogisticRegression(C=0.5,
class_weight="balanced")), the feature definition (fixed below, not touched after seeing
results), the train/val/test physical groups (identical to
docs/VERSION2_CROSS_RIG_PREREG.md), the threshold procedure (Youden on training-rig
validation only, frozen before the test rig is scored), and the bootstrap procedure
(cluster bootstrap over physical groups, 2000 draws). This script runs the comparison
once; no band, C, classifier, or threshold change is made after seeing the AUROC below.

DATA NOTE: datasets/Accelerometer/Accelerometer/ has no Branched/ folder on this machine.
A time-boxed search (repo, git history, local zip archives, sibling directories,
documented download sources) did not recover it. Per instruction this run freezes
Mendeley = Looped-only, 20 physical groups (16 leak / 4 no-leak) -- NOT the 40-group set
used in the originally audited docs/VERSION2_CROSS_RIG_BASELINE.md. Direction A's test
rig (Sheffield, 11 groups) is unaffected. Model F and the full-feature spectral baseline
are RE-EVALUATED here (inference / refit only, no retraining of Model F) on this reduced
set so the three-way comparison is apples-to-apples; their numbers will not exactly match
the audited 40-group write-up, and that is reported explicitly below, not hidden.

    python experiments/xrig_band_intervention.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Model_F"))
import cross_dataset as CD                                   # noqa: E402
import metrics                                                  # noqa: E402
import public_data as P                                        # noqa: E402
import rigs                                                     # noqa: E402
from xrig_eval import full_metrics, pair_and_score            # noqa: E402
from _common import MODELS_DIR, REPO_ROOT, _jsonable, get_device, load_model, record_run   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "v2_cross_rig"
PAIRING_SEED = 0
N_BOOT = 2000
BANDS_3 = ((400, 600), (600, 800), (1600, 2000))
DIRECTIONS = {
    "A_mendeley_to_sheffield": {"train_rig": "mendeley_acc", "test_rig": "sheffield",
                                "ckpt_prefix": "xrig_mendeley_acc_foldx_seed"},
    "B_sheffield_to_mendeley": {"train_rig": "sheffield", "test_rig": "mendeley_acc",
                                "ckpt_prefix": "xrig_sheffield_foldx_seed"},
}


def features_3band(x: np.ndarray, fs: int = P.FS) -> np.ndarray:
    """log10 normalised-PSD energy fraction in the three bands identified in
    xrig_mechanism.py as leak-discriminative with the SAME sign on both rigs. Same
    convention as cross_dataset.features_1ch's band columns, restricted to 3 bands."""
    x = x.astype(np.float64)
    f, pxx = welch(x, fs=fs, nperseg=256, axis=-1)
    tot = pxx.sum(axis=1, keepdims=True) + 1e-30
    cols = [np.log10(pxx[:, (f >= lo) & (f < hi)].sum(axis=1) / tot[:, 0] + 1e-12)
           for lo, hi in BANDS_3]
    return np.column_stack(cols)


def features_full(x: np.ndarray) -> np.ndarray:
    """The existing baseline's feature set, unchanged (CD.features_1ch on standardised x)."""
    return CD.features_1ch(CD.standardise(x))


def fit_logreg(feat_fn, train_rw, val_rw):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    clf.fit(feat_fn(train_rw.x), train_rw.y)
    score_fn = lambda rw: clf.decision_function(feat_fn(rw.x))      # noqa: E731
    thr = rigs.youden_threshold(val_rw.y, score_fn(val_rw))
    return score_fn, thr


def run_direction(name: str, cfg: dict) -> dict:
    train_rig, test_rig, prefix = cfg["train_rig"], cfg["test_rig"], cfg["ckpt_prefix"]
    tj0 = json.loads((OUT_DIR / f"{prefix}0.json").read_text())
    rec_train_groups, rec_val_groups = tj0["train_groups"], tj0["val_groups"]

    train_rw_full = rigs.load_rig(train_rig)
    tr = train_rw_full.subset(np.isin(train_rw_full.group, rec_train_groups))
    va = train_rw_full.subset(np.isin(train_rw_full.group, rec_val_groups))
    data_note = {
        "train_rig_groups_recorded_at_training_time": len(set(rec_train_groups)),
        "train_rig_groups_available_now": len(set(tr.group.tolist())),
        "val_rig_groups_recorded_at_training_time": len(set(rec_val_groups)),
        "val_rig_groups_available_now": len(set(va.group.tolist())),
    }
    test_rw = rigs.load_rig(test_rig)
    print(f"\n== {name}: train rig {train_rig} (train groups now {data_note['train_rig_groups_available_now']}"
         f"/{data_note['train_rig_groups_recorded_at_training_time']} recorded, val groups now "
         f"{data_note['val_rig_groups_available_now']}/{data_note['val_rig_groups_recorded_at_training_time']} "
         f"recorded) -> test rig {test_rig} ({test_rw.summary()['n_groups']} groups, "
         f"{test_rw.summary()['n_windows']} windows) ==")

    full_score_fn, full_thr = fit_logreg(features_full, tr, va)
    band_score_fn, band_thr = fit_logreg(features_3band, tr, va)

    device = get_device()
    model_f_rows = {}
    for seed in (0, 1, 2):
        ckpt = f"{prefix}{seed}.pt"
        model, ck = load_model(ckpt, device)
        tjs = json.loads((OUT_DIR / f"{prefix}{seed}.json").read_text())
        thr_f = tjs["threshold_val_youden"]
        s = pair_and_score(model, test_rw, PAIRING_SEED, device)
        model_f_rows[seed] = full_metrics(test_rw.y, s, test_rw.group, thr_f, N_BOOT, PAIRING_SEED)

    results = {
        "data_note": data_note,
        "model_f": {str(k): v for k, v in model_f_rows.items()},
        "model_f_mean_window_auroc": float(np.mean([r["auroc"] for r in model_f_rows.values()])),
        "full_spectral_logreg": full_metrics(test_rw.y, full_score_fn(test_rw), test_rw.group,
                                             full_thr, N_BOOT, PAIRING_SEED),
        "band3_logreg": full_metrics(test_rw.y, band_score_fn(test_rw), test_rw.group,
                                     band_thr, N_BOOT, PAIRING_SEED),
        "thresholds": {"model_f_per_seed": {str(s): json.loads((OUT_DIR / f"{prefix}{s}.json").read_text())
                                            ["threshold_val_youden"] for s in (0, 1, 2)},
                      "full_spectral_logreg": full_thr, "band3_logreg": band_thr},
    }
    print(f"  model_f mean window AUROC: {results['model_f_mean_window_auroc']:.3f}")
    print(f"  full_spectral_logreg window AUROC: {metrics.fmt(results['full_spectral_logreg'])}")
    print(f"  band3_logreg window AUROC: {metrics.fmt(results['band3_logreg'])}")
    return results


def main():
    t0 = time.time()
    out = {d: run_direction(d, cfg) for d, cfg in DIRECTIONS.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "band_intervention.json"
    path.write_text(json.dumps(_jsonable(out), indent=1))
    record_run("v2_xrig_band_intervention", {"bands": BANDS_3, "pairing_seed": PAIRING_SEED}, out,
              "locked 3-band logistic-regression intervention vs Model F and the full spectral baseline")
    print(f"\nSaved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
