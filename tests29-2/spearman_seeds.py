"""
spearman_seeds.py - Model F seed replication: between-seed agreement (descriptive)
=================================================================================
DESCRIPTIVE ONLY. No pass rule. This implements the item "Descriptive, no pass
rule: the Spearman correlation of per-recording mean logits between each pair of
seeds, on both sensors" from section "Reported whatever the outcome" of
docs/MODEL_F_PREREG_ADDENDUM_1_SEEDS.md (addendum 1). Nothing here changes any
hypothesis or pass rule.

How it reuses E11 (experiments/model_f_eval.py):
  * load_mendeley_looped()  - same Mendeley Looped-only window sets (both sensors)
  * prepare()               - same 2 kHz band limit + joint z-score (Model_F/augment_f.py)
  * load_model()            - from experiments/_common.py, as in E11
  * predict_proba(..., logits=True) - logits, never sigmoid probabilities
  * the recording ids `g` returned by the loader are the same groups E11 uses
    for its recording-level bootstrap.
No E11 code is re-implemented. E11's channel-pairing seed applies only to public
(one-sensor) sources, which are not used here; Mendeley windows are already
two-channel, so no seed enters the scoring.

Per sensor and seed: per-window logits -> mean logit per recording. Then Spearman
rho / p between each pair of seeds over recordings. As a cross-check against the
E11 JSON, the window-level AUROC on logits (sklearn) is also reported per
seed/sensor.

Side-effect policy: does NOT call record_run(); writes only under tests29-2/spearman/.
LEAKNET_OUT is redirected to a scratch folder before imports as a safeguard.

    .venv/Scripts/python.exe tests29-2/spearman_seeds.py
    .venv/Scripts/python.exe tests29-2/spearman_seeds.py --ckpt-template best_model_f_seed{}.pt --seeds 0 1 2
"""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "spearman"
os.environ["LEAKNET_OUT"] = str(OUT / "_scratch")   # safeguard: no writes to results/ or plots/

sys.path.insert(0, str(REPO / "experiments"))

import numpy as np                                    # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

import model_f_eval as E11                            # noqa: E402
from _common import load_model, predict_proba         # noqa: E402


def per_recording_mean(logit, y, g):
    """Mean logit and label per recording id (sorted by id)."""
    ids = np.unique(g)
    mean = np.array([logit[g == i].mean() for i in ids])
    lab = np.array([y[g == i].mean() for i in ids])
    if not np.all((lab == 0) | (lab == 1)):
        raise ValueError("a recording has mixed window labels")
    return ids, mean, lab.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-template", default="best_model_f_seed{}.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading Mendeley Looped test data (as E11)...")
    raw = E11.load_mendeley_looped()                    # {name: (x, y, g)}
    data = {k: (E11.prepare(x), np.asarray(y), np.asarray(g)) for k, (x, y, g) in raw.items()}
    if not data:
        sys.exit("no Mendeley Looped data found")
    del raw

    per_rec = {k: {} for k in data}                     # sensor -> seed -> (ids, mean, lab)
    auroc = {k: {} for k in data}
    for s in args.seeds:
        ck = args.ckpt_template.format(s)
        model, c = load_model(ck)
        if c.get("cfg", {}).get("input_norm") != "zscore":
            sys.exit(f"{ck} is not a Model F checkpoint")
        print(f"seed {s}: {ck} (epoch {c.get('epoch')})")
        for k, (x, y, g) in data.items():
            logit = predict_proba(model, x, logits=True)
            per_rec[k][s] = per_recording_mean(logit, y, g)
            auroc[k][s] = float(roc_auc_score(y, logit))
        del model

    # per-recording CSV
    with open(OUT / "per_recording_logits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "recording_id", "label"] + [f"seed{s}" for s in args.seeds])
        for k in data:
            ref_ids, _, ref_lab = per_rec[k][args.seeds[0]]
            for s in args.seeds:
                assert np.array_equal(per_rec[k][s][0], ref_ids)
            for j, rid in enumerate(ref_ids):
                w.writerow([k, rid, int(ref_lab[j])] +
                           [f"{per_rec[k][s][1][j]:.6f}" for s in args.seeds])

    # Spearman per pair
    results = {"seeds": args.seeds, "window_auroc_logits": auroc, "spearman": {}}
    for k in data:
        results["spearman"][k] = {}
        for a, b in itertools.combinations(args.seeds, 2):
            rho, p = spearmanr(per_rec[k][a][1], per_rec[k][b][1])
            results["spearman"][k][f"{a}-{b}"] = {
                "rho": float(rho), "p": float(p), "n_recordings": int(len(per_rec[k][a][1]))}
    with open(OUT / "spearman_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSpearman rho of per-recording mean logits (descriptive, no pass rule)")
    print(f"{'sensor':24s} {'pair':5s} {'rho':>7s} {'p':>10s} {'n_rec':>6s}")
    for k, d in results["spearman"].items():
        for pair, r in d.items():
            print(f"{k:24s} {pair:5s} {r['rho']:7.3f} {r['p']:10.3g} {r['n_recordings']:6d}")
    print("\nWindow-level AUROC on logits (cross-check vs E11 JSON)")
    for k, d in auroc.items():
        print(f"{k:24s} " + "  ".join(f"seed{s} {v:.3f}" for s, v in d.items()))
    print(f"\nWrote {OUT / 'per_recording_logits.csv'} and {OUT / 'spearman_results.json'}")


if __name__ == "__main__":
    main()
