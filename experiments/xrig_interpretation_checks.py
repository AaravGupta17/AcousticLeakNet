"""
xrig_interpretation_checks.py — descriptive checks for docs/V2_CROSS_RIG_FINAL_INTERPRETATION.md
================================================================================================
Reads only saved scores (no fitting, no thresholds chosen, nothing selected). Newly computed,
descriptive, written after the Phase 7 results were seen.

  1. Per-group one-group AUROC side by side (3-band, full14, Model F seed mean) with flags
     "all < 0.5", "full > 3-band + 0.05", "3-band > full + 0.05".
  2. Share of window-score variance explained by stream (recording x channel) identity.
  3. Per-channel (Mendeley ch1 / ch2) window AUROC and flagged fraction in direction B.

    python experiments/xrig_interpretation_checks.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import REPO_ROOT, _jsonable                    # noqa: E402

V2 = REPO_ROOT / "results" / "v2_cross_rig"
P7 = V2 / "phase7"
OUT = P7 / "interpretation_checks.json"
TRAIN_RIG = {"A": "mendeley_acc", "B": "sheffield"}
TEST_RIG = {"A": "sheffield", "B": "mendeley_acc"}


def stream_r2(s, stream):
    mu = {k: s[stream == k].mean() for k in np.unique(stream)}
    fitted = np.array([mu[k] for k in stream])
    return float(1 - ((s - fitted) ** 2).sum() / ((s - s.mean()) ** 2).sum())


def main():
    frozen = json.loads((P7 / "three_band_results.json").read_text())
    out = {"note": "descriptive only; computed after results were seen; nothing selected from it",
           "per_group": {}, "stream_variance_share": {}, "per_channel_B": {}}

    rows = list(csv.DictReader(open(P7 / "group_analysis" / "per_group.csv")))
    table = defaultdict(dict)
    for r in rows:
        table[(r["direction"], r["group"], int(r["label"]))][r["method"]] = float(r["one_group_auroc"])
    for (d, g, lab), v in sorted(table.items()):
        mf = float(np.mean([v[f"model_f_seed{i}"] for i in range(3)]))
        tb, fl = v["three_band"], v["full14"]
        out["per_group"].setdefault(d, []).append(
            {"group": g, "label": lab, "three_band": tb, "full14": fl, "model_f_mean": mf,
             "all_below_0.5": max(tb, fl, mf) < 0.5, "full_gt_3band": fl > tb + 0.05,
             "3band_gt_full": tb > fl + 0.05})
    for d, lst in out["per_group"].items():
        out[f"per_group_counts_{d}"] = {
            k: int(sum(e[k] for e in lst)) for k in ("all_below_0.5", "full_gt_3band", "3band_gt_full")}
        out[f"per_group_counts_{d}"]["n_groups"] = len(lst)

    for d in "AB":
        z = np.load(P7 / f"scores_{d}.npz", allow_pickle=True)
        y, st = z["y"].astype(int), z["stream"]
        mf = [np.load(V2 / f"scores_xrig_{TRAIN_RIG[d]}_foldx_seed{i}.npz", allow_pickle=True)
              for i in range(3)]
        for i, m in enumerate(mf):
            assert np.array_equal(m[f"{TEST_RIG[d]}__y"].astype(int), y), "Model F order mismatch"
        scores = {"three_band": z["three_band"].astype(np.float64), "full14": z["full14"].astype(np.float64)}
        scores.update({f"model_f_seed{i}": m[f"{TEST_RIG[d]}__model_f"].astype(np.float64)
                       for i, m in enumerate(mf)})
        out["stream_variance_share"][d] = {"n_streams": int(len(np.unique(st))),
                                           **{k: stream_r2(s, st) for k, s in scores.items()}}
        if d == "B":
            ch = np.array([s.rsplit("|", 1)[1] for s in st])
            thr = frozen["directions"]["B"]["thresholds"]
            thr.update({f"model_f_seed{i}": frozen["directions"]["B"]["comparators"]["model_f"][f"seed{i}"]["threshold"]
                        for i in range(3)})
            for k, s in scores.items():
                out["per_channel_B"][k] = {
                    c: {"window_auroc": float(roc_auc_score(y[ch == c], s[ch == c])),
                        "frac_flagged_all": float((s[ch == c] >= thr[k]).mean()),
                        "frac_flagged_no_leak": float((s[(ch == c) & (y == 0)] >= thr[k]).mean())}
                    for c in sorted(set(ch))}
    OUT.write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_group"}, indent=1))


if __name__ == "__main__":
    main()
