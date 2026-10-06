"""
xrig_band_external.py -- Phase 7 step 3: score the FROZEN Phase 7 logistic regressions on the two
sources that played no part in choosing the three bands (Dongguan, Hong Kong).
=================================================================================================
Protocol and decision rule: science/phase7/PHASE7_AUDIT.md, "Addendum 1". Written and committed
before experiments/xrig_band_intervention.py was run.

Nothing is fitted here. The four classifiers (3-band and full spectral logreg, fitted on the
Mendeley train groups and on the Sheffield train groups) are rebuilt with the SAME function
(xrig_band_intervention.fit_logreg) on the SAME groups. The fit is deterministic (lbfgs), and the
script ABORTS unless the rebuilt models reproduce the test-rig AUROCs and thresholds stored in
results/v2_cross_rig/band_intervention.json to 1e-9. This proves they are the Phase 7 models.

External windows go through the same fixed per-stream steps as the rig windows
(public_data.to_common: 5 kHz, 2 kHz low-pass, 0.4 s windows; then, per file, bank_f.usable
dead-window drop and unit-RMS scaling, exactly as in rigs.stream_windows). No statistic is fitted to them, and thresholds
stay at the training-rig Youden value.

    python experiments/xrig_band_external.py      (after xrig_band_intervention.py)
"""
import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import xrig_band_intervention as BI
from xrig_eval import full_metrics
import bank_f                                                    # noqa: E402 (Model_F on path via BI)
import public_data as P                                          # noqa: E402
import rigs                                                       # noqa: E402
from _common import _jsonable, record_run                        # noqa: E402

SOURCES = {                      # primary sources, fixed in Addendum 1
    "dongguan": lambda: P.load_dongguan(),                                   # env-noise folder excluded (loader default)
    "hongkong": lambda: P.load_hongkong(),                                   # noise loggers + hydrophones, grouped by site
}
SECONDARY = {                    # descriptive only, never part of the decision
    "hk_noiselogger": lambda: P.load_hongkong(sensors=("Noise Loggers",)),
    "hk_hydrophone": lambda: P.load_hongkong(sensors=("Hydrophones",)),
}
BAND_NAMES = [f"{lo}-{hi} Hz" for lo, hi in BI.BANDS_3]
TOL = 1e-9


_windows_of = P.windows_of
N_DROPPED = {"dead": 0}


def _stream_windows(x: np.ndarray) -> np.ndarray:
    """rigs.stream_windows applied to ONE file (one stream), after the loader's to_common:
    windows -> bank_f.usable (threshold relative to THIS stream's median RMS) -> unit RMS."""
    w = _windows_of(x).astype(np.float64)
    if not len(w):
        return w.astype(np.float32)
    keep = bank_f.usable(w)
    N_DROPPED["dead"] += int((~keep).sum())
    w = w[keep]
    return (w / np.sqrt(np.mean(w ** 2, axis=1))[:, None]).astype(np.float32)


def rig_preprocess(load) -> P.Windows:
    """Run a public_data loader with the rig pipeline's per-stream steps substituted for its
    windowing (public_data._pack calls windows_of once per file)."""
    P.windows_of = _stream_windows
    try:
        return load()
    finally:
        P.windows_of = _windows_of


def structure(w: P.Windows) -> dict:
    g = {}
    for gg, yy in zip(w.group, w.y):
        g.setdefault(str(gg), set()).add(int(yy))
    return {"n_windows": len(w), "n_windows_leak": int((w.y == 1).sum()),
            "n_windows_no_leak": int((w.y == 0).sum()),
            "n_groups_leak": sum(v == {1} for v in g.values()),
            "n_groups_no_leak": sum(v == {0} for v in g.values()),
            "n_groups_mixed_class": sum(len(v) > 1 for v in g.values()),
            "no_leak_groups": sorted(k for k, v in g.items() if v == {0})}


def clf_of(score_fn):
    """The fitted Pipeline captured by fit_logreg's score_fn closure."""
    cells = dict(zip(score_fn.__code__.co_freevars, (c.cell_contents for c in score_fn.__closure__)))
    return cells["clf"]


def rebuild(direction: str, cfg: dict, stored: dict) -> dict:
    """Refit exactly as BI.run_direction does, then verify against the stored Phase 7 result."""
    tj0 = json.loads((BI.OUT_DIR / f"{cfg['ckpt_prefix']}0.json").read_text())
    full = rigs.load_rig(cfg["train_rig"])
    tr = full.subset(np.isin(full.group, tj0["train_groups"]))
    va = full.subset(np.isin(full.group, tj0["val_groups"]))
    test = rigs.load_rig(cfg["test_rig"])
    out = {}
    for key, feat in (("full_spectral_logreg", BI.features_full), ("band3_logreg", BI.features_3band)):
        score_fn, thr = BI.fit_logreg(feat, tr, va)
        d_auc = abs(roc_auc_score(test.y, score_fn(test)) - stored[key]["auroc"])
        d_thr = abs(thr - stored["thresholds"][key])
        if d_auc > TOL or d_thr > TOL:
            sys.exit(f"ABORT: rebuilt {direction}/{key} does not reproduce band_intervention.json "
                     f"(|dAUROC|={d_auc:.2e}, |dthr|={d_thr:.2e}). Not the Phase 7 model; nothing scored.")
        clf = clf_of(score_fn)
        lr = clf[-1]
        out[key] = {"score_fn": score_fn, "thr": thr,
                    "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    print(f"  {direction}: rebuilt models reproduce band_intervention.json (tol {TOL})")
    return out


def main():
    t0 = time.time()
    stored_all = json.loads((BI.OUT_DIR / "band_intervention.json").read_text())
    models = {d: rebuild(d, cfg, stored_all[d]) for d, cfg in BI.DIRECTIONS.items()}
    fit_on = {"A_mendeley_to_sheffield": "fit_on_mendeley", "B_sheffield_to_mendeley": "fit_on_sheffield"}

    coefs = {fit_on[d]: dict(zip(BAND_NAMES, m["band3_logreg"]["coef"])) for d, m in models.items()}
    signs = {b: [int(np.sign(coefs[f][b])) for f in coefs] for b in BAND_NAMES}
    sign_agree = all(len(set(v)) == 1 for v in signs.values())
    print(f"\n3-band coefficients (standardised features): {json.dumps(coefs, indent=1)}")
    print(f"sign agreement across training rigs: {sign_agree}")

    results, data = {}, {}
    for src, loader in {**SOURCES, **SECONDARY}.items():
        N_DROPPED["dead"] = 0
        w = rig_preprocess(loader)
        data[src] = {**structure(w), "n_dropped_dead": N_DROPPED["dead"]}
        print(f"\n== {src}: {json.dumps({k: v for k, v in data[src].items() if k != 'no_leak_groups'})}")
        results[src] = {}
        for d, m in models.items():
            for key in ("band3_logreg", "full_spectral_logreg"):
                s = m[key]["score_fn"](w)
                rep = full_metrics(w.y, s, w.group, m[key]["thr"], BI.N_BOOT, BI.PAIRING_SEED)
                results[src][f"{fit_on[d]}/{key}"] = rep
                print(f"  {fit_on[d]:17s} {key:21s} window AUROC {rep['auroc']:.3f} "
                      f"CI {np.round(rep.get('auroc_ci95', [np.nan, np.nan]), 3).tolist()} "
                      f"group AUROC {rep['group']['auroc']}")

    # ── locked decision rule (Addendum 1, section E) ─────────────────────────────
    cells = []
    for src in SOURCES:
        for f in fit_on.values():
            b, fu = results[src][f"{f}/band3_logreg"], results[src][f"{f}/full_spectral_logreg"]
            gb, gf = b["group"]["auroc"], fu["group"]["auroc"]
            cells.append({"source": src, "fit": f, "band3_window": b["auroc"], "band3_group": gb,
                          "full_window": fu["auroc"], "full_group": gf,
                          "support_cell": b["auroc"] >= 0.65 and gb is not None and gb > 0.5,
                          "band3_group_ge_full": gb is not None and gf is not None and gb >= gf,
                          "fail_cell": b["auroc"] <= 0.55 or b["auroc"] <= fu["auroc"]})
    n_fail = sum(c["fail_cell"] for c in cells)
    supported = (sign_agree and all(c["support_cell"] for c in cells)
                 and sum(c["band3_group_ge_full"] for c in cells) >= 3)
    falsified = (not sign_agree) or n_fail >= 2
    verdict = "SUPPORTED (exploratory)" if supported else "FALSIFIED" if falsified else "INCONCLUSIVE"
    print(f"\ncells: {json.dumps(cells, indent=1)}\nfailing cells: {n_fail}/4\nEXTERNAL VERDICT: {verdict}")

    out = {"addendum": "science/phase7/PHASE7_AUDIT.md#addendum-1", "data_structure": data,
           "band3_coefficients": coefs, "band3_coefficient_signs": signs, "sign_agreement": sign_agree,
           "thresholds": {fit_on[d]: {k: m[k]["thr"] for k in ("band3_logreg", "full_spectral_logreg")}
                          for d, m in models.items()},
           "results": results, "decision_cells": cells, "n_fail_cells": n_fail, "verdict": verdict}
    path = BI.OUT_DIR / "band_external.json"
    path.write_text(json.dumps(_jsonable(out), indent=1))
    record_run("v2_xrig_band_external", {"bands": BI.BANDS_3, "sources": list(SOURCES),
                                         "secondary": list(SECONDARY)}, out,
               "frozen Phase 7 logregs scored on Dongguan and Hong Kong, no refit")
    print(f"\nSaved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
