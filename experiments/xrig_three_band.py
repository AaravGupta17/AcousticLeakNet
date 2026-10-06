"""
xrig_three_band.py — Phase 7 three-band normalised-PSD cross-rig intervention
==============================================================================
Protocol: docs/PHASE7_THREE_BAND_PREREG.md (locked). Exploratory, not confirmatory.

Two logistic models on the existing spectral features (cross_dataset.features_1ch):
  three_band  columns 4, 5, 8  (400-600, 600-800, 1600-2000 Hz log10 power fractions)
  full14      all 14 columns   (reproduction check of the frozen logreg baseline)
Directions: A mendeley_acc -> sheffield, B sheffield -> mendeley_acc. Train/val groups come
from the frozen xrig training records; the threshold is fixed on val before the test rig is
loaded. Scores are raw decision values.

    python experiments/xrig_three_band.py
"""

import gc
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cross_dataset as CD                                  # noqa: E402
import metrics                                              # noqa: E402
import rigs                                                 # noqa: E402
import xrig_eval as XE                                      # noqa: E402
from _common import REPO_ROOT, _jsonable, record_run         # noqa: E402

V2_DIR = REPO_ROOT / "results" / "v2_cross_rig"
OUT_DIR = V2_DIR / "phase7"
THREE_BAND_COLS = (4, 5, 8)
EXPECTED_BANDS = ((400, 600), (600, 800), (1600, 2000))
N_FEATURES = 14
CLF_PARAMS = dict(max_iter=3000, class_weight="balanced", C=0.5)
DIRECTIONS = (("A", "mendeley_acc", "sheffield"), ("B", "sheffield", "mendeley_acc"))
N_BOOT, SEED, CHUNK = 2000, 0, 20000
ASSERTIONS = []


def check(cond, msg):
    assert cond, f"ASSERTION FAILED: {msg}"
    ASSERTIONS.append(msg)
    print(f"[assert ok] {msg}")


def make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(**CLF_PARAMS))


def feats14(x) -> np.ndarray:
    out = [CD.features_1ch(CD.standardise(x[i:i + CHUNK])).astype(np.float64)
           for i in range(0, len(x), CHUNK)]
    return np.concatenate(out)


def paired_diff_bootstrap(y, s1, s2, groups, n_boot=N_BOOT, seed=SEED) -> dict:
    """Stratified (within class) cluster bootstrap over groups; same resample for both scores."""
    y, groups = np.asarray(y).astype(int), np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    cls = {g: int(y[idx_of[g]][0]) for g in uniq}
    assert all(len(np.unique(y[idx_of[g]])) == 1 for g in uniq), "mixed-class group"
    strata = [[g for g in uniq if cls[g] == c] for c in (0, 1)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        chosen = []
        for gs in strata:
            chosen.extend(gs[i] for i in rng.choice(len(gs), size=len(gs), replace=True))
        sel = np.concatenate([idx_of[g] for g in chosen])
        if len(np.unique(y[sel])) < 2:
            continue
        diffs.append(roc_auc_score(y[sel], s1[sel]) - roc_auc_score(y[sel], s2[sel]))
    d = np.asarray(diffs)
    point = float(roc_auc_score(y, s1) - roc_auc_score(y, s2))
    return {"point_diff": point, "boot_mean_diff": float(d.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "n_boot": n_boot, "n_valid_draws": len(d), "seed": seed}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"unavailable: {e}"


def peak_rss_mb():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return float(getattr(mi, "peak_wset", mi.rss) / 2 ** 20)
    except Exception:                                        # noqa: BLE001
        return None


def frozen_entry(r: dict) -> dict:
    return {"auroc": r["auroc"], "ci95": r["ci95"]["auroc"], "auprc": r["auprc"],
            "balanced_accuracy": r["balanced_accuracy"], "sensitivity": r["detection_rate"],
            "specificity": 1 - r["false_alarm_rate"], "group_auroc": r["group"]["auroc"],
            "threshold": r["threshold"]}


def comparators(train_rig, test_rig) -> dict:
    out = {"model_f": {}, "logreg_frozen": None}
    for sd in (0, 1, 2):
        d = json.loads((V2_DIR / f"eval_xrig_{train_rig}_foldx_seed{sd}.json").read_text())
        m = d["test_sets"][test_rig]["methods"]
        out["model_f"][f"seed{sd}"] = frozen_entry(m["model_f"]["val_youden"])
        if sd == 0:
            out["logreg_frozen"] = frozen_entry(m["logreg"]["val_youden"])
    arr = {k: np.array([v[k] for v in out["model_f"].values()]) for k in
           ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "group_auroc")}
    out["model_f_summary"] = {k: {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}
                              for k, a in arr.items()}
    return out


def row(name, r, thr):
    g = r["group"]["auroc"]
    return (f"{name:16s} {metrics.fmt(r):24s} {'n/a' if g is None else f'{g:.3f}':>8s} {r['auprc']:.3f}  "
            f"{r['balanced_accuracy']:.3f}  {r['detection_rate']:.3f}  {1 - r['false_alarm_rate']:.3f}  {thr:+.4f}")


def frow(name, e):
    g = e["group_auroc"]
    ci = e["ci95"]
    return (f"{name:16s} {e['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]".ljust(41)
            + f" {'n/a' if g is None else f'{g:.3f}':>8s} {e['auprc']:.3f}  {e['balanced_accuracy']:.3f}  "
              f"{e['sensitivity']:.3f}  {e['specificity']:.3f}  {e['threshold']:+.4f}")


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── feature-config assertions ──
    for c, band in zip(THREE_BAND_COLS, EXPECTED_BANDS):
        check((CD.BAND_EDGES[c], CD.BAND_EDGES[c + 1]) == band, f"col {c} is band {band} Hz")
    probe = CD.features_1ch(CD.standardise(np.random.default_rng(0).standard_normal((8, 2000))))
    check(probe.shape[1] == N_FEATURES, f"features_1ch has {N_FEATURES} columns")
    check(probe[:, list(THREE_BAND_COLS)].shape == (8, 3), "three-band selection has shape (N, 3)")
    for nm, m in (("three_band", make_clf()), ("full14", make_clf())):
        p = m.named_steps["logisticregression"].get_params()
        check(p["C"] == 0.5 and p["class_weight"] == "balanced" and p["max_iter"] == 3000,
              f"{nm} clf params C=0.5, class_weight=balanced, max_iter=3000")

    cols3 = list(THREE_BAND_COLS)
    results = {"directions": {}}
    timings = {}
    for tag, train_rig, test_rig in DIRECTIONS:
        t0 = time.time()
        print(f"\n######## Direction {tag}: train {train_rig} -> test {test_rig}")
        rec = json.loads((V2_DIR / f"xrig_{train_rig}_foldx_seed0.json").read_text())
        train_groups, val_groups = set(rec["train_groups"]), set(rec["val_groups"])
        rw = rigs.load_rig(train_rig)
        all_groups = set(map(str, rw.group))
        vm = rigs.val_split(rw.group, rw.y)
        check(set(map(str, rw.group[vm])) == val_groups, f"{tag}: recomputed val_split == stored val_groups")
        check(train_groups | val_groups == all_groups and not (train_groups & val_groups),
              f"{tag}: train U val == all {train_rig} groups, disjoint")
        tr = rw.subset(np.isin(rw.group, sorted(train_groups)))
        va = rw.subset(np.isin(rw.group, sorted(val_groups)))
        n_tr, n_va = len(tr), len(va)
        del rw
        ftr, fva = feats14(tr.x), feats14(va.x)
        check(ftr.shape[1] == N_FEATURES and ftr[:, cols3].shape == (n_tr, 3), f"{tag}: train feature shapes")
        models = {"three_band": (make_clf(), cols3), "full14": (make_clf(), list(range(N_FEATURES)))}
        thr, val_auc = {}, {}
        for nm, (clf, cols) in models.items():
            clf.fit(ftr[:, cols], tr.y)
            sv = clf.decision_function(fva[:, cols])
            thr[nm] = rigs.youden_threshold(va.y, sv)
            val_auc[nm] = float(roc_auc_score(va.y, sv))
            print(f"  {nm}: val AUROC {val_auc[nm]:.4f}, frozen threshold {thr[nm]:+.5f}")
        train_summary = {"n_train_windows": n_tr, "n_val_windows": n_va,
                         "n_train_groups": len(train_groups), "n_val_groups": len(val_groups)}
        del tr, va, ftr, fva
        gc.collect()

        # ── only now load the test rig ──
        check(test_rig != train_rig, f"{tag}: test rig != train rig")
        te = rigs.load_rig(test_rig)
        check(not (set(map(str, te.group)) & all_groups), f"{tag}: test groups disjoint from train-rig groups")
        fte = feats14(te.x)
        check(fte.shape[1] == N_FEATURES and fte[:, cols3].shape == (len(te), 3), f"{tag}: test feature shapes")
        scores = {nm: clf.decision_function(fte[:, cols]).astype(np.float64)
                  for nm, (clf, cols) in models.items()}
        del fte
        gc.collect()

        entry = {"train_rig": train_rig, "test_rig": test_rig, "thresholds": thr, "val_auroc": val_auc,
                 "train_val": train_summary, "test_summary": te.summary(), "methods": {}, "breakdowns": {},
                 "clf_params": {nm: m[0].named_steps["logisticregression"].get_params()
                                for nm, m in models.items()}}
        if test_rig == "mendeley_acc":
            topo = np.array([m.split("/")[0] for m in te.meta])
            entry["test_topology_groups"] = {t: {"leak": len({g for g, yy in zip(te.group[topo == t], te.y[topo == t]) if yy == 1}),
                                                 "no_leak": len({g for g, yy in zip(te.group[topo == t], te.y[topo == t]) if yy == 0})}
                                             for t in sorted(set(topo))}
        for nm, s in scores.items():
            entry["methods"][nm] = XE.full_metrics(te.y, s, te.group, thr[nm], N_BOOT, SEED)
            entry["breakdowns"][nm] = XE.breakdowns(te, s, test_rig)
        entry["paired_diff_three_band_minus_full14"] = paired_diff_bootstrap(
            te.y, scores["three_band"], scores["full14"], te.group)
        comp = comparators(train_rig, test_rig)
        entry["comparators"] = comp

        # reproduction check vs frozen logreg
        frozen_auc = comp["logreg_frozen"]["auroc"]
        diff = abs(entry["methods"]["full14"]["auroc"] - frozen_auc)
        entry["full14_reproduction"] = {"refit_auroc": entry["methods"]["full14"]["auroc"],
                                        "frozen_auroc": frozen_auc, "abs_diff": diff}
        np.savez_compressed(OUT_DIR / f"scores_{tag}.npz",
                            three_band=scores["three_band"].astype(np.float32),
                            full14=scores["full14"].astype(np.float32),
                            y=te.y, group=te.group, stream=te.stream, meta=te.meta)

        # ── table ──
        s = entry["test_summary"]
        print(f"\n== Direction {tag}: {train_rig} -> {test_rig}: {s['n_windows']} windows, groups leak/no-leak "
              f"{s['n_groups_leak']}/{s['n_groups_no_leak']}, prevalence {s['n_windows_leak'] / s['n_windows']:.3f}")
        print(f"{'method':16s} {'win AUROC [95% CI]':24s} {'grp AUROC':>8s} {'AUPRC'}  BalAcc Sens   Spec   thr")
        for nm in ("three_band", "full14"):
            print(row(nm, entry["methods"][nm], thr[nm]))
        print(frow("frozen_logreg", comp["logreg_frozen"]))
        for k, v in comp["model_f"].items():
            print(frow(f"model_f_{k}", v))
        ms = comp["model_f_summary"]
        print("model_f mean/range: " + ", ".join(
            f"{k} {v['mean']:.3f} [{v['min']:.3f}, {v['max']:.3f}]" for k, v in ms.items()))
        pd_ = entry["paired_diff_three_band_minus_full14"]
        print(f"paired diff (three_band - full14) window AUROC: point {pd_['point_diff']:+.4f}, "
              f"boot mean {pd_['boot_mean_diff']:+.4f}, 95% CI [{pd_['ci95'][0]:+.4f}, {pd_['ci95'][1]:+.4f}] "
              f"({pd_['n_valid_draws']} draws)")
        print(f"full14 reproduction: refit {entry['full14_reproduction']['refit_auroc']:.10f} vs frozen "
              f"{frozen_auc:.10f}, |diff| {diff:.2e}")
        results["directions"][tag] = entry
        check(diff <= 1e-6, f"{tag}: refit full14 window AUROC matches frozen logreg within 1e-6 (diff {diff:.2e})")
        del te, scores
        gc.collect()
        timings[f"direction_{tag}_s"] = time.time() - t0

    timings["total_s"] = time.time() - t_start
    prov = {"git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain"),
            "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "scipy": scipy.__version__, "hostname": socket.gethostname(),
            "sha256_cache_xrig_mendeley_acc": sha256(rigs.XRIG_CACHE / "mendeley_acc.npz"),
            "sha256_cache_xrig_sheffield": sha256(rigs.XRIG_CACHE / "sheffield.npz"),
            "peak_rss_mb": peak_rss_mb(), "timings": timings}
    try:
        import torch
        prov["gpu_name_not_used_for_compute"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:                                        # noqa: BLE001
        prov["gpu_name_not_used_for_compute"] = None
    config = {"prereg": "docs/PHASE7_THREE_BAND_PREREG.md", "three_band_cols": list(THREE_BAND_COLS),
              "three_band_edges_hz": [list(b) for b in EXPECTED_BANDS], "n_features_full": N_FEATURES,
              "band_edges": list(CD.BAND_EDGES), "classifier": CLF_PARAMS, "n_boot": N_BOOT, "seed": SEED,
              "directions": [list(d) for d in DIRECTIONS]}
    results.update(provenance=prov, config=config, assertions=ASSERTIONS)
    (OUT_DIR / "three_band_results.json").write_text(json.dumps(_jsonable(results), indent=1))
    a, b = results["directions"]["A"], results["directions"]["B"]
    summ = (f"three_band AUROC A {a['methods']['three_band']['auroc']:.3f} B {b['methods']['three_band']['auroc']:.3f}; "
            f"full14 A {a['methods']['full14']['auroc']:.3f} B {b['methods']['full14']['auroc']:.3f}")
    record_run("v2_xrig_three_band", config, results, summ)
    print(f"\nSaved {OUT_DIR / 'three_band_results.json'}\n{summ}")


if __name__ == "__main__":
    main()
