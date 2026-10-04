"""
xrig_eval.py — frozen evaluation of the Version 2 cross-rig benchmark
======================================================================
Protocol: docs/VERSION2_CROSS_RIG_PREREG.md. For each checkpoint trained by
Model_F/xrig_train.py this scores the held-out test windows with Model F and with
two baselines, and reports window and group metrics with cluster-bootstrap CIs.

  cross-rig (fold -1)   test = every group of the OTHER rig; a Sheffield-trained
                        model is also scored on the Mendeley hydrophone set
  within-rig (fold f)   test = the training rig's groups with fold_assign == f

Everything fitted (network, logistic regression, RMS sign, thresholds) comes from
the training rig's train / val groups only. Scores are raw logits / decision values,
never sigmoid probabilities. Test windows are paired within their stream with a
fixed seed and get the joint z-score only (no EQ).

    python experiments/xrig_eval.py --ckpt xrig_mendeley_acc_foldx_seed0.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Model_F"))
import augment_f as A                                      # noqa: E402
import cross_dataset as CD                                 # noqa: E402
import mendeley as M                                       # noqa: E402
import metrics                                             # noqa: E402
import rigs                                                # noqa: E402
from _common import (MODELS_DIR, REPO_ROOT, _jsonable, get_device, load_model,   # noqa: E402
                     predict_proba, record_run)
from xrig_train import RigBank                             # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "v2_cross_rig"
MAX_SCORES_MB = 20


def pair_and_score(model, rw, seed: int, device, chunk: int = 4096) -> np.ndarray:
    """Model F logits for every window: same-stream partner, pairing, joint z-score."""
    bank, rng = RigBank(rw), np.random.default_rng(seed)
    out = []
    for lo in range(0, len(rw), chunk):
        xb = np.stack([A.finish(bank.two_channel(i, rng).astype(np.float64), rng, eq=False)
                       for i in range(lo, min(lo + chunk, len(rw)))])
        out.append(predict_proba(model, xb, device, batch_size=512, logits=True))
    return np.concatenate(out)


def fit_baselines(train: "rigs.RigWindows", val: "rigs.RigWindows"):
    """Returns {name: score function}, thresholds {name: t}, fitted on train, thresholded on val."""
    sign = 1.0 if roc_auc_score(train.y, train.log_rms) >= 0.5 else -1.0
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    clf.fit(CD.features_1ch(CD.standardise(train.x)), train.y)
    fns = {"rms": lambda w: sign * w.log_rms.astype(np.float64),
           "logreg": lambda w: clf.decision_function(CD.features_1ch(CD.standardise(w.x)))}
    thr = {k: rigs.youden_threshold(val.y, f(val)) for k, f in fns.items()}
    return fns, thr, {"rms_sign": sign}


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def full_metrics(y, s, groups, thr, n_boot, seed) -> dict:
    y, s, groups = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64), np.asarray(groups)
    rep = metrics.detection_report(y, s, groups, threshold=thr, n_boot=n_boot, seed=seed)
    pred = s >= thr
    rep.update(auprc=float(average_precision_score(y, s)), prevalence=float(y.mean()),
               confusion={"tp": int((pred & (y == 1)).sum()), "fn": int((~pred & (y == 1)).sum()),
                          "tn": int((~pred & (y == 0)).sum()), "fp": int((pred & (y == 0)).sum()),
                          "windows": len(y)})
    ug = np.unique(groups)
    gm = np.array([s[groups == g].mean() for g in ug])
    gy = np.array([int(y[groups == g][0]) for g in ug])
    gp = gm >= thr
    rep["group"] = {"auroc": _auroc(gy, gm), "n_groups_leak": int((gy == 1).sum()),
                    "n_groups_no_leak": int((gy == 0).sum()),
                    "confusion": {"tp": int((gp & (gy == 1)).sum()), "fn": int((~gp & (gy == 1)).sum()),
                                  "tn": int((~gp & (gy == 0)).sum()), "fp": int((gp & (gy == 0)).sum())}}
    return rep


def breakdowns(rw: "rigs.RigWindows", score: np.ndarray, rig: str) -> dict:
    """Descriptive window AUROCs; never used to select anything."""
    y, out = rw.y.astype(int), {}

    def part(name, mask):
        out[name] = {"auroc": _auroc(y[mask], score[mask]), "n_leak": int((y[mask] == 1).sum()),
                     "n_no_leak": int((y[mask] == 0).sum())}

    if rig == "sheffield":
        pos = np.array([int(m.split("=")[1]) for m in rw.meta])
        for name, sel in (("pos0", pos == 0), ("pos1_10", (pos >= 1) & (pos <= 10)), ("pos15plus", pos >= 15)):
            part(f"distance/{name}", (y == 0) | ((y == 1) & sel))
    else:
        topo = np.array([m.split("/")[0] for m in rw.meta])
        cond = np.array([m.split("/")[1] for m in rw.meta])
        for t in M.TOPOLOGIES:
            part(f"topology/{t}", topo == t)
        for c in M.LEAK_TYPES:
            part(f"leak_type/{c}", (y == 0) | (cond == c))
    return out


def build_test_sets(cfg: dict, override=None) -> dict:
    rig, fold = cfg["train_rig"], cfg["fold"]
    if fold >= 0:
        rw = rigs.load_rig(rig)
        f = rigs.fold_assign(rw.group, rw.y)
        held = np.array([f[str(g)] == fold for g in rw.group])
        assert not (set(rw.group[held]) & (set(cfg["train_groups"]) | set(cfg["val_groups"]))), \
            "test groups overlap the checkpoint's train/val groups"
        return {f"{rig}_fold{fold}": (rw.subset(held), rig)}
    names = override or (["mendeley_acc"] if rig == "sheffield" else ["sheffield"])
    if rig == "sheffield" and not override:
        names.append("mendeley_hyd")
    assert rig not in names, "cross-rig test must not include the training rig"
    return {n: (rigs.load_rig(n), n) for n in names}


def evaluate_ckpt(name: str, args) -> dict:
    device = get_device()
    model, ck = load_model(name, device)
    cfg, stem = ck["cfg"], Path(name).stem
    tj = OUT_DIR / f"{stem}.json"
    if not tj.exists():
        sys.exit(f"missing training record {tj}")
    thr_f = json.loads(tj.read_text())["threshold_val_youden"]
    train_rw = rigs.load_rig(cfg["train_rig"])
    tr = train_rw.subset(np.isin(train_rw.group, cfg["train_groups"]))
    va = train_rw.subset(np.isin(train_rw.group, cfg["val_groups"]))
    fns, thr_b, extra = fit_baselines(tr, va)
    thresholds = {"model_f": thr_f, **thr_b}
    del train_rw

    res = {"checkpoint": name, "train_rig": cfg["train_rig"], "fold": cfg["fold"], "seed_pairing": args.seed,
           "thresholds": thresholds, **extra, "test_sets": {}}
    scores_npz = {}
    for tname, (rw, rig) in build_test_sets(cfg, args.test_rigs).items():
        sc = {"model_f": pair_and_score(model, rw, args.seed, device)}
        sc.update({k: f(rw) for k, f in fns.items()})
        entry = {"summary": rw.summary(), "methods": {}, "breakdowns": {}}
        for m, s in sc.items():
            kinds = {"val_youden": thresholds[m]}
            if m == "model_f":
                kinds["logit0"] = 0.0
            entry["methods"][m] = {k: full_metrics(rw.y, s, rw.group, t, args.n_boot, args.seed)
                                   for k, t in kinds.items()}
            entry["breakdowns"][m] = breakdowns(rw, s, rig)
            scores_npz[f"{tname}__{m}"] = s.astype(np.float32)
        for f in ("y", "group", "stream", "meta"):
            scores_npz[f"{tname}__{f}"] = getattr(rw, f)
        res["test_sets"][tname] = entry
        print_table(tname, entry)
    return res, scores_npz, stem


def print_table(tname: str, entry: dict):
    s = entry["summary"]
    print(f"\n== test set {tname}: {s['n_windows']} windows, groups leak/no-leak "
          f"{s['n_groups_leak']}/{s['n_groups_no_leak']}, prevalence "
          f"{s['n_windows_leak'] / s['n_windows']:.2f}")
    print(f"{'method':8s} {'thr':10s} {'win AUROC [95% CI]':26s} {'grp AUROC':9s} {'AUPRC':6s} "
          f"{'BalAcc':6s} {'Det':5s} {'FA':5s}")
    for m, kinds in entry["methods"].items():
        for k, r in kinds.items():
            g = r["group"]["auroc"]
            print(f"{m:8s} {k:10s} {metrics.fmt(r):26s} {'n/a' if g is None else f'{g:.3f}':9s} "
                  f"{r['auprc']:.3f}  {r['balanced_accuracy']:.3f}  {r['detection_rate']:.2f}  "
                  f"{r['false_alarm_rate']:.2f}")


def unique_path(p: Path) -> Path:
    return p if not p.exists() else p.with_name(f"{p.stem}_{time.strftime('%Y%m%d_%H%M%S')}{p.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint names in models/")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="pairing / bootstrap seed")
    ap.add_argument("--test-rigs", nargs="*", default=None, help="override cross-rig test rigs")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.ckpt:
        res, scores, stem = evaluate_ckpt(name, args)
        path = unique_path(OUT_DIR / f"eval_{stem}.json")
        path.write_text(json.dumps(_jsonable(res), indent=1))
        sp = unique_path(OUT_DIR / f"scores_{stem}.npz")
        np.savez_compressed(sp, **scores)
        if sp.stat().st_size > MAX_SCORES_MB * 2 ** 20:
            sp.unlink()
            print(f"scores file over {MAX_SCORES_MB} MB, not kept")
        first = next(iter(res["test_sets"].values()))["methods"]["model_f"]["val_youden"]
        record_run("v2_xrig_eval", {"checkpoint": name, "n_boot": args.n_boot, "seed": args.seed},
                   res, f"{stem}: model_f window AUROC {metrics.fmt(first)} on "
                        f"{next(iter(res['test_sets']))}")
        print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
