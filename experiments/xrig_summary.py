"""
xrig_summary.py — aggregate the Version 2 cross-rig results
============================================================
Pure aggregation of existing records in results/v2_cross_rig/ (eval_xrig_*.json,
xrig_*.json training records, scores_*.npz). Nothing is trained or re-evaluated and
no run record is written. The outcome classes are applied exactly as fixed in
docs/VERSION2_CROSS_RIG_PREREG.md ("Outcome classes"); they are not tuned here.

  direction A: train mendeley_acc -> test sheffield
  direction B: train sheffield    -> test mendeley_acc (primary), mendeley_hyd (secondary)
  within-rig : grouped 3-fold CV per rig (fold >= 0), seed 0

Only a checkpoint with a training record counts. If several eval files exist for one
checkpoint stem the newest is used and the others are listed as ignored duplicates.
Writes results/v2_cross_rig/summary.json (or <LEAKNET_OUT>/v2_cross_rig/), never
overwriting an existing file.

    python experiments/xrig_summary.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v2_cross_rig"
METHODS = ("model_f", "rms", "logreg")
EXPECTED_SEEDS = (0, 1, 2)
TOL = 1e-9


# --------------------------------------------------------------------------- classes
def classify_direction(auroc, base):
    """Per-direction class from the prereg. Returns (label, case_d_flag)."""
    if auroc < 0.40:
        label = "Collapse (inverted ranking)"
    elif auroc <= 0.60:
        label = "Collapse"
    elif auroc >= 0.80 and auroc - base >= 0.05:
        label = "Strong"
    else:
        label = "Moderate/asymmetric"
    return label, bool(base >= auroc - 0.05)


def overall_class(label_a, label_b):
    """Strong needs both directions; Collapse if either collapses; else Moderate."""
    if label_a.startswith("Collapse") or label_b.startswith("Collapse"):
        return "Collapse"
    if label_a == "Strong" and label_b == "Strong":
        return "Strong"
    return "Moderate/asymmetric"


# --------------------------------------------------------------------------- loading
def stem_of(ckpt):
    return ckpt[:-3] if ckpt.endswith(".pt") else ckpt


def load_runs(res=RES):
    """Newest eval per checkpoint stem, paired with its training record."""
    by_stem = {}
    for p in sorted(res.glob("eval_xrig_*.json")):
        d = json.loads(p.read_text())
        by_stem.setdefault(stem_of(d["checkpoint"]), []).append((p.stat().st_mtime, p, d))
    runs, ignored, no_train = {}, {}, []
    for stem, lst in by_stem.items():
        lst.sort(key=lambda t: t[0])
        _, p, d = lst[-1]
        if len(lst) > 1:
            ignored[stem] = [q.name for _, q, _ in lst[:-1]]
        tp = res / f"{stem}.json"
        if not tp.exists():
            no_train.append(stem)          # no training record: does not count
            continue
        t = json.loads(tp.read_text())
        runs[stem] = {
            "eval_file": p.name, "eval": d,
            "train": {k: t.get(k) for k in
                      ("seed", "best_epoch", "val_auroc", "threshold_val_youden", "wall_time_s")},
        }
    return runs, ignored, no_train


def _m(d, test, method, thr):
    return d["test_sets"][test]["methods"][method][thr]


def _mrow(m):
    c = m["confusion"]
    return {
        "auroc": m["auroc"], "auroc_ci95": m["ci95"]["auroc"],
        "group_auroc": m["group"]["auroc"],
        "auprc": m["auprc"], "prevalence": m["prevalence"],
        "balanced_accuracy": m["balanced_accuracy"],
        "sensitivity": m["detection_rate"],
        "specificity": 1.0 - m["false_alarm_rate"],
        "confusion": {k: c[k] for k in ("tp", "fn", "tn", "fp")},
        "n_groups_leak": m["n_groups_leak"], "n_groups_no_leak": m["n_groups_no_leak"],
    }


def _rng(v):
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}


def cross_rig(runs, train_rig, test):
    sel = {}
    for stem, r in runs.items():
        d = r["eval"]
        if d["fold"] == -1 and d["train_rig"] == train_rig and test in d["test_sets"]:
            sel[int(r["train"]["seed"])] = (stem, d)
    seeds = sorted(sel)
    out = {"train_rig": train_rig, "test": test, "seeds_available": seeds,
           "seeds_missing": [s for s in EXPECTED_SEEDS if s not in seeds],
           "incomplete": any(s not in seeds for s in EXPECTED_SEEDS), "methods": {}}
    if not seeds:
        return out
    for meth in METHODS:
        per = {s: _mrow(_m(sel[s][1], test, meth, "val_youden")) for s in seeds}
        entry = {"val_youden": None, "logit0": None}
        if meth == "model_f":
            entry["val_youden"] = {"per_seed": per,
                                   "window_auroc": _rng([per[s]["auroc"] for s in seeds]),
                                   "group_auroc": _rng([per[s]["group_auroc"] for s in seeds])}
            entry["logit0"] = {"per_seed": {s: _mrow(_m(sel[s][1], test, meth, "logit0"))
                                            for s in seeds}}
        else:
            ref = per[seeds[0]]
            for s in seeds[1:]:   # deterministic given the split
                assert abs(per[s]["auroc"] - ref["auroc"]) < TOL, (meth, s, "window auroc")
                assert abs(per[s]["group_auroc"] - ref["group_auroc"]) < TOL, (meth, s, "group")
            entry["val_youden"] = {"single": ref,
                                   "window_auroc": _rng([ref["auroc"]]),
                                   "group_auroc": _rng([ref["group_auroc"]]),
                                   "identical_across_seeds_checked": seeds}
        out["methods"][meth] = entry
    return out


def base_of(block):
    m = block["methods"]
    return max(m["rms"]["val_youden"]["window_auroc"]["mean"],
               m["logreg"]["val_youden"]["window_auroc"]["mean"])


def outcome(dirA, dirB):
    """Apply prereg classes. Primary test sets only (A: sheffield, B: mendeley_acc)."""
    res = {}
    if dirA["seeds_available"]:
        af = dirA["methods"]["model_f"]["val_youden"]["window_auroc"]["mean"]
        ba = base_of(dirA)
        la, da = classify_direction(af, ba)
        res["A"] = {"AF": af, "Base": ba, "class": la, "case_D": da,
                    "incomplete": dirA["incomplete"]}
    if dirB["seeds_available"]:
        ab = dirB["methods"]["model_f"]["val_youden"]["window_auroc"]["mean"]
        bb = base_of(dirB)
        lb, db = classify_direction(ab, bb)
        res["B"] = {"AB": ab, "Base": bb, "class": lb, "case_D": db,
                    "incomplete": dirB["incomplete"]}
    if "A" in res and "B" in res:
        res["overall"] = overall_class(res["A"]["class"], res["B"]["class"])
        res["overall_provisional"] = bool(res["A"]["incomplete"] or res["B"]["incomplete"])
    else:
        res["overall"] = None
    return res


# --------------------------------------------------------------------------- within-rig
def _group_auroc(scores, y, groups):
    gy, gm = [], []
    for g in np.unique(groups):
        mk = groups == g
        gy.append(float(y[mk].mean()))
        gm.append(float(scores[mk].mean()))
    gy, gm = np.array(gy), np.array(gm)
    keep = (gy == 0) | (gy == 1)
    nl, nn = int((gy == 1).sum()), int((gy == 0).sum())
    if len(set(gy[keep])) < 2:
        return None, nl, nn
    return float(roc_auc_score(gy[keep].astype(int), gm[keep])), nl, nn


def within_rig(runs, res=RES):
    out = {}
    for rig in ("mendeley_acc", "sheffield"):
        folds, missing, sc = {}, [], {}
        for k in range(3):
            stem = f"xrig_{rig}_fold{k}_seed0"
            test = f"{rig}_fold{k}"
            if stem not in runs or test not in runs[stem]["eval"]["test_sets"]:
                missing.append(k)
                continue
            d = runs[stem]["eval"]
            row = {}
            for meth in METHODS:
                m = _m(d, test, meth, "val_youden")
                row[meth] = {"auroc": m["auroc"], "auroc_ci95": m["ci95"]["auroc"],
                             "group_auroc": m["group"]["auroc"],
                             "n_groups_leak": m["n_groups_leak"],
                             "n_groups_no_leak": m["n_groups_no_leak"]}
            folds[k] = row
            z = res / f"scores_{stem}.npz"
            if z.exists():
                sc[k] = np.load(z, allow_pickle=True)
        entry = {"folds": folds, "missing_folds": missing,
                 "fold_mean": ({m: float(np.mean([f[m]["auroc"] for f in folds.values()]))
                                for m in METHODS} if folds else None),
                 "pooled": None}
        if sc:
            ks = sorted(sc)
            pooled = {"folds_pooled": ks, "caveat": (
                "scores pooled across different fold models (and differently fitted "
                "baselines); scales need not be comparable, so treat as descriptive")}
            for meth in METHODS:
                y = np.concatenate([sc[k][f"{rig}_fold{k}__y"] for k in ks]).astype(int)
                s = np.concatenate([sc[k][f"{rig}_fold{k}__{meth}"] for k in ks])
                g = np.concatenate([sc[k][f"{rig}_fold{k}__group"] for k in ks])
                ga, nl, nn = _group_auroc(s, y, g)
                pooled[meth] = {"window_auroc": float(roc_auc_score(y, s)),
                                "group_auroc": ga, "n_groups_leak": nl, "n_groups_no_leak": nn}
            entry["pooled"] = pooled
        out[rig] = entry
    return out


# --------------------------------------------------------------------------- report
def _f(x):
    return "  n/a " if x is None else f"{x:.3f}"


def report(S):
    L = ["=== xrig summary (pure aggregation; no record_run) ===",
         f"runs used: {len(S['runs_used'])}; eval without training record (not counted): "
         f"{S['eval_without_training_record'] or 'none'}"]
    if S["ignored_duplicates"]:
        L.append(f"ignored duplicate evals: {S['ignored_duplicates']}")
    for name, blk in S["cross_rig"].items():
        L.append("")
        flag = "  ** INCOMPLETE **" if blk["incomplete"] else ""
        L.append(f"--- {name}: train {blk['train_rig']} -> test {blk['test']} | seeds available: "
                 f"{blk['seeds_available']}{flag}")
        if not blk["seeds_available"]:
            continue
        for meth, e in blk["methods"].items():
            v = e["val_youden"]
            if meth == "model_f":
                for s, r in v["per_seed"].items():
                    ci, c = r["auroc_ci95"], r["confusion"]
                    L.append(f"  model_f seed{s}: AUROC {_f(r['auroc'])} [{_f(ci[0])},{_f(ci[1])}] "
                             f"grp {_f(r['group_auroc'])} AUPRC {_f(r['auprc'])} (prev {_f(r['prevalence'])}) "
                             f"| youden: BA {_f(r['balanced_accuracy'])} sens {_f(r['sensitivity'])} "
                             f"spec {_f(r['specificity'])} tp/fn/tn/fp {c['tp']}/{c['fn']}/{c['tn']}/{c['fp']}")
                for s, r in e["logit0"]["per_seed"].items():
                    c = r["confusion"]
                    L.append(f"    logit0 seed{s}: BA {_f(r['balanced_accuracy'])} sens {_f(r['sensitivity'])} "
                             f"spec {_f(r['specificity'])} tp/fn/tn/fp {c['tp']}/{c['fn']}/{c['tn']}/{c['fp']}")
            else:
                r = v["single"]
                ci, c = r["auroc_ci95"], r["confusion"]
                L.append(f"  {meth} (seed-invariant): AUROC {_f(r['auroc'])} [{_f(ci[0])},{_f(ci[1])}] "
                         f"grp {_f(r['group_auroc'])} AUPRC {_f(r['auprc'])} | youden: BA "
                         f"{_f(r['balanced_accuracy'])} sens {_f(r['sensitivity'])} spec {_f(r['specificity'])} "
                         f"tp/fn/tn/fp {c['tp']}/{c['fn']}/{c['tn']}/{c['fp']}")
            w, g = v["window_auroc"], v["group_auroc"]
            L.append(f"    -> {meth}: window AUROC mean {_f(w['mean'])} [{_f(w['min'])}-{_f(w['max'])}]; "
                     f"group AUROC mean {_f(g['mean'])} [{_f(g['min'])}-{_f(g['max'])}]")
    L.append("")
    L.append("--- outcome classes (prereg; primary test sets)")
    o = S["outcome"]
    if "A" in o:
        a = o["A"]
        L.append(f"  A (test sheffield): AF={_f(a['AF'])} Base={_f(a['Base'])} -> {a['class']}; "
                 f"Case D (Base >= AF-0.05): {a['case_D']}{'; INCOMPLETE' if a['incomplete'] else ''}")
    else:
        L.append("  A: no runs")
    if "B" in o:
        b = o["B"]
        L.append(f"  B (test mendeley_acc): AB={_f(b['AB'])} Base={_f(b['Base'])} -> {b['class']}; "
                 f"Case D (Base >= AB-0.05): {b['case_D']}{'; INCOMPLETE' if b['incomplete'] else ''}")
    else:
        L.append("  B: no runs")
    L.append(f"  overall: {o['overall']}"
             f"{' (PROVISIONAL: seeds missing)' if o.get('overall_provisional') else ''}")
    L.append("")
    L.append("--- within-rig (grouped 3-fold CV, seed 0)")
    for rig, e in S["within_rig"].items():
        L.append(f"  {rig}: folds present {sorted(e['folds'])}, missing {e['missing_folds']}")
        for k, row in sorted(e["folds"].items()):
            parts = []
            for m in METHODS:
                r = row[m]
                ci = r["auroc_ci95"]
                parts.append(f"{m} {_f(r['auroc'])} [{_f(ci[0])},{_f(ci[1])}] grp {_f(r['group_auroc'])}")
            r0 = row["model_f"]
            L.append(f"    fold{k} (groups leak/no-leak {r0['n_groups_leak']}/{r0['n_groups_no_leak']}): "
                     + " | ".join(parts))
        if e["fold_mean"]:
            L.append("    fold mean window AUROC: " + ", ".join(f"{m} {_f(v)}" for m, v in e["fold_mean"].items()))
        if e["pooled"]:
            p = e["pooled"]
            L.append(f"    pooled out-of-fold (folds {p['folds_pooled']}) CAVEAT: {p['caveat']}")
            for m in METHODS:
                L.append(f"      {m}: window {_f(p[m]['window_auroc'])} group {_f(p[m]['group_auroc'])} "
                         f"(groups leak/no-leak {p[m]['n_groups_leak']}/{p[m]['n_groups_no_leak']})")
    return "\n".join(L)


def build(res=RES):
    runs, ignored, no_train = load_runs(res)
    A = cross_rig(runs, "mendeley_acc", "sheffield")
    B1 = cross_rig(runs, "sheffield", "mendeley_acc")
    B2 = cross_rig(runs, "sheffield", "mendeley_hyd")
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prereg": "docs/VERSION2_CROSS_RIG_PREREG.md",
        "runs_used": {s: {"eval_file": r["eval_file"], "train": r["train"]}
                      for s, r in sorted(runs.items())},
        "ignored_duplicates": ignored,
        "eval_without_training_record": no_train,
        "cross_rig": {"A_mendeley_acc_to_sheffield": A,
                      "B_sheffield_to_mendeley_acc_primary": B1,
                      "B_sheffield_to_mendeley_hyd_secondary": B2},
        "outcome": outcome(A, B1),
        "within_rig": within_rig(runs, res),
    }


def main():
    S = build()
    base = os.environ.get("LEAKNET_OUT")
    outdir = (Path(base) / "v2_cross_rig") if base else RES
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "summary.json"
    if path.exists():
        path = outdir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(S, indent=2, default=str))
    print(report(S))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
