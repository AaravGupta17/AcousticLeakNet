"""
xrig_three_band_groups.py — Phase 7 addendum 1: group-level and failure analysis
================================================================================
Protocol: docs/PHASE7_ADDENDUM_1_GROUP_ANALYSIS.md (locked). Descriptive analysis of the frozen
Phase 7 results; nothing is chosen from it. Analyses 1-3 use saved scores, 4 uses the existing
rig caches (deterministic refit asserted to reproduce the frozen validation AUROC), 5 is a
secondary within-rig reference.

    python experiments/xrig_three_band_groups.py
"""

import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cross_dataset as CD                                  # noqa: E402
import rigs                                                 # noqa: E402
import xrig_eval as XE                                      # noqa: E402
import xrig_three_band as TB                                # noqa: E402
from _common import REPO_ROOT, _jsonable, record_run         # noqa: E402

V2_DIR = TB.V2_DIR
P7 = TB.OUT_DIR
OUT = P7 / "group_analysis"
COLS3 = list(TB.THREE_BAND_COLS)
FEAT_NAMES = {4: "band_400_600", 5: "band_600_800", 8: "band_1600_2000"}
N_BOOT, SEED = 2000, 0
DIRECTIONS = TB.DIRECTIONS
check = TB.check
ASSERTIONS = TB.ASSERTIONS


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def load_scores(tag, train_rig, test_rig, frozen):
    z = np.load(P7 / f"scores_{tag}.npz", allow_pickle=False)
    y, g, st, me = z["y"].astype(int), z["group"], z["stream"], z["meta"]
    S = {"three_band": z["three_band"].astype(np.float64), "full14": z["full14"].astype(np.float64)}
    thr = {"three_band": frozen["thresholds"]["three_band"], "full14": frozen["thresholds"]["full14"]}
    for sd in (0, 1, 2):
        f = np.load(V2_DIR / f"scores_xrig_{train_rig}_foldx_seed{sd}.npz", allow_pickle=False)
        p = f"{test_rig}__"
        fy, fg, fs, fm = f[p + "y"].astype(int), f[p + "group"], f[p + "stream"], f[p + "meta"]
        sc = f[p + "model_f"].astype(np.float64)
        if (np.array_equal(fy, y) and np.array_equal(fg, g) and np.array_equal(fs, st) and np.array_equal(fm, me)):
            check(True, f"{tag}: model_f seed{sd} window order identical to phase7 scores")
        else:
            o1 = np.lexsort((me, st, g))
            o2 = np.lexsort((fm, fs, fg))
            check(np.array_equal(fg[o2], g[o1]) and np.array_equal(fs[o2], st[o1])
                  and np.array_equal(fm[o2], me[o1]) and np.array_equal(fy[o2], y[o1]),
                  f"{tag}: model_f seed{sd} aligns after sorting on (group, stream, meta)")
            inv = np.empty_like(o1)
            inv[o1] = np.arange(len(o1))
            sc = sc[o2][inv]
        S[f"model_f_seed{sd}"] = sc
        e = json.loads((V2_DIR / f"eval_xrig_{train_rig}_foldx_seed{sd}.json").read_text())
        thr_c = frozen["comparators"]["model_f"][f"seed{sd}"]["threshold"]
        thr_e = e["test_sets"][test_rig]["methods"]["model_f"]["val_youden"]["threshold"]
        check(abs(thr_c - thr_e) < 1e-12, f"{tag}: model_f seed{sd} comparator threshold == eval json threshold")
        if "thresholds" in e and "model_f" in e["thresholds"]:
            check(abs(e["thresholds"]["model_f"] - thr_c) < 1e-9,
                  f"{tag}: model_f seed{sd} eval_xrig thresholds.model_f == comparator threshold")
        thr[f"model_f_seed{sd}"] = thr_c
    return y, g, S, thr


def group_stats(y, g, s, thr):
    out = {}
    for gg in np.unique(g):
        m = g == gg
        lab = int(y[m][0])
        opp = y == (1 - lab)
        ys = np.concatenate([np.full(m.sum(), lab), np.full(opp.sum(), 1 - lab)])
        ss = np.concatenate([s[m], s[opp]])
        out[str(gg)] = dict(label=lab, n_windows=int(m.sum()), mean_score=float(s[m].mean()),
                            median_score=float(np.median(s[m])), frac_flagged=float((s[m] >= thr).mean()),
                            one_group_auroc=auroc(ys, ss))
    return out


def analyse_method(y, g, s, thr, frozen_auc, frozen_gauc, frozen_conf=None):
    ug = np.unique(g)
    gy = np.array([int(y[g == u][0]) for u in ug])
    n_g = {u: (g == u).sum() for u in ug}
    pooled = auroc(y, s)
    check(abs(pooled - frozen_auc) < 1e-6, f"pooled window AUROC reproduces frozen ({pooled:.6f} vs {frozen_auc:.6f})")
    w = np.array([1.0 / n_g[x] for x in g])
    gb = float(roc_auc_score(y, s, sample_weight=w))
    loo = []
    for u in ug:
        m = g != u
        loo.append({"dropped_group": str(u), "label": int(y[g == u][0]), "auroc": auroc(y[m], s[m])})
    la = [d["auroc"] for d in loo if d["auroc"] is not None]
    gs = group_stats(y, g, s, thr)
    one = {c: [v["one_group_auroc"] for v in gs.values() if v["label"] == c and v["one_group_auroc"] is not None]
           for c in (0, 1)}
    spread = {("leak" if c else "no_leak"): {"min": float(min(v)), "median": float(np.median(v)),
                                             "max": float(max(v)), "n": len(v)} for c, v in one.items()}
    gm = np.array([s[g == u].mean() for u in ug])
    gauc = auroc(gy, gm)
    check(gauc is not None and abs(gauc - frozen_gauc) < 1e-9, f"group-level AUROC reproduces frozen ({gauc} vs {frozen_gauc})")
    gp = gm >= thr
    conf = {"tp": int((gp & (gy == 1)).sum()), "fn": int((~gp & (gy == 1)).sum()),
            "tn": int((~gp & (gy == 0)).sum()), "fp": int((gp & (gy == 0)).sum())}
    if frozen_conf is not None:
        check(conf == frozen_conf, f"group confusion reproduces frozen {conf}")
    valid = [d for d in loo if d["auroc"] is not None]
    return {"pooled_window_auroc": pooled, "group_balanced_window_auroc": gb,
            "balanced_minus_pooled": gb - pooled, "loo_group_auroc": loo,
            "loo_min": float(min(la)), "loo_max": float(max(la)),
            "loo_min_dropped": min(valid, key=lambda d: d["auroc"])["dropped_group"],
            "loo_max_dropped": max(valid, key=lambda d: d["auroc"])["dropped_group"],
            "one_group_auroc_spread": spread, "group_level_auroc": gauc, "group_confusion": conf,
            "threshold": float(thr), "n_groups": {"leak": int((gy == 1).sum()), "no_leak": int((gy == 0).sum())}}, gs


def feats_rig(name):
    rw = rigs.load_rig(name)
    f = TB.feats14(rw.x)
    rw.x = None
    return rw, f


def gmeans(g, v):
    ug = np.unique(g)
    return ug, np.array([v[g == u].mean() for u in ug])


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((P7 / "three_band_results.json").read_text())
    for c, band in zip(TB.THREE_BAND_COLS, TB.EXPECTED_BANDS):
        check((CD.BAND_EDGES[c], CD.BAND_EDGES[c + 1]) == band, f"col {c} is band {band} Hz")

    # ---------- analyses 1-3 ----------
    rows, ga = [], {"directions": {}}
    plot_data = {}
    for tag, train_rig, test_rig in DIRECTIONS:
        fz = res["directions"][tag]
        y, g, S, thr = load_scores(tag, train_rig, test_rig, fz)
        ga["directions"][tag] = {"train_rig": train_rig, "test_rig": test_rig, "methods": {}}
        plot_data[tag] = (y, g, S, thr)
        print(f"\n######## Direction {tag}: {train_rig} -> {test_rig}")
        print(f"{'method':14s} {'pooled':>7s} {'grpbal':>7s} {'LOGO[min,max]':>16s} {'grpAUC':>7s} "
              f"{'1grp leak min/med/max':>22s} {'1grp noleak min/med/max':>24s}  grp-confusion")
        for nm, s in S.items():
            if nm in ("three_band", "full14"):
                m = fz["methods"][nm]
                fa_, fga, fc = m["auroc"], m["group"]["auroc"], m["group"]["confusion"]
            else:
                c = fz["comparators"]["model_f"][nm.split("_", 2)[2]]
                fa_, fga, fc = c["auroc"], c["group_auroc"], None
            r, gs = analyse_method(y, g, s, thr[nm], fa_, fga, fc)
            ga["directions"][tag]["methods"][nm] = r
            for gg, v in gs.items():
                rows.append(dict(direction=tag, method=nm, group=gg, label=v["label"], n_windows=v["n_windows"],
                                 mean_score=v["mean_score"], median_score=v["median_score"],
                                 frac_flagged=v["frac_flagged"], one_group_auroc=v["one_group_auroc"]))
            sp = r["one_group_auroc_spread"]
            f3 = lambda d: f"{d['min']:.2f}/{d['median']:.2f}/{d['max']:.2f}" if d else "n/a"   # noqa: E731
            print(f"{nm:14s} {r['pooled_window_auroc']:7.3f} {r['group_balanced_window_auroc']:7.3f} "
                  f"[{r['loo_min']:.3f},{r['loo_max']:.3f}] {r['group_level_auroc']:7.3f} "
                  f"{f3(sp.get('leak')):>22s} {f3(sp.get('no_leak')):>24s}  {r['group_confusion']}")
            print(f"   LOGO min when dropping {r['loo_min_dropped']}; max when dropping {r['loo_max_dropped']}")
        ga["directions"][tag]["n_groups"] = ga["directions"][tag]["methods"]["three_band"]["n_groups"]
    with open(OUT / "per_group.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "method", "group", "label", "n_windows", "mean_score",
                                          "median_score", "frac_flagged", "one_group_auroc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT / 'per_group.csv'} ({len(rows)} rows)")

    # ---------- analysis 4 ----------
    print("\n######## Failure analysis (features)")
    rigdat, fa = {}, {"per_rig_feature": {}, "refit": {}, "class_means": {}, "shift": {}}
    for rname in ("mendeley_acc", "sheffield"):
        rw, F = feats_rig(rname)
        X = {FEAT_NAMES[c]: F[:, c] for c in COLS3}
        X["log_rms"] = rw.log_rms.astype(np.float64)
        rigdat[rname] = (rw, F, X)
    for rname, (rw, F, X) in rigdat.items():
        fa["per_rig_feature"][rname] = {}
        for fn, v in X.items():
            ug, gm = gmeans(rw.group, v)
            gy = np.array([int(rw.y[rw.group == u][0]) for u in ug])
            ga_ = auroc(gy, gm)
            wa = auroc(rw.y, v)
            fa["per_rig_feature"][rname][fn] = {
                "group_auroc": ga_, "window_auroc": wa,
                "direction_group": "leak higher" if ga_ > 0.5 else "leak lower",
                "direction_window": "leak higher" if wa > 0.5 else "leak lower",
                "mean_leak_groups": float(gm[gy == 1].mean()), "mean_no_leak_groups": float(gm[gy == 0].mean())}
    print(f"{'rig':13s} {'feature':15s} {'grpAUC':>7s} {'winAUC':>7s}  direction(group)  leak-mean  noleak-mean")
    for rname, d in fa["per_rig_feature"].items():
        for fn, v in d.items():
            print(f"{rname:13s} {fn:15s} {v['group_auroc']:7.3f} {v['window_auroc']:7.3f}  {v['direction_group']:16s} "
                  f"{v['mean_leak_groups']:+9.4f} {v['mean_no_leak_groups']:+9.4f}")
    # (b) refit
    for tag, train_rig, test_rig in DIRECTIONS:
        rec = json.loads((V2_DIR / f"xrig_{train_rig}_foldx_seed0.json").read_text())
        tg, vg = set(rec["train_groups"]), set(rec["val_groups"])
        rw, F, _ = rigdat[train_rig]
        mt, mv = np.isin(rw.group, sorted(tg)), np.isin(rw.group, sorted(vg))
        clf = TB.make_clf()
        clf.fit(F[mt][:, COLS3], rw.y[mt])
        va = auroc(rw.y[mv], clf.decision_function(F[mv][:, COLS3]))
        frz = res["directions"][tag]["val_auroc"]["three_band"]
        check(abs(va - frz) < 1e-9, f"{tag}: refit three_band val AUROC {va:.12f} == frozen {frz:.12f}")
        sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
        coef_s, icpt = lr.coef_[0], float(lr.intercept_[0])
        coef_raw = coef_s / sc.scale_
        icpt_raw = icpt - float(np.sum(coef_s * sc.mean_ / sc.scale_))
        fa["refit"][tag] = {"train_rig": train_rig, "features": [FEAT_NAMES[c] for c in COLS3],
                            "val_auroc_refit": va, "val_auroc_frozen": frz,
                            "coef_standardised": coef_s, "coef_raw": coef_raw, "intercept_scaled_space": icpt,
                            "intercept_raw_space": icpt_raw, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_}
        print(f"\nRefit {tag} (train {train_rig}): val AUROC {va:.6f}")
        print(f"  {'feature':15s} {'coef_std':>9s} {'coef_raw':>9s} {'mean':>8s} {'scale':>8s}")
        for i, c in enumerate(COLS3):
            print(f"  {FEAT_NAMES[c]:15s} {coef_s[i]:+9.4f} {coef_raw[i]:+9.4f} {sc.mean_[i]:+8.4f} {sc.scale_[i]:8.4f}")
        print(f"  intercept (scaled space) {icpt:+.4f}; raw space {icpt_raw:+.4f}")
    # (c) class-conditional group means and shift
    cell_sd = {}
    for rname, (rw, F, X) in rigdat.items():
        fa["class_means"][rname] = {}
        for fn, v in X.items():
            ug, gm = gmeans(rw.group, v)
            gy = np.array([int(rw.y[rw.group == u][0]) for u in ug])
            fa["class_means"][rname][fn] = {}
            for c, cn in ((1, "leak"), (0, "no_leak")):
                a = gm[gy == c]
                cell_sd[(rname, fn, c)] = (a.var(ddof=1), len(a))
                fa["class_means"][rname][fn][cn] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n_groups": len(a)}
    print("\nClass-conditional group means; shift = (sheffield - mendeley_acc)/pooled SD of group means "
          "(pooled over the 4 rig x class cells, df-weighted)")
    print(f"{'feature':15s} {'class':8s} {'mend_mean':>9s} {'mend_sd':>8s} {'sheff_mean':>10s} {'sheff_sd':>8s} {'pooledSD':>8s} {'shift':>7s}")
    for fn in X:
        num = sum(cell_sd[(r, fn, c)][0] * (cell_sd[(r, fn, c)][1] - 1) for r in rigdat for c in (0, 1))
        den = sum(cell_sd[(r, fn, c)][1] - 1 for r in rigdat for c in (0, 1))
        psd = float(np.sqrt(num / den))
        fa["shift"][fn] = {"pooled_within_cell_sd": psd}
        for c, cn in ((1, "leak"), (0, "no_leak")):
            m_, s_ = fa["class_means"]["mendeley_acc"][fn][cn], fa["class_means"]["sheffield"][fn][cn]
            sh = (s_["mean"] - m_["mean"]) / psd
            fa["shift"][fn][cn + "_shift_sd_units"] = sh
            print(f"{fn:15s} {cn:8s} {m_['mean']:+9.4f} {m_['sd']:8.4f} {s_['mean']:+10.4f} {s_['sd']:8.4f} {psd:8.4f} {sh:+7.2f}")
        gaps = {r: fa["class_means"][r][fn]["leak"]["mean"] - fa["class_means"][r][fn]["no_leak"]["mean"] for r in rigdat}
        fa["shift"][fn]["leak_minus_noleak_gap"] = {r: float(v) for r, v in gaps.items()}
        fa["shift"][fn]["gap_in_sd_units"] = {r: float(v / psd) for r, v in gaps.items()}
        print(f"{'':15s} leak-noleak gap: mend {gaps['mendeley_acc']:+.4f} ({gaps['mendeley_acc'] / psd:+.2f} SD), "
              f"sheff {gaps['sheffield']:+.4f} ({gaps['sheffield'] / psd:+.2f} SD)")

    # ---------- analysis 5 ----------
    print("\n######## Within-rig reference (3-band LR, existing grouped 3-fold)")
    wr, wr_scores = [], {}
    print(f"{'rig':13s} fold  nTest(leak/no)  AUROC   BalAcc  Sens   Spec   thr | frozen full-LR AUROC/BalAcc | frozen model_f AUROC/BalAcc")
    for rname in ("mendeley_acc", "sheffield"):
        rw, F, _ = rigdat[rname]
        allg = set(map(str, rw.group))
        cls = rigs._group_class(rw.group, rw.y)
        fold_of = rigs.fold_assign(rw.group, rw.y)
        for f in (0, 1, 2):
            rec = json.loads((V2_DIR / f"xrig_{rname}_fold{f}_seed0.json").read_text())
            tg, vg = set(rec["train_groups"]), set(rec["val_groups"])
            te_g = {k for k, v in fold_of.items() if v == f}
            check(not (te_g & tg) and not (te_g & vg) and not (tg & vg), f"{rname} fold{f}: train/val/test groups disjoint")
            check(tg | vg | te_g == allg, f"{rname} fold{f}: train U val U test == all groups")
            gstr = rw.group.astype(str)
            mt, mv, me = np.isin(gstr, sorted(tg)), np.isin(gstr, sorted(vg)), np.isin(gstr, sorted(te_g))
            clf = TB.make_clf()
            clf.fit(F[mt][:, COLS3], rw.y[mt])
            sv = clf.decision_function(F[mv][:, COLS3])
            thr = rigs.youden_threshold(rw.y[mv], sv)
            ste = clf.decision_function(F[me][:, COLS3]).astype(np.float64)
            yt, gt = rw.y[me].astype(int), rw.group[me]
            nl, nn = int((yt == 1).sum()), int((yt == 0).sum())
            ev = json.loads((V2_DIR / f"eval_xrig_{rname}_fold{f}_seed0.json").read_text())["test_sets"][f"{rname}_fold{f}"]["methods"]
            fl, fm = ev["logreg"]["val_youden"], ev["model_f"]["val_youden"]
            entry = {"rig": rname, "fold": f, "n_train_groups": len(tg), "n_val_groups": len(vg),
                     "n_test_groups_leak": len({x for x, yy in zip(gt, yt) if yy == 1}),
                     "n_test_groups_no_leak": len({x for x, yy in zip(gt, yt) if yy == 0}),
                     "n_val_groups_no_leak": int(sum(cls[k] == 0 for k in vg)),
                     "n_test_windows_leak": nl, "n_test_windows_no_leak": nn, "threshold": thr,
                     "frozen_logreg": {"auroc": fl["auroc"], "balanced_accuracy": fl["balanced_accuracy"]},
                     "frozen_model_f": {"auroc": fm["auroc"], "balanced_accuracy": fm["balanced_accuracy"]}}
            try:
                m = XE.full_metrics(yt, ste, gt, thr, N_BOOT, SEED)
                entry["three_band"] = {"auroc": m["auroc"], "ci95": m["ci95"]["auroc"],
                                       "balanced_accuracy": m["balanced_accuracy"], "sensitivity": m["detection_rate"],
                                       "specificity": 1 - m["false_alarm_rate"], "group_auroc": m["group"]["auroc"],
                                       "group_confusion": m["group"]["confusion"]}
            except Exception as e:                                          # noqa: BLE001
                entry["three_band"] = {"error": repr(e), "auroc": auroc(yt, ste)}
                print(f"  full_metrics failed for {rname} fold{f}: {e!r}")
            wr.append(entry)
            wr_scores[f"{rname}_fold{f}_score"] = ste.astype(np.float32)
            wr_scores[f"{rname}_fold{f}_y"] = yt
            wr_scores[f"{rname}_fold{f}_group"] = gt
            tb = entry["three_band"]
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"             # noqa: E731
            print(f"{rname:13s} {f:4d}  {nl:6d}/{nn:<6d}  {fmt(tb.get('auroc'))}  {fmt(tb.get('balanced_accuracy'))}  "
                  f"{fmt(tb.get('sensitivity'))}  {fmt(tb.get('specificity'))}  {thr:+.3f} | "
                  f"{fl['auroc']:.3f}/{fl['balanced_accuracy']:.3f} | {fm['auroc']:.3f}/{fm['balanced_accuracy']:.3f}"
                  f"   [test groups leak/no {entry['n_test_groups_leak']}/{entry['n_test_groups_no_leak']}, "
                  f"val no-leak groups {entry['n_val_groups_no_leak']}]")
    np.savez_compressed(OUT / "within_rig_scores.npz", **wr_scores)
    for rname in rigdat:
        rigdat[rname] = None
    gc.collect()

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meths = [("three_band", "3-band LR"), ("full14", "full-14 LR"), ("model_f_seed0", "Model F seed0")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    rng = np.random.default_rng(0)
    for i, (tag, train_rig, test_rig) in enumerate(DIRECTIONS):
        y, g, S, thr = plot_data[tag]
        for j, (nm, title) in enumerate(meths):
            ax = axes[i, j]
            for u in np.unique(g):
                lab = int(y[g == u][0])
                ax.scatter(lab + rng.uniform(-0.15, 0.15), S[nm][g == u].mean(), s=28,
                           c="tab:red" if lab else "tab:blue", alpha=0.8, edgecolors="none")
            ax.axhline(thr[nm], color="k", ls="--", lw=1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["no leak", "leak"])
            ax.set_title(f"{tag}: {train_rig}->{test_rig}\n{title}", fontsize=9)
            ax.set_ylabel("group mean decision value")
    fig.suptitle("Per-group mean decision value; dashed = frozen val threshold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_group.png", dpi=130)
    plt.close(fig)

    # ---------- provenance + save ----------
    inputs = [P7 / "three_band_results.json", P7 / "scores_A.npz", P7 / "scores_B.npz"]
    for sd in (0, 1, 2):
        for r in ("mendeley_acc", "sheffield"):
            inputs += [V2_DIR / f"scores_xrig_{r}_foldx_seed{sd}.npz", V2_DIR / f"eval_xrig_{r}_foldx_seed{sd}.json"]
    for r in ("mendeley_acc", "sheffield"):
        inputs.append(V2_DIR / f"xrig_{r}_foldx_seed0.json")
        for f in (0, 1, 2):
            inputs += [V2_DIR / f"xrig_{r}_fold{f}_seed0.json", V2_DIR / f"eval_xrig_{r}_fold{f}_seed0.json"]
    prov = {"git_commit": TB.git("rev-parse", "HEAD"), "git_status_porcelain": TB.git("status", "--porcelain"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "sha256_cache_xrig_mendeley_acc": TB.sha256(rigs.XRIG_CACHE / "mendeley_acc.npz"),
            "sha256_cache_xrig_sheffield": TB.sha256(rigs.XRIG_CACHE / "sheffield.npz"),
            "input_sha256": {str(p.relative_to(REPO_ROOT)).replace("\\", "/"): TB.sha256(p) for p in inputs},
            "bootstrap_seed": SEED, "n_boot": N_BOOT, "wall_s": time.time() - t_start}
    out = {"analysis_1_to_3": ga, "failure_analysis": fa, "within_rig_reference": wr, "provenance": prov,
           "assertions_passed": ASSERTIONS}
    (OUT / "group_analysis.json").write_text(json.dumps(_jsonable(out), indent=1))
    config = {"addendum": "docs/PHASE7_ADDENDUM_1_GROUP_ANALYSIS.md", "n_boot": N_BOOT, "seed": SEED,
              "three_band_cols": COLS3, "classifier": TB.CLF_PARAMS}
    summ = "group-level/failure analysis of frozen Phase 7 results (descriptive)"
    record_run("v2_xrig_three_band_groups", config, _jsonable(out), summ)
    print(f"\nSaved {OUT / 'group_analysis.json'}, per_group.csv, fig_per_group.png, within_rig_scores.npz")


if __name__ == "__main__":
    main()
