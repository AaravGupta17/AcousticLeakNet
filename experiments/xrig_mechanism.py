"""
xrig_mechanism.py -- exploratory mechanism analysis for the Version 2 cross-rig collapse
==========================================================================================
Follow-up to the audited, pre-registered cross-rig result (docs/VERSION2_CROSS_RIG_BASELINE.md):
Model F collapses Mendeley->Sheffield (0.345, inverted) and is chance-level Sheffield->Mendeley
(0.525). This script does NOT train or select anything. It is pure descriptive analysis of the
SAME recordings and physical groups already used in that benchmark, to find out which acoustic
cues are leak-discriminative WITHIN a rig but rig-INVARIANT across rigs, as a candidate for a
follow-up representation. No new dataset, no test-set tuning: every number here is descriptive
(oof_auroc uses grouped folds entirely within one rig, or pools both rigs' LABELS freely since
no model selection follows from it).

Four parts:
  1. Per-feature leak-AUROC (within each rig) vs rig-AUROC (pooled), for every scalar feature in
     cross_dataset.features_1ch plus log_rms -- the per-feature version of the existing
     xrig_shift.py separability analysis (which only reports the FULL feature vector).
  2. Per-condition (not just per-rig) spectral contrast, to check the inversion is not driven by
     one outlier group.
  3. Two-sensor relational features.
       - Mendeley ch1/ch2: checked for simultaneity (cross-correlation sharpness) first, then
         (if simultaneous) paired per-window: log-amplitude difference, spectral log-ratio
         D(f) = log10 P1(f) - log10 P2(f) in the same coarse bands as features_1ch, zero-lag and
         best-lag correlation. Tested for leak separability within Mendeley only.
       - Sheffield Acc0/Acc1 columns are NOT simultaneous (separate 30 s sessions; pre-reg).
         A true window-paired relational feature cannot be built. Reported as a hard limitation,
         not worked around. A purely descriptive near-vs-far spectral difference per leak test
         condition is computed instead, clearly labelled as non-simultaneous and not comparable
         to the Mendeley feature.
  4. Ranking table + the primary/backup candidate (printed, not fabricated).

    python experiments/xrig_mechanism.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Model_F"))
import cross_dataset as CD                                   # noqa: E402
import mendeley as M                                          # noqa: E402
import public_data as P                                        # noqa: E402
import rigs                                                     # noqa: E402
from xrig_shift import (contrast_correlation, group_then_class_mean,   # noqa: E402
                        normalised_psd, oof_auroc, spectral_centroid)
from _common import REPO_ROOT, _jsonable                        # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "v2_cross_rig"
FEATURE_NAMES = [f"band_{lo}_{hi}" for lo, hi in zip(CD.BAND_EDGES[:-1], CD.BAND_EDGES[1:])] + \
                ["centroid_khz", "flatness", "kurtosis", "crest", "zero_cross_rate"]


# ── part 1: per-feature leak vs rig AUROC ───────────────────────────────────────

def _group_auroc(y, s, groups):
    groups = np.asarray(groups).astype(str)
    ug = np.unique(groups)
    gm = np.array([s[groups == g].mean() for g in ug])
    gy = np.array([int(y[groups == g][0]) for g in ug])
    return float(roc_auc_score(gy, gm)) if len(np.unique(gy)) == 2 else None


def per_feature_table(rws: dict) -> dict:
    """rws: {rig_name: RigWindows}. Returns per-feature leak-AUROC (per rig, window & group
    level) and rig-AUROC (pooled, window level), for log_rms plus every features_1ch column."""
    feats, logrms = {}, {}
    for r, rw in rws.items():
        feats[r] = CD.features_1ch(CD.standardise(rw.x))
        logrms[r] = rw.log_rms.astype(np.float64)
    names = ["log_rms"] + FEATURE_NAMES
    cols = {r: np.column_stack([logrms[r]] + [feats[r][:, i] for i in range(feats[r].shape[1])])
            for r in rws}
    rig_names = list(rws)
    pooled_rig_y = np.concatenate([np.full(len(rws[r]), int(r == "sheffield")) for r in rig_names])
    out = {}
    for j, name in enumerate(names):
        row = {"leak_auroc": {}, "leak_group_auroc": {}}
        for r, rw in rws.items():
            s = cols[r][:, j]
            row["leak_auroc"][r] = float(roc_auc_score(rw.y, s)) if len(np.unique(rw.y)) == 2 else None
            row["leak_group_auroc"][r] = _group_auroc(rw.y, s, rw.group)
        pooled_s = np.concatenate([cols[r][:, j] for r in rig_names])
        row["rig_auroc"] = float(roc_auc_score(pooled_rig_y, pooled_s))
        # orientation-free: a feature that separates rigs with AUROC 0.1 is just as rig-telling as 0.9
        row["rig_auroc_abs"] = max(row["rig_auroc"], 1 - row["rig_auroc"])
        out[name] = row
    return out


# ── part 2: per-condition spectral contrast ─────────────────────────────────────

def per_condition_contrast(rw, rig: str) -> dict:
    """Leak - no-leak log10-normalised-PSD contrast, one curve per NO-LEAK group against the
    pooled leak groups (Sheffield has only 3 no-leak groups; Mendeley topology/condition acts as
    the natural sub-grouping instead, since there are 8 no-leak RECORDINGS already)."""
    freqs, psd = normalised_psd(rw.x)
    log_psd = np.log10(psd + 1e-30)
    leak_mean = np.mean([log_psd[rw.group == g].mean(axis=0)
                         for g in sorted(set(rw.group[rw.y == 1]))], axis=0)
    out = {"freqs_hz": freqs, "leak_mean": leak_mean, "per_no_leak_group": {}}
    for g in sorted(set(rw.group[rw.y == 0])):
        contrast = leak_mean - log_psd[rw.group == g].mean(axis=0)
        out["per_no_leak_group"][str(g)] = {
            "contrast": contrast, "freq_of_max_hz": float(freqs[int(np.argmax(contrast))]),
            "freq_of_min_hz": float(freqs[int(np.argmin(contrast))])}
    curves = [v["contrast"] for v in out["per_no_leak_group"].values()]
    if len(curves) >= 2:
        cors = [float(np.corrcoef(curves[i], curves[j])[0, 1])
                for i in range(len(curves)) for j in range(i + 1, len(curves))]
        out["pairwise_pearson_between_no_leak_groups"] = cors
    return out


# ── part 3: two-sensor relational features ──────────────────────────────────────

def mendeley_simultaneity_check(n_check: int = 6) -> dict:
    """Cross-correlation of raw ch1/ch2 for a handful of no-leak Mendeley recordings: a sharp
    peak near lag 0 supports simultaneous dual-sensor recording; a flat/noisy profile would not."""
    recs = [r for r in M.discover(rigs.MENDELEY_ROOTS["accelerometer"], "accelerometer")
            if r.condition == M.NO_LEAK][:n_check]
    out = []
    for rec in recs:
        x1, x2 = M.load_recording(rec, fs_out=P.FS, cache_dir=rigs.MENDELEY_CACHE)
        n = min(len(x1), len(x2))
        x1, x2 = (x1[:n] - x1[:n].mean()) / (x1[:n].std() + 1e-9), (x2[:n] - x2[:n].mean()) / (x2[:n].std() + 1e-9)
        max_lag = int(0.5 * P.FS)
        xc = np.correlate(x1, x2[max_lag:n - max_lag], mode="valid") / n
        lag = int(np.argmax(np.abs(xc))) - max_lag
        out.append({"rec": rec.rec_id, "n_samples": n, "peak_corr": float(xc[np.argmax(np.abs(xc))]),
                    "peak_lag_samples": lag, "zero_lag_corr": float(xc[max_lag])})
    return {"per_recording": out,
           "mean_abs_peak_corr": float(np.mean([abs(o["peak_corr"]) for o in out])),
           "mean_abs_zero_lag_corr": float(np.mean([abs(o["zero_lag_corr"]) for o in out]))}


def mendeley_two_sensor_features() -> dict:
    """True simultaneous per-window pairing of ch1/ch2 for every Mendeley accelerometer
    recording. Returns per-window: log-rms difference, zero-lag correlation, spectral
    log-ratio in features_1ch's coarse bands; and per-recording group id / label."""
    recs = M.discover(rigs.MENDELEY_ROOTS["accelerometer"], "accelerometer")
    rows_dlogrms, rows_xcorr, rows_dspec, groups, ys = [], [], [], [], []
    for rec in recs:
        x1, x2 = M.load_recording(rec, fs_out=P.FS, cache_dir=rigs.MENDELEY_CACHE)
        w1, w2 = P.windows_of(P.to_common(x1, P.FS)), P.windows_of(P.to_common(x2, P.FS))
        n = min(len(w1), len(w2))
        if n == 0:
            continue
        w1, w2 = w1[:n].astype(np.float64), w2[:n].astype(np.float64)
        r1, r2 = np.sqrt((w1 ** 2).mean(1)), np.sqrt((w2 ** 2).mean(1))
        rows_dlogrms.append(np.log10(r1 + 1e-12) - np.log10(r2 + 1e-12))
        u1, u2 = w1 / (r1[:, None] + 1e-12), w2 / (r2[:, None] + 1e-12)
        rows_xcorr.append((u1 * u2).mean(1))
        f, p1 = welch(w1, fs=P.FS, nperseg=256, axis=-1)
        _, p2 = welch(w2, fs=P.FS, nperseg=256, axis=-1)
        m = f <= P.BAND_HZ
        f, p1, p2 = f[m], p1[:, m], p2[:, m]
        bands = [np.log10(p1[:, (f >= lo) & (f < hi)].sum(1) + 1e-30) -
                np.log10(p2[:, (f >= lo) & (f < hi)].sum(1) + 1e-30)
                for lo, hi in zip(CD.BAND_EDGES[:-1], CD.BAND_EDGES[1:])]
        rows_dspec.append(np.column_stack(bands))
        groups.extend([f"md:{rec.rec_id}"] * n)
        ys.extend([rec.label] * n)
    dlogrms, xcorr = np.concatenate(rows_dlogrms), np.concatenate(rows_xcorr)
    dspec = np.concatenate(rows_dspec)
    groups, ys = np.array(groups), np.array(ys)
    F = np.column_stack([dlogrms, xcorr, dspec])
    names = ["d_log_rms", "zero_lag_corr"] + [f"d_band_{lo}_{hi}" for lo, hi in
                                              zip(CD.BAND_EDGES[:-1], CD.BAND_EDGES[1:])]
    k = min(5, int((ys == 0).sum()))
    full = oof_auroc(F, ys, groups, k, stratified=True)
    per = {name: {"leak_auroc": float(roc_auc_score(ys, F[:, i]))
                 if len(np.unique(ys)) == 2 else None} for i, name in enumerate(names)}
    return {"n_windows": len(ys), "n_groups": len(set(groups.tolist())),
           "full_relational_feature_oof_auroc": full, "per_feature": per}


def sheffield_position_contrast() -> dict:
    """Descriptive only, NOT simultaneous: for each Sheffield leak test condition with more than
    one distinct sensor-distance file, the normalised-PSD difference between its nearest and
    farthest position. Different time sessions of the same leak, not a coherence feature."""
    import re
    import pandas as pd
    out = {}
    for folder in sorted(d for d in rigs.SHEFFIELD_ROOT.iterdir()
                         if d.is_dir() and re.match(r"\(test#\d+\)", d.name)):
        k = int(re.match(r"\(test#(\d+)\)", folder.name).group(1))
        pos_psd = {}
        for f in sorted(folder.glob("*.csv")):
            if "noleak" in f.stem.lower():
                continue
            df = pd.read_csv(f, dtype=np.float32)
            for col in df.columns:
                if not re.fullmatch(r"Acc\d+", col):
                    continue
                d = int(col[3:])
                w, _, _ = rigs.stream_windows(df[col].to_numpy(), rigs.SHEFFIELD_FS)
                if len(w) == 0:
                    continue
                _, psd = normalised_psd(w)
                pos_psd.setdefault(d, []).append(np.log10(psd + 1e-30).mean(0))
        if len(pos_psd) < 2:
            continue
        near, far = min(pos_psd), max(pos_psd)
        c_near = np.mean(pos_psd[near], axis=0)
        c_far = np.mean(pos_psd[far], axis=0)
        out[f"test{k}"] = {"near_m": near, "far_m": far, "near_minus_far": (c_near - c_far).tolist()}
    return out


def main():
    t0 = time.time()
    print("Loading rigs (cached after first run)...")
    rws = {"mendeley_acc": rigs.load_rig("mendeley_acc"), "sheffield": rigs.load_rig("sheffield")}

    print("\n[1] per-feature leak-AUROC vs rig-AUROC")
    feat_table = per_feature_table(rws)
    for name, row in feat_table.items():
        print(f"  {name:16s} leak(mendeley)={row['leak_auroc']['mendeley_acc']:.3f} "
              f"leak(sheffield)={row['leak_auroc']['sheffield']:.3f} "
              f"rig_abs={row['rig_auroc_abs']:.3f}")

    print("\n[2] per-condition spectral contrast consistency")
    cond = {r: per_condition_contrast(rws[r], r) for r in rws}
    for r, c in cond.items():
        pw = c.get("pairwise_pearson_between_no_leak_groups", [])
        print(f"  {r}: {len(c['per_no_leak_group'])} no-leak groups, "
              f"pairwise contrast-curve Pearson {['%.2f' % p for p in pw]}")

    print("\n[3a] Mendeley ch1/ch2 simultaneity check")
    sim = mendeley_simultaneity_check()
    print(f"  mean |peak corr| {sim['mean_abs_peak_corr']:.3f}, "
          f"mean |zero-lag corr| {sim['mean_abs_zero_lag_corr']:.3f}")
    for o in sim["per_recording"]:
        print(f"    {o['rec']}: peak_corr={o['peak_corr']:.3f} at lag {o['peak_lag_samples']} samples "
              f"(zero-lag {o['zero_lag_corr']:.3f})")

    print("\n[3b] Mendeley two-sensor relational features (leak separability, within-rig only)")
    rel = mendeley_two_sensor_features()
    print(f"  n_windows={rel['n_windows']} n_groups={rel['n_groups']}")
    print(f"  full relational-feature oof AUROC: {rel['full_relational_feature_oof_auroc']['auroc']}")
    for name, v in rel["per_feature"].items():
        print(f"    {name:16s} leak_auroc={v['leak_auroc']:.3f}")

    print("\n[3c] Sheffield near-vs-far spectral contrast (descriptive, NOT simultaneous)")
    shpos = sheffield_position_contrast()
    for k, v in shpos.items():
        print(f"  {k}: near {v['near_m']} m vs far {v['far_m']} m")

    results = {"per_feature": feat_table, "per_condition_contrast": cond,
              "mendeley_simultaneity": sim, "mendeley_two_sensor": rel,
              "sheffield_position_contrast": shpos}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mechanism_analysis.json"
    out_path.write_text(json.dumps(_jsonable(results), indent=1))
    print(f"\nSaved {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
