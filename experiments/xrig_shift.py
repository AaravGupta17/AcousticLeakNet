"""
xrig_shift.py — pre-specified failure analysis for the Version 2 cross-rig benchmark
====================================================================================
Three descriptive analyses from the "Failure analysis" section of
docs/VERSION2_CROSS_RIG_PREREG.md. None of them selects or tunes anything; CPU only.

  1. Spectra      Welch PSD (fs 5000, nperseg 256) of every unit-RMS window, normalised to
                  sum 1 over 0-2000 Hz, averaged within each independence group first and
                  then across groups, per rig and class. Contrast = mean log10 PSD(leak) -
                  mean log10 PSD(no-leak). Pearson / Spearman between the two rigs'
                  contrast curves. Median log_rms by group is listed per rig and class;
                  units differ between rigs (Sheffield calibrated m/s^2, Mendeley units
                  undocumented), so absolute levels are not comparable across rigs.
  2. Separability Loudness-free features (cross_dataset.features_1ch) -> standardised
                  logistic regression. (a) predict the RIG (Sheffield = 1), GroupKFold(5)
                  over groups pooled from both rigs. (b) predict the LABEL within each rig,
                  grouped folds (n_splits = min(5, no-leak groups)). Out-of-fold window
                  AUROC, descriptive only.
  3. Score tracking  Spearman rho of the transferred Model F score with log_rms and with
                  spectral centroid, within each class and pooled, for every
                  results/v2_cross_rig/scores_xrig_*_foldx_seed*.npz, plus per-group mean
                  scores. Skipped when no such file exists (or with --skip-scores).

Outputs: shift_analysis.json (timestamp suffix if it exists), plots/v2_xrig_shift.png and a
run record. With LEAKNET_OUT set, the JSON goes to <LEAKNET_OUT>/v2_cross_rig/.

    python experiments/xrig_shift.py [--skip-scores]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:                                           # pragma: no cover
    StratifiedGroupKFold = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Model_F"))
import cross_dataset as CD                                    # noqa: E402
import rigs                                                   # noqa: E402
from _common import PLOTS_DIR, REPO_ROOT, _jsonable, record_run   # noqa: E402

SCORES_DIR = REPO_ROOT / "results" / "v2_cross_rig"
_OUT = os.environ.get("LEAKNET_OUT")
OUT_DIR = Path(_OUT).resolve() / "v2_cross_rig" if _OUT else SCORES_DIR
SPECTRA_RIGS = ("sheffield", "mendeley_acc")
FS = 5000
BAND_HZ = 2000


# ── helpers (pure, tested in tests/test_xrig_shift.py) ─────────────────────────

def normalised_psd(x: np.ndarray):
    """(N, T) windows -> (freqs <= 2000 Hz, (N, F) PSD normalised to sum 1 over those bins)."""
    f, p = welch(x.astype(np.float64), fs=FS, nperseg=256, axis=-1)
    m = f <= BAND_HZ
    p = p[:, m]
    return f[m], p / (p.sum(axis=1, keepdims=True) + 1e-30)


def group_then_class_mean(values: np.ndarray, groups, y) -> dict:
    """Mean of `values` (N, ...) within each group, then across the groups of each class.
    Returns {class: (mean array, n_groups)}. Each group must have a single class."""
    groups, y = np.asarray(groups).astype(str), np.asarray(y).astype(int)
    out = {}
    for c in (0, 1):
        gs = sorted(set(groups[y == c]))
        if not gs:
            continue
        assert all(len(set(y[groups == g])) == 1 for g in gs), "group with both classes"
        out[c] = (np.mean([values[groups == g].mean(axis=0) for g in gs], axis=0), len(gs))
    return out


def contrast_correlation(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson and Spearman correlation between two contrast curves."""
    return {"pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b)[0])}


def spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return (psd * freqs).sum(axis=1)


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))


def oof_auroc(F, y, groups, n_splits, stratified=False) -> dict:
    """Out-of-fold decision-value AUROC with grouped folds."""
    y, groups = np.asarray(y).astype(int), np.asarray(groups).astype(str)
    if stratified and StratifiedGroupKFold is not None:
        cv, kind = StratifiedGroupKFold(n_splits=n_splits), "StratifiedGroupKFold"
    else:
        cv, kind = GroupKFold(n_splits=n_splits), "GroupKFold"
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(F, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        oof[te] = _clf().fit(F[tr], y[tr]).decision_function(F[te])
    ok = ~np.isnan(oof)
    return {"auroc": float(roc_auc_score(y[ok], oof[ok])) if len(np.unique(y[ok])) == 2 else None,
            "cv": kind, "n_splits": n_splits, "n_windows_scored": int(ok.sum()),
            "n_windows": int(len(y)), "n_groups": int(len(set(groups)))}


def median_log_rms_by_group(rw) -> dict:
    out = {}
    for c, name in ((0, "no_leak"), (1, "leak")):
        m = rw.y == c
        out[name] = {str(g): float(np.median(rw.log_rms[m & (rw.group == g)])) for g in sorted(set(rw.group[m]))}
    return out


# ── analyses ───────────────────────────────────────────────────────────────────

def analysis_spectra(data: dict) -> dict:
    res = {"note": "log_rms units differ between rigs (Sheffield calibrated m/s^2, Mendeley undocumented); "
                   "absolute levels are not comparable across rigs.", "rigs": {}}
    contrasts = {}
    for rig in SPECTRA_RIGS:
        rw, (freqs, psd) = data[rig]["rw"], data[rig]["psd"]
        gm = group_then_class_mean(np.log10(psd + 1e-30), rw.group, rw.y)
        # log10 of each window's normalised PSD, then group mean, then class mean
        contrast = gm[1][0] - gm[0][0]
        contrasts[rig] = contrast
        res["rigs"][rig] = {
            "freqs_hz": freqs, "mean_log10_psd_no_leak": gm[0][0], "mean_log10_psd_leak": gm[1][0],
            "n_groups_no_leak": gm[0][1], "n_groups_leak": gm[1][1], "contrast": contrast,
            "freq_of_max_contrast_hz": float(freqs[int(np.argmax(contrast))]),
            "median_log_rms_by_group": median_log_rms_by_group(rw)}
    res["contrast_correlation"] = contrast_correlation(contrasts["sheffield"], contrasts["mendeley_acc"])
    return res


def analysis_separability(data: dict) -> dict:
    feats = {r: CD.features_1ch(CD.standardise(data[r]["rw"].x)) for r in SPECTRA_RIGS}
    F = np.concatenate([feats[r] for r in SPECTRA_RIGS])
    rig_y = np.concatenate([np.full(len(feats[r]), int(r == "sheffield")) for r in SPECTRA_RIGS])
    grp = np.concatenate([data[r]["rw"].group for r in SPECTRA_RIGS]).astype(str)
    res = {"rig_id": oof_auroc(F, rig_y, grp, 5), "label_within_rig": {},
           "note": "out-of-fold window AUROC, descriptive only; label folds are few and uneven"}
    for r in SPECTRA_RIGS:
        rw = data[r]["rw"]
        k = min(5, len(set(rw.group[rw.y == 0])))
        res["label_within_rig"][r] = oof_auroc(feats[r], rw.y, rw.group, k, stratified=True)
    return res


def analysis_scores(data_cache: dict, files: list) -> dict:
    out = {}
    for fp in files:
        z = np.load(fp, allow_pickle=False)
        tnames = sorted(k[:-3] for k in z.files if k.endswith("__y"))
        entry = {}
        for t in tnames:
            if t not in rigs.RIGS:
                print(f"  {fp.name}: test set {t!r} is not a rig, skipped")
                continue
            if t not in data_cache:
                rw = rigs.load_rig(t)
                data_cache[t] = {"rw": rw, "psd": normalised_psd(rw.x)}
            rw, (freqs, psd) = data_cache[t]["rw"], data_cache[t]["psd"]
            y, grp, s = z[f"{t}__y"], z[f"{t}__group"], z[f"{t}__model_f"].astype(np.float64)
            assert np.array_equal(y, rw.y) and np.array_equal(grp.astype(str), rw.group.astype(str)), \
                f"{fp.name}/{t}: stored y/group differ from the reloaded rig"
            cen = spectral_centroid(psd, freqs)
            rho = {}
            for name, m in (("class0", y == 0), ("class1", y == 1), ("pooled", np.ones(len(y), bool))):
                rho[name] = {"vs_log_rms": float(spearmanr(s[m], rw.log_rms[m])[0]),
                             "vs_centroid": float(spearmanr(s[m], cen[m])[0]), "n": int(m.sum())}
            per_group = [{"group": str(g), "label": int(y[grp == g][0]), "mean_score": float(s[grp == g].mean()),
                          "n_windows": int((grp == g).sum())} for g in sorted(set(grp.astype(str)))]
            entry[t] = {"spearman": rho, "per_group_mean_score": per_group}
        out[fp.name] = entry
    return out


# ── figure ─────────────────────────────────────────────────────────────────────

def make_figure(spec: dict, sep: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"sheffield": "tab:blue", "mendeley_acc": "tab:orange"}
    for rig, r in spec["rigs"].items():
        f = np.asarray(r["freqs_hz"])
        ax[0].plot(f, r["mean_log10_psd_no_leak"], "--", color=colors[rig], label=f"{rig} no-leak")
        ax[0].plot(f, r["mean_log10_psd_leak"], "-", color=colors[rig], label=f"{rig} leak")
        ax[1].plot(f, r["contrast"], color=colors[rig], label=rig)
    ax[0].set(xlabel="Hz", ylabel="mean log10 normalised PSD", title="Group-then-class mean spectra")
    ax[1].axhline(0, color="gray", lw=0.5)
    cc = spec["contrast_correlation"]
    ax[1].set(xlabel="Hz", ylabel="leak - no-leak (log10)",
              title=f"Contrast (Pearson {cc['pearson']:.2f}, Spearman {cc['spearman']:.2f})")
    for a in ax[:2]:
        a.legend(fontsize=7)
    names = ["rig ID"] + [f"label: {r}" for r in sep["label_within_rig"]]
    vals = [sep["rig_id"]["auroc"]] + [v["auroc"] for v in sep["label_within_rig"].values()]
    ax[2].bar(names, [np.nan if v is None else v for v in vals], color="tab:gray")
    ax[2].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[2].set(ylim=(0, 1), ylabel="out-of-fold window AUROC", title="Separability (descriptive)")
    ax[2].tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def unique_path(p: Path) -> Path:
    return p if not p.exists() else p.with_name(f"{p.stem}_{time.strftime('%Y%m%d_%H%M%S')}{p.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scores", action="store_true", help="skip analysis 3 (score tracking)")
    args = ap.parse_args()

    data = {}
    for r in SPECTRA_RIGS:
        rw = rigs.load_rig(r)
        data[r] = {"rw": rw, "psd": normalised_psd(rw.x)}

    spec = analysis_spectra(data)
    cc = spec["contrast_correlation"]
    print(f"\n[1] contrast correlation sheffield vs mendeley_acc: Pearson {cc['pearson']:.4f}, "
          f"Spearman {cc['spearman']:.4f}")
    for r in SPECTRA_RIGS:
        print(f"    {r}: max contrast at {spec['rigs'][r]['freq_of_max_contrast_hz']:.1f} Hz "
              f"(groups no-leak/leak {spec['rigs'][r]['n_groups_no_leak']}/{spec['rigs'][r]['n_groups_leak']})")
        for cls, d in spec["rigs"][r]["median_log_rms_by_group"].items():
            print(f"    {r} {cls} median log_rms by group: " + ", ".join(f"{g}={v:.3f}" for g, v in d.items()))

    sep = analysis_separability(data)
    print(f"\n[2] rig-ID out-of-fold AUROC: {sep['rig_id']['auroc']:.4f} "
          f"({sep['rig_id']['cv']}, {sep['rig_id']['n_splits']} folds)")
    for r, v in sep["label_within_rig"].items():
        a = v["auroc"]
        print(f"    label within {r}: out-of-fold AUROC {'n/a' if a is None else f'{a:.4f}'} "
              f"({v['cv']}, {v['n_splits']} folds, {v['n_windows_scored']}/{v['n_windows']} windows scored; descriptive)")

    scores = None
    if args.skip_scores:
        print("\n[3] skipped (--skip-scores)")
    else:
        files = sorted(SCORES_DIR.glob("scores_xrig_*_foldx_seed*.npz"))
        if not files:
            print(f"\n[3] no scores files matching scores_xrig_*_foldx_seed*.npz in {SCORES_DIR}; skipped")
        else:
            scores = analysis_scores(data, files)
            for fn, entry in scores.items():
                for t, e in entry.items():
                    for k, v in e["spearman"].items():
                        print(f"[3] {fn} / {t} / {k}: rho vs log_rms {v['vs_log_rms']:.3f}, "
                              f"vs centroid {v['vs_centroid']:.3f} (n={v['n']})")

    results = {"spectra": spec, "separability": sep, "score_tracking": scores}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = unique_path(OUT_DIR / "shift_analysis.json")
    jp.write_text(json.dumps(_jsonable(results), indent=1))
    fig_path = unique_path(PLOTS_DIR / "v2_xrig_shift.png")
    make_figure(spec, sep, fig_path)
    record_run("v2_xrig_shift", {"skip_scores": args.skip_scores}, results,
               f"contrast Pearson {cc['pearson']:.2f}, rig-ID AUROC {sep['rig_id']['auroc']:.3f}")
    print(f"\nSaved {jp}\nSaved {fig_path}")


if __name__ == "__main__":
    main()
