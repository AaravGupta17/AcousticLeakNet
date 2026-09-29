"""
r2_held_out_groups.py - AcousticLeakNet R2: held-out real-group stability
==========================================================================
Pre-registered in docs/ACOUSTICLEAKNET_R2_PREREG.md (frozen before this script
was run). Uses ONLY the three already-trained Model F checkpoints
(best_model_f_seed{0,1,2}.pt) and the already-built cache_f/bank.npz. No new
training. Does not change docs/MODEL_F_PREREG.md, its addendum, or H0-H4.

Question: is Model F's ranking of never-trained-on real groups from its own
training sources (Hong Kong, Dongguan) stable across seeds, or seed-noise?

Rows with s_val == True in bank.npz were excluded from EVERY training step of
EVERY seed (Model_F/train_f.py's Bank class filters by s_val; is_val_group in
bank_f.py hashes only the group string, independent of --seed). This script
re-evaluates them post hoc, at ONE shared evaluation seed across all three
checkpoints (a deliberate deviation from train_f.py's own per-seed real-val
construction, needed so seed-to-seed differences reflect the model and not a
different random channel-pairing draw).

Primary endpoint: Spearman correlation of per-group mean logits between each
pair of seeds (low selection-bias: checkpoint selection optimised a POOLED
window AUROC, not this ranking statistic).
Secondary (descriptive; caveated as optimistically selected by best-of-30
checkpoint picking): group-level AUROC per seed per source, with a group
cluster-bootstrap 95% CI, vs. a trivial-feature logistic-regression baseline
trained on the TRAIN-split groups of the same source (mirrors the existing
H2/H3 logreg baseline).

    python Model_F/r2_held_out_groups.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "Model_F"))
import augment_f as A                                      # noqa: E402
from cross_dataset import features_1ch, standardise         # noqa: E402
from _common import load_model, predict_proba, record_run   # noqa: E402
import train_f as TF                                        # noqa: E402  (Bank class)

EVAL_SEED = 0          # ONE fixed seed for channel-pairing/EQ, shared by all checkpoints
N_BOOT = 2000
SEEDS = (0, 1, 2)
CKPTS = {s: f"best_model_f_seed{s}.pt" for s in SEEDS}
BANK = ROOT / "cache_f" / "bank.npz"


def build_real_eval_windows(bank: TF.Bank, seed: int) -> np.ndarray:
    """Two-channel windows for every row in `bank`, at ONE fixed seed:
    A.finish(bank.two_channel(i, rng), rng), exactly train_f.py's real-val
    construction, but with a single shared rng so all checkpoints see
    identical inputs."""
    rng = np.random.default_rng(seed)
    x = np.stack([A.finish(bank.two_channel(i, rng).astype(np.float64), rng)
                 for i in range(len(bank.single))])
    return x.astype(np.float32)


def group_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC over GROUPS (each group = one point): fraction of (leak, no-leak)
    pairs correctly ranked by score, with 0.5 credit for ties."""
    leak, noleak = scores[labels == 1], scores[labels == 0]
    if not len(leak) or not len(noleak):
        return float("nan")
    diff = leak[:, None] - noleak[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def group_auroc_ci(labels: np.ndarray, scores: np.ndarray, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    rng = np.random.default_rng(seed)
    leak_idx, noleak_idx = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    point = group_auroc(labels, scores)
    boots = [group_auroc(labels[bi], scores[bi]) for bi in
            (np.concatenate([rng.choice(leak_idx, len(leak_idx), replace=True),
                            rng.choice(noleak_idx, len(noleak_idx), replace=True)])
             for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": point, "ci95": [float(lo), float(hi)], "n_boot": n_boot,
            "n_leak_groups": int(len(leak_idx)), "n_noleak_groups": int(len(noleak_idx))}


def per_group_mean(scores: np.ndarray, groups: np.ndarray):
    order = {}
    for g, s in zip(groups, scores):
        order.setdefault(g, []).append(s)
    keys = sorted(order)
    return keys, np.array([np.mean(order[k]) for k in keys])


def logreg_group_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """H2/H3-style trivial-feature baseline: fit on TRAIN-split single-channel
    windows of this source, score held-out single-channel windows (channel 1
    only, as model_f_eval.py's logreg_baseline does)."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    clf.fit(features_1ch(standardise(train_x)), train_y)
    return clf.decision_function(features_1ch(standardise(test_x)))


def main():
    z = np.load(BANK, allow_pickle=False)
    val_mask = z["s_val"] == True
    tr_mask = ~val_mask
    sources = sorted(set(z["s_source"].tolist()))
    print(f"Held-out (s_val=True) rows: {val_mask.sum()} | sources: {sources}")

    bank_va = TF.Bank(BANK, val=True, exclude=())         # identical for every seed (no --seed dependence)
    x_eval = build_real_eval_windows(bank_va, EVAL_SEED)   # SAME inputs for all 3 checkpoints
    src_va = z["s_source"][val_mask]

    # ---- one forward pass per seed; reused for every downstream statistic ----
    seed_logit = {}
    for s in SEEDS:
        model, ckpt = load_model(CKPTS[s])
        assert ckpt["cfg"].get("input_norm") == "zscore", f"{CKPTS[s]} is not a Model F checkpoint"
        seed_logit[s] = predict_proba(model, x_eval, logits=True)
        print(f"seed {s}: epoch {ckpt.get('epoch')}, held-out windows scored = {len(seed_logit[s])}")

    results = {"eval_seed": EVAL_SEED, "n_boot": N_BOOT, "sources": {}}
    for src in sources:
        sm = src_va == src
        groups_src, y_src = bank_va.group[sm], bank_va.y[sm]
        ref_keys, ref_y = None, None
        seed_mean_logit = {}
        for s in SEEDS:
            keys, mean_logit = per_group_mean(seed_logit[s][sm], groups_src)
            grp_y = np.array([y_src[groups_src == k][0] for k in keys])
            if ref_keys is None:
                ref_keys, ref_y = keys, grp_y
            else:
                assert keys == ref_keys and (grp_y == ref_y).all(), "held-out group set differs across seeds"
            seed_mean_logit[s] = mean_logit

        entry = {"n_groups": len(ref_keys), "groups": ref_keys, "labels": ref_y.tolist(),
                 "n_leak_groups": int(ref_y.sum()), "n_noleak_groups": int((ref_y == 0).sum())}

        # Primary: cross-seed Spearman of per-group mean logits
        entry["spearman"] = {}
        for a, b in itertools.combinations(SEEDS, 2):
            rho, p = spearmanr(seed_mean_logit[a], seed_mean_logit[b])
            entry["spearman"][f"{a}-{b}"] = {"rho": float(rho), "p": float(p)}
        rhos = [v["rho"] for v in entry["spearman"].values()]
        n_sig = sum(v["p"] < 0.05 for v in entry["spearman"].values())
        med_rho = float(np.median(rhos))
        stability = ("Stable" if med_rho >= 0.5 and n_sig == 3 else
                    "Unstable" if med_rho < 0.3 or n_sig <= 1 else "Mixed")
        entry["median_rho"], entry["stability_classification"] = med_rho, stability

        # Secondary (descriptive; caveated): group-level AUROC per seed
        entry["model_auroc"] = {str(s): group_auroc_ci(ref_y, seed_mean_logit[s]) for s in SEEDS}

        # Control: trivial-feature logreg baseline, fit on TRAIN-split single-channel windows
        # of the same source, scored on the held-out single-channel windows (channel 1 only)
        tr_sm = z["s_source"][tr_mask] == src
        train_x = z["single"][tr_mask][tr_sm].astype(np.float32)
        train_y = z["s_y"][tr_mask][tr_sm].astype(int)
        base_scores = logreg_group_scores(train_x, train_y, bank_va.single[sm])
        _, base_mean = per_group_mean(base_scores, groups_src)
        entry["logreg_baseline_auroc"] = group_auroc_ci(ref_y, base_mean)

        # Informational only: window-pooled AUROC, same construction as train_f.py's
        # checkpoint-selection metric -- INFLATED by best-of-30 selection, not a fresh estimate
        entry["window_pooled_auroc_INFLATED_BY_SELECTION"] = {
            str(s): float(roc_auc_score(y_src, seed_logit[s][sm])) for s in SEEDS}

        results["sources"][src] = entry
        print(f"\n=== {src}: {entry['n_groups']} held-out groups "
              f"({entry['n_leak_groups']} leak / {entry['n_noleak_groups']} no-leak) ===")
        print("  Spearman rho: " + ", ".join(f"{k}={v['rho']:.3f} (p={v['p']:.3g})"
                                             for k, v in entry["spearman"].items()))
        print(f"  -> {stability} (median rho {med_rho:.3f}, {n_sig}/3 significant)")
        for s in SEEDS:
            a = entry["model_auroc"][str(s)]
            print(f"  seed {s} group-AUROC {a['auroc']:.3f} {a['ci95']} "
                  f"(window-pooled, inflated: {entry['window_pooled_auroc_INFLATED_BY_SELECTION'][str(s)]:.3f})")
        b = entry["logreg_baseline_auroc"]
        print(f"  logreg baseline group-AUROC {b['auroc']:.3f} {b['ci95']}")

    out_dir = ROOT / "results" / "r2_held_out_groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r2_results.json").write_text(json.dumps(results, indent=1))
    summary = "; ".join(f"{src}: {e['stability_classification']} (median rho {e['median_rho']:.2f})"
                        for src, e in results["sources"].items())
    record_run("r2_held_out_groups", {"eval_seed": EVAL_SEED, "n_boot": N_BOOT, "seeds": list(SEEDS)},
              results, summary)
    print(f"\nWrote {out_dir / 'r2_results.json'}")


if __name__ == "__main__":
    main()
