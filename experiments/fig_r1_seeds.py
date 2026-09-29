"""
fig_r1_seeds.py — Figure 4: Model F seed replication (R1), accelerometer vs.
hydrophone AUROC across seeds 0/1/2, for the CJSJ manuscript.

Reads ONLY the two already-committed R1 run records (no model loaded, no
data touched, nothing re-evaluated):
  results/runs/2026-09-29_000514_e11_model_f_eval.json   (seed 0)
  results/runs/2026-09-29_193019_e11_model_f_eval.json   (seeds 1, 2)

    python experiments/fig_r1_seeds.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import PLOTS_DIR, REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"
SEED0_FILE = RUNS / "2026-09-29_000514_e11_model_f_eval.json"
SEED12_FILE = RUNS / "2026-09-29_193019_e11_model_f_eval.json"
CKPT = {0: "best_model_f_seed0.pt", 1: "best_model_f_seed1.pt", 2: "best_model_f_seed2.pt"}
TESTS = {"mendeley_looped_acc": "Accelerometer", "mendeley_looped_hyd": "Hydrophone"}


def load():
    seed0 = json.loads(SEED0_FILE.read_text())["results"][CKPT[0]]["tests"]
    seed12 = json.loads(SEED12_FILE.read_text())["results"]
    per_test = {t: {} for t in TESTS}
    for s, block in ((0, seed0), (1, seed12[CKPT[1]]["tests"]), (2, seed12[CKPT[2]]["tests"])):
        for t in TESTS:
            m = block[t]["model"]
            per_test[t][s] = (m["auroc"], m["ci95"]["auroc"])
    return per_test


def main():
    per_test = load()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    seeds = [0, 1, 2]
    for ax, (test_key, title) in zip(axes, TESTS.items()):
        aurocs = [per_test[test_key][s][0] for s in seeds]
        los = [per_test[test_key][s][1][0] for s in seeds]
        his = [per_test[test_key][s][1][1] for s in seeds]
        err_lo = [a - lo for a, lo in zip(aurocs, los)]
        err_hi = [hi - a for a, hi in zip(aurocs, his)]
        ax.errorbar(seeds, aurocs, yerr=[err_lo, err_hi], fmt="o", color="crimson",
                    capsize=5, markersize=7, linewidth=1.5)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (0.5)")
        ax.set_xticks(seeds)
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{title} (Mendeley Looped)")
        ax.set_xlabel("Training seed")
    axes[0].set_ylabel("AUROC (logits, 95% CI)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Model F seed replication (R1): real-data AUROC is seed-dependent\n"
                  "for the hydrophone sensor, not stably above chance for either sensor",
                  fontsize=10)
    plt.tight_layout()
    out = PLOTS_DIR / "fig4_r1_seed_replication.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")
    for t, title in TESTS.items():
        print(f"  {title}: " + ", ".join(f"seed{s}={per_test[t][s][0]:.3f} {per_test[t][s][1]}"
                                          for s in seeds))


if __name__ == "__main__":
    main()
