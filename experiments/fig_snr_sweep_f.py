"""
fig_snr_sweep_f.py — Figure 2: Model F SNR robustness sweep, for the CJSJ manuscript.

Reads ONLY the already-committed, pre-registered SNR sweep result record (no model
loaded, no data touched, nothing re-evaluated):
    tests30/runs/snr_sweep_f_results.json   (== results/snr_sweep_f/snr_sweep_f_results.json)

    python experiments/fig_snr_sweep_f.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import PLOTS_DIR, REPO_ROOT

RESULTS_FILE = REPO_ROOT / "tests30" / "runs" / "snr_sweep_f_results.json"
SEED_COLORS = {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"}


def load():
    d = json.loads(RESULTS_FILE.read_text())
    grid = d["snr_grid_db"]
    native_floor = d["native_floor_db"]
    seeds = {}
    for s, block in d["seeds"].items():
        aurocs, los, his = [], [], []
        for snr in grid:
            lv = block["levels"][str(snr)]
            aurocs.append(lv["auroc"])
            los.append(lv["ci95"]["auroc"][0])
            his.append(lv["ci95"]["auroc"][1])
        seeds[s] = (aurocs, los, his)
    rms = d["rms_baseline"]
    rms_auroc = [rms[str(snr)]["auroc"] for snr in grid]
    rms_lo = [rms[str(snr)]["ci95"]["auroc"][0] for snr in grid]
    rms_hi = [rms[str(snr)]["ci95"]["auroc"][1] for snr in grid]
    return grid, native_floor, seeds, (rms_auroc, rms_lo, rms_hi)


def main():
    grid, native_floor, seeds, (rms_auroc, rms_lo, rms_hi) = load()

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for s in ("0", "1", "2"):
        aurocs, los, his = seeds[s]
        err_lo = [a - lo for a, lo in zip(aurocs, los)]
        err_hi = [hi - a for a, hi in zip(aurocs, his)]
        ax.errorbar(grid, aurocs, yerr=[err_lo, err_hi], fmt="o-", capsize=3,
                    markersize=5, linewidth=1.6, color=SEED_COLORS[s], label=f"Model F, seed {s}")

    err_lo = [a - lo for a, lo in zip(rms_auroc, rms_lo)]
    err_hi = [hi - a for a, hi in zip(rms_auroc, rms_hi)]
    ax.errorbar(grid, rms_auroc, yerr=[err_lo, err_hi], fmt="s--", capsize=3,
                markersize=5, linewidth=1.4, color="gray", label="RMS baseline")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.axvline(native_floor, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(native_floor, 0.515, " native\n training\n floor", fontsize=7, va="bottom", ha="left")

    ax.set_xlabel("Injected SNR (dB) — synthetic evaluation")
    ax.set_ylabel("AUROC on logits (95% CI)")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(grid)
    ax.set_title("Model F ranks synthetic leak windows above chance down to\n"
                  "-20 dB, well below its native training floor (in-domain only)",
                  fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.5, 0.005,
              "Ranking (AUROC) only — thresholded detection at logit ≥ 0 is 0.26-0.32 at -20 dB, "
              "rising to 0.81-0.83 at 0 dB (Results 3.1b).",
              ha="center", fontsize=7, style="italic")
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    out = PLOTS_DIR / "fig2_snr_sweep_f.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")
    for s in ("0", "1", "2"):
        aurocs, los, his = seeds[s]
        print(f"  seed {s}: " + ", ".join(f"{snr}dB={a:.3f} [{lo:.3f},{hi:.3f}]"
                                           for snr, a, lo, hi in zip(grid, aurocs, los, his)))
    print("  rms:    " + ", ".join(f"{snr}dB={a:.3f} [{lo:.3f},{hi:.3f}]"
                                    for snr, a, lo, hi in zip(grid, rms_auroc, rms_lo, rms_hi)))


if __name__ == "__main__":
    main()
