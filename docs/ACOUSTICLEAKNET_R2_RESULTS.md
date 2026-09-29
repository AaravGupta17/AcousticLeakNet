# AcousticLeakNet R2 results: held-out real-source group stability

Pre-registered: `docs/ACOUSTICLEAKNET_R2_PREREG.md` (frozen at commit `fed18cb`, before this was
run). Script: `Model_F/r2_held_out_groups.py`. Raw output:
`results/r2_held_out_groups/r2_results.json`; run record:
`results/runs/2026-09-29_214254_r2_held_out_groups.json`. No new training — reuses
`best_model_f_seed{0,1,2}.pt` exactly as committed for R1.

## What was run

Evaluated all three seeds on the 1,159 real windows `bank_f.py` already excludes from every
training step of every seed (`s_val == True`, seed-independent): 8 Hong Kong sites (5 leak, 3
no-leak, 675 windows) and 79 Dongguan trial-groups (73 leak, 6 no-leak, 484 windows). One fixed
evaluation seed (0) built the two-channel pairing for all three checkpoints, so differences
between seeds reflect the model, not a different random pairing draw.

## Primary endpoint: cross-seed rank stability (not the quantity checkpoint selection optimised)

| Source | Spearman rho (0-1, 0-2, 1-2) | Median rho | Significant (p<0.05) | Classification |
|---|---|---|---|---|
| Hong Kong | 0.929, 0.976, 0.976 | 0.976 | 3/3 | **Stable** |
| Dongguan | 0.672, 0.667, 0.631 | 0.667 | 3/3 | **Stable** |

Both sources are **Stable** by the frozen rule (median rho >= 0.5, all three p < 0.05). This is
the opposite of R1's finding for the out-of-domain Mendeley hydrophone target (rho 0.16-0.60,
inconsistent significance): whatever Model F learned about its own training sources ranks
never-trained-on groups the same way regardless of training seed.

## Secondary endpoint (descriptive; group-level AUROC is inflated by best-of-30 checkpoint
selection, since checkpoint selection maximised a pooled-window version of this same held-out set)

| Source | Seed 0 | Seed 1 | Seed 2 | Logreg baseline (loudness-free spectral features) |
|---|---|---|---|---|
| Hong Kong (8 groups) | 1.000 [1.0, 1.0] | 1.000 [1.0, 1.0] | 1.000 [1.0, 1.0] | 0.800 [0.40, 1.00] |
| Dongguan (79 groups) | 0.991 [0.966, 1.0] | 0.993 [0.973, 1.0] | 0.993 [0.973, 1.0] | 0.973 [0.913, 1.0] |

Window-pooled AUROC (the inflated number, for continuity with `train_f.py`'s own metric): Hong
Kong 0.90-0.93, Dongguan 0.945-0.967 across seeds. Not used for any claim here.

## Interpretation

**Stable, not evidence of a transferable leak cue.** As pre-registered, this design cannot
separate "detects the physical leak" from "recognises this acoustic environment," because every
held-out group is single-class (a site or trial was monitored during a leak, or was not, never
both) — the same limitation as any field leak survey. The control shows why this matters here
specifically: a trivial, loudness-free spectral-texture classifier (`features_1ch`: band powers,
centroid, flatness, kurtosis, crest factor, zero-crossing rate — no leak-specific engineering)
already reaches AUROC 0.800 on Hong Kong and 0.973 on Dongguan on the SAME held-out groups. Model
F's margin over that baseline is real (clearest on Hong Kong: 1.000 vs 0.800, non-overlapping
point estimates though the baseline's CI is wide at n=8) but the baseline itself is already close
to ceiling on Dongguan (a controlled lab rig where a leak is a genuine, large acoustic energy
event, so near-perfect separability by any reasonable feature is expected, mirroring why the
Mendeley Looped-only synthetic-style tests are also easy in-domain). Dongguan's held-out set is
also heavily leak-skewed (73 leak vs. 6 no-leak groups).

**What this adds:** a stable, non-seed-dependent, above-trivial-baseline signal on Model F's own
training sources — genuinely different from the out-of-domain Mendeley instability R1 found — but
one best read as evidence that Hong Kong/Dongguan sites are acoustically distinct enough that a
model (and to a lesser extent, simple spectral features) can tell them apart, not as evidence the
model has learned a physically transferable leak signature. This mirrors, on the training-domain
data, the same site/trial-specific-structure story the project already documents for Mendeley (E4:
loudness inverted between leak/no-leak; E6: confounding by flow condition).

**What would still be needed to separate the two explanations:** a frozen retrain with several
Hong Kong sites moved into a checkpoint-selection-blind test split (removes the best-of-30
selection bias entirely) — flagged in the pre-registration as a possible R3, not run here because
the current within-source held-out design cannot resolve the site-vs-physics question regardless
of how much more compute is spent on it; the limitation is the data's site=label confound, not the
sample size.

## Reproducibility

- Code: commit `1551414` (checkpoints/bank unchanged; `fed18cb` adds only the prereg and this
  script/results).
- `python Model_F/r2_held_out_groups.py` from the repo root, CPU (torch 2.13.0+cpu on this
  machine; no CUDA needed for evaluation-only). ~1,159 windows x 3 checkpoints, seconds of
  inference plus the 2000-resample bootstrap per source/seed.
- `cache_f/bank.npz` as checked out at `1551414`; no `cache_f` file was modified.
