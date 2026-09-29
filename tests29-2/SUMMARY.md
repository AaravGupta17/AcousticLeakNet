# tests29-2: Model F seed replication (R1)

Replication defined in `docs/MODEL_F_PREREG_ADDENDUM_1_SEEDS.md` (written after the seed-0 results
were known and before any other seed was trained; the motivation is post hoc, the rules were fixed
before the runs). Only `--seed` changes (seeds 1 and 2). Runs from the repo root **without**
`LEAKNET_OUT`. Nothing in the addendum or in `docs/MODEL_F_PREREG.md` was changed after seeing
results.

## 1. What was run

    python Model_F/train_f.py --seed 1
    python Model_F/train_f.py --seed 2
    python experiments/model_f_eval.py --ckpt best_model_f_seed1.pt best_model_f_seed2.pt

Plus one descriptive analysis: `python -u tests29-2/spearman_seeds.py` (log `logs/05_spearman_seeds.log`).

- Code: commit `e24cb892d1dbc74eac45a083c6f0d85e84c19653`, `git_dirty` False in all three run
  records. Python 3.12.10, torch 2.7.1+cu118, CUDA, NVIDIA GeForce RTX 3060, driver (NVIDIA-SMI
  header) 617.14 (`logs/00_preflight.log`, `logs/04_nvidia_smi_after.log`).
- Preflight (`logs/00_preflight.log`, `logs/00_preflight_hashes.log`): clean working tree at
  start; `LEAKNET_OUT` empty; every `cache_f` file's sha256 equals its Git LFS id, and the rebuilt
  `train/leak.npy` (the concatenation of its four parts) equals the sha256 listed in
  `cache_f/README.md`. No `.py` file changed between `2dbc6ae` (the seed-0 training commit) and
  this commit. The `config` blocks of the seed 0, 1 and 2 training records are identical except
  for `seed`.
- Wall times: seed 1 training 8683 s, seed 2 training 7670 s, E11 93 s.
- No crashes and no reruns; each run finished once and its result is used.

Best epoch (checkpoint rule: best mean of synthetic and real validation AUROC on held-out groups
of the training sources). Records: seed 0 `results/runs/2026-09-26_044252_train_f.json`, seed 1
`tests29-2/runs/2026-09-29_172028_train_f.json`, seed 2 `tests29-2/runs/2026-09-29_192831_train_f.json`.

| Seed | Best epoch | Val AUROC synthetic | Val AUROC real |
|---|---|---|---|
| 0 (official) | 10 | 0.941 | 0.940 |
| 1 | 7 | 0.931 | 0.951 |
| 2 | 15 | 0.944 | 0.933 |

## 2. Decision rule result (R1)

Source: `results/runs/2026-09-29_193019_e11_model_f_eval.json` (identical copy in `runs/`).
`mendeley_looped_hyd` (Looped-only), values as written in the JSON.

| Seed | AUROC | 95% CI lower bound | AUROC > 0.5 | CI lower bound > 0.5 |
|---|---|---|---|---|
| 1 | 0.8123230288350081 | 0.6933234646357614 | yes | yes |
| 2 | 0.5080173633594742 | 0.29183334406058026 | yes | **no** (0.2918... <= 0.5) |

Both seeds have AUROC > 0.5 and at least one CI lower bound is <= 0.5, so:

**R1: Direction only.**

Seed 2's AUROC is only 0.008 above 0.5 and its CI (0.292 to 0.718) spans it widely. Per the
addendum the GPU run record is authoritative and no run is repeated.

**H2's official result is unchanged:** fail on both sensors, on seed 0
(`results/runs/2026-09-29_000514_e11_model_f_eval.json`). R1 is reported separately and cannot
change it.

## 3. All seeds, both sensors

Sources: seed 0 `results/runs/2026-09-29_000514_e11_model_f_eval.json` (checkpoint
`best_model_f_seed0.pt`, unchanged); seeds 1 and 2 `results/runs/2026-09-29_193019_e11_model_f_eval.json`.
AUROC on logits, 95% CIs bootstrapped over recordings (2000 resamples), Mendeley Looped-only.
Detection and false alarm at threshold 0 on the logit. The logreg baseline is the same in all
three (same real sources), so its AUROC and CI do not change with the seed.

| Seed | Test | AUROC [95% CI] | Detect | False alarm | Balanced acc. | Logreg AUROC [95% CI] |
|---|---|---|---|---|---|---|
| 0 | mendeley_looped_acc | 0.424 [0.193, 0.680] | 0.705 | 0.816 | 0.444 | 0.295 [0.050, 0.627] |
| 1 | mendeley_looped_acc | 0.318 [0.116, 0.552] | 0.698 | 0.912 | 0.393 | 0.295 [0.050, 0.627] |
| 2 | mendeley_looped_acc | 0.445 [0.212, 0.685] | 0.700 | 0.740 | 0.480 | 0.295 [0.050, 0.627] |
| 0 | mendeley_looped_hyd | 0.630 [0.500, 0.762] | 0.063 | 0.028 | 0.517 | 0.188 [0.106, 0.288] |
| 1 | mendeley_looped_hyd | 0.812 [0.693, 0.908] | 0.025 | 0.014 | 0.506 | 0.188 [0.106, 0.288] |
| 2 | mendeley_looped_hyd | 0.508 [0.292, 0.718] | 0.126 | 0.196 | 0.465 | 0.188 [0.106, 0.288] |

Seed 0 hydrophone CI lower bound is 0.49997 (0.4999704...), as in the addendum.

Mean and range (min to max) over seeds 0, 1, 2:

| Sensor | AUROC mean (range) | Detect mean (range) | False alarm mean (range) |
|---|---|---|---|
| mendeley_looped_acc | 0.395 (0.318 to 0.445) | 0.701 (0.698 to 0.705) | 0.823 (0.740 to 0.912) |
| mendeley_looped_hyd | 0.650 (0.508 to 0.812) | 0.072 (0.025 to 0.126) | 0.080 (0.014 to 0.196) |

## 4. H2 rule applied per seed (DESCRIPTIVE ONLY)

H2 (`docs/MODEL_F_PREREG.md`): on each Looped sensor, CI lower bound > 0.5 **and** AUROC > logreg
AUROC; both sensors = pass, one = partial. Applied here to seeds 1 and 2 (and to seed 0 for
comparison) only as a description. It is **not** a pre-registered test for these seeds and does not
alter H2's verdict.

| Seed | acc: lower bound > 0.5 | acc: AUROC > logreg | hyd: lower bound > 0.5 | hyd: AUROC > logreg | Descriptive H2 |
|---|---|---|---|---|---|
| 0 | no (0.193) | yes (0.424 > 0.295) | no (0.49997) | yes (0.630 > 0.188) | fail on both sensors |
| 1 | no (0.116) | yes (0.318 > 0.295) | yes (0.693) | yes (0.812 > 0.188) | partial (hyd only) |
| 2 | no (0.212) | yes (0.445 > 0.295) | no (0.292) | yes (0.508 > 0.188) | fail on both sensors |

The logreg baseline itself is below 0.5 on both Looped sensors (0.295 and 0.188), so "AUROC >
logreg" is a low bar here.

## 5. Spearman correlation of per-recording mean logits (descriptive, no pass rule)

Source: `spearman/spearman_results.json` (per-recording values in `spearman/per_recording_logits.csv`;
log `logs/05_spearman_seeds.log`). Comparison of cross-seed rankings is descriptive only.

| Sensor | Pair | rho | p | n recordings |
|---|---|---|---|---|
| mendeley_looped_acc | 0-1 | 0.914 | 1.73e-08 | 20 |
| mendeley_looped_acc | 0-2 | 0.952 | 1.1e-10 | 20 |
| mendeley_looped_acc | 1-2 | 0.920 | 9.2e-09 | 20 |
| mendeley_looped_hyd | 0-1 | 0.596 | 0.00051 | 30 |
| mendeley_looped_hyd | 0-2 | 0.374 | 0.0415 | 30 |
| mendeley_looped_hyd | 1-2 | 0.157 | 0.406 | 30 |

Cross-check: the window-level AUROCs on logits recomputed by `spearman_seeds.py`
(`window_auroc_logits` in `spearman_results.json`) match the E11 JSON AUROCs. Maximum absolute
difference over all six (seed, sensor) cells: 0.0.

The accelerometer seeds rank the recordings almost identically to each other (rho 0.91 to 0.95),
but all three have AUROC below 0.5. On the hydrophone the rankings agree only partly (rho 0.16 to
0.60), so the seeds do not order the recordings the same way.

## 6. Accelerometer (no pass rule)

Reported in the section 3 table. AUROC is 0.318 to 0.445 across seeds, all three CIs include 0.5,
and at threshold 0 the models flag most recordings as leak (false alarm 0.740 to 0.912).

## 7. Interpretation

- The seed-0 hydrophone result does not give a stable estimate of the method. The three seeds span
  AUROC 0.508 to 0.812 on the same test data with the same training data, code and hyperparameters,
  and only one of the three (seed 1) has a CI lower bound above 0.5.
- R1 is "Direction only": all hydrophone AUROCs are above 0.5, but that does not reproduce as a
  reliable result. The accelerometer AUROCs are below 0.5 for all three seeds.
- Detection and false-alarm rates at logit 0 are very low for hydrophone seed 1 (0.025 and
  0.014), and low for seeds 0 and 2 too (0.063/0.028 and 0.126/0.196). The hydrophone AUROC
  reflects a ranking of recordings, not a usable threshold.
- Limits stated in the addendum: three seeds describe the spread of the method and do not give a
  tight estimate; even a Replicated result would not show leak detection on the hydrophone data,
  because the Mendeley no-leak recordings differ from the leak recordings in recording setup and
  session (E4, E6); GPU non-determinism alone moves AUROC by about 0.01 (INTEGRITY_LOG #22).
- This does not show that the model generalises to real pipes.

## Contents

| Path | What |
|---|---|
| `SUMMARY.md` | this file |
| `spearman_seeds.py` | descriptive per-recording Spearman analysis over seeds 0, 1, 2 |
| `checkpoint_sha256.txt` | sha256 of the two new checkpoints |
| `final_head.txt` | git HEAD at the end of the runs |
| `logs/00_preflight.log` | HEAD, clean-tree check, GPU and torch info before training |
| `logs/00_preflight_hashes.log` | `cache_f` sha256 against Git LFS ids |
| `logs/01_train_f_seed1.log` | `python Model_F/train_f.py --seed 1` |
| `logs/02_train_f_seed2.log` | `python Model_F/train_f.py --seed 2` |
| `logs/03_e11_seeds12.log` | E11 for seeds 1 and 2 |
| `logs/04_nvidia_smi_after.log` | nvidia-smi after the runs |
| `logs/05_spearman_seeds.log` | Spearman analysis output |
| `runs/` | copies of the `results/runs/` records: two `train_f` records and the E11 record |
| `histories/` | training histories of seeds 1 and 2 |
| `spearman/` | `spearman_results.json`, `per_recording_logits.csv` |

Checkpoints (in `models/`, sha256 from `checkpoint_sha256.txt`):

- `best_model_f_seed1.pt`: `bf434b3dc5baec8a5c5e4426935f8c44a155da8f441228eaf20bfa60957230e0`
- `best_model_f_seed2.pt`: `2f36bbc556e3353d12f23436b9ad6b1d9312d3acf1ddb3a24ce223672988792d`

Final HEAD: `e24cb892d1dbc74eac45a083c6f0d85e84c19653` (unchanged; nothing was committed).
