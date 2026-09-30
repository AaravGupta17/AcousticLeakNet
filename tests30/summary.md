# tests30: Model F SNR robustness sweep (pre-registered)

Run from the repo root **without** `LEAKNET_OUT` (real results). Protocol:
`docs/MODEL_F_SNR_SWEEP_PREREG.md`, run once as written; nothing in the pre-registration, script,
grid, seeds, sample counts, baseline or criterion was changed before or after the run.

Code: commit `4b20090` (pre-registration, script and tests added in one commit, before any
result existed). Tree clean apart from untracked output. GPU: RTX 3060, torch 2.7.1+cu118,
Python 3.12.10.

## Contents

| Path | What |
|---|---|
| `logs/01_pytest_full.log` | `python -m pytest tests -v` (93 passed) |
| `logs/02_snr_sweep_f_dryrun.log` | `python Model_F/snr_sweep_f.py --dry-run` (builds 8 windows, no checkpoint, no AUROC) |
| `logs/03_snr_sweep_f.log` | `python Model_F/snr_sweep_f.py`: **the pre-registered run** (default args) |
| `logs/04_criterion_check.log` | full metrics read from the results JSON and the frozen success criterion applied |
| `runs/2026-09-30_155956_snr_sweep_f.json` | copy of the `results/runs/` record |
| `runs/snr_sweep_f_results.json` | copy of `results/snr_sweep_f/snr_sweep_f_results.json` |

Absolute user paths in the logs were replaced with `<repo>/`; nothing else was edited.

## 1. Data

- `datasets/NetworkList/`: `Network_1`-`Network_9` + `NetworkMend`, each with `CI/DI/PVC/STEEL`.
  Not modified.
- `data/csv/val_sampled.csv`: 91,896 rows (all `Network_7`; 45,948 leak / 45,948 base),
  12,165 unique referenced files, **0 missing** (paths resolve relative to `model_C/`).
- `LeakDatasetE` cached 91,896/91,896 rows; 45,948 leak scenarios loaded.
- `cache_f/bank.npz` (`val=True` slice): 1,159 real background windows, 91 Mendeley pairs.
- Checkpoints: `best_model_f_seed0.pt` (epoch 10), `seed1` (epoch 7), `seed2` (epoch 15), committed at `1551414`.

## 2. Configuration (as found in the pre-registration and script)

| Item | Value |
|---|---|
| SNR grid | -20, -15, -10, -5, 0, 5, 10 dB (native training floor -10 dB) |
| Windows per level | 2,000 (1,000 leak / 1,000 no-leak), same windows for every checkpoint |
| Checkpoints | Model F seeds 0, 1, 2 |
| Evaluation seed | `EVAL_SEED = 0`; leak jitter seeded by `crc32("leak:i")` |
| Primary | AUROC on logits |
| Secondary | detection rate and false-alarm rate at logit ≥ 0 |
| Baseline | RMS over both channels, on the same generated windows |
| Bootstrap | 2,000 resamples, `seed=0`, window-level (`groups = arange(N)`) |

## 3. Tests

93/93 passed, including the 12 tests in `tests/test_snr_sweep_f.py`.

## 4. Results: AUROC [95% CI]

| SNR (dB) | Seed 0 | Seed 1 | Seed 2 | RMS baseline |
|---|---|---|---|---|
| -20 | 0.697 [0.675, 0.720] | 0.691 [0.667, 0.714] | 0.733 [0.712, 0.753] | 0.501 [0.477, 0.525] |
| -15 | 0.758 [0.737, 0.779] | 0.759 [0.738, 0.780] | 0.793 [0.774, 0.811] | 0.502 [0.477, 0.526] |
| -10 | 0.829 [0.811, 0.847] | 0.839 [0.822, 0.857] | 0.856 [0.840, 0.871] | 0.506 [0.481, 0.530] |
| -5 | 0.894 [0.879, 0.908] | 0.898 [0.884, 0.912] | 0.904 [0.890, 0.918] | 0.517 [0.492, 0.542] |
| 0 | 0.919 [0.905, 0.933] | 0.924 [0.911, 0.936] | 0.924 [0.911, 0.937] | 0.537 [0.513, 0.562] |
| 5 | 0.928 [0.915, 0.941] | 0.936 [0.923, 0.948] | 0.934 [0.922, 0.946] | 0.569 [0.544, 0.594] |
| 10 | 0.937 [0.925, 0.949] | 0.946 [0.934, 0.956] | 0.943 [0.930, 0.954] | 0.613 [0.589, 0.638] |

## 5. Detection rate / false-alarm rate at logit ≥ 0

The false-alarm rate does not depend on SNR, because no-leak windows are bit-identical across levels:
seed 0 **0.038** [0.026, 0.051], seed 1 **0.062** [0.047, 0.077], seed 2 **0.059** [0.045, 0.073].

| SNR (dB) | Seed 0 DR | Seed 1 DR | Seed 2 DR |
|---|---|---|---|
| -20 | 0.263 [0.235, 0.290] | 0.284 [0.255, 0.313] | 0.322 [0.292, 0.351] |
| -15 | 0.388 [0.357, 0.417] | 0.398 [0.368, 0.427] | 0.441 [0.409, 0.470] |
| -10 | 0.503 [0.471, 0.533] | 0.532 [0.500, 0.563] | 0.582 [0.552, 0.611] |
| -5 | 0.700 [0.671, 0.727] | 0.730 [0.703, 0.756] | 0.766 [0.739, 0.792] |
| 0 | 0.812 [0.787, 0.836] | 0.824 [0.801, 0.847] | 0.831 [0.808, 0.854] |
| 5 | 0.850 [0.827, 0.872] | 0.854 [0.832, 0.876] | 0.859 [0.837, 0.880] |
| 10 | 0.866 [0.844, 0.887] | 0.871 [0.850, 0.891] | 0.875 [0.854, 0.895] |

The RMS baseline uses the median RMS as its threshold, so its detection and false-alarm rates are
about 0.50/0.50 at -20 dB and 0.63/0.37 at +10 dB (full values in `logs/04_criterion_check.log`).

## 6. Pre-registered success criterion

Model F AUROC CI lower bound > 0.5 **and** > RMS baseline AUROC CI upper bound.

| Seed | -20 dB | -15 dB |
|---|---|---|
| 0 | PASS (0.675 vs RMS 0.525) | PASS (0.737 vs RMS 0.526) |
| 1 | PASS (0.667 vs RMS 0.525) | PASS (0.738 vs RMS 0.526) |
| 2 | PASS (0.712 vs RMS 0.525) | PASS (0.774 vs RMS 0.526) |

**Overall: PASS.** No falsification condition in the pre-registration applies: all three seeds
pass, the effect is present below the native training floor, and no CI overlaps 0.5 or the RMS
baseline.

## 7. Notes and caveats

- **The RMS baseline is weak by construction.** It is scored on the windows after
  `augment_f.finish()`, whose last step (`joint_zscore`) sets every window to unit RMS over both
  channels. The only differences left are from clipping (`CLIP`), so the baseline sits at chance
  at low SNR (0.50) and only reaches 0.61 at +10 dB. In practice, condition 2 of the
  criterion adds little beyond condition 1. This is what the pre-registration specifies ("same
  generated windows"), and it was not changed. The result shows the model does not rely on
  absolute loudness, which z-scoring removes for the model as well. It does not rule out relative
  cues such as the channel RMS ratio, which this baseline does not test.
- AUROC falls steadily as SNR drops (about 0.94 at +10 dB to about 0.70 at -20 dB), with no collapse.
  The drop steepens below 0 dB. The spread between seeds is at most 0.04 at every level.
- Detection at the fixed threshold of 0 is conservative. At -20 dB only 26-32% of leaks are flagged,
  at a 4-6% false-alarm rate.
- The statistical unit is the window, as pre-registered. All 1,000 leak windows per level use
  scenarios 0-999 of the val split (`i % n_scenarios`), all from `Network_7`.
- This is in-domain synthetic evidence only. It does not change R1 or R2 and says nothing about
  transfer to real pipes.

## 8. Technical deviations

- pytest was listed in `requirements.txt` but not installed in `.venv`; it was installed
  (9.1.1) before running the tests. No code changes.
- The pre-registration's Data section says `datasets/NetworkList` is absent; on this machine it is
  present and complete (see section 1). The document was left unedited.
- The pre-registration's status line still reads "NOT yet run". It was left unedited; a
  post-run note is up to the authors.
- Run once. No re-runs and no bug fixes.
