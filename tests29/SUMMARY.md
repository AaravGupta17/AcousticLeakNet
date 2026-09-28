# tests29: Model F (pre-registered) and E9 Looped-only rerun

Runs from the repo root **without** `LEAKNET_OUT` (real results). Default hyperparameters, seed 0,
`--workers` left at the default (6-core CPU). Results are judged against `docs/MODEL_F_PREREG.md`
as written; nothing there was changed after seeing results.

Code: commit `2dbc6ae` (clean tree for all training runs). GPU: RTX 3060, torch 2.7.1+cu118.

## Contents

| Path | What |
|---|---|
| `logs/02_bank_f.log` | `python Model_F/bank_f.py` |
| `logs/03_pregen_f.log` | `python Model_F/pregen_f.py` (default `--split train val`) |
| `logs/04_audit_f.log` | `python Model_F/audit_f.py` (H0 gate) |
| `logs/05_train_f.log` | `python Model_F/train_f.py` |
| `logs/06_train_f_nohk.log` | `python Model_F/train_f.py --exclude-source hongkong --prefix f_nohk` |
| `logs/07_train_f_nodg.log` | `python Model_F/train_f.py --exclude-source dongguan --prefix f_nodg` |
| `logs/08_train_f_synonly.log` | `python Model_F/train_f.py --real-frac 0 --prefix f_synonly` (**incomplete**, see caveats) |
| `logs/09_model_f_eval_crashed.log` | first E11 attempt, killed by a GPU driver fault (kept for the record) |
| `logs/11_model_f_eval.log` | E11: `python experiments/model_f_eval.py --ckpt best_model_f_seed0.pt best_model_f_nohk_seed0.pt best_model_f_nodg_seed0.pt best_model_f_synonly_seed0.pt` |
| `logs/10a_cross_dataset.log` | E9: `python experiments/cross_dataset.py` (Looped-only) |
| `logs/10b_cross_dataset_model_d.log` | E9: `python experiments/cross_dataset.py --ckpt best_model_d.pt` |
| `runs/` | copies of the `results/runs/` JSON records for these runs |

Checkpoints (Git LFS, in `models/`): `best_model_f_seed0.pt`, `best_model_f_nohk_seed0.pt`,
`best_model_f_nodg_seed0.pt`, `best_model_f_synonly_seed0.pt`.

## 1. Training data (`bank_f.py`, `pregen_f.py`)

| Source | Windows | Groups | Leak |
|---|---|---|---|
| hk_noiselogger | 2175 | 32 | 46% |
| hk_hydrophone | 2000 | 18 | 50% |
| dongguan | 1772 | 293 | 56% |
| Mendeley Branched no-leak (backgrounds only) | 632 two-sensor windows | 6 recordings | 0% |

Saved to `cache_f/bank.npz`; train and val caches written to `cache_f/train`, `cache_f/val`.

## 2. H0 gate: shortcut audit (`audit_f.py`)

Record: `runs/2026-09-25_234449_f_shortcut_audit.json`

| Rows | rms | peak | ch. RMS ratio | ch. correlation | kurtosis | spectral logreg (5-fold, reference) |
|---|---|---|---|---|---|---|
| synthetic | 0.621 | 0.507 | 0.564 | 0.548 | 0.505 | 0.673 |
| real | 0.597 | 0.610 | 0.607 | 0.528 | **0.668** | 0.857 |

**H0 passes**: every trivial feature on synthetic rows is < 0.65 (max: rms 0.621), which is the
pre-registered rule. Note: on *real* rows kurtosis is 0.668 (≥ 0.65). This is outside the H0 rule
(which covers synthetic rows only), but it was flagged and training went ahead on explicit approval.

## 3. Training

Checkpoint chosen by score = mean of synthetic and real validation AUROC (held-out groups of the
training sources).

| Model | Real data from | Best epoch | Val AUROC at best (synthetic / real) | Record |
|---|---|---|---|---|
| `f` | Hong Kong, Dongguan | 10 | 0.941 / 0.940 | `runs/2026-09-26_044252_train_f.json` |
| `f_nohk` | Dongguan | 3 | 0.897 / 0.962 | `runs/2026-09-26_064018_train_f_nohk.json` |
| `f_nodg` | Hong Kong | 5 | 0.946 / 0.946 | `runs/2026-09-26_083643_train_f_nodg.json` |
| `f_synonly` | none (real backgrounds only) | 11 | 0.950 / n/a | **no record** (log only: `logs/08_train_f_synonly.log`) |

## 4. E11: Model F evaluation

Record: `runs/2026-09-29_000514_e11_model_f_eval.json`. AUROC on logits, 95% CIs bootstrapped
over recordings. Mendeley is Looped-only. Baseline = logreg trained on the same real sources.
Detect / false alarm are at the model's default threshold.

| Model | Test | Model F AUROC [95% CI] | Detect | False alarm | Logreg AUROC [95% CI] |
|---|---|---|---|---|---|
| `f` | mendeley_looped_acc | 0.424 [0.193, 0.680] | 0.70 | 0.82 | 0.295 [0.050, 0.627] |
| `f` | mendeley_looped_hyd | 0.630 [0.500, 0.762] | 0.06 | 0.03 | 0.188 [0.106, 0.288] |
| `f_nohk` | mendeley_looped_acc | 0.348 [0.162, 0.558] | 0.91 | 0.96 | 0.189 [0.058, 0.362] |
| `f_nohk` | mendeley_looped_hyd | 0.197 [0.105, 0.304] | 0.01 | 0.01 | 0.428 [0.243, 0.599] |
| `f_nohk` | hk_noiselogger (held out) | 0.389 [0.292, 0.596] | 0.91 | 0.94 | 0.699 [0.471, 0.838] |
| `f_nohk` | hk_hydrophone (held out) | 0.631 [0.486, 0.774] | 0.29 | 0.07 | 0.375 [0.166, 0.599] |
| `f_nodg` | mendeley_looped_acc | 0.273 [0.047, 0.557] | 0.49 | 0.77 | 0.263 [0.039, 0.565] |
| `f_nodg` | mendeley_looped_hyd | 0.704 [0.526, 0.859] | 0.00 | 0.01 | 0.113 [0.039, 0.204] |
| `f_nodg` | dongguan (held out) | 0.612 [0.543, 0.684] | 0.81 | 0.75 | 0.777 [0.724, 0.822] |
| `f_synonly` | mendeley_looped_acc | 0.315 [0.104, 0.544] | 0.53 | 0.77 | 0.295 [0.050, 0.627] |
| `f_synonly` | mendeley_looped_hyd | 0.396 [0.256, 0.526] | 0.08 | 0.08 | 0.188 [0.106, 0.288] |

Noise flagged as leak (band-limited white / coloured):

| Model | White | Coloured |
|---|---|---|
| `f` | 0% | 0% |
| `f_nohk` | 0% | 0% |
| `f_nodg` | 1% | 0% |
| `f_synonly` | 0% | 0% |

(Models C/D: 100%, `tests24/runs/2026-09-25_000006_e7_texture_probe.json`.)

The public-dataset tests pair two windows of one single-sensor recording, so they measure
single-sensor sound recognition, not the two-sensor timing idea. Only Mendeley tests both sensors.

## 5. Pre-registered hypotheses

| | Rule (from `docs/MODEL_F_PREREG.md`) | Result | Verdict |
|---|---|---|---|
| H0 | every trivial feature on synthetic rows < 0.65 | max 0.621 (rms) | **Pass** |
| H1 | `f` flags < 20% of white and coloured noise | 0% / 0% | **Pass** |
| H2 | `f` on both Looped sensors: CI lower bound > 0.5 **and** AUROC > logreg | acc: 0.424, lower 0.193 (fails); hyd: 0.630 > 0.188 but lower bound 0.500 is not > 0.5 (fails) | **Fail** (not even partial) |
| H3 | Model F beats same-source logreg in ≥ 2 of 3 held-out tests | hk_noiselogger lose (0.389 vs 0.699), hk_hydrophone win (0.631 vs 0.375), dongguan lose (0.612 vs 0.777): 1 of 3 | **Fail** |
| H4 | exploratory: `f` > `f_synonly` on both Looped sensors | acc 0.424 vs 0.315, hyd 0.630 vs 0.396 | Higher on both (CIs overlap; no pass rule) |

Reading: the noise shortcut is gone (H1), but the added realism did not produce a reliable transfer
to a new real network (H2) or to held-out public sources (H3). Several Model F AUROCs are below 0.5
(`f_nohk` on Looped hydrophone 0.197 is below chance with its whole CI). At the default threshold
the models are poorly calibrated on real data: they flag nearly everything (acc) or nearly nothing
(hyd). This does not show generalisation to real pipes.

## 6. E9: cross-dataset rerun (Looped-only Mendeley)

_In progress when this file was written; the logs are in `logs/10a_cross_dataset.log` and
`logs/10b_cross_dataset_model_d.log`, records in `runs/`._

## Integrity caveats

- **`f_synonly` is incomplete.** The PC shut down unexpectedly (no bugcheck) during epoch 30 of 30.
  The log ends at epoch 29 and no run record was written. The saved checkpoint is the best epoch so
  far (epoch 11, synthetic val AUROC 0.950); epochs 12–29 all scored lower on synthetic val, but
  epoch 30 was never evaluated. It has no real validation windows (`--real-frac 0`), so real val
  AUROC is `nan` and selection used the synthetic score. It was not re-run.
- **First E11 attempt crashed** (exit 127) from an NVIDIA driver fault (nvlddmkm event 153) partway
  through `f_nohk`. Its partial log is kept as `logs/09_model_f_eval_crashed.log`. The rerun
  (`logs/11_model_f_eval.log`) reproduced every number the crashed run had reached.
- **Epoch timing in `05_train_f.log`:** epoch 12 shows 10801 s because training was paused
  (process suspended) overnight during that epoch. It does not reflect compute time; results are
  unaffected.
- **Real-row kurtosis 0.668** in the audit (see §2): above 0.65 but outside the H0 rule.
- One seed per model; GPU non-determinism alone moves AUROC by about 0.01 (INTEGRITY_LOG #22).
  Mendeley has only 4–6 no-leak recordings per topology and sensor, so its CIs are wide.
