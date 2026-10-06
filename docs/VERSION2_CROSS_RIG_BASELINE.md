# Version 2 cross-rig baseline: Mendeley ↔ Sheffield

Protocol: `docs/VERSION2_CROSS_RIG_PREREG.md`, written on 2 Oct 2026 before any model in this
benchmark was trained. Every number below comes from a file in `results/v2_cross_rig/` or
`results/runs/`; the aggregate is `results/v2_cross_rig/summary.json`
(`experiments/xrig_summary.py`).

**Result in one line.** Under the pre-registered rules, the Model F recipe **collapses in both
directions**:
- **Mendeley → Sheffield:** mean window AUROC **0.345**, an inverted ranking.
- **Sheffield → Mendeley:** **0.525**, chance.

A loudness-free spectral logistic regression transfers better than Model F from Mendeley to
Sheffield (0.738). Within each rig, Model F learns something (Mendeley 0.70–0.79 on held-out
groups), so the failure is in transfer, not in learning.

## 1. Scientific question

Does the leak representation that the Model F recipe learns on one physical laboratory rig
transfer to a second, physically independent rig? This is a **Level-2 independent-rig** test. It
is not a test of transfer to buried municipal networks, and nothing here says anything about
them.

## 2. Mendeley rig

Mendeley testbed, accelerometer set (`datasets/Accelerometer/Accelerometer`):

- **Topologies and conditions:** two pipe topologies (Looped, Branched) × five conditions (four
  leak types plus no-leak) × four flow regimes (0.18 LPS, 0.47 LPS, ND, Transient).
- **Recordings:** 40, of which 32 are leak and 8 are no-leak.
- **Sensors:** two accelerometer channels (A1, A2) per recording, about 36 s at 25.6 kHz.
- **Secondary test set:** the hydrophone recordings of the same testbed (60 recordings: 48 leak,
  12 no-leak; 8 kHz).

## 3. Sheffield rig

University of Sheffield CID lab, `All_Data` folder (README checked 2 Oct 2026):

- **Pipe:** 63 mm MDPE at 2.8 or 4.2 bar.
- **Sensors:** two PCB 393B12 accelerometers, 8192 Hz, 30 s per session. One sensor stays at the
  leak; the other is moved to D m. In the CSVs, each column `AccD` is a separate session, and
  the columns are not simultaneous (Acc0 vs Acc1, r = −0.07).
- **Leak data:** 8 leak conditions (test #1–#8: 1/2/3 mm valve, 1 and 3.6 mm direct holes,
  longitudinal and transverse slits) in 14 leak files.
- **No-leak data:** 4 files, of which only **3 are distinct recordings**. Two are byte-identical
  copies by MD5.
- **Excluded folder:** the `Coherence` folder holds leak recordings only, with no no-leak
  counterpart, so it was excluded before training.

## 4. Physical differences between the rigs

| | Mendeley | Sheffield |
|---|---|---|
| Pipe | lab testbed, looped and branched layouts | 63 mm MDPE loop |
| Sensor | accelerometer (same runs also on hydrophones) | calibrated seismic accelerometer, m/s² |
| Native rate | 25.6 kHz | 8192 Hz |
| Operating variable | flow regime (4 levels) | pressure (2.8 / 4.2 bar), leak size and shape |
| Sensor–leak geometry | fixed sensor positions | 0–50 m from the leak |
| Independent no-leak recordings | 8 | 3 |

The raw amplitude units differ (Sheffield is calibrated in m/s²; Mendeley's units are not
documented), so absolute loudness cannot be compared across rigs.

## 5. Data loading

`experiments/rigs.py`:
- **Sheffield:** one CSV at a time, float32, an fs check against 8192 Hz, and no-leak
  deduplication by MD5.
- **Mendeley:** the existing `experiments/mendeley.py` reader with its resample cache.
- **Cache:** windows go to the git-ignored `cache_xrig/` (132 MB for Sheffield, 55 MB for
  Mendeley accelerometers).
- **Not loaded:** Hong Kong, Dongguan and the synthetic cache (`cache_f`). The training script
  structurally loads only its training rig, and a unit test checks this
  (`tests/test_rigs.py`).

## 6. Grouping and independence

- **Stream:** one sensor channel of one recording. Windows are paired only within a stream.
- **Independence group:** used for splits and the bootstrap.
  - **Sheffield leak:** the test condition (8 groups). Both repeat files and every sensor
    position stay together.
  - **Sheffield no-leak:** the distinct recording (3 groups: `NoLeak_4.2bar`, `NoLeak_2.8bar`,
    `noLeak_2.8bar_newer`).
  - **Mendeley:** the recording (40 groups).

| | groups (leak / no-leak) | streams | windows (leak / no-leak) |
|---|---|---|---|
| Mendeley accelerometer | 40 (32 / 8) | 80 | 7148 (5722 / 1426) |
| Sheffield | 11 (8 / 3) | 230 | 17250 (15150 / 2100) |
| Mendeley hydrophone (secondary test) | 60 (48 / 12) | 120 | 11516 (9122 / 2394) |

No dead windows were dropped on any rig.

## 7. Preprocessing (Model F, unchanged)

**Per stream:**
- **Filtering and rate:** demean, resample to 5 kHz (`resample_poly`), then an 8th-order
  Butterworth low-pass at 2 kHz, zero-phase (`public_data.to_common`).
- **Windows:** non-overlapping 2000-sample windows of 0.4 s.
- **Clean-up and storage:** drop dead windows (`bank_f.usable`), scale each window to unit RMS,
  store as float32. The raw log10 RMS is kept for the RMS baseline only.

**Model input:**
- **Pairing:** each window is paired with another window of the same stream through
  `augment_f.pair_channels`, which adds random coherence and lag [F5].
- **Training rows:** random EQ [F4] plus joint z-score with clip 5 [F2].
- **Validation:** the same construction with a fixed seed, as in `train_f.py`.
- **Test:** joint z-score only, with a fixed pairing seed.

**No fitted statistics:** no preprocessing step fits anything to data. The things that are
fitted (weights, logistic regression, RMS sign, thresholds) use the training rig only.

## 8. Memory-safe implementation and hardware

**Hardware:**
- **GPU:** NVIDIA RTX 3060, 12 GB.
- **System RAM:** 32 GB.
- **Software:** torch 2.7.1 + CUDA 11.8.

**Measured usage:**
- **Peak GPU memory:** 1.66 GB (`torch.cuda.max_memory_allocated`, in every training record).
- **Peak RAM during training:** about 4.0 GB. That is the trainer at about 1.2 GB plus 6
  DataLoader workers at about 0.47 GB each, read from the process working sets on 4 Oct.
- **Memory was never the bottleneck.** The earlier out-of-memory failure came from loading the
  3.9 GB synthetic cache, which this path never opens.

**Engineering choices that do not affect results:**
- **Workers:** 6 DataLoader workers, the Model F default. Every training item is drawn from a
  generator seeded by (seed, epoch, index), so the worker count does not change the data.
- **Smoke test:** run before training on both rigs, on GPU and on CPU. On CPU in fp32 the loss
  and all gradients were finite. On GPU, the first AMP steps overflow and are skipped by the
  GradScaler, which is the same behaviour as Model F.

**Wall time:** about 1.6–2.7 h per run when the GPU was not shared (`wall_time_s` in each
training record). Some `wall_time_s` values are inflated by things unrelated to the protocol:
- **Shared GPU:** an unrelated process shared the GPU during part of `sheffield_foldx_seed1`
  (9.25 h).
- **Pauses:** the operator suspended training for 1 h during `mendeley_acc_fold1_seed0` and for
  about 3 h in total during `mendeley_acc_fold2_seed0`. Suspension does not change the computation.

## 9. Train / validation / test construction

**Validation rule:** within each class, sort groups by `crc32(group)`. The first
`ceil(0.2 × n)` groups (at least 1) go to validation.

| direction | train | validation | test |
|---|---|---|---|
| A: Mendeley → Sheffield | 31 Mendeley groups (25 leak / 6 no-leak) | 9 (7 / 2) | all 11 Sheffield groups |
| B: Sheffield → Mendeley | 8 Sheffield groups: tests #2, 3, 4, 6, 7, 8; `NoLeak_2.8bar`, `noLeak_2.8bar_newer` | 3: tests #1, #5; `NoLeak_4.2bar` | all 40 Mendeley groups (+ 60 hydrophone groups, secondary) |

- **What the test rig never influenced:** the test rig was not opened during training,
  validation, checkpoint selection or threshold selection.
- **Validation consequence for direction B:** Sheffield's only 4.2 bar recordings (test #1 and
  its background) fall in validation under this rule. The Sheffield-trained models therefore
  never trained on 4.2 bar data.

## 10. Model F configuration

The recipe is `train_f.py` with `--real-frac 1.0`; the implementation is `Model_F/xrig_train.py`.

**Settings carried over unchanged from Model F:**
- **Architecture:** AcousticLeakNet with base 64, dropout 0.3, CCA fusion.
- **Optimiser:** AdamW, lr 3e-4, weight decay 1e-2, OneCycle (pct_start 0.05), gradient clip
  1.0, AMP.
- **Schedule:** 30 epochs × 1500 steps × batch 256.
- **Loss:** BCE with label smoothing 0.05.
- **Row sampling:** 50/50 leak per row, with a window drawn uniformly within its class.
- **Seeds:** 0, 1 and 2 per direction.

**Only change:** there are no synthetic rows, so the checkpoint is chosen by the **training
rig's validation AUROC** (window level, on logits) instead of mean(synthetic, real). Ties keep
the earlier epoch.

**Checkpoint selected per run:**

| run | best epoch | val AUROC (training rig) |
|---|---|---|
| Mendeley seed 0 / 1 / 2 | 1 / 1 / 1 | 0.832 / 0.872 / 0.925 |
| Sheffield seed 0 / 1 / 2 | 29 / 21 / 17 | 0.936 / 0.939 / 0.940 |

On Mendeley, validation AUROC peaks at epoch 1 and then falls as the training loss reaches its
floor (e.g. seed 1: 0.87 at epoch 1, 0.63–0.71 at epochs 12–19). The model memorises 31 training
recordings quickly.

## 11. Baselines (same splits)

- **RMS:** score = ± log10 RMS of the band-limited window. The sign is fitted on the training
  rig's train groups: −1 on Mendeley (leaks are not louder there), +1 on Sheffield.
- **Spectral logistic regression (logreg):** the loudness-free features of E9/E11 with
  StandardScaler and LogisticRegression (C = 0.5, balanced), fitted on the training rig's train
  groups.
- **Thresholds:** every method's threshold is the score that maximises balanced accuracy on the
  training rig's validation windows, then frozen. Model F is also reported at logit 0.

## 12. Results: Mendeley → Sheffield (direction A)

Test: 17250 windows, 8 leak / 3 no-leak groups, prevalence 0.878.
Sources: `results/v2_cross_rig/eval_xrig_mendeley_acc_foldx_seed{0,1,2}.json`.

| method | window AUROC [95% CI] | group AUROC | AUPRC | bal. acc | sens. | spec. |
|---|---|---|---|---|---|---|
| Model F seed 0 | 0.293 [0.114, 0.446] | 0.125 | 0.833 | 0.312 | 0.38 | 0.24 |
| Model F seed 1 | 0.369 [0.182, 0.555] | 0.250 | 0.858 | 0.390 | 0.48 | 0.30 |
| Model F seed 2 | 0.374 [0.197, 0.563] | 0.292 | 0.861 | 0.345 | 0.59 | 0.10 |
| **Model F mean (range)** | **0.345 (0.293–0.374)** | 0.222 (0.125–0.292) | | | | |
| RMS | 0.232 [0.011, 0.466] | 0.083 | 0.784 | 0.313 | 0.43 | 0.20 |
| Spectral logreg | **0.738** [0.630, 0.939] | 0.958 | 0.951 | 0.671 | 0.64 | 0.70 |

- **Thresholds:** balanced accuracy, sensitivity and specificity are at the frozen validation
  threshold. Window confusion counts for every run are in `summary.json`. At logit 0, Model F's
  balanced accuracy is 0.36–0.38.
- **AUPRC needs care here:** with 88% prevalence, a random scorer already gets about 0.88, so
  Model F's 0.83–0.86 is *below* chance.
- **By sensor distance** (window AUROC, leak windows at that distance vs all no-leak):
  - **Leak at the sensor (0 m):** 0.42 / 0.70 / 0.70 for seeds 0, 1, 2. Not consistent across
    seeds.
  - **1–10 m:** 0.30 / 0.35 / 0.36.
  - **≥15 m:** 0.25 / 0.34 / 0.33.
  - **Spectral logreg:** 0.95, 0.72 and 0.73 for the same three subsets.

## 13. Results: Sheffield → Mendeley (direction B)

Primary test: Mendeley accelerometer, 7148 windows, 32 / 8 groups, prevalence 0.801.
Sources: `results/v2_cross_rig/eval_xrig_sheffield_foldx_seed{0,1,2}.json`.

| method | window AUROC [95% CI] | group AUROC | AUPRC | bal. acc | sens. | spec. |
|---|---|---|---|---|---|---|
| Model F seed 0 | 0.515 [0.432, 0.605] | 0.512 | 0.817 | 0.471 | 0.89 | 0.06 |
| Model F seed 1 | 0.524 [0.414, 0.639] | 0.512 | 0.832 | 0.459 | 0.79 | 0.13 |
| Model F seed 2 | 0.535 [0.455, 0.623] | 0.547 | 0.823 | 0.479 | 0.89 | 0.07 |
| **Model F mean (range)** | **0.525 (0.515–0.535)** | 0.523 (0.512–0.547) | | | | |
| RMS | 0.339 [0.214, 0.459] | 0.293 | 0.725 | 0.464 | 0.13 | 0.80 |
| Spectral logreg | 0.473 [0.369, 0.574] | 0.430 | 0.810 | 0.524 | 0.82 | 0.23 |

- **Threshold transfer fails:** at the frozen threshold, Model F flags nearly everything as a
  leak (specificity 0.06–0.13). At logit 0, specificity is 0.02–0.04.
- **Breakdowns (Model F, seeds 0/1/2):**
  - **Looped-only** (the AGENTS.md clean set): 0.59 / 0.60 / 0.61.
  - **Branched-only:** 0.45 / 0.43 / 0.47.
  - **By leak type:** 0.48–0.61.
- **Secondary, Mendeley hydrophone** (60 groups): Model F 0.564 / 0.564 / 0.580 (group AUROC
  0.65–0.66). RMS 0.638 and logreg 0.526 on the same windows.

## 14. Within-rig references (secondary)

Grouped 3-fold CV, seed 0. Each fold's held-out groups were never seen by that fold's model.
Source: `summary.json`.

| fold | Mendeley: test groups (L/N) | Model F | RMS | logreg | Sheffield: test groups (L/N) | Model F | RMS | logreg |
|---|---|---|---|---|---|---|---|---|
| 0 | 11 / 3 | 0.698 | 0.712 | 0.615 | 3 / 1 | 0.584 | 0.531 | 0.849 |
| 1 | 11 / 3 | 0.753 | 0.660 | 0.700 | 3 / 1 | 0.759 | 0.989 | 0.874 |
| 2 | 10 / 2 | 0.792 | 0.603 | 0.448 | 2 / 1 | 0.548 | 0.789 | 0.889 |
| fold mean | | **0.748** | 0.658 | 0.588 | | **0.630** | 0.770 | 0.871 |

All values are window AUROC.

- **Mendeley:** Model F is the best of the three methods within the rig (fold mean 0.748). Its
  per-fold CIs are wide; fold 0 is [0.42, 0.96].
- **Sheffield:** each fold tests on **one** no-leak group, so these numbers say how one
  background recording ranks against 2–3 leak conditions. Spectral logreg is the strongest
  method there.
- **Pooled out-of-fold scores:** these mix scores from different fold models, so they are
  descriptive only (in `summary.json`).

## 15. Statistical treatment

- **Primary metric:** window AUROC on raw logits or scores, never on sigmoid outputs.
- **Confidence intervals:** 95% intervals come from a cluster bootstrap over independence
  groups, stratified by class (2000 draws, `metrics.cluster_bootstrap`).
- **Group AUROC:** ranks the per-group mean score over all leak × no-leak group pairs.
- **Sheffield has 3 no-leak groups.** Any CI on a Sheffield test resamples those 3 recordings,
  so it is not a meaningful confidence statement. That applies to direction A and to all
  within-Sheffield folds; the within-Sheffield folds rest on a single no-leak group each. The
  Sheffield conclusions rest on the consistency across seeds and methods, not on the intervals.
- **Mendeley (40 groups) is better but still small:** 8 no-leak recordings.
- **Seeds:** the three seeds per direction measure training variability only, not data
  variability.

## 16. Failure analysis (pre-specified)

Source: `results/v2_cross_rig/shift_analysis.json`, `plots/v2_xrig_shift.png`.

1. **The leak signatures sit in different places.**
   - **Sheffield:** relative to background, leaks add energy at about 150–650 Hz (maximum
     contrast at 566 Hz). The quiet Sheffield backgrounds carry narrow peaks near 1080 and
     1500 Hz, so there the contrast is negative.
   - **Mendeley:** leaks add broadband energy above about 600 Hz (maximum at 1504 Hz).
   - **The two rigs' leak − no-leak contrast curves are slightly anti-correlated** (Pearson
     −0.18, Spearman −0.13). Near 1500 Hz they point in opposite directions. A cue learned on
     one rig is wrong, not just weak, on the other.
2. **The rigs are near-perfectly separable from the same loudness-free features** (rig-ID
   out-of-fold AUROC 0.982), while the label inside each rig is only partly separable
   (out-of-fold 0.827 Sheffield, 0.662 Mendeley). The rig identity is a much stronger signal in
   these features than the leak.
3. **Loudness points the opposite way on the two rigs.**
   - **Sheffield:** leak windows are louder than background (median log10 RMS: leak groups −3.08
     to −1.68, no-leak −3.17 to −2.73).
   - **Mendeley:** they are not (Looped leak groups are often quieter than Looped no-leak).
   - **Consequence for the RMS baseline:** it inverts in both directions (0.23, 0.34).
   - **Consequence for Model F:** its inputs are z-scored, yet the Mendeley-trained score
     correlates *negatively* with loudness on Sheffield (Spearman −0.24 to −0.31 pooled). The
     Sheffield-trained score correlates positively with loudness and spectral centroid on
     Mendeley (+0.14 to +0.33). Each model has absorbed its training rig's loudness/spectral
     relation, which inverts on the other rig.

**Interpretation, not separately tested:** these three observations are enough to explain an
inverted transfer in direction A and a chance-level transfer in direction B.

## 17. Interpretation

**Pre-registered outcome** (rules fixed before training; `summary.json`):
- **A:** AF = 0.345 → **Collapse, inverted ranking.** The Case-D flag is raised: spectral logreg
  0.738 ≥ Model F − 0.05.
- **B:** AB = 0.525 → **Collapse.** No Case-D flag; the best baseline (0.473) is not better than
  Model F.
- **Overall: Case C, cross-rig collapse,** with a Case-D finding in direction A.

**What this means:**
- **The current representation does not survive this independent-rig shift.** The Model F
  recipe, trained on one rig's real labelled windows, learns that rig's leak/background contrast
  (within-Mendeley 0.70–0.79). That contrast does not carry over.
- **Simple acoustic features are not better in general.** The loudness-free spectral classifier
  transfers from Mendeley to Sheffield (0.738) but not back (0.473).
- **The original hypothesis is not disproved.** This is one pair of rigs, one sensor-pairing
  scheme, and tiny numbers of independent no-leak recordings.

## 18. Limitations

- **Very few independent negatives:** 3 Sheffield no-leak recordings, 8 Mendeley. Sheffield CIs
  are not interpretable as confidence statements.
- **One-sensor recognition only:** Model F's pairing turns one-sensor windows into two channels.
  This benchmark tests single-sensor leak recognition, not the two-sensor timing idea, because
  Sheffield's `All_Data` columns are not simultaneous.
- **Label confounds inside Sheffield:** pressure (4.2 bar appears in only one leak condition and
  one background) and sensor position (leak files reach 50 m, backgrounds only 20 m) are partly
  tied to the label. They are reported, not removed.
- **Unchanged Model F hyperparameters:** 30 × 1500 steps on a few thousand windows lets the
  model memorise its training recordings. Checkpoint selection on validation groups handles
  this, but on Mendeley it picks epoch 1 in every seed.
- **Seed 0 of direction A was trained twice.** The first attempt was killed by a 2 h tool time
  limit before any evaluation; it was rerun unchanged (`docs/INTEGRITY_LOG.md`).
- **Code provenance:** the run records carry commit `4b20090`, but the benchmark code was
  uncommitted (untracked) while the runs were made. Commit the code together with this document so
  that commit identifies the code that produced these results.

## 19. Recommended Version-2 direction (based on this evidence only)

The evidence points at **rig-specific spectral colouration and loudness relations**, not at
model capacity:
- **Capacity is not the problem:** Model F learns within a rig.
- **The cues invert across rigs:** the leak-vs-background contrast and the loudness direction
  differ or flip between them.
- **What Version 2 needs:** invariance to the sensor and pipe transfer function, and to the
  background's own spectral structure.

Concretely, the next experiments should test whether a representation that does not depend on
absolute spectral shape survives this same frozen split:
- **Per-recording spectral whitening** against that rig's own background, or relative and
  differential features.
- **Training on more than one rig.**

The split, metrics and baselines here should be reused unchanged as the yardstick. Within-rig
numbers should not be used as the target; the Sheffield ones rest on 3 no-leak recordings.

## Files

- **Code:** `experiments/rigs.py`, `Model_F/xrig_train.py`, `experiments/xrig_eval.py`,
  `experiments/xrig_shift.py`, `experiments/xrig_summary.py`. Tests: `tests/test_rigs.py`,
  `tests/test_xrig_shift.py`, `tests/test_xrig_summary.py`.
- **Results:** `results/v2_cross_rig/`:
  - `xrig_*.json`: training records.
  - `eval_*.json`: evaluations.
  - `scores_*.npz`: per-window scores.
  - `summary.json`: the aggregate.
  - `shift_analysis.json`: the failure analysis.
- **Run records:** `results/runs/*v2_xrig*`.
- **Figure:** `plots/v2_xrig_shift.png`.
- **Checkpoints:** `models/xrig_*.pt` (12).
- **Logs:** `logs_v2/`.
