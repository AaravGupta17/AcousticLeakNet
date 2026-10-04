# Version 2 cross-rig baseline: pre-registration

Written on 2 Oct 2026, **before** any model in this benchmark was trained and before any
cross-rig score was computed. The splits, rules and outcome classes below are fixed. Results are
reported whatever they show. Any change after results are seen gets logged in
`docs/INTEGRITY_LOG.md` under a new name.

The physical-domain audit documents named in the handoff (`VERSION2_PHYSICAL_DOMAIN_AUDIT.md`,
`VERSION2_CRITICAL_REVIEW.md`) are not in this repository. Every physical fact used here was
checked against the raw data and the dataset README instead (see "Data facts checked").

## Question

Does the Model F recipe learn a leak representation that transfers between two physically
independent laboratory rigs? This is a **Level-2 independent-rig transfer** test. It is not a
test of transfer to municipal networks.

- **A:** train on Mendeley, test on Sheffield.
- **B:** train on Sheffield, test on Mendeley.

## Data facts checked (2 Oct 2026, from the raw files)

**Sheffield** (`datasets/public/sheffield/extracted/Acoustic Data (Leakage Experiments)/All_Data`):

- 8192 Hz, 30 s per column. Two PCB 393B12 accelerometers (README). The CSVs are "combined":
  column `AccD` is a separate 30 s session with the moving sensor at D metres, so the columns
  are not simultaneous. For example, Acc0 and Acc1 of `NoLeak_2.8bar` have r = −0.07.
- There are 8 leak conditions (test #1–#8) and 14 leak files (#3 and #4 have one file each,
  the others two).
- There are 4 no-leak files but only **3 distinct recordings**. By MD5,
  `(test#5)/NoLeak_2.8bar.csv` duplicates `(test#2)/NoLeak_2.8bar.csv`, and
  `(test#4)/noLeak.csv` duplicates `(test#3)/noLeak_2.8bar_newer.csv`. The three that remain
  are not copies of each other (same-time r ≤ 0.04).
- The `Coherence (simultaneous recording)` folder holds **leak recordings only**. It has no
  no-leak data, so including it would tie the label to the recording session. **It is
  excluded.**

**Mendeley accelerometer** (`datasets/Accelerometer/Accelerometer`):

- 25.6 kHz, about 36 s, two channels (A1/A2) per recording.
- 40 recordings = 2 topologies × 5 conditions × 4 flow regimes, of which 32 are leak and 8 are
  no-leak.
- No synthetic data and no earlier checkpoint is used here. The Branched contamination (only
  relevant to synthetic training, INTEGRITY_LOG #7) therefore does not apply, and all 40
  recordings are eligible. A Looped-only breakdown is still reported (AGENTS.md).

## Units

- **Stream** = one sensor channel of one recording. That is one Sheffield CSV column, or one
  Mendeley A1/A2 channel. Windows come from streams.
- **Independence group** (splits and bootstrap) = the physical condition.
  - **Sheffield leak:** the test number. There are 8 groups; both repeat files and every sensor
    position stay together.
  - **Sheffield no-leak:** the distinct background recording, deduplicated by content. There
    are 3 groups: `NoLeak_4.2bar`, `NoLeak_2.8bar`, `noLeak_2.8bar_newer`.
  - **Mendeley:** the recording (both channels together). There are 40 groups: 32 leak and
    8 no-leak.
- Windows are never treated as independent samples.

## Preprocessing (Model F, unchanged)

Per stream:
1. Subtract the mean.
2. Resample to 5 kHz (`resample_poly`).
3. Apply an 8th-order Butterworth 2 kHz zero-phase low-pass (`public_data.to_common`).
4. Cut non-overlapping 2000-sample windows (0.4 s).
5. Drop dead windows with `bank_f.usable`.
6. Scale each window to unit RMS and store it as float32. The raw log-RMS is kept for the RMS
   baseline only.

For the model input:
- A window is paired with a second window of the **same stream** via `augment_f.pair_channels`,
  which adds random coherence and lag [F5]. This is how Model F turns one-sensor real data into
  two channels. Both rigs go through the same path, so the benchmark measures one-sensor leak
  recognition, not two-sensor timing (see Limitations).
- **Training rows:** `augment_f.finish` = random EQ [F4] + joint z-score (clip 5) [F2].
- **Validation:** the same, with a fixed seed. This is `train_f.py`'s real-validation
  construction.
- **Test:** joint z-score only (no EQ) and a fixed pairing seed. Every test window is scored.

None of these steps fits any statistic to data. The only fitted objects in this benchmark are
the network weights, the logistic-regression scaler and weights, the RMS sign, and the
thresholds. All of them are fitted on training-rig data only. The training script never opens a
file of the test rig.

## Splits

Training-rig validation split: within each class, sort the groups by `crc32(group)`. The first
`ceil(0.2 × n_class)` groups (at least 1) go to validation.
- **Mendeley:** 2 of 8 no-leak and 7 of 32 leak groups go to validation.
- **Sheffield:** 1 of 3 no-leak and 2 of 8 leak groups go to validation.

The test set is **every** eligible group of the other rig.

## Model F

The recipe is `Model_F/train_f.py` with `--real-frac 1.0`. Settings, unchanged from Model F:
- Architecture `build_model` with base 64, dropout 0.3, fusion cca.
- AdamW, lr 3e-4, weight decay 1e-2. OneCycle with pct_start 0.05. Gradient clip 1.0.
- AMP on CUDA, as in Model F.
- 30 epochs × 1500 steps × batch 256.
- BCE with label smoothing 0.05. Each row is 50/50 leak/no-leak, with a window drawn uniformly
  from the class pool.

The position and flow loss terms are inactive because there are no synthetic rows.

**Only change:** there are no synthetic rows, so the synthetic cache is never opened, and the
checkpoint score is the **training-rig validation AUROC** (window-level, on logits) instead of
mean(synthetic, real). Ties keep the earlier epoch.

**Seeds:** 0, 1 and 2 for each direction. All seeds are reported. The headline is the mean
across seeds with the per-seed range.

## Baselines (same splits)

1. **RMS:** score = ± log10 RMS of the band-limited window. The sign is fitted on the training
   rig's train groups.
2. **Spectral logistic regression:** the loudness-free features of E9/E11
   (`cross_dataset.features_1ch` on standardised windows) with StandardScaler and
   LogisticRegression(C=0.5, balanced). Fitted on the training rig's train groups.

## Metrics

Primary metric: **window-level AUROC on the test rig** (logits or raw scores).

Also reported:
- AUPRC, with the test prevalence stated.
- **Group-level AUROC:** the mean score per independence group, ranked over groups.
- Balanced accuracy, sensitivity, specificity and the confusion matrix (windows and groups).
- These threshold metrics are reported at two thresholds: (a) the score maximising balanced
  accuracy on the training rig's validation windows, frozen; (b) logit 0 for Model F, an
  a-priori choice because training is class-balanced.

Uncertainty: a cluster bootstrap over independence groups, stratified by class (2000 draws,
`metrics.cluster_bootstrap`), plus the number of groups per class. Sheffield has only
**3 no-leak groups**, so its CIs cannot be meaningful. They are reported, but no inference rests
on them.

Secondary breakdowns (descriptive only, never used to select anything):
- Sheffield by sensor distance: 0 m, 1–10 m and ≥15 m leak windows, each against all no-leak
  windows.
- Mendeley by topology (including Looped-only) and by leak type.
- Direction B also scored on the Mendeley **hydrophone** set, as a sensor-modality shift.

## Within-rig references (secondary)

Grouped 3-fold CV within each rig, stratified by class. Groups of each class are sorted by
crc32 and dealt to folds round-robin. In each fold, the remaining groups are split train/val by
the rule above. The fold uses Model F (seed 0) and both baselines, with the same checkpoint and
threshold rules. Folds are reported separately and pooled out-of-fold. These references are
never used to tune the cross-rig runs.

## Outcome classes (fixed now)

Let AF and AB be Model F's mean window AUROC for directions A and B, and Base the better
baseline's AUROC in the same direction.

- **Collapse (C):** AF or AB lies in [0.40, 0.60], or below 0.40 (inverted ranking, reported as
  such).
- **Strong (A):** AF ≥ 0.80 and AB ≥ 0.80, and Model F beats Base by ≥ 0.05 in both directions.
- **Moderate/asymmetric (B):** anything else with both directions > 0.60.
- **Baselines match (D):** an additional flag, raised in any direction where Base ≥ Model F − 0.05.

The class applies per direction where the directions differ. "Collapse" means the current
representation does not survive this rig shift. It does not mean the project hypothesis is
disproved.

## Failure analysis (pre-specified, limited)

1. Per-rig mean normalised PSD (0–2 kHz) for each class, and each rig's leak-minus-no-leak
   spectral contrast. The question is whether the leak signature sits in the same bands on
   both rigs.
2. Rig separability: logistic regression on the same features predicting the rig, compared with
   how separable the label is within each rig.
3. Whether the transferred Model F score on the test rig tracks RMS or spectral centroid
   (Spearman rank correlation, per class).
