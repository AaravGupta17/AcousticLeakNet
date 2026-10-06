# Phase 7: three-band normalised-PSD cross-rig intervention

- **Pre-specification:** `docs/PHASE7_THREE_BAND_PREREG.md`. Committed as `726e860` before any
  Phase 7 score was computed.
- **Script:** `experiments/xrig_three_band.py`
- **Raw results:** `results/v2_cross_rig/phase7/three_band_results.json`
- **Raw scores:** `results/v2_cross_rig/phase7/scores_A.npz` (M→S) and `scores_B.npz` (S→M)
- **Run record:** `results/runs/2026-10-06_231340_v2_xrig_three_band.json`

The Version 2 cross-rig results (`results/v2_cross_rig/*`, apart from `phase7/`) are frozen. They
were read but not modified.

## Status: exploratory

The three bands were chosen by a mechanism analysis that looked at both rigs, including the
rig each direction is tested on. This is therefore a hypothesis-driven **exploratory**
intervention, not an independently selected confirmatory feature set. Any gain it shows is
optimistically biased by that selection.

## Environment and data

- **Machine:** Windows 11, NVIDIA RTX 3060 (12 GB), 32 GB RAM. The whole run used the CPU
  (scikit-learn) and took about 60 s. The GPU was recorded but not needed.
- **Software:** Python 3.12.10, numpy 2.4.2, scipy 1.17.0, scikit-learn 1.8.0, torch 2.7.1+cu118
  (not used).
- **Code version:** run at commit `726e860` on branch `phase7-three-band-xrig`, which branches
  from `224ca5d` (the Version 2 baseline). The run JSON stores the SHA-256 of both window caches
  (`cache_xrig/mendeley_acc.npz`, `cache_xrig/sheffield.npz`).
- **Mendeley:** the accelerometer set with **both Looped and Branched** topologies. It has
  **40 groups** (32 leak, 8 no-leak) and 7148 windows. This is the original audited 40-group
  protocol.
- **Sheffield:** the `All_Data` set with MD5-deduplicated no-leak files. It has **11 groups**
  (8 leak, 3 no-leak) and 17250 windows. This is the original audited protocol.

## Feature and classifier

- **Features:** the log10 normalised PSD energy fraction in each of three bands: [400, 600),
  [600, 800) and [1600, 2000) Hz.
  - Each window is per-window standardised, then its PSD is estimated with Welch
    (fs = 5 kHz, nperseg = 256).
  - Each band's fraction is taken relative to the total power over 0–2500 Hz.
  - These are columns 4, 5 and 8 of the existing `cross_dataset.features_1ch`, used as they are.
  - The feature dimension is 3. The script asserts both the dimension and the band edges.
- **Classifier:** StandardScaler followed by LogisticRegression(C=0.5, class_weight="balanced",
  max_iter=3000). This is identical to the existing spectral baseline.
- **Full spectral baseline:** the same pipeline on all **14** columns of `features_1ch`. These
  are 9 band fractions plus centroid, flatness, kurtosis, crest and zero-crossing rate. The
  handoff called this the 13-feature baseline, but the code has 14 features.

## Protocol

The protocol is unchanged from `docs/VERSION2_CROSS_RIG_PREREG.md`.

- **Train and validation groups** are read from the frozen training records
  (`xrig_<rig>_foldx_seed0.json`).
  - The script recomputes them with `rigs.val_split` and asserts they are identical.
  - It asserts that train ∪ val is every group of the training rig and that train and val are
    disjoint.
- The **classifier** is fitted on the training groups only.
- The **threshold** maximises balanced accuracy on the validation groups and is frozen.
- The **test rig is loaded only after** the classifier and threshold are fixed.
  - The script asserts that the test rig differs from the training rig and that the two share
    no groups.
  - Test labels are used only to compute metrics.
- **Directions** were run one after the other, and the arrays were released between them.
- **Uncertainty:** cluster bootstrap over independence groups, stratified by class
  (2000 draws, seed 0).
  - Paired difference between the three-band and full models: the same group resamples score
    both models.
- **Reproduction check:** the refit 14-feature baseline reproduced the frozen Version 2
  `logreg` AUROC exactly in both directions (|Δ| = 0).

## Results

Window AUROC is the primary metric. AUROC is computed on raw decision values or logits. The
threshold-based metrics (balanced accuracy, sensitivity, specificity) use the
validation-frozen threshold. Group AUROC ranks the mean score of each group. Model F is shown
as the mean over seeds 0–2 with the [min, max] range.

### A. Mendeley → Sheffield

The test set has 11 groups (8 leak, 3 no-leak) and 17250 windows (15150 leak, 2100 no-leak).
Prevalence is 0.878.

| Method | Window AUROC [95% CI] | Group AUROC | AUPRC | Bal. acc | Sens | Spec | Threshold (val) |
|---|---|---|---|---|---|---|---|
| **3-band LR** | **0.801 [0.687, 0.906]** | 1.000 | 0.959 | 0.615 | 0.336 | 0.895 | +1.242 |
| Full 14-feature LR | 0.738 [0.630, 0.939] | 0.958 | 0.951 | 0.671 | 0.641 | 0.701 | +1.347 |
| Model F (3 seeds) | 0.345 [0.293, 0.374] | 0.222 | 0.851 | 0.349 | 0.482 | 0.216 | per seed |

- **Paired difference** in window AUROC (3-band − full): **+0.064**, with 95% CI
  [−0.051, +0.122].
- **Validation AUROC on Mendeley** (the training rig): 3-band **0.498**, full 0.599.
- **3-band group confusion at the frozen threshold:** TP 0, FN 8, TN 3, FP 0.
  - The group ranking is perfect, but the threshold sits above every group mean.
  - The threshold came from a validation set on which the 3-band model was at chance.
- **Descriptive breakdowns** (window AUROC by sensor distance):

  | Distance | 3-band | Full |
  |---|---|---|
  | 0 m | 0.972 | 0.947 |
  | 1–10 m | 0.760 | 0.718 |
  | ≥15 m | 0.872 | 0.732 |

### B. Sheffield → Mendeley

The test set has 40 groups (32 leak, 8 no-leak) and 7148 windows (5722 leak, 1426 no-leak).
Prevalence is 0.801.

| Method | Window AUROC [95% CI] | Group AUROC | AUPRC | Bal. acc | Sens | Spec | Threshold (val) |
|---|---|---|---|---|---|---|---|
| **3-band LR** | **0.506 [0.355, 0.653]** | 0.465 | 0.788 | 0.514 | 0.868 | 0.159 | −1.166 |
| Full 14-feature LR | 0.473 [0.369, 0.574] | 0.430 | 0.810 | 0.524 | 0.822 | 0.225 | −2.764 |
| Model F (3 seeds) | 0.525 [0.515, 0.535] | 0.523 | 0.824 | 0.470 | 0.856 | 0.084 | per seed |

- **Paired difference** in window AUROC (3-band − full): **+0.033**, with 95% CI
  [−0.126, +0.182].
- **Validation AUROC on Sheffield** (the training rig): 3-band 0.862, full 0.883.
- **Descriptive breakdowns:**
  - By topology: Looped-only 3-band 0.391 vs full 0.336; Branched 0.584 vs 0.650.
  - By leak type: 3-band ranges from 0.38 to 0.58.

## Interpretation

Against the outcome classes in the task brief, the result is **partial and partly unexpected**
(B/D). It is not a strong improvement (A).

- **M→S:** the point estimate rises from 0.738 to 0.801.
  - The paired CI for the gain includes 0, so the improvement is not established.
  - With only 3 Sheffield no-leak groups, every CI in this direction is weak.
- **S→M:** 0.506 is still in the pre-registered collapse range [0.40, 0.60]. Removing the
  unstable cues does not recover transfer in this direction.
- **Mismatch with the hypothesis in direction A:**
  - The 3-band model does **not** separate leaks within Mendeley (validation AUROC 0.498), yet
    it ranks Sheffield windows at 0.80.
  - So the M→S gain does not come from a leak signature learned and checked on the training
    rig. The weights fitted on Mendeley happen to point in a direction that works on Sheffield.
  - That is consistent with the bands having been chosen partly by looking at Sheffield.
  - It is not evidence of a transferable representation.
- **Consequence for the frozen threshold:** because the validation AUROC was at chance, the
  threshold is uninformative. On Sheffield this gives sensitivity 0.34 at specificity 0.90, and
  all 8 leak groups fall below the threshold.
- **Comparison with Model F:**
  - Both logistic models clearly beat Model F in M→S, where Model F's ranking is inverted
    (0.345).
  - In S→M, all three methods are at chance level.
- **Hypothesis support:** the hypothesis that restricting to direction-consistent bands
  recovers cross-rig transfer is **not supported** as a general claim. What remains is a
  non-significant point improvement in one direction, which the selection bias described above
  can explain.

## Attribution

The pre-approved attribution comparison (full 14-feature vs 3-band) was produced in the same
run. It is the paired difference reported above. No other ablation, band search, change to C,
classifier change or threshold change was run.

## Limitations

- The bands were selected using both rigs, including each test rig.
- There are only 3 Sheffield no-leak groups, so the direction A CIs (and Sheffield's role as a
  training rig in direction B) rest on very few independent negatives.
- The Sheffield streams are one-sensor columns that were not recorded simultaneously. Like the
  Version 2 benchmark, this measures one-sensor leak recognition.
- Window AUROC pools windows inside groups. The CIs come from a group bootstrap, but the point
  estimates are weighted by how many windows each group has.
- Group AUROC 1.000 in direction A rests on 8 × 3 = 24 group pairs.
- None of this is evidence that the model generalises to real buried pipes.

## Protocol violations

None found.
- All assertions passed.
- Test data was loaded only after thresholds were frozen.
- The frozen baseline was reproduced exactly.
- The script ran once, with no code fixes and no reruns.
- No setting was changed after the results were seen.
