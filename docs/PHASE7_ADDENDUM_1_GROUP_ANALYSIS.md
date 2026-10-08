# Phase 7 addendum 1: group-level and failure analysis (pre-specification)

Written on 8 Oct 2026. The Phase 7 primary results (`results/v2_cross_rig/phase7/three_band_results.json`,
commit `ce423a2`) have already been seen. This addendum therefore pre-specifies only **descriptive**
analyses of those frozen results. It does not change the hypothesis, features, classifier, splits,
thresholds or the outcome classification in `docs/PHASE7_THREE_BAND_PREREG.md`.

## What is not done

- The primary experiment is **not refit or rerun**. Every primary metric comes from the frozen
  Phase 7 and Version 2 files.
- No new bands, normalisations, C values, classifiers or thresholds.
- No result from this addendum is used to choose anything.

## Analyses (all from saved scores or existing data)

1. **Per-test-group results** for each direction and each method:
   - Methods: 3-band LR, full 14-feature LR, Model F seeds 0–2.
   - Scores come from `phase7/scores_{A,B}.npz` and `scores_xrig_<rig>_foldx_seed{0,1,2}.npz`.
   - Per group:
     - number of windows
     - mean and median decision value
     - fraction of windows at or above the frozen validation threshold
     - "one-group AUROC": that group's windows against all windows of the opposite class
2. **Test-group variability:**
   - Leave-one-test-group-out window AUROC (minimum and maximum over the dropped group).
   - Spread of the one-group AUROCs.
3. **Window-count sanity check:** a group-balanced window AUROC, where each window is weighted
   1 / (that group's window count), compared with the pooled window AUROC.
4. **Failure analysis on the features** (existing caches, no test-label tuning):
   - (a) Per rig, the univariate group-level AUROC of each of the 3 band features and of log RMS.
     A group is scored by its mean feature value. This checks whether each band's leak direction
     is the same on both rigs.
   - (b) The fitted 3-band LR coefficients in each direction.
     - This uses a deterministic refit of the identical Phase 7 pipeline on the identical train
       groups.
     - The refit is asserted to reproduce the frozen validation AUROC within 1e-9. It is not a
       new model.
   - (c) Shift in the class-conditional group means of the 3 features between rigs, in units of
     pooled within-rig group SD.
5. **Optional within-rig reference (secondary):**
   - 3-band LR on the existing grouped 3-fold within-rig split (`rigs.fold_assign`).
   - Same recipe, with train/val from each existing fold record (`xrig_<rig>_fold{0,1,2}_seed0.json`).
   - The full-LR within-rig numbers are read from the frozen fold evaluations.
   - This is a reference only. It is not a primary claim.

Group-level CIs are the existing stratified cluster bootstrap (2000 draws, seed 0) where they
are computed. The independent unit is the physical group:
- Mendeley accelerometer: 40 groups (32 leak / 8 no-leak).
- Sheffield: 11 groups (8 leak / 3 no-leak).
