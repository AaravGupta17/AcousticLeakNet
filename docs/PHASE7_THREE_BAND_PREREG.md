# Phase 7: three-band normalised-PSD intervention (pre-specification)

Written on 6 Oct 2026, **before** any Phase 7 score was computed. Results are reported whatever
they show. Any change made after results are seen gets logged in `docs/INTEGRITY_LOG.md`.

## Status: exploratory, not confirmatory

The three bands were picked by looking at how cues are associated with leaks on **both** rigs,
including each test rig. That was the post-hoc mechanism analysis that followed the Version 2
failure. It is not saved as a file in this repository. It extends the per-rig spectra in
`results/v2_cross_rig/shift_analysis.json`. So this is a hypothesis-driven exploratory
intervention. It is not an independently selected feature set, and any improvement is
optimistically biased by that selection.

## Hypothesis

Cross-rig collapse is partly caused by cues whose leak association reverses between the two
rigs (log RMS, the low-frequency bands, kurtosis and crest). The normalised PSD energy in
400–600, 600–800 and 1600–2000 Hz has the same leak-associated direction on both rigs. A
representation restricted to those three bands may therefore transfer better than one that
also contains the unstable cues.

## Feature definition (fixed)

The features are columns 4, 5 and 8 of the existing spectral baseline's `cross_dataset.features_1ch`,
taken without any change. That baseline is also called the "full spectral/logistic" baseline.
- The input is windows from `rigs.load_rig` (5 kHz, 2 kHz low-pass, 2000 samples, unit RMS).
  Each window is then per-window standardised with `cross_dataset.standardise`.
- Each window's PSD is computed with Welch (`fs=5000`, `nperseg=256`). Its total power `tot` is
  summed over all Welch bins (0–2500 Hz).
- Each feature is `log10(sum(Pxx[lo <= f < hi]) / tot + 1e-12)`, for these bands:
  - [400, 600) Hz
  - [600, 800) Hz
  - [1600, 2000) Hz

The baseline stores these fractions as log10, so the intervention uses the same log10
fractions. Using them as they are means the attribution comparison differs in exactly one way:
which columns are kept. The feature dimension is 3.

The "full" baseline has **14** features, not the 13 sometimes quoted:
- 9 band fractions (edges 10, 50, 100, 200, 400, 600, 800, 1200, 1600, 2000 Hz)
- centroid, flatness, kurtosis, crest and zero-crossing rate

## Classifier (fixed)

`make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced",
max_iter=3000))`. This is identical to the existing baseline (`xrig_eval.fit_baselines`).

## Protocol (unchanged from `docs/VERSION2_CROSS_RIG_PREREG.md`)

- **Directions:**
  - A: train on Mendeley accelerometer (all 40 groups eligible), test on Sheffield (11 groups).
  - B: train on Sheffield, test on Mendeley accelerometer (40 groups).
- **Train and validation groups** are the `train_groups` and `val_groups` stored in the existing
  training records `results/v2_cross_rig/xrig_<rig>_foldx_seed0.json`. They come from the crc32
  `val_split` rule and contain no randomness.
- The **classifier** is fitted on the train groups only.
- The **threshold** maximises balanced accuracy (`rigs.youden_threshold`) on the validation
  groups. It is frozen before the test rig is loaded.
- The **test set** is every group of the other rig.

**Metrics:**
- Primary: window-level AUROC on decision values.
- Secondary: AUPRC (prevalence stated), group-level AUROC, and at the frozen threshold, balanced
  accuracy, sensitivity and specificity.
- Uncertainty: stratified cluster bootstrap over independence groups, 2000 draws, seed 0
  (`metrics.detection_report`).

The comparators are taken from the frozen Version 2 results and are not recomputed for the
headline:
- Model F (mean and range over seeds 0–2)
- the full spectral baseline

The full spectral baseline is also refit in the same script, as a check that it reproduces the
frozen numbers.

**Not done:**
- other bands
- a different C
- other classifiers or preprocessing
- neural networks or DANN
- any threshold chosen on the test rig

If the three-band model clearly improves transfer, the only follow-up is the full (14) vs three-band
attribution comparison. That comparison is already produced by this run.
