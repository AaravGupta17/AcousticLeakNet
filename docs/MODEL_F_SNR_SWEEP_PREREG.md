# Model F SNR robustness sweep: pre-registration

Status: **built and validated, NOT yet run.** Written before any result from
`Model_F/snr_sweep_f.py` exists. Uses only already-trained checkpoints
(`best_model_f_seed{0,1,2}.pt`, committed at `1551414`). No new training.
Does not change `docs/MODEL_F_PREREG.md`, its addendum, H0-H4, R1, or R2.

## Why this experiment

R1 and R2 already establish that Model F's real-data behaviour does not
support a transferable leak-specific detection claim. This experiment does
not attempt to answer that question again, and a positive result here would
**not** change that conclusion. It answers a narrower, in-domain question
that nothing in the project has tested for Model F specifically: having been
redesigned to remove a synthetic noise-texture shortcut (H0/H1), did Model F
retain genuine, non-loudness discrimination of the synthetic leak signal
itself, and how far does that hold as SNR degrades? `experiments/snr_sweep.py`
(E3) answered this for the superseded original model; it has never been run
on Model F, and cannot be pointed at a Model F checkpoint as-is, because it
feeds raw `dataset_c.LeakDataset` output (Model C's fixed-scale convention)
directly into `predict_proba` with no normalization step. Model F requires
its own pipeline (`Model_F/augment_f.py`: 2 kHz band limit, joint z-score,
random EQ). Using E3 unmodified on a Model F checkpoint would silently
mis-evaluate it, not test it.

## Hypothesis

**Not simply "AUROC > 0.5."** Model F maintains AUROC on synthetic leak
detection that (a) stays meaningfully above the RMS-loudness baseline and
(b) does not collapse to the baseline's level, specifically at SNR levels
**below Model F's native training floor** (`SNR_DB_MIN_E = -10 dB` in
`Model_E/dataset_e.py`) -- the regime checkpoint selection had no
opportunity to reward, since checkpoint selection's own synthetic-validation
metric (`auc_syn` in `Model_F/train_f.py`) is computed at the native,
non-overridden SNR distribution, not at a fixed out-of-range override.

## Data, exactly

- Leak scenarios: `leak_status == 1` rows from `data/csv/val_sampled.csv`
  (Model F's own synthetic validation split), loaded via
  `Model_E/dataset_e.LeakDatasetE`, exactly as `Model_F/pregen_f.py` already
  does for training. **Blocked in the current environment**: `_build_cache`
  reports 0/91,896 rows cached because `datasets/NetworkList` (the raw
  EPANET per-network CSVs each scenario's `file_path` points to) is not
  present locally -- it is large, git-ignored, local-only data per
  `AGENTS.md`. `Model_F/snr_sweep_f.py` now fails loudly with an explicit
  message naming this cause (`load_val_leak_scenarios`'s assertion) instead
  of the bare `ZeroDivisionError` it originally produced. **This blocks
  running the sweep, not building or validating it** -- `cache_f/val/leak.npy`
  itself exists locally (pre-generated at the *native* SNR distribution when
  the raw data was available), but leak components at an *overridden* SNR
  cannot be produced without regenerating from the raw CSVs.
- Backgrounds/interferers/EQ: `Model_F/train_f.py`'s `Bank`/`background()`
  and `Model_F/augment_f.py`'s `interferers()`/`band_limit()`/`finish()`,
  loaded from `cache_f/bank.npz` as committed (`val=True` slice, matching
  R2's convention of using the held-out real-source rows for any evaluation
  that isn't training).
- Checkpoints: `best_model_f_seed0.pt`, `best_model_f_seed1.pt`,
  `best_model_f_seed2.pt`. No retraining.

## Method

For window index `i` in `[0, N)`, `i < N/2` is a leak window:

1. **Nuisance variables** (background, interferers, random EQ, channel
   pairing): seeded from `np.random.default_rng([EVAL_SEED=0, i])` alone.
   Identical across every SNR level and every checkpoint for a given `i` --
   verified directly (`tests/test_snr_sweep_f.py::test_no_leak_windows_are_bit_identical_across_snr_levels`,
   `::test_leak_window_background_is_independent_of_snr`).
2. **Leak physics jitter** (wave speed, attenuation, position noise inside
   `dataset_e.py`'s `_leak()`, which uses the *global* `np.random` state, not
   a passed `Generator`): seeded via `np.random.seed(crc32(f"leak:{i}"))`
   immediately before generation, depending on `i` alone, never on the SNR
   level -- verified directly
   (`::test_leak_window_physics_jitter_is_independent_of_snr`, which confirms
   the RMS ratio between two SNR levels of the *same* window matches the
   predicted `10**(ΔdB/20)` factor to within 2%, i.e. the jitter truly did
   not change).
3. **Leak component**: `Model_F/pregen_f.leak_component(ds, scenario)` with
   `ds.snr_override_db` set to the grid level, in the documented
   background-RMS units. Underlying scenario assigned by `i % n_scenarios`,
   fixed across SNR levels.
4. **Assembly**, identical to `Model_F/train_f.py`'s `MixDataset` synthetic
   branch: `x = background + band_limit(interferers [+ leak]); x = finish(x, rng)`.
5. **Baseline**: RMS over both channels (`experiments/features.py:rms`
   convention), same generated windows, same labels.
6. Run all three checkpoints on the identical `2N`-window set per SNR level
   (construction does not depend on the checkpoint).

## Preprocessing validation (Phase 2) -- already run, all 12 checks pass

`tests/test_snr_sweep_f.py`, synthetic fixtures only (no raw data, no
checkpoints needed):

| # | Check | Result |
|---|---|---|
| 1 | Window shape `(2, 2000)` | PASS |
| 2 | Labels exactly balanced, identical across SNR levels | PASS |
| 3 | No-leak windows bit-identical across SNR levels | PASS |
| 4 | Leak-window background/EQ independent of SNR (leak zeroed via monkeypatch) | PASS |
| 5 | Leak physics jitter independent of SNR (RMS ratio matches `10**(ΔdB/20)` to 2%) | PASS |
| 6 | `snr_override_db` scaling law exact across the full grid (2% tolerance) | PASS |
| 7 | `snr_override_db` ignores `torricelli_amp` (confirms override truly bypasses the native-SNR formula) | PASS |
| 8 | Received amplitude *does* still depend on distance/material (documented, not a bug -- see below) | PASS |
| 9 | Output band-limited to 2 kHz (using a band-limited bank fixture, matching real `bank.npz`'s own construction) | PASS |
| 10 | Output is joint-z-scored (mean ~0, std in `[0.5, 1.5]`), not Model C's fixed ~0.1-RMS convention | PASS |
| 11 | `main()` contains the Model-F-checkpoint guard (`cfg["input_norm"] == "zscore"`) before scoring any checkpoint | PASS |
| 12 | Background/interferer code path never branches on label | PASS |

**One non-obvious, documented property (check #8), not a bug**:
`snr_override_db` sets the leak's *source* amplitude before propagation, so
two scenarios at the same override but different distance/material still
receive different amplitude at the sensor (attenuation is applied after the
SNR-based source amplitude, exactly as `dataset_e.py`'s `_leak()` has always
worked for every prior model in this project). This means the SNR grid
controls the *distribution* of received amplitude at each nominal level, not
an exact received value -- consistent with how E3 and H0/H1 already use this
same quantity, so it does not need a design change, only this explicit
disclosure.

## Frozen protocol

- **Primary metric**: AUROC on logits (never sigmoid probabilities).
- **SNR grid** (fixed before running, matches E3's own grid for
  comparability): -20, -15, -10, -5, 0, 5, 10 dB. -10 dB is Model F's native
  training floor; -15 and -20 dB are the out-of-selection-distribution
  points the primary hypothesis rests on.
- **Sample size**: 2000 windows per level (1000 leak / 1000 no-leak),
  matching `experiments/snr_sweep.py`'s own `--n` default.
- **Seeds**: all three of `best_model_f_seed{0,1,2}.pt`. No seed is dropped
  or chosen after seeing results.
- **Randomness**: `EVAL_SEED = 0` for nuisance variables, shared across every
  checkpoint and SNR level (see Method, step 1); leak physics jitter seeded
  per-window index (step 2). Both fixed and recorded in
  `Model_F/snr_sweep_f.py`.
- **Statistical unit**: the window. Unlike real Mendeley/Hong Kong/Dongguan
  data, each synthetic window here is an independently generated sample (own
  random background, own random leak draw) rather than a fragment of one
  shared recording, so window-level bootstrapping is appropriate --
  consistent with how `experiments/snr_sweep.py` (E3) already treats
  synthetic rows, and unlike every real-data analysis in this project
  (E4, R1, R2), which groups by recording/site because that grouping
  assumption would be false there.
- **Confidence intervals**: `experiments/metrics.py:detection_report`'s
  cluster bootstrap (2000 resamples, `seed=0`), with `groups = arange(N)`
  (equivalent to a plain window bootstrap given the point above).
- **Secondary metrics**: detection rate and false-alarm rate at the
  pre-registered threshold (logit ≥ 0), reported alongside AUROC, not
  headlined.

## Success criterion (fixed before running)

Both of the following, for **all three seeds**, at **both** -15 dB and
-20 dB:
1. Model F's AUROC CI lower bound exceeds 0.5 (above-chance discrimination
   below its own native training range).
2. Model F's AUROC CI lower bound exceeds the RMS baseline's AUROC CI upper
   bound (a reproducible advantage over loudness alone, not just "not
   chance").

No numeric threshold was chosen to guarantee this passes; both conditions
are the same structure R1/R2 already use (CI-lower-bound comparisons), and
the -15/-20 dB requirement specifically targets the two grid points outside
checkpoint selection's own distribution.

## Falsification (fixed before running)

Any of: Model F's CI overlapping the RMS baseline's at -15/-20 dB in any
seed; Model F's CI overlapping 0.5 at those levels in any seed; the effect
present in only one or two of three seeds; or the effect present only at
SNR levels inside the native training range (0, 5, 10 dB), which would
indicate the result reflects the checkpoint-selection-adjacent distribution
rather than genuine low-SNR robustness.

## Stopping rule

Run once, at the grid and checkpoints above. No re-running with a different
grid, threshold, or seed subset if the first result disappoints. If a
script or data-loading bug is found before any result is inspected, it may
be fixed and the run repeated once; any such fix is logged here.

## What a positive result would and would not mean

Would mean: Model F, evaluated entirely in-domain on synthetic data, retains
non-loudness leak discrimination as SNR degrades toward and below its
training range, and the H0/H1 shortcut-removal did not come at the cost of
genuine discriminative power on the synthetic task it was trained for.

Would **not** mean: anything about real-world transfer. R1 (seed-unstable
Mendeley hydrophone result) and R2 (confounded held-out-group separability)
are unaffected either way and are not superseded by this experiment. A
positive result here is a new Results entry alongside R1/R2, not a
replacement for either.

## Known limitations, stated in advance

- Grid points inside the native training range (0, 5, 10 dB) are not fully
  independent of checkpoint selection, since selection's own synthetic
  metric is computed on a similar (though not identical -- distributional,
  not fixed-override) SNR draw from the same range. This is why the primary
  success criterion rests on -15/-20 dB specifically, not the whole curve.
- Received leak amplitude still depends on scenario distance/material even
  at a fixed SNR override (documented above); this is intrinsic to how SNR
  has always been defined in this codebase's synthetic pipeline, not
  specific to this experiment.
- Requires `datasets/NetworkList` to be present locally to run; it is not
  present in the current environment (see Data section). Someone with that
  data (or the means to regenerate it) must run this, or it must be
  fetched/rebuilt first.
