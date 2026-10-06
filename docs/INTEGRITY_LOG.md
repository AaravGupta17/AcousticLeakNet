# Integrity log

A record of errors found in this project and how each was corrected. Keep adding to it. A
dated list of self-found mistakes shows judges that the evidence has been checked.

## 23 Sep 2026: pre-submission audit

| # | Problem found | Correction |
|---|---|---|
| 1 | `model_C/evaluate_c.py` (and the archived `evalnew.py`) labelled the test networks "Anytown / Richmond / Kentucky". Networks 6 and 8 were **swapped**: `scripts/inptocsv.py` builds Network 6 from KY15 and Network 8 from Richmond (44 pipes, 1,056 test samples). | Labels fixed in the scripts. The `"network"` names in `results/test_results_c.json` and `results/test_results.json` were corrected in place. The numbers are unchanged; each result was already stored under the correct split. |
| 2 | The documentation said L-TOWN is in the Czech Republic. | Corrected to Limassol, Cyprus (BattLeDIM). |
| 3 | The `model_C/mendely_exp_2.py` docstring listed "AUROC ~0.79+" as the result for Mendeley-matched retraining. That was a pre-run expectation; the actual result is 0.515 (`results/experiment2_results.json`). | Docstring now states the actual result. |
| 4 | The model docstrings said the input was 8 kHz × 8000 samples with 9 scalars. The pipeline uses 5 kHz × 2000 samples with 11 scalars. | Corrected. |
| 5 | The cross-channel attention was described as "equivalent to cross-correlation TDOA". The gate is time-constant (squeeze-and-excitation style). | Docstrings rewritten. Added `tests/test_model.py::test_gate_is_constant_over_time` and a `fusion="concat"` ablation option. |
| 6 | Real-data evaluation z-scored every window (RMS 1.0), while training inputs used a fixed scale (leak-free RMS ≈ 0.1). E1 shows the Model C/D checkpoints score leak-free noise with RMS ≥ 0.05 as a leak. The `mendely_exp_2.py` docstring wrongly said its evaluation matched training. | Documented. `experiments/mendeley_eval.py` evaluates both scalings on a clean test set. |
| 7 | Mendeley Branched no-leak hydrophone recordings were used as training noise and were also part of the real-data test. | New evaluations flag all Branched no-leak recordings as contaminated and use Looped-only as the clean test. |
| 8 | `model_C/mendely_eval.py` printed automatic "conclusions" (e.g. "flow scale mismatch is dominant") based only on AUROC thresholds. | Removed. |
| 9 | Identifying information (school name, city) in `docs/script.md` and `docs/TO_DO.md`. This is not allowed in IRIS submissions. | Removed from `script.md`. `TO_DO.md` moved to the git-ignored `private/` folder. **Note: older git commits still contain it** (see below). |
| 10 | `docs/research_paper.pdf` was an empty (0-byte) file. | Removed. Commit the real paper when it's final. |
| 11 | Scripts still pointed at the old `../csv/` and `../inp/` folders after the move to `data/`. | Paths updated. |
| 12 | AUROC was computed on sigmoid probabilities. The checkpoints output such large logits that many float32 probabilities are exactly 1.0 (or 0.0). Tied scores push AUROC toward 0.5, so earlier real-data AUROCs (0.501, 0.515) may understate how well the model ranks windows. | experiments/ compute AUROC on logits; E4 also reports the old-style AUROC and the saturated fraction, so the size of the effect is measured. Test: `tests/test_metrics.py::test_saturated_sigmoid_hides_ranking_but_logits_do_not`. |

## 23 Sep 2026: after the E1–E5 runs

| # | Problem found | Correction |
|---|---|---|
| 13 | Hypothesis in #6 (scaling mismatch causes the real-data false alarms) was **tested and rejected** by E4: matched scaling reduced false alarms only from 100% to 88–91%. | README and CLAIMS updated. #6 stays as a record of the preprocessing mismatch, not as the explanation. |
| 14 | Model C's synthesiser adds 60% of the leak signal to both channels with **zero delay** (`dataset_c.py`, `CORRELATION_ALPHA`). In noise-free synthetic leaks, cross-correlation lag vs true TDOA gives r = 0.00 (r = 1.00 without that component). The synthetic data never contained a usable time-delay cue, so the model's localisation cannot be timing-based, and GCC-PHAT (MAE ≈ 0.25) was structurally unable to work. | Documented; test `tests/test_dataset_e.py::test_model_c_synthetic_leak_has_no_recoverable_tdoa`. Model E uses physical fractional delays. Claims about TDOA-based localisation removed. |
| 15 | `np.roll` delays in Model C/D wrap the end of the window back to the start. | Model E uses a linear fractional delay (`tests/test_dataset_e.py::test_fractional_delay_is_linear_not_circular`). |
| 16 | Leak rows with no valid leak distance were labelled "leak" but synthesised as pure noise. | Model E drops them (`test_leak_rows_without_source_are_dropped`). E3 reports how many exist per network. |
| 17 | Rows whose synthesis failed are stored with label −1, and `train_c.py` / `evaluate_c.py` would train and evaluate on that −1 target. | Both now skip rows with label < 0 and print how many were skipped. |
| 18 | `pregen_c.py` never seeded the random generator, so caches can't be regenerated identically. | `Model_E/pregen_e.py` seeds per split. Model C's existing caches are unchanged. |

## 25 Sep 2026: after the tests24/tests25 runs

| # | Problem found | Correction |
|---|---|---|
| 19 | E9 (`cross_dataset.py`) used all Mendeley recordings, including the Branched no-leak files that the synthetic checkpoints saw as background noise (#7). The `cnn_pre` and `probe` rows on Mendeley in `tests25/` are therefore contaminated; `logreg`, `cnn_scratch` and the Hong Kong/Dongguan rows are not. | E9 now keeps only Looped Mendeley recordings, for every method (dropping only Branched no-leak would make "Branched" mean "leak"). `--mendeley-topology all` reproduces the old behaviour. Test: `tests/test_public_data.py::test_mendeley_branched_is_dropped_but_other_datasets_kept`. E9 to be rerun. |
| 20 | A relative `LEAKNET_OUT` broke after scripts that `os.chdir()` (E8 crashed in tests24). | Resolved to an absolute path on import; `tests/test_common.py`. |
| 21 | The anonymity scanner (`scripts/make_submission_copy.py`) and its test contained the school and city names they were meant to catch, plus both student names. | Those terms now live in the git-ignored `private/forbidden_terms.txt`; the test uses made-up names and checks the script names no place. |
| 22 | Same seed, same data: `cnn_scratch` never uses the checkpoint, yet its AUROCs differ by up to ≈0.01 between the two full E9 runs in `tests25/runs/` (GPU non-determinism). | Not a bug, but differences of this size are treated as noise, not findings. |

## 2–4 Oct 2026: Version 2 cross-rig baseline (`docs/VERSION2_CROSS_RIG_BASELINE.md`)

| # | Problem found | Correction |
|---|---|---|
| 23 | The first training run of `xrig_mendeley_acc_foldx_seed0` was killed by a 2-hour time limit in the tool that launched it. It left only an epoch-1 checkpoint and no training record, and it was never evaluated. | Rerun with an identical configuration (`--overwrite`), after the other runs. Only the rerun is reported. No result had been seen when the rerun was queued, so this is not a selection. |
| 24 | The handoff named two audit documents (`VERSION2_PHYSICAL_DOMAIN_AUDIT.md`, `VERSION2_CRITICAL_REVIEW.md`) as the benchmark definition. Neither exists in this repository. | Every physical fact was checked against the raw data instead, and the protocol was written down before training (`docs/VERSION2_CROSS_RIG_PREREG.md`). One check changed the data: Sheffield has 3 distinct no-leak recordings, not 4, because two files are byte-identical copies. |
| 25 | The cross-rig runs were made while the benchmark code was untracked, so the commit hash in each run record (`4b20090`) does not identify that code. | Commit the code together with the results, so the commit that adds `results/v2_cross_rig/` identifies the code version. Not committed yet (awaiting the go-ahead). |

## 5–6 Oct 2026: Phase 7 band intervention (`science/phase7/PHASE7_AUDIT.md`)

| # | Problem found | Correction |
|---|---|---|
| 26 | The Mendeley Branched accelerometer recordings are missing on the machine used for Phase 7, and no cache holds them. Phase 7 therefore uses 20 Looped Mendeley groups, not the 40 of the cross-rig baseline. | Stated in the audit's Addendum 1 before the run. All Phase 7 comparisons are made within one run on identical data. None is compared with the 40-group baseline numbers. |
| 27 | A first launch of `xrig_band_intervention.py` on 5 Oct left an empty log and no output. No score was produced or seen. | Recorded here. The single counted run is the one made after Addendum 1 was committed. |
| 28 | The three Phase 7 bands were chosen using leak labels from both rigs' full data (`xrig_mechanism.py`), which are also the Phase 7 test sets. | Phase 7 is labelled exploratory. Only the no-refit Dongguan/Hong Kong scoring bears on the decision, under rules fixed in Addendum 1. |

## Still open

- `L-TOWN.inp` is not committed (it's over 100 MB). The README says where to get it.
- Git history still contains the school name (older versions of `docs/script.md` and `TO_DO.md`). If the repository link goes into the IRIS submission, publish a fresh copy without that history, or keep the repo private.
- No LICENSE chosen yet.
- Material parameters, partly resolved by E10 (Sheffield MDPE, measured). *At the leak* the spectrum (centroid ~770 Hz) roughly matches Model C's PVC band (650–950 Hz), so the earlier note that the values "look inverted" was only partly right. The real error is **attenuation**: measured 0.6–2.5 dB/m against ≈0.004 dB/m in the synthesiser. Model E now uses the measured curve for PVC (`Model_E/calibration_mdpe.json`, traceable to the E10 run record); metals are unchanged. Caveat: MDPE is softer than PVC, so the measured loss may overstate PVC attenuation.
