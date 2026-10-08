# V2 cross-rig three-band normalised-PSD experiment: results

**Status: exploratory / hypothesis-generating.** It is not a confirmatory test (see §4).

This document consolidates three sources:

- **Primary experiment:** Phase 7.
  - Pre-specified in `docs/PHASE7_THREE_BAND_PREREG.md` (commit `726e860`).
  - Run once, results in commit `ce423a2`.
- **Pre-specified follow-up:** `docs/PHASE7_ADDENDUM_1_GROUP_ANALYSIS.md` (commit `ed5cf64`, written
  before that analysis was run). It adds the per-group, uncertainty, failure-analysis and within-rig
  outputs.
- **No refit of the primary experiment.** The primary experiment was **not** refit for this document.
  It had already been run once under the identical feature, classifier and split specification.
  Running it again would be a repeat run, so every primary number below comes from the frozen files.

## Artifacts

| What | Where |
|---|---|
| Primary script | `experiments/xrig_three_band.py` |
| Primary metrics and provenance | `results/v2_cross_rig/phase7/three_band_results.json` |
| Raw test scores (3-band, full LR, labels, groups) | `results/v2_cross_rig/phase7/scores_A.npz` (M→S), `scores_B.npz` (S→M) |
| Model F raw test scores (seeds 0–2) | `results/v2_cross_rig/scores_xrig_<train_rig>_foldx_seed{0,1,2}.npz` |
| Group / failure / within-rig script | `experiments/xrig_three_band_groups.py` |
| Per-group table | `results/v2_cross_rig/phase7/group_analysis/per_group.csv` |
| Group, failure and within-rig metrics, provenance | `results/v2_cross_rig/phase7/group_analysis/group_analysis.json` |
| Within-rig raw scores | `results/v2_cross_rig/phase7/group_analysis/within_rig_scores.npz` |
| Per-group figure | `results/v2_cross_rig/phase7/group_analysis/fig_per_group.png` |
| Run logs and records | `group_analysis/run_log.txt`, `results/runs/2026-10-06_231340_v2_xrig_three_band.json`, `results/runs/2026-10-08_215302_v2_xrig_three_band_groups.json` |
| Earlier write-up | `report06/REPORT.md` |

### Provenance

- **Primary run:**
  - Date: 6 Oct 2026.
  - Code: commit `726e860`, clean tree.
- **Group analysis:**
  - Date: 8 Oct 2026, 21:53.
  - Code: commit `ed5cf64`, with the analysis script not yet committed at run time. It is committed
    unchanged together with this document.
- **Environment:**
  - Windows 11.
  - Python 3.12.10, numpy 2.4.2, scipy 1.17.0, scikit-learn 1.8.0.
  - CPU only.
- **Data versions:** pinned by SHA-256 of the window caches.
  - `cache_xrig/mendeley_acc.npz`: `7e4d2d52…7475`.
  - `cache_xrig/sheffield.npz`: `1239cbf8…29ed0`.
  - The caches are built by `experiments/rigs.py` from the Mendeley accelerometer set and the
    Sheffield `All_Data` set, with MD5-deduplicated no-leak files.
- **Seeds:**
  - The logistic regressions are deterministic (lbfgs).
  - Bootstrap: 2000 draws, seed 0.
  - Model F: seeds 0, 1 and 2 (frozen Version 2 checkpoints).

## 1. Research question

Can a simple normalised spectral representation, built from bands that behave the same on both
rigs, improve leak-detection transfer between two independent physical rigs (Mendeley and Sheffield)?

This is a Level-2 test across two laboratory rigs. It is not a test of generalisation to municipal
networks.

## 2. Hypothesis

Rig-dependent loudness and low-frequency waveform cues drive much of the cross-rig failure. The
normalised PSD energy in 400–600, 600–800 and 1600–2000 Hz carries more stable leak information,
so a classifier restricted to those bands should transfer better.

## 3. Why it was proposed

The Version 2 benchmark (`docs/VERSION2_CROSS_RIG_BASELINE.md`) found that Model F collapses in
both directions:

| Direction | Model F window AUROC, mean | Seed range |
|---|---|---|
| M→S | 0.345 | 0.293–0.374 |
| S→M | 0.525 | 0.515–0.535 |

Spectral baselines did better in M→S. The mechanism analysis (`results/v2_cross_rig/shift_analysis.json`
and its post-hoc extension) found:

- **Rig-dependent cues:** log RMS, the low-frequency bands, kurtosis and crest factor.
- **Reversals:** some of these cues reverse their leak direction between the two rigs.
- **Stable bands:** the three bands above appeared direction-stable.

## 4. Caveat: band selection used both rigs

The three bands were chosen after inspecting leak-associated behaviour on **both** rigs, including
each direction's test rig. There is no locked holdout rig that played no part in the selection.

The experiment is therefore **exploratory and hypothesis-generating**. Any improvement is biased
upward by the selection.

## 5. Feature definition (exact)

The features are columns 4, 5 and 8 of `cross_dataset.features_1ch`. This is the same function that
produces the full spectral baseline's features.

- **Input windows:** from `rigs.load_rig`.
  - 5 kHz sampling, 2 kHz low-pass, 2000 samples per window, unit RMS.
  - Each window is then standardised with `cross_dataset.standardise`.
- **PSD:** Welch, `fs=5000`, `nperseg=256`.
- **Normalisation:** each band's power is divided by `tot`, the summed PSD over all Welch bins
  (0–2500 Hz).
- **Feature value:** `log10(sum(Pxx[lo <= f < hi]) / tot + 1e-12)`.
- **Bands:** [400, 600), [600, 800) and [1600, 2000) Hz.
- **Dimension:** 3. The script asserts both the dimension and the band edges.

This is the normalisation the existing baseline and mechanism analysis already use. No other
normalisation was tried.

## 6. Experimental protocol

**Splits.** The Version 2 protocol is unchanged.

- The independence unit is the physical group: one rig condition or recording set.
- Train and validation groups were read from the frozen training records
  (`results/v2_cross_rig/xrig_<rig>_foldx_seed0.json`). They come from the deterministic crc32 rule
  `rigs.val_split`.
- The script asserts that:
  - the stored splits match a recomputation;
  - train and validation are disjoint;
  - train and validation together cover every group of the training rig;
  - the test rig shares no groups with the training rig.

| Direction | Train groups (leak / no-leak) | Val groups (leak / no-leak) | Test groups (leak / no-leak) | Test windows (leak / no-leak) |
|---|---|---|---|---|
| A: Mendeley → Sheffield | 31 (25 / 6) | 9 (7 / 2) | 11 (8 / 3) | 17250 (15150 / 2100) |
| B: Sheffield → Mendeley | 8 (6 / 2) | 3 (2 / 1) | 40 (32 / 8) | 7148 (5722 / 1426) |

- **Mendeley:** the accelerometer set, Looped and Branched topologies.
- **Sheffield:** Sheffield has only **3 independent no-leak groups** in total. As a training rig it
  therefore contributes 2 no-leak groups to training and 1 to validation.

**Classifier.**

- Pipeline: `StandardScaler()` followed by `LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)`.
- This is identical to the existing baseline.
- No search over C, no feature selection, no other classifiers.

**Threshold.** The threshold maximises balanced accuracy on the training rig's validation groups
(`rigs.youden_threshold`). It is frozen before the test rig is loaded.

**Scores.** AUROC uses raw decision values (logits), never probabilities.

**Comparators.**

- **Model F:** the frozen Version 2 results for seeds 0–2, each with its own validation-frozen
  threshold.
- **Full spectral LR:** the same pipeline on all 14 `features_1ch` columns:
  - 9 band fractions;
  - centroid, flatness, kurtosis, crest and zero-crossing rate.

  It was refit in the Phase 7 script and reproduced the frozen Version 2 AUROC exactly (|Δ| = 0)
  in both directions.

## 7. Mendeley → Sheffield results

Prevalence is 0.878 (AUPRC chance level). Window metrics are on the test set. Sensitivity and
specificity use the frozen validation threshold. CIs are 95% stratified cluster-bootstrap CIs over
the 11 Sheffield groups.

| Direction | Method | AUROC [95% CI] | AUPRC | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| M→S | Model F (mean of 3 seeds; range) | 0.345 (0.293–0.374) | 0.851 | 0.349 | 0.482 | 0.216 |
| M→S | Full spectral LR (14) | 0.738 [0.630, 0.939] | 0.951 | 0.671 [0.586, 0.801] | 0.641 | 0.701 |
| M→S | **3-band normalised-PSD LR** | **0.801 [0.687, 0.906]** | 0.959 | 0.615 [0.572, 0.668] | 0.336 | 0.895 |

**Validation AUROC on Mendeley (the training rig):**

| Method | Validation AUROC |
|---|---|
| 3-band | **0.498** (chance) |
| Full 14-feature | 0.599 |

**Group-level results:**

| Method | Group AUROC | Group confusion at the frozen threshold |
|---|---|---|
| 3-band | 1.000 | TP 0, FN 8, TN 3, FP 0 |
| Full 14-feature | 0.958 | TP 7, FN 1, TN 3, FP 0 |

The 3-band model ranks the groups perfectly. Its threshold, however, was learned on a validation
set where the model had no skill, and it sits above every Sheffield group mean (threshold +1.242;
highest group mean +1.147). As a result, no leak group is detected.

## 8. Sheffield → Mendeley results

Prevalence is 0.801. CIs are taken over the 40 Mendeley groups.

| Direction | Method | AUROC [95% CI] | AUPRC | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| S→M | Model F (mean of 3 seeds; range) | 0.525 (0.515–0.535) | 0.824 | 0.470 | 0.856 | 0.084 |
| S→M | Full spectral LR (14) | 0.473 [0.369, 0.574] | 0.810 | 0.524 [0.444, 0.604] | 0.822 | 0.225 |
| S→M | **3-band normalised-PSD LR** | **0.506 [0.355, 0.653]** | 0.788 | 0.514 [0.455, 0.585] | 0.868 | 0.159 |

**Validation AUROC on Sheffield:**

| Method | Validation AUROC |
|---|---|
| 3-band | 0.862 |
| Full 14-feature | 0.883 |

**Group-level results:**

| Method | Group AUROC | Group confusion at the frozen threshold |
|---|---|---|
| 3-band | 0.465 | 7 of 8 no-leak groups flagged as leaks |
| Full 14-feature | 0.430 | — |

All three methods sit in the pre-registered collapse range [0.40, 0.60].

## 9. Comparison against Model F

- **M→S:** both logistic models clearly beat Model F, whose ranking is inverted (0.345).
  - The 3-band model is +0.456 above the Model F mean.
  - Its bootstrap CI does not overlap any Model F seed's CI (Model F seed CIs, upper bounds
    0.446–0.563).
  - However, the full 14-feature LR achieves most of that gain (+0.393).
  - So the gain over Model F comes mainly from **using a small spectral LR instead of Model F**,
    not from restricting it to the three bands.
- **S→M:** there is no material difference. The 3-band model scores 0.506 and Model F 0.525; both
  are at chance.

## 10. Comparison against the full spectral LR (the relevant test of the hypothesis)

Paired cluster bootstrap: the same group resamples score both models.

| Direction | 3-band − full window AUROC | 95% CI |
|---|---|---|
| M→S | +0.064 | [−0.051, +0.122] |
| S→M | +0.033 | [−0.126, +0.182] |

Both point estimates are positive, but both CIs include 0. At the frozen thresholds, the 3-band model
has the **lower** balanced accuracy in both directions (M→S 0.615 vs 0.671; S→M 0.514 vs 0.524).

## 11. Group-level uncertainty and variability

Independent physical groups are the unit of uncertainty:

- **Sheffield:** 11 groups (8 leak, 3 no-leak).
- **Mendeley accelerometer:** 40 groups (32 leak, 8 no-leak).

Every M→S estimate rests on 3 independent no-leak groups. Group AUROC 1.000 in M→S is 24 of 24
leak/no-leak group pairs. The source is `group_analysis.json`.

| Dir | Method | Pooled window AUROC | Group-balanced window AUROC | Leave-one-test-group-out [min, max] | One-group AUROC, leak groups (min / median / max) | One-group AUROC, no-leak groups (min / median / max) |
|---|---|---|---|---|---|---|
| M→S | 3-band | 0.801 | 0.831 | [0.728, 0.887] | 0.72 / 0.83 / 0.87 | 0.70 / 0.89 / 0.89 |
| M→S | Full 14 | 0.738 | 0.819 | [0.693, 0.811] | 0.60 / 0.75 / 0.92 | 0.65 / 0.79 / 0.95 |
| M→S | Model F s0 / s1 / s2 | 0.293 / 0.369 / 0.374 | 0.257 / 0.340 / 0.364 | [0.242, 0.418] | medians 0.30–0.44 | medians 0.27–0.36 |
| S→M | 3-band | 0.506 | 0.505 | [0.454, 0.552] | 0.21 / 0.51 / 0.90 | 0.18 / 0.50 / 0.86 |
| S→M | Full 14 | 0.473 | 0.473 | [0.453, 0.509] | 0.17 / 0.43 / 0.80 | 0.22 / 0.50 / 0.61 |
| S→M | Model F s0 / s1 / s2 | 0.515 / 0.524 / 0.535 | 0.516 / 0.525 / 0.535 | [0.484, 0.553] | medians 0.51–0.54 | medians 0.48–0.52 |

- **M→S:** the 3-band ordering is consistent across Sheffield groups. Every leak group's
  one-group AUROC is at least 0.72.
  - The leave-one-out range is driven by which no-leak group is dropped:
    - dropping NoLeak_4.2bar gives the minimum;
    - dropping NoLeak_2.8bar gives the maximum.
- **S→M:** the per-group results spread from 0.18 to 0.90 and centre on 0.5. Group means overlap
  completely between classes.
  - For example, the Looped no-leak group LO_NL_0.47 scores above most leak groups.
- **Window counts:** group-balanced and pooled AUROCs agree within 0.03 except in M→S. There,
  balancing raises both LR models.

The per-group means, medians and flagged fractions for every method are in `per_group.csv` and are
plotted in `fig_per_group.png`.

## 12. Failure analysis

These analyses are descriptive and use existing data only. Nothing below was used to change the
representation.

**(a) Do the three bands keep their leak direction on both rigs?** Yes.

| Rig | Feature | Group AUROC | Window AUROC |
|---|---|---|---|
| Mendeley | 400–600 | 0.750 | 0.647 |
| Mendeley | 600–800 | 0.719 | 0.674 |
| Mendeley | 1600–2000 | 0.730 | 0.667 |
| Mendeley | log RMS | **0.293** | 0.339 |
| Sheffield | 400–600 | 1.000 | 0.802 |
| Sheffield | 600–800 | 0.833 | 0.577 |
| Sheffield | 1600–2000 | 0.750 | 0.644 |
| Sheffield | log RMS | **0.917** | 0.768 |

- All three band fractions are higher in leaks on both rigs.
- log RMS reverses between the rigs. This confirms that loudness is a rig-dependent cue and that
  the chosen bands are direction-stable **individually**.

**(b) Why does a direction-stable input not give a transferable model?** Look at the fitted
coefficients. These come from a deterministic refit that reproduced the frozen validation AUROC
within 1e-9.

| Direction | Train rig | Standardised coefficient: 400–600 | 600–800 | 1600–2000 |
|---|---|---|---|---|
| A | Mendeley | +0.62 | **−0.57** | +0.90 |
| B | Sheffield | +1.43 | **−1.00** | +0.36 |

In both directions the multivariate fit puts a **negative** weight on 600–800 Hz, even though that
band is higher in leaks on both rigs.

- The three log-fractions are correlated. Logistic regression exploits their *differences*, and
  those differences are rig-specific:
  - On Sheffield, the leak signal is concentrated in 400–600 Hz (leak–no-leak gap 2.96 pooled
    group-SD units).
  - On Mendeley, it is spread evenly across the three bands (gaps 0.84, 1.05, 0.96 SD).
  - So the weight pattern learned on one rig does not encode the other rig's leak contrast.
- The Mendeley-fitted model has no validation skill on Mendeley (0.498).
- Within-rig 3-fold on Mendeley (below) gives AUROC 0.48–0.70.

**The direction-stable marginal behaviour therefore does not make the multivariate representation
invariant.** The M→S score of 0.80 is not a leak signature learned and checked on the training rig.
It is consistent with a fortunate weight direction plus band selection informed by Sheffield.

**(c) Cross-rig shift of the features.** Class-conditional group means move between rigs by up to
1.7 pooled SD:

| Feature | Leak-group shift (S − M) | No-leak-group shift (S − M) |
|---|---|---|
| 400–600 | +1.72 SD | −0.41 SD |
| 600–800 | +0.59 SD | +1.19 SD |
| 1600–2000 | +0.11 SD | +0.54 SD |
| log RMS | +0.93 SD | −1.48 SD |

So the normalised fractions themselves shift between rigs. That breaks a training-rig threshold
(see the M→S group confusion and the S→M specificity of 0.16) even where the ranking survives.

**Which failure mechanisms do the data support?**

| Mechanism | Assessment |
|---|---|
| Insufficient spectral invariance | **Supported.** Fractions shift by 0.1–1.7 SD, and the leak contrast's spectral shape differs between rigs. |
| Remaining rig-specific cues | **Supported.** They enter through the rig-specific multivariate weighting (b), not through any single band. |
| Insufficient group diversity | **Supported.** Sheffield has 3 no-leak groups, so S→M trains on 2 no-leak groups and thresholds on 1. Mendeley validation has 2 no-leak groups. |
| Sensor differences | Plausible but not separable with these data: Mendeley accelerometer vs Sheffield calibrated sensors, with uncalibrated Mendeley units. |
| Recording conditions | Plausible but not separable with these data. |
| Operating regime | Plausible but not separable with these data: different pressures and flows. Each is confounded with rig identity. |

**(d) Secondary within-rig reference** (existing grouped 3-fold split). Values are AUROC per fold.

| Rig | 3-band LR | Full LR (frozen) | Model F (frozen) |
|---|---|---|---|
| Mendeley | 0.699 / 0.637 / 0.479 | 0.615 / 0.700 / 0.448 | 0.698 / 0.753 / 0.792 |
| Sheffield | 0.885 / 0.770 / 0.860 | 0.849 / 0.874 / 0.889 | 0.584 / 0.759 / 0.548 |

- Each Sheffield fold has a single no-leak test group.
- The 3-band representation is weak within Mendeley. It is a reference only, not a claim.

## 13. Interpretation

The hypothesis has two parts.

**Part 1: rig-dependent loudness cues contribute to failure.** This part is supported descriptively:
- log RMS reverses direction between the rigs;
- Model F inverts in M→S.

**Part 2: the selected normalised bands transfer better.** This part is **not supported**:
- **Against the relevant baseline:** the full spectral LR shows a gain in both directions, but each
  gain's CI includes zero.
- **S→M:** the result stays at chance.
- **M→S:** the gain arises from a model with no skill on its own training rig.
- **Selection bias:** the bands were selected using both rigs, which biases the point estimates
  upward.

## 14. Limitations

- **Band selection:** the bands were chosen using both rigs, so this is exploratory.
- **Few independent groups:** 3 Sheffield no-leak groups; 8 Mendeley no-leak groups, of which only
  2 are in validation for direction A.
- **Weighting:** window metrics pool windows within groups. Group-balanced versions are reported
  alongside them.
- **One-sensor streams:** Sheffield streams are one-sensor columns recorded at different times, so
  this measures one-sensor leak recognition, not two-sensor correlation.
- **Confounding:** rig, sensor, pipe material and operating regime are confounded, and only two rigs
  exist.
- **Scope:** no claim about municipal networks, other pipe materials, real buried pipes or
  universal invariance follows from these data.

## 15. Final scientific conclusion

**Classification: C. Null result.**

- **Overall:** the three-band normalised-PSD representation does not materially improve cross-rig
  transfer over the full spectral baseline.
- **S→M:** it is at chance (0.506).
- **M→S:** its non-significant gain (+0.064, CI [−0.051, +0.122]) is not backed by within-rig skill
  on the training rig.
- **Partial-support reading:** one could read B only against Model F in M→S. That gain belongs to
  spectral logistic regression in general, not to the three-band restriction.

The negative result is still informative:

- The individual bands are direction-stable across rigs.
- A multivariate fit on them is not, because the spectral shape of the leak contrast and the
  absolute band fractions both differ between rigs.
- With two rigs and three independent Sheffield no-leak groups, the data cannot separate sensor,
  material and operating-regime causes.

The defensible claim is:

> "We identified rig-dependent acoustic cues associated with poor cross-rig transfer (notably a
> loudness cue that reverses between rigs). Restricting a simple classifier to normalised spectral
> bands that are individually direction-stable did not reliably improve transfer between the two
> evaluated physical rigs."

## 16. Recommendation

**FREEZE AND WRITE**

- **The question is answered.** The pre-specified intervention was run once and evaluated against
  both comparators, with the required uncertainty, per-group and failure analyses. The answer is null.
- **Remaining ideas would be post-hoc.** Other ideas suggested by §12 include:
  - an unweighted band sum;
  - per-rig fraction normalisation;
  - other bands.

  Each of these would be a new representation chosen after seeing both rigs' test results. Each
  would be judged against the same two rigs with the same three Sheffield no-leak groups, so it
  could not produce confirmatory evidence.
- **What a genuine next step needs:** a third, independent rig. That is outside the current scope.
- **The paper:** report the Version 2 collapse, the loudness-reversal mechanism and this null
  invariance result as they stand.
