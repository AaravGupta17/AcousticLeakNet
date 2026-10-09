# V2 cross-rig experiments: final interpretation

- **Inspected:** branch `phase7-three-band-xrig`, commit `e7cbae9`, clean tree.
- **Task:** analysis only. No model was trained, no threshold or feature was chosen, and no
  frozen result was changed.

**Authoritative sources:**

| Short name | Path |
|---|---|
| V2-BASE | `docs/VERSION2_CROSS_RIG_BASELINE.md`, `results/v2_cross_rig/summary.json`, `eval_xrig_*_foldx_seed{0,1,2}.json`, `shift_analysis.json` |
| P7 | `docs/V2_THREE_BAND_CROSS_RIG_RESULTS.md` (identical to `oct8/REPORT.md`, checked with `diff`), `results/v2_cross_rig/phase7/three_band_results.json` |
| P7-GRP | `results/v2_cross_rig/phase7/group_analysis/group_analysis.json`, `per_group.csv` (= `oct8/per_group.md`), `fig_per_group.png` (= `oct8/fig_per_group.png`) |
| CHK | `results/v2_cross_rig/phase7/interpretation_checks.json`, from `experiments/xrig_interpretation_checks.py` |

CHK is **new for this document and descriptive only**. It reads saved scores, fits nothing and
selects nothing. It was computed after all results were known.

## 1. Executive conclusion

**The Model F recipe does not transfer between the two independent physical rigs.**
- From Mendeley to Sheffield it is inverted (window AUROC 0.345).
- From Sheffield to Mendeley it is at chance (0.525).

**A simple spectral logistic regression transfers in one direction only.**
- Mendeley → Sheffield: 0.738.
- Sheffield → Mendeley: 0.473.

**Restricting that classifier to three normalised PSD bands does not establish an improvement.**
- Its paired AUROC gains over the full spectral model are small, and both CIs include 0.
- Sheffield → Mendeley stays at chance.
- In Mendeley → Sheffield, the 3-band model had no skill on its own training rig (validation
  AUROC 0.498), so its 0.801 cannot be read as a transferable leak signature.

**What is established:**
- A strong, direction-dependent cross-rig domain shift.
- One rig-dependent cue that is directly measured: loudness reverses its leak association
  between the rigs.
- A null result for the band-restriction hypothesis.

**What is not established:**
- Why the shift happens. Sensor, pipe, operating regime and recording conditions are all
  confounded with rig identity.

## 2. Exact results across both directions

Window AUROC is computed on raw decision values. The threshold-based metrics use the threshold
frozen on the training rig's validation groups.

| Direction (test prevalence) | Method | AUROC [95% group CI] | AUPRC | Bal. acc | Sens | Spec | Group AUROC | Train-rig val AUROC |
|---|---|---|---|---|---|---|---|---|
| M→S (0.878) | Model F, mean of 3 seeds (range) | 0.345 (0.293–0.374) | 0.851 | 0.349 | 0.482 | 0.216 | 0.222 | — |
| M→S | Full spectral LR (14 features) | 0.738 [0.630, 0.939] | 0.951 | 0.671 | 0.641 | 0.701 | 0.958 | 0.599 |
| M→S | 3-band LR | 0.801 [0.687, 0.906] | 0.959 | 0.615 | 0.336 | 0.895 | 1.000 | **0.498** |
| S→M (0.801) | Model F, mean of 3 seeds (range) | 0.525 (0.515–0.535) | 0.824 | 0.470 | 0.856 | 0.084 | 0.523 | — |
| S→M | Full spectral LR | 0.473 [0.369, 0.574] | 0.810 | 0.524 | 0.822 | 0.225 | 0.430 | 0.883 |
| S→M | 3-band LR | 0.506 [0.355, 0.653] | 0.788 | 0.514 | 0.868 | 0.159 | 0.465 | 0.862 |

Sources: P7 §7–8 and `three_band_results.json` (`directions.{A,B}`). The Model F rows are the
V2-BASE tables in §12–13.

**Paired 3-band − full difference in window AUROC** (cluster bootstrap, P7 §10):

| Direction | Difference | 95% CI |
|---|---|---|
| M→S | +0.064 | [−0.051, +0.122] |
| S→M | +0.033 | [−0.126, +0.182] |

**Answers to question A:**
- **AUROC:** the 3-band model's point estimate is higher than the full model's in both
  directions, but neither gain is supported by its CI.
- **AUPRC:** compare against prevalence, which is what a random scorer achieves.
  - **M→S:** both logistic models (0.95–0.96) are above prevalence (0.878). Model F (0.851) is
    below it.
  - **S→M:** all three methods are at or near prevalence (0.801): 3-band 0.788 (below), full 0.810
    and Model F 0.824 (slightly above).
  - The 3-band model is not better than the full model on AUPRC in S→M.
- **Threshold-based metrics:** the 3-band model has the *lower* balanced accuracy in both
  directions. In S→M, every method flags most no-leak windows as leaks (specificity 0.08–0.23).
- **Consistency across directions:** none. Both logistic models work only in M→S, and Model F
  works in neither direction.
- **Verdict on the 3-band intervention:** null, not a stronger negative conclusion. The 3-band
  model is not systematically worse than the full model (its point estimates are higher), but it
  is also not reliably better.

## 3. Per-group failure patterns

Sources: P7-GRP, P7 §11 and CHK `per_group`. The test sets contain 11 Sheffield groups (8 leak,
3 no-leak) and 40 Mendeley groups (32 leak, 8 no-leak). The patterns below are **descriptive**;
none of them is a statistical test.

### Mendeley → Sheffield

**Group ranking is good and consistent for both logistic models.**

| Method | Smallest one-group AUROC (any group) | Group AUROC | Leave-one-group-out window AUROC |
|---|---|---|---|
| 3-band | 0.70 | 1.000 | 0.728–0.887 |
| Full | 0.60 | 0.958 | 0.693–0.811 |

- The 3-band model beats the full model by more than 0.05 on 6 of the 11 groups. The full model
  wins on 3: test3, test4 and noLeak_2.8bar_newer (CHK `per_group_counts_A`).
- No single group drives the aggregate result. The spread comes mainly from **which of the 3
  no-leak groups** is dropped.

**Model F is poor on almost every Sheffield group.** Its seed-mean one-group AUROC is 0.14–0.52,
except test2 at 0.73.

**The frozen threshold creates systematic false negatives for the 3-band model.**
- The threshold (+1.242) lies above every group mean (the largest is +1.147).
- As a result, 0 of 8 leak groups are detected: window sensitivity is 0.336.
- See `fig_per_group.png`, top left.

### Sheffield → Mendeley

**There is a broad failure with large spread between groups.**
- The 3-band one-group AUROC ranges from 0.18 to 0.90, with median 0.51 for both leak and
  no-leak groups.
- Five groups are below 0.5 for **all** methods (CHK):
  - BR_GL_0.18
  - BR_GL_0.47
  - BR_OL_0.18
  - LO_NL_0.47
  - LO_NL_ND

**The full model beats the 3-band model on 10 groups, and the 3-band model beats the full model
on 20 (CHK).** The pattern splits by topology:
- **Full model ahead:** mostly Branched leak groups.
- **3-band model ahead:** mostly Looped leak groups.
- Topology breakdowns (P7 §8):

  | Topology | 3-band | Full |
  |---|---|---|
  | Looped | 0.391 | 0.336 |
  | Branched | 0.584 | 0.650 |

- **On the Looped-only subset, both logistic models are inverted.**
  - Model F scores 0.59–0.61 on that subset (V2-BASE §13).
  - Each topology has only 4 no-leak groups.

**The frozen threshold creates systematic false positives.** The 3-band model flags 7 of 8
no-leak groups (P7-GRP group confusion: TP 31, FN 1, TN 1, FP 7).

### Both directions

**Scores are mostly set by the recording channel (stream), not by the individual window**
(CHK `stream_variance_share`). Stream identity explains this share of window-score variance:

| Direction | Logistic models | Model F |
|---|---|---|
| M→S (230 streams) | 0.88–0.90 | 0.79–0.83 |
| S→M (80 streams) | 0.74–0.94 | 0.40–0.59 |

- The effective sample is therefore streams and groups, not windows.
- Window counts mostly measure recording length.

**In S→M the full model flags 99.9% of Mendeley channel-2 windows, including 99.9% of ch2 no-leak
windows, but only 55% of ch1 no-leak windows** (CHK `per_channel_B`). This is a sensor-channel
offset: the fitted model responds to which accelerometer recorded the window.

## 4. What the mechanism analysis supports

**Directly measured observations:**

| Observation | Value | Source |
|---|---|---|
| Loudness (log RMS) reverses its leak association between rigs (group AUROC) | Mendeley 0.293, Sheffield 0.917 | P7 §12a; V2-BASE §16.3 |
| RMS baseline inverts in both directions | 0.232 and 0.339 | V2-BASE §12–13 |
| The rigs are near-perfectly separable from loudness-free features | rig-ID out-of-fold AUROC 0.982 | `shift_analysis.json` `separability` |
| Within-rig label separability from the same features | 0.827 Sheffield, 0.662 Mendeley | same |
| Leak − no-leak spectral contrast curves are weakly anti-correlated between rigs | Pearson −0.18 | `shift_analysis.json` `contrast_correlation` |
| Each of the three selected band fractions is leak-higher on both rigs (group AUROC 0.72–1.00) | — | P7 §12a |
| The fitted 3-band logistic weights are rig-specific; 600–800 Hz gets a negative weight in both directions | — | P7 §12b |
| Class-conditional band means shift between rigs | up to 1.72 pooled SD | P7 §12c |
| Window scores are dominated by stream identity; channel offset in S→M | — | CHK |

**Consistent with the proposed mechanism.** The following fit the idea that rig-dependent
loudness and spectral colouration drive the failure:
- Model F's inverted M→S ranking;
- its negative correlation with loudness on Sheffield (V2-BASE §16.3);
- the RMS inversion.

**Plausible but unverified:**
- **Physical source of the shift:** that it comes from sensor transfer functions, pipe material
  or diameter, or operating pressure and flow. These factors are confounded with rig identity, so
  they cannot be separated.
- **Channel effect:** the ch1/ch2 offset suggests sensor-specific response matters, but it was
  not tested.

**Not supported by the data:**
- **Band restriction:** that restricting to direction-stable bands yields a rig-invariant
  representation.
- **Kurtosis and crest factor:** that they reverse between rigs. This is stated in
  `docs/PHASE7_THREE_BAND_PREREG.md` but **no saved result in `results/` records it**. See §7,
  discrepancy D3.

## 5. What the three-band null result establishes

1. **Individually stable cues do not make a stable model.** Each band keeps its leak direction on
   both rigs, yet:
   - the multivariate fit weights them differently on each rig;
   - their absolute levels shift between rigs;
   - so neither the ranking (S→M) nor the threshold (both directions) transfers.
2. **The M→S gain over Model F belongs to spectral logistic regression in general, not to the
   three bands.**
   - The full model already gains +0.393 over Model F.
   - The band restriction adds a non-significant +0.064.
3. **Within-rig skill does not predict transfer, and the reverse also holds.**
   - Model F is the best method within Mendeley (fold mean 0.748, V2-BASE §14) and the worst
     transferring out of it.
   - The 3-band model has no skill on its own Mendeley validation groups but ranks Sheffield
     groups perfectly.
   - Training-rig validation therefore says little about transfer, in either direction.

The null result does **not** show that loudness cues are irrelevant. It also does not show that
no normalised spectral representation could transfer. It shows that this pre-specified one did
not, on these two rigs.

## 6. What remains unresolved

- **Why the shift happens.** Which physical factor drives it (sensor, pipe, regime or background
  structure) is unknown, because each is confounded with rig.
- **Whether the S→M failure is a property of Sheffield as a training rig.** Sheffield provides only
  2 no-leak training groups and 1 validation group, so this cannot be separated from the transfer
  question.
- **Whether any representation transfers in both directions.** Answering this needs a third
  independent rig (§11).

## 7. Limitations and uncertainty

**Independent groups.** There are 51 groups in all:
- Sheffield: 11 (8 leak, 3 no-leak).
- Mendeley accelerometer: 40 (32 leak, 8 no-leak).

**Sheffield no-leak groups.** There are only 3.
- In M→S, a class-stratified group bootstrap can draw only **10 distinct no-leak resamples**
  (multisets of 3 from 3).
- So the M→S CIs, including the paired difference CI, rest on very little information.
- V2-BASE §15 already states that these intervals "are not a meaningful confidence statement".
- M→S conclusions should rest on consistency across methods and groups, not on interval width.

**Band selection used both rigs.** That makes P7 exploratory. It can only bias the 3-band result
upward, so it strengthens, rather than weakens, the null reading.

**Rig transfer ≠ network generalisation.** This is a test between two laboratory rigs (Level 2).
It says nothing about municipal networks, buried pipes, other materials or field conditions.

**Mendeley topology.**
- The AGENTS.md Looped-only rule exists because Branched no-leak recordings were used as
  synthesiser background noise (INTEGRITY_LOG #7, #19).
- The V2 Model F and logistic models were trained on real rig windows only, with no synthetic
  cache (`Model_F/xrig_train.py` header), so the 40-group test is not contaminated for them.
- Looped-only breakdowns are reported (§3), but each topology has only 4 no-leak groups.

**Uncertainty structure.**
- All CIs are cluster bootstraps over groups, so they do not treat windows as independent.
- Point estimates are window-weighted, but the group-balanced versions agree within 0.03, with
  two exceptions in M→S (P7-GRP):
  - the full model (0.738 vs 0.819);
  - Model F seed 0 (0.293 vs 0.257).
- Model F seed ranges reflect training variability only.

**One-sensor recognition only.** Sheffield channels are not simultaneous, so the two-sensor idea
is not tested.

### Discrepancies found (documented, not reconciled)

| # | Discrepancy | Assessment |
|---|---|---|
| D1 | `report06/REPORT.md` and `summary.md` classify the outcome as "partial and partly unexpected (B/D)". P7 classifies it as **C, null**. | Same numbers, different framing. P7 is authoritative. Neither says the hypothesis is supported. |
| D2 | The Model F column of `report06/summary.md` writes the seed range as "[0.293–0.374]", in CI-style brackets. | It is a seed range, not a CI. P7 and this document use parentheses. |
| D3 | The post-hoc mechanism analysis that **selected** the bands, and that reported kurtosis and crest reversals, "is not saved as a file in this repository" (PHASE7 prereg). | The band-selection step cannot be audited. Only log RMS reversal and the three bands' direction stability are recorded (P7 §12a). Kurtosis and crest claims must not be cited as measured. |
| D4 | V2-BASE §16.1 reports a Sheffield sign reversal near 1500 Hz (background peaks near 1080 and 1500 Hz). The selected [1600, 2000) band starts just above it. | Not a contradiction, but the "direction-stable" label is edge-sensitive. That is a further reason not to generalise the band choice. |
| D5 | P7 §12 listed sensor differences as "not separable". CHK now shows a ch1/ch2 score offset in S→M. | This adds descriptive support for a sensor-channel effect. It is still untested as a cause. |

D3 is the only provenance problem. It does not affect any reported number, because the bands
were fixed in the pre-specification before scoring and the result is null. It does limit what may
be said about *why* the bands were chosen.

## 8. Strongest defensible scientific contribution

Ranked by strength of evidence:

1. **An empirical, pre-registered demonstration of cross-rig domain shift in acoustic leak
   detection.**
   - A model that learns within one rig (Mendeley, fold mean 0.748) collapses or inverts on an
     independent rig.
   - This is evaluated with group-level splits, frozen thresholds and group bootstraps.
2. **A neural-vs-simple-baseline comparison under shift.**
   - The direction-dependent result: simple spectral features transfer M→S and the network does
     not, while nothing transfers S→M.
   - "The model is too weak" is ruled out as the explanation, because Model F is the strongest
     method within Mendeley.
3. **A documented failure of a physically motivated invariant-feature hypothesis**, with a
   specific diagnosis: features that are stable one at a time can still give a non-invariant
   fitted model, and absolute levels shift.
4. **A reproducible protocol** for transfer between independent rigs: group splits, frozen
   validation thresholds, group bootstraps, pre-registration and exact reproduction checks.

Not claimed: a universal principle, a working invariant representation, or a novel method.

## 9. Claims to use in an IRIS submission

Each claim is followed by its evidence.

- "A leak detector trained on one laboratory rig's recordings did not transfer to an independent
  rig. Its ranking inverted from Mendeley to Sheffield (AUROC 0.345, range 0.293–0.374 over 3
  seeds) and was at chance from Sheffield to Mendeley (0.525)."
  - Evidence: V2-BASE §12–13.
- "Within the Mendeley rig, the same recipe reached fold-mean AUROC 0.748. The failure is one of
  transfer, not of capacity."
  - Evidence: V2-BASE §14.
- "Loudness reversed its association with leaks between the two rigs. Rig identity was far more
  predictable from the features than the leak label (out-of-fold AUROC 0.982 vs 0.66–0.83)."
  - Evidence: P7 §12a; `shift_analysis.json`.
- "A simple spectral logistic regression transferred from Mendeley to Sheffield (0.738) but not
  back (0.473)."
  - Evidence: V2-BASE §12–13.
- "An exploratory classifier restricted to three normalised frequency bands, chosen because they
  behaved the same way on both rigs, did not reliably improve transfer. Gains over the full
  spectral model were +0.064 and +0.033 AUROC, with group-bootstrap intervals that include zero."
  - Evidence: P7 §10.
- "The evaluation has only 3 independent no-leak recordings on one rig. The results are limited to
  two laboratory rigs."
  - Evidence: §7.

## 10. Claims to avoid

| Avoid | Why |
|---|---|
| "The three-band representation improves cross-rig transfer" or "achieves 0.80 cross-rig AUROC" | The CI includes 0, S→M is at chance, and the M→S model has no skill on its own training rig. |
| "We identified rig-invariant frequency bands" | The bands are stable one at a time but shift in level, and they were selected using both rigs. |
| Any statement of generalisation to municipal networks, buried pipes, other pipe materials, or real-world leak detection | Only two laboratory rigs were tested. |
| "Kurtosis and crest factor reverse between rigs" | There is no saved result (D3). |
| "Sensor differences cause the failure" | Confounded with rig and untested. The channel offset (CHK) is descriptive. |
| Narrow-CI language for M→S, or treating window counts as sample size | 3 no-leak groups and 10 distinct bootstrap resamples. |
| "Simple features beat deep learning" | True in one direction only. In the other, all methods fail. |
| Calling the band experiment confirmatory or pre-registered without qualification | It was pre-specified, but the bands were chosen post hoc using both rigs. |

## 11. Are further experiments necessary before writing?

**No.** The question the experiments set out to answer is answered with the recorded evidence:
- whether Model F transfers between independent rigs, and
- whether the pre-specified invariant representation fixes it.

The remaining unknowns (§6) concern *causes*. They need a third independent rig, or a rig with
controlled sensor and regime changes, and those data are not available. More analysis of the same
two rigs would mean choosing representations after seeing both test sets, which cannot produce
confirmatory evidence.

D3 is a documentation gap, not a missing result. Handle it in the text by not claiming the
unsaved kurtosis/crest findings.

## Final decision

**A. FREEZE AND WRITE.** The existing evidence supports an honest, coherent submission:
- the cross-rig collapse;
- the measured loudness reversal and rig separability;
- the direction-dependent baseline comparison;
- the null invariant-feature result with its diagnosis.

No integrity problem was found that changes any reported number. No new experiment is necessary.
