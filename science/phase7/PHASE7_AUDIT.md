# Phase 7 audit: 3-band cross-rig intervention

Audit date: 6 Oct 2026. Scope: the Phase 7 "cross-rig-invariant normalised-PSD band"
intervention for Mendeley ↔ Sheffield. No computation was run for this audit except reading
existing result files and counting groups in existing split records.

**Headline: Phase 7 has not been run. No Phase 7 result exists, locally or on the remote.**
The verdict is therefore **INCONCLUSIVE (not executed)**. There is also a second finding that
matters more: even if the locked script is run as written, its Mendeley/Sheffield numbers are
largely predetermined by how the bands were chosen (§10), so they cannot count as evidence
for the hypothesis.

---

## 1. Repository state

*Re-verified on 6 Oct after a review comment that the baseline lives on `v2-cross-rig-baseline`.
Every ref was checked; the conclusions below did not change.*

### A. Completed and committed cross-rig baseline: authoritative

- **Branch / commit:** `v2-cross-rig-baseline` → `224ca5d` ("Add Version 2 cross-rig baseline:
  Mendeley <-> Sheffield transfer benchmark", 4 Oct). `git rev-parse` gives the same SHA for the
  local branch, `origin/v2-cross-rig-baseline` and the current `HEAD`.
- **Verified committed artifacts at `224ca5d`:**
  - **Code:** `experiments/rigs.py`, `Model_F/xrig_train.py`, `experiments/xrig_eval.py`,
    `xrig_shift.py`, `xrig_summary.py`.
  - **Docs:** `docs/VERSION2_CROSS_RIG_PREREG.md`, `docs/VERSION2_CROSS_RIG_BASELINE.md`, and
    `docs/INTEGRITY_LOG.md` entries #23–25.
  - **Results:** `results/v2_cross_rig/`, holding 12 training records, 12 eval JSONs, 12 score
    `.npz` files, `summary.json` and `shift_analysis.json`.
  - **Run records:** 25 `results/runs/*v2_xrig*`.
  - **Checkpoints:** 12 `models/xrig_*.pt` (LFS).
  - **Figure:** `plots/v2_xrig_shift.png`.
- **Integrity of what this audit used:** the working tree has no tracked diff against `224ca5d`.
  Every baseline number in this audit was read from these committed files and matches the
  write-up.
- **One gap:** the write-up lists `logs_v2/` as the log location, but that directory is neither
  committed nor present locally.

### B. Phase 7 exploratory band intervention: never committed, never run

- **No ref contains it.** No commit on any ref (all local branches, all remote branches, the
  stash) touches `xrig_band_intervention.py`, `xrig_mechanism.py`, `mechanism_analysis.json` or
  `band_intervention.json`.
- **No branch tip contains the identifiers.** A grep of every branch tip and the stash for
  `band_intervention` / `BANDS_3` / `xrig_mechanism` returns nothing.
- **The remote has no Phase 7 branch.** It holds 7 heads, none of them Phase 7.
- **The files exist only as untracked files in this working copy** (table below).
- **Status by part:**
  - The **mechanism analysis** (exploratory, selects nothing) was run.
  - The **band intervention** was not run.

### C. Current working branch

- **`cross-rig-mechanism`:** a local-only branch created at `224ca5d`, with no commits of its own.
  It is therefore content-identical to `v2-cross-rig-baseline`, and all Phase 7 files are
  untracked on top of it.
- **Other refs:** `origin/main` is `4b20090`; local `main` is `4f1eeb8` (= `origin/model-f-snr-sweep`).

Untracked Phase 7 material in the working copy:

| artifact | timestamp | content |
|---|---|---|
| `experiments/xrig_mechanism.py` | 4 Oct 23:49 | exploratory per-feature / per-condition / two-sensor analysis |
| `mechanism_run.log`, `results/v2_cross_rig/mechanism_analysis.json` | 4 Oct 23:50 | its output (complete; "Saved … (60s)") |
| `experiments/xrig_band_intervention.py` | 5 Oct 00:48:21 | the locked Phase 7 script |
| `band_intervention_run.log` | 5 Oct 00:48:27 | **0 bytes** |
| `results/v2_cross_rig/band_intervention.json` | — | **does not exist** |
| `results/runs/*v2_xrig_band_intervention*` | — | **does not exist** |

The log was created 6 s after the script was saved and never written to. No Python process is
running. The run was either never started properly or died before its first print, which comes
after both rigs load. **The pushed/committed state contains no Phase 7 experiment, and the
local state contains only an unexecuted protocol.**

There is no Phase 7 protocol document: no `.md` file mentions the 3-band intervention. The
only protocol is the docstring of `xrig_band_intervention.py`.

---

## 2. What was intended (reconstructed from the script)

| question | answer (source: `xrig_band_intervention.py`) |
|---|---|
| Model / features | (a) **3-band**: log10 normalised-PSD energy fraction in 400–600, 600–800, 1600–2000 Hz (Welch, nperseg 256, same convention as `cross_dataset.features_1ch`, verified). (b) **Full spectral logreg**: `features_1ch` (9 bands + centroid, flatness, kurtosis, crest, ZCR). (c) **Model F** cross-rig checkpoints `xrig_*_foldx_seed{0,1,2}.pt`, inference only. |
| Classifier | `StandardScaler + LogisticRegression(C=0.5, class_weight="balanced")` for (a) and (b) |
| Training rig / test rig | A: Mendeley acc → Sheffield; B: Sheffield → Mendeley acc. Both directions. |
| Splits | Training/validation groups taken from the recorded `xrig_*_foldx_seed0.json`, so they match `docs/VERSION2_CROSS_RIG_PREREG.md`; test = every group of the other rig |
| Preprocessing | Model F pipeline (5 kHz, 2 kHz low-pass, 0.4 s windows, unit RMS). Nothing fitted except the scaler and logreg, fitted on training-rig train groups |
| Threshold | Youden on training-rig validation windows, frozen before test scoring |
| Uncertainty | cluster bootstrap over physical groups, 2000 draws |
| Seeds | Model F seeds 0/1/2; logregs are deterministic (one fit); pairing seed 0 |
| Within-rig references | **not recomputed** for the 3-band model; only the baseline's within-rig numbers exist |
| Pass/fail rule | **none written down** |

**Data change versus the audited baseline.** `datasets/Accelerometer/Accelerometer/` on this
machine contains only `Looped/`. Branched is missing, and neither `cache_mendeley/` (20 files,
all `LO_*`) nor `cache_xrig/mendeley_acc.npz` (rebuilt 4 Oct 23:47) holds Branched windows.
So Phase 7 runs on **Mendeley Looped-only: 20 groups (16 leak / 4 no-leak), 3572 windows**,
not the 40 groups of the baseline. Consequences, counted from the recorded split files:

- **Direction A training set:** 14 of the 31 recorded train groups remain, **2 of them no-leak**.
  Validation keeps 6 of 9 groups, **2 no-leak**. The 3-band and full logregs are therefore refit
  on 2 negative recordings, and the threshold is chosen on 2 others.
- **Direction B test set:** 20 Looped groups (4 no-leak) instead of 40 (8 no-leak).
- **Model F direction A checkpoints** were trained on all 40 groups (inference only, which is fine),
  but the logregs they are compared against now see half the data. **The comparison is not
  like-for-like on training data.**

---

## 3. Protocol integrity audit

| check | status |
|---|---|
| Physical groups intact across train/val/test | **Yes.** Group-level splits come from the recorded split files; Sheffield leak groups = test condition, no-leak = deduplicated recording; Mendeley = recording |
| Windows from one group across splits | No (cross-rig: the test rig is entirely separate) |
| Preprocessing fitted on test data | No |
| Threshold fitted on test data | No |
| **Feature/band choice used test-rig labels** | **Yes, in both directions** (see §10) |
| Pass/fail rule fixed before results | **No.** The script says "locked", but no decision rule was written |
| Independent physical groups | Sheffield 11 (8 leak / **3 no-leak**); Mendeley Looped 20 (16 / **4**); direction-A training 2 no-leak, validation 2 no-leak |
| Baselines | Model F (3 seeds), full spectral logreg; RMS is not included in the script |
| Both directions | Yes |
| Within-rig references | From the baseline only (40-group Mendeley); none for the 3-band model |
| Code committed | No (all untracked) |
| Data deviation logged | Only in the docstring; nothing in `docs/INTEGRITY_LOG.md` |

---

## 4. Actual results

**There are no Phase 7 results.** The table below shows only what exists, so that the slot the
3-band model would fill is visible. Window AUROC on logits/scores; group AUROC in brackets
where available.

| method | M→S (Sheffield, 11 groups) | S→M, 40 groups (baseline) | S→M, Looped-only, 20 groups (Phase 7 test set) |
|---|---|---|---|
| Model F seeds 0/1/2 | 0.293 / 0.369 / 0.374 | 0.515 / 0.524 / 0.535 | 0.586 / ≈0.60 / ≈0.61 |
| Model F mean | **0.345** (group 0.222) | **0.525** (group 0.523) | ≈0.60 |
| Full spectral logreg | **0.738** [0.630, 0.939]* (group 0.958) | 0.473 (group 0.430) | **0.336** |
| RMS | 0.232 (group 0.083) | 0.339 (group 0.293) | 0.261 |
| **3-band logreg** | **NOT RUN** | **NOT RUN** | **NOT RUN** |
| chance | 0.500 | 0.500 | 0.500 |

\*This bootstrap resamples 3 Sheffield no-leak recordings. It is not a meaningful confidence
interval (stated in the baseline pre-reg) and is shown only for completeness.

Sources: `results/v2_cross_rig/summary.json`; Looped-only breakdowns in
`results/v2_cross_rig/eval_xrig_sheffield_foldx_seed0.json` (`test_sets/mendeley_acc/breakdowns`);
seeds 1 and 2 Looped values (0.60, 0.61) from `docs/VERSION2_CROSS_RIG_BASELINE.md` §13.

**Note:** the full-logreg M→S value of 0.738 was fitted on 40 Mendeley groups. Under the Phase 7
data it would be refit on 14, so the Phase 7 run would not reproduce 0.738.

Within-rig references (baseline, grouped 3-fold, window AUROC, fold means): Mendeley (40) Model F
0.748, logreg 0.588, RMS 0.658. Sheffield Model F 0.630, logreg 0.871, RMS 0.770 (each Sheffield
fold tests **one** no-leak recording).

---

## 5. M→S analysis

There is no 3-band measurement. What is already known:

- **The full spectral logreg already transfers in this direction** (0.738, group 0.958). The
  "baseline transfers, Model F inverts" pattern is the existing Case-D finding.
- **On Sheffield, the three selected bands score 0.802 / 0.577 / 0.644 one at a time**
  (window AUROC), with group AUROC **1.000 / 0.833 / 0.750**. These were computed on all 11
  Sheffield groups, which are exactly the direction-A test set (`mechanism_analysis.json`).
  - **Implication:** if the Mendeley-fitted weights come out all positive, a 3-band score will land
    near this range, close to the existing 0.738. So the expected gain over the full logreg is
    small, and whatever it is, it was set by the selection.
- **The training set for this direction shrank to 2 no-leak recordings.** The refit is more fragile
  than the audited baseline.

## 6. S→M analysis

There is no 3-band measurement. What is already known:

- **On the Looped-only test set, the full spectral logreg is inverted (0.336)** and Model F is
  about 0.60.
- **On the same 20 Looped groups, the selected bands score 0.753 / 0.842 / 0.822 one at a time**
  (window), with group AUROC **0.906 / 0.953 / 0.859**.
- **Implication:** a 3-band result of 0.75–0.85 in this direction would look like a dramatic rescue
  (+0.4 over the full logreg, +0.2 over Model F). It would be almost entirely the selection showing
  up again: each band was picked because it already separates these exact recordings.

## 7. Comparison with Model F

Not possible: there is no measurement. Model F's cross-rig numbers stand as audited: 0.345
(inverted) for M→S and 0.525 (chance) for S→M. Looped-only S→M is about 0.60.

## 8. Comparison with baselines

Not possible for the 3-band model. Among existing methods, the best baseline is the full logreg
in A (0.738). In B, every method is at or below chance on 40 groups (best: Model F 0.525). On
Looped-only, Model F (about 0.60) beats the full logreg (0.336).

---

## 9. Statistical / physical-group limitations

- **Negatives are the binding constraint.** Sheffield has 3 independent no-leak recordings,
  Mendeley Looped has 4, and direction-A training has 2 + 2 for validation.
- **Group AUROC on Sheffield is ranked over 8 × 3 = 24 leak/no-leak pairs**, so one background
  recording moves it by up to about 0.33. On Mendeley Looped it is 16 × 4 = 64 pairs.
- **No legitimate CI exists for a Sheffield-test AUROC.** The bootstrap resamples 3 negatives.
  Mendeley Looped (4 negatives) is barely better.
- **Thousands of windows (17 250 Sheffield; 3572 Mendeley Looped) add no independent evidence**
  about the leak/no-leak contrast: they are 0.4 s slices of 7 background recordings.
- **Seeds:**
  - For the logregs, seeds are irrelevant: they are deterministic.
  - For Model F, the three seeds measure training noise, not data variability.
  - "Robust across seeds" therefore cannot be shown for the 3-band model, and it would not mean
    much if it could.

## 10. Selection-bias / exploratory-status assessment

This is the central problem, and it is worse than "exploratory".

1. **The bands were chosen from leak-vs-no-leak AUROCs computed on the full label sets of both
   rigs:** all 11 Sheffield groups and all 20 Mendeley Looped groups. These are exactly the two
   Phase 7 test sets. Both directions' test labels informed the feature choice.
2. **The selected features already separate the test sets at group level** (Sheffield 400–600 Hz:
   group AUROC 1.000; Mendeley 600–800 Hz: 0.953). A logistic regression only learns weights
   and an offset.
   - **AUROC ignores a per-rig offset.** If the fitted weights come out positive, which the
     training rig's own univariate AUROCs predict in both directions, the test AUROC is mostly
     fixed in advance.
   - **The only open parts** are whether correlated, compositional band fractions produce a
     negative coefficient, and the threshold metrics, which are sensitive to the rig offset.
3. **The selection rule is undocumented and had alternatives.** Other features also had
   same-sign leak AUROC above 0.6 on both rigs: 100–200 Hz (0.627 / 0.712) and flatness
   (0.836 / 0.601). 400–600 Hz is no more rig-invariant than 100–200 Hz (|rig AUROC| 0.623 vs
   0.631). This is a forking path.

**Permissible claims if the locked script is run:**
- *Allowed:* "In an exploratory analysis, three normalised-PSD bands chosen post hoc from both
  rigs' labels separate leak from background on both rigs. A logistic regression on them, fitted
  on one rig, ranks the other rig's windows at X."
- *Not allowed:* "Restricting to 400–2000 Hz rescues cross-rig generalisation"; "these bands
  contain rig-invariant leak information"; "evidence that the invariance hypothesis is correct".
  A positive Mendeley/Sheffield result **carries almost no evidential weight**. A negative result
  **would** be informative: features chosen to work on the test set still failed once weights
  had to come from the other rig.

**Is there an independent validation set?** Not between Mendeley and Sheffield; both are spent.
Two sources were **not used in the band selection**:
- **Dongguan** (outdoor test base; 79 trial-groups in the R2 held-out set: 73 leak / 6 no-leak).
- **Hong Kong** (buried networks; 8 sites: 5 leak / 3 no-leak).

They are only **partly** suitable:
- **Earlier exposure:** both were seen in E9/R2 with the full feature vector, though no per-band
  analysis of them is on record (searched `docs/`, `experiments/`, `reports/`, `research_notes/`).
- **Each site or trial group is single-class**, so label and site are confounded.
- **The no-leak counts are again tiny** (6 and 3).
- **Hong Kong is a different physical level:** buried municipal pipe, not a lab rig.

Within those limits they are the only non-circular test available.

---

## 11. Mechanistic interpretation (existing evidence only)

**OBSERVED** (sources: `summary.json`, `shift_analysis.json`, `mechanism_analysis.json`):
- **Loudness inverts between rigs.** log-RMS leak AUROC is 0.261 on Mendeley Looped and 0.768 on
  Sheffield (group 0.125 vs 0.917). RMS transfer is inverted in both directions.
- **The leak − no-leak normalised-PSD contrast curves of the two rigs are weakly anti-correlated**
  (Pearson −0.18). Sheffield's leak energy is concentrated at about 150–650 Hz; Mendeley's is
  broadband above about 600 Hz.
- **Rig identity is almost perfectly separable** from the loudness-free features (out-of-fold
  0.982), more so than the label within either rig.
- **Some features keep the same sign on both rigs** (400–600, 600–800, 1600–2000, 100–200 Hz,
  flatness), but with very different strengths. Example: 600–800 Hz gives 0.842 on Mendeley but
  0.577 on Sheffield.
- **Mendeley ch1/ch2 are essentially uncorrelated** (mean |zero-lag r| 0.019, peaks at lags of
  600–2400 samples). The "relational" features that separate Mendeley within-rig (out-of-fold
  0.884; d_log_rms 0.123, i.e. strongly inverted-informative) are therefore **inter-sensor level
  and spectral-tilt differences, not coherence or timing**.
- **The Sheffield columns are not simultaneous.** A matched relational feature cannot be built
  on Sheffield, so the relational hypothesis cannot be tested cross-rig with these data.

**SUPPORTED INFERENCE:**
- The cross-rig collapse is driven by **rig-specific spectral colouration and loudness relations
  that change sign**, not by model capacity: Model F learns within Mendeley at 0.70–0.79.
- With **3 and 4 independent background recordings**, any per-band "invariance" is a statement
  about a handful of recordings. A band that looks same-sign could easily be same-sign by chance
  of which backgrounds were recorded.
- **Sheffield's label is partly confounded** with pressure (4.2 bar in one leak and one no-leak
  group) and sensor distance (leaks up to 50 m, backgrounds up to 20 m).

**UNTESTED HYPOTHESES:**
- That 400–2000 Hz normalised bands carry leak information that is invariant across rigs
  (not testable on Mendeley/Sheffield any more).
- That two-sensor relational information would transfer (no simultaneous Sheffield data).
- That per-recording background whitening would fix the transfer. This was proposed in baseline
  §19, but it needs a no-leak reference from the same installation, which in deployment amounts
  to calibration.

---

## 12. Final verdict

**INCONCLUSIVE: the experiment was not executed.**

There is no Phase 7 measurement in either direction. In addition, as designed, its
Mendeley/Sheffield arm **cannot** yield a CLEAR or PARTIAL IMPROVEMENT that counts as evidence:
the bands were selected on both test sets. That arm can only produce an exploratory description,
or an informative failure.

---

## 13. One recommended next experiment

**Run the locked Phase 7 script once, and add a frozen evaluation on the two sources the bands
never saw (Dongguan, Hong Kong), under a decision rule written before running.**

Concretely, as one CPU-scale run with no training:
1. **Before running:** commit a short addendum with the data deviation (Looped-only, 20
   groups; direction-A training now 2 no-leak) and the decision rules in §15–16. Also add an
   `INTEGRITY_LOG` entry for the Branched data loss.
2. **Run `xrig_band_intervention.py` unchanged** for M→S and S→M. Report it as exploratory, with
   the per-band test-set AUROCs from §5–6 printed beside it so readers can see the circularity.
3. **Untouched-source check.** Take the four already-fitted logistic regressions (3-band and full,
   each fitted on Mendeley train groups and on Sheffield train groups). Score Dongguan and Hong
   Kong **with no refit and no threshold change**. Report window and group AUROC with group
   counts, and the fitted coefficient signs from each training rig.

Model F inference on Dongguan/Hong Kong is **not** part of this. The original Model F saw those
sources, and the cross-rig checkpoints are not needed to test the band hypothesis.

## 14. Why this is worth doing

- **Step 3 is the only non-circular test of the band hypothesis** that the existing data allow,
  and it costs minutes, not GPU-hours.
- **Steps 1–2 close an open loop.** A designed-and-abandoned experiment otherwise sits in the
  record unexplained.
- **It decides whether the band line continues at all.** Without step 3, any further band or
  feature work is selection on Mendeley+Sheffield.

## 15. What result would support the hypothesis (fixed before running)

All of these must hold:
- **Same-sign coefficients:** the 3-band coefficients fitted independently on Mendeley and on
  Sheffield have the same sign for every band. This is what invariance predicts.
- **Above chance on both untouched sources:** the 3-band group AUROC exceeds 0.5 on both Dongguan
  and Hong Kong, for **both** training-rig fits (4 of 4 cells), with window AUROC ≥ 0.65.
- **Better than the full features:** the 3-band model is ≥ the full-feature logreg's group AUROC
  in at least 3 of the 4 cells.

Even then, the claim is "**consistent with** band invariance on two further sources with
3 and 6 negative groups and single-class sites", not "confirmed".

## 16. What result would falsify it

- **Coefficient signs disagree** between the Mendeley and Sheffield fits for any selected band, or
- **3-band window AUROC ≤ 0.55, or ≤ the full-feature logreg,** in 2 or more of the 4
  untouched-source cells.

Anything in between is recorded as **inconclusive**, with no follow-up band search.

## 17. What we should stop doing if it fails (or is inconclusive)

- **Stop searching feature subsets, bands or representations on Mendeley + Sheffield.** Both rigs'
  labels are spent, and every further choice is another fork on 7 background recordings.
- **Stop training new cross-rig models.** Model F capacity is not the bottleneck; the data are.
- **Write up the pre-registered cross-rig collapse and its mechanism as the AcousticLeakNet
  Version-2 result,** and move the time to Ferrite:
  - Model F learns within a rig.
  - Loudness and spectral contrast invert across rigs.
  - A loudness-free baseline transfers in one direction only.
  - Independent negatives are the binding limit.
- **Further cross-rig work only with new data:** a rig with ≥ 10 independent background recordings,
  or simultaneous two-sensor recordings on a second rig. Not more analysis of these two.

---

## 18. IRIS strategic assessment

1. **Credible research story?** Yes, but it is an honest-negative / domain-shift story, not a
   generalisation story:
   - The synthetic in-domain result is 1.000 (in-domain only).
   - Real held-out Mendeley performance is modest.
   - The pre-registered independent-rig test collapses.
   - There is a documented physical mechanism (sign-flipping loudness and spectral cues).
   - The integrity record is careful (pre-registration, integrity log, group-level statistics).

   That is defensible and fairly rare at this level.
2. **Is the cross-rig result strong enough for IRIS?** The *collapse and its mechanism* are: they
   are pre-registered, both directions were run, and they are reproducible from committed files.
   *Phase 7* is not, and should not be in the submission unless step 3 above passes. Even then it
   belongs in the paper as an exploratory panel.
3. **Can one more experiment materially improve the story?** Only marginally:
   - **If step 3 passes:** the story gains "a candidate invariant cue, consistent on two unseen
     sources". That is a nice paragraph, not a new claim.
   - **If it fails:** the story gains a clean negative: "even test-informed band restriction does
     not survive".

   Neither changes the headline.
4. **Diminishing returns?** Yes. The project is past the point where more analysis of
   Mendeley + Sheffield can strengthen a claim. The recommended run is justified only because it
   is cheap and closes the loop. Anything larger is not justified while Ferrite is the primary
   project.

**Deadline check:** the research pipeline notes list IRIS as **3 October**, and today is
**6 October**. If that date is right, this work now serves the later deadlines (INSEF 25 Oct,
Conrad 30 Oct, JEI) rather than IRIS. That makes it even less important to fit Phase 7 in before
a submission. Please confirm the actual IRIS date.

---

---

## Addendum 1: locked protocol for the single Phase 7 run

Written on 6 Oct 2026, **before** `experiments/xrig_band_intervention.py` or
`experiments/xrig_band_external.py` produced any score. This addendum, both scripts and the
mechanism analysis are committed together, and that commit is the lock. Nothing below may be
changed after results are seen. Each script is run **exactly once**. A crash is reported as a
crash, not rerun until it works.

### F. Status: exploratory, not confirmatory

- **The bands were chosen with test labels.** The three bands (400–600, 600–800, 1600–2000 Hz)
  were selected in `xrig_mechanism.py` using leak labels from **all** Sheffield groups and
  **all** Mendeley Looped groups. Those are the same recordings that form the two cross-rig test
  sets below.
- **What a strong M↔S result would and would not show.** A strong Mendeley ↔ Sheffield result
  therefore **does not establish cross-rig generalisation** or band invariance. It shows only
  that bands picked to work on these recordings still work once the weights come from the other
  rig.
- **Dongguan and Hong Kong are the only sources not used to choose the bands.** Even they are
  not a clean confirmation:
  - Each site or trial group is single-class.
  - The no-leak groups are few.
  - Both were seen earlier (E9/R2) with the full feature vector.

  A pass there is reported as "consistent with", never "confirmed".

### G. Data: Mendeley is Looped-only on this machine

- **What is missing.** `datasets/Accelerometer/Accelerometer/` has no `Branched/` folder, and no
  cache holds Branched windows. Mendeley in this run is **20 groups (16 leak / 4 no-leak)**, not
  the 40 of the audited baseline.
- **Direction A training data.** The logregs are refit on the Looped part of the recorded train
  groups: 14 groups, 2 no-leak. Their thresholds come from the Looped part of the recorded
  validation groups: 6 groups, 2 no-leak.
- **Model F direction-A checkpoints.** These were trained on all 40 groups. They are re-scored
  (inference only), so their training data is **not** matched to the logregs'.
- **Direction B test set.** It is Looped-only for **every** method, so Model F and the logregs are
  compared on identical test data.
- **Comparability with the baseline.** None of these numbers is directly comparable to the 40-group
  numbers in `docs/VERSION2_CROSS_RIG_BASELINE.md`. Comparisons are made **only within this run**.

### A. Primary metric

- **Primary metric:** window AUROC on the raw decision function (logreg) or logits (Model F).
- **Reported alongside:**
  - Group AUROC (per-group mean score ranked over leak × no-leak group pairs).
  - Cluster-bootstrap CI over physical groups (2000 draws).
  - Group counts per class.
  - Threshold metrics at the frozen training-rig Youden threshold.
- **CIs are not used for any decision.** No-leak group counts are 3 (Sheffield), 4 (Mendeley
  Looped) and few on the external sources.

### B. Mendeley → Sheffield comparison (exploratory)

- **Setup:** fit on the Looped Mendeley train groups, threshold on the Looped Mendeley validation
  groups, test on all 11 Sheffield groups (8 leak / 3 no-leak).
- **Compared:** 3-band logreg vs the full spectral logreg (same fit data, same run) and vs the
  Model F mean over seeds 0/1/2 (re-scored in the same run).

### C. Sheffield → Mendeley comparison (exploratory)

- **Setup:** fit on the recorded Sheffield train groups, threshold on the Sheffield validation
  groups, test on the 20 Looped Mendeley groups (16 leak / 4 no-leak).
- **Compared:** the same methods as in B.

### D. Baselines

- **Full spectral logreg:** `cross_dataset.features_1ch`, same classifier, same fit data. This is
  the primary comparator.
- **Model F (3 seeds, mean):** the cross-rig checkpoints, re-scored. A secondary comparator in
  direction A, because it was trained on 40 groups.
- **Chance:** 0.5.
- **On the external sources, the comparator is the full spectral logreg only.** Model F is not
  scored there; its re-scoring is not part of this test.

### E. Decision rules (fixed now)

**M↔S arm, per direction. Descriptive only, cannot "pass".** Let
`best = max(full logreg, Model F mean)` in that direction.

| label | condition |
|---|---|
| *exploratory gain* | 3-band ≥ best + 0.05 |
| *no meaningful change* | \|3-band − best\| < 0.05 |
| *worse* | 3-band ≤ best − 0.05 |
| *informative negative* (extra flag) | 3-band < 0.60 in that direction, although its bands were chosen on that test set |

**External arm (the only decision-bearing part).** A cell is one training-rig fit
(Mendeley-fit or Sheffield-fit) × one external source (Dongguan, Hong Kong). There are 4 cells.

- **Sources:**
  - **Dongguan:** `public_data.load_dongguan()`, leak + no-leak folders; the environmental-noise
    folder is excluded (loader default).
  - **Hong Kong:** `public_data.load_hongkong()`, noise loggers + hydrophones, grouped by site.
- **Preprocessing:** the same per-file steps as the rigs. No refit of anything: the models are
  rebuilt deterministically and must reproduce the stored Phase 7 test AUROCs and thresholds to
  1e-9, or the script aborts.
- **SUPPORTED (exploratory)** requires all of:
  1. **Same-sign coefficients:** each 3-band coefficient has the same sign in the Mendeley fit and
     the Sheffield fit.
  2. **Every cell clears the bar:** 3-band window AUROC ≥ 0.65 **and** group AUROC > 0.50 in all
     4 cells.
  3. **Beats the full features:** 3-band group AUROC ≥ full-logreg group AUROC in at least 3 of
     the 4 cells.
- **FALSIFIED** if either:
  - the coefficient signs disagree for any band, **or**
  - 2 or more cells fail, where a cell fails if its 3-band window AUROC ≤ 0.55 **or** ≤ that cell's
    full-logreg window AUROC.
- **INCONCLUSIVE:** anything else.
- **Secondary, descriptive only:** the per-sensor Hong Kong splits (noise logger, hydrophone).

**Consequence (fixed now):**
- **FALSIFIED or INCONCLUSIVE:** stop all band/feature engineering on AcousticLeakNet. Record:
  "The exploratory band-restriction intervention did not provide sufficient evidence to
  overturn the Version-2 cross-rig collapse."
- **SUPPORTED:** no automatic follow-up. The claim is limited to "consistent with a
  same-sign band cue on two further single-class-site sources". Another experiment is proposed
  only if it answers a specific remaining question cleanly.

### Commands (run once each, in this order)

```
.venv/Scripts/python.exe experiments/xrig_band_intervention.py   > science/phase7/run_band_intervention.log
.venv/Scripts/python.exe experiments/xrig_band_external.py        > science/phase7/run_band_external.log
```

---

**NEXT ACTION (as written before Addendum 1):** Write a short addendum that freezes the decision rule, then run the existing
locked `xrig_band_intervention.py` once, plus a no-refit scoring of its fitted models on Dongguan
and Hong Kong (about 1 h total, CPU, no training). If the §16 falsification rule triggers, or the
result is inconclusive, stop the band/feature line and write up the cross-rig collapse as the
final AcousticLeakNet Version-2 result.
