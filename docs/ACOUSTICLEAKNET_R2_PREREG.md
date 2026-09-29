# AcousticLeakNet R2 pre-registration: held-out real-source group stability

Written 2026-09-29, **before** any of this analysis is run, using only already-trained
checkpoints (`best_model_f_seed{0,1,2}.pt`, committed at `1551414`) and the already-built
`cache_f/bank.npz`. No new training. This is a separate, exploratory experiment; it does not
change `docs/MODEL_F_PREREG.md`, its addendum, H0–H4, or any existing verdict.

## Why this experiment, and why not the alternatives

**Dataset audit (Phase 1), summarised:**

| Data | Independent grouping available? | Already tested by an existing experiment? |
|---|---|---|
| Synthetic (EPANET networks) | Yes, many networks | Yes — Model C AUROC 1.000 on 3 held-out networks |
| Mendeley (Accelerometer/Hydrophone) | No site field: **one physical testbed**, grouped only by recording (both channels of one run). Every window from the same testbed shares undocumented site-level acoustic structure (room, pipe run, pump) | Yes — Looped-only zero-shot (E4), R1 seed replication |
| Hong Kong (noise loggers, hydrophones) | Yes: 50 distinct physical sites (32 noise-logger, 18 hydrophone), parsed from the file-naming scheme in `experiments/public_data.py:hk_site`. **Every site is single-class** (checked directly against the raw files): no site has both leak and no-leak recordings, because a site was monitored during a leak event or was not, never both. Site identity and label are therefore perfectly confounded in this dataset, as in essentially any field leak survey | Cross-SOURCE only (H3: `f_nohk` excludes Hong Kong entirely) |
| Dongguan | "NA" for a geographic site field throughout; grouped by experimental trial (material, pressure, sensor). 293 groups, each again single-class | Cross-SOURCE only (H3: `f_nodg`) |
| Sheffield | Not a leak/no-leak classification dataset (attenuation/spectrum calibration only, E10) | N/A — excluded from this analysis |

**What is not yet possible:** because Hong Kong has only two sources total (`SOURCES = ("hongkong",
"dongguan")` in `Model_F/train_f.py`), leave-one-source-out is exhausted by the existing `f_nohk`/
`f_nodg` runs (H3). A finer within-source, **leave-several-sites-out** test has never been run for
the main `f` checkpoints: `experiments/model_f_eval.py` only evaluates a checkpoint on the public
sources it was trained *without* (`cfg["exclude_source"]`), so `f` (excluded = none) has never been
evaluated on Hong Kong or Dongguan at all.

**What already exists, unused:** `Model_F/bank_f.py` deterministically marks 20% of groups per
source as held out (`is_val_group`, a CRC32 hash of the group string, **independent of `--seed`**),
and `Model_F/train_f.py`'s `Bank` class excludes those windows from every training row for every
seed. This gives 8 Hong Kong sites (5 leak, 3 no-leak, 675 windows) and 79 Dongguan trial-groups
(73 leak, 6 no-leak, 484 windows) that no seed-0/1/2 checkpoint has ever trained on. Confirmed by
inspecting `cache_f/bank.npz` directly (its own sha256 matches the committed Git LFS id) and by
reading `Bank.__init__`'s `keep = (z["s_val"] == val) & ...` filter.

**The one real problem with reusing it:** `train_f.py` selects the saved checkpoint as the epoch
that maximises `score = (auc_syn + auc_real) / 2`, where `auc_real` is exactly
`roc_auc_score` on these same held-out windows, computed every epoch and maximised over up to 30
epochs. This is a best-of-30 selection on the held-out set, so **the resulting pooled AUROC is
optimistically biased** and cannot be treated as a fresh, unbiased generalisation estimate — the
same caveat this project already applies to best-of-k search elsewhere (`README.md` E5, and by
analogy to the Ferrite best-of-32 caveats). Intermediate-epoch weights were not saved, so this
bias cannot be removed without retraining with a proper three-way split (train / checkpoint-dev /
frozen-test); that is flagged in Limits below as a possible R3, not run here.

## Candidate experiments considered

| # | Design | Training runs | Distinguishes transfer vs. site-specific structure? | Cost | Selected? |
|---|---|---|---|---|---|
| 1 | Leave-one-source-out (Hong Kong ↔ Dongguan) | 0 (already run) | Coarse, already answered (H3: 1/3 wins) | none | Already done, not new |
| 2 | **This experiment**: reuse the existing held-out groups; cross-seed rank stability as the primary (low-selection-bias) endpoint, group-level AUROC and a trivial-feature control as secondary (caveated) endpoints | 0 | Directly tests within-source generalisation to unseen sites/trials, and whether it is a repeatable property of the method or seed noise | ~minutes, CPU/GPU eval only | **Yes** |
| 3 | Fresh retrain with sites moved into a frozen, checkpoint-selection-blind test split (removes the selection bias) | 4–6 (2–3 seeds × ~2–2.5 h each, from R1's wall times) | Cleanest possible version of #2 | hours of GPU | No — held for a possible R3 if #2 is ambiguous |
| 4 | Leave-one-material-out within Dongguan (ductile iron / PE / steel) | 3+ seeds × 3 materials | Tests a different confound (pipe material, not site) | hours of GPU | No — separate question, not run |
| 5 | Cross-sensor transfer within Hong Kong (train on noise-logger, test on hydrophone or vice versa) | 1–2 new runs | Tests sensor-general vs. sensor-specific learning | ~1 h GPU | No — separate question, not run |

Ranked by information value, not expected result: #2 and #3 answer the same question with
different rigour/cost trade-offs; #2 is selected now because it needs no new training and its
primary endpoint (rank stability) is not the quantity checkpoint selection optimised for. #4 and
#5 ask genuinely different questions and are left for a future prereg if warranted.

## Hypothesis

**R2.** Model F's ranking of never-trained-on real groups from its own training sources (Hong
Kong, Dongguan) is a stable, seed-independent property, not an artefact of one training run's
initialisation — mirroring the question R1 asked of the Mendeley hydrophone result, but applied to
the training-domain held-out data instead of the out-of-domain target.

This is exploratory and descriptive; there is no H0-style pass/fail gate, for the same reason R1's
addendum used a three-way outcome rather than a binary pass/fail: a null or negative result is
exactly as reportable as a positive one.

## Data, exactly

- `cache_f/bank.npz` as committed at `1551414` (verified against its Git LFS sha256 before use).
- Rows with `s_val == True`: 8 Hong Kong groups (`hk:05569, hk:12835, hk:172499, hk:h1.3, hk:h3.3`
  = leak; `hk:h2.1, hk:h2.3, hk:h2.7` = no-leak) and 79 Dongguan groups (73 leak, 6 no-leak).
- These rows were excluded from every training step of every seed (`Bank.__init__`'s `s_val`
  filter), independently of `--seed` (`is_val_group` does not depend on the seed).
- Checkpoints: `best_model_f_seed0.pt`, `best_model_f_seed1.pt`, `best_model_f_seed2.pt`, exactly
  as trained for the original prereg and its addendum. No retraining.

## Method

1. For each held-out row `i`, build a two-channel window exactly as `train_f.py` builds its real
   validation rows: `A.finish(bank.two_channel(i, rng).astype(np.float64), rng)`
   (`Model_F/augment_f.py`), with **one fixed evaluation seed shared across all three
   checkpoints** (seed 0, matching `experiments/model_f_eval.py`'s default), so any difference
   between seeds' scores reflects the model, not a different random channel-pairing/EQ draw. This
   is the one deliberate deviation from `train_f.py`'s own construction (which uses `--seed + 2`,
   different for each training seed) — needed for a fair seed-to-seed comparison, and is a
   between-checkpoints control, not a new held-out set.
2. Band-limit + joint z-score (`prepare`, identical to E11).
3. `predict_proba(model, x, logits=True)` for each of the three checkpoints (never sigmoid
   probabilities, per `AGENTS.md`).
4. Per held-out group: mean logit.
5. **Primary endpoint (not affected by checkpoint-selection bias):** Spearman correlation of
   per-group mean logits between each pair of seeds (0-1, 0-2, 1-2), separately for Hong Kong and
   Dongguan — the same statistic and structure as R1's Mendeley analysis
   (`tests29-2/spearman_seeds.py`).
6. **Secondary endpoint (descriptive; caveated as optimistically selected):** group-level AUROC
   per seed per source, computed as the fraction of (leak-group, no-leak-group) pairs correctly
   ranked by mean logit (equivalent to Mann–Whitney U on groups, not windows), with a group
   cluster-bootstrap 95% CI (2000 resamples of groups, stratified by class, matching
   `experiments/metrics.py:cluster_bootstrap`).
7. **Control (diagnostic, not the primary contribution):** the same H2/H3-style logistic-
   regression baseline (`experiments/cross_dataset.py:features_1ch`, trained on the *train*-split
   groups of the same source, exactly as `model_f_eval.py:logreg_baseline`), scored identically at
   the group level, to check whether a trivial acoustic feature alone separates these held-out
   groups as well as Model F.
8. Window-pooled AUROC (same construction, informational only) is also reported for continuity
   with the numbers `train_f.py` already computed, explicitly labelled as **inflated by best-of-30
   checkpoint selection** and not used for any claim of generalisation.

No threshold is tuned on this data. No group is added, dropped, or re-defined after seeing a
result.

## Statistical method

- Spearman rho and its exact two-sided p-value (`scipy.stats.spearmanr`), reported per pair per
  source; Hong Kong n = 8 groups, Dongguan n = 79 groups.
- Group cluster-bootstrap 95% CI for the group-level AUROC (`np.random.default_rng(0)`, 2000
  resamples), reported per seed per source, model and logreg baseline.

## Interpretation rules (frozen)

**Primary (per source, Hong Kong and Dongguan reported separately, never pooled):**
- **Stable:** the median of the three pairwise Spearman rho is ≥ 0.5 and all three p < 0.05.
- **Unstable:** the median rho is < 0.3, or at least two of the three pairs have p ≥ 0.05.
- **Mixed:** anything between the two.

**Secondary (descriptive only, no pass/fail):** report each seed's group-level AUROC and CI next
to the logreg baseline's. A model AUROC that does not clearly exceed the logreg baseline's CI is
reported as "no evidence of a non-trivial signal beyond the baseline feature," not as a failure of
a pre-registered gate — there is no gate here.

**What a Stable + high-AUROC result would support:** Model F's real-data behaviour on training
sources is a repeatable property of the method, not seed noise — but it would **not**, by itself,
distinguish a transferable physical leak cue from a consistently memorised site/trial acoustic
fingerprint, since (as the audit found) every held-out group is single-class. That distinction
would require a frozen, selection-blind retrain (candidate #3 above), not this experiment.

**What an Unstable or Mixed result would support:** the seed-sensitivity R1 found for the
out-of-domain Mendeley target extends to Model F's own training-domain held-out data, i.e., the
aggregate real-validation AUROC used for checkpoint selection does not reflect a repeatable,
group-general signal even before asking whether it transfers out of domain.

## Stopping rule

Run once, on the three existing checkpoints and the existing bank, at the fixed evaluation seed
above. No re-running with a different seed if the first result is unfavourable. If a script or
data-loading bug is found before any result is inspected, it may be fixed and the run repeated
once; any such fix is logged.

## Limits known in advance

- The secondary/AUROC endpoint is optimistically biased by checkpoint selection; only the primary
  (Spearman) endpoint is treated as a clean test of stability.
- Hong Kong site labels are perfectly confounded with site identity (single-class sites), as in
  essentially all field leak surveys; this experiment cannot separate "detects the physical leak"
  from "recognises this acoustic environment as one of the leaky-sounding ones," in either
  direction of outcome.
- Dongguan's 79 groups are lab-trial replicates (material × pressure × sensor combinations), not
  geographic sites; they are reported separately from Hong Kong for this reason.
- Dongguan's held-out groups are heavily leak-skewed (73 leak vs. 6 no-leak groups), so its
  group-level AUROC is imprecise on the no-leak side.
- Three seeds bound the spread of the method; they do not tightly estimate it (as R1 already
  notes for Mendeley).
