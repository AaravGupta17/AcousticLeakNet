# AcousticLeakNet Version 2: real-world generalization research program

Written 2026-10-02, before any new training. This is a planning document, not a
pre-registration — it sets direction; individual experiments inside it still get their own
frozen prereg (`docs/*_PREREG.md`) before running, per this project's existing practice.
It does not change any existing claim in `docs/CLAIMS.md`, `docs/INTEGRITY_LOG.md`, or any
Model F result. Dataset and literature findings below come from a five-thread research pass;
full notes and citations are in `research_notes/Acoustic leak detection generalization/` and
`reports/Acoustic leak detection generalization.md`.

## A. Original research question, precisely

Can we build a leak-detection representation that generalizes across independent real-world
water-distribution acoustic/vibration recording domains — different testbeds, sensors, pipe
materials, and sites — without requiring a large labeled dataset collected from each new target
domain?

Restated as the falsifiable version this program actually tests: *does training on synthetic
data plus some real source domains produce a model whose leak/no-leak ranking on a held-out,
previously untouched real domain beats chance and beats a trivial same-source-trained baseline,
at zero or very few (≤10) labeled target examples?*

This is not a new question for the project — it is the question Model F's H2/H3 and the R1/R2
follow-ups already asked. Version 2 does not replace that question; it changes what we do in
response to the answer those experiments already gave.

## B. What previous work actually established

- **Model C/D** reach AUROC 1.000 on three held-out *synthetic* EPANET networks, but peak
  amplitude alone scores 0.92–0.97 on the same data (E2) — the synthetic task has a shortcut.
  Zero-shot transfer to real Mendeley data is at chance (AUROC ≈ 0.5, E4). Synthetic pretraining
  made real-data label efficiency *worse*, not better (E5, negative transfer).
- **Model E** fixed two concrete physics bugs (circular vs. linear delay; attenuation calibrated
  to measured Sheffield MDPE data instead of an arbitrary constant), closing specific
  shortcut/realism gaps without yet testing transfer.
- **Model F** redesigned the data pipeline (loudness removed via joint z-score, 2 kHz band
  limit, real labeled windows mixed into training, fixed loss weights, checkpoint selection by
  detection AUROC) specifically to remove the amplitude/noise-texture shortcuts E1–E10 diagnosed.
  Its own pre-registered tests (H0–H4, `docs/MODEL_F_PREREG.md`) and two follow-ups are the most
  important inputs to this program:
  - **H2** (zero-shot transfer to Mendeley Looped, a real domain excluded from training): failed
    on both sensors in the official seed-0 run. The one borderline hydrophone result (AUROC
    0.630) turned out to sit at its bootstrap CI lower bound of ~0.5 within numerical noise, and
    **R1** (seed replication) found checkpoints trained with different real sources ranked the
    same held-out recordings very differently (`f_nohk` 0.197 vs. `f_nodg` 0.705) — i.e., not a
    reproducible property of the method.
  - **R2** (held-out-group stability on Model F's *own* training sources, Hong Kong/Dongguan)
    found the ranking of never-trained-on groups *is* stable across seeds (Spearman ρ median
    0.67–0.98) and beats a trivial spectral-feature baseline on Hong Kong (1.000 vs. 0.800). But
    by design this cannot separate "detects the physical leak" from "recognizes this acoustic
    environment," because **every Hong Kong site and Dongguan trial group is single-class** — a
    site was monitored during a leak or it wasn't, never both. The secondary AUROC numbers are
    also inflated by best-of-30 checkpoint selection on that same held-out set.
  - Cross-dataset leave-one-source-out evaluation (E9) has repeatedly landed near chance to
    modestly above it, never a clear, stable win.
  - The project's own current manuscript framing (`docs/CJSJ_HUMAN_REVIEW_CHECKLIST.md`) states
    the honest conclusion plainly: *"none of \[the model's] apparent real-data results survive
    seed replication or a simple baseline check."*

**The established result, stated once and not re-litigated:** this project has already built
and rigorously tested a synthetic-to-real transfer pipeline, with correct statistics (bootstrap
over recordings, not windows; AUROC on logits; multi-seed replication), and it does not
transfer. That is real, reportable, pre-registered work — not a mistake to route around quietly.

## C. What remains unsolved

1. **The actual target question** — zero/few-label transfer to a genuinely unseen real domain —
   has never been attempted with a method built for it. Model F is a better *synthetic* model
   with some real data mixed into training; it was never given a domain-generalization or
   domain-adaptation objective (no CORAL/MMD/DANN-style alignment loss, no explicit few-shot
   calibration step).
2. **The single-class-per-domain confound** (every real site/trial is all-leak or all-no-leak)
   is a structural property of every real source this project has, and — per the literature
   review below — of essentially every field leak survey. No experiment has yet asked what a
   method *designed around* this confound (rather than one that assumes balanced domains) would
   do.
3. **Checkpoint-selection bias** in the held-out real groups (R2's caveat) has never been removed
   by a proper three-way split (train / checkpoint-dev / frozen test).
4. Whether **some* labeled target data (not zero) resolves transfer, and how much, is untested
   here — Model F's design never included a calibration/fine-tuning step.

## D. Real dataset inventory

Full detail, every citation, and explicit confirmed/unconfirmed flags are in
`reports/Acoustic leak detection generalization.md`. Headline result: **no new, independently
downloadable, both-class real water-leak dataset surfaced** beyond the four already in use.

| Dataset | Status | Verdict |
|---|---|---|
| Mendeley Aghashahi/Sela/Banks PVC testbed | Already used | Confirmed origin (DOI 10.17632/tbrnp6vrnj.1); ≥2 earlier companion Mendeley DOIs from the *same* testbed exist — not new data. |
| Hong Kong (noise logger + hydrophone) | Already used | Confirmed match to Tijani et al., Mendeley `hkn8mxcjyz`, ~90 sites/12 months. Every site single-class. |
| Dongguan (lab trials) | Already used | — |
| Sheffield MDPE rig | Already used, calibration only | Confirmed (Shekofteh et al., *Sensors* 2026); single-material, not a leak/no-leak classification set. |
| Dongguan Zenodo release (DOI 10.5281/zenodo.18631450, Feb 2026) | **Unresolved duplicate risk** | Same city, same "training base" framing as the existing Dongguan source. CC BY 4.0, 500 leak/386 no-leak/114 noise clips, currently downloadable. **Action: diff this DOI against the project's current Dongguan citation before counting it as a 5th domain.** |
| Sydney "Lift and Shift" utility deployment (multi-year, 6 metro areas, >94% reported CNN accuracy, ~70% real pre-existing leaks) | **Not available** | Papers state the raw data "cannot be made publicly available due to confidentiality." Access only via contacting the authors — worth a low-cost email, not a dependency. |
| AI Hub (Korea) corpus (~30,000 cases, 11,000+ sensor locations, Gwangju/Goheung) | **Promising but degraded/unverified** | Largest distinct real lead found. Published as FFT magnitude spectra, not raw waveform (loses the two-channel timing structure this project's architecture is built around); portal access/license unverified (403 on fetch). Timebox a access check to ≤2 days; do not plan around it. |
| GPLA-12 (gas pipeline acoustic emission) | Open, real, downloadable (GitHub + arXiv:2106.10277) | **Different medium** (compressible gas-flow orifice acoustics, no hydrophone/liquid coupling) — useful only as an architecture/pipeline testbed, not as water-domain validation evidence. |
| Several Chinese-city datasets (HZ city, Jiangsu-Zhejiang-Shanghai, WDN sets) | **Unconfirmed** | Surfaced only as search-snippet fragments inside a paywalled survey/paper; not verified as real, accessible, or distinct. Do not cite or plan around these without reading the primary PDF. |

**Implication:** the field appears to systematically withhold or under-release real leak-acoustic
data (confidential utility data, or single-use datasets nobody re-releases), not merely to be
hard to search. This project's existing four-domain set is close to the practical ceiling of
what's publicly obtainable right now. **Data acquisition is not the critical path for October.**

## E. Cross-domain benchmark design

The benchmark this program needs is **already mostly built**. Model F's H3 ablations
(`f_nohk` excludes Hong Kong, `f_nodg` excludes Dongguan) plus the main `f` checkpoint (which
already excludes Mendeley from training entirely, per `docs/MODEL_F_PREREG.md`) are exactly the
three leave-one-real-domain-out rotations this program needs:

| Held-out (test) domain | Trained on | Existing checkpoint |
|---|---|---|
| Mendeley (Looped, multi-class) | synthetic + Hong Kong + Dongguan | `best_model_f_seed{0,1,2}.pt` (the main `f` models) |
| Hong Kong (single-class sites) | synthetic + Dongguan | `f_nodg` |
| Dongguan (single-class trials) | synthetic + Hong Kong | `f_nohk` |

**What's new is the method layered on top, not the rotation structure.** Three additions:

1. **Fix the checkpoint-selection bias** (R2's flagged gap): for each rotation, split the
   *retained* real sources' held-out groups into a small **checkpoint-dev** slice (used only for
   epoch selection) and a separate **frozen-test** slice (touched exactly once, at the end). This
   was never done — R2's AUROC numbers are explicitly the best-of-30-on-the-same-set the project
   already flags as inflated.
2. **A domain-alignment loss** (CORAL/MMD, see F/G below) computed between synthetic features and
   each real source's features — class-conditional where both classes exist (synthetic, and
   Mendeley Looped, which — unlike Hong Kong/Dongguan — does have a small multi-class real
   sample, ~4 no-leak + leak recordings per topology), marginal-only where they don't (Hong Kong,
   Dongguan).
3. **A k-shot calibration step** on the held-out domain: 0 (zero-shot, what Model F already
   tested), then 1, 5, and 10 labeled target recordings, used only to recalibrate a frozen
   feature extractor (linear probe or BatchNorm affine recalibration — not full fine-tuning,
   given how little target data exists).

Train/validation/test discipline: no held-out domain's frozen-test labels are used for
architecture choice, hyperparameters, threshold, or checkpoint selection at any k, including
k≥1 (the k labeled shots used for calibration are a *disjoint* small sample from the frozen-test
set, never the same recordings).

## F. Literature review (condensed; full citations in the report)

**Domain generalization / adaptation.** DANN, Deep CORAL, and MMD-style alignment have direct,
reusable precedent on vibration time-series (not just vision) via the public, maintained
`TL-Fault-Diagnosis-Library` (CWRU/MFPT/PU/XJTU/IMS/JNU bearing datasets, cross-dataset
protocols) — the single best "applicable code + benchmark" resource found, and this project's
most defensible starting point over porting vision-benchmark DG code. **IRM is specifically
flagged against**: it provably needs more environments than feature dimensionality to recover
true invariance, and can select a lower-empirical-risk *non-invariant* solution instead
(Rosenfeld & Ravikumar; Kamath et al.) — with at most 3–4 "environments" (synthetic + 2–3 real
sources), this project sits in IRM's documented failure regime.

**The load-bearing finding for method selection**: every DG/DA method surveyed assumes each
source/target domain mixes both classes, so its alignment loss is computed class-conditionally.
This project's real sources violate that outright (single-class sites). No source found studies
DG/DA under this exact confound. The inferred, not directly sourced, implication: class-
conditional alignment is only valid on the synthetic domain (and the small Mendeley Looped
sample); single-class real domains can only be targets for *marginal* statistics matching
(CORAL/MMD on pooled features, or AdaBN-style batch-norm recalibration). **Tent and SHOT
(entropy minimization / pseudo-labeling) are flagged as actively dangerous here**: both
reinforce a model's own confident predictions within a batch, and an all-no-leak target domain
has no counter-signal to correct a confidently wrong "leak" prediction. This is a mechanical
consequence of their definitions, not a hypothesis — treat it as a hard constraint on method
choice, not a tuning knob.

**Self-supervised / foundation-model pretraining.** Thin, mostly indirect evidence for
speech/music-derived architectures (SSAST, BEATs, wav2vec2, CLAP) on industrial acoustic/vibration
tasks. Stronger, narrower precedent for AudioSet-pretrained CNN embeddings (VGGish/YAMNet/PANNs)
in bearing/gear/weld-defect classification. The one concrete piece of evidence that pretraining
helps *genuine cross-dataset* (not just within-dataset) transfer in a vibration task is
FreqCondNorm (82.1% zero-shot accuracy on MFPT after joint pretraining across five public
predictive-maintenance datasets) — a single abstract-level claim, not independently audited
against a from-scratch baseline. **No paper was found that pretrains self-supervised
representations on synthetic/simulated data and tests transfer to real recordings** — this
project's exact train/test structure is an open gap in the literature, not solved territory to
import. Given this project's real audio volume (hundreds of recordings, not thousands) and GPU
budget, a from-scratch audio-foundation-model build is not a credible October deliverable; it is
noted as a stretch/future direction, not a near-term candidate.

**Analogous-domain cross-dataset case studies — the most important finding of the whole review.**
Every field checked (bearing/gearbox fault diagnosis, structural health monitoring, pipeline leak
detection itself, underwater acoustics, seismology) shows the same shape: **zero-label transfer
to an independently collected rig collapses**, and every published fix needed at least some
labeled or auto-labeled target access.
- Bearing diagnosis: CWRU→Paderborn with no adaptation ≈ 54% accuracy; +5 labeled target
  examples/class → 96.5%.
- Underwater acoustics (closest structural analogue: passive acoustic, few-class, independently
  collected train/test corpora): USS8→ShipsEar zero-shot collapses from 96.35% to 15.31%,
  recovering only to ~52% with feature-alignment + margin loss — still ~45 points below in-domain.
- Pipeline leak detection itself (the most directly relevant precedent): a global multi-region
  model beats a single-region model by only ~2 points; fine-tuning to 96–97% accuracy still
  needs roughly half the target domain's labels, not zero.

**Conclusion for this project's planning, stated plainly:** nothing in this literature supports
a path to strong *zero*-label cross-rig transfer. Treating zero-shot failure as the expected,
field-consistent outcome (not a pipeline flaw) is itself scientifically defensible and matches
this project's existing evidence-integrity practice. The realistic, literature-supported target
is **few-shot** (1–10 labeled target recordings), not zero-shot.

## G. Top 3 candidate strategies

Ranked by (1) likelihood of genuine effect, (2) fit to this project's actual data (small,
confounded, single-class-per-site real sources), (3) feasibility in the time available.

1. **Marginal feature-statistics alignment (CORAL/MMD) across all sources, safe under the
   single-class confound.** Add a CORAL or MMD loss between pooled features of synthetic
   (balanced) and each real source (class-conditional only where both classes exist — synthetic
   and Mendeley Looped; marginal-only for Hong Kong/Dongguan). Directly reuses a validated,
   adjacent-domain (vibration, not vision) open-source implementation
   (`TL-Fault-Diagnosis-Library`). Cheapest to implement and the only method class in this review
   with both a working reference implementation on similar data *and* no documented failure mode
   under this project's specific confound (unlike IRM, Tent, SHOT).
2. **k-shot calibration on the held-out domain (1/5/10 labeled target recordings).** Directly
   operationalizes the literature's actual success stories (bearing few-shot, pipeline
   fine-tuning). Keep the adaptation itself cheap and low-capacity (linear probe or BatchNorm
   affine recalibration on frozen features) to avoid overfitting ~10 real recordings, consistent
   with the field's preference for parameter-efficient adaptation over full fine-tuning on small
   industrial datasets.
3. **Joint multi-source pretraining (synthetic + every available real source simultaneously,
   FreqCondNorm-style) instead of the current practice of fully excluding the eventual test
   domain from all pretraining.** Exploratory/stretch: the one encouraging data point
   (FreqCondNorm's 82.1% zero-shot MFPT result) came from pooling five heterogeneous datasets in
   pretraining, more than this project currently does. Higher novelty, less certain payoff, and
   only pursued after 1–2 show results, given the time budget.

Not selected, with reasons: full self-supervised foundation-model pretraining (no synthetic→real
transfer evidence found, infeasible compute/data in this timeframe); IRM (documented failure mode
at this project's environment count); Tent/SHOT/pseudo-labeling as a target-only adaptation step
(mechanically unsafe on single-class domains, as argued in F).

## H. Recommended flagship architecture

Keep Model F's existing dual-channel CNN encoder — capacity was never the diagnosed problem
(the same encoder reaches 0.971 AUROC trained and tested within Dongguan). Add, not replace:

- A **CORAL/MMD alignment head**: a linear projection of the pooled penultimate-layer features,
  with a covariance-matching (CORAL) or kernel (MMD) loss computed per-batch between synthetic
  and each real source present in that batch, weighted against the main detection loss.
- A **frozen-feature calibration interface**: the existing classifier head replaced, after the
  main training run, by a small linear probe (or just BatchNorm running-statistics
  recalibration) that can be refit on k labeled target recordings without touching the encoder —
  this is what makes the k-shot protocol in E/J cheap and low-risk of overfitting.
- No architecture change for the encoder/fusion mechanism itself (the existing cross-channel
  gate, already tested to be time-constant and not a timing mechanism, is unaffected by this
  work and out of scope for Version 2).

## I. Recommended training pipeline

1. **Phase 1 — synthetic base training**, unchanged from Model F's existing synthesiser
   (Model E physics, Model F's loudness/band-limit/z-score augmentation).
2. **Phase 2 — joint multi-source training with alignment loss**: for each leave-one-domain-out
   rotation, train on synthetic + the two retained real sources, with the CORAL/MMD loss from H
   added to Model F's existing detection + label-smoothing loss. Checkpoint selection uses only
   the new checkpoint-dev slice (E), never the frozen-test slice.
3. **Phase 3 — k-shot calibration**: for k ∈ {0, 1, 5, 10}, draw k labeled recordings from the
   held-out domain (disjoint from frozen-test), refit only the linear probe / BatchNorm affine
   parameters, freeze the encoder.
4. **Phase 4 — frozen evaluation**: score the untouched frozen-test slice of the held-out domain
   at each k, report once, no re-running after seeing the result (this project's existing
   stopping-rule discipline, per every `*_PREREG.md` so far).
5. Repeat across the 3 rotations (Mendeley / Hong Kong / Dongguan held out in turn) × 3 seeds.

## J. Recommended evaluation protocol

- **Primary metric**: AUROC on logits (never sigmoid probabilities — this project's own
  documented saturation bug, `docs/INTEGRITY_LOG.md` #12), bootstrapped over **recordings/groups**
  (never windows), exactly as `experiments/metrics.py` already does for R1/R2.
- **Secondary**: AUPRC, detection rate, false-alarm rate at a fixed pre-registered threshold.
- **Baselines required for every rotation and every k**: RMS/loudness baseline
  (`baselines/`/`experiments/features.py`), spectral logreg baseline
  (`experiments/cross_dataset.py:features_1ch`, already built), current Model F (no alignment
  loss, no k-shot calibration) as the "prior approach" control, and the new model.
- **Pass/fail rule, pre-registered before running** (see M for the exact criterion).
- **3 seeds per rotation**, matching R1's existing discipline, reported as a range/stability
  statistic, not a single number.
- Every run still goes through `record_run()` into `results/runs/`, same as every existing
  experiment.

## K. Compute requirements

- Synthetic generation: CPU-bound, already cached (`cache_f/`); no new cost.
- Each Model-F-scale training run: ~2–2.5 GPU-hours (observed wall time from R1). Adding a
  CORAL/MMD loss term is a per-batch covariance/kernel computation on already-computed features —
  negligible additional cost (<10%).
- Grid: 3 rotations × 3 seeds × ~2.5 h ≈ **22–25 GPU-hours** for Phase 2 alone. Phase 3 (k-shot
  calibration: linear-probe/BatchNorm refits on ≤10 recordings) is minutes per run, not hours.
  Total new compute for the core program: **well under 40 GPU-hours**, feasible on the project's
  existing single-GPU CUDA setup within October.
- Candidate 3 (joint multi-source pretraining, stretch) adds roughly one more full grid if
  pursued — budget it only after Candidates 1–2 produce a result, per the directive's own "don't
  build 20 models" instruction.
- No budget allocated to large-scale self-supervised pretraining from raw audio — data volume
  (hundreds, not thousands, of real recordings) and literature evidence both argue against it as
  a near-term investment.

## L. October execution plan (today is 2026-10-02; IRIS is 2026-10-03, INSEF is 2026-10-25)

- **Oct 2–3 (today/tomorrow): IRIS.** Untouched by this program — submit the existing, honestly
  negative Model F / R1 / R2 result as planned. Do not let Version 2 planning delay or alter it.
- **Week 1 (Oct 4–10): infrastructure.** Resolve the Dongguan-Zenodo duplicate question (diff
  DOIs); timebox a 1–2 day AI Hub Korea access check (not a dependency if it fails); build the
  checkpoint-dev/frozen-test three-way split for each real source; port a CORAL/MMD loss module
  from `TL-Fault-Diagnosis-Library`'s approach into this codebase; pre-register the Version 2
  benchmark experiment (hypotheses, pass/fail rule, stopping rule) before any training, per
  existing project practice.
- **Week 2 (Oct 11–17): Phase 2 training.** Run the 3-rotation × 3-seed alignment-loss grid
  (Candidate 1).
- **Week 3 (Oct 18–24): Phase 3 + analysis.** Run k-shot calibration (0/1/5/10) on each held-out
  domain; compute all pre-registered statistics; write up results exactly as they come out,
  whichever direction they point.
- **Oct 25: INSEF.** Report the Version 2 result honestly — a k-shot-only success, a partial
  pass, or a replicated null result are all legitimate, reportable outcomes (this mirrors the
  project's own existing practice with R1/R2 and the directive's explicit instruction that a
  "zero-shot fails, 5-shot works" finding is scientifically valuable on its own).
- **Oct 26 onward (toward NASA Space Apps Nov 15, ArXiv preprint, JEI rolling):** deepen whichever
  candidate showed signal; consider Candidate 3 (joint multi-source pretraining) as a follow-up
  only if Candidates 1–2 show a clear positive direction worth extending.

This timeline does not promise a solved cross-domain generalization result by INSEF — the
literature review's own conclusion (F) is that no field has solved zero-label transfer, so
promising it here would repeat exactly the kind of pre-run-expectation-stated-as-result error
`docs/INTEGRITY_LOG.md` already corrects twice. The plan is scoped to produce a real,
pre-registered test of the best available methods within the time and compute actually available.

## M. Definition of success (fixed before any Version 2 run)

**Meaningful cross-domain real-world generalization** = for **at least 2 of the 3**
leave-one-domain-out rotations (Mendeley / Hong Kong / Dongguan held out in turn), at **some**
k ∈ {0, 1, 5, 10}: the new model's AUROC 95% CI lower bound (bootstrapped over recordings/groups)
exceeds **both** (a) 0.5 and (b) the matched-data logistic-regression baseline's CI upper bound,
**replicated across all 3 seeds**, on the frozen-test slice never used for checkpoint or
threshold selection.

- A rotation that only clears this bar at k=5 or k=10 (not k=0) is reported as a **k-shot
  success**, explicitly distinguished from zero-shot success, and is still a positive,
  publishable result.
- If no rotation clears this bar at any k, the result is reported as **"negative result
  replicated under the new method"** — consistent with, not contradicting, this project's
  existing honest reporting of Model F's own H2/H3 outcomes.
- No numeric threshold here was chosen to guarantee a pass; it mirrors the CI-lower-bound
  structure R1/R2 already use.
