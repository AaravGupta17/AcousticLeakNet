# Model F pre-registration, addendum 1: seed replication (R1)

Written on 29 Sep 2026, **after** the seed-0 results of `docs/MODEL_F_PREREG.md` were known and
**before** any model with another seed was trained. The decision to run this replication was made
because of the seed-0 hydrophone result, so the motivation is post hoc; the rules below are fixed
before the new runs exist and are reported whether they pass or fail.

This addendum does not change `docs/MODEL_F_PREREG.md`, its hypotheses H0–H4, its pass rules or
its verdicts. H2 is defined on seed 0 and its official result is the committed GPU run
`results/runs/2026-09-29_000514_e11_model_f_eval.json`: **fail** on both sensors. Nothing in this
addendum can turn H2 into a pass. The result of this addendum is reported separately, as R1.

## Why

Seed 0 of `f` on `mendeley_looped_hyd`: AUROC 0.630, 95% CI lower bound 0.49997 (official GPU
run). A CPU re-run of the same evaluation gave a lower bound of 0.50006, so the seed-0 bound is at
0.5 within numerical noise. Checkpoints trained with different real sources rank the same
hydrophone recordings very differently (`f_nohk` 0.197, `f_nodg` 0.705). With one seed it is not
known whether 0.630 is a property of the method or of one training run.

## Question

**R1.** Does the Mendeley Looped hydrophone result of `f` reproduce when only the training seed
changes?

## What changes and what does not

Only `--seed` changes: seeds 1 and 2. In `Model_F/train_f.py` the seed sets

- the network's initial weights and dropout (`torch.manual_seed(seed)`),
- the training mixtures drawn at each step (`MixDataset(..., seed)`),
- the draws that build the fixed synthetic validation mixtures (`seed + 1`),
- the channel pairing and random EQ of the real validation windows (`seed + 2`).

Which recordings and sites are used for training and for validation is set by `cache_f/bank.npz`
and `cache_f/{train,val}`, not by the seed, and does not change.

Held fixed, identical to the seed-0 run:

- the Model F code (`Model_F/`, `experiments/model_f_eval.py` and the modules they import), the
  same as the commit the seed-0 models were trained from;
- `cache_f/bank.npz` and `cache_f/{train,val}` as committed on `main` (Git LFS object ids are the
  sha256 of the contents; they are checked before training);
- all default hyperparameters of `train_f.py`, the checkpoint rule (best mean of synthetic and real
  validation detection AUROC) and the 30-epoch schedule;
- E11 unchanged, with its defaults: evaluation seed 0, 2000 bootstrap resamples over recordings,
  threshold 0 on the logit, same test sets, preprocessing and logreg baseline.

## Commands

    python Model_F/train_f.py --seed 1
    python Model_F/train_f.py --seed 2
    python experiments/model_f_eval.py --ckpt best_model_f_seed1.pt best_model_f_seed2.pt

The checkpoints are `models/best_model_f_seed1.pt` and `models/best_model_f_seed2.pt`. Seed 0 is
not retrained or re-evaluated; its official numbers are the ones in the run record above.

## Decision rule

On `mendeley_looped_hyd`, for each of seeds 1 and 2, take the AUROC and the 95% CI lower bound
exactly as written in the E11 run record (full precision, no rounding):

- **Replicated:** both seeds have AUROC > 0.5 **and** CI lower bound > 0.5.
- **Direction only:** both seeds have AUROC > 0.5, and at least one CI lower bound is ≤ 0.5.
- **Not replicated:** at least one seed has AUROC ≤ 0.5.

Inequalities are strict. The GPU run record is authoritative; no run is repeated to move a value
that lies near 0.5.

## Reported whatever the outcome

- For each seed and both Mendeley sensors: AUROC, CI, detection rate, false-alarm rate, and the
  logreg baseline; the H2 rule applied to each seed, marked descriptive.
- Mean and range over seeds 0, 1 and 2 on both sensors.
- Descriptive, no pass rule: the Spearman correlation of per-recording mean logits between each
  pair of seeds, on both sensors.
- The accelerometer results, with no pass rule.

## Rules for the runs

- Exactly seeds 1 and 2. No further seeds are added because of their results.
- A run that crashes for a hardware or driver reason is restarted from the beginning with the same
  command, and the crash is logged. A run that finishes is used, whatever its result.
- Run records go to `results/runs/`; console logs are kept.

## Limits known in advance

- Three seeds describe the spread of the method; they do not give a tight estimate of it.
- Even a **Replicated** result would not show that the model detects leaks on the hydrophone data:
  the Mendeley no-leak recordings differ from the leak recordings in recording setup and session,
  and simple features that separate them point in opposite directions on the Branched and Looped
  topologies (E4, E6).
- GPU non-determinism alone moves AUROC by about 0.01 (INTEGRITY_LOG #22).
