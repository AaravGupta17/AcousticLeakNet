# Phase 7 summary: three-band normalised-PSD cross-rig test

- **Data:**
  - Mendeley: 40 groups (Looped + Branched).
  - Sheffield: 11 groups.
- **Protocol:** the locked Version 2 cross-rig protocol.
- **Features:** log10 PSD fraction in 400–600, 600–800 and 1600–2000 Hz.
- **Classifier:** StandardScaler + LogisticRegression(C=0.5, balanced).
- **Pre-specification:** `docs/PHASE7_THREE_BAND_PREREG.md`.
- **Full write-up:** `REPORT.md`.

Window AUROC with 95% group-bootstrap CIs:

| Direction | 3-band LR | Full 14-feature LR | Model F (mean of 3 seeds) | 3-band − full (paired) |
|---|---|---|---|---|
| Mendeley → Sheffield | **0.801** [0.687, 0.906] | 0.738 [0.630, 0.939] | 0.345 [0.293–0.374] | +0.064 [−0.051, +0.122] |
| Sheffield → Mendeley | **0.506** [0.355, 0.653] | 0.473 [0.369, 0.574] | 0.525 [0.515–0.535] | +0.033 [−0.126, +0.182] |

## Verdict

The result is a partial and partly unexpected outcome. The hypothesis is **not supported** as a
general fix.

- **M→S:** the point estimate improves, but the gain's CI includes 0.
- **S→M:** the score stays in the collapse range.
- **Within Mendeley:** the 3-band model is at chance (validation AUROC 0.498). Its M→S gain is
  therefore not a leak signature learned on the training rig.
- **Selection bias:** the bands were chosen by looking at both rigs, which biases any gain
  upward.

## Integrity

- No protocol violations.
- The frozen Version 2 baseline was reproduced exactly.
- The script ran once, with no tuning after the results were seen.
