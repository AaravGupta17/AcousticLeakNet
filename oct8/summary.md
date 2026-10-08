# 8 Oct 2026 summary: V2 three-band cross-rig experiment

- **Status:** exploratory. The bands were chosen using both rigs.
- **Full write-up:** `REPORT.md`, a copy of `docs/V2_THREE_BAND_CROSS_RIG_RESULTS.md`.
- **Per-group tables:** `per_group.md`.
- **Figure:** `fig_per_group.png` shows each group's mean decision value, with the frozen threshold
  drawn in.

## Setup

- **Rigs:**
  - Mendeley accelerometer: 40 groups (32 leak / 8 no-leak).
  - Sheffield: 11 groups (8 leak / 3 no-leak).
- **Features:** log10 normalised PSD fraction in 400–600, 600–800 and 1600–2000 Hz.
- **Classifier:** StandardScaler + LogisticRegression(C=0.5, balanced).
- **Protocol:** the locked Version 2 group splits. The threshold is frozen on the training rig's
  validation groups.

## Results

Window AUROC with 95% group-bootstrap CIs:

| Direction | 3-band LR | Full spectral LR | Model F (3 seeds) | 3-band − full (paired) |
|---|---|---|---|---|
| Mendeley → Sheffield | 0.801 [0.687, 0.906] | 0.738 [0.630, 0.939] | 0.345 (range 0.293–0.374) | +0.064 [−0.051, +0.122] |
| Sheffield → Mendeley | 0.506 [0.355, 0.653] | 0.473 [0.369, 0.574] | 0.525 (range 0.515–0.535) | +0.033 [−0.126, +0.182] |

## Main findings

- **No gain over the full spectral LR.** The paired CI includes 0 in both directions.
- **Sheffield → Mendeley:** every method is at chance.
- **Mendeley → Sheffield:**
  - The 3-band model is at chance on its own Mendeley validation groups (0.498).
  - Its frozen threshold flags 0 of the 8 Sheffield leak groups, even though its group ranking is
    perfect.
- **Individual bands vs the fitted model:**
  - Each band is higher in leaks on both rigs.
  - log RMS reverses its leak direction between the rigs.
  - The fitted model puts a negative weight on 600–800 Hz in both directions, and the band
    fractions shift by up to 1.7 SD between the rigs. So the combined representation is not
    invariant.

## Verdict

**C: null result.**

**FREEZE AND WRITE.** Any further band or normalisation variant would be chosen after seeing both
rigs' results. A real next step needs a third independent rig.
