# Phase 7 post-run audit

Audit date: 6 Oct 2026. Requested on the premise that Phase 7 had been executed.

**Finding: no Phase 7 execution exists on this machine or on the remote. There are no results to
audit.** Every result field below is empty on purpose. Nothing in this file was estimated,
inferred or filled in.

## 1. Execution verification

| check | finding |
|---|---|
| Locked protocol | Commit `d0145b1` on local branch `cross-rig-mechanism` (6 Oct 19:24:51, author ArmaanGuha), parent `224ca5d`. It contains Addendum 1, `xrig_band_intervention.py`, `xrig_band_external.py`, `xrig_mechanism.py`, `mechanism_analysis.json` and INTEGRITY_LOG #26–28. Working files are identical to the commit. **Not pushed:** no remote ref contains `d0145b1`. |
| `results/v2_cross_rig/band_intervention.json` | **absent** |
| `results/v2_cross_rig/band_external.json` | **absent** |
| `results/runs/*v2_xrig_band_*` | **absent**; the newest run record is `2026-10-04_072952_v2_xrig_shift.json` |
| `science/phase7/run_band_intervention.log`, `run_band_external.log` (the commands in Addendum 1) | **absent** |
| `band_intervention_run.log` (5 Oct) | still 0 bytes (INTEGRITY_LOG #27) |
| Score files / figures for Phase 7 | none |
| Bytecode `experiments/__pycache__/xrig_band_*.cpython-313.pyc` | 6 Oct 19:23:48, from a compile-only syntax check (`py_compile`) made while the protocol was being written. That check executes no code. Nothing has been written since. |
| Running Python processes | none |
| Remote (`git fetch --all`) | no new branches or commits since `224ca5d` |
| "Step 9" attribution comparison | **not part of the locked protocol**, and no such output exists. Addendum 1 has no Step 9. |

**Possible explanation, not verified:** the run was made on another machine (for example one
that still has the Mendeley Branched data) and its outputs have not been pushed or copied here.
If so:
- **Commit:** the run record's `git_commit` must read `d0145b1` (and `git_dirty: false`) for the
  run to count as the locked run.
- **Data:** that machine's Mendeley group count must be checked against Addendum 1 §G, which fixed
  Looped-only, 20 groups. A run on 40 groups would be a deviation and must be reported as one.

## 2. Primary results

| direction | method | AUROC | AUPRC | bal. acc | sens. | spec. |
|---|---|---|---|---|---|---|
| M → S | Model F (mean of 3 seeds) | — | — | — | — | — |
| M → S | full spectral logreg | — | — | — | — | — |
| M → S | 3-band logreg | — | — | — | — | — |
| S → M | Model F (mean of 3 seeds) | — | — | — | — | — |
| S → M | full spectral logreg | — | — | — | — | — |
| S → M | 3-band logreg | — | — | — | — | — |
| Dongguan / Hong Kong (4 cells) | 3-band vs full logreg | — | | | | |

**No measurements exist.** The audited Version-2 baseline numbers (`224ca5d`, 40 Mendeley groups)
are **not** substitutes. Addendum 1 §G rules out comparing Phase 7 with them.

## 3. Did Phase 7 help?

- **M → S:** INCONCLUSIVE (not executed)
- **S → M:** INCONCLUSIVE (not executed)
- **External (decision-bearing) arm:** INCONCLUSIVE (not executed)

## 4. Test-set feature selection (unchanged, applies whenever it is run)

- **The status is fixed regardless of outcome.** Phase 7 is exploratory, and the three bands were
  chosen with leak labels from all Mendeley Looped and all Sheffield groups.
- **What the M↔S evaluation cannot do.** It cannot be independent confirmation of the band
  hypothesis, and a strong result on those datasets does not remove this limitation.
- **The only decision-bearing evaluation** is the no-refit Dongguan/Hong Kong scoring, under the
  rule in Addendum 1 §E.

## 5. Mechanism prediction

- **Not assessable:** there is no Phase 7 measurement.
- **Still stands:** the OBSERVED / SUPPORTED / UNTESTED breakdown in `PHASE7_AUDIT.md` §11.

## 6. Attribution comparison (full 13/14-feature vs 3-band)

- **Not triggered.** It is not part of the locked protocol, and no output exists.
- **The data it would need is already planned.** The locked run reports the full spectral logreg
  and the 3-band logreg side by side on identical fit and test data in every cell, which is the
  comparison the request describes. Any further attribution analysis would be a new, unlocked
  analysis.

## 7. Decision

There is no new evidence, so the decision rests on what existed before:
- the pre-registered Version-2 collapse;
- the audit's finding that the M↔S arm is circular.

Two legitimate options:

1. **Run the locked protocol exactly once** (the two commands in Addendum 1, at commit `d0145b1`).
   - **Cost:** about an hour on CPU, plus GPU inference for Model F.
   - **Outcome:** its rule decides the band question once and for all.
2. **Freeze AcousticLeakNet now without running it.** This is also defensible. The audit already
   showed that the best possible outcome adds an exploratory paragraph, not a new claim. The
   locked protocol stays in the repository as a documented, unexecuted follow-up.

Either way, **no band/feature engineering beyond the locked protocol** is justified.

## 8. Limitations of this report

- **Scope:** it verifies only this machine and the remote.
- **Outputs held elsewhere:** if outputs exist on another machine, this report must be replaced
  by an audit of those files. Their provenance must first be checked against `d0145b1` and
  Addendum 1 §G.

---

PHASE 7 VERDICT: NOT EXECUTED — no Phase 7 output exists on this machine or the remote; nothing to audit.
NEXT ACTION: Either supply the outputs from the machine where Phase 7 was run (`band_intervention.json`, `band_external.json`, both run logs, both run records) for audit, or run the two locked commands once at commit `d0145b1`. If neither happens, freeze AcousticLeakNet and move to Ferrite.
