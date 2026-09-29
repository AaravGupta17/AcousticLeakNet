# CJSJ human review checklist — AcousticLeakNet

For the author(s) and mentor to work through before submission. Source draft:
`private/cjsj/draft.md` (gitignored; not in the repository history). Written 2026-09-30 —
**today is the tracked deadline** for this journal cycle (see the draft's own header;
[VERIFY] the exact time on CJSJ's site).

## A. Scientific correctness

- [ ] Can you state the paper's contribution in one sentence? (Suggested: "We audited a
      synthetic-to-real acoustic leak detector end to end and found that none of its apparent
      real-data results survive seed replication or a simple baseline check.")
- [ ] Does every number in the draft still match its cited evidence file? (Spot-check at least
      the R1/R2 numbers in §3.6–3.7 against `tests29-2/SUMMARY.md` and
      `docs/ACOUSTICLEAKNET_R2_RESULTS.md` — these are the newest additions and haven't had a
      second pass.)
- [ ] Does the Results section follow initial result → seed replication → held-out-group
      analysis → baseline → confound → constrained conclusion, in that order? (It does in the
      current draft — confirm it still does after any trim.)

## B. Claim scope

- [ ] Does any sentence say or imply the model "detects leaks," "generalizes," or "works in
      real-world conditions" without a hedge? (Automated scan in this pass found none — but
      re-scan after any editing, since a trim pass is still needed and could reintroduce one.)
- [ ] Does the title avoid implying a working detector? (Current: "What a High Score Doesn't
      Prove: Auditing a Synthetic-to-Real Acoustic Leak Detector" — investigative, not
      sensational.)
- [ ] Does the abstract make the central limitation (source/site confounding; seed instability)
      impossible to miss in the first read?
- [ ] Is the distinction between "classification performance / separability" and "transferable
      leak-specific detection" stated explicitly at least once in Results and once in Discussion?

## C. R1/R2 interpretation

- [ ] Is R1 described as testing whether a *single seed's* result is reproducible, not as a
      second independent experiment?
- [ ] Is R2's AUROC explicitly labeled as optimistically biased (checkpoint selection used the
      same held-out set), with Spearman stability as the actual primary endpoint?
- [ ] Is it clear that R1 and R2 answer *different* questions (out-of-domain seed stability vs.
      in-domain held-out-group stability), not the same one twice?

## D. Confounding / limitations

- [ ] Does the text say "source- or environment-specific acoustic structure," not "geographic
      multi-site validation," for Dongguan?
- [ ] Is it explicit that Hong Kong sites and Dongguan trial groups are each single-class, so
      site identity and leak label are perfectly confounded — this is a property of the *data*,
      not something R2 "found" causally?
- [ ] Does Limitations avoid apologizing, and avoid proposing a new experiment? (Both are
      currently satisfied.)

## E. Figures

- [ ] **Fig. 4 (R1 seed comparison) does not exist yet.** No script currently generates it. Build
      it from `results/runs/2026-09-29_193019_e11_model_f_eval.json` and
      `results/runs/2026-09-29_000514_e11_model_f_eval.json` before the figures deck is final.
- [ ] **Fig. 2's current candidate plots are stale** (`plots/exp2_mendeley_eval.png` and three
      others, all dated 13 Aug — before the 23 Sep E4 rerun this draft actually cites). Regenerate
      from `results/runs/2026-09-23_202146_e4_mendeley_eval.json` before using any figure titled
      "Mendeley" from `plots/`.
- [ ] Fig. 1 and Fig. 3 (`plots/e3_snr_sweep.png`, `plots/e5_label_efficiency.png`) exist at
      usable resolution (2250×1200 and 1050×675) — confirm they still match the numbers quoted in
      §3.1/3.4 before reuse.
- [ ] Decide whether a 5th figure (R2 group-AUROC vs. baseline) is worth the page budget, or
      whether the in-text table in §3.7 is enough — see the draft's own note.
- [ ] Every figure: axis labels with units, legible font at print size, caption below the figure,
      no school/city name anywhere in a plot title or axis label.

## F. Writing

- [ ] The draft is currently ~2,500 prose words (tables excluded); the CJSJ template at this
      density fits roughly 1,300–1,700 words in 2–3 pages. **A trim pass is required** — see the
      `(TRIM?)` markers in `private/cjsj/draft.md` (the E1–E10 methods table and the Discussion's
      second paragraph are the two largest candidates).
- [ ] Read titles/headers for consistent capitalization and terminology (e.g., always "Model F,"
      always "R1"/"R2," not mixed with "replication study" elsewhere).
- [ ] No passive-voice results sentences that hide which experiment produced a number.

## G. Submission metadata

- [ ] Author names, order, school, city, graduation year, contact — all still `[CONFIRM]`
      placeholders in the draft.
- [ ] Mentor/PI name and title — `[CONFIRM]`.
- [ ] Mentor/PI signature on `~/Downloads/CJSJ+Permission+to+Publish+Form.pdf`.
- [ ] Run `scripts/make_submission_copy.py` (uses `private/forbidden_terms.txt`) on the final
      files before upload, to confirm no school/city name leaked into the paper or figures.
- [ ] File names on submission: `LastnameFirstname_paper.docx`, `_figures.ppt`, `_form.pdf`.
- [ ] Code-availability link: decide whether a public copy of the repository (without exposing
      `private/`) will be provided, and what to say if not.

## H. AI disclosure

- [ ] The draft's AI-use disclosure paragraph is factually accurate as of today: it covers R1,
      R2, and this drafting/audit pass, and states that prompts behind the earlier E1–E10
      diagnostic work are not confirmed to be retained. **Do not let this be softened or
      generalized away** in later edits — CJSJ requires the full prompts and tool/version, and
      requires disclosure in the cover letter as well as the manuscript.
- [ ] Confirm whether historical prompts (E1–E10 era) actually exist before the cover letter
      claims either that they do or that they don't.

## I. Author/mentor sign-off

- [ ] At least one author has read every sentence of the final draft against its cited source.
- [ ] The mentor has read the Results/Discussion/Limitations and confirms the framing (finding,
      not failure report) is accurate to their understanding of the work.
- [ ] Final `git status`/diff review confirms no scientific file (checkpoints, `results/`,
      `docs/MODEL_F_PREREG*.md`, `docs/ACOUSTICLEAKNET_R2*`) was altered during manuscript work.
