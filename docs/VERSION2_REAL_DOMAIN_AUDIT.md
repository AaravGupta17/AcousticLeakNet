# Real-domain audit: choosing the flagship benchmark after the Mendeley correction

Written 2026-10-02, in direct response to your correction that `docs/VERSION2_CRITICAL_REVIEW.md`
was wrong to make Mendeley the flagship target. That correction is accepted without reservation:
Mendeley is a 47 m, 152.4 mm, low-pressure lab PVC loop, and a model that only has to transfer into
it is not answering "does this generalize across real-world municipal/network domains." This
document re-audits every real dataset the project has — including two sources I had not
previously examined at the file level, which change the answer — and proposes a new flagship.

**This document supersedes §3, §4, §5, §9 and §11 of `VERSION2_REAL_DOMAIN_AUDIT`'s predecessor
(`docs/VERSION2_CRITICAL_REVIEW.md`).** It does not touch that document's §1 (literature
verification), §6 (CORAL/MMD vs. alternatives), §7 (representation candidates) or §8 (the
informational-bottleneck argument) — those conclusions do not depend on which real dataset is the
test domain, and are referenced, not repeated, below. No coding has started. Nothing here changes
any existing claim in `docs/CLAIMS.md`, `docs/INTEGRITY_LOG.md`, or any Model F/R1/R2 result.

Method: direct inspection of every file in `datasets/public/` (not descriptions of them) —
including the Hong Kong MEMS accelerometer `.xlsx` files and the Sheffield CID-lab CSVs, neither
of which the prior review opened — plus two targeted web lookups for the one piece of external
metadata (MEMS sampling rate) that no file on disk contains.

## 1. The corrected, per-dataset classification

"Domain" is deliberately not equated with "dataset" here — see §4 for why. This table classifies
each *source* first; sub-domain structure within a source (site, trial, test condition) is spelled
out in the sections that follow.

| Source | A. Physical similarity to municipal target | B. Within-domain leak/no-leak variation | C. Independence | D. Site/label confound | E. Train usefulness | F. Validation usefulness | G. Test-domain usefulness |
|---|---|---|---|---|---|---|---|
| **Hong Kong — Noise Logger** | **High** — real buried municipal pipes, ~90 field sites, standard utility noise-logger equipment | None within a site (every site single-class) | n/a (reference source) | **Total** — 0/32 sites overlap (file audit, §2) | Good — real field acoustic texture, largest real N | Secondary only | **Invalid** as a generalization test; valid only as a diagnostic (can it rank unseen sites of a source it trained on) |
| **Hong Kong — Hydrophone** | **High** | None within a site | n/a | **Total** — 0/18 sites overlap | Good | Secondary only | **Invalid**, same reason |
| **Hong Kong — MEMS accelerometer** | **High** — same real field campaign as the two sources above | **Yes, newly found** — all 5 sites appear in both classes; one site has same-day leak/no-leak recordings (§2) | Same underlying field source as the two above, but a structurally different (unconfounded) labeling pattern | **None at the site level** (pending one engineering blocker, §2) | Good, once unblocked | Good, once unblocked | **Potentially the strongest test domain in the entire project** — currently blocked on sampling rate, not on data structure |
| **Dongguan — full dataset** | **Medium** — a dedicated real outdoor leak-training facility, multiple pipe materials, induced leaks (not a live customer network) | None at trial level, **except** an 8-trial matched subset (§3) | Independent facility, different city/curator from Hong Kong | Confounded by construction, except the 8-trial subset | Good — large N, 4 materials | Full dataset: secondary only. 8-trial subset: a genuine (if small) validation check | Full dataset: **invalid**. 8-trial subset: **conditionally valid**, with the caveats in §3 |
| **Sheffield — CID lab rig** | **Low-medium** — indoor lab pipe loop, but *real* 63 mm MDPE pipe, *real* seismic accelerometers (PCB Piezotronics 393B12), realistic 2.8–4.2 bar operating pressure | **Yes — previously unused for classification** (§5): 8 independent orifice/pressure test conditions, each with its own leak recording(s) and a matched no-leak background | Fully independent rig, institution, sensor hardware and curator from every other source | **Essentially none at the rig level** (small no-leak population, reused across some tests — stated honestly below) | Good — a second genuinely balanced real source | Good | **Valid**, and specifically valuable paired with Mendeley (§6) |
| **Mendeley — Looped** | **Low** — 47 m lab PVC loop, lower pressure/flow than a municipal main (your original objection, accepted) | Yes (already established) | Fully independent rig | None | Good | Good | **Valid, but secondary/stress-test** per your instruction — not the flagship |
| Mendeley — Branched | n/a | n/a | n/a | n/a | **Excluded** (contaminated — used as synthesiser background noise, `AGENTS.md`) | — | — |

Datasets considered and **not added** as a new source (full detail already in
`reports/Acoustic leak detection generalization.md`, not re-litigated here): a confidential Sydney
commercial deployment (not downloadable); a South Korean AI Hub corpus (~30,000 cases, but
released only as FFT magnitude spectra, not waveform, and the portal 403'd on direct fetch); a
2026 Dongguan Zenodo release (unconfirmed whether it duplicates the Dongguan source already in
use — same city, same "training base" framing); GPLA-12 (real, open, but **gas** pipeline acoustic
emission — different physics, usable only as an architecture-transfer reference, not a water
validation domain). None of these change §6's benchmark design; none is a new usable water source.

## 2. Hong Kong is not thrown out — the MEMS files, examined directly

`experiments/public_data.py`'s own docstring already flagged the MEMS `.xlsx` files as unused and
hypothesized they were "before/after recordings with the same sensors." I opened all 20 files
directly (10 Leak, 10 No-Leak) rather than trusting that docstring. They are more useful than the
docstring guessed:

**Site overlap (direct file audit, not inferred):** every one of the 5 distinct sensor-unit IDs
present in this dataset — `A50`, `AD6`, `AD8`, `AD9`, `ACB` — appears in **both** the Leak and the
No-Leak folder. This is the opposite of the Noise Logger/Hydrophone confound: site identity does
**not** determine label here.

**Visit pattern (from filenames, which encode `<unit>_<month>_<day>_<year>`):**

| Site | Visits, chronological (label, date) |
|---|---|
| A50 | No-Leak 2020-11-25 → No-Leak 2020-12-17 → Leak 2020-12-30 → Leak 2021-03-02 |
| AD6 | No-Leak 2020-12-09 → Leak 2021-01-13 |
| AD8 | Leak 2020-12-04 (short, ~4,400 samples) → No-Leak 2020-12-03 *(sic — see caveat below)* → No-Leak 2021-01-05 |
| AD9 | No-Leak 2021-01-13 → No-Leak 2021-02-02 → Leak 2021-04-08 (recorded as two sequential files, same date) |
| ACB | No-Leak 2020-12-23 → No-Leak 2020-12-30 → **Leak 2021-01-26 and No-Leak 2021-01-26 — the same calendar date** → Leak 2021-03-17 (two sequential files) → Leak 2021-03-23 |

This is a real, ~4-month (Nov 2020–Apr 2021) longitudinal monitoring campaign at fixed sites, with
leak status changing over calendar time — consistent with ~90 real municipal sites being visited
repeatedly, not an induced-leak lab protocol. ACB's same-day Leak/No-Leak pair (2021-01-26) is the
single best same-site, minimal-time-gap real comparison available anywhere in this project's data,
better in kind than anything Mendeley or Sheffield can offer, because it is a real field site, not
a rig.

*(Caveat on the AD8 row: the Leak visit, 2020-12-04, precedes both No-Leak visits in the table —
2020-12-03 is one day *before* it, 2021-01-05 is after. Read literally this says a leak was present
on Dec 4 and absent the day before and a month later, which is physically plausible [a leak that
was found and then fixed] but worth independently sanity-checking before relying on this site
specifically; it is not worth more than this one-line flag given the site isn't pivotal to the
argument.)*

**Row counts and the Excel-row-limit explanation.** File sizes and row counts (765,330 up to
1,048,520 data points, i.e. up against Excel's 1,048,576-row ceiling) plus the fact that two
same-date files per site (AD9 2021-04-08; ACB 2021-03-17) have `Data point` indices that pick up
roughly where the other leaves off, indicate these are long (likely multi-hour) continuous
recordings split across files purely because of Excel's row limit — not 20 independent short
clips. One file (`Leak/4. AD8_12_04_2020.xlsx`) is a clear outlier at only 4,410 rows.

**The sampling-rate blocker — now has a specific, named lead, not a dead end.** Two independent
searches converge on the same answer: a companion paper by an overlapping author group (Liu,
Tariq, Tijani, Fan, Abdelmageed, Fares, Zayed — "Data-driven application of MEMS-based
accelerometers for leak detection in water distribution networks," *IWA Water Practice &
Technology* 20(9)) states the accelerometer used was an **AX3-class MEMS unit**, with a documented
maximum sampling rate of **1000 Hz if recording all three axes, or ~3000 Hz if recording a single
axis**. Every file here has exactly one data column (not three), consistent with single-axis
recording at or near that upper rate. **This is a strong, specific, not-yet-fully-confirmed lead,
not a settled fact** — the paper describes the same research group's methodology and cites the same
~90-site Hong Kong campaign, but I could not get full-text access (IWA's site returned HTTP 403;
a related open-access PolyU PDF was unreadable as fetched) to confirm it is describing literally
this released dataset rather than a related deployment. **Concrete Week-1 action**: get full-text
access to that specific paper (library access, a mentor's institutional login, or emailing the
corresponding author) and confirm the rate against it; if confirmed, resample at ~3000 Hz (or
whatever the paper states) and this dataset unblocks immediately — no new data collection required.

**Net assessment**: Hong Kong MEMS is not just "not thrown out" — it is the single most promising
real-domain lead this audit surfaced, because it is the only available source that is both
physically a real municipal field deployment *and* structurally unconfounded by site. It should be
treated as a priority unblock, not a parallel nice-to-have, specifically because unblocking it
would hand the project the one benchmark the original V2 objective actually asked for.

## 3. Dongguan's 8 matched trials — forensic audit, not an assumption

The prior review found these by matching material+zone+sensor, stripped of a pressure/delay field
that is always `NA` on the no-leak side. That alone does not establish the 8 "matches" are
physically comparable, only that they share a filename prefix. This section tests that directly.

**What the pressure/delay field actually encodes.** Leak-folder files record a specific measured
pressure (e.g. "0.31 MPa") and acoustic arrival delay (e.g. "1.16 ms") *per leak trial*. The 8
matched files are leak-folder entries where this field is `NA-NA` instead of a number — the same
material+zone+sensor combinations also have *other*, clearly-genuine leak trials with real pressure
values (confirmed directly: e.g. `ductile iron-zone 2-0.31 MPa-1.16 ms-hydrophone` exists alongside
`ductile iron-zone 2-NA-NA-hydrophone`). So the open question is whether the `NA-NA` leak-folder
files are (a) real leak recordings whose pressure/delay simply wasn't logged, or (b) mislabeled
baseline segments that don't belong in the leak folder at all.

**Acoustic test.** For every one of the 8 matched material/zone/sensor combinations, I computed RMS
amplitude for every available clip in three groups: the ambiguous (`NA-NA`) leak-folder files, the
matched no-leak-folder files, and — where available — the *confirmed* (real-pressure) leak-folder
files sharing the same material+zone+sensor, as a positive control for "what does a verified leak
sound like at this exact physical setup."

| | n (clips, not independent recordings — see caveat) | Mean RMS | Median RMS |
|---|---|---|---|
| Ambiguous (`NA-NA`, leak folder) | 29 | 10,287 | 8,534 |
| No-leak (matched folder) | 299 | 2,997 | 1,953 |
| Confirmed leak (real pressure, same material/zone/sensor) | 359 | 9,937 | 8,753 |

Mann-Whitney U: ambiguous clips are louder than their matched no-leak counterparts with
**p = 1.3×10⁻¹⁴**; ambiguous clips are **not** significantly different from confirmed-pressure leak
clips at the same site (**p = 0.77**). This pattern held in 6 of 7 directly-comparable
material/zone/sensor groups (the one exception, `pe-zone 2/hydrophone`, had the ambiguous group
slightly quieter than its no-leak match, within a small sample).

**Interpretation, stated honestly, both ways.** This is evidence that the 8 matched-trial subset
is acoustically consistent with being genuine leak recordings, not a mislabeling artifact — the
ambiguous files sound like the rest of the leak population at the same site, not like the no-leak
population. **It is not proof.** RMS/loudness is exactly the metric this project has repeatedly
flagged as an unreliable, confound-prone signal (`AGENTS.md`'s loudness-shortcut warning,
`docs/INTEGRITY_LOG.md`): a systematic gain or recording-session difference between "the leak-test
campaign" and "the no-leak baseline campaign" at the same nominal site would produce an
**identical** statistical signature — louder clips in the leak folder, matching other leak-folder
clips, differing from no-leak-folder clips — without any genuine leak-vs-no-leak information being
present at all. This check rules out one specific, checkable failure mode (naive filename
mislabeling) and cannot rule out the deeper one (session/gain confound), because Dongguan's
filenames carry no timestamp or session ID the way the Hong Kong MEMS files do.

**Sample-size caveat, stated precisely.** The "n=29" ambiguous clips are not 29 independent
samples — they are sequential 1-second sub-clips of a small number of underlying recordings (the
existing `dongguan_recording()` grouping already treats consecutive clips of one recording as one
group for exactly this reason). The real independent sample size behind this subset is on the
order of **8 matched trials**, most of the weight coming from just 2 of them
(`ductile iron-zone 2` and `pe-zone 2`); the `ductile iron-NA`/`pe-NA` and `NA-NA` pairs each have
only a single ambiguous clip, which is barely a trial at all, not a sub-benchmark.

**Conclusion for this subset**: usable as a small, honestly-caveated secondary check (§6), not as
a primary generalization claim, and any result from it must report n at the trial level (≈8), not
the clip level (29), and must disclose the un-rulable-out session-confound risk alongside the
result — exactly the posture the prior review already took for Dongguan generally, now backed by
a direct measurement instead of a filename argument.

## 4. What "domain" should mean here

Per your instruction, a domain is not automatically a whole dataset. The project's actual data
supports (at least) this hierarchy, from coarsest to finest:

1. **Network** — a real distribution system with many physical sites (Hong Kong only; Dongguan's
   "training base" is a dedicated test facility, not a live customer network, and Sheffield/Mendeley
   are single lab rigs, so this level only has one instance in the data).
2. **Site / rig** — a specific physical pipe installation: a Hong Kong field site, the Dongguan
   training base (treated as one rig, since its "zones" are sub-areas of one facility), the
   Sheffield CID-lab loop, the Mendeley 47 m loop.
3. **Material** — PVC (Mendeley), MDPE (Sheffield), ductile iron / PE / steel / PVC (Dongguan,
   mixed within one rig), metal / non-metal (Hong Kong, mixed within one network).
4. **Sensor** — hydrophone, noise logger, MEMS accelerometer, seismic accelerometer.
5. **Operating condition** — pressure, orifice size/type, flow regime (varies *within* Sheffield
   and Dongguan; mostly unrecorded for Hong Kong).
6. **Campaign / session / visit** — a specific date a site was visited (meaningful for Hong Kong
   MEMS specifically, since the same site was visited repeatedly over ~4 months; not reconstructable
   for Dongguan, which has no timestamps).

**Why this matters for benchmark design**: the original V2 document (and my own prior review)
implicitly worked at level 2 ("dataset = domain"), which is why Hong Kong and Dongguan looked like
dead ends — at level 2, site/trial identity swallows the label. Hong Kong MEMS only becomes usable
once you work at **level 6** (visit), because the thing that varies independently of site there is
the date of the visit, not the site itself. Sheffield and Dongguan's matched subset only become
usable once you stop requiring level-2 (whole-dataset) class balance and instead look for
class-balanced structure at **level 5** (operating condition) or **level 3** (material/zone). The
practical rule going forward: before declaring a source confounded or usable, check every level,
not just "does this dataset contain both labels."

## 5. Sheffield, re-examined directly (not taken on the "calibration only" framing)

`AGENTS.md` and the prior review both described Sheffield as "used only for attenuation/wave-speed
calibration, not classification." That framing is **accurate about how the project has used it so
far, and wrong about what the data supports.** Opening the actual files:

- `All_Data/` has 8 test conditions (orifice type/size × pressure, e.g. "1 mm valve, 4.2 bar,"
  "longitudinal slit, 2.8 bar"), each a 30-second, 13-channel (0–20 m spacing), 8192 Hz recording —
  **confirmed sampling rate**, real seismic accelerometers (PCB Piezotronics 393B12), a documented
  sensor sensitivity, and (per the dataset's own `ReadME.docx`) real measured flow rates per test.
- Each test condition has 1–2 **Leak** recordings and is matched (per the ReadME, quoted directly:
  *"NoLeak_4.2bar is the background noise. It is used for all tests as the noise in the lab was
  similar"*) against a **No-Leak** background recording.
- This is a genuine, if small, real both-class dataset on a rig the project has never used that
  way — exactly the kind of source §6 of the original V2 document assumed didn't exist beyond
  Mendeley.

**Honest limitations, stated before using it as a benchmark component:**
- It is an indoor lab pipe loop, not a field site — the same category of limitation as Mendeley
  (just on a different material and pressure range), not a field validation.
- The no-leak population is genuinely small and partly **reused by the curators themselves**: the
  ReadME states one no-leak recording was used as the background for multiple test conditions. Counting
  distinct (non-duplicated) no-leak files: at most 5 across all 8 tests, some of which may still be
  near-duplicates of each other rather than independently informative. Any benchmark using Sheffield
  must treat its no-leak side as low-N and partly non-independent, exactly as Mendeley's own 4
  no-leak recordings are already treated.
- The `Coherence (simultaneous recording)/` subfolder (10 repeat trials × 2 orifice configurations)
  is **not** additional independent leak/no-leak domains — per its own `info_new.txt`, these are
  repeated measurements of the same leak at different sensor spacings, collected for wave-speed
  estimation, the same role this data already plays in the project. Do not double-count it as new
  class-balanced data.

**What it's good for**: a second, independent, real, genuinely-both-class rig — different
institution, different material (MDPE vs. Mendeley's PVC), different pressure regime, different
sensor hardware and curator from Mendeley in every respect. That independence is exactly what a
real cross-rig generalization test needs, and exactly what the project did not have when the prior
review concluded Mendeley was the *only* valid real test domain.

## 6. The revised benchmark, tiered by what it can actually prove

- **Tier 1 — aspirational, highest-value, currently blocked**: Hong Kong MEMS as a held-out,
  cross-visit real-field test (train on some site/visits, test on held-out site/visits at the same
  real municipal sites). This is the closest available approximation to the project's original,
  un-watered-down objective — a real buried network, not a rig — and should become the flagship the
  moment §2's sampling-rate lead is confirmed. **Not codeable yet.**
- **Tier 2 — new primary flagship, usable now**: **Mendeley ↔ Sheffield cross-rig leave-one-rig-out.**
  Train (including synthetic) with Mendeley Looped as the real source, hold out Sheffield entirely
  as the test rig, and the reverse. Both directions are genuine instances of Problem A (§4 of the
  prior review) with **no site/label confound in either direction** — a strictly stronger benchmark
  than the previous Mendeley-only design, because success now has to transfer across two
  independently-curated real rigs with different material, pressure and sensor hardware, not just
  into one. This directly answers your Section 6 requirement ("at least two or more physically
  meaningful real source domains... independent held-out domain(s)... minimal confounding").
- **Tier 3 — secondary/diagnostic, real-world training signal**: Hong Kong (Noise Logger +
  Hydrophone, full dataset) and Dongguan (full dataset) used as additional real training sources
  (more field-realistic acoustic texture than either lab rig) and reported, honestly labeled, as
  "can the model rank never-trained groups of a source it already trained on" — never as a
  generalization claim, per the prior review's already-settled reasoning.
- **Tier 4 — small supplementary check**: Dongguan's 8-trial matched subset (§3), reported with its
  n≈8 and residual session-confound caveat attached to any result, never headlined alone.

**Benchmarks explicitly marked impossible, not worked around**: a true cross-*network* test
(independent municipal systems, not rigs) is not constructible from current data — Hong Kong is the
only real network-level source in hand, and it has no confound-free held-out split at the site
level for Noise Logger/Hydrophone. This gap is real and is what §2's unblock is for, not something
Tier 2 papers over.

## 7. Model development still comes after this (not revised by this document)

`docs/VERSION2_CRITICAL_REVIEW.md` §6 (AdaBN promoted to co-primary; CORAL/MMD gated on reading
`TL-Fault-Diagnosis-Library`'s own tables; Tent/SHOT excluded), §7 (cross-channel coherence feature;
self-supervised pretraining on unlabeled real audio; log-mel ablation) and §8 (the informational
bottleneck is representation-invariance, not capacity) are unaffected by which real dataset is the
test domain, and still apply directly to the Tier 2 benchmark above. One upgrade worth noting:
because Sheffield, like Mendeley, is genuinely both-class, **class-conditional alignment (not just
marginal CORAL/MMD) can now be computed against two independent real sources, not one** — a
materially stronger setup than the prior review's "class-conditional alignment restricted to
synthetic and a small Mendeley sample only," since Sheffield adds a second, independently-sourced
balanced real anchor.

## 8. Don't let "no public data" end the project — an optional, not-yet-decided track

The literature search (`reports/Acoustic leak detection generalization.md`) already confirmed no
new public, both-class, independent real water-leak dataset exists beyond what's catalogued above.
Given that, and given Tier 1 is blocked on an external confirmation rather than new data: a small,
self-collected third rig (e.g. a garden-hose or home plumbing segment with an induced leak,
instrumented with an inexpensive contact-mic/accelerometer and recorded under a couple of
pressure/flow conditions) would not resolve the "real municipal network" gap either — it would be
a third lab rig, not a field site — but it **would** test something neither Mendeley nor Sheffield
alone can: whether cross-rig transfer generalizes beyond two specific academic labs' equipment and
protocols, by adding a rig with no shared curator, instrumentation, or methodology bias with either.
This is genuinely optional, low-cost, and not something I can size correctly without knowing what
tools/space/time you actually have available — **flagging it as a question, not a plan**: worth
pursuing in parallel, or not worth the time relative to unblocking Hong Kong MEMS? Your call.

## 9. Final answers

**"What is the strongest scientifically valid real-world generalization experiment we can perform
with the data we currently have?"** The Tier 2 Mendeley↔Sheffield cross-rig leave-one-rig-out
benchmark (§6): two independent, genuinely both-class, unconfounded real pipe rigs, different
material/pressure/sensor/curator, usable starting now with no new data collection. It is honestly
scoped as a cross-rig (lab-to-lab) test, not a cross-network (municipal) test, and should be
reported as such — this is a narrower claim than the original project objective, but it is the
strongest one the current data actually supports without inventing a workaround.

**"What is the minimum additional data required to make the original municipal-generalization
question genuinely testable?"** Not new data collection — **external confirmation of one fact**:
the Hong Kong MEMS accelerometer sampling rate (§2), via full-text access to the Liu/Tariq/Tijani
et al. IWA paper or direct author contact. If confirmed, Hong Kong MEMS unblocks as a Tier 1
benchmark that is a real buried municipal network, not a rig, with no site/label confound and a
real same-site, cross-time comparison already sitting in the downloaded data (including one
same-day leak/no-leak pair). That is a materially lower bar than "collect a new dataset," and
should be the first thing done in Week 1, ahead of — not parallel to — starting the Tier 2
implementation.

Do not start Week 1 implementation of either tier until you've reacted to this; in particular,
confirm whether the Tier 2 flagship (Mendeley↔Sheffield) is the right one to build first while the
Hong Kong MEMS lead is being chased, or whether you'd rather block on the MEMS confirmation before
writing any code at all.
