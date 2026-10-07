# Physical-domain audit — execution, not planning

Written 2026-10-02. This is the consolidated, authoritative audit requested to replace the two
preceding working documents (`docs/VERSION2_REAL_DOMAIN_AUDIT.md`,
`docs/VERSION2_DOMAIN_HIERARCHY.md`) as the single reference. Those two are not deleted — they
contain supporting detail (the HK-MEMS file-by-file table, the Dongguan RMS numbers, the Sheffield
CSV structure) that is cited here rather than reproduced line-for-line. **Two claims in those
documents are corrected below** (§7); everything else there stands.

No model code was written. No architecture, CORAL/MMD/IRM/Tent/SHOT, or hyperparameter search was
touched. The only code run in this pass was read-only inspection (`find`, direct `openpyxl`/`wavfile`
reads, grep) plus two external lookups to verify facts no file on disk contains.

**Tagging discipline used throughout**: every non-trivial claim is marked
**[FACT]** (directly observed in a file, or stated in the project's own committed documentation),
**[INFERENCE]** (derived from FACTs plus a stated, checkable argument — e.g. a statistical test),
**[ASSUMPTION]** (adopted for now, not independently verified, flagged as such), or
**[UNKNOWN]** (genuinely not resolvable from available material). Where a claim changed category
between the prior documents and this one, that's stated explicitly in §7.

---

## 1. Executive summary

The ultimate research question — does a learned leak representation transfer across physically
meaningful environment changes, ideally toward unseen municipal networks, without large labeled
target data — **cannot currently be tested at the level that matters most (independent municipal
networks)**, because the project has exactly **one** Level-1 network in its data
**[FACT]**, and that network's only structurally unconfounded within-source comparison (the MEMS
accelerometer files) is blocked on one external fact, not a data problem **[FACT, see §9]**. It
**can** be tested one level down: the project has **three** independent Level-2 rigs
(Mendeley, Sheffield, Dongguan) **[FACT]**, of which two (Mendeley, Sheffield) have full,
unconfounded, both-class real data **[FACT]** and one (Dongguan) has a small, acoustically-
supported-but-unproven partially-deconfounded subset **[INFERENCE]**. This places the project in
**Case B** (§13): a limited but genuine benchmark is constructible now, with an honestly stated
ceiling on what it can conclude.

## 2. Physical-domain hierarchy

Unchanged from `docs/VERSION2_DOMAIN_HIERARCHY.md`, restated for completeness:

1. **Level 1 — physical site/network.** A genuinely independent distribution installation serving
   real customers.
2. **Level 2 — independent physical rig.** A physically separate experimental installation, not a
   live network.
3. **Level 3 — operating regime within one physical system.** Pressure, flow, orifice, material
   zone, topology.
4. **Level 4 — sensor/recording campaign within one physical system.** Instrumentation or visit
   date, physical system unchanged.

Priority for a generalization claim: **Level 1 > Level 2 > Level 3 > Level 4.** A second campaign
on the same rig is never a new Level-2 instance; a sensor change is never a new Level-1 instance.

## 3. Complete physical-source inventory

Full repository search performed (`find datasets -maxdepth 2`, `find datasets/public -maxdepth 3`,
`grep` of `scripts/download_public_data.py`), not a re-read of prior descriptions **[FACT]**.
Confirmed present: `datasets/Accelerometer/`, `datasets/Hydrophone/` (raw Mendeley, consumed by
`experiments/mendeley.py`), `datasets/_mendeley_zips/` (download cache, not a data source), and
`datasets/public/{hongkong,dongguan,sheffield}/`. **No other real dataset exists in the
repository** **[FACT]** — three public sources plus the original Mendeley download, matching
`scripts/download_public_data.py`'s own docstring exactly.

| # | Source name | Cited origin (from the repo's own download script) | Level |
|---|---|---|---|
| 1 | Hong Kong (Noise Logger, Hydrophone, MEMS) | Tijani, Tariq, Zayed et al. (2022), Mendeley Data DOI 10.17632/hkn8mxcjyz.1 **[FACT — quoted directly from `scripts/download_public_data.py`'s docstring]** | 1 |
| 2 | Dongguan | Wang, Mei, Zhan & Chen (2026), "Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity," Zenodo DOI 10.5281/zenodo.18631450 **[FACT — confirmed by direct web lookup of the Zenodo record, matching file counts: 500 leak / 386 no-leak / 114 environmental-noise clips, exactly matching the counts on disk]** | 2 |
| 3 | Sheffield | Shekofteh, M.R. (2026), "Acoustic data for the leakage experiments in the CID Lab, University of Sheffield," ORDA DOI 10.15131/shef.data.32229270.v1 **[FACT]** | 2 |
| 4 | Mendeley (Looped only; Branched excluded as contaminated) | Aghashahi, Sela & Banks (2023), *Data in Brief* 48, 109148 **[FACT, already established in this project's existing docs]** | 2 |

No other real acoustic/leak dataset is present anywhere in the repository.

## 4. Dataset-by-dataset audit (the 21-field table, condensed to what's non-redundant)

Fields 1–4, 6–8 (name, network, rig, level, material, dimensions, installation type) are already
fixed by §3 and the citations above; the table below gives the fields that actually differ by
source and drive the benchmark decision.

| Field | Hong Kong — Noise Logger / Hydrophone | Hong Kong — MEMS | Dongguan | Sheffield | Mendeley (Looped) |
|---|---|---|---|---|---|
| **5. Independent from every other source?** | Yes (Level 1, unique) | Yes (same network as 5a, different instrumentation — not independent *of Hong Kong*, independent *of the other 3 sources*) | Yes | Yes | Yes |
| **9. Pressure/flow regime** | Not tracked **[FACT — absent from the loader/filenames]** | Not tracked | Logged per leak trial (e.g. "0.31 MPa"), **[UNKNOWN]** for the ambiguous no-pressure subset (§7 of the prior review, restated in §6 below) | 2.8–4.2 bar, 8 documented test conditions **[FACT, from `ReadME.docx`]** | Flow condition tracked, suspected unresolved confound, `E7` never run **[FACT, from `docs/CLAIMS.md`]** |
| **10/11. Leak / no-leak conditions** | 22/32 sites leak, 10/32 no-leak (noise logger); 10/18 leak, 8/18 no-leak (hydrophone) **[FACT, file count]** | All 5 sites have both labels; one same-day pair (ACB, 2021-01-26) **[FACT, file count + filename dates]** | 500 leak clips, 386 no-leak clips, 114 env-noise clips **[FACT]** | 8 test conditions, each with 1–2 leak recordings and a matched no-leak background; ≤5 distinct no-leak recordings total, some explicitly reused by the curators **[FACT, from `ReadME.docx`]** | 16 leak, 4 no-leak recordings (Looped) **[FACT, established]** |
| **12/13. Sensors / placement** | Noise logger; hydrophone | MEMS accelerometer, single-axis column, unit IDs A50/AD6/AD8/AD9/ACB **[FACT]** | Hydrophone, noise logger **[FACT]** | 2 seismic accelerometers (PCB Piezotronics 393B12), one fixed at the leak, one moved across 13 positions **[FACT, from `ReadME.docx`]** | Accelerometer, hydrophone **[FACT, established]** |
| **14. Recording campaigns** | **[UNKNOWN]** — no timestamps in filenames | ~4 months, Nov 2020–Apr 2021, repeat visits per site **[FACT, from filename dates]** | **[UNKNOWN]** — no timestamps | One lab campaign **[ASSUMPTION — not explicitly stated, but no evidence of multiple campaigns]** | One lab campaign **[ASSUMPTION, same basis]** |
| **17. Leak/no-leak coexist within the same physical source?** | **No** — 0/32 and 0/18 site overlap, confirmed by direct file audit **[FACT]** | **Yes** — all 5 sites **[FACT]** | No at the trial level, except an 8-recording matched subset **[FACT for the filename match; see §6 for the comparability caveat]** | Yes — 8 test conditions each matched to a no-leak background **[FACT]** | Yes **[FACT, established]** |
| **18. Physically comparable?** | n/a (no coexistence) | **[ASSUMPTION]** — same site, different calendar date; physical pipe condition between visits is not independently verified, only inferred from the label | **[INFERENCE]** — RMS-based acoustic test (§6) supports "real leak, not mislabeled," cannot rule out a session/gain confound | **[ASSUMPTION]** — curators' own stated protocol ("noise in the lab was similar") is the only basis for treating the shared no-leak background as comparable across tests | **[FACT, established]** — same rig, independently-run sessions |
| **19. Potential confounders** | Site = label, total | Visit date/session (cannot separate from true leak-status change without more metadata) **[UNKNOWN]** | Trial identity (by construction); residual session/gain-level confound cannot be ruled out for the 8-trial subset | Reused no-leak background across multiple test conditions — low no-leak N, possible non-independence | Flow condition (documented, unresolved, `E7` pending) |
| **20/21. Usable as / claim supported** | Train (texture), secondary test only — domain-recognition diagnostic, not generalization | **Potential Level-1 test**, blocked pending §9 | Train (full); small Level-2-internal secondary check (8-trial subset) | Train + **Level-2 cross-rig test** | Train + **Level-2 cross-rig test** |

## 5. Leak/no-leak comparability — summary of the confound analysis

This restates (does not redo) `docs/VERSION2_REAL_DOMAIN_AUDIT.md` §2–3 and
`docs/VERSION2_DOMAIN_HIERARCHY.md`'s per-source table. The headline cases of
`LEAK = SITE A, NO-LEAK = SITE B` (or equivalent):

- **Hong Kong Noise Logger and Hydrophone**: exactly this pattern, total, 0% site overlap
  **[FACT, direct file audit]**.
- **Dongguan, full dataset**: the same pattern by construction (trial identity bakes in label),
  **except** 8 of 18 no-leak base recordings, which share material+zone+sensor with a leak-folder
  counterpart **[FACT, filename match]** — see §6 for whether this is genuinely deconfounded or an
  artifact.
- **Sheffield and Mendeley**: **not** this pattern — both have genuine within-rig coexistence
  **[FACT]**.
- **Hong Kong MEMS**: **not** this pattern at the site level — all 5 sites have both labels
  **[FACT]** — but the within-site leak/no-leak comparison is now a *campaign* (visit-date)
  comparison, which **is** a confound risk of its own (§4, field 18) that the project cannot
  currently resolve from the data alone **[UNKNOWN]**.

## 6. Dongguan's 8-trial subset — comparability, restated precisely

`docs/VERSION2_REAL_DOMAIN_AUDIT.md` §3 ran an RMS comparison across all available clips in the
8 matched material/zone/sensor combinations: ambiguous (no-pressure-logged) leak-folder clips had
mean RMS 10,287 vs. 2,997 for matched no-leak clips (Mann-Whitney p = 1.3×10⁻¹⁴) and were
statistically indistinguishable from confirmed-pressure leak clips at the same site (p = 0.77)
**[INFERENCE — a specific, reproducible statistical test, not a guess]**. This supports, but does
not prove, that the subset is genuine leak data rather than a mislabeling artifact: an identical
signature would also arise from a systematic recording-session/gain difference between "the leak
test campaign" and "the no-leak baseline campaign" at the same nominal site, which filenames cannot
rule out **[UNKNOWN — the source paper, Wang/Mei/Zhan/Chen 2026, was not obtainable in full text
this pass; it is the one document that could resolve this directly and is now a named, concrete
reading target]**. The real independent sample size behind this subset is **≈8 trials**, not the
29 clips used for the statistical test (consecutive 1-second clips of one recording are not
independent samples) **[FACT, from the existing `dongguan_recording()` grouping convention]**.

## 7. Reclassification of previous cross-source experiments — two corrections

**Correction 1 (to `docs/VERSION2_DOMAIN_HIERARCHY.md`).** That document stated E9
(`experiments/cross_dataset.py`) "runs leave-one-source-out between [hk_hydrophone and
hk_noiselogger]," implying they are held out against each other. **This is wrong for the `loso`
protocol**, which is the project's main, emphasized protocol. Direct code inspection
(`cross_dataset.py` lines 225–256) **[FACT]** shows `loso` holds out an entire *source*
(`SOURCE = {"hk_noiselogger": "hongkong", "hk_hydrophone": "hongkong", ...}`) — all of Hong Kong
together — and only *reports* the held-out source's test metrics broken down by original dataset
for readability (`# report each held-out dataset separately`). **The project's existing `loso`
protocol already respects the Level-1 grouping correctly.** The `lodo` (leave-one-*dataset*-out)
protocol is the one that would hold hydrophone out against noise logger — but the project's own
docstring already labels `lodo` "for completeness only; a sibling dataset from the same source can
leak into training," i.e. the codebase already carries the correct warning, and the specific run
inspected (`results/runs/2026-09-29_010352_e9_cross_dataset.json`) did not even execute `lodo` (its
`lodo` dict is empty). **Net effect**: the E9 numbers already quoted in
`docs/VERSION2_CRITICAL_REVIEW.md` §2A (the "held out: Dongguan / HK hydrophone / HK noise logger /
Mendeley acc / Mendeley hyd" table) are correctly described as a leave-one-**source**-out result,
not a leave-one-sensor-out result — no retraction of that specific table is needed. The error was
in the *hierarchy document's* general description of the infrastructure, now fixed here.

**Correction 2 (to `reports/Acoustic leak detection generalization.md`, the earlier literature
search).** That report flagged "a 2026 Zenodo release from Guangdong University of Technology's
Dongguan 'outdoor leakage detection training base' (DOI 10.5281/zenodo.18631450)... shares the same
city and training-base framing as the project's existing Dongguan source and could not be
confirmed as independent." **Resolved**: this DOI is not a second, possibly-independent dataset —
it is the citation for the Dongguan source **already in use** (`scripts/download_public_data.py`
cites the identical DOI, and the record's stated file counts — 500 leak / 386 no-leak / 114
environmental clips — match what's on disk exactly) **[FACT]**. There is no unconfirmed second
Dongguan-area dataset; this was one dataset double-counted by the literature search, now corrected.

No other prior result required reclassification. The `within`-dataset numbers (`docs/MODEL_F_PREREG.md`'s
0.971 Dongguan in-domain AUROC, the per-source `features_1ch` logreg numbers quoted throughout)
were already correctly scoped as within-source and are unaffected by any of this.

## 8. Mendeley ↔ Sheffield assessment

**Genuinely independent Level-2 rigs** **[FACT]**: different institution (PolyU vs. University of
Sheffield), different pipe material (PVC vs. 63 mm MDPE), different pressure regime (Mendeley's
documented range is lower than Sheffield's 2.8–4.2 bar), different sensor hardware (Mendeley's
accelerometer/hydrophone vs. Sheffield's PCB Piezotronics 393B12 seismic accelerometers), different
curator and methodology. Both have genuine within-rig leak/no-leak coexistence **[FACT]**. This
pair supports a **Level-2 independent-rig transfer** experiment in both directions
(Mendeley→Sheffield, Sheffield→Mendeley). **What it can establish**: whether a representation
survives a change in pipe material, pressure regime, and sensor hardware, holding "it's a
controlled lab rig" fixed. **What it cannot establish**: anything about a live, buried,
customer-connected network — neither rig is one **[FACT, by construction of both studies]**.
**Engineering note**: Sheffield is not currently loaded by `experiments/public_data.py`'s common
window format — it is read only by `experiments/sheffield_calibration.py` for calibration, as
raw per-test CSVs. Building this benchmark requires a new loader (resample 8192→5000 Hz, 2 kHz
band limit, 0.4 s windowing, group-by-test-condition) — **benchmark construction tooling**,
explicitly permitted by your Part 11, not modeling.

## 9. Hong Kong assessment

**Level**: 1, verified, not assumed. Verification chain, stated precisely: (a) the project's own
`scripts/download_public_data.py` docstring states "Real buried networks, ~90 leak sites," citing
Tijani, Tariq, Zayed et al. (2022) **[FACT — this is the project's existing, committed
documentation, not a new claim]**; (b) an independent web search for the underlying peer-reviewed
papers by overlapping authors found multiple abstracts using "real WDNs," "in-service and buried
WDNs," and "real urban water networks in Hong Kong" **[INFERENCE — corroborating, not primary
full-text confirmation; one abstract's site count ("four... sites") does not match the "~90" figure,
most likely reflecting a sub-study using a subset, left as **[UNKNOWN]** rather than resolved]**.
**Leak/no-leak availability**: Noise Logger and Hydrophone — present but totally confounded with
site **[FACT]**. MEMS — present, not confounded with site **[FACT, file audit, §3 of the prior
review]**, but the within-site comparison is across calendar visits, whose physical comparability
is **[UNKNOWN]** (§4, §5). **Sampling rate**: Noise Logger confirmed 4096 Hz (documented in
`experiments/public_data.py`) **[FACT]**; MEMS **unconfirmed on this dataset specifically**
**[UNKNOWN]**, with a named, specific, not-yet-verified lead — an AX3-class sensor, ~1000 Hz
3-axis / ~3000 Hz 1-axis, from a companion paper by an overlapping author group
**[INFERENCE — plausible, not confirmed; full text of that specific paper was not obtained this
pass, blocked by a 403]**. **Can MEMS legitimately contribute now?** Not yet — resampling at a
guessed rate would shift every frequency, exactly as `experiments/public_data.py`'s existing
docstring already states. **Does MEMS represent a new physical source, or another campaign/sensor
on the same source?** The latter **[FACT]** — same ~90-site Hong Kong campaign, different
instrumentation (Level 4), which is precisely why it is valuable: it is the *same* Level-1 network,
observed in a way that happens not to be confounded.

## 10. Dongguan assessment

**Level**: 2 (a dedicated outdoor leak-training facility built by Guangdong University of
Technology, not a live customer network) **[FACT, per the Wang/Mei/Zhan/Chen 2026 Zenodo record's
own description, §3]**. **Zones**: Level-3 sub-locations within this one facility (zone 1, zone 2,
and an unzoned "NA" condition) **[FACT, from filenames]**, co-occurring with 4 materials (ductile
iron, PE, steel, PVC) **[FACT, from filenames]** and 2 sensors (hydrophone, noise logger)
**[FACT]**. **Leak/no-leak structure**: 500/386/114 clips as above; confounded by trial identity
except the 8-trial subset (§6). **Do the partially-deconfounded recordings share physical source
characteristics?** Yes at the material+zone+sensor level by filename match **[FACT]**; comparability
of the *leak event itself* is supported but not proven by the RMS test (§6) **[INFERENCE]**.
**Can they support a legitimate held-out physical-source experiment?** Only as a **within-rig**
(Level-2-internal) secondary check, n≈8, caveats stated — not as a second independent physical
source (it is still the same Dongguan facility, not a new rig).

## 11. Sheffield assessment

Covered in §8 (paired with Mendeley) and in detail in `docs/VERSION2_REAL_DOMAIN_AUDIT.md` §5.
Restated once, precisely: Level 2, one rig, 8 Level-3 operating-regime test conditions, genuine
leak/no-leak coexistence, confirmed 8192 Hz sampling and documented sensor hardware (all **[FACT]**
from direct file/ReadME inspection), with a stated no-leak-population caveat (≤5 distinct
recordings, some explicitly reused across tests per the curators' own stated protocol
**[FACT, quoted from `ReadME.docx`]**, raising the comparability of that pooled no-leak reference
across different tests to **[ASSUMPTION]** rather than FACT).

## 12. Any other datasets — none found beyond the four already catalogued

Already searched exhaustively in the prior literature pass
(`reports/Acoustic leak detection generalization.md`) and re-confirmed by this pass's repository
search (§3): no other real water-leak dataset exists on disk, and no new independent one was found
in the literature beyond what that report already catalogued (a confidential Sydney deployment,
not downloadable; a Korean AI Hub corpus released only as FFT spectra, access blocked; GPLA-12, a
real but **gas**-pipeline dataset, useful only for architecture-transfer reference). One new, named
lead surfaced this pass worth flagging for method design (not data acquisition): the Dongguan
source's own origin paper (Wang, Mei, Zhan & Chen, 2026) is titled "Self-supervised acoustic
leakage detection for water distribution systems... under data scarcity" — i.e. its authors already
attempted self-supervised learning on exactly this data; reading it in full is now a concrete,
high-value action for `docs/VERSION2_CRITICAL_REVIEW.md` §7's self-supervised-pretraining candidate,
separate from and in addition to the already-flagged Hong Kong↔Dalian paper.

## 13. Independent physical-source count

- **Level 1 (networks): 1** — Hong Kong. **0 of 1 currently has a usable, unconfounded within-source
  leak/no-leak comparison** (blocked pending §9).
- **Level 2 (rigs): 3** — Mendeley, Sheffield, Dongguan. **2 of 3 (Mendeley, Sheffield) are fully
  usable now; 1 of 3 (Dongguan) has an n≈8 partial subset**, plus full-dataset use as a confounded
  training-only source.
- **Level 3 (regimes), approximate, not independently re-verified for Mendeley this pass**:
  Sheffield 8; Dongguan 8 material×zone combinations (ductile iron × {NA, zone 1, zone 2}, PE ×
  {NA, zone 1, zone 2}, PVC × zone 1, steel × zone 1); Mendeley's topology×flow-condition count is
  **[UNKNOWN in this pass — already documented elsewhere in the project, not re-derived here]**.
- **Level 4 (sensor/session instances)**: Hong Kong — 3 sensor types × ~4 months of repeat visits
  per site (MEMS only); Dongguan — 2 sensor types; Mendeley — 2 sensor types; Sheffield — 1 sensor
  type (2 physical units).

## 14. Which sources have valid within-source leak/no-leak comparisons

**Valid now, unconfirmed-free**: Sheffield, Mendeley. **Valid, small, caveated**: Dongguan's
8-trial subset. **Potentially valid, blocked**: Hong Kong MEMS. **Invalid**: Hong Kong Noise
Logger, Hong Kong Hydrophone, Dongguan full dataset.

## 15. Primary benchmark decision: B

**Case B — only one or two genuinely useful independent physical sources at the level that matters
most.** Precisely: zero Level-1 sources are currently usable; two-to-three Level-2 rigs are. This
is not Case A (we do not have multiple independent *networks*) and not Case C (a real, non-
manufactured benchmark does exist at Level 2). Stated explicitly, what Case B prevents us from
concluding: **no experiment this project can run today can support the sentence "this
generalizes to unseen municipal networks."** The best available sentence is narrower: "this
generalizes across independent laboratory pipe rigs of different material, pressure and sensor
hardware" — itself a real, non-trivial, previously-untested claim, just not the original one at
face value.

## 16. Exact proposed benchmark

**Leave-one-independent-physical-source-out, at Level 2**:

- **Training physical sources**: the Mendeley 47 m PVC loop (PolyU) *or* the Sheffield 63 mm MDPE
  CID-lab loop (University of Sheffield) — whichever is not held out — plus, as auxiliary
  Level-1/Level-2 real training texture (not evaluated as test domains): the Hong Kong buried
  network (all three sensor types) and the Dongguan outdoor training facility, each explicitly
  labeled by source when reporting composition (per the no-silent-pooling rule,
  `docs/VERSION2_DOMAIN_HIERARCHY.md`).
- **Validation physical source**: a held-out slice of the *training* rig's own groups (not the test
  rig), disjoint from its frozen-test slice, for checkpoint selection — avoiding the
  best-of-N-on-the-test-set bias already flagged for Model F/R2.
- **Held-out test physical source**: the other rig (Sheffield, if training included Mendeley; or
  Mendeley, if training included Sheffield) — entirely held out, zero windows of any kind used in
  training or checkpoint selection.
- **Exact hierarchy level**: Level 2 (independent physical rig), explicitly not Level 1.
- **Why the physical shift is meaningful**: material (PVC vs. MDPE), diameter, pressure regime,
  sensor hardware, and curating institution all differ — a model that only works on its own
  training rig has learned rig-specific acoustic texture, not a transferable leak cue.
- **What this experiment can and cannot establish**: can establish whether a representation
  generalizes across independently-curated real lab rigs. Cannot establish anything about live
  municipal networks — stated as a hard ceiling, not a hedge, every time a result is reported.

## 17. Minimum additional physical data required (if a stronger claim is wanted)

Not new data collection — **one external confirmation**: the Hong Kong MEMS accelerometer sampling
rate (§9), via full-text access to the Liu/Tariq/Tijani et al. IWA paper or direct author contact.
If confirmed, Hong Kong MEMS becomes a genuine Level-1 benchmark (held-out sites/visits within the
one real network this project has) — still not cross-*network* evidence (no second network exists
or is obtainable), but a materially stronger claim than anything available today, and the only one
available without new data collection. If a true cross-*network* claim is ever wanted, the minimum
additional physical data is exactly what §17 of the prior audit document already stated: a second,
independent, real, in-service distribution network with its own genuine within-network leak/no-leak
coexistence — nothing short of that satisfies Level 1 at a second instance, and nothing in hand or
in the literature search provides it.

## 18. Recommended next experiment

Build the Mendeley↔Sheffield Level-2 leave-one-rig-out benchmark (§16) as the primary deliverable,
using Model F's existing encoder and the baselines/statistical protocol already established in
`docs/VERSION2_CRITICAL_REVIEW.md` §9–10 (unchanged by this document). In parallel, not blocking:
pursue the Hong Kong MEMS sampling-rate confirmation (§9/§17) and read the Wang/Mei/Zhan/Chen (2026)
Dongguan self-supervised paper (§12). Do not begin either until you confirm which of these two
parallel tracks you want prioritized this week.

---

## Final answers (as requested, concise)

1. **Independent Level-1 physical networks: 1** (Hong Kong), **0 currently benchmark-ready**.
2. **Independent Level-2 rigs: 3** (Mendeley, Sheffield, Dongguan), **2 fully usable now**
   (Mendeley, Sheffield).
3. **Level-3 regimes**: Sheffield 8, Dongguan 8 (material×zone), Mendeley unknown-in-this-pass
   (already documented elsewhere, not re-derived).
4. **Level-4 sensor/session domains**: Hong Kong 3 sensor types × repeat visits (MEMS only);
   Dongguan 2 sensors; Mendeley 2 sensors; Sheffield 1 sensor type (2 units).
5. **Valid within-source leak/no-leak comparisons**: Sheffield and Mendeley (clean); Dongguan
   8-trial subset (small, caveated); Hong Kong MEMS (potentially clean, blocked); Hong Kong Noise
   Logger/Hydrophone and Dongguan full dataset (invalid — confounded).
6. **Legitimate multi-physical-source benchmark exists**: **Yes, at Level 2** (not at Level 1).
7. **Mendeley↔Sheffield runnable**: **Yes**, pending one engineering task (a Sheffield loader in
   `experiments/public_data.py`'s common window format — it is not there yet).
8. **Recommended benchmark**: Level-2 leave-one-rig-out, Mendeley↔Sheffield, Hong Kong + Dongguan
   as labeled auxiliary real training texture, reported per §16's exact specification.
9. **Single most important limitation**: the project has only one real network and zero currently-
   usable Level-1 test domains — every result this benchmark can produce is rig-level evidence,
   not network-level evidence, and no sentence reporting it should imply otherwise.
10. **What to do next**: build the Sheffield loader and run the Level-2 benchmark (§16); in
    parallel, chase the Hong Kong MEMS sampling-rate confirmation, since it is the only currently
    identified path to ever obtaining Level-1 evidence. Tell me which of these two you want
    prioritized before any of it starts.
