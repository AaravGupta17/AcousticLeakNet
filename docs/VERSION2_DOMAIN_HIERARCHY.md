# Domain hierarchy, corrected: physical source first, file structure never

Written 2026-10-02, refining `docs/VERSION2_REAL_DOMAIN_AUDIT.md` per your instruction that
"domain" must be defined from the physical source of distribution shift, not from how files
happen to be organized, and ranked strictly:
**PHYSICAL NETWORK/SITE > INDEPENDENT PHYSICAL RIG > OPERATING REGIME > SENSOR/RECORDING CAMPAIGN.**

This document **replaces §4 and the Tier framing of §6/§9** in `VERSION2_REAL_DOMAIN_AUDIT.md`.
It does not repeat the forensic work already done there (Hong Kong MEMS, the Dongguan 8-trial
acoustic check, Sheffield's actual contents) — it reclassifies that evidence under a stricter
hierarchy and makes one thing explicit that the prior document left implicit: **we have at most
one physical network in the entire project, and it is not currently unblocked.** Everything called
a "domain" below that line is a rig, not a network, and must say so every time.

## The four levels, applied to data this project actually has

**Level 1 — physical site / network.** A genuinely independent water-distribution installation
serving real customers. **Only one instance exists in the whole project: Hong Kong.** This label
is verified, not assumed — the Mendeley page's own language ("leak sites") was not treated as
sufficient by itself, since "site" could in principle describe a test facility, as it does for
Dongguan. A separate search for the group's peer-reviewed papers behind this dataset (Tijani,
Fares, Zayed and co-authors) found multiple independent abstracts that explicitly use the words
"**real WDNs**," "**in-service and buried WDNs**," and "**real urban water networks in Hong
Kong**" to describe this exact data — consistent, specific language across more than one
publication, not a single ambiguous phrase. This is corroboration, not a primary-source
confirmation of the full methodology (full text of these papers was not obtained; one site count
mentioned in an abstract, "four... sites," does not match the Mendeley page's "~90 sites" figure,
most likely because different sub-studies within the same multi-year campaign used different
subsets — noted as an open, low-stakes detail, not a reason to doubt the Level-1 classification
itself). Dongguan is **not** this level — see below. There is no second network anywhere in the
data.

**Level 2 — independent physical rig.** A physically separate experimental installation, not a
live network, but independently built/curated/instrumented. Three instances: **Mendeley** (47 m
PVC loop, PolyU-curated), **Sheffield** (63 mm MDPE loop, University of Sheffield CID lab), and
**Dongguan** (the Guangdong outdoor "training base" — a dedicated leak-test facility, built to run
controlled trials across materials/zones, not a customer network). **Correction from the prior
document**: Dongguan was described there with language ("Medium... a dedicated real outdoor leak
test facility") that blurred toward Level 1. It belongs at **Level 2**, squarely alongside Mendeley
and Sheffield, not between them and Hong Kong.

**Sub-rule, stated explicitly**: a second recording campaign on the *same* physical rig is **not**
a second Level-2 domain — it stays Level 3 or 4 depending on what actually changed. Applied to
every Level-2 source in this project: Sheffield's 8 test conditions are one rig with Level-3
(orifice/pressure) variation inside it, not 8 rigs. Dongguan's materials/zones are one facility
with Level-3 variation inside it, not multiple facilities. Mendeley's topologies/flow conditions
are one loop with Level-3 variation inside it. None of these internal splits should ever be
reported as "N independent rigs"; there are exactly three Level-2 rigs in this project
(Mendeley, Sheffield, Dongguan), full stop.

**Level 3 — operating regime within one physical system.** Pressure, flow, orifice size/type,
material zone, topology. This is where Sheffield's 8 test conditions live (one rig, Level 3
variation across tests), where Dongguan's material/zone variation lives (one rig, Level 3 variation
across zones and pipe materials), and where Mendeley's topology/flow-condition variation lives.
None of this is cross-rig or cross-network evidence by itself.

**Level 4 — sensor / recording campaign within one physical system.** Hong Kong's Noise Logger,
Hydrophone, and MEMS datasets are **three Level-4 instances of the same single Level-1 domain** —
different instrumentation deployed on the same real network, not three independent domains to
validate against each other. **This is a correction to how the project's own existing
infrastructure already treats the data**: `experiments/cross_dataset.py` (E9) lists
`hk_noiselogger` and `hk_hydrophone` as separate "sources" and runs leave-one-source-out between
them (holding out hydrophone while training on noise logger, and vice versa). Under this
hierarchy, that specific rotation is a Level-4 (sensor-invariance) test *within* Hong Kong, not a
cross-domain generalization test — it should keep running (sensor invariance is a real and useful
thing to measure), but its result must never be reported as evidence about cross-network transfer.
The already-reported E9 LOSO numbers (hk_hydrophone 0.16–0.69 across methods, §2A of the prior
review) happen to be near chance anyway, so no earlier claim in this project relied on
over-reading that rotation — this is a forward-looking rule, not a retraction of a past result.

## Per-source breakdown, the seven questions

| Source | 1. Network | 2. Rig | 3. Operating regimes | 4. Sensors | 5. Campaigns | 6. Leak/no-leak within-source | 7. Train / Val / Test |
|---|---|---|---|---|---|---|---|
| **Hong Kong** | Yes — the only Level-1 instance | n/a (it is the network) | Not tracked by the loader | Noise logger, hydrophone, MEMS (3 Level-4 instances of one network) | MEMS: dated visits, Nov 2020–Apr 2021. Noise logger/hydrophone: unknown, not timestamped | Noise logger/hydrophone: **none** (site=label confound, confirmed). MEMS: **yes**, pending sampling-rate fix | Train: good (all 3 sensors). Val: good. **Test: invalid now; the only possible future Level-1 test, once MEMS unblocks — and even then, a within-network (cross-site) test, not a cross-network one, since no second network exists** |
| **Dongguan** | No — a dedicated test facility, not a customer network | Yes — Level 2 | Material (ductile iron/PE/steel/PVC) × zone × pressure/delay | Hydrophone, noise logger | Not timestamped; campaign identity unrecoverable from filenames | Full dataset: none (confounded by construction). 8-trial subset: yes, acoustically supported, n≈8 (§3 of the prior document) | Train: good. Val: the 8-trial subset, small. Test: full dataset invalid; 8-trial subset conditionally valid as a **Level-2, within-rig** secondary check |
| **Sheffield** | No | Yes — Level 2 | 8 orifice/pressure test conditions | One seismic accelerometer type (two units, one fixed at the leak as reference) | Single lab campaign | Yes — 8 conditions, each with leak recording(s) + matched no-leak background | Train: good. Val: good. Test: valid, **Level-2, cross-rig** against Mendeley |
| **Mendeley (Looped)** | No | Yes — Level 2 | Topology × flow condition | Accelerometer, hydrophone | Single lab campaign | Yes (established) | Train: good. Val: good. Test: valid, **Level-2, cross-rig** against Sheffield |

## Mendeley, reclassified per your instruction

Mendeley is **an independent laboratory physical rig (Level 2)** — a legitimate domain-shift test
at that level, and nothing higher. Stated exactly as you framed it:

- **Success on Mendeley (or Sheffield) = evidence of transfer across an independent laboratory
  physical system.** Real, worth having, not nothing.
- **Failure = evidence of difficulty under that particular laboratory domain shift.** Also real,
  also worth reporting honestly.
- **Neither result, alone or together, establishes success or failure on an operating municipal
  network.** The project has zero Level-1 cross-network evidence today and will have at most one
  *within*-network result even if Hong Kong MEMS fully unblocks, because there is only one network
  in the data to begin with.

Mendeley stays in the benchmark (paired with Sheffield, §6 of the prior document, Tier 2 there) —
it is simply never the sole definition of, or stand-in for, municipal generalization, and no
sentence describing a Mendeley/Sheffield result should say "generalizes to real networks" without
the Level-2 qualifier attached.

## The A/B/C decision

**We are in case B: only one or two physically valid domains at the level that actually matters
most (Level 1), with a workable set of domains one level down (Level 2).** Precisely:

- **Level 1 (physical network)**: **zero** currently usable instances. One instance exists (Hong
  Kong) but its only structurally-unconfounded within-network comparison (MEMS) is blocked on an
  external fact, not a data-structure problem (`VERSION2_REAL_DOMAIN_AUDIT.md` §2). There is no
  second network anywhere in the project's data or in the literature search
  (`reports/Acoustic leak detection generalization.md` — confidential/format-degraded/duplicate-
  candidate sources only, none a usable second network).
- **Level 2 (independent rig)**: **two fully valid instances (Mendeley, Sheffield) and one
  partially valid instance (Dongguan's 8-trial subset)** — enough to construct a genuine
  leave-one-rig-out benchmark now.

Per your Section 7 instruction: **do not pretend a sophisticated ML method solves the original
question** at a level the data can't support. The strongest defensible primary benchmark today is
therefore a **Level-2, leave-one-rig-out** benchmark (Mendeley ↔ Sheffield, Dongguan's subset as a
supplementary check) — this is `VERSION2_REAL_DOMAIN_AUDIT.md` §6's Tier 2, now relabeled
explicitly as **Level-2 evidence, not Level-1 evidence**, in every place it is reported. The
minimum additional *fact* (not data collection) needed to ever reach Level 1 is still exactly what
§2 of that document already identified: confirming the Hong Kong MEMS sampling rate. No benchmark
proposed here claims more than its level actually supports.

## Applying the "what physical factor differs, and why does it matter" test

Required by your Section 8, applied to every split this project now proposes or already runs:

| Split | What physical factor differs | Why it matters to the original application | Valid? |
|---|---|---|---|
| Train on Mendeley, test on Sheffield (or reverse) | Pipe material (PVC vs. MDPE), diameter, pressure/flow regime, sensor hardware, curating institution | An operating network will differ from any single lab rig in exactly these ways; a model that only works on the rig it trained on has learned rig-specific acoustic texture, not leak physics | **Valid — Level 2 cross-rig** |
| Dongguan 8-trial subset, leak vs. no-leak at matched material/zone/sensor | The leak itself, at a fixed material/zone/sensor combination (§3 of the prior document) | The closest available approximation to "same physical setup, leak present vs. absent" outside Mendeley/Sheffield | **Valid — within-rig confirmatory check, not cross-rig**, small n, residual session-confound risk disclosed |
| Hong Kong MEMS, held-out site(s)/visit(s) vs. trained site(s)/visit(s) (future, pending unblock) | Physical site within one real network, and calendar time of visit | The literal practical deployment question: does the model work on a customer's pipe it has not seen, at a time it has not seen | **Valid — Level 1, within-network** (not cross-network; only one network exists) |
| Hong Kong noise logger vs. Hong Kong hydrophone, held out against each other | Sensor instrumentation only — same real network, same general site population | Useful as a sensor-invariance check; does **not** test generalization across physical sources | **Valid only as a Level-4 sensor-invariance check; invalid as a cross-domain generalization claim** — correct the framing of this existing E9 rotation going forward |
| Any split within Dongguan by "zone" or within Sheffield by "test#" alone, without a leak/no-leak axis | Operating regime only (Level 3), not source identity | Robustness-relevant, not generalization-relevant | **Valid only as a robustness check, explicitly not a domain-generalization claim** |

## Reporting rule: name the physical entity, not the dataset

Every benchmark result from here on reports **the exact physical entity held out**, not a dataset
label. Not "trained on real data, tested on Dataset X" — and not even "trained on Mendeley +
Sheffield + Dongguan + Hong Kong, tested on Sheffield," since that still reads as dataset
bookkeeping. The required form is, e.g.:

> Trained on: the Mendeley 47 m PVC loop (PolyU), the Dongguan outdoor leak-training facility
> (Guangdong University of Technology), and the Hong Kong buried distribution network (noise
> logger + hydrophone + MEMS instrumentation). Held out entirely: the Sheffield CID-lab 63 mm MDPE
> loop (University of Sheffield).

This is not a formatting preference — it is what makes Level-2/Level-1 labeling (above) checkable
by a reader instead of asserted. Every results table and every sentence of prose reporting a
cross-domain number must be traceable to a specific rig or network, named as such.

## Rule: no silent pooling of incompatible physical sources

The flagship design (`VERSION2_REAL_DOMAIN_AUDIT.md` §6) trains on everything except the held-out
rig — which means Mendeley, Sheffield (when not held out), Dongguan, and Hong Kong get combined
into one training pool. **That pooling must never be silent.** Concretely, before any training run:

- Every window in the training pool carries its physical provenance (network/rig identity,
  material, sensor type, and — where known — operating pressure) as metadata that survives into
  any results file or log, not just as an internal loader detail that gets lost once windows are
  concatenated. `experiments/public_data.py`'s existing `dataset` and `meta` fields already carry
  most of this; the rule is that nothing downstream (training script, results JSON, written
  report) is allowed to flatten it down to an undifferentiated "real" count without also recording
  the per-source breakdown.
- Any claim of the form "trained on real data" must be immediately followed by which physical
  sources that means, by name, at the level defined above (e.g. "three Level-2 rigs and one
  Level-1 network," not "four real datasets").
- If a future experiment pools sources whose physical differences are large enough to matter
  (e.g. mixing Hong Kong's unconfirmed-sampling-rate MEMS data with everything else once it
  unblocks), that difference is stated in the same document that reports the result, not left for
  a reader to discover by checking the loader code.

## Net effect on the plan

No change to which experiments are runnable today — the Level-2 Mendeley↔Sheffield benchmark from
`VERSION2_REAL_DOMAIN_AUDIT.md` §6 remains the right thing to build first. What changes is
precision of claim: every result from it must be reported as **Level-2 (independent-rig)
evidence**, Dongguan's subset as a smaller same-level confirmatory check, and Hong Kong MEMS kept
explicitly labeled as the project's only possible path to **Level-1 (network)** evidence — itself
still only within one network, never across networks, because a second network does not exist in
any data this project has or can currently obtain. This is the honest ceiling on what "generalizes
across real-world domains" can mean until a second real network surfaces; nothing proposed here
should be allowed to blur past it.
