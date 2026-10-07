# Real-World Acoustic/Vibration Leak-Detection Datasets (Academic Literature Inventory)

Scope: datasets distinct from the four already in use by the project (Mendeley Aghashahi/Sela/Banks
2023 testbed; Hong Kong noise-logger/hydrophone dataset; Dongguan lab-trial dataset; Sheffield
MDPE rig used for calibration). Note: during research, the primary sources for several of the
project's *existing* datasets were located and confirmed — these are reported below for
verification purposes only, clearly marked as **ALREADY IN USE**, not as new finds.

## What papers describe real (non-synthetic) water-pipe leak acoustic/vibration datasets with both leak and no-leak recordings, and are they actually the same ones already in use?

### Takeaway
Confirmed, via primary-source fetches, that three of the project's four existing datasets have
identifiable origin papers/DOIs (Aghashahi/Sela/Banks benchmarking dataset, a Dongguan dataset from
Guangdong University of Technology, and the Sheffield Shekofteh MDPE rig). No new *downloadable*
water-specific dataset with confirmed both-class real field recordings and public access was found
beyond these — the strongest new lead (a large commercial Sydney deployment) is explicitly
**not** publicly available. Several other candidate real datasets were only found second-hand
(cited inside a 2026 IEEE survey) and could not be traced to a primary, verifiable, downloadable
source within the research budget — these are flagged as gaps, not confirmed entries.

### Cited Findings
- **Aghashahi, Sela & Banks (2023)** — "Benchmarking dataset for leak detection and localization in
  water distribution systems," *Data in Brief* / Mendeley Data, DOI 10.17632/tbrnp6vrnj.1. Confirmed
  via PMC full text (PMC10147960): 152.4 mm schedule-80 PVC testbed, 47 m total pipe length, looped
  and branched topologies; accelerometers (PCB 333B50, 51.2 kHz), hydrophones (Aquarian H2C, 8 kHz),
  dynamic pressure sensors (PCB 102B16, 51.2 kHz); 280 recordings of 30 s each; 4 leak types (orifice,
  longitudinal crack, circumferential crack, gasket) × no-leak, across 6 background/demand
  conditions. CC BY license, publicly downloadable on Mendeley Data now. Each recording is a single
  scenario (one topology, one leak type/location, one background condition), so **leak/no-leak labels
  are confounded with topology, location, and background condition across trials** — this is the
  dataset already in use (the project's own documentation already flags the Branched no-leak
  contamination issue). — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10147960/), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923002676), [Mendeley Data](https://data.mendeley.com/datasets/tbrnp6vrnj/1)
- **ALREADY IN USE (Dongguan).** Found a 2026 Zenodo record, "Acoustic data for the manuscript
  entitled 'Self-supervised acoustic leakage detection for water distribution systems: A real-time
  diagnosis framework under data scarcity,'" Wang Qi (contact), Mei Zhongyi, Zhan Fan, Chen Jiongxi,
  Guangdong University of Technology. Collected at an "outdoor leakage detection training base
  located at Dongguan in Southern China." 500 leak clips, 386 no-leak clips, 114 environmental-noise
  clips, each 1 s. CC BY 4.0, published 2026-02-13, DOI 10.5281/zenodo.18631450. Pipe material,
  diameter, and sensor/sampling-rate details were **not stated** in the dataset record itself. This
  is almost certainly the primary source behind the project's existing "Dongguan lab-trial dataset"
  entry (same city, same leak/no-leak clip structure) — worth cross-checking against the project's
  current citation to confirm it's the same resource and to backfill the missing pipe-material/sensor
  metadata. — [Zenodo record](https://zenodo.org/records/18631450)
- **ALREADY IN USE (Sheffield).** Shekofteh et al., "Calibrated Acoustic Leak Signatures in
  Pressurised Plastic Water Pipes: A Laboratory Analysis," *Sensors* 2026, 26, 4325 (DOI
  10.3390/s26144325). University of Sheffield CID (Contaminant Ingress into Distribution) rig, 63 mm
  MDPE, calibrated accelerometers at 0–50 m from the leak with simultaneous sensor pairs. Data openly
  deposited in the University of Sheffield's ORDA (Figshare-based) repository, DOI
  10.15131/shef.data.32229270.v1. Matches the project's description of "Sheffield MDPE pipe rig used
  only for calibration" — single pipe material, so not a multi-material or generalization-test
  resource. — [Sensors paper](https://www.mdpi.com/1424-8220/26/14/4325), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13419278/)
- **New lead, but NOT publicly downloadable — Sydney, Australia vibro-acoustic mains monitoring.**
  Two companion papers describe a large real-world water-utility deployment: "Vibro-Acoustic
  Distributed Sensing for Large-Scale Data-Driven Leak Detection on Urban Distribution Mains"
  (*Sensors* 2022, 22(18), 6897) and "Detection of Water Leaks in Suburban Distribution Mains with
  Lift and Shift Vibro-Acoustic Sensors" (*Vibration* 2022, 5(2), 21, DOI 10.3390/vibration5020021).
  Commercial "Lift and Shift" semi-permanent vibro-acoustic loggers deployed across six areas of
  metropolitan/suburban Sydney over deployments of up to ~24 months, covering "a wide variety of
  pipeline sizes, materials and surrounding soils." Leak and no-leak audio recordings from real leak
  sources (hydrants, valves, service lines, water-main failures, meter couplings/taps); ~70% of
  detected leaks were pre-existing/hidden (some up to 10 years old) — i.e., real field leaks, not
  induced ones. CNN+STFT classification reported >94% accuracy. **However, a search specifically on
  data availability found that the underlying raw dataset "cannot be made publicly available due to
  confidentiality," with readers directed to contact the corresponding author** — so this is a
  described-but-unavailable dataset. — [Sensors 2022 paper (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9504692/), [Vibration 2022 companion paper](https://doi.org/10.3390/vibration5020021), [industry writeup](https://info.awa.asn.au/water-e-journal/enhancing-sydney-waters-leak-prevention-through-acoustic-monitoring)
- **Second-hand leads only (not independently verified) — cited inside a 2026 IEEE survey.** Tu,
  Rayhana, Bai, Ling & Liu, "A Survey of Water Pipeline Leak Detection using Acoustic Sensing," *IEEE
  Transactions on Instrumentation and Measurement*, 2026 (survey landing page:
  https://ieeexplore.ieee.org/abstract/document/11408040), is reported by search snippets to
  reference several real datasets, including: (a) an "HZ city" pipeline dataset of 12,316 audio
  samples from >10 real pipeline sections, galvanized steel and ductile iron pipe, DN100–DN400mm,
  3,435 leak / 8,881 normal samples; (b) a "real pipeline acoustic signal dataset" of 3,812 samples;
  (c) a water distribution network dataset of 1,004 samples (439 leak / 565 non-leak, then windowed
  into 4,390/5,650 one-second segments). **I could not locate or confirm the primary source papers,
  authors, sensor specs, or download links for any of these three** — they only surfaced as text
  fragments in search-engine summaries, not from a page I could independently fetch and verify. Flag
  as unconfirmed/gap, not fact. — [survey DOI landing page](https://ieeexplore.ieee.org/abstract/document/11408040)
- **Possible multi-material real AE dataset — conflicting/unverified details.** One search pass
  described a dataset with pipes of "cast iron, steel, and concrete located in Jiangsu, Zhejiang, and
  Shanghai, China," collected 2019–2021 under 13/18 bar with leak sizes 0.3/0.5/1 mm (1,080 AE
  samples). A direct follow-up search attributed to the specific paper "Frequency Characteristic
  Analysis of Acoustic Emission Signals of Pipeline Leakage" (*Water* 2022, 14, 3992, DOI
  10.3390/w14243992) instead described "over 6800 leak detection signals from cast-iron pipelines"
  only (3,280 leak / remainder no-leak), with no mention of steel or concrete. **These two summaries
  conflict** — either they are two different papers/datasets that got merged in search snippets, or
  one summary is simply wrong. I was not able to fetch either paper's full text directly (access
  blocked) to resolve this, so neither the multi-material claim nor the single-material claim should
  be treated as confirmed without reading the primary text. Flagged as a gap requiring direct PDF
  access. — [Water 2022 paper DOI](https://doi.org/10.3390/w14243992), [ResearchGate copy](https://www.researchgate.net/publication/366116611_Frequency_Characteristic_Analysis_of_Acoustic_Emission_Signals_of_Pipeline_Leakage)

### Inferences
- Among real (not simulated) datasets with explicit public access, the only ones with full
  independently-verified metadata are the Aghashahi/Sela/Banks Mendeley dataset, the Guangdong Univ.
  Tech Dongguan Zenodo release, and the Sheffield ORDA release — all three already underlie the
  project's existing data sources. No genuinely *new*, independently downloadable, multi-class
  real-world dataset was confirmed in this pass.
- Utility-scale real deployments (Sydney) exist and are the most ecologically valid real-world
  evidence of acoustic leak-detection performance in the literature, but such datasets are
  systematically withheld as confidential/proprietary by the utilities and logger vendors that
  collect them — this appears to be a structural, not incidental, barrier (contact-author-only
  access was stated explicitly), consistent with the difficulty the project has had sourcing
  field data at all.
- Where Chinese-city real pipeline datasets are mentioned in secondary literature (HZ city, Jiangsu/
  Zhejiang/Shanghai), these are typically embedded as training data in a single group's
  classification paper rather than released as a standalone citable, licensed dataset artifact —
  meaning even if located, they would likely fall into the "described in a paper, data unavailable"
  category rather than the "actually downloadable" category.

### Gaps
- Could not confirm sensor type/sampling rate, independent-site count, or single-class-per-site
  status for the "HZ city," "real pipeline acoustic signal," and "1,004-sample WDN" datasets cited
  secondhand via the 2026 IEEE survey — the survey's own full text was not fetched (only
  search-snippet fragments were available), so these three entries could be mischaracterized or
  even duplicates of each other.
- Could not resolve the conflict between the "cast iron/steel/concrete, Jiangsu/Zhejiang/Shanghai"
  description and the "cast-iron only" description both nominally pointing at or near the *Water*
  2022 paper (DOI 10.3390/w14243992) — direct fetch of ScienceDirect/MDPI full text was blocked
  (403) in this session.
- Did not verify whether the Dongguan Zenodo record (DOI 10.5281/zenodo.18631450) is in fact the
  exact source the project currently cites as its "Dongguan lab-trial dataset," versus a newer/
  different release from the same training base — worth a quick manual check against the project's
  existing citation.

## Is there a publicly downloadable, water-specific, multi-material real dataset (PVC, ductile iron, steel, PE/MDPE, asbestos cement, concrete) with both leak and no-leak classes, distinct from the four already used?

### Takeaway
No such dataset was confirmed. The closest thing — a benchmarking/survey claim of a Chinese-city
real pipeline dataset spanning galvanized steel and ductile iron (DN100–400mm, 12,316 samples) —
could not be traced to a primary, fetchable, licensed, downloadable source, so it must be treated as
unconfirmed. All currently-verified public real datasets (Aghashahi/Sela/Banks, Dongguan, Sheffield)
are single-material rigs (PVC, and MDPE respectively; Dongguan's material was not stated in the
record itself despite the project description implying ductile iron/PE/steel/PVC coverage).

### Cited Findings
- Dataset-summary text (not independently verified against primary source) states acoustic noise
  loggers "work best on cast iron, ductile iron, steel, concrete, and transite pipes, while PVC pipes
  require longer periods of recorded data and sensors must be closer together" — this is a general
  claim about detector physics across materials, not itself a dataset citation. — [search synthesis, no single primary source identified]
- The Aghashahi/Sela/Banks benchmarking testbed is single-material: 152.4 mm schedule-80 PVC only. — [PMC10147960](https://pmc.ncbi.nlm.nih.gov/articles/PMC10147960/)
- The Sheffield Shekofteh rig is single-material: 63 mm MDPE only. — [Sensors 2026, 26, 4325](https://www.mdpi.com/1424-8220/26/14/4325)
- The Dongguan Zenodo record (DOI 10.5281/zenodo.18631450) does not state pipe material in its own
  metadata record, despite the project's AGENTS.md characterizing "Dongguan" as covering multiple
  materials — this specific claim could not be verified from the dataset page itself. — [Zenodo record](https://zenodo.org/records/18631450)

### Inferences
- A genuinely multi-material, public, real, both-class acoustic leak dataset does not appear to
  exist in the literature as of this search — or if it exists, it is not well-indexed/discoverable
  via general web search, which itself is notable given how often "multi-material generalization" is
  raised as a gap in leak-detection papers.

### Gaps
- Did not check Scientific Data (Nature) or Journal of Water Resources Planning and Management
  directly (only via general web search aggregation) for a dedicated multi-material dataset paper;
  a targeted journal-site search was not completed within the tool-call budget.

## Is there a non-water (gas or general industrial pipeline) acoustic-emission dataset that is at least publicly downloadable and could be informationally useful?

### Takeaway
Yes — GPLA-12 is a confirmed, publicly downloadable, real (not simulated) **gas pipeline** acoustic
leakage dataset, clearly not water-specific, but worth flagging since it is one of the few leak
acoustic datasets with an open GitHub release.

### Cited Findings
- **GPLA-12** ("An Acoustic Signal Dataset of Gas Pipeline Leakage"), Jie Li et al., arXiv:2106.10277.
  12 categories, 684 total training/testing acoustic signals, collected from "an intact gas pipe
  system with external artificial leakages." Publicly released at
  github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12 (updated to a third version,
  data_v3) and at www.daip.club, alongside pretrained models. Sampling rate and exact sensor hardware
  were **not confirmed** (PDF full-text fetch failed in this session; only indirect summaries were
  available). This is a **gas pipeline**, not water pipeline, dataset — gas acoustics differ
  physically (different fluid compressibility/sound speed, no hydrophone/liquid-coupling physics),
  so any use would be for architecture/methodology transfer only, not as a stand-in for water-pipe
  acoustic physics. — [arXiv abstract](https://arxiv.org/abs/2106.10277), [GitHub release](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12)

### Inferences
- GPLA-12 being one of the only truly open-source, versioned, GitHub-hosted real leak acoustic
  datasets found in this entire search (across both water and gas domains) suggests the broader field
  is unusually closed/proprietary about raw acoustic data, reinforcing the earlier Sydney-dataset
  finding (confidential, contact-author-only).

### Gaps
- Did not confirm GPLA-12's sensor type/sampling rate, pipe diameter/material, or leak-size
  conditions — the arXiv PDF could not be parsed as text in this session (binary/image content), and
  only secondary summaries were available.
- Did not check IEEE DataPort directly (general web search did not surface a working platform-level
  search of IEEE DataPort itself) for any gas- or general industrial-pipeline AE datasets beyond
  GPLA-12.
