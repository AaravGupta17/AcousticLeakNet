# Repository Dataset Search: Real-World Acoustic/Vibration Leak Detection Datasets

Scope note: the project already uses the Mendeley "Accelerometer and Hydrophone Signals
Dataset for Leak Detection" (Aghashahi, Sela & Banks, *Data in Brief* 48, 109148, 2023), a
Hong Kong noise-logger/hydrophone dataset, a Dongguan lab-trial dataset, and a Sheffield MDPE
rig dataset. This note flags overlaps with those explicitly where found.

## What relevant datasets exist on Mendeley Data, Zenodo, IEEE DataPort, Kaggle, government open-data portals, and GitHub?

### Takeaway
Mendeley Data hosts a cluster of datasets from the same Aghashahi/Sela/Banks testbed (likely
precursors/companions to the dataset already in use, not independent data) plus one genuinely
separate real-world dataset (Hong Kong — already in the project's list). Zenodo has one
promising, currently-downloadable, real-sensor dataset collected at a Dongguan training base
(possibly the same site as the project's existing Dongguan data) plus a classic simulated
benchmark (LeakDB) and a gas-pipeline acoustic dataset (GPLA-12). Kaggle's water-leak
listings are tabular IoT telemetry, not raw acoustic/vibration waveforms. The single
strongest new lead is a large real in-situ vibration-sensor corpus from South Korea's AI Hub
portal (30,000 cases, 11,000+ sensor locations), distinct from all four datasets already in
use, though its data format is FFT magnitude spectra rather than raw audio and portal
access/license terms could not be fully verified.

### Cited Findings
- Mendeley dataset "Dataset of Leak Simulations in Experimental Testbed Water Distribution System" (DOI 10.17632/tbrnp6vrnj.1, published 2022-12-12) describes a 47 m, 152.4 mm-diameter PVC testbed, 280 signals, two accelerometers + two hydrophones + two dynamic pressure sensors, 8 kHz hydrophone sampling, 30 s recordings, five categories (longitudinal crack, circumferential crack, gasket leak, orifice leak, no-leak), CC BY 4.0, CSV + RAW files, authors Mohsen Aghashahi, Lina Sela, M. Katherine Banks — [Mendeley Data](https://data.mendeley.com/datasets/tbrnp6vrnj/1)
- Mendeley dataset "Dataset of Leak Simulations in Experimental Testbed Water Network" (DOI 10.17632/xw44wv2g88.1, published 2022-03-18) has an essentially identical description: same 47 m/152.4 mm PVC testbed, same two accelerometers/hydrophones/pressure sensors, same 8 kHz sampling, 280 signals, CC BY 4.0, same author trio — [Mendeley Data](https://data.mendeley.com/datasets/xw44wv2g88/1)
- The *Data in Brief* 2023 paper describing the project's already-used dataset (same authors) reports the same testbed, same sensor set, and the same experimental factors (network topology looped/branched, four leak types + no-leak, background flow levels, traffic/tool background noise, three sensor types) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923002676) (full text blocked by a 403 on direct fetch; details corroborated via search snippets and the PMC mirror [PMC10147960](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10147960/))
- A third, earlier Mendeley entry, "Dataset for Leak Detection and Localization in Water Distribution Systems" (DOI 10.17632/z8gxwdt5dj.1, published 2021-10-12, contributor Mohsen Aghashahi), exists but its page gave no sensor/format/size detail beyond license (CC BY 4.0) and categories ("Change Detection", "Pipeline Leak Detection") — [Mendeley Data](https://data.mendeley.com/datasets/z8gxwdt5dj/1)
- "Acoustic Based Data Acquisition for Leak Detection of Water Distribution Networks" (DOI via Mendeley, published 2022-02-07, authors Tijani, Tariq, Zayed, Hu, Bakhtawar, Abdulmageed, Liu, Fares) is real sensor data from ~90 real leak locations in Hong Kong over a 12-month period, using noise loggers, hydrophones, and MEMS accelerometers on both metallic and non-metallic pipes, CC BY 4.0 — [Mendeley Data](https://data.mendeley.com/datasets/hkn8mxcjyz/1). This matches the description of the Hong Kong dataset already in the project's pipeline, i.e. it is very likely the same dataset already accounted for, not a new one.
- "Single leakage dataset for Hanoi and Anytown water distribution networks" (Mendeley, DOI 10.17632/r9bzzpbh8t.1, published 2024-09-24, contributor Arvin Ajoodani) is purely simulated: leakage was generated with the EPANET emitter tool on the classic synthetic Hanoi and Anytown benchmark networks under 20 demand-multiplier variations; it contains hydraulic/pressure simulation output only, no acoustic or vibration data, CC BY 4.0 — [Mendeley Data](https://data.mendeley.com/datasets/r9bzzpbh8t/1)
- Zenodo record "Acoustic data for the manuscript entitled 'Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity'" (DOI 10.5281/zenodo.18631450, published 2026-02-13) is real sensor data from an "outdoor leakage detection training base" in Dongguan, Southern China, plus separately sourced environmental noise clips; 500 one-second leak clips, 386 one-second no-leak clips, 114 one-second noise clips (1,000 total), labeled with pipe material, region, pressure (MPa), flow rate (m/s) and collection device (some fields missing), three RAR archives, CC BY 4.0, currently listed "Open" and downloadable — [Zenodo](https://zenodo.org/records/18631450). Because this is also sited in Dongguan, it may overlap with — or be a newer/expanded release from — the same training base as the project's existing "Dongguan lab-trial dataset"; this could not be confirmed or ruled out from the page content alone.
- A companion GitHub repo reproduces an acoustic pipeline-leak-detection pipeline (ICASSP submission) built on the above Zenodo dataset — [GitHub: Steven-Yang05/acoustic-leak-fingerprint](https://github.com/Steven-Yang05/acoustic-leak-fingerprint)
- LeakDB ("A benchmark dataset for leakage diagnosis in water distribution networks") on Zenodo (DOI 10.5281/zenodo.1313116, WDSA/CCWI 2018) is simulated/EPANET-generated ("artificially created but realistic leakage scenarios"), CC BY 4.0, ~1.2 GB, with 4,580 views / 3,194 downloads recorded on Zenodo and a companion GitHub repo (KIOS-Research/LeakDB) and MATLAB scoring toolkit — it functions as a widely reused community benchmark, but is not real sensor data and is hydraulic (pressure/demand), not acoustic — [Zenodo](https://zenodo.org/records/1313116)
- GPLA-12 ("An Acoustic Signal Dataset of Gas Pipeline Leakage") is a GAS pipeline dataset: 684 acoustic signals across 12 categories, collected on an intact gas pipe system with artificial external leaks, released in three successive versions (data_v1–v3), EPL-2.0 license, with a paper on arXiv — [GitHub](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12), [arXiv:2106.10277](https://arxiv.org/pdf/2106.10277). Pipe material/diameter, pressure, sensor type and sampling rate were not stated on the pages retrieved.
- Kaggle "Smart Water Leak Detection Dataset" (talha97s) is described as simulated/IoT telemetry (pressure, flow rate, vibration, RPM, operational hours, geolocation) for a smart water transport network rather than raw acoustic waveforms — [Kaggle](https://www.kaggle.com/datasets/talha97s/smart-water-leak-detection-dataset) (page content could not be fully fetched; description drawn from search snippet only — treat as low-confidence)
- A second Kaggle listing, "Water Leak Dataset" (ziya07), appeared in search results but its page could not be fetched (JS-rendered, no detail retrieved) — [Kaggle](https://www.kaggle.com/datasets/ziya07/water-leak-dataset/versions/1) — flagged as a gap, not verified
- A large real-world, in-situ vibration-sensor dataset exists on South Korea's AI Hub ("The Open AI Dataset Project") open-data portal: ~30,000 labeled cases from sensors installed at 11,000+ locations in Gwangju and Goheung, Korea, categories of outdoor leak / indoor leak / electric noise / other noise / normal, two sensor types (LTE and embedded units) mounted on water-meter boxes and valve rooms spaced ~150–300 m apart, with the published data format being FFT magnitude spectral density rather than raw waveform — reported via the paper "Machine Learning Model for Leak Detection Using Water Pipeline Vibration Sensor" (MDPI Sensors 23(21):8935, DOI 10.3390/s23218935), which cites AI Hub (aihub.or.kr) as the source — [MDPI paper](https://www.mdpi.com/1424-8220/23/21/8935). The AI Hub portal page itself could not be directly fetched (403) to confirm current download access, exact license, or registration requirements.
- IEEE DataPort did not surface a dedicated, confirmed-downloadable water/gas pipeline acoustic leak dataset in search results; only a generic "Defect detection dataset" listing appeared, unrelated to pipeline acoustics — [IEEE DataPort](https://ieee-dataport.org/documents/defect-detection-dataset-0) — treated as a gap (no verified IEEE DataPort leak-acoustic dataset found)
- UK Ofwat publishes an official annual water-company "Leakage Dataset" (1992–93 through 2024–25), but this is utility-level aggregate leakage volume/statistics, not sensor waveform data — [Ofwat, November 2025](https://www.ofwat.gov.uk/publication/leakage-dataset-november-2025/)

### Inferences
- The three Mendeley entries (z8gxwdt5dj 2021, xw44wv2g88 March 2022, tbrnp6vrnj December 2022) are almost certainly earlier/companion releases from the same Aghashahi/Sela/Banks 47 m PVC testbed that produced the dataset already used by the project (2023 *Data in Brief* paper); the overlapping testbed dimensions, sensor set, 8 kHz sampling, 280-signal count, and author list make it very unlikely these represent independent evidence. Treat them as the same underlying source, not additional datasets, unless a direct DOI-level diff against the project's currently-cited dataset says otherwise.
- "Acoustic Based Data Acquisition for Leak Detection of Water Distribution Networks" (hkn8mxcjyz) matches the project's existing "Hong Kong noise-logger/hydrophone dataset" description closely enough (Hong Kong, noise loggers + hydrophones + MEMS accelerometers, ~90 leak sites) that it is very likely the same dataset already accounted for.
- LeakDB and the Hanoi/Anytown Mendeley dataset are useful only as simulated hydraulic benchmarks (no acoustic/vibration signal), consistent with this project's own synthetic/EPANET pipeline rather than as held-out real-world test data — they are not candidates for a "real-world generalization" test.
- Of everything found, the AI Hub Korea corpus is the most promising genuinely new, large-scale, real-sensor, labeled leak/no-leak dataset not already represented in the project's pipeline — but its pre-processed (FFT spectra, not raw audio) format and unconfirmed open-access terms are real constraints that need direct portal verification before relying on it.
- The new 2026 Dongguan Zenodo release is the second most promising new lead, being small but clearly real-world, labeled, CC BY 4.0, and currently downloadable — however its site overlap with the project's existing Dongguan dataset must be checked directly (e.g., by comparing collection dates/authors/device names) before treating it as independent evidence.

### Gaps
- Could not directly diff the Mendeley z8gxwdt5dj/xw44wv2g88/tbrnp6vrnj DOIs against whichever exact Mendeley DOI the project currently cites for "Aghashahi/Sela/Banks 2023" — the ScienceDirect Data in Brief article (which would state the canonical DOI) returned HTTP 403 on fetch. Recommend the project team cross-check its own citation's DOI against these three IDs directly.
- Could not confirm whether the new Zenodo Dongguan dataset (18631450) is collected at the same physical "training base" as the project's existing Dongguan lab-trial dataset, or a different/independent site in the same city.
- AI Hub Korea portal page (aihub.or.kr) returned a 403 on fetch; could not verify current downloadability, exact license terms, whether non-Korean/international accounts can register, or whether raw waveform files (vs. only FFT spectra) are available.
- Kaggle "Water Leak Dataset" (ziya07) could not be characterized (JS-rendered page, no content retrieved) — unknown whether real or synthetic, unknown format.
- No IEEE DataPort dataset specific to pipeline acoustic/vibration leak detection was found; this may mean none exists, or that it is not well-indexed by general web search — direct IEEE DataPort site search (via an authenticated session or their own search UI) would be needed to rule this out conclusively.
- Did not find a dedicated "benchmark" acoustic-emission dataset for gas pipelines beyond GPLA-12 that shows clear signs of repeated reuse across multiple independent papers (citation/reuse evidence beyond the original authors was not established within the search budget).
- Did not check data.gov (US) or EU open-data portals directly (e.g., data.europa.eu) beyond general web search; no relevant pipeline acoustic dataset surfaced for these portals, but a direct portal-native search was not performed, so absence is not confirmed.

## For each candidate, is it real sensor data, currently downloadable, what license/format/size?

### Takeaway
Among all candidates, clearly real AND currently downloadable with a stated open license are: the Hong Kong dataset (hkn8mxcjyz, likely already known to the project), the new Dongguan Zenodo release (18631450), and the AI Hub Korea corpus (access terms unverified). GPLA-12 (gas) is real but a different domain (gas, not water) and its exact current accessibility/format could not be confirmed beyond "versions available." LeakDB and the Hanoi/Anytown Mendeley dataset are confirmed simulated, not real sensor data, despite being open and downloadable.

### Cited Findings
- Dongguan Zenodo dataset: real, CC BY 4.0, currently "Open"/downloadable, three RAR archives (5.7 MB compressed footprint figure given, with "data volume" listed as 837.1 MB), published 2026-02-13 — [Zenodo](https://zenodo.org/records/18631450)
- Hong Kong Mendeley dataset: real (90 real leak sites over 12 months), CC BY 4.0, published 2022-02-07 — [Mendeley Data](https://data.mendeley.com/datasets/hkn8mxcjyz/1); file formats/exact size not stated on the fetched page (gap)
- LeakDB: simulated, CC BY 4.0, ~1.2 GB, hosted on Zenodo with a GitHub companion toolkit, confirmed downloadable (view/download counts visible on page) — [Zenodo](https://zenodo.org/records/1313116)
- Hanoi/Anytown Mendeley dataset: simulated (EPANET emitter-tool output only), CC BY 4.0, published 2024-09-24 — [Mendeley Data](https://data.mendeley.com/datasets/r9bzzpbh8t/1); exact file size/format not stated on the fetched page (gap)
- GPLA-12: real acoustic recordings from an intact gas pipe with artificial leaks, EPL-2.0 license, three successive data versions hosted on GitHub — [GitHub](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12); current direct download link and exact file size were not confirmed from the page content retrieved (gap)
- AI Hub Korea corpus: real, in-situ; license and current public-download terms could not be verified (portal fetch returned 403) — flagged as a gap, not confirmed

### Inferences
- The clearest "ready to use today" real-world, open-license, water-specific acoustic/vibration dataset not already in the project's pipeline is the Dongguan Zenodo release (18631450), pending resolution of the site-overlap question above.
- Simulated/hydraulic-only datasets (LeakDB, Hanoi/Anytown) should not be treated as real-world validation data regardless of their popularity as algorithm benchmarks.

### Gaps
- Exact file formats and sizes for the Hong Kong and Hanoi/Anytown Mendeley datasets were not available from the fetched page content (likely present in linked supplementary PDFs not retrieved in this pass).
- GPLA-12's current live download link/working status was not directly verified by following through to a file listing; only the GitHub repo's existence and version history were confirmed.

## What are the physical setup details (pipe material/diameter/length, pressure/flow, sensor type/sampling rate, leak vs no-leak counts, site/label confounding)?

### Takeaway
Full physical setup detail was recoverable for the Aghashahi-group testbed datasets (47 m, 152.4 mm PVC, 8 kHz hydrophones, 280 signals) and the AI Hub Korea corpus (two sensor types, 150–300 m sensor spacing, 30,000 cases/11,000+ sites). For the Hong Kong, Dongguan Zenodo, and GPLA-12 datasets, only partial setup detail (site count/duration or category counts) was retrievable from the pages fetched; sampling rate and precise sensor model were gaps for most of them.

### Cited Findings
- Aghashahi testbed (tbrnp6vrnj/xw44wv2g88): 47 m, 152.4 mm diameter PVC pipe; two accelerometers (A1/A2), two hydrophones (H1/H2), two dynamic pressure sensors (P1/P2); 8 kHz hydrophone sampling; 30 s recordings; 280 total signals across longitudinal crack, circumferential crack, gasket leak, orifice leak, and no-leak categories — [Mendeley Data](https://data.mendeley.com/datasets/tbrnp6vrnj/1)
- Same research group's full factorial design (from the related *Data in Brief* paper): network topology (looped vs. branched), leak type (orifice/longitudinal/circumferential/gasket) vs. no-leak, background flow (0, 0.18, 0.47 L/s, plus a transient 0.47→0 L/s step), background noise (traffic, tool), three sensor types — [search snippet, ScienceDirect paper](https://www.sciencedirect.com/science/article/pii/S2352340923002676)
- Hong Kong dataset: ~90 real leak locations, 12-month collection window, noise loggers + hydrophones + MEMS accelerometers, both metallic and non-metallic pipe materials; sampling rate, pipe diameter/length, and exact leak/no-leak recording counts were not stated on the fetched page — [Mendeley Data](https://data.mendeley.com/datasets/hkn8mxcjyz/1)
- Dongguan Zenodo dataset: 500 leak / 386 no-leak / 114 noise one-second clips; metadata fields include pipe material, region, pressure (MPa), flow rate (m/s), collection device, though some entries have missing values; sampling rate and exact sensor model were not stated on the fetched page — [Zenodo](https://zenodo.org/records/18631450)
- AI Hub Korea corpus: two vibration sensor types (LTE-connected and embedded units), mounted in water-meter boxes and valve rooms spaced approximately 150–300 m apart across sites in Gwangju and Goheung; data published as FFT magnitude spectral density, not raw time-domain waveform; ~30,000 total cases across outdoor leak / indoor leak / electric noise / other noise / normal categories — [MDPI Sensors 23(21):8935](https://www.mdpi.com/1424-8220/23/21/8935)
- GPLA-12 (gas): 684 total acoustic signals across 12 categories on an "intact gas pipe system with external artificial leakages"; pipe material, diameter, pressure, sensor type and sampling rate were not stated in the pages retrieved — [GitHub](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12), [arXiv:2106.10277](https://arxiv.org/pdf/2106.10277)

### Inferences
- For the Aghashahi-testbed family of datasets, site/trial identity is not confounded with label in the way a single real pipeline would be — leak type is an experimentally controlled factor on one fixed rig, which is a strength for controlled comparison but means all data comes from one physical pipe run (a limitation already presumably known to the project given it already uses this dataset lineage).
- For the Hong Kong and Dongguan real-world datasets, site/label confounding could not be ruled out or confirmed from the pages fetched — this is exactly the kind of check the project's own AGENTS.md rules emphasize (e.g., the Mendeley Branched no-leak contamination issue already documented for the existing dataset), so any future use of either source should explicitly verify whether no-leak recordings come from the same physical sites as leak recordings or from different, systematically different locations.

### Gaps
- Sampling rate was not confirmed for the Hong Kong dataset, the Dongguan Zenodo dataset, or GPLA-12.
- Exact leak vs. no-leak recording counts (not just category labels) were not confirmed for the Hong Kong dataset.
- Whether site identity is confounded with label was not directly stated for any of the three non-Aghashahi real-world datasets (Hong Kong, Dongguan Zenodo, AI Hub Korea) — would require reading the full linked papers, which was outside this pass's budget.

## Are there gas-pipeline acoustic-emission / SCADA-linked datasets, and buried-pipe asset-monitoring datasets more broadly?

### Takeaway
One clearly real, downloadable gas-pipeline acoustic dataset was found and verified (GPLA-12). No SCADA-linked acoustic sensor dataset or broader buried-pipe asset-monitoring dataset (beyond leak-specific ones already covered) was found with a verifiable, checkable public link within the search budget.

### Cited Findings
- GPLA-12 is explicitly a GAS pipeline acoustic leakage dataset (not water): "acoustic leakage signals... collected on the basis of an intact gas pipe system with external artificial leakages," 684 signals, 12 categories, three released versions, EPL-2.0 license — [GitHub](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12), [arXiv:2106.10277](https://arxiv.org/pdf/2106.10277)
- A separate, non-dataset finding: a paper on gas-pipeline leak-sample dataset *construction* using Pipeline Studio simulation software (not a public released dataset, a methodology) exists at IEEE Xplore but no downloadable dataset artifact was confirmed — [IEEE Xplore](https://ieeexplore.ieee.org/document/9661577/)

### Inferences
- GPLA-12 should be flagged to the research team explicitly as a gas, not water, analogue if used — the physical acoustic generation mechanism for gas leaks (compressible flow through an orifice) differs materially from water pipe leaks, which is a relevant caveat for any cross-domain transfer claims.

### Gaps
- No SCADA-linked public acoustic sensor dataset for oil/gas pipeline monitoring was found with a checkable link; this may reflect that such data is commercially sensitive and rarely released publicly rather than its absence from the literature.
- No broader "buried-pipe asset-monitoring" dataset (e.g., for general infrastructure condition monitoring beyond leak-specific acoustic/vibration) was identified as a distinct, downloadable artifact within the search budget — this remains an open gap.

## Do any widely-cited "benchmark" leak-detection datasets exist (reused across papers) versus one-off datasets?

### Takeaway
LeakDB is the clearest case of a genuinely reused community benchmark (explicit download/view counts, a companion GitHub toolkit, and conference-paper origin designed for reproducible comparison), but it is simulated hydraulic data, not acoustic. The Aghashahi/Sela/Banks testbed family is the most-established real-sensor acoustic/vibration benchmark in this space (multiple Mendeley releases from the same group over 2021–2023, already used by this project), but direct third-party reuse evidence (independent papers using it) was not verified in this pass. No other dataset in this search showed comparable multi-paper reuse signals.

### Cited Findings
- LeakDB: 4,580 views and 3,194 downloads recorded on its Zenodo page, an associated GitHub repository (KIOS-Research/LeakDB) with a MATLAB scoring toolkit, and origin in a joint WDSA/CCWI 2018 conference paper explicitly aimed at reproducible algorithm benchmarking — [Zenodo](https://zenodo.org/records/1313116)
- The Aghashahi/Sela/Banks testbed data was released across at least three/four Mendeley entries between 2021 and 2023 by the same group, converging in a formal *Data in Brief* 48, 109148 (2023) publication — a pattern consistent with a dataset the originating group intended as a standing benchmark, though independent third-party reuse was not directly confirmed in this search pass — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923002676)

### Inferences
- None of the other datasets found (Hong Kong, Dongguan Zenodo, GPLA-12, AI Hub Korea, Kaggle listings) showed any visible reuse signal (citation counts, companion toolkits, multiple independent papers) within the material retrieved — they currently read as one-off releases rather than established benchmarks.

### Gaps
- Did not run citation-count lookups (e.g., Google Scholar "cited by") for GPLA-12, the Hong Kong dataset, or the Aghashahi testbed family to quantify independent reuse — this would be the most direct way to confirm or refute "benchmark" status and was outside this pass's tool budget.
