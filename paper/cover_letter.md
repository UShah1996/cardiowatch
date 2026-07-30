# PSB 2027 — Cover Letter

**To:** PSB 2027 Program Committee
**Re:** Submission to the session *AI and Machine Learning in Clinical Medicine*

Dear Program Committee,

Please consider our manuscript, *"In-Domain Accuracy Does Not Predict Cross-Device
Variability: A Pre-Registered Multi-Cohort Benchmark for Wearable Atrial
Fibrillation Screening,"* for the PSB 2027 session **AI and Machine Learning in Clinical
Medicine**.

Atrial fibrillation is a leading, largely preventable cause of ischemic stroke,
and a large fraction of cases are silent until a stroke occurs. Consumer wearables
that record a single-lead ECG make opportunistic population-scale screening
possible, but only if a screening model works, unchanged, across the heterogeneous
fleet of devices people already own. Our paper asks a question about *evaluation*
rather than architecture: does in-domain held-out accuracy — the metric on which
such models are routinely selected — predict which model survives deployment?

We find that it does not, in a specific and practically important way. On a
pre-registered, leakage-safe, patient-grouped hold-out a deep waveform model holds
a small but reproducible advantage over a lightweight device-agnostic timing model
(AUC 0.949 vs 0.935; significant on 14 of 20 alternate hold-out splits). That
advantage carries no information about off-device behaviour. Across four external
cohorts the deep model's AUC ranges from 0.642 to 0.935 (SD 0.121) while the
timing model stays within 0.792–0.873 (SD 0.041) — a 2.9x difference in spread —
and the sign of the gap between them flips from cohort to cohort. Notably, our two
ambulatory Holter cohorts belong to the same hardware class yet rank the two
models in opposite orders, so the variability is not explained by device class
alone and could not have been predicted by testing one dataset per device type.

Our contribution is therefore a cautionary benchmark result together with the
reusable protocol that exposes it — identical windows, zero-shot external cohorts,
paired tests, patient-clustered intervals, and a 20-split sensitivity analysis —
responsive to documented leakage and reproducibility problems in clinical machine
learning. We report our limitations plainly: four cohorts cannot separate the
factors that co-vary with them, one cohort rests on 23 patients, one is evaluated
on a bounded per-record subsample, and the Apple Watch cohort contains a single
confirmed positive and supports plausibility only.

We believe this fits the Clinical Medicine session's emphasis on clinically
grounded, methodologically rigorous machine learning, and speaks directly to
stroke prevention through earlier AFib detection.

**Corresponding author:** Urmi Shah, Department of Computer Engineering, San Jose
State University — urmimanishkumar.shah@sjsu.edu

**Originality and authorship.** We confirm that this work is original, has not
been published elsewhere, and is not under review at another venue. All co-authors
have seen and approved the submission.

**Disclosure of the role of large language models (LLMs) in authoring.**
In accordance with the PSB 2027 policy, we disclose that an LLM-based coding
assistant (Anthropic Claude, used via Claude Code) was used during this project as
a tool for: (i) software engineering support — drafting and refactoring analysis
and evaluation code, cross-device scoring/statistics scripts, and HPC batch
scripts; (ii) editorial assistance — copy-editing and reorganizing manuscript
prose; and (iii) literature-search support. All study design decisions, the
pre-registered analysis plan, the choice of statistical tests, the experiments,
and the interpretation of results are the authors' own. All quantitative results
were produced by the authors' code on the cited datasets and were verified by the
authors; the LLM did not generate, alter, or estimate any reported numbers. The
authors take full responsibility for the content of the manuscript.

Thank you for your consideration.

Sincerely,
Urmi Shah
