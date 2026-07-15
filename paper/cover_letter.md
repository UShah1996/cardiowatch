# PSB 2027 — Cover Letter

**To:** PSB 2027 Program Committee
**Re:** Submission to the session *AI and Machine Learning in Clinical Medicine*

Dear Program Committee,

Please consider our manuscript, *"Device-Agnostic Screening for Atrial
Fibrillation on Consumer Wearables: A Pre-Registered, Leakage-Safe Cross-Device
Evaluation for Stroke Prevention,"* for the PSB 2027 session **AI and Machine
Learning in Clinical Medicine**.

Atrial fibrillation is a leading, largely preventable cause of ischemic stroke,
and a large fraction of cases are silent until a stroke occurs. Consumer
smartwatches that record a single-lead ECG make opportunistic population-scale
screening possible, but only if a screening model works, unchanged, across the
heterogeneous fleet of devices people already own. Our paper treats this
cross-device gap as the central obstacle and evaluates it rigorously: on a
pre-registered, leakage-safe, patient-grouped hold-out we show that a lightweight
device-agnostic timing model is statistically indistinguishable from a deep
waveform model on matched data (and stable across 20 seeds), and that under the
realistic zero-shot constraint the timing model transfers across unseen hardware
(hospital, ambulatory Holter, single-lead AliveCor, and Apple Watch) while the
deep model degrades. We frame the contribution as a reusable screening-evaluation
methodology, responsive to the documented leakage/reproducibility problems in
clinical machine learning, rather than as a new detector.

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
