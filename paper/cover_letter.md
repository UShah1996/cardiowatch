# PSB 2027 — Cover Letter

**To:** PSB 2027 Program Committee
**Re:** Submission to the session *AI and Machine Learning in Clinical Medicine*

Dear Program Committee,

Please consider our manuscript, *"In-Domain Accuracy Does Not Predict Cross-Device
Robustness: A Pre-Registered Benchmark for Wearable Atrial Fibrillation
Screening,"* for the PSB 2027 session **AI and Machine Learning in Clinical
Medicine**.

Atrial fibrillation is a leading, largely preventable cause of ischemic stroke,
and a large fraction of cases are silent until a stroke occurs. Consumer wearables
that record a single-lead ECG make opportunistic population-scale screening
possible, but only if a screening model works, unchanged, across the heterogeneous
fleet of devices people already own. Our paper asks a question about *evaluation*
rather than architecture: does in-domain held-out accuracy — the metric on which
such models are routinely selected — predict which model survives deployment?

We find that it does not, and that it can order candidate models incorrectly. On a
pre-registered, leakage-safe, patient-grouped hold-out, a deep waveform model is
nominally ahead of a lightweight device-agnostic timing model (AUC 0.949 vs 0.935,
not significant). Moved zero-shot onto ambulatory Holter recordings, that ranking
inverts: the timing model reaches 0.871 (95% CI 0.81–0.92) while the same deep
model falls to 0.773 (0.67–0.87). On single-lead AliveCor recordings the two are
indistinguishable (p=0.93), so the effect appears where device shift is largest. A
deep model trained *across* devices recovers and leads, showing the failure is one
of training distribution rather than architecture.

Our contribution is therefore a cautionary benchmark result together with the
reusable protocol that exposes it — identical windows, zero-shot external cohorts,
paired tests, and patient-clustered intervals — responsive to documented leakage
and reproducibility problems in clinical machine learning. We report our
limitations plainly, including a pre-registered cohort we could not score within
budget and a pre-registered sensitivity analysis that did not complete.

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
