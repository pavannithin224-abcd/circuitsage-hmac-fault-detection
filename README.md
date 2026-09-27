# Graph-Aware Fault Detection for an OpenTitan HMAC Circuit

[![DOI](https://zenodo.org/badge/DOI/10.5281%2Fzenodo.22986785.svg)](https://doi.org/10.5281/zenodo.22986785)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-frozen%20prototype-555.svg)](#project-status)

I built this project to explore a practical question: **can information from a synthesized circuit graph help us predict whether a digital fault will be detected?**

The answer from this experiment is yes—but with important limits. The graph-aware hybrid model performed much better than the conventional and plain deep-learning baselines on this circuit. It also lost more performance than allowed when I moved from development vectors to separate validation vectors. I have kept both results in this repository because I want this to be useful as an honest research baseline, not just a collection of impressive numbers.

This repository is intended for students, researchers and hardware enthusiasts who want to reproduce the experiment, try a smaller model, improve the graph representation, add new fault types or adapt the workflow to another digital circuit.

## Project status

**Completed and frozen research prototype — version 1.**

The RTL, synthesized golden netlist, fault catalog, datasets, feature policies, model selections, thresholds and evaluation results for this version are frozen. Changing any of them creates a new experiment and should use a new version number.

This is not advertised as a production-ready fault detector or a fabricated-silicon result.

## What was tested

| Item | Frozen scope |
|---|---:|
| Circuit | OpenTitan HMAC-SHA256, fixed 32-byte-message wrapper |
| OpenTitan revision | `83fc48ed3a727399056772d12be8c7d4a8a276f0` |
| Golden-netlist cells / legal sites | 22,839 |
| Directed graph edges | 47,617 |
| Fault models | Single persistent SA0 and SA1 |
| Fault instances | 45,678 |
| Development TRAIN vectors | 64 |
| Development enabled samples | 2,923,392 |
| Separate VALIDATION vectors | 48 |
| Validation enabled samples | 2,192,544 |
| HOLDOUT vectors used | 0 |

The supervised task is **pre-simulation binary detectability**. Given a candidate circuit site, SA0/SA1 type, test-vector information and leakage-safe structural features, the model predicts whether that fault is likely to be detected. The main hybrid model is therefore not a universal “unknown fault finder.”

A separate behavior-diagnosis pilot takes 64 observed HMAC responses without receiving the hidden fault identity. It compares the responses against a frozen reference catalog and returns no anomaly, one matching site, an ambiguous candidate set or no catalog match. That pilot is a reference checker and lookup system, not a newly trained neural network.

## Models compared

| Model | DEV_SITE_TEST MCC | VALIDATION MCC |
|---|---:|---:|
| Incremental logistic regression | 0.14208149 | 0.13614390 |
| Feedforward MLP | 0.14832380 | 0.08894234 |
| Directed bidirectional SGC | 0.33662641 | 0.36085870 |
| **Hybrid graph-feature MLP** | **0.62982119** | **0.50934869** |

MCC means Matthews correlation coefficient; it is not percentage accuracy.

The frozen final development model is:

```text
527 sample features + 119 graph features = 646 input features
646 → 128 → 64 → 32 → 1
Trainable parameters: 93,185
Frozen threshold: 0.4965
```

The hybrid exceeded the GNN by `0.14848999` MCC on validation, so graph-feature fusion was valuable in this experiment.

## Honest validation result

The hybrid development MCC was `0.62982119`, while its separate-vector validation MCC was `0.50934869`. The drop was `0.12047250`. My predeclared maximum acceptable drop was `0.10`, so the validation-retention criterion was **NOT_MET** by `0.02047250` MCC.

The evaluation itself completed correctly and the hybrid remained the strongest model. The result was frozen without retraining on validation, changing the threshold or weakening the acceptance rule. The cause of the retention gap has not been established. Possible explanations can be studied, but they should not be reported as proven without a new controlled experiment.

The untouched HOLDOUT partition was not opened because the validation advancement gate did not pass.

## Behavior-based diagnosis demonstration

The blinded known-catalog pilot used 12 committed cases: two fault-free and ten injected.

| Pilot result | Score |
|---|---:|
| False alarms on fault-free cases | 0 / 2 |
| Observable injected faults detected | 8 / 8 |
| All injected faults detected | 8 / 10 |
| Unique-category cases localized exactly | 4 / 4 |
| All injected cases localized exactly | 4 / 10 |
| True site retained in observable candidate sets | 8 / 8 |

Two injected faults were indistinguishable from normal under the fixed 64 tests. Several observable signatures matched multiple sites. These are observability limits, not cases where a hidden identity should be guessed.

This demonstration used the same HMAC circuit and a known fault catalog. It is **not independent generalization accuracy**.

## What you can experiment with

Please treat this project as a starting point rather than a finished universal solution. Useful experiments include:

- Retrain the models on a different circuit such as AES, SHA, Ibex or an ISCAS benchmark.
- Compare SGC with GCN, GraphSAGE, GAT or another message-passing network.
- Train a smaller student model using distillation from the hybrid model.
- Prune the MLP or reduce the 646 input features.
- Create an INT8 or lower-precision derivative and measure MCC, calibration, speed, memory and model size against the frozen model.
- Improve robustness across test-vector distributions without tuning on the frozen validation labels.
- Study SA0 and SA1 separately or introduce a properly specified multi-task objective.
- Add transient, bridging, delay or multiple faults under a new fault-injection and evaluation contract.
- Rank ambiguous locations instead of returning an unranked candidate set.
- Test real runtime observability on an FPGA.
- Build a clean command-line or frontend interface after reproducing the model results.

If you retrain, change the features, change the decision threshold, quantize the weights or add another fault type, please publish it as a derivative model with a new name and its own evaluation results. Do not replace the frozen V1 file and keep its name.

## What this model cannot currently claim

- It does not accept arbitrary RTL and immediately diagnose it without circuit-specific preprocessing.
- It has not demonstrated transfer to an unseen circuit or chip.
- It covers persistent single SA0/SA1 faults only.
- It does not cover transient, bridging, delay, analog or simultaneous multiple faults.
- It is not validated on an FPGA or fabricated silicon.
- It does not establish production safety, reliability or real-time performance.
- The main supervised model expects a candidate fault description; unknown-fault behavior diagnosis is handled by the separate response-catalog pilot.
- Some observable fault signatures are inherently ambiguous across many physical sites.
- The validation-retention criterion did not pass.
- HOLDOUT results are intentionally absent.

## Repository layout

```text
rtl/                 HMAC wrapper RTL released for this experiment
scripts/             Dataset, graph, training and evaluation utilities
config/              Frozen task, feature, model and evaluation contracts
models/              Frozen model artifact and integrity record
schemas/             Dataset, feature and graph schemas
reports/             Model card and aggregate evaluation reports
examples/            Small non-private usage examples
SHA256SUMS           Integrity hashes for every released file
THIRD_PARTY_NOTICE.md Upstream attribution and licensing notes
```

Large raw campaigns, build directories, virtual environments, simulator binaries, private blinded-case truth files and machine-specific logs are deliberately excluded. The schemas and generation instructions are included so these artifacts can be rebuilt locally.

## Installation

Use a clean Linux or WSL environment. The exact dependency versions recorded by the release should be installed before loading the model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-lock.txt
```

RTL regeneration and fault simulation also require the documented OSS-CAD-Suite tools, including Yosys and Verilator. Internet access is not needed after all dependencies and the pinned OpenTitan revision are available locally.

## Model-file security

The frozen model uses Python `joblib` serialization. **Never load a joblib or pickle file from an untrusted source.** Verify the file against `SHA256SUMS` before loading it, and only use the model distributed through this repository’s verified release.

## Reproducibility rules

1. Verify `SHA256SUMS` before running anything.
2. Keep the OpenTitan revision pinned.
3. Regenerate large datasets instead of committing them to ordinary Git history.
4. Preserve physical-site grouping so SA0 and SA1 versions of one site never cross data partitions.
5. Fit scalers only on the training partition.
6. Do not use post-simulation outcome fields as model inputs.
7. Freeze model selection and thresholds before opening an evaluation partition.
8. Report MCC, confidence intervals and limitations—not only accuracy.

## Quantization policy

The original frozen hybrid model is distributed unchanged. A quantized version must be stored as a separately named derivative, for example:

```text
hmac_hybrid_v1_original.joblib
hmac_hybrid_v1_int8_experimental.<format>
```

Before publishing a quantized derivative, compare it with the original using identical samples and threshold rules. At minimum, report exact prediction agreement, MCC change, PR-AUC change, model size, peak memory and inference time. A smaller file is not automatically a better model.

## License and attribution

The original project code is prepared for release under the Apache License 2.0. OpenTitan and every other third-party component retain their own copyright, license and attribution. Review `LICENSE`, `THIRD_PARTY_NOTICE.md` and the pinned upstream revision before redistributing anything.

## Citation

If this repository helps your work, please cite the release using the included `CITATION.cff`. If you publish an extension, I would genuinely like to see what you build and how it performs on a different circuit.

## Final note

I am publishing this because VLSI fault-detection work becomes more useful when the full experiment—including the limitations—is reproducible. There is plenty left to improve here. If you can make the model smaller, more robust, transferable to another circuit or useful on real hardware, that is exactly the kind of follow-up this project is meant to encourage.

