# OpenTitan HMAC VLSI Fault-Detection Research Prototype

## Final disposition

This project is complete and frozen as a research prototype. The completion statement means that the planned hardware, fault-campaign, model-comparison, validation, failure-review, and blinded behavior-diagnosis workflows were executed and preserved. It does not mean that every predeclared performance criterion passed or that the system is production-ready.

Maintainer and project release owner: **Pavan Nithin**. OpenTitan and other third-party components remain the property of their respective authors and must retain their original attribution and licenses.

## Circuit and fault scope

- Circuit: OpenTitan HMAC-SHA256 32-byte-message wrapper.
- Frozen golden netlist: 22,839 cells/sites and 47,617 collapsed directed graph edges.
- Fault catalog: 45,678 single persistent SA0/SA1 instances.
- Development: 64 TRAIN vectors and 2,923,392 enabled samples.
- Validation: 48 separate VALIDATION vectors and 2,192,544 enabled samples.
- HOLDOUT: not opened or executed.

## Frozen diagnostic models

| Model | DEV_SITE_TEST MCC | VALIDATION MCC |
|---|---:|---:|
| Conventional logistic baseline | 0.14208149 | 0.13614390 |
| Deep MLP | 0.14832380 | 0.08894234 |
| Directed bidirectional SGC | 0.33662641 | 0.36085870 |
| Hybrid graph-feature MLP | **0.62982119** | **0.50934869** |

The final development model is the hybrid graph-augmented feature-fusion MLP with architecture `646 → 128 → 64 → 32 → 1`, 93,185 trainable parameters, and frozen threshold 0.4965.

## Validation result

The validation execution and evaluation passed technically, and the hybrid remained the best model. Its MCC exceeded the GNN by 0.14848999. However, hybrid MCC fell by 0.12047250 from development site testing to validation, exceeding the predeclared maximum permitted drop of 0.10. Therefore, the original validation acceptance target is **NOT_MET**. The cause of this retention failure was not established, and HOLDOUT remained blocked.

## Behavior-based detection and localization pilot

The separate blinded pilot diagnoses from 64 observed HMAC responses without giving fault identity to the diagnosis worker. In 12 committed cases it produced zero false alarms on two fault-free cases, detected all eight observable injected faults, exactly localized all four uniquely identifiable cases, and retained the true site in all eight observable candidate sets. Two injected faults were behaviorally indistinguishable from normal under the fixed vectors.

This is a same-circuit known-catalog demonstration, not independent generalization accuracy. It does not establish transient, bridging, multiple, analog, unseen-circuit, out-of-catalog, FPGA, or fabricated-silicon performance.

## Release status

The frozen research project authorizes preparation of a sanitized reproducible GitHub bundle. Public push is not yet authorized until the bundle passes secret, privacy, licensing, size, dependency, manifest, and clean-environment checks. Dashboard and frontend work should begin only after that verified model release.
