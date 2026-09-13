# Stage 11E-1G — Behavior-Diagnosis Pilot Disposition

## Disposition

The blinded behavior-diagnosis pilot is **demonstrated and frozen**. It accepts only the ordered responses from 64 HMAC tests. The diagnosis worker does not receive the injected fault identity. Predictions were committed before the private truth was opened for scoring.

## Frozen demonstration result

| Measure | Result |
|---|---:|
| Fault-free false alarms | 0/2 |
| Observable injected faults detected | 8/8 |
| All injected faults detected | 8/10 |
| Unique-category exact-site localization | 4/4 |
| All injected faults exactly localized | 4/10 |
| Observable faults whose candidate set contained the true site | 8/8 |
| Canonical/replay disagreements | 0 |

The two undetected injected faults were deliberately selected from faults whose response is indistinguishable from the normal response under the fixed 64-vector suite. Four observable faults had unique catalog signatures. Four observable faults had signatures shared by multiple sites and were therefore correctly returned as candidate sets rather than false exact locations.

## Valid scope

This result is a small, stratified, same-circuit, known-catalog reproducibility demonstration for single persistent SA0/SA1 faults in the frozen OpenTitan HMAC netlist. It is not independent generalization accuracy and does not establish support for transient, bridging, multiple, analog, unseen-circuit, or out-of-catalog faults.

## Project relationship

The original four-model validation result remains **NOT_MET** and is unchanged. No model was retrained, no threshold changed, and no VALIDATION or HOLDOUT sample was accessed in this stage. The overall project final freeze is not declared by Stage 11E-1G.

Next gate: **Stage 11E-1H — Final Project Disposition and Reproducible-Release Readiness Freeze**.
