# Validation Failure Review and Project Disposition — Stage 11D-3J

**Review completed. Current experiment: acceptance NOT_MET; stop before HOLDOUT.**

The hybrid is the strongest of the four evaluated models. Only the predeclared MCC-retention criterion failed. The evaluation and deterministic replay passed; this is a performance shortfall, not an execution failure.

## Evidence and acceptance

| Model | Validation MCC | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.50934869 | 0.75626316 | 0.68186985 | 0.80936047 | 0.74016535 | 0.83262812 | 0.85633393 | 0.17022065 |
| gnn | 0.36085870 | 0.64246092 | 0.81052102 | 0.34906883 | 0.48797879 | 0.69775720 | 0.71453098 | 0.23949327 |
| baseline | 0.13614390 | 0.56426897 | 0.53733704 | 0.39775413 | 0.45712768 | 0.52640103 | 0.59169292 | 0.38866364 |
| deep | 0.08894234 | 0.54474415 | 0.48639181 | 0.52661577 | 0.50570519 | 0.49844587 | 0.56044913 | 0.24933299 |

The hybrid MCC dropped from 0.62982119 to 0.50934869. The drop of 0.12047250 exceeded the allowed 0.10000000; the retention threshold was missed by 0.02047250 MCC.

The hybrid–GNN gain remains 0.14848999, with paired-site 95% interval [0.14111904393687372, 0.1556260512510558]. This supports a gain over GNN under the stated same-circuit evaluation, but cannot override the failed retention criterion.

| Acceptance family | Frozen result |
|---|---|
| absolute_performance | PASS |
| comparator_confirmation | PASS |
| generalization_retention | NOT_MET |
| minimum_validity | PASS |

## Site-population comparison

The acceptance contract compares a development score on 3,426 DEV_SITE_TEST sites with validation across all 22,839 sites. The following table separates validation by the original site partitions. These descriptive results do not change the acceptance population or threshold.

| Original development site partition | Validation samples | Hybrid MCC | GNN MCC | MCC difference |
|---|---:|---:|---:|---:|
| DEV_TRAIN | 1534752 | 0.51261893 | 0.35781528 | 0.15480365 |
| DEV_CALIBRATION | 328896 | 0.49660879 | 0.35560951 | 0.14099928 |
| DEV_SITE_TEST | 328896 | 0.50686271 | 0.38009922 | 0.12676349 |

On the original DEV_SITE_TEST sites, validation hybrid MCC is 0.50686271; the earlier development score was 0.62982119. The difference is -0.12295848. The vector sets and their sizes differ, so this does not isolate a causal effect.

## Fault and cell groups

All saved SA0/SA1, site-category, and driver-cell groups are shown below. Comparisons are descriptive; no per-group confidence intervals or multiple-comparison significance tests were added.

| Family | Group | Samples | Hybrid MCC | GNN MCC | MCC difference |
|---|---|---:|---:|---:|---:|
| fault_model | `SA0` | 1096272 | 0.50485130 | 0.37204203 | 0.13280927 |
| fault_model | `SA1` | 1096272 | 0.50890480 | 0.34559976 | 0.16330504 |
| site_category | `COMBINATIONAL_LOGIC_STEM` | 1842624 | 0.51887953 | 0.38649962 | 0.13237991 |
| site_category | `SEQUENTIAL_STATE_STEM` | 349920 | 0.44822922 | 0.23769187 | 0.21053735 |
| driver_cell_type | `$_AND_` | 569568 | 0.41604616 | 0.28111659 | 0.13492957 |
| driver_cell_type | `$_DFFE_PN0P_` | 291936 | 0.49237505 | 0.25925538 | 0.23311967 |
| driver_cell_type | `$_DFFE_PN1P_` | 96 | 0.00000000 | 0.00000000 | 0.00000000 |
| driver_cell_type | `$_DFFE_PP_` | 49152 | 0.05384468 | 0.00750824 | 0.04633644 |
| driver_cell_type | `$_DFF_PN0_` | 8736 | 0.91372763 | 0.46515563 | 0.44857199 |
| driver_cell_type | `$_MUX_` | 546912 | 0.69108964 | 0.48550088 | 0.20558876 |
| driver_cell_type | `$_NOT_` | 42336 | 0.42907211 | 0.25519049 | 0.17388163 |
| driver_cell_type | `$_OR_` | 288864 | 0.43323934 | 0.30867955 | 0.12455979 |
| driver_cell_type | `$_XOR_` | 394944 | 0.44282411 | 0.35263660 | 0.09018751 |

## Validation vectors

Across 48 vector groups, hybrid MCC ranges from 0.33475244 to 0.67189261. Hybrid point MCC exceeds GNN in 46 groups. These are repeated measurements on the same circuit sites, not independent chips.

| Validation vector ID | Samples | Hybrid MCC | GNN MCC | MCC difference |
|---|---:|---:|---:|---:|
| 1 | 45678 | 0.36653961 | 0.34367288 | 0.02286673 |
| 4 | 45678 | 0.60623466 | 0.31543770 | 0.29079696 |
| 13 | 45678 | 0.66825757 | 0.30311165 | 0.36514592 |
| 14 | 45678 | 0.67189261 | 0.29795505 | 0.37393755 |
| 16 | 45678 | 0.66778055 | 0.30046699 | 0.36731355 |
| 26 | 45678 | 0.61883934 | 0.33636817 | 0.28247117 |
| 37 | 45678 | 0.53955466 | 0.35536034 | 0.18419432 |
| 38 | 45678 | 0.50855647 | 0.39174052 | 0.11681595 |
| 41 | 45678 | 0.54555594 | 0.39784130 | 0.14771464 |
| 46 | 45678 | 0.56377920 | 0.38726968 | 0.17650953 |
| 54 | 45678 | 0.55539163 | 0.37700793 | 0.17838370 |
| 60 | 45678 | 0.44690621 | 0.35725060 | 0.08965560 |
| 62 | 45678 | 0.58017185 | 0.36675484 | 0.21341701 |
| 64 | 45678 | 0.55468131 | 0.39155415 | 0.16312716 |
| 72 | 45678 | 0.48992198 | 0.30961427 | 0.18030771 |
| 75 | 45678 | 0.33475244 | 0.37095286 | -0.03620043 |
| 79 | 45678 | 0.43427246 | 0.31165478 | 0.12261768 |
| 86 | 45678 | 0.50973681 | 0.40032962 | 0.10940719 |
| 91 | 45678 | 0.53449339 | 0.36604339 | 0.16845000 |
| 109 | 45678 | 0.57133177 | 0.37764717 | 0.19368460 |
| 114 | 45678 | 0.57399773 | 0.38953686 | 0.18446087 |
| 116 | 45678 | 0.52666514 | 0.39833298 | 0.12833216 |
| 120 | 45678 | 0.52198794 | 0.39371826 | 0.12826968 |
| 122 | 45678 | 0.52078042 | 0.38160848 | 0.13917194 |
| 124 | 45678 | 0.51094331 | 0.38897557 | 0.12196774 |
| 137 | 45678 | 0.52596060 | 0.39203277 | 0.13392783 |
| 139 | 45678 | 0.59397421 | 0.39362968 | 0.20034453 |
| 143 | 45678 | 0.46478516 | 0.37045334 | 0.09433183 |
| 153 | 45678 | 0.48181454 | 0.39887037 | 0.08294417 |
| 171 | 45678 | 0.57787230 | 0.40183309 | 0.17603922 |
| 173 | 45678 | 0.52930312 | 0.38778942 | 0.14151371 |
| 178 | 45678 | 0.55612199 | 0.41206001 | 0.14406199 |
| 179 | 45678 | 0.41797391 | 0.38123918 | 0.03673474 |
| 181 | 45678 | 0.55471428 | 0.27290133 | 0.28181296 |
| 188 | 45678 | 0.38119763 | 0.32917543 | 0.05202220 |
| 190 | 45678 | 0.49254990 | 0.38651796 | 0.10603193 |
| 191 | 45678 | 0.51027087 | 0.36720280 | 0.14306807 |
| 192 | 45678 | 0.34871754 | 0.39119521 | -0.04247767 |
| 214 | 45678 | 0.44889078 | 0.31844363 | 0.13044714 |
| 232 | 45678 | 0.53449237 | 0.38743173 | 0.14706064 |
| 240 | 45678 | 0.48943604 | 0.39029899 | 0.09913706 |
| 241 | 45678 | 0.55128010 | 0.37732859 | 0.17395151 |
| 242 | 45678 | 0.55787610 | 0.39424801 | 0.16362809 |
| 245 | 45678 | 0.52192651 | 0.40342821 | 0.11849830 |
| 246 | 45678 | 0.50718422 | 0.35936182 | 0.14782240 |
| 247 | 45678 | 0.53785656 | 0.39336976 | 0.14448680 |
| 249 | 45678 | 0.53508414 | 0.38076894 | 0.15431520 |
| 251 | 45678 | 0.57528182 | 0.39745634 | 0.17782548 |

## What this review establishes

The recorded failure is the retention threshold breach. The underlying cause is not established. Overfitting, stimulus distribution differences, and site-population differences must not be presented as proven explanations from these aggregate reports.

new key/message pairs on the same circuit; report all-site metrics and separate original DEV_TRAIN/DEV_CALIBRATION/DEV_SITE_TEST site strata

Conditional site-bootstrap intervals on one circuit and 48 shared validation vectors. Development site tests were used previously across model families; this is a new-vector validation, not a new-chip test.

The task predicts whether a simulated persistent SA0/SA1 fault will be detected, before simulation. It is not a claim of fault localization, transient-fault coverage, or physical-chip validation. MCC is not percentage accuracy.

## Project disposition

1. Preserve the current run as an experiment with unmet validation acceptance. The hybrid remains the best observed comparator, without approval for advancement.
2. Keep HOLDOUT unopened and preserve all models, thresholds, data, and acceptance rules.
3. Use this report and the frozen audit in the mentor review. Final project completion and deployment readiness have not been established.
4. If further research is chosen, discuss a separately scoped, prospectively specified study with independent evaluation. This review does not authorize retraining, tuning on validation, a threshold waiver, or HOLDOUT execution.

The requested failure review is complete. No additional automatic execution stage is created.

## Suggested mentor explanation

“We completed the HMAC SA0/SA1 development campaign and evaluated four frozen models on 48 unseen validation vectors. The hybrid achieved MCC 0.5093 and outperformed GNN by 0.1485. It passed the other acceptance checks but exceeded the permitted development-to-validation MCC drop. We preserved the failed result, kept HOLDOUT unopened, and documented the limitations. The broader project is not yet declared complete.”

## Verification limits

This review verifies the frozen report chain, saved output hashes, acceptance reconstruction, confusion-metric consistency, group totals, and bootstrap summary consistency through the prior verifier. It reads summary JSON only for interpretation. It does not regenerate predictions or recompute rank metrics from samples.
