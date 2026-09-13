#!/usr/bin/env python3
"""Stage 11E-1H — final project disposition and release-readiness freeze.

This stage verifies frozen milestone evidence and writes a deterministic final
research-prototype disposition.  It does not train, infer, simulate, open
HOLDOUT data, construct a release bundle, contact GitHub, or modify prior work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


VERSION = "HMAC-FINAL-PROJECT-DISPOSITION-v1"

INPUTS = {
    # Functional RTL, regression, and synthesis anchors.
    "rtl/hmac/opentitan_hmac_sha256_msg32.sv":
        "d56d0835d7609eaafbd850e76a50ac70b20d7bf0e18e24ab53d185fe34b82e9e",
    "results/hmac_regression_freeze_11a7c.json":
        "3c5e3e536e16ee87378ddda5bfdf4c8ae2bef55a6e97053b419787fa997674d1",
    "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json":
        "0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1",
    "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.v":
        "28a92a28b8ae09ef84c781904f0df950f2c2a5f052ad10d50b76a53fe5286574",

    # Development campaign, canonical data, and leakage-safe features.
    "results/hmac_fault_campaign_11c5/hmac_full_campaign_dataset_integrity_freeze_11c5b.json":
        "d3d346301674d8ea84eee60d72a1377dc53916682ad1bed3461664af6ff09b5c",
    "results/hmac_fault_campaign_11c5/hmac_canonical_dataset_schema_freeze_11c5c.json":
        "a43de512ffab41cddaee286b2f459e9064f9588ee9c4b4621107485445ed3729",
    "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/hmac_leakage_safe_feature_matrix_schema_11c5e.json":
        "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    "results/hmac_fault_campaign_11c5/hmac_leakage_safe_feature_matrix_freeze_11c5e.json":
        "6320bb732b84ae56c937e188189fb87a159d35a164eed892ec3da9d1ee27e540",

    # Graph, hybrid architecture, selected model, and development disposition.
    "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json":
        "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    "results/hmac_fault_campaign_11d1/hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json":
        "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
    "config/diagnostic_model/hmac_hybrid_architecture_11d2a.json":
        "d213a89c53769c217ca9cca71700b03abc23f94d5302bc4309ea63b035016063",
    "config/diagnostic_model/hmac_hybrid_training_contract_11d2a.json":
        "1ccd69a6ee179dbdbce9cc295324c85f28bc4fd98f22b5df010fdcbf21cda41e",
    "results/hmac_fault_campaign_11d2/hmac_hybrid_environment_11d2a.json":
        "1849274a8006e77cced8695be784e70102abceec940fee036163f0a62aa65a57",
    "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib":
        "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_hybrid_selection_lock_11d2b.json":
        "005cecbd94456d3d1aa6805b2dc898e895cc178564673b6b0aa269fcd64260c6",
    "results/hmac_fault_campaign_11d2/hmac_hybrid_training_calibration_freeze_11d2b.json":
        "39f459b3dbb0270810af8b8d39819ad2efecf982ebd929b3683d44cfa2570722",
    "results/hmac_fault_campaign_11d2/hybrid_evaluation_11d2c/hmac_hybrid_vs_frozen_comparators_11d2c.json":
        "a1713dfbaf51227952712d88f7f5aac95388dfef70b3187735d1ca0c7ff47595",
    "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json":
        "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    "results/hmac_fault_campaign_11d2/hmac_hybrid_disposition_final_diagnostic_model_freeze_11d2d.json":
        "8114786a4e026e8f5b7ef482af13c0ff3eb78c8f7815fcd5ae8e0390684d0c05",

    # Independent-vector validation and the preserved failed acceptance result.
    "results/hmac_fault_campaign_11d3/hmac_validation_dataset_integrity_freeze_11d3e.json":
        "ed7b77c65419b2319ccd7a39732a7cabcfe63ff109a37da40373879846a80825",
    "results/hmac_fault_campaign_11d3/hmac_canonical_validation_dataset_schema_freeze_11d3f.json":
        "a0fd16b8cafaac0f1014df56fe20d1d843aebeef8250c56e84f3861a32ae2410",
    "results/hmac_fault_campaign_11d3/hmac_validation_feature_inference_contract_freeze_11d3g.json":
        "04b0ecada12085ba9c9f66cb19b5f01933dab885d8117bfa98dd6c3818f42347",
    "results/hmac_fault_campaign_11d3/hmac_locked_validation_four_model_freeze_11d3h.json":
        "d914ddf75d7477bb024dbc8badeef5c47e23504e9098a82ba3c5fb8e210cd902",
    "results/hmac_fault_campaign_11d3/hmac_validation_result_disposition_holdout_readiness_freeze_11d3i.json":
        "fde310f3c49cbbfc38b5c3e669d2a2206ee5ef2ab326a63e777532ae8663ed5a",
    "results/hmac_fault_campaign_11d3/hmac_validation_failure_review_project_disposition_freeze_11d3j.json":
        "f0771b64be274e00aff66865fedb72233d141b1f178d1ffd5819f37f266d9dae",

    # Blinded behavior-based detection/localization demonstration.
    "stage_11e1g_behavior_pilot_disposition.py":
        "5ff6cbf9e130b5b339b96f42fd02b48794e8a43e3ef8af5c00189e490a3a979d",
    "config/diagnostic_model/hmac_behavior_diagnosis_pilot_disposition_policy_11e1g.json":
        "9e46bdc7973d27046f6169ef84cfc59e9aff46a81043a3f1f841919c6a20f45e",
    "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_report_11e1g.md":
        "588b58ea47b046b99325daa912d6c370999a80346d0d39daf86a82ddbeb4f06b",
    "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_disposition_freeze_11e1g.json":
        "3a402e789346bdc99fbe0ef3629031e45d5f4b9556ece5b91de28e5bace1c7a4",
}

RESULT_ROOT = "results/hmac_project_final_11e1h"
DISPOSITION = "config/release/hmac_final_project_disposition_11e1h.json"
RELEASE_POLICY = "config/release/hmac_reproducible_release_policy_11e1h.json"
INVENTORY = RESULT_ROOT + "/hmac_reproducible_release_inventory_11e1h.json"
MODEL_CARD = RESULT_ROOT + "/hmac_final_research_prototype_model_card_11e1h.md"
AUDIT = RESULT_ROOT + "/hmac_final_project_release_readiness_freeze_11e1h.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "JSON root is not an object: " + Path(path).name)
    return value


def close(actual, expected, tolerance=5e-8):
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def record(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "artifact outside project root")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "existing final output differs: " + path.name)
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def scalar_values(value):
    """Yield scalar values without assuming an earlier stage's field spelling."""
    if isinstance(value, dict):
        for item in value.values():
            yield from scalar_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from scalar_values(item)
    else:
        yield value


def same_scalar(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and close(actual, expected)
    return type(actual) is type(expected) and actual == expected


def require_values(document, expected, label):
    values = list(scalar_values(document))
    for wanted in expected:
        require(any(same_scalar(actual, wanted) for actual in values),
                f"{label}: required value absent: {wanted!r}")


def verify_semantics(root, docs):
    """Bind the final disposition to exact files and their essential meanings.

    Earlier stages did not use one uniform JSON key vocabulary.  INPUTS pins
    every accepted byte sequence; these checks intentionally verify semantic
    values without inventing a new field-name contract for already-frozen data.
    """
    regression = docs["results/hmac_regression_freeze_11a7c.json"]
    require_values(regression, ["PASS"], "functional regression")

    train_integrity = docs[
        "results/hmac_fault_campaign_11c5/hmac_full_campaign_dataset_integrity_freeze_11c5b.json"]
    require_values(train_integrity,
                   ["PASS", "FROZEN", 45, 22839, 45678, 64, 2926272, 0],
                   "development dataset freeze")

    train_canonical = docs[
        "results/hmac_fault_campaign_11c5/hmac_canonical_dataset_schema_freeze_11c5c.json"]
    require_values(train_canonical, ["PASS", "FROZEN", 2926272, 22839],
                   "canonical development dataset")

    features = docs[
        "results/hmac_fault_campaign_11c5/hmac_leakage_safe_feature_matrix_freeze_11c5e.json"]
    require_values(features, ["PASS", "FROZEN", 2923392, 527, False, 0],
                   "development feature freeze")

    graph = docs[
        "results/hmac_fault_campaign_11d1/hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"]
    require_values(graph, ["PASS", "FROZEN", 22839, 47617, False],
                   "golden-netlist graph freeze")

    model_lock = docs["config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json"]
    model_relative = (
        "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/"
        "hmac_selected_hybrid_model_11d2b.joblib")
    require_values(model_lock,
                   ["PASS", "FROZEN",
                    "HYBRID_SGC3_MLP_128_64_32_A1E4_LR1E3", 0.4965,
                    INPUTS[model_relative], False],
                   "final development model lock")
    require(sha(root / model_relative) == INPUTS[model_relative],
            "final model SHA binding")

    dev = docs[
        "results/hmac_fault_campaign_11d2/hmac_hybrid_disposition_final_diagnostic_model_freeze_11d2d.json"]
    require_values(dev, ["PASS", "FROZEN", "HYBRID", 0.62982119, 0.33662641,
                         "MET", False],
                   "hybrid development disposition")

    validation_data = docs[
        "results/hmac_fault_campaign_11d3/hmac_validation_dataset_integrity_freeze_11d3e.json"]
    require_values(validation_data, ["PASS", "FROZEN", 45, 48, 2194704, 0],
                   "validation dataset freeze")

    validation = docs[
        "results/hmac_fault_campaign_11d3/hmac_locked_validation_four_model_freeze_11d3h.json"]
    require_values(validation, ["PASS", "FROZEN", "NOT_MET", 0.50934869,
                                0.36085870, False, True, 0],
                   "locked validation result")

    disposition = docs[
        "results/hmac_fault_campaign_11d3/hmac_validation_result_disposition_holdout_readiness_freeze_11d3i.json"]
    require_values(disposition, ["PASS", "NOT_MET", "mcc_retention", 0.12047250,
                                 0.1, "NOT_AUTHORIZED", "NOT_DECLARED"],
                   "validation disposition")

    review = docs[
        "results/hmac_fault_campaign_11d3/hmac_validation_failure_review_project_disposition_freeze_11d3j.json"]
    require_values(review, ["PASS", "FROZEN", True,
                            "ACCEPTANCE_NOT_MET; STOP_BEFORE_HOLDOUT",
                            "NOT_ESTABLISHED_BY_THIS_REVIEW", "BLOCKED", False],
                   "validation failure review")

    behavior = docs[
        "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_disposition_freeze_11e1g.json"]
    require_values(behavior, ["PASS", "FROZEN", "DEMONSTRATED_AND_FROZEN", 12,
                              False],
                   "behavior-diagnosis pilot disposition")


def main(root):
    root = root.resolve()
    require(root.name == "vlsi_fault_detection_v2" and Path(__file__).resolve().parent == root,
            "copy this script to ~/vlsi_fault_detection_v2 and run it from there")
    source_sha = sha(Path(__file__))
    print("STAGE 11E-1H — FINAL PROJECT DISPOSITION AND REPRODUCIBLE-RELEASE READINESS")
    print("FROZEN EVIDENCE VERIFICATION")
    evidence = {}
    docs = {}
    for relative, expected in INPUTS.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), "missing frozen input: " + relative)
        require(sha(path) == expected, "SHA mismatch: " + relative)
        evidence[relative] = record(root, path)
        if path.suffix == ".json":
            docs[relative] = load_json(path)
        print("  " + Path(relative).name.ljust(76) + ": OK", flush=True)

    verify_semantics(root, docs)
    require(sha(Path(__file__)) == source_sha, "Stage 11E-1H source changed during verification")
    evidence_commitment = hashlib.sha256(encode(evidence)).hexdigest()

    disposition = {
        "stage": "11E-1H",
        "version": VERSION,
        "status": "PASS",
        "project_lifecycle_status": "COMPLETED_AND_FROZEN_RESEARCH_PROTOTYPE",
        "final_project_freeze": "DECLARED",
        "completion_interpretation":
            "The defined research workflow is complete and immutable; completion does not convert an unmet acceptance criterion into a pass.",
        "project_owner_and_release_maintainer": "Pavan Nithin",
        "upstream_attribution": "OpenTitan and all third-party components retain their original ownership and licenses.",
        "hardware_target": "OpenTitan HMAC-SHA256 message-32 wrapper",
        "fault_scope": "Single persistent net-stem SA0/SA1 faults",
        "golden_netlist_cells": 22839,
        "persistent_fault_instances": 45678,
        "development_train_vectors": 64,
        "development_model_samples": 2923392,
        "validation_vectors": 48,
        "validation_model_samples": 2192544,
        "holdout_vectors_executed": 0,
        "final_development_model": {
            "model_id": "HYBRID_FUSION_MLP_11D2C",
            "family": "Graph-augmented feature-fusion MLP",
            "architecture": "646 -> 128 -> 64 -> 32 -> 1",
            "trainable_parameters": 93185,
            "threshold": 0.4965,
            "artifact": evidence[
                "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"],
            "dev_site_test_mcc": 0.62982119,
            "validation_mcc": 0.50934869,
            "validation_gnn_mcc": 0.36085870,
            "validation_hybrid_minus_gnn_mcc": 0.14848999,
        },
        "development_outcome": "TARGET_MET",
        "validation_execution": "PASS_AND_FROZEN",
        "validation_acceptance": "NOT_MET",
        "failed_validation_criterion": "mcc_retention",
        "validation_mcc_drop": 0.12047250,
        "maximum_allowed_mcc_drop": 0.10000000,
        "cause_of_retention_failure": "NOT_ESTABLISHED",
        "behavior_diagnosis_pilot": {
            "status": "DEMONSTRATED_AND_FROZEN",
            "method": "Reference checker plus exact response-catalog lookup",
            "fault_identity_supplied_to_diagnosis": False,
            "observable_fault_detections": "8/8",
            "all_injected_fault_detections": "8/10",
            "unique_exact_site": "4/4",
            "observable_candidate_coverage": "8/8",
            "independent_generalization": "NOT_ESTABLISHED",
        },
        "holdout_disposition": "BLOCKED_AND_NOT_EXECUTED",
        "retraining_or_threshold_change": "PROHIBITED_FOR_THIS_FROZEN_VERSION",
        "production_or_silicon_deployment_readiness": "NOT_ESTABLISHED",
        "public_release_claim_level": "COMPLETED RESEARCH PROTOTYPE WITH DOCUMENTED VALIDATION LIMITATION",
        "release_bundle_generation": "AUTHORIZED",
        "public_github_push": "NOT YET AUTHORIZED; VERIFY THE SANITIZED RELEASE BUNDLE FIRST",
        "dashboard_or_frontend": "DEFER UNTIL THE VERIFIED GITHUB RELEASE IS CREATED",
        "next_gate": "STAGE 11E-1I — REPRODUCIBLE GITHUB RELEASE BUNDLE GENERATION AND SANITIZATION FREEZE",
    }

    release_policy = {
        "stage": "11E-1H",
        "version": VERSION,
        "status": "PASS",
        "release_readiness_status": "FROZEN",
        "release_bundle_generation_authorized": True,
        "public_push_authorized": False,
        "frontend_work_authorized": False,
        "release_owner": "Pavan Nithin",
        "required_release_components": [
            "README with setup, training, inference, limitations, and reproduction commands",
            "Project LICENSE plus third-party attribution and OpenTitan license/commit notice",
            "Frozen hybrid model artifact and safe-use warning for joblib deserialization",
            "Model card containing development and validation results",
            "Architecture, feature, graph, split, training, and inference contracts",
            "Dataset schemas, commitments, and regeneration instructions—not multi-gigabyte raw data",
            "Source scripts required to rebuild datasets, train comparators, evaluate, and reproduce the behavior pilot",
            "Python requirements and OSS-CAD-Suite/OpenTitan environment documentation",
            "SHA-256 release manifest covering every published file",
            "CITATION.cff and clear maintainer/authorship information",
        ],
        "mandatory_exclusions": [
            ".venv/**",
            "build/**",
            "third_party/opentitan/** as copied source; use a pinned upstream commit/submodule instead",
            "results/**/private/**",
            "**/entropy.json",
            "**/truth.json",
            "**/capture_plan.json",
            "raw campaign CSV files and generated simulator binaries",
            "large NPZ/CSV/GZIP datasets unless separately released with explicit size/license review",
            "credentials, tokens, private keys, shell history, machine-specific paths, and personal files",
        ],
        "mandatory_pre_push_checks": [
            "Secret scan passes",
            "Private blinded-case artifacts are absent",
            "Every release file matches the release manifest",
            "A clean-environment smoke test passes",
            "Model-loading and example inference work without retraining",
            "License and upstream attribution review passes",
            "Large-file policy and Git LFS decision are documented",
            "No claim describes MCC as percentage accuracy",
            "README states that validation retention was NOT_MET and HOLDOUT was not opened",
        ],
        "model_change_policy":
            "Any retraining, threshold change, feature change, new fault type, or new circuit must use a new version and new evaluation protocol.",
        "next_gate": disposition["next_gate"],
    }

    inventory = {
        "stage": "11E-1H",
        "version": VERSION,
        "status": "PASS",
        "purpose": "Input plan for a later sanitized release-bundle generator; this is not the bundle itself.",
        "frozen_evidence_commitment": evidence_commitment,
        "verified_input_count": len(evidence),
        "verified_inputs": evidence,
        "release_primary_model": disposition["final_development_model"]["artifact"],
        "release_by_reference_or_regeneration": [
            "OpenTitan source at commit 83fc48ed3a727399056772d12be8c7d4a8a276f0",
            "Canonical development and validation datasets",
            "Golden netlist and fault-batch build products",
            "Feature matrices and graph NPZ products",
        ],
        "never_publish_from_private_pilot_directories": True,
        "public_aggregate_behavior_report_allowed":
            "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_report_11e1g.md",
        "release_bundle_created": False,
        "github_contacted": False,
    }

    model_card = """# OpenTitan HMAC VLSI Fault-Detection Research Prototype

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
"""

    disposition_path = root / DISPOSITION
    release_policy_path = root / RELEASE_POLICY
    inventory_path = root / INVENTORY
    model_card_path = root / MODEL_CARD
    audit_path = root / AUDIT
    write_once(disposition_path, encode(disposition))
    write_once(release_policy_path, encode(release_policy))
    write_once(inventory_path, encode(inventory))
    write_once(model_card_path, model_card.encode())

    outputs = {
        "disposition": record(root, disposition_path),
        "release_policy": record(root, release_policy_path),
        "release_inventory": record(root, inventory_path),
        "model_card": record(root, model_card_path),
    }
    # Recheck every bound byte immediately before publishing the final audit.
    for relative, expected in INPUTS.items():
        require(sha(root / relative) == expected, "final input recheck failed: " + relative)
    require(sha(Path(__file__)) == source_sha, "Stage 11E-1H source changed")

    audit = {
        "stage": "11E-1H",
        "version": VERSION,
        "status": "PASS",
        "project_lifecycle_status": "COMPLETED_AND_FROZEN_RESEARCH_PROTOTYPE",
        "project_final_freeze": "DECLARED",
        "final_diagnostic_model_status": "FROZEN",
        "behavior_diagnosis_pilot_status": "FROZEN",
        "validation_execution_status": "PASS_AND_FROZEN",
        "original_validation_acceptance": "NOT_MET",
        "failed_validation_criteria": ["mcc_retention"],
        "holdout_status": "BLOCKED_AND_NOT_EXECUTED",
        "production_readiness": "NOT_ESTABLISHED",
        "release_readiness_status": "FROZEN",
        "release_bundle_generation": "AUTHORIZED",
        "public_github_push": "NOT_YET_AUTHORIZED",
        "dashboard_frontend": "DEFERRED_UNTIL_VERIFIED_GITHUB_RELEASE",
        "source_sha256": source_sha,
        "verified_input_count": len(evidence),
        "verified_input_commitment": evidence_commitment,
        "input_evidence": evidence,
        "outputs": outputs,
        "model_training_calls": 0,
        "model_inference_calls": 0,
        "simulation_runs": 0,
        "validation_holdout_payloads_read": 0,
        "frozen_inputs_modified": False,
        "completion_rule": "THIS_AUDIT_AND_ALL_MATCHING_OUTPUT_HASHES_REQUIRED",
        "next_gate": disposition["next_gate"],
    }
    write_once(audit_path, encode(audit))

    print("\nSTAGE 11E-1H — FINAL PROJECT DISPOSITION AND REPRODUCIBLE-RELEASE READINESS FREEZE")
    print("Status                         : PASS")
    print("Project lifecycle              : COMPLETED AND FROZEN RESEARCH PROTOTYPE")
    print("Final project freeze           : DECLARED")
    print("Final development model        : HYBRID FUSION MLP")
    print("Model architecture             : 646 -> 128 -> 64 -> 32 -> 1")
    print("Trainable parameters           : 93185")
    print("Development MCC                : 0.62982119")
    print("Validation MCC                 : 0.50934869")
    print("Original validation acceptance : NOT_MET (mcc_retention)")
    print("HOLDOUT                        : BLOCKED / NOT EXECUTED")
    print("Behavior diagnosis pilot       : DEMONSTRATED AND FROZEN")
    print("Independent generalization     : NOT ESTABLISHED")
    print("Production readiness           : NOT ESTABLISHED")
    print("Release-bundle generation      : AUTHORIZED")
    print("Public GitHub push             : NOT YET AUTHORIZED")
    print("Dashboard / frontend           : DEFERRED UNTIL VERIFIED RELEASE")
    print("Disposition                    :", disposition_path)
    print("Disposition SHA                :", sha(disposition_path))
    print("Release policy                 :", release_policy_path)
    print("Release policy SHA             :", sha(release_policy_path))
    print("Release inventory              :", inventory_path)
    print("Release inventory SHA          :", sha(inventory_path))
    print("Model card                     :", model_card_path)
    print("Model card SHA                 :", sha(model_card_path))
    print("Audit                          :", audit_path)
    print("Audit SHA                      :", sha(audit_path))
    print("Next gate                      :", disposition["next_gate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        main(args.project_root)
    except KeyboardInterrupt:
        print("STOP: interrupted; no frozen input was modified.")
        raise SystemExit(130)
    except Exception as error:
        print("STOP:", error)
        print("Preserve all existing project evidence. Public release remains unauthorized.")
        raise SystemExit(1)
