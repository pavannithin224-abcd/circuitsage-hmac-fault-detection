#!/usr/bin/env python3
"""Stage 11E-1G: verify and freeze the blinded behavior-diagnosis pilot.

This is a read-only verification/disposition stage.  It performs no simulation,
model training, threshold selection, validation/HOLDOUT access, or new case
selection.  Run from the vlsi_fault_detection_v2 project root.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import types


VERSION = "HMAC-BEHAVIOR-PILOT-DISPOSITION-v1"
EXECUTION_SOURCE = "stage_11e1f_blinded_execution.py"
EXECUTION_SOURCE_SHA = "b0a66d1e2824dde6a34d9ce889a8b53847848c43f979c2dacdcece44623d3daf"
EXECUTION = "results/hmac_behavior_diagnosis_pilot/execution_11e1f"
EXECUTION_FREEZE_SHA = "b4a83f640bf0ea024625dc8c2e59d6d30f6048e72b0eb7cb979a911f968a2677"
PREDICTION_COMMITMENT_SHA = "893ad7096decbcd47dcaab26404174fdbbbbf3d1fe9651d498f3dc02f979c523"
SCORING_REPORT_SHA = "4c1dd51b8d1e9c152b98b0d6842039994bf8d8bd4a7b6e147516d2f503598c5c"

POLICY = "config/diagnostic_model/hmac_behavior_diagnosis_pilot_disposition_policy_11e1g.json"
REPORT = "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_report_11e1g.md"
AUDIT = "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_disposition_freeze_11e1g.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verified_module(path, expected_sha):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha,
            "source SHA mismatch: " + Path(path).name)
    result = types.ModuleType("_verified_" + Path(path).stem)
    result.__file__ = str(Path(path).resolve())
    exec(compile(data, str(path), "exec"), result.__dict__)
    return result


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data,
                "existing output differs; preserve it and stop: " + path.name)
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def relative_record(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "artifact escapes project root")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def verify_record(record, allowed_root=None):
    require(set(record) == {"path", "sha256", "bytes"}, "invalid artifact record schema")
    path = Path(record["path"])
    require(path.is_absolute(), "execution artifact record is not absolute")
    resolved = path.resolve()
    require(str(resolved) == record["path"], "execution artifact path is not canonical")
    if allowed_root is not None:
        require(resolved.is_relative_to(allowed_root.resolve()), "execution artifact escapes its bundle")
    require(resolved.is_file() and not resolved.is_symlink(), "execution artifact missing or linked")
    require(resolved.stat().st_size == record["bytes"] and sha(resolved) == record["sha256"],
            "execution artifact changed: " + resolved.name)
    return resolved


def exact_fraction(report, key, numerator, denominator):
    require(report[key] == {"numerator": numerator, "denominator": denominator},
            "unexpected scoring metric: " + key)


def main(root):
    root = root.resolve()
    source_path = root / EXECUTION_SOURCE
    source_sha = sha(Path(__file__))
    execution_module = verified_module(source_path, EXECUTION_SOURCE_SHA)

    # This verified call rechecks the complete 11E-1A through 11E-1E chain,
    # including the private preparation commitment.  It performs no capture.
    (preparer, runner_inputs, baseline, contract, evidence, suite, vectors,
     batch_netlists, template, truth, capture_plan) = execution_module.verified_inputs(root)

    execution_dir = evidence.path(EXECUTION)
    public_dir = execution_dir / "public"
    private_dir = execution_dir / "private"
    require(execution_dir.is_dir() and public_dir.is_dir() and private_dir.is_dir(),
            "Stage 11E-1F execution bundle is incomplete")

    freeze_path = evidence.check(EXECUTION + "/freeze.json", EXECUTION_FREEZE_SHA)
    prediction_commitment_path = evidence.check(
        EXECUTION + "/public/prediction_commitment.json", PREDICTION_COMMITMENT_SHA)
    scoring_report_path = evidence.check(
        EXECUTION + "/public/scoring_report.json", SCORING_REPORT_SHA)
    evidence.check(EXECUTION_SOURCE, EXECUTION_SOURCE_SHA)

    freeze = read_json(freeze_path)
    require(freeze["stage"] == "11E-1F" and freeze["version"] == execution_module.VERSION and
            freeze["status"] == "PASS", "11E-1F closure status")
    require(freeze["cases"] == 12 and freeze["canonical_rows"] == 768 and
            freeze["replay_rows"] == 768, "11E-1F execution dimensions")
    require(freeze["worker_denied_probes"] == 7 and freeze["worker_exact_replay"] is True and
            freeze["predictions_committed_before_scoring"] is True,
            "11E-1F blinded-worker or commitment controls")
    require(freeze["frozen_inputs_changed"] is False and
            freeze["model_training_performed"] is False and
            freeze["holdout_validation_samples_selected"] == 0 and
            freeze["independent_accuracy_claim"] is False and
            freeze["original_validation_target"] == "NOT_MET; PRESERVED" and
            freeze["project_final_freeze_declared"] is False,
            "11E-1F scope or project disposition")
    require(freeze["binding"]["source_sha256"] == EXECUTION_SOURCE_SHA and
            freeze["binding"]["truth_commitment"] == execution_module.TRUTH_SHA and
            freeze["binding"]["authorization_sha256"] == execution_module.AUTH_SHA,
            "11E-1F source/truth/authorization binding")

    # verified_inputs recreated the exact evidence state that existed at the
    # beginning of 11E-1F.  The closure must bind that complete state.
    require(freeze["input_evidence"] == {
        key: value for key, value in evidence.records.items()
        if key not in {EXECUTION_SOURCE, EXECUTION + "/freeze.json",
                       EXECUTION + "/public/prediction_commitment.json",
                       EXECUTION + "/public/scoring_report.json"}
    }, "11E-1F frozen-input evidence binding")

    expected_public = {"truth_commitment.json", "reference_suite.json",
                       "capture_manifest.json", "prediction_commitment.json",
                       "scoring_report.json"}
    case_ids = [case["case_id"] for case in truth["cases"]]
    for case_id in case_ids:
        expected_public.update({case_id + "_canonical.csv", case_id + "_replay.csv",
                                case_id + "_prediction.json"})
    require(set(freeze["artifacts"]) == expected_public and
            {path.name for path in public_dir.iterdir() if path.is_file()} == expected_public,
            "11E-1F public artifact inventory")

    verified_public = {}
    for name, record in freeze["artifacts"].items():
        path = verify_record(record, public_dir)
        require(path.name == name, "public artifact name/record mismatch")
        verified_public[name] = relative_record(root, path)

    checkpoint_path = verify_record(freeze["checkpoint"], private_dir)
    require(checkpoint_path.name == "checkpoint.json", "checkpoint path")
    checkpoint = read_json(checkpoint_path)
    require(checkpoint["binding"] == freeze["binding"] and
            checkpoint["active_capture"] is None and len(checkpoint["captures"]) == 24,
            "checkpoint closure")
    expected_batches = {str(case["batch_id"]) for case in capture_plan["cases"]}
    require(set(checkpoint["builds"]) == expected_batches, "simulator build coverage")
    for build in checkpoint["builds"].values():
        require(set(build) == {"binary", "record", "testbench", "log"}, "build record schema")
        for record in build.values():
            verify_record(record, private_dir)

    truth_commitment = read_json(public_dir / "truth_commitment.json")
    require(truth_commitment == {
        "truth_commitment": execution_module.TRUTH_SHA,
        "authorization_sha256": execution_module.AUTH_SHA,
        "case_ids": case_ids,
    }, "public truth commitment")
    require(read_json(public_dir / "reference_suite.json") == suite, "reference suite replay")

    capture_manifest = read_json(public_dir / "capture_manifest.json")
    require(capture_manifest["version"] == execution_module.VERSION and
            capture_manifest["suite_commitment"] == suite["suite_commitment"] and
            capture_manifest["truth_commitment"] == execution_module.TRUTH_SHA and
            [item["case_id"] for item in capture_manifest["cases"]] == case_ids,
            "capture manifest binding")

    prediction_manifest = read_json(prediction_commitment_path)
    require(prediction_manifest["version"] == execution_module.VERSION and
            prediction_manifest["truth_commitment"] == execution_module.TRUTH_SHA and
            prediction_manifest["suite_commitment"] == suite["suite_commitment"] and
            prediction_manifest["catalog_freeze_sha256"] == execution_module.CATALOG_FREEZE_SHA and
            [item["case_id"] for item in prediction_manifest["cases"]] == case_ids,
            "prediction commitment binding")

    report = read_json(scoring_report_path)
    require(report["status"] == "PASS" and
            report["predictions_committed_before_truth_read"] is True and
            report["replay_disagreements"] == 0 and report["no_match_cases"] == 0,
            "committed scoring closure")
    exact_fraction(report, "false_alarms_fault_free", 0, 2)
    exact_fraction(report, "observable_fault_detections", 8, 8)
    exact_fraction(report, "all_injected_fault_detections", 8, 10)
    exact_fraction(report, "unique_category_exact_site", 4, 4)
    exact_fraction(report, "all_injected_exact_site", 4, 10)
    exact_fraction(report, "observable_candidate_coverage", 8, 8)
    require(len(report["cases"]) == 12 and all(row["pass"] for row in report["cases"]),
            "per-case scoring closure")

    manifest_by_id = {item["case_id"]: item for item in capture_manifest["cases"]}
    prediction_by_id = {item["case_id"]: item for item in prediction_manifest["cases"]}
    score_by_id = {item["case_id"]: item for item in report["cases"]}
    require(set(manifest_by_id) == set(prediction_by_id) == set(score_by_id) == set(case_ids),
            "case coverage across commitments")

    conn = baseline.open_catalog(evidence.path(baseline.DEST + "/catalog.sqlite3"))
    try:
        for case in truth["cases"]:
            case_id = case["case_id"]
            canonical_record = checkpoint["captures"][case_id + ":canonical"]
            replay_record = checkpoint["captures"][case_id + ":replay"]
            canonical_path = verify_record(canonical_record["observation"], public_dir)
            replay_path = verify_record(replay_record["observation"], public_dir)
            canonical_log = verify_record(canonical_record["log"], private_dir)
            replay_log = verify_record(replay_record["log"], private_dir)
            require("BLINDED_CAPTURE_ROWS=64" in canonical_log.read_text() and
                    "BLINDED_CAPTURE_ROWS=64" in replay_log.read_text(), "capture log closure")
            canonical_profile = baseline.read_observations(
                io.StringIO(canonical_path.read_text()), suite)
            replay_profile = baseline.read_observations(io.StringIO(replay_path.read_text()), suite)
            require(canonical_profile == replay_profile and
                    canonical_record["normalized_sha256"] == replay_record["normalized_sha256"] ==
                    hashlib.sha256(canonical_profile).hexdigest(), "normalized capture replay")

            manifest_item = manifest_by_id[case_id]
            require(manifest_item["observation_path"] == str(canonical_path) and
                    manifest_item["observation_sha256"] == sha(canonical_path),
                    "capture manifest case binding")
            prediction_path = verify_record(prediction_by_id[case_id]["prediction"], public_dir)
            prediction = read_json(prediction_path)
            require(prediction["case_id"] == case_id and
                    prediction["observation"] == canonical_record["observation"] and
                    prediction["replay"] == replay_record["observation"],
                    "prediction/capture binding")

            expected_profile = baseline.golden_profile(suite)
            if case["fault_present"]:
                site_index = int(case["site_id"].split("-")[-1]) - 1
                fault_instance = 2 * site_index + case["stuck_value"]
                rows = conn.execute(
                    "SELECT p.response FROM profiles p JOIN faults f "
                    "ON p.profile_id=f.profile_id WHERE f.fault_instance=?",
                    (fault_instance,),
                ).fetchall()
                require(len(rows) == 1, "truth absent from frozen response catalog")
                expected_profile = bytes(rows[0][0])
            require(canonical_profile == expected_profile, "fresh capture/catalog mismatch")
            diagnosis = baseline.classify(conn, canonical_profile, suite)
            require(prediction["diagnosis"] == diagnosis, "committed diagnosis replay")
            scored = score_by_id[case_id]
            require(scored["category"] == case["selection_category"] and
                    scored["fault_present"] == case["fault_present"] and
                    scored["site_id"] == case["site_id"] and
                    scored["stuck_value"] == case["stuck_value"] and
                    scored["result"] == diagnosis["result"] and
                    scored["observable_anomaly"] == diagnosis["observable_anomaly"] and
                    all(scored["checks"].values()), "scoring/truth replay")
    finally:
        conn.close()

    require(Counter(case["selection_category"] for case in truth["cases"]) ==
            {"fault_free": 2, "unique": 4, "ambiguous": 4, "normal": 2},
            "private case stratification")
    evidence.recheck()
    require(sha(source_path) == EXECUTION_SOURCE_SHA and sha(Path(__file__)) == source_sha,
            "verification source changed during execution")

    input_records = {
        EXECUTION_SOURCE: relative_record(root, source_path),
        EXECUTION + "/freeze.json": relative_record(root, freeze_path),
        EXECUTION + "/private/checkpoint.json": relative_record(root, checkpoint_path),
        EXECUTION + "/public/prediction_commitment.json": relative_record(root, prediction_commitment_path),
        EXECUTION + "/public/scoring_report.json": relative_record(root, scoring_report_path),
    }
    public_commitment = hashlib.sha256(encode(verified_public)).hexdigest()

    policy = {
        "stage": "11E-1G",
        "version": VERSION,
        "status": "PASS",
        "disposition_status": "FROZEN",
        "behavior_diagnosis_pilot_status": "DEMONSTRATED_AND_FROZEN",
        "task": "Behavior-based anomaly detection and catalog-constrained physical-site localization",
        "diagnosis_input": "64 ordered HMAC response observations; unknown fault identity excluded",
        "fault_scope": "Single persistent SA0/SA1 faults in the frozen OpenTitan HMAC catalog",
        "reference_method": "Golden-response checker plus exact frozen response-catalog lookup",
        "demonstration": {
            "private_cases": 12,
            "fault_free_cases": 2,
            "injected_fault_cases": 10,
            "observable_injected_cases": 8,
            "normal_compatible_injected_cases": 2,
            "canonical_transactions": 768,
            "replay_transactions": 768,
            "false_alarms_fault_free": "0/2",
            "observable_fault_detections": "8/8",
            "all_injected_fault_detections": "8/10",
            "unique_category_exact_site": "4/4",
            "all_injected_exact_site": "4/10",
            "observable_candidate_coverage": "8/8",
            "normalized_capture_replay": "PASS",
            "prediction_replay": "PASS",
        },
        "authorized_claims": [
            "The frozen workflow detected all eight observable committed known-catalog cases.",
            "It exactly localized all four committed cases having a unique frozen-catalog signature.",
            "The true site was retained in the candidate set for all eight observable committed cases.",
            "Both committed fault-free cases produced no false alarm.",
        ],
        "prohibited_claims": [
            "Independent generalization accuracy",
            "Detection of every physical fault",
            "Universal exact-site localization",
            "Coverage of transient, bridging, multiple, analog, or out-of-catalog faults",
            "Applicability to an unseen circuit without rebuilding and independently testing its catalog",
        ],
        "normal_behavior_interpretation":
            "No observed anomaly under the fixed 64-vector suite; it does not prove fault absence.",
        "ambiguous_behavior_interpretation":
            "The returned candidate set is exact for the frozen catalog but is not a confidence ranking.",
        "new_model_training": False,
        "new_threshold_selection": False,
        "validation_or_holdout_access": False,
        "original_validation_target": "NOT_MET; PRESERVED",
        "holdout_execution": "BLOCKED / NOT AUTHORIZED",
        "project_final_freeze": "NOT DECLARED",
        "public_github_release": "NOT YET AUTHORIZED; RELEASE-READINESS REVIEW REQUIRED",
        "next_gate": "STAGE 11E-1H — FINAL PROJECT DISPOSITION AND REPRODUCIBLE-RELEASE READINESS FREEZE",
    }

    report_markdown = f"""# Stage 11E-1G — Behavior-Diagnosis Pilot Disposition

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
"""

    policy_path = root / POLICY
    report_path = root / REPORT
    audit_path = root / AUDIT
    write_once(policy_path, encode(policy))
    write_once(report_path, report_markdown.encode())
    outputs = {
        POLICY: relative_record(root, policy_path),
        REPORT: relative_record(root, report_path),
    }
    audit = {
        "stage": "11E-1G",
        "version": VERSION,
        "status": "PASS",
        "disposition_status": "FROZEN",
        "behavior_diagnosis_pilot_status": "DEMONSTRATED_AND_FROZEN",
        "source_sha256": source_sha,
        "verified_inputs": input_records,
        "verified_public_artifact_count": len(verified_public),
        "verified_public_artifact_commitment": public_commitment,
        "outputs": outputs,
        "private_cases": 12,
        "fresh_canonical_transactions": 768,
        "fresh_replay_transactions": 768,
        "blinded_diagnosis": "PASS",
        "prediction_commitment_before_truth": "PASS",
        "deterministic_replay": "PASS",
        "false_alarms_fault_free": "0/2",
        "observable_fault_detections": "8/8",
        "all_injected_fault_detections": "8/10",
        "unique_category_exact_site": "4/4",
        "all_injected_exact_site": "4/10",
        "observable_candidate_coverage": "8/8",
        "new_simulation_performed": False,
        "new_model_training_performed": False,
        "threshold_changed": False,
        "validation_holdout_accessed": False,
        "frozen_inputs_modified": False,
        "independent_generalization_claim": False,
        "original_validation_target": "NOT_MET; PRESERVED",
        "holdout_execution": "NOT AUTHORIZED",
        "project_final_freeze_declared": False,
        "public_github_release_authorized": False,
        "next_gate": policy["next_gate"],
    }
    write_once(audit_path, encode(audit))

    print("STAGE 11E-1G — BLINDED DEMONSTRATION DISPOSITION AND BEHAVIOR-DIAGNOSIS PILOT FREEZE")
    print("Status                         : PASS")
    print("Disposition status             : FROZEN")
    print("Behavior-diagnosis pilot       : DEMONSTRATED AND FROZEN")
    print("Diagnosis identity input       : PROHIBITED / NOT PROVIDED")
    print("Fault scope                    : SINGLE PERSISTENT SA0/SA1; FROZEN HMAC CATALOG")
    print("Private cases                  : 12 (2 fault-free + 10 injected)")
    print("Canonical / replay transactions: 768 / 768")
    print("Fault-free false alarms        : 0/2")
    print("Observable fault detections    : 8/8")
    print("All injected fault detections  : 8/10")
    print("Unique exact-site localization : 4/4")
    print("All injected exact-site        : 4/10")
    print("Observable candidate coverage  : 8/8")
    print("Deterministic replay           : PASS")
    print("Independent generalization     : NOT ESTABLISHED")
    print("New model training             : NO")
    print("VALIDATION / HOLDOUT access    : NO / NO")
    print("Original validation target     : NOT_MET; PRESERVED")
    print("Project final freeze           : NOT DECLARED")
    print("Public GitHub release          : NOT YET AUTHORIZED")
    print("Policy                         :", policy_path)
    print("Policy SHA                     :", sha(policy_path))
    print("Report                         :", report_path)
    print("Report SHA                     :", sha(report_path))
    print("Audit                          :", audit_path)
    print("Audit SHA                      :", sha(audit_path))
    print("Next gate                      :", policy["next_gate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        main(arguments.project_root)
    except KeyboardInterrupt:
        print("STOP: interrupted; no frozen input was modified.")
        raise SystemExit(130)
    except Exception as error:
        print("STOP:", error)
        print("Preserve all Stage 11E artifacts; do not rerun captures or change predictions.")
        raise SystemExit(1)
