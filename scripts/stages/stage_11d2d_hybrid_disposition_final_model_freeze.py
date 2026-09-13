#!/usr/bin/env python3
"""Freeze the hybrid as final development diagnostic model and preserve blind gates."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-HYBRID-DISPOSITION-FINAL-MODEL-FREEZER-v1"

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
RESULT_11D1 = ROOT / "results/hmac_fault_campaign_11d1"
RESULT_11D2 = ROOT / "results/hmac_fault_campaign_11d2"
CONFIG_ROOT = ROOT / "config/diagnostic_model"

BASELINE_AUDIT = RESULT_11C5 / "hmac_conventional_baseline_site_test_freeze_11c5h.json"
DEEP_AUDIT = RESULT_11C5 / "hmac_deep_diagnostic_site_test_comparison_freeze_11c5k.json"
GNN_EVALUATION_AUDIT = RESULT_11D1 / "hmac_gnn_site_test_comparator_freeze_11d1d.json"
GNN_DISPOSITION_AUDIT = RESULT_11D1 / "hmac_gnn_disposition_hybrid_readiness_freeze_11d1e.json"

HYBRID_ARCHITECTURE_AUDIT = RESULT_11D2 / "hmac_hybrid_architecture_training_contract_freeze_11d2a.json"
HYBRID_MODEL = RESULT_11D2 / "hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"
HYBRID_SELECTION_LOCK = RESULT_11D2 / "hybrid_training_11d2b/hmac_hybrid_selection_lock_11d2b.json"
HYBRID_TRAINING_MANIFEST = RESULT_11D2 / "hmac_hybrid_training_manifest_11d2b.json"
HYBRID_TRAINING_AUDIT = RESULT_11D2 / "hmac_hybrid_training_calibration_freeze_11d2b.json"

HYBRID_EVALUATOR = ROOT / "stage_11d2c_locked_hybrid_site_test.py"
HYBRID_COMPARISON = RESULT_11D2 / "hybrid_evaluation_11d2c/hmac_hybrid_vs_frozen_comparators_11d2c.json"
HYBRID_EVALUATION_LOCK = RESULT_11D2 / "hybrid_evaluation_11d2c/hmac_hybrid_site_test_evaluation_lock_11d2c.json"
HYBRID_EVALUATION_MANIFEST = RESULT_11D2 / "hmac_hybrid_site_test_comparator_manifest_11d2c.json"
HYBRID_EVALUATION_AUDIT = RESULT_11D2 / "hmac_hybrid_site_test_comparator_freeze_11d2c.json"

DISPOSITION_POLICY = CONFIG_ROOT / "hmac_hybrid_final_diagnostic_disposition_policy_11d2d.json"
FINAL_MODEL_LOCK = CONFIG_ROOT / "hmac_final_diagnostic_model_lock_11d2d.json"
VALIDATION_READINESS_POLICY = CONFIG_ROOT / "hmac_validation_readiness_policy_11d2d.json"
COMPARATOR_REGISTRY_CSV = RESULT_11D2 / "hmac_final_diagnostic_comparator_registry_11d2d.csv"
COMPARATOR_REGISTRY_JSON = RESULT_11D2 / "hmac_final_diagnostic_comparator_registry_11d2d.json"
AUDIT = RESULT_11D2 / "hmac_hybrid_disposition_final_diagnostic_model_freeze_11d2d.json"

EXPECTED_INPUTS = {
    BASELINE_AUDIT: "54d86a12e605ce8f2b920c55e87c6ad9a3e30b41df8420730adfff4be4fbee19",
    DEEP_AUDIT: "f5df0c081c75a66178a362ccbfb6cfdeb612634e6d2292891307a2826db8aab8",
    GNN_EVALUATION_AUDIT: "3555783dc613aec71a6f89c09b6c487b514aa0cfc68a6c7b42489018dbdb5a1b",
    GNN_DISPOSITION_AUDIT: "de89484ec1c244b141950ad259ad69163562ac70fd31e9659c8943c007e48339",
    HYBRID_ARCHITECTURE_AUDIT: "cbe3f362986a86506f9252253e06a2a8a2ec3541e9bc92655b75e05fcc082e65",
    HYBRID_MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    HYBRID_SELECTION_LOCK: "005cecbd94456d3d1aa6805b2dc898e895cc178564673b6b0aa269fcd64260c6",
    HYBRID_TRAINING_MANIFEST: "c5792cae623c0098165e691446c590235f50e1a5adc753dd133247b19adba41a",
    HYBRID_TRAINING_AUDIT: "39f459b3dbb0270810af8b8d39819ad2efecf982ebd929b3683d44cfa2570722",
    HYBRID_EVALUATOR: "2d0ce6457f280ac274ffbae0dbcc95001a6b8cc7af8483b9af80e8cc6f402225",
    HYBRID_COMPARISON: "a1713dfbaf51227952712d88f7f5aac95388dfef70b3187735d1ca0c7ff47595",
    HYBRID_EVALUATION_LOCK: "ee4d4b200027a756d9b8b699a4cc962127edabd4e67f1bac4c7706e6f4c2a0be",
    HYBRID_EVALUATION_MANIFEST: "ebe62b16ec9bc9a551ecefafc76d44cf589fdac4d53dcc4bf351cea49aa11bbc",
    HYBRID_EVALUATION_AUDIT: "6ae184f7eaef5070871b4ac54e280c01029ffd1d32168d5b44d8b76a5efa227e",
}

EXPECTED_HYBRID_CANDIDATE = "HYBRID_SGC3_MLP_128_64_32_A1E4_LR1E3"
EXPECTED_HYBRID_THRESHOLD = 0.4965
EXPECTED_HYBRID_MCC = 0.62982119
EXPECTED_HYBRID_MCC_CI = (0.61083264, 0.65040174)
EXPECTED_GNN_MCC = 0.33662641
EXPECTED_DEEP_MCC = 0.14832380
EXPECTED_BASELINE_MCC = 0.14208149
EXPECTED_HYBRID_GNN_DELTA = 0.29319478
EXPECTED_HYBRID_GNN_CI = (0.26987311, 0.31594325)
EXPECTED_HYBRID_DEEP_DELTA = 0.48149739
EXPECTED_HYBRID_DEEP_CI = (0.45002344, 0.51223916)
EXPECTED_HYBRID_BASELINE_DELTA = 0.48773970
EXPECTED_HYBRID_BASELINE_CI = (0.46198045, 0.51275537)
REQUIRED_HYBRID_GAIN = 0.02
EXPECTED_DEV_SITE_TEST_SITES = 3_426
EXPECTED_DEV_SITE_TEST_SAMPLES = 438_528


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def close(actual: object, expected: float, tolerance: float = 5e-8) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def interval_close(actual: object, expected: tuple[float, float], tolerance: float = 5e-8) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == 2
        and close(actual[0], expected[0], tolerance)
        and close(actual[1], expected[1], tolerance)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        stop(f"artifact outside project root: {path}")


def record(path: Path) -> dict:
    return {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not read JSON {relative(path)}: {error}")
    require(isinstance(value, dict), f"JSON root must be an object: {relative(path)}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {relative(path)}")
    require(path.stat().st_size > 0, f"empty frozen input: {relative(path)}")
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual}",
    )
    print(f"  {path.name:<80}: OK", flush=True)
    return record(path)


def resolve_artifact(item: dict, label: str) -> dict:
    require(isinstance(item, dict), f"{label} record")
    path_text = item.get("path")
    expected_sha = item.get("sha256")
    require(isinstance(path_text, str) and path_text, f"{label} path")
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, f"{label} SHA")
    path = (ROOT / path_text).resolve()
    relative(path)
    require(path.is_file() and path.stat().st_size > 0, f"{label} missing")
    require(sha256(path) == expected_sha, f"{label} SHA mismatch")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size")
    return record(path)


def verify_outputs_absent() -> None:
    for path in (
        DISPOSITION_POLICY,
        FINAL_MODEL_LOCK,
        VALIDATION_READINESS_POLICY,
        COMPARATOR_REGISTRY_CSV,
        COMPARATOR_REGISTRY_JSON,
        AUDIT,
    ):
        require(not path.exists(), f"refusing to overwrite Stage 11D-2D output: {relative(path)}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        DISPOSITION_POLICY,
        FINAL_MODEL_LOCK,
        VALIDATION_READINESS_POLICY,
        COMPARATOR_REGISTRY_CSV,
        COMPARATOR_REGISTRY_JSON,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def verify_governance_only_source() -> None:
    tree = ast.parse(SOURCE.read_text())
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name in {
            "load", "fit", "partial_fit", "predict", "predict_proba",
            "select_threshold", "train_candidate",
        }:
            forbidden.append((node.lineno, name))
    require(not forbidden, f"training, inference, or dataset-load calls present: {forbidden}")


def verify_semantics() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    baseline = load_json(BASELINE_AUDIT)
    deep = load_json(DEEP_AUDIT)
    gnn = load_json(GNN_EVALUATION_AUDIT)
    prior_disposition = load_json(GNN_DISPOSITION_AUDIT)
    training_lock = load_json(HYBRID_SELECTION_LOCK)
    training_audit = load_json(HYBRID_TRAINING_AUDIT)
    comparison = load_json(HYBRID_COMPARISON)
    evaluation_lock = load_json(HYBRID_EVALUATION_LOCK)
    evaluation_manifest = load_json(HYBRID_EVALUATION_MANIFEST)
    evaluation_audit = load_json(HYBRID_EVALUATION_AUDIT)

    require(baseline.get("status") == "PASS", "baseline audit status")
    require(baseline.get("baseline_status") == "FROZEN COMPARATOR", "baseline frozen comparator")
    require(close(baseline.get("mcc"), EXPECTED_BASELINE_MCC), "baseline MCC")
    require(deep.get("status") == "PASS", "deep audit status")
    require(deep.get("deep_model_status") == "FROZEN COMPARATOR", "deep frozen comparator")
    require(close(deep.get("deep_mcc"), EXPECTED_DEEP_MCC), "deep MCC")
    require(gnn.get("status") == "PASS", "GNN audit status")
    require(gnn.get("gnn_status") == "FROZEN COMPARATOR", "GNN frozen comparator")
    require(close(gnn.get("gnn_mcc"), EXPECTED_GNN_MCC), "GNN MCC")
    require(prior_disposition.get("status") == "PASS", "GNN disposition status")
    require(prior_disposition.get("winner") == "GNN", "prior frozen winner")
    require(prior_disposition.get("hybrid_architecture_contract") == "AUTHORIZED", "hybrid contract authorization")

    require(training_lock.get("status") == "PASS", "hybrid selection-lock status")
    require(training_lock.get("lock_status") == "FROZEN", "hybrid selection lock")
    require(training_lock.get("selected_candidate_id") == EXPECTED_HYBRID_CANDIDATE, "hybrid candidate")
    require(close(training_lock.get("selected_threshold"), EXPECTED_HYBRID_THRESHOLD), "hybrid threshold")
    require(training_lock.get("minimum_validity_status") == "PASS", "hybrid calibration validity")
    require(training_lock.get("absolute_performance_status_on_calibration") == "PASS", "hybrid calibration absolute target")
    require(training_lock.get("dev_site_test_opened_for_training_or_selection") is False, "hybrid training leakage")
    require(training_lock.get("retraining_after_site_test_authorized") is False, "post-test retraining gate")
    require(training_lock.get("selected_model") == record(HYBRID_MODEL), "hybrid selected-model record")
    replay = training_lock.get("deterministic_replay", {})
    require(replay.get("status") == "PASS", "hybrid replay status")
    for key in ("weights_exact", "probabilities_exact", "threshold_exact", "metrics_exact"):
        require(replay.get(key) is True, f"hybrid replay {key}")
    require(training_audit.get("status") == "PASS", "hybrid training audit status")
    require(training_audit.get("training_status") == "FROZEN", "hybrid training freeze")
    require(training_audit.get("model_selection_status") == "FROZEN", "hybrid model-selection freeze")
    require(training_audit.get("threshold_status") == "FROZEN", "hybrid threshold freeze")
    require(training_audit.get("dev_site_test_opened_during_training") is False, "hybrid training site-test state")

    require(comparison.get("status") == "PASS", "hybrid comparison status")
    require(comparison.get("comparison_status") == "FROZEN", "hybrid comparison freeze")
    require(comparison.get("partition") == "DEV_SITE_TEST", "hybrid comparison partition")
    metrics = comparison.get("metrics", {})
    require(close(metrics.get("hybrid", {}).get("mcc"), EXPECTED_HYBRID_MCC), "comparison hybrid MCC")
    require(close(metrics.get("gnn", {}).get("mcc"), EXPECTED_GNN_MCC), "comparison GNN MCC")
    require(close(metrics.get("deep", {}).get("mcc"), EXPECTED_DEEP_MCC), "comparison deep MCC")
    require(close(metrics.get("baseline", {}).get("mcc"), EXPECTED_BASELINE_MCC), "comparison baseline MCC")
    require(comparison.get("point_winner") == "HYBRID", "hybrid point winner")
    require(close(comparison.get("hybrid_minus_gnn_mcc"), EXPECTED_HYBRID_GNN_DELTA), "hybrid-GNN delta")
    require(interval_close(comparison.get("hybrid_minus_gnn_mcc_ci_95"), EXPECTED_HYBRID_GNN_CI), "hybrid-GNN CI")
    require(close(comparison.get("hybrid_minus_deep_mcc"), EXPECTED_HYBRID_DEEP_DELTA), "hybrid-deep delta")
    require(interval_close(comparison.get("hybrid_minus_deep_mcc_ci_95"), EXPECTED_HYBRID_DEEP_CI), "hybrid-deep CI")
    require(close(comparison.get("hybrid_minus_baseline_mcc"), EXPECTED_HYBRID_BASELINE_DELTA), "hybrid-baseline delta")
    require(interval_close(comparison.get("hybrid_minus_baseline_mcc_ci_95"), EXPECTED_HYBRID_BASELINE_CI), "hybrid-baseline CI")
    require(comparison.get("hybrid_improvement_over_gnn_statistically_supported") is True, "hybrid statistical gain over GNN")
    require(comparison.get("hybrid_improvement_over_deep_statistically_supported") is True, "hybrid statistical gain over deep")
    require(comparison.get("hybrid_improvement_over_baseline_statistically_supported") is True, "hybrid statistical gain over baseline")
    require(close(comparison.get("required_hybrid_mcc_improvement"), REQUIRED_HYBRID_GAIN), "hybrid required gain")
    require(comparison.get("required_hybrid_mcc_improvement_over_gnn_met") is True, "hybrid required gain result")
    require(comparison.get("hybrid_advancement_target_met") is True, "hybrid advancement target")

    require(evaluation_lock.get("status") == "PASS", "hybrid evaluation-lock status")
    require(evaluation_lock.get("lock_status") == "FROZEN", "hybrid evaluation lock")
    require(evaluation_lock.get("candidate_id") == EXPECTED_HYBRID_CANDIDATE, "evaluation-lock candidate")
    require(close(evaluation_lock.get("threshold"), EXPECTED_HYBRID_THRESHOLD), "evaluation-lock threshold")
    require(evaluation_lock.get("partition") == "DEV_SITE_TEST", "evaluation-lock partition")
    require(evaluation_lock.get("model_sha256") == EXPECTED_INPUTS[HYBRID_MODEL], "evaluation-lock model SHA")
    for key in (
        "model_fit_calls", "graph_scaler_fit_calls", "candidate_selection_calls",
        "threshold_selection_calls",
    ):
        require(evaluation_lock.get(key) == 0, f"evaluation-lock {key}")
    require(evaluation_lock.get("model_retrained") is False, "hybrid model retraining")
    require(evaluation_lock.get("threshold_changed") is False, "hybrid threshold changed")
    require(evaluation_lock.get("post_site_test_retraining_authorized") is False, "post-test retraining authorization")
    require(evaluation_lock.get("comparison") == comparison, "evaluation-lock comparison")

    require(evaluation_manifest.get("status") == "PASS", "hybrid evaluation manifest status")
    require(evaluation_manifest.get("partition") == "DEV_SITE_TEST", "hybrid evaluation manifest partition")
    require(evaluation_manifest.get("physical_sites") == EXPECTED_DEV_SITE_TEST_SITES, "site-test site count")
    require(evaluation_manifest.get("samples") == EXPECTED_DEV_SITE_TEST_SAMPLES, "site-test sample count")
    require(evaluation_manifest.get("model_fit_calls") == 0, "manifest model fitting")
    require(evaluation_manifest.get("threshold_selection_calls") == 0, "manifest threshold selection")
    require(evaluation_manifest.get("validation_vectors_exposed") == 0, "manifest VALIDATION exposure")
    require(evaluation_manifest.get("holdout_vectors_exposed") == 0, "manifest HOLDOUT exposure")
    require(evaluation_manifest.get("comparison") == comparison, "manifest comparison")
    outputs = evaluation_manifest.get("outputs", {})
    require(isinstance(outputs, dict) and outputs, "evaluation output records")
    resolved_outputs = {
        name: resolve_artifact(item, f"evaluation output {name}")
        for name, item in sorted(outputs.items())
    }
    require(evaluation_lock.get("outputs") == outputs, "lock/manifest output agreement")

    require(evaluation_audit.get("status") == "PASS", "hybrid evaluation audit status")
    require(evaluation_audit.get("evaluation_status") == "FROZEN", "hybrid evaluation freeze")
    require(evaluation_audit.get("comparison_status") == "FROZEN", "hybrid audit comparison freeze")
    require(evaluation_audit.get("point_winner") == "HYBRID", "hybrid audit winner")
    require(close(evaluation_audit.get("hybrid_mcc"), EXPECTED_HYBRID_MCC), "hybrid audit MCC")
    require(interval_close(evaluation_audit.get("hybrid_mcc_ci_95"), EXPECTED_HYBRID_MCC_CI), "hybrid MCC CI")
    require(evaluation_audit.get("hybrid_advancement_target_status") == "MET", "hybrid audit advancement")
    require(evaluation_audit.get("minimum_validity_status") == "PASS", "hybrid audit minimum validity")
    require(evaluation_audit.get("absolute_performance_status") == "PASS", "hybrid audit absolute target")
    require(evaluation_audit.get("dev_site_test_state") == "CONSUMED AND FROZEN FOR HYBRID", "hybrid site-test state")
    require(evaluation_audit.get("validation_vectors_exposed") == 0, "audit VALIDATION exposure")
    require(evaluation_audit.get("holdout_vectors_exposed") == 0, "audit HOLDOUT exposure")
    require(evaluation_audit.get("post_site_test_retraining_authorized") is False, "audit retraining gate")
    require(evaluation_audit.get("evaluation_lock") == record(HYBRID_EVALUATION_LOCK), "audit evaluation-lock record")
    require(evaluation_audit.get("manifest") == record(HYBRID_EVALUATION_MANIFEST), "audit manifest record")
    require(evaluation_audit.get("next_gate", "").startswith("STAGE 11D-2D"), "final disposition gate")

    return (
        baseline, deep, gnn, training_lock, comparison, evaluation_audit,
        {"evaluation_outputs": resolved_outputs},
    )


def comparator_row(
    comparator_id: str,
    family: str,
    role: str,
    candidate: str,
    threshold: object,
    metrics: dict,
    status: str,
    winner: bool,
) -> dict:
    return {
        "comparator_id": comparator_id,
        "family": family,
        "role": role,
        "candidate": candidate,
        "threshold": format(float(threshold), ".17g"),
        "mcc": format(float(metrics["mcc"]), ".17g"),
        "balanced_accuracy": format(float(metrics["balanced_accuracy"]), ".17g"),
        "precision": format(float(metrics["precision"]), ".17g"),
        "recall": format(float(metrics["recall"]), ".17g"),
        "f1_score": format(float(metrics["f1_score"]), ".17g"),
        "pr_auc": format(float(metrics["pr_auc"]), ".17g"),
        "roc_auc": format(float(metrics["roc_auc"]), ".17g"),
        "brier_score": format(float(metrics["brier_score"]), ".17g"),
        "status": status,
        "selection_winner": "YES" if winner else "NO",
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-2D script in project root")
    verify_outputs_absent()
    verify_governance_only_source()

    print("STAGE 11D-2D — HYBRID DISPOSITION AND FINAL DIAGNOSTIC MODEL")
    print("FROZEN INPUT VERIFICATION", flush=True)
    inputs = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    baseline, deep, gnn, training_lock, comparison, evaluation_audit, handoff = verify_semantics()
    print("  Prior-stage semantic contracts                                               : PASS")
    print("  Winner and paired-confidence decision                                       : PASS")
    print("  Dataset, model training, or inference performed                             : NO")
    print("  VALIDATION or HOLDOUT data opened                                           : NO")

    metrics = comparison["metrics"]
    hybrid_metrics = metrics["hybrid"]
    gnn_metrics = metrics["gnn"]
    deep_metrics = metrics["deep"]
    baseline_metrics = metrics["baseline"]
    created_at = datetime.now(timezone.utc).isoformat()

    comparator_rows = [
        comparator_row(
            "HYBRID_FUSION_MLP_11D2C",
            "GRAPH_AUGMENTED_FEATURE_FUSION_MLP",
            "FINAL_DEVELOPMENT_DIAGNOSTIC_MODEL",
            EXPECTED_HYBRID_CANDIDATE,
            EXPECTED_HYBRID_THRESHOLD,
            hybrid_metrics,
            "FROZEN_WINNING_MODEL",
            True,
        ),
        comparator_row(
            "DIR_SGC_11D1D",
            "DIRECTED_BIDIRECTIONAL_SGC",
            "PRIMARY_GRAPH_COMPARATOR",
            gnn["gnn_candidate"],
            gnn["gnn_threshold"],
            gnn_metrics,
            "FROZEN_COMPARATOR",
            False,
        ),
        comparator_row(
            "DEEP_MLP_11C5K",
            "FEEDFORWARD_MULTILAYER_PERCEPTRON",
            "SECONDARY_NEURAL_COMPARATOR",
            deep["deep_candidate"],
            deep["deep_threshold"],
            deep_metrics,
            "FROZEN_COMPARATOR",
            False,
        ),
        comparator_row(
            "CONVENTIONAL_LOGREG_11C5H",
            "INCREMENTAL_LOGISTIC_REGRESSION",
            "PRIMARY_SIMPLE_COMPARATOR",
            baseline["candidate_id"],
            baseline["threshold"],
            baseline_metrics,
            "FROZEN_COMPARATOR",
            False,
        ),
    ]
    atomic_csv(COMPARATOR_REGISTRY_CSV, list(comparator_rows[0]), comparator_rows)
    atomic_json(COMPARATOR_REGISTRY_JSON, {
        "stage": "11D-2D",
        "status": "PASS",
        "registry_status": "FROZEN",
        "evaluation_partition": "DEV_SITE_TEST",
        "primary_metric": "MCC",
        "winner_declared": True,
        "winner": "HYBRID_FUSION_MLP_11D2C",
        "winner_reason": (
            "Hybrid exceeds the frozen GNN by at least +0.02 MCC, the paired "
            "physical-site 95% MCC-delta confidence interval is entirely above "
            "zero, and the absolute locked-site-test performance target passes."
        ),
        "comparators": comparator_rows,
    })

    atomic_json(DISPOSITION_POLICY, {
        "stage": "11D-2D",
        "title": "HYBRID FINAL DIAGNOSTIC DISPOSITION POLICY",
        "status": "PASS",
        "policy_status": "FROZEN",
        "created_at_utc": created_at,
        "primary_task": "PRE-SIMULATION BINARY DETECTABILITY",
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY",
        "primary_metric": "MCC",
        "winner_declared": True,
        "winning_model_id": "HYBRID_FUSION_MLP_11D2C",
        "winning_candidate": EXPECTED_HYBRID_CANDIDATE,
        "winning_threshold": EXPECTED_HYBRID_THRESHOLD,
        "hybrid_mcc": hybrid_metrics["mcc"],
        "gnn_mcc": gnn_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "baseline_mcc": baseline_metrics["mcc"],
        "hybrid_minus_gnn_mcc": comparison["hybrid_minus_gnn_mcc"],
        "hybrid_minus_gnn_mcc_ci_95": comparison["hybrid_minus_gnn_mcc_ci_95"],
        "required_hybrid_mcc_gain": REQUIRED_HYBRID_GAIN,
        "statistical_gain_over_gnn": True,
        "required_hybrid_gain_met": True,
        "minimum_validity_status": "PASS",
        "absolute_performance_status": "PASS",
        "development_project_target_status": "MET",
        "governance_disposition": "FREEZE_HYBRID_AS_FINAL_DEVELOPMENT_DIAGNOSTIC_MODEL",
        "retraining_authorized": False,
        "threshold_change_authorized": False,
        "architecture_change_authorized": False,
        "dev_site_test_use_for_design": "PROHIBITED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "comparator_registry_csv": record(COMPARATOR_REGISTRY_CSV),
        "comparator_registry_json": record(COMPARATOR_REGISTRY_JSON),
    })

    atomic_json(FINAL_MODEL_LOCK, {
        "stage": "11D-2D",
        "title": "FINAL DEVELOPMENT DIAGNOSTIC MODEL LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": created_at,
        "model_id": "HYBRID_FUSION_MLP_11D2C",
        "model_family": "GRAPH_AUGMENTED_FEATURE_FUSION_MLP",
        "candidate_id": EXPECTED_HYBRID_CANDIDATE,
        "threshold": EXPECTED_HYBRID_THRESHOLD,
        "model_artifact": record(HYBRID_MODEL),
        "training_selection_lock": record(HYBRID_SELECTION_LOCK),
        "training_manifest": record(HYBRID_TRAINING_MANIFEST),
        "training_audit": record(HYBRID_TRAINING_AUDIT),
        "evaluation_comparison": record(HYBRID_COMPARISON),
        "evaluation_lock": record(HYBRID_EVALUATION_LOCK),
        "evaluation_manifest": record(HYBRID_EVALUATION_MANIFEST),
        "evaluation_audit": record(HYBRID_EVALUATION_AUDIT),
        "evaluation_partition": "DEV_SITE_TEST",
        "evaluation_metrics": hybrid_metrics,
        "model_retraining_allowed": False,
        "threshold_reselection_allowed": False,
        "feature_or_graph_refitting_allowed": False,
        "validation_use_for_tuning_allowed": False,
        "holdout_use_for_tuning_allowed": False,
    })

    atomic_json(VALIDATION_READINESS_POLICY, {
        "stage": "11D-2D",
        "title": "FROZEN MODEL VALIDATION READINESS POLICY",
        "status": "PASS",
        "readiness_status": "READY_FOR_VALIDATION_CAMPAIGN_CONTRACT_ONLY",
        "created_at_utc": created_at,
        "frozen_final_development_model": "HYBRID_FUSION_MLP_11D2C",
        "validation_campaign_contract_authorized": True,
        "validation_campaign_execution_authorized": False,
        "holdout_campaign_contract_authorized": False,
        "holdout_campaign_execution_authorized": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "validation_purpose": "ONE_LOCKED_GENERALIZATION_EVALUATION_WITHOUT_TUNING",
        "required_validation_controls": [
            "USE_FROZEN_MODEL_BYTES",
            "USE_FROZEN_THRESHOLD",
            "NO_PARAMETER_OR_FEATURE_REFITTING",
            "NO_CANDIDATE_SELECTION",
            "NO_THRESHOLD_SELECTION",
            "PHYSICAL_SITE_ACCOUNTABILITY",
            "PRESERVE_HOLDOUT_TEST_BLINDNESS",
        ],
        "prohibited_actions": [
            "RETRAIN_ANY_FROZEN_COMPARATOR",
            "CHANGE_ANY_FROZEN_THRESHOLD",
            "USE_VALIDATION_TO_MODIFY_THE_FINAL_MODEL",
            "OPEN_HOLDOUT_TEST",
            "MODIFY_FROZEN_RTL_OR_GOLDEN_NETLIST",
        ],
        "next_gate": "STAGE 11D-3A — VALIDATION CAMPAIGN ARCHITECTURE AND EXECUTION-CONTRACT FREEZE",
    })

    outputs = {
        "disposition_policy": record(DISPOSITION_POLICY),
        "final_model_lock": record(FINAL_MODEL_LOCK),
        "validation_readiness_policy": record(VALIDATION_READINESS_POLICY),
        "comparator_registry_csv": record(COMPARATOR_REGISTRY_CSV),
        "comparator_registry_json": record(COMPARATOR_REGISTRY_JSON),
    }
    atomic_json(AUDIT, {
        "stage": "11D-2D",
        "title": "HYBRID DISPOSITION AND FINAL DIAGNOSTIC MODEL FREEZE",
        "status": "PASS",
        "disposition_status": "FROZEN",
        "final_model_status": "FROZEN",
        "comparator_registry_status": "FROZEN",
        "validation_readiness_status": "FROZEN",
        "created_at_utc": created_at,
        "winner_declared": True,
        "winner": "HYBRID",
        "final_development_model": "HYBRID_FUSION_MLP_11D2C",
        "hybrid_status": "FROZEN FINAL DEVELOPMENT MODEL",
        "gnn_status": "FROZEN PRIMARY GRAPH COMPARATOR",
        "deep_status": "FROZEN SECONDARY NEURAL COMPARATOR",
        "baseline_status": "FROZEN PRIMARY SIMPLE COMPARATOR",
        "hybrid_candidate": EXPECTED_HYBRID_CANDIDATE,
        "hybrid_threshold": EXPECTED_HYBRID_THRESHOLD,
        "hybrid_mcc": hybrid_metrics["mcc"],
        "hybrid_mcc_ci_95": comparison["metrics"]["hybrid"].get(
            "mcc_ci_95", evaluation_audit["hybrid_mcc_ci_95"]
        ),
        "gnn_mcc": gnn_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "baseline_mcc": baseline_metrics["mcc"],
        "hybrid_minus_gnn_mcc": comparison["hybrid_minus_gnn_mcc"],
        "hybrid_minus_gnn_mcc_ci_95": comparison["hybrid_minus_gnn_mcc_ci_95"],
        "hybrid_minus_deep_mcc": comparison["hybrid_minus_deep_mcc"],
        "hybrid_minus_deep_mcc_ci_95": comparison["hybrid_minus_deep_mcc_ci_95"],
        "hybrid_minus_baseline_mcc": comparison["hybrid_minus_baseline_mcc"],
        "hybrid_minus_baseline_mcc_ci_95": comparison["hybrid_minus_baseline_mcc_ci_95"],
        "statistical_gain_over_gnn": "SUPPORTED",
        "statistical_gain_over_deep": "SUPPORTED",
        "statistical_gain_over_baseline": "SUPPORTED",
        "required_hybrid_mcc_improvement": "MET",
        "minimum_validity_status": "PASS",
        "absolute_performance_status": "PASS",
        "development_project_target_status": "MET",
        "dataset_or_model_execution_performed": False,
        "dev_site_test_state": "CONSUMED_AND_FROZEN_FOR_ALL_DEVELOPMENT_MODELS",
        "model_retraining_authorized": False,
        "threshold_change_authorized": False,
        "validation_campaign_contract": "AUTHORIZED",
        "validation_campaign_execution": "NOT_YET_AUTHORIZED",
        "holdout_campaign": "NOT_YET_AUTHORIZED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "post_simulation_features": 0,
        "input_evidence": inputs,
        "handoff": handoff,
        "outputs": outputs,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "hybrid_model_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
        "next_gate": "STAGE 11D-3A — VALIDATION CAMPAIGN ARCHITECTURE AND EXECUTION-CONTRACT FREEZE",
    })
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-2D — HYBRID DISPOSITION AND FINAL DIAGNOSTIC MODEL FREEZE")
    print("Status                         : PASS")
    print("Disposition status             : FROZEN")
    print("Final model status             : FROZEN")
    print("Winner declared                : HYBRID")
    print("Final development model        : HYBRID FUSION MLP")
    print("Hybrid MCC                     :", f"{hybrid_metrics['mcc']:.8f}")
    print("Hybrid MCC 95% site CI         :", f"[{evaluation_audit['hybrid_mcc_ci_95'][0]:.8f}, {evaluation_audit['hybrid_mcc_ci_95'][1]:.8f}]")
    print("GNN MCC                        :", f"{gnn_metrics['mcc']:.8f}")
    print("Deep MLP MCC                   :", f"{deep_metrics['mcc']:.8f}")
    print("Conventional baseline MCC      :", f"{baseline_metrics['mcc']:.8f}")
    print("MCC delta (hybrid-GNN)         :", f"{comparison['hybrid_minus_gnn_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{comparison['hybrid_minus_gnn_mcc_ci_95'][0]:.8f}, {comparison['hybrid_minus_gnn_mcc_ci_95'][1]:.8f}]")
    print("Statistical gain vs GNN        : SUPPORTED")
    print("Required +0.02 MCC gain        : MET")
    print("Minimum validity               : PASS")
    print("Absolute performance target    : PASS")
    print("Development project target     : MET")
    print("Model retraining               : PROHIBITED")
    print("Threshold changes              : PROHIBITED")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Validation contract            : AUTHORIZED")
    print("Validation execution           : NOT YET AUTHORIZED")
    print("Holdout campaign               : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Hybrid model modified          : NO")
    print("Disposition policy             :", DISPOSITION_POLICY)
    print("Disposition policy SHA         :", outputs["disposition_policy"]["sha256"])
    print("Final model lock               :", FINAL_MODEL_LOCK)
    print("Final model lock SHA           :", outputs["final_model_lock"]["sha256"])
    print("Validation readiness policy    :", VALIDATION_READINESS_POLICY)
    print("Validation readiness SHA       :", outputs["validation_readiness_policy"]["sha256"])
    print("Comparator registry            :", COMPARATOR_REGISTRY_CSV)
    print("Comparator registry SHA        :", outputs["comparator_registry_csv"]["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-3A — VALIDATION CAMPAIGN ARCHITECTURE AND EXECUTION-CONTRACT FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
