#!/usr/bin/env python3
"""Freeze the winning GNN disposition and authorize hybrid contract design."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-GNN-DISPOSITION-HYBRID-READINESS-v1"

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
RESULT_11D1 = ROOT / "results/hmac_fault_campaign_11d1"
CONFIG_ROOT = ROOT / "config/diagnostic_model"

BASELINE_DISPOSITION = CONFIG_ROOT / "hmac_diagnostic_baseline_disposition_policy_11c5l.json"
GNN_READINESS_11C5L = CONFIG_ROOT / "hmac_gnn_readiness_policy_11c5l.json"
COMPARATOR_REGISTRY_11C5L = RESULT_11C5 / "hmac_frozen_diagnostic_comparator_registry_11c5l.csv"
BASELINE_DISPOSITION_AUDIT = (
    RESULT_11C5 / "hmac_diagnostic_baseline_disposition_gnn_readiness_freeze_11c5l.json"
)

BASELINE_AUDIT = RESULT_11C5 / "hmac_conventional_baseline_site_test_freeze_11c5h.json"
DEEP_AUDIT = RESULT_11C5 / "hmac_deep_diagnostic_site_test_comparison_freeze_11c5k.json"

GRAPH_AUDIT = (
    RESULT_11D1 / "hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"
)
GNN_CONTRACT_AUDIT = (
    RESULT_11D1 / "hmac_gnn_architecture_training_contract_freeze_11d1b.json"
)
GNN_MODEL = (
    RESULT_11D1 / "gnn_training_11d1c/hmac_selected_gnn_model_11d1c.joblib"
)
GNN_SELECTION_LOCK = (
    RESULT_11D1 / "gnn_training_11d1c/hmac_gnn_selection_lock_11d1c.json"
)
GNN_TRAINING_AUDIT = RESULT_11D1 / "hmac_gnn_training_calibration_freeze_11d1c.json"

GNN_EVALUATOR = ROOT / "stage_11d1d_locked_gnn_site_test.py"
GNN_PREDICTIONS = (
    RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_site_test_predictions_11d1d.npz"
)
GNN_COMPARISON = (
    RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_vs_frozen_comparators_11d1d.json"
)
GNN_EVALUATION_LOCK = (
    RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_site_test_evaluation_lock_11d1d.json"
)
GNN_EVALUATION_MANIFEST = RESULT_11D1 / "hmac_gnn_site_test_comparator_manifest_11d1d.json"
GNN_EVALUATION_AUDIT = RESULT_11D1 / "hmac_gnn_site_test_comparator_freeze_11d1d.json"

GNN_DISPOSITION_POLICY = CONFIG_ROOT / "hmac_gnn_disposition_policy_11d1e.json"
HYBRID_READINESS_POLICY = CONFIG_ROOT / "hmac_hybrid_readiness_policy_11d1e.json"
COMPARATOR_REGISTRY_CSV = RESULT_11D1 / "hmac_frozen_diagnostic_comparator_registry_11d1e.csv"
COMPARATOR_REGISTRY_JSON = RESULT_11D1 / "hmac_frozen_diagnostic_comparator_registry_11d1e.json"
AUDIT = RESULT_11D1 / "hmac_gnn_disposition_hybrid_readiness_freeze_11d1e.json"

EXPECTED_INPUTS = {
    BASELINE_DISPOSITION: "26553f2c402c2723c1a702fb23dc7ad8b897c0301a6c67d36a02f059ef171747",
    GNN_READINESS_11C5L: "702eb89f214aa2e70b1c95850d622ea15678ec71667639b0e7edc5f6cf3dd9fe",
    COMPARATOR_REGISTRY_11C5L: "41dca9b233950636fbd93c0011e94fbc21fc237baf72faf242de410c8386bc2c",
    BASELINE_DISPOSITION_AUDIT: "9fca71d7ab347e0c20c09ad2da5d42c0d4bbb7c47a9858e38680eb98dfcd49d1",
    BASELINE_AUDIT: "54d86a12e605ce8f2b920c55e87c6ad9a3e30b41df8420730adfff4be4fbee19",
    DEEP_AUDIT: "f5df0c081c75a66178a362ccbfb6cfdeb612634e6d2292891307a2826db8aab8",
    GRAPH_AUDIT: "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
    GNN_CONTRACT_AUDIT: "b5a5b8b6b8d7f6f6926a17440ff6f7859e422e9182c948695afa47bf0f36ef01",
    GNN_MODEL: "c345ab00a45bceb7b7c375d9c5483daa067a01bb8149b30c2325b056a3724e91",
    GNN_SELECTION_LOCK: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
    GNN_TRAINING_AUDIT: "afe2c204fe2208f3a5805bf9e7bf7c7000bdef1a81dac8d651876d2cb633c079",
    GNN_EVALUATOR: "f5126b50d8f089e5fb1cea84de5e3375d7c648c0878dd62fbef29f0e7a5eb599",
    GNN_PREDICTIONS: "3849103f43dede6043c6b1bfc5265f7a427aec16917a95d0d156955668f830a6",
    GNN_COMPARISON: "5bfce5f1cb49b29dc3a17343dbd2ba5f5777e3ec11d9679efdd24b9bfbdaf0e1",
    GNN_EVALUATION_LOCK: "e09b0c093566e4f22a7f7e9ae92fd73325d4092b25baeca2357dd85618f78398",
    GNN_EVALUATION_MANIFEST: "a97fe3ca5c9a000f8e514ad88d263ea5a686df5dabdbf17c637676db8083789a",
    GNN_EVALUATION_AUDIT: "3555783dc613aec71a6f89c09b6c487b514aa0cfc68a6c7b42489018dbdb5a1b",
}

EXPECTED_GNN_CANDIDATE = "DIR_SGC_K3_L2_A1E5"
EXPECTED_GNN_THRESHOLD = 0.86327695216798483
EXPECTED_GNN_MCC = 0.33662641
EXPECTED_DEEP_MCC = 0.14832380
EXPECTED_BASELINE_MCC = 0.14208149
EXPECTED_GNN_BASELINE_DELTA = 0.19454492
EXPECTED_GNN_BASELINE_CI = (0.17047836, 0.21671921)
EXPECTED_GNN_DEEP_DELTA = 0.18830261
EXPECTED_GNN_DEEP_CI = (0.15953297, 0.21494971)
EXPECTED_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


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
        stop(f"could not read JSON {path}: {error}")
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
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
    require(path.is_file(), f"missing frozen input: {path}")
    actual_sha = sha256(path)
    require(
        actual_sha == expected_sha,
        f"SHA mismatch for {path}: expected {expected_sha}, actual {actual_sha}",
    )
    print(f"  {path.name:<76}: OK")
    return record(path)


def close(actual: object, expected: float, tolerance: float = 5e-8) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def interval_close(actual: object, expected: tuple[float, float]) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == 2
        and close(actual[0], expected[0])
        and close(actual[1], expected[1])
    )


def verify_semantics() -> tuple[dict, dict, dict, dict, dict]:
    baseline = load_json(BASELINE_AUDIT)
    deep = load_json(DEEP_AUDIT)
    graph = load_json(GRAPH_AUDIT)
    comparison = load_json(GNN_COMPARISON)
    gnn = load_json(GNN_EVALUATION_AUDIT)

    require(baseline.get("status") == "PASS", "baseline status")
    require(baseline.get("evaluation_status") == "FROZEN", "baseline evaluation freeze")
    require(close(baseline.get("mcc"), EXPECTED_BASELINE_MCC), "baseline MCC")
    require(deep.get("status") == "PASS", "deep comparator status")
    require(deep.get("evaluation_status") == "FROZEN", "deep comparator freeze")
    require(close(deep.get("deep_mcc"), EXPECTED_DEEP_MCC), "deep MCC")

    require(graph.get("status") == "PASS", "graph dataset status")
    require(graph.get("graph_dataset_status") == "FROZEN", "graph dataset freeze")
    require(graph.get("topology_integrity_status") == "PASS", "topology integrity")
    require(graph.get("legal_nodes") == EXPECTED_SITES, "legal graph node count")
    require(graph.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "fault instance count")

    require(comparison.get("status") == "PASS", "GNN comparison status")
    require(comparison.get("comparison_status") == "FROZEN", "GNN comparison freeze")
    require(comparison.get("primary_metric") == "MCC", "GNN primary metric")
    metrics = comparison.get("metrics", {})
    require(close(metrics.get("gnn", {}).get("mcc"), EXPECTED_GNN_MCC), "comparison GNN MCC")
    require(close(metrics.get("deep", {}).get("mcc"), EXPECTED_DEEP_MCC), "comparison deep MCC")
    require(close(metrics.get("baseline", {}).get("mcc"), EXPECTED_BASELINE_MCC), "comparison baseline MCC")
    require(comparison.get("point_winner") == "GNN", "GNN point winner")
    require(close(comparison.get("gnn_minus_baseline_mcc"), EXPECTED_GNN_BASELINE_DELTA), "GNN-baseline delta")
    require(interval_close(comparison.get("gnn_minus_baseline_mcc_ci_95"), EXPECTED_GNN_BASELINE_CI), "GNN-baseline CI")
    require(close(comparison.get("gnn_minus_deep_mcc"), EXPECTED_GNN_DEEP_DELTA), "GNN-deep delta")
    require(interval_close(comparison.get("gnn_minus_deep_mcc_ci_95"), EXPECTED_GNN_DEEP_CI), "GNN-deep CI")
    require(comparison.get("gnn_improvement_over_baseline_statistically_supported") is True, "GNN-baseline statistical gain")
    require(comparison.get("gnn_improvement_over_deep_statistically_supported") is True, "GNN-deep statistical gain")
    require(comparison.get("required_mcc_improvement_over_baseline_met") is True, "required MCC improvement")
    require(comparison.get("project_target_met") is True, "project target")

    require(gnn.get("status") == "PASS", "GNN evaluation status")
    require(gnn.get("evaluation_status") == "FROZEN", "GNN evaluation freeze")
    require(gnn.get("comparison_status") == "FROZEN", "GNN comparator freeze")
    require(gnn.get("gnn_candidate") == EXPECTED_GNN_CANDIDATE, "GNN candidate")
    require(close(gnn.get("gnn_threshold"), EXPECTED_GNN_THRESHOLD, 1e-14), "GNN threshold")
    require(close(gnn.get("gnn_mcc"), EXPECTED_GNN_MCC), "GNN audit MCC")
    require(gnn.get("point_winner") == "GNN", "GNN audit winner")
    require(gnn.get("project_target_status") == "MET", "GNN project target status")
    require(gnn.get("absolute_performance_status") == "NOT_MET", "absolute performance status")
    require(gnn.get("hybrid_training_authorized") is False, "prior hybrid training gate")
    require(gnn.get("validation_vectors_exposed") == 0, "validation exposure")
    require(gnn.get("holdout_vectors_exposed") == 0, "holdout exposure")
    return baseline, deep, graph, comparison, gnn


def verify_outputs_absent() -> None:
    for path in (
        GNN_DISPOSITION_POLICY,
        HYBRID_READINESS_POLICY,
        COMPARATOR_REGISTRY_CSV,
        COMPARATOR_REGISTRY_JSON,
        AUDIT,
    ):
        require(not path.exists(), f"refusing to overwrite Stage 11D-1E output: {path}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        GNN_DISPOSITION_POLICY,
        HYBRID_READINESS_POLICY,
        COMPARATOR_REGISTRY_CSV,
        COMPARATOR_REGISTRY_JSON,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-1E script in project root")
    verify_outputs_absent()

    print("STAGE 11D-1E — GNN DISPOSITION AND HYBRID READINESS")
    print("FROZEN INPUT VERIFICATION", flush=True)
    inputs = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    baseline, deep, graph, comparison, gnn = verify_semantics()
    print("  Prior-stage semantic contracts                                             : PASS")
    print("  Winner and confidence-interval decision                                   : PASS")
    print("  DEV_SITE_TEST data opened in this stage                                   : NO")

    created_at = datetime.now(timezone.utc).isoformat()
    metrics = comparison["metrics"]
    gnn_metrics = metrics["gnn"]
    deep_metrics = metrics["deep"]
    baseline_metrics = metrics["baseline"]

    comparator_rows = [
        {
            "comparator_id": "DIR_SGC_11D1D",
            "family": "DIRECTED_BIDIRECTIONAL_SGC",
            "role": "PRIMARY_GRAPH_COMPARATOR",
            "candidate": EXPECTED_GNN_CANDIDATE,
            "threshold": format(EXPECTED_GNN_THRESHOLD, ".17g"),
            "mcc": format(float(gnn_metrics["mcc"]), ".17g"),
            "balanced_accuracy": format(float(gnn_metrics["balanced_accuracy"]), ".17g"),
            "precision": format(float(gnn_metrics["precision"]), ".17g"),
            "recall": format(float(gnn_metrics["recall"]), ".17g"),
            "f1_score": format(float(gnn_metrics["f1_score"]), ".17g"),
            "pr_auc": format(float(gnn_metrics["pr_auc"]), ".17g"),
            "roc_auc": format(float(gnn_metrics["roc_auc"]), ".17g"),
            "status": "FROZEN_WINNING_COMPARATOR",
            "selection_winner": "YES",
        },
        {
            "comparator_id": "DEEP_MLP_11C5K",
            "family": "FEEDFORWARD_MULTILAYER_PERCEPTRON",
            "role": "SECONDARY_NEURAL_COMPARATOR",
            "candidate": deep["deep_candidate"],
            "threshold": format(float(deep["deep_threshold"]), ".17g"),
            "mcc": format(float(deep_metrics["mcc"]), ".17g"),
            "balanced_accuracy": format(float(deep_metrics["balanced_accuracy"]), ".17g"),
            "precision": format(float(deep_metrics["precision"]), ".17g"),
            "recall": format(float(deep_metrics["recall"]), ".17g"),
            "f1_score": format(float(deep_metrics["f1_score"]), ".17g"),
            "pr_auc": format(float(deep_metrics["pr_auc"]), ".17g"),
            "roc_auc": format(float(deep_metrics["roc_auc"]), ".17g"),
            "status": "FROZEN_COMPARATOR",
            "selection_winner": "NO",
        },
        {
            "comparator_id": "CONVENTIONAL_LOGREG_11C5H",
            "family": "INCREMENTAL_LOGISTIC_REGRESSION",
            "role": "PRIMARY_SIMPLE_COMPARATOR",
            "candidate": baseline["candidate_id"],
            "threshold": format(float(baseline["threshold"]), ".17g"),
            "mcc": format(float(baseline_metrics["mcc"]), ".17g"),
            "balanced_accuracy": format(float(baseline_metrics["balanced_accuracy"]), ".17g"),
            "precision": format(float(baseline_metrics["precision"]), ".17g"),
            "recall": format(float(baseline_metrics["recall"]), ".17g"),
            "f1_score": format(float(baseline_metrics["f1_score"]), ".17g"),
            "pr_auc": format(float(baseline_metrics["pr_auc"]), ".17g"),
            "roc_auc": format(float(baseline_metrics["roc_auc"]), ".17g"),
            "status": "FROZEN_COMPARATOR",
            "selection_winner": "NO",
        },
    ]
    atomic_csv(COMPARATOR_REGISTRY_CSV, list(comparator_rows[0]), comparator_rows)
    atomic_json(COMPARATOR_REGISTRY_JSON, {
        "stage": "11D-1E",
        "status": "PASS",
        "registry_status": "FROZEN",
        "primary_metric": "MCC",
        "evaluation_partition": "DEV_SITE_TEST",
        "winner_declared": True,
        "winner": "DIR_SGC_11D1D",
        "winner_reason": (
            "GNN exceeds both frozen comparators; both paired physical-site "
            "95% MCC-delta confidence intervals are entirely above zero."
        ),
        "comparators": comparator_rows,
    })

    atomic_json(GNN_DISPOSITION_POLICY, {
        "stage": "11D-1E",
        "title": "GNN DISPOSITION POLICY",
        "status": "PASS",
        "policy_status": "FROZEN",
        "created_at_utc": created_at,
        "primary_task": "PRE-SIMULATION BINARY DETECTABILITY",
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY",
        "primary_metric": "MCC",
        "winner_declared": True,
        "winning_comparator": "DIR_SGC_11D1D",
        "winning_candidate": EXPECTED_GNN_CANDIDATE,
        "winning_threshold": EXPECTED_GNN_THRESHOLD,
        "gnn_mcc": gnn_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "baseline_mcc": baseline_metrics["mcc"],
        "gnn_minus_baseline_mcc": comparison["gnn_minus_baseline_mcc"],
        "gnn_minus_baseline_mcc_ci_95": comparison["gnn_minus_baseline_mcc_ci_95"],
        "gnn_minus_deep_mcc": comparison["gnn_minus_deep_mcc"],
        "gnn_minus_deep_mcc_ci_95": comparison["gnn_minus_deep_mcc_ci_95"],
        "statistical_gain_over_baseline": True,
        "statistical_gain_over_deep": True,
        "required_mcc_gain_met": True,
        "project_target_status": "MET",
        "absolute_performance_status": "NOT_MET",
        "governance_disposition": "FREEZE_GNN_AS_PRIMARY_GRAPH_COMPARATOR",
        "gnn_retraining_authorized": False,
        "gnn_threshold_change_authorized": False,
        "dev_site_test_state": "CONSUMED_AND_FROZEN_FOR_ALL_EXISTING_COMPARATORS",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "comparator_registry_csv": record(COMPARATOR_REGISTRY_CSV),
        "comparator_registry_json": record(COMPARATOR_REGISTRY_JSON),
    })

    atomic_json(HYBRID_READINESS_POLICY, {
        "stage": "11D-1E",
        "title": "HYBRID MODEL READINESS POLICY",
        "status": "PASS",
        "readiness_status": "READY_FOR_ARCHITECTURE_AND_TRAINING_CONTRACT_FREEZE_ONLY",
        "created_at_utc": created_at,
        "rationale": {
            "gnn_is_statistically_supported_winner": True,
            "relative_project_target_met": True,
            "absolute_performance_target_met": False,
            "improvement_objective": (
                "Improve recall, F1, balanced accuracy, and MCC without losing "
                "the GNN topology advantage or violating split isolation."
            ),
        },
        "frozen_components": [
            "DIR_SGC_11D1D",
            "DEEP_MLP_11C5K",
            "CONVENTIONAL_LOGREG_11C5H",
        ],
        "development_contract": {
            "training_partition": "DEV_TRAIN",
            "selection_and_threshold_partition": "DEV_CALIBRATION",
            "dev_site_test_use": "FINAL_LOCKED_EVALUATION_ONLY_AFTER_HYBRID_FREEZE",
            "site_grouping_unit": "PHYSICAL_FAULT_SITE_WITH_SA0_SA1_GROUPED",
            "primary_metric": "MCC",
            "graph_source": "FROZEN_GOLDEN_NETLIST_GRAPH_11D1A",
            "post_simulation_features": 0,
            "target_present_in_input": False,
        },
        "stacking_safety_contract": {
            "dev_train_meta_features": "OUT_OF_FOLD_ONLY_IF_MODEL_SCORE_STACKING_IS_USED",
            "dev_calibration_meta_features": "GENERATED_WITH_MODELS_NOT_FIT_ON_DEV_CALIBRATION",
            "dev_site_test_predictions_as_training_features": "PROHIBITED",
            "in_sample_dev_train_scores_for_meta_learner": "PROHIBITED",
            "validation_or_holdout_for_design_selection": "PROHIBITED",
        },
        "authorized_actions": [
            "DEFINE_HYBRID_ARCHITECTURE",
            "DEFINE_LEAKAGE_SAFE_OUT_OF_FOLD_SCORE_PROTOCOL",
            "DEFINE_HYBRID_CANDIDATE_GRID",
            "DEFINE_TRAINING_CALIBRATION_AND_REPLAY_CONTRACT",
            "FREEZE_HYBRID_ENVIRONMENT_AND_ACCEPTANCE_RULES",
        ],
        "prohibited_actions": [
            "FIT_HYBRID_PARAMETERS",
            "SELECT_HYBRID_CANDIDATE",
            "SELECT_HYBRID_THRESHOLD",
            "REOPEN_DEV_SITE_TEST_FOR_DESIGN",
            "USE_DEV_SITE_TEST_PREDICTIONS_AS_TRAINING_FEATURES",
            "EXPOSE_VALIDATION_VECTORS",
            "EXPOSE_HOLDOUT_VECTORS",
            "MODIFY_FROZEN_COMPARATOR_MODELS_OR_THRESHOLDS",
            "USE_POST_SIMULATION_FEATURES",
        ],
        "hybrid_architecture_contract_authorized": True,
        "hybrid_training_authorized": False,
        "hybrid_site_test_evaluation_authorized": False,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "next_gate": "STAGE 11D-2A — HYBRID MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
    })

    outputs = {
        "gnn_disposition_policy": record(GNN_DISPOSITION_POLICY),
        "hybrid_readiness_policy": record(HYBRID_READINESS_POLICY),
        "comparator_registry_csv": record(COMPARATOR_REGISTRY_CSV),
        "comparator_registry_json": record(COMPARATOR_REGISTRY_JSON),
    }
    atomic_json(AUDIT, {
        "stage": "11D-1E",
        "title": "GNN DISPOSITION AND HYBRID READINESS FREEZE",
        "status": "PASS",
        "disposition_status": "FROZEN",
        "hybrid_readiness_status": "FROZEN",
        "comparator_registry_status": "FROZEN",
        "created_at_utc": created_at,
        "winner_declared": True,
        "winner": "GNN",
        "gnn_status": "FROZEN PRIMARY GRAPH COMPARATOR",
        "deep_status": "FROZEN SECONDARY NEURAL COMPARATOR",
        "baseline_status": "FROZEN PRIMARY SIMPLE COMPARATOR",
        "gnn_candidate": EXPECTED_GNN_CANDIDATE,
        "gnn_threshold": EXPECTED_GNN_THRESHOLD,
        "gnn_mcc": gnn_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "baseline_mcc": baseline_metrics["mcc"],
        "gnn_minus_baseline_mcc": comparison["gnn_minus_baseline_mcc"],
        "gnn_minus_baseline_mcc_ci_95": comparison["gnn_minus_baseline_mcc_ci_95"],
        "gnn_minus_deep_mcc": comparison["gnn_minus_deep_mcc"],
        "gnn_minus_deep_mcc_ci_95": comparison["gnn_minus_deep_mcc_ci_95"],
        "statistical_gain_over_baseline": "SUPPORTED",
        "statistical_gain_over_deep": "SUPPORTED",
        "required_mcc_improvement": "MET",
        "minimum_validity_status": "PASS",
        "absolute_performance_status": "NOT_MET",
        "project_target_status": "MET",
        "dev_site_test_opened": False,
        "hybrid_architecture_contract": "AUTHORIZED",
        "hybrid_training": "NOT_YET_AUTHORIZED",
        "hybrid_site_test_evaluation": "NOT_YET_AUTHORIZED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "post_simulation_features": 0,
        "input_evidence": inputs,
        "outputs": outputs,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
        "next_gate": "STAGE 11D-2A — HYBRID MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
    })
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-1E — GNN DISPOSITION AND HYBRID READINESS FREEZE")
    print("Status                         : PASS")
    print("Disposition status             : FROZEN")
    print("Hybrid readiness status        : FROZEN")
    print("Winner declared                : GNN")
    print("GNN                            : FROZEN PRIMARY GRAPH COMPARATOR")
    print("Deep MLP                       : FROZEN SECONDARY NEURAL COMPARATOR")
    print("Conventional baseline          : FROZEN PRIMARY SIMPLE COMPARATOR")
    print("GNN MCC                        :", f"{gnn_metrics['mcc']:.8f}")
    print("Deep MLP MCC                   :", f"{deep_metrics['mcc']:.8f}")
    print("Conventional baseline MCC      :", f"{baseline_metrics['mcc']:.8f}")
    print("MCC delta (GNN-baseline)       :", f"{comparison['gnn_minus_baseline_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{comparison['gnn_minus_baseline_mcc_ci_95'][0]:.8f}, {comparison['gnn_minus_baseline_mcc_ci_95'][1]:.8f}]")
    print("MCC delta (GNN-deep)           :", f"{comparison['gnn_minus_deep_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{comparison['gnn_minus_deep_mcc_ci_95'][0]:.8f}, {comparison['gnn_minus_deep_mcc_ci_95'][1]:.8f}]")
    print("Statistical gain vs baseline   : SUPPORTED")
    print("Statistical gain vs deep       : SUPPORTED")
    print("Required +0.05 MCC gain        : MET")
    print("Project target                 : MET")
    print("Absolute performance target    : NOT_MET")
    print("DEV_SITE_TEST opened           : NO")
    print("Hybrid architecture contract   : AUTHORIZED")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("GNN model modified             : NO")
    print("GNN disposition policy         :", GNN_DISPOSITION_POLICY)
    print("GNN disposition policy SHA     :", outputs["gnn_disposition_policy"]["sha256"])
    print("Hybrid readiness policy        :", HYBRID_READINESS_POLICY)
    print("Hybrid readiness policy SHA    :", outputs["hybrid_readiness_policy"]["sha256"])
    print("Comparator registry            :", COMPARATOR_REGISTRY_CSV)
    print("Comparator registry SHA        :", outputs["comparator_registry_csv"]["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-2A — HYBRID MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
