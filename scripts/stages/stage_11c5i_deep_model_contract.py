#!/usr/bin/env python3
"""Freeze the non-GNN deep diagnostic architecture and training contract."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.neural_network import MLPClassifier
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error


VERSION = "HMAC-DEEP-DIAGNOSTIC-CONTRACT-v1"
RANDOM_SEED = 20_260_904

EXPECTED_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512
EXPECTED_VECTORS = 64

EXPECTED_TRAIN_SITES = 15_987
EXPECTED_CALIBRATION_SITES = 3_426
EXPECTED_SITE_TEST_SITES = 3_426
EXPECTED_TOTAL_SITES = 22_839

EXPECTED_TRAIN_ROWS = 2_046_336
EXPECTED_CALIBRATION_ROWS = 438_528
EXPECTED_SITE_TEST_ROWS = 438_528
EXPECTED_TOTAL_ROWS = 2_923_392

EXPECTED_TRAIN_POSITIVES = 894_000
EXPECTED_CALIBRATION_POSITIVES = 192_943
EXPECTED_SITE_TEST_POSITIVES = 191_945

EXPECTED_BASELINE_CANDIDATE = "SGD_LOGREG_L2_A1E6_BAL"
EXPECTED_BASELINE_THRESHOLD = 0.98536854982376099
EXPECTED_BASELINE_SITE_TEST_MCC = 0.14208149

HIDDEN_LAYERS = (128, 64, 32)
TRAINABLE_PARAMETERS = (
    (EXPECTED_FEATURES + 1) * HIDDEN_LAYERS[0]
    + (HIDDEN_LAYERS[0] + 1) * HIDDEN_LAYERS[1]
    + (HIDDEN_LAYERS[1] + 1) * HIDDEN_LAYERS[2]
    + (HIDDEN_LAYERS[2] + 1)
)

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"
BASELINE_ROOT = RESULT_ROOT / "baseline_training_11c5g"
BASELINE_EVAL_ROOT = RESULT_ROOT / "baseline_evaluation_11c5h"
CONFIG_ROOT = ROOT / "config/diagnostic_model"

TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CALIBRATION_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
FEATURE_MANIFEST = RESULT_ROOT / "hmac_leakage_safe_feature_matrix_manifest_11c5e.json"
FEATURE_AUDIT = RESULT_ROOT / "hmac_leakage_safe_feature_matrix_freeze_11c5e.json"

BASELINE_CONTRACT = CONFIG_ROOT / "hmac_conventional_baseline_training_contract_11c5f.json"
BASELINE_GRID = CONFIG_ROOT / "hmac_conventional_baseline_candidate_grid_11c5f.csv"
BASELINE_ENVIRONMENT = RESULT_ROOT / "hmac_conventional_baseline_environment_11c5f.json"
BASELINE_CONTRACT_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_training_contract_freeze_11c5f.json"

BASELINE_MODEL = BASELINE_ROOT / "hmac_selected_conventional_baseline_11c5g.joblib"
BASELINE_SELECTION_LOCK = BASELINE_ROOT / "hmac_conventional_baseline_selection_lock_11c5g.json"
BASELINE_TRAINING_MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_training_manifest_11c5g.json"
BASELINE_TRAINING_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_training_calibration_freeze_11c5g.json"

BASELINE_EVALUATOR = ROOT / "stage_11c5h_locked_site_test.py"
BASELINE_PREDICTIONS = BASELINE_EVAL_ROOT / "hmac_conventional_baseline_site_test_predictions_11c5h.npz"
BASELINE_EVALUATION_LOCK = BASELINE_EVAL_ROOT / "hmac_conventional_baseline_site_test_evaluation_lock_11c5h.json"
BASELINE_EVALUATION_MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_site_test_manifest_11c5h.json"
BASELINE_EVALUATION_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_site_test_freeze_11c5h.json"

ARCHITECTURE = CONFIG_ROOT / "hmac_deep_diagnostic_architecture_11c5i.json"
TRAINING_CONTRACT = CONFIG_ROOT / "hmac_deep_diagnostic_training_contract_11c5i.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_deep_diagnostic_candidate_grid_11c5i.csv"
ENVIRONMENT = RESULT_ROOT / "hmac_deep_diagnostic_environment_11c5i.json"
AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_architecture_training_contract_freeze_11c5i.json"

EXPECTED_INPUTS = {
    TRAIN_MATRIX: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CALIBRATION_MATRIX: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    FEATURE_MANIFEST: "413cbad01681b4f209de6fce44d3777bccfe424c50c3c95582c627e44965e186",
    FEATURE_AUDIT: "6320bb732b84ae56c937e188189fb87a159d35a164eed892ec3da9d1ee27e540",
    BASELINE_CONTRACT: "610211065fd81538183472377b0a69aab2f4967cc4f99ada6393712fc090c007",
    BASELINE_GRID: "5901d7e99a6fc08bfe7e68969caeeb8fd03c3ec0121e80c27d22ebf545e7e132",
    BASELINE_ENVIRONMENT: "9e42cceccff6c016f9af5c237a30f400c16ed626d5681159c8348357f7db8e3f",
    BASELINE_CONTRACT_AUDIT: "b94314756d4576df9cecc81ecf7e6b11c12c7cab757544bbe693c96a50292fb1",
    BASELINE_MODEL: "8220ae569958e3d072b1ac40ac3c2958c7428419fb0ebb4acab8f324eaea5833",
    BASELINE_SELECTION_LOCK: "1532a30fb96a5f142ba05a778643bdb9aa93c058eba568e3df92c09f47e523fc",
    BASELINE_TRAINING_MANIFEST: "9c612c42d3ba97c47372a27927fc31ff374fa75e7f19577bbda458061166ada7",
    BASELINE_TRAINING_AUDIT: "f9f77bdc5c07cec91ca180f3d8b2cfb93c744e029445b6a226e94b5add8df5f1",
    BASELINE_EVALUATOR: "8f720d2f5a2eff1d964507424af502fba9aca60abf09d50fb897d33075d290e3",
    BASELINE_PREDICTIONS: "ec6d1b20e024b9739d1e79d7ef19bf0a5cfcb3292b16ae355c70f4e014248f76",
    BASELINE_EVALUATION_LOCK: "5ff4987169c2b885826db13cf721164b65e009a83290d9aa21da9eeb13b213c4",
    BASELINE_EVALUATION_MANIFEST: "cfe75842e7b081230c29d5162ec0292381e574b8679f58118c9a26bf6a7cc1ef",
    BASELINE_EVALUATION_AUDIT: "54d86a12e605ce8f2b920c55e87c6ad9a3e30b41df8420730adfff4be4fbee19",
}

CANDIDATE_COLUMNS = [
    "candidate_id",
    "hidden_layer_sizes",
    "activation",
    "solver",
    "alpha",
    "learning_rate_init",
    "batch_size",
    "epochs",
    "shuffle",
    "early_stopping",
    "random_seed",
    "trainable_parameters",
]

CANDIDATES = [
    {
        "candidate_id": "DEEP_MLP_128_64_32_A1E4_LR1E3",
        "hidden_layer_sizes": "128;64;32",
        "activation": "relu",
        "solver": "adam",
        "alpha": "0.0001",
        "learning_rate_init": "0.001",
        "batch_size": "8192",
        "epochs": "4",
        "shuffle": "false",
        "early_stopping": "false",
        "random_seed": str(RANDOM_SEED),
        "trainable_parameters": str(TRAINABLE_PARAMETERS),
    },
    {
        "candidate_id": "DEEP_MLP_128_64_32_A1E5_LR3E4",
        "hidden_layer_sizes": "128;64;32",
        "activation": "relu",
        "solver": "adam",
        "alpha": "0.00001",
        "learning_rate_init": "0.0003",
        "batch_size": "8192",
        "epochs": "4",
        "shuffle": "false",
        "early_stopping": "false",
        "random_seed": str(RANDOM_SEED),
        "trainable_parameters": str(TRAINABLE_PARAMETERS),
    },
]


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
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        stop(f"artifact outside project root: {path}")


def record(path: Path) -> dict:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not read JSON {path}: {error}")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {name}")


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {relative(path)}")
    require(path.stat().st_size > 0, f"empty frozen input: {relative(path)}")
    actual_sha = sha256(path)
    require(
        actual_sha == expected_sha,
        f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual_sha}",
    )
    print(f"  {path.name:<72}: OK", flush=True)
    return record(path)


def verify_outputs_absent() -> None:
    for path in (ARCHITECTURE, TRAINING_CONTRACT, CANDIDATE_GRID, ENVIRONMENT, AUDIT):
        require(not path.exists(), f"Stage 11C-5I output already exists: {relative(path)}")


def resolve_manifest_outputs(manifest: dict) -> dict:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and outputs, "Stage 11C-5H manifest output records")
    resolved = {}
    for name, item in sorted(outputs.items()):
        require(isinstance(item, dict), f"manifest output record {name}")
        path_text = item.get("path")
        expected_sha = item.get("sha256")
        require(isinstance(path_text, str) and path_text, f"manifest output path {name}")
        require(isinstance(expected_sha, str) and len(expected_sha) == 64, f"manifest output SHA {name}")
        path = (ROOT / path_text).resolve()
        relative(path)
        require(path.is_file() and path.stat().st_size > 0, f"missing manifest output {name}")
        require(sha256(path) == expected_sha, f"manifest output SHA mismatch {name}")
        if "bytes" in item:
            require(path.stat().st_size == int(item["bytes"]), f"manifest output size {name}")
        resolved[name] = record(path)
    return resolved


def verify_semantics() -> tuple[dict, dict]:
    feature_audit = load_json(FEATURE_AUDIT)
    baseline_contract = load_json(BASELINE_CONTRACT)
    baseline_training_audit = load_json(BASELINE_TRAINING_AUDIT)
    baseline_lock = load_json(BASELINE_SELECTION_LOCK)
    baseline_eval_manifest = load_json(BASELINE_EVALUATION_MANIFEST)
    baseline_eval_lock = load_json(BASELINE_EVALUATION_LOCK)
    baseline_eval_audit = load_json(BASELINE_EVALUATION_AUDIT)

    require(feature_audit.get("status") == "PASS", "Stage 11C-5E status")
    require(baseline_contract.get("status") == "PASS", "Stage 11C-5F status")
    require(baseline_contract.get("gnn_training_authorized") is False, "Stage 11C-5F GNN lock")
    require(baseline_training_audit.get("status") == "PASS", "Stage 11C-5G status")
    require(baseline_training_audit.get("execution_status") == "FROZEN", "Stage 11C-5G execution freeze")
    require(baseline_training_audit.get("model_selection_status") == "FROZEN", "Stage 11C-5G selection freeze")
    require(baseline_lock.get("selected_candidate_id") == EXPECTED_BASELINE_CANDIDATE, "baseline candidate")
    require(float(baseline_lock.get("selected_threshold")) == EXPECTED_BASELINE_THRESHOLD, "baseline threshold")

    require(baseline_eval_manifest.get("status") == "PASS", "Stage 11C-5H manifest status")
    require(baseline_eval_manifest.get("minimum_validity_status") == "PASS", "baseline minimum validity")
    require(baseline_eval_manifest.get("project_target_status") == "NOT_MET", "baseline project-target status")
    require(baseline_eval_lock.get("lock_status") == "FROZEN", "baseline evaluation lock")
    require(baseline_eval_lock.get("model_retrained") is False, "baseline retraining")
    require(baseline_eval_lock.get("threshold_changed") is False, "baseline threshold change")
    require(baseline_eval_audit.get("status") == "PASS", "Stage 11C-5H audit status")
    require(baseline_eval_audit.get("evaluation_status") == "FROZEN", "baseline evaluation status")
    require(baseline_eval_audit.get("baseline_status") == "FROZEN COMPARATOR", "baseline comparator status")
    require(int(baseline_eval_audit.get("site_test_sites")) == EXPECTED_SITE_TEST_SITES, "site-test sites")
    require(int(baseline_eval_audit.get("site_test_samples")) == EXPECTED_SITE_TEST_ROWS, "site-test rows")
    require(abs(float(baseline_eval_audit.get("mcc")) - EXPECTED_BASELINE_SITE_TEST_MCC) < 1e-8, "baseline MCC")
    require(baseline_eval_audit.get("dev_site_test_state") == "CONSUMED AND FROZEN", "site-test state")
    require(baseline_eval_audit.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(baseline_eval_audit.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")
    require(baseline_eval_audit.get("gnn_training_authorized") is False, "Stage 11C-5H GNN lock")

    manifest_outputs = resolve_manifest_outputs(baseline_eval_manifest)
    return baseline_eval_audit, manifest_outputs


def verify_environment() -> dict:
    frozen = load_json(BASELINE_ENVIRONMENT)
    require(frozen.get("status") == "PASS", "frozen environment status")
    expected_packages = frozen.get("packages")
    actual_packages = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": package_version("threadpoolctl"),
    }
    require(actual_packages == expected_packages, f"package environment changed: {actual_packages}")

    estimator = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=8192,
        learning_rate_init=1e-3,
        max_iter=1,
        shuffle=False,
        random_state=RANDOM_SEED,
        early_stopping=False,
    )
    require(estimator.hidden_layer_sizes == HIDDEN_LAYERS, "MLP architecture probe")
    require(estimator.shuffle is False, "MLP deterministic shuffle probe")

    return {
        "status": "PASS",
        "environment_status": "FROZEN",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": actual_packages,
        "framework": "scikit-learn",
        "estimator": "sklearn.neural_network.MLPClassifier",
        "framework_probe": "PASS",
        "device": "CPU",
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "thread_limit": 1,
    }


def main() -> None:
    require(SOURCE.parent == ROOT, "run the Stage 11C-5I script from the project root")
    verify_outputs_absent()

    print("STAGE 11C-5I — DEEP DIAGNOSTIC MODEL ARCHITECTURE AND TRAINING CONTRACT")
    print("FROZEN INPUT VERIFICATION", flush=True)
    evidence = {
        relative(path): verify_input(path, expected_sha)
        for path, expected_sha in EXPECTED_INPUTS.items()
    }
    baseline_audit, baseline_outputs = verify_semantics()
    environment_document = verify_environment()
    created_at = datetime.now(timezone.utc).isoformat()

    architecture_document = {
        "stage": "11C-5I",
        "title": "DEEP DIAGNOSTIC MODEL ARCHITECTURE FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "created_at_utc": created_at,
        "model_family": "FEEDFORWARD MULTILAYER PERCEPTRON",
        "model_role": "NON-GNN DEEP DIAGNOSTIC COMPARATOR",
        "framework": "scikit-learn",
        "estimator": "sklearn.neural_network.MLPClassifier",
        "task": "PRE-SIMULATION BINARY DETECTABILITY",
        "fault_scope": "PERSISTENT NET-STEM SA0/SA1 ONLY",
        "input_features": EXPECTED_FEATURES,
        "input_partition": {
            "stuck_value": 1,
            "physical_site_features": EXPECTED_SITE_FEATURES,
            "test_vector_features": EXPECTED_VECTOR_FEATURES,
        },
        "hidden_layer_sizes": list(HIDDEN_LAYERS),
        "hidden_layer_count": len(HIDDEN_LAYERS),
        "hidden_activation": "RELU",
        "output_units": 1,
        "output_semantics": "DETECTABILITY PROBABILITY",
        "objective": "BINARY LOG LOSS",
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "post_simulation_features": 0,
        "target_present_in_x": False,
        "graph_message_passing": False,
        "gnn_model": False,
        "architecture_change_after_freeze_authorized": False,
    }
    atomic_json(ARCHITECTURE, architecture_document)
    architecture_record = record(ARCHITECTURE)

    atomic_csv(CANDIDATE_GRID, CANDIDATES)
    grid_record = record(CANDIDATE_GRID)

    environment_document.update(
        {
            "stage": "11C-5I",
            "title": "DEEP DIAGNOSTIC TRAINING ENVIRONMENT FREEZE",
            "created_at_utc": created_at,
            "baseline_environment": evidence[relative(BASELINE_ENVIRONMENT)],
        }
    )
    atomic_json(ENVIRONMENT, environment_document)
    environment_record = record(ENVIRONMENT)

    contract_document = {
        "stage": "11C-5I",
        "title": "DEEP DIAGNOSTIC MODEL TRAINING CONTRACT FREEZE",
        "status": "PASS",
        "training_contract_status": "FROZEN",
        "created_at_utc": created_at,
        "contract_version": VERSION,
        "architecture": architecture_record,
        "candidate_grid": grid_record,
        "environment": environment_record,
        "model_family": "FEEDFORWARD MULTILAYER PERCEPTRON",
        "task": "PRE-SIMULATION BINARY DETECTABILITY",
        "target": "detected",
        "fault_scope": "PERSISTENT NET-STEM SA0/SA1 ONLY",
        "model_features": EXPECTED_FEATURES,
        "physical_site_groups": EXPECTED_TOTAL_SITES,
        "fault_instances": EXPECTED_TOTAL_SITES * 2,
        "partitions": {
            "DEV_TRAIN": {
                "sites": EXPECTED_TRAIN_SITES,
                "samples": EXPECTED_TRAIN_ROWS,
                "positive_samples": EXPECTED_TRAIN_POSITIVES,
                "use": "PARAMETER FITTING ONLY",
            },
            "DEV_CALIBRATION": {
                "sites": EXPECTED_CALIBRATION_SITES,
                "samples": EXPECTED_CALIBRATION_ROWS,
                "positive_samples": EXPECTED_CALIBRATION_POSITIVES,
                "use": "CANDIDATE AND THRESHOLD SELECTION ONLY",
            },
            "DEV_SITE_TEST": {
                "sites": EXPECTED_SITE_TEST_SITES,
                "samples": EXPECTED_SITE_TEST_ROWS,
                "positive_samples": EXPECTED_SITE_TEST_POSITIVES,
                "use": "LOCKED EVALUATION ONLY AFTER DEEP MODEL SELECTION FREEZE",
                "state": "LOCKED FOR DEEP TRAINING AND SELECTION",
            },
        },
        "site_group_overlap": 0,
        "sa0_sa1_group_separations": 0,
        "trainable_candidates": len(CANDIDATES),
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "training_batch_size": 8192,
        "inference_batch_size": 32768,
        "epochs_per_candidate": 4,
        "optimizer": "ADAM",
        "loss": "BINARY LOG LOSS",
        "class_balance_policy": "NATURAL DEV_TRAIN PREVALENCE; NO RESAMPLING",
        "external_shuffle": "DETERMINISTIC PCG64 PERMUTATION PER EPOCH",
        "estimator_shuffle": False,
        "early_stopping": False,
        "random_seed": RANDOM_SEED,
        "checkpoint_after_each_epoch": True,
        "resume_supported": True,
        "selection_partition": "DEV_CALIBRATION",
        "primary_selection_metric": "MCC",
        "selection_tie_breakers": ["PR_AUC DESC", "BRIER ASC", "CANDIDATE_ID ASC"],
        "threshold_selection": {
            "partition": "DEV_CALIBRATION",
            "metric": "MCC",
            "grid_start": 0.0,
            "grid_stop": 1.0,
            "grid_points": 2001,
            "tie_breakers": ["BALANCED_ACCURACY DESC", "DISTANCE_TO_0.5 ASC", "THRESHOLD ASC"],
        },
        "minimum_validity_checks": [
            "MCC > 0",
            "BALANCED_ACCURACY > 0.5",
            "PR_AUC > PARTITION POSITIVE PREVALENCE",
            "FINITE PROBABILITIES AND LOSS",
        ],
        "project_target_checks": [
            "MCC >= 0.40",
            "BALANCED_ACCURACY >= 0.70",
            "F1 >= 0.70",
            "RECALL >= 0.70",
        ],
        "baseline_comparison": {
            "status": "FROZEN COMPARATOR",
            "candidate": EXPECTED_BASELINE_CANDIDATE,
            "site_test_mcc": float(baseline_audit["mcc"]),
            "minimum_required_mcc_improvement": 0.05,
            "comparison_partition": "DEV_SITE_TEST",
            "comparison_permitted_only_after_deep_selection_freeze": True,
        },
        "required_evaluation_breakdowns": [
            "FAULT_MODEL SA0/SA1",
            "SITE_CATEGORY SEQUENTIAL/COMBINATIONAL",
            "DRIVER_CELL_TYPE WHEN REPRESENTED",
        ],
        "site_group_bootstrap_replicates": 1000,
        "site_group_bootstrap_confidence": 0.95,
        "dev_train_authorized": True,
        "dev_calibration_authorized_for_selection": True,
        "dev_site_test_authorized_during_training": False,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
        "architecture_tuning_after_site_test": False,
        "retraining_after_site_test": False,
        "post_simulation_features": 0,
        "target_present_in_x": False,
        "frozen_inputs": evidence,
        "baseline_evaluation_outputs": baseline_outputs,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
    }
    atomic_json(TRAINING_CONTRACT, contract_document)
    contract_record = record(TRAINING_CONTRACT)

    audit_document = {
        "stage": "11C-5I",
        "title": "DEEP DIAGNOSTIC MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "environment_status": "FROZEN",
        "candidate_grid_status": "FROZEN",
        "contract_version": VERSION,
        "model_family": "FEEDFORWARD MULTILAYER PERCEPTRON",
        "deep_learning_model": True,
        "gnn_model": False,
        "hidden_layers": list(HIDDEN_LAYERS),
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_candidates": len(CANDIDATES),
        "model_features": EXPECTED_FEATURES,
        "dev_train_samples": EXPECTED_TRAIN_ROWS,
        "dev_calibration_samples": EXPECTED_CALIBRATION_ROWS,
        "dev_site_test_samples": EXPECTED_SITE_TEST_ROWS,
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "epochs_per_candidate": 4,
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "dev_site_test": "LOCKED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "deep_model_training": "AUTHORIZED FOR DEV_TRAIN",
        "gnn_training": "NOT YET AUTHORIZED",
        "hybrid_training": "NOT YET AUTHORIZED",
        "baseline_status": "FROZEN COMPARATOR",
        "baseline_project_target": "NOT_MET",
        "architecture": architecture_record,
        "training_contract": contract_record,
        "candidate_grid": grid_record,
        "environment": environment_record,
        "contract_source": record(SOURCE),
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "next_gate": "DEEP DIAGNOSTIC MODEL TRAINING AND CALIBRATION EXECUTION",
    }
    atomic_json(AUDIT, audit_document)
    audit_record = record(AUDIT)

    print("\nSTAGE 11C-5I — DEEP DIAGNOSTIC MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE")
    print("Status                         : PASS")
    print("Architecture status            : FROZEN")
    print("Training contract status       : FROZEN")
    print("Environment status             : FROZEN")
    print("Candidate grid status          : FROZEN")
    print("Model family                   : FEEDFORWARD MULTILAYER PERCEPTRON")
    print("Deep learning model            : YES")
    print("GNN model                      : NO")
    print("Architecture                   : 527 -> 128 -> 64 -> 32 -> 1")
    print("Trainable parameters           :", TRAINABLE_PARAMETERS)
    print("Trainable candidates           :", len(CANDIDATES))
    print("DEV_TRAIN samples              :", EXPECTED_TRAIN_ROWS)
    print("DEV_CALIBRATION samples        :", EXPECTED_CALIBRATION_ROWS)
    print("DEV_SITE_TEST samples          :", EXPECTED_SITE_TEST_ROWS)
    print("Candidate execution            : SEQUENTIAL")
    print("Parallel candidates            : 1")
    print("Epochs per candidate           : 4")
    print("Selection partition            : DEV_CALIBRATION")
    print("Primary metric                 : MCC")
    print("DEV_SITE_TEST                  : LOCKED")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Deep model training            : AUTHORIZED FOR DEV_TRAIN")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Canonical dataset modified     : NO")
    print("Feature matrices modified      : NO")
    print("Baseline model modified        : NO")
    print("Architecture                   :", ARCHITECTURE)
    print("Architecture SHA               :", architecture_record["sha256"])
    print("Training contract              :", TRAINING_CONTRACT)
    print("Training contract SHA          :", contract_record["sha256"])
    print("Candidate grid                 :", CANDIDATE_GRID)
    print("Candidate grid SHA             :", grid_record["sha256"])
    print("Environment                    :", ENVIRONMENT)
    print("Environment SHA                :", environment_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : DEEP DIAGNOSTIC MODEL TRAINING AND CALIBRATION EXECUTION")


if __name__ == "__main__":
    main()
