#!/usr/bin/env python3
"""Freeze the conventional HMAC diagnostic baseline training contract."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.linear_model import SGDClassifier
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the project .venv before running."
    ) from error


VERSION = "HMAC-CONVENTIONAL-BASELINE-CONTRACT-FREEZER-v1"
CONTRACT_VERSION = "HMAC-CONVENTIONAL-BASELINE-TRAINING-CONTRACT-v1"
RANDOM_SEED = 20_260_903

ROOT = Path.cwd().resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"

FEATURE_GENERATOR = ROOT / "stage_11c5e_feature_matrix.py"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
FEATURE_MANIFEST = RESULT_ROOT / "hmac_leakage_safe_feature_matrix_manifest_11c5e.json"
FEATURE_AUDIT = RESULT_ROOT / "hmac_leakage_safe_feature_matrix_freeze_11c5e.json"

MATRIX_PATHS = {
    "DEV_TRAIN": FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz",
    "DEV_CALIBRATION": FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz",
    "DEV_SITE_TEST": FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz",
}

CONTRACT = CONFIG_ROOT / "hmac_conventional_baseline_training_contract_11c5f.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_conventional_baseline_candidate_grid_11c5f.csv"
ENVIRONMENT = RESULT_ROOT / "hmac_conventional_baseline_environment_11c5f.json"
AUDIT = RESULT_ROOT / "hmac_conventional_baseline_training_contract_freeze_11c5f.json"

EXPECTED_INPUTS = {
    FEATURE_GENERATOR:
        "e0b4f72c1b6c334f683ce84e24b11adafed3786d46340e16f1e7d99084ccd8bb",
    FEATURE_SCHEMA:
        "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    FEATURE_MANIFEST:
        "413cbad01681b4f209de6fce44d3777bccfe424c50c3c95582c627e44965e186",
    FEATURE_AUDIT:
        "6320bb732b84ae56c937e188189fb87a159d35a164eed892ec3da9d1ee27e540",
    MATRIX_PATHS["DEV_TRAIN"]:
        "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    MATRIX_PATHS["DEV_CALIBRATION"]:
        "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    MATRIX_PATHS["DEV_SITE_TEST"]:
        "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    ROOT / "config/diagnostic_model/hmac_diagnostic_task_feature_split_policy_11c5d.json":
        "74243fa3b40f047a9f25a23c87eaccf6ac5a71a07241efef58df6b9e8e5322be",
    RESULT_ROOT / "hmac_fault_site_group_split_11c5d.csv":
        "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    RESULT_ROOT / "hmac_fault_site_group_split_manifest_11c5d.json":
        "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
    RESULT_ROOT / "hmac_diagnostic_task_feature_split_freeze_11c5d.json":
        "ff381a785c12281de3aeb1cda0e1d2b05d56444144b47dfa626b7334ee5b2985",
}

PARTITIONS = ("DEV_TRAIN", "DEV_CALIBRATION", "DEV_SITE_TEST")
EXPECTED_ROWS = {
    "DEV_TRAIN": 2_046_336,
    "DEV_CALIBRATION": 438_528,
    "DEV_SITE_TEST": 438_528,
}
EXPECTED_SITES = {
    "DEV_TRAIN": 15_987,
    "DEV_CALIBRATION": 3_426,
    "DEV_SITE_TEST": 3_426,
}
EXPECTED_POSITIVES = {
    "DEV_TRAIN": 894_000,
    "DEV_CALIBRATION": 192_943,
    "DEV_SITE_TEST": 191_945,
}
EXPECTED_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512

NPZ_MEMBERS = {
    "feature_columns",
    "format_version",
    "partition",
    "record_id",
    "site_feature_columns",
    "site_features",
    "site_ids",
    "site_indices",
    "site_row",
    "stuck_value",
    "target_detected",
    "vector_features",
    "vector_ids",
    "vector_row",
}

FORBIDDEN_FEATURES = {
    "detected",
    "activity",
    "cycles",
    "baseline_cycles",
    "latency_delta",
    "timed_out",
    "unknown",
    "expected_digest",
    "actual_digest",
    "digest_hamming_distance",
    "activated_vectors",
    "detected_vectors",
    "timeout_vectors",
    "final_classification",
}

CANDIDATE_HEADER = [
    "candidate_id",
    "model_family",
    "loss",
    "penalty",
    "alpha",
    "sample_weighting",
    "epochs",
    "batch_size",
    "average",
    "random_seed",
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
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        stop(f"artifact outside project root: {path}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not read JSON {path}: {error}")


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def output_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_file(path: Path, expected: str) -> dict:
    require(path.is_file(), f"missing frozen input: {path}")
    require(path.stat().st_size > 0, f"empty frozen input: {path}")
    actual = sha256(path)
    require(
        actual == expected,
        f"SHA mismatch for {relative(path)}: expected {expected}, actual {actual}",
    )
    return output_record(path)


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    relative(path)
    return path


def verify_manifest_artifact(record: dict, label: str) -> dict:
    require(isinstance(record, dict), f"missing manifest record for {label}")
    require(set(("path", "sha256", "bytes")) <= set(record), f"manifest fields for {label}")
    path = project_path(str(record["path"]))
    require(path.is_file(), f"missing manifest artifact {label}: {path}")
    require(path.stat().st_size == int(record["bytes"]), f"byte count for {label}")
    actual = sha256(path)
    require(actual == str(record["sha256"]), f"manifest SHA for {label}")
    return output_record(path)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required Python package is missing: {distribution}")


def physical_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            return int(fields[1]) * 1024
    return None


def inspect_feature_contract() -> tuple[dict, dict, dict]:
    schema = load_json(FEATURE_SCHEMA)
    manifest = load_json(FEATURE_MANIFEST)
    audit = load_json(FEATURE_AUDIT)

    require(schema.get("status") == "PASS", "feature schema status")
    require(schema.get("schema_version") == "HMAC-DIAGNOSTIC-FEATURE-SCHEMA-v1", "feature schema version")
    require(schema.get("matrix_format_version") == "HMAC-FACTORIZED-FEATURE-MATRIX-v1", "matrix format")
    require(schema.get("task") == "pre_simulation_persistent_fault_detectability", "diagnostic task")
    require(schema.get("model_feature_count") == EXPECTED_FEATURES, "feature count")
    require(schema.get("site_feature_count") == EXPECTED_SITE_FEATURES, "site-feature count")
    require(schema.get("stimulus_feature_count") == EXPECTED_VECTOR_FEATURES, "stimulus-feature count")
    require(schema.get("target", {}).get("not_a_feature") is True, "target isolation")
    leakage = schema.get("leakage_contract", {})
    require(leakage.get("post_simulation_fields_in_X") == [], "post-simulation leakage")
    require(leakage.get("target_in_X") is False, "target present in X")
    require(leakage.get("scaler_fit_partition") == "DEV_TRAIN", "scaler fit partition")
    require(leakage.get("site_group_overlap") == 0, "site overlap")
    require(leakage.get("sa0_sa1_group_separations") == 0, "SA0/SA1 separation")
    require(leakage.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(leakage.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")

    require(manifest.get("status") == "PASS", "feature manifest status")
    require(manifest.get("model_feature_count") == EXPECTED_FEATURES, "manifest feature count")
    require(manifest.get("validation_vectors_exposed") == 0, "manifest VALIDATION exposure")
    require(manifest.get("holdout_vectors_exposed") == 0, "manifest HOLDOUT exposure")
    require(manifest.get("post_simulation_fields_in_features") == [], "manifest leakage fields")
    require(audit.get("status") == "PASS", "feature audit status")
    require(audit.get("feature_matrix_status") == "FROZEN", "feature-matrix freeze")
    require(audit.get("schema_status") == "FROZEN", "feature-schema freeze")
    require(audit.get("development_training_authorization") == "DEV_TRAIN_ONLY", "development authorization")
    require(audit.get("validation_campaign_authorized") is False, "VALIDATION authorization")
    require(audit.get("holdout_campaign_authorized") is False, "HOLDOUT authorization")
    require(audit.get("gnn_training_authorized") is False, "GNN authorization")
    require(audit.get("post_simulation_features") == 0, "audit leakage count")
    require(audit.get("target_in_feature_columns") is False, "audit target isolation")

    verify_manifest_artifact(manifest.get("site_feature_catalog"), "site feature catalog")
    verify_manifest_artifact(manifest.get("vector_feature_catalog"), "vector feature catalog")
    verify_manifest_artifact(manifest.get("feature_schema"), "feature schema")
    for partition in PARTITIONS:
        record = manifest.get("partitions", {}).get(partition, {})
        require(record.get("sites") == EXPECTED_SITES[partition], f"{partition} manifest sites")
        require(record.get("samples") == EXPECTED_ROWS[partition], f"{partition} manifest samples")
        require(record.get("positive_samples") == EXPECTED_POSITIVES[partition], f"{partition} manifest positives")
        require(record.get("negative_samples") == EXPECTED_ROWS[partition] - EXPECTED_POSITIVES[partition], f"{partition} manifest negatives")
        verify_manifest_artifact(record.get("matrix"), f"{partition} matrix")

    return schema, manifest, audit


def inspect_matrix(path: Path, partition: str) -> dict:
    expected_rows = EXPECTED_ROWS[partition]
    expected_sites = EXPECTED_SITES[partition]
    expected_positives = EXPECTED_POSITIVES[partition]
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == NPZ_MEMBERS, f"{partition} NPZ member set")
        require(archive["partition"].tolist() == [partition], f"{partition} identity")
        require(
            archive["format_version"].tolist()
            == ["HMAC-FACTORIZED-FEATURE-MATRIX-v1"],
            f"{partition} format version",
        )
        columns = [str(value) for value in archive["feature_columns"].tolist()]
        site_columns = [str(value) for value in archive["site_feature_columns"].tolist()]
        require(len(columns) == EXPECTED_FEATURES, f"{partition} feature width")
        require(len(set(columns)) == EXPECTED_FEATURES, f"{partition} unique features")
        require(len(site_columns) == EXPECTED_SITE_FEATURES, f"{partition} site-feature width")
        require(columns[0] == "stuck_value", f"{partition} first feature")
        require(not (set(columns) & FORBIDDEN_FEATURES), f"{partition} leakage feature")

        one_dimensional = (
            "record_id",
            "site_row",
            "vector_row",
            "stuck_value",
            "target_detected",
        )
        for name in one_dimensional:
            require(archive[name].shape == (expected_rows,), f"{partition}:{name} shape")
        require(archive["site_ids"].shape == (expected_sites,), f"{partition}:site_ids shape")
        require(archive["site_indices"].shape == (expected_sites,), f"{partition}:site_indices shape")
        require(archive["site_features"].shape == (expected_sites, 14), f"{partition}:site_features shape")
        require(archive["vector_ids"].shape == (64,), f"{partition}:vector_ids shape")
        require(archive["vector_features"].shape == (64, 512), f"{partition}:vector_features shape")

        require(np.isfinite(archive["site_features"]).all(), f"{partition} nonfinite site feature")
        require(np.isin(archive["stuck_value"], (0, 1)).all(), f"{partition} stuck values")
        require(np.isin(archive["target_detected"], (0, 1)).all(), f"{partition} targets")
        require(np.isin(archive["vector_features"], (0, 1)).all(), f"{partition} stimulus bits")
        require(int(archive["target_detected"].sum(dtype=np.uint64)) == expected_positives, f"{partition} positives")
        require(int(archive["site_row"].max()) < expected_sites, f"{partition} site lookup range")
        require(int(archive["vector_row"].max()) < 64, f"{partition} vector lookup range")
        require(len(set(archive["site_ids"].tolist())) == expected_sites, f"{partition} unique sites")

        # Reconstruct a deterministic, bounded feature sample without training.
        probe_count = min(4096, expected_rows)
        probe_indices = np.linspace(0, expected_rows - 1, probe_count, dtype=np.int64)
        probe = np.concatenate(
            (
                archive["stuck_value"][probe_indices, None].astype(np.float32),
                archive["site_features"][archive["site_row"][probe_indices]],
                archive["vector_features"][archive["vector_row"][probe_indices]].astype(np.float32),
            ),
            axis=1,
        )
        require(probe.shape == (probe_count, EXPECTED_FEATURES), f"{partition} reconstructed probe")
        require(np.isfinite(probe).all(), f"{partition} reconstructed finite values")

        return {
            "path": relative(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "sites": expected_sites,
            "samples": expected_rows,
            "positive_samples": expected_positives,
            "negative_samples": expected_rows - expected_positives,
            "positive_prevalence": expected_positives / expected_rows,
            "feature_count": len(columns),
            "probe_rows_reconstructed": probe_count,
            "target_in_features": False,
            "post_simulation_features": [],
            "site_ids": set(archive["site_ids"].tolist()),
            "feature_columns": columns,
            "site_feature_columns": site_columns,
            "vector_ids": archive["vector_ids"].astype(int).tolist(),
        }


def candidate_rows() -> list[dict]:
    rows = []
    for weighting in ("NONE", "BALANCED_FROM_DEV_TRAIN"):
        suffix = "RAW" if weighting == "NONE" else "BAL"
        for alpha_text in ("1e-06", "1e-05", "0.0001"):
            alpha = float(alpha_text)
            exponent = int(round(-math.log10(alpha)))
            rows.append(
                {
                    "candidate_id": f"SGD_LOGREG_L2_A1E{exponent}_{suffix}",
                    "model_family": "sklearn.linear_model.SGDClassifier",
                    "loss": "log_loss",
                    "penalty": "l2",
                    "alpha": alpha_text,
                    "sample_weighting": weighting,
                    "epochs": 6,
                    "batch_size": 32_768,
                    "average": 1,
                    "random_seed": RANDOM_SEED,
                }
            )
    rows.sort(key=lambda row: row["candidate_id"])
    require(len(rows) == 6, "candidate count")
    require(len({row["candidate_id"] for row in rows}) == 6, "candidate IDs")
    return rows


def verify_estimator_contract(rows: list[dict]) -> dict:
    signature = inspect.signature(SGDClassifier)
    required_parameters = {
        "loss",
        "penalty",
        "alpha",
        "fit_intercept",
        "max_iter",
        "tol",
        "shuffle",
        "random_state",
        "learning_rate",
        "early_stopping",
        "average",
    }
    require(required_parameters <= set(signature.parameters), "SGDClassifier API contract")
    for row in rows:
        estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(row["alpha"]),
            fit_intercept=True,
            max_iter=1,
            tol=None,
            shuffle=False,
            random_state=RANDOM_SEED,
            learning_rate="optimal",
            early_stopping=False,
            average=True,
        )
        params = estimator.get_params(deep=False)
        require(params["loss"] == "log_loss", "estimator loss")
        require(params["penalty"] == "l2", "estimator penalty")
        require(params["shuffle"] is False, "estimator internal shuffle")
        require(params["early_stopping"] is False, "estimator early stopping")
        require(params["average"] is True, "estimator averaging")
    return {
        "class": "sklearn.linear_model.SGDClassifier",
        "api_parameters_verified": sorted(required_parameters),
        "partial_fit_classes": [0, 1],
        "probability_interface": "predict_proba positive-class column",
    }


def environment_record() -> dict:
    disk = shutil.disk_usage(ROOT)
    memory = physical_memory_bytes()
    return {
        "stage": "11C-5F",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": package_version("threadpoolctl"),
        },
        "resources": {
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": memory,
            "free_disk_bytes": disk.free,
            "minimum_required_free_disk_bytes": 5 * 1024**3,
            "minimum_required_physical_memory_bytes": 4 * 1024**3,
        },
        "training_runtime_contract": {
            "parallel_model_candidates": 1,
            "n_jobs": 1,
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
    }


def training_contract(
    matrix_records: dict[str, dict],
    candidate_grid_record: dict,
    environment_output: dict,
    positive_weight: float,
    negative_weight: float,
) -> dict:
    return {
        "stage": "11C-5F",
        "title": "CONVENTIONAL BASELINE MODEL AND TRAINING-CONTRACT FREEZE",
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": {
            "name": "pre_simulation_persistent_fault_detectability",
            "type": "binary_classification",
            "fault_scope": "persistent net-stem SA0/SA1 only",
            "target": "target_detected",
            "positive_label": 1,
            "negative_label": 0,
        },
        "baseline_role": {
            "name": "CPU-safe transparent conventional baseline",
            "claim": "Baseline comparator for later diagnostic models and GNN; not automatically the final model.",
            "model_family": "incremental linear logistic regression",
            "implementation": "sklearn.linear_model.SGDClassifier.partial_fit",
        },
        "data_access": {
            "DEV_TRAIN": {
                "samples": EXPECTED_ROWS["DEV_TRAIN"],
                "sites": EXPECTED_SITES["DEV_TRAIN"],
                "allowed": ["fit model parameters", "fit class weights", "fit preprocessing already frozen in 11C-5E"],
            },
            "DEV_CALIBRATION": {
                "samples": EXPECTED_ROWS["DEV_CALIBRATION"],
                "sites": EXPECTED_SITES["DEV_CALIBRATION"],
                "allowed": ["candidate comparison", "threshold selection", "probability assessment"],
                "forbidden": ["model coefficient fitting", "feature fitting", "class-weight fitting"],
            },
            "DEV_SITE_TEST": {
                "samples": EXPECTED_ROWS["DEV_SITE_TEST"],
                "sites": EXPECTED_SITES["DEV_SITE_TEST"],
                "state": "LOCKED",
                "open_condition": "Chosen candidate, trained-model SHA, threshold, and calibration-selection record must be atomically frozen first.",
                "forbidden": ["training", "candidate selection", "threshold selection", "retraining after inspection"],
            },
            "VALIDATION": {"vectors_exposed": 0, "state": "NOT AUTHORIZED"},
            "HOLDOUT_TEST": {"vectors_exposed": 0, "state": "NOT AUTHORIZED"},
        },
        "features": {
            "count": EXPECTED_FEATURES,
            "storage": "factorized NPZ",
            "batch_reconstruction": (
                "X = concatenate(stuck_value, site_features[site_row], "
                "vector_features[vector_row]); float32"
            ),
            "target_in_X": False,
            "post_simulation_features": [],
            "feature_selection": "NONE",
            "PCA": "NOT USED BY THIS BASELINE",
        },
        "dummy_references": [
            {
                "id": "DUMMY_MOST_FREQUENT",
                "prediction": 0,
                "purpose": "majority-class lower bound",
            },
            {
                "id": "DUMMY_STRATIFIED_TRAIN_PREVALENCE",
                "positive_probability": EXPECTED_POSITIVES["DEV_TRAIN"] / EXPECTED_ROWS["DEV_TRAIN"],
                "random_seed": RANDOM_SEED,
                "purpose": "deterministic prevalence-matched random lower bound",
            },
        ],
        "candidate_grid": candidate_grid_record,
        "candidate_count": 6,
        "fixed_estimator_parameters": {
            "loss": "log_loss",
            "penalty": "l2",
            "fit_intercept": True,
            "max_iter_per_partial_fit_call": 1,
            "tol": None,
            "shuffle": False,
            "random_state": RANDOM_SEED,
            "learning_rate": "optimal",
            "early_stopping": False,
            "average": True,
            "classes": [0, 1],
        },
        "training_schedule": {
            "epochs": 6,
            "batch_size": 32_768,
            "sample_population": "all DEV_TRAIN samples; no subsampling",
            "candidate_execution": "sequential",
            "external_shuffle": {
                "algorithm": "numpy.random.Generator with PCG64",
                "seed": RANDOM_SEED,
                "rule": "Identical epoch permutation for every candidate.",
            },
            "checkpoint": "after every epoch and candidate",
            "resume": "required and deterministic",
        },
        "sample_weighting": {
            "NONE": {"negative": 1.0, "positive": 1.0},
            "BALANCED_FROM_DEV_TRAIN": {
                "formula": "N / (2 * class_count)",
                "negative": negative_weight,
                "positive": positive_weight,
                "fit_partition": "DEV_TRAIN only",
            },
        },
        "candidate_and_threshold_selection": {
            "partition": "DEV_CALIBRATION",
            "probability_source": "SGDClassifier.predict_proba[:, 1]",
            "threshold_search": "Exact sorted probability boundaries, including all-negative and all-positive endpoints.",
            "primary_maximization": "Matthews correlation coefficient (MCC)",
            "tie_break_order": [
                "higher balanced_accuracy",
                "higher f1_score",
                "higher recall",
                "threshold closest to 0.5",
                "lexicographically smaller candidate_id",
            ],
            "probability_calibration": "NOT APPLIED in the linear baseline; report Brier score and calibration curve as diagnostics.",
            "selection_lock_required_before_site_test": True,
        },
        "metrics": {
            "primary": "matthews_correlation_coefficient",
            "secondary": [
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "pr_auc",
                "roc_auc",
                "brier_score",
                "confusion_matrix",
            ],
            "required_breakdowns": [
                "SA0 versus SA1",
                "sequential versus combinational stem",
                "driver_cell_type when supported",
            ],
            "confidence_intervals": {
                "method": "site-group bootstrap",
                "confidence": 0.95,
                "replicates": 1_000,
                "random_seed": RANDOM_SEED,
            },
        },
        "acceptance": {
            "minimum_validity": {
                "mcc": "> 0",
                "balanced_accuracy": "> 0.50",
                "pr_auc": "> partition positive prevalence",
                "dummy_comparison": "outperform both frozen dummy references",
            },
            "project_target": {
                "mcc": ">= 0.40",
                "balanced_accuracy": ">= 0.70",
                "f1_score": ">= 0.70",
                "recall": ">= 0.70",
            },
            "failure_rule": "Report honestly. Do not inspect HOLDOUT or change the threshold using DEV_SITE_TEST.",
        },
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "single_job": True,
            "deterministic_replay_required": True,
            "exact_prediction_commitment_required": True,
            "model_coefficient_sha256_required": True,
            "checkpoint_resume_equivalence_required": True,
            "environment": environment_output,
        },
        "matrix_inputs": matrix_records,
        "training_execution_authorized": True,
        "authorized_partition": "DEV_TRAIN",
        "dev_calibration_use_authorized": True,
        "dev_site_test_authorized": False,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "gnn_training_authorized": False,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (CONTRACT, CANDIDATE_GRID, ENVIRONMENT, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage 11C-5F output already exists: {path}")
        require(not path.with_name(path.name + ".tmp").exists(), f"stale temporary output: {path}.tmp")

    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11C-5F — CONVENTIONAL BASELINE MODEL AND TRAINING CONTRACT")
    print("FROZEN INPUT VERIFICATION")
    input_evidence = {}
    for path, expected in EXPECTED_INPUTS.items():
        record = verify_file(path, expected)
        input_evidence[record["path"]] = record
        print(f"  {path.name:<66} : OK", flush=True)

    schema, manifest, feature_audit = inspect_feature_contract()

    print("\nFEATURE MATRIX CONTRACT CHECKS")
    matrix_details = {}
    site_sets = {}
    reference_columns = None
    reference_site_columns = None
    reference_vectors = None
    for partition in PARTITIONS:
        details = inspect_matrix(MATRIX_PATHS[partition], partition)
        site_sets[partition] = details.pop("site_ids")
        columns = details.pop("feature_columns")
        site_columns = details.pop("site_feature_columns")
        vector_ids = details.pop("vector_ids")
        if reference_columns is None:
            reference_columns = columns
            reference_site_columns = site_columns
            reference_vectors = vector_ids
        else:
            require(columns == reference_columns, f"{partition} feature-column contract")
            require(site_columns == reference_site_columns, f"{partition} site-feature contract")
            require(vector_ids == reference_vectors, f"{partition} vector contract")
        matrix_details[partition] = details
        print(
            f"  {partition:<16} rows={details['samples']:7d} "
            f"features={details['feature_count']:3d} "
            f"positives={details['positive_samples']:7d} : PASS",
            flush=True,
        )

    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1:]:
            require(not (site_sets[left] & site_sets[right]), f"site overlap: {left}/{right}")
    require(sum(len(value) for value in site_sets.values()) == 22_839, "site partition closure")

    rows = candidate_rows()
    estimator_contract = verify_estimator_contract(rows)

    candidate_temporary = CANDIDATE_GRID.with_name(CANDIDATE_GRID.name + ".tmp")
    with candidate_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    candidate_temporary.replace(CANDIDATE_GRID)
    candidate_grid_record = output_record(CANDIDATE_GRID)

    environment = environment_record()
    resources = environment["resources"]
    if resources["physical_memory_bytes"] is not None:
        require(
            resources["physical_memory_bytes"]
            >= resources["minimum_required_physical_memory_bytes"],
            "less than 4 GiB physical memory available to WSL",
        )
    require(
        resources["free_disk_bytes"] >= resources["minimum_required_free_disk_bytes"],
        "less than 5 GiB free disk",
    )
    environment["estimator_api_contract"] = estimator_contract
    environment["input_evidence"] = input_evidence
    atomic_json(ENVIRONMENT, environment)
    environment_output = output_record(ENVIRONMENT)

    train_positive = EXPECTED_POSITIVES["DEV_TRAIN"]
    train_negative = EXPECTED_ROWS["DEV_TRAIN"] - train_positive
    positive_weight = EXPECTED_ROWS["DEV_TRAIN"] / (2.0 * train_positive)
    negative_weight = EXPECTED_ROWS["DEV_TRAIN"] / (2.0 * train_negative)

    matrix_records = {
        partition: {
            key: value
            for key, value in matrix_details[partition].items()
            if key not in ("probe_rows_reconstructed", "target_in_features", "post_simulation_features")
        }
        for partition in PARTITIONS
    }
    contract = training_contract(
        matrix_records,
        candidate_grid_record,
        environment_output,
        positive_weight,
        negative_weight,
    )
    contract["input_evidence"] = input_evidence
    contract["feature_schema"] = output_record(FEATURE_SCHEMA)
    contract["feature_manifest"] = output_record(FEATURE_MANIFEST)
    contract["feature_audit"] = output_record(FEATURE_AUDIT)
    atomic_json(CONTRACT, contract)
    contract_record = output_record(CONTRACT)

    audit = {
        "stage": "11C-5F",
        "title": "CONVENTIONAL BASELINE MODEL AND TRAINING-CONTRACT FREEZE",
        "status": "PASS",
        "contract_status": "FROZEN",
        "environment_status": "FROZEN",
        "candidate_grid_status": "FROZEN",
        "baseline_model_family": "INCREMENTAL LOGISTIC REGRESSION",
        "trainable_candidates": 6,
        "dummy_references": 2,
        "model_features": EXPECTED_FEATURES,
        "dev_train_samples": EXPECTED_ROWS["DEV_TRAIN"],
        "dev_calibration_samples": EXPECTED_ROWS["DEV_CALIBRATION"],
        "dev_site_test_samples": EXPECTED_ROWS["DEV_SITE_TEST"],
        "site_overlap": 0,
        "target_in_features": False,
        "post_simulation_features": 0,
        "training_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "build_jobs": 1,
        "batch_size": 32_768,
        "epochs_per_candidate": 6,
        "random_seed": RANDOM_SEED,
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "dev_site_test_state": "LOCKED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "training_execution_authorized": True,
        "authorized_training_partition": "DEV_TRAIN",
        "dev_calibration_use_authorized": True,
        "dev_site_test_authorized": False,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "gnn_training_authorized": False,
        "contract": contract_record,
        "candidate_grid": candidate_grid_record,
        "environment": environment_output,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "next_gate": "CONVENTIONAL BASELINE TRAINING AND CALIBRATION EXECUTION",
    }
    atomic_json(AUDIT, audit)
    audit_record = output_record(AUDIT)

    print("\nSTAGE 11C-5F — CONVENTIONAL BASELINE MODEL AND TRAINING-CONTRACT FREEZE")
    print("Status                         : PASS")
    print("Training contract status       : FROZEN")
    print("Environment status             : FROZEN")
    print("Candidate grid status          : FROZEN")
    print("Baseline model                 : INCREMENTAL LOGISTIC REGRESSION")
    print("Trainable candidates           : 6")
    print("Dummy references               : 2")
    print("Model features                 :", EXPECTED_FEATURES)
    print("DEV_TRAIN samples              :", EXPECTED_ROWS["DEV_TRAIN"])
    print("DEV_CALIBRATION samples        :", EXPECTED_ROWS["DEV_CALIBRATION"])
    print("DEV_SITE_TEST samples          :", EXPECTED_ROWS["DEV_SITE_TEST"])
    print("Target present in X            : NO")
    print("Post-simulation features       : 0")
    print("Candidate execution            : SEQUENTIAL")
    print("Parallel candidates            : 1")
    print("Training batch size            : 32768")
    print("Epochs per candidate           : 6")
    print("Selection partition            : DEV_CALIBRATION")
    print("Primary metric                 : MCC")
    print("DEV_SITE_TEST                  : LOCKED")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Baseline training              : AUTHORIZED FOR DEV_TRAIN")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Canonical dataset modified     : NO")
    print("Feature matrices modified      : NO")
    print("Contract                       :", CONTRACT)
    print("Contract SHA                   :", contract_record["sha256"])
    print("Candidate grid                 :", CANDIDATE_GRID)
    print("Candidate grid SHA             :", candidate_grid_record["sha256"])
    print("Environment                    :", ENVIRONMENT)
    print("Environment SHA                :", environment_output["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : CONVENTIONAL BASELINE TRAINING AND CALIBRATION EXECUTION")


if __name__ == "__main__":
    main()
