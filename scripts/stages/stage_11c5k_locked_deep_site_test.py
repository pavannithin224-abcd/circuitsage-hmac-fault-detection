#!/usr/bin/env python3
"""Evaluate the frozen deep MLP once and compare it with the frozen baseline."""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import zipfile
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

try:
    import stage_11c5h_locked_site_test as baseline_helpers
except ImportError as error:
    raise SystemExit(
        "STOP: frozen Stage 11C-5H evaluator is required in the project root."
    ) from error


VERSION = "HMAC-LOCKED-DEEP-SITE-TEST-EVALUATOR-v1"
RANDOM_SEED = 20_260_905
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_CONFIDENCE = 0.95
INFERENCE_BATCH_SIZE = 32_768

EXPECTED_ROWS = 438_528
EXPECTED_SITES = 3_426
EXPECTED_POSITIVES = 191_945
EXPECTED_NEGATIVES = 246_583
EXPECTED_FEATURES = 527
EXPECTED_SAMPLES_PER_SITE = 128
EXPECTED_CANDIDATE = "DEEP_MLP_128_64_32_A1E5_LR3E4"
EXPECTED_THRESHOLD = 0.442
EXPECTED_HIDDEN_LAYERS = (128, 64, 32)
EXPECTED_PARAMETERS = 77_953
EXPECTED_BASELINE_CANDIDATE = "SGD_LOGREG_L2_A1E6_BAL"
EXPECTED_BASELINE_THRESHOLD = 0.98536854982376099
EXPECTED_BASELINE_MCC = 0.14208149
REQUIRED_MCC_IMPROVEMENT = 0.05

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"
DEEP_TRAIN_ROOT = RESULT_ROOT / "deep_training_11c5j"
DEEP_EVAL_ROOT = RESULT_ROOT / "deep_evaluation_11c5k"
BASELINE_EVAL_ROOT = RESULT_ROOT / "baseline_evaluation_11c5h"

SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"

CONTRACT_SOURCE = ROOT / "stage_11c5i_deep_model_contract.py"
ARCHITECTURE = ROOT / "config/diagnostic_model/hmac_deep_diagnostic_architecture_11c5i.json"
TRAINING_CONTRACT = ROOT / "config/diagnostic_model/hmac_deep_diagnostic_training_contract_11c5i.json"
CONTRACT_AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_architecture_training_contract_freeze_11c5i.json"

TRAINER = ROOT / "stage_11c5j_deep_model_train.py"
SELECTED_MODEL = DEEP_TRAIN_ROOT / "hmac_selected_deep_diagnostic_model_11c5j.joblib"
SELECTION_LOCK = DEEP_TRAIN_ROOT / "hmac_deep_diagnostic_selection_lock_11c5j.json"
TRAINING_MANIFEST = RESULT_ROOT / "hmac_deep_diagnostic_training_manifest_11c5j.json"
TRAINING_AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_training_calibration_freeze_11c5j.json"

BASELINE_EVALUATOR = ROOT / "stage_11c5h_locked_site_test.py"
BASELINE_PREDICTIONS = BASELINE_EVAL_ROOT / "hmac_conventional_baseline_site_test_predictions_11c5h.npz"
BASELINE_EVALUATION_LOCK = BASELINE_EVAL_ROOT / "hmac_conventional_baseline_site_test_evaluation_lock_11c5h.json"
BASELINE_MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_site_test_manifest_11c5h.json"
BASELINE_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_site_test_freeze_11c5h.json"

PREDICTIONS = DEEP_EVAL_ROOT / "hmac_deep_diagnostic_site_test_predictions_11c5k.npz"
METRICS = DEEP_EVAL_ROOT / "hmac_deep_diagnostic_site_test_metrics_11c5k.json"
BREAKDOWN_CSV = DEEP_EVAL_ROOT / "hmac_deep_diagnostic_site_test_breakdowns_11c5k.csv"
BREAKDOWN_JSON = DEEP_EVAL_ROOT / "hmac_deep_diagnostic_site_test_breakdowns_11c5k.json"
BOOTSTRAP_CSV = DEEP_EVAL_ROOT / "hmac_deep_baseline_paired_site_bootstrap_11c5k.csv"
BOOTSTRAP_JSON = DEEP_EVAL_ROOT / "hmac_deep_baseline_paired_site_bootstrap_11c5k.json"
BOOTSTRAP_DISTRIBUTION = DEEP_EVAL_ROOT / "hmac_deep_baseline_paired_site_bootstrap_distribution_11c5k.npz"
CALIBRATION_CURVE = DEEP_EVAL_ROOT / "hmac_deep_site_test_calibration_curve_11c5k.csv"
COMPARISON_CSV = DEEP_EVAL_ROOT / "hmac_deep_vs_conventional_baseline_comparison_11c5k.csv"
COMPARISON_JSON = DEEP_EVAL_ROOT / "hmac_deep_vs_conventional_baseline_comparison_11c5k.json"
EVALUATION_LOCK = DEEP_EVAL_ROOT / "hmac_deep_diagnostic_site_test_evaluation_lock_11c5k.json"
MANIFEST = RESULT_ROOT / "hmac_deep_diagnostic_site_test_manifest_11c5k.json"
AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_site_test_comparison_freeze_11c5k.json"

EXPECTED_INPUTS = {
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    CONTRACT_SOURCE: "a58ae217b3f2f1c78627d93650b2e18b75f3708a5e7143842a0a99a7b05dbca6",
    ARCHITECTURE: "fad4bf5caf653f1e7712d16114ca8a9aa85982a992e21e19ee2c09606583ce6f",
    TRAINING_CONTRACT: "e6175b3921737ec78bf88a1d8020b17ab2e37f44137da211dd20414e6dbb0894",
    CONTRACT_AUDIT: "0ce03891c718f62e1bf47746db298a6338582997727eb9d862b6f7b829f6c05c",
    TRAINER: "9101eab7fc704ddd7baa3b8619a95f967eef5bec0f2ee96926eb9608e13c4470",
    SELECTED_MODEL: "6e25e901f7703e22c069ab21b15e1e526db6bfab79f4d205d9e36fea4cafb7fa",
    SELECTION_LOCK: "12319c2d3b067ca04862209512aad9e42b35be97068557d03bc1f1906b4f024d",
    TRAINING_MANIFEST: "48c4fe476843a821a565af70de8d9f30ef72b0bd9dfa48f53add85b8cf191551",
    TRAINING_AUDIT: "d9d10c4781ade4aff544f57012ae1e8218717f13242f003126e90bcb9d5ade84",
    BASELINE_EVALUATOR: "8f720d2f5a2eff1d964507424af502fba9aca60abf09d50fb897d33075d290e3",
    BASELINE_PREDICTIONS: "ec6d1b20e024b9739d1e79d7ef19bf0a5cfcb3292b16ae355c70f4e014248f76",
    BASELINE_EVALUATION_LOCK: "5ff4987169c2b885826db13cf721164b65e009a83290d9aa21da9eeb13b213c4",
    BASELINE_MANIFEST: "cfe75842e7b081230c29d5162ec0292381e574b8679f58118c9a26bf6a7cc1ef",
    BASELINE_AUDIT: "54d86a12e605ce8f2b920c55e87c6ad9a3e30b41df8420730adfff4be4fbee19",
}

COMPARISON_METRICS = (
    "mcc",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "brier_score",
)

BOOTSTRAP_METRICS = (
    "mcc",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "brier_score",
)


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


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    temporary.replace(path)


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {relative(path)}")
    require(path.stat().st_size > 0, f"empty frozen input: {relative(path)}")
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual}",
    )
    print(f"  {path.name:<74}: OK", flush=True)
    return record(path)


def verify_outputs_absent() -> None:
    for path in (
        PREDICTIONS,
        METRICS,
        BREAKDOWN_CSV,
        BREAKDOWN_JSON,
        BOOTSTRAP_CSV,
        BOOTSTRAP_JSON,
        BOOTSTRAP_DISTRIBUTION,
        CALIBRATION_CURVE,
        COMPARISON_CSV,
        COMPARISON_JSON,
        EVALUATION_LOCK,
        MANIFEST,
        AUDIT,
    ):
        require(not path.exists(), f"Stage 11C-5K output already exists: {relative(path)}")


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


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {name}")


def verify_environment() -> dict:
    frozen = load_json(RESULT_ROOT / "hmac_deep_diagnostic_environment_11c5i.json")
    expected = frozen.get("packages")
    actual = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": package_version("threadpoolctl"),
    }
    require(actual == expected, f"Python package environment changed: expected {expected}, actual {actual}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"thread contract {name}")
    return {"status": "PASS", "packages": actual, "threads": 1, "inference_device": "CPU"}


def verify_handoff() -> tuple[dict, dict, dict, dict]:
    contract = load_json(TRAINING_CONTRACT)
    training_lock = load_json(SELECTION_LOCK)
    training_manifest = load_json(TRAINING_MANIFEST)
    training_audit = load_json(TRAINING_AUDIT)
    baseline_lock = load_json(BASELINE_EVALUATION_LOCK)
    baseline_audit = load_json(BASELINE_AUDIT)

    require(contract.get("status") == "PASS", "Stage 11C-5I contract status")
    require(contract.get("selection_partition") == "DEV_CALIBRATION", "deep selection partition")
    require(contract.get("dev_site_test_authorized_during_training") is False, "training site-test lock")
    require(contract.get("architecture_tuning_after_site_test") is False, "architecture tuning lock")
    require(contract.get("retraining_after_site_test") is False, "retraining lock")
    require(contract.get("validation_campaign_authorized") is False, "VALIDATION lock")
    require(contract.get("holdout_campaign_authorized") is False, "HOLDOUT lock")
    require(contract.get("gnn_training_authorized") is False, "GNN lock")
    require(contract.get("hybrid_training_authorized") is False, "hybrid lock")

    require(training_lock.get("status") == "PASS", "deep selection-lock status")
    require(training_lock.get("lock_status") == "FROZEN", "deep selection lock")
    require(training_lock.get("selected_candidate_id") == EXPECTED_CANDIDATE, "deep candidate")
    require(float(training_lock.get("selected_threshold")) == EXPECTED_THRESHOLD, "deep threshold")
    require(training_lock.get("selection_partition") == "DEV_CALIBRATION", "deep lock partition")
    require(training_lock.get("selection_metric") == "MCC", "deep selection metric")
    require(training_lock.get("dev_site_test_opened_for_training_or_selection") is False, "deep site-test exposure")
    require(training_lock.get("validation_vectors_exposed") == 0, "deep VALIDATION exposure")
    require(training_lock.get("holdout_vectors_exposed") == 0, "deep HOLDOUT exposure")
    replay = training_lock.get("deterministic_replay", {})
    require(replay.get("status") == "PASS", "deep replay status")
    require(replay.get("weights_exact") is True, "deep replay weights")
    require(replay.get("probabilities_exact") is True, "deep replay probabilities")
    require(replay.get("threshold_exact") is True, "deep replay threshold")
    require(replay.get("metrics_exact") is True, "deep replay metrics")
    require(training_lock.get("selected_model", {}).get("sha256") == EXPECTED_INPUTS[SELECTED_MODEL], "deep model lock SHA")

    require(training_manifest.get("status") == "PASS", "deep training manifest status")
    require(training_manifest.get("selected_candidate_id") == EXPECTED_CANDIDATE, "deep manifest candidate")
    require(float(training_manifest.get("selected_threshold")) == EXPECTED_THRESHOLD, "deep manifest threshold")
    require(training_manifest.get("minimum_validity_status") == "PASS", "deep minimum validity")
    require(training_manifest.get("deterministic_replay") == "PASS", "deep manifest replay")
    require(training_manifest.get("dev_site_test_opened") is False, "deep manifest site-test lock")

    require(training_audit.get("status") == "PASS", "deep training audit status")
    require(training_audit.get("training_status") == "FROZEN", "deep training freeze")
    require(training_audit.get("model_selection_status") == "FROZEN", "deep selection freeze")
    require(training_audit.get("threshold_status") == "FROZEN", "deep threshold freeze")
    require(training_audit.get("dev_site_test_state") == "AUTHORIZED FOR ONE LOCKED DEEP-MODEL EVALUATION", "deep evaluation authorization")
    require(training_audit.get("dev_site_test_opened_during_training") is False, "deep audit site-test exposure")

    require(baseline_lock.get("status") == "PASS", "baseline evaluation lock status")
    require(baseline_lock.get("lock_status") == "FROZEN", "baseline evaluation lock")
    require(baseline_lock.get("candidate_id") == EXPECTED_BASELINE_CANDIDATE, "baseline candidate")
    require(float(baseline_lock.get("threshold")) == EXPECTED_BASELINE_THRESHOLD, "baseline threshold")
    require(baseline_audit.get("status") == "PASS", "baseline audit status")
    require(baseline_audit.get("baseline_status") == "FROZEN COMPARATOR", "baseline comparator")
    require(abs(float(baseline_audit.get("mcc")) - EXPECTED_BASELINE_MCC) < 1e-8, "baseline MCC")
    require(baseline_audit.get("dev_site_test_state") == "CONSUMED AND FROZEN", "baseline test state")

    deep_outputs = {
        name: resolve_artifact(item, f"deep training output {name}")
        for name, item in sorted(training_manifest.get("outputs", {}).items())
    }
    baseline_outputs = {
        name: resolve_artifact(item, f"baseline evaluation output {name}")
        for name, item in sorted(load_json(BASELINE_MANIFEST).get("outputs", {}).items())
    }
    require(deep_outputs, "deep training outputs")
    require(baseline_outputs, "baseline evaluation outputs")
    return training_lock, training_audit, baseline_audit, {
        "deep_training_outputs": deep_outputs,
        "baseline_evaluation_outputs": baseline_outputs,
    }


def load_model() -> MLPClassifier:
    model = joblib.load(SELECTED_MODEL)
    require(isinstance(model, MLPClassifier), "selected deep estimator type")
    require(tuple(model.hidden_layer_sizes) == EXPECTED_HIDDEN_LAYERS, "selected deep hidden layers")
    require(int(model.n_features_in_) == EXPECTED_FEATURES, "selected deep feature count")
    require(np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])), "selected deep classes")
    parameters = sum(value.size for value in model.coefs_) + sum(value.size for value in model.intercepts_)
    require(parameters == EXPECTED_PARAMETERS, "selected deep parameter count")
    return model


def predict_once(model: MLPClassifier, bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    probability = np.empty(EXPECTED_ROWS, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    batches = 0
    total_batches = math.ceil(EXPECTED_ROWS / INFERENCE_BATCH_SIZE)
    for start in range(0, EXPECTED_ROWS, INFERENCE_BATCH_SIZE):
        stop_index = min(start + INFERENCE_BATCH_SIZE, EXPECTED_ROWS)
        indices = np.arange(start, stop_index, dtype=np.int64)
        matrix = baseline_helpers.feature_batch(bundle, indices)
        probability[start:stop_index] = model.predict_proba(matrix)[:, positive_column]
        batches += 1
        print(f"  inference batch {batches}/{total_batches}", flush=True)
    require(np.isfinite(probability).all(), "finite deep probabilities")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), "deep probability range")
    return probability, batches


def load_baseline_predictions(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(BASELINE_PREDICTIONS, allow_pickle=False) as archive:
        required = {"record_id", "site_row", "vector_row", "stuck_value", "target_detected", "probability", "prediction", "threshold"}
        require(required <= set(archive.files), "baseline prediction members")
        result = {name: np.asarray(archive[name]) for name in required}
    for name in ("record_id", "site_row", "vector_row", "stuck_value", "target_detected", "probability", "prediction"):
        require(result[name].shape == (EXPECTED_ROWS,), f"baseline {name} shape")
    require(result["threshold"].shape == (1,), "baseline threshold shape")
    require(float(result["threshold"][0]) == EXPECTED_BASELINE_THRESHOLD, "baseline prediction threshold")
    require(np.array_equal(result["record_id"], bundle["record_id"]), "baseline record alignment")
    require(np.array_equal(result["site_row"], bundle["site_row"]), "baseline site alignment")
    require(np.array_equal(result["vector_row"], bundle["vector_row"]), "baseline vector alignment")
    require(np.array_equal(result["stuck_value"], bundle["stuck_value"]), "baseline stuck alignment")
    require(np.array_equal(result["target_detected"], bundle["target"]), "baseline target alignment")
    require(np.isfinite(result["probability"]).all(), "baseline finite probabilities")
    require(np.isin(result["prediction"], (0, 1)).all(), "baseline binary predictions")
    return result


def assessment(metrics: dict) -> tuple[str, str, dict]:
    minimum = {
        "mcc_gt_zero": metrics["mcc"] > 0.0,
        "balanced_accuracy_gt_half": metrics["balanced_accuracy"] > 0.5,
        "pr_auc_gt_prevalence": metrics["pr_auc"] > metrics["positive_prevalence"],
    }
    project = {
        "mcc_ge_0_40": metrics["mcc"] >= 0.40,
        "balanced_accuracy_ge_0_70": metrics["balanced_accuracy"] >= 0.70,
        "f1_ge_0_70": metrics["f1_score"] >= 0.70,
        "recall_ge_0_70": metrics["recall"] >= 0.70,
    }
    return (
        "PASS" if all(minimum.values()) else "NOT_MET",
        "PASS" if all(project.values()) else "NOT_MET",
        {"minimum_validity_checks": minimum, "project_target_checks": project},
    )


def verify_baseline_metrics(bundle: dict[str, np.ndarray], baseline: dict[str, np.ndarray], audit: dict) -> dict:
    metrics = baseline_helpers.calculate_metrics(
        bundle["target"],
        baseline["probability"].astype(np.float64),
        baseline["prediction"].astype(np.uint8),
        EXPECTED_BASELINE_THRESHOLD,
    )
    audit_keys = {
        "mcc": "mcc",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "specificity": "specificity",
        "f1_score": "f1_score",
        "pr_auc": "pr_auc",
        "roc_auc": "roc_auc",
        "brier_score": "brier_score",
    }
    for metric_name, audit_name in audit_keys.items():
        require(abs(float(metrics[metric_name]) - float(audit[audit_name])) < 1e-12, f"baseline metric replay {metric_name}")
    return metrics


def site_components(
    site_row: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    confusion = np.stack(
        [
            np.bincount(site_row, weights=(target == 0) & (prediction == 0), minlength=EXPECTED_SITES),
            np.bincount(site_row, weights=(target == 0) & (prediction == 1), minlength=EXPECTED_SITES),
            np.bincount(site_row, weights=(target == 1) & (prediction == 0), minlength=EXPECTED_SITES),
            np.bincount(site_row, weights=(target == 1) & (prediction == 1), minlength=EXPECTED_SITES),
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    squared_error = np.bincount(
        site_row,
        weights=np.square(probability - target, dtype=np.float64),
        minlength=EXPECTED_SITES,
    ).astype(np.float64, copy=False)
    require(np.all(confusion.sum(axis=1) == EXPECTED_SAMPLES_PER_SITE), "site confusion closure")
    return confusion, squared_error


def paired_bootstrap(
    bundle: dict[str, np.ndarray],
    deep_probability: np.ndarray,
    deep_prediction: np.ndarray,
    deep_metrics: dict,
    baseline_probability: np.ndarray,
    baseline_prediction: np.ndarray,
    baseline_metrics: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    deep_confusion, deep_squared = site_components(
        bundle["site_row"], bundle["target"], deep_probability, deep_prediction
    )
    base_confusion, base_squared = site_components(
        bundle["site_row"], bundle["target"], baseline_probability, baseline_prediction
    )
    arrays = {}
    for metric in BOOTSTRAP_METRICS:
        arrays[f"deep_{metric}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        arrays[f"baseline_{metric}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        arrays[f"delta_{metric}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, EXPECTED_SITES, size=EXPECTED_SITES)
        multiplicity = np.bincount(sampled, minlength=EXPECTED_SITES).astype(np.float64)
        deep_hard = baseline_helpers.hard_metrics_from_counts(*(multiplicity @ deep_confusion))
        base_hard = baseline_helpers.hard_metrics_from_counts(*(multiplicity @ base_confusion))
        deep_values = {**deep_hard, "brier_score": float(multiplicity @ deep_squared) / EXPECTED_ROWS}
        base_values = {**base_hard, "brier_score": float(multiplicity @ base_squared) / EXPECTED_ROWS}
        for metric in BOOTSTRAP_METRICS:
            arrays[f"deep_{metric}"][replicate] = deep_values[metric]
            arrays[f"baseline_{metric}"][replicate] = base_values[metric]
            arrays[f"delta_{metric}"][replicate] = deep_values[metric] - base_values[metric]
        if (replicate + 1) % 100 == 0:
            print(f"  paired bootstrap replicate {replicate + 1}/{BOOTSTRAP_REPLICATES}", flush=True)

    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    intervals = []
    for metric in BOOTSTRAP_METRICS:
        for model_name, point in (
            ("deep", deep_metrics[metric]),
            ("baseline", baseline_metrics[metric]),
            ("delta_deep_minus_baseline", deep_metrics[metric] - baseline_metrics[metric]),
        ):
            key = f"delta_{metric}" if model_name.startswith("delta") else f"{model_name}_{metric}"
            lower, upper = np.quantile(arrays[key], [alpha, 1.0 - alpha], method="linear")
            intervals.append(
                {
                    "metric": metric,
                    "estimate": model_name,
                    "point_estimate": float(point),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "confidence": BOOTSTRAP_CONFIDENCE,
                    "replicates": BOOTSTRAP_REPLICATES,
                }
            )
    summary = {
        "stage": "11C-5K",
        "status": "PASS",
        "method": "PAIRED PHYSICAL-SITE-GROUP NONPARAMETRIC PERCENTILE BOOTSTRAP",
        "sampling_unit": "physical fault site",
        "sites_per_replicate": EXPECTED_SITES,
        "records_per_site": EXPECTED_SAMPLES_PER_SITE,
        "random_seed": RANDOM_SEED,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "replicates": BOOTSTRAP_REPLICATES,
        "intervals": intervals,
    }
    return summary, arrays


def write_breakdowns(records: list[dict], unrepresented: list[dict]) -> None:
    atomic_csv(
        BREAKDOWN_CSV,
        baseline_helpers.BREAKDOWN_HEADER,
        [baseline_helpers.breakdown_csv_row(item) for item in records],
    )
    atomic_json(
        BREAKDOWN_JSON,
        {
            "stage": "11C-5K",
            "status": "PASS",
            "partition": "DEV_SITE_TEST",
            "threshold_source": "FROZEN STAGE 11C-5J SELECTION LOCK",
            "threshold": EXPECTED_THRESHOLD,
            "represented_group_count": len(records),
            "unrepresented_group_count": len(unrepresented),
            "unrepresented_groups": unrepresented,
            "groups": records,
        },
    )


def write_bootstrap(summary: dict, arrays: dict[str, np.ndarray]) -> None:
    atomic_json(BOOTSTRAP_JSON, summary)
    atomic_csv(
        BOOTSTRAP_CSV,
        ["metric", "estimate", "point_estimate", "ci_lower", "ci_upper", "confidence", "replicates"],
        summary["intervals"],
    )
    payload = {name: values.astype("<f8", copy=False) for name, values in arrays.items()}
    payload.update(
        {
            "confidence": np.asarray([BOOTSTRAP_CONFIDENCE], dtype="<f8"),
            "random_seed": np.asarray([RANDOM_SEED], dtype="<u8"),
            "replicates": np.asarray([BOOTSTRAP_REPLICATES], dtype="<u4"),
            "site_count": np.asarray([EXPECTED_SITES], dtype="<u4"),
        }
    )
    deterministic_npz(BOOTSTRAP_DISTRIBUTION, payload)


def write_calibration_curve(target: np.ndarray, probability: np.ndarray) -> None:
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, 20)
    rows = []
    for index, indices in enumerate(groups):
        scores = probability[indices]
        labels = target[indices]
        rows.append(
            {
                "bin_index": index,
                "sample_count": len(indices),
                "minimum_probability": float(scores[0]),
                "maximum_probability": float(scores[-1]),
                "mean_predicted_probability": float(scores.mean(dtype=np.float64)),
                "observed_positive_fraction": float(labels.mean(dtype=np.float64)),
            }
        )
    atomic_csv(
        CALIBRATION_CURVE,
        ["bin_index", "sample_count", "minimum_probability", "maximum_probability", "mean_predicted_probability", "observed_positive_fraction"],
        rows,
    )


def comparison_document(
    deep_metrics: dict,
    baseline_metrics: dict,
    bootstrap: dict,
) -> tuple[dict, list[dict]]:
    rows = []
    for metric in COMPARISON_METRICS:
        rows.append(
            {
                "metric": metric,
                "deep_value": deep_metrics[metric],
                "baseline_value": baseline_metrics[metric],
                "deep_minus_baseline": deep_metrics[metric] - baseline_metrics[metric],
                "preferred_direction": "LOWER" if metric == "brier_score" else "HIGHER",
            }
        )
    mcc_delta = deep_metrics["mcc"] - baseline_metrics["mcc"]
    mcc_delta_interval = next(
        item
        for item in bootstrap["intervals"]
        if item["metric"] == "mcc" and item["estimate"] == "delta_deep_minus_baseline"
    )
    point_outcome = (
        "DEEP_OUTPERFORMS_BASELINE"
        if mcc_delta > 0.0
        else "BASELINE_OUTPERFORMS_DEEP"
        if mcc_delta < 0.0
        else "POINT_TIE"
    )
    statistically_supported = mcc_delta_interval["ci_lower"] > 0.0
    return (
        {
            "stage": "11C-5K",
            "status": "PASS",
            "partition": "DEV_SITE_TEST",
            "comparison_status": "FROZEN",
            "primary_metric": "MCC",
            "deep_candidate": EXPECTED_CANDIDATE,
            "baseline_candidate": EXPECTED_BASELINE_CANDIDATE,
            "deep_metrics": deep_metrics,
            "baseline_metrics": baseline_metrics,
            "metric_deltas": {item["metric"]: item["deep_minus_baseline"] for item in rows},
            "point_outcome": point_outcome,
            "mcc_delta": mcc_delta,
            "mcc_delta_ci_95": [mcc_delta_interval["ci_lower"], mcc_delta_interval["ci_upper"]],
            "statistically_supported_mcc_improvement": statistically_supported,
            "required_mcc_improvement": REQUIRED_MCC_IMPROVEMENT,
            "required_mcc_improvement_met": mcc_delta >= REQUIRED_MCC_IMPROVEMENT,
            "decision_rule": (
                "Deep improvement is statistically supported only when the paired "
                "physical-site bootstrap 95% MCC-delta interval is entirely above zero."
            ),
        },
        rows,
    )


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        PREDICTIONS,
        METRICS,
        BREAKDOWN_CSV,
        BREAKDOWN_JSON,
        BOOTSTRAP_CSV,
        BOOTSTRAP_JSON,
        BOOTSTRAP_DISTRIBUTION,
        CALIBRATION_CURVE,
        COMPARISON_CSV,
        COMPARISON_JSON,
        EVALUATION_LOCK,
        MANIFEST,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11C-5K evaluator in project root")
    verify_outputs_absent()

    print("STAGE 11C-5K — LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION AND BASELINE COMPARISON")
    print("FROZEN INPUT VERIFICATION", flush=True)
    input_evidence = {
        relative(path): verify_input(path, expected_sha)
        for path, expected_sha in EXPECTED_INPUTS.items()
    }
    training_lock, training_audit, baseline_audit, handoff = verify_handoff()
    environment = verify_environment()

    print("\nOPENING LOCKED DEV_SITE_TEST")
    print("  Model fitting calls          : 0")
    print("  Candidate-selection calls    : 0")
    print("  Threshold-selection calls    : 0")
    print("  Frozen deep candidate        :", EXPECTED_CANDIDATE)
    print("  Frozen deep threshold        :", format(EXPECTED_THRESHOLD, ".17g"))
    print("  Frozen baseline candidate    :", EXPECTED_BASELINE_CANDIDATE)
    print("  Frozen baseline threshold    :", format(EXPECTED_BASELINE_THRESHOLD, ".17g"))

    bundle = baseline_helpers.load_site_test()
    require(len(bundle["target"]) == EXPECTED_ROWS, "site-test row count")
    require(int(bundle["target"].sum(dtype=np.uint64)) == EXPECTED_POSITIVES, "site-test positives")
    model = load_model()
    baseline = load_baseline_predictions(bundle)

    print("\nLOCKED DEEP MODEL INFERENCE", flush=True)
    probability, inference_batches = predict_once(model, bundle)
    prediction = (probability >= EXPECTED_THRESHOLD).astype(np.uint8)
    deep_metrics = baseline_helpers.calculate_metrics(
        bundle["target"], probability, prediction, EXPECTED_THRESHOLD
    )
    baseline_metrics = verify_baseline_metrics(bundle, baseline, baseline_audit)
    minimum_status, project_status, acceptance = assessment(deep_metrics)

    print("\nREQUIRED DEEP-MODEL BREAKDOWNS")
    breakdowns, unrepresented = baseline_helpers.make_breakdowns(
        bundle, probability, prediction, EXPECTED_THRESHOLD
    )
    for item in breakdowns:
        print(
            f"  {item['group_dimension']:<18} {item['group_value']:<28} "
            f"samples={item['sample_count']:7d} MCC={item['mcc']:.6f}"
        )
    for item in unrepresented:
        print(f"  {item['group_dimension']:<18} {item['group_value']:<28} NOT REPRESENTED")

    print("\nPAIRED PHYSICAL-SITE BOOTSTRAP", flush=True)
    bootstrap_summary, bootstrap_arrays = paired_bootstrap(
        bundle,
        probability,
        prediction,
        deep_metrics,
        baseline["probability"].astype(np.float64),
        baseline["prediction"].astype(np.uint8),
        baseline_metrics,
    )
    comparison, comparison_rows = comparison_document(deep_metrics, baseline_metrics, bootstrap_summary)

    DEEP_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    deterministic_npz(
        PREDICTIONS,
        {
            "candidate_id": np.asarray([EXPECTED_CANDIDATE]),
            "model_sha256": np.asarray([EXPECTED_INPUTS[SELECTED_MODEL]]),
            "partition": np.asarray(["DEV_SITE_TEST"]),
            "prediction": prediction,
            "probability": probability.astype("<f8", copy=False),
            "record_id": bundle["record_id"].astype("<u4", copy=False),
            "site_ids": bundle["site_ids"],
            "site_indices": bundle["site_indices"].astype("<u2", copy=False),
            "site_row": bundle["site_row"].astype("<u2", copy=False),
            "stuck_value": bundle["stuck_value"].astype(np.uint8, copy=False),
            "target_detected": bundle["target"].astype(np.uint8, copy=False),
            "threshold": np.asarray([EXPECTED_THRESHOLD], dtype="<f8"),
            "vector_ids": bundle["vector_ids"].astype("<u2", copy=False),
            "vector_row": bundle["vector_row"].astype(np.uint8, copy=False),
        },
    )
    atomic_json(
        METRICS,
        {
            "stage": "11C-5K",
            "status": "PASS",
            "partition": "DEV_SITE_TEST",
            "candidate": EXPECTED_CANDIDATE,
            "threshold": EXPECTED_THRESHOLD,
            "metrics": deep_metrics,
            "minimum_validity_status": minimum_status,
            "project_target_status": project_status,
            "acceptance_checks": acceptance,
        },
    )
    write_breakdowns(breakdowns, unrepresented)
    write_bootstrap(bootstrap_summary, bootstrap_arrays)
    write_calibration_curve(bundle["target"], probability)
    atomic_csv(
        COMPARISON_CSV,
        ["metric", "deep_value", "baseline_value", "deep_minus_baseline", "preferred_direction"],
        comparison_rows,
    )
    atomic_json(COMPARISON_JSON, comparison)

    outputs = {
        "predictions": record(PREDICTIONS),
        "metrics": record(METRICS),
        "breakdown_csv": record(BREAKDOWN_CSV),
        "breakdown_json": record(BREAKDOWN_JSON),
        "paired_bootstrap_csv": record(BOOTSTRAP_CSV),
        "paired_bootstrap_json": record(BOOTSTRAP_JSON),
        "paired_bootstrap_distribution": record(BOOTSTRAP_DISTRIBUTION),
        "calibration_curve": record(CALIBRATION_CURVE),
        "comparison_csv": record(COMPARISON_CSV),
        "comparison_json": record(COMPARISON_JSON),
    }
    created_at = datetime.now(timezone.utc).isoformat()
    evaluation_lock = {
        "stage": "11C-5K",
        "title": "LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": created_at,
        "candidate_id": EXPECTED_CANDIDATE,
        "model_sha256": EXPECTED_INPUTS[SELECTED_MODEL],
        "threshold": EXPECTED_THRESHOLD,
        "partition": "DEV_SITE_TEST",
        "inference_batches": inference_batches,
        "deep_metrics": deep_metrics,
        "minimum_validity_status": minimum_status,
        "project_target_status": project_status,
        "comparison": comparison,
        "model_fit_calls": 0,
        "candidate_selection_calls": 0,
        "threshold_selection_calls": 0,
        "model_retrained": False,
        "threshold_changed": False,
        "architecture_changed": False,
        "outputs": outputs,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
    }
    atomic_json(EVALUATION_LOCK, evaluation_lock)
    evaluation_lock_record = record(EVALUATION_LOCK)

    manifest = {
        "stage": "11C-5K",
        "title": "LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION AND BASELINE COMPARISON",
        "status": "PASS",
        "evaluator_version": VERSION,
        "evaluator_source": record(SOURCE),
        "created_at_utc": created_at,
        "partition": "DEV_SITE_TEST",
        "physical_sites": EXPECTED_SITES,
        "fault_instances": EXPECTED_SITES * 2,
        "samples": EXPECTED_ROWS,
        "positive_samples": EXPECTED_POSITIVES,
        "negative_samples": EXPECTED_NEGATIVES,
        "deep_candidate": EXPECTED_CANDIDATE,
        "deep_threshold": EXPECTED_THRESHOLD,
        "baseline_candidate": EXPECTED_BASELINE_CANDIDATE,
        "baseline_threshold": EXPECTED_BASELINE_THRESHOLD,
        "deep_metrics": deep_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "minimum_validity_status": minimum_status,
        "project_target_status": project_status,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "input_evidence": input_evidence,
        "handoff_artifacts": handoff,
        "environment": environment,
        "outputs": outputs,
        "evaluation_lock": evaluation_lock_record,
        "model_fit_calls": 0,
        "candidate_selection_calls": 0,
        "threshold_selection_calls": 0,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = record(MANIFEST)

    audit = {
        "stage": "11C-5K",
        "title": "LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION AND BASELINE COMPARISON FREEZE",
        "status": "PASS",
        "evaluation_status": "FROZEN",
        "comparison_status": "FROZEN",
        "deep_model_status": "FROZEN COMPARATOR",
        "baseline_status": "FROZEN COMPARATOR",
        "deep_candidate": EXPECTED_CANDIDATE,
        "deep_threshold": EXPECTED_THRESHOLD,
        "deep_mcc": deep_metrics["mcc"],
        "deep_balanced_accuracy": deep_metrics["balanced_accuracy"],
        "deep_precision": deep_metrics["precision"],
        "deep_recall": deep_metrics["recall"],
        "deep_specificity": deep_metrics["specificity"],
        "deep_f1": deep_metrics["f1_score"],
        "deep_pr_auc": deep_metrics["pr_auc"],
        "deep_roc_auc": deep_metrics["roc_auc"],
        "deep_brier": deep_metrics["brier_score"],
        "baseline_mcc": baseline_metrics["mcc"],
        "mcc_delta": comparison["mcc_delta"],
        "mcc_delta_ci_95": comparison["mcc_delta_ci_95"],
        "point_outcome": comparison["point_outcome"],
        "statistically_supported_mcc_improvement": comparison["statistically_supported_mcc_improvement"],
        "required_mcc_improvement_met": comparison["required_mcc_improvement_met"],
        "minimum_validity_status": minimum_status,
        "project_target_status": project_status,
        "required_breakdowns": "PASS",
        "paired_site_bootstrap": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "model_retrained": False,
        "threshold_changed": False,
        "architecture_changed": False,
        "dev_site_test_state": "CONSUMED AND FROZEN FOR DEEP MODEL",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
        "evaluation_lock": evaluation_lock_record,
        "manifest": manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
        "next_gate": "DIAGNOSTIC BASELINE DISPOSITION AND GNN READINESS FREEZE",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    mcc_deep_interval = next(
        item for item in bootstrap_summary["intervals"]
        if item["metric"] == "mcc" and item["estimate"] == "deep"
    )
    print("\nSTAGE 11C-5K — LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION AND BASELINE COMPARISON FREEZE")
    print("Status                         : PASS")
    print("Evaluation status              : FROZEN")
    print("Comparison status              : FROZEN")
    print("Deep candidate                 :", EXPECTED_CANDIDATE)
    print("Deep threshold                 :", format(EXPECTED_THRESHOLD, ".17g"))
    print("DEV_SITE_TEST sites            :", EXPECTED_SITES)
    print("DEV_SITE_TEST samples          :", EXPECTED_ROWS)
    print("Deep MCC                       :", f"{deep_metrics['mcc']:.8f}")
    print("Deep MCC 95% site CI           :", f"[{mcc_deep_interval['ci_lower']:.8f}, {mcc_deep_interval['ci_upper']:.8f}]")
    print("Deep balanced accuracy         :", f"{deep_metrics['balanced_accuracy']:.8f}")
    print("Deep precision                 :", f"{deep_metrics['precision']:.8f}")
    print("Deep recall                    :", f"{deep_metrics['recall']:.8f}")
    print("Deep specificity               :", f"{deep_metrics['specificity']:.8f}")
    print("Deep F1                        :", f"{deep_metrics['f1_score']:.8f}")
    print("Deep PR-AUC                    :", f"{deep_metrics['pr_auc']:.8f}")
    print("Deep ROC-AUC                   :", f"{deep_metrics['roc_auc']:.8f}")
    print("Deep Brier                     :", f"{deep_metrics['brier_score']:.8f}")
    print("Baseline MCC                   :", f"{baseline_metrics['mcc']:.8f}")
    print("MCC delta (deep-baseline)      :", f"{comparison['mcc_delta']:.8f}")
    print("MCC delta 95% paired-site CI   :", f"[{comparison['mcc_delta_ci_95'][0]:.8f}, {comparison['mcc_delta_ci_95'][1]:.8f}]")
    print("Point comparison               :", comparison["point_outcome"])
    print("Statistical improvement        :", "SUPPORTED" if comparison["statistically_supported_mcc_improvement"] else "NOT_SUPPORTED")
    print("Required +0.05 MCC improvement :", "MET" if comparison["required_mcc_improvement_met"] else "NOT_MET")
    print("Minimum validity               :", minimum_status)
    print("Project target                 :", project_status)
    print("Model fitting calls            : 0")
    print("Threshold changed              : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Deep model modified            : NO")
    print("Baseline model modified        : NO")
    print("Comparison                     :", COMPARISON_JSON)
    print("Comparison SHA                 :", outputs["comparison_json"]["sha256"])
    print("Evaluation lock                :", EVALUATION_LOCK)
    print("Evaluation lock SHA            :", evaluation_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : DIAGNOSTIC BASELINE DISPOSITION AND GNN READINESS FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
