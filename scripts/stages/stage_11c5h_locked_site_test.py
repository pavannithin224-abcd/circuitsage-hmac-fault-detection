#!/usr/bin/env python3
"""Evaluate the frozen conventional HMAC baseline on DEV_SITE_TEST once."""
from __future__ import annotations

import os

# Preserve the single-worker numerical environment frozen in Stage 11C-5F.
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
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error


VERSION = "HMAC-LOCKED-SITE-TEST-EVALUATOR-v1-R1"
RANDOM_SEED = 20_260_903
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_CONFIDENCE = 0.95
INFERENCE_BATCH_SIZE = 32_768

EXPECTED_ROWS = 438_528
EXPECTED_SITES = 3_426
EXPECTED_POSITIVES = 191_945
EXPECTED_NEGATIVES = EXPECTED_ROWS - EXPECTED_POSITIVES
EXPECTED_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512
EXPECTED_VECTORS = 64
EXPECTED_SAMPLES_PER_SITE = 128
EXPECTED_CANDIDATE = "SGD_LOGREG_L2_A1E6_BAL"
EXPECTED_THRESHOLD = 0.98536854982376099

ROOT = Path.cwd().resolve()
EVALUATOR_SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"
TRAINING_ROOT = RESULT_ROOT / "baseline_training_11c5g"
WORK_ROOT = RESULT_ROOT / "baseline_evaluation_11c5h"
CONFIG_ROOT = ROOT / "config/diagnostic_model"

SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
TRAINING_CONTRACT = CONFIG_ROOT / "hmac_conventional_baseline_training_contract_11c5f.json"
ENVIRONMENT = RESULT_ROOT / "hmac_conventional_baseline_environment_11c5f.json"

TRAINER = ROOT / "stage_11c5g_baseline_train.py"
TRAINING_LOG = RESULT_ROOT / "stage_11c5g_baseline_training_calibration.log"
TRAINING_RESOURCE_LOG = RESULT_ROOT / "stage_11c5g_baseline_training_calibration_resources.log"
SELECTION_LOCK = TRAINING_ROOT / "hmac_conventional_baseline_selection_lock_11c5g.json"
TRAINING_MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_training_manifest_11c5g.json"
TRAINING_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_training_calibration_freeze_11c5g.json"
SELECTED_MODEL = TRAINING_ROOT / "hmac_selected_conventional_baseline_11c5g.joblib"
CALIBRATION_CURVE = TRAINING_ROOT / "hmac_selected_calibration_curve_11c5g.csv"
CANDIDATE_METRICS = TRAINING_ROOT / "hmac_baseline_candidate_calibration_metrics_11c5g.csv"

PREDICTIONS = WORK_ROOT / "hmac_conventional_baseline_site_test_predictions_11c5h.npz"
OVERALL_METRICS = WORK_ROOT / "hmac_conventional_baseline_site_test_metrics_11c5h.json"
BREAKDOWN_CSV = WORK_ROOT / "hmac_conventional_baseline_site_test_breakdowns_11c5h.csv"
BREAKDOWN_JSON = WORK_ROOT / "hmac_conventional_baseline_site_test_breakdowns_11c5h.json"
BOOTSTRAP_CSV = WORK_ROOT / "hmac_conventional_baseline_site_bootstrap_ci_11c5h.csv"
BOOTSTRAP_JSON = WORK_ROOT / "hmac_conventional_baseline_site_bootstrap_ci_11c5h.json"
BOOTSTRAP_DISTRIBUTION = WORK_ROOT / "hmac_conventional_baseline_site_bootstrap_distribution_11c5h.npz"
SITE_TEST_CURVE = WORK_ROOT / "hmac_conventional_baseline_site_test_calibration_curve_11c5h.csv"
EVALUATION_LOCK = WORK_ROOT / "hmac_conventional_baseline_site_test_evaluation_lock_11c5h.json"
MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_site_test_manifest_11c5h.json"
AUDIT = RESULT_ROOT / "hmac_conventional_baseline_site_test_freeze_11c5h.json"

EXPECTED_INPUTS = {
    TRAINER:
        "baae204689f803bf175809dd526d23bbc40e5ec1fa0eee492d9b8015869f9870",
    TRAINING_LOG:
        "2aef29094f76cb11831f2ac71a0ca0d473b8b041526f7f70c0f0faf0a91e5597",
    TRAINING_RESOURCE_LOG:
        "a7b796b210ca5b778a2e02b49000ffb379609d507197d70601b6ec7948e22e5e",
    SELECTION_LOCK:
        "1532a30fb96a5f142ba05a778643bdb9aa93c058eba568e3df92c09f47e523fc",
    TRAINING_MANIFEST:
        "9c612c42d3ba97c47372a27927fc31ff374fa75e7f19577bbda458061166ada7",
    TRAINING_AUDIT:
        "f9f77bdc5c07cec91ca180f3d8b2cfb93c744e029445b6a226e94b5add8df5f1",
    SELECTED_MODEL:
        "8220ae569958e3d072b1ac40ac3c2958c7428419fb0ebb4acab8f324eaea5833",
    CALIBRATION_CURVE:
        "08a072249a9772456ca16156e646e05c300c779e2f891230aee0fc6c25e96b34",
    CANDIDATE_METRICS:
        "75a745a0608de9441fca0531ba13a14020fcbb8629cff01803c34246b3cd7098",
    SITE_TEST_MATRIX:
        "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA:
        "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    TRAINING_CONTRACT:
        "610211065fd81538183472377b0a69aab2f4967cc4f99ada6393712fc090c007",
    ENVIRONMENT:
        "9e42cceccff6c016f9af5c237a30f400c16ed626d5681159c8348357f7db8e3f",
}

EXPECTED_DRIVER_TYPES = (
    "$_AND_",
    "$_DFFE_PN0P_",
    "$_DFFE_PN1P_",
    "$_DFFE_PP_",
    "$_DFF_PN0_",
    "$_MUX_",
    "$_NOT_",
    "$_OR_",
    "$_XOR_",
)

BREAKDOWN_HEADER = [
    "group_dimension",
    "group_value",
    "site_count",
    "sample_count",
    "positive_samples",
    "negative_samples",
    "positive_prevalence",
    "threshold",
    "mcc",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "brier_score",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]

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


def output_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_file(path: Path, expected_sha256: str) -> dict:
    require(path.is_file(), f"missing frozen input: {path}")
    require(path.stat().st_size > 0, f"empty frozen input: {path}")
    actual = sha256(path)
    require(
        actual == expected_sha256,
        f"SHA mismatch for {relative(path)}: expected {expected_sha256}, actual {actual}",
    )
    return output_record(path)


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
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
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


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {distribution}")


def verify_environment() -> dict:
    frozen = load_json(ENVIRONMENT)
    require(frozen.get("status") == "PASS", "frozen environment status")
    expected = frozen.get("packages", {})
    actual = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": package_version("threadpoolctl"),
    }
    require(
        actual == expected,
        f"Python package environment changed: expected {expected}, actual {actual}",
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        require(os.environ.get(name) == "1", f"thread contract {name}")
    return {"frozen": output_record(ENVIRONMENT), "packages": actual, "threads": 1}


def resolve_record(record: dict, label: str) -> dict:
    require(isinstance(record, dict), f"{label} record")
    record_path = record.get("path")
    record_sha = record.get("sha256")
    require(isinstance(record_path, str) and record_path, f"{label} path")
    require(isinstance(record_sha, str) and len(record_sha) == 64, f"{label} SHA")
    path = (ROOT / record_path).resolve()
    relative(path)
    require(path.is_file() and path.stat().st_size > 0, f"{label} artifact missing")
    require(sha256(path) == record_sha, f"{label} artifact SHA")
    if "bytes" in record:
        require(path.stat().st_size == int(record["bytes"]), f"{label} artifact size")
    return output_record(path)


def verify_frozen_handoff() -> tuple[dict, dict, dict, dict]:
    contract = load_json(TRAINING_CONTRACT)
    lock = load_json(SELECTION_LOCK)
    manifest = load_json(TRAINING_MANIFEST)
    audit = load_json(TRAINING_AUDIT)

    require(contract.get("status") == "PASS", "Stage 11C-5F contract status")
    require(contract.get("dev_site_test_authorized") is False, "pre-selection site-test lock")
    require(contract.get("validation_campaign_authorized") is False, "VALIDATION lock")
    require(contract.get("holdout_campaign_authorized") is False, "HOLDOUT lock")
    require(contract.get("gnn_training_authorized") is False, "GNN lock")

    require(lock.get("status") == "PASS", "selection-lock status")
    require(lock.get("lock_status") == "FROZEN", "selection-lock freeze")
    require(lock.get("selected_candidate_id") == EXPECTED_CANDIDATE, "selected candidate")
    require(
        float(lock.get("selected_threshold")) == EXPECTED_THRESHOLD,
        "selected threshold changed",
    )
    require(lock.get("selection_partition") == "DEV_CALIBRATION", "selection partition")
    require(lock.get("selection_metric") == "MCC", "selection metric")
    require(lock.get("dev_site_test_opened_for_evaluation") is False, "site test already opened")
    require(lock.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(lock.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")
    require(lock.get("retraining_after_site_test_authorized") is False, "retraining lock")
    replay = lock.get("deterministic_replay", {})
    require(replay.get("status") == "PASS", "training replay status")
    require(replay.get("coefficients_exact") is True, "coefficient replay")
    require(replay.get("probabilities_exact") is True, "probability replay")
    require(replay.get("predictions_exact") is True, "prediction replay")

    locked_artifacts = {
        "selected_model": resolve_record(lock.get("selected_model"), "selected model"),
        "selected_weights": resolve_record(lock.get("selected_weights"), "selected weights"),
        "selected_calibration_predictions": resolve_record(
            lock.get("selected_calibration_predictions"),
            "selected calibration predictions",
        ),
        "selected_calibration_curve": resolve_record(
            lock.get("selected_calibration_curve"),
            "selected calibration curve",
        ),
        "candidate_metrics_csv": resolve_record(
            lock.get("candidate_metrics_csv"),
            "candidate metrics CSV",
        ),
        "candidate_metrics_json": resolve_record(
            lock.get("candidate_metrics_json"),
            "candidate metrics JSON",
        ),
        "dummy_metrics": resolve_record(lock.get("dummy_metrics"), "dummy metrics"),
    }
    replay_records = replay
    for key, label in (
        ("replay_model", "replay model"),
        ("replay_weights", "replay weights"),
        ("replay_predictions", "replay predictions"),
    ):
        locked_artifacts[key] = resolve_record(replay_records.get(key), label)

    require(manifest.get("status") == "PASS", "training manifest status")
    require(manifest.get("selected_candidate_id") == EXPECTED_CANDIDATE, "manifest candidate")
    require(float(manifest.get("selected_threshold")) == EXPECTED_THRESHOLD, "manifest threshold")
    require(manifest.get("deterministic_replay") == "PASS", "manifest replay")
    require(manifest.get("dev_site_test_opened_for_evaluation") is False, "manifest site-test lock")

    require(audit.get("status") == "PASS", "training audit status")
    require(audit.get("execution_status") == "FROZEN", "training execution freeze")
    require(audit.get("model_selection_status") == "FROZEN", "model selection freeze")
    require(audit.get("threshold_status") == "FROZEN", "threshold freeze")
    require(audit.get("selected_candidate_id") == EXPECTED_CANDIDATE, "audit candidate")
    require(float(audit.get("selected_threshold")) == EXPECTED_THRESHOLD, "audit threshold")
    require(audit.get("dev_site_test_opened_for_evaluation") is False, "audit site-test lock")
    require(audit.get("dev_site_test_state") == "AUTHORIZED FOR ONE LOCKED EVALUATION", "site-test authorization")
    require(audit.get("validation_vectors_exposed") == 0, "audit VALIDATION exposure")
    require(audit.get("holdout_vectors_exposed") == 0, "audit HOLDOUT exposure")
    require(audit.get("gnn_training_authorized") is False, "audit GNN lock")
    return contract, lock, manifest, {"audit": audit, "artifacts": locked_artifacts}


def load_site_test() -> dict[str, np.ndarray]:
    with np.load(SITE_TEST_MATRIX, allow_pickle=False) as archive:
        required = {
            "partition",
            "site_features",
            "vector_features",
            "site_row",
            "vector_row",
            "stuck_value",
            "target_detected",
            "record_id",
            "site_ids",
            "site_indices",
            "vector_ids",
            "feature_columns",
            "site_feature_columns",
        }
        require(required <= set(archive.files), "DEV_SITE_TEST matrix members")
        require(archive["partition"].tolist() == ["DEV_SITE_TEST"], "partition identity")
        result = {
            "site_features": np.asarray(archive["site_features"], dtype=np.float32),
            "vector_features": np.asarray(archive["vector_features"], dtype=np.float32),
            "site_row": np.asarray(archive["site_row"], dtype=np.uint16),
            "vector_row": np.asarray(archive["vector_row"], dtype=np.uint8),
            "stuck_value": np.asarray(archive["stuck_value"], dtype=np.uint8),
            "target": np.asarray(archive["target_detected"], dtype=np.uint8),
            "record_id": np.asarray(archive["record_id"], dtype=np.uint32),
            "site_ids": np.asarray(archive["site_ids"]),
            "site_indices": np.asarray(archive["site_indices"], dtype=np.uint16),
            "vector_ids": np.asarray(archive["vector_ids"], dtype=np.uint16),
            "feature_columns": np.asarray(archive["feature_columns"]),
            "site_feature_columns": np.asarray(archive["site_feature_columns"]),
        }

    require(result["site_features"].shape == (EXPECTED_SITES, EXPECTED_SITE_FEATURES), "site feature shape")
    require(result["vector_features"].shape == (EXPECTED_VECTORS, EXPECTED_VECTOR_FEATURES), "vector feature shape")
    require(result["feature_columns"].shape == (EXPECTED_FEATURES,), "feature-column shape")
    require(result["site_feature_columns"].shape == (EXPECTED_SITE_FEATURES,), "site-column shape")
    require(result["site_ids"].shape == (EXPECTED_SITES,), "site-ID shape")
    require(result["site_indices"].shape == (EXPECTED_SITES,), "site-index shape")
    require(result["vector_ids"].shape == (EXPECTED_VECTORS,), "vector-ID shape")
    for name in ("site_row", "vector_row", "stuck_value", "target", "record_id"):
        require(result[name].shape == (EXPECTED_ROWS,), f"{name} row count")
    require(np.isfinite(result["site_features"]).all(), "finite site features")
    require(np.isfinite(result["vector_features"]).all(), "finite vector features")
    require(
        np.all((result["vector_features"] == 0.0) | (result["vector_features"] == 1.0)),
        "binary vector features",
    )
    require(np.isin(result["target"], (0, 1)).all(), "binary target")
    require(np.isin(result["stuck_value"], (0, 1)).all(), "binary stuck value")
    require(int(result["target"].sum(dtype=np.uint64)) == EXPECTED_POSITIVES, "positive count")
    require(int((result["target"] == 0).sum()) == EXPECTED_NEGATIVES, "negative count")
    require(int(result["site_row"].max()) == EXPECTED_SITES - 1, "site-row range")
    require(int(result["vector_row"].max()) == EXPECTED_VECTORS - 1, "vector-row range")
    site_counts = np.bincount(result["site_row"], minlength=EXPECTED_SITES)
    require(np.all(site_counts == EXPECTED_SAMPLES_PER_SITE), "samples per physical site")
    sample_code = (
        result["site_row"].astype(np.uint64) * (2 * EXPECTED_VECTORS)
        + result["stuck_value"].astype(np.uint64) * EXPECTED_VECTORS
        + result["vector_row"].astype(np.uint64)
    )
    require(len(np.unique(sample_code)) == EXPECTED_ROWS, "site/SA/vector sample uniqueness")
    require(result["feature_columns"][0] == "stuck_value", "feature 0 contract")
    forbidden = {
        "detected",
        "target_detected",
        "activity",
        "cycles",
        "baseline_cycles",
        "latency_delta",
        "timed_out",
        "unknown",
        "digest_hamming_distance",
        "expected_digest",
        "actual_digest",
    }
    require(
        not (set(result["feature_columns"].tolist()) & forbidden),
        "post-simulation leakage field in model features",
    )
    require(
        result["feature_columns"][1:15].tolist()
        == result["site_feature_columns"].tolist(),
        "site-feature column alignment",
    )
    return result


def feature_batch(bundle: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    matrix = np.empty((len(indices), EXPECTED_FEATURES), dtype=np.float32)
    matrix[:, 0] = bundle["stuck_value"][indices]
    matrix[:, 1:15] = bundle["site_features"][bundle["site_row"][indices]]
    matrix[:, 15:] = bundle["vector_features"][bundle["vector_row"][indices]]
    require(np.isfinite(matrix).all(), "finite materialized features")
    return matrix


def load_locked_model(lock: dict) -> SGDClassifier:
    model = joblib.load(SELECTED_MODEL)
    require(isinstance(model, SGDClassifier), "selected estimator type")
    require(np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])), "model classes")
    require(int(model.n_features_in_) == EXPECTED_FEATURES, "model feature count")
    require(np.asarray(model.coef_).shape == (1, EXPECTED_FEATURES), "model coefficient shape")
    require(np.asarray(model.intercept_).shape == (1,), "model intercept shape")
    require(lock["selected_model"]["sha256"] == sha256(SELECTED_MODEL), "lock/model SHA")
    return model


def predict_once(model: SGDClassifier, bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    probabilities = np.empty(EXPECTED_ROWS, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    batches = 0
    for start in range(0, EXPECTED_ROWS, INFERENCE_BATCH_SIZE):
        stop_index = min(start + INFERENCE_BATCH_SIZE, EXPECTED_ROWS)
        indices = np.arange(start, stop_index, dtype=np.int64)
        matrix = feature_batch(bundle, indices)
        probabilities[start:stop_index] = model.predict_proba(matrix)[:, positive_column]
        batches += 1
        print(
            f"  inference batch {batches}/{math.ceil(EXPECTED_ROWS / INFERENCE_BATCH_SIZE)}",
            flush=True,
        )
    require(np.isfinite(probabilities).all(), "finite probabilities")
    require(((0.0 <= probabilities) & (probabilities <= 1.0)).all(), "probability range")
    return probabilities, batches


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def hard_metrics_from_counts(tn: float, fp: float, fn: float, tp: float) -> dict:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_ratio(tp * tn - fp * fn, denominator)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    precision = safe_ratio(tp, tp + fp)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "mcc": mcc,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
    }


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict:
    require(len(y_true) > 0, "empty metric population")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = hard_metrics_from_counts(float(tn), float(fp), float(fn), float(tp))
    positive_count = int(y_true.sum(dtype=np.uint64))
    negative_count = len(y_true) - positive_count
    metrics.update(
        {
            "threshold": float(threshold),
            "sample_count": int(len(y_true)),
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "positive_prevalence": positive_count / len(y_true),
            "pr_auc": (
                float(average_precision_score(y_true, probability))
                if positive_count and negative_count
                else None
            ),
            "roc_auc": (
                float(roc_auc_score(y_true, probability))
                if positive_count and negative_count
                else None
            ),
            "brier_score": float(brier_score_loss(y_true, probability)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "predicted_positive": int(prediction.sum(dtype=np.uint64)),
            "predicted_negative": int(len(prediction) - prediction.sum(dtype=np.uint64)),
        }
    )
    return metrics


def dummy_references(y_true: np.ndarray) -> list[dict]:
    train_prevalence = 894_000 / 2_046_336
    constant_score = np.full(len(y_true), train_prevalence, dtype=np.float64)
    majority_prediction = np.zeros(len(y_true), dtype=np.uint8)
    majority = calculate_metrics(
        y_true,
        constant_score,
        majority_prediction,
        threshold=0.5,
    )
    majority["dummy_id"] = "DUMMY_MOST_FREQUENT"

    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    stratified_prediction = (rng.random(len(y_true)) < train_prevalence).astype(np.uint8)
    stratified = calculate_metrics(
        y_true,
        constant_score,
        stratified_prediction,
        threshold=train_prevalence,
    )
    stratified["dummy_id"] = "DUMMY_STRATIFIED_TRAIN_PREVALENCE"
    return [majority, stratified]


def assessment(metrics: dict, dummy_results: list[dict]) -> tuple[str, str, dict]:
    best_dummy_mcc = max(item["mcc"] for item in dummy_results)
    minimum_checks = {
        "mcc_gt_zero": metrics["mcc"] > 0.0,
        "balanced_accuracy_gt_half": metrics["balanced_accuracy"] > 0.5,
        "pr_auc_gt_site_test_prevalence": metrics["pr_auc"] > metrics["positive_prevalence"],
        "mcc_gt_both_dummy_references": metrics["mcc"] > best_dummy_mcc,
    }
    target_checks = {
        "mcc_ge_0_40": metrics["mcc"] >= 0.40,
        "balanced_accuracy_ge_0_70": metrics["balanced_accuracy"] >= 0.70,
        "f1_ge_0_70": metrics["f1_score"] >= 0.70,
        "recall_ge_0_70": metrics["recall"] >= 0.70,
    }
    checks = {
        "minimum_validity_checks": minimum_checks,
        "project_target_checks": target_checks,
    }
    return (
        "PASS" if all(minimum_checks.values()) else "NOT_MET",
        "PASS" if all(target_checks.values()) else "NOT_MET",
        checks,
    )


def site_attributes(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    columns = bundle["site_feature_columns"].tolist()
    require("is_sequential_stem" in columns, "sequential site feature")
    sequential_index = columns.index("is_sequential_stem")
    sequential = np.rint(bundle["site_features"][:, sequential_index]).astype(np.uint8)
    require(np.isin(sequential, (0, 1)).all(), "sequential feature values")

    driver_columns = [
        (index, name.split("::", 1)[1])
        for index, name in enumerate(columns)
        if name.startswith("driver_cell_type::")
    ]
    require(tuple(name for _, name in driver_columns) == EXPECTED_DRIVER_TYPES, "driver vocabulary")
    driver_matrix = bundle["site_features"][:, [index for index, _ in driver_columns]]
    require(np.all((driver_matrix == 0.0) | (driver_matrix == 1.0)), "driver one-hot values")
    require(np.all(driver_matrix.sum(axis=1) == 1.0), "driver one-hot closure")
    driver_index = np.argmax(driver_matrix, axis=1)
    driver = np.asarray([EXPECTED_DRIVER_TYPES[index] for index in driver_index])
    return sequential, driver


def make_breakdowns(
    bundle: dict[str, np.ndarray],
    probability: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    sequential_by_site, driver_by_site = site_attributes(bundle)
    sample_sequential = sequential_by_site[bundle["site_row"]]
    sample_driver = driver_by_site[bundle["site_row"]]
    groups = [
        ("fault_model", "SA0", bundle["stuck_value"] == 0),
        ("fault_model", "SA1", bundle["stuck_value"] == 1),
        ("site_category", "COMBINATIONAL_LOGIC_STEM", sample_sequential == 0),
        ("site_category", "SEQUENTIAL_STATE_STEM", sample_sequential == 1),
    ]
    records = []
    for dimension, value, mask in groups:
        require(bool(mask.any()), f"empty breakdown {dimension}/{value}")
        metrics = calculate_metrics(
            bundle["target"][mask],
            probability[mask],
            prediction[mask],
            threshold,
        )
        records.append(
            {
                "group_dimension": dimension,
                "group_value": value,
                "site_count": int(np.unique(bundle["site_row"][mask]).size),
                **metrics,
            }
        )

    unrepresented = []
    driver_records = []
    for driver_type in EXPECTED_DRIVER_TYPES:
        mask = sample_driver == driver_type
        if not bool(mask.any()):
            unrepresented.append(
                {
                    "group_dimension": "driver_cell_type",
                    "group_value": driver_type,
                    "status": "NOT_REPRESENTED_IN_DEV_SITE_TEST",
                    "site_count": 0,
                    "sample_count": 0,
                    "disposition": "NO METRIC COMPUTED; BREAKDOWN REQUIRED WHEN SUPPORTED",
                }
            )
            continue
        metrics = calculate_metrics(
            bundle["target"][mask],
            probability[mask],
            prediction[mask],
            threshold,
        )
        driver_records.append(
            {
                "group_dimension": "driver_cell_type",
                "group_value": driver_type,
                "site_count": int(np.unique(bundle["site_row"][mask]).size),
                **metrics,
            }
        )

    require(driver_records, "no represented driver-cell breakdowns")
    require(
        sum(record["site_count"] for record in driver_records) == EXPECTED_SITES,
        "driver-cell site closure",
    )
    require(
        sum(record["sample_count"] for record in driver_records) == EXPECTED_ROWS,
        "driver-cell sample closure",
    )
    records.extend(driver_records)
    require(len(records) + len(unrepresented) == 13, "breakdown vocabulary closure")
    return records, unrepresented


def format_optional(value) -> str:
    return "" if value is None else format(float(value), ".17g")


def breakdown_csv_row(record: dict) -> dict:
    confusion = record["confusion_matrix"]
    row = {
        "group_dimension": record["group_dimension"],
        "group_value": record["group_value"],
        "site_count": record["site_count"],
        "sample_count": record["sample_count"],
        "positive_samples": record["positive_samples"],
        "negative_samples": record["negative_samples"],
        "positive_prevalence": format_optional(record["positive_prevalence"]),
        "threshold": format_optional(record["threshold"]),
        "mcc": format_optional(record["mcc"]),
        "balanced_accuracy": format_optional(record["balanced_accuracy"]),
        "precision": format_optional(record["precision"]),
        "recall": format_optional(record["recall"]),
        "specificity": format_optional(record["specificity"]),
        "f1_score": format_optional(record["f1_score"]),
        "pr_auc": format_optional(record["pr_auc"]),
        "roc_auc": format_optional(record["roc_auc"]),
        "brier_score": format_optional(record["brier_score"]),
        "true_negative": confusion["true_negative"],
        "false_positive": confusion["false_positive"],
        "false_negative": confusion["false_negative"],
        "true_positive": confusion["true_positive"],
    }
    return row


def write_breakdowns(records: list[dict], unrepresented: list[dict]) -> None:
    temporary = BREAKDOWN_CSV.with_name(BREAKDOWN_CSV.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BREAKDOWN_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(breakdown_csv_row(record) for record in records)
    temporary.replace(BREAKDOWN_CSV)
    atomic_json(
        BREAKDOWN_JSON,
        {
            "stage": "11C-5H",
            "status": "PASS",
            "partition": "DEV_SITE_TEST",
            "threshold_source": "FROZEN STAGE 11C-5G SELECTION LOCK",
            "threshold": EXPECTED_THRESHOLD,
            "required_breakdowns": [
                "SA0 versus SA1",
                "sequential versus combinational stem",
                "driver_cell_type when represented in DEV_SITE_TEST",
            ],
            "represented_group_count": len(records),
            "unrepresented_group_count": len(unrepresented),
            "unrepresented_groups": unrepresented,
            "groups": records,
        },
    )


def site_bootstrap(
    bundle: dict[str, np.ndarray],
    probability: np.ndarray,
    prediction: np.ndarray,
    point_metrics: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    site_row = bundle["site_row"]
    target = bundle["target"]
    confusion_by_site = np.stack(
        [
            np.bincount(site_row, weights=(target == 0) & (prediction == 0), minlength=EXPECTED_SITES),
            np.bincount(site_row, weights=(target == 0) & (prediction == 1), minlength=EXPECTED_SITES),
            np.bincount(site_row, weights=(target == 1) & (prediction == 0), minlength=EXPECTED_SITES),
            np.bincount(site_row, weights=(target == 1) & (prediction == 1), minlength=EXPECTED_SITES),
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    squared_error_by_site = np.bincount(
        site_row,
        weights=np.square(probability - target, dtype=np.float64),
        minlength=EXPECTED_SITES,
    ).astype(np.float64, copy=False)
    require(np.all(confusion_by_site.sum(axis=1) == EXPECTED_SAMPLES_PER_SITE), "site confusion closure")

    distributions = {
        name: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for name in BOOTSTRAP_METRICS
    }
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_sites = rng.integers(0, EXPECTED_SITES, size=EXPECTED_SITES)
        multiplicity = np.bincount(sampled_sites, minlength=EXPECTED_SITES).astype(np.float64)
        tn, fp, fn, tp = multiplicity @ confusion_by_site
        hard = hard_metrics_from_counts(tn, fp, fn, tp)
        for name in BOOTSTRAP_METRICS:
            if name != "brier_score":
                distributions[name][replicate] = hard[name]
        distributions["brier_score"][replicate] = (
            float(multiplicity @ squared_error_by_site) / EXPECTED_ROWS
        )
        if (replicate + 1) % 100 == 0:
            print(
                f"  bootstrap replicate {replicate + 1}/{BOOTSTRAP_REPLICATES}",
                flush=True,
            )

    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    rows = []
    for name in BOOTSTRAP_METRICS:
        lower, upper = np.quantile(
            distributions[name],
            [alpha, 1.0 - alpha],
            method="linear",
        )
        rows.append(
            {
                "metric": name,
                "point_estimate": float(point_metrics[name]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence": BOOTSTRAP_CONFIDENCE,
                "replicates": BOOTSTRAP_REPLICATES,
                "method": "physical-site-group nonparametric percentile bootstrap",
            }
        )
    summary = {
        "stage": "11C-5H",
        "status": "PASS",
        "partition": "DEV_SITE_TEST",
        "sampling_unit": "physical fault site",
        "sites_per_replicate": EXPECTED_SITES,
        "records_per_site": EXPECTED_SAMPLES_PER_SITE,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "interval_method": "percentile",
        "auc_interval_disposition": (
            "PR-AUC and ROC-AUC are reported as exact point estimates; confidence "
            "intervals are limited to the primary MCC, hard-decision metrics, and Brier score."
        ),
        "intervals": rows,
    }
    return summary, distributions


def write_bootstrap(summary: dict, distributions: dict[str, np.ndarray]) -> None:
    atomic_json(BOOTSTRAP_JSON, summary)
    temporary = BOOTSTRAP_CSV.with_name(BOOTSTRAP_CSV.name + ".tmp")
    header = [
        "metric",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "confidence",
        "replicates",
        "method",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for record in summary["intervals"]:
            row = dict(record)
            for name in ("point_estimate", "ci_lower", "ci_upper", "confidence"):
                row[name] = format(float(row[name]), ".17g")
            writer.writerow(row)
    temporary.replace(BOOTSTRAP_CSV)
    arrays = {
        name: values.astype("<f8", copy=False)
        for name, values in distributions.items()
    }
    arrays.update(
        {
            "confidence": np.asarray([BOOTSTRAP_CONFIDENCE], dtype="<f8"),
            "random_seed": np.asarray([RANDOM_SEED], dtype="<u8"),
            "replicates": np.asarray([BOOTSTRAP_REPLICATES], dtype="<u4"),
            "site_count": np.asarray([EXPECTED_SITES], dtype="<u4"),
        }
    )
    deterministic_npz(BOOTSTRAP_DISTRIBUTION, arrays)


def write_calibration_curve(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 20,
) -> None:
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, bins)
    temporary = SITE_TEST_CURVE.with_name(SITE_TEST_CURVE.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        header = [
            "bin_index",
            "sample_count",
            "minimum_probability",
            "maximum_probability",
            "mean_predicted_probability",
            "observed_positive_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for bin_index, indices in enumerate(groups):
            require(len(indices) > 0, "empty calibration-curve bin")
            scores = probability[indices]
            labels = y_true[indices]
            writer.writerow(
                {
                    "bin_index": bin_index,
                    "sample_count": len(indices),
                    "minimum_probability": format(float(scores[0]), ".17g"),
                    "maximum_probability": format(float(scores[-1]), ".17g"),
                    "mean_predicted_probability": format(float(scores.mean(dtype=np.float64)), ".17g"),
                    "observed_positive_fraction": format(float(labels.mean(dtype=np.float64)), ".17g"),
                }
            )
    temporary.replace(SITE_TEST_CURVE)


def write_predictions(
    bundle: dict[str, np.ndarray],
    probability: np.ndarray,
    prediction: np.ndarray,
) -> None:
    deterministic_npz(
        PREDICTIONS,
        {
            "candidate_id": np.asarray([EXPECTED_CANDIDATE]),
            "model_sha256": np.asarray([EXPECTED_INPUTS[SELECTED_MODEL]]),
            "partition": np.asarray(["DEV_SITE_TEST"]),
            "prediction": prediction.astype(np.uint8, copy=False),
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


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(EVALUATOR_SOURCE.parent == ROOT, "place evaluator source in the project root")
    require(not AUDIT.exists(), f"Stage 11C-5H already frozen: {AUDIT}")
    require(not MANIFEST.exists(), f"Stage 11C-5H manifest already exists: {MANIFEST}")
    require(not EVALUATION_LOCK.exists(), f"Stage 11C-5H evaluation already locked: {EVALUATION_LOCK}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11C-5H — LOCKED DEV_SITE_TEST BASELINE EVALUATION")
    print("FROZEN INPUT VERIFICATION")
    input_evidence = {}
    for path, expected in EXPECTED_INPUTS.items():
        record = verify_file(path, expected)
        input_evidence[record["path"]] = record
        print(f"  {path.name:<72} : OK", flush=True)

    contract, selection_lock, training_manifest, handoff = verify_frozen_handoff()
    environment = verify_environment()
    require(contract["data_access"]["DEV_SITE_TEST"]["state"] == "LOCKED", "contract site-test state")
    require(training_manifest.get("minimum_validity_status") == "PASS", "calibration minimum validity")

    print("\nOPENING LOCKED DEV_SITE_TEST", flush=True)
    print("  Model fitting calls          : 0", flush=True)
    print("  Candidate-selection calls    : 0", flush=True)
    print("  Threshold-selection calls    : 0", flush=True)
    print("  Frozen candidate             :", EXPECTED_CANDIDATE, flush=True)
    print("  Frozen threshold             :", format(EXPECTED_THRESHOLD, ".17g"), flush=True)
    site_test = load_site_test()
    model = load_locked_model(selection_lock)

    print("\nLOCKED MODEL INFERENCE", flush=True)
    probability, inference_batches = predict_once(model, site_test)
    prediction = (probability >= EXPECTED_THRESHOLD).astype(np.uint8)

    overall = calculate_metrics(
        site_test["target"],
        probability,
        prediction,
        EXPECTED_THRESHOLD,
    )
    dummy_results = dummy_references(site_test["target"])
    minimum_status, target_status, acceptance_checks = assessment(overall, dummy_results)

    print("\nREQUIRED BREAKDOWN METRICS", flush=True)
    breakdowns, unrepresented_breakdowns = make_breakdowns(
        site_test,
        probability,
        prediction,
        EXPECTED_THRESHOLD,
    )
    write_breakdowns(breakdowns, unrepresented_breakdowns)
    for record in breakdowns:
        print(
            f"  {record['group_dimension']:<18} {record['group_value']:<28} "
            f"samples={record['sample_count']:7d} MCC={record['mcc']:.6f}",
            flush=True,
        )
    for record in unrepresented_breakdowns:
        print(
            f"  {record['group_dimension']:<18} {record['group_value']:<28} "
            "NOT REPRESENTED",
            flush=True,
        )

    print("\nPHYSICAL-SITE GROUP BOOTSTRAP", flush=True)
    bootstrap_summary, bootstrap_distributions = site_bootstrap(
        site_test,
        probability,
        prediction,
        overall,
    )

    write_predictions(site_test, probability, prediction)
    write_bootstrap(bootstrap_summary, bootstrap_distributions)
    write_calibration_curve(site_test["target"], probability)

    overall_document = {
        "stage": "11C-5H",
        "title": "LOCKED DEV_SITE_TEST BASELINE EVALUATION",
        "status": "PASS",
        "partition": "DEV_SITE_TEST",
        "candidate_id": EXPECTED_CANDIDATE,
        "model_sha256": EXPECTED_INPUTS[SELECTED_MODEL],
        "threshold": EXPECTED_THRESHOLD,
        "threshold_source": "FROZEN STAGE 11C-5G SELECTION LOCK",
        "probability_calibration_applied": False,
        "metrics": overall,
        "dummy_references": dummy_results,
        "minimum_validity_status": minimum_status,
        "project_target_status": target_status,
        "acceptance_checks": acceptance_checks,
    }
    atomic_json(OVERALL_METRICS, overall_document)

    output_artifacts = {
        "predictions": output_record(PREDICTIONS),
        "overall_metrics": output_record(OVERALL_METRICS),
        "breakdown_csv": output_record(BREAKDOWN_CSV),
        "breakdown_json": output_record(BREAKDOWN_JSON),
        "bootstrap_csv": output_record(BOOTSTRAP_CSV),
        "bootstrap_json": output_record(BOOTSTRAP_JSON),
        "bootstrap_distribution": output_record(BOOTSTRAP_DISTRIBUTION),
        "site_test_calibration_curve": output_record(SITE_TEST_CURVE),
    }

    evaluation_lock = {
        "stage": "11C-5H",
        "title": "CONVENTIONAL BASELINE DEV_SITE_TEST EVALUATION LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "DEV_SITE_TEST",
        "evaluation_policy": "ONE LOCKED DEVELOPMENT-SITE TEST EVALUATION",
        "candidate_id": EXPECTED_CANDIDATE,
        "model_sha256": EXPECTED_INPUTS[SELECTED_MODEL],
        "threshold": EXPECTED_THRESHOLD,
        "threshold_changed": False,
        "model_retrained": False,
        "model_parameters_modified": False,
        "feature_policy_modified": False,
        "repair_classification": "EVALUATOR BREAKDOWN-REPRESENTATION ASSUMPTION ONLY",
        "prior_attempt_frozen": False,
        "technical_rerun_disposition": (
            "AUTHORIZED AFTER PRE-FREEZE REPORTING FAILURE; NO MODEL, THRESHOLD, "
            "FEATURE, OR LABEL CHANGE"
        ),
        "inference_batches": inference_batches,
        "overall_metrics": overall,
        "minimum_validity_status": minimum_status,
        "project_target_status": target_status,
        "represented_breakdown_group_count": len(breakdowns),
        "unrepresented_breakdown_group_count": len(unrepresented_breakdowns),
        "unrepresented_breakdown_groups": unrepresented_breakdowns,
        "site_group_bootstrap": {
            "status": "PASS",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "sampling_unit": "physical fault site",
        },
        "outputs": output_artifacts,
        "stage_11c5g_selection_lock_sha256": EXPECTED_INPUTS[SELECTION_LOCK],
        "site_test_matrix_sha256": EXPECTED_INPUTS[SITE_TEST_MATRIX],
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "retraining_after_site_test_authorized": False,
        "gnn_training_authorized": False,
    }
    atomic_json(EVALUATION_LOCK, evaluation_lock)
    evaluation_lock_record = output_record(EVALUATION_LOCK)

    manifest = {
        "stage": "11C-5H",
        "title": "LOCKED DEV_SITE_TEST BASELINE EVALUATION AND FREEZE",
        "status": "PASS",
        "evaluator_version": VERSION,
        "evaluator_source": output_record(EVALUATOR_SOURCE),
        "repair_classification": "EVALUATOR BREAKDOWN-REPRESENTATION ASSUMPTION ONLY",
        "prior_attempt_frozen": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_simulation_persistent_fault_detectability",
        "fault_scope": "persistent net-stem SA0/SA1 only",
        "partition": "DEV_SITE_TEST",
        "physical_sites": EXPECTED_SITES,
        "fault_instances": EXPECTED_SITES * 2,
        "samples": EXPECTED_ROWS,
        "positive_samples": EXPECTED_POSITIVES,
        "negative_samples": EXPECTED_NEGATIVES,
        "candidate_id": EXPECTED_CANDIDATE,
        "model_sha256": EXPECTED_INPUTS[SELECTED_MODEL],
        "threshold": EXPECTED_THRESHOLD,
        "overall_metrics": overall,
        "minimum_validity_status": minimum_status,
        "project_target_status": target_status,
        "breakdown_groups": len(breakdowns),
        "unrepresented_breakdown_group_count": len(unrepresented_breakdowns),
        "unrepresented_breakdown_groups": unrepresented_breakdowns,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "model_fit_calls": 0,
        "candidate_selection_calls": 0,
        "threshold_selection_calls": 0,
        "input_evidence": input_evidence,
        "locked_handoff_artifacts": handoff["artifacts"],
        "environment": environment,
        "outputs": output_artifacts,
        "evaluation_lock": evaluation_lock_record,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "trained_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = output_record(MANIFEST)

    audit = {
        "stage": "11C-5H",
        "title": "LOCKED DEV_SITE_TEST BASELINE EVALUATION AND FREEZE",
        "status": "PASS",
        "evaluation_status": "FROZEN",
        "baseline_status": "FROZEN COMPARATOR",
        "evaluator_version": VERSION,
        "evaluator_source": output_record(EVALUATOR_SOURCE),
        "repair_classification": "EVALUATOR BREAKDOWN-REPRESENTATION ASSUMPTION ONLY",
        "prior_attempt_frozen": False,
        "candidate_id": EXPECTED_CANDIDATE,
        "threshold": EXPECTED_THRESHOLD,
        "site_test_samples": EXPECTED_ROWS,
        "site_test_sites": EXPECTED_SITES,
        "site_test_positive_samples": EXPECTED_POSITIVES,
        "site_test_negative_samples": EXPECTED_NEGATIVES,
        "mcc": overall["mcc"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "specificity": overall["specificity"],
        "f1_score": overall["f1_score"],
        "pr_auc": overall["pr_auc"],
        "roc_auc": overall["roc_auc"],
        "brier_score": overall["brier_score"],
        "minimum_validity_status": minimum_status,
        "project_target_status": target_status,
        "site_group_bootstrap": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "required_breakdowns": "PASS",
        "breakdown_groups": len(breakdowns),
        "unrepresented_breakdown_group_count": len(unrepresented_breakdowns),
        "unrepresented_breakdown_groups": unrepresented_breakdowns,
        "model_retrained": False,
        "threshold_changed": False,
        "dev_site_test_state": "CONSUMED AND FROZEN",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized": False,
        "evaluation_lock": evaluation_lock_record,
        "manifest": manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "trained_model_modified": False,
        "next_gate": "DEEP DIAGNOSTIC MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
    }
    atomic_json(AUDIT, audit)
    audit_record = output_record(AUDIT)

    mcc_interval = next(
        record for record in bootstrap_summary["intervals"] if record["metric"] == "mcc"
    )
    print("\nSTAGE 11C-5H — LOCKED DEV_SITE_TEST BASELINE EVALUATION AND FREEZE")
    print("Status                         : PASS")
    print("Evaluation status              : FROZEN")
    print("Baseline status                : FROZEN COMPARATOR")
    print("Repair classification          : EVALUATOR ASSUMPTION ONLY")
    print("Prior attempt frozen           : NO")
    print("Candidate                      :", EXPECTED_CANDIDATE)
    print("Threshold                      :", format(EXPECTED_THRESHOLD, ".17g"))
    print("DEV_SITE_TEST sites            :", EXPECTED_SITES)
    print("DEV_SITE_TEST samples          :", EXPECTED_ROWS)
    print("Positive samples               :", EXPECTED_POSITIVES)
    print("Negative samples               :", EXPECTED_NEGATIVES)
    print("MCC                            :", f"{overall['mcc']:.8f}")
    print(
        "MCC 95% site-bootstrap CI      :",
        f"[{mcc_interval['ci_lower']:.8f}, {mcc_interval['ci_upper']:.8f}]",
    )
    print("Balanced accuracy              :", f"{overall['balanced_accuracy']:.8f}")
    print("Precision                      :", f"{overall['precision']:.8f}")
    print("Recall                         :", f"{overall['recall']:.8f}")
    print("Specificity                    :", f"{overall['specificity']:.8f}")
    print("F1                             :", f"{overall['f1_score']:.8f}")
    print("PR-AUC                         :", f"{overall['pr_auc']:.8f}")
    print("ROC-AUC                        :", f"{overall['roc_auc']:.8f}")
    print("Brier score                    :", f"{overall['brier_score']:.8f}")
    print("Minimum validity               :", minimum_status)
    print("Project target                 :", target_status)
    print("Required breakdowns            : PASS")
    print("Breakdown groups               :", len(breakdowns))
    print("Unrepresented driver groups    :", len(unrepresented_breakdowns))
    for record in unrepresented_breakdowns:
        print("  Not represented              :", record["group_value"])
    print("Site bootstrap                 : PASS")
    print("Bootstrap replicates           :", BOOTSTRAP_REPLICATES)
    print("Model retrained                : NO")
    print("Threshold changed              : NO")
    print("DEV_SITE_TEST state            : CONSUMED AND FROZEN")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Canonical dataset modified     : NO")
    print("Feature matrices modified      : NO")
    print("Trained model modified         : NO")
    print("Predictions                    :", PREDICTIONS)
    print("Predictions SHA                :", sha256(PREDICTIONS))
    print("Evaluation lock                :", EVALUATION_LOCK)
    print("Evaluation lock SHA            :", evaluation_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : DEEP DIAGNOSTIC MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE")


if __name__ == "__main__":
    main()
