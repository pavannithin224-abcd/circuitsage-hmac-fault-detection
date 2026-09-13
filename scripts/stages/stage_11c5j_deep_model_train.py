#!/usr/bin/env python3
"""Train, calibrate, replay, select, and freeze the Stage 11C-5I deep MLP."""
from __future__ import annotations

import os

# Freeze numerical execution before importing NumPy/scikit-learn.
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
import platform
import shutil
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error


VERSION = "HMAC-DEEP-DIAGNOSTIC-TRAINER-v1"
RANDOM_SEED = 20_260_904
EXPECTED_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512
EXPECTED_VECTORS = 64
EXPECTED_SAMPLES_PER_SITE = 128
EXPECTED_TRAIN_SITES = 15_987
EXPECTED_CALIBRATION_SITES = 3_426
EXPECTED_TRAIN_ROWS = 2_046_336
EXPECTED_CALIBRATION_ROWS = 438_528
EXPECTED_TRAIN_POSITIVES = 894_000
EXPECTED_CALIBRATION_POSITIVES = 192_943
TRAINING_BATCH_SIZE = 8_192
INFERENCE_BATCH_SIZE = 32_768
EPOCHS = 4
THRESHOLD_GRID_POINTS = 2_001
HIDDEN_LAYERS = (128, 64, 32)
EXPECTED_PARAMETERS = 77_953

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
WORK_ROOT = RESULT_ROOT / "deep_training_11c5j"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CALIBRATION_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"

CONTRACT_SOURCE = ROOT / "stage_11c5i_deep_model_contract.py"
ARCHITECTURE = CONFIG_ROOT / "hmac_deep_diagnostic_architecture_11c5i.json"
TRAINING_CONTRACT = CONFIG_ROOT / "hmac_deep_diagnostic_training_contract_11c5i.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_deep_diagnostic_candidate_grid_11c5i.csv"
CONTRACT_ENVIRONMENT = RESULT_ROOT / "hmac_deep_diagnostic_environment_11c5i.json"
CONTRACT_AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_architecture_training_contract_freeze_11c5i.json"

CANDIDATE_METRICS_CSV = WORK_ROOT / "hmac_deep_candidate_calibration_metrics_11c5j.csv"
CANDIDATE_METRICS_JSON = WORK_ROOT / "hmac_deep_candidate_calibration_metrics_11c5j.json"
TRAINING_HISTORY = WORK_ROOT / "hmac_deep_training_history_11c5j.csv"
SELECTED_MODEL = WORK_ROOT / "hmac_selected_deep_diagnostic_model_11c5j.joblib"
SELECTED_WEIGHTS = WORK_ROOT / "hmac_selected_deep_diagnostic_weights_11c5j.npz"
SELECTED_PREDICTIONS = WORK_ROOT / "hmac_selected_deep_calibration_predictions_11c5j.npz"
CALIBRATION_CURVE = WORK_ROOT / "hmac_selected_deep_calibration_curve_11c5j.csv"
REPLAY_MODEL = WORK_ROOT / "hmac_selected_deep_diagnostic_replay_model_11c5j.joblib"
REPLAY_WEIGHTS = WORK_ROOT / "hmac_selected_deep_diagnostic_replay_weights_11c5j.npz"
REPLAY_PREDICTIONS = WORK_ROOT / "hmac_selected_deep_calibration_replay_predictions_11c5j.npz"
SELECTION_LOCK = WORK_ROOT / "hmac_deep_diagnostic_selection_lock_11c5j.json"
MANIFEST = RESULT_ROOT / "hmac_deep_diagnostic_training_manifest_11c5j.json"
AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_training_calibration_freeze_11c5j.json"

EXPECTED_INPUTS = {
    TRAIN_MATRIX: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CALIBRATION_MATRIX: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    CONTRACT_SOURCE: "a58ae217b3f2f1c78627d93650b2e18b75f3708a5e7143842a0a99a7b05dbca6",
    ARCHITECTURE: "fad4bf5caf653f1e7712d16114ca8a9aa85982a992e21e19ee2c09606583ce6f",
    TRAINING_CONTRACT: "e6175b3921737ec78bf88a1d8020b17ab2e37f44137da211dd20414e6dbb0894",
    CANDIDATE_GRID: "98b2919c718644935edac8719915d4935d876a19fe52e80870404fdf3b263e24",
    CONTRACT_ENVIRONMENT: "9b0cf9cd372c6effbcfd442d1d1763fb641489c2c17aa13768238e520214301f",
    CONTRACT_AUDIT: "0ce03891c718f62e1bf47746db298a6338582997727eb9d862b6f7b829f6c05c",
}

METRIC_FIELDS = [
    "candidate_id",
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
    "positive_prevalence",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "minimum_validity_status",
    "project_target_status",
    "final_batch_loss",
]

HISTORY_FIELDS = [
    "run",
    "candidate_id",
    "epoch",
    "batches",
    "samples_seen",
    "final_batch_loss",
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


def atomic_joblib(path: Path, model: MLPClassifier) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary, compress=3, protocol=5)
    temporary.replace(path)


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {relative(path)}")
    require(path.stat().st_size > 0, f"empty frozen input: {relative(path)}")
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual}",
    )
    print(f"  {path.name:<72}: OK", flush=True)
    return record(path)


def verify_outputs_absent() -> None:
    paths = (
        CANDIDATE_METRICS_CSV,
        CANDIDATE_METRICS_JSON,
        TRAINING_HISTORY,
        SELECTED_MODEL,
        SELECTED_WEIGHTS,
        SELECTED_PREDICTIONS,
        CALIBRATION_CURVE,
        REPLAY_MODEL,
        REPLAY_WEIGHTS,
        REPLAY_PREDICTIONS,
        SELECTION_LOCK,
        MANIFEST,
        AUDIT,
    )
    for path in paths:
        require(not path.exists(), f"Stage 11C-5J output already exists: {relative(path)}")


def checkpoint_paths(candidate_id: str, replay: bool) -> tuple[Path, Path]:
    prefix = "replay" if replay else "canonical"
    stem = f"{prefix}_{candidate_id.lower()}"
    return (
        CHECKPOINT_ROOT / f"{stem}.joblib",
        CHECKPOINT_ROOT / f"{stem}.json",
    )


def load_checkpoint(candidate: dict, replay: bool) -> tuple[MLPClassifier | None, int, list[dict]]:
    model_path, metadata_path = checkpoint_paths(candidate["candidate_id"], replay)
    if not model_path.exists() and not metadata_path.exists():
        return None, 0, []
    require(model_path.is_file() and metadata_path.is_file(), "incomplete training checkpoint")
    metadata = load_json(metadata_path)
    require(metadata.get("status") == "IN_PROGRESS", "checkpoint status")
    require(metadata.get("trainer_version") == VERSION, "checkpoint trainer version")
    require(metadata.get("candidate") == candidate, "checkpoint candidate contract")
    require(metadata.get("replay") is replay, "checkpoint replay identity")
    require(metadata.get("model_sha256") == sha256(model_path), "checkpoint model SHA")
    completed_epoch = int(metadata.get("completed_epoch", -1))
    require(0 <= completed_epoch <= EPOCHS, "checkpoint epoch")
    history = metadata.get("history")
    require(isinstance(history, list) and len(history) == completed_epoch, "checkpoint history")
    model = joblib.load(model_path)
    require(isinstance(model, MLPClassifier), "checkpoint estimator type")
    require(tuple(model.hidden_layer_sizes) == HIDDEN_LAYERS, "checkpoint hidden layers")
    require(int(model.n_features_in_) == EXPECTED_FEATURES, "checkpoint feature count")
    print(
        f"  RESUME {candidate['candidate_id']} "
        f"{'replay' if replay else 'canonical'} after epoch {completed_epoch}/{EPOCHS}",
        flush=True,
    )
    return model, completed_epoch, history


def save_checkpoint(
    candidate: dict,
    replay: bool,
    model: MLPClassifier,
    completed_epoch: int,
    history: list[dict],
) -> None:
    model_path, metadata_path = checkpoint_paths(candidate["candidate_id"], replay)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_joblib(model_path, model)
    atomic_json(
        metadata_path,
        {
            "status": "IN_PROGRESS",
            "trainer_version": VERSION,
            "candidate": candidate,
            "replay": replay,
            "completed_epoch": completed_epoch,
            "history": history,
            "model_sha256": sha256(model_path),
            "model_bytes": model_path.stat().st_size,
        },
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {name}")


def verify_environment() -> dict:
    frozen = load_json(CONTRACT_ENVIRONMENT)
    require(frozen.get("status") == "PASS", "Stage 11C-5I environment status")
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
    return {
        "status": "PASS",
        "python": platform.python_version(),
        "packages": actual,
        "device": "CPU",
        "threads": 1,
    }


def parse_candidates() -> list[dict]:
    with CANDIDATE_GRID.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2, "frozen candidate count")
    expected_ids = [
        "DEEP_MLP_128_64_32_A1E4_LR1E3",
        "DEEP_MLP_128_64_32_A1E5_LR3E4",
    ]
    require([row.get("candidate_id") for row in rows] == expected_ids, "candidate IDs/order")
    for row in rows:
        require(row.get("hidden_layer_sizes") == "128;64;32", "hidden-layer contract")
        require(row.get("activation") == "relu", "activation contract")
        require(row.get("solver") == "adam", "solver contract")
        require(int(row.get("batch_size", 0)) == TRAINING_BATCH_SIZE, "batch-size contract")
        require(int(row.get("epochs", 0)) == EPOCHS, "epoch contract")
        require(row.get("shuffle") == "false", "shuffle contract")
        require(row.get("early_stopping") == "false", "early-stopping contract")
        require(int(row.get("random_seed", 0)) == RANDOM_SEED, "seed contract")
        require(int(row.get("trainable_parameters", 0)) == EXPECTED_PARAMETERS, "parameter contract")
    return rows


def verify_contract() -> tuple[dict, dict]:
    architecture = load_json(ARCHITECTURE)
    contract = load_json(TRAINING_CONTRACT)
    audit = load_json(CONTRACT_AUDIT)
    require(architecture.get("status") == "PASS", "architecture status")
    require(architecture.get("architecture_status") == "FROZEN", "architecture freeze")
    require(tuple(architecture.get("hidden_layer_sizes", [])) == HIDDEN_LAYERS, "architecture layers")
    require(int(architecture.get("trainable_parameters", 0)) == EXPECTED_PARAMETERS, "architecture parameters")
    require(architecture.get("gnn_model") is False, "non-GNN contract")
    require(architecture.get("post_simulation_features") == 0, "post-simulation feature contract")
    require(architecture.get("target_present_in_x") is False, "target leakage contract")

    require(contract.get("status") == "PASS", "training contract status")
    require(contract.get("training_contract_status") == "FROZEN", "training contract freeze")
    require(contract.get("dev_train_authorized") is True, "DEV_TRAIN authorization")
    require(contract.get("dev_calibration_authorized_for_selection") is True, "calibration authorization")
    require(contract.get("dev_site_test_authorized_during_training") is False, "site-test lock")
    require(contract.get("validation_campaign_authorized") is False, "VALIDATION lock")
    require(contract.get("holdout_campaign_authorized") is False, "HOLDOUT lock")
    require(contract.get("gnn_training_authorized") is False, "GNN lock")
    require(contract.get("hybrid_training_authorized") is False, "hybrid lock")
    require(contract.get("selection_partition") == "DEV_CALIBRATION", "selection partition")
    require(contract.get("primary_selection_metric") == "MCC", "primary metric")
    require(int(contract.get("training_batch_size", 0)) == TRAINING_BATCH_SIZE, "contract batch size")
    require(int(contract.get("epochs_per_candidate", 0)) == EPOCHS, "contract epochs")
    require(contract.get("candidate_execution") == "SEQUENTIAL", "candidate execution")
    require(contract.get("parallel_candidates") == 1, "parallel-candidate contract")
    require(contract.get("checkpoint_after_each_epoch") is True, "checkpoint contract")
    require(contract.get("resume_supported") is True, "resume contract")

    require(audit.get("status") == "PASS", "Stage 11C-5I audit status")
    require(audit.get("architecture_status") == "FROZEN", "audit architecture freeze")
    require(audit.get("training_contract_status") == "FROZEN", "audit training freeze")
    require(audit.get("deep_model_training") == "AUTHORIZED FOR DEV_TRAIN", "training authorization")
    require(audit.get("dev_site_test") == "LOCKED", "audit site-test lock")
    require(audit.get("gnn_training") == "NOT YET AUTHORIZED", "audit GNN lock")
    require(audit.get("hybrid_training") == "NOT YET AUTHORIZED", "audit hybrid lock")
    return architecture, contract


def load_partition(path: Path, partition: str, rows: int, sites: int, positives: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
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
        require(required <= set(archive.files), f"{partition} matrix members")
        require(archive["partition"].tolist() == [partition], f"{partition} identity")
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
    require(result["site_features"].shape == (sites, EXPECTED_SITE_FEATURES), f"{partition} site shape")
    require(result["vector_features"].shape == (EXPECTED_VECTORS, EXPECTED_VECTOR_FEATURES), f"{partition} vector shape")
    for name in ("site_row", "vector_row", "stuck_value", "target", "record_id"):
        require(result[name].shape == (rows,), f"{partition} {name} rows")
    require(result["feature_columns"].shape == (EXPECTED_FEATURES,), f"{partition} feature columns")
    require(result["site_feature_columns"].shape == (EXPECTED_SITE_FEATURES,), f"{partition} site columns")
    require(result["site_ids"].shape == (sites,), f"{partition} site IDs")
    require(result["site_indices"].shape == (sites,), f"{partition} site indices")
    require(result["vector_ids"].shape == (EXPECTED_VECTORS,), f"{partition} vector IDs")
    require(np.isfinite(result["site_features"]).all(), f"{partition} finite site features")
    require(np.isfinite(result["vector_features"]).all(), f"{partition} finite vector features")
    require(np.isin(result["target"], (0, 1)).all(), f"{partition} binary target")
    require(np.isin(result["stuck_value"], (0, 1)).all(), f"{partition} binary stuck value")
    require(int(result["target"].sum(dtype=np.uint64)) == positives, f"{partition} positives")
    require(int(result["site_row"].max()) == sites - 1, f"{partition} site-row range")
    require(int(result["vector_row"].max()) == EXPECTED_VECTORS - 1, f"{partition} vector-row range")
    require(np.all(np.bincount(result["site_row"], minlength=sites) == EXPECTED_SAMPLES_PER_SITE), f"{partition} site closure")
    sample_code = (
        result["site_row"].astype(np.uint64) * (2 * EXPECTED_VECTORS)
        + result["stuck_value"].astype(np.uint64) * EXPECTED_VECTORS
        + result["vector_row"].astype(np.uint64)
    )
    require(len(np.unique(sample_code)) == rows, f"{partition} unique samples")
    require(result["feature_columns"][0] == "stuck_value", f"{partition} feature zero")
    forbidden = {
        "detected", "target_detected", "activity", "cycles", "baseline_cycles",
        "latency_delta", "timed_out", "unknown", "digest_hamming_distance",
        "expected_digest", "actual_digest",
    }
    require(not (set(result["feature_columns"].tolist()) & forbidden), f"{partition} leakage field")
    require(
        result["feature_columns"][1:15].tolist() == result["site_feature_columns"].tolist(),
        f"{partition} feature alignment",
    )
    return result


def materialize(bundle: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.empty((len(indices), EXPECTED_FEATURES), dtype=np.float32)
    matrix[:, 0] = bundle["stuck_value"][indices]
    matrix[:, 1:15] = bundle["site_features"][bundle["site_row"][indices]]
    matrix[:, 15:] = bundle["vector_features"][bundle["vector_row"][indices]]
    require(np.isfinite(matrix).all(), "finite materialized features")
    return matrix, bundle["target"][indices]


def build_model(candidate: dict) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=float(candidate["alpha"]),
        batch_size=TRAINING_BATCH_SIZE,
        learning_rate_init=float(candidate["learning_rate_init"]),
        max_iter=1,
        shuffle=False,
        random_state=RANDOM_SEED,
        tol=0.0,
        early_stopping=False,
        n_iter_no_change=EPOCHS + 1,
        warm_start=False,
    )


def epoch_order(row_count: int, epoch: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED + epoch))
    return rng.permutation(row_count).astype(np.int64, copy=False)


def train_candidate(candidate: dict, train: dict[str, np.ndarray], replay: bool = False) -> tuple[MLPClassifier, list[dict]]:
    candidate_id = candidate["candidate_id"]
    model, start_epoch, history = load_checkpoint(candidate, replay)
    if model is None:
        model = build_model(candidate)
    total_batches = math.ceil(EXPECTED_TRAIN_ROWS / TRAINING_BATCH_SIZE)
    prefix = "REPLAY" if replay else "TRAIN"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(start_epoch, EPOCHS):
            order = epoch_order(EXPECTED_TRAIN_ROWS, epoch)
            batches = 0
            for start in range(0, EXPECTED_TRAIN_ROWS, TRAINING_BATCH_SIZE):
                indices = order[start : start + TRAINING_BATCH_SIZE]
                matrix, target = materialize(train, indices)
                if epoch == 0 and start == 0:
                    model.partial_fit(matrix, target, classes=np.asarray([0, 1], dtype=np.uint8))
                else:
                    model.partial_fit(matrix, target)
                batches += 1
                if batches % 50 == 0 or batches == total_batches:
                    print(
                        f"  {prefix} {candidate_id} epoch {epoch + 1}/{EPOCHS} "
                        f"batch {batches}/{total_batches}",
                        flush=True,
                    )
            loss = float(model.loss_)
            require(math.isfinite(loss), f"non-finite training loss for {candidate_id}")
            history.append(
                {
                    "run": "REPLAY" if replay else "CANONICAL",
                    "candidate_id": candidate_id,
                    "epoch": epoch + 1,
                    "batches": batches,
                    "samples_seen": EXPECTED_TRAIN_ROWS,
                    "final_batch_loss": format(loss, ".17g"),
                }
            )
            save_checkpoint(candidate, replay, model, epoch + 1, history)
    require(tuple(model.hidden_layer_sizes) == HIDDEN_LAYERS, "trained hidden layers")
    require(int(model.n_features_in_) == EXPECTED_FEATURES, "trained feature count")
    require(np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])), "trained classes")
    parameter_count = sum(value.size for value in model.coefs_) + sum(
        value.size for value in model.intercepts_
    )
    require(parameter_count == EXPECTED_PARAMETERS, "trained parameter count")
    return model, history


def predict_probability(model: MLPClassifier, bundle: dict[str, np.ndarray], label: str) -> np.ndarray:
    rows = len(bundle["target"])
    probability = np.empty(rows, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    total_batches = math.ceil(rows / INFERENCE_BATCH_SIZE)
    for batch_number, start in enumerate(range(0, rows, INFERENCE_BATCH_SIZE), start=1):
        stop_index = min(start + INFERENCE_BATCH_SIZE, rows)
        indices = np.arange(start, stop_index, dtype=np.int64)
        matrix, _ = materialize(bundle, indices)
        probability[start:stop_index] = model.predict_proba(matrix)[:, positive_column]
        print(f"  {label} inference batch {batch_number}/{total_batches}", flush=True)
    require(np.isfinite(probability).all(), f"{label} finite probabilities")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), f"{label} probability range")
    return probability


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def hard_metrics(tn: int, fp: int, fn: int, tp: int) -> dict:
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


def select_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    positive_scores = np.sort(probability[target == 1])
    negative_scores = np.sort(probability[target == 0])
    thresholds = np.linspace(0.0, 1.0, THRESHOLD_GRID_POINTS, dtype=np.float64)
    positives = len(positive_scores)
    negatives = len(negative_scores)
    best = None
    for threshold in thresholds:
        tp = positives - int(np.searchsorted(positive_scores, threshold, side="left"))
        fp = negatives - int(np.searchsorted(negative_scores, threshold, side="left"))
        fn = positives - tp
        tn = negatives - fp
        metrics = hard_metrics(tn, fp, fn, tp)
        key = (
            metrics["mcc"],
            metrics["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), {**metrics, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    require(best is not None, "threshold selection")
    return best[1], best[2]


def full_metrics(target: np.ndarray, probability: np.ndarray, threshold: float, candidate_id: str, final_loss: float) -> dict:
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(target, prediction, labels=[0, 1]).ravel()]
    result = {
        "candidate_id": candidate_id,
        "threshold": float(threshold),
        **hard_metrics(tn, fp, fn, tp),
        "pr_auc": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "positive_prevalence": float(target.mean()),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "final_batch_loss": float(final_loss),
    }
    minimum = {
        "mcc_gt_zero": result["mcc"] > 0.0,
        "balanced_accuracy_gt_half": result["balanced_accuracy"] > 0.5,
        "pr_auc_gt_prevalence": result["pr_auc"] > result["positive_prevalence"],
        "finite_metrics": all(math.isfinite(float(result[name])) for name in (
            "mcc", "balanced_accuracy", "precision", "recall", "specificity",
            "f1_score", "pr_auc", "roc_auc", "brier_score", "final_batch_loss",
        )),
    }
    project = {
        "mcc_ge_0_40": result["mcc"] >= 0.40,
        "balanced_accuracy_ge_0_70": result["balanced_accuracy"] >= 0.70,
        "f1_ge_0_70": result["f1_score"] >= 0.70,
        "recall_ge_0_70": result["recall"] >= 0.70,
    }
    result["minimum_validity_checks"] = minimum
    result["project_target_checks"] = project
    result["minimum_validity_status"] = "PASS" if all(minimum.values()) else "NOT_MET"
    result["project_target_status"] = "PASS" if all(project.values()) else "NOT_MET"
    return result


def metric_csv_row(metrics: dict) -> dict:
    return {name: metrics[name] for name in METRIC_FIELDS}


def selection_key(metrics: dict) -> tuple:
    return (
        float(metrics["mcc"]),
        float(metrics["pr_auc"]),
        -float(metrics["brier_score"]),
    )


def model_arrays(model: MLPClassifier) -> dict[str, np.ndarray]:
    arrays = {
        "classes": np.asarray(model.classes_),
        "n_features_in": np.asarray([model.n_features_in_], dtype=np.int32),
        "n_iter": np.asarray([model.n_iter_], dtype=np.int32),
        "t": np.asarray([model.t_], dtype=np.float64),
    }
    for index, value in enumerate(model.coefs_):
        arrays[f"coef_{index}"] = np.asarray(value)
    for index, value in enumerate(model.intercepts_):
        arrays[f"intercept_{index}"] = np.asarray(value)
    return arrays


def compare_models(first: MLPClassifier, second: MLPClassifier) -> bool:
    if len(first.coefs_) != len(second.coefs_) or len(first.intercepts_) != len(second.intercepts_):
        return False
    return all(np.array_equal(a, b) for a, b in zip(first.coefs_, second.coefs_)) and all(
        np.array_equal(a, b) for a, b in zip(first.intercepts_, second.intercepts_)
    )


def write_calibration_curve(path: Path, target: np.ndarray, probability: np.ndarray) -> None:
    rows = []
    edges = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    bin_index = np.minimum(np.searchsorted(edges, probability, side="right") - 1, 19)
    bin_index = np.maximum(bin_index, 0)
    for index in range(20):
        mask = bin_index == index
        count = int(mask.sum())
        rows.append(
            {
                "bin_index": index,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "sample_count": count,
                "mean_probability": float(probability[mask].mean()) if count else "",
                "observed_positive_rate": float(target[mask].mean()) if count else "",
            }
        )
    atomic_csv(
        path,
        ["bin_index", "lower_bound", "upper_bound", "sample_count", "mean_probability", "observed_positive_rate"],
        rows,
    )


def cleanup_partial_outputs() -> None:
    # Preserve valid epoch checkpoints, but remove incomplete final-freeze outputs.
    if not AUDIT.exists():
        for path in (
            CANDIDATE_METRICS_CSV,
            CANDIDATE_METRICS_JSON,
            TRAINING_HISTORY,
            SELECTED_MODEL,
            SELECTED_WEIGHTS,
            SELECTED_PREDICTIONS,
            CALIBRATION_CURVE,
            REPLAY_MODEL,
            REPLAY_WEIGHTS,
            REPLAY_PREDICTIONS,
            SELECTION_LOCK,
            MANIFEST,
        ):
            if path.exists():
                path.unlink()
        if WORK_ROOT.exists():
            for temporary in WORK_ROOT.rglob("*.tmp"):
                temporary.unlink()


def main() -> None:
    require(SOURCE.parent == ROOT, "run the Stage 11C-5J script from the project root")
    verify_outputs_absent()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11C-5J — DEEP DIAGNOSTIC MODEL TRAINING AND CALIBRATION")
    print("FROZEN INPUT VERIFICATION", flush=True)
    input_evidence = {
        relative(path): verify_input(path, expected_sha)
        for path, expected_sha in EXPECTED_INPUTS.items()
    }
    architecture, contract = verify_contract()
    environment = verify_environment()
    candidates = parse_candidates()

    print("\nOPENING AUTHORIZED DEVELOPMENT PARTITIONS", flush=True)
    train = load_partition(
        TRAIN_MATRIX, "DEV_TRAIN", EXPECTED_TRAIN_ROWS, EXPECTED_TRAIN_SITES, EXPECTED_TRAIN_POSITIVES
    )
    calibration = load_partition(
        CALIBRATION_MATRIX,
        "DEV_CALIBRATION",
        EXPECTED_CALIBRATION_ROWS,
        EXPECTED_CALIBRATION_SITES,
        EXPECTED_CALIBRATION_POSITIVES,
    )
    require(
        set(train["site_ids"].tolist()).isdisjoint(set(calibration["site_ids"].tolist())),
        "DEV_TRAIN/DEV_CALIBRATION site overlap",
    )
    require(np.array_equal(train["feature_columns"], calibration["feature_columns"]), "feature vocabulary")
    print("  DEV_TRAIN       : OPENED FOR PARAMETER FITTING")
    print("  DEV_CALIBRATION : OPENED FOR SELECTION")
    print("  DEV_SITE_TEST   : NOT OPENED")

    trained = []
    metric_records = []
    history_records = []
    print("\nSEQUENTIAL DEEP CANDIDATE TRAINING", flush=True)
    for candidate in candidates:
        print(f"\nCANDIDATE {candidate['candidate_id']}", flush=True)
        model, history = train_candidate(candidate, train)
        probability = predict_probability(model, calibration, candidate["candidate_id"])
        threshold, _ = select_threshold(calibration["target"], probability)
        metrics = full_metrics(
            calibration["target"], probability, threshold, candidate["candidate_id"], float(model.loss_)
        )
        print(
            f"  threshold={threshold:.6f} MCC={metrics['mcc']:.8f} "
            f"PR-AUC={metrics['pr_auc']:.8f} status={metrics['minimum_validity_status']}",
            flush=True,
        )
        trained.append({"candidate": candidate, "model": model, "probability": probability, "metrics": metrics})
        metric_records.append(metrics)
        history_records.extend(history)

    selected = max(trained, key=lambda item: (*selection_key(item["metrics"]), item["candidate"]["candidate_id"] == candidates[0]["candidate_id"]))
    selected_id = selected["candidate"]["candidate_id"]
    selected_model = selected["model"]
    selected_probability = selected["probability"]
    selected_metrics = selected["metrics"]
    require(selected_metrics["minimum_validity_status"] == "PASS", "selected deep model minimum validity")

    print("\nSELECTED DEEP CANDIDATE")
    print("  Candidate :", selected_id)
    print("  Threshold :", format(selected_metrics["threshold"], ".17g"))
    print("  MCC       :", f"{selected_metrics['mcc']:.8f}")

    print("\nDETERMINISTIC SELECTED-CANDIDATE REPLAY", flush=True)
    replay_model, replay_history = train_candidate(selected["candidate"], train, replay=True)
    replay_probability = predict_probability(replay_model, calibration, "REPLAY")
    replay_threshold, _ = select_threshold(calibration["target"], replay_probability)
    replay_metrics = full_metrics(
        calibration["target"], replay_probability, replay_threshold, selected_id, float(replay_model.loss_)
    )
    weights_exact = compare_models(selected_model, replay_model)
    probabilities_exact = np.array_equal(selected_probability, replay_probability)
    threshold_exact = selected_metrics["threshold"] == replay_threshold
    metrics_exact = all(
        selected_metrics[name] == replay_metrics[name]
        for name in ("mcc", "balanced_accuracy", "precision", "recall", "specificity", "f1_score", "pr_auc", "roc_auc", "brier_score")
    )
    require(weights_exact, "deterministic replay weights")
    require(probabilities_exact, "deterministic replay probabilities")
    require(threshold_exact, "deterministic replay threshold")
    require(metrics_exact, "deterministic replay metrics")

    created_at = datetime.now(timezone.utc).isoformat()
    atomic_csv(CANDIDATE_METRICS_CSV, METRIC_FIELDS, [metric_csv_row(item) for item in metric_records])
    atomic_json(
        CANDIDATE_METRICS_JSON,
        {
            "stage": "11C-5J",
            "status": "PASS",
            "selection_partition": "DEV_CALIBRATION",
            "primary_metric": "MCC",
            "candidates": metric_records,
        },
    )
    atomic_csv(TRAINING_HISTORY, HISTORY_FIELDS, history_records + replay_history)
    atomic_joblib(SELECTED_MODEL, selected_model)
    deterministic_npz(SELECTED_WEIGHTS, model_arrays(selected_model))
    selected_prediction = (selected_probability >= selected_metrics["threshold"]).astype(np.uint8)
    deterministic_npz(
        SELECTED_PREDICTIONS,
        {
            "record_id": calibration["record_id"],
            "site_row": calibration["site_row"],
            "vector_row": calibration["vector_row"],
            "stuck_value": calibration["stuck_value"],
            "target_detected": calibration["target"],
            "probability": selected_probability,
            "prediction": selected_prediction,
            "threshold": np.asarray([selected_metrics["threshold"]], dtype=np.float64),
        },
    )
    write_calibration_curve(CALIBRATION_CURVE, calibration["target"], selected_probability)
    atomic_joblib(REPLAY_MODEL, replay_model)
    deterministic_npz(REPLAY_WEIGHTS, model_arrays(replay_model))
    deterministic_npz(
        REPLAY_PREDICTIONS,
        {
            "record_id": calibration["record_id"],
            "probability": replay_probability,
            "prediction": (replay_probability >= replay_threshold).astype(np.uint8),
            "threshold": np.asarray([replay_threshold], dtype=np.float64),
        },
    )

    outputs = {
        "candidate_metrics_csv": record(CANDIDATE_METRICS_CSV),
        "candidate_metrics_json": record(CANDIDATE_METRICS_JSON),
        "training_history": record(TRAINING_HISTORY),
        "selected_model": record(SELECTED_MODEL),
        "selected_weights": record(SELECTED_WEIGHTS),
        "selected_calibration_predictions": record(SELECTED_PREDICTIONS),
        "selected_calibration_curve": record(CALIBRATION_CURVE),
        "replay_model": record(REPLAY_MODEL),
        "replay_weights": record(REPLAY_WEIGHTS),
        "replay_predictions": record(REPLAY_PREDICTIONS),
    }

    selection_lock = {
        "stage": "11C-5J",
        "title": "DEEP DIAGNOSTIC MODEL SELECTION LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": created_at,
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_metrics["threshold"],
        "selection_partition": "DEV_CALIBRATION",
        "selection_metric": "MCC",
        "selected_metrics": selected_metrics,
        "minimum_validity_status": selected_metrics["minimum_validity_status"],
        "project_target_status": selected_metrics["project_target_status"],
        "selected_model": outputs["selected_model"],
        "selected_weights": outputs["selected_weights"],
        "selected_calibration_predictions": outputs["selected_calibration_predictions"],
        "selected_calibration_curve": outputs["selected_calibration_curve"],
        "candidate_metrics_csv": outputs["candidate_metrics_csv"],
        "candidate_metrics_json": outputs["candidate_metrics_json"],
        "deterministic_replay": {
            "status": "PASS",
            "weights_exact": weights_exact,
            "probabilities_exact": probabilities_exact,
            "threshold_exact": threshold_exact,
            "metrics_exact": metrics_exact,
            "replay_model": outputs["replay_model"],
            "replay_weights": outputs["replay_weights"],
            "replay_predictions": outputs["replay_predictions"],
        },
        "dev_site_test_opened_for_training_or_selection": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
        "retraining_after_site_test_authorized": False,
    }
    atomic_json(SELECTION_LOCK, selection_lock)
    selection_lock_record = record(SELECTION_LOCK)

    manifest = {
        "stage": "11C-5J",
        "title": "DEEP DIAGNOSTIC MODEL TRAINING AND CALIBRATION EXECUTION",
        "status": "PASS",
        "trainer_version": VERSION,
        "trainer_source": record(SOURCE),
        "created_at_utc": created_at,
        "architecture": input_evidence[relative(ARCHITECTURE)],
        "training_contract": input_evidence[relative(TRAINING_CONTRACT)],
        "candidate_grid": input_evidence[relative(CANDIDATE_GRID)],
        "environment": environment,
        "task": "PRE-SIMULATION BINARY DETECTABILITY",
        "fault_scope": "PERSISTENT NET-STEM SA0/SA1 ONLY",
        "model_family": "FEEDFORWARD MULTILAYER PERCEPTRON",
        "input_features": EXPECTED_FEATURES,
        "hidden_layers": list(HIDDEN_LAYERS),
        "trainable_parameters": EXPECTED_PARAMETERS,
        "candidate_count": len(candidates),
        "epochs_per_candidate": EPOCHS,
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_metrics["threshold"],
        "selected_metrics": selected_metrics,
        "minimum_validity_status": selected_metrics["minimum_validity_status"],
        "project_target_status": selected_metrics["project_target_status"],
        "deterministic_replay": "PASS",
        "input_evidence": input_evidence,
        "outputs": outputs,
        "selection_lock": selection_lock_record,
        "dev_train_samples": EXPECTED_TRAIN_ROWS,
        "dev_calibration_samples": EXPECTED_CALIBRATION_ROWS,
        "dev_site_test_opened": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = record(MANIFEST)

    audit = {
        "stage": "11C-5J",
        "title": "DEEP DIAGNOSTIC MODEL TRAINING AND CALIBRATION FREEZE",
        "status": "PASS",
        "training_status": "FROZEN",
        "model_selection_status": "FROZEN",
        "threshold_status": "FROZEN",
        "model_family": "FEEDFORWARD MULTILAYER PERCEPTRON",
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_metrics["threshold"],
        "calibration_mcc": selected_metrics["mcc"],
        "calibration_balanced_accuracy": selected_metrics["balanced_accuracy"],
        "calibration_precision": selected_metrics["precision"],
        "calibration_recall": selected_metrics["recall"],
        "calibration_specificity": selected_metrics["specificity"],
        "calibration_f1": selected_metrics["f1_score"],
        "calibration_pr_auc": selected_metrics["pr_auc"],
        "calibration_roc_auc": selected_metrics["roc_auc"],
        "calibration_brier": selected_metrics["brier_score"],
        "minimum_validity_status": selected_metrics["minimum_validity_status"],
        "project_target_status": selected_metrics["project_target_status"],
        "deterministic_replay": "PASS",
        "weights_exact": weights_exact,
        "probabilities_exact": probabilities_exact,
        "threshold_exact": threshold_exact,
        "metrics_exact": metrics_exact,
        "dev_site_test_state": "AUTHORIZED FOR ONE LOCKED DEEP-MODEL EVALUATION",
        "dev_site_test_opened_during_training": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
        "selection_lock": selection_lock_record,
        "manifest": manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "next_gate": "LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION AND BASELINE COMPARISON",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)

    print("\nSTAGE 11C-5J — DEEP DIAGNOSTIC MODEL TRAINING AND CALIBRATION FREEZE")
    print("Status                         : PASS")
    print("Training status                : FROZEN")
    print("Model selection status         : FROZEN")
    print("Threshold status               : FROZEN")
    print("Candidates trained             :", len(candidates))
    print("Epochs per candidate           :", EPOCHS)
    print("Selected candidate             :", selected_id)
    print("Selected threshold             :", format(selected_metrics["threshold"], ".17g"))
    print("Calibration MCC                :", f"{selected_metrics['mcc']:.8f}")
    print("Calibration balanced accuracy  :", f"{selected_metrics['balanced_accuracy']:.8f}")
    print("Calibration precision          :", f"{selected_metrics['precision']:.8f}")
    print("Calibration recall             :", f"{selected_metrics['recall']:.8f}")
    print("Calibration specificity        :", f"{selected_metrics['specificity']:.8f}")
    print("Calibration F1                 :", f"{selected_metrics['f1_score']:.8f}")
    print("Calibration PR-AUC             :", f"{selected_metrics['pr_auc']:.8f}")
    print("Calibration ROC-AUC            :", f"{selected_metrics['roc_auc']:.8f}")
    print("Calibration Brier              :", f"{selected_metrics['brier_score']:.8f}")
    print("Minimum validity               :", selected_metrics["minimum_validity_status"])
    print("Project target                 :", selected_metrics["project_target_status"])
    print("Deterministic replay           : PASS")
    print("Weights exact                  : YES")
    print("Probabilities exact            : YES")
    print("DEV_SITE_TEST opened           : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Canonical dataset modified     : NO")
    print("Feature matrices modified      : NO")
    print("Baseline model modified        : NO")
    print("Selected model                 :", SELECTED_MODEL)
    print("Selected model SHA             :", outputs["selected_model"]["sha256"])
    print("Selection lock                 :", SELECTION_LOCK)
    print("Selection lock SHA             :", selection_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : LOCKED DEEP MODEL DEV_SITE_TEST EVALUATION AND BASELINE COMPARISON")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
