#!/usr/bin/env python3
"""Train, select, replay, and freeze the conventional HMAC baseline."""
from __future__ import annotations

import os

# The frozen contract requires a single numerical worker. These must be set
# before NumPy, SciPy, or scikit-learn is imported.
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
import shutil
import sys
import time
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
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error


VERSION = "HMAC-CONVENTIONAL-BASELINE-TRAINER-v1"
RANDOM_SEED = 20_260_903
CLASSES = np.asarray([0, 1], dtype=np.uint8)

ROOT = Path.cwd().resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"
WORK_ROOT = RESULT_ROOT / "baseline_training_11c5g"
MODEL_ROOT = WORK_ROOT / "candidate_models"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CALIBRATION_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"

CONTRACT_FREEZER = ROOT / "stage_11c5f_baseline_contract.py"
CONTRACT = CONFIG_ROOT / "hmac_conventional_baseline_training_contract_11c5f.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_conventional_baseline_candidate_grid_11c5f.csv"
ENVIRONMENT = RESULT_ROOT / "hmac_conventional_baseline_environment_11c5f.json"
CONTRACT_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_training_contract_freeze_11c5f.json"

CANDIDATE_METRICS_CSV = WORK_ROOT / "hmac_baseline_candidate_calibration_metrics_11c5g.csv"
CANDIDATE_METRICS_JSON = WORK_ROOT / "hmac_baseline_candidate_calibration_metrics_11c5g.json"
DUMMY_METRICS_JSON = WORK_ROOT / "hmac_baseline_dummy_metrics_11c5g.json"
SELECTED_MODEL = WORK_ROOT / "hmac_selected_conventional_baseline_11c5g.joblib"
SELECTED_WEIGHTS = WORK_ROOT / "hmac_selected_conventional_baseline_weights_11c5g.npz"
SELECTED_PREDICTIONS = WORK_ROOT / "hmac_selected_calibration_predictions_11c5g.npz"
CALIBRATION_CURVE = WORK_ROOT / "hmac_selected_calibration_curve_11c5g.csv"
REPLAY_MODEL = WORK_ROOT / "hmac_selected_conventional_baseline_replay_11c5g.joblib"
REPLAY_WEIGHTS = WORK_ROOT / "hmac_selected_conventional_baseline_replay_weights_11c5g.npz"
REPLAY_PREDICTIONS = WORK_ROOT / "hmac_selected_calibration_predictions_replay_11c5g.npz"
SELECTION_LOCK = WORK_ROOT / "hmac_conventional_baseline_selection_lock_11c5g.json"
MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_training_manifest_11c5g.json"
AUDIT = RESULT_ROOT / "hmac_conventional_baseline_training_calibration_freeze_11c5g.json"

EXPECTED_INPUTS = {
    CONTRACT_FREEZER:
        "f6f726e7cde13365b7f7f749c92055e7daa27de0a1b10d2ef58c4ad4c5947a5e",
    CONTRACT:
        "610211065fd81538183472377b0a69aab2f4967cc4f99ada6393712fc090c007",
    CANDIDATE_GRID:
        "5901d7e99a6fc08bfe7e68969caeeb8fd03c3ec0121e80c27d22ebf545e7e132",
    ENVIRONMENT:
        "9e42cceccff6c016f9af5c237a30f400c16ed626d5681159c8348357f7db8e3f",
    CONTRACT_AUDIT:
        "b94314756d4576df9cecc81ecf7e6b11c12c7cab757544bbe693c96a50292fb1",
    ROOT / "stage_11c5e_feature_matrix.py":
        "e0b4f72c1b6c334f683ce84e24b11adafed3786d46340e16f1e7d99084ccd8bb",
    RESULT_ROOT / "hmac_leakage_safe_feature_matrix_manifest_11c5e.json":
        "413cbad01681b4f209de6fce44d3777bccfe424c50c3c95582c627e44965e186",
    RESULT_ROOT / "hmac_leakage_safe_feature_matrix_freeze_11c5e.json":
        "6320bb732b84ae56c937e188189fb87a159d35a164eed892ec3da9d1ee27e540",
    FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json":
        "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    TRAIN_MATRIX:
        "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CALIBRATION_MATRIX:
        "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    SITE_TEST_MATRIX:
        "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
}

EXPECTED_TRAIN_ROWS = 2_046_336
EXPECTED_CALIBRATION_ROWS = 438_528
EXPECTED_SITE_TEST_ROWS = 438_528
EXPECTED_TRAIN_POSITIVES = 894_000
EXPECTED_CALIBRATION_POSITIVES = 192_943
EXPECTED_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512
EXPECTED_CANDIDATES = 6
EXPECTED_EPOCHS = 6
EXPECTED_BATCH_SIZE = 32_768

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

METRICS_CSV_HEADER = [
    "candidate_id",
    "alpha",
    "sample_weighting",
    "epochs",
    "calibration_threshold",
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
    "model_sha256",
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


def atomic_joblib(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3, protocol=4)
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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
    require(actual == expected, f"Python package environment changed: expected {expected}, actual {actual}")
    runtime = frozen.get("training_runtime_contract", {})
    require(runtime.get("parallel_model_candidates") == 1, "parallel candidate contract")
    require(runtime.get("n_jobs") == 1, "n_jobs contract")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        require(os.environ.get(name) == "1", f"thread contract {name}")
    return {"frozen": output_record(ENVIRONMENT), "packages": actual, "threads": 1}


def verify_contract() -> dict:
    contract = load_json(CONTRACT)
    audit = load_json(CONTRACT_AUDIT)
    require(contract.get("status") == "PASS", "training contract status")
    require(
        contract.get("contract_version")
        == "HMAC-CONVENTIONAL-BASELINE-TRAINING-CONTRACT-v1",
        "training contract version",
    )
    require(contract.get("training_execution_authorized") is True, "training authorization")
    require(contract.get("authorized_partition") == "DEV_TRAIN", "authorized partition")
    require(contract.get("dev_calibration_use_authorized") is True, "calibration authorization")
    require(contract.get("dev_site_test_authorized") is False, "site-test authorization")
    require(contract.get("validation_campaign_authorized") is False, "VALIDATION authorization")
    require(contract.get("holdout_campaign_authorized") is False, "HOLDOUT authorization")
    require(contract.get("gnn_training_authorized") is False, "GNN authorization")
    features = contract.get("features", {})
    require(features.get("count") == EXPECTED_FEATURES, "feature count")
    require(features.get("target_in_X") is False, "target isolation")
    require(features.get("post_simulation_features") == [], "post-simulation leakage")
    schedule = contract.get("training_schedule", {})
    require(schedule.get("epochs") == EXPECTED_EPOCHS, "epoch contract")
    require(schedule.get("batch_size") == EXPECTED_BATCH_SIZE, "batch-size contract")
    require(schedule.get("candidate_execution") == "sequential", "candidate execution")
    require(audit.get("status") == "PASS", "contract audit status")
    require(audit.get("contract_status") == "FROZEN", "contract freeze")
    require(audit.get("candidate_grid_status") == "FROZEN", "candidate-grid freeze")
    require(audit.get("dev_site_test_state") == "LOCKED", "site-test lock")
    return contract


def load_candidates(contract: dict) -> list[dict]:
    rows = []
    with CANDIDATE_GRID.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CANDIDATE_HEADER, "candidate-grid header")
        for row in reader:
            parsed = dict(row)
            parsed["alpha"] = float(row["alpha"])
            parsed["epochs"] = int(row["epochs"])
            parsed["batch_size"] = int(row["batch_size"])
            parsed["average"] = int(row["average"])
            parsed["random_seed"] = int(row["random_seed"])
            require(parsed["model_family"] == "sklearn.linear_model.SGDClassifier", "candidate model family")
            require(parsed["loss"] == "log_loss", "candidate loss")
            require(parsed["penalty"] == "l2", "candidate penalty")
            require(parsed["sample_weighting"] in ("NONE", "BALANCED_FROM_DEV_TRAIN"), "candidate weighting")
            require(parsed["epochs"] == EXPECTED_EPOCHS, "candidate epochs")
            require(parsed["batch_size"] == EXPECTED_BATCH_SIZE, "candidate batch size")
            require(parsed["average"] == 1, "candidate averaging")
            require(parsed["random_seed"] == RANDOM_SEED, "candidate seed")
            rows.append(parsed)
    require(len(rows) == EXPECTED_CANDIDATES, "candidate count")
    require(len({row["candidate_id"] for row in rows}) == EXPECTED_CANDIDATES, "candidate IDs")
    require(
        contract.get("candidate_grid", {}).get("sha256") == EXPECTED_INPUTS[CANDIDATE_GRID],
        "contract candidate-grid SHA",
    )
    return rows


def load_matrix(path: Path, partition: str, load_record_ids: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(archive["partition"].tolist() == [partition], f"{partition} identity")
        result = {
            "site_features": np.asarray(archive["site_features"], dtype=np.float32),
            "vector_features": np.asarray(archive["vector_features"], dtype=np.float32),
            "site_row": np.asarray(archive["site_row"], dtype=np.uint16),
            "vector_row": np.asarray(archive["vector_row"], dtype=np.uint8),
            "stuck_value": np.asarray(archive["stuck_value"], dtype=np.uint8),
            "target": np.asarray(archive["target_detected"], dtype=np.uint8),
            "site_ids": np.asarray(archive["site_ids"]),
            "site_indices": np.asarray(archive["site_indices"], dtype=np.uint16),
            "vector_ids": np.asarray(archive["vector_ids"], dtype=np.uint16),
            "feature_columns": np.asarray(archive["feature_columns"]),
        }
        if load_record_ids:
            result["record_id"] = np.asarray(archive["record_id"], dtype=np.uint32)

    expected_rows = (
        EXPECTED_TRAIN_ROWS if partition == "DEV_TRAIN" else EXPECTED_CALIBRATION_ROWS
    )
    expected_sites = 15_987 if partition == "DEV_TRAIN" else 3_426
    expected_positives = (
        EXPECTED_TRAIN_POSITIVES
        if partition == "DEV_TRAIN"
        else EXPECTED_CALIBRATION_POSITIVES
    )
    require(result["site_features"].shape == (expected_sites, EXPECTED_SITE_FEATURES), f"{partition} site matrix")
    require(result["vector_features"].shape == (64, EXPECTED_VECTOR_FEATURES), f"{partition} vector matrix")
    require(result["feature_columns"].shape == (EXPECTED_FEATURES,), f"{partition} feature columns")
    for name in ("site_row", "vector_row", "stuck_value", "target"):
        require(result[name].shape == (expected_rows,), f"{partition} {name} rows")
    require(int(result["target"].sum(dtype=np.uint64)) == expected_positives, f"{partition} positive labels")
    require(np.isin(result["target"], (0, 1)).all(), f"{partition} binary labels")
    require(np.isin(result["stuck_value"], (0, 1)).all(), f"{partition} stuck values")
    require(np.isfinite(result["site_features"]).all(), f"{partition} finite site features")
    return result


def feature_batch(bundle: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    count = len(indices)
    matrix = np.empty((count, EXPECTED_FEATURES), dtype=np.float32)
    matrix[:, 0] = bundle["stuck_value"][indices]
    matrix[:, 1:15] = bundle["site_features"][bundle["site_row"][indices]]
    matrix[:, 15:] = bundle["vector_features"][bundle["vector_row"][indices]]
    require(np.isfinite(matrix).all(), "nonfinite materialized feature batch")
    return matrix


def new_estimator(candidate: dict) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(candidate["alpha"]),
        fit_intercept=True,
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=RANDOM_SEED,
        learning_rate="optimal",
        early_stopping=False,
        average=True,
    )


def epoch_permutation(sample_count: int, epoch_index: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED + epoch_index))
    return rng.permutation(sample_count)


def model_arrays(model: SGDClassifier) -> dict[str, np.ndarray]:
    return {
        "classes": np.asarray(model.classes_),
        "coef": np.asarray(model.coef_),
        "intercept": np.asarray(model.intercept_),
        "n_features_in": np.asarray([model.n_features_in_], dtype="<u4"),
        "t": np.asarray([model.t_], dtype="<f8"),
    }


def models_exact(left: SGDClassifier, right: SGDClassifier) -> bool:
    left_arrays = model_arrays(left)
    right_arrays = model_arrays(right)
    return all(
        left_arrays[name].dtype == right_arrays[name].dtype
        and np.array_equal(left_arrays[name], right_arrays[name])
        for name in left_arrays
    )


def train_candidate(
    candidate: dict,
    train: dict[str, np.ndarray],
    positive_weight: float,
    negative_weight: float,
    resume: bool,
    replay: bool = False,
) -> tuple[SGDClassifier, dict]:
    candidate_id = candidate["candidate_id"]
    model_path = MODEL_ROOT / f"{candidate_id}.joblib"
    state_path = CHECKPOINT_ROOT / f"{candidate_id}.json"
    committed_model_path = None

    if replay:
        model = new_estimator(candidate)
        start_epoch = 0
    elif resume and state_path.is_file():
        state = load_json(state_path)
        require(state.get("candidate_id") == candidate_id, f"{candidate_id} checkpoint identity")
        checkpoint_model = state.get("epoch_model", {})
        require(isinstance(checkpoint_model, dict), f"{candidate_id} checkpoint model record")
        checkpoint_relative = checkpoint_model.get("path")
        require(isinstance(checkpoint_relative, str), f"{candidate_id} checkpoint model path")
        committed_model_path = ROOT / checkpoint_relative
        require(committed_model_path.is_file(), f"{candidate_id} checkpoint model missing")
        require(
            checkpoint_model.get("sha256") == sha256(committed_model_path),
            f"{candidate_id} checkpoint model SHA",
        )
        start_epoch = int(state.get("completed_epochs", -1))
        require(0 <= start_epoch <= EXPECTED_EPOCHS, f"{candidate_id} checkpoint epoch")
        require(
            committed_model_path.name
            == f"{candidate_id}.epoch_{start_epoch:02d}.joblib",
            f"{candidate_id} checkpoint epoch/model agreement",
        )
        model = joblib.load(committed_model_path)
        print(f"  {candidate_id}: RESUME after epoch {start_epoch}/{EXPECTED_EPOCHS}", flush=True)
    else:
        require(
            replay or not model_path.exists(),
            f"uncommitted canonical model without checkpoint for {candidate_id}",
        )
        model = new_estimator(candidate)
        start_epoch = 0

    sample_count = len(train["target"])
    updates_before = start_epoch * sample_count
    start_time = time.monotonic()
    fitted = hasattr(model, "classes_")

    for epoch_index in range(start_epoch, EXPECTED_EPOCHS):
        permutation = epoch_permutation(sample_count, epoch_index)
        batch_count = math.ceil(sample_count / EXPECTED_BATCH_SIZE)
        for batch_number, start in enumerate(
            range(0, sample_count, EXPECTED_BATCH_SIZE),
            start=1,
        ):
            indices = permutation[start:start + EXPECTED_BATCH_SIZE]
            x_batch = feature_batch(train, indices)
            y_batch = train["target"][indices]
            if candidate["sample_weighting"] == "BALANCED_FROM_DEV_TRAIN":
                sample_weight = np.where(
                    y_batch == 1,
                    positive_weight,
                    negative_weight,
                ).astype(np.float64, copy=False)
            else:
                sample_weight = None

            if fitted:
                model.partial_fit(x_batch, y_batch, sample_weight=sample_weight)
            else:
                model.partial_fit(
                    x_batch,
                    y_batch,
                    classes=CLASSES,
                    sample_weight=sample_weight,
                )
                fitted = True

            if batch_number % 16 == 0 or batch_number == batch_count:
                label = "REPLAY" if replay else candidate_id
                print(
                    f"    {label} epoch={epoch_index + 1}/{EXPECTED_EPOCHS} "
                    f"batch={batch_number}/{batch_count}",
                    flush=True,
                )

        del permutation
        if not replay:
            committed_model_path = (
                MODEL_ROOT / f"{candidate_id}.epoch_{epoch_index + 1:02d}.joblib"
            )
            atomic_joblib(committed_model_path, model)
            checkpoint = {
                "stage": "11C-5G",
                "status": "IN_PROGRESS" if epoch_index + 1 < EXPECTED_EPOCHS else "TRAINED",
                "candidate_id": candidate_id,
                "completed_epochs": epoch_index + 1,
                "sample_updates": (epoch_index + 1) * sample_count,
                "epoch_model": output_record(committed_model_path),
                "contract_sha256": EXPECTED_INPUTS[CONTRACT],
                "train_matrix_sha256": EXPECTED_INPUTS[TRAIN_MATRIX],
                "random_seed": RANDOM_SEED,
            }
            atomic_json(state_path, checkpoint)
            print(
                f"  {candidate_id}: CHECKPOINT epoch {epoch_index + 1}/{EXPECTED_EPOCHS} "
                f"sha256={checkpoint['epoch_model']['sha256']}",
                flush=True,
            )

    elapsed = time.monotonic() - start_time
    state = {
        "candidate_id": candidate_id,
        "completed_epochs": EXPECTED_EPOCHS,
        "sample_updates": EXPECTED_EPOCHS * sample_count,
        "sample_updates_this_invocation": EXPECTED_EPOCHS * sample_count - updates_before,
        "elapsed_seconds_this_invocation": elapsed,
    }
    if not replay:
        require(committed_model_path is not None, f"{candidate_id} committed model")
        atomic_copy(committed_model_path, model_path)
        state["model"] = output_record(model_path)
        state["epoch_model"] = output_record(committed_model_path)
        state["checkpoint"] = output_record(state_path)
    return model, state


def predict_probabilities(
    model: SGDClassifier,
    bundle: dict[str, np.ndarray],
) -> np.ndarray:
    sample_count = len(bundle["target"])
    probabilities = np.empty(sample_count, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    for start in range(0, sample_count, EXPECTED_BATCH_SIZE):
        stop_index = min(start + EXPECTED_BATCH_SIZE, sample_count)
        indices = np.arange(start, stop_index, dtype=np.int64)
        x_batch = feature_batch(bundle, indices)
        probabilities[start:stop_index] = model.predict_proba(x_batch)[:, positive_column]
    require(np.isfinite(probabilities).all(), "nonfinite probabilities")
    require(((0.0 <= probabilities) & (probabilities <= 1.0)).all(), "probability range")
    return probabilities


def optimal_threshold(y_true: np.ndarray, probability: np.ndarray) -> dict:
    require(len(y_true) == len(probability), "threshold input length")
    order = np.argsort(-probability, kind="stable")
    sorted_score = probability[order]
    sorted_target = y_true[order].astype(np.int64, copy=False)
    changes = np.flatnonzero(sorted_score[:-1] != sorted_score[1:])
    ends = np.concatenate((changes, np.asarray([len(sorted_score) - 1], dtype=np.int64)))
    cumulative_positive = np.cumsum(sorted_target, dtype=np.int64)
    tp = cumulative_positive[ends].astype(np.float64)
    predicted_positive = (ends + 1).astype(np.float64)
    fp = predicted_positive - tp
    positive_total = float(y_true.sum(dtype=np.uint64))
    negative_total = float(len(y_true) - int(positive_total))
    fn = positive_total - tp
    tn = negative_total - fp
    thresholds = sorted_score[ends]

    all_negative_threshold = np.nextafter(float(sorted_score[0]), math.inf)
    thresholds = np.concatenate(([all_negative_threshold], thresholds))
    tp = np.concatenate(([0.0], tp))
    fp = np.concatenate(([0.0], fp))
    fn = np.concatenate(([positive_total], fn))
    tn = np.concatenate(([negative_total], tn))

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    numerator = tp * tn - fp * fn
    mcc = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) != 0)
    balanced = (recall + specificity) / 2.0
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    ranking = np.lexsort(
        (
            -thresholds,
            np.abs(thresholds - 0.5),
            -recall,
            -f1,
            -balanced,
            -mcc,
        )
    )
    index = int(ranking[0])
    return {
        "threshold": float(thresholds[index]),
        "search_boundaries": int(len(thresholds)),
        "optimized_mcc": float(mcc[index]),
        "optimized_balanced_accuracy": float(balanced[index]),
        "optimized_f1_score": float(f1[index]),
        "optimized_recall": float(recall[index]),
    }


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
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
    return metrics, prediction


def dummy_metrics(y_true: np.ndarray) -> list[dict]:
    prevalence = EXPECTED_TRAIN_POSITIVES / EXPECTED_TRAIN_ROWS
    constant_score = np.full(len(y_true), prevalence, dtype=np.float64)
    majority_prediction = np.zeros(len(y_true), dtype=np.uint8)
    majority = metrics_for_fixed_prediction(y_true, majority_prediction, constant_score)
    majority["dummy_id"] = "DUMMY_MOST_FREQUENT"

    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    stratified_prediction = (rng.random(len(y_true)) < prevalence).astype(np.uint8)
    stratified = metrics_for_fixed_prediction(y_true, stratified_prediction, constant_score)
    stratified["dummy_id"] = "DUMMY_STRATIFIED_TRAIN_PREVALENCE"
    return [majority, stratified]


def metrics_for_fixed_prediction(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def metrics_csv_row(result: dict) -> dict:
    metrics = result["calibration_metrics"]
    confusion = metrics["confusion_matrix"]
    return {
        "candidate_id": result["candidate_id"],
        "alpha": format(result["alpha"], ".17g"),
        "sample_weighting": result["sample_weighting"],
        "epochs": result["epochs"],
        "calibration_threshold": format(metrics["threshold"], ".17g"),
        "mcc": format(metrics["mcc"], ".17g"),
        "balanced_accuracy": format(metrics["balanced_accuracy"], ".17g"),
        "precision": format(metrics["precision"], ".17g"),
        "recall": format(metrics["recall"], ".17g"),
        "specificity": format(metrics["specificity"], ".17g"),
        "f1_score": format(metrics["f1_score"], ".17g"),
        "pr_auc": format(metrics["pr_auc"], ".17g"),
        "roc_auc": format(metrics["roc_auc"], ".17g"),
        "brier_score": format(metrics["brier_score"], ".17g"),
        "true_negative": confusion["true_negative"],
        "false_positive": confusion["false_positive"],
        "false_negative": confusion["false_negative"],
        "true_positive": confusion["true_positive"],
        "model_sha256": result["model"]["sha256"],
    }


def selection_key(result: dict) -> tuple:
    metrics = result["calibration_metrics"]
    return (
        -metrics["mcc"],
        -metrics["balanced_accuracy"],
        -metrics["f1_score"],
        -metrics["recall"],
        abs(metrics["threshold"] - 0.5),
        result["candidate_id"],
    )


def write_prediction_artifact(
    path: Path,
    calibration: dict[str, np.ndarray],
    probability: np.ndarray,
    prediction: np.ndarray,
    candidate_id: str,
    threshold: float,
) -> None:
    deterministic_npz(
        path,
        {
            "candidate_id": np.asarray([candidate_id]),
            "partition": np.asarray(["DEV_CALIBRATION"]),
            "prediction": prediction.astype(np.uint8, copy=False),
            "probability": probability.astype("<f8", copy=False),
            "record_id": calibration["record_id"].astype("<u4", copy=False),
            "site_row": calibration["site_row"].astype("<u2", copy=False),
            "stuck_value": calibration["stuck_value"].astype(np.uint8, copy=False),
            "target_detected": calibration["target"].astype(np.uint8, copy=False),
            "threshold": np.asarray([threshold], dtype="<f8"),
            "vector_row": calibration["vector_row"].astype(np.uint8, copy=False),
        },
    )


def write_calibration_curve(
    path: Path,
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 20,
) -> None:
    """Write a deterministic equal-count calibration curve diagnostic."""
    require(len(y_true) == len(probability), "calibration-curve input length")
    require(bins > 1 and bins <= len(y_true), "calibration-curve bin count")
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, bins)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "bin_index",
                "sample_count",
                "minimum_probability",
                "maximum_probability",
                "mean_predicted_probability",
                "observed_positive_fraction",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for bin_index, indices in enumerate(groups):
            require(len(indices) > 0, "empty calibration-curve bin")
            bin_probability = probability[indices]
            bin_target = y_true[indices]
            writer.writerow(
                {
                    "bin_index": bin_index,
                    "sample_count": len(indices),
                    "minimum_probability": format(float(bin_probability[0]), ".17g"),
                    "maximum_probability": format(float(bin_probability[-1]), ".17g"),
                    "mean_predicted_probability": format(
                        float(bin_probability.mean(dtype=np.float64)),
                        ".17g",
                    ),
                    "observed_positive_fraction": format(
                        float(bin_target.mean(dtype=np.float64)),
                        ".17g",
                    ),
                }
            )
    temporary.replace(path)


def assessment(metrics: dict, dummy_results: list[dict], prevalence: float) -> tuple[str, str, dict]:
    best_dummy_mcc = max(item["mcc"] for item in dummy_results)
    minimum_checks = {
        "mcc_gt_zero": metrics["mcc"] > 0.0,
        "balanced_accuracy_gt_half": metrics["balanced_accuracy"] > 0.5,
        "pr_auc_gt_prevalence": metrics["pr_auc"] > prevalence,
        "mcc_gt_both_dummy_references": metrics["mcc"] > best_dummy_mcc,
    }
    target_checks = {
        "mcc_ge_0_40": metrics["mcc"] >= 0.40,
        "balanced_accuracy_ge_0_70": metrics["balanced_accuracy"] >= 0.70,
        "f1_ge_0_70": metrics["f1_score"] >= 0.70,
        "recall_ge_0_70": metrics["recall"] >= 0.70,
    }
    status = {
        "minimum_validity_checks": minimum_checks,
        "project_target_checks": target_checks,
    }
    return (
        "PASS" if all(minimum_checks.values()) else "NOT_MET",
        "PASS" if all(target_checks.values()) else "NOT_MET",
        status,
    )


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(not AUDIT.exists(), f"Stage 11C-5G already frozen: {AUDIT}")
    require(not MANIFEST.exists(), f"Stage 11C-5G manifest already exists: {MANIFEST}")
    require(not SELECTION_LOCK.exists(), f"Stage 11C-5G selection already locked: {SELECTION_LOCK}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11C-5G — CONVENTIONAL BASELINE TRAINING AND CALIBRATION")
    print("FROZEN INPUT VERIFICATION")
    input_evidence = {}
    for path, expected in EXPECTED_INPUTS.items():
        record = verify_file(path, expected)
        input_evidence[record["path"]] = record
        print(f"  {path.name:<66} : OK", flush=True)

    contract = verify_contract()
    environment = verify_environment()
    candidates = load_candidates(contract)

    print("\nLOADING AUTHORIZED MATRICES", flush=True)
    train = load_matrix(TRAIN_MATRIX, "DEV_TRAIN", load_record_ids=False)
    calibration = load_matrix(
        CALIBRATION_MATRIX,
        "DEV_CALIBRATION",
        load_record_ids=True,
    )
    require(
        np.array_equal(train["feature_columns"], calibration["feature_columns"]),
        "TRAIN/CALIBRATION feature columns",
    )
    require(
        np.array_equal(train["vector_ids"], calibration["vector_ids"]),
        "TRAIN/CALIBRATION vector identities",
    )
    require(
        not (set(train["site_ids"].tolist()) & set(calibration["site_ids"].tolist())),
        "TRAIN/CALIBRATION site overlap",
    )
    print("  DEV_TRAIN       : LOADED FOR MODEL FITTING", flush=True)
    print("  DEV_CALIBRATION : LOADED FOR SELECTION AND THRESHOLD", flush=True)
    print("  DEV_SITE_TEST   : NOT OPENED FOR EVALUATION", flush=True)
    print("  VALIDATION      : NOT PRESENT", flush=True)
    print("  HOLDOUT_TEST    : NOT PRESENT", flush=True)

    weighting = contract["sample_weighting"]["BALANCED_FROM_DEV_TRAIN"]
    positive_weight = float(weighting["positive"])
    negative_weight = float(weighting["negative"])
    require(math.isfinite(positive_weight) and positive_weight > 0, "positive class weight")
    require(math.isfinite(negative_weight) and negative_weight > 0, "negative class weight")

    print("\nSEQUENTIAL CANDIDATE TRAINING", flush=True)
    candidate_results = []
    models = {}
    probabilities = {}
    for candidate_number, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["candidate_id"]
        print(
            f"CANDIDATE {candidate_number}/{EXPECTED_CANDIDATES}: {candidate_id} "
            f"alpha={candidate['alpha']:.1e} weighting={candidate['sample_weighting']}",
            flush=True,
        )
        model, training_state = train_candidate(
            candidate,
            train,
            positive_weight,
            negative_weight,
            resume=True,
        )
        score = predict_probabilities(model, calibration)
        threshold_search = optimal_threshold(calibration["target"], score)
        metrics, prediction = classification_metrics(
            calibration["target"],
            score,
            threshold_search["threshold"],
        )
        model_path = MODEL_ROOT / f"{candidate_id}.joblib"
        result = {
            "candidate_id": candidate_id,
            "alpha": float(candidate["alpha"]),
            "sample_weighting": candidate["sample_weighting"],
            "epochs": EXPECTED_EPOCHS,
            "training": training_state,
            "threshold_search": threshold_search,
            "calibration_metrics": metrics,
            "model": output_record(model_path),
        }
        candidate_results.append(result)
        models[candidate_id] = model
        probabilities[candidate_id] = score
        print(
            f"  CALIBRATION threshold={metrics['threshold']:.9g} "
            f"MCC={metrics['mcc']:.6f} balanced={metrics['balanced_accuracy']:.6f} "
            f"F1={metrics['f1_score']:.6f} recall={metrics['recall']:.6f}",
            flush=True,
        )

    dummy_results = dummy_metrics(calibration["target"])
    atomic_json(
        DUMMY_METRICS_JSON,
        {
            "stage": "11C-5G",
            "status": "PASS",
            "partition": "DEV_CALIBRATION",
            "train_positive_prevalence": EXPECTED_TRAIN_POSITIVES / EXPECTED_TRAIN_ROWS,
            "random_seed": RANDOM_SEED,
            "dummy_results": dummy_results,
        },
    )

    metrics_temporary = CANDIDATE_METRICS_CSV.with_name(
        CANDIDATE_METRICS_CSV.name + ".tmp"
    )
    with metrics_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=METRICS_CSV_HEADER,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metrics_csv_row(result) for result in candidate_results)
    metrics_temporary.replace(CANDIDATE_METRICS_CSV)
    atomic_json(
        CANDIDATE_METRICS_JSON,
        {
            "stage": "11C-5G",
            "status": "PASS",
            "partition": "DEV_CALIBRATION",
            "selection_metric": "MCC",
            "candidate_results": candidate_results,
        },
    )

    selected = sorted(candidate_results, key=selection_key)[0]
    selected_id = selected["candidate_id"]
    selected_model_source = MODEL_ROOT / f"{selected_id}.joblib"
    selected_model = models[selected_id]
    selected_probability = probabilities[selected_id]
    selected_threshold = float(selected["calibration_metrics"]["threshold"])
    selected_prediction = (selected_probability >= selected_threshold).astype(np.uint8)

    atomic_copy(selected_model_source, SELECTED_MODEL)
    deterministic_npz(SELECTED_WEIGHTS, model_arrays(selected_model))
    write_prediction_artifact(
        SELECTED_PREDICTIONS,
        calibration,
        selected_probability,
        selected_prediction,
        selected_id,
        selected_threshold,
    )
    write_calibration_curve(
        CALIBRATION_CURVE,
        calibration["target"],
        selected_probability,
    )

    print(f"\nDETERMINISTIC SELECTED-MODEL REPLAY: {selected_id}", flush=True)
    selected_candidate = next(
        candidate for candidate in candidates if candidate["candidate_id"] == selected_id
    )
    replay_model, replay_state = train_candidate(
        selected_candidate,
        train,
        positive_weight,
        negative_weight,
        resume=False,
        replay=True,
    )
    replay_probability = predict_probabilities(replay_model, calibration)
    replay_prediction = (replay_probability >= selected_threshold).astype(np.uint8)
    require(models_exact(selected_model, replay_model), "selected model coefficient replay")
    require(np.array_equal(selected_probability, replay_probability), "selected probability replay")
    require(np.array_equal(selected_prediction, replay_prediction), "selected prediction replay")

    atomic_joblib(REPLAY_MODEL, replay_model)
    deterministic_npz(REPLAY_WEIGHTS, model_arrays(replay_model))
    write_prediction_artifact(
        REPLAY_PREDICTIONS,
        calibration,
        replay_probability,
        replay_prediction,
        selected_id,
        selected_threshold,
    )
    require(sha256(SELECTED_WEIGHTS) == sha256(REPLAY_WEIGHTS), "weight artifact replay")
    require(
        sha256(SELECTED_PREDICTIONS) == sha256(REPLAY_PREDICTIONS),
        "prediction artifact replay",
    )

    calibration_prevalence = EXPECTED_CALIBRATION_POSITIVES / EXPECTED_CALIBRATION_ROWS
    minimum_status, target_status, checks = assessment(
        selected["calibration_metrics"],
        dummy_results,
        calibration_prevalence,
    )

    selection_lock = {
        "stage": "11C-5G",
        "title": "CONVENTIONAL BASELINE SELECTION AND THRESHOLD LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate_id": selected_id,
        "selection_partition": "DEV_CALIBRATION",
        "selection_metric": "MCC",
        "selected_threshold": selected_threshold,
        "selected_calibration_metrics": selected["calibration_metrics"],
        "threshold_search": selected["threshold_search"],
        "minimum_validity_status": minimum_status,
        "project_target_status_on_calibration": target_status,
        "acceptance_checks": checks,
        "selected_model": output_record(SELECTED_MODEL),
        "selected_weights": output_record(SELECTED_WEIGHTS),
        "selected_calibration_predictions": output_record(SELECTED_PREDICTIONS),
        "selected_calibration_curve": output_record(CALIBRATION_CURVE),
        "deterministic_replay": {
            "status": "PASS",
            "coefficients_exact": True,
            "probabilities_exact": True,
            "predictions_exact": True,
            "replay_model": output_record(REPLAY_MODEL),
            "replay_weights": output_record(REPLAY_WEIGHTS),
            "replay_predictions": output_record(REPLAY_PREDICTIONS),
            "replay_training": replay_state,
        },
        "candidate_metrics_csv": output_record(CANDIDATE_METRICS_CSV),
        "candidate_metrics_json": output_record(CANDIDATE_METRICS_JSON),
        "dummy_metrics": output_record(DUMMY_METRICS_JSON),
        "dev_site_test_opened_for_evaluation": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "retraining_after_site_test_authorized": False,
        "contract_sha256": EXPECTED_INPUTS[CONTRACT],
        "train_matrix_sha256": EXPECTED_INPUTS[TRAIN_MATRIX],
        "calibration_matrix_sha256": EXPECTED_INPUTS[CALIBRATION_MATRIX],
    }
    atomic_json(SELECTION_LOCK, selection_lock)
    selection_lock_record = output_record(SELECTION_LOCK)

    candidate_model_records = {
        result["candidate_id"]: result["model"] for result in candidate_results
    }
    manifest = {
        "stage": "11C-5G",
        "title": "CONVENTIONAL BASELINE TRAINING AND CALIBRATION EXECUTION",
        "status": "PASS",
        "trainer_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_simulation_persistent_fault_detectability",
        "fault_scope": "persistent net-stem SA0/SA1 only",
        "train_samples": EXPECTED_TRAIN_ROWS,
        "calibration_samples": EXPECTED_CALIBRATION_ROWS,
        "candidate_count": EXPECTED_CANDIDATES,
        "epochs_per_candidate": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "candidate_training_sample_updates": (
            EXPECTED_CANDIDATES * EXPECTED_EPOCHS * EXPECTED_TRAIN_ROWS
        ),
        "selected_replay_sample_updates": EXPECTED_EPOCHS * EXPECTED_TRAIN_ROWS,
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_threshold,
        "selected_calibration_metrics": selected["calibration_metrics"],
        "minimum_validity_status": minimum_status,
        "project_target_status_on_calibration": target_status,
        "deterministic_replay": "PASS",
        "candidate_models": candidate_model_records,
        "candidate_metrics_csv": output_record(CANDIDATE_METRICS_CSV),
        "candidate_metrics_json": output_record(CANDIDATE_METRICS_JSON),
        "dummy_metrics": output_record(DUMMY_METRICS_JSON),
        "selected_model": output_record(SELECTED_MODEL),
        "selected_weights": output_record(SELECTED_WEIGHTS),
        "selected_calibration_predictions": output_record(SELECTED_PREDICTIONS),
        "selected_calibration_curve": output_record(CALIBRATION_CURVE),
        "replay_model": output_record(REPLAY_MODEL),
        "replay_weights": output_record(REPLAY_WEIGHTS),
        "replay_calibration_predictions": output_record(REPLAY_PREDICTIONS),
        "selection_lock": selection_lock_record,
        "input_evidence": input_evidence,
        "environment": environment,
        "dev_site_test_opened_for_evaluation": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = output_record(MANIFEST)

    audit = {
        "stage": "11C-5G",
        "title": "CONVENTIONAL BASELINE TRAINING AND CALIBRATION EXECUTION FREEZE",
        "status": "PASS",
        "execution_status": "FROZEN",
        "model_selection_status": "FROZEN",
        "threshold_status": "FROZEN",
        "baseline_model_family": "INCREMENTAL LOGISTIC REGRESSION",
        "candidates_trained": EXPECTED_CANDIDATES,
        "candidate_epochs": EXPECTED_EPOCHS,
        "candidate_training_sample_updates": (
            EXPECTED_CANDIDATES * EXPECTED_EPOCHS * EXPECTED_TRAIN_ROWS
        ),
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_threshold,
        "calibration_mcc": selected["calibration_metrics"]["mcc"],
        "calibration_balanced_accuracy": selected["calibration_metrics"]["balanced_accuracy"],
        "calibration_precision": selected["calibration_metrics"]["precision"],
        "calibration_recall": selected["calibration_metrics"]["recall"],
        "calibration_specificity": selected["calibration_metrics"]["specificity"],
        "calibration_f1": selected["calibration_metrics"]["f1_score"],
        "calibration_pr_auc": selected["calibration_metrics"]["pr_auc"],
        "calibration_roc_auc": selected["calibration_metrics"]["roc_auc"],
        "calibration_brier_score": selected["calibration_metrics"]["brier_score"],
        "calibration_curve": output_record(CALIBRATION_CURVE),
        "minimum_validity_status": minimum_status,
        "project_target_status_on_calibration": target_status,
        "deterministic_replay": "PASS",
        "coefficient_replay": "EXACT",
        "probability_replay": "EXACT",
        "prediction_replay": "EXACT",
        "target_in_features": False,
        "post_simulation_features": 0,
        "dev_train_used_for_fitting": True,
        "dev_calibration_used_for_selection": True,
        "dev_site_test_opened_for_evaluation": False,
        "dev_site_test_state": "AUTHORIZED FOR ONE LOCKED EVALUATION",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized": False,
        "selection_lock": selection_lock_record,
        "manifest": manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "next_gate": "LOCKED DEV_SITE_TEST BASELINE EVALUATION",
    }
    atomic_json(AUDIT, audit)
    audit_record = output_record(AUDIT)

    print("\nSTAGE 11C-5G — CONVENTIONAL BASELINE TRAINING AND CALIBRATION EXECUTION FREEZE")
    print("Status                         : PASS")
    print("Execution status               : FROZEN")
    print("Candidates trained             : 6/6")
    print("Epochs per candidate           :", EXPECTED_EPOCHS)
    print("Training samples               :", EXPECTED_TRAIN_ROWS)
    print("Calibration samples            :", EXPECTED_CALIBRATION_ROWS)
    print("Selected candidate             :", selected_id)
    print("Selected threshold             :", format(selected_threshold, ".17g"))
    print("Calibration MCC                :", f"{selected['calibration_metrics']['mcc']:.8f}")
    print("Calibration balanced accuracy  :", f"{selected['calibration_metrics']['balanced_accuracy']:.8f}")
    print("Calibration precision          :", f"{selected['calibration_metrics']['precision']:.8f}")
    print("Calibration recall             :", f"{selected['calibration_metrics']['recall']:.8f}")
    print("Calibration specificity        :", f"{selected['calibration_metrics']['specificity']:.8f}")
    print("Calibration F1                 :", f"{selected['calibration_metrics']['f1_score']:.8f}")
    print("Calibration PR-AUC             :", f"{selected['calibration_metrics']['pr_auc']:.8f}")
    print("Calibration ROC-AUC            :", f"{selected['calibration_metrics']['roc_auc']:.8f}")
    print("Calibration Brier score        :", f"{selected['calibration_metrics']['brier_score']:.8f}")
    print("Minimum validity               :", minimum_status)
    print("Project target on calibration  :", target_status)
    print("Deterministic replay           : PASS")
    print("Coefficient replay             : EXACT")
    print("Probability replay             : EXACT")
    print("Prediction replay              : EXACT")
    print("DEV_SITE_TEST evaluated        : NO")
    print("DEV_SITE_TEST next use         : ONE LOCKED EVALUATION")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Canonical dataset modified     : NO")
    print("Feature matrices modified      : NO")
    print("Selected model                 :", SELECTED_MODEL)
    print("Selected model SHA             :", sha256(SELECTED_MODEL))
    print("Calibration curve              :", CALIBRATION_CURVE)
    print("Calibration curve SHA          :", sha256(CALIBRATION_CURVE))
    print("Selection lock                 :", SELECTION_LOCK)
    print("Selection lock SHA             :", selection_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : LOCKED DEV_SITE_TEST BASELINE EVALUATION")


if __name__ == "__main__":
    main()
