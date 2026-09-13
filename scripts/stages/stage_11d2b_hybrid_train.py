#!/usr/bin/env python3
"""Train, calibrate, replay, select, and freeze the Stage 11D-2A hybrid MLP."""
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
import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error

try:
    import stage_11c5j_deep_model_train as deep_helpers
except ImportError as error:
    raise SystemExit(
        "STOP: frozen Stage 11C-5J training helpers are required in the project root."
    ) from error


VERSION = "HMAC-HYBRID-TRAINER-v1"
RANDOM_SEED = 20_260_910
EXPECTED_SAMPLE_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512
EXPECTED_GRAPH_FEATURES = 119
EXPECTED_COMBINED_FEATURES = 646
EXPECTED_VECTORS = 64
EXPECTED_SAMPLES_PER_SITE = 128
EXPECTED_TRAIN_SITES = 15_987
EXPECTED_CALIBRATION_SITES = 3_426
EXPECTED_TRAIN_ROWS = 2_046_336
EXPECTED_CALIBRATION_ROWS = 438_528
EXPECTED_TRAIN_POSITIVES = 894_000
EXPECTED_CALIBRATION_POSITIVES = 192_943
EXPECTED_GRAPH_NODES = 22_839
TRAINING_BATCH_SIZE = 8_192
INFERENCE_BATCH_SIZE = 32_768
EPOCHS = 4
HIDDEN_LAYERS = (128, 64, 32)
EXPECTED_PARAMETERS = 93_185
EXPECTED_CANDIDATES = 2
FROZEN_GNN_MCC = 0.33662641
REQUIRED_HYBRID_GAIN = 0.02

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
RESULT_11D1 = ROOT / "results/hmac_fault_campaign_11d1"
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11d2"
FEATURE_ROOT = RESULT_11C5 / "feature_matrix_11c5e"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
WORK_ROOT = RESULT_ROOT / "hybrid_training_11d2b"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CALIBRATION_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"

PROPAGATION_CACHE = (
    RESULT_11D1 / "gnn_training_11d1c/hmac_directed_sgc_graph_features_11d1c.npz"
)
GNN_SELECTION_LOCK = (
    RESULT_11D1 / "gnn_training_11d1c/hmac_gnn_selection_lock_11d1c.json"
)
GNN_EVALUATION_AUDIT = RESULT_11D1 / "hmac_gnn_site_test_comparator_freeze_11d1d.json"
HYBRID_READINESS_AUDIT = RESULT_11D1 / "hmac_gnn_disposition_hybrid_readiness_freeze_11d1e.json"

DEEP_HELPERS = ROOT / "stage_11c5j_deep_model_train.py"
CONTRACT_SOURCE = ROOT / "stage_11d2a_hybrid_contract.py"
ARCHITECTURE = CONFIG_ROOT / "hmac_hybrid_architecture_11d2a.json"
TRAINING_CONTRACT = CONFIG_ROOT / "hmac_hybrid_training_contract_11d2a.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_hybrid_candidate_grid_11d2a.csv"
CONTRACT_ENVIRONMENT = RESULT_ROOT / "hmac_hybrid_environment_11d2a.json"
CONTRACT_AUDIT = RESULT_ROOT / "hmac_hybrid_architecture_training_contract_freeze_11d2a.json"

CANDIDATE_METRICS_CSV = WORK_ROOT / "hmac_hybrid_candidate_calibration_metrics_11d2b.csv"
CANDIDATE_METRICS_JSON = WORK_ROOT / "hmac_hybrid_candidate_calibration_metrics_11d2b.json"
TRAINING_HISTORY = WORK_ROOT / "hmac_hybrid_training_history_11d2b.csv"
SELECTED_MODEL = WORK_ROOT / "hmac_selected_hybrid_model_11d2b.joblib"
SELECTED_WEIGHTS = WORK_ROOT / "hmac_selected_hybrid_weights_11d2b.npz"
SELECTED_PREDICTIONS = WORK_ROOT / "hmac_selected_hybrid_calibration_predictions_11d2b.npz"
CALIBRATION_CURVE = WORK_ROOT / "hmac_selected_hybrid_calibration_curve_11d2b.csv"
REPLAY_MODEL = WORK_ROOT / "hmac_selected_hybrid_replay_model_11d2b.joblib"
REPLAY_WEIGHTS = WORK_ROOT / "hmac_selected_hybrid_replay_weights_11d2b.npz"
REPLAY_PREDICTIONS = WORK_ROOT / "hmac_selected_hybrid_calibration_replay_predictions_11d2b.npz"
SELECTION_LOCK = WORK_ROOT / "hmac_hybrid_selection_lock_11d2b.json"
MANIFEST = RESULT_ROOT / "hmac_hybrid_training_manifest_11d2b.json"
AUDIT = RESULT_ROOT / "hmac_hybrid_training_calibration_freeze_11d2b.json"

EXPECTED_INPUTS = {
    TRAIN_MATRIX: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CALIBRATION_MATRIX: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    GNN_SELECTION_LOCK: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
    GNN_EVALUATION_AUDIT: "3555783dc613aec71a6f89c09b6c487b514aa0cfc68a6c7b42489018dbdb5a1b",
    HYBRID_READINESS_AUDIT: "de89484ec1c244b141950ad259ad69163562ac70fd31e9659c8943c007e48339",
    DEEP_HELPERS: "9101eab7fc704ddd7baa3b8619a95f967eef5bec0f2ee96926eb9608e13c4470",
    CONTRACT_SOURCE: "efee27d3a3ab38be66462a20b084d820eb28744c57b4d427b9b8239a0da1b748",
    ARCHITECTURE: "d213a89c53769c217ca9cca71700b03abc23f94d5302bc4309ea63b035016063",
    TRAINING_CONTRACT: "1ccd69a6ee179dbdbce9cc295324c85f28bc4fd98f22b5df010fdcbf21cda41e",
    CANDIDATE_GRID: "3aaea24ec28c4680ebd24ecef345ab82ab910c07a6f5240dc762f196c8032bbd",
    CONTRACT_ENVIRONMENT: "1849274a8006e77cced8695be784e70102abceec940fee036163f0a62aa65a57",
    CONTRACT_AUDIT: "cbe3f362986a86506f9252253e06a2a8a2ec3541e9bc92655b75e05fcc082e65",
}

CANDIDATE_COLUMNS = [
    "candidate_id", "model_family", "fusion_mode", "sample_features",
    "graph_features", "combined_features", "hidden_layer_sizes", "activation",
    "solver", "alpha", "learning_rate_init", "batch_size", "epochs", "shuffle",
    "early_stopping", "random_seed", "trainable_parameters",
]

METRIC_FIELDS = deep_helpers.METRIC_FIELDS
HISTORY_FIELDS = deep_helpers.HISTORY_FIELDS


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def close(actual: object, expected: float, tolerance: float = 5e-8) -> bool:
    """Match the numerical tolerance used by the frozen 11D-1E/11D-2A gates."""
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


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
    print(f"  {path.name:<76}: OK", flush=True)
    return record(path)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {name}")


def verify_outputs_absent() -> None:
    for path in (
        CANDIDATE_METRICS_CSV, CANDIDATE_METRICS_JSON, TRAINING_HISTORY,
        SELECTED_MODEL, SELECTED_WEIGHTS, SELECTED_PREDICTIONS, CALIBRATION_CURVE,
        REPLAY_MODEL, REPLAY_WEIGHTS, REPLAY_PREDICTIONS, SELECTION_LOCK, MANIFEST, AUDIT,
    ):
        require(not path.exists(), f"Stage 11D-2B output already exists: {relative(path)}")


def checkpoint_paths(candidate_id: str, replay: bool) -> tuple[Path, Path]:
    suffix = "replay" if replay else "canonical"
    base = CHECKPOINT_ROOT / f"{candidate_id}_{suffix}"
    return base.with_suffix(".joblib"), base.with_suffix(".json")


def load_checkpoint(
    candidate: dict, replay: bool
) -> tuple[MLPClassifier | None, int, list[dict]]:
    model_path, metadata_path = checkpoint_paths(candidate["candidate_id"], replay)
    if not model_path.exists() and not metadata_path.exists():
        return None, 0, []
    require(model_path.is_file() and metadata_path.is_file(), "incomplete hybrid checkpoint")
    metadata = load_json(metadata_path)
    require(metadata.get("status") == "IN_PROGRESS", "checkpoint status")
    require(metadata.get("candidate") == candidate, "checkpoint candidate")
    require(metadata.get("replay") is replay, "checkpoint replay identity")
    completed = int(metadata.get("completed_epochs", -1))
    require(0 <= completed <= EPOCHS, "checkpoint epoch range")
    history = metadata.get("history")
    require(isinstance(history, list) and len(history) == completed, "checkpoint history")
    require(metadata.get("model_sha256") == sha256(model_path), "checkpoint model SHA")
    model = joblib.load(model_path)
    require(isinstance(model, MLPClassifier), "checkpoint model type")
    print(
        f"  RESUME {candidate['candidate_id']} "
        f"{'replay' if replay else 'canonical'} after epoch {completed}",
        flush=True,
    )
    return model, completed, history


def save_checkpoint(
    candidate: dict,
    replay: bool,
    model: MLPClassifier,
    completed_epochs: int,
    history: list[dict],
) -> None:
    model_path, metadata_path = checkpoint_paths(candidate["candidate_id"], replay)
    deep_helpers.atomic_joblib(model_path, model)
    atomic_json(metadata_path, {
        "stage": "11D-2B",
        "status": "IN_PROGRESS",
        "candidate": candidate,
        "replay": replay,
        "completed_epochs": completed_epochs,
        "history": history,
        "model_sha256": sha256(model_path),
        "contract_sha256": EXPECTED_INPUTS[TRAINING_CONTRACT],
        "train_matrix_sha256": EXPECTED_INPUTS[TRAIN_MATRIX],
        "propagation_cache_sha256": sha256(PROPAGATION_CACHE),
    })


def verify_environment() -> dict:
    frozen = load_json(CONTRACT_ENVIRONMENT)
    require(frozen.get("status") == "PASS", "contract environment status")
    require(frozen.get("environment_status") == "FROZEN", "contract environment freeze")
    actual = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": package_version("threadpoolctl"),
    }
    require(actual == frozen.get("packages"), f"package environment changed: {actual}")
    require(
        frozen.get("estimator_backend") == "sklearn.neural_network.MLPClassifier",
        "hybrid estimator backend",
    )
    return {
        "status": "PASS",
        "packages": actual,
        "thread_limits": frozen.get("thread_limits"),
        "contract_environment": record(CONTRACT_ENVIRONMENT),
    }


def verify_contract() -> tuple[dict, dict]:
    architecture = load_json(ARCHITECTURE)
    contract = load_json(TRAINING_CONTRACT)
    audit = load_json(CONTRACT_AUDIT)
    gnn = load_json(GNN_EVALUATION_AUDIT)
    readiness = load_json(HYBRID_READINESS_AUDIT)

    require(architecture.get("status") == "PASS", "hybrid architecture status")
    require(architecture.get("architecture_status") == "FROZEN", "architecture freeze")
    require(architecture.get("model_family") == "GRAPH_AUGMENTED_FEATURE_FUSION_MLP", "model family")
    require(architecture.get("is_hybrid_model") is True, "hybrid identity")
    require(architecture.get("fusion_level") == "FEATURE_LEVEL", "fusion level")
    require(architecture.get("fusion", {}).get("input_features") == EXPECTED_COMBINED_FEATURES, "fusion width")
    require(architecture.get("fusion", {}).get("hidden_layers") == list(HIDDEN_LAYERS), "hidden layers")
    require(architecture.get("fusion", {}).get("trainable_parameters") == EXPECTED_PARAMETERS, "parameters")
    require(architecture.get("frozen_comparator_scores_used_as_inputs") is False, "score stacking blocked")
    require(architecture.get("dev_site_test_predictions_used_as_inputs") is False, "site-test prediction leakage")

    require(contract.get("status") == "PASS", "hybrid contract status")
    require(contract.get("training_contract_status") == "FROZEN", "training contract freeze")
    require(contract.get("selection_partition") == "DEV_CALIBRATION", "selection partition")
    require(contract.get("primary_metric") == "MCC", "selection metric")
    require(contract.get("epochs_per_candidate") == EPOCHS, "epochs")
    require(contract.get("training_batch_size") == TRAINING_BATCH_SIZE, "batch size")
    require(contract.get("class_weighting") == "GLOBAL_DEV_TRAIN_BALANCED_RESAMPLING", "balancing")
    require(contract.get("dev_site_test_opening_authorized") is False, "site-test lock")
    require(contract.get("validation_vectors_exposed") == 0, "validation exposure")
    require(contract.get("holdout_vectors_exposed") == 0, "holdout exposure")
    require(
        contract.get("hybrid_advancement_target", {}).get("required_point_improvement")
        == REQUIRED_HYBRID_GAIN,
        "hybrid gain target",
    )

    require(audit.get("status") == "PASS", "contract audit status")
    require(audit.get("hybrid_training_authorized_for_dev_train") is True, "training authorization")
    require(audit.get("dev_site_test_state") == "LOCKED", "audit site-test lock")
    require(close(audit.get("frozen_gnn_mcc"), FROZEN_GNN_MCC), "frozen GNN MCC")
    require(close(gnn.get("gnn_mcc"), FROZEN_GNN_MCC), "GNN audit MCC")
    require(gnn.get("project_target_status") == "MET", "GNN project target")
    require(readiness.get("winner") == "GNN", "readiness winner")
    require(readiness.get("hybrid_architecture_contract") == "AUTHORIZED", "readiness authorization")
    return architecture, contract


def parse_candidates() -> list[dict]:
    candidates = []
    with CANDIDATE_GRID.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CANDIDATE_COLUMNS, "candidate-grid header")
        for row in reader:
            candidate = dict(row)
            for key in (
                "sample_features", "graph_features", "combined_features", "batch_size",
                "epochs", "random_seed", "trainable_parameters",
            ):
                candidate[key] = int(candidate[key])
            candidate["alpha"] = float(candidate["alpha"])
            candidate["learning_rate_init"] = float(candidate["learning_rate_init"])
            require(candidate["model_family"] == "GRAPH_AUGMENTED_FEATURE_FUSION_MLP", "candidate family")
            require(candidate["fusion_mode"] == "FEATURE_LEVEL_CONCATENATION", "candidate fusion")
            require(candidate["sample_features"] == EXPECTED_SAMPLE_FEATURES, "candidate sample width")
            require(candidate["graph_features"] == EXPECTED_GRAPH_FEATURES, "candidate graph width")
            require(candidate["combined_features"] == EXPECTED_COMBINED_FEATURES, "candidate input width")
            require(candidate["hidden_layer_sizes"] == "128;64;32", "candidate layers")
            require(candidate["activation"] == "relu" and candidate["solver"] == "adam", "candidate backend")
            require(candidate["batch_size"] == TRAINING_BATCH_SIZE, "candidate batch size")
            require(candidate["epochs"] == EPOCHS, "candidate epochs")
            require(candidate["shuffle"] == "false", "candidate internal shuffle")
            require(candidate["early_stopping"] == "false", "candidate early stopping")
            require(candidate["random_seed"] == RANDOM_SEED, "candidate seed")
            require(candidate["trainable_parameters"] == EXPECTED_PARAMETERS, "candidate parameters")
            candidates.append(candidate)
    require(len(candidates) == EXPECTED_CANDIDATES, "candidate count")
    require(len({item["candidate_id"] for item in candidates}) == EXPECTED_CANDIDATES, "candidate IDs")
    return candidates


def load_graph_features() -> tuple[np.ndarray, np.ndarray, dict]:
    lock = load_json(GNN_SELECTION_LOCK)
    frozen = lock.get("propagation_cache", {})
    require(frozen.get("path") == relative(PROPAGATION_CACHE), "propagation-cache path")
    require(PROPAGATION_CACHE.is_file(), "missing propagation cache")
    require(frozen.get("sha256") == sha256(PROPAGATION_CACHE), "propagation-cache SHA")
    with np.load(PROPAGATION_CACHE, allow_pickle=False) as archive:
        required = {"k3_features", "k3_columns", "node_site_index", "node_partition_code"}
        require(required.issubset(set(archive.files)), "propagation-cache members")
        features = np.asarray(archive["k3_features"], dtype=np.float32)
        columns = np.asarray(archive["k3_columns"])
        site_index = np.asarray(archive["node_site_index"], dtype=np.uint32)
        partition = np.asarray(archive["node_partition_code"], dtype=np.uint8)
    require(features.shape == (EXPECTED_GRAPH_NODES, EXPECTED_GRAPH_FEATURES), "graph feature shape")
    require(columns.shape == (EXPECTED_GRAPH_FEATURES,), "graph feature columns")
    require(site_index.shape == (EXPECTED_GRAPH_NODES,), "graph site-index shape")
    require(partition.shape == (EXPECTED_GRAPH_NODES,), "graph partition shape")
    require(np.isfinite(features).all(), "finite graph features")
    require(len(np.unique(site_index)) == EXPECTED_GRAPH_NODES, "unique graph site indices")
    lookup = np.full(int(site_index.max()) + 1, -1, dtype=np.int32)
    lookup[site_index] = np.arange(EXPECTED_GRAPH_NODES, dtype=np.int32)
    return features, lookup, record(PROPAGATION_CACHE)


def load_partition(path: Path, partition: str, rows: int, sites: int, positives: int) -> dict:
    return deep_helpers.load_partition(path, partition, rows, sites, positives)


def materialize(
    bundle: dict[str, np.ndarray],
    graph_features: np.ndarray,
    graph_lookup: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.empty((len(indices), EXPECTED_COMBINED_FEATURES), dtype=np.float32)
    local_site_rows = bundle["site_row"][indices]
    global_site_indices = bundle["site_indices"][local_site_rows].astype(np.int64, copy=False)
    require(int(global_site_indices.max()) < len(graph_lookup), "graph lookup bounds")
    graph_rows = graph_lookup[global_site_indices]
    require(np.all(graph_rows >= 0), "complete graph feature mapping")
    matrix[:, 0] = bundle["stuck_value"][indices]
    matrix[:, 1:15] = bundle["site_features"][local_site_rows]
    matrix[:, 15:527] = bundle["vector_features"][bundle["vector_row"][indices]]
    matrix[:, 527:] = graph_features[graph_rows]
    require(np.isfinite(matrix).all(), "finite hybrid features")
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


def balanced_epoch_order(target: np.ndarray, epoch: int) -> np.ndarray:
    positive = np.flatnonzero(target == 1).astype(np.int64, copy=False)
    negative = np.flatnonzero(target == 0).astype(np.int64, copy=False)
    require(len(positive) == EXPECTED_TRAIN_POSITIVES, "training positive count")
    require(len(negative) > len(positive), "expected minority positive class")
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED + epoch))
    positive_order = rng.permutation(positive)
    extra = rng.choice(positive, size=len(negative) - len(positive), replace=True)
    negative_order = rng.permutation(negative)
    balanced = np.concatenate((positive_order, extra, negative_order))
    return balanced[rng.permutation(len(balanced))].astype(np.int64, copy=False)


def train_candidate(
    candidate: dict,
    train: dict[str, np.ndarray],
    graph_features: np.ndarray,
    graph_lookup: np.ndarray,
    replay: bool = False,
) -> tuple[MLPClassifier, list[dict]]:
    candidate_id = candidate["candidate_id"]
    model, start_epoch, history = load_checkpoint(candidate, replay)
    if model is None:
        model = build_model(candidate)
    balanced_rows = 2 * (EXPECTED_TRAIN_ROWS - EXPECTED_TRAIN_POSITIVES)
    total_batches = math.ceil(balanced_rows / TRAINING_BATCH_SIZE)
    prefix = "REPLAY" if replay else "TRAIN"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(start_epoch, EPOCHS):
            order = balanced_epoch_order(train["target"], epoch)
            require(len(order) == balanced_rows, "balanced epoch rows")
            batches = 0
            for start in range(0, len(order), TRAINING_BATCH_SIZE):
                indices = order[start : start + TRAINING_BATCH_SIZE]
                matrix, target = materialize(train, graph_features, graph_lookup, indices)
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
            history.append({
                "run": "REPLAY" if replay else "CANONICAL",
                "candidate_id": candidate_id,
                "epoch": epoch + 1,
                "batches": batches,
                "samples_seen": len(order),
                "final_batch_loss": format(loss, ".17g"),
            })
            save_checkpoint(candidate, replay, model, epoch + 1, history)
    require(tuple(model.hidden_layer_sizes) == HIDDEN_LAYERS, "trained hidden layers")
    require(int(model.n_features_in_) == EXPECTED_COMBINED_FEATURES, "trained feature count")
    require(np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])), "trained classes")
    parameter_count = sum(value.size for value in model.coefs_) + sum(
        value.size for value in model.intercepts_
    )
    require(parameter_count == EXPECTED_PARAMETERS, "trained parameter count")
    return model, history


def predict_probability(
    model: MLPClassifier,
    bundle: dict[str, np.ndarray],
    graph_features: np.ndarray,
    graph_lookup: np.ndarray,
    label: str,
) -> np.ndarray:
    rows = len(bundle["target"])
    probability = np.empty(rows, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    total_batches = math.ceil(rows / INFERENCE_BATCH_SIZE)
    for batch_number, start in enumerate(range(0, rows, INFERENCE_BATCH_SIZE), start=1):
        stop_index = min(start + INFERENCE_BATCH_SIZE, rows)
        indices = np.arange(start, stop_index, dtype=np.int64)
        matrix, _ = materialize(bundle, graph_features, graph_lookup, indices)
        probability[start:stop_index] = model.predict_proba(matrix)[:, positive_column]
        print(f"  {label} inference batch {batch_number}/{total_batches}", flush=True)
    require(np.isfinite(probability).all(), f"{label} finite probabilities")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), f"{label} probability range")
    return probability


def selection_key(metrics: dict) -> tuple:
    return (
        float(metrics["mcc"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["recall"]),
    )


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        CANDIDATE_METRICS_CSV, CANDIDATE_METRICS_JSON, TRAINING_HISTORY,
        SELECTED_MODEL, SELECTED_WEIGHTS, SELECTED_PREDICTIONS, CALIBRATION_CURVE,
        REPLAY_MODEL, REPLAY_WEIGHTS, REPLAY_PREDICTIONS, SELECTION_LOCK, MANIFEST,
    ):
        if path.exists():
            path.unlink()
    if WORK_ROOT.exists():
        for temporary in WORK_ROOT.rglob("*.tmp"):
            temporary.unlink()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-2B script in project root")
    verify_outputs_absent()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11D-2B — HYBRID MODEL TRAINING AND CALIBRATION")
    print("FROZEN INPUT VERIFICATION", flush=True)
    input_evidence = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    architecture, contract = verify_contract()
    environment = verify_environment()
    candidates = parse_candidates()
    graph_features, graph_lookup, propagation_record = load_graph_features()
    input_evidence[relative(PROPAGATION_CACHE)] = propagation_record

    print("\nOPENING AUTHORIZED DEVELOPMENT PARTITIONS", flush=True)
    train = load_partition(
        TRAIN_MATRIX, "DEV_TRAIN", EXPECTED_TRAIN_ROWS,
        EXPECTED_TRAIN_SITES, EXPECTED_TRAIN_POSITIVES,
    )
    calibration = load_partition(
        CALIBRATION_MATRIX, "DEV_CALIBRATION", EXPECTED_CALIBRATION_ROWS,
        EXPECTED_CALIBRATION_SITES, EXPECTED_CALIBRATION_POSITIVES,
    )
    require(
        set(train["site_ids"].tolist()).isdisjoint(set(calibration["site_ids"].tolist())),
        "DEV_TRAIN/DEV_CALIBRATION site overlap",
    )
    require(np.array_equal(train["feature_columns"], calibration["feature_columns"]), "feature vocabulary")
    print("  DEV_TRAIN       : OPENED FOR PARAMETER FITTING")
    print("  DEV_CALIBRATION : OPENED FOR SELECTION")
    print("  DEV_SITE_TEST   : NOT OPENED")
    print("  VALIDATION      : NOT EXPOSED")
    print("  HOLDOUT_TEST    : NOT EXPOSED")
    print("  Graph features  : FROZEN K3 CACHE")

    trained = []
    metric_records = []
    history_records = []
    print("\nSEQUENTIAL HYBRID CANDIDATE TRAINING", flush=True)
    for candidate in candidates:
        print(f"\nCANDIDATE {candidate['candidate_id']}", flush=True)
        model, history = train_candidate(candidate, train, graph_features, graph_lookup)
        probability = predict_probability(
            model, calibration, graph_features, graph_lookup, candidate["candidate_id"]
        )
        threshold, _ = deep_helpers.select_threshold(calibration["target"], probability)
        metrics = deep_helpers.full_metrics(
            calibration["target"], probability, threshold,
            candidate["candidate_id"], float(model.loss_),
        )
        print(
            f"  threshold={threshold:.6f} MCC={metrics['mcc']:.8f} "
            f"balanced={metrics['balanced_accuracy']:.8f} "
            f"recall={metrics['recall']:.8f} status={metrics['minimum_validity_status']}",
            flush=True,
        )
        trained.append({
            "candidate": candidate,
            "model": model,
            "probability": probability,
            "metrics": metrics,
        })
        metric_records.append(metrics)
        history_records.extend(history)

    selected = max(
        trained,
        key=lambda item: (
            *selection_key(item["metrics"]),
            item["candidate"]["candidate_id"] == candidates[0]["candidate_id"],
        ),
    )
    selected_id = selected["candidate"]["candidate_id"]
    selected_model = selected["model"]
    selected_probability = selected["probability"]
    selected_metrics = selected["metrics"]
    require(selected_metrics["minimum_validity_status"] == "PASS", "selected hybrid minimum validity")

    print("\nSELECTED HYBRID CANDIDATE")
    print("  Candidate :", selected_id)
    print("  Threshold :", format(selected_metrics["threshold"], ".17g"))
    print("  MCC       :", f"{selected_metrics['mcc']:.8f}")

    print("\nDETERMINISTIC SELECTED-CANDIDATE REPLAY", flush=True)
    replay_model, replay_history = train_candidate(
        selected["candidate"], train, graph_features, graph_lookup, replay=True
    )
    replay_probability = predict_probability(
        replay_model, calibration, graph_features, graph_lookup, "REPLAY"
    )
    replay_threshold, _ = deep_helpers.select_threshold(calibration["target"], replay_probability)
    replay_metrics = deep_helpers.full_metrics(
        calibration["target"], replay_probability, replay_threshold,
        selected_id, float(replay_model.loss_),
    )
    weights_exact = deep_helpers.compare_models(selected_model, replay_model)
    probabilities_exact = np.array_equal(selected_probability, replay_probability)
    threshold_exact = selected_metrics["threshold"] == replay_threshold
    metrics_exact = all(
        selected_metrics[name] == replay_metrics[name]
        for name in (
            "mcc", "balanced_accuracy", "precision", "recall", "specificity",
            "f1_score", "pr_auc", "roc_auc", "brier_score",
        )
    )
    require(weights_exact, "deterministic replay weights")
    require(probabilities_exact, "deterministic replay probabilities")
    require(threshold_exact, "deterministic replay threshold")
    require(metrics_exact, "deterministic replay metrics")

    created_at = datetime.now(timezone.utc).isoformat()
    atomic_csv(
        CANDIDATE_METRICS_CSV,
        METRIC_FIELDS,
        [deep_helpers.metric_csv_row(item) for item in metric_records],
    )
    atomic_json(CANDIDATE_METRICS_JSON, {
        "stage": "11D-2B",
        "status": "PASS",
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "frozen_gnn_site_test_mcc_for_later_comparison": FROZEN_GNN_MCC,
        "candidates": metric_records,
    })
    atomic_csv(TRAINING_HISTORY, HISTORY_FIELDS, history_records + replay_history)
    deep_helpers.atomic_joblib(SELECTED_MODEL, selected_model)
    deep_helpers.deterministic_npz(SELECTED_WEIGHTS, deep_helpers.model_arrays(selected_model))
    selected_prediction = (selected_probability >= selected_metrics["threshold"]).astype(np.uint8)
    deep_helpers.deterministic_npz(SELECTED_PREDICTIONS, {
        "record_id": calibration["record_id"],
        "site_row": calibration["site_row"],
        "vector_row": calibration["vector_row"],
        "stuck_value": calibration["stuck_value"],
        "target_detected": calibration["target"],
        "probability": selected_probability,
        "prediction": selected_prediction,
        "threshold": np.asarray([selected_metrics["threshold"]], dtype=np.float64),
    })
    deep_helpers.write_calibration_curve(
        CALIBRATION_CURVE, calibration["target"], selected_probability
    )
    deep_helpers.atomic_joblib(REPLAY_MODEL, replay_model)
    deep_helpers.deterministic_npz(REPLAY_WEIGHTS, deep_helpers.model_arrays(replay_model))
    deep_helpers.deterministic_npz(REPLAY_PREDICTIONS, {
        "record_id": calibration["record_id"],
        "probability": replay_probability,
        "prediction": (replay_probability >= replay_threshold).astype(np.uint8),
        "threshold": np.asarray([replay_threshold], dtype=np.float64),
    })

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
        "stage": "11D-2B",
        "title": "HYBRID MODEL SELECTION AND THRESHOLD LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": created_at,
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_metrics["threshold"],
        "selection_partition": "DEV_CALIBRATION",
        "selection_metric": "MCC",
        "selected_metrics": selected_metrics,
        "minimum_validity_status": selected_metrics["minimum_validity_status"],
        "absolute_performance_status_on_calibration": selected_metrics["project_target_status"],
        "hybrid_advancement_status": "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON",
        "frozen_gnn_site_test_mcc": FROZEN_GNN_MCC,
        "required_hybrid_mcc_gain": REQUIRED_HYBRID_GAIN,
        "selected_model": outputs["selected_model"],
        "selected_weights": outputs["selected_weights"],
        "selected_calibration_predictions": outputs["selected_calibration_predictions"],
        "selected_calibration_curve": outputs["selected_calibration_curve"],
        "candidate_metrics_csv": outputs["candidate_metrics_csv"],
        "candidate_metrics_json": outputs["candidate_metrics_json"],
        "propagation_cache": propagation_record,
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
        "retraining_after_site_test_authorized": False,
    }
    atomic_json(SELECTION_LOCK, selection_lock)
    selection_lock_record = record(SELECTION_LOCK)

    manifest = {
        "stage": "11D-2B",
        "title": "HYBRID MODEL TRAINING AND CALIBRATION EXECUTION",
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
        "model_family": "GRAPH-AUGMENTED FEATURE-FUSION MLP",
        "input_features": EXPECTED_COMBINED_FEATURES,
        "sample_features": EXPECTED_SAMPLE_FEATURES,
        "graph_features": EXPECTED_GRAPH_FEATURES,
        "hidden_layers": list(HIDDEN_LAYERS),
        "trainable_parameters": EXPECTED_PARAMETERS,
        "candidate_count": len(candidates),
        "epochs_per_candidate": EPOCHS,
        "balanced_training_samples_per_epoch": 2 * (EXPECTED_TRAIN_ROWS - EXPECTED_TRAIN_POSITIVES),
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_metrics["threshold"],
        "selected_metrics": selected_metrics,
        "minimum_validity_status": selected_metrics["minimum_validity_status"],
        "hybrid_advancement_status": "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON",
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
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = record(MANIFEST)

    audit = {
        "stage": "11D-2B",
        "title": "HYBRID MODEL TRAINING AND CALIBRATION FREEZE",
        "status": "PASS",
        "training_status": "FROZEN",
        "model_selection_status": "FROZEN",
        "threshold_status": "FROZEN",
        "model_family": "GRAPH-AUGMENTED FEATURE-FUSION MLP",
        "candidates_trained": len(candidates),
        "epochs_per_candidate": EPOCHS,
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
        "absolute_performance_status_on_calibration": selected_metrics["project_target_status"],
        "hybrid_advancement_status": "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON",
        "deterministic_replay": "PASS",
        "weights_exact": weights_exact,
        "probabilities_exact": probabilities_exact,
        "threshold_exact": threshold_exact,
        "metrics_exact": metrics_exact,
        "dev_site_test_state": "AUTHORIZED FOR ONE LOCKED HYBRID EVALUATION",
        "dev_site_test_opened_during_training": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "selected_model": outputs["selected_model"],
        "selection_lock": selection_lock_record,
        "manifest": manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
        "next_gate": "STAGE 11D-2C — LOCKED HYBRID DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)

    print("\nSTAGE 11D-2B — HYBRID MODEL TRAINING AND CALIBRATION FREEZE")
    print("Status                         : PASS")
    print("Training status                : FROZEN")
    print("Model selection status         : FROZEN")
    print("Threshold status               : FROZEN")
    print("Candidates trained             :", len(candidates))
    print("Epochs per candidate           :", EPOCHS)
    print("Balanced samples per epoch     :", 2 * (EXPECTED_TRAIN_ROWS - EXPECTED_TRAIN_POSITIVES))
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
    print("Absolute target on calibration :", selected_metrics["project_target_status"])
    print("Hybrid gain over GNN           : PENDING LOCKED SITE TEST")
    print("Deterministic replay           : PASS")
    print("Weights exact                  : YES")
    print("Probabilities exact            : YES")
    print("DEV_SITE_TEST opened           : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Graph dataset modified         : NO")
    print("Frozen comparator models changed: NO")
    print("Selected model                 :", SELECTED_MODEL)
    print("Selected model SHA             :", outputs["selected_model"]["sha256"])
    print("Selection lock                 :", SELECTION_LOCK)
    print("Selection lock SHA             :", selection_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-2C — LOCKED HYBRID DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
