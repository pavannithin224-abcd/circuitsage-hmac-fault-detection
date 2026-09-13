#!/usr/bin/env python3
"""Train, calibrate, replay, and freeze the directed bidirectional SGC model."""
from __future__ import annotations

import os

# Freeze numerical worker counts before importing NumPy/SciPy/scikit-learn.
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
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from scipy import sparse
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


VERSION = "HMAC-DIRECTED-SGC-TRAINER-v1"
MODEL_FAMILY = "DIRECTED_BIDIRECTIONAL_SIMPLIFIED_GRAPH_CONVOLUTION"
RANDOM_SEED = 20_260_906
CLASSES = np.asarray([0, 1], dtype=np.uint8)

EXPECTED_NODES = 22_839
EXPECTED_EDGES = 47_617
EXPECTED_RAW_BRANCHES = 47_631
EXPECTED_SAMPLE_FEATURES = 527
EXPECTED_SITE_FEATURES = 14
EXPECTED_VECTOR_FEATURES = 512
EXPECTED_TRAIN_ROWS = 2_046_336
EXPECTED_CALIBRATION_ROWS = 438_528
EXPECTED_SITE_TEST_ROWS = 438_528
EXPECTED_TRAIN_POSITIVES = 894_000
EXPECTED_CALIBRATION_POSITIVES = 192_943
EXPECTED_TRAIN_SITES = 15_987
EXPECTED_CALIBRATION_SITES = 3_426
EXPECTED_SITE_TEST_SITES = 3_426
EXPECTED_CANDIDATES = 2
EXPECTED_EPOCHS = 6
EXPECTED_BATCH_SIZE = 32_768
EXPECTED_GRAPH_COMMITMENT = (
    "2020aa2959c0919df336da93087d0cad5086711cc9f9bcb185673d2a93db614f"
)

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11d1"
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
GRAPH_ROOT = RESULT_ROOT / "graph_dataset_11d1a"
FEATURE_ROOT = RESULT_11C5 / "feature_matrix_11c5e"
WORK_ROOT = RESULT_ROOT / "gnn_training_11d1c"
MODEL_ROOT = WORK_ROOT / "candidate_models"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

CONTRACT_FREEZER = ROOT / "stage_11d1b_gnn_contract.py"
ARCHITECTURE = CONFIG_ROOT / "hmac_gnn_architecture_11d1b.json"
CONTRACT = CONFIG_ROOT / "hmac_gnn_training_contract_11d1b.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_gnn_candidate_grid_11d1b.csv"
ENVIRONMENT = RESULT_ROOT / "hmac_gnn_environment_11d1b.json"
CONTRACT_AUDIT = RESULT_ROOT / "hmac_gnn_architecture_training_contract_freeze_11d1b.json"

GRAPH_NPZ = GRAPH_ROOT / "hmac_golden_netlist_graph_11d1a.npz"
GRAPH_SCHEMA = GRAPH_ROOT / "hmac_golden_netlist_graph_schema_11d1a.json"
GRAPH_MANIFEST = RESULT_ROOT / "hmac_golden_netlist_graph_manifest_11d1a.json"
GRAPH_AUDIT = RESULT_ROOT / "hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"

TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CALIBRATION_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
FEATURE_MANIFEST = RESULT_11C5 / "hmac_leakage_safe_feature_matrix_manifest_11c5e.json"
FEATURE_AUDIT = RESULT_11C5 / "hmac_leakage_safe_feature_matrix_freeze_11c5e.json"

PROPAGATION_CACHE = WORK_ROOT / "hmac_directed_sgc_graph_features_11d1c.npz"
CANDIDATE_METRICS_CSV = WORK_ROOT / "hmac_gnn_candidate_calibration_metrics_11d1c.csv"
CANDIDATE_METRICS_JSON = WORK_ROOT / "hmac_gnn_candidate_calibration_metrics_11d1c.json"
SELECTED_MODEL = WORK_ROOT / "hmac_selected_gnn_model_11d1c.joblib"
SELECTED_WEIGHTS = WORK_ROOT / "hmac_selected_gnn_weights_11d1c.npz"
SELECTED_PREDICTIONS = WORK_ROOT / "hmac_selected_gnn_calibration_predictions_11d1c.npz"
CALIBRATION_CURVE = WORK_ROOT / "hmac_selected_gnn_calibration_curve_11d1c.csv"
REPLAY_MODEL = WORK_ROOT / "hmac_selected_gnn_model_replay_11d1c.joblib"
REPLAY_WEIGHTS = WORK_ROOT / "hmac_selected_gnn_weights_replay_11d1c.npz"
REPLAY_PREDICTIONS = WORK_ROOT / "hmac_selected_gnn_calibration_predictions_replay_11d1c.npz"
SELECTION_LOCK = WORK_ROOT / "hmac_gnn_selection_lock_11d1c.json"
MANIFEST = RESULT_ROOT / "hmac_gnn_training_manifest_11d1c.json"
AUDIT = RESULT_ROOT / "hmac_gnn_training_calibration_freeze_11d1c.json"

EXPECTED_INPUTS = {
    CONTRACT_FREEZER: "498f59600c014b025ffa4fb58c73ffb95a47e0607332b6f0e15d7dda4304877b",
    ARCHITECTURE: "120033cb61ab8af5c5895aa71adf44395c9aeae6055cba56c1c6f5269796cdba",
    CONTRACT: "3662b5f579c1a910c4336ec1173e592391419fa17cc05813a54575bb9d5350e1",
    CANDIDATE_GRID: "f00c701a739f09c8431505a2021565b8b09bb6113ae43cb70f0ebbafc0adc54f",
    ENVIRONMENT: "23dc4852f5ca047c7f382d16b591cad2f8d6adf67724c09fc25e49b5d4776ddd",
    CONTRACT_AUDIT: "b5a5b8b6b8d7f6f6926a17440ff6f7859e422e9182c948695afa47bf0f36ef01",
    GRAPH_NPZ: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    GRAPH_SCHEMA: "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    GRAPH_MANIFEST: "b719e33941460ce9c0186c29c18f4585adf044ae3afd0493ad40214f42b9828c",
    GRAPH_AUDIT: "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
    TRAIN_MATRIX: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CALIBRATION_MATRIX: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    FEATURE_MANIFEST: "413cbad01681b4f209de6fce44d3777bccfe424c50c3c95582c627e44965e186",
    FEATURE_AUDIT: "6320bb732b84ae56c937e188189fb87a159d35a164eed892ec3da9d1ee27e540",
}

CANDIDATE_COLUMNS = [
    "candidate_id", "model_family", "propagation_hops", "direction_channels",
    "self_channel", "base_node_features", "graph_features", "sample_features",
    "combined_features", "classifier", "loss", "alpha", "learning_rate",
    "initial_learning_rate", "epochs", "batch_size", "class_weighting",
    "shuffle", "random_seed",
]

METRICS_COLUMNS = [
    "candidate_id", "propagation_hops", "graph_features", "combined_features",
    "alpha", "epochs", "calibration_threshold", "mcc", "balanced_accuracy",
    "precision", "recall", "specificity", "f1_score", "pr_auc", "roc_auc",
    "brier_score", "true_negative", "false_positive", "false_negative",
    "true_positive", "model_sha256",
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
    return {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not read JSON {relative(path)}: {error}")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=6, allowZip64=True,
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
                info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    temporary.replace(path)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {distribution}")


def verify_file(path: Path, expected: str) -> dict:
    require(path.is_file(), f"missing frozen input: {relative(path)}")
    require(path.stat().st_size > 0, f"empty frozen input: {relative(path)}")
    actual = sha256(path)
    require(actual == expected, f"SHA mismatch for {relative(path)}: expected {expected}, actual {actual}")
    print(f"  {path.name:<74}: OK", flush=True)
    return output_record(path)


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
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"thread contract {name}")
    return {"frozen": output_record(ENVIRONMENT), "packages": actual, "threads": 1}


def verify_contracts() -> tuple[dict, dict, dict]:
    architecture = load_json(ARCHITECTURE)
    contract = load_json(CONTRACT)
    audit = load_json(CONTRACT_AUDIT)
    graph_manifest = load_json(GRAPH_MANIFEST)
    graph_audit = load_json(GRAPH_AUDIT)
    feature_audit = load_json(FEATURE_AUDIT)

    require(architecture.get("status") == "PASS", "architecture status")
    require(architecture.get("architecture_status") == "FROZEN", "architecture freeze")
    require(architecture.get("model_family") == MODEL_FAMILY, "architecture family")
    require(architecture.get("is_gnn") is True, "GNN identity")
    require(architecture.get("is_deep_backpropagated_gnn") is False, "SGC architecture identity")
    require(architecture.get("target_present_in_input") is False, "architecture target leakage")
    require(architecture.get("post_simulation_features") == 0, "architecture post-simulation features")

    require(contract.get("status") == "PASS", "training contract status")
    require(contract.get("training_contract_status") == "FROZEN", "training contract freeze")
    require(contract.get("model_family") == MODEL_FAMILY, "training model family")
    require(contract.get("fault_scope") == "PERSISTENT_NET_STEM_SA0_SA1_ONLY", "fault scope")
    require(contract.get("candidate_execution") == "SEQUENTIAL", "candidate execution")
    require(contract.get("parallel_candidates") == 1, "parallel candidates")
    require(contract.get("epochs_per_candidate") == EXPECTED_EPOCHS, "epochs")
    require(contract.get("training_batch_size") == EXPECTED_BATCH_SIZE, "batch size")
    require(contract.get("selection_partition") == "DEV_CALIBRATION", "selection partition")
    require(contract.get("primary_metric") == "MCC", "selection metric")
    require(contract.get("dev_site_test_opening_authorized") is False, "site-test lock")
    require(contract.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(contract.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")
    require(contract.get("hybrid_training_authorized") is False, "hybrid gate")

    partitions = contract.get("partitions", {})
    require(partitions.get("DEV_TRAIN", {}).get("samples") == EXPECTED_TRAIN_ROWS, "train samples")
    require(partitions.get("DEV_CALIBRATION", {}).get("samples") == EXPECTED_CALIBRATION_ROWS, "calibration samples")
    require(partitions.get("DEV_SITE_TEST", {}).get("samples") == EXPECTED_SITE_TEST_ROWS, "site-test samples")

    require(audit.get("status") == "PASS", "11D-1B audit status")
    require(audit.get("gnn_training_authorized_for_dev_train") is True, "GNN training authorization")
    require(audit.get("dev_site_test_state") == "LOCKED", "11D-1B site-test lock")
    require(audit.get("hybrid_training_authorized") is False, "11D-1B hybrid gate")

    require(graph_manifest.get("graph_commitment") == EXPECTED_GRAPH_COMMITMENT, "graph commitment")
    require(graph_manifest.get("node_count") == EXPECTED_NODES, "graph nodes")
    require(graph_manifest.get("collapsed_edge_count") == EXPECTED_EDGES, "graph edges")
    require(graph_audit.get("topology_integrity_status") == "PASS", "topology integrity")
    require(graph_audit.get("target_present_in_graph_features") is False, "graph target leakage")
    require(feature_audit.get("target_in_feature_columns") is False, "sample target leakage")
    require(feature_audit.get("post_simulation_features") == 0, "sample post-simulation leakage")
    return architecture, contract, audit


def load_candidates(contract: dict) -> list[dict]:
    rows = []
    with CANDIDATE_GRID.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CANDIDATE_COLUMNS, "candidate-grid header")
        for row in reader:
            parsed = dict(row)
            for key in (
                "propagation_hops", "base_node_features", "graph_features",
                "sample_features", "combined_features", "epochs", "batch_size",
                "random_seed",
            ):
                parsed[key] = int(row[key])
            parsed["alpha"] = float(row["alpha"])
            require(parsed["model_family"] == MODEL_FAMILY, "candidate model family")
            require(parsed["direction_channels"] == "INBOUND+OUTBOUND", "candidate directions")
            require(parsed["self_channel"] == "BASE_FEATURES", "candidate self channel")
            require(parsed["classifier"] == "sklearn.linear_model.SGDClassifier", "candidate classifier")
            require(parsed["loss"] == "log_loss", "candidate loss")
            require(parsed["learning_rate"] == "optimal", "candidate learning rate")
            require(parsed["epochs"] == EXPECTED_EPOCHS, "candidate epochs")
            require(parsed["batch_size"] == EXPECTED_BATCH_SIZE, "candidate batch size")
            require(parsed["class_weighting"] == "GLOBAL_DEV_TRAIN_BALANCED_SAMPLE_WEIGHT", "candidate weighting")
            require(parsed["random_seed"] == RANDOM_SEED, "candidate random seed")
            require(parsed["propagation_hops"] in (2, 3), "candidate propagation hops")
            rows.append(parsed)
    require(len(rows) == EXPECTED_CANDIDATES, "candidate count")
    require(len({row["candidate_id"] for row in rows}) == EXPECTED_CANDIDATES, "candidate IDs")
    require(contract.get("candidate_grid", {}).get("sha256") == EXPECTED_INPUTS[CANDIDATE_GRID], "contract grid SHA")
    return rows


def load_graph() -> tuple[dict[str, np.ndarray], dict]:
    schema = load_json(GRAPH_SCHEMA)
    expected_members = set(schema.get("npz_arrays", {}))
    require(expected_members, "graph NPZ schema members")
    with np.load(GRAPH_NPZ, allow_pickle=False) as archive:
        require(set(archive.files) == expected_members, "graph NPZ members")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, specification in schema["npz_arrays"].items():
        require(list(arrays[name].shape) == specification["shape"], f"graph shape {name}")
        require(arrays[name].dtype.str == specification["dtype"], f"graph dtype {name}")
    edge_index = arrays["edge_index"].astype(np.int64, copy=False)
    require(edge_index.shape == (2, EXPECTED_EDGES), "edge-index shape")
    require(int(edge_index.min()) >= 0 and int(edge_index.max()) < EXPECTED_NODES, "edge-index bounds")
    require(int(arrays["edge_multiplicity"].sum(dtype=np.uint64)) == EXPECTED_RAW_BRANCHES, "edge multiplicity")
    require(np.array_equal(arrays["node_site_index"], np.arange(1, EXPECTED_NODES + 1)), "node/site indexing")
    partitions = np.bincount(arrays["node_partition_code"].astype(np.int64), minlength=3)
    require(partitions.tolist() == [EXPECTED_TRAIN_SITES, EXPECTED_CALIBRATION_SITES, EXPECTED_SITE_TEST_SITES], "graph partitions")
    return arrays, schema


def base_node_features(arrays: dict[str, np.ndarray], schema: dict) -> tuple[np.ndarray, list[str]]:
    driver_code = arrays["node_driver_type_code"].astype(np.int64)
    category_code = arrays["node_site_category_code"].astype(np.int64)
    driver_map = schema.get("driver_cell_type_codes", {})
    category_map = schema.get("site_category_codes", {})
    require(sorted(driver_map.values()) == list(range(len(driver_map))), "driver-code vocabulary")
    require(sorted(category_map.values()) == list(range(len(category_map))), "category-code vocabulary")
    require(len(driver_map) == 9 and len(category_map) == 2, "base categorical feature count")
    driver_names = [name for name, _ in sorted(driver_map.items(), key=lambda item: item[1])]
    category_names = [name for name, _ in sorted(category_map.items(), key=lambda item: item[1])]
    numeric_names = [
        "log1p_is_primary_output", "log1p_cell_fanout", "log1p_in_degree_branches",
        "log1p_out_degree_branches", "log1p_in_neighbor_count",
        "log1p_out_neighbor_count",
    ]
    driver_onehot = np.eye(len(driver_names), dtype=np.float32)[driver_code]
    category_onehot = np.eye(len(category_names), dtype=np.float32)[category_code]
    numeric_raw = np.column_stack([
        arrays["node_is_primary_output"], arrays["node_cell_fanout"],
        arrays["node_in_degree_branches"], arrays["node_out_degree_branches"],
        arrays["node_in_neighbor_count"], arrays["node_out_neighbor_count"],
    ]).astype(np.float32)
    # Match the frozen 11D-1B encoder exactly: LOG1P applies to all six
    # numeric columns, including the binary primary-output indicator.
    numeric = np.log1p(numeric_raw)
    features = np.concatenate([driver_onehot, category_onehot, numeric], axis=1)
    columns = (
        [f"driver_type={name}" for name in driver_names]
        + [f"site_category={name}" for name in category_names]
        + numeric_names
    )
    require(features.shape == (EXPECTED_NODES, 17), "base-node feature shape")
    require(len(columns) == 17 and np.isfinite(features).all(), "base-node features")
    return features, columns


def normalized_adjacencies(arrays: dict[str, np.ndarray]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    edge_index = arrays["edge_index"].astype(np.int64, copy=False)
    source, destination = edge_index[0], edge_index[1]
    weight = arrays["edge_multiplicity"].astype(np.float32)
    identity = np.arange(EXPECTED_NODES, dtype=np.int64)
    identity_weight = np.ones(EXPECTED_NODES, dtype=np.float32)
    inbound = sparse.coo_matrix(
        (np.concatenate([weight, identity_weight]),
         (np.concatenate([destination, identity]), np.concatenate([source, identity]))),
        shape=(EXPECTED_NODES, EXPECTED_NODES), dtype=np.float32,
    ).tocsr()
    outbound = sparse.coo_matrix(
        (np.concatenate([weight, identity_weight]),
         (np.concatenate([source, identity]), np.concatenate([destination, identity]))),
        shape=(EXPECTED_NODES, EXPECTED_NODES), dtype=np.float32,
    ).tocsr()

    def normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        require(np.all(row_sum > 0.0), "positive adjacency row sums")
        result = (sparse.diags((1.0 / row_sum).astype(np.float32)) @ matrix).tocsr()
        result.sort_indices()
        return result

    return normalize(inbound), normalize(outbound)


def raw_graph_features(
    base: np.ndarray,
    inbound: sparse.csr_matrix,
    outbound: sparse.csr_matrix,
    hops: int,
) -> np.ndarray:
    inbound_state = base.copy()
    outbound_state = base.copy()
    inbound_hops = []
    outbound_hops = []
    for _ in range(hops):
        inbound_state = np.asarray(inbound @ inbound_state, dtype=np.float32)
        outbound_state = np.asarray(outbound @ outbound_state, dtype=np.float32)
        require(np.isfinite(inbound_state).all(), "finite inbound propagation")
        require(np.isfinite(outbound_state).all(), "finite outbound propagation")
        inbound_hops.append(inbound_state.copy())
        outbound_hops.append(outbound_state.copy())
    result = np.concatenate([base, *inbound_hops, *outbound_hops], axis=1).astype(np.float32, copy=False)
    require(result.shape == (EXPECTED_NODES, 17 * (1 + 2 * hops)), "propagated graph shape")
    return result


def graph_feature_columns(base_columns: list[str], hops: int) -> list[str]:
    columns = [f"self::{name}" for name in base_columns]
    for hop in range(1, hops + 1):
        columns.extend(f"inbound_h{hop}::{name}" for name in base_columns)
    for hop in range(1, hops + 1):
        columns.extend(f"outbound_h{hop}::{name}" for name in base_columns)
    return columns


def build_graph_feature_cache(
    arrays: dict[str, np.ndarray],
    schema: dict,
    candidates: list[dict],
) -> tuple[dict[str, dict], dict]:
    base, base_columns = base_node_features(arrays, schema)
    inbound, outbound = normalized_adjacencies(arrays)
    train_mask = arrays["node_partition_code"].astype(np.uint8) == 0
    require(int(train_mask.sum()) == EXPECTED_TRAIN_SITES, "graph-scaler train nodes")

    cache_arrays: dict[str, np.ndarray] = {
        "node_site_index": arrays["node_site_index"].astype("<u4", copy=False),
        "node_partition_code": arrays["node_partition_code"].astype(np.uint8, copy=False),
    }
    feature_sets: dict[str, dict] = {}
    replay_exact = True
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        hops = candidate["propagation_hops"]
        print(f"  PROPAGATE {candidate_id}: hops={hops}", flush=True)
        raw = raw_graph_features(base, inbound, outbound, hops)
        replay = raw_graph_features(base, inbound, outbound, hops)
        replay_exact = replay_exact and np.array_equal(raw, replay)
        require(replay_exact, f"{candidate_id} propagation replay")
        mean = raw[train_mask].mean(axis=0, dtype=np.float64)
        scale = raw[train_mask].std(axis=0, dtype=np.float64)
        scale[scale == 0.0] = 1.0
        scaled = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
        require(np.isfinite(scaled).all(), f"{candidate_id} finite scaled graph features")
        require(scaled.shape[1] == candidate["graph_features"], f"{candidate_id} graph width")
        key = f"k{hops}"
        columns = graph_feature_columns(base_columns, hops)
        cache_arrays[f"{key}_features"] = scaled
        cache_arrays[f"{key}_mean"] = mean.astype("<f8")
        cache_arrays[f"{key}_scale"] = scale.astype("<f8")
        cache_arrays[f"{key}_columns"] = np.asarray(columns)
        feature_sets[candidate_id] = {
            "features": scaled,
            "mean": mean,
            "scale": scale,
            "columns": columns,
            "hops": hops,
        }
    deterministic_npz(PROPAGATION_CACHE, cache_arrays)
    return feature_sets, {
        "status": "PASS",
        "replay_exact": replay_exact,
        "train_nodes_used_for_scaler": EXPECTED_TRAIN_SITES,
        "calibration_labels_used": False,
        "site_test_labels_used": False,
        "inbound_nnz": int(inbound.nnz),
        "outbound_nnz": int(outbound.nnz),
        "artifact": output_record(PROPAGATION_CACHE),
    }


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
    expected_rows = EXPECTED_TRAIN_ROWS if partition == "DEV_TRAIN" else EXPECTED_CALIBRATION_ROWS
    expected_sites = EXPECTED_TRAIN_SITES if partition == "DEV_TRAIN" else EXPECTED_CALIBRATION_SITES
    expected_positives = EXPECTED_TRAIN_POSITIVES if partition == "DEV_TRAIN" else EXPECTED_CALIBRATION_POSITIVES
    require(result["site_features"].shape == (expected_sites, EXPECTED_SITE_FEATURES), f"{partition} site features")
    require(result["vector_features"].shape == (64, EXPECTED_VECTOR_FEATURES), f"{partition} vector features")
    require(result["feature_columns"].shape == (EXPECTED_SAMPLE_FEATURES,), f"{partition} feature columns")
    for name in ("site_row", "vector_row", "stuck_value", "target"):
        require(result[name].shape == (expected_rows,), f"{partition} {name} rows")
    require(result["site_indices"].shape == (expected_sites,), f"{partition} site indices")
    require(int(result["target"].sum(dtype=np.uint64)) == expected_positives, f"{partition} positives")
    require(np.isin(result["target"], (0, 1)).all(), f"{partition} binary target")
    require(np.isin(result["stuck_value"], (0, 1)).all(), f"{partition} stuck values")
    return result


def feature_batch(bundle: dict[str, np.ndarray], graph_features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    width = EXPECTED_SAMPLE_FEATURES + graph_features.shape[1]
    matrix = np.empty((len(indices), width), dtype=np.float32)
    site_rows = bundle["site_row"][indices]
    graph_rows = bundle["site_indices"][site_rows].astype(np.int64) - 1
    require(int(graph_rows.min()) >= 0 and int(graph_rows.max()) < EXPECTED_NODES, "graph row bounds")
    matrix[:, 0] = bundle["stuck_value"][indices]
    matrix[:, 1:15] = bundle["site_features"][site_rows]
    matrix[:, 15:527] = bundle["vector_features"][bundle["vector_row"][indices]]
    matrix[:, 527:] = graph_features[graph_rows]
    require(np.isfinite(matrix).all(), "finite combined feature batch")
    return matrix


def new_estimator(candidate: dict) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=float(candidate["alpha"]),
        fit_intercept=True, max_iter=1, tol=None, shuffle=False,
        random_state=RANDOM_SEED, learning_rate="optimal", early_stopping=False,
        average=False,
    )


def epoch_permutation(sample_count: int, epoch_index: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(RANDOM_SEED + epoch_index)).permutation(sample_count)


def estimator_arrays(model: SGDClassifier) -> dict[str, np.ndarray]:
    return {
        "classes": np.asarray(model.classes_),
        "coef": np.asarray(model.coef_, dtype="<f8"),
        "intercept": np.asarray(model.intercept_, dtype="<f8"),
        "n_features_in": np.asarray([model.n_features_in_], dtype="<u4"),
        "t": np.asarray([model.t_], dtype="<f8"),
    }


def model_bundle(candidate: dict, model: SGDClassifier, graph_set: dict) -> dict:
    return {
        "version": VERSION,
        "model_family": MODEL_FAMILY,
        "candidate": dict(candidate),
        "estimator": model,
        "graph_scaler_mean": np.asarray(graph_set["mean"], dtype=np.float64),
        "graph_scaler_scale": np.asarray(graph_set["scale"], dtype=np.float64),
        "graph_feature_columns": list(graph_set["columns"]),
        "graph_commitment": EXPECTED_GRAPH_COMMITMENT,
        "sample_feature_count": EXPECTED_SAMPLE_FEATURES,
        "target_in_input": False,
        "post_simulation_features": 0,
    }


def bundles_exact(left: dict, right: dict) -> bool:
    for name in estimator_arrays(left["estimator"]):
        if not np.array_equal(estimator_arrays(left["estimator"])[name], estimator_arrays(right["estimator"])[name]):
            return False
    return (
        left["candidate"] == right["candidate"]
        and left["graph_commitment"] == right["graph_commitment"]
        and np.array_equal(left["graph_scaler_mean"], right["graph_scaler_mean"])
        and np.array_equal(left["graph_scaler_scale"], right["graph_scaler_scale"])
    )


def train_candidate(
    candidate: dict,
    train: dict[str, np.ndarray],
    graph_set: dict,
    positive_weight: float,
    negative_weight: float,
    resume: bool,
    replay: bool = False,
) -> tuple[dict, dict]:
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
        start_epoch = int(state.get("completed_epochs", -1))
        require(0 <= start_epoch <= EXPECTED_EPOCHS, f"{candidate_id} checkpoint epoch")
        model_record = state.get("epoch_model", {})
        committed_model_path = ROOT / str(model_record.get("path", ""))
        require(committed_model_path.is_file(), f"{candidate_id} checkpoint model missing")
        require(sha256(committed_model_path) == model_record.get("sha256"), f"{candidate_id} checkpoint SHA")
        bundle = joblib.load(committed_model_path)
        require(bundle.get("candidate", {}).get("candidate_id") == candidate_id, f"{candidate_id} model identity")
        require(np.array_equal(bundle["graph_scaler_mean"], graph_set["mean"]), f"{candidate_id} scaler mean")
        require(np.array_equal(bundle["graph_scaler_scale"], graph_set["scale"]), f"{candidate_id} scaler scale")
        model = bundle["estimator"]
        print(f"  {candidate_id}: RESUME after epoch {start_epoch}/{EXPECTED_EPOCHS}", flush=True)
    else:
        require(not model_path.exists(), f"uncommitted candidate model without checkpoint: {candidate_id}")
        model = new_estimator(candidate)
        start_epoch = 0

    sample_count = len(train["target"])
    fitted = hasattr(model, "classes_")
    started = time.monotonic()
    for epoch_index in range(start_epoch, EXPECTED_EPOCHS):
        permutation = epoch_permutation(sample_count, epoch_index)
        batch_count = math.ceil(sample_count / EXPECTED_BATCH_SIZE)
        for batch_number, start in enumerate(range(0, sample_count, EXPECTED_BATCH_SIZE), start=1):
            indices = permutation[start:start + EXPECTED_BATCH_SIZE]
            x_batch = feature_batch(train, graph_set["features"], indices)
            y_batch = train["target"][indices]
            sample_weight = np.where(y_batch == 1, positive_weight, negative_weight).astype(np.float64)
            if fitted:
                model.partial_fit(x_batch, y_batch, sample_weight=sample_weight)
            else:
                model.partial_fit(x_batch, y_batch, classes=CLASSES, sample_weight=sample_weight)
                fitted = True
            if batch_number % 16 == 0 or batch_number == batch_count:
                label = "REPLAY" if replay else candidate_id
                print(
                    f"    {label} epoch={epoch_index + 1}/{EXPECTED_EPOCHS} "
                    f"batch={batch_number}/{batch_count}", flush=True,
                )
        if not replay:
            committed_model_path = MODEL_ROOT / f"{candidate_id}.epoch_{epoch_index + 1:02d}.joblib"
            atomic_joblib(committed_model_path, model_bundle(candidate, model, graph_set))
            checkpoint = {
                "stage": "11D-1C",
                "status": "TRAINED" if epoch_index + 1 == EXPECTED_EPOCHS else "IN_PROGRESS",
                "candidate_id": candidate_id,
                "completed_epochs": epoch_index + 1,
                "sample_updates": (epoch_index + 1) * sample_count,
                "epoch_model": output_record(committed_model_path),
                "contract_sha256": EXPECTED_INPUTS[CONTRACT],
                "train_matrix_sha256": EXPECTED_INPUTS[TRAIN_MATRIX],
                "graph_sha256": EXPECTED_INPUTS[GRAPH_NPZ],
                "random_seed": RANDOM_SEED,
            }
            atomic_json(state_path, checkpoint)
            print(
                f"  {candidate_id}: CHECKPOINT epoch {epoch_index + 1}/{EXPECTED_EPOCHS} "
                f"sha256={checkpoint['epoch_model']['sha256']}", flush=True,
            )
    elapsed = time.monotonic() - started
    bundle = model_bundle(candidate, model, graph_set)
    state = {
        "candidate_id": candidate_id,
        "completed_epochs": EXPECTED_EPOCHS,
        "sample_updates": EXPECTED_EPOCHS * sample_count,
        "sample_updates_this_invocation": (EXPECTED_EPOCHS - start_epoch) * sample_count,
        "elapsed_seconds_this_invocation": elapsed,
    }
    if not replay:
        require(committed_model_path is not None, f"{candidate_id} committed model")
        atomic_copy(committed_model_path, model_path)
        state.update({
            "model": output_record(model_path),
            "epoch_model": output_record(committed_model_path),
            "checkpoint": output_record(state_path),
        })
    return bundle, state


def predict_probabilities(bundle: dict, matrix: dict[str, np.ndarray], graph_features: np.ndarray) -> np.ndarray:
    model = bundle["estimator"]
    sample_count = len(matrix["target"])
    probability = np.empty(sample_count, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    total_batches = math.ceil(sample_count / EXPECTED_BATCH_SIZE)
    for batch_number, start in enumerate(range(0, sample_count, EXPECTED_BATCH_SIZE), start=1):
        stop_index = min(start + EXPECTED_BATCH_SIZE, sample_count)
        indices = np.arange(start, stop_index, dtype=np.int64)
        x_batch = feature_batch(matrix, graph_features, indices)
        probability[start:stop_index] = model.predict_proba(x_batch)[:, positive_column]
        if batch_number % 7 == 0 or batch_number == total_batches:
            print(f"    inference batch {batch_number}/{total_batches}", flush=True)
    require(np.isfinite(probability).all(), "finite probabilities")
    require(((probability >= 0.0) & (probability <= 1.0)).all(), "probability range")
    return probability


def optimal_threshold(y_true: np.ndarray, probability: np.ndarray) -> dict:
    order = np.argsort(-probability, kind="stable")
    score = probability[order]
    target = y_true[order].astype(np.int64, copy=False)
    changes = np.flatnonzero(score[:-1] != score[1:])
    ends = np.concatenate([changes, np.asarray([len(score) - 1], dtype=np.int64)])
    cumulative_positive = np.cumsum(target, dtype=np.int64)
    tp = cumulative_positive[ends].astype(np.float64)
    predicted_positive = (ends + 1).astype(np.float64)
    fp = predicted_positive - tp
    positives = float(y_true.sum(dtype=np.uint64))
    negatives = float(len(y_true) - int(positives))
    fn = positives - tp
    tn = negatives - fp
    thresholds = score[ends]
    thresholds = np.concatenate([[np.nextafter(float(score[0]), math.inf)], thresholds])
    tp = np.concatenate([[0.0], tp])
    fp = np.concatenate([[0.0], fp])
    fn = np.concatenate([[positives], fn])
    tn = np.concatenate([[negatives], tn])
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator != 0)
    # Contract tie-break: higher MCC, closest to 0.5, then lower threshold.
    ranking = np.lexsort((thresholds, np.abs(thresholds - 0.5), -mcc))
    index = int(ranking[0])
    return {
        "threshold": float(thresholds[index]),
        "search_boundaries": int(len(thresholds)),
        "optimized_mcc": float(mcc[index]),
        "tie_break": ["HIGHER_MCC", "CLOSEST_TO_0.5", "LOWER_THRESHOLD"],
    }


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics = {
        "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "predicted_classes": sorted(int(value) for value in np.unique(prediction)),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }
    return metrics, prediction


def selection_key(result: dict) -> tuple:
    metrics = result["calibration_metrics"]
    return (
        -metrics["mcc"], -metrics["balanced_accuracy"],
        result["propagation_hops"], result["candidate_id"],
    )


def metrics_row(result: dict) -> dict:
    metrics = result["calibration_metrics"]
    confusion = metrics["confusion_matrix"]
    return {
        "candidate_id": result["candidate_id"],
        "propagation_hops": result["propagation_hops"],
        "graph_features": result["graph_features"],
        "combined_features": result["combined_features"],
        "alpha": format(result["alpha"], ".17g"),
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


def write_predictions(path: Path, matrix: dict, probability: np.ndarray, prediction: np.ndarray, candidate_id: str, threshold: float) -> None:
    deterministic_npz(path, {
        "candidate_id": np.asarray([candidate_id]),
        "partition": np.asarray(["DEV_CALIBRATION"]),
        "prediction": prediction.astype(np.uint8, copy=False),
        "probability": probability.astype("<f8", copy=False),
        "record_id": matrix["record_id"].astype("<u4", copy=False),
        "site_row": matrix["site_row"].astype("<u2", copy=False),
        "stuck_value": matrix["stuck_value"].astype(np.uint8, copy=False),
        "target_detected": matrix["target"].astype(np.uint8, copy=False),
        "threshold": np.asarray([threshold], dtype="<f8"),
        "vector_row": matrix["vector_row"].astype(np.uint8, copy=False),
    })


def write_weights(path: Path, bundle: dict) -> None:
    arrays = estimator_arrays(bundle["estimator"])
    arrays.update({
        "candidate_id": np.asarray([bundle["candidate"]["candidate_id"]]),
        "graph_scaler_mean": np.asarray(bundle["graph_scaler_mean"], dtype="<f8"),
        "graph_scaler_scale": np.asarray(bundle["graph_scaler_scale"], dtype="<f8"),
        "graph_feature_columns": np.asarray(bundle["graph_feature_columns"]),
    })
    deterministic_npz(path, arrays)


def write_calibration_curve(path: Path, y_true: np.ndarray, probability: np.ndarray, bins: int = 20) -> None:
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, bins)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "bin_index", "sample_count", "minimum_probability",
            "maximum_probability", "mean_predicted_probability",
            "observed_positive_fraction",
        ], lineterminator="\n")
        writer.writeheader()
        for bin_index, indices in enumerate(groups):
            values = probability[indices]
            targets = y_true[indices]
            writer.writerow({
                "bin_index": bin_index,
                "sample_count": len(indices),
                "minimum_probability": format(float(values[0]), ".17g"),
                "maximum_probability": format(float(values[-1]), ".17g"),
                "mean_predicted_probability": format(float(values.mean(dtype=np.float64)), ".17g"),
                "observed_positive_fraction": format(float(targets.mean(dtype=np.float64)), ".17g"),
            })
    temporary.replace(path)


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-1C script in project root")
    require(not AUDIT.exists(), f"Stage 11D-1C already frozen: {relative(AUDIT)}")
    require(not MANIFEST.exists(), f"Stage 11D-1C manifest already exists: {relative(MANIFEST)}")
    require(not SELECTION_LOCK.exists(), f"Stage 11D-1C selection already locked: {relative(SELECTION_LOCK)}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11D-1C — GNN MODEL TRAINING AND CALIBRATION")
    print("FROZEN INPUT VERIFICATION", flush=True)
    evidence = {relative(path): verify_file(path, expected) for path, expected in EXPECTED_INPUTS.items()}
    _, contract, _ = verify_contracts()
    environment = verify_environment()
    candidates = load_candidates(contract)

    print("\nLOADING AUTHORIZED MATRICES", flush=True)
    train = load_matrix(TRAIN_MATRIX, "DEV_TRAIN", load_record_ids=False)
    calibration = load_matrix(CALIBRATION_MATRIX, "DEV_CALIBRATION", load_record_ids=True)
    require(np.array_equal(train["feature_columns"], calibration["feature_columns"]), "feature-column identity")
    require(np.array_equal(train["vector_ids"], calibration["vector_ids"]), "vector identity")
    require(not (set(train["site_ids"].tolist()) & set(calibration["site_ids"].tolist())), "site split overlap")
    print("  DEV_TRAIN       : LOADED FOR MODEL FITTING")
    print("  DEV_CALIBRATION : LOADED FOR SELECTION AND THRESHOLD")
    print("  DEV_SITE_TEST   : NOT OPENED FOR EVALUATION")
    print("  VALIDATION      : NOT PRESENT")
    print("  HOLDOUT_TEST    : NOT PRESENT")

    print("\nBUILDING DIRECTED SGC FEATURES", flush=True)
    graph_arrays, graph_schema = load_graph()
    graph_sets, propagation = build_graph_feature_cache(graph_arrays, graph_schema, candidates)
    print("  Propagation replay           : PASS")
    print("  Graph scaler fit             : DEV_TRAIN NODES ONLY")
    print("  Propagation cache SHA        :", propagation["artifact"]["sha256"])

    positives = int(train["target"].sum(dtype=np.uint64))
    negatives = len(train["target"]) - positives
    positive_weight = len(train["target"]) / (2.0 * positives)
    negative_weight = len(train["target"]) / (2.0 * negatives)

    print("\nSEQUENTIAL GNN CANDIDATE TRAINING", flush=True)
    candidate_results = []
    bundles = {}
    probabilities = {}
    for number, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["candidate_id"]
        print(
            f"CANDIDATE {number}/{EXPECTED_CANDIDATES}: {candidate_id} "
            f"hops={candidate['propagation_hops']} alpha={candidate['alpha']:.1e}",
            flush=True,
        )
        bundle, training = train_candidate(
            candidate, train, graph_sets[candidate_id], positive_weight,
            negative_weight, resume=True,
        )
        print("  DEV_CALIBRATION INFERENCE", flush=True)
        probability = predict_probabilities(bundle, calibration, graph_sets[candidate_id]["features"])
        threshold_search = optimal_threshold(calibration["target"], probability)
        metrics, _ = classification_metrics(calibration["target"], probability, threshold_search["threshold"])
        model_path = MODEL_ROOT / f"{candidate_id}.joblib"
        result = {
            "candidate_id": candidate_id,
            "propagation_hops": candidate["propagation_hops"],
            "graph_features": candidate["graph_features"],
            "combined_features": candidate["combined_features"],
            "alpha": candidate["alpha"],
            "epochs": candidate["epochs"],
            "training": training,
            "threshold_search": threshold_search,
            "calibration_metrics": metrics,
            "model": output_record(model_path),
        }
        candidate_results.append(result)
        bundles[candidate_id] = bundle
        probabilities[candidate_id] = probability
        print(
            f"  CALIBRATION threshold={metrics['threshold']:.9g} "
            f"MCC={metrics['mcc']:.6f} balanced={metrics['balanced_accuracy']:.6f} "
            f"F1={metrics['f1_score']:.6f} recall={metrics['recall']:.6f}",
            flush=True,
        )

    metrics_tmp = CANDIDATE_METRICS_CSV.with_name(CANDIDATE_METRICS_CSV.name + ".tmp")
    with metrics_tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRICS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_row(result) for result in candidate_results)
    metrics_tmp.replace(CANDIDATE_METRICS_CSV)
    atomic_json(CANDIDATE_METRICS_JSON, {
        "stage": "11D-1C", "status": "PASS", "partition": "DEV_CALIBRATION",
        "selection_metric": "MCC", "candidate_results": candidate_results,
    })

    selected = sorted(candidate_results, key=selection_key)[0]
    selected_id = selected["candidate_id"]
    selected_bundle = bundles[selected_id]
    selected_probability = probabilities[selected_id]
    selected_threshold = float(selected["calibration_metrics"]["threshold"])
    selected_prediction = (selected_probability >= selected_threshold).astype(np.uint8)
    atomic_copy(MODEL_ROOT / f"{selected_id}.joblib", SELECTED_MODEL)
    write_weights(SELECTED_WEIGHTS, selected_bundle)
    write_predictions(SELECTED_PREDICTIONS, calibration, selected_probability, selected_prediction, selected_id, selected_threshold)
    write_calibration_curve(CALIBRATION_CURVE, calibration["target"], selected_probability)

    print(f"\nDETERMINISTIC SELECTED-GNN REPLAY: {selected_id}", flush=True)
    selected_candidate = next(item for item in candidates if item["candidate_id"] == selected_id)
    replay_bundle, replay_training = train_candidate(
        selected_candidate, train, graph_sets[selected_id], positive_weight,
        negative_weight, resume=False, replay=True,
    )
    replay_probability = predict_probabilities(replay_bundle, calibration, graph_sets[selected_id]["features"])
    replay_prediction = (replay_probability >= selected_threshold).astype(np.uint8)
    require(bundles_exact(selected_bundle, replay_bundle), "selected coefficient/scaler replay")
    require(np.array_equal(selected_probability, replay_probability), "selected probability replay")
    require(np.array_equal(selected_prediction, replay_prediction), "selected prediction replay")
    atomic_joblib(REPLAY_MODEL, replay_bundle)
    write_weights(REPLAY_WEIGHTS, replay_bundle)
    write_predictions(REPLAY_PREDICTIONS, calibration, replay_probability, replay_prediction, selected_id, selected_threshold)
    require(sha256(SELECTED_WEIGHTS) == sha256(REPLAY_WEIGHTS), "selected weight-artifact replay")
    require(sha256(SELECTED_PREDICTIONS) == sha256(REPLAY_PREDICTIONS), "selected prediction-artifact replay")

    selected_metrics = selected["calibration_metrics"]
    minimum_checks = {
        "mcc_greater_than_zero": selected_metrics["mcc"] > 0.0,
        "both_predicted_classes_present": selected_metrics["predicted_classes"] == [0, 1],
        "finite_probabilities": bool(np.isfinite(selected_probability).all()),
        "unknown_labels": 0,
        "deterministic_replay_exact": True,
    }
    minimum_status = "PASS" if all(
        value is True or (key == "unknown_labels" and value == 0)
        for key, value in minimum_checks.items()
    ) else "NOT_MET"

    created_at = datetime.now(timezone.utc).isoformat()
    atomic_json(SELECTION_LOCK, {
        "stage": "11D-1C",
        "title": "GNN MODEL SELECTION AND THRESHOLD LOCK",
        "status": "PASS",
        "lock_status": "FROZEN",
        "created_at_utc": created_at,
        "selected_candidate_id": selected_id,
        "selected_propagation_hops": selected["propagation_hops"],
        "selection_partition": "DEV_CALIBRATION",
        "selection_metric": "MCC",
        "selected_threshold": selected_threshold,
        "selected_calibration_metrics": selected_metrics,
        "threshold_search": selected["threshold_search"],
        "minimum_validity_status": minimum_status,
        "minimum_validity_checks": minimum_checks,
        "project_target_status": "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON",
        "selected_model": output_record(SELECTED_MODEL),
        "selected_weights": output_record(SELECTED_WEIGHTS),
        "selected_calibration_predictions": output_record(SELECTED_PREDICTIONS),
        "selected_calibration_curve": output_record(CALIBRATION_CURVE),
        "deterministic_replay": {
            "status": "PASS", "weights_exact": True,
            "probabilities_exact": True, "predictions_exact": True,
            "replay_model": output_record(REPLAY_MODEL),
            "replay_weights": output_record(REPLAY_WEIGHTS),
            "replay_predictions": output_record(REPLAY_PREDICTIONS),
            "replay_training": replay_training,
        },
        "candidate_metrics_csv": output_record(CANDIDATE_METRICS_CSV),
        "candidate_metrics_json": output_record(CANDIDATE_METRICS_JSON),
        "propagation_cache": output_record(PROPAGATION_CACHE),
        "dev_site_test_opened_for_evaluation": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "hybrid_training_authorized": False,
        "contract_sha256": EXPECTED_INPUTS[CONTRACT],
        "graph_sha256": EXPECTED_INPUTS[GRAPH_NPZ],
        "train_matrix_sha256": EXPECTED_INPUTS[TRAIN_MATRIX],
        "calibration_matrix_sha256": EXPECTED_INPUTS[CALIBRATION_MATRIX],
    })
    lock_record = output_record(SELECTION_LOCK)

    manifest = {
        "stage": "11D-1C",
        "title": "GNN MODEL TRAINING AND CALIBRATION EXECUTION",
        "status": "PASS",
        "trainer_version": VERSION,
        "created_at_utc": created_at,
        "model_family": MODEL_FAMILY,
        "fault_scope": "persistent net-stem SA0/SA1 only",
        "graph_commitment": EXPECTED_GRAPH_COMMITMENT,
        "graph_nodes": EXPECTED_NODES,
        "graph_edges": EXPECTED_EDGES,
        "train_samples": EXPECTED_TRAIN_ROWS,
        "calibration_samples": EXPECTED_CALIBRATION_ROWS,
        "candidate_count": EXPECTED_CANDIDATES,
        "epochs_per_candidate": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "candidate_training_sample_updates": EXPECTED_CANDIDATES * EXPECTED_EPOCHS * EXPECTED_TRAIN_ROWS,
        "selected_replay_sample_updates": EXPECTED_EPOCHS * EXPECTED_TRAIN_ROWS,
        "selected_candidate_id": selected_id,
        "selected_threshold": selected_threshold,
        "selected_calibration_metrics": selected_metrics,
        "minimum_validity_status": minimum_status,
        "project_target_status": "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON",
        "propagation": propagation,
        "candidate_models": {item["candidate_id"]: item["model"] for item in candidate_results},
        "candidate_metrics_csv": output_record(CANDIDATE_METRICS_CSV),
        "candidate_metrics_json": output_record(CANDIDATE_METRICS_JSON),
        "selected_model": output_record(SELECTED_MODEL),
        "selected_weights": output_record(SELECTED_WEIGHTS),
        "selected_calibration_predictions": output_record(SELECTED_PREDICTIONS),
        "selected_calibration_curve": output_record(CALIBRATION_CURVE),
        "replay_model": output_record(REPLAY_MODEL),
        "replay_weights": output_record(REPLAY_WEIGHTS),
        "replay_calibration_predictions": output_record(REPLAY_PREDICTIONS),
        "selection_lock": lock_record,
        "input_evidence": evidence,
        "environment": environment,
        "dev_site_test_opened_for_evaluation": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "hybrid_training_authorized": False,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "deep_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = output_record(MANIFEST)

    audit = {
        "stage": "11D-1C",
        "title": "GNN MODEL TRAINING AND CALIBRATION FREEZE",
        "status": "PASS",
        "training_status": "FROZEN",
        "model_selection_status": "FROZEN",
        "threshold_status": "FROZEN",
        "model_family": MODEL_FAMILY,
        "candidates_trained": EXPECTED_CANDIDATES,
        "epochs_per_candidate": EXPECTED_EPOCHS,
        "selected_candidate_id": selected_id,
        "selected_propagation_hops": selected["propagation_hops"],
        "selected_graph_features": selected["graph_features"],
        "selected_combined_features": selected["combined_features"],
        "selected_threshold": selected_threshold,
        "calibration_mcc": selected_metrics["mcc"],
        "calibration_balanced_accuracy": selected_metrics["balanced_accuracy"],
        "calibration_precision": selected_metrics["precision"],
        "calibration_recall": selected_metrics["recall"],
        "calibration_specificity": selected_metrics["specificity"],
        "calibration_f1": selected_metrics["f1_score"],
        "calibration_pr_auc": selected_metrics["pr_auc"],
        "calibration_roc_auc": selected_metrics["roc_auc"],
        "calibration_brier_score": selected_metrics["brier_score"],
        "minimum_validity_status": minimum_status,
        "project_target_status_on_calibration": "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON",
        "deterministic_propagation_replay": "PASS",
        "deterministic_training_replay": "PASS",
        "weights_exact": True,
        "probabilities_exact": True,
        "predictions_exact": True,
        "dev_site_test_state": "LOCKED",
        "dev_site_test_opened_for_evaluation": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_model_frozen": True,
        "hybrid_training_authorized": False,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "deep_model_modified": False,
        "selected_model": output_record(SELECTED_MODEL),
        "selection_lock": lock_record,
        "manifest": manifest_record,
        "next_gate": "STAGE 11D-1D — LOCKED GNN DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS",
    }
    atomic_json(AUDIT, audit)
    audit_record = output_record(AUDIT)

    print("\nSTAGE 11D-1C — GNN MODEL TRAINING AND CALIBRATION FREEZE")
    print("Status                         : PASS")
    print("Training status                : FROZEN")
    print("Model selection status         : FROZEN")
    print("Threshold status               : FROZEN")
    print("Model family                   : DIRECTED BIDIRECTIONAL SGC")
    print("Candidates trained             :", EXPECTED_CANDIDATES)
    print("Epochs per candidate           :", EXPECTED_EPOCHS)
    print("Selected candidate             :", selected_id)
    print("Selected propagation hops      :", selected["propagation_hops"])
    print("Selected graph features        :", selected["graph_features"])
    print("Selected combined features     :", selected["combined_features"])
    print("Selected threshold             :", format(selected_threshold, ".17g"))
    print("Calibration MCC                :", f"{selected_metrics['mcc']:.8f}")
    print("Calibration balanced accuracy  :", f"{selected_metrics['balanced_accuracy']:.8f}")
    print("Calibration precision          :", f"{selected_metrics['precision']:.8f}")
    print("Calibration recall             :", f"{selected_metrics['recall']:.8f}")
    print("Calibration specificity        :", f"{selected_metrics['specificity']:.8f}")
    print("Calibration F1                 :", f"{selected_metrics['f1_score']:.8f}")
    print("Calibration PR-AUC             :", f"{selected_metrics['pr_auc']:.8f}")
    print("Calibration ROC-AUC            :", f"{selected_metrics['roc_auc']:.8f}")
    print("Calibration Brier              :", f"{selected_metrics['brier_score']:.8f}")
    print("Minimum validity               :", minimum_status)
    print("Project target                 : PENDING LOCKED SITE TEST")
    print("Propagation replay             : PASS")
    print("Training replay                : PASS")
    print("Weights exact                  : YES")
    print("Probabilities exact            : YES")
    print("DEV_SITE_TEST opened           : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Graph dataset modified         : NO")
    print("Selected model                 :", SELECTED_MODEL)
    print("Selected model SHA             :", sha256(SELECTED_MODEL))
    print("Selection lock                 :", SELECTION_LOCK)
    print("Selection lock SHA             :", lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-1D — LOCKED GNN DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS")


if __name__ == "__main__":
    main()
