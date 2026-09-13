#!/usr/bin/env python3
"""Freeze the directed SGC architecture and leakage-safe GNN training contract."""
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
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from scipy import sparse
    from sklearn.linear_model import SGDClassifier
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, SciPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error


VERSION = "HMAC-GNN-ARCHITECTURE-CONTRACT-v1"
MODEL_FAMILY = "DIRECTED_BIDIRECTIONAL_SIMPLIFIED_GRAPH_CONVOLUTION"
RANDOM_SEED = 20_260_906
EXPECTED_NODES = 22_839
EXPECTED_EDGES = 47_617
EXPECTED_RAW_BRANCHES = 47_631
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_SAMPLE_FEATURES = 527
EXPECTED_TRAIN_SAMPLES = 2_046_336
EXPECTED_CALIBRATION_SAMPLES = 438_528
EXPECTED_SITE_TEST_SAMPLES = 438_528
EXPECTED_TRAIN_SITES = 15_987
EXPECTED_CALIBRATION_SITES = 3_426
EXPECTED_SITE_TEST_SITES = 3_426
EXPECTED_GRAPH_COMMITMENT = "2020aa2959c0919df336da93087d0cad5086711cc9f9bcb185673d2a93db614f"
BASE_NODE_NUMERIC_FEATURES = 6

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11d1"
GRAPH_ROOT = RESULT_ROOT / "graph_dataset_11d1a"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
FEATURE_ROOT = ROOT / "results/hmac_fault_campaign_11c5/feature_matrix_11c5e"
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"

GRAPH_BUILDER = ROOT / "stage_11d1a_graph_dataset.py"
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
TASK_SPLIT_POLICY = ROOT / "config/diagnostic_model/hmac_diagnostic_task_feature_split_policy_11c5d.json"
SITE_SPLIT = RESULT_11C5 / "hmac_fault_site_group_split_11c5d.csv"
GNN_READINESS_POLICY = ROOT / "config/diagnostic_model/hmac_gnn_readiness_policy_11c5l.json"
COMPARATOR_REGISTRY = RESULT_11C5 / "hmac_frozen_diagnostic_comparator_registry_11c5l.csv"
READINESS_AUDIT = RESULT_11C5 / "hmac_diagnostic_baseline_disposition_gnn_readiness_freeze_11c5l.json"

ARCHITECTURE = CONFIG_ROOT / "hmac_gnn_architecture_11d1b.json"
TRAINING_CONTRACT = CONFIG_ROOT / "hmac_gnn_training_contract_11d1b.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_gnn_candidate_grid_11d1b.csv"
ENVIRONMENT = RESULT_ROOT / "hmac_gnn_environment_11d1b.json"
AUDIT = RESULT_ROOT / "hmac_gnn_architecture_training_contract_freeze_11d1b.json"

EXPECTED_INPUTS = {
    GRAPH_BUILDER: "cbf2a524ad6c2db5ea5ada161d8b94a23bd26c0b1d7ed1ad4978e44b763f1b5f",
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
    TASK_SPLIT_POLICY: "74243fa3b40f047a9f25a23c87eaccf6ac5a71a07241efef58df6b9e8e5322be",
    SITE_SPLIT: "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    GNN_READINESS_POLICY: "702eb89f214aa2e70b1c95850d622ea15678ec71667639b0e7edc5f6cf3dd9fe",
    COMPARATOR_REGISTRY: "41dca9b233950636fbd93c0011e94fbc21fc237baf72faf242de410c8386bc2c",
    READINESS_AUDIT: "9fca71d7ab347e0c20c09ad2da5d42c0d4bbb7c47a9858e38680eb98dfcd49d1",
}

CANDIDATE_COLUMNS = [
    "candidate_id",
    "model_family",
    "propagation_hops",
    "direction_channels",
    "self_channel",
    "base_node_features",
    "graph_features",
    "sample_features",
    "combined_features",
    "classifier",
    "loss",
    "alpha",
    "learning_rate",
    "initial_learning_rate",
    "epochs",
    "batch_size",
    "class_weighting",
    "shuffle",
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


def record(path: Path) -> dict:
    return {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}


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
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual}",
    )
    print(f"  {path.name:<74}: OK", flush=True)
    return record(path)


def verify_outputs_absent() -> None:
    for path in (ARCHITECTURE, TRAINING_CONTRACT, CANDIDATE_GRID, ENVIRONMENT, AUDIT):
        require(not path.exists(), f"Stage 11D-1B output already exists: {relative(path)}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (ARCHITECTURE, TRAINING_CONTRACT, CANDIDATE_GRID, ENVIRONMENT):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def verify_semantics() -> tuple[dict, dict, dict, dict]:
    graph_schema = load_json(GRAPH_SCHEMA)
    graph_manifest = load_json(GRAPH_MANIFEST)
    graph_audit = load_json(GRAPH_AUDIT)
    readiness = load_json(GNN_READINESS_POLICY)
    readiness_audit = load_json(READINESS_AUDIT)
    feature_audit = load_json(FEATURE_AUDIT)
    task_policy = load_json(TASK_SPLIT_POLICY)

    require(graph_schema.get("status") == "PASS", "graph schema status")
    require(graph_schema.get("schema_version") == "HMAC-GOLDEN-NETLIST-GRAPH-SCHEMA-v1", "graph schema version")
    require(graph_schema.get("physical_site_count") == EXPECTED_NODES, "graph schema node count")
    require(graph_schema.get("fault_instance_count") == EXPECTED_FAULT_INSTANCES, "graph schema fault instances")
    require(graph_schema.get("target_in_graph_features") is False, "graph target leakage")
    require(graph_schema.get("post_simulation_features") == 0, "graph post-simulation features")

    require(graph_manifest.get("status") == "PASS", "graph manifest status")
    require(graph_manifest.get("dataset_status") == "FROZEN", "graph dataset freeze")
    require(graph_manifest.get("schema_status") == "FROZEN", "graph schema freeze")
    require(graph_manifest.get("topology_status") == "FROZEN", "graph topology freeze")
    require(graph_manifest.get("graph_commitment") == EXPECTED_GRAPH_COMMITMENT, "graph commitment")
    require(graph_manifest.get("node_count") == EXPECTED_NODES, "graph node count")
    require(graph_manifest.get("collapsed_edge_count") == EXPECTED_EDGES, "graph edge count")
    require(graph_manifest.get("raw_internal_branch_count") == EXPECTED_RAW_BRANCHES, "raw branch count")
    require(graph_manifest.get("deterministic_replay") == "PASS", "graph deterministic replay")

    require(graph_audit.get("status") == "PASS", "graph audit status")
    require(graph_audit.get("graph_dataset_status") == "FROZEN", "graph audit dataset freeze")
    require(graph_audit.get("topology_integrity_status") == "PASS", "topology integrity")
    require(graph_audit.get("undriven_input_branches") == 0, "undriven graph branches")
    require(graph_audit.get("fanout_mismatches") == 0, "graph fanout mismatches")
    require(graph_audit.get("overlapping_sites") == 0, "graph split overlap")
    require(graph_audit.get("sa0_sa1_separations") == 0, "graph SA0/SA1 split")
    require(graph_audit.get("gnn_architecture_contract_generation") == "AUTHORIZED", "GNN contract gate")
    require(graph_audit.get("gnn_training") == "NOT_YET_AUTHORIZED", "prior GNN training gate")

    require(readiness.get("status") == "PASS", "GNN readiness status")
    require(readiness.get("graph_dataset_construction_authorized") is True, "graph construction authorization")
    require(readiness.get("gnn_training_authorized") is False, "11C-5L GNN training gate")
    require(readiness_audit.get("status") == "PASS", "readiness audit status")
    require(readiness_audit.get("winner_declared") is False, "comparator winner disposition")

    require(feature_audit.get("status") == "PASS", "feature-matrix status")
    require(feature_audit.get("feature_matrix_status") == "FROZEN", "feature-matrix freeze")
    require(feature_audit.get("model_feature_count") == EXPECTED_SAMPLE_FEATURES, "sample feature count")
    partition_samples = feature_audit.get("partition_samples", {})
    require(partition_samples.get("DEV_TRAIN") == EXPECTED_TRAIN_SAMPLES, "DEV_TRAIN samples")
    require(partition_samples.get("DEV_CALIBRATION") == EXPECTED_CALIBRATION_SAMPLES, "DEV_CALIBRATION samples")
    require(partition_samples.get("DEV_SITE_TEST") == EXPECTED_SITE_TEST_SAMPLES, "DEV_SITE_TEST samples")
    require(feature_audit.get("post_simulation_features") == 0, "sample post-simulation features")
    require(feature_audit.get("target_in_feature_columns") is False, "sample target leakage")

    task_features = task_policy.get("diagnostic_task_and_features", {})
    task = task_features.get("task", {})
    require(task.get("name") == "pre_simulation_persistent_fault_detectability", "diagnostic task")
    require(task.get("target") == "detected", "diagnostic target")
    require(task_policy.get("validation_campaign_authorized") is False, "validation gate")
    require(task_policy.get("holdout_campaign_authorized") is False, "holdout gate")
    return graph_schema, graph_manifest, graph_audit, feature_audit


def load_graph_arrays(graph_schema: dict) -> dict[str, np.ndarray]:
    expected_members = set(graph_schema.get("npz_arrays", {}))
    require(expected_members, "graph NPZ schema members")
    with np.load(GRAPH_NPZ, allow_pickle=False) as archive:
        require(set(archive.files) == expected_members, "graph NPZ member set")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, specification in graph_schema["npz_arrays"].items():
        array = arrays[name]
        require(list(array.shape) == specification["shape"], f"graph array shape {name}")
        require(array.dtype.str == specification["dtype"], f"graph array dtype {name}")

    edge_index = arrays["edge_index"].astype(np.int64, copy=False)
    require(edge_index.shape == (2, EXPECTED_EDGES), "edge-index shape")
    require(int(edge_index.min()) >= 0, "negative edge index")
    require(int(edge_index.max()) < EXPECTED_NODES, "edge index outside node universe")
    require(int(arrays["edge_multiplicity"].sum(dtype=np.uint64)) == EXPECTED_RAW_BRANCHES, "edge multiplicity total")
    require(int(arrays["edge_is_self_loop"].sum(dtype=np.uint64)) == 0, "self-loop count")
    require(np.array_equal(arrays["node_site_index"], np.arange(1, EXPECTED_NODES + 1, dtype=arrays["node_site_index"].dtype)), "node/site indexing")
    partitions = np.bincount(arrays["node_partition_code"].astype(np.int64), minlength=3)
    require(partitions.tolist() == [EXPECTED_TRAIN_SITES, EXPECTED_CALIBRATION_SITES, EXPECTED_SITE_TEST_SITES], "node partition counts")
    return arrays


def base_node_features(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    driver_code = arrays["node_driver_type_code"].astype(np.int64)
    category_code = arrays["node_site_category_code"].astype(np.int64)
    driver_count = int(driver_code.max()) + 1
    category_count = int(category_code.max()) + 1
    require(driver_count >= 2, "driver-type vocabulary")
    require(category_count == 2, "site-category vocabulary")

    driver_onehot = np.eye(driver_count, dtype=np.float32)[driver_code]
    category_onehot = np.eye(category_count, dtype=np.float32)[category_code]
    numeric_raw = np.column_stack([
        arrays["node_is_primary_output"],
        arrays["node_cell_fanout"],
        arrays["node_in_degree_branches"],
        arrays["node_out_degree_branches"],
        arrays["node_in_neighbor_count"],
        arrays["node_out_neighbor_count"],
    ]).astype(np.float32)
    numeric = np.log1p(numeric_raw)
    features = np.concatenate([driver_onehot, category_onehot, numeric], axis=1)
    require(features.shape[0] == EXPECTED_NODES, "base-node feature rows")
    require(np.isfinite(features).all(), "finite base-node features")
    return features, {
        "driver_cell_type_features": driver_count,
        "site_category_features": category_count,
        "numeric_structural_features": BASE_NODE_NUMERIC_FEATURES,
        "base_node_features": features.shape[1],
        "numeric_transforms": "LOG1P",
        "partition_used_as_feature": False,
        "site_id_used_as_feature": False,
        "target_used_as_feature": False,
    }


def normalized_adjacencies(arrays: dict[str, np.ndarray]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    edge_index = arrays["edge_index"].astype(np.int64, copy=False)
    source = edge_index[0]
    destination = edge_index[1]
    weight = arrays["edge_multiplicity"].astype(np.float32)
    identity_index = np.arange(EXPECTED_NODES, dtype=np.int64)
    identity_weight = np.ones(EXPECTED_NODES, dtype=np.float32)

    inbound = sparse.coo_matrix(
        (
            np.concatenate([weight, identity_weight]),
            (
                np.concatenate([destination, identity_index]),
                np.concatenate([source, identity_index]),
            ),
        ),
        shape=(EXPECTED_NODES, EXPECTED_NODES),
        dtype=np.float32,
    ).tocsr()
    outbound = sparse.coo_matrix(
        (
            np.concatenate([weight, identity_weight]),
            (
                np.concatenate([source, identity_index]),
                np.concatenate([destination, identity_index]),
            ),
        ),
        shape=(EXPECTED_NODES, EXPECTED_NODES),
        dtype=np.float32,
    ).tocsr()

    def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        require(np.all(row_sum > 0.0), "positive adjacency row sums")
        result = sparse.diags((1.0 / row_sum).astype(np.float32)) @ matrix
        result.sort_indices()
        return result.tocsr()

    return row_normalize(inbound), row_normalize(outbound)


def propagation_canary(
    base: np.ndarray,
    inbound: sparse.csr_matrix,
    outbound: sparse.csr_matrix,
) -> dict:
    canary = base[:, : min(4, base.shape[1])].astype(np.float32, copy=True)

    def execute() -> tuple[np.ndarray, np.ndarray]:
        incoming = canary.copy()
        outgoing = canary.copy()
        for _ in range(3):
            incoming = np.asarray(inbound @ incoming, dtype=np.float32)
            outgoing = np.asarray(outbound @ outgoing, dtype=np.float32)
        return incoming, outgoing

    first_in, first_out = execute()
    second_in, second_out = execute()
    require(np.array_equal(first_in, second_in), "inbound propagation replay")
    require(np.array_equal(first_out, second_out), "outbound propagation replay")
    require(np.isfinite(first_in).all() and np.isfinite(first_out).all(), "finite propagation canary")
    digest = hashlib.sha256()
    digest.update(first_in.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(first_out.astype("<f4", copy=False).tobytes(order="C"))
    return {
        "status": "PASS",
        "replay_exact": True,
        "hops": 3,
        "canary_features": canary.shape[1],
        "inbound_nnz": int(inbound.nnz),
        "outbound_nnz": int(outbound.nnz),
        "sha256": digest.hexdigest(),
    }


def candidates(base_features: int) -> list[dict]:
    rows = []
    for hops, alpha, alpha_id in ((2, 1e-6, "A1E6"), (3, 1e-5, "A1E5")):
        graph_features = base_features * (1 + 2 * hops)
        rows.append({
            "candidate_id": f"DIR_SGC_K{hops}_L2_{alpha_id}",
            "model_family": MODEL_FAMILY,
            "propagation_hops": hops,
            "direction_channels": "INBOUND+OUTBOUND",
            "self_channel": "BASE_FEATURES",
            "base_node_features": base_features,
            "graph_features": graph_features,
            "sample_features": EXPECTED_SAMPLE_FEATURES,
            "combined_features": EXPECTED_SAMPLE_FEATURES + graph_features,
            "classifier": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss",
            "alpha": format(alpha, ".17g"),
            "learning_rate": "optimal",
            "initial_learning_rate": "0",
            "epochs": 6,
            "batch_size": 32768,
            "class_weighting": "GLOBAL_DEV_TRAIN_BALANCED_SAMPLE_WEIGHT",
            "shuffle": "DETERMINISTIC_EPOCH_PERMUTATION",
            "random_seed": RANDOM_SEED,
        })
    return rows


def environment_document(canary: dict) -> dict:
    packages = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": package_version("threadpoolctl"),
    }
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-6,
        learning_rate="optimal",
        random_state=RANDOM_SEED,
        shuffle=False,
        max_iter=1,
        tol=None,
    )
    require(estimator.loss == "log_loss", "SGDClassifier log-loss support")
    require(sparse.isspmatrix_csr(sparse.eye(2, format="csr")), "SciPy CSR support")
    memory_bytes = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        memory_bytes = int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "stage": "11D-1B",
        "status": "PASS",
        "environment_status": "FROZEN",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
        "packages": packages,
        "required_external_gnn_framework": "NONE",
        "pytorch_required": False,
        "torch_geometric_required": False,
        "sparse_backend": "scipy.sparse.csr_matrix",
        "classifier_backend": "sklearn.linear_model.SGDClassifier",
        "thread_limits": {
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        "propagation_canary": canary,
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-1B script in project root")
    verify_outputs_absent()

    print("STAGE 11D-1B — GNN MODEL ARCHITECTURE AND TRAINING-CONTRACT")
    print("FROZEN INPUT VERIFICATION", flush=True)
    evidence = {
        relative(path): verify_input(path, expected_sha)
        for path, expected_sha in EXPECTED_INPUTS.items()
    }
    graph_schema, graph_manifest, graph_audit, feature_audit = verify_semantics()
    print("  Frozen graph and leakage contracts                                      : PASS")

    print("\nGRAPH BACKEND AND PROPAGATION CANARY", flush=True)
    graph_arrays = load_graph_arrays(graph_schema)
    base_features, base_feature_schema = base_node_features(graph_arrays)
    inbound, outbound = normalized_adjacencies(graph_arrays)
    canary = propagation_canary(base_features, inbound, outbound)
    print("  Sparse backend              : scipy.sparse CSR")
    print("  Base node features          :", base_features.shape[1])
    print("  Inbound normalized nnz      :", inbound.nnz)
    print("  Outbound normalized nnz     :", outbound.nnz)
    print("  Three-hop deterministic test: PASS")
    print("  Canary SHA                  :", canary["sha256"])

    candidate_rows = candidates(base_features.shape[1])
    require(len(candidate_rows) == 2, "candidate count")
    atomic_csv(CANDIDATE_GRID, candidate_rows)
    candidate_grid_record = record(CANDIDATE_GRID)

    created_at = datetime.now(timezone.utc).isoformat()
    architecture = {
        "stage": "11D-1B",
        "title": "GNN MODEL ARCHITECTURE FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "architecture_version": VERSION,
        "created_at_utc": created_at,
        "model_family": MODEL_FAMILY,
        "is_gnn": True,
        "is_deep_backpropagated_gnn": False,
        "design_rationale": (
            "Simplified graph convolution is a recognized GNN that propagates "
            "features over the frozen netlist before fitting a sample classifier. "
            "It is selected for deterministic CPU execution on a 16-GB laptop."
        ),
        "graph": {
            "commitment": EXPECTED_GRAPH_COMMITMENT,
            "nodes": EXPECTED_NODES,
            "collapsed_directed_edges": EXPECTED_EDGES,
            "raw_internal_branches": EXPECTED_RAW_BRANCHES,
            "direction": "DRIVER_TO_SINK",
            "parallel_edges": "MULTIPLICITY_WEIGHTED",
            "self_loops": "ADDED_FOR_PROPAGATION_ONLY",
        },
        "node_feature_encoder": base_feature_schema,
        "propagation": {
            "operator": "ROW_NORMALIZED_MULTIPLICITY_WEIGHTED_ADJACENCY",
            "channels": ["SELF", "INBOUND", "OUTBOUND"],
            "candidate_hops": [2, 3],
            "concatenation": "SELF + EACH_INBOUND_HOP + EACH_OUTBOUND_HOP",
            "labels_used_during_propagation": False,
            "partition_used_during_propagation": False,
            "topology_setting": "TRANSDUCTIVE_STRUCTURE_ONLY",
        },
        "sample_head": {
            "estimator": "sklearn.linear_model.SGDClassifier",
            "loss": "LOG_LOSS",
            "penalty": "L2",
            "input": "527_LEAKAGE_SAFE_SAMPLE_FEATURES + PROPAGATED_NODE_FEATURES",
            "output": "DETECTABILITY_PROBABILITY",
        },
        "candidate_grid": candidate_grid_record,
        "propagation_canary": canary,
        "target_present_in_input": False,
        "post_simulation_features": 0,
    }
    atomic_json(ARCHITECTURE, architecture)
    architecture_record = record(ARCHITECTURE)

    training_contract = {
        "stage": "11D-1B",
        "title": "GNN TRAINING CONTRACT FREEZE",
        "status": "PASS",
        "training_contract_status": "FROZEN",
        "created_at_utc": created_at,
        "model_family": MODEL_FAMILY,
        "primary_task": "PRE_SIMULATION_BINARY_DETECTABILITY",
        "target": "DETECTED",
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY",
        "sample_features": EXPECTED_SAMPLE_FEATURES,
        "physical_graph_nodes": EXPECTED_NODES,
        "fault_instances": EXPECTED_FAULT_INSTANCES,
        "partitions": {
            "DEV_TRAIN": {"sites": EXPECTED_TRAIN_SITES, "samples": EXPECTED_TRAIN_SAMPLES, "use": "FIT_PARAMETERS"},
            "DEV_CALIBRATION": {"sites": EXPECTED_CALIBRATION_SITES, "samples": EXPECTED_CALIBRATION_SAMPLES, "use": "SELECT_CANDIDATE_AND_THRESHOLD"},
            "DEV_SITE_TEST": {"sites": EXPECTED_SITE_TEST_SITES, "samples": EXPECTED_SITE_TEST_SAMPLES, "use": "LOCKED_UNTIL_SELECTION_FREEZE"},
        },
        "grouping_unit": "PHYSICAL_SITE",
        "sa0_sa1_grouped": True,
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "epochs_per_candidate": 6,
        "training_batch_size": 32768,
        "random_seed": RANDOM_SEED,
        "shuffle": "DETERMINISTIC_EPOCH_PERMUTATION",
        "class_weighting": "GLOBAL_DEV_TRAIN_BALANCED_SAMPLE_WEIGHT",
        "graph_feature_scaler": "FIT_ON_DEV_TRAIN_NODES_ONLY",
        "sample_feature_scaler": "REUSE_FROZEN_DEV_TRAIN_ONLY_SCALER_CONTRACT",
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "candidate_tie_break": ["HIGHER_MCC", "HIGHER_BALANCED_ACCURACY", "LOWER_HOPS", "LEXICAL_CANDIDATE_ID"],
        "threshold_selection": "DETERMINISTIC_MCC_MAXIMIZATION_ON_DEV_CALIBRATION",
        "threshold_tie_break": ["HIGHER_MCC", "CLOSEST_TO_0.5", "LOWER_THRESHOLD"],
        "minimum_validity": [
            "MCC_GREATER_THAN_ZERO",
            "BOTH_PREDICTED_CLASSES_PRESENT",
            "FINITE_PROBABILITIES",
            "NO_UNKNOWN_LABELS",
            "DETERMINISTIC_REPLAY_EXACT",
        ],
        "project_target": {
            "metric": "MCC",
            "comparison": "GNN_MINUS_FROZEN_CONVENTIONAL_BASELINE",
            "required_point_improvement": 0.05,
            "statistical_rule": "PAIRED_PHYSICAL_SITE_95_PERCENT_CI_ENTIRELY_ABOVE_ZERO",
        },
        "locked_comparators": ["CONVENTIONAL_LOGREG_11C5H", "DEEP_MLP_11C5K"],
        "estimated_cpu_training_time": "20-90 MINUTES",
        "maximum_resident_memory_policy": "LESS_THAN_8_GIB",
        "checkpoint_after_each_candidate": True,
        "resume_supported": True,
        "internet_required": False,
        "gpu_required": False,
        "dev_site_test_opening_authorized": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "hybrid_training_authorized": False,
        "forbidden_features": [
            "detected",
            "activated",
            "actual_digest",
            "expected_digest",
            "digest_mismatch",
            "latency_mismatch",
            "timeout",
            "final_fault_classification",
            "partition_code_or_name",
            "site_id_or_site_index_identity",
        ],
        "candidate_grid": candidate_grid_record,
        "architecture": architecture_record,
    }
    atomic_json(TRAINING_CONTRACT, training_contract)
    training_contract_record = record(TRAINING_CONTRACT)

    environment = environment_document(canary)
    environment.update({
        "created_at_utc": created_at,
        "input_evidence": evidence,
        "graph_npz": record(GRAPH_NPZ),
        "graph_schema": record(GRAPH_SCHEMA),
    })
    atomic_json(ENVIRONMENT, environment)
    environment_record = record(ENVIRONMENT)

    audit = {
        "stage": "11D-1B",
        "title": "GNN MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "environment_status": "FROZEN",
        "candidate_grid_status": "FROZEN",
        "model_family": MODEL_FAMILY,
        "gnn_model": True,
        "deep_backpropagated_gnn": False,
        "graph_commitment": EXPECTED_GRAPH_COMMITMENT,
        "graph_nodes": EXPECTED_NODES,
        "graph_edges": EXPECTED_EDGES,
        "base_node_features": base_features.shape[1],
        "sample_features": EXPECTED_SAMPLE_FEATURES,
        "trainable_candidates": len(candidate_rows),
        "candidate_hops": [2, 3],
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "epochs_per_candidate": 6,
        "training_batch_size": 32768,
        "dev_train_samples": EXPECTED_TRAIN_SAMPLES,
        "dev_calibration_samples": EXPECTED_CALIBRATION_SAMPLES,
        "dev_site_test_samples": EXPECTED_SITE_TEST_SAMPLES,
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "propagation_canary": "PASS",
        "deterministic_propagation_replay": "PASS",
        "target_present_in_input": False,
        "post_simulation_features": 0,
        "dev_site_test_state": "LOCKED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_training_authorized_for_dev_train": True,
        "hybrid_training_authorized": False,
        "internet_required": False,
        "gpu_required": False,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "deep_model_modified": False,
        "architecture": architecture_record,
        "training_contract": training_contract_record,
        "candidate_grid": candidate_grid_record,
        "environment": environment_record,
        "next_gate": "STAGE 11D-1C — GNN MODEL TRAINING AND CALIBRATION EXECUTION",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    maximum_graph_features = max(int(row["graph_features"]) for row in candidate_rows)
    maximum_combined_features = max(int(row["combined_features"]) for row in candidate_rows)
    print("\nSTAGE 11D-1B — GNN MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE")
    print("Status                         : PASS")
    print("Architecture status            : FROZEN")
    print("Training contract status       : FROZEN")
    print("Environment status             : FROZEN")
    print("Candidate grid status          : FROZEN")
    print("Model family                   : DIRECTED BIDIRECTIONAL SGC")
    print("GNN model                      : YES")
    print("Deep backpropagated GNN        : NO")
    print("Graph nodes                    :", EXPECTED_NODES)
    print("Graph edges                    :", EXPECTED_EDGES)
    print("Base node features             :", base_features.shape[1])
    print("Propagation hops               : 2, 3")
    print("Direction channels             : INBOUND + OUTBOUND")
    print("Maximum graph features         :", maximum_graph_features)
    print("Sample features                :", EXPECTED_SAMPLE_FEATURES)
    print("Maximum combined features      :", maximum_combined_features)
    print("Trainable candidates           :", len(candidate_rows))
    print("Candidate execution            : SEQUENTIAL")
    print("Parallel candidates            : 1")
    print("Epochs per candidate           : 6")
    print("Training batch size            : 32768")
    print("DEV_TRAIN samples              :", EXPECTED_TRAIN_SAMPLES)
    print("DEV_CALIBRATION samples        :", EXPECTED_CALIBRATION_SAMPLES)
    print("DEV_SITE_TEST                  : LOCKED")
    print("Selection partition            : DEV_CALIBRATION")
    print("Primary metric                 : MCC")
    print("Propagation canary             : PASS")
    print("Deterministic replay           : PASS")
    print("Estimated CPU training time    : 20-90 MINUTES")
    print("Internet required              : NO")
    print("GPU required                   : NO")
    print("Target present in input        : NO")
    print("Post-simulation features       : 0")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("GNN training                   : AUTHORIZED FOR DEV_TRAIN")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Graph dataset modified         : NO")
    print("Architecture                   :", ARCHITECTURE)
    print("Architecture SHA               :", architecture_record["sha256"])
    print("Training contract              :", TRAINING_CONTRACT)
    print("Training contract SHA          :", training_contract_record["sha256"])
    print("Candidate grid                 :", CANDIDATE_GRID)
    print("Candidate grid SHA             :", candidate_grid_record["sha256"])
    print("Environment                    :", ENVIRONMENT)
    print("Environment SHA                :", environment_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-1C — GNN MODEL TRAINING AND CALIBRATION EXECUTION")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
