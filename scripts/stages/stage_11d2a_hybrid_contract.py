#!/usr/bin/env python3
"""Freeze the leakage-safe graph-augmented hybrid architecture and contract."""
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

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

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


VERSION = "HMAC-HYBRID-ARCHITECTURE-CONTRACT-v1"
MODEL_FAMILY = "GRAPH_AUGMENTED_FEATURE_FUSION_MLP"
RANDOM_SEED = 20_260_910

EXPECTED_NODES = 22_839
EXPECTED_EDGES = 47_617
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_SAMPLE_FEATURES = 527
EXPECTED_GRAPH_FEATURES = 119
EXPECTED_COMBINED_FEATURES = 646
EXPECTED_TRAIN_SAMPLES = 2_046_336
EXPECTED_CALIBRATION_SAMPLES = 438_528
EXPECTED_SITE_TEST_SAMPLES = 438_528
EXPECTED_TRAIN_SITES = 15_987
EXPECTED_CALIBRATION_SITES = 3_426
EXPECTED_SITE_TEST_SITES = 3_426
EXPECTED_GNN_CANDIDATE = "DIR_SGC_K3_L2_A1E5"
EXPECTED_GNN_MCC = 0.33662641
EXPECTED_DEEP_MCC = 0.14832380
EXPECTED_BASELINE_MCC = 0.14208149
HIDDEN_LAYERS = (128, 64, 32)
TRAINABLE_PARAMETERS = (
    (EXPECTED_COMBINED_FEATURES + 1) * HIDDEN_LAYERS[0]
    + (HIDDEN_LAYERS[0] + 1) * HIDDEN_LAYERS[1]
    + (HIDDEN_LAYERS[1] + 1) * HIDDEN_LAYERS[2]
    + (HIDDEN_LAYERS[2] + 1)
)

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
RESULT_11D1 = ROOT / "results/hmac_fault_campaign_11d1"
RESULT_11D2 = ROOT / "results/hmac_fault_campaign_11d2"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
FEATURE_ROOT = RESULT_11C5 / "feature_matrix_11c5e"
GRAPH_ROOT = RESULT_11D1 / "graph_dataset_11d1a"
GNN_TRAIN_ROOT = RESULT_11D1 / "gnn_training_11d1c"

TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CALIBRATION_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
FEATURE_AUDIT = RESULT_11C5 / "hmac_leakage_safe_feature_matrix_freeze_11c5e.json"

GRAPH_NPZ = GRAPH_ROOT / "hmac_golden_netlist_graph_11d1a.npz"
GRAPH_SCHEMA = GRAPH_ROOT / "hmac_golden_netlist_graph_schema_11d1a.json"
GRAPH_AUDIT = RESULT_11D1 / "hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"

GNN_SELECTION_LOCK = GNN_TRAIN_ROOT / "hmac_gnn_selection_lock_11d1c.json"
GNN_TRAINING_MANIFEST = RESULT_11D1 / "hmac_gnn_training_manifest_11d1c.json"
GNN_TRAINING_AUDIT = RESULT_11D1 / "hmac_gnn_training_calibration_freeze_11d1c.json"
PROPAGATION_CACHE = GNN_TRAIN_ROOT / "hmac_directed_sgc_graph_features_11d1c.npz"

GNN_COMPARISON = (
    RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_vs_frozen_comparators_11d1d.json"
)
GNN_EVALUATION_AUDIT = RESULT_11D1 / "hmac_gnn_site_test_comparator_freeze_11d1d.json"

READINESS_SOURCE = ROOT / "stage_11d1e_gnn_disposition_hybrid_readiness.py"
GNN_DISPOSITION = CONFIG_ROOT / "hmac_gnn_disposition_policy_11d1e.json"
HYBRID_READINESS = CONFIG_ROOT / "hmac_hybrid_readiness_policy_11d1e.json"
COMPARATOR_REGISTRY_CSV = RESULT_11D1 / "hmac_frozen_diagnostic_comparator_registry_11d1e.csv"
COMPARATOR_REGISTRY_JSON = RESULT_11D1 / "hmac_frozen_diagnostic_comparator_registry_11d1e.json"
READINESS_AUDIT = RESULT_11D1 / "hmac_gnn_disposition_hybrid_readiness_freeze_11d1e.json"

ARCHITECTURE = CONFIG_ROOT / "hmac_hybrid_architecture_11d2a.json"
TRAINING_CONTRACT = CONFIG_ROOT / "hmac_hybrid_training_contract_11d2a.json"
CANDIDATE_GRID = CONFIG_ROOT / "hmac_hybrid_candidate_grid_11d2a.csv"
ENVIRONMENT = RESULT_11D2 / "hmac_hybrid_environment_11d2a.json"
AUDIT = RESULT_11D2 / "hmac_hybrid_architecture_training_contract_freeze_11d2a.json"

EXPECTED_INPUTS = {
    TRAIN_MATRIX: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CALIBRATION_MATRIX: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    FEATURE_AUDIT: "6320bb732b84ae56c937e188189fb87a159d35a164eed892ec3da9d1ee27e540",
    GRAPH_NPZ: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    GRAPH_SCHEMA: "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    GRAPH_AUDIT: "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
    GNN_SELECTION_LOCK: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
    GNN_TRAINING_MANIFEST: "c40b118f9142e619218107397b324c0e14753d295952e45e32dd875534763fdf",
    GNN_TRAINING_AUDIT: "afe2c204fe2208f3a5805bf9e7bf7c7000bdef1a81dac8d651876d2cb633c079",
    GNN_COMPARISON: "5bfce5f1cb49b29dc3a17343dbd2ba5f5777e3ec11d9679efdd24b9bfbdaf0e1",
    GNN_EVALUATION_AUDIT: "3555783dc613aec71a6f89c09b6c487b514aa0cfc68a6c7b42489018dbdb5a1b",
    READINESS_SOURCE: "014465393c16d30270ebdc120595925e05e1298faee1e312762999d423dac0e0",
    GNN_DISPOSITION: "be8df4b5b5e4d0bd8a7337f27d48179e9a636ffce5a2c7b394805390cd4f4197",
    HYBRID_READINESS: "c961db485923bbd90efc09ad2bd804948a9392462aaa5436f4780da4069ffda5",
    COMPARATOR_REGISTRY_CSV: "5b6d9e3972c08e1043ed403b48dc96e612dd3b10a7890630b0d75bd099a52085",
    COMPARATOR_REGISTRY_JSON: "e611003eb2c011b9315d505013acbf66be5b09cb08321308ea2c209f28020c49",
    READINESS_AUDIT: "de89484ec1c244b141950ad259ad69163562ac70fd31e9659c8943c007e48339",
}

CANDIDATE_COLUMNS = [
    "candidate_id",
    "model_family",
    "fusion_mode",
    "sample_features",
    "graph_features",
    "combined_features",
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
        "candidate_id": "HYBRID_SGC3_MLP_128_64_32_A1E4_LR1E3",
        "model_family": MODEL_FAMILY,
        "fusion_mode": "FEATURE_LEVEL_CONCATENATION",
        "sample_features": str(EXPECTED_SAMPLE_FEATURES),
        "graph_features": str(EXPECTED_GRAPH_FEATURES),
        "combined_features": str(EXPECTED_COMBINED_FEATURES),
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
        "candidate_id": "HYBRID_SGC3_MLP_128_64_32_A1E5_LR3E4",
        "model_family": MODEL_FAMILY,
        "fusion_mode": "FEATURE_LEVEL_CONCATENATION",
        "sample_features": str(EXPECTED_SAMPLE_FEATURES),
        "graph_features": str(EXPECTED_GRAPH_FEATURES),
        "combined_features": str(EXPECTED_COMBINED_FEATURES),
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
    print(f"  {path.name:<76}: OK", flush=True)
    return record(path)


def close(actual: object, expected: float, tolerance: float = 5e-8) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def verify_outputs_absent() -> None:
    for path in (ARCHITECTURE, TRAINING_CONTRACT, CANDIDATE_GRID, ENVIRONMENT, AUDIT):
        require(not path.exists(), f"Stage 11D-2A output already exists: {relative(path)}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (ARCHITECTURE, TRAINING_CONTRACT, CANDIDATE_GRID, ENVIRONMENT):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def verify_semantics() -> tuple[dict, dict, dict]:
    feature = load_json(FEATURE_AUDIT)
    graph = load_json(GRAPH_AUDIT)
    lock = load_json(GNN_SELECTION_LOCK)
    comparison = load_json(GNN_COMPARISON)
    gnn_evaluation = load_json(GNN_EVALUATION_AUDIT)
    readiness = load_json(HYBRID_READINESS)
    readiness_audit = load_json(READINESS_AUDIT)

    require(feature.get("status") == "PASS", "feature status")
    require(feature.get("feature_matrix_status") == "FROZEN", "feature matrix freeze")
    require(feature.get("model_feature_count") == EXPECTED_SAMPLE_FEATURES, "sample feature count")
    require(feature.get("target_in_feature_columns") is False, "sample target leakage")
    require(feature.get("post_simulation_features") == 0, "post-simulation sample features")
    require(feature.get("partition_samples", {}).get("DEV_TRAIN") == EXPECTED_TRAIN_SAMPLES, "train rows")
    require(feature.get("partition_samples", {}).get("DEV_CALIBRATION") == EXPECTED_CALIBRATION_SAMPLES, "calibration rows")
    require(feature.get("partition_samples", {}).get("DEV_SITE_TEST") == EXPECTED_SITE_TEST_SAMPLES, "site-test rows")

    require(graph.get("status") == "PASS", "graph status")
    require(graph.get("graph_dataset_status") == "FROZEN", "graph freeze")
    require(graph.get("topology_integrity_status") == "PASS", "topology integrity")
    require(graph.get("legal_nodes") == EXPECTED_NODES, "graph nodes")
    require(graph.get("collapsed_directed_edges") == EXPECTED_EDGES, "graph edges")
    require(graph.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "fault instances")
    require(graph.get("target_present_in_graph_features") is False, "graph target leakage")
    require(graph.get("post_simulation_features") == 0, "post-simulation graph features")

    require(lock.get("status") == "PASS", "GNN selection lock status")
    require(lock.get("lock_status") == "FROZEN", "GNN selection lock")
    require(lock.get("selected_candidate_id") == EXPECTED_GNN_CANDIDATE, "selected GNN")
    require(lock.get("selected_propagation_hops") == 3, "selected GNN hops")
    propagation = lock.get("propagation_cache", {})
    require(propagation.get("path") == relative(PROPAGATION_CACHE), "propagation-cache path")
    require(PROPAGATION_CACHE.is_file(), "missing propagation cache")
    require(propagation.get("sha256") == sha256(PROPAGATION_CACHE), "propagation-cache SHA")

    require(comparison.get("status") == "PASS", "GNN comparison status")
    require(comparison.get("comparison_status") == "FROZEN", "GNN comparison freeze")
    require(comparison.get("point_winner") == "GNN", "GNN winner")
    metrics = comparison.get("metrics", {})
    require(close(metrics.get("gnn", {}).get("mcc"), EXPECTED_GNN_MCC), "GNN MCC")
    require(close(metrics.get("deep", {}).get("mcc"), EXPECTED_DEEP_MCC), "deep MCC")
    require(close(metrics.get("baseline", {}).get("mcc"), EXPECTED_BASELINE_MCC), "baseline MCC")
    require(comparison.get("project_target_met") is True, "GNN project target")
    require(gnn_evaluation.get("absolute_performance_status") == "NOT_MET", "absolute target")

    require(readiness.get("status") == "PASS", "hybrid readiness status")
    require(
        readiness.get("readiness_status")
        == "READY_FOR_ARCHITECTURE_AND_TRAINING_CONTRACT_FREEZE_ONLY",
        "hybrid readiness state",
    )
    require(readiness.get("hybrid_architecture_contract_authorized") is True, "hybrid contract authorization")
    require(readiness.get("hybrid_training_authorized") is False, "prior hybrid training gate")
    require(readiness.get("validation_vectors_exposed") == 0, "readiness validation exposure")
    require(readiness.get("holdout_vectors_exposed") == 0, "readiness holdout exposure")
    require(readiness_audit.get("status") == "PASS", "readiness audit status")
    require(readiness_audit.get("winner") == "GNN", "readiness audit winner")
    require(readiness_audit.get("project_target_status") == "MET", "readiness project target")
    require(readiness_audit.get("dev_site_test_opened") is False, "readiness site-test state")
    return comparison, lock, propagation


def fusion_canary(propagation: dict) -> dict:
    with np.load(PROPAGATION_CACHE, allow_pickle=False) as archive:
        required = {"k3_features", "k3_columns", "node_site_index", "node_partition_code"}
        require(required.issubset(set(archive.files)), "K3 propagation-cache members")
        graph_features = np.asarray(archive["k3_features"][:16], dtype=np.float32)
        columns = np.asarray(archive["k3_columns"])
    require(graph_features.shape == (16, EXPECTED_GRAPH_FEATURES), "K3 graph feature shape")
    require(len(columns) == EXPECTED_GRAPH_FEATURES, "K3 graph column count")
    require(np.isfinite(graph_features).all(), "finite K3 graph features")

    sample = (
        (np.arange(16 * EXPECTED_SAMPLE_FEATURES, dtype=np.float32).reshape(16, -1) % 97)
        / np.float32(97.0)
    )

    def execute() -> np.ndarray:
        return np.concatenate((sample, graph_features), axis=1).astype(np.float32, copy=False)

    first = execute()
    second = execute()
    require(first.shape == (16, EXPECTED_COMBINED_FEATURES), "fusion width")
    require(np.array_equal(first, second), "fusion replay")
    require(np.isfinite(first).all(), "finite fusion canary")
    return {
        "status": "PASS",
        "replay_exact": True,
        "records": 16,
        "sample_features": EXPECTED_SAMPLE_FEATURES,
        "graph_features": EXPECTED_GRAPH_FEATURES,
        "combined_features": EXPECTED_COMBINED_FEATURES,
        "propagation_cache_sha256": propagation["sha256"],
        "sha256": hashlib.sha256(first.astype("<f4", copy=False).tobytes(order="C")).hexdigest(),
    }


def environment_document(canary: dict) -> dict:
    estimator = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=1e-5,
        batch_size=8192,
        learning_rate_init=3e-4,
        max_iter=1,
        shuffle=False,
        random_state=RANDOM_SEED,
        early_stopping=False,
    )
    require(estimator.hidden_layer_sizes == HIDDEN_LAYERS, "MLP architecture support")
    require(estimator.solver == "adam", "MLP Adam support")
    return {
        "stage": "11D-2A",
        "status": "PASS",
        "environment_status": "FROZEN",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "threadpoolctl": package_version("threadpoolctl"),
        },
        "required_external_gnn_framework": "NONE",
        "pytorch_required": False,
        "torch_geometric_required": False,
        "estimator_backend": "sklearn.neural_network.MLPClassifier",
        "thread_limits": {
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        "fusion_canary": canary,
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-2A script in project root")
    verify_outputs_absent()

    print("STAGE 11D-2A — HYBRID MODEL ARCHITECTURE AND TRAINING-CONTRACT")
    print("FROZEN INPUT VERIFICATION", flush=True)
    evidence = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    comparison, _, propagation = verify_semantics()
    evidence[relative(PROPAGATION_CACHE)] = record(PROPAGATION_CACHE)
    print("  Frozen GNN disposition and hybrid authorization                         : PASS")
    print("  Frozen graph and sample-feature leakage contracts                       : PASS")
    print("  DEV_SITE_TEST matrix content opened                                     : NO")

    print("\nHYBRID FUSION CANARY", flush=True)
    canary = fusion_canary(propagation)
    print("  Fusion mode                  : FEATURE-LEVEL CONCATENATION")
    print("  Sample features              :", EXPECTED_SAMPLE_FEATURES)
    print("  Graph features               :", EXPECTED_GRAPH_FEATURES)
    print("  Combined features            :", EXPECTED_COMBINED_FEATURES)
    print("  Deterministic replay         : PASS")
    print("  Canary SHA                   :", canary["sha256"])

    require(len(CANDIDATES) == 2, "candidate count")
    require(len({row["candidate_id"] for row in CANDIDATES}) == 2, "candidate IDs")
    require(TRAINABLE_PARAMETERS == 93_185, "trainable parameter calculation")
    atomic_csv(CANDIDATE_GRID, CANDIDATES)
    candidate_grid_record = record(CANDIDATE_GRID)
    created_at = datetime.now(timezone.utc).isoformat()

    architecture = {
        "stage": "11D-2A",
        "title": "HYBRID MODEL ARCHITECTURE FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "architecture_version": VERSION,
        "created_at_utc": created_at,
        "model_family": MODEL_FAMILY,
        "is_hybrid_model": True,
        "is_deep_learning_model": True,
        "uses_graph_topology": True,
        "fusion_level": "FEATURE_LEVEL",
        "design_rationale": (
            "Fuse the winning frozen three-hop directed-SGC topology representation "
            "with the leakage-safe sample features, then learn nonlinear interactions "
            "with the already-qualified CPU MLP backend. Feature fusion avoids in-sample "
            "score-stacking leakage and preserves the three frozen comparators."
        ),
        "sample_branch": {
            "source": "FROZEN_11C5E_LEAKAGE_SAFE_FEATURE_MATRIX",
            "features": EXPECTED_SAMPLE_FEATURES,
            "post_simulation_features": 0,
            "target_present": False,
        },
        "graph_branch": {
            "source": "FROZEN_11D1C_DIR_SGC_K3_PROPAGATION_CACHE",
            "selected_gnn": EXPECTED_GNN_CANDIDATE,
            "propagation_hops": 3,
            "direction_channels": ["SELF", "INBOUND", "OUTBOUND"],
            "features": EXPECTED_GRAPH_FEATURES,
            "graph_scaler": "FROZEN_DEV_TRAIN_NODE_SCALER_FROM_11D1C",
            "labels_used_during_propagation": False,
        },
        "fusion": {
            "operation": "CONCATENATE_SAMPLE_AND_GRAPH_FEATURES",
            "input_features": EXPECTED_COMBINED_FEATURES,
            "hidden_layers": list(HIDDEN_LAYERS),
            "activation": "RELU",
            "output": "BINARY_DETECTABILITY_PROBABILITY",
            "trainable_parameters": TRAINABLE_PARAMETERS,
        },
        "frozen_comparator_scores_used_as_inputs": False,
        "dev_site_test_predictions_used_as_inputs": False,
        "candidate_grid": candidate_grid_record,
        "fusion_canary": canary,
        "target_present_in_input": False,
        "post_simulation_features": 0,
    }
    atomic_json(ARCHITECTURE, architecture)
    architecture_record = record(ARCHITECTURE)

    training_contract = {
        "stage": "11D-2A",
        "title": "HYBRID MODEL TRAINING CONTRACT FREEZE",
        "status": "PASS",
        "training_contract_status": "FROZEN",
        "created_at_utc": created_at,
        "model_family": MODEL_FAMILY,
        "primary_task": "PRE_SIMULATION_BINARY_DETECTABILITY",
        "target": "DETECTED",
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY",
        "sample_features": EXPECTED_SAMPLE_FEATURES,
        "graph_features": EXPECTED_GRAPH_FEATURES,
        "combined_features": EXPECTED_COMBINED_FEATURES,
        "physical_graph_nodes": EXPECTED_NODES,
        "fault_instances": EXPECTED_FAULT_INSTANCES,
        "partitions": {
            "DEV_TRAIN": {
                "sites": EXPECTED_TRAIN_SITES,
                "samples": EXPECTED_TRAIN_SAMPLES,
                "use": "FIT_HYBRID_PARAMETERS",
            },
            "DEV_CALIBRATION": {
                "sites": EXPECTED_CALIBRATION_SITES,
                "samples": EXPECTED_CALIBRATION_SAMPLES,
                "use": "SELECT_CANDIDATE_AND_THRESHOLD",
            },
            "DEV_SITE_TEST": {
                "sites": EXPECTED_SITE_TEST_SITES,
                "samples": EXPECTED_SITE_TEST_SAMPLES,
                "use": "LOCKED_UNTIL_HYBRID_SELECTION_FREEZE",
            },
        },
        "grouping_unit": "PHYSICAL_SITE",
        "sa0_sa1_grouped": True,
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "epochs_per_candidate": 4,
        "training_batch_size": 8192,
        "random_seed": RANDOM_SEED,
        "shuffle": "DETERMINISTIC_FIXED_RECORD_ORDER",
        "class_weighting": "GLOBAL_DEV_TRAIN_BALANCED_RESAMPLING",
        "graph_feature_scaler": "REUSE_FROZEN_11D1C_DEV_TRAIN_NODE_SCALER",
        "sample_feature_scaler": "REUSE_FROZEN_11C5E_DEV_TRAIN_ONLY_SCALER",
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "candidate_tie_break": [
            "HIGHER_MCC",
            "HIGHER_BALANCED_ACCURACY",
            "HIGHER_RECALL",
            "LEXICAL_CANDIDATE_ID",
        ],
        "threshold_selection": "DETERMINISTIC_MCC_MAXIMIZATION_ON_DEV_CALIBRATION",
        "threshold_tie_break": ["HIGHER_MCC", "HIGHER_RECALL", "CLOSEST_TO_0.5", "LOWER_THRESHOLD"],
        "minimum_validity": [
            "MCC_GREATER_THAN_ZERO",
            "BALANCED_ACCURACY_GREATER_THAN_HALF",
            "BOTH_PREDICTED_CLASSES_PRESENT",
            "FINITE_PROBABILITIES",
            "NO_UNKNOWN_LABELS",
            "DETERMINISTIC_REPLAY_EXACT",
        ],
        "hybrid_advancement_target": {
            "comparison": "HYBRID_MINUS_FROZEN_GNN",
            "primary_metric": "MCC",
            "required_point_improvement": 0.02,
            "statistical_rule": "PAIRED_PHYSICAL_SITE_95_PERCENT_CI_ENTIRELY_ABOVE_ZERO",
            "failure_disposition": "RETAIN_FROZEN_GNN_AS_PROJECT_WINNER",
        },
        "absolute_performance_target": {
            "mcc_minimum": 0.40,
            "balanced_accuracy_minimum": 0.70,
            "f1_minimum": 0.70,
            "recall_minimum": 0.70,
        },
        "locked_comparators": [
            "DIR_SGC_11D1D",
            "DEEP_MLP_11C5K",
            "CONVENTIONAL_LOGREG_11C5H",
        ],
        "estimated_cpu_training_time": "45-180 MINUTES",
        "maximum_resident_memory_policy": "LESS_THAN_8_GIB",
        "checkpoint_after_each_candidate": True,
        "resume_supported": True,
        "internet_required": False,
        "gpu_required": False,
        "dev_site_test_opening_authorized": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
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
            "dev_site_test_comparator_predictions",
            "in_sample_frozen_model_scores",
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
        "propagation_cache": record(PROPAGATION_CACHE),
    })
    atomic_json(ENVIRONMENT, environment)
    environment_record = record(ENVIRONMENT)

    audit = {
        "stage": "11D-2A",
        "title": "HYBRID MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "environment_status": "FROZEN",
        "candidate_grid_status": "FROZEN",
        "model_family": MODEL_FAMILY,
        "hybrid_model": True,
        "deep_learning_model": True,
        "graph_aware": True,
        "fusion_mode": "FEATURE_LEVEL_CONCATENATION",
        "sample_features": EXPECTED_SAMPLE_FEATURES,
        "graph_features": EXPECTED_GRAPH_FEATURES,
        "combined_features": EXPECTED_COMBINED_FEATURES,
        "architecture": f"{EXPECTED_COMBINED_FEATURES} -> 128 -> 64 -> 32 -> 1",
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_candidates": len(CANDIDATES),
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "epochs_per_candidate": 4,
        "training_batch_size": 8192,
        "dev_train_samples": EXPECTED_TRAIN_SAMPLES,
        "dev_calibration_samples": EXPECTED_CALIBRATION_SAMPLES,
        "dev_site_test_samples": EXPECTED_SITE_TEST_SAMPLES,
        "selection_partition": "DEV_CALIBRATION",
        "primary_metric": "MCC",
        "fusion_canary": "PASS",
        "deterministic_replay": "PASS",
        "target_present_in_input": False,
        "post_simulation_features": 0,
        "dev_site_test_state": "LOCKED",
        "dev_site_test_opened": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "hybrid_training_authorized_for_dev_train": True,
        "hybrid_site_test_evaluation_authorized": False,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "internet_required": False,
        "gpu_required": False,
        "frozen_gnn_mcc": comparison["metrics"]["gnn"]["mcc"],
        "hybrid_required_mcc_improvement_over_gnn": 0.02,
        "failure_disposition": "RETAIN_FROZEN_GNN_AS_PROJECT_WINNER",
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
        "architecture_artifact": architecture_record,
        "training_contract": training_contract_record,
        "candidate_grid": candidate_grid_record,
        "environment": environment_record,
        "next_gate": "STAGE 11D-2B — HYBRID MODEL TRAINING AND CALIBRATION EXECUTION",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-2A — HYBRID MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE")
    print("Status                         : PASS")
    print("Architecture status            : FROZEN")
    print("Training contract status       : FROZEN")
    print("Environment status             : FROZEN")
    print("Candidate grid status          : FROZEN")
    print("Model family                   : GRAPH-AUGMENTED FEATURE-FUSION MLP")
    print("Hybrid model                   : YES")
    print("Deep learning model            : YES")
    print("Graph aware                    : YES")
    print("Fusion mode                    : FEATURE-LEVEL CONCATENATION")
    print("Sample features                :", EXPECTED_SAMPLE_FEATURES)
    print("Graph features                 :", EXPECTED_GRAPH_FEATURES)
    print("Combined features              :", EXPECTED_COMBINED_FEATURES)
    print("Architecture                   : 646 -> 128 -> 64 -> 32 -> 1")
    print("Trainable parameters           :", TRAINABLE_PARAMETERS)
    print("Trainable candidates           :", len(CANDIDATES))
    print("Epochs per candidate           : 4")
    print("Training batch size            : 8192")
    print("Candidate execution            : SEQUENTIAL")
    print("Parallel candidates            : 1")
    print("DEV_TRAIN samples              :", EXPECTED_TRAIN_SAMPLES)
    print("DEV_CALIBRATION samples        :", EXPECTED_CALIBRATION_SAMPLES)
    print("DEV_SITE_TEST                  : LOCKED")
    print("Selection partition            : DEV_CALIBRATION")
    print("Primary metric                 : MCC")
    print("Frozen GNN comparator MCC      :", f"{EXPECTED_GNN_MCC:.8f}")
    print("Required hybrid MCC gain       : +0.02 OVER GNN")
    print("Fusion canary                  : PASS")
    print("Deterministic replay           : PASS")
    print("Estimated CPU training time    : 45-180 MINUTES")
    print("Internet required              : NO")
    print("GPU required                   : NO")
    print("Target present in input        : NO")
    print("Post-simulation features       : 0")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Hybrid training                : AUTHORIZED FOR DEV_TRAIN")
    print("Hybrid site-test evaluation    : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Graph dataset modified         : NO")
    print("Frozen comparator models changed: NO")
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
    print("Next gate                      : STAGE 11D-2B — HYBRID MODEL TRAINING AND CALIBRATION EXECUTION")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
