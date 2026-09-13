#!/usr/bin/env python3
"""Evaluate the frozen directed SGC once and compare all frozen diagnostics."""
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
    from sklearn.linear_model import SGDClassifier
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


VERSION = "HMAC-LOCKED-GNN-SITE-TEST-EVALUATOR-v1"
RANDOM_SEED = 20_260_907
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_CONFIDENCE = 0.95
INFERENCE_BATCH_SIZE = 32_768

EXPECTED_ROWS = 438_528
EXPECTED_SITES = 3_426
EXPECTED_POSITIVES = 191_945
EXPECTED_NEGATIVES = 246_583
EXPECTED_SAMPLES_PER_SITE = 128
EXPECTED_SAMPLE_FEATURES = 527
EXPECTED_GRAPH_FEATURES = 119
EXPECTED_COMBINED_FEATURES = 646
EXPECTED_GRAPH_NODES = 22_839
EXPECTED_GRAPH_EDGES = 47_617
EXPECTED_GNN_CANDIDATE = "DIR_SGC_K3_L2_A1E5"
EXPECTED_GNN_HOPS = 3
EXPECTED_GNN_THRESHOLD = 0.86327695216798483
EXPECTED_BASELINE_CANDIDATE = "SGD_LOGREG_L2_A1E6_BAL"
EXPECTED_BASELINE_MCC = 0.14208149
EXPECTED_DEEP_CANDIDATE = "DEEP_MLP_128_64_32_A1E5_LR3E4"
EXPECTED_DEEP_MCC = 0.14832380
REQUIRED_MCC_IMPROVEMENT = 0.05
GRAPH_COMMITMENT = "2020aa2959c0919df336da93087d0cad5086711cc9f9bcb185673d2a93db614f"

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_11D1 = ROOT / "results/hmac_fault_campaign_11d1"
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
FEATURE_ROOT = RESULT_11C5 / "feature_matrix_11c5e"
TRAIN_ROOT = RESULT_11D1 / "gnn_training_11d1c"
EVAL_ROOT = RESULT_11D1 / "gnn_evaluation_11d1d"
BASELINE_ROOT = RESULT_11C5 / "baseline_evaluation_11c5h"
DEEP_ROOT = RESULT_11C5 / "deep_evaluation_11c5k"

SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"

GRAPH_NPZ = RESULT_11D1 / "graph_dataset_11d1a/hmac_golden_netlist_graph_11d1a.npz"
GRAPH_MANIFEST = RESULT_11D1 / "hmac_golden_netlist_graph_manifest_11d1a.json"
GRAPH_AUDIT = RESULT_11D1 / "hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"

GNN_CONTRACT_SOURCE = ROOT / "stage_11d1b_gnn_contract.py"
GNN_ARCHITECTURE = ROOT / "config/diagnostic_model/hmac_gnn_architecture_11d1b.json"
GNN_CONTRACT = ROOT / "config/diagnostic_model/hmac_gnn_training_contract_11d1b.json"
GNN_CONTRACT_AUDIT = RESULT_11D1 / "hmac_gnn_architecture_training_contract_freeze_11d1b.json"
GNN_TRAINER = ROOT / "stage_11d1c_gnn_train.py"
GNN_MODEL = TRAIN_ROOT / "hmac_selected_gnn_model_11d1c.joblib"
GNN_SELECTION_LOCK = TRAIN_ROOT / "hmac_gnn_selection_lock_11d1c.json"
GNN_PROPAGATION_CACHE = TRAIN_ROOT / "hmac_directed_sgc_graph_features_11d1c.npz"
GNN_TRAINING_MANIFEST = RESULT_11D1 / "hmac_gnn_training_manifest_11d1c.json"
GNN_TRAINING_AUDIT = RESULT_11D1 / "hmac_gnn_training_calibration_freeze_11d1c.json"

BASELINE_EVALUATOR = ROOT / "stage_11c5h_locked_site_test.py"
BASELINE_PREDICTIONS = BASELINE_ROOT / "hmac_conventional_baseline_site_test_predictions_11c5h.npz"
BASELINE_LOCK = BASELINE_ROOT / "hmac_conventional_baseline_site_test_evaluation_lock_11c5h.json"
BASELINE_MANIFEST = RESULT_11C5 / "hmac_conventional_baseline_site_test_manifest_11c5h.json"
BASELINE_AUDIT = RESULT_11C5 / "hmac_conventional_baseline_site_test_freeze_11c5h.json"

DEEP_EVALUATOR = ROOT / "stage_11c5k_locked_deep_site_test.py"
DEEP_PREDICTIONS = DEEP_ROOT / "hmac_deep_diagnostic_site_test_predictions_11c5k.npz"
DEEP_COMPARISON = DEEP_ROOT / "hmac_deep_vs_conventional_baseline_comparison_11c5k.json"
DEEP_LOCK = DEEP_ROOT / "hmac_deep_diagnostic_site_test_evaluation_lock_11c5k.json"
DEEP_MANIFEST = RESULT_11C5 / "hmac_deep_diagnostic_site_test_manifest_11c5k.json"
DEEP_AUDIT = RESULT_11C5 / "hmac_deep_diagnostic_site_test_comparison_freeze_11c5k.json"

PREDICTIONS = EVAL_ROOT / "hmac_gnn_site_test_predictions_11d1d.npz"
METRICS = EVAL_ROOT / "hmac_gnn_site_test_metrics_11d1d.json"
BREAKDOWN_CSV = EVAL_ROOT / "hmac_gnn_site_test_breakdowns_11d1d.csv"
BREAKDOWN_JSON = EVAL_ROOT / "hmac_gnn_site_test_breakdowns_11d1d.json"
BOOTSTRAP_CSV = EVAL_ROOT / "hmac_gnn_comparator_paired_site_bootstrap_11d1d.csv"
BOOTSTRAP_JSON = EVAL_ROOT / "hmac_gnn_comparator_paired_site_bootstrap_11d1d.json"
BOOTSTRAP_DISTRIBUTION = EVAL_ROOT / "hmac_gnn_comparator_paired_site_bootstrap_distribution_11d1d.npz"
RELIABILITY_CURVE = EVAL_ROOT / "hmac_gnn_site_test_reliability_curve_11d1d.csv"
COMPARISON_CSV = EVAL_ROOT / "hmac_gnn_vs_frozen_comparators_11d1d.csv"
COMPARISON_JSON = EVAL_ROOT / "hmac_gnn_vs_frozen_comparators_11d1d.json"
EVALUATION_LOCK = EVAL_ROOT / "hmac_gnn_site_test_evaluation_lock_11d1d.json"
MANIFEST = RESULT_11D1 / "hmac_gnn_site_test_comparator_manifest_11d1d.json"
AUDIT = RESULT_11D1 / "hmac_gnn_site_test_comparator_freeze_11d1d.json"

EXPECTED_INPUTS = {
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    GRAPH_NPZ: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    GRAPH_MANIFEST: "b719e33941460ce9c0186c29c18f4585adf044ae3afd0493ad40214f42b9828c",
    GRAPH_AUDIT: "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
    GNN_CONTRACT_SOURCE: "498f59600c014b025ffa4fb58c73ffb95a47e0607332b6f0e15d7dda4304877b",
    GNN_ARCHITECTURE: "120033cb61ab8af5c5895aa71adf44395c9aeae6055cba56c1c6f5269796cdba",
    GNN_CONTRACT: "3662b5f579c1a910c4336ec1173e592391419fa17cc05813a54575bb9d5350e1",
    GNN_CONTRACT_AUDIT: "b5a5b8b6b8d7f6f6926a17440ff6f7859e422e9182c948695afa47bf0f36ef01",
    GNN_TRAINER: "cbd19604847c47cbba5327598fd7108e4702b00ef49500dcf7a25c647a315f6c",
    GNN_MODEL: "c345ab00a45bceb7b7c375d9c5483daa067a01bb8149b30c2325b056a3724e91",
    GNN_SELECTION_LOCK: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
    GNN_TRAINING_MANIFEST: "c40b118f9142e619218107397b324c0e14753d295952e45e32dd875534763fdf",
    GNN_TRAINING_AUDIT: "afe2c204fe2208f3a5805bf9e7bf7c7000bdef1a81dac8d651876d2cb633c079",
    BASELINE_EVALUATOR: "8f720d2f5a2eff1d964507424af502fba9aca60abf09d50fb897d33075d290e3",
    BASELINE_PREDICTIONS: "ec6d1b20e024b9739d1e79d7ef19bf0a5cfcb3292b16ae355c70f4e014248f76",
    BASELINE_LOCK: "5ff4987169c2b885826db13cf721164b65e009a83290d9aa21da9eeb13b213c4",
    BASELINE_MANIFEST: "cfe75842e7b081230c29d5162ec0292381e574b8679f58118c9a26bf6a7cc1ef",
    BASELINE_AUDIT: "54d86a12e605ce8f2b920c55e87c6ad9a3e30b41df8420730adfff4be4fbee19",
    DEEP_EVALUATOR: "51f23c9e5b27af8463aeb2b902e7c9ef90d553a47e9b81fdde6c81943cd86d7c",
    DEEP_COMPARISON: "47094ad6b03ca97d1aa3726e8b0e6a017667c532ea24f4aa49a3e54e9c3b9865",
    DEEP_LOCK: "678be80a8e9077136fd4e66b4c6863b263f96883294cc700d61b215bc3209250",
    DEEP_MANIFEST: "57ca6ec6fec634b7fe09f07631648353c29ed8d4b84559fe029f1ee7f418080f",
    DEEP_AUDIT: "f5df0c081c75a66178a362ccbfb6cfdeb612634e6d2292891307a2826db8aab8",
}

COMPARISON_METRICS = (
    "mcc", "balanced_accuracy", "precision", "recall", "specificity",
    "f1_score", "pr_auc", "roc_auc", "brier_score",
)
BOOTSTRAP_METRICS = (
    "mcc", "balanced_accuracy", "precision", "recall", "specificity",
    "f1_score", "brier_score",
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
        stop(f"could not read JSON {relative(path)}: {error}")


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


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {relative(path)}")
    require(path.stat().st_size > 0, f"empty frozen input: {relative(path)}")
    actual = sha256(path)
    require(actual == expected_sha, f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual}")
    print(f"  {path.name:<78}: OK", flush=True)
    return record(path)


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


def verify_outputs_absent() -> None:
    for path in (
        PREDICTIONS, METRICS, BREAKDOWN_CSV, BREAKDOWN_JSON, BOOTSTRAP_CSV,
        BOOTSTRAP_JSON, BOOTSTRAP_DISTRIBUTION, RELIABILITY_CURVE,
        COMPARISON_CSV, COMPARISON_JSON, EVALUATION_LOCK, MANIFEST, AUDIT,
    ):
        require(not path.exists(), f"Stage 11D-1D output already exists: {relative(path)}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        PREDICTIONS, METRICS, BREAKDOWN_CSV, BREAKDOWN_JSON, BOOTSTRAP_CSV,
        BOOTSTRAP_JSON, BOOTSTRAP_DISTRIBUTION, RELIABILITY_CURVE,
        COMPARISON_CSV, COMPARISON_JSON, EVALUATION_LOCK, MANIFEST,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        stop(f"required package missing: {name}")


def verify_environment() -> dict:
    frozen = load_json(RESULT_11D1 / "hmac_gnn_environment_11d1b.json")
    expected = frozen.get("packages", {})
    actual = {
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
        "threadpoolctl": package_version("threadpoolctl"),
    }
    require(actual == expected, f"Python package environment changed: expected {expected}, actual {actual}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"thread contract {name}")
    return {"status": "PASS", "packages": actual, "threads": 1, "inference_device": "CPU"}


def verify_handoff() -> tuple[dict, dict, dict, dict, dict]:
    contract = load_json(GNN_CONTRACT)
    lock = load_json(GNN_SELECTION_LOCK)
    training_manifest = load_json(GNN_TRAINING_MANIFEST)
    training_audit = load_json(GNN_TRAINING_AUDIT)
    graph_manifest = load_json(GRAPH_MANIFEST)
    graph_audit = load_json(GRAPH_AUDIT)
    baseline_manifest = load_json(BASELINE_MANIFEST)
    baseline_audit = load_json(BASELINE_AUDIT)
    deep_manifest = load_json(DEEP_MANIFEST)
    deep_audit = load_json(DEEP_AUDIT)

    require(contract.get("status") == "PASS", "GNN contract status")
    require(contract.get("training_contract_status") == "FROZEN", "GNN contract freeze")
    require(contract.get("selection_partition") == "DEV_CALIBRATION", "GNN selection partition")
    require(contract.get("dev_site_test_opening_authorized") is False, "GNN training site-test lock")
    require(contract.get("validation_vectors_exposed") == 0, "GNN contract VALIDATION exposure")
    require(contract.get("holdout_vectors_exposed") == 0, "GNN contract HOLDOUT exposure")
    require(contract.get("hybrid_training_authorized") is False, "GNN contract hybrid gate")

    require(lock.get("status") == "PASS" and lock.get("lock_status") == "FROZEN", "GNN selection lock")
    require(lock.get("selected_candidate_id") == EXPECTED_GNN_CANDIDATE, "GNN candidate")
    require(int(lock.get("selected_propagation_hops")) == EXPECTED_GNN_HOPS, "GNN hops")
    require(float(lock.get("selected_threshold")) == EXPECTED_GNN_THRESHOLD, "GNN threshold")
    require(lock.get("selection_partition") == "DEV_CALIBRATION", "GNN lock partition")
    require(lock.get("minimum_validity_status") == "PASS", "GNN calibration validity")
    require(lock.get("dev_site_test_opened_for_evaluation") is False, "GNN site-test exposure")
    require(lock.get("validation_vectors_exposed") == 0, "GNN lock VALIDATION exposure")
    require(lock.get("holdout_vectors_exposed") == 0, "GNN lock HOLDOUT exposure")
    require(lock.get("hybrid_training_authorized") is False, "GNN lock hybrid gate")
    replay = lock.get("deterministic_replay", {})
    require(replay.get("status") == "PASS", "GNN replay status")
    require(replay.get("weights_exact") is True, "GNN replay weights")
    require(replay.get("probabilities_exact") is True, "GNN replay probabilities")
    require(replay.get("predictions_exact") is True, "GNN replay predictions")
    require(lock.get("selected_model", {}).get("sha256") == EXPECTED_INPUTS[GNN_MODEL], "GNN model lock SHA")

    require(training_manifest.get("status") == "PASS", "GNN training manifest status")
    require(training_manifest.get("selected_candidate_id") == EXPECTED_GNN_CANDIDATE, "GNN manifest candidate")
    require(float(training_manifest.get("selected_threshold")) == EXPECTED_GNN_THRESHOLD, "GNN manifest threshold")
    require(training_manifest.get("minimum_validity_status") == "PASS", "GNN manifest validity")
    require(training_manifest.get("dev_site_test_opened_for_evaluation") is False, "GNN manifest site-test exposure")
    require(training_audit.get("status") == "PASS", "GNN training audit status")
    require(training_audit.get("training_status") == "FROZEN", "GNN training freeze")
    require(training_audit.get("model_selection_status") == "FROZEN", "GNN selection freeze")
    require(training_audit.get("threshold_status") == "FROZEN", "GNN threshold freeze")
    require(training_audit.get("selected_candidate_id") == EXPECTED_GNN_CANDIDATE, "GNN audit candidate")
    require(training_audit.get("dev_site_test_state") == "LOCKED", "GNN site-test state")
    require(training_audit.get("next_gate", "").startswith("STAGE 11D-1D"), "GNN evaluation gate")

    propagation_record = lock.get("propagation_cache", {})
    require(propagation_record == training_manifest.get("propagation", {}).get("artifact"), "propagation record agreement")
    resolved_propagation = resolve_artifact(propagation_record, "GNN propagation cache")
    require(resolved_propagation["path"] == relative(GNN_PROPAGATION_CACHE), "propagation cache path")

    require(graph_manifest.get("graph_commitment") == GRAPH_COMMITMENT, "graph commitment")
    require(graph_manifest.get("node_count") == EXPECTED_GRAPH_NODES, "graph node count")
    require(graph_manifest.get("collapsed_edge_count") == EXPECTED_GRAPH_EDGES, "graph edge count")
    require(graph_audit.get("topology_integrity_status") == "PASS", "graph topology integrity")

    require(baseline_audit.get("status") == "PASS", "baseline audit status")
    require(baseline_audit.get("baseline_status") == "FROZEN COMPARATOR", "baseline comparator status")
    require(abs(float(baseline_audit.get("mcc")) - EXPECTED_BASELINE_MCC) < 1e-8, "baseline MCC")
    require(baseline_audit.get("dev_site_test_state") == "CONSUMED AND FROZEN", "baseline site-test state")
    require(deep_audit.get("status") == "PASS", "deep audit status")
    require(deep_audit.get("deep_model_status") == "FROZEN COMPARATOR", "deep comparator status")
    require(abs(float(deep_audit.get("deep_mcc")) - EXPECTED_DEEP_MCC) < 1e-8, "deep MCC")
    require(deep_audit.get("dev_site_test_state") == "CONSUMED AND FROZEN FOR DEEP MODEL", "deep site-test state")

    baseline_outputs = {
        name: resolve_artifact(item, f"baseline output {name}")
        for name, item in sorted(baseline_manifest.get("outputs", {}).items())
    }
    deep_outputs = {
        name: resolve_artifact(item, f"deep output {name}")
        for name, item in sorted(deep_manifest.get("outputs", {}).items())
    }
    require(baseline_outputs.get("predictions", {}).get("path") == relative(BASELINE_PREDICTIONS), "baseline predictions record")
    require(deep_outputs.get("predictions", {}).get("path") == relative(DEEP_PREDICTIONS), "deep predictions record")
    return lock, training_audit, baseline_audit, deep_audit, {
        "propagation_cache": resolved_propagation,
        "baseline_outputs": baseline_outputs,
        "deep_outputs": deep_outputs,
    }


def load_model_and_graph(lock: dict) -> tuple[dict, np.ndarray]:
    bundle = joblib.load(GNN_MODEL)
    require(isinstance(bundle, dict), "GNN model bundle type")
    require(bundle.get("version") == "HMAC-DIRECTED-SGC-TRAINER-v1", "GNN bundle version")
    require(bundle.get("model_family") == "DIRECTED_BIDIRECTIONAL_SIMPLIFIED_GRAPH_CONVOLUTION", "GNN bundle family")
    require(bundle.get("graph_commitment") == GRAPH_COMMITMENT, "GNN bundle graph commitment")
    require(bundle.get("candidate", {}).get("candidate_id") == EXPECTED_GNN_CANDIDATE, "GNN bundle candidate")
    require(int(bundle.get("candidate", {}).get("propagation_hops")) == EXPECTED_GNN_HOPS, "GNN bundle hops")
    require(int(bundle.get("candidate", {}).get("graph_features")) == EXPECTED_GRAPH_FEATURES, "GNN graph width")
    require(int(bundle.get("candidate", {}).get("combined_features")) == EXPECTED_COMBINED_FEATURES, "GNN combined width")
    require(bundle.get("target_in_input") is False, "GNN target leakage")
    require(bundle.get("post_simulation_features") == 0, "GNN post-simulation features")
    model = bundle.get("estimator")
    require(isinstance(model, SGDClassifier), "GNN sample-head estimator type")
    require(np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])), "GNN classes")
    require(int(model.n_features_in_) == EXPECTED_COMBINED_FEATURES, "GNN estimator width")
    require(np.asarray(model.coef_).shape == (1, EXPECTED_COMBINED_FEATURES), "GNN coefficient shape")
    require(lock.get("selected_model", {}).get("sha256") == sha256(GNN_MODEL), "GNN model SHA")

    with np.load(GNN_PROPAGATION_CACHE, allow_pickle=False) as archive:
        required = {"node_site_index", "node_partition_code", "k3_features", "k3_mean", "k3_scale", "k3_columns"}
        require(required <= set(archive.files), "GNN propagation-cache members")
        graph_features = np.asarray(archive["k3_features"], dtype=np.float32)
        node_site_index = np.asarray(archive["node_site_index"], dtype=np.uint32)
        mean = np.asarray(archive["k3_mean"], dtype=np.float64)
        scale = np.asarray(archive["k3_scale"], dtype=np.float64)
        columns = np.asarray(archive["k3_columns"])
    require(graph_features.shape == (EXPECTED_GRAPH_NODES, EXPECTED_GRAPH_FEATURES), "cached graph-feature shape")
    require(node_site_index.shape == (EXPECTED_GRAPH_NODES,), "cached node-index shape")
    require(np.array_equal(node_site_index, np.arange(1, EXPECTED_GRAPH_NODES + 1)), "cached node indexing")
    require(np.isfinite(graph_features).all(), "finite cached graph features")
    require(np.array_equal(mean, bundle["graph_scaler_mean"]), "frozen graph-scaler mean")
    require(np.array_equal(scale, bundle["graph_scaler_scale"]), "frozen graph-scaler scale")
    require(columns.tolist() == bundle["graph_feature_columns"], "frozen graph-feature columns")
    return bundle, graph_features


def combined_feature_batch(site_test: dict[str, np.ndarray], graph_features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    matrix = np.empty((len(indices), EXPECTED_COMBINED_FEATURES), dtype=np.float32)
    site_rows = site_test["site_row"][indices]
    graph_rows = site_test["site_indices"][site_rows].astype(np.int64) - 1
    require(int(graph_rows.min()) >= 0 and int(graph_rows.max()) < EXPECTED_GRAPH_NODES, "graph lookup bounds")
    matrix[:, 0] = site_test["stuck_value"][indices]
    matrix[:, 1:15] = site_test["site_features"][site_rows]
    matrix[:, 15:527] = site_test["vector_features"][site_test["vector_row"][indices]]
    matrix[:, 527:] = graph_features[graph_rows]
    require(np.isfinite(matrix).all(), "finite combined features")
    return matrix


def predict_once(bundle: dict, graph_features: np.ndarray, site_test: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    model = bundle["estimator"]
    probability = np.empty(EXPECTED_ROWS, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    batches = math.ceil(EXPECTED_ROWS / INFERENCE_BATCH_SIZE)
    completed = 0
    for start in range(0, EXPECTED_ROWS, INFERENCE_BATCH_SIZE):
        stop_index = min(start + INFERENCE_BATCH_SIZE, EXPECTED_ROWS)
        indices = np.arange(start, stop_index, dtype=np.int64)
        matrix = combined_feature_batch(site_test, graph_features, indices)
        probability[start:stop_index] = model.predict_proba(matrix)[:, positive_column]
        completed += 1
        print(f"  inference batch {completed}/{batches}", flush=True)
    require(np.isfinite(probability).all(), "finite GNN probabilities")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), "GNN probability range")
    return probability, completed


def load_frozen_predictions(path: Path, site_test: dict[str, np.ndarray], label: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"record_id", "site_row", "vector_row", "stuck_value", "target_detected", "probability", "prediction", "threshold"}
        require(required <= set(archive.files), f"{label} prediction members")
        result = {name: np.asarray(archive[name]) for name in required}
    for name in ("record_id", "site_row", "vector_row", "stuck_value", "target_detected", "probability", "prediction"):
        require(result[name].shape == (EXPECTED_ROWS,), f"{label} {name} shape")
    require(result["threshold"].shape == (1,), f"{label} threshold shape")
    require(np.array_equal(result["record_id"], site_test["record_id"]), f"{label} record alignment")
    require(np.array_equal(result["site_row"], site_test["site_row"]), f"{label} site alignment")
    require(np.array_equal(result["vector_row"], site_test["vector_row"]), f"{label} vector alignment")
    require(np.array_equal(result["stuck_value"], site_test["stuck_value"]), f"{label} fault alignment")
    require(np.array_equal(result["target_detected"], site_test["target"]), f"{label} target alignment")
    require(np.isfinite(result["probability"]).all(), f"{label} finite probabilities")
    require(np.isin(result["prediction"], (0, 1)).all(), f"{label} binary predictions")
    return result


def replay_comparator_metrics(site_test: dict, predictions: dict, audit: dict, prefix: str) -> dict:
    metrics = baseline_helpers.calculate_metrics(
        site_test["target"], predictions["probability"].astype(np.float64),
        predictions["prediction"].astype(np.uint8), float(predictions["threshold"][0]),
    )
    if prefix == "deep":
        keys = {
            "mcc": "deep_mcc", "balanced_accuracy": "deep_balanced_accuracy",
            "precision": "deep_precision", "recall": "deep_recall",
            "specificity": "deep_specificity", "f1_score": "deep_f1",
            "pr_auc": "deep_pr_auc", "roc_auc": "deep_roc_auc",
            "brier_score": "deep_brier",
        }
    else:
        keys = {
            "mcc": "mcc", "balanced_accuracy": "balanced_accuracy",
            "precision": "precision", "recall": "recall",
            "specificity": "specificity", "f1_score": "f1_score",
            "pr_auc": "pr_auc", "roc_auc": "roc_auc",
            "brier_score": "brier_score",
        }
    for metric, audit_key in keys.items():
        require(abs(float(metrics[metric]) - float(audit[audit_key])) < 1e-8, f"{prefix} metric replay {metric}")
    return metrics


def site_components(site_row: np.ndarray, target: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confusion = np.stack([
        np.bincount(site_row, weights=(target == 0) & (prediction == 0), minlength=EXPECTED_SITES),
        np.bincount(site_row, weights=(target == 0) & (prediction == 1), minlength=EXPECTED_SITES),
        np.bincount(site_row, weights=(target == 1) & (prediction == 0), minlength=EXPECTED_SITES),
        np.bincount(site_row, weights=(target == 1) & (prediction == 1), minlength=EXPECTED_SITES),
    ], axis=1).astype(np.float64, copy=False)
    squared = np.bincount(
        site_row, weights=np.square(probability - target, dtype=np.float64),
        minlength=EXPECTED_SITES,
    ).astype(np.float64, copy=False)
    require(np.all(confusion.sum(axis=1) == EXPECTED_SAMPLES_PER_SITE), "site confusion closure")
    return confusion, squared


def paired_bootstrap(
    site_test: dict,
    model_data: dict[str, dict],
    point_metrics: dict[str, dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    components = {
        name: site_components(site_test["site_row"], site_test["target"], data["probability"], data["prediction"])
        for name, data in model_data.items()
    }
    comparisons = (("gnn", "baseline"), ("gnn", "deep"), ("deep", "baseline"))
    arrays: dict[str, np.ndarray] = {}
    for metric in BOOTSTRAP_METRICS:
        for model in ("gnn", "baseline", "deep"):
            arrays[f"{model}_{metric}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for left, right in comparisons:
            arrays[f"delta_{left}_minus_{right}_{metric}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, EXPECTED_SITES, size=EXPECTED_SITES)
        multiplicity = np.bincount(sampled, minlength=EXPECTED_SITES).astype(np.float64)
        replicate_metrics = {}
        for model in ("gnn", "baseline", "deep"):
            confusion, squared = components[model]
            hard = baseline_helpers.hard_metrics_from_counts(*(multiplicity @ confusion))
            replicate_metrics[model] = {
                **hard,
                "brier_score": float(multiplicity @ squared) / EXPECTED_ROWS,
            }
            for metric in BOOTSTRAP_METRICS:
                arrays[f"{model}_{metric}"][replicate] = replicate_metrics[model][metric]
        for left, right in comparisons:
            for metric in BOOTSTRAP_METRICS:
                arrays[f"delta_{left}_minus_{right}_{metric}"][replicate] = (
                    replicate_metrics[left][metric] - replicate_metrics[right][metric]
                )
        if (replicate + 1) % 100 == 0:
            print(f"  paired bootstrap replicate {replicate + 1}/{BOOTSTRAP_REPLICATES}", flush=True)

    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    intervals = []
    for metric in BOOTSTRAP_METRICS:
        for model in ("gnn", "baseline", "deep"):
            values = arrays[f"{model}_{metric}"]
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
            intervals.append({
                "metric": metric, "estimate": model,
                "point_estimate": float(point_metrics[model][metric]),
                "ci_lower": float(lower), "ci_upper": float(upper),
                "confidence": BOOTSTRAP_CONFIDENCE, "replicates": BOOTSTRAP_REPLICATES,
            })
        for left, right in comparisons:
            key = f"delta_{left}_minus_{right}_{metric}"
            lower, upper = np.quantile(arrays[key], [alpha, 1.0 - alpha], method="linear")
            intervals.append({
                "metric": metric, "estimate": f"{left}_minus_{right}",
                "point_estimate": float(point_metrics[left][metric] - point_metrics[right][metric]),
                "ci_lower": float(lower), "ci_upper": float(upper),
                "confidence": BOOTSTRAP_CONFIDENCE, "replicates": BOOTSTRAP_REPLICATES,
            })
    return ({
        "stage": "11D-1D", "status": "PASS",
        "method": "PAIRED PHYSICAL-SITE-GROUP NONPARAMETRIC PERCENTILE BOOTSTRAP",
        "sampling_unit": "physical fault site", "sites_per_replicate": EXPECTED_SITES,
        "records_per_site": EXPECTED_SAMPLES_PER_SITE, "random_seed": RANDOM_SEED,
        "confidence": BOOTSTRAP_CONFIDENCE, "replicates": BOOTSTRAP_REPLICATES,
        "intervals": intervals,
    }, arrays)


def interval(bootstrap: dict, metric: str, estimate: str) -> dict:
    return next(
        item for item in bootstrap["intervals"]
        if item["metric"] == metric and item["estimate"] == estimate
    )


def comparison_documents(metrics: dict[str, dict], bootstrap: dict) -> tuple[dict, list[dict]]:
    rows = []
    for metric in COMPARISON_METRICS:
        rows.append({
            "metric": metric,
            "gnn_value": metrics["gnn"][metric],
            "deep_value": metrics["deep"][metric],
            "baseline_value": metrics["baseline"][metric],
            "gnn_minus_deep": metrics["gnn"][metric] - metrics["deep"][metric],
            "gnn_minus_baseline": metrics["gnn"][metric] - metrics["baseline"][metric],
            "preferred_direction": "LOWER" if metric == "brier_score" else "HIGHER",
        })
    gnn_baseline = interval(bootstrap, "mcc", "gnn_minus_baseline")
    gnn_deep = interval(bootstrap, "mcc", "gnn_minus_deep")
    delta_baseline = metrics["gnn"]["mcc"] - metrics["baseline"]["mcc"]
    delta_deep = metrics["gnn"]["mcc"] - metrics["deep"]["mcc"]
    point_winner = max(("gnn", "deep", "baseline"), key=lambda name: metrics[name]["mcc"])
    document = {
        "stage": "11D-1D", "status": "PASS", "partition": "DEV_SITE_TEST",
        "comparison_status": "FROZEN", "primary_metric": "MCC",
        "models": {
            "gnn": EXPECTED_GNN_CANDIDATE,
            "deep": EXPECTED_DEEP_CANDIDATE,
            "baseline": EXPECTED_BASELINE_CANDIDATE,
        },
        "metrics": metrics,
        "point_winner": point_winner.upper(),
        "gnn_minus_baseline_mcc": delta_baseline,
        "gnn_minus_baseline_mcc_ci_95": [gnn_baseline["ci_lower"], gnn_baseline["ci_upper"]],
        "gnn_minus_deep_mcc": delta_deep,
        "gnn_minus_deep_mcc_ci_95": [gnn_deep["ci_lower"], gnn_deep["ci_upper"]],
        "gnn_improvement_over_baseline_statistically_supported": gnn_baseline["ci_lower"] > 0.0,
        "gnn_improvement_over_deep_statistically_supported": gnn_deep["ci_lower"] > 0.0,
        "required_mcc_improvement": REQUIRED_MCC_IMPROVEMENT,
        "required_mcc_improvement_over_baseline_met": delta_baseline >= REQUIRED_MCC_IMPROVEMENT,
        "project_target_met": (
            delta_baseline >= REQUIRED_MCC_IMPROVEMENT and gnn_baseline["ci_lower"] > 0.0
        ),
        "decision_rule": (
            "The frozen GNN project target requires at least +0.05 MCC over the "
            "conventional baseline and a paired physical-site 95% MCC-delta "
            "confidence interval entirely above zero."
        ),
    }
    return document, rows


def write_breakdowns(records: list[dict], unrepresented: list[dict]) -> None:
    atomic_csv(
        BREAKDOWN_CSV, baseline_helpers.BREAKDOWN_HEADER,
        [baseline_helpers.breakdown_csv_row(item) for item in records],
    )
    atomic_json(BREAKDOWN_JSON, {
        "stage": "11D-1D", "status": "PASS", "partition": "DEV_SITE_TEST",
        "threshold_source": "FROZEN STAGE 11D-1C SELECTION LOCK",
        "threshold": EXPECTED_GNN_THRESHOLD,
        "represented_group_count": len(records),
        "unrepresented_group_count": len(unrepresented),
        "unrepresented_groups": unrepresented, "groups": records,
    })


def write_bootstrap(summary: dict, arrays: dict[str, np.ndarray]) -> None:
    atomic_json(BOOTSTRAP_JSON, summary)
    atomic_csv(
        BOOTSTRAP_CSV,
        ["metric", "estimate", "point_estimate", "ci_lower", "ci_upper", "confidence", "replicates"],
        summary["intervals"],
    )
    payload = {name: values.astype("<f8", copy=False) for name, values in arrays.items()}
    payload.update({
        "confidence": np.asarray([BOOTSTRAP_CONFIDENCE], dtype="<f8"),
        "random_seed": np.asarray([RANDOM_SEED], dtype="<u8"),
        "replicates": np.asarray([BOOTSTRAP_REPLICATES], dtype="<u4"),
        "site_count": np.asarray([EXPECTED_SITES], dtype="<u4"),
    })
    deterministic_npz(BOOTSTRAP_DISTRIBUTION, payload)


def write_reliability_curve(target: np.ndarray, probability: np.ndarray) -> None:
    order = np.argsort(probability, kind="stable")
    rows = []
    for index, indices in enumerate(np.array_split(order, 20)):
        values = probability[indices]
        labels = target[indices]
        rows.append({
            "bin_index": index, "sample_count": len(indices),
            "minimum_probability": float(values[0]),
            "maximum_probability": float(values[-1]),
            "mean_predicted_probability": float(values.mean(dtype=np.float64)),
            "observed_positive_fraction": float(labels.mean(dtype=np.float64)),
        })
    atomic_csv(RELIABILITY_CURVE, [
        "bin_index", "sample_count", "minimum_probability", "maximum_probability",
        "mean_predicted_probability", "observed_positive_fraction",
    ], rows)


def assessment(metrics: dict) -> tuple[str, str, dict]:
    minimum = {
        "mcc_gt_zero": metrics["mcc"] > 0.0,
        "balanced_accuracy_gt_half": metrics["balanced_accuracy"] > 0.5,
        "pr_auc_gt_prevalence": metrics["pr_auc"] > metrics["positive_prevalence"],
        "both_predicted_classes_present": (
            metrics["predicted_positive"] > 0 and metrics["predicted_negative"] > 0
        ),
    }
    absolute = {
        "mcc_ge_0_40": metrics["mcc"] >= 0.40,
        "balanced_accuracy_ge_0_70": metrics["balanced_accuracy"] >= 0.70,
        "f1_ge_0_70": metrics["f1_score"] >= 0.70,
        "recall_ge_0_70": metrics["recall"] >= 0.70,
    }
    return (
        "PASS" if all(minimum.values()) else "NOT_MET",
        "PASS" if all(absolute.values()) else "NOT_MET",
        {"minimum_validity_checks": minimum, "absolute_performance_checks": absolute},
    )


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-1D evaluator in project root")
    verify_outputs_absent()

    print("STAGE 11D-1D — LOCKED GNN DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS")
    print("FROZEN INPUT VERIFICATION", flush=True)
    input_evidence = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    lock, training_audit, baseline_audit, deep_audit, handoff = verify_handoff()
    environment = verify_environment()

    print("\nOPENING LOCKED DEV_SITE_TEST")
    print("  Model fitting calls          : 0")
    print("  Graph-scaler fitting calls   : 0")
    print("  Candidate-selection calls    : 0")
    print("  Threshold-selection calls    : 0")
    print("  Frozen GNN candidate         :", EXPECTED_GNN_CANDIDATE)
    print("  Frozen GNN threshold         :", format(EXPECTED_GNN_THRESHOLD, ".17g"))

    site_test = baseline_helpers.load_site_test()
    require(len(site_test["target"]) == EXPECTED_ROWS, "site-test rows")
    require(int(site_test["target"].sum(dtype=np.uint64)) == EXPECTED_POSITIVES, "site-test positives")
    require(int((site_test["target"] == 0).sum()) == EXPECTED_NEGATIVES, "site-test negatives")
    bundle, graph_features = load_model_and_graph(lock)
    baseline_predictions = load_frozen_predictions(BASELINE_PREDICTIONS, site_test, "baseline")
    deep_predictions = load_frozen_predictions(DEEP_PREDICTIONS, site_test, "deep")

    print("\nLOCKED GNN INFERENCE", flush=True)
    gnn_probability, inference_batches = predict_once(bundle, graph_features, site_test)
    gnn_prediction = (gnn_probability >= EXPECTED_GNN_THRESHOLD).astype(np.uint8)
    gnn_metrics = baseline_helpers.calculate_metrics(
        site_test["target"], gnn_probability, gnn_prediction, EXPECTED_GNN_THRESHOLD,
    )
    baseline_metrics = replay_comparator_metrics(site_test, baseline_predictions, baseline_audit, "baseline")
    deep_metrics = replay_comparator_metrics(site_test, deep_predictions, deep_audit, "deep")
    minimum_status, absolute_status, acceptance = assessment(gnn_metrics)

    print("\nREQUIRED GNN BREAKDOWNS")
    breakdowns, unrepresented = baseline_helpers.make_breakdowns(
        site_test, gnn_probability, gnn_prediction, EXPECTED_GNN_THRESHOLD,
    )
    for item in breakdowns:
        print(
            f"  {item['group_dimension']:<18} {item['group_value']:<28} "
            f"samples={item['sample_count']:7d} MCC={item['mcc']:.6f}"
        )
    for item in unrepresented:
        print(f"  {item['group_dimension']:<18} {item['group_value']:<28} NOT REPRESENTED")

    print("\nPAIRED PHYSICAL-SITE COMPARATOR BOOTSTRAP", flush=True)
    model_data = {
        "gnn": {"probability": gnn_probability, "prediction": gnn_prediction},
        "baseline": {
            "probability": baseline_predictions["probability"].astype(np.float64),
            "prediction": baseline_predictions["prediction"].astype(np.uint8),
        },
        "deep": {
            "probability": deep_predictions["probability"].astype(np.float64),
            "prediction": deep_predictions["prediction"].astype(np.uint8),
        },
    }
    all_metrics = {"gnn": gnn_metrics, "baseline": baseline_metrics, "deep": deep_metrics}
    bootstrap, bootstrap_arrays = paired_bootstrap(site_test, model_data, all_metrics)
    comparison, comparison_rows = comparison_documents(all_metrics, bootstrap)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    deterministic_npz(PREDICTIONS, {
        "candidate_id": np.asarray([EXPECTED_GNN_CANDIDATE]),
        "model_sha256": np.asarray([EXPECTED_INPUTS[GNN_MODEL]]),
        "partition": np.asarray(["DEV_SITE_TEST"]),
        "prediction": gnn_prediction,
        "probability": gnn_probability.astype("<f8", copy=False),
        "record_id": site_test["record_id"].astype("<u4", copy=False),
        "site_ids": site_test["site_ids"],
        "site_indices": site_test["site_indices"].astype("<u2", copy=False),
        "site_row": site_test["site_row"].astype("<u2", copy=False),
        "stuck_value": site_test["stuck_value"].astype(np.uint8, copy=False),
        "target_detected": site_test["target"].astype(np.uint8, copy=False),
        "threshold": np.asarray([EXPECTED_GNN_THRESHOLD], dtype="<f8"),
        "vector_ids": site_test["vector_ids"].astype("<u2", copy=False),
        "vector_row": site_test["vector_row"].astype(np.uint8, copy=False),
    })
    atomic_json(METRICS, {
        "stage": "11D-1D", "status": "PASS", "partition": "DEV_SITE_TEST",
        "candidate": EXPECTED_GNN_CANDIDATE, "threshold": EXPECTED_GNN_THRESHOLD,
        "metrics": gnn_metrics, "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status, "acceptance_checks": acceptance,
    })
    write_breakdowns(breakdowns, unrepresented)
    write_bootstrap(bootstrap, bootstrap_arrays)
    write_reliability_curve(site_test["target"], gnn_probability)
    atomic_csv(COMPARISON_CSV, [
        "metric", "gnn_value", "deep_value", "baseline_value",
        "gnn_minus_deep", "gnn_minus_baseline", "preferred_direction",
    ], comparison_rows)
    atomic_json(COMPARISON_JSON, comparison)

    outputs = {
        "predictions": record(PREDICTIONS), "metrics": record(METRICS),
        "breakdown_csv": record(BREAKDOWN_CSV), "breakdown_json": record(BREAKDOWN_JSON),
        "paired_bootstrap_csv": record(BOOTSTRAP_CSV),
        "paired_bootstrap_json": record(BOOTSTRAP_JSON),
        "paired_bootstrap_distribution": record(BOOTSTRAP_DISTRIBUTION),
        "reliability_curve": record(RELIABILITY_CURVE),
        "comparison_csv": record(COMPARISON_CSV), "comparison_json": record(COMPARISON_JSON),
    }
    created_at = datetime.now(timezone.utc).isoformat()
    atomic_json(EVALUATION_LOCK, {
        "stage": "11D-1D", "title": "LOCKED GNN DEV_SITE_TEST EVALUATION LOCK",
        "status": "PASS", "lock_status": "FROZEN", "created_at_utc": created_at,
        "candidate_id": EXPECTED_GNN_CANDIDATE,
        "model_sha256": EXPECTED_INPUTS[GNN_MODEL],
        "threshold": EXPECTED_GNN_THRESHOLD, "partition": "DEV_SITE_TEST",
        "inference_batches": inference_batches, "gnn_metrics": gnn_metrics,
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "comparison": comparison,
        "model_fit_calls": 0, "graph_scaler_fit_calls": 0,
        "candidate_selection_calls": 0, "threshold_selection_calls": 0,
        "model_retrained": False, "threshold_changed": False,
        "graph_scaler_changed": False, "architecture_changed": False,
        "outputs": outputs, "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0, "hybrid_training_authorized": False,
    })
    evaluation_lock_record = record(EVALUATION_LOCK)

    manifest = {
        "stage": "11D-1D",
        "title": "LOCKED GNN DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS",
        "status": "PASS", "evaluator_version": VERSION,
        "evaluator_source": record(SOURCE), "created_at_utc": created_at,
        "partition": "DEV_SITE_TEST", "physical_sites": EXPECTED_SITES,
        "fault_instances": EXPECTED_SITES * 2, "samples": EXPECTED_ROWS,
        "positive_samples": EXPECTED_POSITIVES, "negative_samples": EXPECTED_NEGATIVES,
        "gnn_candidate": EXPECTED_GNN_CANDIDATE,
        "gnn_threshold": EXPECTED_GNN_THRESHOLD,
        "gnn_metrics": gnn_metrics, "deep_metrics": deep_metrics,
        "baseline_metrics": baseline_metrics, "comparison": comparison,
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "input_evidence": input_evidence, "handoff_artifacts": handoff,
        "environment": environment, "outputs": outputs,
        "evaluation_lock": evaluation_lock_record,
        "model_fit_calls": 0, "graph_scaler_fit_calls": 0,
        "candidate_selection_calls": 0, "threshold_selection_calls": 0,
        "validation_vectors_exposed": 0, "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False, "golden_netlist_modified": False,
        "graph_dataset_modified": False, "canonical_dataset_modified": False,
        "feature_matrices_modified": False, "gnn_model_modified": False,
        "deep_model_modified": False, "baseline_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = record(MANIFEST)

    gnn_interval = interval(bootstrap, "mcc", "gnn")
    delta_baseline_interval = interval(bootstrap, "mcc", "gnn_minus_baseline")
    delta_deep_interval = interval(bootstrap, "mcc", "gnn_minus_deep")
    project_status = "MET" if comparison["project_target_met"] else "NOT_MET"
    hybrid_readiness = (
        "AUTHORIZED_FOR_CONTRACT_FREEZE"
        if comparison["gnn_improvement_over_baseline_statistically_supported"]
        else "REQUIRES_DISPOSITION_REVIEW"
    )
    atomic_json(AUDIT, {
        "stage": "11D-1D",
        "title": "LOCKED GNN DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS FREEZE",
        "status": "PASS", "evaluation_status": "FROZEN",
        "comparison_status": "FROZEN", "gnn_status": "FROZEN COMPARATOR",
        "deep_status": "FROZEN COMPARATOR", "baseline_status": "FROZEN COMPARATOR",
        "gnn_candidate": EXPECTED_GNN_CANDIDATE,
        "gnn_threshold": EXPECTED_GNN_THRESHOLD,
        "gnn_mcc": gnn_metrics["mcc"],
        "gnn_mcc_ci_95": [gnn_interval["ci_lower"], gnn_interval["ci_upper"]],
        "gnn_balanced_accuracy": gnn_metrics["balanced_accuracy"],
        "gnn_precision": gnn_metrics["precision"], "gnn_recall": gnn_metrics["recall"],
        "gnn_specificity": gnn_metrics["specificity"], "gnn_f1": gnn_metrics["f1_score"],
        "gnn_pr_auc": gnn_metrics["pr_auc"], "gnn_roc_auc": gnn_metrics["roc_auc"],
        "gnn_brier": gnn_metrics["brier_score"],
        "deep_mcc": deep_metrics["mcc"], "baseline_mcc": baseline_metrics["mcc"],
        "gnn_minus_baseline_mcc": comparison["gnn_minus_baseline_mcc"],
        "gnn_minus_baseline_mcc_ci_95": [delta_baseline_interval["ci_lower"], delta_baseline_interval["ci_upper"]],
        "gnn_minus_deep_mcc": comparison["gnn_minus_deep_mcc"],
        "gnn_minus_deep_mcc_ci_95": [delta_deep_interval["ci_lower"], delta_deep_interval["ci_upper"]],
        "point_winner": comparison["point_winner"],
        "gnn_improvement_over_baseline_statistically_supported": comparison["gnn_improvement_over_baseline_statistically_supported"],
        "gnn_improvement_over_deep_statistically_supported": comparison["gnn_improvement_over_deep_statistically_supported"],
        "required_mcc_improvement_met": comparison["required_mcc_improvement_over_baseline_met"],
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "project_target_status": project_status,
        "required_breakdowns": "PASS", "paired_site_bootstrap": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "model_retrained": False, "threshold_changed": False,
        "graph_scaler_changed": False, "architecture_changed": False,
        "dev_site_test_state": "CONSUMED AND FROZEN FOR GNN",
        "validation_vectors_exposed": 0, "holdout_vectors_exposed": 0,
        "hybrid_readiness": hybrid_readiness,
        "hybrid_training_authorized": False,
        "evaluation_lock": evaluation_lock_record, "manifest": manifest_record,
        "frozen_rtl_modified": False, "golden_netlist_modified": False,
        "graph_dataset_modified": False, "canonical_dataset_modified": False,
        "feature_matrices_modified": False, "gnn_model_modified": False,
        "deep_model_modified": False, "baseline_model_modified": False,
        "next_gate": "STAGE 11D-1E — GNN DISPOSITION AND HYBRID READINESS FREEZE",
    })
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-1D — LOCKED GNN DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS FREEZE")
    print("Status                         : PASS")
    print("Evaluation status              : FROZEN")
    print("Comparison status              : FROZEN")
    print("GNN candidate                  :", EXPECTED_GNN_CANDIDATE)
    print("GNN threshold                  :", format(EXPECTED_GNN_THRESHOLD, ".17g"))
    print("DEV_SITE_TEST sites            :", EXPECTED_SITES)
    print("DEV_SITE_TEST samples          :", EXPECTED_ROWS)
    print("GNN MCC                        :", f"{gnn_metrics['mcc']:.8f}")
    print("GNN MCC 95% site CI            :", f"[{gnn_interval['ci_lower']:.8f}, {gnn_interval['ci_upper']:.8f}]")
    print("GNN balanced accuracy          :", f"{gnn_metrics['balanced_accuracy']:.8f}")
    print("GNN precision                  :", f"{gnn_metrics['precision']:.8f}")
    print("GNN recall                     :", f"{gnn_metrics['recall']:.8f}")
    print("GNN specificity                :", f"{gnn_metrics['specificity']:.8f}")
    print("GNN F1                         :", f"{gnn_metrics['f1_score']:.8f}")
    print("GNN PR-AUC                     :", f"{gnn_metrics['pr_auc']:.8f}")
    print("GNN ROC-AUC                    :", f"{gnn_metrics['roc_auc']:.8f}")
    print("GNN Brier                      :", f"{gnn_metrics['brier_score']:.8f}")
    print("Deep MLP MCC                   :", f"{deep_metrics['mcc']:.8f}")
    print("Conventional baseline MCC      :", f"{baseline_metrics['mcc']:.8f}")
    print("MCC delta (GNN-baseline)       :", f"{comparison['gnn_minus_baseline_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{delta_baseline_interval['ci_lower']:.8f}, {delta_baseline_interval['ci_upper']:.8f}]")
    print("MCC delta (GNN-deep)           :", f"{comparison['gnn_minus_deep_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{delta_deep_interval['ci_lower']:.8f}, {delta_deep_interval['ci_upper']:.8f}]")
    print("Point winner                   :", comparison["point_winner"])
    print("Statistical gain vs baseline   :", "SUPPORTED" if comparison["gnn_improvement_over_baseline_statistically_supported"] else "NOT_SUPPORTED")
    print("Statistical gain vs deep       :", "SUPPORTED" if comparison["gnn_improvement_over_deep_statistically_supported"] else "NOT_SUPPORTED")
    print("Required +0.05 MCC gain        :", "MET" if comparison["required_mcc_improvement_over_baseline_met"] else "NOT_MET")
    print("Minimum validity               :", minimum_status)
    print("Absolute performance target    :", absolute_status)
    print("Project target                 :", project_status)
    print("Model fitting calls            : 0")
    print("Graph-scaler fitting calls     : 0")
    print("Threshold changed              : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("GNN model modified             : NO")
    print("Comparison                     :", COMPARISON_JSON)
    print("Comparison SHA                 :", outputs["comparison_json"]["sha256"])
    print("Evaluation lock                :", EVALUATION_LOCK)
    print("Evaluation lock SHA            :", evaluation_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-1E — GNN DISPOSITION AND HYBRID READINESS FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
