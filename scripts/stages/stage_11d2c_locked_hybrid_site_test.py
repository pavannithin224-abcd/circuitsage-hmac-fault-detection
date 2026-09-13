#!/usr/bin/env python3
"""Evaluate the frozen hybrid once and compare all frozen diagnostic models."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import zipfile
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
    from sklearn.neural_network import MLPClassifier
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy, joblib, and scikit-learn are required. "
        "Activate the frozen project .venv before running."
    ) from error

try:
    import stage_11d1d_locked_gnn_site_test as comparator_helpers
    import stage_11d2b_hybrid_train as hybrid_helpers
except ImportError as error:
    raise SystemExit(
        "STOP: frozen Stage 11D-1D and repaired Stage 11D-2B helpers "
        "must be present in the project root."
    ) from error


VERSION = "HMAC-LOCKED-HYBRID-SITE-TEST-EVALUATOR-v1"
RANDOM_SEED = 20_260_911
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_CONFIDENCE = 0.95

EXPECTED_ROWS = 438_528
EXPECTED_SITES = 3_426
EXPECTED_POSITIVES = 191_945
EXPECTED_NEGATIVES = 246_583
EXPECTED_SAMPLES_PER_SITE = 128
EXPECTED_SAMPLE_FEATURES = 527
EXPECTED_GRAPH_FEATURES = 119
EXPECTED_COMBINED_FEATURES = 646
EXPECTED_PARAMETERS = 93_185
EXPECTED_HYBRID_CANDIDATE = "HYBRID_SGC3_MLP_128_64_32_A1E4_LR1E3"
EXPECTED_HYBRID_THRESHOLD = 0.4965
EXPECTED_GNN_CANDIDATE = "DIR_SGC_K3_L2_A1E5"
EXPECTED_GNN_MCC = 0.33662641
EXPECTED_DEEP_MCC = 0.14832380
EXPECTED_BASELINE_MCC = 0.14208149
REQUIRED_HYBRID_GAIN = 0.02

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
RESULT_11D1 = ROOT / "results/hmac_fault_campaign_11d1"
RESULT_11D2 = ROOT / "results/hmac_fault_campaign_11d2"
FEATURE_ROOT = RESULT_11C5 / "feature_matrix_11c5e"
TRAIN_ROOT = RESULT_11D2 / "hybrid_training_11d2b"
EVAL_ROOT = RESULT_11D2 / "hybrid_evaluation_11d2c"

SITE_TEST_MATRIX = FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"

HYBRID_CONTRACT_SOURCE = ROOT / "stage_11d2a_hybrid_contract.py"
HYBRID_ARCHITECTURE = ROOT / "config/diagnostic_model/hmac_hybrid_architecture_11d2a.json"
HYBRID_CONTRACT = ROOT / "config/diagnostic_model/hmac_hybrid_training_contract_11d2a.json"
HYBRID_GRID = ROOT / "config/diagnostic_model/hmac_hybrid_candidate_grid_11d2a.csv"
HYBRID_ENVIRONMENT = RESULT_11D2 / "hmac_hybrid_environment_11d2a.json"
HYBRID_CONTRACT_AUDIT = RESULT_11D2 / "hmac_hybrid_architecture_training_contract_freeze_11d2a.json"

HYBRID_TRAINER = ROOT / "stage_11d2b_hybrid_train.py"
HYBRID_MODEL = TRAIN_ROOT / "hmac_selected_hybrid_model_11d2b.joblib"
HYBRID_SELECTION_LOCK = TRAIN_ROOT / "hmac_hybrid_selection_lock_11d2b.json"
HYBRID_TRAINING_MANIFEST = RESULT_11D2 / "hmac_hybrid_training_manifest_11d2b.json"
HYBRID_TRAINING_AUDIT = RESULT_11D2 / "hmac_hybrid_training_calibration_freeze_11d2b.json"

GNN_EVALUATOR = ROOT / "stage_11d1d_locked_gnn_site_test.py"
GNN_PREDICTIONS = RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_site_test_predictions_11d1d.npz"
GNN_COMPARISON = RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_vs_frozen_comparators_11d1d.json"
GNN_EVALUATION_LOCK = RESULT_11D1 / "gnn_evaluation_11d1d/hmac_gnn_site_test_evaluation_lock_11d1d.json"
GNN_EVALUATION_MANIFEST = RESULT_11D1 / "hmac_gnn_site_test_comparator_manifest_11d1d.json"
GNN_EVALUATION_AUDIT = RESULT_11D1 / "hmac_gnn_site_test_comparator_freeze_11d1d.json"

PREDICTIONS = EVAL_ROOT / "hmac_hybrid_site_test_predictions_11d2c.npz"
METRICS = EVAL_ROOT / "hmac_hybrid_site_test_metrics_11d2c.json"
BREAKDOWN_CSV = EVAL_ROOT / "hmac_hybrid_site_test_breakdowns_11d2c.csv"
BREAKDOWN_JSON = EVAL_ROOT / "hmac_hybrid_site_test_breakdowns_11d2c.json"
BOOTSTRAP_CSV = EVAL_ROOT / "hmac_hybrid_comparator_paired_site_bootstrap_11d2c.csv"
BOOTSTRAP_JSON = EVAL_ROOT / "hmac_hybrid_comparator_paired_site_bootstrap_11d2c.json"
BOOTSTRAP_DISTRIBUTION = EVAL_ROOT / "hmac_hybrid_comparator_paired_site_bootstrap_distribution_11d2c.npz"
RELIABILITY_CURVE = EVAL_ROOT / "hmac_hybrid_site_test_reliability_curve_11d2c.csv"
COMPARISON_CSV = EVAL_ROOT / "hmac_hybrid_vs_frozen_comparators_11d2c.csv"
COMPARISON_JSON = EVAL_ROOT / "hmac_hybrid_vs_frozen_comparators_11d2c.json"
EVALUATION_LOCK = EVAL_ROOT / "hmac_hybrid_site_test_evaluation_lock_11d2c.json"
MANIFEST = RESULT_11D2 / "hmac_hybrid_site_test_comparator_manifest_11d2c.json"
AUDIT = RESULT_11D2 / "hmac_hybrid_site_test_comparator_freeze_11d2c.json"

EXPECTED_INPUTS = {
    SITE_TEST_MATRIX: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    HYBRID_CONTRACT_SOURCE: "efee27d3a3ab38be66462a20b084d820eb28744c57b4d427b9b8239a0da1b748",
    HYBRID_ARCHITECTURE: "d213a89c53769c217ca9cca71700b03abc23f94d5302bc4309ea63b035016063",
    HYBRID_CONTRACT: "1ccd69a6ee179dbdbce9cc295324c85f28bc4fd98f22b5df010fdcbf21cda41e",
    HYBRID_GRID: "3aaea24ec28c4680ebd24ecef345ab82ab910c07a6f5240dc762f196c8032bbd",
    HYBRID_ENVIRONMENT: "1849274a8006e77cced8695be784e70102abceec940fee036163f0a62aa65a57",
    HYBRID_CONTRACT_AUDIT: "cbe3f362986a86506f9252253e06a2a8a2ec3541e9bc92655b75e05fcc082e65",
    HYBRID_TRAINER: "756366ce8715e54f12eb2ddbc6adae3065303a1875b18da0e8dbf245d001c91f",
    HYBRID_MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    HYBRID_SELECTION_LOCK: "005cecbd94456d3d1aa6805b2dc898e895cc178564673b6b0aa269fcd64260c6",
    HYBRID_TRAINING_MANIFEST: "c5792cae623c0098165e691446c590235f50e1a5adc753dd133247b19adba41a",
    HYBRID_TRAINING_AUDIT: "39f459b3dbb0270810af8b8d39819ad2efecf982ebd929b3683d44cfa2570722",
    GNN_EVALUATOR: "f5126b50d8f089e5fb1cea84de5e3375d7c648c0878dd62fbef29f0e7a5eb599",
    GNN_PREDICTIONS: "3849103f43dede6043c6b1bfc5265f7a427aec16917a95d0d156955668f830a6",
    GNN_COMPARISON: "5bfce5f1cb49b29dc3a17343dbd2ba5f5777e3ec11d9679efdd24b9bfbdaf0e1",
    GNN_EVALUATION_LOCK: "e09b0c093566e4f22a7f7e9ae92fd73325d4092b25baeca2357dd85618f78398",
    GNN_EVALUATION_MANIFEST: "a97fe3ca5c9a000f8e514ad88d263ea5a686df5dabdbf17c637676db8083789a",
    GNN_EVALUATION_AUDIT: "3555783dc613aec71a6f89c09b6c487b514aa0cfc68a6c7b42489018dbdb5a1b",
}

MODEL_NAMES = ("hybrid", "gnn", "deep", "baseline")
PAIR_NAMES = (
    ("hybrid", "gnn"),
    ("hybrid", "deep"),
    ("hybrid", "baseline"),
    ("gnn", "deep"),
    ("gnn", "baseline"),
    ("deep", "baseline"),
)
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


def close(actual: object, expected: float, tolerance: float = 5e-8) -> bool:
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
    require(
        actual == expected_sha,
        f"SHA mismatch for {relative(path)}: expected {expected_sha}, actual {actual}",
    )
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
        require(not path.exists(), f"Stage 11D-2C output already exists: {relative(path)}")


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


def verify_inference_only_source() -> None:
    tree = ast.parse(SOURCE.read_text())
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name in {"fit", "partial_fit", "select_threshold", "train_candidate"}:
            forbidden.append((node.lineno, name))
    require(not forbidden, f"training or threshold-selection calls present: {forbidden}")


def verify_hybrid_handoff() -> tuple[dict, dict, dict]:
    architecture, contract = hybrid_helpers.verify_contract()
    lock = load_json(HYBRID_SELECTION_LOCK)
    manifest = load_json(HYBRID_TRAINING_MANIFEST)
    audit = load_json(HYBRID_TRAINING_AUDIT)

    require(lock.get("status") == "PASS" and lock.get("lock_status") == "FROZEN", "hybrid selection lock")
    require(lock.get("selected_candidate_id") == EXPECTED_HYBRID_CANDIDATE, "hybrid candidate")
    require(close(lock.get("selected_threshold"), EXPECTED_HYBRID_THRESHOLD), "hybrid threshold")
    require(lock.get("selection_partition") == "DEV_CALIBRATION", "hybrid selection partition")
    require(lock.get("selection_metric") == "MCC", "hybrid selection metric")
    require(lock.get("minimum_validity_status") == "PASS", "hybrid calibration validity")
    require(lock.get("absolute_performance_status_on_calibration") == "PASS", "hybrid calibration target")
    require(lock.get("hybrid_advancement_status") == "PENDING_LOCKED_DEV_SITE_TEST_COMPARISON", "hybrid advancement gate")
    require(close(lock.get("frozen_gnn_site_test_mcc"), EXPECTED_GNN_MCC), "frozen GNN MCC")
    require(close(lock.get("required_hybrid_mcc_gain"), REQUIRED_HYBRID_GAIN), "hybrid gain requirement")
    require(lock.get("dev_site_test_opened_for_training_or_selection") is False, "hybrid site-test training leakage")
    require(lock.get("validation_vectors_exposed") == 0, "hybrid lock VALIDATION exposure")
    require(lock.get("holdout_vectors_exposed") == 0, "hybrid lock HOLDOUT exposure")
    require(lock.get("retraining_after_site_test_authorized") is False, "hybrid retraining gate")
    replay = lock.get("deterministic_replay", {})
    for key in ("weights_exact", "probabilities_exact", "threshold_exact", "metrics_exact"):
        require(replay.get(key) is True, f"hybrid replay {key}")
    require(lock.get("selected_model", {}).get("sha256") == EXPECTED_INPUTS[HYBRID_MODEL], "hybrid model lock SHA")

    require(manifest.get("status") == "PASS", "hybrid training manifest status")
    require(manifest.get("selected_candidate_id") == EXPECTED_HYBRID_CANDIDATE, "hybrid manifest candidate")
    require(close(manifest.get("selected_threshold"), EXPECTED_HYBRID_THRESHOLD), "hybrid manifest threshold")
    require(manifest.get("dev_site_test_opened") is False, "hybrid manifest site-test state")
    require(manifest.get("validation_vectors_exposed") == 0, "hybrid manifest VALIDATION exposure")
    require(manifest.get("holdout_vectors_exposed") == 0, "hybrid manifest HOLDOUT exposure")
    require(manifest.get("deterministic_replay") == "PASS", "hybrid manifest replay")

    require(audit.get("status") == "PASS", "hybrid training audit status")
    require(audit.get("training_status") == "FROZEN", "hybrid training freeze")
    require(audit.get("model_selection_status") == "FROZEN", "hybrid selection freeze")
    require(audit.get("threshold_status") == "FROZEN", "hybrid threshold freeze")
    require(audit.get("selected_candidate_id") == EXPECTED_HYBRID_CANDIDATE, "hybrid audit candidate")
    require(close(audit.get("selected_threshold"), EXPECTED_HYBRID_THRESHOLD), "hybrid audit threshold")
    require(audit.get("deterministic_replay") == "PASS", "hybrid audit replay")
    require(audit.get("weights_exact") is True and audit.get("probabilities_exact") is True, "hybrid exact replay")
    require(audit.get("dev_site_test_opened_during_training") is False, "hybrid audit site-test state")
    require(audit.get("next_gate", "").startswith("STAGE 11D-2C"), "hybrid evaluation gate")

    output_records = manifest.get("outputs", {})
    require(isinstance(output_records, dict), "hybrid manifest output records")
    resolved = {
        name: resolve_artifact(item, f"hybrid output {name}")
        for name, item in sorted(output_records.items())
    }
    require(resolved.get("selected_model", {}).get("path") == relative(HYBRID_MODEL), "hybrid model path")
    require(manifest.get("selection_lock") == record(HYBRID_SELECTION_LOCK), "hybrid manifest selection-lock record")
    require(audit.get("selected_model") == lock.get("selected_model"), "hybrid audit model record")
    require(audit.get("selection_lock") == record(HYBRID_SELECTION_LOCK), "hybrid audit selection-lock record")
    require(audit.get("manifest") == record(HYBRID_TRAINING_MANIFEST), "hybrid audit manifest record")
    propagation = resolve_artifact(lock.get("propagation_cache", {}), "hybrid propagation cache")
    return lock, audit, {
        "architecture_status": architecture.get("status"),
        "contract_status": contract.get("status"),
        "training_outputs": resolved,
        "propagation_cache": propagation,
    }


def load_hybrid_model(lock: dict) -> MLPClassifier:
    model = joblib.load(HYBRID_MODEL)
    require(isinstance(model, MLPClassifier), "hybrid estimator type")
    require(tuple(model.hidden_layer_sizes) == (128, 64, 32), "hybrid hidden layers")
    require(int(model.n_features_in_) == EXPECTED_COMBINED_FEATURES, "hybrid input width")
    require(np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])), "hybrid classes")
    parameters = sum(value.size for value in model.coefs_) + sum(
        value.size for value in model.intercepts_
    )
    require(parameters == EXPECTED_PARAMETERS, "hybrid parameter count")
    require(lock.get("selected_model", {}).get("sha256") == sha256(HYBRID_MODEL), "hybrid model SHA")
    return model


def replay_metric_record(site_test: dict, predictions: dict, audit: dict, prefix: str) -> dict:
    metrics = comparator_helpers.baseline_helpers.calculate_metrics(
        site_test["target"], predictions["probability"].astype(np.float64),
        predictions["prediction"].astype(np.uint8), float(predictions["threshold"][0]),
    )
    if prefix == "gnn":
        keys = {
            "mcc": "gnn_mcc", "balanced_accuracy": "gnn_balanced_accuracy",
            "precision": "gnn_precision", "recall": "gnn_recall",
            "specificity": "gnn_specificity", "f1_score": "gnn_f1",
            "pr_auc": "gnn_pr_auc", "roc_auc": "gnn_roc_auc",
            "brier_score": "gnn_brier",
        }
    elif prefix == "deep":
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
        require(close(metrics[metric], audit.get(audit_key), 1e-8), f"{prefix} metric replay {metric}")
    return metrics


def paired_bootstrap(site_test: dict, model_data: dict, point_metrics: dict) -> tuple[dict, dict]:
    components = {
        name: comparator_helpers.site_components(
            site_test["site_row"], site_test["target"],
            data["probability"], data["prediction"],
        )
        for name, data in model_data.items()
    }
    arrays = {}
    for metric in BOOTSTRAP_METRICS:
        for model in MODEL_NAMES:
            arrays[f"{model}_{metric}"] = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for left, right in PAIR_NAMES:
            arrays[f"delta_{left}_minus_{right}_{metric}"] = np.empty(
                BOOTSTRAP_REPLICATES, dtype=np.float64
            )

    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, EXPECTED_SITES, size=EXPECTED_SITES)
        multiplicity = np.bincount(sampled, minlength=EXPECTED_SITES).astype(np.float64)
        replicate_metrics = {}
        for model in MODEL_NAMES:
            confusion, squared = components[model]
            hard = comparator_helpers.baseline_helpers.hard_metrics_from_counts(
                *(multiplicity @ confusion)
            )
            replicate_metrics[model] = {
                **hard,
                "brier_score": float(multiplicity @ squared) / EXPECTED_ROWS,
            }
            for metric in BOOTSTRAP_METRICS:
                arrays[f"{model}_{metric}"][replicate] = replicate_metrics[model][metric]
        for left, right in PAIR_NAMES:
            for metric in BOOTSTRAP_METRICS:
                arrays[f"delta_{left}_minus_{right}_{metric}"][replicate] = (
                    replicate_metrics[left][metric] - replicate_metrics[right][metric]
                )
        if (replicate + 1) % 100 == 0:
            print(f"  paired bootstrap replicate {replicate + 1}/{BOOTSTRAP_REPLICATES}", flush=True)

    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    intervals = []
    for metric in BOOTSTRAP_METRICS:
        for model in MODEL_NAMES:
            values = arrays[f"{model}_{metric}"]
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
            intervals.append({
                "metric": metric, "estimate": model,
                "point_estimate": float(point_metrics[model][metric]),
                "ci_lower": float(lower), "ci_upper": float(upper),
                "confidence": BOOTSTRAP_CONFIDENCE,
                "replicates": BOOTSTRAP_REPLICATES,
            })
        for left, right in PAIR_NAMES:
            key = f"delta_{left}_minus_{right}_{metric}"
            lower, upper = np.quantile(arrays[key], [alpha, 1.0 - alpha], method="linear")
            intervals.append({
                "metric": metric, "estimate": f"{left}_minus_{right}",
                "point_estimate": float(point_metrics[left][metric] - point_metrics[right][metric]),
                "ci_lower": float(lower), "ci_upper": float(upper),
                "confidence": BOOTSTRAP_CONFIDENCE,
                "replicates": BOOTSTRAP_REPLICATES,
            })
    return ({
        "stage": "11D-2C", "status": "PASS",
        "method": "PAIRED PHYSICAL-SITE-GROUP NONPARAMETRIC PERCENTILE BOOTSTRAP",
        "sampling_unit": "physical fault site",
        "sites_per_replicate": EXPECTED_SITES,
        "records_per_site": EXPECTED_SAMPLES_PER_SITE,
        "random_seed": RANDOM_SEED,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "replicates": BOOTSTRAP_REPLICATES,
        "intervals": intervals,
    }, arrays)


def interval(bootstrap: dict, metric: str, estimate: str) -> dict:
    return next(
        item for item in bootstrap["intervals"]
        if item["metric"] == metric and item["estimate"] == estimate
    )


def comparison_documents(metrics: dict, bootstrap: dict) -> tuple[dict, list[dict]]:
    rows = []
    for metric in COMPARISON_METRICS:
        rows.append({
            "metric": metric,
            "hybrid_value": metrics["hybrid"][metric],
            "gnn_value": metrics["gnn"][metric],
            "deep_value": metrics["deep"][metric],
            "baseline_value": metrics["baseline"][metric],
            "hybrid_minus_gnn": metrics["hybrid"][metric] - metrics["gnn"][metric],
            "hybrid_minus_deep": metrics["hybrid"][metric] - metrics["deep"][metric],
            "hybrid_minus_baseline": metrics["hybrid"][metric] - metrics["baseline"][metric],
            "preferred_direction": "LOWER" if metric == "brier_score" else "HIGHER",
        })
    hybrid_gnn = interval(bootstrap, "mcc", "hybrid_minus_gnn")
    hybrid_deep = interval(bootstrap, "mcc", "hybrid_minus_deep")
    hybrid_baseline = interval(bootstrap, "mcc", "hybrid_minus_baseline")
    delta_gnn = metrics["hybrid"]["mcc"] - metrics["gnn"]["mcc"]
    delta_deep = metrics["hybrid"]["mcc"] - metrics["deep"]["mcc"]
    delta_baseline = metrics["hybrid"]["mcc"] - metrics["baseline"]["mcc"]
    point_winner = max(MODEL_NAMES, key=lambda name: metrics[name]["mcc"])
    target_met = delta_gnn >= REQUIRED_HYBRID_GAIN and hybrid_gnn["ci_lower"] > 0.0
    document = {
        "stage": "11D-2C", "status": "PASS", "partition": "DEV_SITE_TEST",
        "comparison_status": "FROZEN", "primary_metric": "MCC",
        "models": {
            "hybrid": EXPECTED_HYBRID_CANDIDATE,
            "gnn": EXPECTED_GNN_CANDIDATE,
            "deep": comparator_helpers.EXPECTED_DEEP_CANDIDATE,
            "baseline": comparator_helpers.EXPECTED_BASELINE_CANDIDATE,
        },
        "metrics": metrics,
        "point_winner": point_winner.upper(),
        "hybrid_minus_gnn_mcc": delta_gnn,
        "hybrid_minus_gnn_mcc_ci_95": [hybrid_gnn["ci_lower"], hybrid_gnn["ci_upper"]],
        "hybrid_minus_deep_mcc": delta_deep,
        "hybrid_minus_deep_mcc_ci_95": [hybrid_deep["ci_lower"], hybrid_deep["ci_upper"]],
        "hybrid_minus_baseline_mcc": delta_baseline,
        "hybrid_minus_baseline_mcc_ci_95": [hybrid_baseline["ci_lower"], hybrid_baseline["ci_upper"]],
        "hybrid_improvement_over_gnn_statistically_supported": hybrid_gnn["ci_lower"] > 0.0,
        "hybrid_improvement_over_deep_statistically_supported": hybrid_deep["ci_lower"] > 0.0,
        "hybrid_improvement_over_baseline_statistically_supported": hybrid_baseline["ci_lower"] > 0.0,
        "required_hybrid_mcc_improvement": REQUIRED_HYBRID_GAIN,
        "required_hybrid_mcc_improvement_over_gnn_met": delta_gnn >= REQUIRED_HYBRID_GAIN,
        "hybrid_advancement_target_met": target_met,
        "decision_rule": (
            "Hybrid advancement requires at least +0.02 MCC over the frozen GNN "
            "and a paired physical-site 95% MCC-delta confidence interval entirely above zero."
        ),
    }
    return document, rows


def write_breakdowns(records: list[dict], unrepresented: list[dict]) -> None:
    atomic_csv(
        BREAKDOWN_CSV, comparator_helpers.baseline_helpers.BREAKDOWN_HEADER,
        [comparator_helpers.baseline_helpers.breakdown_csv_row(item) for item in records],
    )
    atomic_json(BREAKDOWN_JSON, {
        "stage": "11D-2C", "status": "PASS", "partition": "DEV_SITE_TEST",
        "threshold_source": "FROZEN STAGE 11D-2B SELECTION LOCK",
        "threshold": EXPECTED_HYBRID_THRESHOLD,
        "represented_group_count": len(records),
        "unrepresented_group_count": len(unrepresented),
        "unrepresented_groups": unrepresented,
        "groups": records,
    })


def write_bootstrap(summary: dict, arrays: dict) -> None:
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
            "bin_index": index,
            "sample_count": len(indices),
            "minimum_probability": float(values[0]),
            "maximum_probability": float(values[-1]),
            "mean_predicted_probability": float(values.mean(dtype=np.float64)),
            "observed_positive_fraction": float(labels.mean(dtype=np.float64)),
        })
    atomic_csv(RELIABILITY_CURVE, [
        "bin_index", "sample_count", "minimum_probability", "maximum_probability",
        "mean_predicted_probability", "observed_positive_fraction",
    ], rows)


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-2C evaluator in project root")
    verify_outputs_absent()
    verify_inference_only_source()

    print("STAGE 11D-2C — LOCKED HYBRID DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS")
    print("FROZEN INPUT VERIFICATION", flush=True)
    input_evidence = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    hybrid_lock, hybrid_training_audit, hybrid_handoff = verify_hybrid_handoff()
    _, _, baseline_audit, deep_audit, comparator_handoff = comparator_helpers.verify_handoff()
    gnn_audit = load_json(GNN_EVALUATION_AUDIT)
    environment = hybrid_helpers.verify_environment()

    require(gnn_audit.get("status") == "PASS", "GNN comparator audit")
    require(close(gnn_audit.get("gnn_mcc"), EXPECTED_GNN_MCC), "GNN comparator MCC")
    require(gnn_audit.get("dev_site_test_state") == "CONSUMED AND FROZEN FOR GNN", "GNN site-test state")
    require(gnn_audit.get("hybrid_readiness") == "AUTHORIZED_FOR_CONTRACT_FREEZE", "GNN hybrid readiness")

    print("\nOPENING LOCKED DEV_SITE_TEST")
    print("  Model fitting calls          : 0")
    print("  Graph-scaler fitting calls   : 0")
    print("  Candidate-selection calls    : 0")
    print("  Threshold-selection calls    : 0")
    print("  Frozen hybrid candidate      :", EXPECTED_HYBRID_CANDIDATE)
    print("  Frozen hybrid threshold      :", format(EXPECTED_HYBRID_THRESHOLD, ".17g"))

    site_test = hybrid_helpers.load_partition(
        SITE_TEST_MATRIX, "DEV_SITE_TEST", EXPECTED_ROWS, EXPECTED_SITES, EXPECTED_POSITIVES
    )
    require(len(site_test["target"]) == EXPECTED_ROWS, "site-test rows")
    require(int(site_test["target"].sum(dtype=np.uint64)) == EXPECTED_POSITIVES, "site-test positives")
    require(int((site_test["target"] == 0).sum()) == EXPECTED_NEGATIVES, "site-test negatives")

    graph_features, graph_lookup, propagation_record = hybrid_helpers.load_graph_features()
    require(propagation_record == hybrid_handoff["propagation_cache"], "hybrid propagation agreement")
    model = load_hybrid_model(hybrid_lock)

    baseline_predictions = comparator_helpers.load_frozen_predictions(
        comparator_helpers.BASELINE_PREDICTIONS, site_test, "baseline"
    )
    deep_predictions = comparator_helpers.load_frozen_predictions(
        comparator_helpers.DEEP_PREDICTIONS, site_test, "deep"
    )
    gnn_predictions = comparator_helpers.load_frozen_predictions(
        GNN_PREDICTIONS, site_test, "gnn"
    )
    require(close(gnn_predictions["threshold"][0], comparator_helpers.EXPECTED_GNN_THRESHOLD), "GNN prediction threshold")

    print("\nLOCKED HYBRID INFERENCE", flush=True)
    hybrid_probability = hybrid_helpers.predict_probability(
        model, site_test, graph_features, graph_lookup, "LOCKED HYBRID"
    )
    hybrid_prediction = (hybrid_probability >= EXPECTED_HYBRID_THRESHOLD).astype(np.uint8)
    hybrid_metrics = comparator_helpers.baseline_helpers.calculate_metrics(
        site_test["target"], hybrid_probability, hybrid_prediction, EXPECTED_HYBRID_THRESHOLD
    )
    gnn_metrics = replay_metric_record(site_test, gnn_predictions, gnn_audit, "gnn")
    deep_metrics = replay_metric_record(site_test, deep_predictions, deep_audit, "deep")
    baseline_metrics = replay_metric_record(site_test, baseline_predictions, baseline_audit, "baseline")
    require(close(gnn_metrics["mcc"], EXPECTED_GNN_MCC), "replayed GNN MCC")
    require(close(deep_metrics["mcc"], EXPECTED_DEEP_MCC), "replayed deep MCC")
    require(close(baseline_metrics["mcc"], EXPECTED_BASELINE_MCC), "replayed baseline MCC")
    minimum_status, absolute_status, acceptance = comparator_helpers.assessment(hybrid_metrics)

    print("\nREQUIRED HYBRID BREAKDOWNS")
    breakdowns, unrepresented = comparator_helpers.baseline_helpers.make_breakdowns(
        site_test, hybrid_probability, hybrid_prediction, EXPECTED_HYBRID_THRESHOLD
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
        "hybrid": {"probability": hybrid_probability, "prediction": hybrid_prediction},
        "gnn": {
            "probability": gnn_predictions["probability"].astype(np.float64),
            "prediction": gnn_predictions["prediction"].astype(np.uint8),
        },
        "deep": {
            "probability": deep_predictions["probability"].astype(np.float64),
            "prediction": deep_predictions["prediction"].astype(np.uint8),
        },
        "baseline": {
            "probability": baseline_predictions["probability"].astype(np.float64),
            "prediction": baseline_predictions["prediction"].astype(np.uint8),
        },
    }
    all_metrics = {
        "hybrid": hybrid_metrics,
        "gnn": gnn_metrics,
        "deep": deep_metrics,
        "baseline": baseline_metrics,
    }
    bootstrap, bootstrap_arrays = paired_bootstrap(site_test, model_data, all_metrics)
    comparison, comparison_rows = comparison_documents(all_metrics, bootstrap)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    deterministic_npz(PREDICTIONS, {
        "candidate_id": np.asarray([EXPECTED_HYBRID_CANDIDATE]),
        "model_sha256": np.asarray([EXPECTED_INPUTS[HYBRID_MODEL]]),
        "partition": np.asarray(["DEV_SITE_TEST"]),
        "prediction": hybrid_prediction,
        "probability": hybrid_probability.astype("<f8", copy=False),
        "record_id": site_test["record_id"].astype("<u4", copy=False),
        "site_ids": site_test["site_ids"],
        "site_indices": site_test["site_indices"].astype("<u2", copy=False),
        "site_row": site_test["site_row"].astype("<u2", copy=False),
        "stuck_value": site_test["stuck_value"].astype(np.uint8, copy=False),
        "target_detected": site_test["target"].astype(np.uint8, copy=False),
        "threshold": np.asarray([EXPECTED_HYBRID_THRESHOLD], dtype="<f8"),
        "vector_ids": site_test["vector_ids"].astype("<u2", copy=False),
        "vector_row": site_test["vector_row"].astype(np.uint8, copy=False),
    })
    atomic_json(METRICS, {
        "stage": "11D-2C", "status": "PASS", "partition": "DEV_SITE_TEST",
        "candidate": EXPECTED_HYBRID_CANDIDATE,
        "threshold": EXPECTED_HYBRID_THRESHOLD,
        "metrics": hybrid_metrics,
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "acceptance_checks": acceptance,
    })
    write_breakdowns(breakdowns, unrepresented)
    write_bootstrap(bootstrap, bootstrap_arrays)
    write_reliability_curve(site_test["target"], hybrid_probability)
    atomic_csv(COMPARISON_CSV, [
        "metric", "hybrid_value", "gnn_value", "deep_value", "baseline_value",
        "hybrid_minus_gnn", "hybrid_minus_deep", "hybrid_minus_baseline",
        "preferred_direction",
    ], comparison_rows)
    atomic_json(COMPARISON_JSON, comparison)

    outputs = {
        "predictions": record(PREDICTIONS),
        "metrics": record(METRICS),
        "breakdown_csv": record(BREAKDOWN_CSV),
        "breakdown_json": record(BREAKDOWN_JSON),
        "paired_bootstrap_csv": record(BOOTSTRAP_CSV),
        "paired_bootstrap_json": record(BOOTSTRAP_JSON),
        "paired_bootstrap_distribution": record(BOOTSTRAP_DISTRIBUTION),
        "reliability_curve": record(RELIABILITY_CURVE),
        "comparison_csv": record(COMPARISON_CSV),
        "comparison_json": record(COMPARISON_JSON),
    }
    created_at = datetime.now(timezone.utc).isoformat()
    inference_batches = math.ceil(EXPECTED_ROWS / hybrid_helpers.INFERENCE_BATCH_SIZE)
    atomic_json(EVALUATION_LOCK, {
        "stage": "11D-2C",
        "title": "LOCKED HYBRID DEV_SITE_TEST EVALUATION LOCK",
        "status": "PASS", "lock_status": "FROZEN", "created_at_utc": created_at,
        "candidate_id": EXPECTED_HYBRID_CANDIDATE,
        "model_sha256": EXPECTED_INPUTS[HYBRID_MODEL],
        "threshold": EXPECTED_HYBRID_THRESHOLD,
        "partition": "DEV_SITE_TEST",
        "inference_batches": inference_batches,
        "hybrid_metrics": hybrid_metrics,
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "comparison": comparison,
        "model_fit_calls": 0,
        "graph_scaler_fit_calls": 0,
        "candidate_selection_calls": 0,
        "threshold_selection_calls": 0,
        "model_retrained": False,
        "threshold_changed": False,
        "graph_scaler_changed": False,
        "architecture_changed": False,
        "outputs": outputs,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "post_site_test_retraining_authorized": False,
    })
    evaluation_lock_record = record(EVALUATION_LOCK)

    manifest = {
        "stage": "11D-2C",
        "title": "LOCKED HYBRID DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS",
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
        "hybrid_candidate": EXPECTED_HYBRID_CANDIDATE,
        "hybrid_threshold": EXPECTED_HYBRID_THRESHOLD,
        "hybrid_metrics": hybrid_metrics,
        "gnn_metrics": gnn_metrics,
        "deep_metrics": deep_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "input_evidence": input_evidence,
        "hybrid_handoff": hybrid_handoff,
        "comparator_handoff": comparator_handoff,
        "environment": environment,
        "outputs": outputs,
        "evaluation_lock": evaluation_lock_record,
        "model_fit_calls": 0,
        "graph_scaler_fit_calls": 0,
        "candidate_selection_calls": 0,
        "threshold_selection_calls": 0,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "hybrid_model_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = record(MANIFEST)

    hybrid_interval = interval(bootstrap, "mcc", "hybrid")
    delta_gnn_interval = interval(bootstrap, "mcc", "hybrid_minus_gnn")
    delta_deep_interval = interval(bootstrap, "mcc", "hybrid_minus_deep")
    delta_baseline_interval = interval(bootstrap, "mcc", "hybrid_minus_baseline")
    advancement_status = "MET" if comparison["hybrid_advancement_target_met"] else "NOT_MET"
    atomic_json(AUDIT, {
        "stage": "11D-2C",
        "title": "LOCKED HYBRID DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS FREEZE",
        "status": "PASS",
        "evaluation_status": "FROZEN",
        "comparison_status": "FROZEN",
        "hybrid_status": "FROZEN COMPARATOR",
        "gnn_status": "FROZEN COMPARATOR",
        "deep_status": "FROZEN COMPARATOR",
        "baseline_status": "FROZEN COMPARATOR",
        "hybrid_candidate": EXPECTED_HYBRID_CANDIDATE,
        "hybrid_threshold": EXPECTED_HYBRID_THRESHOLD,
        "hybrid_mcc": hybrid_metrics["mcc"],
        "hybrid_mcc_ci_95": [hybrid_interval["ci_lower"], hybrid_interval["ci_upper"]],
        "hybrid_balanced_accuracy": hybrid_metrics["balanced_accuracy"],
        "hybrid_precision": hybrid_metrics["precision"],
        "hybrid_recall": hybrid_metrics["recall"],
        "hybrid_specificity": hybrid_metrics["specificity"],
        "hybrid_f1": hybrid_metrics["f1_score"],
        "hybrid_pr_auc": hybrid_metrics["pr_auc"],
        "hybrid_roc_auc": hybrid_metrics["roc_auc"],
        "hybrid_brier": hybrid_metrics["brier_score"],
        "gnn_mcc": gnn_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "baseline_mcc": baseline_metrics["mcc"],
        "hybrid_minus_gnn_mcc": comparison["hybrid_minus_gnn_mcc"],
        "hybrid_minus_gnn_mcc_ci_95": [delta_gnn_interval["ci_lower"], delta_gnn_interval["ci_upper"]],
        "hybrid_minus_deep_mcc": comparison["hybrid_minus_deep_mcc"],
        "hybrid_minus_deep_mcc_ci_95": [delta_deep_interval["ci_lower"], delta_deep_interval["ci_upper"]],
        "hybrid_minus_baseline_mcc": comparison["hybrid_minus_baseline_mcc"],
        "hybrid_minus_baseline_mcc_ci_95": [delta_baseline_interval["ci_lower"], delta_baseline_interval["ci_upper"]],
        "point_winner": comparison["point_winner"],
        "hybrid_improvement_over_gnn_statistically_supported": comparison["hybrid_improvement_over_gnn_statistically_supported"],
        "hybrid_improvement_over_deep_statistically_supported": comparison["hybrid_improvement_over_deep_statistically_supported"],
        "hybrid_improvement_over_baseline_statistically_supported": comparison["hybrid_improvement_over_baseline_statistically_supported"],
        "required_hybrid_mcc_improvement_met": comparison["required_hybrid_mcc_improvement_over_gnn_met"],
        "hybrid_advancement_target_status": advancement_status,
        "minimum_validity_status": minimum_status,
        "absolute_performance_status": absolute_status,
        "required_breakdowns": "PASS",
        "paired_site_bootstrap": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "model_retrained": False,
        "threshold_changed": False,
        "graph_scaler_changed": False,
        "architecture_changed": False,
        "dev_site_test_state": "CONSUMED AND FROZEN FOR HYBRID",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "post_site_test_retraining_authorized": False,
        "evaluation_lock": evaluation_lock_record,
        "manifest": manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "graph_dataset_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "hybrid_model_modified": False,
        "gnn_model_modified": False,
        "deep_model_modified": False,
        "baseline_model_modified": False,
        "next_gate": "STAGE 11D-2D — HYBRID DISPOSITION AND FINAL DIAGNOSTIC MODEL FREEZE",
    })
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-2C — LOCKED HYBRID DEV_SITE_TEST EVALUATION AND COMPARATOR ANALYSIS FREEZE")
    print("Status                         : PASS")
    print("Evaluation status              : FROZEN")
    print("Comparison status              : FROZEN")
    print("Hybrid candidate               :", EXPECTED_HYBRID_CANDIDATE)
    print("Hybrid threshold               :", format(EXPECTED_HYBRID_THRESHOLD, ".17g"))
    print("DEV_SITE_TEST sites            :", EXPECTED_SITES)
    print("DEV_SITE_TEST samples          :", EXPECTED_ROWS)
    print("Hybrid MCC                     :", f"{hybrid_metrics['mcc']:.8f}")
    print("Hybrid MCC 95% site CI         :", f"[{hybrid_interval['ci_lower']:.8f}, {hybrid_interval['ci_upper']:.8f}]")
    print("Hybrid balanced accuracy       :", f"{hybrid_metrics['balanced_accuracy']:.8f}")
    print("Hybrid precision               :", f"{hybrid_metrics['precision']:.8f}")
    print("Hybrid recall                  :", f"{hybrid_metrics['recall']:.8f}")
    print("Hybrid specificity             :", f"{hybrid_metrics['specificity']:.8f}")
    print("Hybrid F1                      :", f"{hybrid_metrics['f1_score']:.8f}")
    print("Hybrid PR-AUC                  :", f"{hybrid_metrics['pr_auc']:.8f}")
    print("Hybrid ROC-AUC                 :", f"{hybrid_metrics['roc_auc']:.8f}")
    print("Hybrid Brier                   :", f"{hybrid_metrics['brier_score']:.8f}")
    print("GNN MCC                        :", f"{gnn_metrics['mcc']:.8f}")
    print("Deep MLP MCC                   :", f"{deep_metrics['mcc']:.8f}")
    print("Conventional baseline MCC      :", f"{baseline_metrics['mcc']:.8f}")
    print("MCC delta (hybrid-GNN)         :", f"{comparison['hybrid_minus_gnn_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{delta_gnn_interval['ci_lower']:.8f}, {delta_gnn_interval['ci_upper']:.8f}]")
    print("MCC delta (hybrid-deep)        :", f"{comparison['hybrid_minus_deep_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{delta_deep_interval['ci_lower']:.8f}, {delta_deep_interval['ci_upper']:.8f}]")
    print("MCC delta (hybrid-baseline)    :", f"{comparison['hybrid_minus_baseline_mcc']:.8f}")
    print("Delta 95% paired-site CI       :", f"[{delta_baseline_interval['ci_lower']:.8f}, {delta_baseline_interval['ci_upper']:.8f}]")
    print("Point winner                   :", comparison["point_winner"])
    print("Statistical gain vs GNN        :", "SUPPORTED" if comparison["hybrid_improvement_over_gnn_statistically_supported"] else "NOT_SUPPORTED")
    print("Required +0.02 MCC over GNN    :", "MET" if comparison["required_hybrid_mcc_improvement_over_gnn_met"] else "NOT_MET")
    print("Hybrid advancement target      :", advancement_status)
    print("Minimum validity               :", minimum_status)
    print("Absolute performance target    :", absolute_status)
    print("Model fitting calls            : 0")
    print("Graph-scaler fitting calls     : 0")
    print("Threshold changed              : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Hybrid model modified          : NO")
    print("Comparison                     :", COMPARISON_JSON)
    print("Comparison SHA                 :", outputs["comparison_json"]["sha256"])
    print("Evaluation lock                :", EVALUATION_LOCK)
    print("Evaluation lock SHA            :", evaluation_lock_record["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-2D — HYBRID DISPOSITION AND FINAL DIAGNOSTIC MODEL FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
