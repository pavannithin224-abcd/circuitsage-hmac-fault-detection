#!/usr/bin/env python3
"""Freeze the HMAC diagnostic task, feature policy, and group-safe site split."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-DIAGNOSTIC-POLICY-FREEZER-v1"
POLICY_VERSION = "HMAC-DIAGNOSTIC-TASK-POLICY-v1"
SPLIT_VERSION = "HMAC-FAULT-SITE-GROUP-SPLIT-v1"
SPLIT_SEED = (
    "HMAC-FAULT-SITE-GROUP-SPLIT-v1|"
    "OpenTitan-83fc48ed3a727399056772d12be8c7d4a8a276f0"
)

ROOT = Path.cwd().resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
DATASET_ROOT = RESULT_ROOT / "canonical_dataset_11c5c"

CANONICAL_DATASET = (
    DATASET_ROOT / "hmac_fault_campaign_canonical_train_11c5c.csv.gz"
)
INSTANCE_SUMMARY = (
    DATASET_ROOT / "hmac_fault_instance_summary_train_11c5c.csv"
)
CANONICAL_SCHEMA = (
    DATASET_ROOT / "hmac_canonical_dataset_schema_11c5c.json"
)
CANONICAL_MANIFEST = (
    RESULT_ROOT / "hmac_canonical_dataset_manifest_11c5c.json"
)
CANONICAL_AUDIT = (
    RESULT_ROOT / "hmac_canonical_dataset_schema_freeze_11c5c.json"
)
LEGAL_SITES = (
    ROOT / "results/hmac_fault_campaign_11c1/hmac_legal_fault_sites_11c1b.json"
)
VECTOR_SELECTION = (
    ROOT / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json"
)
VECTOR_POOL = (
    ROOT / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"
)
GOLDEN_NETLIST = (
    ROOT / "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json"
)
CONSOLIDATOR = ROOT / "stage_11c5c_consolidator.py"

POLICY_PATH = (
    CONFIG_ROOT / "hmac_diagnostic_task_feature_split_policy_11c5d.json"
)
SPLIT_PATH = (
    RESULT_ROOT / "hmac_fault_site_group_split_11c5d.csv"
)
SPLIT_MANIFEST_PATH = (
    RESULT_ROOT / "hmac_fault_site_group_split_manifest_11c5d.json"
)
AUDIT_PATH = (
    RESULT_ROOT / "hmac_diagnostic_task_feature_split_freeze_11c5d.json"
)

EXPECTED_INPUTS = {
    CANONICAL_DATASET:
        "dcc79c94fa3a8d592b516e5f5683b1bf007682745103a9419ee20ef20a7ec78f",
    INSTANCE_SUMMARY:
        "2fb6953097ee7590942f07b2e9ed99103f559503d3f8afce6fcb090978b877c1",
    CANONICAL_SCHEMA:
        "2a08594f9573ff833a872d409783cee984bbb7499c7b013af4c72377ede3f772",
    CANONICAL_MANIFEST:
        "da5437f592362b2675e68abfde93857451f355e22b07948de2d8390a733173d4",
    CANONICAL_AUDIT:
        "a43de512ffab41cddaee286b2f459e9064f9588ee9c4b4621107485445ed3729",
    LEGAL_SITES:
        "d9a39366bb35a277125376c6dc26c83273b45c0d7732fb93a3403401a071cb45",
    VECTOR_SELECTION:
        "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    VECTOR_POOL:
        "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
    GOLDEN_NETLIST:
        "0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1",
    CONSOLIDATOR:
        "16d437ba7a9ac1771d3c8fc444309d5020e01eff934e7503e68de5274e209bb9",
}

EXPECTED_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_ENABLED_ROWS = 2_923_392
EXPECTED_TRAIN_VECTORS = 64
EXPECTED_DETECTED_ROWS = 1_278_888
EXPECTED_NONDETECTED_ROWS = 1_644_504
EXPECTED_DETECTED_INSTANCES = 22_930
EXPECTED_ACTIVATED_UNOBSERVED = 15_378
EXPECTED_UNACTIVATED = 7_370

PARTITIONS = (
    ("DEV_TRAIN", 15_987),
    ("DEV_CALIBRATION", 3_426),
    ("DEV_SITE_TEST", 3_426),
)

INSTANCE_HEADER = [
    "fault_instance_id",
    "site_id",
    "site_index",
    "batch_id",
    "selector",
    "stuck_value",
    "site_category",
    "driver_cell_type",
    "train_vectors",
    "activated_vectors",
    "detected_vectors",
    "timeout_vectors",
    "final_classification",
]

SPLIT_HEADER = [
    "site_id",
    "site_index",
    "partition",
    "partition_rank",
    "split_hash_sha256",
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
        stop(f"artifact is outside project root: {path}")


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


def output_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_file(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(path.stat().st_size > 0, f"empty {label}: {path}")
    actual = sha256(path)
    require(
        actual == expected,
        f"SHA mismatch for {label}: expected {expected}, actual {actual}",
    )


def integer(row: dict[str, str], field: str, location: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        stop(f"invalid integer {field} at {location}")


def find_sites(value):
    if isinstance(value, dict):
        for child in value.values():
            result = find_sites(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        if (
            value
            and isinstance(value[0], dict)
            and "fault_site_id" in value[0]
            and (
                "driver_cell_type" in value[0]
                or "site_category" in value[0]
            )
        ):
            return value
        for child in value:
            result = find_sites(child)
            if result is not None:
                return result
    return None


def load_instance_summary() -> tuple[dict[str, dict], dict[str, dict[int, dict]]]:
    instances = {}
    by_site = defaultdict(dict)
    with INSTANCE_SUMMARY.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == INSTANCE_HEADER, "fault-instance summary header")
        for line_number, row in enumerate(reader, start=2):
            location = f"instance-summary line {line_number}"
            instance_id = row["fault_instance_id"]
            site_id = row["site_id"]
            stuck_value = integer(row, "stuck_value", location)
            require(stuck_value in (0, 1), f"{location} stuck value")
            require(instance_id == f"{site_id}-SA{stuck_value}", f"{location} instance ID")
            require(instance_id not in instances, f"duplicate instance {instance_id}")
            require(stuck_value not in by_site[site_id], f"duplicate {site_id} SA{stuck_value}")
            train_vectors = integer(row, "train_vectors", location)
            activated = integer(row, "activated_vectors", location)
            detected = integer(row, "detected_vectors", location)
            timeouts = integer(row, "timeout_vectors", location)
            require(train_vectors == 64, f"{location} train-vector count")
            require(0 <= detected <= activated <= train_vectors, f"{location} response counts")
            require(0 <= timeouts <= detected, f"{location} timeout count")
            classification = row["final_classification"]
            expected_classification = (
                "DETECTED"
                if detected
                else "ACTIVATED_UNOBSERVED"
                if activated
                else "UNACTIVATED"
            )
            require(classification == expected_classification, f"{location} classification")
            record = {
                "fault_instance_id": instance_id,
                "site_id": site_id,
                "site_index": integer(row, "site_index", location),
                "batch_id": integer(row, "batch_id", location),
                "selector": integer(row, "selector", location),
                "stuck_value": stuck_value,
                "site_category": row["site_category"],
                "driver_cell_type": row["driver_cell_type"],
                "train_vectors": train_vectors,
                "activated_vectors": activated,
                "detected_vectors": detected,
                "timeout_vectors": timeouts,
                "final_classification": classification,
            }
            instances[instance_id] = record
            by_site[site_id][stuck_value] = record
    require(len(instances) == EXPECTED_FAULT_INSTANCES, "fault-instance summary count")
    require(len(by_site) == EXPECTED_SITES, "fault-site summary count")
    require(all(set(pair) == {0, 1} for pair in by_site.values()), "SA0/SA1 site pairing")
    return instances, dict(by_site)


def verify_legal_site_features(by_site: dict[str, dict[int, dict]]) -> dict[str, dict]:
    legal_data = load_json(LEGAL_SITES)
    legal_records = find_sites(legal_data)
    require(isinstance(legal_records, list), "legal-site records not found")
    require(len(legal_records) == EXPECTED_SITES, "legal-site count")
    legal_by_id = {}
    required_fields = {
        "fault_site_id",
        "site_index",
        "site_category",
        "driver_cell_type",
        "cell_fanout",
    }
    for record in legal_records:
        require(required_fields <= set(record), f"legal-site fields for {record.get('fault_site_id')}")
        site_id = str(record["fault_site_id"])
        require(site_id not in legal_by_id, f"duplicate legal site {site_id}")
        require(site_id in by_site, f"legal site absent from canonical summary: {site_id}")
        match = re.fullmatch(r"HMAC-STEM-(\d{6})", site_id)
        require(match is not None, f"invalid legal site ID {site_id}")
        require(int(record["site_index"]) == int(match.group(1)), f"legal site index {site_id}")
        require(int(record["cell_fanout"]) >= 0, f"negative fanout {site_id}")
        summary_record = by_site[site_id][0]
        require(
            str(record["site_category"]) == summary_record["site_category"],
            f"site category differs for {site_id}",
        )
        require(
            str(record["driver_cell_type"]) == summary_record["driver_cell_type"],
            f"driver cell type differs for {site_id}",
        )
        legal_by_id[site_id] = record
    require(set(legal_by_id) == set(by_site), "legal/canonical site-set mismatch")
    return legal_by_id


def verify_vectors() -> tuple[list[int], dict]:
    selection = load_json(VECTOR_SELECTION)
    pool = load_json(VECTOR_POOL)
    selected_ids = selection.get("selected_vector_ids")
    selected_positions = selection.get("selected_schedule_positions")
    require(isinstance(selected_ids, list) and len(selected_ids) == 64, "selected vector IDs")
    require(selected_positions == list(range(64)), "selected vector positions")
    vectors = pool.get("vectors")
    require(isinstance(vectors, list) and len(vectors) == 256, "vector pool")
    split_counts = Counter(str(item.get("split")) for item in vectors)
    require(
        split_counts == Counter({"TRAIN": 160, "VALIDATION": 48, "HOLDOUT_TEST": 48}),
        "vector-pool split counts",
    )
    by_id = {int(item["vector_id"]): item for item in vectors}
    require(len(by_id) == 256, "duplicate vector IDs")
    for raw_id in selected_ids:
        vector_id = int(raw_id)
        record = by_id.get(vector_id)
        require(record is not None, f"selected vector {vector_id} missing")
        require(record.get("split") == "TRAIN", f"selected vector {vector_id} is not TRAIN")
        for field in ("key", "message", "expected_hmac_sha256"):
            value = str(record.get(field, "")).lower().removeprefix("0x")
            require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"vector {vector_id} {field}")
    return [int(value) for value in selected_ids], dict(split_counts)


def split_assignments(by_site: dict[str, dict[int, dict]]) -> tuple[list[dict], dict[str, str]]:
    ranked = []
    for site_id in by_site:
        digest = hashlib.sha256(f"{SPLIT_SEED}|{site_id}".encode()).hexdigest()
        ranked.append((digest, site_id))
    ranked.sort()
    assignments = []
    partition_by_site = {}
    start = 0
    for partition, count in PARTITIONS:
        end = start + count
        for partition_rank, (digest, site_id) in enumerate(ranked[start:end]):
            site_index = by_site[site_id][0]["site_index"]
            assignments.append(
                {
                    "site_id": site_id,
                    "site_index": site_index,
                    "partition": partition,
                    "partition_rank": partition_rank,
                    "split_hash_sha256": digest,
                }
            )
            partition_by_site[site_id] = partition
        start = end
    require(start == EXPECTED_SITES, "partition allocation count")
    require(len(partition_by_site) == EXPECTED_SITES, "partition assignment count")
    assignments.sort(key=lambda item: item["site_index"])
    require(
        [item["site_index"] for item in assignments] == list(range(1, EXPECTED_SITES + 1)),
        "split assignment site-index coverage",
    )
    return assignments, partition_by_site


def partition_statistics(
    by_site: dict[str, dict[int, dict]],
    legal_by_id: dict[str, dict],
    partition_by_site: dict[str, str],
) -> dict:
    stats = {}
    for partition, expected_sites in PARTITIONS:
        stats[partition] = {
            "sites": 0,
            "fault_instances": 0,
            "enabled_rows": 0,
            "detected_rows": 0,
            "nondetected_rows": 0,
            "activated_rows": 0,
            "timeout_rows": 0,
            "instance_classes": Counter(),
            "site_categories": Counter(),
            "driver_cell_types": Counter(),
        }
    for site_id, pair in by_site.items():
        partition = partition_by_site[site_id]
        entry = stats[partition]
        entry["sites"] += 1
        entry["fault_instances"] += 2
        entry["enabled_rows"] += 128
        legal = legal_by_id[site_id]
        entry["site_categories"][str(legal["site_category"])] += 1
        entry["driver_cell_types"][str(legal["driver_cell_type"])] += 1
        for instance in pair.values():
            entry["detected_rows"] += instance["detected_vectors"]
            entry["activated_rows"] += instance["activated_vectors"]
            entry["timeout_rows"] += instance["timeout_vectors"]
            entry["instance_classes"][instance["final_classification"]] += 1
    for partition, expected_sites in PARTITIONS:
        entry = stats[partition]
        require(entry["sites"] == expected_sites, f"{partition} site count")
        require(entry["fault_instances"] == expected_sites * 2, f"{partition} instance count")
        require(entry["enabled_rows"] == expected_sites * 128, f"{partition} row count")
        entry["nondetected_rows"] = entry["enabled_rows"] - entry["detected_rows"]
        require(entry["detected_rows"] > 0, f"{partition} has no positive labels")
        require(entry["nondetected_rows"] > 0, f"{partition} has no negative labels")
        require(len(entry["site_categories"]) >= 2, f"{partition} site-category diversity")
        entry["detected_prevalence"] = entry["detected_rows"] / entry["enabled_rows"]
        for field in ("instance_classes", "site_categories", "driver_cell_types"):
            entry[field] = dict(sorted(entry[field].items()))
    require(
        sum(value["sites"] for value in stats.values()) == EXPECTED_SITES,
        "partition site closure",
    )
    require(
        sum(value["enabled_rows"] for value in stats.values()) == EXPECTED_ENABLED_ROWS,
        "partition row closure",
    )
    require(
        sum(value["detected_rows"] for value in stats.values()) == EXPECTED_DETECTED_ROWS,
        "partition detected-row closure",
    )
    return stats


def feature_policy() -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "status": "FROZEN",
        "task": {
            "name": "pre_simulation_persistent_fault_detectability",
            "type": "binary_classification",
            "sample_grain": "one enabled persistent-fault instance and one input vector",
            "population": "ENABLED rows only",
            "target": "detected",
            "positive_label": 1,
            "negative_label": 0,
            "purpose": "Predict whether an SA0/SA1 fault-vector pair will be observable before executing its faulty simulation.",
            "baseline_rows": "Excluded from model samples and retained only as reference evidence.",
        },
        "allowed_feature_contract": {
            "fault_features": [
                {
                    "name": "stuck_value",
                    "encoding": "binary 0=SA0, 1=SA1",
                    "source": "canonical dataset",
                }
            ],
            "structural_features": [
                {
                    "name": "driver_cell_type",
                    "encoding": "one-hot categorical with frozen vocabulary",
                    "source": "legal fault-site inventory",
                },
                {
                    "name": "site_category",
                    "encoding": "one-hot categorical",
                    "source": "legal fault-site inventory",
                },
                {
                    "name": "cell_fanout",
                    "encoding": "numeric; scaler fitted on DEV_TRAIN only",
                    "source": "legal fault-site inventory",
                },
                {
                    "name": "is_primary_output_stem",
                    "encoding": "binary; derive only from frozen primary-output mapping",
                    "source": "legal fault-site inventory",
                },
                {
                    "name": "is_sequential_stem",
                    "encoding": "binary; derive from frozen site_category",
                    "source": "legal fault-site inventory",
                },
            ],
            "stimulus_features": [
                {
                    "name": "key_bits",
                    "width": 256,
                    "encoding": "binary, bit 255 through bit 0",
                    "source": "frozen vector pool",
                },
                {
                    "name": "message_bits",
                    "width": 256,
                    "encoding": "binary, bit 255 through bit 0",
                    "source": "frozen vector pool",
                },
            ],
            "total_dense_features_before_one_hot_expansion": 518,
        },
        "provenance_only_not_features": [
            "record_id",
            "batch_id",
            "site_selection",
            "selector",
            "site_id",
            "fault_instance_id",
            "vector_slot",
            "vector_id",
            "vector_split",
            "run_type",
            "fault_enable",
        ],
        "forbidden_target_leakage_features": [
            "detected",
            "activity",
            "cycles",
            "baseline_cycles",
            "latency_delta",
            "timed_out",
            "unknown",
            "expected_digest",
            "actual_digest",
            "digest_hamming_distance",
            "activated_vectors",
            "detected_vectors",
            "timeout_vectors",
            "final_classification",
        ],
        "preprocessing": {
            "fit_scope": "DEV_TRAIN only",
            "numeric_scalers": "Fit on DEV_TRAIN; serialize parameters and SHA256.",
            "categorical_vocabulary": "Freeze from structural source schema without using target labels.",
            "missing_values": "Not permitted; fail closed.",
            "class_balancing": "Training partition only; never resample calibration or test data.",
            "random_seed_required": True,
            "deterministic_execution_required": True,
        },
    }


def split_policy(partition_stats: dict) -> dict:
    return {
        "split_version": SPLIT_VERSION,
        "status": "FROZEN",
        "group_key": "site_id",
        "group_rule": "SA0 and SA1 and all 64 TRAIN-vector rows for one physical site remain together.",
        "algorithm": "Sort sites by SHA256(split_seed + '|' + site_id), then slice exact counts.",
        "split_seed": SPLIT_SEED,
        "partitions": {
            "DEV_TRAIN": {
                "sites": 15_987,
                "percentage": 70.0,
                "allowed_use": "Fit model parameters and preprocessing only.",
            },
            "DEV_CALIBRATION": {
                "sites": 3_426,
                "percentage": 15.0,
                "allowed_use": "Hyperparameter comparison, probability calibration, and threshold selection.",
            },
            "DEV_SITE_TEST": {
                "sites": 3_426,
                "percentage": 15.0,
                "allowed_use": "Locked unseen-site evaluation; no fitting or threshold selection.",
            },
        },
        "actual_partition_statistics": partition_stats,
        "reserved_vector_partitions": {
            "VALIDATION": {
                "vectors": 48,
                "campaign_records_present": 0,
                "allowed_use": "Later vector-generalization evaluation and final model selection.",
            },
            "HOLDOUT_TEST": {
                "vectors": 48,
                "campaign_records_present": 0,
                "allowed_use": "One-time final evaluation after architecture, features, hyperparameters, and threshold are locked.",
            },
        },
        "leakage_guards": [
            "No site_id may appear in more than one development partition.",
            "SA0 and SA1 for the same site may not be separated.",
            "No VALIDATION or HOLDOUT_TEST vector may enter development artifacts.",
            "DEV_SITE_TEST may not influence fitting, feature selection, hyperparameters, calibration, or threshold.",
            "HOLDOUT_TEST results may not trigger retraining or configuration changes.",
            "The conventional baseline and future GNN must use the identical frozen assignments.",
        ],
    }


def evaluation_policy() -> dict:
    prevalence = EXPECTED_DETECTED_ROWS / EXPECTED_ENABLED_ROWS
    return {
        "positive_prevalence_in_current_train_campaign": prevalence,
        "primary_model_selection_metric": "Matthews correlation coefficient (MCC)",
        "secondary_metrics": [
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
            "pr_auc",
            "roc_auc",
            "confusion_matrix",
        ],
        "threshold_selection": "DEV_CALIBRATION only; freeze before DEV_SITE_TEST and HOLDOUT_TEST.",
        "confidence_intervals": "Report 95% bootstrap confidence intervals grouped by site_id.",
        "minimum_validity_gate": {
            "mcc": "> 0",
            "balanced_accuracy": "> 0.50",
            "pr_auc": f"> positive prevalence ({prevalence:.8f})",
            "comparison": "Must outperform majority and stratified-random baselines.",
        },
        "project_target_gate": {
            "mcc": ">= 0.40",
            "balanced_accuracy": ">= 0.70",
            "f1_score": ">= 0.70",
            "recall": ">= 0.70",
            "note": "Failure is reported honestly and does not permit threshold changes using holdout data.",
        },
        "reporting_rules": [
            "Report per-partition metrics and aggregate metrics.",
            "Report SA0 and SA1 metrics separately.",
            "Report sequential and combinational stem metrics separately.",
            "Report results by driver_cell_type when sample support is sufficient.",
            "Accuracy alone is not an acceptance metric.",
        ],
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (POLICY_PATH, SPLIT_PATH, SPLIT_MANIFEST_PATH, AUDIT_PATH)
    for path in outputs:
        require(not path.exists(), f"Stage 11C-5D output already exists: {path}")
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()

    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE 11C-5D — DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY")
    print("FROZEN INPUT VERIFICATION")
    input_evidence = {}
    for path, expected in EXPECTED_INPUTS.items():
        verify_file(path, expected, relative(path))
        input_evidence[relative(path)] = {
            "sha256": expected,
            "bytes": path.stat().st_size,
        }
        print(f"  {path.name:<66} : OK", flush=True)

    canonical_manifest = load_json(CANONICAL_MANIFEST)
    canonical_audit = load_json(CANONICAL_AUDIT)
    canonical_schema = load_json(CANONICAL_SCHEMA)
    require(canonical_manifest.get("status") == "PASS", "canonical manifest status")
    require(canonical_audit.get("status") == "PASS", "canonical audit status")
    require(canonical_audit.get("canonical_dataset_status") == "FROZEN", "dataset freeze")
    require(canonical_audit.get("schema_status") == "FROZEN", "schema freeze")
    require(canonical_audit.get("model_training_authorized") is False, "prior training gate")
    require(canonical_schema.get("schema_version") == "HMAC-FAULT-CANONICAL-SCHEMA-v1", "schema version")
    require(canonical_manifest.get("total_records") == 2_926_272, "canonical row count")
    require(canonical_manifest.get("enabled_records") == EXPECTED_ENABLED_ROWS, "enabled row count")
    require(canonical_manifest.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(canonical_manifest.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")

    instances, by_site = load_instance_summary()
    classification_counts = Counter(
        item["final_classification"] for item in instances.values()
    )
    require(classification_counts["DETECTED"] == EXPECTED_DETECTED_INSTANCES, "detected instances")
    require(
        classification_counts["ACTIVATED_UNOBSERVED"]
        == EXPECTED_ACTIVATED_UNOBSERVED,
        "activated-unobserved instances",
    )
    require(classification_counts["UNACTIVATED"] == EXPECTED_UNACTIVATED, "unactivated instances")
    require(
        sum(item["detected_vectors"] for item in instances.values())
        == EXPECTED_DETECTED_ROWS,
        "detected row total",
    )

    legal_by_id = verify_legal_site_features(by_site)
    selected_vector_ids, vector_split_counts = verify_vectors()
    assignments, partition_by_site = split_assignments(by_site)
    stats = partition_statistics(by_site, legal_by_id, partition_by_site)

    split_temporary = SPLIT_PATH.with_name(SPLIT_PATH.name + ".tmp")
    with split_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPLIT_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignments)
    split_temporary.replace(SPLIT_PATH)
    split_record = output_record(SPLIT_PATH)

    policy = {
        "stage": "11C-5D",
        "title": "DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY FREEZE",
        "status": "PASS",
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_task_and_features": feature_policy(),
        "split_policy": split_policy(stats),
        "evaluation_policy": evaluation_policy(),
        "source_dataset": {
            "path": relative(CANONICAL_DATASET),
            "sha256": EXPECTED_INPUTS[CANONICAL_DATASET],
            "enabled_rows": EXPECTED_ENABLED_ROWS,
            "positive_rows": EXPECTED_DETECTED_ROWS,
            "negative_rows": EXPECTED_NONDETECTED_ROWS,
        },
        "site_split_assignment": split_record,
        "selected_train_vector_ids": selected_vector_ids,
        "vector_pool_split_counts": vector_split_counts,
        "development_training_authorization": "DEV_TRAIN_ONLY",
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "gnn_training_authorized": False,
        "input_evidence": input_evidence,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
    }
    atomic_json(POLICY_PATH, policy)
    policy_record = output_record(POLICY_PATH)

    split_manifest = {
        "stage": "11C-5D",
        "status": "PASS",
        "split_version": SPLIT_VERSION,
        "split_seed": SPLIT_SEED,
        "algorithm": "SHA256-ranked exact-count physical-site group split",
        "group_key": "site_id",
        "site_split_assignment": split_record,
        "site_count": EXPECTED_SITES,
        "fault_instance_count": EXPECTED_FAULT_INSTANCES,
        "enabled_row_count": EXPECTED_ENABLED_ROWS,
        "partitions": stats,
        "overlapping_sites": 0,
        "unassigned_sites": 0,
        "sa0_sa1_separations": 0,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
    }
    atomic_json(SPLIT_MANIFEST_PATH, split_manifest)
    split_manifest_record = output_record(SPLIT_MANIFEST_PATH)

    audit = {
        "stage": "11C-5D",
        "title": "DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY FREEZE",
        "status": "PASS",
        "task_status": "FROZEN",
        "feature_policy_status": "FROZEN",
        "split_policy_status": "FROZEN",
        "task": "pre_simulation_persistent_fault_detectability",
        "task_type": "binary_classification",
        "target": "detected",
        "model_samples": EXPECTED_ENABLED_ROWS,
        "positive_samples": EXPECTED_DETECTED_ROWS,
        "negative_samples": EXPECTED_NONDETECTED_ROWS,
        "physical_site_groups": EXPECTED_SITES,
        "fault_instances": EXPECTED_FAULT_INSTANCES,
        "dev_train_sites": 15_987,
        "dev_calibration_sites": 3_426,
        "dev_site_test_sites": 3_426,
        "site_overlap": 0,
        "sa0_sa1_separations": 0,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "forbidden_leakage_features": len(
            policy["diagnostic_task_and_features"]["forbidden_target_leakage_features"]
        ),
        "development_training_authorization": "DEV_TRAIN_ONLY",
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "gnn_training_authorized": False,
        "policy": policy_record,
        "site_split_assignment": split_record,
        "split_manifest": split_manifest_record,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "next_gate": "LEAKAGE-SAFE FEATURE MATRIX GENERATION AND FREEZE",
    }
    atomic_json(AUDIT_PATH, audit)
    audit_record = output_record(AUDIT_PATH)

    print("\nPARTITION SUMMARY")
    for partition, expected_sites in PARTITIONS:
        entry = stats[partition]
        print(
            f"  {partition:<16} sites={entry['sites']:5d} "
            f"faults={entry['fault_instances']:5d} "
            f"rows={entry['enabled_rows']:7d} "
            f"positive={entry['detected_rows']:7d} "
            f"prevalence={entry['detected_prevalence'] * 100:6.2f}%"
        )

    print("\nSTAGE 11C-5D — DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY FREEZE")
    print("Status                       : PASS")
    print("Task status                  : FROZEN")
    print("Feature policy status        : FROZEN")
    print("Split policy status          : FROZEN")
    print("Primary task                 : PRE-SIMULATION BINARY DETECTABILITY")
    print("Target                       : detected")
    print("Model samples                :", EXPECTED_ENABLED_ROWS)
    print("Positive samples             :", EXPECTED_DETECTED_ROWS)
    print("Negative samples             :", EXPECTED_NONDETECTED_ROWS)
    print("Physical site groups         :", EXPECTED_SITES)
    print("DEV_TRAIN sites              : 15987")
    print("DEV_CALIBRATION sites        : 3426")
    print("DEV_SITE_TEST sites          : 3426")
    print("Overlapping sites            : 0")
    print("SA0/SA1 group separations    : 0")
    print("VALIDATION vectors exposed   : 0")
    print("HOLDOUT vectors exposed      : 0")
    print("Post-simulation leakage      : BLOCKED")
    print("Development training         : AUTHORIZED FOR DEV_TRAIN ONLY")
    print("Validation campaign          : NOT YET AUTHORIZED")
    print("Holdout campaign             : NOT YET AUTHORIZED")
    print("GNN training                 : NOT YET AUTHORIZED")
    print("Frozen RTL modified          : NO")
    print("Golden netlist modified      : NO")
    print("Canonical dataset modified   : NO")
    print("Policy                       :", POLICY_PATH)
    print("Policy SHA                   :", policy_record["sha256"])
    print("Site split                   :", SPLIT_PATH)
    print("Site split SHA               :", split_record["sha256"])
    print("Split manifest               :", SPLIT_MANIFEST_PATH)
    print("Split manifest SHA           :", split_manifest_record["sha256"])
    print("Audit                        :", AUDIT_PATH)
    print("Audit SHA                    :", audit_record["sha256"])
    print("Next gate                    : LEAKAGE-SAFE FEATURE MATRIX GENERATION AND FREEZE")


if __name__ == "__main__":
    main()
