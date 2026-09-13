#!/usr/bin/env python3
"""Create and freeze the canonical TRAIN dataset for the HMAC fault campaign."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-CANONICAL-DATASET-CONSOLIDATOR-v1"
SCHEMA_VERSION = "HMAC-FAULT-CANONICAL-SCHEMA-v1"

ROOT = Path.cwd().resolve()
QUALIFICATION_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
DATASET_ROOT = QUALIFICATION_ROOT / "canonical_dataset_11c5c"

SOURCE_MANIFEST = (
    QUALIFICATION_ROOT / "hmac_full_campaign_dataset_manifest_11c5b.json"
)
SOURCE_AUDIT = (
    QUALIFICATION_ROOT / "hmac_full_campaign_dataset_integrity_freeze_11c5b.json"
)
SOURCE_LOG = (
    QUALIFICATION_ROOT / "stage_11c5b_full_campaign_dataset_integrity.log"
)
SOURCE_VERIFIER = ROOT / "stage_11c5b_verifier.py"

CANONICAL_DATASET = (
    DATASET_ROOT / "hmac_fault_campaign_canonical_train_11c5c.csv.gz"
)
INSTANCE_SUMMARY = (
    DATASET_ROOT / "hmac_fault_instance_summary_train_11c5c.csv"
)
SCHEMA_PATH = (
    DATASET_ROOT / "hmac_canonical_dataset_schema_11c5c.json"
)
MANIFEST_PATH = (
    QUALIFICATION_ROOT / "hmac_canonical_dataset_manifest_11c5c.json"
)
AUDIT_PATH = (
    QUALIFICATION_ROOT / "hmac_canonical_dataset_schema_freeze_11c5c.json"
)

EXPECTED_SOURCE = {
    SOURCE_MANIFEST:
        "fe5204404cd8f450c5ac7e72c5eda8cbb84099b837ae25c2b0fcb1936b15fc8d",
    SOURCE_AUDIT:
        "d3d346301674d8ea84eee60d72a1377dc53916682ad1bed3461664af6ff09b5c",
    SOURCE_LOG:
        "4f70ba9f6dff1540c09e1ae5b4d3cead736ce353e83b7c1ac488c54509113afc",
    SOURCE_VERIFIER:
        "6193c249e066d9eb9dcf25c238935575a9d466c6037cc58ea338be638301940d",
    ROOT / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json":
        "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    ROOT / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json":
        "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
}

SOURCE_DATASET_COMMITMENT = (
    "95a3743b1a0c06b5777b0014aebf01966b1535ea401b73786dcd8dad553051c2"
)

EXPECTED_BATCHES = 45
EXPECTED_LEGAL_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_TRAIN_VECTORS = 64
EXPECTED_BASELINE_RECORDS = 2_880
EXPECTED_ENABLED_RECORDS = 2_923_392
EXPECTED_TOTAL_RECORDS = 2_926_272
EXPECTED_ACTIVATED_RECORDS = 2_354_285
EXPECTED_DETECTED_RECORDS = 1_278_888
EXPECTED_DETECTED_INSTANCES = 22_930
EXPECTED_ACTIVATED_UNOBSERVED = 15_378
EXPECTED_UNACTIVATED = 7_370
EXPECTED_BASELINE_CYCLES = 343
TIMEOUT_CYCLES = 2_000

RAW_HEADER = [
    "batch_id",
    "run_type",
    "site_selection",
    "selector",
    "site_id",
    "stuck_value",
    "vector_slot",
    "vector_id",
    "fault_enable",
    "cycles",
    "baseline_cycles",
    "timed_out",
    "activity",
    "detected",
    "unknown",
    "expected_digest",
    "actual_digest",
]

CANONICAL_HEADER = [
    "record_id",
    "vector_split",
    "batch_id",
    "run_type",
    "site_selection",
    "selector",
    "site_id",
    "stuck_value",
    "vector_slot",
    "vector_id",
    "fault_enable",
    "cycles",
    "baseline_cycles",
    "latency_delta",
    "timed_out",
    "activity",
    "detected",
    "unknown",
    "digest_hamming_distance",
    "expected_digest",
    "actual_digest",
]

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


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    relative(path)
    return path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not load JSON {path}: {error}")


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_file(path: Path, expected_sha: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(path.stat().st_size > 0, f"empty {label}: {path}")
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {label}: expected {expected_sha}, actual {actual}",
    )


def integer(row: dict[str, str], field: str, location: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        stop(f"invalid {field} at {location}")


def bit(row: dict[str, str], field: str, location: str) -> int:
    value = integer(row, field, location)
    require(value in (0, 1), f"invalid binary {field} at {location}")
    return value


def normalized_digest(value, field: str, vector_id: int) -> str:
    if isinstance(value, int):
        result = f"{value:064x}"
    else:
        result = str(value).strip().lower()
        if result.startswith("0x"):
            result = result[2:]
    require(
        re.fullmatch(r"[0-9a-f]{64}", result) is not None,
        f"invalid {field} for vector {vector_id}",
    )
    return result


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
            and {"fault_site_id", "selector_code"} <= value[0].keys()
        ):
            return value
        for child in value:
            result = find_sites(child)
            if result is not None:
                return result
    return None


def output_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def schema_definition() -> dict:
    fields = [
        ("record_id", "uint64", "identifier", "Canonical zero-based row number"),
        ("vector_split", "enum", "partition", "Frozen vector split; TRAIN in this artifact"),
        ("batch_id", "uint8", "provenance", "Fault batch number, 0 through 44"),
        ("run_type", "enum", "row_type", "BASELINE or ENABLED"),
        ("site_selection", "int16", "identifier", "Local batch site index; -1 for baseline"),
        ("selector", "int16", "identifier", "Instrumented selector; -1 for baseline"),
        ("site_id", "string", "identifier", "Frozen global fault site ID"),
        ("stuck_value", "int8", "fault_attribute", "SA0=0, SA1=1, baseline=-1"),
        ("vector_slot", "uint8", "identifier", "Selected TRAIN schedule position, 0 through 63"),
        ("vector_id", "uint16", "identifier", "Frozen vector-pool ID"),
        ("fault_enable", "bool", "control", "1 for injected records, otherwise 0"),
        ("cycles", "uint16", "measured_observable", "Observed transaction latency"),
        ("baseline_cycles", "uint16", "reference", "Frozen no-fault latency"),
        ("latency_delta", "int16", "derived_diagnostic", "cycles minus baseline_cycles"),
        ("timed_out", "bool", "measured_observable", "Transaction reached timeout limit"),
        ("activity", "bool", "measured_observable", "Event-latched fault activation"),
        ("detected", "bool", "candidate_target", "Fault effect observed by digest, latency, or timeout"),
        ("unknown", "bool", "quality_flag", "Unknown simulation value; required to be zero"),
        ("digest_hamming_distance", "uint16", "derived_diagnostic", "Bit differences between expected and actual digest"),
        ("expected_digest", "hex256", "reference_restricted", "Frozen golden digest; task policy required before training"),
        ("actual_digest", "hex256", "measured_observable", "Observed 256-bit digest"),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "OpenTitan HMAC-SHA256 persistent stuck-at TRAIN campaign",
        "row_grain": "one baseline or one persistent-fault/vector simulation",
        "format": "CSV",
        "compression": {
            "algorithm": "gzip",
            "level": 6,
            "mtime": 0,
            "text_encoding": "UTF-8",
            "line_ending": "LF",
        },
        "canonical_order": [
            "batch_id ascending",
            "source CSV row order",
            "record_id ascending",
        ],
        "fields": [
            {
                "name": name,
                "type": field_type,
                "role": role,
                "nullable": False,
                "description": description,
            }
            for name, field_type, role, description in fields
        ],
        "constraints": {
            "record_id": "unique contiguous range 0..2926271",
            "vector_split": ["TRAIN"],
            "batch_id": "0..44",
            "run_type": ["BASELINE", "ENABLED"],
            "selected_train_vectors": 64,
            "baseline_cycles": EXPECTED_BASELINE_CYCLES,
            "unknown": 0,
            "detected_without_activity": 0,
            "validation_vectors_exposed": 0,
            "holdout_vectors_exposed": 0,
        },
        "model_usage_policy": {
            "training_authorized_by_this_stage": False,
            "reason": "The diagnostic target and allowed feature set are not yet frozen.",
            "baseline_rows": "Retained as provenance and reference evidence; do not mix into fault-instance training without a later task policy.",
            "leakage_warning": "expected_digest, detected, and derived diagnostics require task-specific handling before model training.",
            "split_rule": "Future dataset splits must be group-safe and follow the frozen vector-pool policy.",
            "next_gate": "DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY FREEZE",
        },
    }


def selected_vectors() -> tuple[list[int], dict[int, dict]]:
    selection = load_json(
        ROOT / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json"
    )
    pool = load_json(
        ROOT / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"
    )
    ids = selection.get("selected_vector_ids")
    positions = selection.get("selected_schedule_positions")
    require(isinstance(ids, list) and len(ids) == 64, "selected vector count")
    require(positions == list(range(64)), "selected vector positions")
    records = pool.get("vectors")
    require(isinstance(records, list) and len(records) == 256, "vector pool")
    by_id = {int(item["vector_id"]): item for item in records}
    require(len(by_id) == 256, "duplicate vector IDs")
    selected = {}
    for slot, raw_id in enumerate(ids):
        vector_id = int(raw_id)
        record = by_id.get(vector_id)
        require(record is not None, f"selected vector {vector_id} missing")
        require(record.get("split") == "TRAIN", f"selected vector {vector_id} is not TRAIN")
        selected[slot] = {
            "vector_id": vector_id,
            "expected_digest": normalized_digest(
                record.get("expected_hmac_sha256"),
                "expected_hmac_sha256",
                vector_id,
            ),
        }
    return [int(value) for value in ids], selected


def clean_temporary_outputs() -> None:
    for target in (
        CANONICAL_DATASET,
        INSTANCE_SUMMARY,
        SCHEMA_PATH,
        MANIFEST_PATH,
        AUDIT_PATH,
    ):
        temporary = target.with_name(target.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    final_outputs = (
        CANONICAL_DATASET,
        INSTANCE_SUMMARY,
        SCHEMA_PATH,
        MANIFEST_PATH,
        AUDIT_PATH,
    )
    for path in final_outputs:
        require(not path.exists(), f"Stage 11C-5C output already exists: {path}")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    QUALIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
    clean_temporary_outputs()

    print("STAGE 11C-5C — CANONICAL DATASET CONSOLIDATION")
    print("FROZEN INPUT VERIFICATION")
    source_evidence = {}
    for path, expected in EXPECTED_SOURCE.items():
        verify_file(path, expected, relative(path))
        source_evidence[relative(path)] = {
            "sha256": expected,
            "bytes": path.stat().st_size,
        }
        print(f"  {path.name:<66} : OK", flush=True)

    source_manifest = load_json(SOURCE_MANIFEST)
    source_audit = load_json(SOURCE_AUDIT)
    require(source_manifest.get("status") == "PASS", "Stage 11C-5B manifest status")
    require(source_audit.get("status") == "PASS", "Stage 11C-5B audit status")
    require(source_audit.get("dataset_status") == "FROZEN", "Stage 11C-5B dataset status")
    require(source_manifest.get("batch_count") == EXPECTED_BATCHES, "source batch count")
    require(source_manifest.get("legal_sites") == EXPECTED_LEGAL_SITES, "source site count")
    require(
        source_manifest.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES,
        "source fault-instance count",
    )
    require(source_manifest.get("total_csv_records") == EXPECTED_TOTAL_RECORDS, "source record count")
    require(
        source_manifest.get("ordered_batch_sha256_commitment")
        == SOURCE_DATASET_COMMITMENT,
        "source dataset commitment",
    )
    require(source_manifest.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(source_manifest.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")
    source_batches = source_manifest.get("batches")
    require(isinstance(source_batches, list) and len(source_batches) == 45, "source batch records")
    require(
        [int(item["batch_id"]) for item in source_batches] == list(range(45)),
        "source batch order",
    )

    selected_ids, selected = selected_vectors()
    totals = Counter()
    global_site_ids = set()
    batch_outputs = []
    instances = []
    record_id = 0

    canonical_temporary = CANONICAL_DATASET.with_name(CANONICAL_DATASET.name + ".tmp")
    print("\nCANONICAL ROW CONSOLIDATION", flush=True)
    with canonical_temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=raw_output,
            mtime=0,
        ) as gzip_output:
            with io.TextIOWrapper(
                gzip_output,
                encoding="utf-8",
                newline="",
            ) as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                writer.writerow(CANONICAL_HEADER)

                for expected_batch_id, batch in enumerate(source_batches):
                    batch_id = int(batch["batch_id"])
                    key = f"{batch_id:03d}"
                    require(batch_id == expected_batch_id, f"Batch {key} order")
                    expected_sites = 311 if batch_id == 44 else 512
                    expected_enabled = expected_sites * 128

                    csv_record = batch.get("csv")
                    mapping_record = batch.get("mapping")
                    require(isinstance(csv_record, dict), f"Batch {key} CSV evidence")
                    require(isinstance(mapping_record, dict), f"Batch {key} mapping evidence")
                    csv_path = project_path(csv_record.get("path", ""))
                    mapping_path = project_path(mapping_record.get("path", ""))
                    verify_file(csv_path, csv_record.get("sha256", ""), f"Batch {key} raw CSV")
                    verify_file(mapping_path, mapping_record.get("sha256", ""), f"Batch {key} mapping")
                    sites = find_sites(load_json(mapping_path))
                    require(isinstance(sites, list) and len(sites) == expected_sites, f"Batch {key} sites")
                    require(
                        [int(item["selector_code"]) for item in sites]
                        == list(range(expected_sites)),
                        f"Batch {key} selectors",
                    )

                    site_ids = [str(item["fault_site_id"]) for item in sites]
                    for local_index, site in enumerate(sites):
                        site_id = site_ids[local_index]
                        require(site_id not in global_site_ids, f"duplicate site {site_id}")
                        global_site_ids.add(site_id)
                        match = re.fullmatch(r"HMAC-STEM-(\d{6})", site_id)
                        require(match is not None, f"invalid site ID {site_id}")
                        parsed_site_index = int(match.group(1))
                        site_index = int(site.get("site_index", parsed_site_index))
                        require(site_index == parsed_site_index, f"site-index mismatch for {site_id}")
                        for stuck_value in (0, 1):
                            instances.append(
                                {
                                    "fault_instance_id": f"{site_id}-SA{stuck_value}",
                                    "site_id": site_id,
                                    "site_index": site_index,
                                    "batch_id": batch_id,
                                    "selector": int(site["selector_code"]),
                                    "stuck_value": stuck_value,
                                    "site_category": str(site.get("site_category", "UNKNOWN")),
                                    "driver_cell_type": str(site.get("driver_cell_type", "UNKNOWN")),
                                    "train_vectors": 0,
                                    "activated_vectors": 0,
                                    "detected_vectors": 0,
                                    "timeout_vectors": 0,
                                }
                            )
                    instance_offset = len(instances) - expected_sites * 2

                    batch_counts = Counter()
                    with csv_path.open(newline="") as input_stream:
                        reader = csv.DictReader(input_stream)
                        require(reader.fieldnames == RAW_HEADER, f"Batch {key} raw schema")
                        for line_number, row in enumerate(reader, start=2):
                            location = f"Batch {key} line {line_number}"
                            require(integer(row, "batch_id", location) == batch_id, f"{location} batch")
                            run_type = row["run_type"]
                            vector_slot = integer(row, "vector_slot", location)
                            require(vector_slot in selected, f"{location} vector slot")
                            vector_id = integer(row, "vector_id", location)
                            require(
                                vector_id == selected[vector_slot]["vector_id"],
                                f"{location} vector ID",
                            )
                            expected_digest = row["expected_digest"].strip().lower()
                            actual_digest = row["actual_digest"].strip().lower()
                            require(
                                expected_digest == selected[vector_slot]["expected_digest"],
                                f"{location} expected digest",
                            )
                            require(
                                re.fullmatch(r"[0-9a-f]{64}", actual_digest) is not None,
                                f"{location} actual digest",
                            )
                            cycles = integer(row, "cycles", location)
                            baseline_cycles = integer(row, "baseline_cycles", location)
                            timed_out = bit(row, "timed_out", location)
                            activity = bit(row, "activity", location)
                            detected = bit(row, "detected", location)
                            unknown = bit(row, "unknown", location)
                            fault_enable = bit(row, "fault_enable", location)
                            require(baseline_cycles == EXPECTED_BASELINE_CYCLES, f"{location} baseline cycles")
                            require(0 <= cycles <= TIMEOUT_CYCLES, f"{location} cycles")
                            require(unknown == 0, f"{location} unknown value")
                            require(not (detected and not activity), f"{location} detected without activity")
                            latency_delta = cycles - baseline_cycles
                            digest_distance = (
                                int(expected_digest, 16) ^ int(actual_digest, 16)
                            ).bit_count()

                            if run_type == "BASELINE":
                                batch_counts["baseline_records"] += 1
                                require(fault_enable == 0, f"{location} baseline fault enable")
                                require(cycles == EXPECTED_BASELINE_CYCLES, f"{location} baseline latency")
                                require(timed_out == activity == detected == 0, f"{location} baseline flags")
                                require(digest_distance == 0, f"{location} baseline digest")
                            elif run_type == "ENABLED":
                                batch_counts["enabled_records"] += 1
                                site_selection = integer(row, "site_selection", location)
                                selector = integer(row, "selector", location)
                                stuck_value = bit(row, "stuck_value", location)
                                require(0 <= site_selection < expected_sites, f"{location} site")
                                require(selector == site_selection, f"{location} selector")
                                require(row["site_id"] == site_ids[site_selection], f"{location} site ID")
                                require(fault_enable == 1, f"{location} enabled flag")
                                computed_detected = int(
                                    timed_out
                                    or digest_distance != 0
                                    or latency_delta != 0
                                )
                                require(detected == computed_detected, f"{location} detected label")
                                instance_index = instance_offset + site_selection * 2 + stuck_value
                                instance = instances[instance_index]
                                instance["train_vectors"] += 1
                                instance["activated_vectors"] += activity
                                instance["detected_vectors"] += detected
                                instance["timeout_vectors"] += timed_out
                                batch_counts["activated_records"] += activity
                                batch_counts["detected_records"] += detected
                                batch_counts["timeout_records"] += timed_out
                            else:
                                stop(f"{location} unsupported run type {run_type!r}")

                            writer.writerow(
                                [
                                    record_id,
                                    "TRAIN",
                                    batch_id,
                                    run_type,
                                    row["site_selection"],
                                    row["selector"],
                                    row["site_id"],
                                    row["stuck_value"],
                                    vector_slot,
                                    vector_id,
                                    fault_enable,
                                    cycles,
                                    baseline_cycles,
                                    latency_delta,
                                    timed_out,
                                    activity,
                                    detected,
                                    unknown,
                                    digest_distance,
                                    expected_digest,
                                    actual_digest,
                                ]
                            )
                            record_id += 1

                    require(batch_counts["baseline_records"] == 64, f"Batch {key} baseline count")
                    require(batch_counts["enabled_records"] == expected_enabled, f"Batch {key} enabled count")
                    batch_counts["total_records"] = (
                        batch_counts["baseline_records"]
                        + batch_counts["enabled_records"]
                    )
                    for instance in instances[instance_offset:]:
                        require(instance["train_vectors"] == 64, f"{instance['fault_instance_id']} vector coverage")
                    totals.update(batch_counts)
                    batch_outputs.append(
                        {
                            "batch_id": batch_id,
                            "raw_csv": {
                                "path": relative(csv_path),
                                "sha256": csv_record["sha256"],
                                "bytes": csv_path.stat().st_size,
                            },
                            "legal_sites": expected_sites,
                            "fault_instances": expected_sites * 2,
                            "baseline_records": batch_counts["baseline_records"],
                            "enabled_records": batch_counts["enabled_records"],
                            "total_records": batch_counts["total_records"],
                            "activated_records": batch_counts["activated_records"],
                            "detected_records": batch_counts["detected_records"],
                            "timeout_records": batch_counts["timeout_records"],
                        }
                    )
                    print(
                        f"  Batch {key}: PASS records={batch_counts['total_records']:6d} "
                        f"canonical_rows={record_id:7d}",
                        flush=True,
                    )

    require(record_id == EXPECTED_TOTAL_RECORDS, "canonical record count")
    expected_site_ids = {f"HMAC-STEM-{index:06d}" for index in range(1, 22_840)}
    require(global_site_ids == expected_site_ids, "global site coverage")
    require(len(instances) == EXPECTED_FAULT_INSTANCES, "fault-instance count")
    require(totals["baseline_records"] == EXPECTED_BASELINE_RECORDS, "baseline total")
    require(totals["enabled_records"] == EXPECTED_ENABLED_RECORDS, "enabled total")
    require(totals["total_records"] == EXPECTED_TOTAL_RECORDS, "record total")
    require(totals["activated_records"] == EXPECTED_ACTIVATED_RECORDS, "activation total")
    require(totals["detected_records"] == EXPECTED_DETECTED_RECORDS, "detection total")

    class_counts = Counter()
    for instance in instances:
        if instance["detected_vectors"]:
            classification = "DETECTED"
        elif instance["activated_vectors"]:
            classification = "ACTIVATED_UNOBSERVED"
        else:
            classification = "UNACTIVATED"
        instance["final_classification"] = classification
        class_counts[classification] += 1

    require(class_counts["DETECTED"] == EXPECTED_DETECTED_INSTANCES, "detected instances")
    require(
        class_counts["ACTIVATED_UNOBSERVED"] == EXPECTED_ACTIVATED_UNOBSERVED,
        "activated-unobserved instances",
    )
    require(class_counts["UNACTIVATED"] == EXPECTED_UNACTIVATED, "unactivated instances")
    require(sum(class_counts.values()) == EXPECTED_FAULT_INSTANCES, "classification closure")

    instance_temporary = INSTANCE_SUMMARY.with_name(INSTANCE_SUMMARY.name + ".tmp")
    with instance_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INSTANCE_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(instances)

    schema_temporary = SCHEMA_PATH.with_name(SCHEMA_PATH.name + ".tmp")
    schema_temporary.write_text(json.dumps(schema_definition(), indent=2, sort_keys=True) + "\n")

    canonical_temporary.replace(CANONICAL_DATASET)
    instance_temporary.replace(INSTANCE_SUMMARY)
    schema_temporary.replace(SCHEMA_PATH)

    canonical_record = output_record(CANONICAL_DATASET)
    instance_record = output_record(INSTANCE_SUMMARY)
    schema_record = output_record(SCHEMA_PATH)
    created_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "stage": "11C-5C",
        "title": "CANONICAL DATASET CONSOLIDATION AND SCHEMA FREEZE",
        "status": "PASS",
        "consolidator_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at,
        "source_stage": "11C-5B",
        "source_dataset_commitment": SOURCE_DATASET_COMMITMENT,
        "source_evidence": source_evidence,
        "canonical_dataset": canonical_record,
        "fault_instance_summary": instance_record,
        "schema": schema_record,
        "batch_count": EXPECTED_BATCHES,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "selected_train_vectors": EXPECTED_TRAIN_VECTORS,
        "selected_vector_ids": selected_ids,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "baseline_records": totals["baseline_records"],
        "enabled_records": totals["enabled_records"],
        "total_records": totals["total_records"],
        "activated_records": totals["activated_records"],
        "detected_records": totals["detected_records"],
        "timeout_records": totals["timeout_records"],
        "instance_classification": {
            "DETECTED": class_counts["DETECTED"],
            "ACTIVATED_UNOBSERVED": class_counts["ACTIVATED_UNOBSERVED"],
            "UNACTIVATED": class_counts["UNACTIVATED"],
            "UNCLASSIFIED": 0,
        },
        "canonical_order": "batch_id ascending, then frozen raw CSV row order",
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "batches": batch_outputs,
        "model_training_authorized": False,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "raw_campaign_dataset_modified": False,
    }
    write_json(MANIFEST_PATH, manifest)
    manifest_record = output_record(MANIFEST_PATH)

    audit = {
        "stage": "11C-5C",
        "title": "CANONICAL DATASET CONSOLIDATION AND SCHEMA FREEZE",
        "status": "PASS",
        "created_at_utc": created_at,
        "canonical_dataset_status": "FROZEN",
        "schema_status": "FROZEN",
        "canonical_dataset": canonical_record,
        "fault_instance_summary": instance_record,
        "schema": schema_record,
        "manifest": manifest_record,
        "source_dataset_commitment": SOURCE_DATASET_COMMITMENT,
        "batch_count": EXPECTED_BATCHES,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "selected_train_vectors": EXPECTED_TRAIN_VECTORS,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "baseline_records": totals["baseline_records"],
        "enabled_records": totals["enabled_records"],
        "total_records": totals["total_records"],
        "unknown_records": 0,
        "detected_without_activity": 0,
        "unclassified_instances": 0,
        "model_training_authorized": False,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "raw_campaign_dataset_modified": False,
        "next_gate": "DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY FREEZE",
    }
    write_json(AUDIT_PATH, audit)
    audit_record = output_record(AUDIT_PATH)

    print("\nSTAGE 11C-5C — CANONICAL DATASET CONSOLIDATION AND SCHEMA FREEZE")
    print("Status                       : PASS")
    print("Canonical dataset status     : FROZEN")
    print("Schema status                : FROZEN")
    print("Batches consolidated         : 45/45")
    print("Canonical rows               :", totals["total_records"])
    print("Baseline rows                :", totals["baseline_records"])
    print("Enabled rows                 :", totals["enabled_records"])
    print("Legal sites                  :", EXPECTED_LEGAL_SITES)
    print("Fault instances              :", EXPECTED_FAULT_INSTANCES)
    print("TRAIN vectors                :", EXPECTED_TRAIN_VECTORS)
    print("VALIDATION vectors exposed   : 0")
    print("HOLDOUT vectors exposed      : 0")
    print("Detected instances           :", class_counts["DETECTED"])
    print("Activated-unobserved         :", class_counts["ACTIVATED_UNOBSERVED"])
    print("Unactivated                  :", class_counts["UNACTIVATED"])
    print("Unclassified                 : 0")
    print("Model training authorized    : NO")
    print("Frozen RTL modified          : NO")
    print("Golden netlist modified      : NO")
    print("Raw campaign dataset modified: NO")
    print("Canonical dataset            :", CANONICAL_DATASET)
    print("Canonical dataset SHA        :", canonical_record["sha256"])
    print("Canonical dataset bytes      :", canonical_record["bytes"])
    print("Fault-instance summary       :", INSTANCE_SUMMARY)
    print("Fault-instance summary SHA   :", instance_record["sha256"])
    print("Schema                       :", SCHEMA_PATH)
    print("Schema SHA                   :", schema_record["sha256"])
    print("Manifest                     :", MANIFEST_PATH)
    print("Manifest SHA                 :", manifest_record["sha256"])
    print("Audit                        :", AUDIT_PATH)
    print("Audit SHA                    :", audit_record["sha256"])
    print("Next gate                    : DIAGNOSTIC TASK, FEATURE, AND SPLIT POLICY FREEZE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop("interrupted by user; no freeze was completed")
