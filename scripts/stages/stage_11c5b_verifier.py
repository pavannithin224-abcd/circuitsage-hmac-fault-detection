#!/usr/bin/env python3
"""Freeze integrity evidence for the completed OpenTitan HMAC fault campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd().resolve()
RAW_ROOT = ROOT / "results/hmac_fault_campaign_full_11c5"
FREEZE_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
CHECKPOINT = RAW_ROOT / "hmac_full_campaign_checkpoint.json"
MASTER_LOG = (
    ROOT
    / "results/hmac_fault_campaign_11c4/"
      "stage_11c5a_full_campaign_master.log"
)
RESOURCE_LOG = (
    ROOT
    / "results/hmac_fault_campaign_11c4/"
      "stage_11c5a_full_campaign_resources.log"
)
MANIFEST = FREEZE_ROOT / "hmac_full_campaign_dataset_manifest_11c5b.json"
AUDIT = FREEZE_ROOT / "hmac_full_campaign_dataset_integrity_freeze_11c5b.json"

EXPECTED_CHECKPOINT_SHA = (
    "47aefecf3d7babbd6e618309e6713be27"
    "a62aebea84be5e72e40f5f7576b07d3"
)

EXPECTED_INPUTS = {
    "config/fault_campaign/hmac_full_campaign_execution_authorization_11c4g_b3.json":
        "b8202b258ff458548e9a2fe7aa991589b749a91882f708823bb8a0bf566ea9cf",
    "results/hmac_fault_campaign_11c4/hmac_full_campaign_execution_authorization_freeze_11c4g_b3.json":
        "7bde7837d08ff37bf453bfa46d05cbf1c9a2d05c5e9583cb5ec4ac3b86ef9554",
    "config/fault_campaign/hmac_full_campaign_execution_policy_11c4g_b1.json":
        "7e43e4b2e4c8c798ca708cbaa7f995f64bdccf4e27ad35478c212c39bce61ad6",
    "results/hmac_fault_campaign_11c4/hmac_full_campaign_execution_policy_freeze_11c4g_b1.json":
        "84ab1ed81f2092ded590c79bacffa1165bb1745e1aa071c878824ed8f111931a",
    "results/hmac_fault_campaign_11c4/hmac_full_campaign_runner_source_freeze_11c4g_b2b.json":
        "71d0a03f3ae813d64caf365210d2143957c6973ae1c27332c47fd3aec97591e9",
    "scripts/hmac/generate_hmac_full_campaign_testbench.py":
        "faae8e8d7bd5096840b103fa0fad6e893e0712b7f41747b2679494558dcb8094",
    "scripts/hmac/run_hmac_full_fault_campaign.py":
        "9bfcc8d98dbc17611966b82ebf0cbb0bb26560bb4f0c5afa1d8df9867a8c21a2",
    "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json":
        "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json":
        "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
    "results/hmac_fault_campaign_11c4/hmac_canonical_batch_generation_index_11c4g_a1.json":
        "7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
    "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json":
        "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
    "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json":
        "0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1",
}

CSV_HEADER = [
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

EXPECTED_BATCHES = list(range(45))
EXPECTED_LEGAL_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_TRAIN_VECTORS = 64
EXPECTED_BASELINE_RECORDS = 2_880
EXPECTED_ENABLED_RECORDS = 2_923_392
EXPECTED_TOTAL_RECORDS = 2_926_272
EXPECTED_BASELINE_CYCLES = 343
TIMEOUT_CYCLES = 2_000


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


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not load JSON {path}: {error}")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        stop(f"artifact is outside project root: {path}")
    return path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        stop(f"cannot make project-relative path: {path}")


def normalized_digest(value, field: str, vector_id: int) -> str:
    if isinstance(value, int):
        result = f"{value:064x}"
    else:
        result = str(value).strip().lower()
        if result.startswith("0x"):
            result = result[2:]
    require(
        re.fullmatch(r"[0-9a-f]{64}", result) is not None,
        f"vector {vector_id} has invalid {field}",
    )
    return result


def integer(row: dict[str, str], field: str, location: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        stop(f"invalid integer field {field} at {location}")


def bit(row: dict[str, str], field: str, location: str) -> int:
    result = integer(row, field, location)
    require(result in (0, 1), f"invalid bit field {field} at {location}")
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


def verify_file(path: Path, expected_sha: str, label: str) -> dict:
    require(path.is_file(), f"missing {label}: {path}")
    require(path.stat().st_size > 0, f"empty {label}: {path}")
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {label}: expected {expected_sha}, actual {actual}",
    )
    return {
        "path": relative(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def verify_frozen_inputs() -> dict:
    evidence = {}
    print("FROZEN INPUT VERIFICATION")
    for name, expected in EXPECTED_INPUTS.items():
        path = ROOT / name
        evidence[name] = verify_file(path, expected, name)
        print(f"  {Path(name).name:<66} : OK")
    evidence[relative(CHECKPOINT)] = verify_file(
        CHECKPOINT,
        EXPECTED_CHECKPOINT_SHA,
        "completed campaign checkpoint",
    )
    print(f"  {CHECKPOINT.name:<66} : OK")
    require(MASTER_LOG.is_file(), f"missing campaign master log: {MASTER_LOG}")
    master_text = MASTER_LOG.read_text(errors="replace")
    for token in (
        "Status               : PASS",
        "Mode                 : FULL",
        "Completed batches    : 45",
        "FULL_CAMPAIGN_RUNNER_RESULT=PASS",
    ):
        require(token in master_text, f"master-log token missing: {token}")
    evidence[relative(MASTER_LOG)] = {
        "path": relative(MASTER_LOG),
        "sha256": sha256(MASTER_LOG),
        "bytes": MASTER_LOG.stat().st_size,
    }
    if RESOURCE_LOG.is_file() and RESOURCE_LOG.stat().st_size:
        resource_text = RESOURCE_LOG.read_text(errors="replace")
        require("Exit status: 0" in resource_text, "campaign resource log is not closed")
        evidence[relative(RESOURCE_LOG)] = {
            "path": relative(RESOURCE_LOG),
            "sha256": sha256(RESOURCE_LOG),
            "bytes": RESOURCE_LOG.stat().st_size,
        }
    print(f"  {MASTER_LOG.name:<66} : PASS CLOSURE")
    return evidence


def selected_vectors() -> tuple[list[int], dict[int, dict]]:
    selection = load_json(
        ROOT / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json"
    )
    pool = load_json(
        ROOT / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"
    )
    vector_ids = selection.get("selected_vector_ids")
    positions = selection.get("selected_schedule_positions")
    require(
        isinstance(vector_ids, list) and len(vector_ids) == EXPECTED_TRAIN_VECTORS,
        "selected vector-ID contract mismatch",
    )
    require(positions == list(range(64)), "selected vector-position contract mismatch")
    records = pool.get("vectors")
    require(isinstance(records, list) and len(records) == 256, "vector-pool contract mismatch")
    by_id = {int(record["vector_id"]): record for record in records}
    require(len(by_id) == 256, "duplicate vector IDs in vector pool")
    selected = {}
    for slot, value in enumerate(vector_ids):
        vector_id = int(value)
        record = by_id.get(vector_id)
        require(record is not None, f"selected vector {vector_id} is missing")
        require(record.get("split") == "TRAIN", f"non-TRAIN vector selected: {vector_id}")
        selected[slot] = {
            "vector_id": vector_id,
            "expected_digest": normalized_digest(
                record.get("expected_hmac_sha256"),
                "expected_hmac_sha256",
                vector_id,
            ),
        }
    return [int(value) for value in vector_ids], selected


def verify_record_artifacts(record: dict, batch_key: str) -> dict:
    artifacts = {}
    for field in (
        "netlist",
        "mapping",
        "testbench",
        "manifest",
        "binary",
        "csv",
        "simulation_log",
        "resource_log",
    ):
        path_value = record.get(field)
        expected = record.get(field + "_sha256")
        require(isinstance(path_value, str) and path_value, f"Batch {batch_key} missing {field}")
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"Batch {batch_key} has invalid {field}_sha256",
        )
        path = project_path(path_value)
        artifacts[field] = verify_file(path, expected, f"Batch {batch_key} {field}")
    return artifacts


def verify_testbench_manifest(
    path: Path,
    batch_id: int,
    sites: list[dict],
    selected_ids: list[int],
) -> None:
    value = load_json(path)
    expected_sites = len(sites)
    require(value.get("status") == "GENERATED", f"Batch {batch_id:03d} manifest status")
    require(value.get("mode") == "full", f"Batch {batch_id:03d} manifest mode")
    require(int(value.get("batch_id", -1)) == batch_id, f"Batch {batch_id:03d} manifest ID")
    require(value.get("vector_positions") == list(range(64)), f"Batch {batch_id:03d} positions")
    require(
        [int(item) for item in value.get("vector_ids", [])] == selected_ids,
        f"Batch {batch_id:03d} vector IDs",
    )
    require(
        value.get("selectors") == list(range(expected_sites)),
        f"Batch {batch_id:03d} manifest selectors",
    )
    require(
        value.get("site_ids") == [item["fault_site_id"] for item in sites],
        f"Batch {batch_id:03d} manifest site IDs",
    )
    require(value.get("baseline_records") == 64, f"Batch {batch_id:03d} baseline count")
    require(
        value.get("enabled_records") == expected_sites * 128,
        f"Batch {batch_id:03d} enabled count",
    )
    require(
        value.get("total_records") == 64 + expected_sites * 128,
        f"Batch {batch_id:03d} total count",
    )
    require(value.get("validation_vectors_exposed") == 0, "VALIDATION vectors exposed")
    require(value.get("holdout_vectors_exposed") == 0, "HOLDOUT vectors exposed")


def verify_csv(
    path: Path,
    batch_id: int,
    sites: list[dict],
    selected: dict[int, dict],
) -> dict:
    key = f"{batch_id:03d}"
    expected_site_count = 311 if batch_id == 44 else 512
    require(len(sites) == expected_site_count, f"Batch {key} mapping site count")
    selectors = [int(item["selector_code"]) for item in sites]
    require(selectors == list(range(expected_site_count)), f"Batch {key} selector sequence")

    expected_site_ids = [str(item["fault_site_id"]) for item in sites]
    require(len(set(expected_site_ids)) == expected_site_count, f"Batch {key} duplicate sites")

    baseline_records = 0
    enabled_records = 0
    activated_records = 0
    detected_records = 0
    timeout_records = 0
    unknown_records = 0
    detected_without_activity = 0
    baseline_slots = set()
    enabled_cases = set()
    instance_activity = [0] * (expected_site_count * 2)
    instance_detection = [0] * (expected_site_count * 2)
    vector_enabled = Counter()

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CSV_HEADER, f"Batch {key} CSV header mismatch")
        for line_number, row in enumerate(reader, start=2):
            location = f"Batch {key} CSV line {line_number}"
            require(integer(row, "batch_id", location) == batch_id, f"{location} batch mismatch")
            run_type = row["run_type"]
            vector_slot = integer(row, "vector_slot", location)
            require(vector_slot in selected, f"{location} invalid vector slot")
            vector_id = integer(row, "vector_id", location)
            expected_vector = selected[vector_slot]
            require(vector_id == expected_vector["vector_id"], f"{location} vector-ID mismatch")

            cycles = integer(row, "cycles", location)
            baseline_cycles = integer(row, "baseline_cycles", location)
            timed_out = bit(row, "timed_out", location)
            activity = bit(row, "activity", location)
            detected = bit(row, "detected", location)
            unknown = bit(row, "unknown", location)
            expected_digest = row["expected_digest"].strip().lower()
            actual_digest = row["actual_digest"].strip().lower()

            require(baseline_cycles == EXPECTED_BASELINE_CYCLES, f"{location} baseline latency")
            require(0 <= cycles <= TIMEOUT_CYCLES, f"{location} cycle count")
            require(
                re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None,
                f"{location} expected digest format",
            )
            require(
                re.fullmatch(r"[0-9a-f]{64}", actual_digest) is not None,
                f"{location} actual digest format",
            )
            require(
                expected_digest == expected_vector["expected_digest"],
                f"{location} expected digest differs from frozen vector pool",
            )
            require(unknown == 0, f"{location} contains unknown output")
            unknown_records += unknown

            if run_type == "BASELINE":
                baseline_records += 1
                require(vector_slot not in baseline_slots, f"{location} duplicate baseline vector")
                baseline_slots.add(vector_slot)
                require(row["site_selection"] == "-1", f"{location} baseline site selection")
                require(row["selector"] == "-1", f"{location} baseline selector")
                require(row["site_id"] == "BASELINE", f"{location} baseline site ID")
                require(row["stuck_value"] == "-1", f"{location} baseline stuck value")
                require(row["fault_enable"] == "0", f"{location} baseline fault enable")
                require(cycles == EXPECTED_BASELINE_CYCLES, f"{location} baseline cycles")
                require(timed_out == 0, f"{location} baseline timeout")
                require(activity == 0 and detected == 0, f"{location} baseline flags")
                require(actual_digest == expected_digest, f"{location} baseline digest mismatch")
            elif run_type == "ENABLED":
                enabled_records += 1
                site_selection = integer(row, "site_selection", location)
                selector = integer(row, "selector", location)
                stuck_value = bit(row, "stuck_value", location)
                fault_enable = bit(row, "fault_enable", location)
                require(0 <= site_selection < expected_site_count, f"{location} site selection")
                require(selector == selectors[site_selection], f"{location} selector mismatch")
                require(row["site_id"] == expected_site_ids[site_selection], f"{location} site ID")
                require(fault_enable == 1, f"{location} enabled fault is disabled")
                case = (site_selection, stuck_value, vector_slot)
                require(case not in enabled_cases, f"{location} duplicate enabled case")
                enabled_cases.add(case)
                if timed_out:
                    require(cycles == TIMEOUT_CYCLES, f"{location} timeout-cycle mismatch")
                computed_detected = int(
                    timed_out == 1
                    or actual_digest != expected_digest
                    or cycles != baseline_cycles
                )
                require(detected == computed_detected, f"{location} detection classification")
                activated_records += activity
                detected_records += detected
                timeout_records += timed_out
                if detected and not activity:
                    detected_without_activity += 1
                instance = site_selection * 2 + stuck_value
                instance_activity[instance] += activity
                instance_detection[instance] += detected
                vector_enabled[vector_slot] += 1
            else:
                stop(f"{location} has unsupported run_type {run_type!r}")

    expected_enabled = expected_site_count * 2 * 64
    expected_cases = {
        (site, stuck, slot)
        for site in range(expected_site_count)
        for stuck in (0, 1)
        for slot in range(64)
    }
    require(baseline_records == 64, f"Batch {key} baseline record count")
    require(baseline_slots == set(range(64)), f"Batch {key} baseline vector coverage")
    require(enabled_records == expected_enabled, f"Batch {key} enabled record count")
    require(enabled_cases == expected_cases, f"Batch {key} Cartesian-case coverage")
    require(unknown_records == 0, f"Batch {key} unknown records")
    require(detected_without_activity == 0, f"Batch {key} detected without activity")
    require(
        vector_enabled == Counter({slot: expected_site_count * 2 for slot in range(64)}),
        f"Batch {key} enabled vector coverage",
    )

    classifications = Counter()
    for activity_count, detection_count in zip(instance_activity, instance_detection):
        if detection_count:
            classifications["DETECTED"] += 1
        elif activity_count:
            classifications["ACTIVATED_UNOBSERVED"] += 1
        else:
            classifications["UNACTIVATED"] += 1
    classified = sum(classifications.values())
    require(classified == expected_site_count * 2, f"Batch {key} unclassified fault")

    return {
        "baseline_records": baseline_records,
        "enabled_records": enabled_records,
        "total_records": baseline_records + enabled_records,
        "activated_records": activated_records,
        "detected_records": detected_records,
        "timeout_records": timeout_records,
        "unknown_records": unknown_records,
        "detected_without_activity": detected_without_activity,
        "fault_instances": expected_site_count * 2,
        "detected_instances": classifications["DETECTED"],
        "activated_unobserved_instances": classifications["ACTIVATED_UNOBSERVED"],
        "unactivated_instances": classifications["UNACTIVATED"],
        "unclassified_instances": 0,
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(not MANIFEST.exists(), f"freeze manifest already exists: {MANIFEST}")
    require(not AUDIT.exists(), f"freeze audit already exists: {AUDIT}")

    print("STAGE 11C-5B — FULL-CAMPAIGN DATASET INTEGRITY")
    frozen_evidence = verify_frozen_inputs()
    selected_ids, selected = selected_vectors()
    checkpoint = load_json(CHECKPOINT)

    require(checkpoint.get("runner_version") == "HMAC-FULL-FAULT-CAMPAIGN-RUNNER-v1", "runner version")
    require(checkpoint.get("mode") == "full", "checkpoint mode")
    require(checkpoint.get("status") == "PASS", "checkpoint status")
    require(checkpoint.get("completed_batches") == EXPECTED_BATCHES, "completed-batch sequence")
    require(
        checkpoint.get("policy_sha256")
        == "7e43e4b2e4c8c798ca708cbaa7f995f64bdccf4e27ad35478c212c39bce61ad6",
        "checkpoint policy SHA",
    )
    require(
        checkpoint.get("generator_sha256")
        == "faae8e8d7bd5096840b103fa0fad6e893e0712b7f41747b2679494558dcb8094",
        "checkpoint testbench-generator SHA",
    )
    require(
        checkpoint.get("runner_sha256")
        == "9bfcc8d98dbc17611966b82ebf0cbb0bb26560bb4f0c5afa1d8df9867a8c21a2",
        "checkpoint runner SHA",
    )
    require(
        checkpoint.get("authorization_sha256")
        == "b8202b258ff458548e9a2fe7aa991589b749a91882f708823bb8a0bf566ea9cf",
        "checkpoint authorization SHA",
    )

    batch_records = checkpoint.get("batches")
    require(isinstance(batch_records, dict), "checkpoint batches are missing")
    require(set(batch_records) == {f"{value:03d}" for value in EXPECTED_BATCHES}, "checkpoint batch keys")

    totals = Counter()
    batch_manifest = []
    global_site_ids = set()
    ordered_commitment = hashlib.sha256()

    print("\nBATCH DATASET VERIFICATION")
    for batch_id in EXPECTED_BATCHES:
        key = f"{batch_id:03d}"
        record = batch_records[key]
        expected_sites = 311 if batch_id == 44 else 512
        expected_enabled = expected_sites * 128
        expected_total = 64 + expected_enabled

        require(record.get("status") == "PASS", f"Batch {key} checkpoint status")
        require(int(record.get("batch_id", -1)) == batch_id, f"Batch {key} checkpoint ID")
        artifacts = verify_record_artifacts(record, key)
        summary = record.get("csv_summary")
        require(isinstance(summary, dict), f"Batch {key} checkpoint CSV summary")
        require(summary.get("baseline_records") == 64, f"Batch {key} checkpoint baseline")
        require(summary.get("enabled_records") == expected_enabled, f"Batch {key} checkpoint enabled")
        require(summary.get("total_records") == expected_total, f"Batch {key} checkpoint records")
        require(summary.get("unknown_records") == 0, f"Batch {key} checkpoint unknown")
        require(summary.get("detected_without_activity") == 0, f"Batch {key} checkpoint activity")
        require(summary.get("selectors") == list(range(expected_sites)), f"Batch {key} checkpoint selectors")
        require(summary.get("vector_slots") == list(range(64)), f"Batch {key} checkpoint vectors")

        mapping_path = project_path(record["mapping"])
        sites = find_sites(load_json(mapping_path))
        require(isinstance(sites, list), f"Batch {key} site mapping not found")
        require(len(sites) == expected_sites, f"Batch {key} mapping count")
        for site in sites:
            site_id = str(site["fault_site_id"])
            require(site_id not in global_site_ids, f"duplicate campaign site {site_id}")
            global_site_ids.add(site_id)

        testbench_manifest_path = project_path(record["manifest"])
        verify_testbench_manifest(testbench_manifest_path, batch_id, sites, selected_ids)
        simulation_text = project_path(record["simulation_log"]).read_text(errors="replace")
        require(
            "FULL_CAMPAIGN_BATCH_RESULT=PASS" in simulation_text,
            f"Batch {key} simulation PASS token",
        )
        resource_text = project_path(record["resource_log"]).read_text(errors="replace")
        require("Exit status: 0" in resource_text, f"Batch {key} resource exit status")

        csv_path = project_path(record["csv"])
        csv_stats = verify_csv(csv_path, batch_id, sites, selected)
        for field, value in csv_stats.items():
            if isinstance(value, int):
                totals[field] += value

        csv_sha = artifacts["csv"]["sha256"]
        ordered_commitment.update(f"{key}:{csv_sha}\n".encode())
        batch_manifest.append(
            {
                "batch_id": batch_id,
                "batch_key": key,
                "first_site_id": sites[0]["fault_site_id"],
                "last_site_id": sites[-1]["fault_site_id"],
                "legal_sites": expected_sites,
                "csv": artifacts["csv"],
                "mapping": artifacts["mapping"],
                "testbench_manifest": artifacts["manifest"],
                "simulation_log": artifacts["simulation_log"],
                "resource_log": artifacts["resource_log"],
                "statistics": csv_stats,
            }
        )
        print(
            f"  Batch {key}: PASS records={csv_stats['total_records']:6d} "
            f"activated={csv_stats['activated_records']:6d} "
            f"detected={csv_stats['detected_records']:6d}"
        )

    expected_site_ids = {f"HMAC-STEM-{index:06d}" for index in range(1, 22_840)}
    require(global_site_ids == expected_site_ids, "global legal-site coverage")
    require(totals["baseline_records"] == EXPECTED_BASELINE_RECORDS, "aggregate baseline records")
    require(totals["enabled_records"] == EXPECTED_ENABLED_RECORDS, "aggregate enabled records")
    require(totals["total_records"] == EXPECTED_TOTAL_RECORDS, "aggregate total records")
    require(totals["fault_instances"] == EXPECTED_FAULT_INSTANCES, "aggregate fault instances")
    require(totals["unknown_records"] == 0, "aggregate unknown records")
    require(totals["detected_without_activity"] == 0, "aggregate detected without activity")
    require(totals["unclassified_instances"] == 0, "aggregate unclassified instances")
    require(
        totals["detected_instances"]
        + totals["activated_unobserved_instances"]
        + totals["unactivated_instances"]
        == EXPECTED_FAULT_INSTANCES,
        "aggregate fault classification closure",
    )

    created_at = datetime.now(timezone.utc).isoformat()
    manifest_value = {
        "stage": "11C-5B",
        "title": "FULL-CAMPAIGN DATASET INTEGRITY AND MANIFEST FREEZE",
        "status": "PASS",
        "created_at_utc": created_at,
        "campaign_mode": "FULL",
        "batch_count": 45,
        "full_batches": 44,
        "partial_batches": 1,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "selected_train_vectors": EXPECTED_TRAIN_VECTORS,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "baseline_records": totals["baseline_records"],
        "enabled_records": totals["enabled_records"],
        "total_csv_records": totals["total_records"],
        "activated_records": totals["activated_records"],
        "detected_records": totals["detected_records"],
        "timeout_records": totals["timeout_records"],
        "unknown_records": totals["unknown_records"],
        "detected_without_activity": totals["detected_without_activity"],
        "detected_instances": totals["detected_instances"],
        "activated_unobserved_instances": totals["activated_unobserved_instances"],
        "unactivated_instances": totals["unactivated_instances"],
        "unclassified_instances": totals["unclassified_instances"],
        "ordered_batch_sha256_commitment": ordered_commitment.hexdigest(),
        "checkpoint": {
            "path": relative(CHECKPOINT),
            "sha256": EXPECTED_CHECKPOINT_SHA,
        },
        "frozen_input_evidence": frozen_evidence,
        "batches": batch_manifest,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "raw_campaign_dataset_modified": False,
    }
    atomic_json(MANIFEST, manifest_value)
    manifest_sha = sha256(MANIFEST)

    audit_value = {
        "stage": "11C-5B",
        "title": "FULL-CAMPAIGN DATASET INTEGRITY AND MANIFEST FREEZE",
        "status": "PASS",
        "created_at_utc": created_at,
        "dataset_manifest": relative(MANIFEST),
        "dataset_manifest_sha256": manifest_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "batch_files_verified": 45,
        "batch_hashes_verified": 45,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "selected_train_vectors": EXPECTED_TRAIN_VECTORS,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "baseline_records": totals["baseline_records"],
        "enabled_records": totals["enabled_records"],
        "total_csv_records": totals["total_records"],
        "unknown_records": totals["unknown_records"],
        "detected_without_activity": totals["detected_without_activity"],
        "unclassified_instances": totals["unclassified_instances"],
        "all_integrity_checks": "PASS",
        "dataset_status": "FROZEN",
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "raw_campaign_dataset_modified": False,
        "next_gate": "CANONICAL DATASET CONSOLIDATION AND SCHEMA FREEZE",
    }
    atomic_json(AUDIT, audit_value)
    audit_sha = sha256(AUDIT)

    print("\nSTAGE 11C-5B — FULL-CAMPAIGN DATASET INTEGRITY AND MANIFEST FREEZE")
    print("Status                       : PASS")
    print("Dataset status               : FROZEN")
    print("Batches verified             : 45/45")
    print("Legal sites                  : 22839/22839")
    print("Persistent fault instances   : 45678/45678")
    print("TRAIN vectors                : 64")
    print("VALIDATION vectors exposed   : 0")
    print("HOLDOUT vectors exposed      : 0")
    print("Baseline records             :", totals["baseline_records"])
    print("Enabled records              :", totals["enabled_records"])
    print("Total CSV records            :", totals["total_records"])
    print("Activated records            :", totals["activated_records"])
    print("Detected records             :", totals["detected_records"])
    print("Detected instances           :", totals["detected_instances"])
    print("Activated-unobserved         :", totals["activated_unobserved_instances"])
    print("Unactivated                  :", totals["unactivated_instances"])
    print("Unclassified                 :", totals["unclassified_instances"])
    print("Unknown records              :", totals["unknown_records"])
    print("Detected without activity    :", totals["detected_without_activity"])
    print("Checkpoint SHA               :", EXPECTED_CHECKPOINT_SHA)
    print("Ordered dataset commitment   :", ordered_commitment.hexdigest())
    print("Frozen RTL modified          : NO")
    print("Golden netlist modified      : NO")
    print("Raw campaign dataset modified: NO")
    print("Manifest                     :", MANIFEST)
    print("Manifest SHA                 :", manifest_sha)
    print("Audit                        :", AUDIT)
    print("Audit SHA                    :", audit_sha)
    print("Next gate                    : CANONICAL DATASET CONSOLIDATION AND SCHEMA FREEZE")


if __name__ == "__main__":
    main()
