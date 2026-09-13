#!/usr/bin/env python3
"""Generate and freeze leakage-safe HMAC diagnostic feature matrices."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy is required. Activate the project .venv before running."
    ) from error


VERSION = "HMAC-LEAKAGE-SAFE-FEATURE-GENERATOR-v1"
FORMAT_VERSION = "HMAC-FACTORIZED-FEATURE-MATRIX-v1"
SCHEMA_VERSION = "HMAC-DIAGNOSTIC-FEATURE-SCHEMA-v1"

ROOT = Path.cwd().resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
CONFIG_ROOT = ROOT / "config/diagnostic_model"
CANONICAL_ROOT = RESULT_ROOT / "canonical_dataset_11c5c"
FEATURE_ROOT = RESULT_ROOT / "feature_matrix_11c5e"

CANONICAL_DATASET = (
    CANONICAL_ROOT / "hmac_fault_campaign_canonical_train_11c5c.csv.gz"
)
CANONICAL_SCHEMA = (
    CANONICAL_ROOT / "hmac_canonical_dataset_schema_11c5c.json"
)
CANONICAL_MANIFEST = RESULT_ROOT / "hmac_canonical_dataset_manifest_11c5c.json"
CANONICAL_AUDIT = RESULT_ROOT / "hmac_canonical_dataset_schema_freeze_11c5c.json"

POLICY = CONFIG_ROOT / "hmac_diagnostic_task_feature_split_policy_11c5d.json"
SITE_SPLIT = RESULT_ROOT / "hmac_fault_site_group_split_11c5d.csv"
SPLIT_MANIFEST = RESULT_ROOT / "hmac_fault_site_group_split_manifest_11c5d.json"
POLICY_AUDIT = RESULT_ROOT / "hmac_diagnostic_task_feature_split_freeze_11c5d.json"
POLICY_GENERATOR = ROOT / "stage_11c5d_policy_freeze.py"

LEGAL_SITES = (
    ROOT / "results/hmac_fault_campaign_11c1/hmac_legal_fault_sites_11c1b.json"
)
VECTOR_SELECTION = (
    ROOT / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json"
)
VECTOR_POOL = ROOT / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"

SITE_CATALOG = FEATURE_ROOT / "hmac_site_feature_catalog_11c5e.csv"
VECTOR_CATALOG = FEATURE_ROOT / "hmac_vector_feature_catalog_11c5e.csv"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
MANIFEST = RESULT_ROOT / "hmac_leakage_safe_feature_matrix_manifest_11c5e.json"
AUDIT = RESULT_ROOT / "hmac_leakage_safe_feature_matrix_freeze_11c5e.json"

MATRIX_PATHS = {
    "DEV_TRAIN": FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz",
    "DEV_CALIBRATION": (
        FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
    ),
    "DEV_SITE_TEST": (
        FEATURE_ROOT / "hmac_dev_site_test_feature_matrix_11c5e.npz"
    ),
}

EXPECTED_INPUTS = {
    CANONICAL_DATASET:
        "dcc79c94fa3a8d592b516e5f5683b1bf007682745103a9419ee20ef20a7ec78f",
    CANONICAL_SCHEMA:
        "2a08594f9573ff833a872d409783cee984bbb7499c7b013af4c72377ede3f772",
    CANONICAL_MANIFEST:
        "da5437f592362b2675e68abfde93857451f355e22b07948de2d8390a733173d4",
    CANONICAL_AUDIT:
        "a43de512ffab41cddaee286b2f459e9064f9588ee9c4b4621107485445ed3729",
    POLICY:
        "74243fa3b40f047a9f25a23c87eaccf6ac5a71a07241efef58df6b9e8e5322be",
    SITE_SPLIT:
        "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    SPLIT_MANIFEST:
        "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
    POLICY_AUDIT:
        "ff381a785c12281de3aeb1cda0e1d2b05d56444144b47dfa626b7334ee5b2985",
    POLICY_GENERATOR:
        "bd075f73357c2bbd0e707593eb5805de5b8409db7c63fe7e44e1dfd8b91fee82",
    LEGAL_SITES:
        "d9a39366bb35a277125376c6dc26c83273b45c0d7732fb93a3403401a071cb45",
    VECTOR_SELECTION:
        "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    VECTOR_POOL:
        "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
}

PARTITIONS = ("DEV_TRAIN", "DEV_CALIBRATION", "DEV_SITE_TEST")
EXPECTED_PARTITION_SITES = {
    "DEV_TRAIN": 15_987,
    "DEV_CALIBRATION": 3_426,
    "DEV_SITE_TEST": 3_426,
}
EXPECTED_PARTITION_ROWS = {
    name: count * 2 * 64 for name, count in EXPECTED_PARTITION_SITES.items()
}

EXPECTED_CANONICAL_ROWS = 2_926_272
EXPECTED_BASELINE_ROWS = 2_880
EXPECTED_ENABLED_ROWS = 2_923_392
EXPECTED_POSITIVE_ROWS = 1_278_888
EXPECTED_NEGATIVE_ROWS = 1_644_504
EXPECTED_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_TRAIN_VECTORS = 64

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

SPLIT_HEADER = [
    "site_id",
    "site_index",
    "partition",
    "partition_rank",
    "split_hash_sha256",
]

SITE_CATALOG_HEADER = [
    "site_id",
    "site_index",
    "partition",
    "partition_row",
    "cell_fanout",
    "cell_fanout_z",
    "is_primary_output_stem",
    "is_sequential_stem",
    "driver_cell_type",
    "driver_cell_type_code",
    "site_category",
    "site_category_code",
]

VECTOR_CATALOG_HEADER = [
    "vector_row",
    "vector_slot",
    "vector_id",
    "name",
    "source",
    "class",
    "key_hex",
    "message_hex",
]

EXPECTED_DRIVER_TYPES = {
    "$_AND_",
    "$_DFFE_PN0P_",
    "$_DFFE_PN1P_",
    "$_DFFE_PP_",
    "$_DFF_PN0_",
    "$_MUX_",
    "$_NOT_",
    "$_OR_",
    "$_XOR_",
}
EXPECTED_SITE_CATEGORIES = {
    "COMBINATIONAL_LOGIC_STEM",
    "SEQUENTIAL_STATE_STEM",
}

FORBIDDEN_LEAKAGE_FIELDS = {
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
}


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_file(path: Path, expected: str) -> dict:
    require(path.is_file(), f"missing frozen input: {path}")
    require(path.stat().st_size > 0, f"empty frozen input: {path}")
    actual = sha256(path)
    require(
        actual == expected,
        f"SHA mismatch for {relative(path)}: expected {expected}, actual {actual}",
    )
    return {
        "path": relative(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def output_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def integer(row: dict[str, str], field: str, location: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        stop(f"invalid integer {field} at {location}")


def binary(row: dict[str, str], field: str, location: str) -> int:
    value = integer(row, field, location)
    require(value in (0, 1), f"invalid binary {field} at {location}")
    return value


def json_binary(value, field: str, site_id: str) -> int:
    if value is True or value == 1:
        return 1
    if value is False or value == 0:
        return 0
    stop(f"invalid {field} for {site_id}")


def normalized_hex256(value, field: str, vector_id: int) -> str:
    if isinstance(value, int):
        text = f"{value:064x}"
    else:
        text = str(value).strip().lower()
        if text.startswith("0x"):
            text = text[2:]
    require(
        re.fullmatch(r"[0-9a-f]{64}", text) is not None,
        f"invalid {field} for vector {vector_id}",
    )
    return text


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
            and "driver_cell_type" in value[0]
        ):
            return value
        for child in value:
            result = find_sites(child)
            if result is not None:
                return result
    return None


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a deterministic, compressed, pickle-free NPZ archive."""
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def verify_npz(path: Path, expected: dict[str, tuple[tuple[int, ...], str]]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected), f"NPZ member set for {path.name}")
        for name, (shape, dtype_string) in expected.items():
            array = archive[name]
            require(array.shape == shape, f"{path.name}:{name} shape")
            require(array.dtype.str == dtype_string, f"{path.name}:{name} dtype")


def inspect_frozen_contracts() -> tuple[dict, dict]:
    policy = load_json(POLICY)
    audit = load_json(POLICY_AUDIT)
    split_manifest = load_json(SPLIT_MANIFEST)
    canonical_manifest = load_json(CANONICAL_MANIFEST)
    canonical_audit = load_json(CANONICAL_AUDIT)

    require(policy.get("status") == "PASS", "11C-5D policy status")
    require(audit.get("status") == "PASS", "11C-5D audit status")
    require(audit.get("task_status") == "FROZEN", "11C-5D task freeze")
    require(audit.get("feature_policy_status") == "FROZEN", "feature-policy freeze")
    require(audit.get("split_policy_status") == "FROZEN", "split-policy freeze")
    require(audit.get("development_training_authorization") == "DEV_TRAIN_ONLY", "training authorization")
    require(audit.get("validation_campaign_authorized") is False, "VALIDATION authorization")
    require(audit.get("holdout_campaign_authorized") is False, "HOLDOUT authorization")
    require(audit.get("gnn_training_authorized") is False, "GNN authorization")

    task_features = policy.get("diagnostic_task_and_features", {})
    task = task_features.get("task", {})
    require(task.get("name") == "pre_simulation_persistent_fault_detectability", "task name")
    require(task.get("target") == "detected", "task target")
    require(task.get("population") == "ENABLED rows only", "task population")
    forbidden = set(task_features.get("forbidden_target_leakage_features", []))
    require(forbidden == FORBIDDEN_LEAKAGE_FIELDS, "frozen forbidden-feature contract")

    require(split_manifest.get("status") == "PASS", "split manifest status")
    require(split_manifest.get("overlapping_sites") == 0, "site overlap")
    require(split_manifest.get("sa0_sa1_separations") == 0, "SA0/SA1 split")
    require(split_manifest.get("validation_vectors_exposed") == 0, "VALIDATION exposure")
    require(split_manifest.get("holdout_vectors_exposed") == 0, "HOLDOUT exposure")
    require(split_manifest.get("site_count") == EXPECTED_SITES, "split site count")
    require(split_manifest.get("enabled_row_count") == EXPECTED_ENABLED_ROWS, "split row count")

    require(canonical_manifest.get("status") == "PASS", "canonical manifest status")
    require(canonical_manifest.get("total_records") == EXPECTED_CANONICAL_ROWS, "canonical rows")
    require(canonical_manifest.get("enabled_records") == EXPECTED_ENABLED_ROWS, "canonical enabled rows")
    require(canonical_audit.get("status") == "PASS", "canonical audit status")
    require(canonical_audit.get("canonical_dataset_status") == "FROZEN", "canonical dataset freeze")
    require(canonical_audit.get("schema_status") == "FROZEN", "canonical schema freeze")

    partition_stats = split_manifest.get("partitions", {})
    require(set(partition_stats) == set(PARTITIONS), "partition-statistics set")
    for partition in PARTITIONS:
        stats = partition_stats[partition]
        require(stats.get("sites") == EXPECTED_PARTITION_SITES[partition], f"{partition} sites")
        require(stats.get("enabled_rows") == EXPECTED_PARTITION_ROWS[partition], f"{partition} rows")
        require(stats.get("detected_rows", 0) > 0, f"{partition} positive labels")
        require(stats.get("nondetected_rows", 0) > 0, f"{partition} negative labels")

    return policy, partition_stats


def load_split() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_site: dict[str, dict] = {}
    by_partition = {partition: [] for partition in PARTITIONS}
    with SITE_SPLIT.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == SPLIT_HEADER, "site-split header")
        for line_number, row in enumerate(reader, start=2):
            location = f"site-split line {line_number}"
            site_id = row["site_id"]
            site_index = integer(row, "site_index", location)
            partition = row["partition"]
            partition_rank = integer(row, "partition_rank", location)
            digest = row["split_hash_sha256"]
            require(re.fullmatch(r"HMAC-STEM-\d{6}", site_id) is not None, f"{location} site ID")
            require(1 <= site_index <= EXPECTED_SITES, f"{location} site index")
            require(partition in by_partition, f"{location} partition")
            require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"{location} split hash")
            require(site_id not in by_site, f"duplicate split site {site_id}")
            record = {
                "site_id": site_id,
                "site_index": site_index,
                "partition": partition,
                "partition_rank": partition_rank,
                "split_hash_sha256": digest,
            }
            by_site[site_id] = record
            by_partition[partition].append(record)

    require(len(by_site) == EXPECTED_SITES, "site-split record count")
    require(
        sorted(record["site_index"] for record in by_site.values())
        == list(range(1, EXPECTED_SITES + 1)),
        "site-split index coverage",
    )
    for partition in PARTITIONS:
        records = by_partition[partition]
        require(len(records) == EXPECTED_PARTITION_SITES[partition], f"{partition} site count")
        ranks = sorted(record["partition_rank"] for record in records)
        require(ranks == list(range(len(records))), f"{partition} rank coverage")
        records.sort(key=lambda item: item["site_index"])
        for partition_row, record in enumerate(records):
            record["partition_row"] = partition_row
    return by_site, by_partition


def load_legal_sites(split_by_site: dict[str, dict]) -> tuple[dict[str, dict], list[str], list[str]]:
    raw_sites = find_sites(load_json(LEGAL_SITES))
    require(isinstance(raw_sites, list), "legal-site list not found")
    require(len(raw_sites) == EXPECTED_SITES, "legal-site record count")

    by_site: dict[str, dict] = {}
    driver_types = set()
    site_categories = set()
    required = {
        "fault_site_id",
        "site_index",
        "driver_cell_type",
        "site_category",
        "cell_fanout",
    }
    for raw in raw_sites:
        require(required <= set(raw), f"legal-site fields for {raw.get('fault_site_id')}")
        site_id = str(raw["fault_site_id"])
        require(site_id in split_by_site, f"legal site absent from split: {site_id}")
        require(site_id not in by_site, f"duplicate legal site {site_id}")
        site_index = int(raw["site_index"])
        require(site_index == split_by_site[site_id]["site_index"], f"site index differs for {site_id}")
        fanout = int(raw["cell_fanout"])
        require(fanout >= 0, f"negative fanout for {site_id}")
        driver_type = str(raw["driver_cell_type"])
        category = str(raw["site_category"])

        if "is_primary_output_stem" in raw:
            is_primary = json_binary(raw["is_primary_output_stem"], "is_primary_output_stem", site_id)
        elif isinstance(raw.get("primary_outputs"), list):
            is_primary = int(bool(raw["primary_outputs"]))
        else:
            stop(f"primary-output classification absent for {site_id}")

        is_sequential = int(category == "SEQUENTIAL_STATE_STEM")
        record = {
            "site_id": site_id,
            "site_index": site_index,
            "driver_cell_type": driver_type,
            "site_category": category,
            "cell_fanout": fanout,
            "is_primary_output_stem": is_primary,
            "is_sequential_stem": is_sequential,
        }
        by_site[site_id] = record
        driver_types.add(driver_type)
        site_categories.add(category)

    require(set(by_site) == set(split_by_site), "legal/split site-set match")
    require(driver_types == EXPECTED_DRIVER_TYPES, "driver-cell vocabulary")
    require(site_categories == EXPECTED_SITE_CATEGORIES, "site-category vocabulary")
    return by_site, sorted(driver_types), sorted(site_categories)


def load_vectors() -> tuple[list[dict], np.ndarray]:
    selection = load_json(VECTOR_SELECTION)
    pool = load_json(VECTOR_POOL)
    selected_ids = selection.get("selected_vector_ids")
    positions = selection.get("selected_schedule_positions")
    require(isinstance(selected_ids, list) and len(selected_ids) == 64, "selected vector IDs")
    require(positions == list(range(64)), "selected vector positions")

    vectors = pool.get("vectors")
    require(isinstance(vectors, list) and len(vectors) == 256, "vector-pool records")
    split_counts = Counter(str(item.get("split")) for item in vectors)
    require(split_counts == Counter({"TRAIN": 160, "VALIDATION": 48, "HOLDOUT_TEST": 48}), "vector-pool splits")
    by_id = {int(item["vector_id"]): item for item in vectors}
    require(len(by_id) == 256, "unique vector IDs")

    selected = []
    feature_rows = []
    for vector_row, raw_id in enumerate(selected_ids):
        vector_id = int(raw_id)
        raw = by_id.get(vector_id)
        require(raw is not None, f"selected vector {vector_id} missing")
        require(raw.get("split") == "TRAIN", f"selected vector {vector_id} is not TRAIN")
        key_hex = normalized_hex256(raw.get("key"), "key", vector_id)
        message_hex = normalized_hex256(raw.get("message"), "message", vector_id)
        key_bits = np.unpackbits(np.frombuffer(bytes.fromhex(key_hex), dtype=np.uint8), bitorder="big")
        message_bits = np.unpackbits(np.frombuffer(bytes.fromhex(message_hex), dtype=np.uint8), bitorder="big")
        bits = np.concatenate((key_bits, message_bits)).astype(np.uint8, copy=False)
        require(bits.shape == (512,), f"vector {vector_id} feature width")
        selected.append(
            {
                "vector_row": vector_row,
                "vector_slot": vector_row,
                "vector_id": vector_id,
                "name": str(raw.get("name", "")),
                "source": str(raw.get("source", "")),
                "class": str(raw.get("class", "")),
                "key_hex": key_hex,
                "message_hex": message_hex,
            }
        )
        feature_rows.append(bits)

    return selected, np.stack(feature_rows, axis=0)


def feature_columns(driver_types: list[str], categories: list[str]) -> tuple[list[str], list[str]]:
    site_columns = [
        "cell_fanout_z",
        "is_primary_output_stem",
        "is_sequential_stem",
    ]
    site_columns.extend(f"driver_cell_type::{value}" for value in driver_types)
    site_columns.extend(f"site_category::{value}" for value in categories)
    stimulus_columns = [f"key_bit_{bit}" for bit in range(255, -1, -1)]
    stimulus_columns.extend(f"message_bit_{bit}" for bit in range(255, -1, -1))
    columns = ["stuck_value"] + site_columns + stimulus_columns
    require(len(site_columns) == 14, "site-feature width")
    require(len(columns) == 527, "model-feature width")
    require(not (set(columns) & FORBIDDEN_LEAKAGE_FIELDS), "leakage field in feature columns")
    return site_columns, columns


def build_site_features(
    legal_by_site: dict[str, dict],
    split_by_partition: dict[str, list[dict]],
    driver_types: list[str],
    categories: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict, list[dict]]:
    train_fanout = np.asarray(
        [legal_by_site[item["site_id"]]["cell_fanout"] for item in split_by_partition["DEV_TRAIN"]],
        dtype=np.float64,
    )
    mean = float(train_fanout.mean())
    scale = float(train_fanout.std(ddof=0))
    require(math.isfinite(mean), "fanout scaler mean")
    require(math.isfinite(scale) and scale > 0.0, "fanout scaler scale")

    driver_code = {value: index for index, value in enumerate(driver_types)}
    category_code = {value: index for index, value in enumerate(categories)}
    site_lookup: dict[str, dict[str, np.ndarray]] = {}
    catalog_rows = []

    for partition in PARTITIONS:
        split_records = split_by_partition[partition]
        count = len(split_records)
        site_ids = np.empty(count, dtype="<U16")
        site_indices = np.empty(count, dtype="<u2")
        site_matrix = np.zeros((count, 14), dtype="<f4")

        for partition_row, split_record in enumerate(split_records):
            site_id = split_record["site_id"]
            raw = legal_by_site[site_id]
            fanout_z = np.float32((raw["cell_fanout"] - mean) / scale)
            dcode = driver_code[raw["driver_cell_type"]]
            ccode = category_code[raw["site_category"]]

            site_ids[partition_row] = site_id
            site_indices[partition_row] = raw["site_index"]
            site_matrix[partition_row, 0] = fanout_z
            site_matrix[partition_row, 1] = raw["is_primary_output_stem"]
            site_matrix[partition_row, 2] = raw["is_sequential_stem"]
            site_matrix[partition_row, 3 + dcode] = 1.0
            site_matrix[partition_row, 3 + len(driver_types) + ccode] = 1.0

            require(site_matrix[partition_row, 3:3 + len(driver_types)].sum() == 1.0, f"driver one-hot {site_id}")
            require(site_matrix[partition_row, 3 + len(driver_types):].sum() == 1.0, f"category one-hot {site_id}")

            catalog_rows.append(
                {
                    "site_id": site_id,
                    "site_index": raw["site_index"],
                    "partition": partition,
                    "partition_row": partition_row,
                    "cell_fanout": raw["cell_fanout"],
                    "cell_fanout_z": format(float(fanout_z), ".9g"),
                    "is_primary_output_stem": raw["is_primary_output_stem"],
                    "is_sequential_stem": raw["is_sequential_stem"],
                    "driver_cell_type": raw["driver_cell_type"],
                    "driver_cell_type_code": dcode,
                    "site_category": raw["site_category"],
                    "site_category_code": ccode,
                }
            )

        site_lookup[partition] = {
            "site_ids": site_ids,
            "site_indices": site_indices,
            "site_features": site_matrix,
        }

    catalog_rows.sort(key=lambda item: int(item["site_index"]))
    scaler = {
        "name": "cell_fanout_standard_scaler",
        "fit_partition": "DEV_TRAIN",
        "fit_grain": "physical site; equivalent to sample weighting because every site has 128 samples",
        "mean_float64": mean,
        "scale_float64": scale,
        "variance_float64": scale * scale,
        "ddof": 0,
        "output_dtype": "float32 little-endian",
    }
    return site_lookup, scaler, catalog_rows


def allocate_samples() -> dict[str, dict[str, np.ndarray | int]]:
    result: dict[str, dict[str, np.ndarray | int]] = {}
    for partition in PARTITIONS:
        count = EXPECTED_PARTITION_ROWS[partition]
        result[partition] = {
            "record_id": np.empty(count, dtype="<u4"),
            "site_row": np.empty(count, dtype="<u2"),
            "vector_row": np.empty(count, dtype="|u1"),
            "stuck_value": np.empty(count, dtype="|u1"),
            "target_detected": np.empty(count, dtype="|u1"),
            "position": 0,
        }
    return result


def consume_canonical_dataset(
    split_by_site: dict[str, dict],
    selected_vectors: list[dict],
    samples: dict[str, dict[str, np.ndarray | int]],
) -> dict:
    vector_id_by_slot = [item["vector_id"] for item in selected_vectors]
    seen = np.zeros((EXPECTED_SITES, 2, 64), dtype=np.bool_)
    canonical_rows = 0
    baseline_rows = 0
    enabled_rows = 0
    positive_rows = 0
    activity_rows = 0
    timeout_rows = 0
    partition_positives = Counter()

    with gzip.open(CANONICAL_DATASET, mode="rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CANONICAL_HEADER, "canonical dataset header")
        for canonical_index, row in enumerate(reader):
            canonical_rows += 1
            location = f"canonical row {canonical_index + 2}"
            record_id = integer(row, "record_id", location)
            require(record_id == canonical_index, f"noncontiguous record_id at {location}")
            require(row["vector_split"] == "TRAIN", f"non-TRAIN vector at {location}")
            run_type = row["run_type"]

            if run_type == "BASELINE":
                baseline_rows += 1
                require(binary(row, "fault_enable", location) == 0, f"baseline fault_enable at {location}")
                require(binary(row, "detected", location) == 0, f"baseline detected at {location}")
                require(binary(row, "unknown", location) == 0, f"baseline unknown at {location}")
            elif run_type == "ENABLED":
                enabled_rows += 1
                require(binary(row, "fault_enable", location) == 1, f"enabled fault_enable at {location}")
                require(binary(row, "unknown", location) == 0, f"enabled unknown at {location}")
                site_id = row["site_id"]
                split_record = split_by_site.get(site_id)
                require(split_record is not None, f"unknown site_id at {location}: {site_id}")
                partition = split_record["partition"]
                site_row = split_record["partition_row"]
                stuck_value = binary(row, "stuck_value", location)
                vector_slot = integer(row, "vector_slot", location)
                require(0 <= vector_slot < 64, f"vector_slot at {location}")
                vector_id = integer(row, "vector_id", location)
                require(vector_id == vector_id_by_slot[vector_slot], f"vector ID/slot at {location}")
                target = binary(row, "detected", location)
                activity = binary(row, "activity", location)
                timed_out = binary(row, "timed_out", location)
                require(target <= activity, f"detected without activity at {location}")

                global_site_row = split_record["site_index"] - 1
                require(not seen[global_site_row, stuck_value, vector_slot], f"duplicate fault-vector sample at {location}")
                seen[global_site_row, stuck_value, vector_slot] = True

                state = samples[partition]
                position = int(state["position"])
                require(position < EXPECTED_PARTITION_ROWS[partition], f"{partition} row overflow")
                state["record_id"][position] = record_id
                state["site_row"][position] = site_row
                state["vector_row"][position] = vector_slot
                state["stuck_value"][position] = stuck_value
                state["target_detected"][position] = target
                state["position"] = position + 1

                positive_rows += target
                activity_rows += activity
                timeout_rows += timed_out
                partition_positives[partition] += target
            else:
                stop(f"invalid run_type at {location}: {run_type}")

            if canonical_rows % 250_000 == 0:
                print(
                    f"  Processed canonical rows: {canonical_rows}/{EXPECTED_CANONICAL_ROWS}",
                    flush=True,
                )

    require(canonical_rows == EXPECTED_CANONICAL_ROWS, "canonical row count")
    require(baseline_rows == EXPECTED_BASELINE_ROWS, "baseline row count")
    require(enabled_rows == EXPECTED_ENABLED_ROWS, "enabled row count")
    require(positive_rows == EXPECTED_POSITIVE_ROWS, "positive label count")
    require(enabled_rows - positive_rows == EXPECTED_NEGATIVE_ROWS, "negative label count")
    require(bool(seen.all()), "fault-site/stuck-value/vector coverage")
    require(int(seen.sum()) == EXPECTED_ENABLED_ROWS, "unique enabled sample count")
    for partition in PARTITIONS:
        require(int(samples[partition]["position"]) == EXPECTED_PARTITION_ROWS[partition], f"{partition} rows written")

    return {
        "canonical_rows": canonical_rows,
        "baseline_rows_excluded": baseline_rows,
        "enabled_model_samples": enabled_rows,
        "positive_samples": positive_rows,
        "negative_samples": enabled_rows - positive_rows,
        "activity_rows_integrity_only": activity_rows,
        "timeout_rows_integrity_only": timeout_rows,
        "partition_positive_samples": dict(partition_positives),
        "unique_site_stuck_vector_samples": int(seen.sum()),
        "duplicate_samples": 0,
        "missing_samples": 0,
        "unknown_samples": 0,
        "detected_without_activity": 0,
    }


def write_catalogs(
    site_path: Path,
    vector_path: Path,
    site_rows: list[dict],
    vector_rows: list[dict],
) -> None:
    with site_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SITE_CATALOG_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(site_rows)
    with vector_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=VECTOR_CATALOG_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(vector_rows)


def schema_document(
    site_columns: list[str],
    columns: list[str],
    driver_types: list[str],
    categories: list[str],
    scaler: dict,
) -> dict:
    return {
        "stage": "11C-5E",
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "matrix_format_version": FORMAT_VERSION,
        "task": "pre_simulation_persistent_fault_detectability",
        "sample_grain": "one persistent SA0/SA1 fault instance and one selected TRAIN vector",
        "target": {
            "array": "target_detected",
            "meaning": "1 if the persistent fault was observed by digest, latency, or timeout; otherwise 0",
            "dtype": "uint8",
            "not_a_feature": True,
        },
        "storage": {
            "type": "factorized NumPy NPZ",
            "reason": "Avoid duplicating 512 stimulus bits and 14 site features across 2,923,392 samples.",
            "pickle": False,
            "compression": "ZIP DEFLATE level 6 with fixed metadata",
            "sample_arrays": {
                "record_id": {"dtype": "uint32", "role": "provenance_only"},
                "site_row": {"dtype": "uint16", "role": "lookup_index_only"},
                "vector_row": {"dtype": "uint8", "role": "lookup_index_only"},
                "stuck_value": {"dtype": "uint8", "role": "model_feature"},
                "target_detected": {"dtype": "uint8", "role": "target_only"},
            },
            "lookup_arrays": {
                "site_ids": {"dtype": "unicode", "role": "provenance_only"},
                "site_indices": {"dtype": "uint16", "role": "provenance_only"},
                "site_features": {"dtype": "float32", "shape_suffix": [14]},
                "vector_ids": {"dtype": "uint16", "role": "provenance_only"},
                "vector_features": {"dtype": "uint8", "shape": [64, 512]},
                "feature_columns": {"dtype": "unicode", "shape": [527]},
                "site_feature_columns": {"dtype": "unicode", "shape": [14]},
            },
            "batch_materialization": (
                "For sample slice s, form X = concatenate([stuck_value[s,None], "
                "site_features[site_row[s]], vector_features[vector_row[s]]], axis=1). "
                "Use float32 and materialize in batches, not as one 2.9M-row dense array."
            ),
        },
        "model_feature_columns": columns,
        "model_feature_count": len(columns),
        "site_feature_columns": site_columns,
        "site_feature_count": len(site_columns),
        "stimulus_feature_count": 512,
        "feature_order": {
            "key_bits": "bit 255 through bit 0",
            "message_bits": "bit 255 through bit 0",
        },
        "categorical_vocabulary": {
            "driver_cell_type": driver_types,
            "site_category": categories,
            "source": "complete frozen legal-site inventory; target labels not consulted",
        },
        "numeric_preprocessing": scaler,
        "provenance_only_not_features": [
            "record_id",
            "site_id",
            "site_index",
            "site_row",
            "vector_id",
            "vector_row",
            "partition",
        ],
        "forbidden_post_simulation_features": sorted(FORBIDDEN_LEAKAGE_FIELDS),
        "leakage_contract": {
            "post_simulation_fields_in_X": [],
            "target_in_X": False,
            "scaler_fit_partition": "DEV_TRAIN",
            "site_group_overlap": 0,
            "sa0_sa1_group_separations": 0,
            "validation_vectors_exposed": 0,
            "holdout_vectors_exposed": 0,
        },
        "split_usage": {
            "DEV_TRAIN": "Fit preprocessing and model parameters.",
            "DEV_CALIBRATION": "Compare frozen candidates, calibrate probabilities, and select threshold; never fit the final model.",
            "DEV_SITE_TEST": "Locked unseen-site evaluation; no fitting, selection, calibration, or threshold changes.",
        },
    }


def matrix_arrays(
    partition: str,
    sample_state: dict[str, np.ndarray | int],
    site_state: dict[str, np.ndarray],
    vector_rows: list[dict],
    vector_features: np.ndarray,
    site_columns: list[str],
    columns: list[str],
) -> dict[str, np.ndarray]:
    return {
        "feature_columns": np.asarray(columns),
        "format_version": np.asarray([FORMAT_VERSION]),
        "partition": np.asarray([partition]),
        "record_id": sample_state["record_id"],
        "site_feature_columns": np.asarray(site_columns),
        "site_features": site_state["site_features"],
        "site_ids": site_state["site_ids"],
        "site_indices": site_state["site_indices"],
        "site_row": sample_state["site_row"],
        "stuck_value": sample_state["stuck_value"],
        "target_detected": sample_state["target_detected"],
        "vector_features": vector_features,
        "vector_ids": np.asarray([item["vector_id"] for item in vector_rows], dtype="<u2"),
        "vector_row": sample_state["vector_row"],
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    all_outputs = [SITE_CATALOG, VECTOR_CATALOG, FEATURE_SCHEMA, MANIFEST, AUDIT]
    all_outputs.extend(MATRIX_PATHS.values())
    for path in all_outputs:
        require(not path.exists(), f"Stage 11C-5E output already exists: {path}")
    require(not FEATURE_ROOT.exists(), f"Stage 11C-5E directory already exists: {FEATURE_ROOT}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    print("STAGE 11C-5E — LEAKAGE-SAFE FEATURE MATRIX GENERATION")
    print("FROZEN INPUT VERIFICATION")
    input_evidence = {}
    for path, expected in EXPECTED_INPUTS.items():
        record = verify_file(path, expected)
        input_evidence[record["path"]] = record
        print(f"  {path.name:<66} : OK", flush=True)

    frozen_policy, frozen_partition_stats = inspect_frozen_contracts()
    split_by_site, split_by_partition = load_split()
    legal_by_site, driver_types, categories = load_legal_sites(split_by_site)
    vector_rows, vector_features = load_vectors()
    site_columns, columns = feature_columns(driver_types, categories)
    site_lookup, scaler, site_catalog_rows = build_site_features(
        legal_by_site,
        split_by_partition,
        driver_types,
        categories,
    )

    print("\nCANONICAL DATASET PASS")
    samples = allocate_samples()
    dataset_stats = consume_canonical_dataset(split_by_site, vector_rows, samples)
    for partition in PARTITIONS:
        expected_positive = int(frozen_partition_stats[partition]["detected_rows"])
        actual_positive = int(samples[partition]["target_detected"].sum(dtype=np.uint64))
        require(actual_positive == expected_positive, f"{partition} positive-label total")

    temporary_root = Path(tempfile.mkdtemp(prefix="stage_11c5e_", dir=RESULT_ROOT))
    temporary_feature_root = temporary_root / FEATURE_ROOT.name
    temporary_feature_root.mkdir()
    try:
        temporary_site_catalog = temporary_feature_root / SITE_CATALOG.name
        temporary_vector_catalog = temporary_feature_root / VECTOR_CATALOG.name
        write_catalogs(
            temporary_site_catalog,
            temporary_vector_catalog,
            site_catalog_rows,
            vector_rows,
        )

        schema = schema_document(
            site_columns,
            columns,
            driver_types,
            categories,
            scaler,
        )
        temporary_schema = temporary_feature_root / FEATURE_SCHEMA.name
        temporary_schema.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")

        print("\nWRITING DETERMINISTIC MATRICES", flush=True)
        temporary_matrices = {}
        for partition in PARTITIONS:
            temporary_path = temporary_feature_root / MATRIX_PATHS[partition].name
            arrays = matrix_arrays(
                partition,
                samples[partition],
                site_lookup[partition],
                vector_rows,
                vector_features,
                site_columns,
                columns,
            )
            deterministic_npz(temporary_path, arrays)
            expected_members = {
                name: (array.shape, array.dtype.str) for name, array in arrays.items()
            }
            verify_npz(temporary_path, expected_members)
            temporary_matrices[partition] = temporary_path
            print(
                f"  {partition:<16} rows={EXPECTED_PARTITION_ROWS[partition]:7d} "
                f"bytes={temporary_path.stat().st_size:10d} sha256={sha256(temporary_path)}",
                flush=True,
            )

        FEATURE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        temporary_feature_root.replace(FEATURE_ROOT)

        matrix_records = {
            partition: output_record(MATRIX_PATHS[partition]) for partition in PARTITIONS
        }
        schema_record = output_record(FEATURE_SCHEMA)
        site_catalog_record = output_record(SITE_CATALOG)
        vector_catalog_record = output_record(VECTOR_CATALOG)

        partition_records = {}
        for partition in PARTITIONS:
            rows = EXPECTED_PARTITION_ROWS[partition]
            positives = int(samples[partition]["target_detected"].sum(dtype=np.uint64))
            site_count = EXPECTED_PARTITION_SITES[partition]
            partition_records[partition] = {
                "sites": site_count,
                "fault_instances": site_count * 2,
                "samples": rows,
                "positive_samples": positives,
                "negative_samples": rows - positives,
                "positive_prevalence": positives / rows,
                "matrix": matrix_records[partition],
            }

        manifest = {
            "stage": "11C-5E",
            "title": "LEAKAGE-SAFE FEATURE MATRIX GENERATION AND FREEZE",
            "status": "PASS",
            "generator_version": VERSION,
            "matrix_format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": "pre_simulation_persistent_fault_detectability",
            "fault_scope": "persistent net-stem SA0/SA1 only",
            "target": "detected",
            "model_feature_count": len(columns),
            "site_feature_count": len(site_columns),
            "stimulus_feature_count": 512,
            "dataset_statistics": dataset_stats,
            "partitions": partition_records,
            "site_feature_catalog": site_catalog_record,
            "vector_feature_catalog": vector_catalog_record,
            "feature_schema": schema_record,
            "input_evidence": input_evidence,
            "post_simulation_fields_in_features": [],
            "validation_vectors_exposed": 0,
            "holdout_vectors_exposed": 0,
            "frozen_rtl_modified": False,
            "golden_netlist_modified": False,
            "canonical_dataset_modified": False,
        }
        atomic_json(MANIFEST, manifest)
        manifest_record = output_record(MANIFEST)

        audit = {
            "stage": "11C-5E",
            "title": "LEAKAGE-SAFE FEATURE MATRIX GENERATION AND FREEZE",
            "status": "PASS",
            "feature_matrix_status": "FROZEN",
            "schema_status": "FROZEN",
            "storage_format": "FACTORIZED NPZ",
            "model_samples": EXPECTED_ENABLED_ROWS,
            "positive_samples": EXPECTED_POSITIVE_ROWS,
            "negative_samples": EXPECTED_NEGATIVE_ROWS,
            "physical_site_groups": EXPECTED_SITES,
            "fault_instances": EXPECTED_FAULT_INSTANCES,
            "model_feature_count": len(columns),
            "partition_samples": {
                name: EXPECTED_PARTITION_ROWS[name] for name in PARTITIONS
            },
            "site_overlap": 0,
            "sa0_sa1_separations": 0,
            "duplicate_samples": 0,
            "missing_samples": 0,
            "unknown_samples": 0,
            "post_simulation_features": 0,
            "target_in_feature_columns": False,
            "scaler_fit_partition": "DEV_TRAIN",
            "validation_vectors_exposed": 0,
            "holdout_vectors_exposed": 0,
            "development_training_authorization": "DEV_TRAIN_ONLY",
            "validation_campaign_authorized": False,
            "holdout_campaign_authorized": False,
            "gnn_training_authorized": False,
            "matrix_manifest": manifest_record,
            "feature_schema": schema_record,
            "matrix_artifacts": matrix_records,
            "frozen_rtl_modified": False,
            "golden_netlist_modified": False,
            "canonical_dataset_modified": False,
            "next_gate": "CONVENTIONAL BASELINE MODEL AND TRAINING-CONTRACT FREEZE",
        }
        atomic_json(AUDIT, audit)
        audit_record = output_record(AUDIT)

    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)

    print("\nPARTITION FEATURE MATRIX SUMMARY")
    for partition in PARTITIONS:
        record = partition_records[partition]
        print(
            f"  {partition:<16} sites={record['sites']:5d} "
            f"samples={record['samples']:7d} positive={record['positive_samples']:7d} "
            f"prevalence={record['positive_prevalence'] * 100:6.2f}%"
        )

    print("\nSTAGE 11C-5E — LEAKAGE-SAFE FEATURE MATRIX GENERATION AND FREEZE")
    print("Status                       : PASS")
    print("Feature matrix status        : FROZEN")
    print("Feature schema status        : FROZEN")
    print("Storage                      : FACTORIZED NPZ")
    print("Persistent fault scope       : SA0/SA1 ONLY")
    print("Model samples                :", EXPECTED_ENABLED_ROWS)
    print("Positive samples             :", EXPECTED_POSITIVE_ROWS)
    print("Negative samples             :", EXPECTED_NEGATIVE_ROWS)
    print("Physical site groups         :", EXPECTED_SITES)
    print("Fault instances              :", EXPECTED_FAULT_INSTANCES)
    print("Model features               :", len(columns))
    print("DEV_TRAIN samples            :", EXPECTED_PARTITION_ROWS["DEV_TRAIN"])
    print("DEV_CALIBRATION samples      :", EXPECTED_PARTITION_ROWS["DEV_CALIBRATION"])
    print("DEV_SITE_TEST samples        :", EXPECTED_PARTITION_ROWS["DEV_SITE_TEST"])
    print("Overlapping sites            : 0")
    print("SA0/SA1 group separations    : 0")
    print("Missing/duplicate samples    : 0/0")
    print("Post-simulation features     : 0")
    print("Target present in X          : NO")
    print("Scaler fit                   : DEV_TRAIN ONLY")
    print("VALIDATION vectors exposed   : 0")
    print("HOLDOUT vectors exposed      : 0")
    print("Development training         : AUTHORIZED FOR DEV_TRAIN ONLY")
    print("GNN training                 : NOT YET AUTHORIZED")
    print("Frozen RTL modified          : NO")
    print("Golden netlist modified      : NO")
    print("Canonical dataset modified   : NO")
    for partition in PARTITIONS:
        record = matrix_records[partition]
        print(f"{partition} matrix           :", ROOT / record["path"])
        print(f"{partition} matrix SHA       :", record["sha256"])
    print("Feature schema               :", FEATURE_SCHEMA)
    print("Feature schema SHA           :", schema_record["sha256"])
    print("Manifest                     :", MANIFEST)
    print("Manifest SHA                 :", manifest_record["sha256"])
    print("Audit                        :", AUDIT)
    print("Audit SHA                    :", audit_record["sha256"])
    print("Next gate                    : CONVENTIONAL BASELINE MODEL AND TRAINING-CONTRACT FREEZE")


if __name__ == "__main__":
    main()
