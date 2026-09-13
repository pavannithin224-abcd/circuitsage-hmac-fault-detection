#!/usr/bin/env python3
"""Freeze the blind validation campaign architecture and execution contract."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-VALIDATION-CAMPAIGN-CONTRACT-v1"

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
CONFIG_FAULT = ROOT / "config/fault_campaign"
CONFIG_MODEL = ROOT / "config/diagnostic_model"
RESULT_11C1 = ROOT / "results/hmac_fault_campaign_11c1"
RESULT_11C3 = ROOT / "results/hmac_fault_campaign_11c3"
RESULT_11C4 = ROOT / "results/hmac_fault_campaign_11c4"
RESULT_11D2 = ROOT / "results/hmac_fault_campaign_11d2"
RESULT_11D3 = ROOT / "results/hmac_fault_campaign_11d3"

VECTOR_SPLIT_POLICY = CONFIG_FAULT / "hmac_vector_split_policy_11c3a.json"
VECTOR_POOL = RESULT_11C3 / "hmac_vector_pool_11c3a.json"
VECTOR_POOL_AUDIT = RESULT_11C3 / "hmac_vector_pool_freeze_11c3a.json"
TRAIN_VECTOR_SELECTION = CONFIG_FAULT / "hmac_campaign_vector_selection_11c3b_b.json"
LEGAL_SITE_FREEZE = RESULT_11C1 / "hmac_legal_fault_site_freeze_11c1b.json"
GENERATOR_LOCK = CONFIG_FAULT / "hmac_reusable_batch_generator_lock_11c4e_e.json"
GENERATION_INDEX = RESULT_11C4 / "hmac_canonical_batch_generation_index_11c4g_a1.json"
SEMANTIC_MANIFEST = RESULT_11C4 / "hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json"
RUNNER_SOURCE_FREEZE = RESULT_11C4 / "hmac_full_campaign_runner_source_freeze_11c4g_b2b.json"
TESTBENCH_GENERATOR = ROOT / "scripts/hmac/generate_hmac_full_campaign_testbench.py"
CAMPAIGN_RUNNER = ROOT / "scripts/hmac/run_hmac_full_fault_campaign.py"

FINAL_DISPOSITION_POLICY = CONFIG_MODEL / "hmac_hybrid_final_diagnostic_disposition_policy_11d2d.json"
FINAL_MODEL_LOCK = CONFIG_MODEL / "hmac_final_diagnostic_model_lock_11d2d.json"
VALIDATION_READINESS = CONFIG_MODEL / "hmac_validation_readiness_policy_11d2d.json"
FINAL_COMPARATOR_REGISTRY = RESULT_11D2 / "hmac_final_diagnostic_comparator_registry_11d2d.json"
FINAL_MODEL_AUDIT = RESULT_11D2 / "hmac_hybrid_disposition_final_diagnostic_model_freeze_11d2d.json"
FINAL_MODEL = RESULT_11D2 / "hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"

VALIDATION_ARCHITECTURE = CONFIG_FAULT / "hmac_validation_campaign_architecture_11d3a.json"
EXECUTION_CONTRACT = CONFIG_FAULT / "hmac_validation_campaign_execution_contract_11d3a.json"
VECTOR_COMMITMENTS_CSV = RESULT_11D3 / "hmac_validation_vector_commitments_11d3a.csv"
VECTOR_COMMITMENTS_JSON = RESULT_11D3 / "hmac_validation_vector_commitments_11d3a.json"
AUDIT = RESULT_11D3 / "hmac_validation_campaign_architecture_execution_contract_freeze_11d3a.json"

EXPECTED_INPUTS = {
    VECTOR_SPLIT_POLICY: "3930631ef81addff1fb5150e4e28e525b0a9e127c3b2c1e051b9d4e2e9a7681a",
    VECTOR_POOL: "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
    VECTOR_POOL_AUDIT: "abd513dc35f6c66bbed2541fd19aea168a8009fd760336e0a8314ead6d9bc9cc",
    TRAIN_VECTOR_SELECTION: "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    LEGAL_SITE_FREEZE: "de4d2c68352a51fd1ef85cd3f304dd7e541e97a9ff44160d4d0b02d5d52a1795",
    GENERATOR_LOCK: "da59b47f95c2d16e648e637b1bacc064a35be1252038376cc96dd3151483763f",
    GENERATION_INDEX: "7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
    SEMANTIC_MANIFEST: "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
    RUNNER_SOURCE_FREEZE: "71d0a03f3ae813d64caf365210d2143957c6973ae1c27332c47fd3aec97591e9",
    TESTBENCH_GENERATOR: "faae8e8d7bd5096840b103fa0fad6e893e0712b7f41747b2679494558dcb8094",
    CAMPAIGN_RUNNER: "9bfcc8d98dbc17611966b82ebf0cbb0bb26560bb4f0c5afa1d8df9867a8c21a2",
    FINAL_DISPOSITION_POLICY: "819219d55c5d29a13b59a1acc8858c203b0efb28bb86e26c49e2b7a230c4ac33",
    FINAL_MODEL_LOCK: "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    VALIDATION_READINESS: "070d72a66b2022b6fc56394748fb6f6c6e5ba8fd80aaac7b29a6bf4dbed4008f",
    FINAL_COMPARATOR_REGISTRY: "39e15600efdf5e65d2155f7ddfcdf4cfc9754fe18dfacdb64bd2cce8cf64e7df",
    FINAL_MODEL_AUDIT: "8114786a4e026e8f5b7ef482af13c0ff3eb78c8f7815fcd5ae8e0390684d0c05",
    FINAL_MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
}

EXPECTED_TOTAL_VECTORS = 256
EXPECTED_TRAIN_VECTORS = 160
EXPECTED_SELECTED_TRAIN_VECTORS = 64
EXPECTED_VALIDATION_VECTORS = 48
EXPECTED_HOLDOUT_VECTORS = 48
EXPECTED_BATCHES = 45
EXPECTED_FULL_BATCHES = 44
EXPECTED_FINAL_BATCH_SITES = 311
EXPECTED_LEGAL_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_VALIDATION_BASELINE_RECORDS = 2_160
EXPECTED_VALIDATION_ENABLED_RECORDS = 2_192_544
EXPECTED_VALIDATION_TOTAL_RECORDS = 2_194_704
EXPECTED_CYCLES = 343
EXPECTED_HYBRID_MCC = 0.62982119
MINIMUM_VALIDATION_MCC = 0.52982119
REQUIRED_HYBRID_GAIN_OVER_GNN = 0.02

HEX_64 = re.compile(r"[0-9a-f]{64}")


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


def require_field_value(data: dict, names: tuple[str, ...], expected: object, label: str) -> str:
    """Accept documented legacy key spellings while enforcing one exact value."""
    present = {name: data[name] for name in names if name in data}
    require(present, f"{label}: none of the accepted fields are present: {', '.join(names)}")
    matching = [name for name, value in present.items() if value == expected]
    require(
        matching,
        f"{label}: expected {expected!r}; observed {present!r}",
    )
    require(
        all(value == expected for value in present.values()),
        f"{label}: conflicting legacy fields {present!r}",
    )
    return matching[0]


def require_optional_pass_status(data: dict, label: str) -> None:
    """Some early lock files encoded only their specific frozen-state field."""
    if "status" in data:
        require(data["status"] == "PASS", label)


def verify_field_value_if_present(
    data: dict,
    names: tuple[str, ...],
    expected: object,
    label: str,
) -> str:
    """Validate a summary field when that historical artifact carries it."""
    present = {name: data[name] for name in names if name in data}
    if not present:
        return "VERIFIED_BY_FROZEN_SHA_AND_CLOSURE"
    require(
        all(value == expected for value in present.values()),
        f"{label}: expected {expected!r}; observed {present!r}",
    )
    return next(iter(present))


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
    print(f"  {path.name:<78}: OK", flush=True)
    return record(path)


def verify_outputs_absent() -> None:
    for path in (
        VALIDATION_ARCHITECTURE,
        EXECUTION_CONTRACT,
        VECTOR_COMMITMENTS_CSV,
        VECTOR_COMMITMENTS_JSON,
        AUDIT,
    ):
        require(not path.exists(), f"refusing to overwrite Stage 11D-3A output: {relative(path)}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        VALIDATION_ARCHITECTURE,
        EXECUTION_CONTRACT,
        VECTOR_COMMITMENTS_CSV,
        VECTOR_COMMITMENTS_JSON,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def verify_contract_only_source() -> None:
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
        if name in {
            "fit", "partial_fit", "predict", "predict_proba", "select_threshold",
            "run", "Popen", "check_call", "check_output",
        }:
            forbidden.append((node.lineno, name))
    require(not forbidden, f"training, inference, or campaign-execution calls present: {forbidden}")


def verify_final_model_governance() -> dict:
    disposition = load_json(FINAL_DISPOSITION_POLICY)
    model_lock = load_json(FINAL_MODEL_LOCK)
    readiness = load_json(VALIDATION_READINESS)
    registry = load_json(FINAL_COMPARATOR_REGISTRY)
    audit = load_json(FINAL_MODEL_AUDIT)

    require(disposition.get("status") == "PASS", "final disposition status")
    require(disposition.get("policy_status") == "FROZEN", "final disposition freeze")
    require(disposition.get("winning_model_id") == "HYBRID_FUSION_MLP_11D2C", "final winning model")
    require(disposition.get("governance_disposition") == "FREEZE_HYBRID_AS_FINAL_DEVELOPMENT_DIAGNOSTIC_MODEL", "final model governance")
    require(disposition.get("retraining_authorized") is False, "retraining prohibition")
    require(disposition.get("threshold_change_authorized") is False, "threshold-change prohibition")
    require(disposition.get("validation_vectors_exposed") == 0, "prior VALIDATION exposure")
    require(disposition.get("holdout_vectors_exposed") == 0, "prior HOLDOUT exposure")

    require(model_lock.get("status") == "PASS" and model_lock.get("lock_status") == "FROZEN", "final model lock")
    require(model_lock.get("model_id") == "HYBRID_FUSION_MLP_11D2C", "final model identity")
    require(model_lock.get("model_artifact") == record(FINAL_MODEL), "final model artifact")
    require(model_lock.get("model_retraining_allowed") is False, "final model retraining")
    require(model_lock.get("threshold_reselection_allowed") is False, "final threshold reselection")
    require(model_lock.get("validation_use_for_tuning_allowed") is False, "validation tuning prohibition")
    require(model_lock.get("holdout_use_for_tuning_allowed") is False, "holdout tuning prohibition")
    require(close(model_lock.get("evaluation_metrics", {}).get("mcc"), EXPECTED_HYBRID_MCC), "locked hybrid MCC")

    require(readiness.get("status") == "PASS", "validation readiness status")
    require(readiness.get("readiness_status") == "READY_FOR_VALIDATION_CAMPAIGN_CONTRACT_ONLY", "validation readiness gate")
    require(readiness.get("validation_campaign_contract_authorized") is True, "validation contract authorization")
    require(readiness.get("validation_campaign_execution_authorized") is False, "validation execution gate")
    require(readiness.get("holdout_campaign_contract_authorized") is False, "holdout contract gate")
    require(readiness.get("holdout_campaign_execution_authorized") is False, "holdout execution gate")
    require(readiness.get("validation_vectors_exposed") == 0, "readiness VALIDATION exposure")
    require(readiness.get("holdout_vectors_exposed") == 0, "readiness HOLDOUT exposure")

    require(registry.get("status") == "PASS" and registry.get("registry_status") == "FROZEN", "comparator registry")
    require(registry.get("winner") == "HYBRID_FUSION_MLP_11D2C", "registry winner")
    comparators = registry.get("comparators")
    require(isinstance(comparators, list) and len(comparators) == 4, "comparator count")
    require(sum(item.get("selection_winner") == "YES" for item in comparators) == 1, "single comparator winner")

    require(audit.get("status") == "PASS", "final model audit status")
    require(audit.get("final_model_status") == "FROZEN", "final model audit freeze")
    require(audit.get("winner") == "HYBRID", "final model audit winner")
    require(audit.get("validation_campaign_contract") == "AUTHORIZED", "validation contract handoff")
    require(audit.get("validation_campaign_execution") == "NOT_YET_AUTHORIZED", "validation execution handoff")
    require(audit.get("holdout_campaign") == "NOT_YET_AUTHORIZED", "holdout handoff")
    require(audit.get("validation_vectors_exposed") == 0, "audit VALIDATION exposure")
    require(audit.get("holdout_vectors_exposed") == 0, "audit HOLDOUT exposure")
    require(audit.get("next_gate", "").startswith("STAGE 11D-3A"), "validation contract gate")
    return {
        "disposition": disposition,
        "model_lock": model_lock,
        "readiness": readiness,
        "registry": registry,
        "audit": audit,
    }


def freeze_validation_vector_commitments() -> tuple[list[dict], str, dict]:
    policy = load_json(VECTOR_SPLIT_POLICY)
    pool_audit = load_json(VECTOR_POOL_AUDIT)
    pool = load_json(VECTOR_POOL)
    selection = load_json(TRAIN_VECTOR_SELECTION)

    require(policy.get("status") == "FROZEN", "vector split policy status")
    require(policy.get("pool_size") == EXPECTED_TOTAL_VECTORS, "vector split policy pool size")
    require(
        policy.get("split_policy", {}).get("TRAIN") == EXPECTED_TRAIN_VECTORS,
        "vector split policy TRAIN count",
    )
    require(
        policy.get("split_policy", {}).get("VALIDATION") == EXPECTED_VALIDATION_VECTORS,
        "vector split policy VALIDATION count",
    )
    require(
        policy.get("split_policy", {}).get("HOLDOUT_TEST") == EXPECTED_HOLDOUT_VECTORS,
        "vector split policy HOLDOUT count",
    )
    require(pool_audit.get("status") == "PASS", "vector-pool audit status")
    require(pool_audit.get("pool_size") == EXPECTED_TOTAL_VECTORS, "vector-pool audit size")
    require(pool.get("status") == "PASS", "vector-pool status")
    require(pool.get("pool_size") == EXPECTED_TOTAL_VECTORS, "vector-pool size")
    require(selection.get("status") == "FROZEN", "TRAIN-vector selection status")
    require(
        selection.get("selected_train_vector_count") == EXPECTED_SELECTED_TRAIN_VECTORS,
        "selected TRAIN count declaration",
    )
    require(selection.get("validation_vectors_reserved") == EXPECTED_VALIDATION_VECTORS, "reserved VALIDATION count")
    require(selection.get("holdout_vectors_reserved") == EXPECTED_HOLDOUT_VECTORS, "reserved HOLDOUT count")
    require(selection.get("validation_status") == "NOT_YET_RUN", "prior validation status")
    require(selection.get("holdout_status") == "SEALED", "HOLDOUT seal")
    records = pool.get("vectors")
    require(isinstance(records, list) and len(records) == EXPECTED_TOTAL_VECTORS, "vector-pool record count")
    counts = Counter(str(item.get("split")) for item in records)
    require(counts == Counter({
        "TRAIN": EXPECTED_TRAIN_VECTORS,
        "VALIDATION": EXPECTED_VALIDATION_VECTORS,
        "HOLDOUT_TEST": EXPECTED_HOLDOUT_VECTORS,
    }), "vector split counts")

    vector_ids = [int(item["vector_id"]) for item in records]
    require(len(set(vector_ids)) == EXPECTED_TOTAL_VECTORS, "unique vector IDs")
    fingerprints = [str(item.get("pair_fingerprint", "")) for item in records]
    commitments = [str(item.get("vector_commitment", "")) for item in records]
    require(len(set(fingerprints)) == EXPECTED_TOTAL_VECTORS, "unique key/message fingerprints")
    require(len(set(commitments)) == EXPECTED_TOTAL_VECTORS, "unique vector commitments")
    require(all(HEX_64.fullmatch(value) for value in fingerprints), "pair-fingerprint format")
    require(all(HEX_64.fullmatch(value) for value in commitments), "vector-commitment format")

    selected_train = [int(value) for value in selection.get("selected_vector_ids", [])]
    require(len(selected_train) == EXPECTED_SELECTED_TRAIN_VECTORS, "selected TRAIN vector count")
    require(len(set(selected_train)) == EXPECTED_SELECTED_TRAIN_VECTORS, "selected TRAIN vector uniqueness")
    by_id = {int(item["vector_id"]): item for item in records}
    require(all(by_id[item].get("split") == "TRAIN" for item in selected_train), "selected TRAIN split")

    validation = sorted(
        (item for item in records if item.get("split") == "VALIDATION"),
        key=lambda item: int(item["vector_id"]),
    )
    require(len(validation) == EXPECTED_VALIDATION_VECTORS, "VALIDATION vector count")
    require(set(selected_train).isdisjoint({int(item["vector_id"]) for item in validation}), "TRAIN/VALIDATION separation")
    for item in validation:
        for field in ("key", "message", "expected_hmac_sha256"):
            value = str(item.get(field, "")).lower().removeprefix("0x")
            require(HEX_64.fullmatch(value) is not None, f"validation vector {item['vector_id']} {field}")

    rows = []
    commitment_hasher = hashlib.sha256()
    for order, item in enumerate(validation):
        row = {
            "validation_order": order,
            "vector_id": int(item["vector_id"]),
            "name": str(item.get("name", "")),
            "class": str(item.get("class", "")),
            "source": str(item.get("source", "")),
            "pair_fingerprint": str(item["pair_fingerprint"]),
            "vector_commitment": str(item["vector_commitment"]),
        }
        require(row["name"] and row["class"] and row["source"], "validation metadata completeness")
        commitment_hasher.update(
            (
                f"{row['validation_order']}|{row['vector_id']}|"
                f"{row['pair_fingerprint']}|{row['vector_commitment']}\n"
            ).encode()
        )
        rows.append(row)

    commitment = commitment_hasher.hexdigest()
    atomic_csv(VECTOR_COMMITMENTS_CSV, list(rows[0]), rows)
    atomic_json(VECTOR_COMMITMENTS_JSON, {
        "stage": "11D-3A",
        "title": "SEALED VALIDATION VECTOR COMMITMENT MANIFEST",
        "status": "PASS",
        "manifest_status": "FROZEN",
        "source_vector_pool": record(VECTOR_POOL),
        "ordering": "ASCENDING VECTOR_ID",
        "validation_vector_count": EXPECTED_VALIDATION_VECTORS,
        "validation_vector_payloads_emitted": 0,
        "holdout_vector_records_emitted": 0,
        "ordered_validation_commitment": commitment,
        "records": rows,
    })
    return rows, commitment, dict(counts)


def batch_plan() -> list[dict]:
    rows = []
    for batch_id in range(EXPECTED_BATCHES):
        sites = EXPECTED_FINAL_BATCH_SITES if batch_id == 44 else 512
        faults = sites * 2
        enabled = faults * EXPECTED_VALIDATION_VECTORS
        baseline = EXPECTED_VALIDATION_VECTORS
        rows.append({
            "batch_id": batch_id,
            "batch_key": f"{batch_id:03d}",
            "legal_sites": sites,
            "fault_instances": faults,
            "validation_vectors": EXPECTED_VALIDATION_VECTORS,
            "baseline_records": baseline,
            "enabled_records": enabled,
            "total_records": baseline + enabled,
        })
    require(sum(item["legal_sites"] for item in rows) == EXPECTED_LEGAL_SITES, "batch legal-site closure")
    require(sum(item["fault_instances"] for item in rows) == EXPECTED_FAULT_INSTANCES, "batch fault closure")
    require(sum(item["baseline_records"] for item in rows) == EXPECTED_VALIDATION_BASELINE_RECORDS, "batch baseline closure")
    require(sum(item["enabled_records"] for item in rows) == EXPECTED_VALIDATION_ENABLED_RECORDS, "batch enabled closure")
    require(sum(item["total_records"] for item in rows) == EXPECTED_VALIDATION_TOTAL_RECORDS, "batch total closure")
    return rows


def verify_campaign_foundation() -> dict:
    legal = load_json(LEGAL_SITE_FREEZE)
    generator = load_json(GENERATOR_LOCK)
    generation = load_json(GENERATION_INDEX)
    semantic = load_json(SEMANTIC_MANIFEST)
    runner = load_json(RUNNER_SOURCE_FREEZE)

    require(legal.get("status") == "PASS", "legal-site freeze status")
    require(legal.get("legal_sites") == EXPECTED_LEGAL_SITES, "legal-site count")
    require(legal.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "fault-instance count")
    require(legal.get("sa0_instances") == EXPECTED_LEGAL_SITES, "SA0 instance count")
    require(legal.get("sa1_instances") == EXPECTED_LEGAL_SITES, "SA1 instance count")

    require(generator.get("status") == "FROZEN", "batch-generator lock status")
    partition = generator.get("partition_contract")
    require(isinstance(partition, dict), "batch-generator partition contract")
    require(partition.get("total_batches") == EXPECTED_BATCHES, "generator batch count")
    require(partition.get("legal_sites") == EXPECTED_LEGAL_SITES, "generator legal-site count")
    require(partition.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "generator fault count")
    require(partition.get("batch_size") == 512, "generator batch size")
    require(partition.get("fault_models_per_site") == 2, "generator fault models per site")

    require(generation.get("status") == "GENERATED_AND_STRUCTURALLY_CHECKED", "canonical generation status")
    require(generation.get("batches") == EXPECTED_BATCHES, "generated batch count")
    require(generation.get("full_batches") == EXPECTED_FULL_BATCHES, "generated full-batch count")
    require(generation.get("partial_batches") == 1, "generated partial-batch count")
    require(generation.get("legal_sites") == EXPECTED_LEGAL_SITES, "generation site count")
    require(generation.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "generation fault count")
    require(generation.get("site_gaps") == 0, "generation site gaps")
    require(generation.get("duplicate_site_ids") == 0, "generation duplicate site IDs")
    require(generation.get("duplicate_site_indices") == 0, "generation duplicate site indices")
    require(generation.get("structural_checks") == "45/45 PASS", "generation structural checks")

    require(semantic.get("status") == "PASS", "semantic manifest status")
    require(semantic.get("batches") == EXPECTED_BATCHES, "semantic batch count")
    require(semantic.get("legal_sites") == EXPECTED_LEGAL_SITES, "semantic site count")
    require(semantic.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "semantic fault count")
    require(semantic.get("mapping_mismatches") == 0, "semantic mapping mismatches")
    require(semantic.get("routing_mismatches") == 0, "semantic routing mismatches")
    require(semantic.get("raw_monitor_mismatches") == 0, "semantic raw-monitor mismatches")
    require(semantic.get("golden_restoration_mismatches") == 0, "semantic restoration mismatches")
    require(semantic.get("site_gaps") == 0, "semantic site gaps")
    require(semantic.get("duplicate_sites") == 0, "semantic duplicate sites")

    require(runner.get("status") == "PASS", "runner source-freeze status")
    require(runner.get("execution") == "SEQUENTIAL", "runner execution mode")
    require(runner.get("parallel_batches") == 1, "runner parallel batches")
    require(runner.get("build_jobs") == 1, "runner build jobs")
    require(runner.get("checkpoint_after_each_batch") is True, "runner checkpointing")
    require(runner.get("resume_supported") is True, "runner resume support")
    require(runner.get("validation_vectors_exposed") == 0, "runner VALIDATION exposure")
    require(runner.get("holdout_vectors_exposed") == 0, "runner HOLDOUT exposure")
    return {
        "legal_site_freeze": legal,
        "legal_site_count_field": "legal_sites",
        "legal_fault_count_field": "persistent_fault_instances",
        "generator_lock": generator,
        "generator_status_field": "status",
        "generator_batch_field": "partition_contract.total_batches",
        "generation_index": generation,
        "generation_batch_field": "batches",
        "generation_site_field": "legal_sites",
        "generation_fault_field": "persistent_fault_instances",
        "semantic_manifest": semantic,
        "semantic_batch_field": "batches",
        "mapping_mismatch_field": "mapping_mismatches",
        "routing_mismatch_field": "routing_mismatches",
        "runner_source_freeze": runner,
        "runner_execution_field": "execution",
        "runner_jobs_field": "build_jobs",
        "runner_resume_field": "resume_supported",
    }


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-3A script in project root")
    verify_outputs_absent()
    verify_contract_only_source()

    print("STAGE 11D-3A — VALIDATION CAMPAIGN ARCHITECTURE AND EXECUTION CONTRACT")
    print("FROZEN INPUT VERIFICATION", flush=True)
    inputs = {
        relative(path): verify_input(path, expected)
        for path, expected in EXPECTED_INPUTS.items()
    }
    model_governance = verify_final_model_governance()
    campaign_foundation = verify_campaign_foundation()
    validation_rows, validation_commitment, split_counts = freeze_validation_vector_commitments()
    batches = batch_plan()

    print("  Final development model governance                                        : PASS")
    print("  Validation vector split and commitment                                    : PASS")
    print("  Canonical fault-batch foundation                                          : PASS")
    print("  Validation simulation or model inference                                  : NOT PERFORMED")
    print("  HOLDOUT_TEST payload emitted                                              : NO")

    created_at = datetime.now(timezone.utc).isoformat()
    acceptance = {
        "integrity_requirements": {
            "batches_complete": EXPECTED_BATCHES,
            "legal_sites_complete": EXPECTED_LEGAL_SITES,
            "fault_instances_complete": EXPECTED_FAULT_INSTANCES,
            "validation_vectors_complete": EXPECTED_VALIDATION_VECTORS,
            "missing_or_duplicate_enabled_records": 0,
            "unknown_records": 0,
            "detected_without_activity_records": 0,
            "baseline_digest_mismatches": 0,
            "baseline_latency_mismatches": 0,
            "baseline_timeouts": 0,
        },
        "hybrid_minimum_validity": {
            "mcc_gt": 0.0,
            "balanced_accuracy_gt": 0.5,
            "pr_auc_gt_positive_prevalence": True,
            "both_predicted_classes_required": True,
        },
        "hybrid_absolute_performance": {
            "mcc_minimum": 0.40,
            "balanced_accuracy_minimum": 0.70,
            "f1_minimum": 0.70,
            "recall_minimum": 0.70,
        },
        "hybrid_generalization_retention": {
            "frozen_dev_site_test_mcc": EXPECTED_HYBRID_MCC,
            "maximum_allowed_mcc_drop": 0.10,
            "validation_mcc_minimum": MINIMUM_VALIDATION_MCC,
        },
        "hybrid_comparator_confirmation": {
            "primary_comparator": "DIR_SGC_11D1D",
            "hybrid_minus_gnn_mcc_minimum": REQUIRED_HYBRID_GAIN_OVER_GNN,
            "paired_physical_site_mcc_delta_ci_lower_must_exceed": 0.0,
        },
        "failure_disposition": (
            "FREEZE_VALIDATION_RESULT_AS_NOT_MET; DO_NOT RETRAIN, RESELECT, "
            "CHANGE THRESHOLDS, OR OPEN HOLDOUT_TEST"
        ),
    }

    atomic_json(VALIDATION_ARCHITECTURE, {
        "stage": "11D-3A",
        "title": "VALIDATION CAMPAIGN ARCHITECTURE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "created_at_utc": created_at,
        "task": "BLIND VALIDATION OF FROZEN PRE-SIMULATION BINARY DETECTABILITY MODELS",
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY",
        "final_development_model": "HYBRID_FUSION_MLP_11D2C",
        "frozen_comparators": [
            "DIR_SGC_11D1D",
            "DEEP_MLP_11C5K",
            "CONVENTIONAL_LOGREG_11C5H",
        ],
        "model_artifact": record(FINAL_MODEL),
        "final_model_lock": record(FINAL_MODEL_LOCK),
        "vector_partition": "VALIDATION",
        "validation_vectors": EXPECTED_VALIDATION_VECTORS,
        "ordered_validation_commitment": validation_commitment,
        "validation_commitment_manifest": record(VECTOR_COMMITMENTS_JSON),
        "holdout_partition": "SEALED_AND_PROHIBITED",
        "batches": EXPECTED_BATCHES,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "baseline_records": EXPECTED_VALIDATION_BASELINE_RECORDS,
        "enabled_records": EXPECTED_VALIDATION_ENABLED_RECORDS,
        "total_records": EXPECTED_VALIDATION_TOTAL_RECORDS,
        "cycles_per_baseline_transaction": EXPECTED_CYCLES,
        "data_flow": [
            "FROZEN_VALIDATION_VECTOR_POOL",
            "CANONICAL_INSTRUMENTED_FAULT_BATCHES",
            "RESUMABLE_SEQUENTIAL_SIMULATION",
            "LEAKAGE_SAFE_FEATURE_TRANSFORMATION_WITH_FROZEN_SCHEMA",
            "FROZEN_FOUR_MODEL_INFERENCE",
            "PHYSICAL_SITE_GROUPED_COMPARATOR_ANALYSIS",
            "IMMUTABLE_VALIDATION_RESULT_FREEZE",
        ],
        "batch_plan": batches,
        "acceptance_contract": acceptance,
        "model_training_permitted": False,
        "candidate_selection_permitted": False,
        "threshold_selection_permitted": False,
        "feature_or_graph_refitting_permitted": False,
        "holdout_access_permitted": False,
    })

    atomic_json(EXECUTION_CONTRACT, {
        "stage": "11D-3A",
        "title": "VALIDATION CAMPAIGN EXECUTION CONTRACT",
        "status": "PASS",
        "contract_status": "FROZEN",
        "created_at_utc": created_at,
        "execution_mode": "SEQUENTIAL",
        "parallel_batches": 1,
        "build_jobs": 1,
        "checkpoint_after_each_batch": True,
        "resume_supported": True,
        "overwrite_default": False,
        "minimum_free_disk_gib": 5,
        "estimated_execution_time": "2-4 HOURS",
        "batches": EXPECTED_BATCHES,
        "validation_vectors": EXPECTED_VALIDATION_VECTORS,
        "baseline_records": EXPECTED_VALIDATION_BASELINE_RECORDS,
        "enabled_records": EXPECTED_VALIDATION_ENABLED_RECORDS,
        "total_records": EXPECTED_VALIDATION_TOTAL_RECORDS,
        "expected_batch_records": batches,
        "required_runner_controls": [
            "EXACT_FROZEN_INPUT_SHA_VERIFICATION",
            "VALIDATION_ONLY_VECTOR_FILTER",
            "NO_HOLDOUT_VECTOR_SERIALIZATION",
            "CANONICAL_BATCH_MAPPING_VERIFICATION",
            "ONE_BUILD_JOB",
            "PER_BATCH_CHECKPOINT_AND_RESUME",
            "CSV_SCHEMA_AND_RECORD_CLOSURE",
            "NO_UNKNOWN_OR_DETECTED_WITHOUT_ACTIVITY",
            "DETERMINISTIC_REPLAY_CANARY",
        ],
        "runner_generation_authorized": True,
        "runner_dry_run_authorized": True,
        "validation_campaign_execution_authorized": False,
        "validation_dataset_consolidation_authorized": False,
        "validation_model_inference_authorized": False,
        "holdout_campaign_contract_authorized": False,
        "holdout_campaign_execution_authorized": False,
        "validation_vectors_exposed_to_training": 0,
        "holdout_vectors_exposed": 0,
        "acceptance_contract": acceptance,
        "next_gate": "STAGE 11D-3B — VALIDATION CAMPAIGN RUNNER GENERATION AND DRY-RUN FREEZE",
    })

    outputs = {
        "validation_architecture": record(VALIDATION_ARCHITECTURE),
        "execution_contract": record(EXECUTION_CONTRACT),
        "validation_vector_commitments_csv": record(VECTOR_COMMITMENTS_CSV),
        "validation_vector_commitments_json": record(VECTOR_COMMITMENTS_JSON),
    }
    atomic_json(AUDIT, {
        "stage": "11D-3A",
        "title": "VALIDATION CAMPAIGN ARCHITECTURE AND EXECUTION-CONTRACT FREEZE",
        "status": "PASS",
        "architecture_status": "FROZEN",
        "execution_contract_status": "FROZEN",
        "validation_commitment_status": "FROZEN",
        "created_at_utc": created_at,
        "final_development_model": "HYBRID_FUSION_MLP_11D2C",
        "validation_vectors_committed": EXPECTED_VALIDATION_VECTORS,
        "validation_vector_payloads_emitted": 0,
        "holdout_vectors_exposed": 0,
        "vector_split_counts": split_counts,
        "ordered_validation_commitment": validation_commitment,
        "batches": EXPECTED_BATCHES,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "baseline_records": EXPECTED_VALIDATION_BASELINE_RECORDS,
        "enabled_records": EXPECTED_VALIDATION_ENABLED_RECORDS,
        "total_records": EXPECTED_VALIDATION_TOTAL_RECORDS,
        "execution": "SEQUENTIAL",
        "parallel_batches": 1,
        "build_jobs": 1,
        "checkpoint_resume": "ENABLED",
        "runner_generation": "AUTHORIZED",
        "runner_dry_run": "AUTHORIZED",
        "validation_campaign_execution": "NOT_YET_AUTHORIZED",
        "validation_dataset_consolidation": "NOT_YET_AUTHORIZED",
        "validation_model_inference": "NOT_YET_AUTHORIZED",
        "holdout_campaign": "NOT_YET_AUTHORIZED",
        "model_training_or_inference_performed": False,
        "acceptance_contract": acceptance,
        "input_evidence": inputs,
        "model_governance": {
            "disposition_status": model_governance["disposition"]["policy_status"],
            "model_lock_status": model_governance["model_lock"]["lock_status"],
            "readiness_status": model_governance["readiness"]["readiness_status"],
        },
        "campaign_foundation": {
            "legal_site_count_field": campaign_foundation["legal_site_count_field"],
            "legal_fault_count_field": campaign_foundation["legal_fault_count_field"],
            "generator_status": "FROZEN",
            "generator_status_field": campaign_foundation["generator_status_field"],
            "generator_batch_field": campaign_foundation["generator_batch_field"],
            "generation_status": "PASS",
            "generation_batch_field": campaign_foundation["generation_batch_field"],
            "generation_site_field": campaign_foundation["generation_site_field"],
            "generation_fault_field": campaign_foundation["generation_fault_field"],
            "semantic_status": "PASS",
            "semantic_batch_field": campaign_foundation["semantic_batch_field"],
            "mapping_mismatch_field": campaign_foundation["mapping_mismatch_field"],
            "routing_mismatch_field": campaign_foundation["routing_mismatch_field"],
            "runner_status": "PASS",
            "runner_execution_field": campaign_foundation["runner_execution_field"],
            "runner_jobs_field": campaign_foundation["runner_jobs_field"],
            "runner_resume_field": campaign_foundation["runner_resume_field"],
        },
        "outputs": outputs,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_fault_batches_modified": False,
        "vector_pool_modified": False,
        "final_model_modified": False,
        "frozen_comparator_models_modified": False,
        "next_gate": "STAGE 11D-3B — VALIDATION CAMPAIGN RUNNER GENERATION AND DRY-RUN FREEZE",
    })
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-3A — VALIDATION CAMPAIGN ARCHITECTURE AND EXECUTION-CONTRACT FREEZE")
    print("Status                         : PASS")
    print("Architecture status            : FROZEN")
    print("Execution contract status      : FROZEN")
    print("Validation commitment status   : FROZEN")
    print("Final development model        : HYBRID FUSION MLP")
    print("Validation vectors committed   :", len(validation_rows))
    print("Validation payloads emitted    : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Batches                        :", EXPECTED_BATCHES)
    print("Legal sites                    :", EXPECTED_LEGAL_SITES)
    print("Persistent fault instances     :", EXPECTED_FAULT_INSTANCES)
    print("Baseline records               :", EXPECTED_VALIDATION_BASELINE_RECORDS)
    print("Enabled validation records     :", EXPECTED_VALIDATION_ENABLED_RECORDS)
    print("Total validation records       :", EXPECTED_VALIDATION_TOTAL_RECORDS)
    print("Execution                      : SEQUENTIAL")
    print("Parallel batches               : 1")
    print("Build jobs                     : 1")
    print("Checkpoint/resume              : ENABLED")
    print("Minimum free disk              : 5 GiB")
    print("Estimated execution time       : 2-4 HOURS")
    print("Frozen validation MCC minimum  :", f"{MINIMUM_VALIDATION_MCC:.8f}")
    print("Required hybrid gain over GNN  : +0.02 MCC WITH POSITIVE 95% SITE CI")
    print("Runner generation              : AUTHORIZED")
    print("Runner dry run                 : AUTHORIZED")
    print("Validation execution           : NOT YET AUTHORIZED")
    print("Validation model inference     : NOT YET AUTHORIZED")
    print("Holdout campaign               : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Final model modified           : NO")
    print("Validation architecture        :", VALIDATION_ARCHITECTURE)
    print("Validation architecture SHA    :", outputs["validation_architecture"]["sha256"])
    print("Execution contract             :", EXECUTION_CONTRACT)
    print("Execution contract SHA         :", outputs["execution_contract"]["sha256"])
    print("Vector commitment manifest     :", VECTOR_COMMITMENTS_JSON)
    print("Vector commitment SHA          :", outputs["validation_vector_commitments_json"]["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-3B — VALIDATION CAMPAIGN RUNNER GENERATION AND DRY-RUN FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
