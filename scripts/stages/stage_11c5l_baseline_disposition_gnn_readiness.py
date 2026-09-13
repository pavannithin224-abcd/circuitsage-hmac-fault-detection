#!/usr/bin/env python3
"""Freeze diagnostic comparator disposition and authorize GNN graph construction."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-BASELINE-DISPOSITION-GNN-READINESS-v1"

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11c5"
CONFIG_ROOT = ROOT / "config/diagnostic_model"

GOLDEN_NETLIST = ROOT / "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json"
LEGAL_SITES = ROOT / "results/hmac_fault_campaign_11c1/hmac_legal_fault_sites_11c1b.json"
CANONICAL_SCHEMA = (
    RESULT_ROOT
    / "canonical_dataset_11c5c/hmac_canonical_dataset_schema_11c5c.json"
)
FEATURE_SCHEMA = (
    RESULT_ROOT
    / "feature_matrix_11c5e/hmac_leakage_safe_feature_matrix_schema_11c5e.json"
)
TASK_SPLIT_POLICY = (
    ROOT
    / "config/diagnostic_model/hmac_diagnostic_task_feature_split_policy_11c5d.json"
)
SITE_SPLIT = RESULT_ROOT / "hmac_fault_site_group_split_11c5d.csv"
SITE_SPLIT_MANIFEST = RESULT_ROOT / "hmac_fault_site_group_split_manifest_11c5d.json"

BASELINE_EVALUATOR = ROOT / "stage_11c5h_locked_site_test.py"
BASELINE_EVAL_LOCK = (
    RESULT_ROOT
    / "baseline_evaluation_11c5h/hmac_conventional_baseline_site_test_evaluation_lock_11c5h.json"
)
BASELINE_MANIFEST = RESULT_ROOT / "hmac_conventional_baseline_site_test_manifest_11c5h.json"
BASELINE_AUDIT = RESULT_ROOT / "hmac_conventional_baseline_site_test_freeze_11c5h.json"

DEEP_CONTRACT_SOURCE = ROOT / "stage_11c5i_deep_model_contract.py"
DEEP_ARCHITECTURE = ROOT / "config/diagnostic_model/hmac_deep_diagnostic_architecture_11c5i.json"
DEEP_TRAINING_CONTRACT = ROOT / "config/diagnostic_model/hmac_deep_diagnostic_training_contract_11c5i.json"
DEEP_CONTRACT_AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_architecture_training_contract_freeze_11c5i.json"

DEEP_TRAINER = ROOT / "stage_11c5j_deep_model_train.py"
DEEP_MODEL = (
    RESULT_ROOT
    / "deep_training_11c5j/hmac_selected_deep_diagnostic_model_11c5j.joblib"
)
DEEP_SELECTION_LOCK = (
    RESULT_ROOT
    / "deep_training_11c5j/hmac_deep_diagnostic_selection_lock_11c5j.json"
)
DEEP_TRAINING_MANIFEST = RESULT_ROOT / "hmac_deep_diagnostic_training_manifest_11c5j.json"
DEEP_TRAINING_AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_training_calibration_freeze_11c5j.json"

DEEP_EVALUATOR = ROOT / "stage_11c5k_locked_deep_site_test.py"
DEEP_COMPARISON = (
    RESULT_ROOT
    / "deep_evaluation_11c5k/hmac_deep_vs_conventional_baseline_comparison_11c5k.json"
)
DEEP_EVAL_LOCK = (
    RESULT_ROOT
    / "deep_evaluation_11c5k/hmac_deep_diagnostic_site_test_evaluation_lock_11c5k.json"
)
DEEP_EVAL_MANIFEST = RESULT_ROOT / "hmac_deep_diagnostic_site_test_manifest_11c5k.json"
DEEP_EVAL_AUDIT = RESULT_ROOT / "hmac_deep_diagnostic_site_test_comparison_freeze_11c5k.json"

DISPOSITION_POLICY = (
    CONFIG_ROOT / "hmac_diagnostic_baseline_disposition_policy_11c5l.json"
)
GNN_READINESS_POLICY = CONFIG_ROOT / "hmac_gnn_readiness_policy_11c5l.json"
COMPARATOR_REGISTRY_CSV = (
    RESULT_ROOT / "hmac_frozen_diagnostic_comparator_registry_11c5l.csv"
)
COMPARATOR_REGISTRY_JSON = (
    RESULT_ROOT / "hmac_frozen_diagnostic_comparator_registry_11c5l.json"
)
AUDIT = (
    RESULT_ROOT
    / "hmac_diagnostic_baseline_disposition_gnn_readiness_freeze_11c5l.json"
)

EXPECTED_INPUTS = {
    GOLDEN_NETLIST: "0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1",
    LEGAL_SITES: "d9a39366bb35a277125376c6dc26c83273b45c0d7732fb93a3403401a071cb45",
    CANONICAL_SCHEMA: "2a08594f9573ff833a872d409783cee984bbb7499c7b013af4c72377ede3f772",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    TASK_SPLIT_POLICY: "74243fa3b40f047a9f25a23c87eaccf6ac5a71a07241efef58df6b9e8e5322be",
    SITE_SPLIT: "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    SITE_SPLIT_MANIFEST: "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
    BASELINE_EVALUATOR: "8f720d2f5a2eff1d964507424af502fba9aca60abf09d50fb897d33075d290e3",
    BASELINE_EVAL_LOCK: "5ff4987169c2b885826db13cf721164b65e009a83290d9aa21da9eeb13b213c4",
    BASELINE_MANIFEST: "cfe75842e7b081230c29d5162ec0292381e574b8679f58118c9a26bf6a7cc1ef",
    BASELINE_AUDIT: "54d86a12e605ce8f2b920c55e87c6ad9a3e30b41df8420730adfff4be4fbee19",
    DEEP_CONTRACT_SOURCE: "a58ae217b3f2f1c78627d93650b2e18b75f3708a5e7143842a0a99a7b05dbca6",
    DEEP_ARCHITECTURE: "fad4bf5caf653f1e7712d16114ca8a9aa85982a992e21e19ee2c09606583ce6f",
    DEEP_TRAINING_CONTRACT: "e6175b3921737ec78bf88a1d8020b17ab2e37f44137da211dd20414e6dbb0894",
    DEEP_CONTRACT_AUDIT: "0ce03891c718f62e1bf47746db298a6338582997727eb9d862b6f7b829f6c05c",
    DEEP_TRAINER: "9101eab7fc704ddd7baa3b8619a95f967eef5bec0f2ee96926eb9608e13c4470",
    DEEP_MODEL: "6e25e901f7703e22c069ab21b15e1e526db6bfab79f4d205d9e36fea4cafb7fa",
    DEEP_SELECTION_LOCK: "12319c2d3b067ca04862209512aad9e42b35be97068557d03bc1f1906b4f024d",
    DEEP_TRAINING_MANIFEST: "48c4fe476843a821a565af70de8d9f30ef72b0bd9dfa48f53add85b8cf191551",
    DEEP_TRAINING_AUDIT: "d9d10c4781ade4aff544f57012ae1e8218717f13242f003126e90bcb9d5ade84",
    DEEP_EVALUATOR: "51f23c9e5b27af8463aeb2b902e7c9ef90d553a47e9b81fdde6c81943cd86d7c",
    DEEP_COMPARISON: "47094ad6b03ca97d1aa3726e8b0e6a017667c532ea24f4aa49a3e54e9c3b9865",
    DEEP_EVAL_LOCK: "678be80a8e9077136fd4e66b4c6863b263f96883294cc700d61b215bc3209250",
    DEEP_EVAL_MANIFEST: "57ca6ec6fec634b7fe09f07631648353c29ed8d4b84559fe029f1ee7f418080f",
    DEEP_EVAL_AUDIT: "f5df0c081c75a66178a362ccbfb6cfdeb612634e6d2292891307a2826db8aab8",
}

EXPECTED_BASELINE_MCC = 0.14208149
EXPECTED_DEEP_MCC = 0.14832380
EXPECTED_MCC_DELTA = 0.00624231
EXPECTED_DELTA_CI = (-0.00568682, 0.01786846)
EXPECTED_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_FEATURES = 527
EXPECTED_DEV_TRAIN_SITES = 15_987
EXPECTED_DEV_CALIBRATION_SITES = 3_426
EXPECTED_DEV_SITE_TEST_SITES = 3_426


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
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


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


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {path}")
    actual_sha = sha256(path)
    require(
        actual_sha == expected_sha,
        f"SHA mismatch for {path}: expected {expected_sha}, actual {actual_sha}",
    )
    print(f"  {path.name:<72}: OK")
    return record(path)


def close(a: float, b: float, tolerance: float = 5e-8) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def verify_semantics() -> tuple[dict, dict, dict]:
    baseline_audit = load_json(BASELINE_AUDIT)
    comparison = load_json(DEEP_COMPARISON)
    deep_audit = load_json(DEEP_EVAL_AUDIT)

    require(baseline_audit.get("status") == "PASS", "baseline evaluation status")
    require(baseline_audit.get("evaluation_status") == "FROZEN", "baseline evaluation freeze")
    require(close(baseline_audit.get("mcc", -1), EXPECTED_BASELINE_MCC), "baseline MCC")
    require(baseline_audit.get("project_target_status") == "NOT_MET", "baseline project target")

    require(comparison.get("status") == "PASS", "deep comparison status")
    require(comparison.get("comparison_status") == "FROZEN", "deep comparison freeze")
    require(comparison.get("primary_metric") == "MCC", "comparison primary metric")
    require(close(comparison.get("deep_metrics", {}).get("mcc", -1), EXPECTED_DEEP_MCC), "deep MCC")
    require(close(comparison.get("baseline_metrics", {}).get("mcc", -1), EXPECTED_BASELINE_MCC), "comparison baseline MCC")
    require(close(comparison.get("mcc_delta", -1), EXPECTED_MCC_DELTA), "MCC delta")
    interval = comparison.get("mcc_delta_ci_95", [])
    require(len(interval) == 2, "MCC delta interval")
    require(close(interval[0], EXPECTED_DELTA_CI[0]), "MCC delta lower bound")
    require(close(interval[1], EXPECTED_DELTA_CI[1]), "MCC delta upper bound")
    require(interval[0] <= 0.0 <= interval[1], "paired MCC interval must cross zero")
    require(comparison.get("point_outcome") == "DEEP_OUTPERFORMS_BASELINE", "point comparison")
    require(comparison.get("statistically_supported_mcc_improvement") is False, "statistical disposition")
    require(comparison.get("required_mcc_improvement_met") is False, "required MCC improvement")

    require(deep_audit.get("status") == "PASS", "deep evaluation status")
    require(deep_audit.get("evaluation_status") == "FROZEN", "deep evaluation freeze")
    require(deep_audit.get("project_target_status") == "NOT_MET", "deep project target")
    require(deep_audit.get("gnn_training_authorized") is False, "prior GNN training gate")
    require(deep_audit.get("hybrid_training_authorized") is False, "prior hybrid training gate")
    require(deep_audit.get("validation_vectors_exposed") == 0, "validation exposure")
    require(deep_audit.get("holdout_vectors_exposed") == 0, "holdout exposure")
    return baseline_audit, comparison, deep_audit


def verify_site_split() -> dict[str, int]:
    counts: dict[str, int] = {}
    sites: set[str] = set()
    fault_sites: set[str] = set()
    with SITE_SPLIT.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "site split header")
        split_field = next(
            (name for name in ("partition", "split", "site_partition") if name in reader.fieldnames),
            None,
        )
        site_field = next(
            (name for name in ("fault_site_id", "site_id", "physical_site_id") if name in reader.fieldnames),
            None,
        )
        require(split_field is not None, "site split partition field")
        require(site_field is not None, "site split site-id field")
        for row in reader:
            site_id = row[site_field]
            require(site_id not in sites, f"duplicate physical site in split: {site_id}")
            sites.add(site_id)
            fault_sites.add(site_id)
            partition = row[split_field]
            counts[partition] = counts.get(partition, 0) + 1
    require(len(fault_sites) == EXPECTED_SITES, "physical site split count")
    require(counts.get("DEV_TRAIN") == EXPECTED_DEV_TRAIN_SITES, "DEV_TRAIN site count")
    require(counts.get("DEV_CALIBRATION") == EXPECTED_DEV_CALIBRATION_SITES, "DEV_CALIBRATION site count")
    require(counts.get("DEV_SITE_TEST") == EXPECTED_DEV_SITE_TEST_SITES, "DEV_SITE_TEST site count")
    return counts


def verify_outputs_absent() -> None:
    for path in (
        DISPOSITION_POLICY,
        GNN_READINESS_POLICY,
        COMPARATOR_REGISTRY_CSV,
        COMPARATOR_REGISTRY_JSON,
        AUDIT,
    ):
        require(not path.exists(), f"refusing to overwrite Stage 11C-5L output: {path}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        DISPOSITION_POLICY,
        GNN_READINESS_POLICY,
        COMPARATOR_REGISTRY_CSV,
        COMPARATOR_REGISTRY_JSON,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11C-5L script in project root")
    verify_outputs_absent()

    print("STAGE 11C-5L — DIAGNOSTIC BASELINE DISPOSITION AND GNN READINESS")
    print("FROZEN INPUT VERIFICATION", flush=True)
    inputs = {
        relative(path): verify_input(path, expected_sha)
        for path, expected_sha in EXPECTED_INPUTS.items()
    }
    baseline_audit, comparison, deep_audit = verify_semantics()
    split_counts = verify_site_split()
    print("  Prior-stage semantic contracts                                           : PASS")
    print("  Physical-site split integrity                                           : PASS")

    created_at = datetime.now(timezone.utc).isoformat()
    baseline_metrics = comparison["baseline_metrics"]
    deep_metrics = comparison["deep_metrics"]
    interval = comparison["mcc_delta_ci_95"]

    comparator_rows = [
        {
            "comparator_id": "CONVENTIONAL_LOGREG_11C5H",
            "family": "INCREMENTAL_LOGISTIC_REGRESSION",
            "role": "PRIMARY_SIMPLE_BASELINE",
            "candidate": baseline_audit["candidate_id"],
            "threshold": format(float(baseline_audit["threshold"]), ".17g"),
            "mcc": format(float(baseline_metrics["mcc"]), ".17g"),
            "balanced_accuracy": format(float(baseline_metrics["balanced_accuracy"]), ".17g"),
            "pr_auc": format(float(baseline_metrics["pr_auc"]), ".17g"),
            "roc_auc": format(float(baseline_metrics["roc_auc"]), ".17g"),
            "status": "FROZEN_COMPARATOR",
            "selection_winner": "NO_WINNER_DECLARED",
        },
        {
            "comparator_id": "DEEP_MLP_11C5K",
            "family": "FEEDFORWARD_MULTILAYER_PERCEPTRON",
            "role": "SECONDARY_DEEP_COMPARATOR",
            "candidate": deep_audit["deep_candidate"],
            "threshold": format(float(deep_audit["deep_threshold"]), ".17g"),
            "mcc": format(float(deep_metrics["mcc"]), ".17g"),
            "balanced_accuracy": format(float(deep_metrics["balanced_accuracy"]), ".17g"),
            "pr_auc": format(float(deep_metrics["pr_auc"]), ".17g"),
            "roc_auc": format(float(deep_metrics["roc_auc"]), ".17g"),
            "status": "FROZEN_COMPARATOR",
            "selection_winner": "NO_WINNER_DECLARED",
        },
    ]
    registry_fields = list(comparator_rows[0])
    atomic_csv(COMPARATOR_REGISTRY_CSV, registry_fields, comparator_rows)
    atomic_json(
        COMPARATOR_REGISTRY_JSON,
        {
            "stage": "11C-5L",
            "status": "PASS",
            "registry_status": "FROZEN",
            "primary_metric": "MCC",
            "selection_partition": "DEV_SITE_TEST",
            "winner_declared": False,
            "disposition": "STATISTICAL_TIE_FOR_GOVERNANCE",
            "reason": "Paired physical-site 95% MCC-delta interval includes zero.",
            "comparators": comparator_rows,
        },
    )

    disposition = {
        "stage": "11C-5L",
        "title": "DIAGNOSTIC BASELINE DISPOSITION POLICY",
        "status": "PASS",
        "policy_status": "FROZEN",
        "created_at_utc": created_at,
        "primary_task": "PRE-SIMULATION BINARY DETECTABILITY",
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY",
        "primary_metric": "MCC",
        "primary_simple_baseline": "CONVENTIONAL_LOGREG_11C5H",
        "secondary_deep_comparator": "DEEP_MLP_11C5K",
        "winner_declared": False,
        "point_outcome": comparison["point_outcome"],
        "baseline_mcc": baseline_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "mcc_delta_deep_minus_baseline": comparison["mcc_delta"],
        "mcc_delta_ci_95": interval,
        "statistically_supported_improvement": False,
        "required_mcc_improvement": comparison["required_mcc_improvement"],
        "required_mcc_improvement_met": False,
        "project_target_status": "NOT_MET",
        "governance_disposition": "RETAIN_BOTH_AS_FROZEN_COMPARATORS",
        "retraining_authorized": False,
        "threshold_change_authorized": False,
        "dev_site_test_state": "CONSUMED_AND_FROZEN",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "comparator_registry_csv": record(COMPARATOR_REGISTRY_CSV),
        "comparator_registry_json": record(COMPARATOR_REGISTRY_JSON),
    }
    atomic_json(DISPOSITION_POLICY, disposition)

    gnn_policy = {
        "stage": "11C-5L",
        "title": "GNN READINESS POLICY",
        "status": "PASS",
        "readiness_status": "READY_FOR_GRAPH_DATASET_CONSTRUCTION_ONLY",
        "created_at_utc": created_at,
        "graph_source": record(GOLDEN_NETLIST),
        "node_universe": {
            "definition": "LEGAL_PERSISTENT_CELL_OUTPUT_NET_STEMS",
            "legal_physical_sites": EXPECTED_SITES,
            "fault_instances": EXPECTED_FAULT_INSTANCES,
            "legal_sites_artifact": record(LEGAL_SITES),
        },
        "edge_contract": {
            "direction": "DRIVER_TO_SINK",
            "source": "FROZEN_GOLDEN_NETLIST_CONNECTIVITY",
            "parallel_edges": "COLLAPSE_WITH_MULTIPLICITY_ATTRIBUTE",
            "self_loops": "RECORD_AND_CLASSIFY",
            "clock_reset_edges": "RETAIN_WITH_EDGE_ROLE_ATTRIBUTE",
            "exact_edge_count": "PENDING_TOPOLOGY_CONSTRUCTION",
        },
        "sample_contract": {
            "task": "PRE-SIMULATION_BINARY_DETECTABILITY",
            "persistent_fault_models": ["SA0", "SA1"],
            "transient_faults_in_scope": False,
            "model_features": EXPECTED_FEATURES,
            "post_simulation_features": 0,
            "target_present_in_input_features": False,
        },
        "split_contract": {
            "unit": "PHYSICAL_FAULT_SITE",
            "sa0_sa1_grouped": True,
            "split_counts": split_counts,
            "overlapping_sites": 0,
            "site_split": record(SITE_SPLIT),
            "site_split_manifest": record(SITE_SPLIT_MANIFEST),
        },
        "authorized_actions": [
            "BUILD_GRAPH_FROM_FROZEN_GOLDEN_NETLIST",
            "MAP_ALL_LEGAL_SITES_TO_GRAPH_NODES",
            "ATTACH_LEAKAGE_SAFE_PRE_SIMULATION_FEATURES",
            "VERIFY_TOPOLOGY_AND_SPLIT_INTEGRITY",
            "FREEZE_GRAPH_DATASET_AND_SCHEMA",
        ],
        "prohibited_actions": [
            "FIT_GNN_PARAMETERS",
            "SELECT_GNN_HYPERPARAMETERS",
            "SELECT_GNN_THRESHOLD",
            "TRAIN_HYBRID_MODEL",
            "EXPOSE_VALIDATION_VECTORS",
            "EXPOSE_HOLDOUT_VECTORS",
            "USE_POST_SIMULATION_FEATURES",
        ],
        "graph_dataset_construction_authorized": True,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
        "validation_campaign_authorized": False,
        "holdout_campaign_authorized": False,
        "next_gate": "STAGE 11D-1A — GOLDEN-NETLIST GRAPH DATASET CONSTRUCTION AND TOPOLOGY-INTEGRITY FREEZE",
    }
    atomic_json(GNN_READINESS_POLICY, gnn_policy)

    output_records = {
        "disposition_policy": record(DISPOSITION_POLICY),
        "gnn_readiness_policy": record(GNN_READINESS_POLICY),
        "comparator_registry_csv": record(COMPARATOR_REGISTRY_CSV),
        "comparator_registry_json": record(COMPARATOR_REGISTRY_JSON),
    }
    audit = {
        "stage": "11C-5L",
        "title": "DIAGNOSTIC BASELINE DISPOSITION AND GNN READINESS FREEZE",
        "status": "PASS",
        "disposition_status": "FROZEN",
        "gnn_readiness_status": "FROZEN",
        "comparator_registry_status": "FROZEN",
        "created_at_utc": created_at,
        "baseline_comparator": "FROZEN",
        "deep_comparator": "FROZEN",
        "winner_declared": False,
        "disposition": "RETAIN_BOTH_AS_FROZEN_COMPARATORS",
        "baseline_mcc": baseline_metrics["mcc"],
        "deep_mcc": deep_metrics["mcc"],
        "mcc_delta": comparison["mcc_delta"],
        "mcc_delta_ci_95": interval,
        "statistical_improvement": "NOT_SUPPORTED",
        "required_point_improvement": "NOT_MET",
        "project_target_status": "NOT_MET",
        "legal_graph_nodes_expected": EXPECTED_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "graph_dataset_construction": "AUTHORIZED",
        "topology_integrity_qualification": "REQUIRED",
        "gnn_training": "NOT_YET_AUTHORIZED",
        "hybrid_training": "NOT_YET_AUTHORIZED",
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "post_simulation_features": 0,
        "input_evidence": inputs,
        "outputs": output_records,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_dataset_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "deep_model_modified": False,
        "next_gate": "STAGE 11D-1A — GOLDEN-NETLIST GRAPH DATASET CONSTRUCTION AND TOPOLOGY-INTEGRITY FREEZE",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    print("\nSTAGE 11C-5L — DIAGNOSTIC BASELINE DISPOSITION AND GNN READINESS FREEZE")
    print("Status                         : PASS")
    print("Disposition status             : FROZEN")
    print("GNN readiness status           : FROZEN")
    print("Conventional baseline          : FROZEN PRIMARY SIMPLE COMPARATOR")
    print("Deep MLP                       : FROZEN SECONDARY COMPARATOR")
    print("Winner declared                : NO")
    print("Baseline MCC                   :", f"{baseline_metrics['mcc']:.8f}")
    print("Deep MCC                       :", f"{deep_metrics['mcc']:.8f}")
    print("MCC delta (deep-baseline)      :", f"{comparison['mcc_delta']:.8f}")
    print("MCC delta 95% paired-site CI   :", f"[{interval[0]:.8f}, {interval[1]:.8f}]")
    print("Statistical improvement        : NOT SUPPORTED")
    print("Required +0.05 improvement     : NOT MET")
    print("Project target                 : NOT MET")
    print("Graph source                   : FROZEN GOLDEN NETLIST")
    print("Expected graph nodes           :", EXPECTED_SITES)
    print("Persistent fault instances     :", EXPECTED_FAULT_INSTANCES)
    print("Graph dataset construction     : AUTHORIZED")
    print("Topology integrity freeze      : REQUIRED BEFORE TRAINING")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Baseline model modified        : NO")
    print("Deep model modified            : NO")
    print("Disposition policy             :", DISPOSITION_POLICY)
    print("Disposition policy SHA         :", output_records["disposition_policy"]["sha256"])
    print("GNN readiness policy           :", GNN_READINESS_POLICY)
    print("GNN readiness policy SHA       :", output_records["gnn_readiness_policy"]["sha256"])
    print("Comparator registry            :", COMPARATOR_REGISTRY_CSV)
    print("Comparator registry SHA        :", output_records["comparator_registry_csv"]["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-1A — GOLDEN-NETLIST GRAPH DATASET CONSTRUCTION AND TOPOLOGY-INTEGRITY FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
