#!/usr/bin/env python3
"""Verify frozen evidence and authorize validation simulation; never launch it.

Standard-library only. Model files and the vector pool are hashed, not executed.
Only the two already-used TRAIN canaries have their payload fields inspected.
Existing authorization/audit files are never overwritten. --check-only writes
no artifacts. A successful normal run creates the authorization last, after
all checks and the audit have completed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "HMAC-VALIDATION-AUTHORIZATION-VERIFIER-v1"
RUNNER_VERSION = "HMAC-VALIDATION-CAMPAIGN-RUNNER-v1"
C = "config/fault_campaign/"
D = "config/diagnostic_model/"
R = "results/hmac_fault_campaign_11d3/"
Q = R + "validation_runner_dry_run_11d3b/"
ARCH = C + "hmac_validation_campaign_architecture_11d3a.json"
CONTRACT = C + "hmac_validation_campaign_execution_contract_11d3a.json"
COMMITMENT = R + "hmac_validation_vector_commitments_11d3a.json"
AUDIT_A = R + "hmac_validation_campaign_architecture_execution_contract_freeze_11d3a.json"
RUNNER = "stage_11d3b_validation_runner.py"
LOCK_B = C + "hmac_validation_campaign_runner_lock_11d3b.json"
AUDIT_B = R + "hmac_validation_campaign_runner_dry_run_freeze_11d3b.json"
CSV_B = Q + "hmac_validation_runner_dry_run_results_11d3b.csv"
CSV_REPLAY = Q + "hmac_validation_runner_dry_run_replay_results_11d3b.csv"
TB_MANIFEST = Q + "hmac_validation_runner_dry_run_testbench_manifest_11d3b.json"
POOL = "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"
TRAIN_SELECTION = C + "hmac_campaign_vector_selection_11c3b_b.json"
SEMANTIC = "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json"
INDEX = "results/hmac_fault_campaign_11c4/hmac_canonical_batch_generation_index_11c4g_a1.json"
MODEL_LOCK = D + "hmac_final_diagnostic_model_lock_11d2d.json"
MODEL = "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"
AUTHORIZATION = C + "hmac_validation_campaign_execution_authorization_11d3c.json"
AUDIT = R + "hmac_validation_campaign_execution_authorization_freeze_11d3c.json"

# These are the exact reported frozen hashes, not recomputed expectations.
EXPECTED = {
    ARCH: "256aca0cba7d14439f985d42a0683912a0e92285282f2fa116eb43728ffe2f8d",
    CONTRACT: "55c3b2c57c8a6a7068812a41a75b01f75292ce0068c4553e3fd35fb972a87420",
    COMMITMENT: "9fff03e0dd00e91d405daee71e98af62408969407547308180475d90d20ba99b",
    AUDIT_A: "ad8d5fc0a2ea6ae66256ac5aa578deea18310a0b15f5b0f55318d9d2e62240c2",
    RUNNER: "71280fdcbe07b66f10f3602630d252b8e92b1816172f7b93689eba95b717a777",
    LOCK_B: "16906b81b29b7fac0196067892034fd3a7ded4e1f2ca63a13b66b5bcf6af0c39",
    AUDIT_B: "e2a728e8ab5bea1e6e2b43c67e4f8abc8b53a9f7f64adcb9d827dccff7b478ea",
    CSV_B: "f22897b6ab641131c171abc1a4ba92e790d9379ccac5d32ccf221f85c72e70e0",
    CSV_REPLAY: "f22897b6ab641131c171abc1a4ba92e790d9379ccac5d32ccf221f85c72e70e0",
    TB_MANIFEST: "fcdf0749c790ea2a0539d409066a8c8ccb10a006cc7e5b420811a3c8d32e63d5",
    R + "stage_11d3b_validation_runner_dry_run.log": "6d3537018ff3741ad77b97cabedfcc1cfa489e1e0562d3ddb830a0f84b92b7a7",
    R + "stage_11d3b_validation_runner_dry_run_resources.log": "d2760f2901bde8e1227ba5169d6fefcb142e573a916ec381c853ad60c4ff5ddc",
    POOL: "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
    TRAIN_SELECTION: "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    SEMANTIC: "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
    INDEX: "7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
    MODEL_LOCK: "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json":
        "0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1",
}
COUNTS = {"batches": 45, "legal_sites": 22839, "persistent_fault_instances": 45678,
          "baseline_records": 2160, "enabled_records": 2192544, "total_records": 2194704}
CSV_HEADER = (
    "batch_id,vector_partition,run_type,site_selection,selector,site_id,stuck_value,"
    "vector_slot,vector_id,fault_enable,cycles,baseline_cycles,timed_out,activity,"
    "detected,unknown,expected_digest,actual_digest"
).split(",")


class VerificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def fields(document, expected, label):
    require(isinstance(document, dict), f"{label}: expected an object")
    errors = []
    for key, value in expected.items():
        if key not in document:
            errors.append(f"missing {key}")
        elif type(document[key]) is not type(value) or document[key] != value:
            errors.append(f"{key}: expected {value!r}, got {document[key]!r}")
    require(not errors, label + ": " + "; ".join(errors))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


class Evidence:
    def __init__(self, root):
        self.root = root.resolve()
        self.records = {}

    def path(self, relative):
        require(isinstance(relative, str) and relative, "missing artifact path")
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts, f"non-project-relative path: {relative}")
        resolved = (self.root / p).resolve()
        require(resolved.is_relative_to(self.root), f"path escapes project: {relative}")
        return resolved

    def verify(self, relative, expected, size=None):
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"malformed expected SHA for {relative}")
        path = self.path(relative)
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty input: {relative}")
        actual = digest(path)
        require(actual == expected, f"SHA mismatch for {relative}: expected {expected}, actual {actual}")
        if size is not None:
            require(type(size) is int and path.stat().st_size == size, f"byte count mismatch: {relative}")
        rec = {"path": relative, "sha256": actual, "bytes": path.stat().st_size}
        if relative in self.records:
            require(self.records[relative] == rec, f"conflicting evidence: {relative}")
        self.records[relative] = rec
        return rec

    def linked(self, record):
        require(isinstance(record, dict) and {"path", "sha256"} <= record.keys(), "malformed linked evidence")
        return self.verify(record["path"], record["sha256"], record.get("bytes"))

    def load(self, relative):
        require(relative in self.records, f"JSON not hash-verified: {relative}")
        obj = json.loads(self.path(relative).read_text(), object_pairs_hook=unique_object)
        require(isinstance(obj, dict), f"JSON object required: {relative}")
        return obj

    def recheck(self):
        for record in list(self.records.values()):
            self.linked(record)


def verify_contracts(e):
    a, c, aa, b, ab = (e.load(p) for p in (ARCH, CONTRACT, AUDIT_A, LOCK_B, AUDIT_B))
    fields(a, {"stage": "11D-3A", "status": "PASS", "architecture_status": "FROZEN",
               "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY", "vector_partition": "VALIDATION",
               "validation_vectors": 48, "cycles_per_baseline_transaction": 343, **COUNTS}, "architecture")
    fields(c, {"stage": "11D-3A", "status": "PASS", "contract_status": "FROZEN",
               "execution_mode": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
               "checkpoint_after_each_batch": True, "resume_supported": True, "overwrite_default": False,
               "minimum_free_disk_gib": 5, "runner_generation_authorized": True, "runner_dry_run_authorized": True,
               "validation_campaign_execution_authorized": False, "validation_model_inference_authorized": False,
               "validation_dataset_consolidation_authorized": False, "holdout_campaign_execution_authorized": False}, "execution contract")
    for flag in ("model_training_permitted", "candidate_selection_permitted", "threshold_selection_permitted",
                 "feature_or_graph_refitting_permitted", "holdout_access_permitted"):
        fields(a, {flag: False}, "architecture prohibitions")
    fields(aa, {"stage": "11D-3A", "status": "PASS", "validation_campaign_execution": "NOT_YET_AUTHORIZED",
                "validation_vector_payloads_emitted": 0, "holdout_vectors_exposed": 0}, "3A freeze")
    fields(b, {"stage": "11D-3B", "status": "FROZEN", "runner_version": RUNNER_VERSION,
               "batch_count": 45, "validation_vector_count": 48, "legal_sites": 22839,
               "persistent_fault_instances": 45678, "expected_baseline_records": 2160,
               "expected_enabled_records": 2192544, "expected_total_records": 2194704,
               "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
               "checkpoint_after_each_batch": True, "resume_supported": True,
               "dry_run_partition": "TRAIN_CANARY", "dry_run_batch": 22, "dry_run_records": 10,
               "validation_vectors_executed": 0, "holdout_vectors_exposed": 0,
               "validation_execution_authorized": False, "validation_model_inference_authorized": False,
               "holdout_execution_authorized": False}, "runner lock")
    fields(ab, {"stage": "11D-3B", "status": "PASS", "runner_status": "FROZEN", "dry_run_status": "PASS",
                "runner_version": RUNNER_VERSION, "dry_run_batch": 22, "dry_run_partition": "TRAIN_CANARY",
                "dry_run_train_vectors": 2, "dry_run_validation_vectors": 0, "dry_run_holdout_vectors": 0,
                "dry_run_sites": 2, "dry_run_fault_instances": 4, "canonical_records": 10, "replay_records": 10,
                "deterministic_replay": "PASS", "exact_csv_match": True, "unknown_records": 0,
                "detected_without_activity": 0, "baseline_failures": 0,
                "validation_vectors_executed": 0, "holdout_vectors_exposed": 0,
                "full_validation_testbenches_generated": 0,
                "validation_campaign_execution": "NOT_YET_AUTHORIZED",
                "validation_model_inference": "NOT_YET_AUTHORIZED", "holdout_campaign": "NOT_YET_AUTHORIZED"}, "3B freeze")
    for doc in (b, ab):
        require(e.linked(doc["runner_source"]) == e.records[RUNNER], "runner-source link")
    require(e.linked(b["stage_11d3a_execution_contract"]) == e.records[CONTRACT], "execution-contract link")
    for key, rec in ab["input_evidence"].items():
        require(key == rec["path"], "3B input-evidence key/path mismatch")
        e.linked(rec)
    required_outputs = {"runner_lock", "testbench", "testbench_manifest", "binary", "canonical_csv", "replay_csv",
                        "build_log", "simulation_log", "simulation_replay_log", "resources_log", "replay_resources_log"}
    require(required_outputs <= ab["outputs"].keys(), "3B output evidence incomplete")
    for rec in ab["outputs"].values():
        e.linked(rec)
    for key, path in (("runner_lock", LOCK_B), ("canonical_csv", CSV_B),
                      ("replay_csv", CSV_REPLAY), ("testbench_manifest", TB_MANIFEST)):
        require(ab["outputs"][key]["path"] == path, f"3B {key} path")
    acceptance = a["acceptance_contract"]
    require(acceptance == c["acceptance_contract"] == aa["acceptance_contract"], "acceptance rules differ")
    require(acceptance["hybrid_generalization_retention"]["validation_mcc_minimum"] == 0.52982119,
            "frozen validation MCC minimum")
    return a, c, aa, b, ab


def verify_model(e, architecture):
    lock = e.load(MODEL_LOCK)
    fields(lock, {"status": "PASS", "lock_status": "FROZEN", "model_id": "HYBRID_FUSION_MLP_11D2C",
                  "candidate_id": "HYBRID_SGC3_MLP_128_64_32_A1E4_LR1E3", "threshold": 0.4965,
                  "model_retraining_allowed": False, "threshold_reselection_allowed": False,
                  "feature_or_graph_refitting_allowed": False, "validation_use_for_tuning_allowed": False,
                  "holdout_use_for_tuning_allowed": False}, "final-model lock")
    for doc in (lock, architecture):
        require(e.linked(doc["model_artifact"]) == e.records[MODEL], "frozen model identity")
    for key in ("training_selection_lock", "training_manifest", "training_audit", "evaluation_comparison",
                "evaluation_lock", "evaluation_manifest", "evaluation_audit"):
        e.linked(lock[key])
    # Deliberately no joblib/pickle loading and no inference.


def verify_vectors(e, documents):
    pool, selection, cm = (e.load(p) for p in (POOL, TRAIN_SELECTION, COMMITMENT))
    records = pool["vectors"]
    require(len(records) == 256, "vector-pool count")
    require(Counter(v["split"] for v in records) == {"TRAIN": 160, "VALIDATION": 48, "HOLDOUT_TEST": 48},
            "vector split counts")
    by_id = {int(v["vector_id"]): v for v in records}
    require(len(by_id) == 256, "duplicate vector IDs")
    train_ids = selection["selected_vector_ids"]
    require(len(train_ids) == len(set(train_ids)) == 64, "selected TRAIN IDs")
    require(all(by_id[i]["split"] == "TRAIN" for i in train_ids), "non-TRAIN canary selection")
    validation_ids = sorted(i for i, v in by_id.items() if v["split"] == "VALIDATION")
    fields(cm, {"status": "PASS", "manifest_status": "FROZEN", "validation_vector_count": 48,
                "validation_vector_payloads_emitted": 0, "holdout_vector_records_emitted": 0}, "commitments")
    require(len(cm["records"]) == 48, "commitment record count")
    require(e.linked(cm["source_vector_pool"]) == e.records[POOL], "commitment pool binding")
    h = hashlib.sha256()
    for pos, (vid, row) in enumerate(zip(validation_ids, cm["records"], strict=True)):
        fields(row, {"validation_order": pos, "vector_id": vid}, "commitment ordering")
        require(not ({"key", "message", "expected_hmac_sha256"} & row.keys()), "payload in commitment metadata")
        for name in ("pair_fingerprint", "vector_commitment"):
            value = row[name]
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "malformed vector commitment")
            require(value == by_id[vid][name], "commitment metadata differs from frozen pool")
        h.update(f"{pos}|{vid}|{row['pair_fingerprint']}|{row['vector_commitment']}\n".encode())
    commitment = h.hexdigest()
    for doc in (cm, *documents):
        require(doc["ordered_validation_commitment"] == commitment, "ordered validation commitment mismatch")
    # Access payload fields only for TRAIN positions already executed in 3B.
    canaries = {pos: by_id[train_ids[pos]] for pos in (0, 63)}
    return commitment, canaries


def verify_batches(e, architecture, contract):
    semantic, index = e.load(SEMANTIC), e.load(INDEX)
    fields(semantic, {"status": "PASS", "batches": 45, "legal_sites": 22839, "persistent_fault_instances": 45678,
                      "mapping_mismatches": 0, "routing_mismatches": 0, "raw_monitor_mismatches": 0,
                      "golden_restoration_mismatches": 0, "site_gaps": 0, "duplicate_sites": 0}, "semantic manifest")
    fields(index, {"status": "GENERATED_AND_STRUCTURALLY_CHECKED", "batches": 45, "full_batches": 44,
                   "partial_batches": 1, "legal_sites": 22839, "persistent_fault_instances": 45678}, "generation index")
    entries = semantic["batch_qualification"]
    require(len(entries) == 45 and sorted(x["batch_id"] for x in entries) == list(range(45)), "batch ID coverage")
    entries = sorted(entries, key=lambda x: x["batch_id"])
    plan, all_ids, batch22, verified_batches = [], set(), None, []
    for entry in entries:
        bid = entry["batch_id"]
        key, size = f"{bid:03d}", (311 if bid == 44 else 512)
        base = f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}"
        net = e.verify(base + ".v", entry["verilog_sha256"])
        js = e.verify(base + ".json", entry["json_sha256"])
        mp = f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
        mapping = e.verify(mp, entry["mapping_sha256"])
        sites = e.load(mp)["sites"]
        require(len(sites) == size, f"Batch {key} site count")
        require([s["selector_code"] for s in sites] == list(range(size)), f"Batch {key} selector coverage")
        for selector, site in enumerate(sites):
            expected_index = bid * 512 + selector + 1
            fields(site, {"fault_site_id": f"HMAC-STEM-{expected_index:06d}", "site_index": expected_index}, f"Batch {key} mapping")
            require(site["fault_site_id"] not in all_ids, "duplicate physical site")
            all_ids.add(site["fault_site_id"])
        if bid == 22:
            batch22 = sites
        plan.append({"batch_id": bid, "batch_key": key, "legal_sites": size, "fault_instances": size * 2,
                     "validation_vectors": 48, "baseline_records": 48, "enabled_records": size * 96,
                     "total_records": size * 96 + 48})
        verified_batches.append({"batch_id": bid, "json": js, "netlist": net, "mapping": mapping})
        print(f"  Batch {key}: PASS sites={size}", flush=True)
    require(len(all_ids) == 22839, "physical-site coverage")
    require(plan == architecture["batch_plan"] == contract["expected_batch_records"], "frozen batch plan mismatch")
    for name, expected in COUNTS.items():
        if name == "batches":
            actual = len(plan)
        else:
            key = "fault_instances" if name == "persistent_fault_instances" else name
            actual = sum(row[key] for row in plan)
        require(actual == expected, f"campaign total {name}")
    return plan, batch22, verified_batches


def csv_summary(text, canaries, sites):
    reader = csv.DictReader(io.StringIO(text))
    require(reader.fieldnames == CSV_HEADER, "dry-run CSV header")
    seen_base, seen_fault, counts = set(), set(), Counter()
    for line, row in enumerate(reader, 2):
        require(None not in row and all(v is not None for v in row.values()), f"CSV field count, line {line}")
        require(row["batch_id"] == "22" and row["vector_partition"] == "TRAIN_CANARY", "dry-run CSV scope")
        pos = int(row["vector_slot"])
        require(pos in canaries and int(row["vector_id"]) == canaries[pos]["vector_id"], "canary vector identity")
        for name in ("timed_out", "activity", "detected", "unknown", "fault_enable"):
            require(row[name] in ("0", "1"), f"non-binary {name}, line {line}")
        require(row["unknown"] == "0", "unknown signal in dry-run CSV")
        for name in ("expected_digest", "actual_digest"):
            require(re.fullmatch(r"[0-9a-fA-F]{64}", row[name]), f"malformed {name}")
        expected_digest = str(canaries[pos]["expected_hmac_sha256"]).lower().removeprefix("0x")
        require(row["expected_digest"].lower() == expected_digest, "canary reference digest")
        cycles = int(row["cycles"])
        require(0 <= cycles <= 2000 and int(row["baseline_cycles"]) == 343, "cycle/timeout bounds")
        require(row["timed_out"] == "0" or cycles == 2000, "timeout before cycle limit")
        changed = row["actual_digest"].lower() != expected_digest or cycles != 343 or row["timed_out"] == "1"
        if row["run_type"] == "BASELINE":
            require(pos not in seen_base, "duplicate baseline sample")
            seen_base.add(pos)
            require(not changed and row["fault_enable"] == "0" and row["activity"] == row["detected"] == "0", "baseline failure")
            require(row["site_id"] == "BASELINE" and all(row[k] == "-1" for k in ("site_selection", "selector", "stuck_value")), "baseline provenance")
        else:
            require(row["run_type"] == "ENABLED" and row["fault_enable"] == "1", "invalid run type")
            selector, stuck = int(row["selector"]), int(row["stuck_value"])
            require(selector in (0, 511) and stuck in (0, 1), "dry-run fault selection")
            require(row["site_id"] == sites[selector]["fault_site_id"], "dry-run site identity")
            require(int(row["site_selection"]) == (0 if selector == 0 else 1), "site-selection mapping")
            sample = (selector, stuck, pos)
            require(sample not in seen_fault, "duplicate enabled sample")
            seen_fault.add(sample)
            require(int(row["detected"]) == int(changed), "detected label inconsistent with observation")
            require(row["detected"] == "0" or row["activity"] == "1", "detected without activity")
            counts["activated"] += int(row["activity"])
            counts["detected"] += int(row["detected"])
    require(seen_base == {0, 63}, "missing baseline vectors")
    require(seen_fault == {(s, f, v) for s in (0, 511) for f in (0, 1) for v in (0, 63)}, "missing fault-vector samples")
    return {"baseline_records": 2, "enabled_records": 8, "total_records": 10,
            "activated_records": counts["activated"], "detected_records": counts["detected"],
            "unknown_records": 0, "detected_without_activity": 0, "baseline_failures": 0}


def verify_replay(e, ab, canaries, sites, commitment):
    a, b = e.path(CSV_B).read_bytes(), e.path(CSV_REPLAY).read_bytes()
    require(a == b, "dry-run CSV replay differs")
    summary = csv_summary(a.decode(), canaries, sites)
    manifest = e.load(TB_MANIFEST)
    fields(manifest, {"status": "GENERATED", "mode": "DRY_RUN_TRAIN_CANARY", "batch_id": 22,
                      "vector_partition": "TRAIN_CANARY", "vectors": 2, "vector_positions": [0, 63],
                      "vector_ids": [canaries[p]["vector_id"] for p in (0, 63)], "sites": 2,
                      "selectors": [0, 511], "site_ids": [sites[i]["fault_site_id"] for i in (0, 511)],
                      "fault_instances": 4, "baseline_records": 2, "enabled_records": 8, "total_records": 10,
                      "validation_vectors_executed": 0, "holdout_vectors_exposed": 0,
                      "ordered_validation_commitment": commitment}, "dry-run generation manifest")
    require(e.linked(manifest["testbench"]) == e.linked(ab["outputs"]["testbench"]), "testbench link")
    for rec in ab["canonical_batch_22"].values():
        e.linked(rec)
    for name in ("simulation_log", "simulation_replay_log"):
        text = e.path(ab["outputs"][name]["path"]).read_text()
        require("VALIDATION_RUNNER_DRY_RUN_RESULT=PASS" in text, "dry-run PASS token missing")
    return summary


def make_authorization(e, source_record, created_at, commitment, plan, architecture):
    return {
        "stage": "11D-3C", "title": "VALIDATION CAMPAIGN EXECUTION AUTHORIZATION",
        "status": "FROZEN", "created_at_utc": created_at, "verifier_version": VERSION,
        "verifier_source": source_record, "runner_version": RUNNER_VERSION,
        "runner_sha256": e.records[RUNNER]["sha256"], "runner_lock_sha256": e.records[LOCK_B]["sha256"],
        "execution_contract_sha256": e.records[CONTRACT]["sha256"],
        "dry_run_audit_sha256": e.records[AUDIT_B]["sha256"], "ordered_validation_commitment": commitment,
        "validation_campaign_execution": "AUTHORIZED", "validation_model_inference": "NOT_AUTHORIZED",
        "validation_dataset_consolidation": "NOT_AUTHORIZED", "holdout_campaign": "NOT_AUTHORIZED",
        "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED", "feature_refitting": "PROHIBITED",
        "vector_partition": "VALIDATION", "validation_vectors": 48, "batch_ids": list(range(45)), **COUNTS,
        "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY", "execution": "SEQUENTIAL",
        "parallel_batches": 1, "build_jobs": 1, "checkpoint_after_each_batch": True, "resume_supported": True,
        "minimum_free_disk_gib": 5, "overwrite_authorized": False,
        "authorization_scope": "SIMULATION_AND_PER_BATCH_RESULT_CHECKPOINTS_ONLY",
        "final_model_lock": e.records[MODEL_LOCK], "final_model": e.records[MODEL],
        "frozen_threshold": 0.4965, "acceptance_contract": architecture["acceptance_contract"],
        "expected_batch_records": plan,
        "prerequisite_audit_path": AUDIT,
        "validation_execution_performed_by_this_stage": False,
        "next_gate": "STAGE 11D-3D — VALIDATION CAMPAIGN EXECUTION",
    }


def publish_once(path, payload):
    """Link a fully flushed temporary file into place; fail if target exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    require(not path.exists() and not path.is_symlink(), f"refusing to replace existing output: {path}")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)  # exclusive publication; cannot replace a prior freeze
    temp.unlink()  # only the temporary hard link created above


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run(root, check_only=False):
    e = Evidence(root)
    source = Path(__file__).resolve()
    require(e.root.name == "vlsi_fault_detection_v2" and source.parent == e.root,
            "place this script in ~/vlsi_fault_detection_v2 and run it there")
    for output in (AUTHORIZATION, AUDIT):
        for suffix in ("", ".pending"):
            p = e.path(output + suffix)
            require(not p.exists() and not p.is_symlink(), f"output exists: {output + suffix}; preserve it and request review")
    for directory in ("results/hmac_validation_campaign_full_11d3", "build/hmac_validation_campaign_full_11d3"):
        p = e.path(directory)
        require(not p.exists() or (p.is_dir() and not any(p.iterdir())), f"validation campaign artifacts already present: {directory}; stop for review")
    print("STAGE 11D-3C — VALIDATION CAMPAIGN EXECUTION AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION", flush=True)
    failures = []
    for relative, expected in EXPECTED.items():
        try:
            e.verify(relative, expected)
            print(f"  {Path(relative).name}: OK", flush=True)
        except (VerificationError, OSError) as error:
            failures.append(str(error))
    require(not failures, "input verification failed:\n  " + "\n  ".join(failures))
    # Parse only. Never import the runner or invoke any of its functions.
    ast.parse(e.path(RUNNER).read_text())
    a, c, aa, b, ab = verify_contracts(e)
    verify_model(e, a)
    commitment, canaries = verify_vectors(e, (a, aa, b))
    print("ALL-BATCH ARTIFACT AND MAPPING VERIFICATION", flush=True)
    plan, batch22, batch_evidence = verify_batches(e, a, c)
    summary = verify_replay(e, ab, canaries, batch22, commitment)
    print("  Exact replay, sample identities, and label consistency: PASS", flush=True)
    require(shutil.which("verilator") and shutil.which("make") and shutil.which("g++"),
            "missing verilator, make, or g++; activate the project tool PATH")
    require(Path("/usr/bin/time").is_file(), "missing /usr/bin/time")
    free_bytes = shutil.disk_usage(e.root).free
    require(free_bytes >= 5 * 1024**3, "less than 5 GiB free disk")
    source_record = e.verify(source.name, digest(source))
    print("  Rechecking verified inputs for changes", flush=True)
    e.recheck()
    if check_only:
        print("CHECK_ONLY_RESULT=PASS; authorization NOT WRITTEN; validation NOT STARTED")
        return 0
    created = datetime.now(timezone.utc).isoformat()
    authorization = make_authorization(e, source_record, created, commitment, plan, a)
    payload = encoded(authorization)
    audit = {
        "stage": "11D-3C", "title": "VALIDATION CAMPAIGN EXECUTION AUTHORIZATION FREEZE",
        "status": "PASS", "authorization_status": "FROZEN", "created_at_utc": created,
        "verifier_version": VERSION, "verifier_source": source_record, **COUNTS,
        "validation_vectors": 48, "ordered_validation_commitment": commitment,
        "batch_artifacts_verified": 45, "mapping_checks": 22839,
        "dry_run_summary": summary, "deterministic_replay": "PASS", "exact_csv_match": True,
        "available_disk_gib": round(free_bytes / 1024**3, 2),
        "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
        "checkpoint_after_each_batch": True, "resume_supported": True,
        "validation_campaign_execution": "AUTHORIZED", "validation_model_inference": "NOT_AUTHORIZED",
        "holdout_campaign": "NOT_AUTHORIZED", "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED",
        "validation_vectors_executed_by_this_stage": 0, "validation_payloads_emitted_by_this_stage": 0,
        "holdout_payloads_emitted_by_this_stage": 0, "model_deserialization_performed": False,
        "model_inference_performed": False, "training_performed": False, "campaign_launched": False,
        "input_artifacts_unchanged": True, "runner_modified": False, "golden_netlist_modified": False,
        "final_model_modified": False, "input_evidence": e.records, "batch_evidence": batch_evidence,
        "acceptance_contract": a["acceptance_contract"],
        "authorization": {"path": AUTHORIZATION, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
        "completion_rule": "BOTH_AUDIT_AND_MATCHING_AUTHORIZATION_MUST_EXIST",
        "next_gate": authorization["next_gate"],
    }
    # Audit first: an interruption cannot leave a usable authorization without an audit.
    publish_once(e.path(AUDIT), encoded(audit))
    publish_once(e.path(AUTHORIZATION), payload)
    require(digest(e.path(AUTHORIZATION)) == audit["authorization"]["sha256"], "authorization readback hash")
    print("\nSTAGE 11D-3C — VALIDATION CAMPAIGN EXECUTION AUTHORIZATION FREEZE")
    for label, value in (
        ("Status", "PASS"), ("Authorization status", "FROZEN"), ("Batches", 45),
        ("Legal sites", 22839), ("Persistent fault instances", 45678), ("VALIDATION vectors", 48),
        ("Baseline records", 2160), ("Enabled validation records", 2192544), ("Total validation records", 2194704),
        ("Batch artifacts verified", "45/45"), ("Deterministic CSV replay", "PASS"),
        ("Execution / build jobs", "SEQUENTIAL / 1"), ("Checkpoint/resume", "ENABLED"),
        ("Available disk", f"{free_bytes / 1024**3:.2f} GiB"),
        ("Validation campaign execution", "AUTHORIZED"), ("Validation campaign started", "NO"),
        ("Validation model inference", "NOT AUTHORIZED"), ("Holdout campaign", "NOT AUTHORIZED"),
        ("Retraining / threshold changes", "PROHIBITED"), ("Frozen inputs modified", "NO"),
        ("Authorization", e.path(AUTHORIZATION)), ("Authorization SHA", digest(e.path(AUTHORIZATION))),
        ("Audit", e.path(AUDIT)), ("Audit SHA", digest(e.path(AUDIT))), ("Next gate", authorization["next_gate"]),
    ):
        print(f"{label:31}: {value}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true", help="verify inputs only; write no authorization or audit")
    args = parser.parse_args()
    try:
        return run(args.project_root, args.check_only)
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        print("Stage 11D-3C did not pass. Do not launch validation. Preserve all existing artifacts.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
