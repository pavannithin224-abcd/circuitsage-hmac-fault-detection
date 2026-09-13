#!/usr/bin/env python3
"""Qualify and later run the frozen OpenTitan HMAC validation campaign."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VERSION = "HMAC-VALIDATION-CAMPAIGN-RUNNER-v1"

EXPECTED_BATCHES = 45
EXPECTED_LEGAL_SITES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_VALIDATION_VECTORS = 48
EXPECTED_BASELINE_RECORDS = 2_160
EXPECTED_ENABLED_RECORDS = 2_192_544
EXPECTED_TOTAL_RECORDS = 2_194_704
EXPECTED_CYCLES = 343
DRY_RUN_BATCH = 22
DRY_RUN_VECTOR_POSITIONS = [0, 63]
DRY_RUN_SELECTORS = [0, 511]

FROZEN_INPUTS = {
    "config/fault_campaign/hmac_validation_campaign_architecture_11d3a.json":
        "256aca0cba7d14439f985d42a0683912a0e92285282f2fa116eb43728ffe2f8d",
    "config/fault_campaign/hmac_validation_campaign_execution_contract_11d3a.json":
        "55c3b2c57c8a6a7068812a41a75b01f75292ce0068c4553e3fd35fb972a87420",
    "results/hmac_fault_campaign_11d3/hmac_validation_vector_commitments_11d3a.json":
        "9fff03e0dd00e91d405daee71e98af62408969407547308180475d90d20ba99b",
    "results/hmac_fault_campaign_11d3/hmac_validation_campaign_architecture_execution_contract_freeze_11d3a.json":
        "ad8d5fc0a2ea6ae66256ac5aa578deea18310a0b15f5b0f55318d9d2e62240c2",
    "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json":
        "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
    "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json":
        "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1",
    "results/hmac_fault_campaign_11c4/hmac_canonical_batch_generation_index_11c4g_a1.json":
        "7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
    "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json":
        "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
    "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json":
        "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
}


def fail(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"could not read JSON {path}: {error}")
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def artifact(root: Path, path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_frozen_inputs(root: Path) -> dict:
    evidence = {}
    print("FROZEN INPUT VERIFICATION", flush=True)
    for relative, expected in FROZEN_INPUTS.items():
        path = root / relative
        require(path.is_file() and path.stat().st_size > 0, f"missing frozen input: {relative}")
        actual = sha256(path)
        require(actual == expected, f"SHA mismatch for {relative}: expected {expected}, actual {actual}")
        evidence[relative] = artifact(root, path)
        print(f"  {path.name:<80}: OK", flush=True)

    architecture = load_json(root / "config/fault_campaign/hmac_validation_campaign_architecture_11d3a.json")
    contract = load_json(root / "config/fault_campaign/hmac_validation_campaign_execution_contract_11d3a.json")
    audit = load_json(root / "results/hmac_fault_campaign_11d3/hmac_validation_campaign_architecture_execution_contract_freeze_11d3a.json")
    require(architecture.get("status") == "PASS" and architecture.get("architecture_status") == "FROZEN", "Stage 11D-3A architecture")
    require(contract.get("status") == "PASS" and contract.get("contract_status") == "FROZEN", "Stage 11D-3A execution contract")
    require(contract.get("runner_generation_authorized") is True, "runner-generation authorization")
    require(contract.get("runner_dry_run_authorized") is True, "runner dry-run authorization")
    require(contract.get("validation_campaign_execution_authorized") is False, "validation execution must remain blocked")
    require(contract.get("validation_model_inference_authorized") is False, "validation inference must remain blocked")
    require(contract.get("holdout_campaign_execution_authorized") is False, "HOLDOUT execution must remain blocked")
    require(audit.get("status") == "PASS", "Stage 11D-3A audit status")
    require(audit.get("validation_campaign_execution") == "NOT_YET_AUTHORIZED", "Stage 11D-3A execution handoff")
    require(audit.get("holdout_vectors_exposed") == 0, "prior HOLDOUT exposure")
    return evidence


def verify_runner_source_safety(source: Path) -> None:
    tree = ast.parse(source.read_text())
    forbidden_calls = {
        "fit", "fit_transform", "partial_fit", "predict", "predict_proba",
        "select_threshold", "backward", "optimizer", "step",
    }
    forbidden_import_roots = {"joblib", "sklearn", "torch", "tensorflow", "keras"}
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = ""
            if name in forbidden_calls:
                findings.append(f"call:{name}@{node.lineno}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden_import_roots:
                    findings.append(f"import:{alias.name}@{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in forbidden_import_roots:
                findings.append(f"import:{node.module}@{node.lineno}")
    require(not findings, f"model-training or inference operations present in runner: {findings}")


def h256(value: object, field: str, vector_id: int) -> str:
    result = f"{value:064x}" if isinstance(value, int) else str(value).strip().lower().removeprefix("0x")
    require(re.fullmatch(r"[0-9a-f]{64}", result) is not None, f"vector {vector_id} {field} is not 256-bit hexadecimal")
    return result


def load_vector_contract(root: Path) -> tuple[list[dict], list[dict], str]:
    pool = load_json(root / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json")
    selection = load_json(root / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json")
    commitments = load_json(root / "results/hmac_fault_campaign_11d3/hmac_validation_vector_commitments_11d3a.json")
    records = pool.get("vectors")
    require(isinstance(records, list) and len(records) == 256, "vector-pool record count")
    by_id = {int(item["vector_id"]): item for item in records}
    require(len(by_id) == 256, "unique vector IDs")

    selected_ids = selection.get("selected_vector_ids")
    require(isinstance(selected_ids, list) and len(selected_ids) == 64, "selected TRAIN vector count")
    train = []
    for position, raw_id in enumerate(selected_ids):
        vector_id = int(raw_id)
        item = by_id.get(vector_id)
        require(item is not None and item.get("split") == "TRAIN", f"selected TRAIN vector {vector_id}")
        train.append({
            "position": position,
            "vector_id": vector_id,
            "partition": "TRAIN_CANARY",
            "key": h256(item["key"], "key", vector_id),
            "message": h256(item["message"], "message", vector_id),
            "digest": h256(item["expected_hmac_sha256"], "digest", vector_id),
        })

    validation_source = sorted(
        (item for item in records if item.get("split") == "VALIDATION"),
        key=lambda item: int(item["vector_id"]),
    )
    require(len(validation_source) == EXPECTED_VALIDATION_VECTORS, "VALIDATION vector count")
    commitment_records = commitments.get("records")
    require(isinstance(commitment_records, list) and len(commitment_records) == EXPECTED_VALIDATION_VECTORS, "validation commitment records")
    validation = []
    ordered = hashlib.sha256()
    for position, (item, sealed) in enumerate(zip(validation_source, commitment_records, strict=True)):
        vector_id = int(item["vector_id"])
        require(int(sealed.get("validation_order", -1)) == position, "validation commitment order")
        require(int(sealed.get("vector_id", -1)) == vector_id, "validation commitment vector ID")
        require(sealed.get("pair_fingerprint") == item.get("pair_fingerprint"), "validation pair fingerprint")
        require(sealed.get("vector_commitment") == item.get("vector_commitment"), "validation vector commitment")
        ordered.update(
            f"{position}|{vector_id}|{item['pair_fingerprint']}|{item['vector_commitment']}\n".encode()
        )
        validation.append({
            "position": position,
            "vector_id": vector_id,
            "partition": "VALIDATION",
            "key": h256(item["key"], "key", vector_id),
            "message": h256(item["message"], "message", vector_id),
            "digest": h256(item["expected_hmac_sha256"], "digest", vector_id),
        })
    ordered_commitment = ordered.hexdigest()
    require(ordered_commitment == commitments.get("ordered_validation_commitment"), "ordered validation commitment")
    require(commitments.get("holdout_vector_records_emitted") == 0, "HOLDOUT commitment records")
    return train, validation, ordered_commitment


def find_sites(value: object) -> list[dict] | None:
    if isinstance(value, dict):
        for child in value.values():
            found = find_sites(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        if value and isinstance(value[0], dict) and {"fault_site_id", "selector_code"} <= value[0].keys():
            return value
        for child in value:
            found = find_sites(child)
            if found is not None:
                return found
    return None


def canonical_batch(root: Path, batch_id: int) -> tuple[Path, Path, list[dict]]:
    key = f"{batch_id:03d}"
    netlist = root / f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.v"
    mapping = root / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
    require(netlist.is_file() and netlist.stat().st_size > 0, f"canonical Batch {key} Verilog missing")
    require(mapping.is_file() and mapping.stat().st_size > 0, f"canonical Batch {key} mapping missing")
    semantic = load_json(root / "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json")
    entries = semantic.get("batch_qualification")
    require(isinstance(entries, list) and len(entries) == EXPECTED_BATCHES, "semantic batch qualification")
    entry = next((item for item in entries if int(item.get("batch_id", -1)) == batch_id), None)
    require(entry is not None, f"semantic qualification for Batch {key}")
    require(sha256(netlist) == entry.get("verilog_sha256"), f"Batch {key} Verilog SHA")
    require(sha256(mapping) == entry.get("mapping_sha256"), f"Batch {key} mapping SHA")
    sites = find_sites(load_json(mapping))
    expected_sites = 311 if batch_id == 44 else 512
    require(isinstance(sites, list) and len(sites) == expected_sites, f"Batch {key} site count")
    require([int(item["selector_code"]) for item in sites] == list(range(expected_sites)), f"Batch {key} selector sequence")
    return netlist, mapping, sites


def generate_testbench(
    path: Path,
    csv_path: Path,
    manifest_path: Path,
    batch_id: int,
    all_vectors: list[dict],
    all_sites: list[dict],
    dry_run: bool,
    root: Path,
    ordered_commitment: str,
) -> dict:
    if dry_run:
        vectors = [all_vectors[position] for position in DRY_RUN_VECTOR_POSITIONS]
        sites = [all_sites[selector] for selector in DRY_RUN_SELECTORS]
        prefix = "VALIDATION_RUNNER_DRY_RUN"
        result_token = "VALIDATION_RUNNER_DRY_RUN_RESULT"
        partition = "TRAIN_CANARY"
    else:
        vectors = all_vectors
        sites = all_sites
        prefix = "VALIDATION_CAMPAIGN_BATCH"
        result_token = "VALIDATION_CAMPAIGN_BATCH_RESULT"
        partition = "VALIDATION"

    vector_count = len(vectors)
    site_count = len(sites)
    fault_count = site_count * 2
    enabled_records = fault_count * vector_count
    total_records = vector_count + enabled_records
    key = f"{batch_id:03d}"
    top = f"tb_hmac_validation_batch_{key}"
    dut = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
    default_csv = str(csv_path).replace("\\", "\\\\").replace('"', '\\"')

    initialization = []
    for index, vector in enumerate(vectors):
        initialization.extend([
            f"        vector_positions[{index}] = {vector['position']};",
            f"        vector_ids[{index}] = {vector['vector_id']};",
            f"        key_vectors[{index}] = 256'h{vector['key']};",
            f"        message_vectors[{index}] = 256'h{vector['message']};",
            f"        digest_vectors[{index}] = 256'h{vector['digest']};",
        ])
    for index, site in enumerate(sites):
        initialization.extend([
            f"        selected_selectors[{index}] = 9'd{int(site['selector_code'])};",
            f"        selected_site_ids[{index}] = \"{site['fault_site_id']}\";",
        ])

    sv = f'''`timescale 1ns/1ps
module {top};
  localparam integer BATCH_ID={batch_id}, VECTOR_COUNT={vector_count}, SITE_COUNT={site_count}, FAULT_COUNT={fault_count};
  localparam integer EXPECTED_ENABLED_RUNS={enabled_records}, EXPECTED_TOTAL_RECORDS={total_records};
  localparam integer TIMEOUT_CYCLES=2000, EXPECTED_BASELINE_CYCLES=343;
  localparam string DEFAULT_CSV="{default_csv}";
  logic clk_i=0,rst_ni=0,start_i=0,busy_o,done_o,fault_enable_i=0,fault_value_i=0,fault_raw_o;
  logic [8:0] fault_selector_i='0; logic [255:0] key_i='0,message_i='0,digest_o;
  logic monitor_clear=1,activity_latched=0;
  integer vector_positions[0:VECTOR_COUNT-1],vector_ids[0:VECTOR_COUNT-1];
  logic [255:0] key_vectors[0:VECTOR_COUNT-1],message_vectors[0:VECTOR_COUNT-1],digest_vectors[0:VECTOR_COUNT-1];
  integer selected_selectors[0:SITE_COUNT-1]; string selected_site_ids[0:SITE_COUNT-1];
  integer baseline_cycles[0:VECTOR_COUNT-1],csv_fd,plusarg_result;
  string csv_path;
  integer baseline_runs,enabled_runs,activated_runs,detected_runs,unknown_runs,detected_without_activity;
  integer baseline_digest_mismatches,baseline_latency_mismatches,baseline_timeouts;
  integer detected_instances,activated_unobserved_instances,unactivated_instances,unclassified_instances;
  integer site_index,vector_index,stuck_index,fault_activity_count,fault_detect_count,cycles_result;
  logic timeout_result,activity_result,unknown_result,detected_result,stuck_value;
  logic [255:0] digest_result;
  {dut} dut(.clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_i),.key_i(key_i),.message_i(message_i),.busy_o(busy_o),.done_o(done_o),.digest_o(digest_o),.fault_enable_i(fault_enable_i),.fault_selector_i(fault_selector_i),.fault_value_i(fault_value_i),.fault_raw_o(fault_raw_o));
  always #5 clk_i=~clk_i;
  always @(monitor_clear or fault_enable_i or fault_value_i or fault_raw_o) begin
    if(monitor_clear) activity_latched=0;
    else if(fault_enable_i && (fault_raw_o !== fault_value_i)) activity_latched=1;
  end
  task automatic reset_dut; begin
    start_i=0; fault_enable_i=0; fault_selector_i='0; fault_value_i=0; monitor_clear=1; rst_ni=0;
    repeat(4) @(posedge clk_i); @(negedge clk_i); rst_ni=1; repeat(2) @(posedge clk_i);
  end endtask
  task automatic run_transaction(input integer vi,input logic en,input integer selector,input logic forced,output integer cycles,output logic timed_out,output logic [255:0] measured_digest,output logic activity_seen,output logic unknown_seen); begin
    reset_dut(); @(negedge clk_i); key_i=key_vectors[vi]; message_i=message_vectors[vi];
    fault_selector_i=selector; fault_value_i=forced; monitor_clear=0; fault_enable_i=en;
    @(negedge clk_i); start_i=1; @(negedge clk_i); start_i=0; cycles=0;
    while(!done_o && cycles<TIMEOUT_CYCLES) begin @(posedge clk_i); #1; cycles=cycles+1; end
    timed_out=!done_o; measured_digest=digest_o; activity_seen=activity_latched;
    unknown_seen=$isunknown({{digest_o,done_o,busy_o,fault_raw_o}});
    fault_enable_i=0; monitor_clear=1; @(posedge clk_i);
  end endtask
  initial begin
{chr(10).join(initialization)}
    baseline_runs=0; enabled_runs=0; activated_runs=0; detected_runs=0; unknown_runs=0;
    detected_without_activity=0; baseline_digest_mismatches=0; baseline_latency_mismatches=0; baseline_timeouts=0;
    detected_instances=0; activated_unobserved_instances=0; unactivated_instances=0; unclassified_instances=0;
    csv_path=DEFAULT_CSV; plusarg_result=$value$plusargs("RESULT_CSV=%s",csv_path); csv_fd=$fopen(csv_path,"w");
    if(csv_fd==0) $fatal(1,"Could not open validation CSV");
    $fdisplay(csv_fd,"batch_id,vector_partition,run_type,site_selection,selector,site_id,stuck_value,vector_slot,vector_id,fault_enable,cycles,baseline_cycles,timed_out,activity,detected,unknown,expected_digest,actual_digest");
    for(vector_index=0;vector_index<VECTOR_COUNT;vector_index=vector_index+1) begin
      run_transaction(vector_index,0,0,0,cycles_result,timeout_result,digest_result,activity_result,unknown_result);
      baseline_cycles[vector_index]=cycles_result; baseline_runs=baseline_runs+1;
      if(timeout_result) baseline_timeouts=baseline_timeouts+1;
      if(digest_result!==digest_vectors[vector_index]) baseline_digest_mismatches=baseline_digest_mismatches+1;
      if(cycles_result!=EXPECTED_BASELINE_CYCLES) baseline_latency_mismatches=baseline_latency_mismatches+1;
      if(unknown_result) unknown_runs=unknown_runs+1;
      $fdisplay(csv_fd,"%0d,{partition},BASELINE,-1,-1,BASELINE,-1,%0d,%0d,0,%0d,%0d,%0d,0,0,%0d,%064h,%064h",BATCH_ID,vector_positions[vector_index],vector_ids[vector_index],cycles_result,EXPECTED_BASELINE_CYCLES,timeout_result,unknown_result,digest_vectors[vector_index],digest_result);
    end
    for(site_index=0;site_index<SITE_COUNT;site_index=site_index+1) begin
      for(stuck_index=0;stuck_index<2;stuck_index=stuck_index+1) begin
        fault_activity_count=0; fault_detect_count=0; stuck_value=(stuck_index!=0);
        for(vector_index=0;vector_index<VECTOR_COUNT;vector_index=vector_index+1) begin
          run_transaction(vector_index,1,selected_selectors[site_index],stuck_value,cycles_result,timeout_result,digest_result,activity_result,unknown_result);
          detected_result=timeout_result||(digest_result!==digest_vectors[vector_index])||(cycles_result!=baseline_cycles[vector_index]); enabled_runs=enabled_runs+1;
          if(activity_result) begin activated_runs=activated_runs+1; fault_activity_count=fault_activity_count+1; end
          if(detected_result) begin detected_runs=detected_runs+1; fault_detect_count=fault_detect_count+1; end
          if(unknown_result) unknown_runs=unknown_runs+1;
          if(detected_result&&!activity_result) detected_without_activity=detected_without_activity+1;
          $fdisplay(csv_fd,"%0d,{partition},ENABLED,%0d,%0d,%s,%0d,%0d,%0d,1,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h",BATCH_ID,site_index,selected_selectors[site_index],selected_site_ids[site_index],stuck_index,vector_positions[vector_index],vector_ids[vector_index],cycles_result,baseline_cycles[vector_index],timeout_result,activity_result,detected_result,unknown_result,digest_vectors[vector_index],digest_result);
        end
        if(fault_detect_count>0) detected_instances=detected_instances+1;
        else if(fault_activity_count>0) activated_unobserved_instances=activated_unobserved_instances+1;
        else unactivated_instances=unactivated_instances+1;
      end
    end
    unclassified_instances=FAULT_COUNT-detected_instances-activated_unobserved_instances-unactivated_instances; $fclose(csv_fd);
    $display("{prefix}_BATCH=%0d",BATCH_ID); $display("{prefix}_BASELINE_RUNS=%0d",baseline_runs);
    $display("{prefix}_ENABLED_RUNS=%0d",enabled_runs); $display("{prefix}_TOTAL_RECORDS=%0d",baseline_runs+enabled_runs);
    $display("{prefix}_FAULT_INSTANCES=%0d",FAULT_COUNT); $display("{prefix}_ACTIVATED_RUNS=%0d",activated_runs);
    $display("{prefix}_DETECTED_RUNS=%0d",detected_runs); $display("{prefix}_DETECTED_INSTANCES=%0d",detected_instances);
    $display("{prefix}_ACTIVATED_UNOBSERVED=%0d",activated_unobserved_instances); $display("{prefix}_UNACTIVATED=%0d",unactivated_instances);
    $display("{prefix}_UNCLASSIFIED=%0d",unclassified_instances); $display("{prefix}_UNKNOWN_RUNS=%0d",unknown_runs);
    $display("{prefix}_DETECTED_WITHOUT_ACTIVITY=%0d",detected_without_activity);
    $display("{prefix}_BASELINE_DIGEST_MISMATCHES=%0d",baseline_digest_mismatches);
    $display("{prefix}_BASELINE_LATENCY_MISMATCHES=%0d",baseline_latency_mismatches); $display("{prefix}_BASELINE_TIMEOUTS=%0d",baseline_timeouts);
    if(baseline_runs==VECTOR_COUNT&&enabled_runs==EXPECTED_ENABLED_RUNS&&(baseline_runs+enabled_runs)==EXPECTED_TOTAL_RECORDS&&baseline_digest_mismatches==0&&baseline_latency_mismatches==0&&baseline_timeouts==0&&unknown_runs==0&&detected_without_activity==0&&unclassified_instances==0) begin
      $display("{result_token}=PASS"); $finish;
    end else $fatal(1,"{result_token}=FAIL");
  end
endmodule
'''
    atomic_text(path, sv)
    info = {
        "runner_version": VERSION,
        "status": "GENERATED",
        "mode": "DRY_RUN_TRAIN_CANARY" if dry_run else "VALIDATION",
        "batch_id": batch_id,
        "vector_partition": partition,
        "vectors": vector_count,
        "vector_positions": [item["position"] for item in vectors],
        "vector_ids": [item["vector_id"] for item in vectors],
        "sites": site_count,
        "selectors": [int(item["selector_code"]) for item in sites],
        "site_ids": [item["fault_site_id"] for item in sites],
        "fault_instances": fault_count,
        "baseline_records": vector_count,
        "enabled_records": enabled_records,
        "total_records": total_records,
        "ordered_validation_commitment": ordered_commitment,
        "testbench": artifact(root, path),
        "result_csv_path": str(csv_path.resolve().relative_to(root)),
        "validation_vectors_executed": 0 if dry_run else vector_count,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
    }
    atomic_json(manifest_path, info)
    return {"top": top, "result_token": result_token, "manifest": info}


def logged(command: list[str], log: Path, timeout_seconds: int, resources: Path | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    actual = list(command)
    if resources is not None:
        actual = ["/usr/bin/time", "-v", "-o", str(resources)] + actual
    with log.open("w") as stream:
        try:
            return subprocess.run(
                actual,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            stream.write(f"\nRUNNER_TIMEOUT_SECONDS={timeout_seconds}\n")
            return 124


def check_csv(path: Path, batch_id: int, site_count: int, vector_count: int, partition: str) -> dict:
    require(path.is_file() and path.stat().st_size > 0, "missing validation-runner CSV")
    baseline = enabled = activated = detected = unknown = detected_without_activity = 0
    baseline_failures = 0
    selectors: set[int] = set()
    slots: set[int] = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "batch_id", "vector_partition", "run_type", "selector", "vector_slot",
            "cycles", "timed_out", "activity", "detected", "unknown",
            "expected_digest", "actual_digest",
        }
        require(reader.fieldnames is not None and required <= set(reader.fieldnames), "validation CSV header")
        for row in reader:
            require(int(row["batch_id"]) == batch_id, "validation CSV batch ID")
            require(row["vector_partition"] == partition, "validation CSV partition")
            unknown += int(row["unknown"])
            if row["run_type"] == "BASELINE":
                baseline += 1
                if (
                    row["timed_out"] != "0"
                    or int(row["cycles"]) != EXPECTED_CYCLES
                    or row["expected_digest"].lower() != row["actual_digest"].lower()
                ):
                    baseline_failures += 1
            elif row["run_type"] == "ENABLED":
                enabled += 1
                selectors.add(int(row["selector"]))
                slots.add(int(row["vector_slot"]))
                activated += int(row["activity"])
                detected += int(row["detected"])
                detected_without_activity += int(row["detected"] == "1" and row["activity"] == "0")
            else:
                fail(f"unknown validation CSV run type: {row['run_type']}")
    require(baseline == vector_count, "validation CSV baseline closure")
    require(enabled == site_count * 2 * vector_count, "validation CSV enabled closure")
    require(unknown == 0, "validation CSV unknown records")
    require(detected_without_activity == 0, "validation CSV detected without activity")
    require(baseline_failures == 0, "validation CSV baseline failures")
    return {
        "baseline_records": baseline,
        "enabled_records": enabled,
        "total_records": baseline + enabled,
        "activated_records": activated,
        "detected_records": detected,
        "unknown_records": unknown,
        "detected_without_activity": detected_without_activity,
        "baseline_failures": baseline_failures,
        "selectors": sorted(selectors),
        "vector_slots": sorted(slots),
    }


def compile_testbench(root: Path, netlist: Path, testbench: Path, top: str, object_dir: Path, log: Path) -> Path:
    verilator = shutil.which("verilator")
    require(verilator is not None, "verilator is not in PATH")
    command = [
        verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal",
        "-j", "1", "--Mdir", str(object_dir), "--top-module", top,
        str(netlist), str(testbench),
    ]
    return_code = logged(command, log, 1_200)
    require(return_code == 0, f"Verilator build failed with return code {return_code}")
    binary = object_dir / f"V{top}"
    require(binary.is_file() and os.access(binary, os.X_OK), "validation-runner executable missing")
    return binary


def simulate(binary: Path, csv_path: Path, log: Path, resources: Path, token: str) -> None:
    return_code = logged([str(binary), f"+RESULT_CSV={csv_path}"], log, 1_800, resources)
    require(return_code == 0, f"validation-runner simulation failed with return code {return_code}")
    require(f"{token}=PASS" in log.read_text(errors="replace"), "validation-runner PASS token missing")


def qualify_dry_run(root: Path, overwrite: bool) -> int:
    require(root.name == "vlsi_fault_detection_v2", f"run from the project root, not {root}")
    source = Path(__file__).resolve()
    require(source.parent == root, "place this runner in the project root")
    verify_runner_source_safety(source)
    evidence = verify_frozen_inputs(root)
    train, validation, ordered_commitment = load_vector_contract(root)
    require(len(train) == 64 and len(validation) == 48, "vector contract closure")

    result_root = root / "results/hmac_fault_campaign_11d3/validation_runner_dry_run_11d3b"
    build_root = root / "build/hmac_validation_runner_dry_run_11d3b"
    runner_lock = root / "config/fault_campaign/hmac_validation_campaign_runner_lock_11d3b.json"
    audit_path = root / "results/hmac_fault_campaign_11d3/hmac_validation_campaign_runner_dry_run_freeze_11d3b.json"
    if runner_lock.exists() or audit_path.exists() or result_root.exists() or build_root.exists():
        require(overwrite, "Stage 11D-3B outputs exist; use --overwrite only to replace an unfrozen failed attempt")
        require(not audit_path.exists(), "refusing to overwrite a completed Stage 11D-3B freeze")
        for directory in (result_root, build_root):
            if directory.exists():
                shutil.rmtree(directory)
        if runner_lock.exists():
            runner_lock.unlink()

    require(shutil.disk_usage(root).free >= 5 * 1024**3, "less than 5 GiB free disk")
    result_root.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)
    netlist, mapping, sites = canonical_batch(root, DRY_RUN_BATCH)
    csv_path = result_root / "hmac_validation_runner_dry_run_results_11d3b.csv"
    replay_csv = result_root / "hmac_validation_runner_dry_run_replay_results_11d3b.csv"
    testbench = build_root / "tb_hmac_validation_batch_022.sv"
    manifest = result_root / "hmac_validation_runner_dry_run_testbench_manifest_11d3b.json"

    generated = generate_testbench(
        testbench, csv_path, manifest, DRY_RUN_BATCH, train, sites, True, root, ordered_commitment
    )
    build_log = result_root / "verilator_build.log"
    object_dir = build_root / "obj_dir"
    binary = compile_testbench(root, netlist, testbench, generated["top"], object_dir, build_log)
    simulation_log = result_root / "simulation.log"
    resources_log = result_root / "resources.log"
    simulate(binary, csv_path, simulation_log, resources_log, generated["result_token"])
    canonical_summary = check_csv(csv_path, DRY_RUN_BATCH, 2, 2, "TRAIN_CANARY")
    require(canonical_summary["selectors"] == DRY_RUN_SELECTORS, "dry-run selectors")
    require(canonical_summary["vector_slots"] == DRY_RUN_VECTOR_POSITIONS, "dry-run vector positions")

    replay_log = result_root / "simulation_replay.log"
    replay_resources = result_root / "resources_replay.log"
    simulate(binary, replay_csv, replay_log, replay_resources, generated["result_token"])
    replay_summary = check_csv(replay_csv, DRY_RUN_BATCH, 2, 2, "TRAIN_CANARY")
    require(canonical_summary == replay_summary, "dry-run replay summary")
    require(csv_path.read_bytes() == replay_csv.read_bytes(), "dry-run exact CSV replay")

    created_at = now()
    lock = {
        "stage": "11D-3B",
        "title": "HMAC VALIDATION CAMPAIGN RUNNER LOCK",
        "status": "FROZEN",
        "created_at_utc": created_at,
        "runner_version": VERSION,
        "runner_source": artifact(root, source),
        "stage_11d3a_execution_contract": evidence[
            "config/fault_campaign/hmac_validation_campaign_execution_contract_11d3a.json"
        ],
        "ordered_validation_commitment": ordered_commitment,
        "validation_vector_count": EXPECTED_VALIDATION_VECTORS,
        "batch_count": EXPECTED_BATCHES,
        "legal_sites": EXPECTED_LEGAL_SITES,
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "expected_baseline_records": EXPECTED_BASELINE_RECORDS,
        "expected_enabled_records": EXPECTED_ENABLED_RECORDS,
        "expected_total_records": EXPECTED_TOTAL_RECORDS,
        "execution": "SEQUENTIAL",
        "parallel_batches": 1,
        "build_jobs": 1,
        "checkpoint_after_each_batch": True,
        "resume_supported": True,
        "dry_run_partition": "TRAIN_CANARY",
        "dry_run_batch": DRY_RUN_BATCH,
        "dry_run_records": 10,
        "validation_vectors_executed": 0,
        "holdout_vectors_exposed": 0,
        "validation_execution_authorized": False,
        "validation_model_inference_authorized": False,
        "holdout_execution_authorized": False,
        "future_authorization_binding": [
            "RUNNER_SOURCE_SHA256",
            "RUNNER_LOCK_SHA256",
            "STAGE_11D3A_EXECUTION_CONTRACT_SHA256",
            "ORDERED_VALIDATION_COMMITMENT",
        ],
    }
    atomic_json(runner_lock, lock)

    outputs = {
        "runner_lock": artifact(root, runner_lock),
        "testbench": artifact(root, testbench),
        "testbench_manifest": artifact(root, manifest),
        "binary": artifact(root, binary),
        "canonical_csv": artifact(root, csv_path),
        "replay_csv": artifact(root, replay_csv),
        "build_log": artifact(root, build_log),
        "simulation_log": artifact(root, simulation_log),
        "simulation_replay_log": artifact(root, replay_log),
        "resources_log": artifact(root, resources_log),
        "replay_resources_log": artifact(root, replay_resources),
    }
    audit = {
        "stage": "11D-3B",
        "title": "VALIDATION CAMPAIGN RUNNER GENERATION AND DRY-RUN FREEZE",
        "status": "PASS",
        "runner_status": "FROZEN",
        "dry_run_status": "PASS",
        "created_at_utc": created_at,
        "runner_version": VERSION,
        "runner_source": artifact(root, source),
        "dry_run_batch": DRY_RUN_BATCH,
        "dry_run_partition": "TRAIN_CANARY",
        "dry_run_train_vectors": 2,
        "dry_run_validation_vectors": 0,
        "dry_run_holdout_vectors": 0,
        "dry_run_sites": 2,
        "dry_run_fault_instances": 4,
        "canonical_records": canonical_summary["total_records"],
        "replay_records": replay_summary["total_records"],
        "deterministic_replay": "PASS",
        "exact_csv_match": True,
        "unknown_records": canonical_summary["unknown_records"],
        "detected_without_activity": canonical_summary["detected_without_activity"],
        "baseline_failures": canonical_summary["baseline_failures"],
        "ordered_validation_commitment_verified": True,
        "validation_filter_static_proof": "PASS",
        "validation_vectors_executed": 0,
        "holdout_vectors_exposed": 0,
        "full_validation_testbenches_generated": 0,
        "validation_campaign_execution": "NOT_YET_AUTHORIZED",
        "validation_model_inference": "NOT_YET_AUTHORIZED",
        "holdout_campaign": "NOT_YET_AUTHORIZED",
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "canonical_fault_batches_modified": False,
        "frozen_models_modified": False,
        "input_evidence": evidence,
        "canonical_batch_22": {
            "netlist": artifact(root, netlist),
            "mapping": artifact(root, mapping),
        },
        "outputs": outputs,
        "next_gate": "STAGE 11D-3C — VALIDATION CAMPAIGN EXECUTION AUTHORIZATION FREEZE",
    }
    atomic_json(audit_path, audit)

    print("\nSTAGE 11D-3B — VALIDATION CAMPAIGN RUNNER GENERATION AND DRY-RUN FREEZE")
    print("Status                         : PASS")
    print("Runner status                  : FROZEN")
    print("Dry-run status                 : PASS")
    print("Runner version                 :", VERSION)
    print("Dry-run batch                  : 22")
    print("Dry-run partition              : TRAIN CANARY")
    print("Dry-run TRAIN vectors          : 2")
    print("Dry-run VALIDATION vectors     : 0")
    print("Dry-run HOLDOUT vectors        : 0")
    print("Dry-run sites                  : 2")
    print("Dry-run fault instances        : 4")
    print("Canonical records              :", canonical_summary["total_records"])
    print("Replay records                 :", replay_summary["total_records"])
    print("Deterministic replay           : PASS")
    print("Exact CSV match                : PASS")
    print("Unknown records                : 0")
    print("Detected without activity      : 0")
    print("Baseline failures              : 0")
    print("Validation filter proof        : PASS")
    print("Committed validation vectors   : 48")
    print("Validation vectors executed    : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("Execution                      : SEQUENTIAL")
    print("Parallel batches               : 1")
    print("Build jobs                     : 1")
    print("Checkpoint/resume              : ENABLED")
    print("Validation campaign execution  : NOT YET AUTHORIZED")
    print("Validation model inference     : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Runner lock                    :", runner_lock)
    print("Runner lock SHA                :", sha256(runner_lock))
    print("Audit                          :", audit_path)
    print("Audit SHA                      :", sha256(audit_path))
    print("Next gate                      : STAGE 11D-3C — VALIDATION CAMPAIGN EXECUTION AUTHORIZATION FREEZE")
    return 0


def authorize_validation(root: Path, source: Path, authorization_text: str | None) -> Path:
    require(authorization_text is not None, "validation execution requires a later frozen --authorization file")
    authorization = Path(authorization_text).resolve()
    require(authorization.is_file(), f"validation authorization missing: {authorization}")
    require(
        authorization.name == "hmac_validation_campaign_execution_authorization_11d3c.json",
        "unexpected validation authorization filename",
    )
    data = load_json(authorization)
    runner_lock = root / "config/fault_campaign/hmac_validation_campaign_runner_lock_11d3b.json"
    require(runner_lock.is_file(), "Stage 11D-3B runner lock missing")
    require(data.get("stage") == "11D-3C" and data.get("status") == "FROZEN", "validation authorization status")
    require(data.get("validation_campaign_execution") == "AUTHORIZED", "validation campaign authorization")
    require(data.get("validation_model_inference") == "NOT_AUTHORIZED", "model inference must remain separate")
    require(data.get("holdout_campaign") == "NOT_AUTHORIZED", "HOLDOUT must remain blocked")
    require(data.get("validation_vectors") == EXPECTED_VALIDATION_VECTORS, "authorized VALIDATION vector count")
    require(data.get("runner_sha256") == sha256(source), "authorization runner SHA")
    require(data.get("runner_lock_sha256") == sha256(runner_lock), "authorization runner-lock SHA")
    require(
        data.get("execution_contract_sha256")
        == FROZEN_INPUTS["config/fault_campaign/hmac_validation_campaign_execution_contract_11d3a.json"],
        "authorization execution-contract SHA",
    )
    return authorization


def execute_validation(root: Path, authorization_text: str | None, resume: bool, overwrite: bool) -> int:
    source = Path(__file__).resolve()
    require(root.name == "vlsi_fault_detection_v2", f"run from the project root, not {root}")
    require(source.parent == root, "place this runner in the project root")
    verify_runner_source_safety(source)
    verify_frozen_inputs(root)
    authorization = authorize_validation(root, source, authorization_text)
    _, validation, ordered_commitment = load_vector_contract(root)
    require(len(validation) == EXPECTED_VALIDATION_VECTORS, "validation vector closure")
    require(shutil.disk_usage(root).free >= 5 * 1024**3, "less than 5 GiB free disk")

    result_root = root / "results/hmac_validation_campaign_full_11d3"
    build_root = root / "build/hmac_validation_campaign_full_11d3"
    checkpoint = result_root / "hmac_validation_campaign_checkpoint.json"
    if result_root.exists() and not (resume or overwrite):
        fail(f"validation results exist: {result_root}; use --resume only for a verified interruption")
    if overwrite:
        require(not resume, "do not combine --overwrite and --resume")
        for directory in (result_root, build_root):
            if directory.exists():
                shutil.rmtree(directory)
    result_root.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    state = {
        "runner_version": VERSION,
        "mode": "VALIDATION",
        "runner_sha256": sha256(source),
        "authorization": artifact(root, authorization),
        "ordered_validation_commitment": ordered_commitment,
        "created_at_utc": now(),
        "updated_at_utc": now(),
        "completed_batches": [],
        "batches": {},
    }
    if resume:
        require(checkpoint.is_file(), "validation resume checkpoint missing")
        state = load_json(checkpoint)
        require(state.get("runner_sha256") == sha256(source), "validation checkpoint runner SHA")
        require(state.get("authorization", {}).get("sha256") == sha256(authorization), "validation checkpoint authorization SHA")
        require(state.get("ordered_validation_commitment") == ordered_commitment, "validation checkpoint commitment")

    print("HMAC VALIDATION FAULT CAMPAIGN RUNNER")
    print("Runner version       :", VERSION)
    print("Mode                 : VALIDATION")
    print("Batches              : 0-44")
    print("Validation vectors   : 48")
    print("Execution            : SEQUENTIAL")
    print("Build jobs           : 1")
    print("Authorization        : PASS")
    print("Model inference      : NOT PERFORMED")
    print("HOLDOUT access       : PROHIBITED")

    for batch_id in range(EXPECTED_BATCHES):
        key = f"{batch_id:03d}"
        prior = state.get("batches", {}).get(key)
        if batch_id in state.get("completed_batches", []):
            prior_csv = root / prior["csv"]["path"]
            require(prior_csv.is_file() and sha256(prior_csv) == prior["csv"]["sha256"], f"Batch {key} resume evidence")
            print(f"Batch {key}: RESUME SKIP (verified)")
            continue
        batch_result = result_root / f"batch_{key}"
        batch_build = build_root / f"batch_{key}"
        batch_result.mkdir(parents=True, exist_ok=True)
        batch_build.mkdir(parents=True, exist_ok=True)
        netlist, mapping, sites = canonical_batch(root, batch_id)
        csv_path = batch_result / f"hmac_validation_batch_{key}_results.csv"
        testbench = batch_build / f"tb_hmac_validation_batch_{key}.sv"
        manifest = batch_result / "testbench_manifest.json"
        generated = generate_testbench(
            testbench, csv_path, manifest, batch_id, validation, sites, False, root, ordered_commitment
        )
        print(f"Batch {key}: BUILD")
        build_log = batch_result / "verilator_build.log"
        binary = compile_testbench(root, netlist, testbench, generated["top"], batch_build / "obj_dir", build_log)
        print(f"Batch {key}: SIMULATE")
        simulation_log = batch_result / "simulation.log"
        resources_log = batch_result / "resources.log"
        simulate(binary, csv_path, simulation_log, resources_log, generated["result_token"])
        summary = check_csv(csv_path, batch_id, len(sites), EXPECTED_VALIDATION_VECTORS, "VALIDATION")
        require(summary["selectors"] == list(range(len(sites))), f"Batch {key} selector closure")
        require(summary["vector_slots"] == list(range(EXPECTED_VALIDATION_VECTORS)), f"Batch {key} vector closure")
        entry = {
            "status": "PASS",
            "batch_id": batch_id,
            "completed_at_utc": now(),
            "netlist": artifact(root, netlist),
            "mapping": artifact(root, mapping),
            "testbench": artifact(root, testbench),
            "manifest": artifact(root, manifest),
            "binary": artifact(root, binary),
            "csv": artifact(root, csv_path),
            "simulation_log": artifact(root, simulation_log),
            "resources_log": artifact(root, resources_log),
            "summary": summary,
        }
        state.setdefault("batches", {})[key] = entry
        state.setdefault("completed_batches", []).append(batch_id)
        state["completed_batches"] = sorted(set(state["completed_batches"]))
        state["updated_at_utc"] = now()
        atomic_json(checkpoint, state)
        print(f"Batch {key}: PASS records={summary['total_records']} csv_sha256={entry['csv']['sha256']}")

    state["status"] = "PASS"
    state["finished_at_utc"] = now()
    state["updated_at_utc"] = now()
    atomic_json(checkpoint, state)
    print("VALIDATION CAMPAIGN RUNNER CLOSURE")
    print("Status               : PASS")
    print("Completed batches    :", len(state["completed_batches"]))
    print("Checkpoint           :", checkpoint)
    print("Checkpoint SHA       :", sha256(checkpoint))
    print("Model inference      : NOT PERFORMED")
    print("HOLDOUT vectors      : 0")
    print("VALIDATION_CAMPAIGN_RUNNER_RESULT=PASS")
    return 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify or later execute the frozen OpenTitan HMAC validation campaign runner."
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--qualify-dry-run", action="store_true")
    mode.add_argument("--execute-validation", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = arguments()
    root = Path(options.project_root).resolve()
    if options.execute_validation:
        return execute_validation(root, options.authorization, options.resume, options.overwrite)
    require(not options.resume, "--resume is only valid with --execute-validation")
    require(options.authorization is None, "--authorization is only valid with --execute-validation")
    return qualify_dry_run(root, options.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
