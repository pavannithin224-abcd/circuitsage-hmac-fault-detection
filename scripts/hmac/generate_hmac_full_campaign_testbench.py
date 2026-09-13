#!/usr/bin/env python3
"""Generate deterministic Verilator testbenches for the HMAC fault campaign."""
from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path

VERSION = "HMAC-FULL-CAMPAIGN-TB-GENERATOR-v1"
SELECTION_SHA = "91ab01eda255a3b162957e067fb112b241a8ca2f95faeb6573450e91233479d1"
POOL_SHA = "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(text); tmp.replace(path)

def h256(value, field, vid):
    result = f"{value:064x}" if isinstance(value, int) else str(value).strip().lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise SystemExit(f"STOP: vector {vid} {field} is not 256-bit hexadecimal")
    return result

def find_sites(value):
    if isinstance(value, dict):
        for child in value.values():
            found = find_sites(child)
            if found is not None: return found
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and {"fault_site_id", "selector_code"} <= value[0].keys(): return value
        for child in value:
            found = find_sites(child)
            if found is not None: return found
    return None

def arguments():
    p = argparse.ArgumentParser(description="Generate one deterministic HMAC campaign testbench.")
    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument("--batch-id", type=int, required=True)
    p.add_argument("--mode", choices=("dry-run", "full"), required=True)
    p.add_argument("--output", required=True); p.add_argument("--result-csv", required=True)
    p.add_argument("--manifest"); p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def main():
    a = arguments(); root = Path(a.project_root).resolve(); bid = a.batch_id
    if bid not in range(45): raise SystemExit("STOP: batch ID must be 0-44")
    out = Path(a.output).resolve(); csv_path = Path(a.result_csv).resolve()
    manifest = Path(a.manifest).resolve() if a.manifest else None
    for target in (out, manifest):
        if target and target.exists() and not a.overwrite: raise SystemExit(f"STOP: output exists: {target}")
    selection_path = root / "config/fault_campaign/hmac_campaign_vector_selection_11c3b_b.json"
    pool_path = root / "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"
    mapping_path = root / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{bid:03d}/hmac_fault_batch_{bid:03d}_mapping.json"
    for path in (selection_path, pool_path, mapping_path):
        if not path.is_file(): raise SystemExit(f"STOP: missing input: {path}")
    if sha(selection_path) != SELECTION_SHA: raise SystemExit("STOP: vector-selection SHA mismatch")
    if sha(pool_path) != POOL_SHA: raise SystemExit("STOP: vector-pool SHA mismatch")
    selection, pool = load(selection_path), load(pool_path)
    ids, positions = selection.get("selected_vector_ids"), selection.get("selected_schedule_positions")
    if not isinstance(ids, list) or len(ids) != 64 or positions != list(range(64)):
        raise SystemExit("STOP: selected TRAIN-vector contract mismatch")
    records = pool.get("vectors")
    if not isinstance(records, list) or len(records) != 256: raise SystemExit("STOP: vector-pool contract mismatch")
    by_id = {int(x["vector_id"]): x for x in records}; selected = []
    for pos, vid in enumerate(ids):
        vid = int(vid); rec = by_id.get(vid)
        if rec is None or rec.get("split") != "TRAIN": raise SystemExit(f"STOP: invalid selected TRAIN vector {vid}")
        selected.append(dict(position=pos, vector_id=vid, key=h256(rec["key"], "key", vid), message=h256(rec["message"], "message", vid), digest=h256(rec["expected_hmac_sha256"], "digest", vid)))
    all_sites = find_sites(load(mapping_path)); expected_sites = 311 if bid == 44 else 512
    if all_sites is None or len(all_sites) != expected_sites: raise SystemExit("STOP: mapping site-count mismatch")
    if [int(x["selector_code"]) for x in all_sites] != list(range(expected_sites)):
        raise SystemExit("STOP: mapping selector contract mismatch")
    if a.mode == "dry-run":
        if bid != 22: raise SystemExit("STOP: frozen dry-run batch is 22")
        vpos, spos, prefix, result = [0, 63], [0, 511], "CAMPAIGN_DRY_RUN", "CAMPAIGN_DRY_RUN_RESULT"
    else:
        vpos, spos, prefix, result = list(range(64)), list(range(expected_sites)), "FULL_CAMPAIGN_BATCH", "FULL_CAMPAIGN_BATCH_RESULT"
    vectors, sites = [selected[x] for x in vpos], [all_sites[x] for x in spos]
    vc, sc, fc = len(vectors), len(sites), len(sites) * 2; enabled = fc * vc; total = vc + enabled
    top = f"tb_hmac_fault_batch_{bid:03d}_campaign"; dut = f"opentitan_hmac_sha256_msg32_faultbatch{bid:03d}"
    init = []
    for i, v in enumerate(vectors):
        init += [f"        vector_positions[{i}] = {v['position']};", f"        vector_ids[{i}] = {v['vector_id']};", f"        key_vectors[{i}] = 256'h{v['key']};", f"        message_vectors[{i}] = 256'h{v['message']};", f"        digest_vectors[{i}] = 256'h{v['digest']};"]
    for i, site in enumerate(sites):
        init += [f"        selected_selectors[{i}] = 9'd{int(site['selector_code'])};", f"        selected_site_ids[{i}] = \"{site['fault_site_id']}\";"]
    default_csv = str(csv_path).replace("\\", "\\\\").replace('"', '\\"')
    sv = f'''`timescale 1ns/1ps
module {top};
  localparam integer BATCH_ID={bid}, VECTOR_COUNT={vc}, SITE_COUNT={sc}, FAULT_COUNT={fc};
  localparam integer EXPECTED_ENABLED_RUNS={enabled}, EXPECTED_TOTAL_RECORDS={total};
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
{chr(10).join(init)}
    baseline_runs=0; enabled_runs=0; activated_runs=0; detected_runs=0; unknown_runs=0;
    detected_without_activity=0; baseline_digest_mismatches=0; baseline_latency_mismatches=0; baseline_timeouts=0;
    detected_instances=0; activated_unobserved_instances=0; unactivated_instances=0; unclassified_instances=0;
    csv_path=DEFAULT_CSV; plusarg_result=$value$plusargs("RESULT_CSV=%s",csv_path); csv_fd=$fopen(csv_path,"w");
    if(csv_fd==0) $fatal(1,"Could not open campaign CSV");
    $fdisplay(csv_fd,"batch_id,run_type,site_selection,selector,site_id,stuck_value,vector_slot,vector_id,fault_enable,cycles,baseline_cycles,timed_out,activity,detected,unknown,expected_digest,actual_digest");
    for(vector_index=0;vector_index<VECTOR_COUNT;vector_index=vector_index+1) begin
      run_transaction(vector_index,0,0,0,cycles_result,timeout_result,digest_result,activity_result,unknown_result);
      baseline_cycles[vector_index]=cycles_result; baseline_runs=baseline_runs+1;
      if(timeout_result) baseline_timeouts=baseline_timeouts+1;
      if(digest_result!==digest_vectors[vector_index]) baseline_digest_mismatches=baseline_digest_mismatches+1;
      if(cycles_result!=EXPECTED_BASELINE_CYCLES) baseline_latency_mismatches=baseline_latency_mismatches+1;
      if(unknown_result) unknown_runs=unknown_runs+1;
      $fdisplay(csv_fd,"%0d,BASELINE,-1,-1,BASELINE,-1,%0d,%0d,0,%0d,%0d,%0d,0,0,%0d,%064h,%064h",BATCH_ID,vector_positions[vector_index],vector_ids[vector_index],cycles_result,EXPECTED_BASELINE_CYCLES,timeout_result,unknown_result,digest_vectors[vector_index],digest_result);
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
          $fdisplay(csv_fd,"%0d,ENABLED,%0d,%0d,%s,%0d,%0d,%0d,1,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h",BATCH_ID,site_index,selected_selectors[site_index],selected_site_ids[site_index],stuck_index,vector_positions[vector_index],vector_ids[vector_index],cycles_result,baseline_cycles[vector_index],timeout_result,activity_result,detected_result,unknown_result,digest_vectors[vector_index],digest_result);
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
      $display("{result}=PASS"); $finish;
    end else $fatal(1,"{result}=FAIL");
  end
endmodule
'''
    atomic(out, sv)
    info = {"generator_version":VERSION,"status":"GENERATED","mode":a.mode,"batch_id":bid,"testbench":str(out),"testbench_sha256":sha(out),"result_csv":str(csv_path),"vector_positions":vpos,"vector_ids":[x["vector_id"] for x in vectors],"selectors":[int(x["selector_code"]) for x in sites],"site_ids":[x["fault_site_id"] for x in sites],"baseline_records":vc,"enabled_records":enabled,"total_records":total,"validation_vectors_exposed":0,"holdout_vectors_exposed":0,"mapping_sha256":sha(mapping_path),"selection_sha256":sha(selection_path),"vector_pool_sha256":sha(pool_path),"frozen_rtl_modified":False,"golden_netlist_modified":False}
    if manifest: atomic(manifest, json.dumps(info,indent=2,sort_keys=True)+"\n")
    print("HMAC FULL-CAMPAIGN TESTBENCH GENERATION"); print("Generator version    :",VERSION); print("Status               : PASS")
    print("Mode                 :",a.mode.upper()); print("Batch ID             :",bid); print("Vectors              :",vc)
    print("Sites                :",sc); print("Fault instances      :",fc); print("Total records        :",total)
    print("VALIDATION exposed   : 0"); print("HOLDOUT exposed      : 0"); print("Testbench            :",out); print("Testbench SHA        :",sha(out))
    return 0

if __name__ == "__main__": raise SystemExit(main())
