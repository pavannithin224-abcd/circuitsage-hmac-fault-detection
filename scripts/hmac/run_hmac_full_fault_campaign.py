#!/usr/bin/env python3
"""Sequential, resumable runner for the frozen OpenTitan HMAC fault campaign."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

VERSION="HMAC-FULL-FAULT-CAMPAIGN-RUNNER-v1"
POLICY_SHA="7e43e4b2e4c8c798ca708cbaa7f995f64bdccf4e27ad35478c212c39bce61ad6"
FROZEN={
 "config/fault_campaign/hmac_full_campaign_execution_policy_11c4g_b1.json":POLICY_SHA,
 "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json":"a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
 "results/hmac_fault_campaign_11c4/hmac_canonical_batch_generation_index_11c4g_a1.json":"7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
 "results/hmac_fault_campaign_11c4/hmac_intermediate_batch_runtime_smoke_freeze_11c4f_c_d.json":"1702ff1382bd2f19d31cdb4bd810619ece9026bb2a6b813e4803890013f5ef3f",
 "config/fault_campaign/hmac_reusable_batch_generator_lock_11c4e_e.json":"da59b47f95c2d16e648e637b1bacc064a35be1252038376cc96dd3151483763f",
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
def verify(path,expected):
 if not path.is_file(): raise SystemExit(f"STOP: missing frozen input: {path}")
 if sha(path)!=expected: raise SystemExit(f"STOP: frozen-input SHA mismatch: {path}")
def logged(command,log,seconds,resource=None):
 log.parent.mkdir(parents=True,exist_ok=True); actual=list(command)
 if resource is not None: actual=["/usr/bin/time","-v","-o",str(resource)]+actual
 with log.open("w") as stream:
  try: return subprocess.run(actual,stdout=stream,stderr=subprocess.STDOUT,timeout=seconds,check=False).returncode
  except subprocess.TimeoutExpired: stream.write(f"\nRUNNER_TIMEOUT_SECONDS={seconds}\n"); return 124
def args():
 p=argparse.ArgumentParser(description="Run the deterministic OpenTitan HMAC fault campaign.")
 p.add_argument("--project-root",default=str(Path.cwd())); mode=p.add_mutually_exclusive_group(required=True)
 mode.add_argument("--dry-run",action="store_true"); mode.add_argument("--execute-full",action="store_true")
 p.add_argument("--authorization"); p.add_argument("--resume",action="store_true"); p.add_argument("--overwrite",action="store_true")
 return p.parse_args()
def authorize(text):
 if not text: raise SystemExit("STOP: full execution requires a later frozen --authorization file")
 path=Path(text).resolve()
 if not path.is_file(): raise SystemExit(f"STOP: authorization missing: {path}")
 data=json.loads(path.read_text())
 if data.get("complete_campaign_execution")!="AUTHORIZED" or data.get("policy_sha256")!=POLICY_SHA or data.get("batch_22_dry_run")!="PASS":
  raise SystemExit("STOP: full-campaign authorization contract is not satisfied")
 return path
def check_csv(path,mode,bid,sites):
 if not path.is_file() or not path.stat().st_size: raise RuntimeError("missing campaign CSV")
 eb,ee=(2,8) if mode=="dry-run" else (64,sites*128); baseline=enabled=unknown=dwa=0; selectors=set(); slots=set()
 with path.open(newline="") as stream:
  reader=csv.DictReader(stream); required={"batch_id","run_type","selector","vector_slot","cycles","timed_out","activity","detected","unknown","expected_digest","actual_digest"}
  if reader.fieldnames is None or not required<=set(reader.fieldnames): raise RuntimeError("CSV header mismatch")
  for row in reader:
   if int(row["batch_id"])!=bid: raise RuntimeError("CSV batch mismatch")
   unknown+=int(row["unknown"])
   if row["run_type"]=="BASELINE":
    baseline+=1
    if row["timed_out"]!="0" or int(row["cycles"])!=343 or row["expected_digest"].lower()!=row["actual_digest"].lower(): raise RuntimeError("baseline failure in CSV")
   elif row["run_type"]=="ENABLED":
    enabled+=1; selectors.add(int(row["selector"])); slots.add(int(row["vector_slot"])); dwa+=int(row["detected"]=="1" and row["activity"]=="0")
   else: raise RuntimeError("unknown CSV run type")
 if (baseline,enabled)!=(eb,ee) or unknown or dwa: raise RuntimeError(f"CSV closure failure baseline={baseline} enabled={enabled} unknown={unknown} detected_without_activity={dwa}")
 if mode=="dry-run" and (selectors!={0,511} or slots!={0,63}): raise RuntimeError("dry-run selector/vector contract mismatch")
 return {"baseline_records":baseline,"enabled_records":enabled,"total_records":baseline+enabled,"unknown_records":unknown,"detected_without_activity":dwa,"selectors":sorted(selectors),"vector_slots":sorted(slots)}
def main():
 a=args(); root=Path(a.project_root).resolve()
 for rel,digest in FROZEN.items(): verify(root/rel,digest)
 generator=root/"scripts/hmac/generate_hmac_full_campaign_testbench.py"; runner=root/"scripts/hmac/run_hmac_full_fault_campaign.py"
 if not generator.is_file(): raise SystemExit(f"STOP: generator missing: {generator}")
 if shutil.disk_usage(root).free<6*1024**3: raise SystemExit("STOP: less than 6 GiB free disk")
 verilator=shutil.which("verilator")
 if not verilator: raise SystemExit("STOP: verilator is not in PATH")
 if a.execute_full:
  auth=authorize(a.authorization); mode="full"; batches=list(range(45)); results=root/"results/hmac_fault_campaign_full_11c5"; builds=root/"build/hmac_fault_campaign_full_11c5"; checkpoint=results/"hmac_full_campaign_checkpoint.json"
 else:
  auth=None; mode="dry-run"; batches=[22]; results=root/"results/hmac_fault_campaign_11c4/full_campaign_dry_run_11c4g_b2"; builds=root/"build/hmac_full_campaign_dry_run_11c4g_b2"; checkpoint=results/"hmac_batch22_dry_run_checkpoint.json"
 if results.exists() and not (a.resume or a.overwrite): raise SystemExit(f"STOP: results exist: {results}; use --resume only for a verified interruption")
 results.mkdir(parents=True,exist_ok=True); builds.mkdir(parents=True,exist_ok=True)
 state={"runner_version":VERSION,"mode":mode,"policy_sha256":POLICY_SHA,"generator_sha256":sha(generator),"runner_sha256":sha(runner) if runner.is_file() else None,"created_at":now(),"updated_at":now(),"completed_batches":[],"batches":{}}
 if auth: state.update(authorization=str(auth),authorization_sha256=sha(auth))
 if a.resume:
  if not checkpoint.is_file(): raise SystemExit("STOP: resume checkpoint missing")
  state=json.loads(checkpoint.read_text())
  if state.get("policy_sha256")!=POLICY_SHA or state.get("generator_sha256")!=sha(generator): raise SystemExit("STOP: checkpoint contract mismatch")
 print("HMAC FULL FAULT CAMPAIGN RUNNER"); print("Runner version       :",VERSION); print("Mode                 :",mode.upper()); print("Batches              :",batches); print("Execution            : SEQUENTIAL"); print("Build jobs           : 1"); print("Full authorization   :","PASS" if auth else "NOT PRESENT")
 for bid in batches:
  key=f"{bid:03d}"; old=state.get("batches",{}).get(key)
  if bid in state.get("completed_batches",[]):
   prior=Path(old["csv"])
   if prior.is_file() and sha(prior)==old["csv_sha256"]: print(f"Batch {key}: RESUME SKIP (verified)"); continue
   raise SystemExit(f"STOP: completed Batch {key} evidence mismatch")
  build=builds/f"batch_{key}"; result=results/f"batch_{key}"; build.mkdir(parents=True,exist_ok=True); result.mkdir(parents=True,exist_ok=True)
  netlist=root/f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.v"
  mapping=root/f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
  if not netlist.is_file() or not mapping.is_file(): raise SystemExit(f"STOP: canonical Batch {key} inputs missing")
  sites=311 if bid==44 else 512; tb=build/f"tb_hmac_fault_batch_{key}_campaign.sv"; manifest=result/"testbench_manifest.json"; csv_path=result/f"hmac_fault_batch_{key}_{mode.replace('-','_')}_results.csv"
  genlog=result/"testbench_generation.log"; cmd=[sys.executable,str(generator),"--project-root",str(root),"--batch-id",str(bid),"--mode",mode,"--output",str(tb),"--result-csv",str(csv_path),"--manifest",str(manifest),"--overwrite"]
  if logged(cmd,genlog,120): raise SystemExit(f"STOP: Batch {key} testbench generation failed")
  top=f"tb_hmac_fault_batch_{key}_campaign"; obj=build/"obj_dir"; buildlog=result/"verilator_build.log"
  command=[verilator,"--binary","--timing","--assert","-Wall","-Wno-fatal","-j","1","--Mdir",str(obj),"--top-module",top,str(netlist),str(tb)]
  print(f"Batch {key}: BUILD")
  rc=logged(command,buildlog,1200)
  if rc: raise SystemExit(f"STOP: Batch {key} build failed rc={rc}")
  binary=obj/f"V{top}"
  if not binary.is_file() or not os.access(binary,os.X_OK): raise SystemExit(f"STOP: Batch {key} executable missing")
  simlog=result/"simulation.log"; resources=result/"resources.log"; print(f"Batch {key}: SIMULATE")
  rc=logged([str(binary),f"+RESULT_CSV={csv_path}"],simlog,1800,resources)
  if rc: raise SystemExit(f"STOP: Batch {key} simulation failed rc={rc}")
  token="CAMPAIGN_DRY_RUN_RESULT=PASS" if mode=="dry-run" else "FULL_CAMPAIGN_BATCH_RESULT=PASS"
  if token not in simlog.read_text(errors="replace"): raise SystemExit(f"STOP: Batch {key} PASS token missing")
  summary=check_csv(csv_path,mode,bid,sites)
  record={"status":"PASS","batch_id":bid,"completed_at":now(),"netlist":str(netlist),"netlist_sha256":sha(netlist),"mapping":str(mapping),"mapping_sha256":sha(mapping),"testbench":str(tb),"testbench_sha256":sha(tb),"manifest":str(manifest),"manifest_sha256":sha(manifest),"binary":str(binary),"binary_sha256":sha(binary),"csv":str(csv_path),"csv_sha256":sha(csv_path),"simulation_log":str(simlog),"simulation_log_sha256":sha(simlog),"resource_log":str(resources),"resource_log_sha256":sha(resources),"csv_summary":summary}
  state.setdefault("batches",{})[key]=record; state.setdefault("completed_batches",[]).append(bid); state["completed_batches"]=sorted(set(state["completed_batches"])); state["updated_at"]=now(); atomic(checkpoint,state)
  print(f"Batch {key}: PASS records={summary['total_records']} csv_sha256={record['csv_sha256']}")
 state.update(status="PASS",finished_at=now(),updated_at=now()); atomic(checkpoint,state)
 print("CAMPAIGN RUNNER CLOSURE"); print("Status               : PASS"); print("Mode                 :",mode.upper()); print("Completed batches    :",len(state["completed_batches"])); print("Checkpoint           :",checkpoint); print("Checkpoint SHA       :",sha(checkpoint))
 if mode=="dry-run": print("Full campaign        : NOT AUTHORIZED"); print("BATCH22_DRY_RUN_RESULT=PASS")
 else: print("FULL_CAMPAIGN_RUNNER_RESULT=PASS")
 return 0
if __name__=="__main__": raise SystemExit(main())
