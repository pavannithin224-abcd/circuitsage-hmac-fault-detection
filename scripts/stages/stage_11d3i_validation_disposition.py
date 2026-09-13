#!/usr/bin/env python3
"""Freeze validation disposition and HOLDOUT readiness from existing evidence.

Run in ~/vlsi_fault_detection_v2. Reads frozen reports and the saved bootstrap
distribution; hashes all linked inputs/predictions. Does not deserialize models,
read sample labels, rerun inference, retrain, select thresholds, or open HOLDOUT.
Disposition PASS verifies documentation; validation performance stays NOT_MET.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import numpy as np

R = "results/hmac_fault_campaign_11d3/"
C = "config/diagnostic_model/"
VERIFIER = "stage_11d3e_validation_dataset_integrity.py"
VERIFIER_SHA = "b2bdcbfc98510aaff4f1bfea55706cf7767a84607f59dbcd2c9267cba02f15dc"
EVALUATOR = "stage_11d3h_locked_validation.py"
AUDIT_H = R + "hmac_locked_validation_four_model_freeze_11d3h.json"
AUTH_H = C + "hmac_validation_inference_authorization_11d3h.json"
CONTRACT = C + "hmac_validation_inference_contract_11d3g.json"
AUDIT_G = R + "hmac_validation_feature_inference_contract_freeze_11d3g.json"
LOG_H = R + "stage_11d3h_evaluation_20260910_003217.log"
RESOURCE_H = R + "stage_11d3h_resources_20260910_003217.log"
EXPECTED = {
    VERIFIER: VERIFIER_SHA,
    EVALUATOR: "9dd2ad2595b84950f26c6a668348934b05e0bc6c1cfd37120645b070bd6c1752",
    AUDIT_H: "d914ddf75d7477bb024dbc8badeef5c47e23504e9098a82ba3c5fb8e210cd902",
    AUTH_H: "7abc756058b24ccd9b5b6612137d1ba7f15121ebfee9e3d63e08a43c2604d5f1",
    CONTRACT: "080b3e4efc940ae6d1abcdc663c6d08bff96dba2ffaae00ddd7a0741a3d1b359",
    AUDIT_G: "04b0ecada12085ba9c9f66cb19b5f01933dab885d8117bfa98dd6c3818f42347",
    LOG_H: "7d5c41e415d70f0db90af9543aa1b60d323a7a8f0190b1cdf97135dd1a0168a0",
    RESOURCE_H: "b27626ae38d1e1fdadf4ee950ce15df4d702cc82b7489531f338a5061121d27b",
}
N, S, V, POSITIVE = 2192544, 22839, 48, 964973
NAMES = ("hybrid", "gnn", "deep", "baseline")
MODEL_IDS = ("HYBRID_FUSION_MLP_11D2C", "DIR_SGC_11D1D", "DEEP_MLP_11C5K", "CONVENTIONAL_LOGREG_11C5H")
REPORTED_MCC = dict(zip(NAMES, (.50934869, .36085870, .08894234, .13614390)))
THRESHOLDS = (.4965, .86327695216798483, .442, .98536854982376099)
OUTPUTS = {
    "disposition_policy": C + "hmac_validation_result_disposition_policy_11d3i.json",
    "holdout_readiness": C + "hmac_holdout_readiness_policy_11d3i.json",
    "criteria": R + "hmac_validation_acceptance_criteria_11d3i.csv",
    "comparator_registry": R + "hmac_validation_comparator_registry_11d3i.csv",
    "report": R + "hmac_validation_result_disposition_report_11d3i.md",
    "audit": R + "hmac_validation_result_disposition_holdout_readiness_freeze_11d3i.json",
}
NEXT_GATE = "VALIDATION FAILURE REVIEW AND PROJECT DISPOSITION; HOLDOUT REMAINS BLOCKED"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load_verifier(root):
    path = root/VERIFIER
    require(path.is_file() and sha(path) == VERIFIER_SHA, "Stage 3E verifier missing or modified")
    module = types.ModuleType("_verified_3e_disposition")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def close(a, b, label, tolerance=1e-12):
    require(isinstance(a, (int,float)) and not isinstance(a,bool) and math.isfinite(a)
            and math.isclose(a,b,rel_tol=0,abs_tol=tolerance), f"numeric agreement: {label}: {a!r} vs {b!r}")


def check_metric_counts(m, label):
    for key in ("tn","fp","fn","tp","samples","positive_samples"):
        require(type(m[key]) is int and m[key] >= 0, f"{label}: invalid count {key}")
    tn, fp, fn, tp = (m[k] for k in ("tn","fp","fn","tp"))
    require(tn+fp+fn+tp == m["samples"] and fn+tp == m["positive_samples"], f"{label}: confusion totals")
    denominator = math.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    close(m["mcc"], (float(tp)*tn-float(fp)*fn)/denominator if denominator else 0.0, label+" MCC")
    def check_ratio(key, a, b):
        if b:
            close(m[key], a/b, label+" "+key)
        else:
            require(m[key] is None, f"{label}: undefined {key} must be null")
    check_ratio("precision", tp, tp+fp)
    check_ratio("recall", tp, tp+fn)
    check_ratio("specificity", tn, tn+fp)
    check_ratio("f1_score", 2*tp, 2*tp+fp+fn)
    if tp+fn and tn+fp:
        close(m["balanced_accuracy"], (tp/(tp+fn)+tn/(tn+fp))/2, label+" balanced accuracy")
    else:
        require(m["balanced_accuracy"] is None, f"{label}: undefined balanced accuracy")
    close(m["positive_prevalence"], m["positive_samples"]/m["samples"], label+" prevalence")
    for key in ("pr_auc","roc_auc","brier_score"):
        value = m[key]
        require(value is None or (type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1), f"{label}: invalid {key}")


def verify_reports(e, v):
    print("FROZEN VALIDATION EVIDENCE VERIFICATION", flush=True)
    for path,digest in EXPECTED.items():
        e.verify(path,digest)
        print(f"  {Path(path).name}: OK", flush=True)
    a, auth, c, g = (e.load(path) for path in (AUDIT_H,AUTH_H,CONTRACT,AUDIT_G))
    v.fields(a, {"stage":"11D-3H","status":"PASS","evaluation_status":"FROZEN","comparison_status":"FROZEN",
        "validation_target":"NOT_MET","samples":N,"sites":S,"validation_vectors":V,
        "positive_samples":POSITIVE,"negative_samples":N-POSITIVE,
        "exact_prediction_replay":"PASS","prediction_npz_replay":"PASS","model_state_unchanged":True,
        "model_fitting_calls":0,"threshold_selection_calls":0,"scaler_fitting_calls":0,"graph_propagation_calls":0,
        "holdout_samples":0,"development_labels_reopened":False,"frozen_inputs_modified":False,
        "holdout_campaign":"NOT_AUTHORIZED"}, "3H audit")
    e.link(a["source"],EVALUATOR); e.link(a["authorization"],AUTH_H)
    require(isinstance(a["input_evidence"],dict), "3H input evidence inventory")
    for i,(path,record) in enumerate(a["input_evidence"].items(),1):
        e.link(record,path)
        if i%100 == 0:
            print(f"  Prior frozen input hashes: {i}/{len(a['input_evidence'])}",flush=True)
    for path in (EVALUATOR,CONTRACT,AUDIT_G,AUTH_H,VERIFIER):
        require(a["input_evidence"][path] == e.records[path], f"3H evidence anchor: {path}")
    v.fields(auth,{"stage":"11D-3H","status":"PASS","authorization_status":"FROZEN",
        "validation_inference_execution_authorized":True,"retraining":"PROHIBITED","threshold_changes":"PROHIBITED",
        "holdout_campaign":"NOT_AUTHORIZED","prior_contract_modified":False},"3H authorization")
    require(auth["evaluator_source"] == a["source"] and auth["inference_contract"] == e.records[CONTRACT]
        and auth["feature_freeze"] == e.records[AUDIT_G], "authorization source/contract binding")
    v.fields(c,{"stage":"11D-3G","status":"PASS","contract_status":"FROZEN","dataset_partition":"VALIDATION",
        "prediction_samples":N,"physical_sites":S,"validation_vectors":V,"samples_per_site":96},"inference contract")
    require(c["acceptance_contract"] == auth["acceptance_contract"] == g["acceptance_contract"], "frozen acceptance binding")
    require(auth["sample_identity_sha256"] == c["sample_identity_sha256"] == g["sample_identity_sha256"], "identity binding")
    require(a["models"] == c["models"] == auth["models"] == g["models"], "frozen four-model binding")
    require(len(a["models"]) == 4, "four comparator count")
    for spec,model_id,threshold in zip(a["models"],MODEL_IDS,THRESHOLDS):
        require(spec["model_id"] == model_id and spec["threshold"] == threshold, "comparator identity/threshold")
        e.link(spec["model_artifact"],spec["model"])
        e.link(spec["selection_lock"],spec["lock"])
    expected_outputs = {name+"_predictions" for name in NAMES} | {"metrics","breakdowns","paired_site_bootstrap","acceptance","bootstrap_distribution"}
    require(set(a["outputs"]) == expected_outputs, "evaluation output inventory")
    for record in a["outputs"].values():
        e.link(record)
    reports = {key:e.load(a["outputs"][key]["path"]) for key in ("metrics","paired_site_bootstrap","acceptance")}
    # Breakdown file is a JSON list, unlike Evidence.load's object-only interface.
    breakdowns = json.loads(e.path(a["outputs"]["breakdowns"]["path"]).read_text(),object_pairs_hook=v.unique_object)
    require(reports["metrics"] == a["metrics"] and set(a["metrics"]) == set(NAMES), "metrics report binding")
    require(reports["paired_site_bootstrap"] == a["paired_site_intervals"], "bootstrap summary binding")
    for key in ("validation_target","checks","failure_disposition","holdout_campaign"):
        require(reports["acceptance"][key] == a[key], f"acceptance/audit binding: {key}")
    for name,m in a["metrics"].items():
        require(m["samples"] == N and m["positive_samples"] == POSITIVE, f"{name}: sample count")
        check_metric_counts(m,name)
        close(m["mcc"],REPORTED_MCC[name],name+" posted MCC",5.1e-9)
    verify_breakdowns(breakdowns,a["metrics"],c["required_breakdowns"])
    b=reports["paired_site_bootstrap"]
    v.fields(b,{"replicates":1000,"seed":20260909,"method":"paired physical-site percentile bootstrap"},"bootstrap summary")
    require(c["site_confidence_intervals"]["replicates"] == 1000 and c["site_confidence_intervals"]["random_seed"] == 20260909
        and c["site_confidence_intervals"]["confidence"] == .95, "frozen bootstrap settings")
    columns=list(NAMES)+["hybrid_minus_"+name for name in NAMES[1:]]
    with np.load(e.path(a["outputs"]["bootstrap_distribution"]["path"]),allow_pickle=False) as z:
        require(set(z.files) == {"mcc_distribution","column_names"},"bootstrap NPZ inventory")
        require(z["column_names"].tolist() == columns,"bootstrap column order")
        dist=z["mcc_distribution"]
    require(dist.shape == (1000,7) and np.isfinite(dist).all(),"bootstrap shape/finite values")
    require(np.all(np.abs(dist[:,:4])<=1+1e-12),"bootstrap MCC range")
    for j in (1,2,3):
        require(np.array_equal(dist[:,j+3],dist[:,0]-dist[:,j]),"paired bootstrap delta arithmetic")
    require(set(b["mcc_intervals"]) == set(columns),"bootstrap interval inventory")
    for j,name in enumerate(columns):
        low,high=np.quantile(dist[:,j],[.025,.975])
        close(b["mcc_intervals"][name][0],float(low),name+" CI lower")
        close(b["mcc_intervals"][name][1],float(high),name+" CI upper")
    v.verify_resource_log(e.path(RESOURCE_H))
    log=e.path(LOG_H).read_text()
    for snippet in ("Evaluation status              : PASS / FROZEN", "Validation target              : NOT_MET", EXPECTED[AUDIT_H]):
        require(snippet in log, "evaluation log/audit agreement")
    return a,c


def verify_breakdowns(rows, scores, required):
    require(isinstance(rows,list) and set(row["group"] for row in rows) == set(required),"required breakdown families")
    for family in required:
        groups=[row for row in rows if row["group"] == family]
        require(len({str(row["value"]) for row in groups}) == len(groups),f"duplicate breakdown: {family}")
        for name in NAMES:
            totals={key:0 for key in ("samples","positive_samples","tn","fp","fn","tp")}
            for row in groups:
                require(set(row["models"]) == set(NAMES),"breakdown comparator inventory")
                metric=row["models"][name]
                if metric["status"] == "NOT_REPRESENTED":
                    require(metric["samples"] == 0 and bool(metric["reason"]),"empty group reporting")
                else:
                    require(metric["status"] == "REPRESENTED" and metric["samples"]>0,"breakdown status")
                    check_metric_counts(metric,f"{family}/{row['value']}/{name}")
                    for key in totals:
                        totals[key]+=metric[key]
            require(all(totals[k] == scores[name][k] for k in totals),f"breakdown totals: {family}/{name}")


def criterion_rows(scores, intervals, policy):
    m=scores["hybrid"]; mini=policy["hybrid_minimum_validity"]; absolute=policy["hybrid_absolute_performance"]
    retention=policy["hybrid_generalization_retention"]; comparator=policy["hybrid_comparator_confirmation"]
    require(mini["pr_auc_gt_positive_prevalence"] is True and mini["both_predicted_classes_required"] is True,"minimum validity policy")
    require(comparator["primary_comparator"] == "DIR_SGC_11D1D","primary comparator policy")
    close(retention["frozen_dev_site_test_mcc"]-retention["maximum_allowed_mcc_drop"],retention["validation_mcc_minimum"],"retention threshold")
    spec=[
        ("positive_mcc","minimum_validity",m["mcc"],">",mini["mcc_gt"]),
        ("balanced_accuracy","minimum_validity",m["balanced_accuracy"],">",mini["balanced_accuracy_gt"]),
        ("pr_auc_over_prevalence","minimum_validity",m["pr_auc"],">",m["positive_prevalence"]),
        ("both_predicted_classes","minimum_validity",int(m["tp"]+m["fp"]>0)+int(m["tn"]+m["fn"]>0),"==",2),
        ("absolute_mcc","absolute_performance",m["mcc"],">=",absolute["mcc_minimum"]),
        ("absolute_balanced_accuracy","absolute_performance",m["balanced_accuracy"],">=",absolute["balanced_accuracy_minimum"]),
        ("absolute_f1","absolute_performance",m["f1_score"],">=",absolute["f1_minimum"]),
        ("absolute_recall","absolute_performance",m["recall"],">=",absolute["recall_minimum"]),
        ("mcc_retention","generalization_retention",m["mcc"],">=",retention["validation_mcc_minimum"]),
        ("hybrid_gain","comparator_confirmation",m["mcc"]-scores["gnn"]["mcc"],">=",comparator["hybrid_minus_gnn_mcc_minimum"]),
        ("positive_paired_ci","comparator_confirmation",intervals["hybrid_minus_gnn"][0],">",comparator["paired_physical_site_mcc_delta_ci_lower_must_exceed"]),
    ]
    rows=[]
    for name,family,actual,op,required in spec:
        require(type(actual) in (int,float) and math.isfinite(actual),f"criterion {name}: undefined actual")
        passed=actual>required if op==">" else actual>=required if op==">=" else actual==required
        rows.append({"criterion":name,"family":family,"actual":actual,"operator":op,"required":required,"passed":bool(passed)})
    return rows


def disposition(a,c):
    policy=c["acceptance_contract"]
    rows=criterion_rows(a["metrics"],a["paired_site_intervals"]["mcc_intervals"],policy)
    checks={row["criterion"]:row["passed"] for row in rows}
    require(checks == a["checks"],"independent acceptance reconstruction disagrees with frozen evaluation")
    failed=[row["criterion"] for row in rows if not row["passed"]]
    require(failed and a["validation_target"] == "NOT_MET","expected frozen validation failure")
    require("mcc_retention" in failed,"reported MCC retention failure absent")
    require(a["failure_disposition"] == policy["failure_disposition"],"frozen failure disposition")
    require("OPEN HOLDOUT_TEST" in policy["failure_disposition"] and "NOT_MET" in policy["failure_disposition"],"unexpected failure policy; review exact contract")
    scores=a["metrics"]; ranking=sorted(NAMES,key=lambda name:scores[name]["mcc"],reverse=True)
    retention=policy["hybrid_generalization_retention"]
    result={"stage":"11D-3I","status":"PASS","disposition_status":"FROZEN","validation_target":"NOT_MET",
        "validation_result_accepted_for_advancement":False,"failed_criteria":failed,"criteria":rows,
        "criterion_family_status":{family:("PASS" if all(r["passed"] for r in rows if r["family"]==family) else "NOT_MET") for family in sorted({r["family"] for r in rows})},
        "model_ranking_by_validation_mcc":ranking,"best_observed_validation_model":ranking[0],
        "best_observed_does_not_override_acceptance":True,"metrics":scores,
        "hybrid_dev_site_test_mcc":retention["frozen_dev_site_test_mcc"],
        "hybrid_validation_mcc":scores["hybrid"]["mcc"],
        "hybrid_mcc_drop":retention["frozen_dev_site_test_mcc"]-scores["hybrid"]["mcc"],
        "allowed_mcc_drop":retention["maximum_allowed_mcc_drop"],
        "retention_shortfall":retention["validation_mcc_minimum"]-scores["hybrid"]["mcc"],
        "hybrid_minus_gnn_mcc":scores["hybrid"]["mcc"]-scores["gnn"]["mcc"],
        "hybrid_minus_gnn_95_site_ci":a["paired_site_intervals"]["mcc_intervals"]["hybrid_minus_gnn"],
        "holdout_readiness":"BLOCKED_VALIDATION_TARGET_NOT_MET","holdout_campaign":"NOT_AUTHORIZED",
        "model_retraining":"PROHIBITED","threshold_changes":"PROHIBITED","acceptance_criteria_changes":"PROHIBITED_WITHIN_THIS_FROZEN_RUN",
        "failure_disposition":policy["failure_disposition"],"project_final_freeze":"NOT_DECLARED",
        "scope":a["scope"],"statistical_limitations":a["statistical_limitations"],
        "next_gate":NEXT_GATE}
    return result,rows


def report_text(d):
    lines=["# Stage 11D-3I — Validation Result Disposition and Holdout Readiness", "",
        "Disposition verification: PASS. Validation performance: NOT_MET. HOLDOUT readiness: BLOCKED.", "",
        "The validation evaluation is preserved as a valid measured result. The hybrid remains the highest-MCC model among the four frozen comparators, but it does not satisfy all predeclared advancement criteria.", "",
        "| Model | Validation MCC | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):return "undefined" if value is None else f"{value:.8f}"
    for name in d["model_ranking_by_validation_mcc"]:
        m=d["metrics"][name]
        lines.append("| "+name+" | "+" | ".join(fmt(m[k]) for k in ("mcc","balanced_accuracy","precision","recall","f1_score","pr_auc","roc_auc","brier_score"))+" |")
    lines.extend(["", "## Acceptance criteria", "", "| Criterion | Actual | Rule | Required | Result |", "|---|---:|---|---:|---|"])
    for r in d["criteria"]:
        lines.append(f"| {r['criterion']} | {r['actual']:.8f} | {r['operator']} | {r['required']:.8f} | {'PASS' if r['passed'] else 'NOT_MET'} |")
    lines.extend(["", "Failed criteria: "+", ".join(d["failed_criteria"])+".", "",
        f"Hybrid MCC fell from {d['hybrid_dev_site_test_mcc']:.8f} on the development site test to {d['hybrid_validation_mcc']:.8f} on validation. The drop is {d['hybrid_mcc_drop']:.8f}; the allowed drop was {d['allowed_mcc_drop']:.8f}. The retention threshold was missed by {d['retention_shortfall']:.8f} MCC.", "",
        f"Hybrid–GNN MCC gain: {d['hybrid_minus_gnn_mcc']:.8f}; paired physical-site 95% interval: {d['hybrid_minus_gnn_95_site_ci']}.", "",
        "## Interpretation and scope", "",d["scope"],"",d["statistical_limitations"],"",
        "MCC is a correlation measure, not percentage accuracy. These samples concern pre-simulation detectability of persistent SA0/SA1 faults on the same HMAC circuit. They do not establish transient-fault performance or generalization to other chips.","",
        "The retention comparison uses a development site-test score and an all-site validation score, as prescribed by the frozen contract. It is not a controlled estimate of a stimulus-only effect. The retained breakdown reports separate original development site strata.","",
        "## Frozen disposition", "",
        "Preserve the models, thresholds, acceptance rules, predictions, and failed validation result. HOLDOUT execution and inference remain unauthorized. No retraining, threshold adjustment, or final project completion is authorized by this disposition.","",
        "The next review should document the measured limitations and decide the project disposition under the frozen protocol. This report does not establish the cause of the performance drop or prescribe tuning using validation results.","",
        "This stage checks report consistency, confusion-derived metrics, breakdown totals, and saved bootstrap quantiles. Prediction files and their prior evidence are hash-verified; inference and rank-based metrics are not recomputed.", ""])
    return "\n".join(lines)


def write_json(path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
        f.flush();os.fsync(f.fileno())


def write_csv(path,rows):
    with path.open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())


def record(path,relative):
    return {"path":relative,"sha256":sha(path),"bytes":path.stat().st_size}


def run(root):
    require(root.name=="vlsi_fault_detection_v2" and Path(__file__).resolve().parent==root,"copy script to ~/vlsi_fault_detection_v2 and run there")
    v=load_verifier(root);e=v.Evidence(root)
    print("STAGE 11D-3I — VALIDATION RESULT DISPOSITION AND HOLDOUT READINESS",flush=True)
    with v.campaign_lock(e):
        for path in OUTPUTS.values():
            require(not (root/path).exists() and not (root/path).is_symlink(),f"output already exists: {path}; preserve completed or partial evidence")
        source=e.verify(Path(__file__).name,sha(Path(__file__)))
        a,c=verify_reports(e,v)
        d,criteria=disposition(a,c)
        print("\nINDEPENDENT ACCEPTANCE RECONSTRUCTION",flush=True)
        for row in criteria:
            print(f"  {row['criterion']:28}: {'PASS' if row['passed'] else 'NOT_MET'} (actual={row['actual']:.8f}; required {row['operator']} {row['required']:.8f})",flush=True)
        stamp=datetime.now(timezone.utc).isoformat()
        d.update(created_at_utc=stamp,source=source,validation_audit=e.records[AUDIT_H],inference_contract=e.records[CONTRACT])
        readiness={"stage":"11D-3I","status":"PASS","readiness_policy_status":"FROZEN",
            "readiness":"BLOCKED_VALIDATION_TARGET_NOT_MET","validation_target":"NOT_MET","failed_criteria":d["failed_criteria"],
            "holdout_campaign_execution_authorized":False,"holdout_model_inference_authorized":False,
            "holdout_vector_access_authorized":False,"model_retraining_authorized":False,"threshold_changes_authorized":False,
            "failure_disposition":d["failure_disposition"],"validation_audit":e.records[AUDIT_H],
            "authorization_issued":False,"project_final_freeze":"NOT_DECLARED","next_gate":NEXT_GATE}
        registry=[]
        for name,spec in zip(NAMES,a["models"]):
            registry.append({"model":name,"model_id":spec["model_id"],"candidate":spec["candidate"],"threshold":spec["threshold"],
                "validation_mcc":a["metrics"][name]["mcc"],"rank":d["model_ranking_by_validation_mcc"].index(name)+1,
                "disposition":"FROZEN_COMPARATOR; NO_ADVANCEMENT_AUTHORIZATION", "model_path":spec["model_artifact"]["path"],"model_sha256":spec["model_artifact"]["sha256"]})
        pending=Path(tempfile.mkdtemp(prefix="stage_11d3i_pending_",dir=root/R))
        try:
            staged={key:pending/Path(path).name for key,path in OUTPUTS.items()}
            write_json(staged["disposition_policy"],d);write_json(staged["holdout_readiness"],readiness)
            write_csv(staged["criteria"],criteria);write_csv(staged["comparator_registry"],registry)
            staged["report"].write_text(report_text(d),encoding="utf-8")
            output_records={key:record(path,OUTPUTS[key]) for key,path in staged.items() if key!="audit"}
            print("RECHECKING FROZEN INPUTS BEFORE DISPOSITION FREEZE",flush=True)
            e.recheck()
            audit={**d,"audit_status":"FROZEN","input_evidence":e.records,"outputs":output_records,
                "checks":{"prior_output_hashes":"PASS","confusion_metric_consistency":"PASS","breakdown_totals":"PASS",
                    "saved_bootstrap_quantiles":"PASS","acceptance_reconstruction":"PASS","resource_and_log_agreement":"PASS","input_recheck":"PASS"},
                "model_objects_deserialized":0,"model_inference_calls":0,"model_fitting_calls":0,"threshold_selection_calls":0,
                "sample_target_arrays_read":False,"prediction_arrays_read":False,"holdout_payloads_read":0,
                "frozen_inputs_modified":False,"original_model_locks_modified":False,
                "completion_rule":"AUDIT_AND_ALL_MATCHING_OUTPUT_HASHES_REQUIRED"}
            write_json(staged["audit"],audit)
            for key in OUTPUTS:
                dest=e.path(OUTPUTS[key]);dest.parent.mkdir(parents=True,exist_ok=True)
                os.link(staged[key],dest)
                require(sha(dest)==sha(staged[key]),f"publication hash: {key}")
            shutil.rmtree(pending)
        except BaseException:
            print(f"STOP: partial evidence preserved at {pending}; completion requires final audit and matching output hashes",file=sys.stderr)
            raise
        print("\nSTAGE 11D-3I — VALIDATION RESULT DISPOSITION AND HOLDOUT READINESS FREEZE")
        for label,value in (("Status","PASS"),("Disposition status","FROZEN"),("Validation target","NOT_MET"),
            ("Failed criteria",", ".join(d["failed_criteria"])),("Best observed validation model",d["best_observed_validation_model"].upper()),
            ("Hybrid validation MCC",f"{d['hybrid_validation_mcc']:.8f}"),("Hybrid MCC drop",f"{d['hybrid_mcc_drop']:.8f}"),
            ("Allowed MCC drop",f"{d['allowed_mcc_drop']:.8f}"),("HOLDOUT readiness","BLOCKED"),("HOLDOUT execution","NOT AUTHORIZED"),
            ("Retraining / threshold changes","PROHIBITED"),("Model inference / label reopening","0 / NO"),
            ("Frozen inputs modified","NO"),("Project final freeze","NOT DECLARED")):
            print(f"{label:32}: {value}")
        for key,path in OUTPUTS.items():
            print(f"{key:32}: {root/path}");print(f"{key+' SHA':32}: {sha(root/path)}")
        print(f"Next gate                       : {NEXT_GATE}")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:run(Path.cwd().resolve())
    except KeyboardInterrupt:
        print("STOP: interrupted; preserve evidence",file=sys.stderr);return 130
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}",file=sys.stderr);return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
