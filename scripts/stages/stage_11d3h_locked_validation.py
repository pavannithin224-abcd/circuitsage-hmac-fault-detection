#!/usr/bin/env python3
"""11D-3H: authorize and evaluate four frozen models on committed VALIDATION.

Run from the project root. No training, threshold selection, graph fitting, or
HOLDOUT access. A completed audit prevents rerunning. Failed attempts remain
available as evidence; a performance target failure is frozen as NOT_MET.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
import types
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

G_SOURCE = "stage_11d3g_validation_features.py"
G_SHA = "d90672e17e737b18d0c936f3cae0528d3684a60fd70bcb4705df7c76394c0ec1"
R = "results/hmac_fault_campaign_11d3/"
D = R + "feature_matrix_11d3g/"
CONTRACT = "config/diagnostic_model/hmac_validation_inference_contract_11d3g.json"
MANIFEST = R + "hmac_validation_feature_matrix_manifest_11d3g.json"
AUDIT_G = R + "hmac_validation_feature_inference_contract_freeze_11d3g.json"
AUTH = "config/diagnostic_model/hmac_validation_inference_authorization_11d3h.json"
AUDIT = R + "hmac_locked_validation_four_model_freeze_11d3h.json"
EXPECTED = {
    G_SOURCE: G_SHA,
    D + "hmac_validation_features_11d3g.npz": "02cd075024ba583611d88eaeaf21a7125c24ca55b11fd4014b8cba23562cce56",
    D + "hmac_validation_targets_11d3g.npz": "e99f434f14f82f97147c0959f57ef62b61f072ff3387bf808ebf92a5b5dda70a",
    D + "hmac_validation_feature_schema_11d3g.json": "098d1df2aeba04085d8d70e733d69a96142ab7864bf3e942acb8ce535a4fc693",
    CONTRACT: "080b3e4efc940ae6d1abcdc663c6d08bff96dba2ffaae00ddd7a0741a3d1b359",
    MANIFEST: "484515028d3e944b04bb4692b7a3e2776df7e90d778c68f5f82afaf8054ffb73",
    AUDIT_G: "04b0ecada12085ba9c9f66cb19b5f01933dab885d8117bfa98dd6c3818f42347",
}
N, S, V = 2192544, 22839, 48
POSITIVE = 964973
NAMES = ("hybrid", "gnn", "deep", "baseline")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_module(root):
    p = root / G_SOURCE
    require(p.is_file() and sha(p) == G_SHA, "Stage 3G source missing or SHA mismatch")
    module = types.ModuleType("_frozen_3g")
    module.__file__ = str(p)
    exec(compile(p.read_bytes(), str(p), "exec"), module.__dict__)
    return module


def write_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def rec(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}


def confusion(y, pred):
    return np.bincount(2*y.astype(np.int64)+pred.astype(np.int64), minlength=4)


def count_metrics(c):
    tn, fp, fn, tp = [int(a) for a in c]
    def divide(a, b):
        return a/b if b else None
    recall, specificity = divide(tp, tp+fn), divide(tn, tn+fp)
    denominator = float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)
    return {"tn":tn, "fp":fp, "fn":fn, "tp":tp,
            "mcc": (float(tp)*tn-float(fp)*fn)/np.sqrt(denominator) if denominator else 0.0,
            "balanced_accuracy": (recall+specificity)/2 if recall is not None and specificity is not None else None,
            "precision":divide(tp,tp+fp), "recall":recall, "specificity":specificity,
            "f1_score":divide(2*tp,2*tp+fp+fn)}


def metrics(y, probability, threshold):
    if len(y) == 0:
        return {"status":"NOT_REPRESENTED", "samples":0, "reason":"no samples in this group"}
    out = count_metrics(confusion(y, probability >= threshold))
    out.update(status="REPRESENTED", samples=len(y), positive_samples=int(y.sum()),
               positive_prevalence=float(y.mean()), brier_score=float(np.mean((probability-y)**2)),
               pr_auc=float(average_precision_score(y, probability)) if y.any() else None,
               roc_auc=float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None)
    out["undefined_reasons"] = {k:"zero denominator or absent required target class" for k,v in out.items() if v is None}
    if not ((out["tp"]+out["fp"])*(out["tp"]+out["fn"])*(out["tn"]+out["fp"])*(out["tn"]+out["fn"])):
        out["mcc_zero_denominator_convention"] = "MCC=0 when a marginal count is zero, matching sklearn"
    return out


def preflight(root, g, v, e):
    print("FROZEN INPUT AND CONTRACT VERIFICATION", flush=True)
    for path, digest in EXPECTED.items():
        e.verify(path, digest)
        print(f"  {Path(path).name}: OK", flush=True)
    a, m, c = e.load(AUDIT_G), e.load(MANIFEST), e.load(CONTRACT)
    for doc in (a,m):
        v.fields(doc, {"stage":"11D-3G", "status":"PASS", "feature_matrix_status":"FROZEN",
                     "inference_contract_status":"FROZEN", "model_samples":N, "physical_sites":S,
                     "validation_vectors":V, "target_in_X":False, "target_file_separate":True,
                     "model_inference_performed":False, "holdout_records":0}, "3G freeze")
    e.link(a["manifest"], MANIFEST)
    for key in ("features","targets","schema","contract"):
        require(a[key] == m[key], f"audit/manifest binding: {key}")
        e.link(a[key], g.OUTPUTS[key])
    for path, evidence in a["input_evidence"].items():
        e.link(evidence, path)
    for path, digest in g.EXPECTED.items():
        require(e.records[path]["sha256"] == digest, f"3G input anchor: {path}")
    v.fields(c, {"stage":"11D-3G", "status":"PASS", "contract_status":"FROZEN",
                 "prediction_samples":N, "physical_sites":S, "validation_vectors":V,
                 "samples_per_site":96, "dataset_partition":"VALIDATION",
                 "runner_generation_authorized":True, "runner_preflight_authorized":True,
                 "validation_inference_execution_authorized":False}, "inference contract")
    require(c["models"] == a["models"] == m["models"], "four-model contract binding")
    require(c["acceptance_contract"] == a["acceptance_contract"] == m["acceptance_contract"], "acceptance binding")
    for key in ("features","targets","schema"):
        require(c[key] == a[key], f"contract artifact binding: {key}")
    require(c["sample_identity_sha256"] == a["sample_identity_sha256"], "sample identity commitment")
    for model, (model_id,spec) in zip(c["models"], g.MODELS.items()):
        require(model["model_id"] == model_id, "model order")
        for key,value in spec.items():
            require(model[key] == value, f"model specification: {model_id}/{key}")
        e.link(model["model_artifact"],spec["model"])
        e.link(model["selection_lock"],spec["lock"])
    require(len(c["models"]) == 4, "model count")
    require(c["execution"]["model_order"] == list(g.MODELS), "execution order")
    require(c["execution"]["inference_batch_size"] == 16384 and c["execution"]["threads"] == 1
            and c["execution"]["parallel_models"] == 1, "execution limits")
    require(c["site_confidence_intervals"]["replicates"] == 1000
            and c["site_confidence_intervals"]["random_seed"] == 20260909
            and c["site_confidence_intervals"]["confidence"] == .95, "bootstrap contract")
    actual = {k:importlib.metadata.version(k) for k in c["environment"]}
    require(actual == c["environment"], f"frozen package versions differ: expected {c['environment']}, actual {actual}")
    x = g.load_members(e.path(c["features"]["path"]), e.load(c["schema"]["path"])["feature_arrays"])
    schema = e.load(c["schema"]["path"])
    for key, info in schema["feature_arrays"].items():
        require(list(x[key].shape) == info["shape"] and x[key].dtype.str == info["dtype"], f"X schema: {key}")
    require(x["partition"].tolist() == ["VALIDATION"] and g.identity_sha(x) == c["sample_identity_sha256"], "X identity/partition")
    require(not(set(x) & g.FORBIDDEN), "forbidden X field")
    require(len(x["site_row"]) == N and np.all(x["site_row"] < S) and np.all(x["vector_row"] < V)
            and np.isin(x["stuck_value"],(0,1)).all(), "sample ranges")
    coverage = ((x["site_row"].astype(np.int64)*2+x["stuck_value"])*V+x["vector_row"])
    require(np.array_equal(np.sort(coverage),np.arange(N)), "exact site/fault/vector sample coverage")
    require(g.canary(x) == schema["materialization_canary"], "materialization canary")
    require(shutil.disk_usage(root).free >= 1024**3, "at least 1 GiB free space required")
    return c, x, schema


def authorize(root, c, e):
    source = rec(root, Path(__file__).resolve())
    e.link(source)
    document = {"stage":"11D-3H", "status":"PASS", "authorization_status":"FROZEN",
        "scope":"one locked four-model VALIDATION evaluation plus exact deterministic replay",
        "validation_inference_execution_authorized":True, "authorization_basis":"user requested Stage 11D-3H after frozen Stage 11D-3G",
        "evaluator_source":source, "inference_contract":e.records[CONTRACT],
        "feature_freeze":e.records[AUDIT_G], "models":c["models"], "sample_identity_sha256":c["sample_identity_sha256"],
        "acceptance_contract":c["acceptance_contract"], "retraining":"PROHIBITED", "threshold_changes":"PROHIBITED",
        "holdout_campaign":"NOT_AUTHORIZED", "prior_contract_modified":False}
    path = root/AUTH
    if path.exists():
        require(json.loads(path.read_text()) == document, "existing authorization differs; preserve it")
    else:
        write_json(path,document)
    e.verify(AUTH,sha(path))
    print("VALIDATION INFERENCE AUTHORIZATION: FROZEN (four locked models only)",flush=True)


def load_estimator(root, spec, x, e, g):
    e.link(spec["model_artifact"],spec["model"])
    obj = joblib.load(root/spec["model"])
    if spec["model_id"] == "DIR_SGC_11D1D":
        require(isinstance(obj,dict), "GNN bundle type")
        require(obj["candidate"]["candidate_id"] == spec["candidate"] and obj["candidate"]["propagation_hops"] == 3, "GNN bundle candidate")
        require(obj["target_in_input"] is False and obj["post_simulation_features"] == 0, "GNN leakage contract")
        cache = g.load_members(root/g.GRAPH_CACHE,("k3_mean","k3_scale","k3_columns","k3_features","node_site_index"))
        for left,right in (("graph_scaler_mean","k3_mean"),("graph_scaler_scale","k3_scale"),("graph_feature_columns","k3_columns")):
            require(np.array_equal(np.asarray(obj[left]),cache[right]), f"GNN bundle/cache mismatch: {left}")
        require(np.array_equal(cache["k3_columns"],x["graph_feature_columns"]), "GNN columns/X mismatch")
        require(np.array_equal(cache["k3_features"],x["graph_features"][cache["node_site_index"].astype(np.int64)-1]), "GNN cache/X mismatch")
        model = obj["estimator"]
    else:
        model = obj
    expected = MLPClassifier if "MLP" in spec["model_id"] else SGDClassifier
    require(isinstance(model,expected), "estimator class")
    require(model.n_features_in_ == spec["features"] and np.array_equal(model.classes_,[0,1]), "model width/classes")
    if isinstance(model,MLPClassifier):
        require(tuple(model.hidden_layer_sizes) == (128,64,32), "MLP hidden layers")
    return obj, model


def predict(model, x, width, g, replay=False):
    probabilities = np.empty(len(x["site_row"]),dtype="<f8")
    for start in range(0,len(probabilities),16384):
        end = min(start+16384,len(probabilities))
        batch = g.materialize(x,np.arange(start,end),width)
        probabilities[start:end] = model.predict_proba(batch)[:,1]
        if start//16384 % 20 == 0:
            print(f"  {'replay' if replay else 'inference'} {end}/{len(probabilities)}",flush=True)
    require(np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1)), "invalid probabilities")
    return probabilities


def bootstrap(y, x, probabilities, specs, replicates=1000, seed=20260909, sites=S):
    tables=[]
    for p,spec in zip(probabilities,specs):
        code=2*y.astype(np.int64)+(p>=spec["threshold"]).astype(np.int64)
        table=np.bincount(x["site_row"].astype(np.int64)*4+code,minlength=sites*4).reshape(sites,4)
        tables.append(table)
    rng=np.random.default_rng(seed)
    distribution=np.empty((replicates,7),dtype="<f8")
    for i in range(replicates):
        draw=rng.integers(0,sites,size=sites)
        scores=[count_metrics(t[draw].sum(axis=0))["mcc"] for t in tables]
        distribution[i]=scores+[scores[0]-scores[j] for j in (1,2,3)]
        if (i+1)%100 == 0:
            print(f"  paired-site bootstrap {i+1}/{replicates}",flush=True)
    names=list(NAMES)+["hybrid_minus_"+n for n in NAMES[1:]]
    intervals={name:np.quantile(distribution[:,i],[.025,.975]).tolist() for i,name in enumerate(names)}
    return {"mcc_intervals":intervals,"replicates":replicates,"seed":seed,
            "method":"paired physical-site percentile bootstrap", "scope":"same frozen circuit and 48 validation vectors; conditional intervals"},distribution


def breakdowns(y,x,probabilities,specs,schema):
    groups=[]
    for stuck in (0,1):
        groups.append(("fault_model",f"SA{stuck}",x["stuck_value"]==stuck))
    cols=x["site_feature_columns"].tolist()
    for kind in ("site_category","driver_cell_type"):
        for value in schema["categorical_vocabulary"][kind]:
            col=cols.index(f"{kind}::{value}")
            groups.append((kind,value,x["site_features"][x["site_row"],col]==1))
    for code,name in enumerate(("DEV_TRAIN","DEV_CALIBRATION","DEV_SITE_TEST")):
        groups.append(("development_site_partition",name,x["site_partition_code"][x["site_row"]]==code))
    for slot,value in enumerate(x["vector_ids"]):
        groups.append(("validation_vector_id",int(value),x["vector_row"]==slot))
    rows=[]
    for kind,value,mask in groups:
        rows.append({"group":kind,"value":value,"models":{
            name:metrics(y[mask],p[mask],spec["threshold"]) for name,p,spec in zip(NAMES,probabilities,specs)}})
    return rows


def assess(m,b,acceptance):
    h=m["hybrid"]; mini=acceptance["hybrid_minimum_validity"]
    absolute=acceptance["hybrid_absolute_performance"]
    retention=acceptance["hybrid_generalization_retention"]
    comp=acceptance["hybrid_comparator_confirmation"]
    checks={"positive_mcc":h["mcc"]>mini["mcc_gt"],
            "balanced_accuracy":h["balanced_accuracy"]>mini["balanced_accuracy_gt"],
            "pr_auc_over_prevalence":h["pr_auc"]>h["positive_prevalence"],
            "both_predicted_classes":(h["tp"]+h["fp"]>0 and h["tn"]+h["fn"]>0),
            "absolute_mcc":h["mcc"]>=absolute["mcc_minimum"],
            "absolute_balanced_accuracy":h["balanced_accuracy"]>=absolute["balanced_accuracy_minimum"],
            "absolute_f1":h["f1_score"]>=absolute["f1_minimum"],
            "absolute_recall":h["recall"]>=absolute["recall_minimum"],
            "mcc_retention":h["mcc"]>=retention["validation_mcc_minimum"],
            "hybrid_gain":h["mcc"]-m["gnn"]["mcc"]>=comp["hybrid_minus_gnn_mcc_minimum"],
            "positive_paired_ci":b["mcc_intervals"]["hybrid_minus_gnn"][0]>comp["paired_physical_site_mcc_delta_ci_lower_must_exceed"]}
    checks={k:bool(v) for k,v in checks.items()}
    return {"validation_target":"MET" if all(checks.values()) else "NOT_MET", "checks":checks,
            "failure_disposition":acceptance["failure_disposition"], "holdout_campaign":"NOT_AUTHORIZED"}


def run(root):
    require(root.name=="vlsi_fault_detection_v2" and Path(__file__).resolve().parent==root, "copy script to ~/vlsi_fault_detection_v2 and run there")
    g=verified_module(root); v=g.load_verifier(root); e=v.Evidence(root)
    print("STAGE 11D-3H — LOCKED FOUR-MODEL VALIDATION EVALUATION",flush=True)
    with v.campaign_lock(e), threadpool_limits(limits=1):
        require(not (root/AUDIT).exists(), "completed Stage 11D-3H audit already exists; preserve the frozen evaluation")
        c,x,schema=preflight(root,g,v,e)
        authorize(root,c,e)
        attempt=Path(tempfile.mkdtemp(prefix="stage_11d3h_attempt_",dir=root/R))
        print(f"Attempt evidence: {attempt}",flush=True)
        try:
            probabilities=[]; outputs={}
            for name,spec in zip(NAMES,c["models"]):
                print(f"\nLOCKED MODEL: {name} threshold={spec['threshold']}",flush=True)
                obj,model=load_estimator(root,spec,x,e,g)
                state=hashlib.sha256(pickle.dumps(obj,protocol=5)).hexdigest()
                p=predict(model,x,spec["features"],g)
                replay=predict(model,x,spec["features"],g,True)
                require(np.array_equal(p,replay), f"{name} prediction replay mismatch")
                require(hashlib.sha256(pickle.dumps(obj,protocol=5)).hexdigest()==state, f"{name} model state changed")
                e.link(spec["model_artifact"],spec["model"])
                dest=attempt/f"{name}_validation_predictions.npz"
                g.write_and_replay(dest,{**{k:x[k] for k in g.IDENTITY},"probability":p,
                    "predicted_detected":(p>=spec["threshold"]).astype("u1")},v)
                outputs[name+"_predictions"]=rec(root,dest)
                probabilities.append(p)
                del obj,model,replay
            print("\nALL PREDICTIONS FROZEN; OPENING SEPARATE VALIDATION TARGETS",flush=True)
            targets=g.load_members(root/c["targets"]["path"],schema["target_arrays"])
            require(g.identity_sha(targets)==c["sample_identity_sha256"], "target identity commitment")
            for key in g.IDENTITY:
                require(np.array_equal(targets[key],x[key]),f"target alignment: {key}")
            y=targets["target_detected"]
            require(y.shape==(N,) and np.isin(y,(0,1)).all() and int(y.sum())==POSITIVE, "target counts/classes")
            scores={name:metrics(y,p,spec["threshold"]) for name,p,spec in zip(NAMES,probabilities,c["models"])}
            print("COMPUTING REQUIRED BREAKDOWNS",flush=True)
            details=breakdowns(y,x,probabilities,c["models"],schema)
            b,distribution=bootstrap(y,x,probabilities,c["models"])
            assessment=assess(scores,b,c["acceptance_contract"])
            for key,data in (("metrics",scores),("breakdowns",details),("paired_site_bootstrap",b),("acceptance",assessment)):
                dest=attempt/f"{key}.json";write_json(dest,data);outputs[key]=rec(root,dest)
            dest=attempt/"paired_site_bootstrap_distribution.npz"
            g.write_and_replay(dest,{"mcc_distribution":distribution,"column_names":np.asarray(list(NAMES)+["hybrid_minus_"+n for n in NAMES[1:]])},v)
            outputs["bootstrap_distribution"]=rec(root,dest)
            print("RECHECKING FROZEN INPUTS AND OUTPUTS",flush=True)
            e.recheck()
            for evidence in outputs.values():
                require(sha(root/evidence["path"])==evidence["sha256"], "output modified")
            audit={"stage":"11D-3H", "status":"PASS", "evaluation_status":"FROZEN", "comparison_status":"FROZEN",
                "created_at_utc":datetime.now(timezone.utc).isoformat(),"source":rec(root,Path(__file__).resolve()),
                "authorization":e.records[AUTH],"input_evidence":e.records,"outputs":outputs,"models":c["models"],
                "samples":N,"sites":S,"validation_vectors":V,"positive_samples":int(y.sum()),
                "negative_samples":N-int(y.sum()),"metrics":scores,"paired_site_intervals":b,**assessment,
                "exact_prediction_replay":"PASS","prediction_npz_replay":"PASS","model_state_unchanged":True,
                "model_fitting_calls":0,"threshold_selection_calls":0,"scaler_fitting_calls":0,"graph_propagation_calls":0,
                "holdout_samples":0,"development_labels_reopened":False,"frozen_inputs_modified":False,
                "metric_implementation":"sklearn average_precision_score / roc_auc_score; explicit binary confusion metrics; Brier mean squared error",
                "scope":c["generalization_scope"],"statistical_limitations":"Conditional site-bootstrap intervals on one circuit and 48 shared validation vectors. Development site tests were used previously across model families; this is a new-vector validation, not a new-chip test.",
                "next_gate":"VALIDATION RESULT DISPOSITION AND HOLDOUT READINESS; separate authorization required"}
            write_json(root/AUDIT,audit)
        except BaseException:
            print(f"STOP: attempt artifacts preserved at {attempt}; no completion PASS",file=sys.stderr)
            raise
        print("\nSTAGE 11D-3H — LOCKED FOUR-MODEL VALIDATION FREEZE")
        print("Evaluation status              : PASS / FROZEN")
        for name in NAMES:
            print(f"{name:30}: MCC={scores[name]['mcc']:.8f}")
        print(f"Hybrid-GNN MCC delta           : {scores['hybrid']['mcc']-scores['gnn']['mcc']:.8f}")
        print(f"Delta 95% paired-site CI        : {b['mcc_intervals']['hybrid_minus_gnn']}")
        print(f"Validation target              : {assessment['validation_target']}")
        print("Deterministic prediction replay: PASS")
        print("Retraining / threshold changes : 0 / 0")
        print("HOLDOUT campaign               : NOT AUTHORIZED")
        print(f"Audit                          : {root/AUDIT}")
        print(f"Audit SHA                      : {sha(root/AUDIT)}")
        print(f"Next gate                      : {audit['next_gate']}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run(Path.cwd().resolve())
    except KeyboardInterrupt:
        print("STOP: interrupted; preserve attempt evidence",file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
