#!/usr/bin/env python3
"""Build VALIDATION inputs and freeze inference rules, without model inference.

Run in ~/vlsi_fault_detection_v2 with the unchanged Stage 11D-3E verifier.
Requires the existing NumPy installation. Copies frozen site and graph features;
never fits preprocessing, loads model objects, trains, or selects thresholds.
Labels are written to a separate NPZ. Existing outputs are never overwritten.
Only committed VALIDATION stimuli are encoded; HOLDOUT stimuli are not emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import types
import zipfile

import numpy as np

VERSION = "HMAC-VALIDATION-FEATURE-CONTRACT-v1"
FORMAT = "HMAC-FACTORIZED-VALIDATION-X-v1"
R = "results/hmac_fault_campaign_11d3/"
D = R + "feature_matrix_11d3g/"
CF = "config/diagnostic_model/"
F5 = "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/"
R5 = "results/hmac_fault_campaign_11c5/"
G1 = "results/hmac_fault_campaign_11d1/"
G2 = "results/hmac_fault_campaign_11d2/"
VERIFIER = "stage_11d3e_validation_dataset_integrity.py"
VERIFIER_SHA = "b2bdcbfc98510aaff4f1bfea55706cf7767a84607f59dbcd2c9267cba02f15dc"
SOURCE_MANIFEST = R + "hmac_canonical_validation_dataset_manifest_11d3f.json"
SOURCE_AUDIT = R + "hmac_canonical_validation_dataset_schema_freeze_11d3f.json"
CANONICAL = R + "canonical_dataset_11d3f/hmac_fault_campaign_canonical_validation_11d3f.csv.gz"
CANONICAL_SCHEMA = R + "canonical_dataset_11d3f/hmac_canonical_validation_dataset_schema_11d3f.json"
FEATURE_SCHEMA = F5 + "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
GRAPH = G1 + "graph_dataset_11d1a/hmac_golden_netlist_graph_11d1a.npz"
GRAPH_CACHE = G1 + "gnn_training_11d1c/hmac_directed_sgc_graph_features_11d1c.npz"
FINAL_LOCK = CF + "hmac_final_diagnostic_model_lock_11d2d.json"
REGISTRY = G2 + "hmac_final_diagnostic_comparator_registry_11d2d.json"
ENVIRONMENT = G2 + "hmac_hybrid_environment_11d2a.json"
MATRICES = {name: F5 + f"hmac_{name.lower()}_feature_matrix_11c5e.npz"
            for name in ("DEV_TRAIN", "DEV_CALIBRATION", "DEV_SITE_TEST")}
SITE_COUNTS = {"DEV_TRAIN": 15987, "DEV_CALIBRATION": 3426, "DEV_SITE_TEST": 3426}
MODELS = {
    "HYBRID_FUSION_MLP_11D2C": {
        "candidate": "HYBRID_SGC3_MLP_128_64_32_A1E4_LR1E3", "threshold": 0.4965, "features": 646,
        "model": G2 + "hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib",
        "lock": G2 + "hybrid_training_11d2b/hmac_hybrid_selection_lock_11d2b.json",
        "object_type": "sklearn.neural_network.MLPClassifier"},
    "DIR_SGC_11D1D": {
        "candidate": "DIR_SGC_K3_L2_A1E5", "threshold": 0.86327695216798483, "features": 646,
        "model": G1 + "gnn_training_11d1c/hmac_selected_gnn_model_11d1c.joblib",
        "lock": G1 + "gnn_training_11d1c/hmac_gnn_selection_lock_11d1c.json",
        "object_type": "dictionary; estimator is sklearn.linear_model.SGDClassifier"},
    "DEEP_MLP_11C5K": {
        "candidate": "DEEP_MLP_128_64_32_A1E5_LR3E4", "threshold": 0.442, "features": 527,
        "model": R5 + "deep_training_11c5j/hmac_selected_deep_diagnostic_model_11c5j.joblib",
        "lock": R5 + "deep_training_11c5j/hmac_deep_diagnostic_selection_lock_11c5j.json",
        "object_type": "sklearn.neural_network.MLPClassifier"},
    "CONVENTIONAL_LOGREG_11C5H": {
        "candidate": "SGD_LOGREG_L2_A1E6_BAL", "threshold": 0.98536854982376099, "features": 527,
        "model": R5 + "baseline_training_11c5g/hmac_selected_conventional_baseline_11c5g.joblib",
        "lock": R5 + "baseline_training_11c5g/hmac_conventional_baseline_selection_lock_11c5g.json",
        "object_type": "sklearn.linear_model.SGDClassifier"},
}
EXPECTED = {
    VERIFIER: VERIFIER_SHA,
    "stage_11d3f_validation_consolidator.py": "361add2f7c9861d59047d72f45aa8bdc1cd0fc863a2d72684b0170dd8afd2b64",
    SOURCE_MANIFEST: "afb00228a31441ab8d534a234ba3c2f355a41c0de0c0566eb9781f2f36c6db64",
    SOURCE_AUDIT: "a0fd16b8cafaac0f1014df56fe20d1d843aebeef8250c56e84f3861a32ae2410",
    CANONICAL: "c14d6788df015def724524c6e45a706f0f691588ff6085ea964cc01afb5cc555",
    CANONICAL_SCHEMA: "9c2fe65ea4e980ae7676c9a30457180b22783301710c81f2eea9fdef1ae57c22",
    R + "stage_11d3f_consolidation_20260909_100634.log": "d68b69eede0df4d6ce59b60593925fa7bb514ffbc94a68bd8725646e305ffe7a",
    R + "stage_11d3f_resources_20260909_100634.log": "2b23ffa748c187b42a092910bfdf0f1965660486931d432b3207f2f0478888d7",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    MATRICES["DEV_TRAIN"]: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    MATRICES["DEV_CALIBRATION"]: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    MATRICES["DEV_SITE_TEST"]: "965f90be0dbdd33a47d85a9f0b0e41b95160410daee8255c5aa04fe92f1767b2",
    GRAPH: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    FINAL_LOCK: "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    REGISTRY: "39e15600efdf5e65d2155f7ddfcdf4cfc9754fe18dfacdb64bd2cce8cf64e7df",
    ENVIRONMENT: "1849274a8006e77cced8695be784e70102abceec940fee036163f0a62aa65a57",
    MODELS["HYBRID_FUSION_MLP_11D2C"]["model"]: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    MODELS["HYBRID_FUSION_MLP_11D2C"]["lock"]: "005cecbd94456d3d1aa6805b2dc898e895cc178564673b6b0aa269fcd64260c6",
    MODELS["DIR_SGC_11D1D"]["model"]: "c345ab00a45bceb7b7c375d9c5483daa067a01bb8149b30c2325b056a3724e91",
    MODELS["DIR_SGC_11D1D"]["lock"]: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
    MODELS["DEEP_MLP_11C5K"]["model"]: "6e25e901f7703e22c069ab21b15e1e526db6bfab79f4d205d9e36fea4cafb7fa",
    MODELS["DEEP_MLP_11C5K"]["lock"]: "12319c2d3b067ca04862209512aad9e42b35be97068557d03bc1f1906b4f024d",
    MODELS["CONVENTIONAL_LOGREG_11C5H"]["model"]: "8220ae569958e3d072b1ac40ac3c2958c7428419fb0ebb4acab8f324eaea5833",
    MODELS["CONVENTIONAL_LOGREG_11C5H"]["lock"]: "1532a30fb96a5f142ba05a778643bdb9aa93c058eba568e3df92c09f47e523fc",
}
OUTPUTS = {
    "features": D + "hmac_validation_features_11d3g.npz",
    "targets": D + "hmac_validation_targets_11d3g.npz",
    "schema": D + "hmac_validation_feature_schema_11d3g.json",
    "contract": CF + "hmac_validation_inference_contract_11d3g.json",
    "manifest": R + "hmac_validation_feature_matrix_manifest_11d3g.json",
    "audit": R + "hmac_validation_feature_inference_contract_freeze_11d3g.json",
}
N_SITES, N_VECTORS, N_ROWS, POSITIVES = 22839, 48, 2192544, 964973
SOURCE_COMMITMENT = "68fc090166650192a05db7efd79e65304f8621b304e96bec2302ebfe31aa03f2"
NEXT_GATE = "VALIDATION INFERENCE AUTHORIZATION AND LOCKED FOUR-MODEL EVALUATION"
FORBIDDEN = {"target", "target_detected", "detected", "activity", "cycles", "baseline_cycles", "latency_delta",
             "timed_out", "unknown", "expected_digest", "actual_digest", "digest_hamming_distance",
             "activated_vectors", "detected_vectors", "timeout_vectors", "final_classification"}
IDENTITY = ("record_id", "site_row", "vector_row", "stuck_value")
BLOCKS = {"validation_model_inference": "NOT_AUTHORIZED", "holdout_campaign": "NOT_AUTHORIZED",
          "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED",
          "model_inference_performed": False, "training_performed": False, "model_deserialization_performed": False,
          "frozen_input_artifacts_modified": False, "raw_campaign_dataset_modified": False,
          "frozen_rtl_modified": False, "golden_netlist_modified": False, "holdout_payloads_emitted": 0}


class FeatureError(Exception):
    pass


def require(ok, message):
    if not ok:
        raise FeatureError(message)


def load_verifier(root):
    path = root / VERIFIER
    require(path.is_file() and not path.is_symlink(), f"missing regular prerequisite: {VERIFIER}")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    require(actual == VERIFIER_SHA, f"prerequisite SHA mismatch: expected {VERIFIER_SHA}, got {actual}")
    module = types.ModuleType("_verified_stage_11d3e")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def load_members(path, names):
    # Select explicit non-object members; never load previous targets or model objects.
    with np.load(path, allow_pickle=False) as archive:
        require(set(names) <= set(archive.files), f"NPZ member(s) missing: {path.name}: {set(names) - set(archive.files)}")
        result = {key: archive[key] for key in names}
    require(all(not a.dtype.hasobject for a in result.values()), "object array prohibited")
    return result


def verify_sources(e, v):
    print("FROZEN DATASET, FEATURES, AND FOUR-MODEL INPUT VERIFICATION", flush=True)
    for path, sha in EXPECTED.items():
        e.verify(path, sha)
        print(f"  {Path(path).name}: OK", flush=True)
    source, audit = e.load(SOURCE_MANIFEST), e.load(SOURCE_AUDIT)
    common = {"stage": "11D-3F", "status": "PASS", "canonical_dataset_status": "FROZEN", "schema_status": "FROZEN",
              "source_dataset_commitment": SOURCE_COMMITMENT, "batch_count": 45, "validation_vectors": N_VECTORS,
              "legal_sites": N_SITES, "persistent_fault_instances": N_SITES * 2, "vector_partition": "VALIDATION", **BLOCKS}
    v.fields(source, common, "3F manifest")
    v.fields(audit, common, "3F audit")
    e.link(audit["manifest"], SOURCE_MANIFEST)
    for name in ("canonical_dataset", "fault_instance_summary", "schema"):
        e.link(audit[name])
        require(audit[name] == source[name], f"3F artifact binding: {name}")
    require(source["canonical_dataset"] == e.records[CANONICAL] and source["schema"] == e.records[CANONICAL_SCHEMA], "3F canonical/schema identities")
    require(source["summary"] == audit["summary"], "3F summary binding")
    v.fields(source["summary"], {"total_records": 2194704, "enabled_records": N_ROWS, "baseline_records": 2160,
                                "detected_records": POSITIVES, "missing_samples": 0, "duplicate_samples": 0,
                                "train_records": 0, "holdout_records": 0, "unknown_records": 0}, "3F counts")
    linked = audit["input_evidence"]
    require(isinstance(linked, dict) and set(v.EXPECTED) <= linked.keys(), "3F prior evidence chain")
    for i, (path, rec) in enumerate(linked.items(), 1):
        e.link(rec, path)
        if i % 100 == 0:
            print(f"  Prior frozen artifacts: {i}/{len(linked)}", flush=True)
    for path, sha in v.EXPECTED.items():
        require(e.records[path]["sha256"] == sha, f"3E anchor mismatch: {path}")
    v.verify_resource_log(e.path(R + "stage_11d3f_resources_20260909_100634.log"))
    vectors, commitment, arch, contract = v.validation_vectors(e, e.load(v.AUTH), e.load(v.AUDIT_C), e.load(v.CHECKPOINT))
    require(source["validation_vector_ids"] == [x["vector_id"] for x in vectors], "validation ID order")
    require(source["ordered_validation_commitment"] == commitment, "validation commitment")
    require(source["acceptance_contract"] == arch["acceptance_contract"] == contract["acceptance_contract"], "frozen acceptance rules differ")
    registry = e.load(REGISTRY)
    v.fields(registry, {"status": "PASS", "registry_status": "FROZEN", "winner": "HYBRID_FUSION_MLP_11D2C"}, "comparator registry")
    rows = {row["comparator_id"]: row for row in registry["comparators"]}
    require(len(rows) == len(registry["comparators"]) == 4 and set(rows) == set(MODELS), "four frozen comparators")
    model_records = []
    for model_id, spec in MODELS.items():
        lock = e.load(spec["lock"])
        v.fields(lock, {"status": "PASS", "lock_status": "FROZEN", "selected_candidate_id": spec["candidate"],
                        "selected_threshold": spec["threshold"]}, f"{model_id} selection lock")
        require(rows[model_id]["candidate"] == spec["candidate"] and float(rows[model_id]["threshold"]) == spec["threshold"], "registry/selection identity")
        e.link(lock["selected_model"], spec["model"])
        if spec["features"] == 646:
            e.link(lock["propagation_cache"], GRAPH_CACHE)
        if model_id == "DIR_SGC_11D1D":
            e.link(lock["selected_weights"])
        model_records.append({"model_id": model_id, **spec, "model_artifact": e.records[spec["model"]],
                              "selection_lock": e.records[spec["lock"]]})
    final = e.load(FINAL_LOCK)
    v.fields(final, {"status": "PASS", "lock_status": "FROZEN", "model_id": "HYBRID_FUSION_MLP_11D2C",
                     "candidate_id": MODELS["HYBRID_FUSION_MLP_11D2C"]["candidate"], "threshold": 0.4965,
                     "model_retraining_allowed": False, "threshold_reselection_allowed": False,
                     "feature_or_graph_refitting_allowed": False, "validation_use_for_tuning_allowed": False}, "final model lock")
    e.link(final["model_artifact"], MODELS["HYBRID_FUSION_MLP_11D2C"]["model"])
    e.link(final["training_selection_lock"], MODELS["HYBRID_FUSION_MLP_11D2C"]["lock"])
    environment = e.load(ENVIRONMENT)
    require(environment["packages"]["numpy"] == np.__version__, "NumPy differs from the frozen hybrid environment; preserve the environment")
    return source, vectors, model_records, environment


def frozen_lookups(e, v):
    schema = e.load(FEATURE_SCHEMA)
    v.fields(schema, {"status": "PASS", "model_feature_count": 527, "site_feature_count": 14, "stimulus_feature_count": 512}, "sample schema")
    v.fields(schema["leakage_contract"], {"target_in_X": False, "post_simulation_fields_in_X": [], "scaler_fit_partition": "DEV_TRAIN"}, "frozen leakage contract")
    columns, site_columns = schema["model_feature_columns"], schema["site_feature_columns"]
    vocabulary = schema["categorical_vocabulary"]
    expected_site = ["cell_fanout_z", "is_primary_output_stem", "is_sequential_stem"]
    expected_site += [f"driver_cell_type::{x}" for x in vocabulary["driver_cell_type"]]
    expected_site += [f"site_category::{x}" for x in vocabulary["site_category"]]
    expected_columns = ["stuck_value"] + expected_site + [f"{kind}_bit_{b}" for kind in ("key", "message") for b in range(255, -1, -1)]
    require(site_columns == expected_site and columns == expected_columns and not (set(columns) & FORBIDDEN), "527-column feature order")
    site_features = np.empty((N_SITES, 14), dtype="<f4")
    partition_codes, covered = np.empty(N_SITES, dtype="u1"), np.zeros(N_SITES, dtype=bool)
    members = ("partition", "site_ids", "site_indices", "site_features", "site_feature_columns", "feature_columns")
    for code, (partition, path) in enumerate(MATRICES.items()):
        data = load_members(e.path(path), members)
        require(data["partition"].tolist() == [partition], "frozen development partition identity")
        require(data["feature_columns"].tolist() == columns and data["site_feature_columns"].tolist() == site_columns, "development feature-order drift")
        count = SITE_COUNTS[partition]
        require(data["site_features"].dtype == np.dtype("<f4") and data["site_features"].shape == (count, 14), "site matrix dtype/shape")
        require(data["site_indices"].shape == data["site_ids"].shape == (count,), "site identity shapes")
        indices = data["site_indices"].astype(np.int64) - 1
        require(np.all((indices >= 0) & (indices < N_SITES)) and len(np.unique(indices)) == count, "invalid/duplicate site index")
        require(not covered[indices].any(), "physical site overlaps development partitions")
        require(data["site_ids"].tolist() == [f"HMAC-STEM-{i+1:06d}" for i in indices], "site index/ID binding")
        site_features[indices] = data["site_features"]
        partition_codes[indices] = code
        require(np.array_equal(site_features[indices], data["site_features"]), "frozen site feature copy differs")
        covered[indices] = True
    require(covered.all() and np.isfinite(site_features).all(), "site feature coverage/finite values")
    require(np.isin(site_features[:, 1:], (0., 1.)).all(), "nonbinary static categorical features")
    require(np.all(site_features[:, 3:12].sum(axis=1) == 1) and np.all(site_features[:, 12:].sum(axis=1) == 1), "one-hot site fields")
    graph = load_members(e.path(GRAPH), ("node_site_index", "node_partition_code"))
    cache = load_members(e.path(GRAPH_CACHE), ("node_site_index", "node_partition_code", "k3_features", "k3_columns", "k3_mean", "k3_scale"))
    require(cache["k3_features"].shape == (N_SITES, 119) and cache["k3_features"].dtype == np.dtype("<f4"), "graph feature shape/dtype")
    graph_indices = cache["node_site_index"].astype(np.int64) - 1
    require(np.array_equal(np.sort(graph_indices), np.arange(N_SITES)), "graph site-index coverage")
    require(np.array_equal(graph["node_site_index"], cache["node_site_index"])
            and np.array_equal(graph["node_partition_code"], cache["node_partition_code"]), "graph/cache node alignment")
    require(np.array_equal(partition_codes[graph_indices], cache["node_partition_code"]), "graph/site partition alignment")
    scaled = np.empty_like(cache["k3_features"])
    scaled[graph_indices] = cache["k3_features"]
    require(np.isfinite(scaled).all() and cache["k3_columns"].shape == (119,), "graph values/columns")
    require(cache["k3_mean"].shape == cache["k3_scale"].shape == (119,)
            and np.isfinite(cache["k3_mean"]).all() and np.isfinite(cache["k3_scale"]).all()
            and (cache["k3_scale"] > 0).all(), "frozen graph scaler")
    weight_record = e.load(MODELS["DIR_SGC_11D1D"]["lock"])["selected_weights"]
    weights = load_members(e.path(weight_record["path"]), ("candidate_id", "graph_scaler_mean", "graph_scaler_scale", "graph_feature_columns"))
    require(weights["candidate_id"].tolist() == [MODELS["DIR_SGC_11D1D"]["candidate"]], "GNN weight export candidate")
    for left, right in (("graph_scaler_mean", "k3_mean"), ("graph_scaler_scale", "k3_scale"), ("graph_feature_columns", "k3_columns")):
        require(np.array_equal(weights[left], cache[right]), f"GNN model/cache metadata differs: {left}")
    graph_columns = cache["k3_columns"].tolist()
    require(len(set(columns + graph_columns)) == 646, "unique combined feature names")
    require(not any(part in FORBIDDEN for name in graph_columns for part in name.split("::")), "graph feature leakage name")
    print("  Frozen static features: 22839/22839 exact; graph/site alignment: PASS", flush=True)
    return {"site_features": site_features, "site_ids": np.asarray([f"HMAC-STEM-{i:06d}" for i in range(1, N_SITES+1)]),
            "site_indices": np.arange(1, N_SITES+1, dtype="<u2"), "site_partition_code": partition_codes,
            "site_feature_columns": np.asarray(site_columns), "feature_columns": np.asarray(columns),
            "graph_features": scaled, "graph_feature_columns": cache["k3_columns"],
            "combined_feature_columns": np.asarray(columns + graph_columns)}, schema


def vector_bits(e, v, vectors):
    pool = e.load(v.POOL)
    selected = {row["vector_id"]: row for row in pool["vectors"] if row["split"] == "VALIDATION"}
    require(set(selected) == {row["vector_id"] for row in vectors} and len(selected) == N_VECTORS, "committed validation stimulus set")
    result = np.empty((N_VECTORS, 512), dtype="u1")
    for slot, record in enumerate(vectors):
        row = selected[record["vector_id"]]
        raw = bytes.fromhex(v.hex256(row["key"], "validation key") + v.hex256(row["message"], "validation message"))
        result[slot] = np.unpackbits(np.frombuffer(raw, dtype="u1"), bitorder="big")
        require(np.packbits(result[slot], bitorder="big").tobytes() == raw, "stimulus endian roundtrip")
    return result


def parse_canonical(e, v, source, vectors):
    arrays = {"record_id": np.empty(N_ROWS, dtype="<u4"), "site_row": np.empty(N_ROWS, dtype="<u2"),
              "vector_row": np.empty(N_ROWS, dtype="u1"), "stuck_value": np.empty(N_ROWS, dtype="u1")}
    target = np.empty(N_ROWS, dtype="u1")
    seen, baselines = np.zeros(N_ROWS, dtype=bool), np.zeros((45, N_VECTORS), dtype=bool)
    totals, row_sha = Counter(), hashlib.sha256()
    count = record_id = 0
    expected_header = source["canonical_header"]
    required = "record_id,vector_split,batch_id,run_type,site_selection,selector,site_id,stuck_value,vector_slot,vector_id,fault_enable,cycles,baseline_cycles,latency_delta,timed_out,activity,detected,unknown,digest_hamming_distance,expected_digest,actual_digest".split(",")
    require(expected_header == required, "frozen canonical header")
    print("\nSTREAMING CANONICAL LABELS AND SAMPLE IDENTITIES", flush=True)
    with gzip.open(e.path(CANONICAL), "rb") as compressed:
        def lines():
            for line in compressed:
                row_sha.update(line)
                yield line.decode("utf-8")
        reader = csv.reader(lines(), strict=True)
        require(next(reader, None) == required, "canonical CSV header")
        for row in reader:
            require(len(row) == 21 and row[0] == str(record_id) and row[1] == "VALIDATION", "canonical row order or partition")
            bid, selection, selector, stuck, slot, vid, enable, cycles, baseline, delta, timeout, active, detected, unknown, distance = (
                int(row[i]) for i in (2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
            require(0 <= bid < 45 and 0 <= slot < N_VECTORS and vid == vectors[slot]["vector_id"], "batch/vector alignment")
            require(all(x in (0, 1) for x in (enable, timeout, active, detected, unknown)) and unknown == 0, "binary flags/unknown")
            require(baseline == 343 and 0 <= cycles <= 2000 and delta == cycles - baseline and (not timeout or cycles == 2000), "latency fields")
            require(v.HEX256.fullmatch(row[19]) and v.HEX256.fullmatch(row[20]), "digest encoding")
            require(row[19] == vectors[slot]["expected_digest"], "validation reference digest")
            require(distance == (int(row[19],16)^int(row[20],16)).bit_count(), "digest-distance mismatch")
            if row[3] == "BASELINE":
                require((selection, selector, stuck, row[6]) == (-1, -1, -1, "BASELINE"), "baseline identity")
                require(enable == timeout == active == detected == delta == distance == 0 and not baselines[bid,slot], "baseline failure/duplicate")
                baselines[bid,slot] = True
                totals["baseline_records"] += 1
            else:
                size = 311 if bid == 44 else 512
                require(row[3] == "ENABLED" and enable == 1 and stuck in (0,1) and 0 <= selector == selection < size, "enabled fault identity")
                index = bid * 512 + selector
                require(row[6] == f"HMAC-STEM-{index+1:06d}", "canonical site mapping")
                sample = (index * 2 + stuck) * N_VECTORS + slot
                require(count < N_ROWS and not seen[sample], "duplicate/extra enabled sample")
                require(detected == int(bool(timeout or delta or distance)) and (not detected or active), "detection-label semantics")
                seen[sample] = True
                for key, value in (("record_id",record_id),("site_row",index),("vector_row",slot),("stuck_value",stuck)):
                    arrays[key][count] = value
                target[count] = detected
                count += 1
                totals["enabled_records"] += 1
                totals["detected_records"] += detected
                totals["activated_records"] += active
                totals["timeout_records"] += timeout
            record_id += 1
            if record_id % 250000 == 0:
                print(f"  Canonical rows: {record_id}/2194704", flush=True)
    require(count == N_ROWS and seen.all() and baselines.all(), "complete site x SA x VALIDATION coverage")
    totals["total_records"] = record_id
    for key, value in totals.items():
        require(source["summary"][key] == value, f"3F aggregate differs: {key}")
    require(int(target.sum(dtype=np.uint64)) == POSITIVES, "positive-label count")
    require(row_sha.hexdigest() == source["checks"]["uncompressed_csv_sha256"], "canonical uncompressed commitment")
    return arrays, target, dict(totals)


def materialize(x, indices, width=646):
    """Prediction X only. Targets are not accepted or consulted by this function."""
    require(width in (527,646) and len(indices) <= 16384, "materialization width/batch limit")
    s, w = x["site_row"][indices], x["vector_row"][indices]
    result = np.empty((len(indices), width), dtype="<f4")
    result[:,0] = x["stuck_value"][indices]
    result[:,1:15] = x["site_features"][s]
    result[:,15:527] = x["vector_features"][w]
    if width == 646:
        result[:,527:] = x["graph_features"][s]
    require(np.isfinite(result).all(), "nonfinite materialized inputs")
    return result


def identity_sha(arrays):
    h = hashlib.sha256()
    for name in IDENTITY:
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}\n".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def canary(x):
    require(not (set(x) & FORBIDDEN), "target or post-simulation field in X archive")
    require(x["vector_features"].shape == (N_VECTORS,512) and np.isin(x["vector_features"], (0,1)).all(), "stimulus feature dimensions")
    mask = (x["vector_row"] == 0) | (x["vector_row"] == N_VECTORS-1) | (x["site_row"] == 0)
    positions = np.flatnonzero(mask)
    digest = hashlib.sha256()
    for start in range(0,len(positions),16384):
        idx = positions[start:start+16384]
        a = materialize(x,idx,646)
        b = materialize(x,idx,527)
        require(np.array_equal(a[:,:527],b), "sample/hybrid feature prefix differs")
        # Independent column stacking confirms the frozen helper's concat order.
        reference = np.column_stack((x["stuck_value"][idx], x["site_features"][x["site_row"][idx]],
                                     x["vector_features"][x["vector_row"][idx]], x["graph_features"][x["site_row"][idx]])).astype("<f4")
        require(np.array_equal(a,reference), "feature fusion canary differs")
        require(np.array_equal(a,materialize(x,idx,646)), "materialization replay differs")
        digest.update(a.tobytes())
    return {"samples": len(positions), "physical_sites": N_SITES, "validation_vectors": N_VECTORS,
            "status": "PASS", "materialized_float32_sha256": digest.hexdigest()}


def deterministic_npz(path, arrays):
    with path.open("xb") as raw:
        with zipfile.ZipFile(raw,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(arrays):
                require(not arrays[name].dtype.hasobject, "object NPZ forbidden")
                payload = io.BytesIO()
                np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(name+".npy", date_time=(1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info,payload.getvalue(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
        raw.flush()
        os.fsync(raw.fileno())


def write_and_replay(path, arrays, v):
    deterministic_npz(path,arrays)
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files) == set(arrays), "NPZ array inventory")
        for key, a in arrays.items():
            b = archive[key]
            require(b.dtype == a.dtype and np.array_equal(b,a), f"NPZ readback differs: {key}")
    replay = path.with_suffix(".replay.npz")
    deterministic_npz(replay,arrays)
    require(v.sha256(path) == v.sha256(replay), "NPZ byte replay differs")
    replay.unlink()


def artifact(path, relative, v):
    return {"path": relative, "sha256": v.sha256(path), "bytes": path.stat().st_size}


def publish(staged, destination):
    require(not destination.exists() and not destination.is_symlink(), f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with staged.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(staged,destination)
    fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def schema_document(x, y, frozen, canary_result):
    return {"stage":"11D-3G","status":"PASS","schema_status":"FROZEN","format_version":FORMAT,
            "task":"PRE-SIMULATION BINARY DETECTABILITY","fault_scope":"PERSISTENT_NET_STEM_SA0_SA1_ONLY",
            "sample_order":"source canonical order after removing BASELINE rows",
            "sample_identity_arrays":list(IDENTITY), "record_id_scope":"Stage 11D-3F canonical row, including gaps where baselines were removed",
            "feature_arrays":{k:{"shape":list(a.shape),"dtype":a.dtype.str} for k,a in x.items()},
            "target_arrays":{k:{"shape":list(a.shape),"dtype":a.dtype.str} for k,a in y.items()},
            "target_file_separate":True,"model_feature_columns":x["feature_columns"].tolist(),
            "graph_feature_columns":x["graph_feature_columns"].tolist(),"combined_feature_columns":x["combined_feature_columns"].tolist(),
            "materialization":{"dtype":"float32","maximum_rows_per_batch":16384,
                               "sample":"[stuck_value, site_features[site_row], vector_features[vector_row]]",
                               "graph_and_hybrid":"[sample_features, graph_features[site_row]]",
                               "already_standardized_graph":True,"apply_graph_scaler_again":False},
            "lookup_indexing":{"site_row":"global site_index - 1", "vector_row":"committed VALIDATION slot 0..47"},
            "site_partition_codes":dict(enumerate(SITE_COUNTS)),
            "categorical_vocabulary":frozen["categorical_vocabulary"],"numeric_preprocessing":frozen["numeric_preprocessing"],
            "preprocessing_action":"EXACT_COPY_OF_FROZEN_SITE_AND_STANDARDIZED_GRAPH_VALUES",
            "scaler_fit_calls":0,"graph_propagation_calls":0,"target_in_X":False,"post_simulation_features":[],
            "forbidden_columns_in_X":sorted(FORBIDDEN),"baseline_rows":"reference evidence only; excluded from prediction samples",
            "static_dev_members_read":["partition","site_ids","site_indices","site_features","site_feature_columns","feature_columns"],
            "development_targets_read":False,"materialization_canary":canary_result,
            "targets":{"array":"target_detected","classes":[0,1],"meaning":"observed simulated fault effect", "role":"evaluation only"}}


def inference_contract(source, models, environment, records, alignment):
    return {"stage":"11D-3G","status":"PASS","contract_status":"FROZEN","primary_model":"HYBRID_FUSION_MLP_11D2C",
            "purpose":"ONE_LOCKED_EVALUATION_ON_COMMITTED_UNSEEN_VALIDATION_STIMULI",
            "models":models,"dataset_partition":"VALIDATION","prediction_samples":N_ROWS,"physical_sites":N_SITES,
            "validation_vectors":N_VECTORS,"samples_per_site":96,"prediction_rule":"probability(class=1) >= frozen threshold",
            "prediction_classes":[0,1],"check_classes_before_probability_column_selection":True,
            "features":records["features"],"targets":records["targets"],"schema":records["schema"],"sample_identity_sha256":alignment,
            "execution":{"model_order":[row["model_id"] for row in models],"parallel_models":1,"threads":1,
                         "inference_batch_size":16384,"probability_dtype":"float64","input_dtype":"float32",
                         "deterministic_prediction_replay_required":True,"model_hash_recheck_before_and_after":True,
                         "assert_estimator_feature_width":True,"assert_gnn_bundle_cache_scaler_and_column_equality":True},
            "environment":environment["packages"],"preprocessing":"reuse frozen arrays; never fit or apply a graph scaler a second time",
            "acceptance_contract":source["acceptance_contract"],
            "metrics":["mcc","balanced_accuracy","precision","recall","specificity","f1_score","pr_auc","roc_auc","brier_score","tn","fp","fn","tp"],
            "pr_auc_definition":"average precision; record the metric implementation",
            "required_breakdowns":["fault_model","site_category","driver_cell_type","development_site_partition","validation_vector_id"],
            "undefined_metric_policy":"record null and reason for mathematically undefined ROC-AUC etc.; empty group is NOT_REPRESENTED; never drop rows",
            "site_confidence_intervals":{"method":"paired physical-site-group nonparametric percentile bootstrap",
                                         "replicates":1000,"random_seed":20260909,"confidence":0.95,
                                         "resampling":"sample 22839 sites with replacement; retain all 96 records per site; use identical draws for all models",
                                         "metrics":["mcc","hybrid_minus_gnn_mcc","hybrid_minus_deep_mcc","hybrid_minus_baseline_mcc"],
                                         "limits":"conditional on the same frozen HMAC circuit and these 48 vectors; not cross-chip evidence"},
            "generalization_scope":"new key/message pairs on the same circuit; report all-site metrics and separate original DEV_TRAIN/DEV_CALIBRATION/DEV_SITE_TEST site strata",
            "success_required_before_holdout":"separate result freeze and explicit holdout authorization",
            "runner_generation_authorized":True,"runner_preflight_authorized":True,
            "validation_inference_execution_authorized":False,"next_gate":NEXT_GATE,**BLOCKS}


def run(root):
    root=root.resolve()
    require(root.name=="vlsi_fault_detection_v2" and Path(__file__).resolve().parent==root,
            "copy the script into ~/vlsi_fault_detection_v2 and run it there")
    v=load_verifier(root); e=v.Evidence(root)
    print("STAGE 11D-3G — VALIDATION FEATURE MATRIX AND INFERENCE CONTRACT",flush=True)
    with v.campaign_lock(e):
        for path in OUTPUTS.values():
            require(not (root/path).exists() and not (root/path).is_symlink(),f"output already exists: {path}; preserve the completed or partial freeze")
        source_record=e.verify(Path(__file__).name,v.sha256(Path(__file__)))
        source,vectors,models,environment=verify_sources(e,v)
        x,frozen_schema=frozen_lookups(e,v)
        x["vector_features"]=vector_bits(e,v,vectors)
        x["vector_ids"]=np.asarray([row["vector_id"] for row in vectors],dtype="<u2")
        ids,target,totals=parse_canonical(e,v,source,vectors)
        x.update(ids)
        x.update(format_version=np.asarray([FORMAT]),partition=np.asarray(["VALIDATION"]))
        y={**{key:ids[key].copy() for key in IDENTITY},"target_detected":target}
        alignment=identity_sha(x)
        require(alignment==identity_sha(y),"X/y identity mismatch")
        canary_result=canary(x)
        groups=[]
        sample_groups=x["site_partition_code"][x["site_row"]]
        for code,(name,count) in enumerate(SITE_COUNTS.items()):
            mask=sample_groups==code
            require(int(mask.sum())==count*96,f"{name} validation samples")
            groups.append({"development_site_partition":name,"sites":count,"samples":int(mask.sum()),
                           "positive_samples":int(target[mask].sum(dtype=np.uint64))})
        require(shutil.disk_usage(e.path(R)).free>=512*1024**2,"at least 512 MiB free disk required")
        staging=Path(tempfile.mkdtemp(prefix=".stage_11d3g_pending_",dir=e.path(R)))
        try:
            staged={name:staging/Path(path).name for name,path in OUTPUTS.items()}
            print("\nWRITING SEPARATE FEATURES AND TARGETS; VERIFYING EXACT NPZ REPLAY",flush=True)
            write_and_replay(staged["features"],x,v)
            write_and_replay(staged["targets"],y,v)
            staged["schema"].write_bytes(v.encoded(schema_document(x,y,frozen_schema,canary_result)))
            records={name:artifact(staged[name],OUTPUTS[name],v) for name in ("features","targets","schema")}
            staged["contract"].write_bytes(v.encoded(inference_contract(source,models,environment,records,alignment)))
            records["contract"]=artifact(staged["contract"],OUTPUTS["contract"],v)
            print("\nRECHECKING FROZEN INPUTS BEFORE PUBLICATION",flush=True)
            e.recheck()
            common={"stage":"11D-3G","status":"PASS","created_at_utc":datetime.now(timezone.utc).isoformat(),
                    "generator_version":VERSION,"generator_source":source_record,"feature_matrix_status":"FROZEN",
                    "feature_schema_status":"FROZEN","inference_contract_status":"FROZEN","storage":"FACTORIZED NPZ; SEPARATE X AND Y",
                    "source_manifest":e.records[SOURCE_MANIFEST],"source_audit":e.records[SOURCE_AUDIT],"canonical_dataset":e.records[CANONICAL],
                    "source_dataset_commitment":SOURCE_COMMITMENT,"ordered_validation_commitment":source["ordered_validation_commitment"],
                    "model_samples":N_ROWS,"positive_samples":POSITIVES,"negative_samples":N_ROWS-POSITIVES,
                    "physical_sites":N_SITES,"fault_instances":N_SITES*2,"validation_vectors":N_VECTORS,"baseline_rows_excluded":2160,
                    "sample_features":527,"graph_features":119,"combined_features":646,"target_in_X":False,
                    "target_file_separate":True,"post_simulation_feature_count":0,"scaler_fit_calls":0,"graph_propagation_calls":0,
                    "sample_identity_sha256":alignment,"development_target_arrays_read":False,"train_records":0,"holdout_records":0,
                    "original_development_site_strata":groups,"canonical_counts":totals,"models":models,"acceptance_contract":source["acceptance_contract"],
                    "checks":{"frozen_site_features_exact":"PASS","graph_site_alignment":"PASS","stimulus_bit_order":"PASS",
                              "exact_sample_coverage":"PASS","target_alignment":"PASS","materialization_canary":"PASS",
                              "npz_readback":"PASS","exact_npz_replay":"PASS","final_input_recheck":"PASS"},
                    "materialization_canary":canary_result,"next_gate":NEXT_GATE,**records,**BLOCKS}
            manifest={**common,"manifest_status":"FROZEN","runtime":{"python":platform.python_version(),"numpy":np.__version__}}
            staged["manifest"].write_bytes(v.encoded(manifest))
            records["manifest"]=artifact(staged["manifest"],OUTPUTS["manifest"],v)
            audit={**common,"audit_status":"FROZEN","manifest":records["manifest"],"input_evidence":e.records,
                   "completion_rule":"AUDIT_AND_ALL_MATCHING_OUTPUT_HASHES_REQUIRED"}
            staged["audit"].write_bytes(v.encoded(audit))
            for name in ("features","targets","schema","contract","manifest"):
                publish(staged[name],e.path(OUTPUTS[name]))
                require(v.sha256(e.path(OUTPUTS[name]))==records[name]["sha256"],f"published hash: {name}")
            publish(staged["audit"],e.path(OUTPUTS["audit"]))
        except BaseException:
            print(f"Incomplete outputs preserved at {staging}; no PASS",file=sys.stderr)
            raise
        else:
            shutil.rmtree(staging)
        print("\nSTAGE 11D-3G — VALIDATION FEATURE MATRIX CONSTRUCTION AND INFERENCE-CONTRACT FREEZE")
        for label,value in (("Status","PASS"),("Feature matrix status","FROZEN"),("Inference contract status","FROZEN"),
                            ("VALIDATION samples",N_ROWS),("Positive / negative samples",f"{POSITIVES} / {N_ROWS-POSITIVES}"),
                            ("Baseline rows excluded",2160),("Physical sites / faults","22839 / 45678"),("VALIDATION vectors",48),
                            ("Sample / graph / combined X","527 / 119 / 646"),("Target in X","NO; SEPARATE TARGET FILE"),
                            ("Frozen static feature copies","EXACT"),("New scaler fits",0),("New graph propagation",0),
                            ("Development labels reopened","NO"),("Model objects deserialized",0),("Model inference performed","NO"),
                            ("Exact NPZ replay","PASS"),("Missing / duplicate samples","0 / 0"),("TRAIN / HOLDOUT samples","0 / 0"),
                            ("Frozen models and inputs changed","NO"),("Retraining / threshold changes","PROHIBITED"),
                            ("Validation model inference","NOT AUTHORIZED"),("HOLDOUT campaign","NOT AUTHORIZED")):
            print(f"{label:33}: {value}")
        for name,path in OUTPUTS.items():
            print(f"{name:33}: {e.path(path)}")
            print(f"{name+' SHA':33}: {v.sha256(e.path(path))}")
        print(f"{'Next gate':33}: {NEXT_GATE}")
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    args=parser.parse_args()
    try:
        require(sys.version_info>=(3,10),"Python 3.10+ required")
        return run(args.project_root)
    except KeyboardInterrupt:
        print("\nSTOP: interrupted. Preserve all inputs and partial outputs; no PASS.",file=sys.stderr)
        return 130
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}",file=sys.stderr)
        print("Stage 11D-3G did not pass. Do not run inference or modify frozen inputs.",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
