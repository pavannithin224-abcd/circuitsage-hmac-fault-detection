#!/usr/bin/env python3
"""Stage 11D-3F: consolidate and freeze the verified VALIDATION dataset.

Run in ~/vlsi_fault_detection_v2 with the unchanged Stage 11D-3E verifier
beside this file. Standard library only; no simulation or model inference.
The pinned verifier supplies its existing read-only integrity primitives.
Raw CSVs and all prior evidence remain untouched. New files are published
exclusively, with the freeze audit last. An audit without all matching output
hashes is invalid. Existing outputs are never overwritten.

--verify-existing checks a completed freeze without regenerating it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import types
import zlib

VERSION = "HMAC-VALIDATION-CANONICAL-CONSOLIDATOR-v1"
SCHEMA_VERSION = "HMAC-FAULT-CANONICAL-VALIDATION-SCHEMA-v1"
VERIFIER = "stage_11d3e_validation_dataset_integrity.py"
VERIFIER_SHA = "b2bdcbfc98510aaff4f1bfea55706cf7767a84607f59dbcd2c9267cba02f15dc"
R = "results/hmac_fault_campaign_11d3/"
D = R + "canonical_dataset_11d3f/"
SOURCE_MANIFEST = R + "hmac_validation_dataset_manifest_11d3e.json"
SOURCE_AUDIT = R + "hmac_validation_dataset_integrity_freeze_11d3e.json"
TRAIN_SCHEMA = "results/hmac_fault_campaign_11c5/canonical_dataset_11c5c/hmac_canonical_dataset_schema_11c5c.json"
EXPECTED = {
    VERIFIER: VERIFIER_SHA,
    SOURCE_MANIFEST: "f6a49c262b75a837405ab53b7a6bf992b7e2e2b546969c364430d0df59e3f223",
    SOURCE_AUDIT: "ed7b77c65419b2319ccd7a39732a7cabcfe63ff109a37da40373879846a80825",
    R + "stage_11d3e_integrity_20260909_093820.log": "8115da157dfbd374b06d6c2188ff83591897cb914625e261b4d16f456d7f79ab",
    R + "stage_11d3e_resources_20260909_093820.log": "c39aed1409d5d7d8271fdfb513450962dbc85ced5110f027c39118d37ca56237",
    TRAIN_SCHEMA: "2a08594f9573ff833a872d409783cee984bbb7499c7b013af4c72377ede3f772",
}
SOURCE_COMMITMENT = "68fc090166650192a05db7efd79e65304f8621b304e96bec2302ebfe31aa03f2"
EXPECTED_COUNTS = {
    "baseline_records": 2160, "enabled_records": 2192544, "total_records": 2194704,
    "fault_instances": 45678, "activated_records": 1771856, "detected_records": 964973,
    "detected_instances": 22922, "activated_unobserved_instances": 15386,
    "unactivated_instances": 7370, "timeout_records": 94320,
    "unclassified_instances": 0, "missing_samples": 0, "duplicate_samples": 0,
    "unknown_records": 0, "detected_without_activity": 0, "baseline_failures": 0,
    "train_records": 0, "holdout_records": 0,
}
OUTPUTS = {
    "canonical_dataset": D + "hmac_fault_campaign_canonical_validation_11d3f.csv.gz",
    "fault_instance_summary": D + "hmac_fault_instance_summary_validation_11d3f.csv",
    "schema": D + "hmac_canonical_validation_dataset_schema_11d3f.json",
    "manifest": R + "hmac_canonical_validation_dataset_manifest_11d3f.json",
    "audit": R + "hmac_canonical_validation_dataset_schema_freeze_11d3f.json",
}
CANONICAL_HEADER = (
    "record_id,vector_split,batch_id,run_type,site_selection,selector,site_id,stuck_value,"
    "vector_slot,vector_id,fault_enable,cycles,baseline_cycles,latency_delta,timed_out,"
    "activity,detected,unknown,digest_hamming_distance,expected_digest,actual_digest"
).split(",")
INSTANCE_HEADER = (
    "fault_instance_id,site_id,site_index,batch_id,selector,stuck_value,site_category,"
    "driver_cell_type,validation_vectors,activated_vectors,detected_vectors,timeout_vectors,final_classification"
).split(",")
NEXT_GATE = "VALIDATION FEATURE MATRIX CONSTRUCTION AND INFERENCE-CONTRACT FREEZE"
BLOCKS = {
    "validation_model_inference": "NOT_AUTHORIZED", "holdout_campaign": "NOT_AUTHORIZED",
    "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED",
    "model_inference_performed": False, "training_performed": False,
    "model_deserialization_performed": False, "holdout_payloads_emitted": 0,
    "frozen_input_artifacts_modified": False, "raw_campaign_dataset_modified": False,
    "frozen_rtl_modified": False, "golden_netlist_modified": False,
}


class ConsolidationError(Exception):
    pass


def require(ok, message):
    if not ok:
        raise ConsolidationError(message)


def load_verifier(root):
    path = root / VERIFIER
    require(path.is_file() and not path.is_symlink(), f"missing regular prerequisite: {VERIFIER}")
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    require(actual == VERIFIER_SHA, f"prerequisite SHA mismatch: {VERIFIER}\nExpected: {VERIFIER_SHA}\nActual:   {actual}")
    # Execute exactly the verified bytes under a non-main name; do not write pycache
    # or call the prior stage's run/main functions.
    module = types.ModuleType("_frozen_stage_11d3e")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def record(path, relative, v):
    return {"path": relative, "sha256": v.sha256(path), "bytes": path.stat().st_size}


def verify_sources(e, v):
    print("FROZEN STAGE 11D-3E AND TRAIN SCHEMA VERIFICATION", flush=True)
    for relative, digest in EXPECTED.items():
        e.verify(relative, digest)
        print(f"  {Path(relative).name}: OK", flush=True)
    manifest, audit = e.load(SOURCE_MANIFEST), e.load(SOURCE_AUDIT)
    common = {"stage": "11D-3E", "status": "PASS", "dataset_status": "FROZEN",
              "vector_partition": "VALIDATION", "batches_verified": 45, "validation_vectors": 48,
              "legal_sites": 22839, "persistent_fault_instances": 45678,
              "ordered_dataset_commitment": SOURCE_COMMITMENT, **BLOCKS}
    v.fields(manifest, {**common, "manifest_status": "FROZEN", "csv_header": v.CSV_HEADER}, "3E manifest")
    v.fields(audit, {**common, "audit_status": "FROZEN"}, "3E audit")
    require(e.link(audit["manifest"], SOURCE_MANIFEST) == e.records[SOURCE_MANIFEST], "3E manifest/audit binding")
    require(manifest["summary"] == audit["summary"], "3E summary binding")
    v.fields(manifest["summary"], EXPECTED_COUNTS, "frozen real validation counts")
    require(set(manifest["summary"]) == set(v.STAT_FIELDS), "3E summary field set")
    linked = audit["input_evidence"]
    require(isinstance(linked, dict) and set(v.EXPECTED) <= linked.keys(), "3E evidence chain incomplete")
    for i, (relative, rec) in enumerate(linked.items(), 1):
        e.link(rec, relative)
        if i % 100 == 0:
            print(f"  Prior frozen artifacts: {i}/{len(linked)}", flush=True)
    for relative, digest in v.EXPECTED.items():
        require(e.records[relative]["sha256"] == digest, f"3E prerequisite anchor: {relative}")
    v.verify_resource_log(e.path(R + "stage_11d3e_resources_20260909_093820.log"))
    require([f["name"] for f in e.load(TRAIN_SCHEMA)["fields"]] == CANONICAL_HEADER,
            "canonical columns differ from the frozen TRAIN schema")
    vectors, commitment, _, _ = v.validation_vectors(e, e.load(v.AUTH), e.load(v.AUDIT_C), e.load(v.CHECKPOINT))
    ids = [x["vector_id"] for x in vectors]
    require(ids == manifest["validation_vector_ids"] and len(set(ids)) == 48, "VALIDATION ID order")
    require(commitment == manifest["ordered_validation_commitment"] == audit["ordered_validation_commitment"],
            "validation commitment binding")
    batches = manifest["batch_records"]
    require(len(batches) == 45 and [b["batch_id"] for b in batches] == list(range(45)), "ordered 45-batch coverage")
    ordered, total = hashlib.sha256(), Counter()
    for batch in batches:
        bid = batch["batch_id"]
        v.fields(batch, {**v.batch_plan(bid), "status": "PASS"}, f"batch {bid:03d}")
        for rec in batch["artifacts"].values():
            require(e.records.get(rec["path"]) == rec, "batch artifact absent from 3E input evidence")
        rec = batch["artifacts"]["csv"]
        ordered.update(f"{bid:03d}|{rec['path']}|{rec['sha256']}|{batch['total_records']}\n".encode())
        total.update(batch["summary"])
    require(ordered.hexdigest() == SOURCE_COMMITMENT, "ordered raw dataset commitment")
    require(dict(total) == manifest["summary"], "per-batch frozen summary aggregation")
    require(manifest["raw_csv_bytes"] == sum(b["artifacts"]["csv"]["bytes"] for b in batches), "raw byte total")
    return manifest, vectors


def canonical_row(row, record_id):
    # Only three columns are added; partition is renamed and digests lowercased.
    cycles, baseline = int(row[10]), int(row[11])
    expected, actual = row[16].lower(), row[17].lower()
    return [str(record_id), row[1], row[0], row[2], row[3], row[4], row[5], row[6],
            row[7], row[8], row[9], row[10], row[11], str(cycles - baseline), row[12],
            row[13], row[14], row[15], str((int(expected, 16) ^ int(actual, 16)).bit_count()), expected, actual]


def encoded_row(row):
    # Frozen values contain no commas, quotes, CR, or LF; reject rather than emit
    # an ambiguous commitment. All serialized data therefore use this exact form.
    require(all(isinstance(x, str) and not any(c in x for c in ',"\r\n') for x in row), "unsafe CSV field")
    return (",".join(row) + "\n").encode("utf-8")


def mapped_sites(e, batch, v):
    bid = batch["batch_id"]
    sites = e.load(batch["artifacts"]["mapping"]["path"])["sites"]
    require(len(sites) == batch["legal_sites"], "mapping site count")
    for selector, site in enumerate(sites):
        index = bid * 512 + selector + 1
        v.fields(site, {"selector_code": selector, "site_index": index,
                        "fault_site_id": f"HMAC-STEM-{index:06d}"}, "site identity")
        for field in ("site_category", "driver_cell_type"):
            require(isinstance(site.get(field), str) and site[field], f"missing mapped {field}: {site['fault_site_id']}")
    return sites


def write_batch(e, v, batch, sites, stream, record_offset):
    bid, counts = batch["batch_id"], [[0, 0, 0, 0] for _ in range(len(sites) * 2)]
    rec = batch["artifacts"]["csv"]
    raw_sha, canonical_sha = hashlib.sha256(), hashlib.sha256()
    nrows = 0
    with e.path(rec["path"]).open("rb") as raw:
        def lines():
            for line in raw:
                raw_sha.update(line)
                yield line.decode("utf-8")
        reader = csv.reader(lines(), strict=True)
        require(next(reader, None) == v.CSV_HEADER, "raw CSV header changed")
        for row in reader:
            require(len(row) == 18 and row[1] == "VALIDATION" and int(row[0]) == bid, "raw row structure changed")
            normalized = canonical_row(row, record_offset + nrows)
            payload = encoded_row(normalized)
            stream.write(payload)
            canonical_sha.update(payload)
            if row[2] == "ENABLED":
                local = int(row[3]) * 2 + int(row[6])
                require(0 <= local < len(counts), "fault instance outside mapping")
                for j, value in enumerate((1, int(row[13]), int(row[14]), int(row[12]))):
                    counts[local][j] += value
            nrows += 1
    require(raw_sha.hexdigest() == rec["sha256"], f"Batch {bid:03d}: input changed during transformation")
    require(nrows == batch["total_records"], "transformed row count")
    instances, classes = [], Counter()
    for site, fault_counts in zip(sites, zip(counts[::2], counts[1::2]), strict=True):
        for stuck, values in enumerate(fault_counts):
            n, active, detected, timeout = values
            require(n == 48 and 0 <= detected <= active <= n and 0 <= timeout <= detected, "instance vector counts")
            kind = "DETECTED" if detected else "ACTIVATED_UNOBSERVED" if active else "UNACTIVATED"
            classes[kind] += 1
            instances.append([f"{site['fault_site_id']}-SA{stuck}", site["fault_site_id"], site["site_index"],
                              bid, site["selector_code"], stuck, site["site_category"], site["driver_cell_type"],
                              n, active, detected, timeout, kind])
    expected = batch["summary"]
    for name, kind in (("detected_instances", "DETECTED"), ("activated_unobserved_instances", "ACTIVATED_UNOBSERVED"),
                       ("unactivated_instances", "UNACTIVATED")):
        require(classes[kind] == expected[name], f"batch classification mismatch: {name}")
    for name, pos in (("enabled_records", 0), ("activated_records", 1), ("detected_records", 2), ("timeout_records", 3)):
        require(sum(c[pos] for c in counts) == expected[name], f"batch instance count mismatch: {name}")
    return {"batch_id": bid, "first_record_id": record_offset, "last_record_id": record_offset + nrows - 1,
            "total_records": nrows, "canonical_rows_sha256": canonical_sha.hexdigest(),
            "raw_csv": rec, "mapping": batch["artifacts"]["mapping"], "summary": expected}, instances


def write_canonical(e, v, source, vectors, destination):
    result, instances, offset = [], [], 0
    print("\nVERIFYING AND CONSOLIDATING ALL 45 BATCHES", flush=True)
    with destination.open("xb") as disk:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, mtime=0, fileobj=disk) as gz:
            gz.write(encoded_row(CANONICAL_HEADER))
            for batch in source["batch_records"]:
                sites = mapped_sites(e, batch, v)
                rec = batch["artifacts"]["csv"]
                stats, per_vector = v.scan_csv(e.path(rec["path"]), batch["batch_id"], sites, vectors, rec["sha256"])
                require(stats == batch["summary"] and per_vector == batch["per_vector"], "raw rows differ from frozen 3E aggregates")
                output, summary = write_batch(e, v, batch, sites, gz, offset)
                result.append(output)
                instances.extend(summary)
                offset += output["total_records"]
                print(f"  Batch {batch['batch_id']:03d}: PASS canonical_rows={offset}/2194704", flush=True)
        disk.flush()
        os.fsync(disk.fileno())
    require(offset == EXPECTED_COUNTS["total_records"] and len(instances) == 45678, "canonical campaign coverage")
    require(len({row[0] for row in instances}) == 45678, "duplicate instance summary identity")
    return result, instances


class HashSink:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, data):
        self.digest.update(data)
        self.bytes += len(data)
        return len(data)

    def flush(self):
        pass


def verify_canonical(path, batches, v):
    print("\nCANONICAL READBACK AND EXACT GZIP REPLAY", flush=True)
    sink, uncompressed, total = HashSink(), hashlib.sha256(), 0
    with gzip.open(path, "rb") as source:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, mtime=0, fileobj=sink) as replay:
            header = next(source, b"")
            require(header == encoded_row(CANONICAL_HEADER), "canonical readback header")
            replay.write(header)
            uncompressed.update(header)
            for batch in batches:
                digest = hashlib.sha256()
                for _ in range(batch["total_records"]):
                    line = next(source, b"")
                    require(line and line.endswith(b"\n"), "canonical file truncated")
                    row = line.decode("utf-8").rstrip("\n").split(",")
                    require(len(row) == 21 and row[0] == str(total) and row[1] == "VALIDATION"
                            and row[2] == str(batch["batch_id"]), "canonical readback identity/order")
                    digest.update(line)
                    uncompressed.update(line)
                    replay.write(line)
                    total += 1
                require(digest.hexdigest() == batch["canonical_rows_sha256"], "canonical/raw normalized-row commitment mismatch")
            require(source.read(1) == b"", "canonical file has extra records")
    require(total == EXPECTED_COUNTS["total_records"], "canonical readback row count")
    require(sink.digest.hexdigest() == v.sha256(path) and sink.bytes == path.stat().st_size, "gzip byte replay mismatch")
    return {"canonical_readback": "PASS", "raw_to_canonical_row_commitments": "PASS",
            "exact_gzip_replay": "PASS", "canonical_data_records": total,
            "uncompressed_csv_sha256": uncompressed.hexdigest(), "replayed_gzip_sha256": sink.digest.hexdigest()}


def schema_definition(train_schema, source):
    definitions = json.loads(json.dumps(train_schema["fields"]))
    for field in definitions:
        if field["name"] == "vector_split":
            field["description"] = "Frozen vector partition; VALIDATION in this artifact"
        elif field["name"] == "vector_slot":
            field["description"] = "Committed VALIDATION order, 0 through 47"
        elif field["name"] == "detected":
            field["role"] = "evaluation_target"
        elif field["name"] == "expected_digest":
            field["description"] = "Simulation reference only; excluded from prediction features"
    return {"schema_version": SCHEMA_VERSION, "dataset_name": "OpenTitan HMAC-SHA256 SA0/SA1 VALIDATION",
            "schema_status": "FROZEN", "fields": definitions, "canonical_header": CANONICAL_HEADER,
            "row_grain": "one baseline or one persistent fault/vector simulation",
            "compression": {"algorithm": "gzip", "level": 6, "mtime": 0, "filename": "",
                            "text_encoding": "UTF-8", "line_ending": "LF"},
            "canonical_order": "batch_id ascending, then frozen raw CSV row order; contiguous zero-based record_id",
            "source_column_mapping": {"vector_partition": "vector_split", "other_raw_fields": "same name; digest hex normalized to lowercase"},
            "added_fields": {"record_id": "zero-based global row ordinal", "latency_delta": "cycles - baseline_cycles",
                             "digest_hamming_distance": "popcount(int(expected_digest,16) XOR int(actual_digest,16))"},
            "constraints": {"record_id": "0..2194703", "vector_split": ["VALIDATION"], "batch_id": "0..44",
                            "run_type": ["BASELINE", "ENABLED"], "baseline_records": 2160, "enabled_records": 2192544,
                            "vector_ids": source["validation_vector_ids"], "baseline_cycles": 343, "timeout_cycles": 2000,
                            "cycles": "0..2000", "latency_delta": "-343..1657", "digest_hamming_distance": "0..256",
                            "baseline_site_id": "BASELINE", "baseline_site_selection_selector_stuck_value": -1,
                            "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY", "unknown": 0},
            "sample_identity": ["batch_id", "site_id", "stuck_value", "vector_id"],
            "baseline_identity": ["batch_id", "vector_id"], "label_definition": source["label_definition"],
            "fault_instance_summary": {
                "header": INSTANCE_HEADER, "rows": 45678,
                "column_types": ["string", "string", "uint32", "uint8", "uint16", "bool", "string", "string",
                                 "uint8", "uint8", "uint8", "uint8", "enum"],
                "order": "global site_index ascending, then SA0, SA1",
                "identity": "fault_instance_id = site_id + '-SA' + stuck_value",
                "classification_scope": "these 48 VALIDATION vectors only",
                "classification_rule": "DETECTED if detected_vectors>0; else ACTIVATED_UNOBSERVED if activated_vectors>0; else UNACTIVATED"},
            "inference_feature_policy": {
                "this_file_is_not_X": True, "fit_scaler_or_model_on_validation": False,
                "baseline_rows": "reference checks only; exclude from fault prediction sample set",
                "post_simulation_columns_excluded_from_X": ["cycles", "latency_delta", "timed_out", "activity", "detected",
                                                           "unknown", "digest_hamming_distance", "actual_digest"],
                "reference_digest_excluded_from_X": "expected_digest",
                "identifiers": "join/alignment only, subject to the frozen feature policy",
                "next_gate": NEXT_GATE, **BLOCKS},
            "timeout_semantics": "retained as observed fault outcomes; baseline timeouts must be zero",
            "detection_reason_counts_overlap": True, "unknown_check_scope": source["unknown_check_scope"]}


def publish_file(staged, destination):
    require(not destination.exists() and not destination.is_symlink(), f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(staged, destination)  # Atomic exclusive link; EEXIST cannot overwrite.
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_existing(e, v, source):
    audit_path, manifest_path = e.path(OUTPUTS["audit"]), e.path(OUTPUTS["manifest"])
    require(audit_path.is_file() and manifest_path.is_file(), "no complete 3F freeze; preserve any partial outputs")
    audit = json.loads(audit_path.read_text(), object_pairs_hook=v.unique_object)
    v.fields(audit, {"stage": "11D-3F", "status": "PASS", "canonical_dataset_status": "FROZEN",
                     "schema_status": "FROZEN", "source_dataset_commitment": SOURCE_COMMITMENT, **BLOCKS}, "existing 3F audit")
    require(audit["summary"] == source["summary"], "existing summary differs from 3E")
    e.link(audit["source_manifest"], SOURCE_MANIFEST)
    e.link(audit["source_audit"], SOURCE_AUDIT)
    e.link(audit["consolidator_source"], Path(__file__).name)
    e.link(audit["manifest"], OUTPUTS["manifest"])
    manifest = e.load(OUTPUTS["manifest"])
    for name in ("canonical_dataset", "fault_instance_summary", "schema"):
        e.link(audit[name], OUTPUTS[name])
        require(audit[name] == manifest[name], f"existing output binding: {name}")
    require(manifest["summary"] == source["summary"] and manifest["checks"] == audit["checks"], "existing manifest/audit consistency")
    verify_canonical(e.path(OUTPUTS["canonical_dataset"]), manifest["batches"], v)
    print("EXISTING_STAGE_11D3F_VERIFICATION=PASS; no outputs replaced")
    return 0


def run(root, verify_only=False):
    root = root.resolve()
    require(root.name == "vlsi_fault_detection_v2" and Path(__file__).resolve().parent == root,
            "copy this script to ~/vlsi_fault_detection_v2 and run it there")
    v = load_verifier(root)
    e = v.Evidence(root)
    print("STAGE 11D-3F — VALIDATION DATASET CONSOLIDATION AND SCHEMA FREEZE", flush=True)
    with v.campaign_lock(e):
        script = e.verify(Path(__file__).name, v.sha256(Path(__file__)))
        if not verify_only:
            for relative in OUTPUTS.values():
                p = root / relative
                require(not p.exists() and not p.is_symlink(), f"output already exists: {relative}; use --verify-existing for a completed freeze")
        source, vectors = verify_sources(e, v)
        if verify_only:
            return verify_existing(e, v, source)
        results = e.path(R)
        # Data are streamed; allow space for gzip plus summaries, without expanding
        # a full uncompressed copy or allocating a feature matrix in memory.
        require(shutil.disk_usage(results).free >= 1024**3, "at least 1 GiB free disk required for new outputs")
        staging = Path(tempfile.mkdtemp(prefix=".stage_11d3f_pending_", dir=results))
        print(f"Working directory: {staging.name} (temporary outputs only)", flush=True)
        try:
            staged = {name: staging / Path(relative).name for name, relative in OUTPUTS.items()}
            batches, instances = write_canonical(e, v, source, vectors, staged["canonical_dataset"])
            checks = verify_canonical(staged["canonical_dataset"], batches, v)
            with staged["fault_instance_summary"].open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(INSTANCE_HEADER)
                writer.writerows(instances)
            with staged["fault_instance_summary"].open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                require(next(reader) == INSTANCE_HEADER, "instance summary readback header")
                for expected_row, actual_row in zip(instances, reader, strict=True):
                    require([str(x) for x in expected_row] == actual_row, "instance summary readback mismatch")
            checks.update(fault_instance_summary_readback="PASS", input_summary_agreement="PASS",
                          exact_sample_coverage="PASS", train_schema_column_compatibility="PASS")
            staged["schema"].write_bytes(v.encoded(schema_definition(e.load(TRAIN_SCHEMA), source)))
            print("\nRECHECKING FROZEN INPUT BYTES BEFORE PUBLICATION", flush=True)
            e.recheck()
            checks["final_input_recheck"] = "PASS"
            artifacts = {name: record(staged[name], OUTPUTS[name], v) for name in ("canonical_dataset", "fault_instance_summary", "schema")}
            common = {"stage": "11D-3F", "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                      "consolidator_version": VERSION, "consolidator_source": script, "schema_version": SCHEMA_VERSION,
                      "canonical_dataset_status": "FROZEN", "schema_status": "FROZEN", "source_stage": "11D-3E",
                      "source_dataset_commitment": SOURCE_COMMITMENT, "source_manifest": e.records[SOURCE_MANIFEST],
                      "source_audit": e.records[SOURCE_AUDIT], "train_schema": e.records[TRAIN_SCHEMA],
                      "ordered_validation_commitment": source["ordered_validation_commitment"],
                      "validation_vector_ids": source["validation_vector_ids"], "validation_vectors": 48,
                      "vector_partition": "VALIDATION", "batch_count": 45, "legal_sites": 22839,
                      "persistent_fault_instances": 45678, "summary": source["summary"],
                      "enabled_positive_labels": source["enabled_positive_labels"],
                      "enabled_negative_labels": source["enabled_negative_labels"],
                      "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY", "acceptance_contract": source["acceptance_contract"],
                      "checks": checks, "next_gate": NEXT_GATE, **BLOCKS, **artifacts}
            manifest = {**common, "manifest_status": "FROZEN", "canonical_header": CANONICAL_HEADER,
                        "fault_instance_summary_header": INSTANCE_HEADER, "batches": batches, "per_vector": source["per_vector"],
                        "canonical_order": "batch_id ascending, then frozen raw CSV row order",
                        "batch_row_commitment_algorithm": "SHA256 of normalized UTF-8 canonical data lines including LF; no header; global record_id",
                        "runtime": {"python": platform.python_version(), "zlib": zlib.ZLIB_RUNTIME_VERSION}}
            staged["manifest"].write_bytes(v.encoded(manifest))
            manifest_record = record(staged["manifest"], OUTPUTS["manifest"], v)
            audit = {**common, "audit_status": "FROZEN", "manifest": manifest_record,
                     "input_evidence": e.records, "completion_rule": "AUDIT_AND_ALL_MATCHING_OUTPUT_HASHES_REQUIRED"}
            staged["audit"].write_bytes(v.encoded(audit))
            # The audit is last: no failure before this point can freeze the data.
            for name in ("canonical_dataset", "fault_instance_summary", "schema", "manifest"):
                publish_file(staged[name], e.path(OUTPUTS[name]))
            for name, rec in {**artifacts, "manifest": manifest_record}.items():
                require(v.sha256(e.path(OUTPUTS[name])) == rec["sha256"], f"published output hash: {name}")
            publish_file(staged["audit"], e.path(OUTPUTS["audit"]))
        except BaseException:
            print(f"STOP: incomplete Stage 11D-3F; preserve pending files at {staging}", file=sys.stderr)
            raise
        else:
            shutil.rmtree(staging)  # Only this run's temporary files; published hardlinks remain.
        print("\nSTAGE 11D-3F — VALIDATION DATASET CONSOLIDATION AND SCHEMA FREEZE")
        for label, value in (("Status", "PASS"), ("Canonical dataset status", "FROZEN"), ("Schema status", "FROZEN"),
                             ("Batches consolidated", "45/45"), ("Canonical rows", source["summary"]["total_records"]),
                             ("Baseline rows", 2160), ("Enabled rows", 2192544), ("VALIDATION vectors", 48),
                             ("Fault-instance summary rows", 45678), ("Detected instances", source["summary"]["detected_instances"]),
                             ("Activated-unobserved", source["summary"]["activated_unobserved_instances"]),
                             ("Unactivated", source["summary"]["unactivated_instances"]),
                             ("Enabled timeout rows retained", source["summary"]["timeout_records"]),
                             ("Canonical columns", 21), ("Raw-to-canonical agreement", "PASS"), ("Exact gzip replay", "PASS"),
                             ("Missing/duplicate samples", "0/0"), ("TRAIN/HOLDOUT rows", "0/0"),
                             ("Validation model inference", "NOT AUTHORIZED"), ("Holdout campaign", "NOT AUTHORIZED"),
                             ("Retraining/threshold changes", "PROHIBITED"), ("Frozen inputs modified", "NO"),
                             ("Raw campaign dataset modified", "NO"), ("Source dataset commitment", SOURCE_COMMITMENT)):
            print(f"{label:31}: {value}")
        for name, relative in OUTPUTS.items():
            print(f"{name:31}: {e.path(relative)}")
            print(f"{name + ' SHA':31}: {v.sha256(e.path(relative))}")
        print(f"{'Next gate':31}: {NEXT_GATE}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-existing", action="store_true", help="verify a completed freeze; never overwrite outputs")
    args = parser.parse_args()
    try:
        require(sys.version_info >= (3, 10), "Python 3.10+ required")
        return run(args.project_root, args.verify_existing)
    except KeyboardInterrupt:
        print("\nSTOP: interrupted; no PASS. Preserve all source and partial output files.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        print("Stage 11D-3F did not pass. Model inference remains unauthorized.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
