#!/usr/bin/env python3
"""Verify and freeze the completed Stage 11D-3D VALIDATION dataset.

Standard-library, streaming, offline verifier. Reads the completed raw campaign;
does not run simulation, import the runner, deserialize models, perform inference,
train, tune, or consolidate CSVs. Only the committed VALIDATION vector payloads
are used for reference checks. No HOLDOUT payload is emitted or simulated.

Run from ~/vlsi_fault_detection_v2. --check-only verifies without writing outputs.
Normal execution publishes a manifest and then an audit, each exclusively.
Both files and a matching manifest hash are required for a complete freeze.
Existing outputs are preserved. Failures and interruptions never report PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys

VERSION = "HMAC-VALIDATION-DATASET-INTEGRITY-v1"
RUNNER_VERSION = "HMAC-VALIDATION-CAMPAIGN-RUNNER-v1"
R = "results/hmac_fault_campaign_11d3/"
C = "config/fault_campaign/"
RAW = "results/hmac_validation_campaign_full_11d3/"
BUILD = "build/hmac_validation_campaign_full_11d3/"
ATTEMPT = R + "stage_11d3d_attempts/20260908T153658Z_pu7c3g_p/"
RECEIPT = ATTEMPT + "completion.json"
CHECKPOINT = RAW + "hmac_validation_campaign_checkpoint.json"
AUTH = C + "hmac_validation_campaign_execution_authorization_11d3c.json"
AUDIT_C = R + "hmac_validation_campaign_execution_authorization_freeze_11d3c.json"
ARCH = C + "hmac_validation_campaign_architecture_11d3a.json"
CONTRACT = C + "hmac_validation_campaign_execution_contract_11d3a.json"
COMMITMENTS = R + "hmac_validation_vector_commitments_11d3a.json"
POOL = "results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json"
SEMANTIC = "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json"
RUNNER = "stage_11d3b_validation_runner.py"
LAUNCHER = "stage_11d3d_validation_launch.py"
MANIFEST = R + "hmac_validation_dataset_manifest_11d3e.json"
AUDIT = R + "hmac_validation_dataset_integrity_freeze_11d3e.json"
LAUNCH_LOCK = R + "hmac_validation_campaign_11d3d.lock"
NEXT_GATE = "VALIDATION DATASET CONSOLIDATION AND SCHEMA FREEZE"
BATCH_IDS = list(range(45))
N_VECTORS = 48
BASELINE_CYCLES = 343
TIMEOUT_CYCLES = 2000
EXPECTED_TOTALS = {"baseline_records": 2160, "enabled_records": 2192544,
                   "total_records": 2194704, "fault_instances": 45678}
EXPECTED = {
    CHECKPOINT: "501b2f89752846480e5963e9ce492e4ce42edc5249bf48bb14ea9e587201ba68",
    RECEIPT: "9146b45dce67ae62e32ac4de3daa0fe60db8c572eb25e197fd7bfcfa6f721ee5",
    AUTH: "c1195366d67ac2c72097903db7fb9f1ab610877e277d2292d79d0c1d9179c247",
    AUDIT_C: "af3c7309df4d4abf547ea2b245a999a552400b3a8df545d5b296a734e6a87875",
    RUNNER: "71280fdcbe07b66f10f3602630d252b8e92b1816172f7b93689eba95b717a777",
    LAUNCHER: "7420621bc596cb4d637d3d1cb5159763c5a6fd64f41e1a96fdc779e948406287",
    "stage_11d3c_validation_authorization.py": "da975aa3848edc501588ec461262a9613915c912f2976e5f77e3e267b47bd874",
    ARCH: "256aca0cba7d14439f985d42a0683912a0e92285282f2fa116eb43728ffe2f8d",
    CONTRACT: "55c3b2c57c8a6a7068812a41a75b01f75292ce0068c4553e3fd35fb972a87420",
    COMMITMENTS: "9fff03e0dd00e91d405daee71e98af62408969407547308180475d90d20ba99b",
    POOL: "9d612ac13160e3bca0e4c8f06946d5e577cb5b5faaaf73c6a20f5b180a2a1e93",
    SEMANTIC: "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
}
CSV_HEADER = (
    "batch_id,vector_partition,run_type,site_selection,selector,site_id,stuck_value,"
    "vector_slot,vector_id,fault_enable,cycles,baseline_cycles,timed_out,activity,"
    "detected,unknown,expected_digest,actual_digest"
).split(",")
HEX256 = re.compile(r"[0-9a-f]{64}")
STAT_FIELDS = (
    "baseline_records", "enabled_records", "total_records", "activated_records", "detected_records",
    "timeout_records", "digest_mismatch_records", "latency_mismatch_records", "unknown_records",
    "detected_without_activity", "baseline_failures", "baseline_digest_mismatches",
    "baseline_latency_mismatches", "baseline_timeouts", "fault_instances", "detected_instances",
    "activated_unobserved_instances", "unactivated_instances", "unclassified_instances",
    "duplicate_samples", "missing_samples", "train_records", "holdout_records",
)


class VerificationError(Exception):
    pass


def require(ok, message):
    if not ok:
        raise VerificationError(message)


def fields(document, wanted, label):
    require(isinstance(document, dict), f"{label}: JSON object required")
    for key, expected in wanted.items():
        require(type(document.get(key)) is type(expected) and document[key] == expected,
                f"{label}/{key}: expected {expected!r}, got {document.get(key)!r}")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
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
        require(not p.is_absolute() and ".." not in p.parts, f"non-project-relative artifact: {relative}")
        resolved = (self.root / p).resolve()
        require(resolved.is_relative_to(self.root), f"artifact escapes project: {relative}")
        return resolved

    def verify(self, relative, expected, size=None):
        require(isinstance(expected, str) and HEX256.fullmatch(expected), f"malformed SHA: {relative}")
        p = self.path(relative)
        require(p.is_file() and p.stat().st_size > 0, f"missing or empty input: {relative}")
        before = p.stat()
        actual = sha256(p)
        after = p.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"input changed while hashing: {relative}")
        require(actual == expected, f"SHA mismatch for {relative}\nExpected: {expected}\nActual:   {actual}")
        if size is not None:
            require(type(size) is int and size == after.st_size, f"byte-count mismatch: {relative}")
        record = {"path": relative, "sha256": actual, "bytes": after.st_size}
        require(relative not in self.records or self.records[relative] == record, f"conflicting evidence: {relative}")
        self.records[relative] = record
        return record

    def link(self, record, expected_path=None):
        require(isinstance(record, dict) and {"path", "sha256", "bytes"} <= record.keys(), "malformed artifact record")
        if expected_path is not None:
            require(record["path"] == expected_path, f"artifact path mismatch: expected {expected_path}, got {record['path']}")
        return self.verify(record["path"], record["sha256"], record["bytes"])

    def load(self, relative):
        require(relative in self.records, f"unverified JSON: {relative}")
        obj = json.loads(self.path(relative).read_text(), object_pairs_hook=unique_object)
        require(isinstance(obj, dict), f"JSON object required: {relative}")
        return obj

    def recheck(self):
        records = list(self.records.values())
        for i, rec in enumerate(records, 1):
            self.link(rec)
            if i % 100 == 0:
                print(f"  Final evidence recheck: {i}/{len(records)}", flush=True)


@contextmanager
def campaign_lock(e):
    # Reuse the existing launcher's lock without editing its file contents.
    path = e.path(LAUNCH_LOCK)
    require(path.is_file(), "campaign launch lock file missing; preserve evidence and request review")
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise VerificationError("campaign or another verifier still holds the launch lock; wait for completion")
        try:
            # Catch a direct execution outside the launcher under this user.
            for p in Path("/proc").iterdir():
                if not p.name.isdecimal():
                    continue
                try:
                    args = [os.fsdecode(x) for x in (p / "cmdline").read_bytes().split(b"\0") if x]
                    if "--execute-validation" not in args or not any(Path(a).name == RUNNER for a in args):
                        continue
                    cwd = (p / "cwd").resolve()
                    same = cwd == e.root
                    if "--project-root" in args:
                        selected = Path(args[args.index("--project-root") + 1])
                        same = same or (cwd / selected).resolve() == e.root
                    require(not same, f"validation runner still active (PID {p.name})")
                except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
                    continue
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_inputs(e):
    print("FROZEN INPUT AND EXECUTION EVIDENCE VERIFICATION", flush=True)
    failures = []
    for relative, expected in EXPECTED.items():
        try:
            e.verify(relative, expected)
            print(f"  {Path(relative).name}: OK", flush=True)
        except (VerificationError, OSError) as error:
            failures.append(str(error))
    require(not failures, "frozen input verification failed:\n" + "\n".join(failures))
    auth, previous, receipt, cp = (e.load(p) for p in (AUTH, AUDIT_C, RECEIPT, CHECKPOINT))
    scope = {"batches": 45, "legal_sites": 22839, "persistent_fault_instances": 45678,
             "validation_vectors": 48, "baseline_records": 2160, "enabled_records": 2192544,
             "total_records": 2194704, "validation_campaign_execution": "AUTHORIZED",
             "validation_model_inference": "NOT_AUTHORIZED", "holdout_campaign": "NOT_AUTHORIZED",
             "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED",
             "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
             "checkpoint_after_each_batch": True, "resume_supported": True}
    fields(auth, {"stage": "11D-3C", "status": "FROZEN", "runner_sha256": EXPECTED[RUNNER],
                  "batch_ids": BATCH_IDS, "vector_partition": "VALIDATION", **scope}, "authorization")
    fields(previous, {"stage": "11D-3C", "status": "PASS", "authorization_status": "FROZEN", **scope}, "3C audit")
    require(e.link(previous["authorization"], AUTH) == e.records[AUTH], "3C authorization binding")
    linked = previous["input_evidence"]
    require(isinstance(linked, dict) and {RUNNER, ARCH, CONTRACT, POOL, SEMANTIC} <= linked.keys(), "3C evidence index incomplete")
    for i, (relative, rec) in enumerate(linked.items(), 1):
        require(relative == rec["path"], "3C evidence key/path mismatch")
        e.link(rec)
        if i % 50 == 0:
            print(f"  Prior frozen evidence: {i}/{len(linked)}", flush=True)
    fields(receipt, {"stage": "11D-3D", "launcher_version": "HMAC-VALIDATION-LAUNCHER-v1",
                     "status": "PASS", "runner_return_code": 0, "integrity_errors": [],
                     "completed_batches": 45, "baseline_records": 2160, "enabled_records": 2192544,
                     "total_records": 2194704, "frozen_inputs_unchanged": True,
                     "dataset_integrity_freeze": "NOT_PERFORMED", "model_inference": "NOT_PERFORMED"}, "completion receipt")
    require(e.link(receipt["checkpoint"], CHECKPOINT) == e.records[CHECKPOINT], "receipt/checkpoint binding")
    e.link(receipt["master_log"], ATTEMPT + "master.log")
    e.link(receipt["resources"], ATTEMPT + "resources.log")
    master = e.path(ATTEMPT + "master.log").read_text()
    for token in ("VALIDATION_CAMPAIGN_RUNNER_RESULT=PASS", "STAGE_11D3D_EXECUTION_RESULT=PASS"):
        require(master.splitlines().count(token) == 1, f"missing or repeated master closure token: {token}")
    fields(cp, {"runner_version": RUNNER_VERSION, "mode": "VALIDATION", "status": "PASS",
                "runner_sha256": EXPECTED[RUNNER], "completed_batches": BATCH_IDS}, "checkpoint")
    require(e.link(cp["authorization"], AUTH) == e.records[AUTH], "checkpoint authorization binding")
    require(isinstance(cp["batches"], dict) and set(cp["batches"]) == {f"{i:03d}" for i in BATCH_IDS}, "checkpoint batch keys")
    actual_dirs = {p.name for p in e.path(RAW).iterdir() if p.is_dir() and p.name.startswith("batch_")}
    require(actual_dirs == {f"batch_{i:03d}" for i in BATCH_IDS}, "raw batch-directory coverage")
    verify_resource_log(e.path(ATTEMPT + "resources.log"))
    return auth, previous, receipt, cp, master


def hex256(value, label):
    if type(value) is int:
        require(0 <= value < 2**256, f"{label}: value outside 256 bits")
        value = f"{value:064x}"
    require(isinstance(value, str), f"{label}: hex string required")
    text = value.strip().lower().removeprefix("0x")
    require(HEX256.fullmatch(text) is not None, f"{label}: 64 hex characters required")
    return text


def validation_vectors(e, auth, previous, cp):
    pool, cm, arch, contract = (e.load(p) for p in (POOL, COMMITMENTS, ARCH, CONTRACT))
    records = pool["vectors"]
    require(isinstance(records, list) and len(records) == 256, "vector-pool count")
    require(all(type(v["vector_id"]) is int for v in records), "vector IDs must be integers")
    require(len({v["vector_id"] for v in records}) == 256, "duplicate vector ID")
    require(Counter(v["split"] for v in records) == {"TRAIN": 160, "VALIDATION": 48, "HOLDOUT_TEST": 48}, "vector-pool splits")
    selected = sorted((v for v in records if v["split"] == "VALIDATION"), key=lambda v: v["vector_id"])
    fields(cm, {"status": "PASS", "manifest_status": "FROZEN", "validation_vector_count": 48,
                "validation_vector_payloads_emitted": 0, "holdout_vector_records_emitted": 0}, "vector commitment manifest")
    require(e.link(cm["source_vector_pool"], POOL) == e.records[POOL], "commitment/vector-pool binding")
    # The historical payload-emission count describes Stage 3A, not this later campaign.
    require(len(cm["records"]) == 48, "validation commitment record count")
    commitment, result = hashlib.sha256(), []
    for slot, (v, sealed) in enumerate(zip(selected, cm["records"], strict=True)):
        vid = v["vector_id"]
        fields(sealed, {"validation_order": slot, "vector_id": vid,
                        "pair_fingerprint": v["pair_fingerprint"], "vector_commitment": v["vector_commitment"]}, "committed vector")
        require(HEX256.fullmatch(v["pair_fingerprint"]) and HEX256.fullmatch(v["vector_commitment"]), "invalid vector commitment hash")
        commitment.update(f"{slot}|{vid}|{v['pair_fingerprint']}|{v['vector_commitment']}\n".encode())
        # Access key/message/digest payloads for these 48 VALIDATION records only.
        expected = hex256(v["expected_hmac_sha256"], f"validation vector {vid} digest")
        key = bytes.fromhex(hex256(v["key"], f"validation vector {vid} key"))
        message = bytes.fromhex(hex256(v["message"], f"validation vector {vid} message"))
        require(hmac.new(key, message, hashlib.sha256).hexdigest() == expected, f"validation vector {vid}: HMAC reference mismatch")
        result.append({"vector_slot": slot, "vector_id": vid, "expected_digest": expected})
    value = commitment.hexdigest()
    for document in (cm, arch, auth, previous, cp):
        require(document["ordered_validation_commitment"] == value, "ordered validation commitment mismatch")
    require(arch["acceptance_contract"] == contract["acceptance_contract"] == auth["acceptance_contract"] == previous["acceptance_contract"], "frozen evaluation acceptance rules differ")
    print("  Committed VALIDATION vectors: 48/48; Python HMAC references: 48/48 PASS", flush=True)
    return result, value, arch, contract


def scan_csv(path, batch_id, sites, vectors, expected_sha):
    """Exhaustive Cartesian coverage using <=49,152 one-byte presence entries."""
    nsites, nvec = len(sites), len(vectors)
    expected_enabled = nsites * 2 * nvec
    seen = bytearray(expected_enabled)
    baseline_seen = bytearray(nvec)
    activity_counts, detection_counts = [0] * (nsites * 2), [0] * (nsites * 2)
    stats = {key: 0 for key in STAT_FIELDS}
    stats["fault_instances"] = nsites * 2
    per_vector = [{"vector_slot": slot, "vector_id": v["vector_id"], "baseline_records": 0,
                   "enabled_records": 0, "activated_records": 0, "detected_records": 0, "timeout_records": 0}
                  for slot, v in enumerate(vectors)]
    digest = hashlib.sha256()
    numeric = (0, 3, 4, 6, 7, 8, 10, 11)
    with path.open("rb") as source:
        def lines():
            for raw in source:
                digest.update(raw)
                yield raw.decode("utf-8")
        reader = csv.reader(lines(), strict=True)
        require(next(reader, None) == CSV_HEADER, f"Batch {batch_id:03d}: CSV header differs from frozen runner")
        for row in reader:
            location = f"Batch {batch_id:03d}, CSV line {reader.line_num}"
            require(len(row) == 18, f"{location}: expected 18 fields, got {len(row)}")
            try:
                values = [int(row[i]) for i in numeric]
            except ValueError:
                raise VerificationError(f"{location}: invalid integer field")
            require(all(str(v) == row[i] for i, v in zip(numeric, values)), f"{location}: noncanonical integer encoding")
            bid, selection, selector, stuck, slot, vector_id, cycles, baseline_cycles = values
            require(bid == batch_id and row[1] == "VALIDATION", f"{location}: wrong batch or non-VALIDATION record")
            require(0 <= slot < nvec and vector_id == vectors[slot]["vector_id"], f"{location}: vector identity mismatch")
            require(all(row[i] in ("0", "1") for i in (9, 12, 13, 14, 15)), f"{location}: nonbinary flag")
            enabled, timed_out, active, detected, unknown = (int(row[i]) for i in (9, 12, 13, 14, 15))
            require(unknown == 0, f"{location}: recorded unknown signal")
            require(baseline_cycles == BASELINE_CYCLES and 0 <= cycles <= TIMEOUT_CYCLES, f"{location}: invalid cycle count")
            require(not timed_out or cycles == TIMEOUT_CYCLES, f"{location}: timeout must occur at 2000 cycles")
            expected, actual = row[16].lower(), row[17].lower()
            require(HEX256.fullmatch(expected) and HEX256.fullmatch(actual), f"{location}: malformed digest")
            require(expected == vectors[slot]["expected_digest"], f"{location}: expected digest differs from validation reference")
            if row[2] == "BASELINE":
                require((selection, selector, stuck, row[5]) == (-1, -1, -1, "BASELINE"), f"{location}: baseline identity")
                require(enabled == active == detected == timed_out == 0, f"{location}: baseline flags")
                require(cycles == BASELINE_CYCLES and actual == expected, f"{location}: baseline digest or latency failure")
                require(not baseline_seen[slot], f"{location}: duplicate baseline vector")
                baseline_seen[slot] = 1
                stats["baseline_records"] += 1
                per_vector[slot]["baseline_records"] += 1
            elif row[2] == "ENABLED":
                require(enabled == 1 and 0 <= selection < nsites and stuck in (0, 1), f"{location}: invalid enabled fault identity")
                site = sites[selection]
                require(selector == site["selector_code"] and row[5] == site["fault_site_id"], f"{location}: selector/site mapping mismatch")
                index = selection * 2 + stuck
                case = index * nvec + slot
                require(not seen[case], f"{location}: duplicate site/stuck/vector combination")
                seen[case] = 1
                digest_error, latency_error = int(actual != expected), int(cycles != baseline_cycles)
                recomputed = int(bool(timed_out or digest_error or latency_error))
                require(detected == recomputed, f"{location}: detection label disagrees with digest/latency/timeout")
                require(not detected or active == 1, f"{location}: detected without activity")
                stats["enabled_records"] += 1
                stats["activated_records"] += active
                stats["detected_records"] += detected
                stats["timeout_records"] += timed_out
                stats["digest_mismatch_records"] += digest_error
                stats["latency_mismatch_records"] += latency_error
                activity_counts[index] += active
                detection_counts[index] += detected
                per_vector[slot]["enabled_records"] += 1
                per_vector[slot]["activated_records"] += active
                per_vector[slot]["detected_records"] += detected
                per_vector[slot]["timeout_records"] += timed_out
            else:
                raise VerificationError(f"{location}: unsupported run_type {row[2]!r}")
    require(digest.hexdigest() == expected_sha, f"Batch {batch_id:03d}: CSV stream SHA changed or differs from checkpoint")
    require(stats["baseline_records"] == nvec and all(baseline_seen), f"Batch {batch_id:03d}: missing baseline vector")
    require(stats["enabled_records"] == expected_enabled and all(seen), f"Batch {batch_id:03d}: missing site/stuck/vector combinations")
    for vector in per_vector:
        fields(vector, {"baseline_records": 1, "enabled_records": nsites * 2}, "per-vector coverage")
    for active, detected in zip(activity_counts, detection_counts):
        kind = "detected_instances" if detected else "activated_unobserved_instances" if active else "unactivated_instances"
        stats[kind] += 1
    stats["total_records"] = stats["baseline_records"] + stats["enabled_records"]
    require(stats["detected_instances"] + stats["activated_unobserved_instances"] + stats["unactivated_instances"] == nsites * 2, "classification closure")
    require(stats["detected_records"] <= stats["activated_records"] <= expected_enabled, "activation/detection count consistency")
    # Reason counts may overlap: a timeout can also change digest and latency.
    return stats, per_vector


def verify_resource_log(path):
    text = path.read_text()
    matches = re.findall(r"^\s*Exit status:\s*(\d+)\s*$", text, re.MULTILINE)
    require(matches == ["0"], f"resource log lacks unique successful exit: {path.name}")


def verify_simulation_log(path, batch_id, stats):
    text = path.read_text()
    require(text.splitlines().count("VALIDATION_CAMPAIGN_BATCH_RESULT=PASS") == 1, f"Batch {batch_id:03d}: simulation PASS token")
    required = {"BATCH": batch_id, "BASELINE_RUNS": stats["baseline_records"], "ENABLED_RUNS": stats["enabled_records"],
                "TOTAL_RECORDS": stats["total_records"], "FAULT_INSTANCES": stats["fault_instances"],
                "ACTIVATED_RUNS": stats["activated_records"], "DETECTED_RUNS": stats["detected_records"],
                "DETECTED_INSTANCES": stats["detected_instances"], "ACTIVATED_UNOBSERVED": stats["activated_unobserved_instances"],
                "UNACTIVATED": stats["unactivated_instances"], "UNCLASSIFIED": 0, "UNKNOWN_RUNS": 0,
                "DETECTED_WITHOUT_ACTIVITY": 0, "BASELINE_DIGEST_MISMATCHES": 0,
                "BASELINE_LATENCY_MISMATCHES": 0, "BASELINE_TIMEOUTS": 0}
    for name, value in required.items():
        matches = re.findall(r"^VALIDATION_CAMPAIGN_BATCH_" + name + r"=(\d+)\s*$", text, re.MULTILINE)
        require(matches == [str(value)], f"Batch {batch_id:03d}: log/CSV mismatch for {name}: expected {value}, got {matches}")


def batch_plan(batch_id):
    size = 311 if batch_id == 44 else 512
    return {"batch_id": batch_id, "batch_key": f"{batch_id:03d}", "legal_sites": size,
            "fault_instances": size * 2, "validation_vectors": 48, "baseline_records": 48,
            "enabled_records": size * 96, "total_records": size * 96 + 48}


def verify_batch(e, bid, cp_entry, semantic_entry, vectors, commitment, master):
    key, plan = f"{bid:03d}", batch_plan(bid)
    fields(cp_entry, {"status": "PASS", "batch_id": bid}, f"Batch {key} checkpoint entry")
    canonical = f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}"
    mapping_path = f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
    batch_root, build_root = RAW + f"batch_{key}/", BUILD + f"batch_{key}/"
    expected_paths = {"netlist": canonical + ".v", "mapping": mapping_path,
                      "testbench": build_root + f"tb_hmac_validation_batch_{key}.sv",
                      "manifest": batch_root + "testbench_manifest.json",
                      "binary": build_root + f"obj_dir/Vtb_hmac_validation_batch_{key}",
                      "csv": batch_root + f"hmac_validation_batch_{key}_results.csv",
                      "simulation_log": batch_root + "simulation.log", "resources_log": batch_root + "resources.log"}
    artifacts = {name: e.link(cp_entry[name], relative) for name, relative in expected_paths.items()}
    e.verify(canonical + ".v", semantic_entry["verilog_sha256"])
    e.verify(canonical + ".json", semantic_entry["json_sha256"])
    e.verify(mapping_path, semantic_entry["mapping_sha256"])
    csv_files = {p.name for p in e.path(batch_root).glob("*.csv")}
    require(csv_files == {Path(expected_paths["csv"]).name}, f"Batch {key}: unexpected CSV files; freeze scope must be unambiguous")
    sites = e.load(mapping_path)["sites"]
    require(isinstance(sites, list) and len(sites) == plan["legal_sites"], f"Batch {key}: mapping site count")
    for selector, site in enumerate(sites):
        global_index = bid * 512 + selector + 1
        fields(site, {"selector_code": selector, "site_index": global_index,
                      "fault_site_id": f"HMAC-STEM-{global_index:06d}"}, f"Batch {key} mapped site")
    info = e.load(expected_paths["manifest"])
    fields(info, {"runner_version": RUNNER_VERSION, "status": "GENERATED", "mode": "VALIDATION",
                  "batch_id": bid, "vector_partition": "VALIDATION", "vectors": 48,
                  "vector_positions": list(range(48)), "vector_ids": [v["vector_id"] for v in vectors],
                  "sites": len(sites), "selectors": list(range(len(sites))),
                  "site_ids": [s["fault_site_id"] for s in sites], "fault_instances": plan["fault_instances"],
                  "baseline_records": 48, "enabled_records": plan["enabled_records"], "total_records": plan["total_records"],
                  "ordered_validation_commitment": commitment, "validation_vectors_executed": 48,
                  "holdout_vectors_exposed": 0, "frozen_rtl_modified": False, "golden_netlist_modified": False,
                  "result_csv_path": expected_paths["csv"]}, f"Batch {key} testbench manifest")
    require(e.link(info["testbench"]) == artifacts["testbench"], f"Batch {key}: testbench identity")
    stats, per_vector = scan_csv(e.path(expected_paths["csv"]), bid, sites, vectors, artifacts["csv"]["sha256"])
    expected_summary = {name: stats[name] for name in ("baseline_records", "enabled_records", "total_records",
        "activated_records", "detected_records", "unknown_records", "detected_without_activity", "baseline_failures")}
    expected_summary.update(selectors=list(range(len(sites))), vector_slots=list(range(48)))
    fields(cp_entry["summary"], expected_summary, f"Batch {key} checkpoint/CSV summary")
    verify_simulation_log(e.path(expected_paths["simulation_log"]), bid, stats)
    verify_resource_log(e.path(expected_paths["resources_log"]))
    line = f"Batch {key}: PASS records={stats['total_records']} csv_sha256={artifacts['csv']['sha256']}"
    require(master.splitlines().count(line) == 1, f"Batch {key}: master log/CSV binding")
    return {**plan, "status": "PASS", "artifacts": artifacts, "summary": stats, "per_vector": per_vector,
            "first_site": sites[0]["fault_site_id"], "last_site": sites[-1]["fault_site_id"]}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish_once(path, payload):
    require(not path.exists() and not path.is_symlink(), f"refusing to replace frozen output: {path}")
    temp = path.with_name(path.name + ".pending")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()  # Remove only the hard link created by this function.


def run(root, check_only=False):
    e = Evidence(root)
    source = Path(__file__).resolve()
    require(e.root.name == "vlsi_fault_detection_v2" and source.parent == e.root,
            "copy this script to ~/vlsi_fault_detection_v2 and run it there")
    print("STAGE 11D-3E — VALIDATION DATASET INTEGRITY", flush=True)
    if not check_only:
        for relative in (MANIFEST, AUDIT):
            for suffix in ("", ".pending"):
                p = e.path(relative + suffix)
                require(not p.exists() and not (e.root / (relative + suffix)).is_symlink(),
                        f"output already exists: {relative + suffix}; preserve it; --check-only can verify without replacing it")
    with campaign_lock(e):
        source_record = e.verify(source.name, sha256(source))
        auth, previous, receipt, cp, master = verify_inputs(e)
        vectors, commitment, arch, contract = validation_vectors(e, auth, previous, cp)
        semantic = e.load(SEMANTIC)
        fields(semantic, {"status": "PASS", "batches": 45, "legal_sites": 22839, "persistent_fault_instances": 45678,
                          "mapping_mismatches": 0, "routing_mismatches": 0, "raw_monitor_mismatches": 0,
                          "golden_restoration_mismatches": 0, "site_gaps": 0, "duplicate_sites": 0}, "semantic manifest")
        entries = semantic["batch_qualification"]
        require(len(entries) == 45 and all(type(x["batch_id"]) is int for x in entries)
                and sorted(x["batch_id"] for x in entries) == BATCH_IDS, "semantic batch coverage")
        entries = {x["batch_id"]: x for x in entries}
        plan = [batch_plan(i) for i in BATCH_IDS]
        require(plan == auth["expected_batch_records"] == arch["batch_plan"] == contract["expected_batch_records"], "batch execution plans differ")
        total = Counter({k: 0 for k in STAT_FIELDS})
        batches, sites_seen = [], set()
        ordered = hashlib.sha256()
        print("\nSTREAMING ALL 45 VALIDATION CSV FILES", flush=True)
        for bid in BATCH_IDS:
            batch = verify_batch(e, bid, cp["batches"][f"{bid:03d}"], entries[bid], vectors, commitment, master)
            for index in range(bid * 512 + 1, bid * 512 + batch["legal_sites"] + 1):
                require(index not in sites_seen, "duplicate physical site across batches")
                sites_seen.add(index)
            batches.append(batch)
            total.update(batch["summary"])
            rec = batch["artifacts"]["csv"]
            ordered.update(f"{bid:03d}|{rec['path']}|{rec['sha256']}|{batch['total_records']}\n".encode())
            s = batch["summary"]
            print(f"  Batch {bid:03d}: PASS rows={s['total_records']} activated={s['activated_records']} detected={s['detected_records']} ({total['total_records']}/2194704)", flush=True)
        require(sites_seen == set(range(1, 22840)), "global legal-site coverage")
        fields(dict(total), EXPECTED_TOTALS, "global dataset counts")
        require(total["detected_instances"] + total["activated_unobserved_instances"] + total["unactivated_instances"] == 45678, "global fault-instance classifications")
        global_vectors = []
        for slot, v in enumerate(vectors):
            aggregate = {k: sum(b["per_vector"][slot][k] for b in batches)
                         for k in ("baseline_records", "enabled_records", "activated_records", "detected_records", "timeout_records")}
            fields(aggregate, {"baseline_records": 45, "enabled_records": 45678}, "global validation-vector coverage")
            global_vectors.append({"vector_slot": slot, "vector_id": v["vector_id"], **aggregate})
        dataset_sha = ordered.hexdigest()
        print("\nRECHECKING FROZEN INPUTS AND VERIFIED DATASET BYTES", flush=True)
        e.recheck()
        if check_only:
            print(f"CHECK_ONLY_RESULT=PASS; verified rows={total['total_records']}; no freeze artifacts written")
            return 0
        created = datetime.now(timezone.utc).isoformat()
        common = {"stage": "11D-3E", "status": "PASS", "dataset_status": "FROZEN", "created_at_utc": created,
                  "verifier_version": VERSION, "verifier_source": source_record, "vector_partition": "VALIDATION",
                  "fault_scope": "PERSISTENT_NET_STEM_SA0_SA1_ONLY", "batches_verified": 45,
                  "legal_sites": 22839, "persistent_fault_instances": 45678, "validation_vectors": 48,
                  "ordered_validation_commitment": commitment, "ordered_dataset_commitment": dataset_sha,
                  "summary": dict(total), "enabled_positive_labels": total["detected_records"],
                  "enabled_negative_labels": total["enabled_records"] - total["detected_records"],
                  "raw_csv_bytes": sum(b["artifacts"]["csv"]["bytes"] for b in batches),
                  "python_validation_reference_checks": 48, "raw_campaign_dataset_modified": False,
                  "frozen_input_artifacts_modified": False, "frozen_rtl_modified": False, "golden_netlist_modified": False,
                  "model_inference_performed": False, "training_performed": False, "model_deserialization_performed": False,
                  "holdout_vectors_executed": 0, "holdout_payloads_emitted": 0,
                  "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED",
                  "validation_model_inference": "NOT_AUTHORIZED", "holdout_campaign": "NOT_AUTHORIZED",
                  "canonical_dataset_consolidation_performed": False, "next_gate": NEXT_GATE}
        manifest = {**common, "title": "VALIDATION DATASET MANIFEST", "manifest_status": "FROZEN",
                    "raw_dataset_root": RAW.rstrip("/"), "csv_header": CSV_HEADER,
                    "sample_identity": ["batch_id", "site_id", "stuck_value", "vector_id"],
                    "baseline_identity": ["batch_id", "vector_id"],
                    "label_definition": "detected = timed_out OR actual_digest != expected_digest OR cycles != baseline_cycles",
                    "classification_scope": "per physical site and SA value, across these 48 VALIDATION vectors only",
                    "unknown_check_scope": "flags recorded at transaction endpoints by the frozen simulator testbench",
                    "detection_reason_counts_overlap": True, "baseline_cycles": 343, "timeout_cycles": 2000,
                    "dataset_commitment_algorithm": "SHA256 of UTF-8 lines: batch_id as 3 digits|project-relative CSV path|CSV SHA256|data records\\n, batches 0..44",
                    "checkpoint": e.records[CHECKPOINT], "completion_receipt": e.records[RECEIPT],
                    "execution_authorization": e.records[AUTH], "validation_commitments": e.records[COMMITMENTS],
                    "validation_vector_ids": [v["vector_id"] for v in vectors], "per_vector": global_vectors,
                    "batch_records": batches, "acceptance_contract": auth["acceptance_contract"]}
        payload = encoded(manifest)
        manifest_record = {"path": MANIFEST, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        audit = {**common, "title": "VALIDATION DATASET INTEGRITY AND MANIFEST FREEZE",
                 "audit_status": "FROZEN", "manifest": manifest_record,
                 "checkpoint": e.records[CHECKPOINT], "completion_receipt": e.records[RECEIPT],
                 "input_evidence": e.records,
                 "checks": {"source_hashes": "PASS", "receipt_checkpoint_binding": "PASS", "baseline_reference": "PASS",
                            "exact_sample_coverage": "PASS", "selector_site_mapping": "PASS", "label_recomputation": "PASS",
                            "checkpoint_log_csv_agreement": "PASS", "vector_partition_membership": "PASS",
                            "physical_site_coverage": "PASS", "final_byte_recheck": "PASS"},
                 "completion_rule": "BOTH_AUDIT_AND_MATCHING_MANIFEST_MUST_EXIST"}
        publish_once(e.path(MANIFEST), payload)
        publish_once(e.path(AUDIT), encoded(audit))
        require(sha256(e.path(MANIFEST)) == manifest_record["sha256"], "manifest readback SHA mismatch")
        print("\nSTAGE 11D-3E — VALIDATION DATASET INTEGRITY AND MANIFEST FREEZE")
        for label, value in (
            ("Status", "PASS"), ("Dataset status", "FROZEN"), ("Batches verified", "45/45"),
            ("Legal sites", "22839/22839"), ("Persistent fault instances", "45678/45678"),
            ("VALIDATION vectors", 48), ("TRAIN/HOLDOUT records", "0/0"),
            ("Baseline records", total["baseline_records"]), ("Enabled records", total["enabled_records"]),
            ("Total CSV records", total["total_records"]), ("Activated records", total["activated_records"]),
            ("Detected records", total["detected_records"]), ("Detected instances", total["detected_instances"]),
            ("Activated-unobserved", total["activated_unobserved_instances"]), ("Unactivated", total["unactivated_instances"]),
            ("Enabled timeout records", total["timeout_records"]), ("Unclassified instances", 0),
            ("Missing/duplicate samples", "0/0"), ("Unknown records", 0), ("Detected without activity", 0),
            ("Baseline failures", 0), ("Checkpoint/log/CSV agreement", "PASS"),
            ("Frozen inputs modified", "NO"), ("Raw campaign dataset modified", "NO"),
            ("Validation model inference", "NOT AUTHORIZED"), ("Holdout campaign", "NOT AUTHORIZED"),
            ("Checkpoint SHA", EXPECTED[CHECKPOINT]), ("Ordered dataset commitment", dataset_sha),
            ("Manifest", e.path(MANIFEST)), ("Manifest SHA", manifest_record["sha256"]),
            ("Audit", e.path(AUDIT)), ("Audit SHA", sha256(e.path(AUDIT))), ("Next gate", NEXT_GATE),
        ):
            print(f"{label:31}: {value}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true", help="verify without publishing manifest or audit")
    args = parser.parse_args()
    try:
        require(sys.version_info >= (3, 10), "Python 3.10+ required")
        return run(args.project_root, args.check_only)
    except KeyboardInterrupt:
        print("\nSTOP: verification interrupted; no PASS. Preserve existing inputs and any partial outputs.", file=sys.stderr)
        return 130
    except (VerificationError, OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"STOP: {error}", file=sys.stderr)
        print("Stage 11D-3E did not pass. Do not consolidate data or run model inference.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
