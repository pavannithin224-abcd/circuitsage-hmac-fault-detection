#!/usr/bin/env python3
"""Launch the frozen validation simulator in WSL; never train or infer.

--start: verify all Stage 11D-3C evidence, refuse existing campaign outputs,
         detach a supervised run, and preserve logs in a unique attempt folder.
--status: read launch/checkpoint state without starting anything.
--resume: explicitly resume an interrupted campaign with a verified checkpoint.
          The frozen runner preserves completed batches but rewrites incomplete
          batch outputs. No automatic resume, cleanup, overwrite, or retry.
--check-only: perform fresh-start preflight without launching a campaign.

Requires Linux/WSL, Python 3.10+, the existing venv, and OSS-CAD-Suite in PATH.
This launcher is separate from, and never edits, the frozen runner.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile


VERSION = "HMAC-VALIDATION-LAUNCHER-v1"
RUNNER_VERSION = "HMAC-VALIDATION-CAMPAIGN-RUNNER-v1"
RUNNER = "stage_11d3b_validation_runner.py"
RESULTS = "results/hmac_fault_campaign_11d3"
CAMPAIGN = "results/hmac_validation_campaign_full_11d3"
BUILD = "build/hmac_validation_campaign_full_11d3"
CHECKPOINT = CAMPAIGN + "/hmac_validation_campaign_checkpoint.json"
AUTH = "config/fault_campaign/hmac_validation_campaign_execution_authorization_11d3c.json"
AUDIT = RESULTS + "/hmac_validation_campaign_execution_authorization_freeze_11d3c.json"
LOCK = RESULTS + "/hmac_validation_campaign_11d3d.lock"
LATEST = RESULTS + "/hmac_validation_campaign_11d3d_latest.json"
PINNED = {
    RUNNER: "71280fdcbe07b66f10f3602630d252b8e92b1816172f7b93689eba95b717a777",
    AUTH: "c1195366d67ac2c72097903db7fb9f1ab610877e277d2292d79d0c1d9179c247",
    AUDIT: "af3c7309df4d4abf547ea2b245a999a552400b3a8df545d5b296a734e6a87875",
    "stage_11d3c_validation_authorization.py":
        "da975aa3848edc501588ec461262a9613915c912f2976e5f77e3e267b47bd874",
    "config/fault_campaign/hmac_validation_campaign_runner_lock_11d3b.json":
        "16906b81b29b7fac0196067892034fd3a7ded4e1f2ca63a13b66b5bcf6af0c39",
    "config/fault_campaign/hmac_validation_campaign_execution_contract_11d3a.json":
        "55c3b2c57c8a6a7068812a41a75b01f75292ce0068c4553e3fd35fb972a87420",
}


class Stop(Exception):
    pass


def require(ok, message):
    if not ok:
        raise Stop(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def local(root, relative):
    require(isinstance(relative, str) and relative, "missing relative artifact path")
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, f"unsafe artifact path: {relative}")
    resolved = (root / p).resolve()
    require(resolved.is_relative_to(root), f"artifact escapes project: {relative}")
    return resolved


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=unique)
    require(isinstance(result, dict), f"JSON object required: {path}")
    return result


def fields(document, wanted, label):
    require(isinstance(document, dict), f"{label}: object required")
    for key, value in wanted.items():
        require(type(document.get(key)) is type(value) and document[key] == value,
                f"{label}/{key}: expected {value!r}, got {document.get(key)!r}")


def verify(root, relative, expected, size=None):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"malformed expected SHA: {relative}")
    path = local(root, relative)
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty file: {relative}")
    actual = sha(path)
    require(actual == expected, f"SHA mismatch: {relative}\nExpected: {expected}\nActual:   {actual}")
    if size is not None:
        require(type(size) is int and path.stat().st_size == size, f"size mismatch: {relative}")


def record(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}


def write_once(path, data):
    require(not path.exists() and not path.is_symlink(), f"refusing to replace evidence: {path}")
    fd, name = tempfile.mkstemp(prefix="." + path.name + "_", suffix=".pending", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(name, path)  # Publish the complete record; never replace an old one.
    Path(name).unlink()  # Only the temporary hard link created by this call.


def write_latest(root, data):
    # Only this launcher's mutable navigation pointer is replaced, never evidence.
    path = local(root, LATEST)
    require(not (root / LATEST).is_symlink(), "latest pointer is a symlink")
    fd, name = tempfile.mkstemp(prefix=".stage_11d3d_latest_", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def check_no_direct_runner(root):
    # The file lock covers this launcher. Also catch a direct runner launched
    # outside it under this user, using /proc arguments and project cwd.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = [os.fsdecode(x) for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
            if "--execute-validation" not in args or not any(Path(a).name == RUNNER for a in args):
                continue
            same = (entry / "cwd").resolve() == root
            if "--project-root" in args:
                i = args.index("--project-root")
                same = same or Path(args[i + 1]).resolve() == root
            require(not same, f"validation runner already active (PID {entry.name}); use --status")
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue


def verify_checkpoint(root, auth, require_incomplete=True):
    path = local(root, CHECKPOINT)
    require(path.is_file(), "resume checkpoint missing; do not delete or overwrite partial outputs; request review")
    cp = read_json(path)
    fields(cp, {"runner_version": RUNNER_VERSION, "mode": "VALIDATION",
                "runner_sha256": PINNED[RUNNER],
                "ordered_validation_commitment": auth["ordered_validation_commitment"]}, "checkpoint")
    fields(cp["authorization"], {"path": AUTH, "sha256": PINNED[AUTH]}, "checkpoint authorization")
    done = cp["completed_batches"]
    require(isinstance(done, list) and all(type(n) is int for n in done)
            and done == list(range(len(done))) and 0 < len(done) <= 45,
            "checkpoint must contain a nonempty sequential prefix of batches 0-44")
    require(set(cp["batches"]) == {f"{n:03d}" for n in done}, "checkpoint batch entries disagree")
    if require_incomplete:
        require(cp.get("status") != "PASS" and len(done) < 45,
                "all batches are recorded or the campaign is complete; do not rerun; request closure review")
    for batch in done:
        entry = cp["batches"][f"{batch:03d}"]
        fields(entry, {"status": "PASS", "batch_id": batch}, "completed batch")
        total = (311 if batch == 44 else 512) * 2 * 48
        fields(entry["summary"], {"baseline_records": 48, "enabled_records": total,
            "total_records": total + 48, "unknown_records": 0, "detected_without_activity": 0,
            "baseline_failures": 0}, "batch summary")
        require(entry["csv"]["path"] == f"{CAMPAIGN}/batch_{batch:03d}/hmac_validation_batch_{batch:03d}_results.csv",
                "checkpoint CSV path")
        for key in ("csv", "netlist", "mapping", "testbench", "manifest", "binary", "simulation_log", "resources_log"):
            rec = entry[key]
            verify(root, rec["path"], rec["sha256"], rec["bytes"])
    return cp


def preflight(root, resume):
    print("STAGE 11D-3D — VALIDATION CAMPAIGN LAUNCH PREFLIGHT", flush=True)
    for relative, expected in PINNED.items():
        verify(root, relative, expected)
        print(f"  {Path(relative).name}: OK", flush=True)
    auth, audit = read_json(local(root, AUTH)), read_json(local(root, AUDIT))
    scope = {"validation_campaign_execution": "AUTHORIZED", "validation_model_inference": "NOT_AUTHORIZED",
             "holdout_campaign": "NOT_AUTHORIZED", "model_retraining": "PROHIBITED", "threshold_changes": "PROHIBITED",
             "validation_vectors": 48, "batches": 45, "legal_sites": 22839,
             "persistent_fault_instances": 45678, "baseline_records": 2160,
             "enabled_records": 2192544, "total_records": 2194704,
             "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
             "checkpoint_after_each_batch": True, "resume_supported": True}
    fields(auth, {"stage": "11D-3C", "status": "FROZEN", "overwrite_authorized": False, **scope}, "authorization")
    fields(audit, {"stage": "11D-3C", "status": "PASS", "authorization_status": "FROZEN", **scope}, "authorization audit")
    fields(audit["authorization"], {"path": AUTH, "sha256": PINNED[AUTH]}, "audit authorization link")
    verify(root, AUTH, audit["authorization"]["sha256"], audit["authorization"]["bytes"])
    evidence = audit["input_evidence"]
    require(isinstance(evidence, dict) and RUNNER in evidence, "Stage 11D-3C evidence index missing")
    print(f"  Rechecking {len(evidence)} frozen evidence files (including all batch artifacts)", flush=True)
    for i, (relative, rec) in enumerate(evidence.items(), 1):
        require(rec["path"] == relative, "evidence key/path mismatch")
        verify(root, relative, rec["sha256"], rec["bytes"])
        if i % 50 == 0:
            print(f"  Evidence verified: {i}/{len(evidence)}", flush=True)
    for tool in ("verilator", "make", "g++"):
        require(shutil.which(tool) is not None, f"{tool} missing from PATH; activate OSS-CAD-Suite")
    require(os.access("/usr/bin/time", os.X_OK), "missing executable /usr/bin/time")
    free = shutil.disk_usage(root).free
    require(free >= 5 * 1024**3, "less than 5 GiB free disk")
    check_no_direct_runner(root)
    if resume:
        verify_checkpoint(root, auth)
    else:
        for relative in (CAMPAIGN, BUILD):
            require(not local(root, relative).exists() and not (root / relative).is_symlink(),
                    f"campaign path already exists: {relative}; do not overwrite; use --status and request resume guidance")
    print(f"  Free disk: {free / 1024**3:.2f} GiB\nLAUNCH_PREFLIGHT=PASS", flush=True)
    return auth, evidence


def command(root, attempt, resume):
    args = ["/usr/bin/time", "-v", "-o", str(attempt / "resources.log"),
            sys.executable, "-u", str(local(root, RUNNER)), "--project-root", str(root),
            "--execute-validation", "--authorization", str(local(root, AUTH))]
    if resume:
        args.append("--resume")
    return args


def worker(root, attempt, cmd, auth, evidence):
    result = {"stage": "11D-3D", "launcher_version": VERSION, "finished_at_utc": None,
              "status": "FAILED", "runner_return_code": None, "integrity_errors": [],
              "dataset_integrity_freeze": "NOT_PERFORMED", "model_inference": "NOT_PERFORMED"}
    try:
        print("STAGE 11D-3D — VALIDATION CAMPAIGN EXECUTION", flush=True)
        completed = subprocess.run(cmd, cwd=root, check=False, stdin=subprocess.DEVNULL)
        result["runner_return_code"] = completed.returncode
        print(f"Validation runner return code: {completed.returncode}", flush=True)
        # Byte-only recheck: never deserialize a model or run inference.
        for relative, rec in evidence.items():
            try:
                verify(root, relative, rec["sha256"], rec["bytes"])
            except (Stop, OSError) as error:
                result["integrity_errors"].append(str(error))
        for relative, expected in PINNED.items():
            verify(root, relative, expected)
        require(completed.returncode == 0, "runner did not exit successfully; preserve partial results")
        require(not result["integrity_errors"], "frozen input integrity changed; stop for review")
        cp = verify_checkpoint(root, auth, require_incomplete=False)
        fields(cp, {"status": "PASS", "completed_batches": list(range(45))}, "campaign closure")
        result.update(status="PASS", completed_batches=45, baseline_records=2160,
                      enabled_records=2192544, total_records=2194704,
                      checkpoint=record(root, local(root, CHECKPOINT)), frozen_inputs_unchanged=True)
    except Exception as error:
        result["error"] = str(error)
        print(f"STOP: {error}", flush=True)
    result["finished_at_utc"] = now()
    resource = attempt / "resources.log"
    if resource.is_file():
        result["resources"] = record(root, resource)
    print(f"STAGE_11D3D_EXECUTION_RESULT={result['status']}", flush=True)
    print("Model inference and HOLDOUT execution remain blocked.", flush=True)
    print("Next: validation dataset integrity verification; do not retrain or change thresholds.", flush=True)
    # Last log write above, then hash it. A completion receipt is execution
    # evidence only, not a dataset or inference freeze.
    result["master_log"] = record(root, attempt / "master.log")
    write_once(attempt / "completion.json", result)
    return 0 if result["status"] == "PASS" else 1


def acquire_lock(root):
    path = local(root, LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise Stop("a validation launch or campaign already holds the lock; use --status")
    return stream


def launch(root, resume, check_only):
    with acquire_lock(root) as held_lock:
        auth, evidence = preflight(root, resume)
        if check_only:
            print("CHECK_ONLY=PASS; campaign NOT STARTED")
            return 0
        attempts = local(root, RESULTS + "/stage_11d3d_attempts")
        attempts.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"), dir=attempts))
        cmd = command(root, attempt, resume)
        launch_record = {"stage": "11D-3D", "status": "LAUNCH_REQUESTED", "created_at_utc": now(),
                         "launcher": record(root, Path(__file__).resolve()), "launcher_version": VERSION,
                         "command": cmd, "resume": resume, "authorization": record(root, local(root, AUTH)),
                         "authorization_audit": record(root, local(root, AUDIT)),
                         "python_executable": sys.executable, "runner_sha256": PINNED[RUNNER]}
        if resume:
            launch_record["checkpoint_before_resume"] = record(root, local(root, CHECKPOINT))
        write_once(attempt / "launch.json", launch_record)
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        # The child inherits the same flock open-file description. The parent
        # closing its copy does not release the lock while the child is alive.
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.setsid()
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
                with open(os.devnull, "rb") as null, (attempt / "master.log").open("xb", buffering=0) as log:
                    os.dup2(null.fileno(), 0)
                    os.dup2(log.fileno(), 1)
                    os.dup2(log.fileno(), 2)
                sys.stdout.reconfigure(line_buffering=True)
                sys.stderr.reconfigure(line_buffering=True)
                write_once(attempt / "supervisor.json", {"pid": os.getpid(), "started_at_utc": now()})
                code = worker(root, attempt, cmd, auth, evidence)
            except BaseException as error:
                print(f"SUPERVISOR_STOP: {error}", file=sys.stderr, flush=True)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                held_lock.close()
                os._exit(code)
        write_latest(root, {"stage": "11D-3D", "supervisor_pid": pid,
                            "attempt": str(attempt.relative_to(root)), "created_at_utc": now()})
        print("\nSTAGE 11D-3D — BACKGROUND LAUNCH REQUESTED")
        print(f"Supervisor PID       : {pid}")
        print("Execution            : SEQUENTIAL; build jobs=1")
        print("Validation vectors   : 48; batches=45; planned records=2194704")
        print(f"Master log           : {attempt / 'master.log'}")
        print(f"Completion receipt   : {attempt / 'completion.json'}")
        print(f"Checkpoint           : {local(root, CHECKPOINT)}")
        print("Use --status to confirm progress. Keep WSL and the laptop awake.")
        return 0


def status(root):
    latest = local(root, LATEST)
    if not latest.is_file():
        print("No launch record from this launcher. No campaign was started by this command.")
        return 0
    pointer = read_json(latest)
    attempt = local(root, pointer["attempt"])
    require(attempt.parent == local(root, RESULTS + "/stage_11d3d_attempts"), "unexpected attempt path")
    busy = False
    lock = local(root, LOCK)
    if lock.is_file():
        with lock.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = True
    receipt = attempt / "completion.json"
    print("STAGE 11D-3D — VALIDATION CAMPAIGN STATUS")
    print(f"Supervisor PID       : {pointer['supervisor_pid']}")
    print(f"Launch lock held     : {'YES (preflight or run active)' if busy else 'NO'}")
    cp = local(root, CHECKPOINT)
    if cp.is_file():
        checkpoint = read_json(cp)
        print(f"Checkpoint batches   : {len(checkpoint.get('completed_batches', []))}/45 (status display, not integrity verification)")
    else:
        print("Checkpoint           : NOT YET WRITTEN (first checkpoint follows Batch 000 PASS)")
    if receipt.is_file():
        done = read_json(receipt)
        print(f"Execution result     : {done.get('status')}")
        print(f"Runner return code   : {done.get('runner_return_code')}")
        if done.get("error"):
            print(f"Error                : {done['error']}")
        print(f"Completion receipt   : {receipt}")
        print(f"Receipt SHA          : {sha(receipt)}")
        print("Dataset integrity freeze: NOT YET PERFORMED")
    elif not busy:
        print("State                : INTERRUPTED OR SUPERVISOR FAILED; preserve all files and request review")
    log = attempt / "master.log"
    print(f"Master log           : {log}")
    if log.is_file():
        print("\nLATEST LOG OUTPUT")
        with log.open(errors="replace") as stream:
            for line in deque(stream, maxlen=25):
                print(line, end="")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--start", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    options = parser.parse_args()
    try:
        root = options.project_root.resolve()
        require(root.name == "vlsi_fault_detection_v2" and Path(__file__).resolve().parent == root,
                "copy this launcher to ~/vlsi_fault_detection_v2 and run it from there")
        require(sys.version_info >= (3, 10), "Python 3.10+ required")
        if options.status:
            return status(root)
        return launch(root, options.resume, options.check_only)
    except (Stop, OSError, ValueError, KeyError, TypeError) as error:
        print(f"STOP: {error}", file=sys.stderr, flush=True)
        print("Do not alter frozen hashes, use --overwrite, or delete campaign files.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
