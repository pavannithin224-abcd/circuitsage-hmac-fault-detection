#!/usr/bin/env python3
"""Freeze a prospective blinded simulator demonstration contract; no capture or training.
Run from ~/vlsi_fault_detection_v2 with Python 3.9+ (standard library only).
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types

SOURCE = 'stage_11e1b_behavior_baseline.py'
SOURCE_SHA = 'c9b26acd5f77b6bdb681cc91c8697199346acd49c678c8d6f180e9f9e74e61f3'
BASE = 'results/hmac_behavior_diagnosis_pilot/baseline_11e1b'
BASE_SHA = 'a6f7312b9cd8a137da711133cd425a706c1705a84eb444df4a45eb2e3129ee71'
DEST = 'results/hmac_behavior_diagnosis_pilot/contract_11e1c'
VERSION = 'HMAC-BLINDED-CAPTURE-CONTRACT-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def load_baseline(root):
    path = (root / SOURCE).resolve()
    require(path.is_relative_to(root), 'baseline helper escapes project')
    data = path.read_bytes()
    require(digest(data) == SOURCE_SHA, '11E-1B source SHA mismatch')
    module = types.ModuleType('_verified_behavior_baseline')
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def documents(suite):
    policy = {
        'stage': '11E-1C', 'version': VERSION, 'status': 'PASS', 'contract_status': 'FROZEN',
        'purpose': 'Fresh simulator capture, identity-blind diagnosis, then truth-revealed scoring of known catalog cases.',
        'baseline_freeze_sha256': BASE_SHA, 'suite_commitment': suite['suite_commitment'],
        'scope': 'Same frozen HMAC circuit; zero or one persistent SA0/SA1 net-stem fault; existing 64 TRAIN tests.',
        'case_plan': {'total_cases': 12, 'fault_free_cases': 2, 'unique_site_fault_cases': 4,
            'ambiguous_site_fault_cases': 4, 'normal_compatible_fault_cases': 2,
            'distinct_injected_fault_instances': 10, 'tests_per_case': 64,
            'canonical_capture_rows': 768, 'separate_replay_rows': 768, 'total_simulated_transactions': 1536,
            'selection': 'Sample without replacement within each catalog category with a recorded seed; include both SA0 and SA1 overall. Fail if infeasible.',
            'selection_freeze': 'Before any capture, commit selected cases, seed, random case-ID mapping and truth. Never choose cases based on fresh outcomes.',
            'case_identifiers': 'Random opaque IDs, randomly permuted; no category, site, stuck value, batch or selector encoded in public paths or order.',
            'no_match_case': 'Separate synthetic software check only; never count it as an injected circuit experiment.'},
        'capture': {'independent_reset_before_each_test': True, 'same_hidden_fault_across_64_tests': True,
            'normal_cycles': 343, 'timeout_cycles': 2000,
            'timing': 'Reuse the frozen campaign start/done sampling and cycle-count conventions exactly.',
            'vectors': 'Use committed TRAIN key/message pairs in suite order; verify stimulus commitments before capture.',
            'measurement_source': 'Fresh simulator output digest, completion cycles and timeout; never copy catalog responses.',
            'forbidden_measurements': ['fault_raw_o', 'activity', 'selector', 'site_id', 'stuck_value', 'fault_enable'],
            'execution': 'Sequential; one build job; new output directory; no overwrite of frozen artifacts.',
            'traceability': 'Privately record tool versions, command, netlist/testbench/binary hashes, logs and observation hashes.',
            'replay': 'Re-execute all 12 cases once; preserve both captures and compare normalized observations exactly.',
            'errors': 'Preserve failed attempts and stop on malformed/missing capture or replay mismatch; never silently substitute a case.'},
        'blinding': {
            'level': 'Process and file-access separation, not cryptographic isolation from the laptop owner.',
            'capture_role': 'May read injection truth, stimulus payloads and simulator logs. Exports only opaque observation files and public manifest.',
            'diagnosis_role': 'May read frozen checker, normal references, full reference catalog and public observations. Cannot read current-case truth, capture logs, selection seed or category map.',
            'catalog_labels': 'Known catalog site labels are necessary to report candidates; current-case injection identity is forbidden.',
            'required_enforcement': 'Future runner must restrict diagnosis file access to an explicit allowlist and test denial of truth/log paths; a filename convention alone is insufficient.',
            'truth_commitment': 'SHA256 over canonical JSON containing full private truth plus a cryptographically random 32-byte salt; publish only commitment before capture.',
            'prediction_commitment': 'Commit all 12 complete diagnosis JSON outputs and their observation hashes before truth is revealed to scoring.',
            'scoring_role': 'Verify truth commitment and prediction commitment, then join by opaque case_id; never feed scoring truth back into diagnosis.'},
        'acceptance': {
            'required': ['12 complete canonical captures and 12 separate replays; exact suite IDs and order.',
                'Public observation schema contains only the five permitted fields.',
                'Demonstrated denial of truth/log access by diagnosis process.',
                'Truth commitment predates capture; predictions committed before truth reveal.',
                'Both fault-free cases return NO_OBSERVED_ANOMALY and never assert certified fault absence.',
                'Fresh normalized responses agree with the expected catalog profiles for these known cases.',
                'Unique cases return the full single-site candidate set containing the injected site.',
                'Ambiguous cases retain all candidates and contain the injected site; no arbitrary tie-break.',
                'Normal-compatible faults return NO_OBSERVED_ANOMALY with NOT_IDENTIFIABLE, never a claim of proven health.',
                'All canonical/replay observations agree; frozen inputs remain byte-identical.'],
            'failure_handling': 'Record observed result and STOP; do not adjust detector, threshold, suite or truth to force acceptance.'},
        'reporting': {
            'metrics': ['Raw 12-case table with expected category, result, anomaly flag, candidate count and truth-in-candidate-set.',
                'False alarms among the two fault-free cases, with numerator/denominator.',
                'Observable-anomaly detections among eight preselected observable faults, with numerator/denominator.',
                'Observed anomalies among all ten injected faults, including the two invisible cases.',
                'Exact-site resolutions among the four unique cases and among all ten faults; report both denominators.',
                'Candidate-set coverage and set size among observable faults; separately report normal-compatible cases.',
                'No-match, invalid-capture and replay-disagreement counts; never drop failures.'],
            'limitations': 'Small stratified known-catalog demonstration. Counts are workflow/reproducibility evidence, not independent generalization accuracy or population estimates.',
            'not_comparable': 'Earlier detectability-prediction MCC values measure a different task.',
            'unsupported_scope': ['transient faults', 'multiple simultaneous faults', 'new test vectors', 'different chips', 'physical silicon validation']},
        'authorizations': {'capture_runner_implementation': True, 'synthetic_runner_tests': True,
            'fresh_simulator_execution': False, 'new_model_training': False, 'old_model_changes': False,
            'validation_sample_access': False, 'holdout_access': False,
            'next_gate': '11E-1D — Blinded Capture Runner Implementation and Preflight; qualify source and access separation before enabling capture.'},
        'original_experiment': {'validation_target': 'NOT_MET; PRESERVED', 'project_final_freeze': 'NOT_DECLARED'},
    }
    interfaces = {
        'suite_commitment': suite['suite_commitment'],
        'observation_csv': {'header_exact': ['vector_slot', 'vector_id', 'actual_digest', 'cycles', 'timed_out'],
            'rows_exact': 64, 'rules': 'Exactly the frozen 11E-1A schema, including timeout digest normalization.',
            'path': 'observations/hmac_behavior_11e1b/<attempt>/<opaque_case_id>.csv'},
        'public_capture_manifest': {'fields_only': ['version', 'suite_commitment', 'truth_commitment', 'cases'],
            'case_fields_only': ['case_id', 'observation_path', 'observation_sha256'], 'cases_exact': 12},
        'private_truth': {'never_diagnosis_input': True, 'fields': ['version', 'salt_hex', 'selection_seed', 'cases'],
            'case_fields': ['case_id', 'fault_present', 'site_id', 'stuck_value', 'selection_category'],
            'fault_free': 'site_id and stuck_value are null; fault_present=false.',
            'storage': 'Private capture workspace outside diagnosis allowlist; access controls are required.'},
        'prediction_manifest': {'fields': ['version', 'suite_commitment', 'catalog_freeze_sha256', 'truth_commitment', 'cases'],
            'case_fields': ['case_id', 'observation_sha256', 'prediction_path', 'prediction_sha256'],
            'case_binding': 'Exact same set of case IDs as public capture manifest; all predictions committed before scoring.'},
        'this_stage_emits_no_truth_or_observations': True,
    }
    note = '''# Blinded behavior diagnosis demonstration

The next implementation will capture fresh simulator responses for 12 cases.
Two cases have no injected fault. Ten contain known catalog faults: four with
unique site signatures, four ambiguous, and two that look normal under this suite.
Each case uses all 64 committed development tests, followed by a separate replay.

Capture knows the injected fault. Diagnosis receives only responses, normal
references and the existing catalog. Scoring reveals truth after predictions
are committed. The future runner must enforce this separation; this contract
does not yet implement isolation or perform any simulation.

Normal-looking faults must remain unresolved. Ambiguous cases must retain every
matching site. A catalog match does not prove the cause of a physical defect.
This small, deliberately selected demonstration cannot establish general accuracy.

Original model results and validation acceptance NOT_MET remain unchanged.
HOLDOUT remains untouched. No model is trained, and no project final freeze is declared.
'''
    return {'capture_contract.json': encode(policy), 'interface_contract.json': encode(interfaces),
            'reference_suite.json': encode(suite), 'mentor_notes.md': note.encode()}


def run(root):
    source_sha = digest(Path(__file__).read_bytes())
    b = load_baseline(root)
    print('STAGE 11E-1C — BLINDED OBSERVATION CAPTURE CONTRACT', flush=True)
    _, e, _, suite, _ = b.contract_inputs(root, full=True)
    e.check(SOURCE, SOURCE_SHA)
    e.check(BASE + '/freeze.json', BASE_SHA)
    audit = b.verify_bundle(e, suite)
    require(audit['catalog_status'] == 'FROZEN', 'catalog not frozen')
    expected = documents(suite)
    dest = e.path(DEST)
    parent = dest.parent
    with (parent / 'contract_11e1c.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if dest.exists():
            freeze = b.read_json(dest / 'freeze.json')
            require(freeze['stage'] == '11E-1C' and freeze['status'] == 'PASS' and
                    freeze['version'] == VERSION and freeze['source_sha256'] == source_sha and
                    freeze['contract_status'] == 'FROZEN', 'existing freeze mismatch')
            for key in ('simulation_performed', 'truth_selected_or_revealed', 'diagnosis_performed',
                        'model_training_performed', 'validation_or_holdout_samples_accessed',
                        'checked_frozen_inputs_changed', 'project_final_freeze_declared'):
                require(freeze[key] is False, 'existing scope mismatch: ' + key)
            require(freeze['original_validation_target'] == 'NOT_MET; PRESERVED', 'original disposition changed')
            require(freeze['input_evidence'] == e.records, 'existing input binding mismatch')
            require(set(freeze['outputs']) == set(expected), 'existing output set mismatch')
            for name, data in expected.items():
                require((dest / name).read_bytes() == data, 'existing contract differs: ' + name)
                require(freeze['outputs'][name] == {'path': DEST + '/' + name,
                    'sha256': digest(data), 'bytes': len(data)}, 'existing output binding: ' + name)
            print('Existing identical contract verified; no overwrite.')
        else:
            with tempfile.TemporaryDirectory(prefix='11e1c_pending_', dir=parent) as pending:
                tmp = Path(pending)
                for name, data in expected.items():
                    (tmp / name).write_bytes(data)
                e.recheck()
                require(digest(Path(__file__).read_bytes()) == source_sha, 'source changed during run')
                freeze = {'stage': '11E-1C', 'version': VERSION, 'status': 'PASS', 'contract_status': 'FROZEN',
                    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_sha,
                    'input_evidence': e.records,
                    'outputs': {name: {'path': DEST + '/' + name, 'sha256': digest(data), 'bytes': len(data)}
                        for name, data in expected.items()},
                    'simulation_performed': False, 'truth_selected_or_revealed': False,
                    'diagnosis_performed': False, 'model_training_performed': False,
                    'validation_or_holdout_samples_accessed': False, 'checked_frozen_inputs_changed': False,
                    'original_validation_target': 'NOT_MET; PRESERVED', 'project_final_freeze_declared': False}
                (tmp / 'freeze.json').write_bytes(encode(freeze))
                require(not dest.exists(), 'output exists; refusing overwrite')
                os.rename(tmp, dest)
        e.recheck()
    print('Status                         : PASS')
    print('Capture / scoring contract     : FROZEN')
    print('Planned cases                  : 12 (2 fault-free + 10 injected)')
    print('Canonical / replay rows        : 768 / 768')
    print('Truth supplied to diagnosis     : PROHIBITED')
    print('Runner implementation          : AUTHORIZED')
    print('Fresh simulation performed     : NO')
    print('Existing experiment changed    : NO')
    print('Independent accuracy measured  : NO')
    print(f'Contract bundle                : {dest}')
    print(f'Audit SHA256                   : {b.sha(dest / "freeze.json")}')
    print('Next: 11E-1D — Blinded Capture Runner Implementation and Preflight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        run(args.project_root.resolve())
    except KeyboardInterrupt:
        print('STOP: interrupted; no simulation started.')
        raise SystemExit(130)
    except Exception as error:
        print(f'STOP: {error}')
        raise SystemExit(1)
