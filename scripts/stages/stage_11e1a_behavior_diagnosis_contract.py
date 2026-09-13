#!/usr/bin/env python3
"""Stage 11E-1A: separate behavior-based detection/localization task contract.

Standard library only. Checks existing evidence and publishes an additive
contract bundle. No simulation, training, model loading or validation/HOLDOUT
sample access. Original experiment remains ACCEPTANCE_NOT_MET.
Run from ~/vlsi_fault_detection_v2. Re-running verifies an existing bundle.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import sys

STAGE = '11E-1A'
VERSION = 'HMAC-BEHAVIOR-DIAGNOSIS-CONTRACT-v1'
REPORT = 'results/hmac_behavior_diagnosis_pilot/observability_7rflb9md/observability_report.json'
REPORT_SHA = 'ba08eee29ab1b994998f049d723a9ac2bf0e0e1a495c16c8cd5cd9f6724955e7'
TRAIN = ('results/hmac_fault_campaign_11c5/canonical_dataset_11c5c/'
         'hmac_fault_campaign_canonical_train_11c5c.csv.gz')
OLD_REVIEW = 'results/hmac_fault_campaign_11d3/hmac_validation_failure_review_project_disposition_freeze_11d3j.json'
OLD_CONTRACT = 'config/diagnostic_model/hmac_validation_inference_contract_11d3g.json'
ANCHORS = {
    'hmac_behavior_observability_audit.py': '7546600ed4914c057f18009a542e5c8e9a5c4ee8580ed2fc6732889b5b6d95ca',
    TRAIN: 'dcc79c94fa3a8d592b516e5f5683b1bf007682745103a9419ee20ef20a7ec78f',
    OLD_REVIEW: 'f0771b64be274e00aff66865fedb72233d141b1f178d1ffd5819f37f266d9dae',
    OLD_CONTRACT: '080b3e4efc940ae6d1abcdc663c6d08bff96dba2ffaae00ddd7a0741a3d1b359',
    'build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json': '0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1',
    'rtl/hmac/opentitan_hmac_sha256_msg32.sv': 'd56d0835d7609eaafbd850e76a50ac70b20d7bf0e18e24ab53d185fe34b82e9e',
}
COUNTS = {
    'physical_sites': 22839, 'fault_instances': 45678, 'test_vectors': 64,
    'canonical_rows': 2926272, 'fault_instances_with_observable_effect': 22930,
    'fault_instances_indistinguishable_from_no_fault': 22748,
    'fault_instances_with_unique_exact_site_within_catalog': 9171,
    'fault_instances_ambiguous_across_sites': 13759,
    'largest_observable_candidate_site_set': 1668,
}
PARENT = 'results/hmac_behavior_diagnosis_pilot'
DEST = PARENT + '/contract_11e1a'
MODEL_IDS = ('HYBRID_FUSION_MLP_11D2C', 'DIR_SGC_11D1D', 'DEEP_MLP_11C5K', 'CONVENTIONAL_LOGREG_11C5H')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def unique(pairs):
    obj = {}
    for k, v in pairs:
        require(k not in obj, f'duplicate JSON key: {k}')
        obj[k] = v
    return obj


def read_json(path):
    with path.open() as f:
        obj = json.load(f, object_pairs_hook=unique)
    require(isinstance(obj, dict), f'expected JSON object: {path.name}')
    return obj


def stable_json(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


class Evidence:
    def __init__(self, root):
        self.root = root
        self.records = {}

    def path(self, relative):
        p = Path(relative)
        require(not p.is_absolute() and '..' not in p.parts, 'expected project-relative path')
        p = (self.root / p).resolve()
        require(p.is_relative_to(self.root), 'path escapes project')
        return p

    def check(self, relative, expected, size=None):
        path = self.path(relative)
        require(path.is_file(), f'missing input: {relative}')
        require(sha(path) == expected, f'SHA mismatch: {relative}')
        n = path.stat().st_size
        require(size is None or n == size, f'size mismatch: {relative}')
        self.records[relative] = {'path': relative, 'sha256': expected, 'bytes': n}
        return path

    def link(self, record):
        return self.check(record['path'], record['sha256'], record.get('bytes'))

    def recheck(self):
        for record in list(self.records.values()):
            self.link(record)


def group_summary(path, fault_count=45678):
    seen = set()
    signatures = set()
    normal_groups = invisible = unique_site = ambiguous = unique_fault = largest = 0
    histogram = Counter()
    with path.open() as f:
        for line in f:
            group = json.loads(line, object_pairs_hook=unique)
            sig = group['profile_sha256']
            require(re.fullmatch('[0-9a-f]{64}', sig) is not None and sig not in signatures,
                    'invalid/duplicate response-profile identifier')
            signatures.add(sig)
            require(type(group['matches_no_fault']) is bool, 'normal-profile marker')
            members = group['fault_instances']
            require(isinstance(members, list) and members, 'empty profile group')
            sites = set()
            for label in members:
                match = re.fullmatch(r'HMAC-STEM-([0-9]{6}):SA([01])', label)
                require(match is not None, 'invalid profile membership label')
                site, stuck = map(int, match.groups())
                k = (site - 1) * 2 + stuck
                require(0 <= k < fault_count and k not in seen, 'duplicate/out-of-range fault membership')
                seen.add(k)
                sites.add(site)
            if group['matches_no_fault']:
                normal_groups += 1
                invisible += len(members)
            else:
                histogram[len(sites)] += 1
                largest = max(largest, len(sites))
                unique_fault += int(len(members) == 1)
                if len(sites) == 1:
                    unique_site += len(members)
                else:
                    ambiguous += len(members)
    require(len(seen) == fault_count, 'missing fault memberships')
    require(normal_groups == 1, 'expected exactly one no-fault-compatible response group')
    return {
        'distinct_fault_response_profiles': len(signatures),
        'fault_instances_indistinguishable_from_no_fault': invisible,
        'fault_instances_with_observable_effect': fault_count - invisible,
        'fault_instances_with_unique_exact_site_within_catalog': unique_site,
        'fault_instances_ambiguous_across_sites': ambiguous,
        'unique_exact_fault_instance_profiles_excluding_no_fault': unique_fault,
        'largest_observable_candidate_site_set': largest,
        'observable_profile_count_by_candidate_site_count': {str(k): v for k, v in sorted(histogram.items())},
    }


def reference_suite(path):
    # Only the first batch's 64 already-used TRAIN baseline rows are decoded.
    # The compressed file is independently hash-verified in full.
    refs = []
    with gzip.open(path, 'rt', newline='') as f:
        reader = csv.DictReader(f)
        for slot in range(64):
            row = next(reader)
            require(row['vector_split'] == 'TRAIN' and row['run_type'] == 'BASELINE', 'reference row type')
            require(int(row['record_id']) == slot and int(row['batch_id']) == 0 and int(row['vector_slot']) == slot,
                    'reference row order')
            require(int(row['fault_enable']) == int(row['timed_out']) == int(row['unknown']) == 0,
                    'reference run flags')
            require(int(row['cycles']) == int(row['baseline_cycles']) == 343, 'reference latency')
            digest = row['expected_digest'].lower()
            require(re.fullmatch('[0-9a-f]{64}', digest) is not None and row['actual_digest'].lower() == digest,
                    'reference digest')
            refs.append({'vector_slot': slot, 'vector_id': int(row['vector_id']),
                         'expected_digest': digest, 'expected_cycles': 343})
    require(len({r['vector_id'] for r in refs}) == 64, 'reference vector identity duplication')
    return refs


def verify_inputs(e, report_path):
    print('FROZEN INPUT VERIFICATION', flush=True)
    for relative, digest in ANCHORS.items():
        e.check(relative, digest)
        print('  ' + Path(relative).name + ': OK', flush=True)
    report = read_json(e.check(report_path, REPORT_SHA))
    require(report['audit_status'] == 'COMPLETED' and report['experiment_status'] == 'EXPLORATORY; NO MODEL TRAINED',
            'observability audit state')
    for key, value in COUNTS.items():
        require(report[key] == value, f'observability count: {key}')
    require(report['source'] == {'path': TRAIN, 'sha256': ANCHORS[TRAIN]}, 'observability source binding')
    require(report['auditor_sha256'] == ANCHORS['hmac_behavior_observability_audit.py'], 'auditor binding')
    require(report['training_performed'] is False and report['validation_or_holdout_opened'] is False,
            'observability scope')
    groups_path = e.link(report['profile_groups'])
    computed = group_summary(groups_path)
    for key, value in computed.items():
        require(report[key] == value, f'profile membership/report disagreement: {key}')
    old = read_json(e.path(OLD_REVIEW))
    require(old['status'] == 'PASS' and old['review_status'] == 'FROZEN', 'original review state')
    require(old['current_experiment_disposition'] == 'ACCEPTANCE_NOT_MET; STOP_BEFORE_HOLDOUT'
            and old['validation_target'] == 'NOT_MET' and old['project_final_freeze'] == 'NOT_DECLARED',
            'original experiment disposition')
    require(old['holdout_campaign_execution_authorized'] is False, 'HOLDOUT must remain blocked')
    prior_contract = read_json(e.path(OLD_CONTRACT))
    models = prior_contract['models']
    require([m['model_id'] for m in models] == list(MODEL_IDS), 'four original model identities')
    for spec in models:
        require(spec['model_artifact']['path'] == spec['model'] and spec['selection_lock']['path'] == spec['lock'],
                'model/selection path binding')
        e.link(spec['model_artifact'])
        e.link(spec['selection_lock'])
    refs = reference_suite(e.path(TRAIN))
    print('  Profile memberships: 45,678/45,678; counts agree', flush=True)
    print('  Four original models and selection locks: SHA verified; objects not loaded', flush=True)
    return report, old, models, refs


def documents(report, old, models, refs):
    suite_sha = hashlib.sha256(stable_json(refs)).hexdigest()
    suite = {'format_version': 'HMAC-FIXED-TRAIN-OBSERVATION-SUITE-v1',
             'vector_count': 64, 'suite_commitment': suite_sha, 'vectors': refs}
    schema = {
        'format_version': 'HMAC-BEHAVIOR-OBSERVATION-CSV-v1', 'suite_commitment': suite_sha,
        'header_exact': ['vector_slot', 'vector_id', 'actual_digest', 'cycles', 'timed_out'],
        'rows_exact': 64, 'row_order': 'vector_slot 0 through 63; vector_id must match suite',
        'actual_digest': '64 hexadecimal characters when completed; empty or 64 hex characters if timed_out=1',
        'cycles': 'integer 0 through 2000', 'timed_out': 'integer 0 or 1; 1 requires cycles=2000',
        'normalization': '35 bytes per test: big-endian uint8 timed_out, uint16 cycles, 32 digest bytes; digest is zeroed at timeout.',
        'intake_rule': 'Reject missing, duplicate, extra or malformed rows/columns; reject unknown X/Z measurements; never guess missing values.',
        'ground_truth_separation': 'Fault site, fault type, injection flags, selectors and activity are forbidden in observation input. Keep experiment truth in a separate scoring file.',
        'label_independence': 'Diagnosis must complete without reading the scoring file or injection controls.',
    }
    contract = {
        'stage': STAGE, 'version': VERSION, 'status': 'PASS', 'contract_status': 'FROZEN',
        'study': 'BEHAVIOR_BASED_DETECTION_AND_LOCALIZATION_PILOT',
        'scope': 'One frozen OpenTitan HMAC-SHA256 msg32 circuit; fixed 64-test suite; zero or one persistent net-stem SA0/SA1 fault.',
        'objective': 'Observe circuit behavior without being told the actual fault identity; detect observable deviations and return all matching fault-site candidates.',
        'prior_experiment': {'disposition': old['current_experiment_disposition'],
            'validation_target': old['validation_target'], 'final_project_freeze': 'NOT_DECLARED',
            'model_retraining': 'PROHIBITED', 'threshold_changes': 'PROHIBITED', 'acceptance_waiver': 'NONE'},
        'catalog_scope': {'source': TRAIN, 'source_sha256': ANCHORS[TRAIN], **COUNTS,
            'existing_site_partitions': 'All already-used development site groups enter this reference catalog. They are not a new held-out evaluation.',
            'labels_role': 'site_id and SA value identify catalog entries only, never supplied as observed features'},
        'measurement_protocol': {'suite_commitment': suite_sha, 'independent_reset_before_each_vector': True,
            'fault_condition': 'Same hidden site/value across all 64 tests, reapplied for every transaction.',
            'timing': 'Match the frozen campaign cycle-count and start/done conventions; fault-free latency=343, timeout=2000.',
            'stimulus': 'Use the existing 64 TRAIN key/message pairs in the committed slot order; do not substitute new stimuli.',
            'measurement_origin': 'Circuit/simulator output digest and timing, not the injection selector or fault_raw monitor.',
            'intake': schema['header_exact']},
        'baseline_algorithm': {'detector': 'Compare every response with the suite golden digest, latency and timeout reference.',
            'locator': 'Compare normalized complete response profiles against the catalog; verify full bytes, not only a hash.',
            'candidate_handling': 'Deduplicate physical sites; retain fault-instance membership. Return every match, with no arbitrary single-site tie break.',
            'normal_profile': 'Return NO_OBSERVED_ANOMALY with localization NOT_IDENTIFIABLE. Include no-fault compatibility and the invisible-fault candidate count.',
            'confidence': 'No learned confidence or probability; catalog uniqueness is not certainty about physical defect cause.',
            'algorithm_class': 'DETERMINISTIC REFERENCE CHECKER AND LOOKUP; NOT A NEW ML MODEL'},
        'result_semantics': {
            'NO_OBSERVED_ANOMALY': 'No measured deviation for this test suite. Does not prove fault absence.',
            'UNIQUE_SITE_IN_CATALOG': 'Anomaly observed; all matched catalog fault instances share one physical site. SA value may remain ambiguous.',
            'AMBIGUOUS_SITES_IN_CATALOG': 'Anomaly observed; more than one candidate physical site.',
            'NO_CATALOG_MATCH': 'Anomaly observed; no known catalog profile fits. Do not invent a location.',
            'INVALID_OBSERVATIONS': 'Wrong suite, incomplete/malformed data or invalid capture; no diagnosis.'},
        'functional_acceptance': {
            'required': ['Normal profile returns NO_OBSERVED_ANOMALY, never CERTIFIED_FAULT_FREE.',
                'All valid catalog profiles return their complete candidate sets; never force a location for ambiguous groups.',
                'All injection/truth columns are rejected at observation intake.',
                'Missing/extra/duplicate vectors and malformed measurements are rejected.',
                'Timeout digest values cannot change normalized diagnosis.',
                'An anomalous complete profile absent from the catalog returns NO_CATALOG_MATCH.',
                'Repeated diagnosis of identical observations gives identical results.'],
            'measurement': 'Report separate rule consistency, exact candidate-set agreement, ambiguous/no-match counts and false-alarm counts. No aggregate ML accuracy claim.'},
        'evaluation_plan': {
            'different_task': 'The earlier detectability MCC values are not directly comparable with behavior-detection or localization accuracy.',
            'software_tests': 'Synthetic fixtures test logic and failure cases; clearly marked synthetic.',
            'catalog_replay': 'Known development profiles with truth stripped at intake; verifies lookup consistency only, not independent generalization.',
            'future_blinded_capture': 'Freeze a separate acquisition and scoring protocol before new captures. Keep injection truth out of the diagnosis process; release it only for scoring.',
            'repeated_simulation_limit': 'Re-simulating deterministic catalog cases is reproducibility evidence, not new independent samples or unseen-vector accuracy.',
            'future_learned_localizer': 'Separate training contract needed, including independent response evaluation, class/site coverage and treatment of indistinguishable labels.',
            'forbidden_claims': ['Detection of every physical fault', 'Unseen-chip generalization',
                'A normal response proves healthy hardware', 'All enabled injections are behaviorally detectable',
                'High accuracy from learning the existing digest-mismatch rule demonstrates new ML capability']},
        'authorizations': {
            'basis': 'User requested step-by-step development of the separate behavior-diagnosis goal.',
            'catalog_and_checker_implementation': True, 'development_catalog_construction': True,
            'synthetic_and_catalog_replay_checks': True, 'old_model_changes': False,
            'new_model_training': False, 'new_simulation_campaign': False,
            'validation_sample_access': False, 'holdout_access': False, 'final_project_freeze': False},
        'preserved_models': [{k: m[k] for k in ('model_id', 'candidate', 'threshold', 'model_artifact', 'selection_lock')} for m in models],
        'observability_report_sha256': REPORT_SHA,
        'next_step': '11E-1B — Response Catalog and Behavior-Based Detector/Locator Baseline',
    }
    notes = f'''# Behavior-based diagnosis: scope for mentor review

The earlier detectability-prediction experiment is complete as an evaluated
experiment, with validation acceptance NOT_MET. Its models, results, thresholds
and failed criterion remain preserved. The broader project is not declared complete.

The user now wants detection and localization from observed circuit behavior.
We are defining a separate additive study using the existing HMAC circuit and
development observations. This contract does not train or evaluate a new model.

The audit found 22,930 observable fault instances, 22,748 indistinguishable from
normal behavior, 9,171 with a unique catalog site and 13,759 ambiguous across
sites. These are finite-catalog distinguishability counts, not model accuracy.

## Order of work

1. Freeze the observation inputs, answer meanings and evaluation boundaries here.
2. Build a golden-reference checker and exact-response catalog lookup baseline.
3. Verify complete candidate sets and demonstrate observations without fault identity.
4. Review remaining ambiguity; additional observations or tests may be needed.
5. Define an independently evaluated learned localizer only after confirming useful information is available.

The initial baseline will be ordinary software, not deep learning. It establishes
what the new task can resolve before adding a trained model. The existing GNN
and hybrid models remain evidence for the earlier, different prediction task.
Their MCC values cannot be directly compared with accuracy on this new task.

New studies take time and add files; no completion deadline or accuracy is promised.
If submission is near, present the existing result honestly and agree with the
mentor how far the extension must go. Do not relabel the earlier failure as a pass.

## Demonstration limits

A normal response means no observed error, not proof of healthy hardware. A unique
match means unique within this catalog and protocol. Several faults may remain
equivalent, and unsupported faults can even mimic known or normal responses.
Deterministic replay is not an independent generalization experiment.

Contract version: {VERSION}
Next step: {contract['next_step']}
'''
    return {'contract.json': stable_json(contract), 'observation_schema.json': stable_json(schema),
            'reference_suite.json': stable_json(suite), 'mentor_scope.md': notes.encode()}


def publish(e, content):
    dest = e.path(DEST)
    own_record = {'path': Path(__file__).name, 'sha256': sha(Path(__file__).resolve())}
    base_audit = {'stage': STAGE, 'version': VERSION, 'status': 'PASS', 'contract_status': 'FROZEN',
        'generator': own_record, 'input_evidence': e.records,
        'checked_frozen_inputs_modified': False, 'model_objects_loaded': 0,
        'model_training_performed': False, 'simulation_performed': False,
        'validation_or_holdout_samples_opened': False, 'original_validation_target': 'NOT_MET',
        'project_final_freeze_declared': False}
    if dest.exists():
        audit = read_json(dest / 'freeze.json')
        require(set(audit) == set(base_audit) | {'outputs', 'created_at_utc'}, 'existing audit field set')
        require(all(audit[k] == value for k, value in base_audit.items()),
                'existing audit/source binding; do not overwrite')
        require(set(audit['outputs']) == set(content), 'existing output set')
        for name, data in content.items():
            require((dest / name).read_bytes() == data, f'existing output differs: {name}')
            require(audit['outputs'][name] == {'path': DEST + '/' + name,
                    'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}, 'existing output record differs')
        e.recheck()
        return dest, True
    parent = e.path(PARENT)
    with tempfile.TemporaryDirectory(prefix='11e1a_pending_', dir=parent) as pending:
        temp = Path(pending)
        output_records = {}
        for name, data in content.items():
            with (temp / name).open('xb') as f:
                f.write(data)
            output_records[name] = {'path': DEST + '/' + name, 'sha256': sha(temp / name), 'bytes': len(data)}
        e.recheck()
        audit = {**base_audit, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'outputs': output_records}
        (temp / 'freeze.json').write_bytes(stable_json(audit))
        require(not dest.exists(), 'destination already exists; no overwrite')
        os.rename(temp, dest)
    return dest, False


def run(root, report_path):
    e = Evidence(root)
    parent = e.path(PARENT)
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / 'contract_11e1a.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another contract writer is active') from None
        report, old, models, refs = verify_inputs(e, report_path)
        dest, existing = publish(e, documents(report, old, models, refs))
    print('\nSTAGE 11E-1A — BEHAVIOR-BASED DETECTION AND LOCALIZATION CONTRACT')
    print('Status                         : PASS' + (' / EXISTING BUNDLE VERIFIED' if existing else ''))
    print('Task / observation contract    : FROZEN')
    print('Unknown fault identity in input: PROHIBITED')
    print('First implementation           : REFERENCE CHECKER + RESPONSE LOOKUP')
    print('Catalog/checker development    : AUTHORIZED')
    print('New model training             : NOT STARTED / NOT AUTHORIZED BY THIS CONTRACT')
    print('Original validation target     : NOT_MET; PRESERVED')
    print('Checked frozen inputs changed  : NO')
    print('Validation/HOLDOUT sample access: NONE')
    print('Project final freeze           : NOT DECLARED')
    print(f'Contract bundle                : {dest}')
    print(f'Audit SHA256                   : {sha(dest / "freeze.json")}')
    print('Next: 11E-1B — Response Catalog and Behavior-Based Detector/Locator Baseline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--observability-report', default=REPORT,
                        help='project-relative path; the expected report SHA remains fixed')
    args = parser.parse_args()
    try:
        run(args.project_root.resolve(), args.observability_report)
        return 0
    except KeyboardInterrupt:
        print('\nStopped. Existing experiment remains unchanged.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'STOP: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
