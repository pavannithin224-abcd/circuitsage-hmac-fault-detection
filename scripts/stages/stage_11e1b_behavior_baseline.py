#!/usr/bin/env python3
"""11E-1B: frozen response catalog and terminal behavior diagnosis baseline.

Commands: build | demo | diagnose --observations <project-relative CSV>
Only the five observation columns from 11E-1A are accepted. No true fault
identity is accepted at diagnosis time. This is deterministic checking and
lookup, not trained ML or an independent accuracy evaluation.
Python 3.9+ on WSL/Linux; standard library only.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys
import tempfile
import types

VERSION = 'HMAC-BEHAVIOR-DETECTOR-LOCATOR-v1'
HELPER = 'stage_11e1a_behavior_diagnosis_contract.py'
HELPER_SHA = 'f31184bf39785bde59f78ec1b66cb6e49cdffac3f43414d9d3dfe750aa9dea33'
AUDITOR = 'hmac_behavior_observability_audit.py'
AUDITOR_SHA = '7546600ed4914c057f18009a542e5c8e9a5c4ee8580ed2fc6732889b5b6d95ca'
PARENT = 'results/hmac_behavior_diagnosis_pilot'
CONTRACT_DIR = PARENT + '/contract_11e1a'
CONTRACT_FREEZE_SHA = 'b6ac3598ce7e27c51885d61ebfb19837ecddf00fd384344f179baa6f788e680d'
DEST = PARENT + '/baseline_11e1b'
HEADER = ['vector_slot', 'vector_id', 'actual_digest', 'cycles', 'timed_out']
STEPS = 64
WIDTH = 35
OUTPUTS = ('catalog.sqlite3', 'summary.json', 'checks.json', 'example_provenance.json',
           'normal.csv', 'unique.csv', 'ambiguous.csv', 'no_match.csv')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def unique_json(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def read_json(path):
    with path.open() as f:
        return json.load(f, object_pairs_hook=unique_json)


def pinned_module(root, relative, expected):
    p = (root / relative).resolve()
    require(p.is_relative_to(root) and p.is_file(), f'missing helper: {relative}')
    source = p.read_bytes()
    require(hashlib.sha256(source).hexdigest() == expected, f'helper SHA mismatch: {relative}')
    module = types.ModuleType('_verified_' + p.stem)
    module.__file__ = str(p)
    exec(compile(source, str(p), 'exec'), module.__dict__)
    return module


def contract_inputs(root, full=False):
    helper = pinned_module(root, HELPER, HELPER_SHA)
    e = helper.Evidence(root)
    e.check(HELPER, HELPER_SHA)
    freeze = read_json(e.check(CONTRACT_DIR + '/freeze.json', CONTRACT_FREEZE_SHA))
    require(freeze['status'] == 'PASS' and freeze['contract_status'] == 'FROZEN', '11E-1A freeze status')
    expected_names = {'contract.json', 'observation_schema.json', 'reference_suite.json', 'mentor_scope.md'}
    require(set(freeze['outputs']) == expected_names, '11E-1A output set')
    for name, record in freeze['outputs'].items():
        require(record['path'] == CONTRACT_DIR + '/' + name, '11E-1A output path')
        e.link(record)
    contract = read_json(e.path(CONTRACT_DIR + '/contract.json'))
    schema = read_json(e.path(CONTRACT_DIR + '/observation_schema.json'))
    suite = read_json(e.path(CONTRACT_DIR + '/reference_suite.json'))
    require(contract['stage'] == '11E-1A' and contract['status'] == 'PASS', 'task contract')
    for key in ('catalog_and_checker_implementation', 'development_catalog_construction', 'synthetic_and_catalog_replay_checks'):
        require(contract['authorizations'][key] is True, f'not authorized: {key}')
    for key in ('old_model_changes', 'new_model_training', 'new_simulation_campaign', 'validation_sample_access', 'holdout_access'):
        require(contract['authorizations'][key] is False, f'unexpected contract scope: {key}')
    require(schema['header_exact'] == HEADER and schema['rows_exact'] == STEPS, 'observation schema')
    require(suite['vector_count'] == STEPS and len(suite['vectors']) == STEPS, 'test suite size')
    require(suite['suite_commitment'] == hashlib.sha256(encoded(suite['vectors'])).hexdigest(), 'suite commitment')
    require(schema['suite_commitment'] == contract['measurement_protocol']['suite_commitment'] == suite['suite_commitment'],
            'suite/schema/contract agreement')
    require([v['vector_slot'] for v in suite['vectors']] == list(range(STEPS)), 'suite slot order')
    report = None
    if full:
        report, old, models, refs = helper.verify_inputs(e, helper.REPORT)
        for name, data in helper.documents(report, old, models, refs).items():
            require(e.path(CONTRACT_DIR + '/' + name).read_bytes() == data, f'contract regeneration: {name}')
        for path, record in freeze['input_evidence'].items():
            require(e.records.get(path) == record, f'11E-1A input binding: {path}')
    return helper, e, contract, suite, report


class ObservationError(ValueError):
    pass


def obs_require(ok, message):
    if not ok:
        raise ObservationError(message)


def read_observations(stream, suite):
    reader = csv.DictReader(stream)
    obs_require(reader.fieldnames == HEADER,
                'Expected exactly: ' + ','.join(HEADER) + '. Injection identity/control columns are forbidden.')
    result = bytearray()
    count = 0
    for row in reader:
        obs_require(count < STEPS, 'extra observation rows')
        obs_require(None not in row and all(v is not None for v in row.values()), 'malformed CSV row')
        for key in ('vector_slot', 'vector_id', 'cycles', 'timed_out'):
            obs_require(re.fullmatch(r'[0-9]+', row[key]) is not None, 'invalid unsigned integer: ' + key)
        slot, vid, cycles, timeout = (int(row[k]) for k in ('vector_slot', 'vector_id', 'cycles', 'timed_out'))
        obs_require(slot == count and vid == suite['vectors'][count]['vector_id'], 'wrong, duplicate or out-of-order test vector')
        obs_require(0 <= cycles <= 2000 and timeout in (0, 1), 'invalid latency/timeout')
        obs_require(not timeout or cycles == 2000, 'timeout must occur at cycle 2000')
        digest = row['actual_digest']
        valid_digest = re.fullmatch('[0-9a-fA-F]{64}', digest) is not None
        obs_require(valid_digest or (timeout == 1 and digest == ''), 'invalid/missing digest; X/Z values are not accepted')
        actual = bytes(32) if timeout else bytes.fromhex(digest)
        result.extend(struct.pack('>BH', timeout, cycles) + actual)
        count += 1
    obs_require(count == STEPS, 'missing observations: exactly 64 tests are required')
    return bytes(result)


def golden_profile(suite):
    return b''.join(struct.pack('>BH', 0, v['expected_cycles']) + bytes.fromhex(v['expected_digest'])
                    for v in suite['vectors'])


def profile_csv(profile, suite):
    require(len(profile) == STEPS * WIDTH, 'profile width')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(HEADER)
    for slot, v in enumerate(suite['vectors']):
        piece = profile[slot * WIDTH:(slot + 1) * WIDTH]
        timeout, cycles = struct.unpack('>BH', piece[:3])
        writer.writerow((slot, v['vector_id'], '' if timeout else piece[3:].hex(), cycles, timeout))
    return stream.getvalue()


def open_catalog(path):
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    conn.execute('PRAGMA query_only=ON')
    conn.execute('PRAGMA trusted_schema=OFF')
    return conn


def classify(conn, profile, suite):
    require(len(profile) == STEPS * WIDTH, 'diagnosis profile width')
    normal = golden_profile(suite)
    deviations = []
    for slot in range(STEPS):
        p, g = profile[slot * WIDTH:(slot + 1) * WIDTH], normal[slot * WIDTH:(slot + 1) * WIDTH]
        timeout, cycles = struct.unpack('>BH', p[:3])
        reasons = []
        if timeout:
            reasons.append('TIMEOUT')
        elif p[3:] != g[3:]:
            reasons.append('DIGEST_MISMATCH')
        if cycles != suite['vectors'][slot]['expected_cycles']:
            reasons.append('LATENCY_MISMATCH')
        if reasons:
            deviations.append({'vector_slot': slot, 'vector_id': suite['vectors'][slot]['vector_id'], 'reasons': reasons})
    # SHA narrows the search; full profile bytes decide equality.
    matching = []
    for pid, saved in conn.execute('SELECT profile_id, response FROM profiles WHERE profile_sha256=?',
                                   (hashlib.sha256(profile).hexdigest(),)):
        if bytes(saved) == profile:
            matching.append(pid)
    require(len(matching) <= 1, 'catalog contains duplicate exact profiles')
    members = []
    if matching:
        members = [k for (k,) in conn.execute('SELECT fault_instance FROM faults WHERE profile_id=? ORDER BY fault_instance',
                                             (matching[0],))]
    sites = sorted({k // 2 + 1 for k in members})
    if not deviations:
        status, localization = 'NO_OBSERVED_ANOMALY', 'NOT_IDENTIFIABLE'
    elif not members:
        status = localization = 'NO_CATALOG_MATCH'
    elif len(sites) == 1:
        status = localization = 'UNIQUE_SITE_IN_CATALOG'
    else:
        status = localization = 'AMBIGUOUS_SITES_IN_CATALOG'
    return {
        'result': status, 'observable_anomaly': bool(deviations), 'localization_status': localization,
        'no_fault_compatible': not deviations, 'candidate_site_count': len(sites),
        'candidate_fault_instance_count': len(members),
        'candidate_sites': [f'HMAC-STEM-{s:06d}' for s in sites],
        'candidate_fault_instances': [f'HMAC-STEM-{k // 2 + 1:06d}:SA{k % 2}' for k in members],
        'deviating_test_count': len(deviations), 'deviating_tests': deviations,
        'observed_profile_sha256': hashlib.sha256(profile).hexdigest(),
        'suite_commitment': suite['suite_commitment'], 'algorithm': VERSION,
        'model_training_or_inference': False, 'candidate_order': 'Ascending site ID; not a confidence ranking.',
        'limitation': 'Finite same-circuit, fixed-test, single-SA-fault catalog. Normal behavior does not prove fault absence; a unique catalog match does not prove physical defect cause.',
    }


def write_catalog(path, groups, suite):
    conn = sqlite3.connect(path)
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        conn.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE profiles (profile_id INTEGER PRIMARY KEY, profile_sha256 TEXT NOT NULL, response BLOB NOT NULL);
            CREATE INDEX profile_lookup ON profiles(profile_sha256);
            CREATE TABLE faults (fault_instance INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL REFERENCES profiles(profile_id));
            CREATE INDEX fault_lookup ON faults(profile_id);
        ''')
        conn.executemany('INSERT INTO metadata VALUES (?,?)', [('version', VERSION), ('suite_commitment', suite['suite_commitment'])])
        for pid, (profile, members) in enumerate(groups.items()):
            require(len(profile) == STEPS * WIDTH, 'catalog profile width')
            conn.execute('INSERT INTO profiles VALUES (?,?,?)', (pid, hashlib.sha256(profile).hexdigest(), profile))
            conn.executemany('INSERT INTO faults VALUES (?,?)', [(k, pid) for k in members])
        conn.commit()
        require(conn.execute('PRAGMA integrity_check').fetchone() == ('ok',), 'SQLite integrity')
        require(not conn.execute('PRAGMA foreign_key_check').fetchall(), 'SQLite membership integrity')
    finally:
        conn.close()


def absent_profile(conn, suite):
    normal = golden_profile(suite)
    count = conn.execute('SELECT COUNT(*) FROM profiles').fetchone()[0]
    for n in range(count + 2):
        digest = n.to_bytes(32, 'big')
        if digest == normal[3:WIDTH]:
            continue
        candidate = normal[:3] + digest + normal[WIDTH:]
        if classify(conn, candidate, suite)['result'] == 'NO_CATALOG_MATCH':
            return candidate
    raise ValueError('could not construct synthetic no-match test')


def intake_checks(suite):
    base = profile_csv(golden_profile(suite), suite)
    lines = base.splitlines()
    bad = [base.replace(lines[0], lines[0] + ',site_id', 1),
           '\n'.join(lines[:-1]) + '\n', base + lines[-1] + '\n',
           '\n'.join([lines[0], lines[1], lines[1]] + lines[3:]) + '\n']
    for column, value in ((1, '999999'), (2, 'x' * 64), (2, ''), (3, '-1'), (3, '2001'), (4, '2'), (4, '1')):
        row = lines[1].split(','); row[column] = value
        bad.append('\n'.join([lines[0], ','.join(row)] + lines[2:]) + '\n')
    for data in bad:
        try:
            read_observations(io.StringIO(data), suite)
        except ObservationError:
            pass
        else:
            raise ValueError('invalid observation fixture accepted')
    p = struct.pack('>BH', 1, 2000) + bytes(32) + golden_profile(suite)[WIDTH:]
    timeout_csv = profile_csv(p, suite)
    changed = timeout_csv.splitlines()
    row = changed[1].split(','); row[2] = 'ab' * 32; changed[1] = ','.join(row)
    require(read_observations(io.StringIO(timeout_csv), suite) ==
            read_observations(io.StringIO('\n'.join(changed) + '\n'), suite) == p, 'timeout digest invariance')
    return {'invalid_intake_cases_rejected': len(bad), 'timeout_digest_invariance': 'PASS',
            'fixture_origin': 'SYNTHETIC SOFTWARE TESTS; NOT CIRCUIT EXPERIMENTS'}


def check_catalog(path, groups, suite):
    conn = open_catalog(path)
    examples = {'normal': (golden_profile(suite), None)}
    outcomes = Counter()
    try:
        for i, (profile, members) in enumerate(groups.items(), 1):
            csv_text = profile_csv(profile, suite)
            observed = read_observations(io.StringIO(csv_text), suite)
            require(observed == profile, 'CSV normalization changed a source profile')
            answer = classify(conn, observed, suite)
            labels = [f'HMAC-STEM-{k // 2 + 1:06d}:SA{k % 2}' for k in sorted(members)]
            require(answer['candidate_fault_instances'] == labels, 'candidate-set mismatch')
            nsites = len({k // 2 for k in members})
            expected = ('NO_OBSERVED_ANOMALY' if profile == golden_profile(suite)
                        else 'UNIQUE_SITE_IN_CATALOG' if nsites == 1 else 'AMBIGUOUS_SITES_IN_CATALOG')
            require(answer['result'] == expected, 'result semantics mismatch')
            require(answer == classify(conn, observed, suite), 'deterministic diagnosis replay')
            outcomes[answer['result']] += len(members)
            if expected in ('UNIQUE_SITE_IN_CATALOG', 'AMBIGUOUS_SITES_IN_CATALOG'):
                name = 'unique' if expected == 'UNIQUE_SITE_IN_CATALOG' else 'ambiguous'
                if name not in examples or min(members) < examples[name][1]:
                    examples[name] = (profile, min(members))
            if i % 2500 == 0:
                print(f'  Verified {i:,} distinct response profiles', flush=True)
        require({'normal', 'unique', 'ambiguous'} <= examples.keys(), 'required example categories absent')
        examples['no_match'] = (absent_profile(conn, suite), None)
        normal_answer = classify(conn, examples['normal'][0], suite)
        require(normal_answer['result'] == 'NO_OBSERVED_ANOMALY' and normal_answer['no_fault_compatible'], 'normal rule')
        require(normal_answer['localization_status'] == 'NOT_IDENTIFIABLE', 'normal localization rule')
        checks = {'status': 'PASS', 'distinct_profiles_checked': len(groups),
                  'fault_memberships_checked': sum(map(len, groups.values())),
                  'fault_instance_results': dict(outcomes), 'exact_candidate_set_agreement': 'PASS',
                  'deterministic_diagnosis_replay': 'PASS', 'normal_reference_false_alarms': 0,
                  'synthetic_no_match': 'PASS', 'accuracy_claim': 'NONE; CATALOG CONSISTENCY ONLY',
                  **intake_checks(suite)}
        return checks, examples
    finally:
        conn.close()


def compare_memberships(e, groups, report):
    by_hash = {}
    for profile, members in groups.items():
        digest = hashlib.sha256(profile).hexdigest()
        require(digest not in by_hash, 'response digest collision in audit identifier; exact profiles remain distinct')
        by_hash[digest] = members
    count = 0
    with e.path(report['profile_groups']['path']).open() as f:
        for line in f:
            row = json.loads(line, object_pairs_hook=unique_json)
            members = by_hash.pop(row['profile_sha256'], None)
            require(members is not None, 'profile absent from audited groups')
            actual = [f'HMAC-STEM-{k // 2 + 1:06d}:SA{k % 2}' for k in members]
            require(actual == row['fault_instances'], 'reconstructed membership differs from audited group')
            count += len(members)
    require(not by_hash and count == 45678, 'profile audit coverage')


def verify_bundle(e, suite):
    dest = e.path(DEST)
    audit = read_json(dest / 'freeze.json')
    require(audit['stage'] == '11E-1B' and audit['status'] == 'PASS' and audit['version'] == VERSION, 'baseline freeze status')
    require(audit['source_sha256'] == sha(Path(__file__).resolve()), 'baseline source changed since build')
    require(audit['contract_freeze_sha256'] == CONTRACT_FREEZE_SHA and audit['suite_commitment'] == suite['suite_commitment'],
            'baseline/contract binding')
    require(set(audit['outputs']) == set(OUTPUTS), 'baseline output set')
    for name, record in audit['outputs'].items():
        require(record['path'] == DEST + '/' + name, 'baseline output path')
        e.link(record)
    for key in ('model_training_performed', 'old_models_changed', 'simulation_performed',
                'validation_or_holdout_samples_opened', 'project_final_freeze_declared'):
        require(audit[key] is False, 'baseline scope changed: ' + key)
    checks = read_json(dest / 'checks.json')
    require(checks['status'] == 'PASS' and checks['fault_memberships_checked'] == 45678, 'baseline checks')
    conn = open_catalog(dest / 'catalog.sqlite3')
    try:
        require(dict(conn.execute('SELECT key,value FROM metadata')) ==
                {'version': VERSION, 'suite_commitment': suite['suite_commitment']}, 'catalog metadata')
        require(conn.execute('SELECT COUNT(*) FROM faults').fetchone()[0] == 45678, 'catalog membership count')
    finally:
        conn.close()
    return audit


def build(root):
    source_sha = sha(Path(__file__).resolve())
    helper, e, contract, suite, report = contract_inputs(root, full=True)
    dest = e.path(DEST)
    parent = e.path(PARENT)
    with (parent / 'baseline_11e1b.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another baseline builder is active') from None
        if dest.exists():
            audit = verify_bundle(e, suite)
            e.recheck()
            print('EXISTING 11E-1B BUNDLE VERIFIED; no files overwritten.')
        else:
            auditor = pinned_module(root, AUDITOR, AUDITOR_SHA)
            e.check(AUDITOR, AUDITOR_SHA)
            print('\nRECONSTRUCTING COMPLETE DEVELOPMENT RESPONSE PROFILES', flush=True)
            groups, normal, refs, rows = auditor.scan(e.path(helper.TRAIN))
            require(normal == golden_profile(suite), 'normal response binding')
            require(rows == contract['catalog_scope']['canonical_rows'], 'source row count')
            for v in suite['vectors']:
                require(refs[v['vector_slot']] == (v['vector_id'], bytes.fromhex(v['expected_digest']), v['expected_cycles']),
                        'reference vector binding')
            summary = auditor.summarize(groups, normal, 22839, STEPS)
            for key, value in summary.items():
                wanted = report[key]
                require(json.loads(json.dumps(value)) == wanted, 'observability summary disagreement: ' + key)
            compare_memberships(e, groups, report)
            with tempfile.TemporaryDirectory(prefix='11e1b_pending_', dir=parent) as pending:
                temp = Path(pending)
                print('\nBUILDING AND CHECKING RESPONSE CATALOG', flush=True)
                write_catalog(temp / 'catalog.sqlite3', groups, suite)
                checks, examples = check_catalog(temp / 'catalog.sqlite3', groups, suite)
                (temp / 'summary.json').write_bytes(encoded(summary))
                (temp / 'checks.json').write_bytes(encoded(checks))
                provenance = {'warning': 'Stored catalog examples and a synthetic no-match case. No live simulation or independent accuracy evaluation.',
                              'selection': 'Lowest fault-instance index in each observable category; normal uses golden reference.', 'examples': {}}
                for name, (profile, k) in examples.items():
                    (temp / (name + '.csv')).write_text(profile_csv(profile, suite))
                    provenance['examples'][name] = {'origin': 'SYNTHETIC SOFTWARE TEST' if name == 'no_match'
                        else 'GOLDEN REFERENCE' if name == 'normal' else 'EXISTING DEVELOPMENT CATALOG PROFILE',
                        'source_fault_instance_for_fixture_audit_only': None if k is None else f'HMAC-STEM-{k // 2 + 1:06d}:SA{k % 2}'}
                (temp / 'example_provenance.json').write_bytes(encoded(provenance))
                e.recheck()
                require(sha(Path(__file__).resolve()) == source_sha, 'baseline source changed during build')
                records = {name: {'path': DEST + '/' + name, 'sha256': sha(temp / name), 'bytes': (temp / name).stat().st_size}
                           for name in OUTPUTS}
                audit = {'stage': '11E-1B', 'version': VERSION, 'status': 'PASS', 'catalog_status': 'FROZEN',
                    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_sha,
                    'contract_freeze_sha256': CONTRACT_FREEZE_SHA, 'suite_commitment': suite['suite_commitment'],
                    'input_evidence': e.records, 'outputs': records,
                    'model_training_performed': False, 'old_models_changed': False, 'simulation_performed': False,
                    'validation_or_holdout_samples_opened': False, 'project_final_freeze_declared': False,
                    'original_validation_target': 'NOT_MET; PRESERVED', 'accuracy_claim': 'NONE; SOFTWARE AND CATALOG CONSISTENCY',
                    'next_step': '11E-1C — Blinded Observation Capture and Diagnosis Demonstration Contract'}
                (temp / 'freeze.json').write_bytes(encoded(audit))
                require(not dest.exists(), 'output already exists; no overwrite')
                os.rename(temp, dest)
    print('\nSTAGE 11E-1B — RESPONSE CATALOG AND BEHAVIOR DETECTOR/LOCATOR BASELINE')
    print('Status                         : PASS')
    print('Reference fault instances      : 45,678')
    print('Normal-compatible faults       : 22,748')
    print('Unique-site fault instances    : 9,171')
    print('Ambiguous-site fault instances : 13,759')
    print('Exact candidate-set checks     : PASS')
    print('Unknown identity in query      : NOT REQUIRED / FORBIDDEN')
    print('New ML model trained           : NO')
    print('Independent accuracy measured  : NO')
    print('Existing experiment changed    : NO')
    print(f'Catalog bundle                 : {dest}')
    print(f'Audit SHA256                   : {sha(dest / "freeze.json")}')
    print('Next: 11E-1C — Blinded Observation Capture and Diagnosis Demonstration Contract')


def show(answer):
    print('Result                         : ' + answer['result'])
    print('Tests with observed errors     : ' + str(answer['deviating_test_count']) + '/64')
    print('Candidate physical sites       : ' + str(answer['candidate_site_count']))
    print('Candidate fault instances      : ' + str(answer['candidate_fault_instance_count']))
    if answer['no_fault_compatible']:
        print('Normal-compatible response; an undetected fault may still be present.')
    for site in answer['candidate_sites'][:10]:
        print('  ' + site)
    if answer['candidate_site_count'] > 10:
        print('  ... first 10 shown in ID order, not ranked. Full candidates are in the JSON result.')
    if answer['result'] == 'NO_CATALOG_MATCH':
        print('Anomaly observed, but no known fault profile matches. Location unresolved.')


def diagnose_command(root, observations=None, demo=False):
    helper, e, contract, suite, _ = contract_inputs(root)
    audit = verify_bundle(e, suite)
    dest = e.path(DEST)
    conn = open_catalog(dest / 'catalog.sqlite3')
    outputs = {}
    if demo:
        print('SAVED RESPONSE DEMONSTRATION — NO LIVE CIRCUIT RUN OR INDEPENDENT ACCURACY')
        print('Three reference/catalog examples plus one explicitly synthetic no-match test.')
        inputs = [(name, DEST + '/' + name + '.csv') for name in ('normal', 'unique', 'ambiguous', 'no_match')]
    else:
        inputs = [('observation', observations)]
    try:
        for name, relative in inputs:
            path = e.path(relative)
            if not demo:
                intake_dir = root / 'observations/hmac_behavior_11e1b'
                require(path.is_relative_to(intake_dir),
                        'Place new observation CSVs under observations/hmac_behavior_11e1b/; frozen datasets are not query inputs')
            with path.open('rb') as f:
                data = f.read(262145)
            obs_require(len(data) <= 262144, 'observation CSV exceeds 256 KiB')
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                raise ObservationError('observation CSV must be UTF-8 text') from None
            profile = read_observations(io.StringIO(text, newline=''), suite)
            answer = classify(conn, profile, suite)
            answer['observation_csv_sha256'] = hashlib.sha256(data).hexdigest()
            answer['input_origin'] = ('STORED DEMONSTRATION; NO NEW MEASUREMENT' if demo
                                      else 'USER-SUPPLIED OBSERVATIONS; MEASUREMENT ORIGIN NOT INDEPENDENTLY VERIFIED')
            outputs[name] = answer
            print('\n' + name.upper())
            show(answer)
        e.recheck()
    finally:
        conn.close()
    parent = e.path(PARENT + '/diagnosis_results_11e1b')
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', prefix='demo_' if demo else 'diagnosis_', suffix='.json', dir=parent, delete=False) as f:
        json.dump({'catalog_freeze_sha256': sha(dest / 'freeze.json'), 'results': outputs}, f, indent=2, sort_keys=True)
        f.write('\n'); result_path = Path(f.name)
    print(f'\nComplete result JSON: {result_path}')
    print('No neural model was run. HOLDOUT and validation samples were not accessed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='build/check the catalog from the frozen TRAIN campaign')
    sub.add_parser('demo', help='show stored catalog examples and a synthetic no-match case')
    query = sub.add_parser('diagnose', help='diagnose a complete 64-row observation CSV')
    query.add_argument('--observations', required=True,
                       help='CSV under observations/hmac_behavior_11e1b/; no fault identity columns')
    args = parser.parse_args()
    try:
        root = args.project_root.resolve()
        if args.command == 'build':
            build(root)
        else:
            diagnose_command(root, getattr(args, 'observations', None), args.command == 'demo')
        return 0
    except ObservationError as error:
        print(json.dumps({'result': 'INVALID_OBSERVATIONS', 'reason': str(error), 'diagnosis_performed': False}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('\nStopped; original experiment unchanged.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'STOP: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
