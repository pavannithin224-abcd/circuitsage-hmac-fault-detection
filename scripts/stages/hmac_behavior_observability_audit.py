#!/usr/bin/env python3
"""Audit whether existing HMAC responses distinguish fault locations.

Exploratory analysis for a new behavior-based diagnosis task. Not a trained
detector, localization accuracy measurement, or project freeze. Uses only the
already-used TRAIN campaign. Does not open validation/HOLDOUT or model files.
Run from ~/vlsi_fault_detection_v2 with Python 3.9+; standard library only.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import tempfile

SOURCE = ('results/hmac_fault_campaign_11c5/canonical_dataset_11c5c/'
          'hmac_fault_campaign_canonical_train_11c5c.csv.gz')
SOURCE_SHA = 'dcc79c94fa3a8d592b516e5f5683b1bf007682745103a9419ee20ef20a7ec78f'
HEADER = ('record_id,vector_split,batch_id,run_type,site_selection,selector,site_id,'
          'stuck_value,vector_slot,vector_id,fault_enable,cycles,baseline_cycles,'
          'latency_delta,timed_out,activity,detected,unknown,digest_hamming_distance,'
          'expected_digest,actual_digest').split(',')
BASELINE_CYCLES = 343
TIMEOUT_CYCLES = 2000
TOKEN_BYTES = 35


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_bytes(value):
    require(re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None, 'invalid digest')
    return bytes.fromhex(value)


def observation(actual, cycles, timed_out):
    """Only observable behavior; a digest after a timeout is not valid output."""
    require(timed_out in (0, 1) and 0 <= cycles <= TIMEOUT_CYCLES,
            'invalid observation timing')
    if timed_out:
        require(cycles == TIMEOUT_CYCLES, 'inconsistent timeout capture')
        actual = bytes(32)
    return struct.pack('>BH', timed_out, cycles) + actual


def scan(path, sites=22839, vectors=64, batch_size=512, progress=True):
    faults = sites * 2
    batches = (sites + batch_size - 1) // batch_size
    profiles = [bytearray(vectors * TOKEN_BYTES) for _ in range(faults)]
    coverage = [0] * faults
    baseline_seen = set()
    refs = {}
    baseline_profiles = {}
    total = 0
    with gzip.open(path, 'rt', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == HEADER, 'unexpected canonical header')
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), 'malformed row')
            require(int(row['record_id']) == total, 'record order/gap')
            total += 1
            require(row['vector_split'] == 'TRAIN', 'only existing TRAIN records permitted')
            slot, batch = int(row['vector_slot']), int(row['batch_id'])
            require(0 <= slot < vectors and 0 <= batch < batches, 'slot/batch out of range')
            expected = digest_bytes(row['expected_digest'])
            ref = (int(row['vector_id']), expected, int(row['baseline_cycles']))
            require(ref[2] == BASELINE_CYCLES, 'baseline latency contract')
            require(refs.setdefault(slot, ref) == ref, 'reference differs for same test vector')
            require(int(row['unknown']) == 0, 'unknown simulator value; cannot audit')
            cycles, timeout = int(row['cycles']), int(row['timed_out'])
            token = observation(digest_bytes(row['actual_digest']), cycles, timeout)
            if row['run_type'] == 'BASELINE':
                require(int(row['fault_enable']) == 0, 'baseline injection enabled')
                require((batch, slot) not in baseline_seen, 'duplicate baseline')
                baseline_seen.add((batch, slot))
                require(token == observation(expected, BASELINE_CYCLES, 0), 'baseline failure')
                require(baseline_profiles.setdefault(slot, token) == token, 'baseline differs across batches')
            else:
                require(row['run_type'] == 'ENABLED' and int(row['fault_enable']) == 1, 'row type')
                require(re.fullmatch(r'HMAC-STEM-[0-9]{6}', row['site_id']) is not None, 'site label format')
                site = int(row['site_id'].split('-')[-1]) - 1
                stuck = int(row['stuck_value'])
                require(0 <= site < sites and stuck in (0, 1), 'fault label range')
                require(batch == site // batch_size and int(row['selector']) == site % batch_size,
                        'site/batch binding')
                require(int(row['site_selection']) == site % batch_size, 'site selection binding')
                k = site * 2 + stuck
                require(not coverage[k] & (1 << slot), 'duplicate fault/vector observation')
                coverage[k] |= 1 << slot
                profiles[k][slot * TOKEN_BYTES:(slot + 1) * TOKEN_BYTES] = token
            if progress and total % 250000 == 0:
                print(f'  Processed {total:,} rows', flush=True)
    require(total == (faults + batches) * vectors, 'canonical row count')
    require(len(baseline_seen) == batches * vectors, 'baseline coverage')
    require(all(mask == (1 << vectors) - 1 for mask in coverage), 'missing fault/vector observations')
    require(len({ref[0] for ref in refs.values()}) == vectors, 'duplicate vector identity')
    normal = b''.join(baseline_profiles[slot] for slot in range(vectors))
    groups = {}
    for k in range(faults):
        profile = bytes(profiles[k])
        profiles[k] = None
        groups.setdefault(profile, []).append(k)
    return groups, normal, refs, total


def summarize(groups, normal, sites, vectors):
    invisible = len(groups.get(normal, []))
    unique_fault = unique_site_instances = ambiguous_instances = 0
    histogram = Counter()
    largest = 0
    for profile, members in groups.items():
        if profile == normal:
            continue
        nsites = len({k // 2 for k in members})
        histogram[nsites] += 1
        largest = max(largest, nsites)
        unique_fault += int(len(members) == 1)
        if nsites == 1:
            unique_site_instances += len(members)
        else:
            ambiguous_instances += len(members)
    require(invisible + unique_site_instances + ambiguous_instances == sites * 2, 'classification totals')
    return {
        'physical_sites': sites, 'fault_instances': sites * 2, 'test_vectors': vectors,
        'distinct_fault_response_profiles': len(groups),
        'fault_instances_indistinguishable_from_no_fault': invisible,
        'fault_instances_with_observable_effect': sites * 2 - invisible,
        'fault_instances_with_unique_exact_site_within_catalog': unique_site_instances,
        'fault_instances_ambiguous_across_sites': ambiguous_instances,
        'unique_exact_fault_instance_profiles_excluding_no_fault': unique_fault,
        'largest_observable_candidate_site_set': largest,
        'observable_profile_count_by_candidate_site_count': dict(sorted(histogram.items())),
        'interpretation': 'Structural distinguishability for this finite test suite and fault catalog; NOT predictive accuracy.',
    }


def run(root):
    source = (root / SOURCE).resolve()
    require(source.is_relative_to(root) and source.is_file(), f'missing project dataset: {SOURCE}')
    print('HMAC BEHAVIOR-BASED DIAGNOSIS: OBSERVABILITY AUDIT', flush=True)
    print('Checking the frozen development dataset...', flush=True)
    require(sha256(source) == SOURCE_SHA, 'source SHA mismatch; preserve the frozen dataset')
    groups, normal, refs, rows = scan(source)
    stats = summarize(groups, normal, 22839, 64)
    require(stats['fault_instances_with_observable_effect'] == 22930, 'disagrees with development detection freeze')
    require(sha256(source) == SOURCE_SHA, 'source changed during analysis')
    report = {
        'audit_status': 'COMPLETED', 'experiment_status': 'EXPLORATORY; NO MODEL TRAINED',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'path': SOURCE, 'sha256': SOURCE_SHA},
        'auditor_sha256': sha256(Path(__file__).resolve()),
        'canonical_rows': rows, **stats,
        'observation_inputs': ['actual_digest_when_not_timed_out', 'cycles', 'timed_out'],
        'known_test_context': 'Fixed ordered 64-vector suite, with golden reference responses.',
        'grouping_labels_only': ['site_id', 'stuck_value'],
        'excluded_from_response_profiles': ['site_id', 'stuck_value', 'selector', 'batch_id',
            'fault_enable', 'site_selection', 'activity', 'detected', 'record_id'],
        'profile_comparison': 'Exact bytes across all test vectors; SHA is for report identifiers only.',
        'timeout_handling': 'Invalid/incomplete digest ignored at timeout; duration and timeout retained.',
        'training_performed': False, 'model_files_opened': False,
        'validation_or_holdout_opened': False, 'project_final_freeze_declared': False,
        'limitations': [
            'No observable error does not establish absence of a physical fault.',
            'Equal profiles cannot reveal a unique site using these observations alone.',
            'Unique catalog profiles are not proof of accuracy on unseen measurements.',
            'All fault instances, including earlier site-test groups, are included in this descriptive audit; none is a new test set.',
            'Assumes one persistent SA0/SA1 fault, identical reset/test protocol, and this one frozen circuit.',
        ],
        'next_work': ['Freeze a separate behavior-based task, observation protocol and independent evaluation plan.',
            'Implement a golden-reference monitor and a fault-signature lookup baseline.',
            'If ambiguous, study additional tests or physically available observation points before model training.'],
    }
    parent = (root / 'results/hmac_behavior_diagnosis_pilot').resolve()
    require(parent.is_relative_to(root), 'output folder escapes project')
    parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='observability_', dir=parent))
    # Store memberships only, not raw digest payloads. All outputs are new files.
    membership_path = attempt / 'response_profile_groups.jsonl'
    with membership_path.open('x') as f:
        for profile, members in groups.items():
            f.write(json.dumps({'profile_sha256': hashlib.sha256(profile).hexdigest(),
                'matches_no_fault': profile == normal,
                'fault_instances': [f'HMAC-STEM-{k // 2 + 1:06d}:SA{k % 2}' for k in members]},
                separators=(',', ':')) + '\n')
    report['profile_groups'] = {'path': str(membership_path.relative_to(root)), 'sha256': sha256(membership_path)}
    report_path = attempt / 'observability_report.json'
    with report_path.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write('\n')
    print('\nAUDIT COMPLETED — NOT A MODEL ACCURACY RESULT')
    for key in ('fault_instances', 'fault_instances_with_observable_effect',
                'fault_instances_indistinguishable_from_no_fault',
                'fault_instances_with_unique_exact_site_within_catalog',
                'fault_instances_ambiguous_across_sites', 'largest_observable_candidate_site_set'):
        print(f'{key}: {stats[key]:,}')
    print(f'Report: {report_path}\nReport SHA256: {sha256(report_path)}')
    print('Model training: NOT PERFORMED. Validation/HOLDOUT access: NONE.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        run(args.project_root.resolve())
        return 0
    except KeyboardInterrupt:
        print('\nStopped; no project freeze declared.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'STOP: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
