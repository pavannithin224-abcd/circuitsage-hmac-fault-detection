#!/usr/bin/env python3
"""Prepare private demonstration cases and authorize bounded capture. No simulation.
Standard library only, Python 3.9+, WSL/Linux. Run from the project root.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import sys
import types

RUNNER = 'stage_11e1d_blinded_runner.py'
RUNNER_SHA = '4fdfeb3ae57c28c9884e403a0fdc36950493d0f546cbe5c1c8b62ca5d77d6919'
PREFLIGHT = 'results/hmac_behavior_diagnosis_pilot/preflight_11e1d_ow_4gax4'
PREFLIGHT_SHA = '4763eafc22456000a970414060633a745a90e208c1deec621a1cec8a9637961a'
DEST = 'results/hmac_behavior_diagnosis_pilot/private_case_preparation_11e1e'
VERSION = 'HMAC-PRIVATE-CASE-AUTHORIZATION-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_once(path, data):
    if path.exists():
        require(path.read_bytes()==data, 'Existing artifact differs; no overwrite: '+path.name)
    else:
        fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())


def load_runner(root):
    path=(root/RUNNER).resolve()
    require(path.is_relative_to(root), 'Runner escapes project')
    source=path.read_bytes()
    require(sha_bytes(source)==RUNNER_SHA, 'Repaired 11E-1D runner SHA mismatch')
    d=types.ModuleType('_verified_blinded_runner');d.__file__=str(path)
    exec(compile(source,str(path),'exec'), d.__dict__)
    return d


def check_preflight(b,e):
    audit=b.read_json(e.check(PREFLIGHT+'/freeze.json',PREFLIGHT_SHA))
    require(audit['stage']=='11E-1D' and audit['status']=='PASS' and audit['preflight_status']=='FROZEN', 'Preflight status')
    require(audit['source_sha256']==RUNNER_SHA, 'Preflight runner binding')
    for key in ('capture_lint','worker_exact_replay','worker_complete_candidate_agreement','synthetic_selection_replay'):
        require(audit[key]=='PASS', 'Preflight check: '+key)
    require(audit['access_denial_probes']==7 and audit['identity_input_rejected'] is True, 'Worker separation checks')
    for key in ('real_cases_selected','simulation_performed','model_training_performed','frozen_inputs_changed',
                'capture_execution_authorized','project_final_freeze_declared'):
        require(audit[key] is False, 'Preflight scope: '+key)
    require(audit['validation_holdout_samples_selected']==0 and
            audit['original_validation_target']=='NOT_MET; PRESERVED', 'Preflight scope/disposition')
    for path,record in audit['input_evidence'].items():
        require(e.records.get(path)==record, 'Preflight input binding: '+path)
    require({'lint.log','lint_capture.sv','worker_results.json'} <= set(audit['outputs']), 'Preflight outputs missing')
    for relative,record in audit['outputs'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Invalid preflight output path')
        e.check(PREFLIGHT+'/'+relative,record['sha256'],record['bytes'])


def validate_truth(truth,conn,b,suite):
    cases=truth['cases']
    require(len(cases)==12 and len({x['case_id'] for x in cases})==12,'Case count/IDs')
    require(all(re.fullmatch('[0-9a-f]{32}',x['case_id']) for x in cases),'Opaque IDs')
    require(Counter(x['selection_category'] for x in cases)=={'unique':4,'ambiguous':4,'normal':2,'fault_free':2},'Category counts')
    used=set();values=set();plan=[]
    normal=b.golden_profile(suite)
    for case in cases:
        if not case['fault_present']:
            require(case['selection_category']=='fault_free' and case['site_id'] is None and case['stuck_value'] is None,'Fault-free truth')
            bid,selector,forced=0,0,0
        else:
            require(re.fullmatch('HMAC-STEM-[0-9]{6}',case['site_id']) is not None,'Site label')
            site=int(case['site_id'].split('-')[-1])-1;forced=case['stuck_value']
            require(0<=site<22839 and type(forced) is int and forced in (0,1),'Fault bounds')
            k=2*site+forced;require(k not in used,'Duplicate injected fault');used.add(k);values.add(forced)
            rows=conn.execute('SELECT p.response FROM profiles p JOIN faults f ON p.profile_id=f.profile_id WHERE f.fault_instance=?',(k,)).fetchall()
            require(len(rows)==1,'Catalog fault membership')
            response=bytes(rows[0][0]);answer=b.classify(conn,response,suite)
            category='normal' if response==normal else 'unique' if answer['candidate_site_count']==1 else 'ambiguous'
            require(category==case['selection_category'],'Private case category mismatch')
            require(case['site_id'] in answer['candidate_sites'],'Truth absent from catalog candidates')
            bid,selector=divmod(site,512)
        plan.append({'case_id':case['case_id'],'batch_id':bid,'selector':selector,
                     'fault_enable':int(case['fault_present']),'forced_value':forced})
    require(len(used)==10 and values=={0,1},'Distinct faults / SA0 and SA1 coverage')
    return plan


def prepare(root):
    source_sha=sha_bytes(Path(__file__).read_bytes())
    d=load_runner(root)
    b,c,e,suite,_,batches,_=d.inputs(root)
    e.check(RUNNER,RUNNER_SHA)
    check_preflight(b,e)
    e.recheck()
    dest=e.path(DEST)
    binding={'runner_sha256':RUNNER_SHA,'preflight_sha256':PREFLIGHT_SHA,
             'catalog_freeze_sha256':c.BASE_SHA,'suite_commitment':suite['suite_commitment'],
             'python_version':platform.python_version()}
    with (dest.parent/'private_preparation_11e1e.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not dest.exists():
            dest.mkdir(mode=0o700)
            (dest/'private').mkdir(mode=0o700)
            # Persist entropy before selection. Failures/reruns never choose a fresh seed.
            entropy={'binding':binding,'selection_seed':secrets.token_hex(32),'salt_hex':secrets.token_hex(32)}
            write_once(dest/'private/entropy.json',encode(entropy))
        require(not dest.is_symlink() and (dest/'private/entropy.json').is_file(),
                'Incomplete preparation without entropy; preserve it for review, do not delete/reseed')
        require((dest.stat().st_mode & 0o077)==0 and ((dest/'private').stat().st_mode & 0o077)==0,
                'Private preparation permissions must be owner-only')
        entropy=b.read_json(dest/'private/entropy.json')
        require(entropy['binding']==binding,'Preparation inputs/environment changed; no reselection')
        require(all(re.fullmatch('[0-9a-f]{64}',entropy[k]) for k in ('selection_seed','salt_hex')),'Entropy format')
        conn=b.open_catalog(e.path(b.DEST+'/catalog.sqlite3'))
        try:
            truth=d.select_truth(conn,b,suite,entropy['selection_seed'],entropy['salt_hex'])
            require(truth==d.select_truth(conn,b,suite,entropy['selection_seed'],entropy['salt_hex']), 'Case selection replay')
            plan=validate_truth(truth,conn,b,suite)
        finally:conn.close()
        truth_bytes=encode(truth);commitment=sha_bytes(truth_bytes)
        write_once(dest/'private/truth.json',truth_bytes)
        private_plan={'binding':binding,'truth_commitment':commitment,'cases':plan,
                      'batch_netlists':{str(x['batch_id']):str(batches[x['batch_id']].relative_to(root)) for x in plan}}
        write_once(dest/'private/capture_plan.json',encode(private_plan))
        public={'version':VERSION,'suite_commitment':suite['suite_commitment'],
                'truth_commitment':commitment,'case_ids':[x['case_id'] for x in truth['cases']]}
        authorization={'stage':'11E-1E','version':VERSION,'status':'PASS','authorization_status':'FROZEN',
            'binding':binding,'truth_commitment':commitment,
            'public_case_manifest_sha256':sha_bytes(encode(public)),
            'capture_execution_authorized':True,'scope':'Only these 12 committed cases and the fixed 64 TRAIN vectors',
            'canonical_transactions':768,'replay_transactions':768,'total_transactions':1536,
            'execution':'SEQUENTIAL','build_jobs':1,'case_substitution_allowed':False,
            'prerequisites_for_execution':['Verify this authorization, private truth commitment, case IDs and frozen input hashes.',
                'Require the completed PASS preparation freeze.json and verify every bound artifact before accepting authorization.',
                'Commit truth before launching any simulator; match binary batch to private capture-plan batch.',
                'Write each fresh capture to a new attempt; preserve failures; never copy catalog responses.',
                'Keep truth, seed, category map, command lines and simulator logs outside the diagnosis allowlist.',
                'Use repaired frozen worker; commit all predictions before scoring reveals truth.',
                'Recheck frozen inputs and exact normalized canonical/replay agreement.'],
            'model_training_authorized':False,'validation_holdout_access_authorized':False,
            'original_validation_target':'NOT_MET; PRESERVED','project_final_freeze_declared':False,
            'measurement_claim':'Known-catalog workflow demonstration; not independent generalization accuracy',
            'next':'11E-1F — Blinded Capture, Diagnosis and Committed Scoring Execution'}
        write_once(dest/'public_case_manifest.json',encode(public))
        write_once(dest/'capture_authorization.json',encode(authorization))
        e.recheck();require(sha_bytes(Path(__file__).read_bytes())==source_sha,'Preparation source changed')
        names=['private/entropy.json','private/truth.json','private/capture_plan.json',
               'public_case_manifest.json','capture_authorization.json']
        outputs={name:{'path':DEST+'/'+name,'sha256':d.sha(dest/name),'bytes':(dest/name).stat().st_size} for name in names}
        audit={'stage':'11E-1E','version':VERSION,'status':'PASS','preparation_status':'FROZEN',
               'source_sha256':source_sha,'input_evidence':e.records,'outputs':outputs,
               'truth_commitment':commitment,'cases':12,'fault_free_cases':2,'injected_fault_cases':10,
               'unique_cases':4,'ambiguous_cases':4,'normal_compatible_cases':2,
               'selection_replay':'PASS','catalog_category_verification':'PASS','sa0_sa1_coverage':'BOTH',
               'truth_displayed':False,'capture_execution_authorized':True,'simulation_performed':False,
               'diagnosis_on_fresh_observations_performed':False,'model_training_performed':False,
               'frozen_inputs_changed':False,'original_validation_target':'NOT_MET; PRESERVED',
               'project_final_freeze_declared':False}
        # Timestamp is set once, so idempotent reruns preserve every output byte.
        if (dest/'freeze.json').exists():
            previous=b.read_json(dest/'freeze.json')
            audit['created_at_utc']=previous['created_at_utc']
        else:audit['created_at_utc']=datetime.now(timezone.utc).isoformat()
        write_once(dest/'freeze.json',encode(audit))
    print('STAGE 11E-1E — PRIVATE CASE PREPARATION AND CAPTURE AUTHORIZATION')
    print('Status                         : PASS')
    print('Private cases                  : FROZEN; identities not displayed')
    print('Cases                          : 12 (2 fault-free + 10 injected)')
    print('Unique / ambiguous / invisible : 4 / 4 / 2')
    print('Canonical / replay transactions : 768 / 768')
    print('Selection replay / SA coverage : PASS / BOTH')
    print('Bounded capture execution      : AUTHORIZED; NOT STARTED')
    print('Truth commitment               :',commitment)
    print('Preparation bundle             :',dest)
    print('Authorization SHA256           :',d.sha(dest/'capture_authorization.json'))
    print('Audit SHA256                   :',d.sha(dest/'freeze.json'))
    print('Next:',authorization['next'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    args=parser.parse_args()
    try:prepare(args.project_root.resolve())
    except KeyboardInterrupt:
        print('STOP: interrupted. Preserve the preparation directory; no capture started.')
        raise SystemExit(130)
    except Exception as error:
        print('STOP:',error)
        print('No capture started. Preserve existing private preparation files; do not delete/reseed.')
        raise SystemExit(1)
