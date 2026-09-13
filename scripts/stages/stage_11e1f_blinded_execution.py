#!/usr/bin/env python3
"""11E-1F: execute only the committed 12-case capture demonstration.
Run once; --resume skips verified completed captures. --status is read-only.
An interrupted active simulation is not automatically rerun. Preserve all evidence.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import types

VERSION='HMAC-BLINDED-EXECUTION-v1'
PREP='results/hmac_behavior_diagnosis_pilot/private_case_preparation_11e1e'
PREP_SHA='3f1c8551aefde2fd94cb692a55417da5a003b89da8eb24f8107392815afa59bd'
AUTH_SHA='56ee0a6a4441d3f58083f7a8a5303a46312ca250ec980e5886cb536252f83b7d'
TRUTH_SHA='7d356fc97a10eca8defcbc7295dc71d2cfa1a32611bd5c0ba1aa6a1f6f92a41a'
PREPARER='stage_11e1e_private_case_authorization.py'
PREPARER_SHA='67959b8fabba906723eff7727ebc74d89594f52f2a3189b2a39cd5198349656e'
BASELINE_SHA='c9b26acd5f77b6bdb681cc91c8697199346acd49c678c8d6f180e9f9e74e61f3'
CATALOG_FREEZE_SHA='a6f7312b9cd8a137da711133cd425a706c1705a84eb444df4a45eb2e3129ee71'
DEST='results/hmac_behavior_diagnosis_pilot/execution_11e1f'


def require(ok,message):
    if not ok:raise ValueError(message)


def enc(obj):return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def module(path,expected):
    data=Path(path).read_bytes();require(hashlib.sha256(data).hexdigest()==expected,'Source hash mismatch: '+Path(path).name)
    m=types.ModuleType('_verified_'+Path(path).stem);m.__file__=str(path)
    exec(compile(data,str(path),'exec'),m.__dict__);return m


def atomic(path,obj):
    fd,name=tempfile.mkstemp(prefix='pending_',dir=path.parent)
    with os.fdopen(fd,'wb') as f:f.write(enc(obj));f.flush();os.fsync(f.fileno())
    os.replace(name,path)


def immutable(path,obj):
    data=enc(obj)
    if path.exists():require(path.read_bytes()==data,'Committed artifact changed: '+path.name)
    else:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def record(path):return {'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size}


def verify(r):
    path=Path(r['path']);require(path.is_file() and sha(path)==r['sha256'] and path.stat().st_size==r['bytes'],'Checkpoint artifact changed: '+path.name)
    return path


def verified_inputs(root):
    m=module(root/PREPARER,PREPARER_SHA);d=m.load_runner(root)
    b,c,e,suite,vectors,batches,source=d.inputs(root)
    e.check(m.RUNNER,m.RUNNER_SHA);m.check_preflight(b,e)
    audit=b.read_json(e.check(PREP+'/freeze.json',PREP_SHA))
    require(audit['stage']=='11E-1E' and audit['status']=='PASS' and audit['preparation_status']=='FROZEN'
            and audit['source_sha256']==PREPARER_SHA,'Preparation freeze')
    require(audit['truth_displayed'] is False and audit['capture_execution_authorized'] is True and
            audit['simulation_performed'] is False and audit['diagnosis_on_fresh_observations_performed'] is False and
            audit['model_training_performed'] is False and audit['frozen_inputs_changed'] is False,
            'Preparation authorization scope')
    require(audit['cases']==12 and audit['fault_free_cases']==2 and audit['injected_fault_cases']==10 and
            audit['unique_cases']==4 and audit['ambiguous_cases']==4 and audit['normal_compatible_cases']==2,
            'Preparation case counts')
    for p,r in audit['input_evidence'].items():require(e.records.get(p)==r,'Preparation input binding: '+p)
    names={'private/entropy.json','private/truth.json','private/capture_plan.json','public_case_manifest.json','capture_authorization.json'}
    require(set(audit['outputs'])==names,'Preparation output set')
    for name,r in audit['outputs'].items():
        require(r['path']==PREP+'/'+name,'Preparation output path');e.link(r)
    e.check(PREPARER,PREPARER_SHA)
    auth=b.read_json(e.check(PREP+'/capture_authorization.json',AUTH_SHA))
    require(auth['capture_execution_authorized'] is True and auth['authorization_status']=='FROZEN','Capture not authorized')
    require(auth['truth_commitment']==audit['truth_commitment']==TRUTH_SHA,'Truth commitment')
    require(auth['canonical_transactions']==auth['replay_transactions']==768 and auth['total_transactions']==1536,'Capture bounds')
    require(auth['build_jobs']==1 and auth['execution']=='SEQUENTIAL' and auth['case_substitution_allowed'] is False,'Execution policy')
    require(auth['model_training_authorized'] is False and auth['validation_holdout_access_authorized'] is False and
            auth['original_validation_target']=='NOT_MET; PRESERVED' and auth['project_final_freeze_declared'] is False,
            'Authorization scope/disposition')
    require(auth['binding']['python_version']==platform.python_version(),'Python changed since selection freeze')
    require(auth['binding']['runner_sha256']==m.RUNNER_SHA and auth['binding']['suite_commitment']==suite['suite_commitment'],'Authorization inputs')
    truthpath=e.check(PREP+'/private/truth.json',TRUTH_SHA)
    truth=b.read_json(truthpath);public=b.read_json(e.path(PREP+'/public_case_manifest.json'))
    require(set(public)=={'version','suite_commitment','truth_commitment','case_ids'} and
            public['truth_commitment']==TRUTH_SHA and public['suite_commitment']==suite['suite_commitment'],
            'Public preparation manifest schema/binding')
    require(public['case_ids']==[x['case_id'] for x in truth['cases']] and len(set(public['case_ids']))==12,'Frozen case IDs')
    require(sha(e.path(PREP+'/public_case_manifest.json'))==auth['public_case_manifest_sha256'],'Public manifest commitment')
    plan=b.read_json(e.path(PREP+'/private/capture_plan.json'))
    require(plan['truth_commitment']==TRUTH_SHA and plan['binding']==auth['binding'],'Private plan binding')
    require([x['case_id'] for x in plan['cases']]==public['case_ids'],'Plan case IDs')
    for case,p in zip(truth['cases'],plan['cases']):
        site=int(case['site_id'].split('-')[-1])-1 if case['fault_present'] else 0
        bid,sel=divmod(site,512)
        require(p=={'case_id':case['case_id'],'batch_id':bid,'selector':sel,'fault_enable':int(case['fault_present']),
                    'forced_value':case['stuck_value'] or 0},'Private injection routing')
        require(plan['batch_netlists'][str(bid)]==str(batches[bid].relative_to(root)),'Batch binary binding')
    e.recheck()
    return m,d,b,c,e,suite,vectors,batches,source,truth,plan


def score(jobpath):
    """New scoring process: verify committed predictions BEFORE opening truth."""
    job=load(jobpath)
    manifest=load(verify(job['prediction_commitment']))
    predictions={}
    for r in manifest['cases']:
        prediction=load(verify(r['prediction']))
        require(prediction['case_id']==r['case_id'],'Prediction ID')
        predictions[r['case_id']]=prediction
    require(len(predictions)==12,'Prediction coverage')
    require(manifest['truth_commitment']==TRUTH_SHA,'Prediction truth binding')
    require(manifest['version']==VERSION and len(manifest['cases'])==12 and
            manifest['suite_commitment']==load(verify(job['suite']))['suite_commitment'],
            'Prediction manifest binding')
    require(manifest['catalog_freeze_sha256']==CATALOG_FREEZE_SHA,
            'Prediction/catalog freeze binding')
    truthpath=verify(job['truth']);require(sha(truthpath)==TRUTH_SHA,'Scoring truth commitment')
    truth=load(truthpath)  # Truth is first read here, after prediction verification.
    require(set(predictions)=={x['case_id'] for x in truth['cases']},'Scoring IDs')
    b=module(Path(job['baseline_source']),BASELINE_SHA)
    suite=load(verify(job['suite']));catalog=verify(job['catalog'])
    conn=b.open_catalog(catalog);rows=[]
    try:
        for case in truth['cases']:
            pred=predictions[case['case_id']];answer=pred['diagnosis']
            observation=verify(pred['observation']);replay=verify(pred['replay'])
            profile=b.read_observations(io.StringIO(observation.read_text()),suite)
            repeat=b.read_observations(io.StringIO(replay.read_text()),suite)
            expected=b.golden_profile(suite)
            if case['fault_present']:
                k=(int(case['site_id'].split('-')[-1])-1)*2+case['stuck_value']
                saved=conn.execute('SELECT p.response FROM profiles p JOIN faults f ON p.profile_id=f.profile_id WHERE f.fault_instance=?',(k,)).fetchall()
                require(len(saved)==1,'Scoring catalog entry');expected=bytes(saved[0][0])
            observed=answer['observable_anomaly'];category=case['selection_category']
            expected_status={'fault_free':'NO_OBSERVED_ANOMALY','normal':'NO_OBSERVED_ANOMALY',
                             'unique':'UNIQUE_SITE_IN_CATALOG','ambiguous':'AMBIGUOUS_SITES_IN_CATALOG'}[category]
            contained=case['site_id'] in answer['candidate_sites'] if case['fault_present'] else None
            checks={'replay_matches':profile==repeat,'catalog_response_matches':profile==expected,
                    'full_answer_matches':answer==b.classify(conn,profile,suite),
                    'expected_result':answer['result']==expected_status,
                    'truth_in_candidates':contained is not False,
                    'normal_not_certified':category not in ('normal','fault_free') or
                        (not observed and answer['localization_status']=='NOT_IDENTIFIABLE' and answer['no_fault_compatible'])}
            rows.append({'case_id':case['case_id'],'category':category,'fault_present':case['fault_present'],
                         'site_id':case['site_id'],'stuck_value':case['stuck_value'],'result':answer['result'],
                         'observable_anomaly':observed,'candidate_sites':answer['candidate_site_count'],
                         'truth_in_candidates':contained,'checks':checks,'pass':all(checks.values())})
    finally:conn.close()
    def fraction(n,den):return {'numerator':n,'denominator':den}
    observable=[x for x in rows if x['category'] in ('unique','ambiguous')]
    faults=[x for x in rows if x['fault_present']]
    unique=[x for x in rows if x['category']=='unique']
    resolved=lambda x:x['result']=='UNIQUE_SITE_IN_CATALOG' and x['truth_in_candidates'] is True
    report={'status':'PASS' if all(x['pass'] for x in rows) else 'FAIL','cases':rows,
            'false_alarms_fault_free':fraction(sum(x['observable_anomaly'] for x in rows if not x['fault_present']),2),
            'observable_fault_detections':fraction(sum(x['observable_anomaly'] for x in observable),8),
            'all_injected_fault_detections':fraction(sum(x['observable_anomaly'] for x in faults),10),
            'unique_category_exact_site':fraction(sum(resolved(x) for x in unique),4),
            'all_injected_exact_site':fraction(sum(resolved(x) for x in faults),10),
            'observable_candidate_coverage':fraction(sum(x['truth_in_candidates'] is True for x in observable),8),
            'no_match_cases':sum(x['result']=='NO_CATALOG_MATCH' for x in rows),
            'replay_disagreements':sum(not x['checks']['replay_matches'] for x in rows),
            'interpretation':'Small stratified known-catalog reproducibility demonstration; not independent generalization accuracy.',
            'predictions_committed_before_truth_read':True}
    print(json.dumps(report,sort_keys=True))


def execute(root,resume=False):
    os.umask(0o077)
    source_sha=sha(__file__)
    m,d,b,c,e,suite,vectors,batches,template,truth,plan=verified_inputs(root)
    tool=shutil.which('verilator');require(tool is not None,'Verilator missing from PATH')
    version=subprocess.run([tool,'--version'],capture_output=True,text=True,check=True).stdout.strip()
    binding={'source_sha256':source_sha,'preparation_sha256':PREP_SHA,'authorization_sha256':AUTH_SHA,
             'truth_commitment':TRUTH_SHA,'runner_sha256':m.RUNNER_SHA,'verilator':version,
             'verilator_sha256':sha(tool),'python_version':platform.python_version()}
    dest=e.path(DEST)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with (dest.parent/'execution_11e1f.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if dest.exists():require(resume,'Execution exists. Use --resume; do not delete previous evidence.')
        else:
            require(not resume,'No execution exists to resume')
            dest.mkdir(mode=0o700);(dest/'private').mkdir();(dest/'public').mkdir()
            atomic(dest/'private/checkpoint.json',{'binding':binding,'builds':{},'captures':{},'active_capture':None})
        state=load(dest/'private/checkpoint.json')
        require(state['binding']==binding,'Execution inputs/tool/source changed')
        require(state['active_capture'] is None,'An active capture was interrupted. Preserve the attempt for review; no automatic repeat.')
        for r in state['builds'].values():
            for item in r.values():verify(item)
        for r in state['captures'].values():
            verify(r['observation']);verify(r['log'])
        def checkpoint():atomic(dest/'private/checkpoint.json',state)
        immutable(dest/'public/truth_commitment.json',{'truth_commitment':TRUTH_SHA,'authorization_sha256':AUTH_SHA,
                    'case_ids':[x['case_id'] for x in truth['cases']]})
        immutable(dest/'public/reference_suite.json',suite)
        immutable(dest/'private/start_binding.json',binding)
        print('STAGE 11E-1F — BLINDED CAPTURE EXECUTION',flush=True)
        for index,(case,p) in enumerate(zip(truth['cases'],plan['cases']),1):
            key=str(p['batch_id'])
            if key not in state['builds']:
                print(f'Case {index:02d}/12: preparing simulator (one build job)',flush=True)
                work=Path(tempfile.mkdtemp(prefix='build_',dir=dest/'private'))/'work'
                binary=d.build_capture(tool,batches[p['batch_id']],template,p['batch_id'],vectors,work)
                state['builds'][key]={'binary':record(binary),'record':record(work/'build_record.json'),
                                      'testbench':record(work/'capture.sv'),'log':record(work/'build.log')};checkpoint()
            binary=verify(state['builds'][key]['binary'])
            build_info=load(verify(state['builds'][key]['record']))
            require(build_info['batch_id']==p['batch_id'] and build_info['netlist_sha256']==sha(batches[p['batch_id']])
                    and build_info['binary_sha256']==sha(binary),'Build/netlist binding')
            for mode in ('canonical','replay'):
                cid=case['case_id'];ck=cid+':'+mode
                if ck in state['captures']:continue
                private=Path(tempfile.mkdtemp(prefix='capture_',dir=dest/'private'))
                output=private/'response.csv';log=private/'simulation.log'
                state['active_capture']={'case_id':cid,'mode':mode,'directory':str(private)};checkpoint()
                print(f'Case {index:02d}/12: {mode.upper()}',flush=True)
                d.capture_case(binary,case,output,log)
                profile=b.read_observations(io.StringIO(output.read_text()),suite)
                require('BLINDED_CAPTURE_ROWS=64' in log.read_text(),'Missing capture closure')
                public=dest/'public'/f'{cid}_{mode}.csv'
                data=output.read_bytes()
                require(not public.exists(),'Public capture already exists without checkpoint; preserve attempt')
                public.write_bytes(data)
                state['captures'][ck]={'observation':record(public),'log':record(log),
                                      'normalized_sha256':hashlib.sha256(profile).hexdigest()}
                state['active_capture']=None;checkpoint()
            a=state['captures'][case['case_id']+':canonical'];r=state['captures'][case['case_id']+':replay']
            require(a['normalized_sha256']==r['normalized_sha256'],'Replay disagreement; stop without replacing captures')
            print(f'Case {index:02d}/12: CAPTURE AND REPLAY PASS',flush=True)
        require(len(state['captures'])==24,'Incomplete capture set')
        e.recheck();require(sha(__file__)==source_sha,'Controller source changed')
        manifest={'version':VERSION,'suite_commitment':suite['suite_commitment'],'truth_commitment':TRUTH_SHA,
                  'cases':[{'case_id':x['case_id'],'observation_path':state['captures'][x['case_id']+':canonical']['observation']['path'],
                            'observation_sha256':state['captures'][x['case_id']+':canonical']['observation']['sha256']} for x in truth['cases']]}
        immutable(dest/'public/capture_manifest.json',manifest)
        probe=dest/'private/denial_canary';probe.write_text('ACCESS DENIAL CANARY; NO TRUTH')
        job={'baseline_source':str(root/c.SOURCE),'baseline_source_sha256':c.SOURCE_SHA,
             'catalog':str(e.path(b.DEST+'/catalog.sqlite3')),'suite':str(dest/'public/reference_suite.json'),
             'cases':[{'case_id':x['case_id'],'path':x['observation_path'],'sha256':x['observation_sha256']} for x in manifest['cases']],
             'denial_probes':[str(e.path(PREP+'/private/truth.json')),str(e.path(PREP+'/private/entropy.json')),str(probe)]}
        print('Restricted diagnosis worker: starting; truth access denied',flush=True)
        work=Path(tempfile.mkdtemp(prefix='worker_',dir=dest/'private'))
        first=d.isolated(job,work/'job.json');second=d.isolated(job,work/'replay_job.json')
        require(first==second and first['denied_probes']==7,'Worker replay/denial failure')
        require([x['case_id'] for x in first['results']]==[x['case_id'] for x in manifest['cases']],'Worker case IDs')
        commitments=[]
        for result in first['results']:
            cid=result['case_id'];a=state['captures'][cid+':canonical']['observation'];r=state['captures'][cid+':replay']['observation']
            require(result['observation_sha256']==a['sha256'],'Worker observation binding')
            pred={'case_id':cid,'diagnosis':result['diagnosis'],'observation':a,'replay':r}
            path=dest/'public'/f'{cid}_prediction.json';immutable(path,pred)
            commitments.append({'case_id':cid,'prediction':record(path)})
        prediction_manifest={'version':VERSION,'truth_commitment':TRUTH_SHA,'suite_commitment':suite['suite_commitment'],
                             'catalog_freeze_sha256':CATALOG_FREEZE_SHA,'cases':commitments}
        immutable(dest/'public/prediction_commitment.json',prediction_manifest)
        print('All 12 predictions committed. Starting separate scoring process.',flush=True)
        scorejob={'prediction_commitment':record(dest/'public/prediction_commitment.json'),
                  'truth':record(e.path(PREP+'/private/truth.json')),'baseline_source':str(root/c.SOURCE),
                  'suite':record(dest/'public/reference_suite.json'),'catalog':record(e.path(b.DEST+'/catalog.sqlite3'))}
        immutable(work/'score_job.json',scorejob)
        proc=subprocess.run([sys.executable,'-I','-S',str(Path(__file__).resolve()),'_score',str(work/'score_job.json')],
                            stdin=subprocess.DEVNULL,capture_output=True,close_fds=True,timeout=120)
        require(proc.returncode==0,'Scoring failed: '+proc.stderr.decode()[-1000:])
        report=json.loads(proc.stdout);immutable(dest/'public/scoring_report.json',report)
        e.recheck();require(sha(__file__)==source_sha,'Controller source changed')
        for r in state['captures'].values():verify(r['observation']);verify(r['log'])
        artifacts={p.name:record(p) for p in (dest/'public').iterdir() if p.is_file()}
        audit={'stage':'11E-1F','version':VERSION,'status':report['status'],'binding':binding,'input_evidence':e.records,
               'artifacts':artifacts,'checkpoint':record(dest/'private/checkpoint.json'),
               'canonical_rows':768,'replay_rows':768,'cases':12,'worker_denied_probes':7,'worker_exact_replay':True,
               'predictions_committed_before_scoring':True,'original_validation_target':'NOT_MET; PRESERVED',
               'frozen_inputs_changed':False,'model_training_performed':False,'holdout_validation_samples_selected':0,
               'independent_accuracy_claim':False,'project_final_freeze_declared':False}
        immutable(dest/'freeze.json',audit)
        print('STAGE 11E-1F — BLINDED CAPTURE, DIAGNOSIS AND SCORING')
        print('Status:',report['status'])
        for key in ('false_alarms_fault_free','observable_fault_detections','all_injected_fault_detections',
                    'unique_category_exact_site','all_injected_exact_site','observable_candidate_coverage'):
            value=report[key];print(key+':',str(value['numerator'])+'/'+str(value['denominator']))
        print('Known-catalog demonstration only; not independent generalization accuracy.')
        print('Scoring report:',dest/'public/scoring_report.json')
        print('Audit SHA256:',sha(dest/'freeze.json'))
        require(report['status']=='PASS','Demonstration acceptance failed; preserve results and review')


def status(root):
    dest=root/DEST;path=dest/'private/checkpoint.json'
    if not path.exists():print('Execution checkpoint: NOT PRESENT');return
    state=load(path)
    print('Completed captures:',len(state['captures']),'/24 (status only; not integrity verification)')
    print('Active capture:', 'YES' if state['active_capture'] else 'NO')
    print('Closure:',load(dest/'freeze.json')['status'] if (dest/'freeze.json').exists() else 'NOT YET FROZEN')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='_score':score(sys.argv[2])
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--project-root',type=Path,default=Path.cwd())
        parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true')
        args=parser.parse_args()
        try:
            if args.status:status(args.project_root.resolve())
            else:execute(args.project_root.resolve(),args.resume)
        except KeyboardInterrupt:
            print('STOP: interrupted. Preserve the attempt; an active capture requires review before repeating.');raise SystemExit(130)
        except Exception as error:
            print('STOP:',error);print('Preserve all execution and private preparation files.');raise SystemExit(1)
