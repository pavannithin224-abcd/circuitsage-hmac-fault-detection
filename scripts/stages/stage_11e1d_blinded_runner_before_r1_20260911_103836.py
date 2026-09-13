#!/usr/bin/env python3
"""11E-1D runner components and preflight. No real circuit execution in this stage.
Python 3.9+, WSL/Linux, standard library; Verilator is required for preflight lint.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import types

VERSION = 'HMAC-BLINDED-RUNNER-v1'
CONTRACT_SOURCE = 'stage_11e1c_blinded_capture_contract.py'
CONTRACT_SHA = '4439a68d1823f1f1a668bbfbb0c1d593ce5ee4705383244fc3e79f35ab55d32c'
GENERATOR = 'scripts/hmac/generate_hmac_full_campaign_testbench.py'
GENERATOR_SHA = 'faae8e8d7bd5096840b103fa0fad6e893e0712b7f41747b2679494558dcb8094'
SEMANTIC = 'results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json'
SEMANTIC_SHA = 'a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0'
PARENT = 'results/hmac_behavior_diagnosis_pilot'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def module(path, expected):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'source SHA mismatch: ' + path.name)
    m = types.ModuleType('_verified_' + path.stem)
    m.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), m.__dict__)
    return m


def inputs(root):
    c = module(root / CONTRACT_SOURCE, CONTRACT_SHA)
    b = c.load_baseline(root)
    _, e, _, suite, _ = b.contract_inputs(root, full=True)
    e.check(c.SOURCE, c.SOURCE_SHA)
    e.check(c.BASE + '/freeze.json', c.BASE_SHA)
    b.verify_bundle(e, suite)
    path = e.path(c.DEST + '/freeze.json')
    require(path.is_file(), '11E-1C freeze missing; run stage_11e1c_blinded_capture_contract.py first')
    freeze = b.read_json(e.check(c.DEST + '/freeze.json', sha(path)))
    require(freeze['stage'] == '11E-1C' and freeze['version'] == c.VERSION and
            freeze['status'] == 'PASS' and freeze['contract_status'] == 'FROZEN' and
            freeze['source_sha256'] == CONTRACT_SHA, '11E-1C freeze binding')
    for k in ('simulation_performed', 'truth_selected_or_revealed', 'diagnosis_performed',
              'model_training_performed', 'validation_or_holdout_samples_accessed',
              'checked_frozen_inputs_changed', 'project_final_freeze_declared'):
        require(freeze[k] is False, '11E-1C scope: ' + k)
    require(freeze['original_validation_target']=='NOT_MET; PRESERVED', 'original validation disposition changed')
    for p, r in freeze['input_evidence'].items():
        require(e.records.get(p) == r, '11E-1C input mismatch: ' + p)
    docs = c.documents(suite)
    require(set(freeze['outputs']) == set(docs), '11E-1C output set')
    for name, data in docs.items():
        record = freeze['outputs'][name]
        require(record['path'] == c.DEST + '/' + name, '11E-1C output path')
        require(e.link(record).read_bytes() == data, '11E-1C regenerated contract mismatch: ' + name)
    e.check(CONTRACT_SOURCE, CONTRACT_SHA)
    genpath = e.check(GENERATOR, GENERATOR_SHA)
    generator = module(genpath, GENERATOR_SHA)
    semantic = b.read_json(e.check(SEMANTIC, SEMANTIC_SHA))
    entries = semantic['batch_qualification']
    require(len(entries) == 45 and {int(x['batch_id']) for x in entries} == set(range(45)), 'batch coverage')
    batches = {}
    for entry in entries:
        bid = int(entry['batch_id']); key = f'{bid:03d}'
        net = e.check(f'build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.v', entry['verilog_sha256'])
        mapping = e.check(f'results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json', entry['mapping_sha256'])
        sites = generator.find_sites(b.read_json(mapping))
        count = 311 if bid == 44 else 512
        require(sites is not None and len(sites) == count, 'mapping count')
        for i, site in enumerate(sites):
            require(int(site['selector_code']) == i and site['fault_site_id'] == f'HMAC-STEM-{bid*512+i+1:06d}', 'site mapping')
        batches[bid] = net
    # Only TRAIN records are selected/used; no validation or holdout payload is emitted.
    pool = b.read_json(e.check('results/hmac_fault_campaign_11c3/hmac_vector_pool_11c3a.json', generator.POOL_SHA))
    train = {int(x['vector_id']): x for x in pool['vectors'] if x['split'] == 'TRAIN'}
    vectors = []
    for v in suite['vectors']:
        record = train[v['vector_id']]
        converted = {k: generator.h256(record[k], k, v['vector_id']) for k in ('key', 'message', 'expected_hmac_sha256')}
        require(converted['expected_hmac_sha256'] == v['expected_digest'], 'suite expected digest')
        vectors.append({**v, **converted})
    print('Frozen inputs, 45 batch mappings and 64 TRAIN vectors: PASS', flush=True)
    return b, c, e, suite, vectors, batches, genpath.read_text()


def render_capture(source, bid, vectors):
    """Keep frozen reset/run tasks verbatim after f-string brace unescaping.
    All injection controls are private plusargs, never CSV columns.
    """
    require(0 <= bid < 45 and len(vectors) == 64, 'capture dimensions')
    start = source.index('  task automatic reset_dut;')
    end = source.index('  initial begin', start)
    tasks = source[start:end].replace('{{', '{').replace('}}', '}')
    require('cycles=0;' in tasks and '#1; cycles=cycles+1' in tasks, 'timing template')
    init = '\n'.join(f"key_vectors[{i}]=256'h{v['key']}; message_vectors[{i}]=256'h{v['message']}; vector_ids[{i}]={v['vector_id']};" for i,v in enumerate(vectors))
    return f'''`timescale 1ns/1ps
module tb_blinded_capture;
localparam TIMEOUT_CYCLES=2000;
logic clk_i=0,rst_ni=0,start_i=0,busy_o,done_o,fault_enable_i=0,fault_value_i=0,fault_raw_o;
logic [8:0] fault_selector_i=0;
logic [255:0] key_i=0,message_i=0,digest_o;
logic monitor_clear=1,activity_latched=0;
logic [255:0] key_vectors[0:63],message_vectors[0:63],measured;
integer vector_ids[0:63]; integer fd,vi,cycles,en,selector,forced;
logic timed_out,activity_seen,unknown_seen;
string csv_path;
opentitan_hmac_sha256_msg32_faultbatch{bid:03d} dut(.clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_i),.key_i(key_i),.message_i(message_i),.busy_o(busy_o),.done_o(done_o),.digest_o(digest_o),.fault_enable_i(fault_enable_i),.fault_selector_i(fault_selector_i),.fault_value_i(fault_value_i),.fault_raw_o(fault_raw_o));
always #5 clk_i=~clk_i;
always @(monitor_clear or fault_enable_i or fault_value_i or fault_raw_o) begin
 if(monitor_clear) activity_latched=0;
 else if(fault_enable_i && (fault_raw_o !== fault_value_i)) activity_latched=1;
end
{tasks}
initial begin
{init}
if(!$value$plusargs("RESULT_CSV=%s",csv_path)) $fatal(1,"missing output");
if(!$value$plusargs("ENABLE=%d",en)) $fatal(1,"missing enable");
if(!$value$plusargs("SELECTOR=%d",selector)) $fatal(1,"missing selector");
if(!$value$plusargs("FORCED=%d",forced)) $fatal(1,"missing forced value");
if(en<0 || en>1 || forced<0 || forced>1 || selector<0 || selector>={311 if bid==44 else 512}) $fatal(1,"invalid controls");
fd=$fopen(csv_path,"w"); if(!fd) $fatal(1,"output open failed");
$fdisplay(fd,"vector_slot,vector_id,actual_digest,cycles,timed_out");
for(vi=0;vi<64;vi=vi+1) begin
 run_transaction(vi,en!=0,selector,forced!=0,cycles,timed_out,measured,activity_seen,unknown_seen);
 if(unknown_seen) $fatal(1,"unknown measurement");
 if(timed_out) $fdisplay(fd,"%0d,%0d,,%0d,1",vi,vector_ids[vi],cycles);
 else $fdisplay(fd,"%0d,%0d,%064h,%0d,0",vi,vector_ids[vi],measured,cycles);
end
$fclose(fd); $display("BLINDED_CAPTURE_ROWS=64"); $finish;
end
endmodule
'''


def select_truth(conn, b, suite, seed, salt):
    """Future private preparation; preflight never calls this on the real catalog."""
    rng = random.Random(seed)
    groups = {'unique': [], 'ambiguous': [], 'normal': []}
    normal = b.golden_profile(suite)
    for pid, response in conn.execute('SELECT profile_id,response FROM profiles ORDER BY profile_id'):
        members = [k for k, in conn.execute('SELECT fault_instance FROM faults WHERE profile_id=? ORDER BY fault_instance', (pid,))]
        category = 'normal' if bytes(response)==normal else 'unique' if len({k//2 for k in members})==1 else 'ambiguous'
        groups[category].extend(members)
    chosen = [(category,k) for category,n in [('unique',4),('ambiguous',4),('normal',2)] for k in rng.sample(groups[category],n)]
    require({k%2 for _,k in chosen}=={0,1}, 'selected set lacks both SA values; stop before capture')
    records = [{'fault_present':True,'site_id':f'HMAC-STEM-{k//2+1:06d}','stuck_value':k%2,'selection_category':category} for category,k in chosen]
    records += [{'fault_present':False,'site_id':None,'stuck_value':None,'selection_category':'fault_free'} for _ in range(2)]
    rng.shuffle(records)
    for r in records:
        r['case_id'] = f'{rng.getrandbits(128):032x}'
    require(len({r['case_id'] for r in records})==12, 'case ID collision')
    require(re.fullmatch('[0-9a-f]{64}',salt) is not None, 'truth salt must be 32 bytes')
    return {'version':VERSION,'salt_hex':salt,'selection_seed':seed,'cases':records}


def build_capture(verilator, netlist, source, bid, vectors, directory):
    """Private build primitive for the later authorized controller; never overwrites."""
    directory.mkdir(parents=True, exist_ok=False)
    tb=directory/'capture.sv';tb.write_text(render_capture(source,bid,vectors))
    obj=directory/'obj_dir'
    command=[verilator,'--binary','--timing','--assert','-Wall','-Wno-fatal','-j','1',
             '--Mdir',str(obj),'--top-module','tb_blinded_capture',str(netlist),str(tb)]
    with (directory/'build.log').open('x') as f:
        rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
    binary=obj/'Vtb_blinded_capture'
    require(rc==0 and binary.is_file(),'capture build failed; preserve build directory')
    (directory/'build_record.json').write_bytes(encode({'batch_id':bid,'command':command,
        'netlist_sha256':sha(netlist),'testbench_sha256':sha(tb),'binary_sha256':sha(binary)}))
    return binary


def capture_case(binary, case, output, log):
    """Private capture primitive; only a subsequent authorized controller may call it."""
    require(not output.exists() and not log.exists(), 'capture output exists')
    site = int(case['site_id'].split('-')[-1])-1 if case['fault_present'] else 0
    require(not case['fault_present'] or (0<=site<22839 and case['stuck_value'] in (0,1)), 'invalid injection truth')
    command = [str(binary), '+RESULT_CSV='+str(output), '+ENABLE='+str(int(case['fault_present'])),
               '+SELECTOR='+str(site%512), '+FORCED='+str(case['stuck_value'] or 0)]
    with log.open('x') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, timeout=300, check=False)
    require(result.returncode==0 and output.is_file(), 'capture failed; preserve attempt')


def install_guard(allowed):
    """File allowlist for reviewed trusted CPython worker, NOT a hostile-code sandbox."""
    allowed = {str(Path(p).resolve()) for p in allowed}
    def guard(event, args):
        if event == 'open':
            path, mode, flags = args
            if isinstance(path,int) or str(Path(os.fsdecode(path)).resolve()) not in allowed:
                raise PermissionError(errno.EACCES, 'worker read not allowlisted')
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                raise PermissionError(errno.EACCES, 'worker write forbidden')
        elif event == 'sqlite3.connect':
            require(str(args[0]) in SQLITE_ALLOWED, 'SQLite path not allowlisted')
        elif event.startswith(('subprocess.', 'socket.', 'ctypes.')) or event in (
            'os.system','os.exec','os.posix_spawn','os.fork','os.forkpty','os.listdir','os.scandir',
            'os.remove','os.rename','os.mkdir','os.rmdir','os.link','os.symlink','os.truncate','os.chmod'):
            raise PermissionError(errno.EACCES, 'worker operation forbidden: '+event)
    sys.addaudithook(guard)


SQLITE_ALLOWED = set()


def worker(jobpath):
    # Started with -I -S, close_fds=True, empty input; no parent case truth in memory.
    job = json.loads(Path(jobpath).read_text())
    b = module(Path(job['baseline_source']), job['baseline_source_sha256'])
    catalog = Path(job['catalog']).resolve()
    uri = catalog.as_uri()+'?mode=ro'
    SQLITE_ALLOWED.add(uri)
    allowed = [catalog, job['suite']] + [x['path'] for x in job['cases']]
    install_guard(allowed)
    denied = 0
    for p in job.get('denial_probes',[]):
        for op in (lambda p: Path(p).read_bytes(),lambda p: os.open(p,os.O_RDONLY)):
            try:
                value=op(p)
                if isinstance(value,int): os.close(value)
            except PermissionError: denied+=1
            else: raise ValueError('private read was permitted')
    try:
        with open(catalog,'ab'): pass
    except PermissionError: denied+=1
    else: raise ValueError('catalog write was permitted')
    suite = json.loads(Path(job['suite']).read_text())
    out=[]
    conn=b.open_catalog(catalog)
    try:
        for case in job['cases']:
            data=Path(case['path']).read_bytes()
            require(hashlib.sha256(data).hexdigest()==case['sha256'], 'observation changed')
            profile=b.read_observations(io.StringIO(data.decode()),suite)
            out.append({'case_id':case['case_id'],'observation_sha256':case['sha256'],
                        'diagnosis':b.classify(conn,profile,suite)})
    finally: conn.close()
    print(json.dumps({'denied_probes':denied,'results':out},sort_keys=True))


def isolated(job, path):
    path.write_bytes(encode(job))
    proc=subprocess.run([sys.executable,'-I','-S',str(Path(__file__).resolve()),'_worker',str(path)],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,
        env={'PATH':os.environ.get('PATH',''),'PYTHONHASHSEED':'0'},timeout=120)
    require(proc.returncode==0, 'isolated worker failed: '+proc.stderr.decode()[-2000:])
    return json.loads(proc.stdout)


def preflight(root):
    source_sha=sha(__file__)
    b,c,e,suite,vectors,batches,source=inputs(root)
    verilator=shutil.which('verilator')
    require(verilator is not None,'verilator missing; export PATH="$HOME/tools/oss-cad-suite/bin:$PATH"')
    parent=e.path(PARENT); parent.mkdir(exist_ok=True)
    with (parent/'runner_11e1d.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=Path(tempfile.mkdtemp(prefix='preflight_11e1d_',dir=parent))
        try:
            script=attempt/'lint_capture.sv'
            script.write_text(render_capture(source,22,vectors))
            command=[verilator,'--lint-only','--timing','--assert','-Wall','-Wno-fatal',
                     '--top-module','tb_blinded_capture',str(batches[22]),str(script)]
            with (attempt/'lint.log').open('w') as f:
                rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
            require(rc==0,'capture testbench lint failed; see '+str(attempt/'lint.log'))
            print('Capture testbench lint: PASS; no binary executed',flush=True)
            public=attempt/'public';public.mkdir()
            (public/'suite.json').write_bytes(encode(suite))
            private=attempt/'private_canaries';private.mkdir()
            for name in ('truth.json','capture.log'): (private/name).write_text('SYNTHETIC DENIAL CANARY')
            (public/'escape.csv').symlink_to(private/'truth.json')
            cases=[]
            for i,name in enumerate(('normal','unique','ambiguous','no_match')):
                path=public/f'{i:032x}.csv'
                shutil.copyfile(e.path(b.DEST+'/'+name+'.csv'),path)
                cases.append({'case_id':f'{i:032x}','path':str(path),'sha256':sha(path)})
            job={'baseline_source':str(root/c.SOURCE),'baseline_source_sha256':c.SOURCE_SHA,
                 'catalog':str(e.path(b.DEST+'/catalog.sqlite3')),'suite':str(public/'suite.json'),
                 'cases':cases,'denial_probes':[str(private/'truth.json'),str(private/'capture.log'),str(public/'escape.csv')]}
            first=isolated(job,attempt/'worker_job.json')
            second=isolated(job,attempt/'worker_replay_job.json')
            require(first==second and first['denied_probes']==7,'worker replay/access denial')
            expected=['NO_OBSERVED_ANOMALY','UNIQUE_SITE_IN_CATALOG','AMBIGUOUS_SITES_IN_CATALOG','NO_CATALOG_MATCH']
            require([x['diagnosis']['result'] for x in first['results']]==expected,'worker results')
            conn=b.open_catalog(e.path(b.DEST+'/catalog.sqlite3'))
            try:
                for case,out in zip(cases,first['results']):
                    p=b.read_observations(io.StringIO(Path(case['path']).read_text()),suite)
                    require(out['diagnosis']==b.classify(conn,p,suite),'complete candidate set changed in worker')
            finally:conn.close()
            # Reject identity-bearing input in the isolated worker.
            bad=public/'malformed.csv';bad.write_text(Path(cases[0]['path']).read_text().replace('timed_out','timed_out,site_id',1))
            badjob={**job,'cases':[{'case_id':'bad','path':str(bad),'sha256':sha(bad)}]}
            try:isolated(badjob,attempt/'invalid_job.json')
            except ValueError as err:require('Injection identity/control columns' in str(err),'unexpected negative-test failure')
            else:raise ValueError('worker accepted injection identity')
            # Selection logic is exercised only on a separate synthetic miniature catalog.
            normal=b.golden_profile(suite)
            synthetic={normal:[0,1]}
            for k in range(2,6):
                synthetic[normal[:3]+k.to_bytes(32,'big')+normal[35:]]=[k]
            synthetic[normal[:3]+(6).to_bytes(32,'big')+normal[35:]]=[6,7,8,9]
            fixture=attempt/'synthetic_selection.sqlite3'
            b.write_catalog(fixture,synthetic,suite)
            conn=b.open_catalog(fixture)
            try:
                truth=select_truth(conn,b,suite,123,'00'*32)
                require(truth==select_truth(conn,b,suite,123,'00'*32),'synthetic selection replay')
                require(len(truth['cases'])==12 and sum(x['fault_present'] for x in truth['cases'])==10,'synthetic selection counts')
            finally:conn.close()
            (attempt/'worker_results.json').write_bytes(encode(first))
            e.recheck();require(sha(__file__)==source_sha,'runner source changed')
            files={str(p.relative_to(attempt)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in attempt.rglob('*') if p.is_file() and not p.is_symlink()}
            audit={'stage':'11E-1D','version':VERSION,'status':'PASS','preflight_status':'FROZEN',
                   'source_sha256':source_sha,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                   'input_evidence':e.records,'outputs':files,'capture_lint':'PASS','access_denial_probes':7,
                   'worker_exact_replay':'PASS','worker_complete_candidate_agreement':'PASS','identity_input_rejected':True,
                   'synthetic_selection_replay':'PASS',
                   'blinding_enforcement':'Trusted CPython worker audit allowlist; not a hostile-code OS sandbox.',
                   'test_origin':'Saved catalog examples plus synthetic no-match and denial canaries; not fresh circuit responses.',
                   'real_cases_selected':False,'simulation_performed':False,'model_training_performed':False,
                   'validation_holdout_samples_selected':0,'frozen_inputs_changed':False,
                   'capture_execution_authorized':False,'original_validation_target':'NOT_MET; PRESERVED',
                   'project_final_freeze_declared':False,
                   'next':'11E-1E — Private Case Preparation and Capture Execution Authorization'}
            (attempt/'freeze.json').write_bytes(encode(audit))
            print('STAGE 11E-1D — BLINDED RUNNER IMPLEMENTATION AND PREFLIGHT')
            print('Status: PASS; testbench lint: PASS; denied probes: 7; worker replay: PASS')
            print('Real simulation: NOT PERFORMED; real cases: NOT SELECTED')
            print('Preflight bundle:',attempt)
            print('Audit SHA256:',sha(attempt/'freeze.json'))
            print('Next:',audit['next'])
        except BaseException:
            print('Attempt preserved for diagnosis:',attempt,flush=True)
            raise


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='_worker':
        worker(sys.argv[2])
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--project-root',type=Path,default=Path.cwd())
        args=parser.parse_args()
        try:preflight(args.project_root.resolve())
        except KeyboardInterrupt:raise SystemExit(130)
        except Exception as error:
            print('STOP:',error)
            raise SystemExit(1)
