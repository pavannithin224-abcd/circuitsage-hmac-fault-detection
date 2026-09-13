#!/usr/bin/env python3
"""11D-3J: review frozen validation results and record project disposition.

No new model execution, sample-label access, tuning, or HOLDOUT access.
Produces a review report, descriptive subgroup tables, a disposition record,
and a final audit. Review PASS never changes the measured validation NOT_MET.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import hashlib

R="results/hmac_fault_campaign_11d3/"
SOURCE_I="stage_11d3i_validation_disposition.py"
SHA_I="d3ce55488398953211aa10b68d13bd9c06b8ef83dbac87bfb14a95823935a93c"
AUDIT_I=R+"hmac_validation_result_disposition_holdout_readiness_freeze_11d3i.json"
POLICY_I="config/diagnostic_model/hmac_validation_result_disposition_policy_11d3i.json"
READINESS_I="config/diagnostic_model/hmac_holdout_readiness_policy_11d3i.json"
RESOURCE_I=R+"stage_11d3i_resources_20260910_005103.log"
LOG_I=R+"stage_11d3i_disposition_20260910_005103.log"
EXPECTED={
    SOURCE_I:SHA_I,
    AUDIT_I:"fde310f3c49cbbfc38b5c3e669d2a2206ee5ef2ab326a63e777532ae8663ed5a",
    POLICY_I:"d300249ae5d670f9a73b1edbc54ff42875fe6737349ded3fce217756be8ea7eb",
    READINESS_I:"a1a501a98741dd0260b61b1983b5536b98510904578691b047b1cee5a5f11633",
    R+"hmac_validation_acceptance_criteria_11d3i.csv":"eb729c1eda58e22940d08a0f2ccd9d51ea0c1b070f3684f514cbbb09fb9c20a0",
    R+"hmac_validation_comparator_registry_11d3i.csv":"af9330e9359ea2afb9c0135fc44e8818dbbb222537e3dcea7ece87c404cdc022",
    R+"hmac_validation_result_disposition_report_11d3i.md":"64efc5ce45ecd25e75592185927a554592126988c337ea945137998ec236260d",
    RESOURCE_I:"13b378036158b09bfecdcdabcb5920e19d2432dfcb4895594ccfc6daedab38e9",
    LOG_I:"2da59a376b6d114a598cc55833ef6c2191ab1be74678eea1273e0a9f97f27c72",
}
OUT={
    "review":R+"hmac_validation_failure_review_11d3j.md",
    "subgroup_metrics":R+"hmac_validation_review_subgroup_metrics_11d3j.csv",
    "subgroup_comparisons":R+"hmac_validation_review_subgroup_comparisons_11d3j.csv",
    "disposition":R+"hmac_project_disposition_after_validation_11d3j.json",
    "audit":R+"hmac_validation_failure_review_project_disposition_freeze_11d3j.json",
}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def helper(root):
    path=root/SOURCE_I
    require(path.is_file() and sha(path)==SHA_I,"Stage 3I source missing or hash mismatch")
    mod=types.ModuleType('_verified_disposition_review');mod.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),mod.__dict__)
    return mod


def verify(e,i,v):
    for path,digest in EXPECTED.items():
        e.verify(path,digest)
        print(f"  {Path(path).name}: OK",flush=True)
    a,p,r=(e.load(path) for path in (AUDIT_I,POLICY_I,READINESS_I))
    v.fields(a,{"stage":"11D-3I","status":"PASS","disposition_status":"FROZEN","audit_status":"FROZEN",
        "validation_target":"NOT_MET","failed_criteria":["mcc_retention"],"best_observed_validation_model":"hybrid",
        "validation_result_accepted_for_advancement":False,"holdout_campaign":"NOT_AUTHORIZED",
        "holdout_readiness":"BLOCKED_VALIDATION_TARGET_NOT_MET","project_final_freeze":"NOT_DECLARED",
        "model_objects_deserialized":0,"model_inference_calls":0,"model_fitting_calls":0,"threshold_selection_calls":0,
        "sample_target_arrays_read":False,"prediction_arrays_read":False,"holdout_payloads_read":0,
        "frozen_inputs_modified":False,"original_model_locks_modified":False},"3I freeze")
    for path,record in a['input_evidence'].items():e.link(record,path)
    require(set(a['outputs'])==set(i.OUTPUTS)-{'audit'},"3I output inventory")
    for key,record in a['outputs'].items():e.link(record,i.OUTPUTS[key])
    for key,value in p.items():require(a[key]==value,f"3I policy/audit agreement: {key}")
    v.fields(r,{"readiness":"BLOCKED_VALIDATION_TARGET_NOT_MET","validation_target":"NOT_MET",
        "failed_criteria":["mcc_retention"],"holdout_campaign_execution_authorized":False,
        "holdout_model_inference_authorized":False,"holdout_vector_access_authorized":False,
        "model_retraining_authorized":False,"threshold_changes_authorized":False,"authorization_issued":False},"HOLDOUT readiness")
    h,c=i.verify_reports(e,v)
    reconstructed,_=i.disposition(h,c)
    for key,value in reconstructed.items():require(p[key]==value,f"disposition reconstruction: {key}")
    v.verify_resource_log(e.path(RESOURCE_I))
    require(EXPECTED[AUDIT_I] in e.path(LOG_I).read_text(),"3I log/audit agreement")
    raw=json.loads(e.path(h['outputs']['breakdowns']['path']).read_text(),object_pairs_hook=v.unique_object)
    schema=e.load(c['schema']['path'])
    wanted={
        'fault_model':{'SA0','SA1'},
        'site_category':set(schema['categorical_vocabulary']['site_category']),
        'driver_cell_type':set(schema['categorical_vocabulary']['driver_cell_type']),
        'development_site_partition':{'DEV_TRAIN','DEV_CALIBRATION','DEV_SITE_TEST'},
    }
    for family,values in wanted.items():
        require({row['value'] for row in raw if row['group']==family}==values,f"breakdown vocabulary: {family}")
    vectors=[row for row in raw if row['group']=='validation_vector_id']
    require(len(vectors)==48 and all(type(row['value']) is int for row in vectors),"48 validation vector groups")
    for row in vectors:
        require(all(m['samples']==45678 for m in row['models'].values()),"validation vector group sample count")
    counts={'DEV_TRAIN':15987*96,'DEV_CALIBRATION':3426*96,'DEV_SITE_TEST':3426*96}
    for row in raw:
        if row['group']=='development_site_partition':
            require(all(m['samples']==counts[row['value']] for m in row['models'].values()),"site stratum sample count")
    return p,h,c,raw


def summarize(p,h,raw):
    comparisons=[];flat=[]
    for group in raw:
        scores=group['models'];hybrid=scores['hybrid'];gnn=scores['gnn']
        represented=hybrid['status']=='REPRESENTED'
        delta=hybrid['mcc']-gnn['mcc'] if represented else None
        comparisons.append({'family':group['group'],'group':group['value'],'samples':hybrid['samples'],
            'status':hybrid['status'],'hybrid_mcc':hybrid.get('mcc'),'gnn_mcc':gnn.get('mcc'),
            'hybrid_minus_gnn_mcc':delta,'interpretation':'DESCRIPTIVE_ONLY; NO_SUBGROUP_SIGNIFICANCE_TEST'})
        for name,m in scores.items():
            row={'family':group['group'],'group':group['value'],'model':name,'status':m['status'],'samples':m['samples']}
            for key in ('positive_samples','positive_prevalence','mcc','balanced_accuracy','precision','recall','specificity','f1_score','pr_auc','roc_auc','brier_score','tn','fp','fn','tp'):
                row[key]=m.get(key)
            flat.append(row)
    strata={row['group']:row for row in comparisons if row['family']=='development_site_partition'}
    require(set(strata)=={'DEV_TRAIN','DEV_CALIBRATION','DEV_SITE_TEST'},'review requires three site strata')
    vectors=[row for row in comparisons if row['family']=='validation_vector_id' and row['status']=='REPRESENTED']
    require(vectors,'no represented validation vector groups')
    all_site=p['hybrid_validation_mcc'];old_sites=strata['DEV_SITE_TEST']['hybrid_mcc']
    result={'stage':'11D-3J','status':'PASS','review_status':'FROZEN','project_disposition_status':'FROZEN',
        'current_experiment_disposition':'ACCEPTANCE_NOT_MET; STOP_BEFORE_HOLDOUT',
        'validation_target':'NOT_MET','failed_criteria':p['failed_criteria'],'best_observed_validation_model':'HYBRID',
        'hybrid_validation_mcc':all_site,'hybrid_dev_site_test_mcc':p['hybrid_dev_site_test_mcc'],
        'contract_mcc_drop':p['hybrid_mcc_drop'],'allowed_mcc_drop':p['allowed_mcc_drop'],'retention_shortfall':p['retention_shortfall'],
        'hybrid_minus_gnn_mcc':p['hybrid_minus_gnn_mcc'],'hybrid_minus_gnn_95_site_ci':p['hybrid_minus_gnn_95_site_ci'],
        'acceptance_family_status':p['criterion_family_status'],'frozen_model_metrics':h['metrics'],
        'validation_original_site_strata':strata,
        'descriptive_population_comparison':{
            'original_dev_site_test_mcc':p['hybrid_dev_site_test_mcc'],
            'validation_on_original_dev_site_test_sites_mcc':old_sites,
            'validation_all_sites_mcc':all_site,
            'validation_original_site_test_minus_development_mcc':old_sites-p['hybrid_dev_site_test_mcc'],
            'all_site_minus_original_site_test_validation_mcc':all_site-old_sites,
            'limitation':'Descriptive population comparisons, not a causal decomposition; 64 development and 48 validation vectors differ.'},
        'descriptive_vector_summary':{
            'groups':len(vectors),'hybrid_mcc_minimum':min(row['hybrid_mcc'] for row in vectors),
            'hybrid_mcc_maximum':max(row['hybrid_mcc'] for row in vectors),
            'hybrid_point_gain_over_gnn_groups':sum(row['hybrid_minus_gnn_mcc']>0 for row in vectors),
            'limitation':'Same sites reused across vectors; no independent-replicate or subgroup significance claim.'},
        'cause_of_retention_failure':'NOT_ESTABLISHED_BY_THIS_REVIEW',
        'holdout_readiness':'BLOCKED','holdout_campaign_execution_authorized':False,'holdout_inference_authorized':False,
        'retraining_authorized':False,'threshold_changes_authorized':False,'acceptance_rule_changes_authorized':False,
        'project_final_freeze':'NOT_DECLARED','deployment_readiness':'NOT_ESTABLISHED',
        'review_complete':True,'automatic_continuation':'NONE',
        'permitted_followup':'Discuss the frozen findings and limitations with the mentor; no new experiment is authorized by this report.',
        'project_decision_for_review':'Document this experiment as a measured result with an unmet acceptance criterion. Any separately scoped future study needs a prospective protocol and independent evaluation plan; preserve this run unchanged.',
        'scope':h['scope'],'statistical_limitations':h['statistical_limitations'],
        'sample_labels_reopened':False,'prediction_arrays_read':False,'model_objects_deserialized':0,
        'model_inference_calls':0,'model_fitting_calls':0,'holdout_payloads_read':0,'frozen_inputs_modified':False}
    return result,flat,comparisons


def report(d,p,comparisons):
    def f(value):return 'not represented' if value is None else f'{value:.8f}'
    lines=['# Validation Failure Review and Project Disposition — Stage 11D-3J','',
        '**Review completed. Current experiment: acceptance NOT_MET; stop before HOLDOUT.**','',
        'The hybrid is the strongest of the four evaluated models. Only the predeclared MCC-retention criterion failed. The evaluation and deterministic replay passed; this is a performance shortfall, not an execution failure.','',
        '## Evidence and acceptance','',
        '| Model | Validation MCC | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('hybrid','gnn','baseline','deep'):
        m=d['frozen_model_metrics'][name]
        lines.append('| '+name+' | '+' | '.join(f(m[key]) for key in ('mcc','balanced_accuracy','precision','recall','f1_score','pr_auc','roc_auc','brier_score'))+' |')
    lines.extend(['',f"The hybrid MCC dropped from {d['hybrid_dev_site_test_mcc']:.8f} to {d['hybrid_validation_mcc']:.8f}. The drop of {d['contract_mcc_drop']:.8f} exceeded the allowed {d['allowed_mcc_drop']:.8f}; the retention threshold was missed by {d['retention_shortfall']:.8f} MCC.",'',
        f"The hybrid–GNN gain remains {d['hybrid_minus_gnn_mcc']:.8f}, with paired-site 95% interval {d['hybrid_minus_gnn_95_site_ci']}. This supports a gain over GNN under the stated same-circuit evaluation, but cannot override the failed retention criterion.",'',
        '| Acceptance family | Frozen result |','|---|---|'])
    for family,status in d['acceptance_family_status'].items():lines.append(f'| {family} | {status} |')
    lines.extend(['','## Site-population comparison','',
        'The acceptance contract compares a development score on 3,426 DEV_SITE_TEST sites with validation across all 22,839 sites. The following table separates validation by the original site partitions. These descriptive results do not change the acceptance population or threshold.','',
        '| Original development site partition | Validation samples | Hybrid MCC | GNN MCC | MCC difference |',
        '|---|---:|---:|---:|---:|'])
    for name in ('DEV_TRAIN','DEV_CALIBRATION','DEV_SITE_TEST'):
        r=d['validation_original_site_strata'][name]
        lines.append(f"| {name} | {r['samples']} | {f(r['hybrid_mcc'])} | {f(r['gnn_mcc'])} | {f(r['hybrid_minus_gnn_mcc'])} |")
    population=d['descriptive_population_comparison']
    lines.extend(['',f"On the original DEV_SITE_TEST sites, validation hybrid MCC is {population['validation_on_original_dev_site_test_sites_mcc']:.8f}; the earlier development score was {population['original_dev_site_test_mcc']:.8f}. The difference is {population['validation_original_site_test_minus_development_mcc']:.8f}. The vector sets and their sizes differ, so this does not isolate a causal effect.",'',
        '## Fault and cell groups','',
        'All saved SA0/SA1, site-category, and driver-cell groups are shown below. Comparisons are descriptive; no per-group confidence intervals or multiple-comparison significance tests were added.','',
        '| Family | Group | Samples | Hybrid MCC | GNN MCC | MCC difference |','|---|---|---:|---:|---:|---:|'])
    for r in comparisons:
        if r['family'] in ('fault_model','site_category','driver_cell_type'):
            lines.append(f"| {r['family']} | `{r['group']}` | {r['samples']} | {f(r['hybrid_mcc'])} | {f(r['gnn_mcc'])} | {f(r['hybrid_minus_gnn_mcc'])} |")
    summary=d['descriptive_vector_summary']
    lines.extend(['','## Validation vectors','',
        f"Across {summary['groups']} vector groups, hybrid MCC ranges from {summary['hybrid_mcc_minimum']:.8f} to {summary['hybrid_mcc_maximum']:.8f}. Hybrid point MCC exceeds GNN in {summary['hybrid_point_gain_over_gnn_groups']} groups. These are repeated measurements on the same circuit sites, not independent chips.",'',
        '| Validation vector ID | Samples | Hybrid MCC | GNN MCC | MCC difference |','|---|---:|---:|---:|---:|'])
    for r in sorted((r for r in comparisons if r['family']=='validation_vector_id'),key=lambda r:r['group']):
        lines.append(f"| {r['group']} | {r['samples']} | {f(r['hybrid_mcc'])} | {f(r['gnn_mcc'])} | {f(r['hybrid_minus_gnn_mcc'])} |")
    lines.extend(['','## What this review establishes','',
        'The recorded failure is the retention threshold breach. The underlying cause is not established. Overfitting, stimulus distribution differences, and site-population differences must not be presented as proven explanations from these aggregate reports.','',
        d['scope'],'',d['statistical_limitations'],'',
        'The task predicts whether a simulated persistent SA0/SA1 fault will be detected, before simulation. It is not a claim of fault localization, transient-fault coverage, or physical-chip validation. MCC is not percentage accuracy.','',
        '## Project disposition','',
        '1. Preserve the current run as an experiment with unmet validation acceptance. The hybrid remains the best observed comparator, without approval for advancement.','2. Keep HOLDOUT unopened and preserve all models, thresholds, data, and acceptance rules.','3. Use this report and the frozen audit in the mentor review. Final project completion and deployment readiness have not been established.','4. If further research is chosen, discuss a separately scoped, prospectively specified study with independent evaluation. This review does not authorize retraining, tuning on validation, a threshold waiver, or HOLDOUT execution.','',
        'The requested failure review is complete. No additional automatic execution stage is created.','',
        '## Suggested mentor explanation','',
        f'“We completed the HMAC SA0/SA1 development campaign and evaluated four frozen models on 48 unseen validation vectors. The hybrid achieved MCC {d["hybrid_validation_mcc"]:.4f} and outperformed GNN by {d["hybrid_minus_gnn_mcc"]:.4f}. It passed the other acceptance checks but exceeded the permitted development-to-validation MCC drop. We preserved the failed result, kept HOLDOUT unopened, and documented the limitations. The broader project is not yet declared complete.”','',
        '## Verification limits','',
        'This review verifies the frozen report chain, saved output hashes, acceptance reconstruction, confusion-metric consistency, group totals, and bootstrap summary consistency through the prior verifier. It reads summary JSON only for interpretation. It does not regenerate predictions or recompute rank metrics from samples.',''])
    return '\n'.join(lines)


def run(root):
    require(root.name=='vlsi_fault_detection_v2' and Path(__file__).resolve().parent==root,'copy script to ~/vlsi_fault_detection_v2 and run there')
    i=helper(root);v=i.load_verifier(root);e=v.Evidence(root)
    print('STAGE 11D-3J — VALIDATION FAILURE REVIEW AND PROJECT DISPOSITION',flush=True)
    with v.campaign_lock(e):
        for path in OUT.values():require(not(root/path).exists() and not(root/path).is_symlink(),f'output exists: {path}; preserve existing review')
        source=e.verify(Path(__file__).name,sha(Path(__file__)))
        p,h,c,raw=verify(e,i,v)
        d,flat,comparisons=summarize(p,h,raw)
        d.update(created_at_utc=datetime.now(timezone.utc).isoformat(),review_source=source,prior_disposition=e.records[AUDIT_I])
        pending=Path(tempfile.mkdtemp(prefix='stage_11d3j_pending_',dir=root/R))
        try:
            staged={key:pending/Path(path).name for key,path in OUT.items()}
            staged['review'].write_text(report(d,p,comparisons),encoding='utf-8')
            i.write_csv(staged['subgroup_metrics'],flat);i.write_csv(staged['subgroup_comparisons'],comparisons)
            i.write_json(staged['disposition'],d)
            records={key:i.record(path,OUT[key]) for key,path in staged.items() if key!='audit'}
            print('RECHECKING FROZEN EVIDENCE BEFORE PUBLICATION',flush=True);e.recheck()
            audit={**d,'audit_status':'FROZEN','outputs':records,'input_evidence':e.records,
                'checks':{'prior_evidence':'PASS','acceptance_reconstruction':'PASS','group_inventory':'PASS','input_recheck':'PASS'},
                'completion_rule':'AUDIT_AND_ALL_MATCHING_OUTPUT_HASHES_REQUIRED'}
            i.write_json(staged['audit'],audit)
            for key in OUT:
                dest=e.path(OUT[key]);dest.parent.mkdir(parents=True,exist_ok=True)
                os.link(staged[key],dest);require(sha(dest)==sha(staged[key]),f'publication hash: {key}')
            shutil.rmtree(pending)
        except BaseException:
            print(f'STOP: partial evidence preserved at {pending}; require final audit and matching hashes',file=sys.stderr)
            raise
        print('\nSTAGE 11D-3J — VALIDATION FAILURE REVIEW AND PROJECT DISPOSITION FREEZE')
        for label,value in [('Status','PASS'),('Review status','FROZEN / COMPLETE'),('Current experiment','ACCEPTANCE_NOT_MET; STOP_BEFORE_HOLDOUT'),
            ('Failed criterion','mcc_retention'),('Hybrid validation MCC',f"{d['hybrid_validation_mcc']:.8f}"),
            ('Cause of retention failure','NOT ESTABLISHED'),('HOLDOUT','BLOCKED / NOT AUTHORIZED'),
            ('Retraining / threshold changes','NOT AUTHORIZED'),('Project final freeze','NOT DECLARED'),('Automatic continuation','NONE')]:
            print(f'{label:32}: {value}')
        for key,path in OUT.items():print(f'{key:32}: {root/path}\n{key+" SHA":32}: {sha(root/path)}')
        print('Follow-up                       : Review the findings and project disposition with your mentor.')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:run(Path.cwd().resolve())
    except KeyboardInterrupt:print('STOP: interrupted; preserve evidence',file=sys.stderr);return 130
    except Exception as error:print(f'STOP: {type(error).__name__}: {error}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
