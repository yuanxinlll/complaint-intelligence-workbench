"""Review statistics for legacy CSV and CIW review exports / JSONL journals.
No model loading, inference, network access, or label modification.
"""
import argparse,csv,json,math,sys
from pathlib import Path
from collections import Counter

def flag(value):
    if value is None or value=='':return False
    if isinstance(value,bool):return value
    v=str(value).strip().lower()
    if v in ('true','1','yes'):return True
    if v in ('false','0','no'):return False
    raise ValueError('Invalid boolean flag: '+repr(value))

def labels(value):
    if value is None:return set()
    if isinstance(value,list):return {str(x).strip() for x in value if str(x).strip()}
    text=str(value).strip()
    if text.startswith('['):return labels(json.loads(text))
    return {x.strip() for x in text.replace(';','|').split('|') if x.strip()}

def read(path,duplicates='error',revisions='error',role='review'):
    path=Path(path)
    if path.suffix.lower()=='.jsonl':rows=[json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    else:
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);header=reader.fieldnames or []
            if len(set(header))!=len(header):raise ValueError('Duplicate CSV headers')
            rows=list(reader)
    ids=[str(r.get('sample_id') or r.get('review_id') or '').strip() for r in rows]
    if any(not x for x in ids):raise ValueError('Missing sample_id / review_id')
    duplicate_ids=sorted(k for k,n in Counter(ids).items() if n>1)
    journal=path.suffix.lower()=='.jsonl'
    if duplicate_ids and not (journal and revisions=='latest') and duplicates=='error':raise ValueError('Duplicate sample IDs: '+', '.join(duplicate_ids))
    packets={r.get('packet_sha256') for r in rows if r.get('packet_sha256')};reviewers={r.get('reviewer') or r.get('reviewer_file') for r in rows if r.get('reviewer') or r.get('reviewer_file')}
    if role=='review' and (len(packets)>1 or len(reviewers)>1):raise ValueError('Mixed reviewer identities or packet versions; split the file first')
    selected={}
    for rid,row in zip(ids,rows):
        if rid in selected:
            if journal and revisions=='latest':
                old=selected[rid]
                if not old.get('event_id') or row.get('supersedes_event_id')!=old['event_id']:raise ValueError('Unlinked or duplicate JSONL event for '+rid)
            elif duplicates=='first':continue
            else:raise ValueError('Duplicate sample ID '+rid)
        selected[rid]=row
    out={}
    for rid,row in selected.items():
        if role=='review':
            if 'primary_category' in row:primary=labels(row['primary_category']);extra=labels(row.get('other_reasonable_categories'))
            elif 'label' in row:primary=labels(row['label']);extra=set()
            else:raise ValueError('Review file needs primary_category or label; author expectations are not independent reviewer answers')
            suff=str(row.get('sufficiency','')).strip().lower()
            if suff not in ('','enough','insufficient','multiple'):raise ValueError('Unknown sufficiency: '+suff)
            choices=primary|extra;multi=flag(row.get('multiple_reasonable')) or suff=='multiple' or len(choices)>1
            out[rid]={'labels':choices,'primary':primary,'insufficient':flag(row.get('insufficient_information')) or suff=='insufficient','multiple':multi,'raw':row}
        else:
            key=next((k for k in (('model_label','predicted_issue') if role=='model' else ('issue',)) if k in row),None)
            if key is None:raise ValueError('Reference needs issue; model predictions need model_label/predicted_issue. Author expectations cannot be substituted.')
            out[rid]={'labels':labels(row[key])}
    return out,{'rows_read':len(rows),'unique_samples':len(out),'duplicate_ids':duplicate_ids,'revision_policy':revisions,'duplicate_policy':duplicates,'reviewer_identities':sorted(reviewers),'packet_versions':sorted(packets)}

def reviewer_summary(rows):
    n=len(rows);return {'samples':n,'labelled_samples':sum(bool(x['labels']) for x in rows.values()),'insufficient_count':sum(x['insufficient'] for x in rows.values()),'multiple_reasonable_count':sum(x['multiple'] for x in rows.values()),'proportions_denominator':'all unique samples in this reviewer file','insufficient_proportion':sum(x['insufficient'] for x in rows.values())/n if n else None,'multiple_reasonable_proportion':sum(x['multiple'] for x in rows.values())/n if n else None}

def compare_reference(a,b):
    ids=[i for i in a if i in b and a[i]['labels'] and b[i]['labels']];return {'comparable':len(ids),'exact_set_matches':sum(a[i]['labels']==b[i]['labels'] for i in ids),'any_overlap_matches':sum(bool(a[i]['labels']&b[i]['labels']) for i in ids),'missing_reference_ids':len(set(a)-set(b)),'note':'Agreement, not adjudicated accuracy. Insufficient/multiple flags are not silently excluded.'}

def stats(args):
    a,am=read(args.a,args.duplicates,args.jsonl_revisions);result={'mode':'single_reviewer' if not args.b else 'two_reviewers','reviewer_A':reviewer_summary(a),'input_A':am,'warnings':[]}
    if not args.b:
        result['human_human']={'computed':False,'reason':'Human-human agreement and kappa require a second independent reviewer; not calculated.'}
    else:
        if Path(args.a).resolve()==Path(args.b).resolve():raise ValueError('A and B must not be the same file')
        b,bm=read(args.b,args.duplicates,args.jsonl_revisions)
        if set(am['reviewer_identities'])&set(bm['reviewer_identities']):raise ValueError('Same reviewer identity detected; not independent reviewers')
        if am['packet_versions'] and bm['packet_versions'] and am['packet_versions']!=bm['packet_versions']:raise ValueError('Reviewers used different packets')
        ids=[i for i in a if i in b and a[i]['labels'] and b[i]['labels']];n=len(ids)
        result.update(reviewer_B=reviewer_summary(b),input_B=bm)
        hh={'comparable':n,'only_A_ids':len(set(a)-set(b)),'only_B_ids':len(set(b)-set(a)),'exact_set_agreement_count':sum(a[i]['labels']==b[i]['labels'] for i in ids),'any_overlap_count':sum(bool(a[i]['labels']&b[i]['labels']) for i in ids),'independence':'Separate identities are not proof of blind independent review; confirm collection procedure.'}
        single=[i for i in ids if len(a[i]['labels'])==len(b[i]['labels'])==1 and not a[i]['multiple'] and not b[i]['multiple']];hh['kappa_single_label_n']=len(single);hh['kappa_multilabel_excluded']=n-len(single)
        if not n:hh.update(message='No samples labelled by both reviewers',exact_set_agreement=None,any_overlap_agreement=None)
        else:hh.update(exact_set_agreement=hh['exact_set_agreement_count']/n,any_overlap_agreement=hh['any_overlap_count']/n)
        if not single:hh.update(kappa=None,kappa_reason='No eligible single-label pairs')
        else:
            x=[next(iter(a[i]['labels'])) for i in single];y=[next(iter(b[i]['labels'])) for i in single];c,d=Counter(x),Counter(y);nn=len(x);po=sum(u==v for u,v in zip(x,y))/nn;pe=sum(c[k]*d[k] for k in c)/nn**2
            hh.update(kappa=None if math.isclose(pe,1) else (po-pe)/(1-pe),kappa_reason='Undefined: expected agreement is one' if math.isclose(pe,1) else 'Single-label pairs only; not an accuracy ceiling')
        result['human_human']=hh
    for opt,role,key in [(args.model,'model','reviewer_A_vs_model'),(args.gold,'reference','reviewer_A_vs_CFPB_original_label')]:
        if opt:
            ref,meta=read(opt,args.duplicates,args.jsonl_revisions,role);result[key]=compare_reference(a,ref)
            if args.b:result[key.replace('reviewer_A','reviewer_B')]=compare_reference(b,ref)
    if not args.model:result['warnings'].append('No matched model predictions provided; model agreement not calculated.')
    if not args.gold:result['warnings'].append('No CFPB original labels provided; reference agreement not calculated. Synthetic author labels are not CFPB labels.')
    result['warnings'].append('A small non-random diagnostic set cannot estimate a population accuracy ceiling. No answers or labels were modified.')
    return result

def output_precheck(parser,path):
    if path and Path(path).exists():parser.exit(2,f"output file already exists, not overwritten: {path} (choose a new name)\n")

def main():
    # Keep the original subcommand spellings while allowing direct stats flags.
    if len(sys.argv)>1 and sys.argv[1]=='make-sheet':
        parser=argparse.ArgumentParser();parser.add_argument('cmd');parser.add_argument('--samples',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
        output_precheck(parser,args.out)
        with Path(args.samples).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        ids=[str(r.get('sample_id') or r.get('review_id') or '').strip() for r in rows]
        if any(not i for i in ids) or len(set(ids))!=len(ids):parser.error('Missing or duplicate sample IDs')
        if any('narrative' not in r and 'text' not in r for r in rows):parser.error('Samples need narrative or text')
        try:
            with Path(args.out).open('x',encoding='utf-8',newline='') as f:
                w=csv.writer(f);w.writerow(['sample_id','narrative','label','sufficiency','confidence'])
                for rid,r in zip(ids,rows):w.writerow([rid,r.get('narrative',r.get('text','')),'','',''])
        except FileExistsError:parser.exit(2,f"output file already exists, not overwritten: {args.out} (choose a new name)\n")
        print('Blank sheet saved; no labels or predictions included.');return
    if len(sys.argv)>1 and sys.argv[1]=='stats':sys.argv.pop(1)
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--a',required=True);ap.add_argument('--b');ap.add_argument('--model');ap.add_argument('--gold',help='CFPB original issue file only, not synthetic author expectations');ap.add_argument('--duplicates',choices=['error','first'],default='error');ap.add_argument('--jsonl-revisions',choices=['error','latest'],default='error');ap.add_argument('--out');args=ap.parse_args()
    output_precheck(ap,args.out)
    try:result=stats(args)
    except (ValueError,KeyError,OSError,json.JSONDecodeError) as e:ap.exit(2,'Input error: '+str(e)+'\n')
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.out:
        try:
            with Path(args.out).open('x',encoding='utf-8') as f:f.write(text+'\n')
        except FileExistsError:ap.exit(2,f"output file already exists, not overwritten: {args.out} (choose a new name)\n")
    print(text)
if __name__=='__main__':main()
