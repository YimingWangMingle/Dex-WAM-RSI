"""Task loss diagnostics: local training batches and held-out decoded samples."""
import json
import math
from collections import defaultdict

from .sampling_coverage import update_index

TRAIN_METRICS=('total_loss','video_loss','depth_loss','action_loss')
EVAL_METRICS=('video_mse','video_psnr','depth_gray_mse','depth_mse_m2','depth_absrel','action_mse')


def get_task_metrics(run_dir, cache_path, task='', split='holdout'):
    metadata=run_dir/'sampling_catalog.json'
    catalog=json.loads(metadata.read_text()) if metadata.exists() else {}
    labels={(e['source'],e['episode_id']):e['task']
            for e in catalog.get('task_labels',catalog.get('episodes',[]))}
    all_tasks=sorted(set(labels.values()))
    db=update_index(run_dir,cache_path)
    try:
        db.execute('CREATE TEMP TABLE labels(source TEXT,episode TEXT,task TEXT,PRIMARY KEY(source,episode))')
        db.executemany('INSERT INTO labels VALUES (?,?,?)',((s,e,t) for (s,e),t in labels.items()))
        first,last=db.execute('SELECT min(step),max(step) FROM task_losses').fetchone()
        summaries={t:dict(task=t,train={},validation={},validation_samples=0) for t in all_tasks}
        q="""SELECT coalesce(l.task,'未知任务'),a.metric,sum(a.value*a.n)/sum(a.n),count(*),sum(a.n)
          FROM task_losses a LEFT JOIN labels l ON a.source=l.source AND a.episode=l.episode
          WHERE a.step>? GROUP BY 1,2"""
        for t,m,mean,n,samples in db.execute(q,(max(0,(last or 0)-1000),)):
            summaries.setdefault(t,dict(task=t,train={},validation={},validation_samples=0))['train'][m]=dict(mean=mean,batches=n,samples=samples)
        width=max(1,((last or 0)+199)//200)
        filt=" AND coalesce(l.task,'未知任务')=?" if task else ''
        params=[width]+([task] if task else [])
        train_curve=defaultdict(dict)
        q="""SELECT (a.step-1)/?,a.metric,sum(a.value*a.n)/sum(a.n),count(*)
          FROM task_losses a LEFT JOIN labels l ON a.source=l.source AND a.episode=l.episode
          WHERE 1=1"""+filt+" GROUP BY 1,2"
        for b,m,v,n in db.execute(q,params):
            train_curve[min((b+1)*width,last)][m]=dict(value=v,n=n)
    finally:
        db.close()
    # Evaluation files are small; preserve the last record for a replayed sample.
    records={}
    for path in sorted((run_dir/'eval').glob('*.jsonl')):
        with path.open() as stream:
            for line in stream:
                try:r=json.loads(line)
                except ValueError:continue
                if r.get('type')!='eval_sample' or r.get('split','holdout')!=split:continue
                key=(r.get('step'),r.get('rank'),r.get('sample_index'),r.get('source'),r.get('split'))
                records[key]=r
    buckets=defaultdict(lambda:defaultdict(list)); sample_counts=defaultdict(int)
    for r in records.values():
        t=labels.get((r.get('source'),r.get('episode_id')),'未知任务'); step=int(r['step'])
        sample_counts[step,t]+=1
        for m in EVAL_METRICS:
            v=r.get('metrics',{}).get(m)
            if isinstance(v,(float,int)) and math.isfinite(v):buckets[step,t][m].append(v)
    eval_last=max((step for step,t in sample_counts),default=None)
    eval_curve=defaultdict(lambda:defaultdict(list))
    for (step,t), metrics in buckets.items():
        row=summaries.setdefault(t,dict(task=t,train={},validation={},validation_samples=0))
        if step==eval_last:
            row['validation_samples']=sample_counts[step,t]
            row['validation']={m:dict(mean=sum(vs)/len(vs),n=len(vs)) for m,vs in metrics.items()}
        if not task or task==t:
            for m,vs in metrics.items():eval_curve[step][m].extend(vs)
    return dict(tasks=sorted(summaries.values(),key=lambda x:x['task']),
        train_first_step=first,train_last_step=last,train_recent_steps=1000,
        train_curve=[dict(step=s,metrics=m) for s,m in sorted(train_curve.items())],
        eval_last_step=eval_last,split=split,
        eval_curve=[dict(step=s,metrics={m:dict(value=sum(v)/len(v),n=len(v)) for m,v in ms.items()})
                    for s,ms in sorted(eval_curve.items())],
        note='训练使用跨卡平均前的本地 batch loss，按样本数加权；被 dropout 的分支不计入对应均值，total 仅统计无模态 dropout 的步。历史未记录本地 loss 时显示缺失，不能由全局均值反推。验证按样本等权汇总，缺失指标不当作 0。')
