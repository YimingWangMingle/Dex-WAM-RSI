"""Incremental, disk-backed sample accounting. No training-process dependencies."""
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def update_index(run_dir, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache_path, timeout=120)
    db.executescript("""
      CREATE TABLE IF NOT EXISTS files(name TEXT PRIMARY KEY, pos INTEGER, last_step INTEGER);
      CREATE TABLE IF NOT EXISTS samples(
        rank INTEGER, step INTEGER, slot INTEGER, source TEXT, episode TEXT,
        start INTEGER, window TEXT, PRIMARY KEY(rank,step,slot));
      CREATE INDEX IF NOT EXISTS ep_idx ON samples(source,episode,window,step);
      CREATE TABLE IF NOT EXISTS task_losses(rank INTEGER,step INTEGER,source TEXT,
        episode TEXT,metric TEXT,value REAL,n INTEGER,PRIMARY KEY(rank,step,metric));
    """)
    paths = sorted((run_dir / 'slices').glob('rank*.jsonl'))
    names = {p.name for p in paths}
    for name, in db.execute('SELECT name FROM files').fetchall():
        if name not in names:
            rank = int(name[4:-6])
            db.execute('DELETE FROM samples WHERE rank=?', (rank,))
            db.execute('DELETE FROM task_losses WHERE rank=?', (rank,))
            db.execute('DELETE FROM files WHERE name=?', (name,))
    for path in paths:
        rank = int(path.stem[4:])
        old = db.execute('SELECT pos,last_step FROM files WHERE name=?', (path.name,)).fetchone()
        pos, last = old or (0, -1)
        size = path.stat().st_size
        if size < pos:
            db.execute('DELETE FROM samples WHERE rank=?', (rank,))
            db.execute('DELETE FROM task_losses WHERE rank=?', (rank,))
            pos, last = 0, -1
        if size == pos:
            continue
        with path.open('rb') as stream:
            stream.seek(pos)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                pos = stream.tell()
                try:
                    record = json.loads(line)
                    step = int(record['step'])
                    if int(record.get('rank', rank)) != rank:
                        continue
                except (ValueError, KeyError, TypeError):
                    continue
                # A rank replaying an earlier step invalidates its abandoned tail.
                if step <= last:
                    db.execute('DELETE FROM samples WHERE rank=? AND step>=?', (rank, step))
                    db.execute('DELETE FROM task_losses WHERE rank=? AND step>=?', (rank, step))
                last = step
                if record.get('optimizer_step_skipped'):
                    continue
                rows = []
                for slot, sample in enumerate(record.get('samples') or []):
                    frames = sample.get('frame_indices') or []
                    start = sample.get('window_start')
                    if start is None and frames:
                        start = frames[0]
                    if start is None or not sample.get('episode_id'):
                        continue
                    identity = json.dumps([frames, sample.get('ratio'), start], separators=(',', ':'))
                    window = hashlib.sha256(identity.encode()).hexdigest()[:24]
                    rows.append((rank, step, slot, sample.get('source') or 'unknown',
                                 str(sample['episode_id']), int(start), window))
                db.executemany('INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?)', rows)
                identities = {(row[3],row[4]) for row in rows}
                # Only attribute homogeneous batches. Never copy a global loss.
                if len(identities) == 1:
                    source, episode = next(iter(identities))
                    metrics = record.get('local_metrics') or {}
                    for metric,modality in [('total_loss',None),('video_loss','video'),
                                            ('action_loss','action'),('depth_loss','depth')]:
                        value=metrics.get(metric)
                        dropped=(metrics.get('modality_dropped_'+modality,False) if modality else
                                 any(metrics.get('modality_dropped_'+m,False) for m in ('video','action','depth','prompt')))
                        if isinstance(value,(int,float)) and math.isfinite(value) and not dropped:
                            db.execute('INSERT OR REPLACE INTO task_losses VALUES (?,?,?,?,?,?,?)',
                                       (rank,step,source,episode,metric,float(value),len(rows)))
        db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?)', (path.name, pos, last))
        db.commit()
    db.commit()
    return db


def report(db, catalog, task='', episode='', max_points=250):
    entries = catalog.get('episodes', [])
    db.execute('CREATE TEMP TABLE IF NOT EXISTS labels(source TEXT,episode TEXT,task TEXT,total INTEGER,PRIMARY KEY(source,episode))')
    db.execute('DELETE FROM labels')
    db.executemany('INSERT OR REPLACE INTO labels VALUES (?,?,?,?)',
                   ((x['source'], x['episode_id'], x['task'], x['total_windows']) for x in entries))
    stats = {(s,e): (n,u,last) for s,e,n,u,last in db.execute(
        'SELECT source,episode,count(*),count(DISTINCT window),max(step) FROM samples GROUP BY source,episode')}
    all_eps = {(x['source'],x['episode_id']): dict(x) for x in entries}
    for key in stats:
        all_eps.setdefault(key, dict(source=key[0],episode_id=key[1],task='未知任务',total_windows=None))
    tasks = {}
    selected = []
    for key, row in all_eps.items():
        n,u,last = stats.get(key, (0,0,None))
        row.update(samples=n, unique_windows=u, last_step=last)
        row['coverage'] = u/row['total_windows'] if row.get('total_windows') else None
        t = tasks.setdefault(row['task'], dict(task=row['task'],samples=0,unique_windows=0,
                            episodes=0,seen_episodes=0,total_windows=0))
        t['samples'] += n; t['unique_windows'] += u; t['episodes'] += 1
        t['seen_episodes'] += int(n>0); t['total_windows'] += row.get('total_windows') or 0
        if not task or row['task'] == task:
            selected.append(row)
    where, params = [], []
    if task:
        where.append("coalesce(l.task,'未知任务')=?"); params.append(task)
    if episode:
        where.append('s.episode=?'); params.append(episode)
    join = ' FROM samples s LEFT JOIN labels l ON s.source=l.source AND s.episode=l.episode'
    condition = (' WHERE ' + ' AND '.join(where)) if where else ''
    first,last = db.execute('SELECT min(s.step),max(s.step)'+join+condition, params).fetchone()
    width = max(1, ((last or 0)+max_points-1)//max_points)
    totals = dict(db.execute('SELECT (s.step-1)/?,count(*)'+join+condition+' GROUP BY 1', [width]+params))
    uniques = dict(db.execute('SELECT (first_step-1)/?,count(*) FROM (SELECT min(s.step) first_step'+
        join+condition+' GROUP BY s.source,s.episode,s.window) GROUP BY 1', [width]+params))
    denom_rows = [x for x in selected if not episode or x['episode_id']==episode]
    denominator = sum(x.get('total_windows') or 0 for x in denom_rows)
    denominator_known = bool(denom_rows) and all(x.get('total_windows') for x in denom_rows)
    curve=[]; n=u=0
    for bucket in range(((last-1)//width+1) if last else 0):
        n += totals.get(bucket,0); u += uniques.get(bucket,0)
        curve.append(dict(step=min((bucket+1)*width,last),samples=n,unique_windows=u,
                          coverage=u/denominator if denominator_known else None))
    return dict(tasks=sorted(tasks.values(),key=lambda x:-x['samples']),
        episodes=sorted(selected,key=lambda x:(x['samples'],x['episode_id'])),curve=curve,
        first_step=first,last_step=last,bin_steps=width,metadata_available=bool(entries),
        ranks=[r for r, in db.execute('SELECT DISTINCT rank FROM samples ORDER BY rank')],
        note='统计已记录的训练窗口，不等同于动作有效更新；各 rank 同 step 重跑覆盖旧记录。缺失 rank 或未记录 step 不作补估。覆盖率分母为该 run ranges 中有效起点数。')


def get_report(run_dir, cache_path, task='', episode=''):
    metadata_path = run_dir / 'sampling_catalog.json'
    catalog = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    db = update_index(run_dir, cache_path)
    try:
        return report(db, catalog, task, episode)
    finally:
        db.close()
