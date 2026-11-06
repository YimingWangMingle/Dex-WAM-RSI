"""Build a selected-task release, preserving splits of retained episodes.

CPU-only build. Refuses to replace an existing release. Config is updated only
separately, after the build and its input/publisher tests pass.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thinker_wm.data_prepare.robodojo_core import FieldMap, RoboDojoHDF5, episode_record
from scripts.build_robodojo_catalog import _validators
from thinker_wm.models.action_stats import ACTION_STATS


ROBOT = 'robodojo_arx_x5'


def read_ranges(path):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute('SELECT * FROM episode_ranges ORDER BY ordinal')]


def extend(args):
    old, out = Path(args.base), Path(args.output)
    if out.exists():
        raise ValueError(f'release already exists: {out}')
    fm = FieldMap.load(args.field_map)
    reader = RoboDojoHDF5(fm)
    validators = _validators()
    if validators is None:
        raise RuntimeError('catalog contract validation is required')
    old_rows = list(map(json.loads, (old / 'robodojo.episode_catalog.jsonl').read_text().splitlines()))
    old_split = json.loads((old / 'robodojo.split.json').read_text())
    old_ranges = {s: read_ranges(old / f'robodojo.{s}.rsi_f12_r2.ranges.sqlite') for s in ('train','holdout')}
    for name, key in (('train','train'), ('holdout','val')):
        assert {r['episode_id'] for r in old_ranges[name]} == set(old_split[key])
    selected_tasks = tuple(args.tasks)
    if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError('tasks must be nonempty and unique')
    original_tasks = {r['meta']['task_name'] for r in old_rows}
    new_tasks = tuple(t for t in selected_tasks if t not in original_tasks)
    removed_tasks = sorted(original_tasks - set(selected_tasks))
    old_rows = [r for r in old_rows if r['meta']['task_name'] in selected_tasks]
    retained_ids = {r['episode_id'] for r in old_rows}
    old_split = dict(old_split, **{k: [eid for eid in old_split[k] if eid in retained_ids]
                                 for k in ('train','val','excluded')})
    old_ranges = {k: [r for r in rows if r['episode_id'] in retained_ids]
                  for k, rows in old_ranges.items()}
    new_rows = []
    for task in new_tasks:
        files = sorted((Path(args.data_root) / task / 'arx_x5/data').glob('*.hdf5'))
        if len(files) != 100:
            raise ValueError(f'{task}: expected 100 episodes, got {len(files)}')
        for path in files:
            row = episode_record(path, fm, reader, robot=ROBOT, task_name=task)
            validators['record'](row)
            new_rows.append(row)
    records = old_rows + new_rows
    ids = {r['episode_id'] for r in records}
    assert len(ids) == len(records)
    split = {k: list(old_split[k]) for k in ('train','val','excluded')}
    for task in new_tasks:
        task_rows = [r for r in new_rows if r['meta']['task_name'] == task]
        ranked = sorted(task_rows, key=lambda r: hashlib.sha256(
            f"{args.split_seed}:{r['episode_id']}".encode()).hexdigest())
        holdout = {r['episode_id'] for r in ranked[:5]}
        for row in task_rows:
            split['val' if row['episode_id'] in holdout else 'train'].append(row['episode_id'])
    train_ids = set(split['train'])
    val_ids = set(split['val'])
    assert not train_ids & val_ids
    assert train_ids | val_ids == ids
    # Existing checkpoint's normalization stays immutable. Audit new tasks in
    # that same coordinate system; six-task train-only stats are diagnostic.
    stats = ACTION_STATS[ROBOT]
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    audit, training_values = defaultdict(list), []
    new_ids = {r['episode_id'] for r in new_rows}
    for index, row in enumerate(records):
        path = row['video']['storage']['path']
        task = row['meta']['task_name']
        n = row['video']['count']
        with h5py.File(path,'r') as f:
            fields = list(fm.state_paths) + list(fm.action_paths) + list(fm.rgb_views.values()) + list(fm.depth_views.values())
            assert all(len(f[k]) == n for k in fields), path
            state = np.concatenate([f[k][:] for k in fm.state_paths],axis=1)
            action = np.concatenate([f[k][:] for k in fm.action_paths],axis=1)
            assert state.shape == action.shape == (n,14), path
            assert np.isfinite(state).all() and np.isfinite(action).all(), path
            assert reader.instruction(path) == row['texts'][0]['text'], path
        if row['episode_id'] in new_ids:
            points = sorted({0,n//2,n-1})
            rgb = reader.read_rgb(path, points)
            depth, valid = reader.read_depth_mm(path, points)
            assert rgb.shape == (len(points),480,640,3)
            assert depth.shape == valid.shape == (len(points),480,640)
            assert valid.reshape(len(points),-1).any(axis=1).all(), path
        maximum = float(max(np.abs((action-mean)/std).max(), np.abs((state-mean)/std).max()))
        if maximum >= 50:
            raise ValueError(f'{path}: normalized action/state reaches outlier threshold: {maximum}')
        audit[task].append(dict(episode_id=row['episode_id'], rows=n, max_abs_z=maximum))
        if row['episode_id'] in train_ids:
            training_values.append(action.astype(np.float64))
        if (index+1) % 50 == 0:
            print(f'AUDIT {index+1}/{len(records)} numeric rows; new episodes RGB/depth first/middle/last',flush=True)
    out.mkdir(parents=True)
    catalog = out / 'robodojo.episode_catalog.jsonl'
    catalog.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records))
    metadata = json.loads((old/'robodojo.episode_catalog.meta.json').read_text())
    metadata.update(catalog_id=out.name, episodes=len(records), sha256=hashlib.sha256(catalog.read_bytes()).hexdigest())
    metadata['meta'].update(parent_catalog=str(old/'robodojo.episode_catalog.jsonl'), added_tasks=list(new_tasks), removed_tasks=removed_tasks, selected_tasks=list(selected_tasks), skipped=0)
    validators['meta'](metadata)
    for row in records: validators['bind'](row,metadata)
    (out/'robodojo.episode_catalog.meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    validators['index'](catalog,force=False)
    split['meta'] = dict(old_split['meta'], catalog_sha256=metadata['sha256'],
        split_source='frozen original split + 5 lowest sha256(seed:episode_id) per new task',
        added_task_split_seed=args.split_seed, parent_split=str(old/'robodojo.split.json'))
    (out/'robodojo.split.json').write_text(json.dumps(split,indent=2)+'\n')
    full = out/'robodojo.all.rsi_f12_r2.ranges.sqlite'
    subprocess.run([sys.executable,'-B',str(ROOT/'third_party/Thinker-WM/data_prepare/build_sampling_ranges.py'),
        '--catalog',str(catalog),'--dataset','robodojo','--robodojo-field-map',args.field_map,
        '--num-video-frames','12','--action-fps-ratio','2','--video-stride',
        '--action-start-offset','0','--tail-pad-steps','16','--all-text-indices',
        '--split','all','--workers','4','--output',str(full)],check=True)
    full_rows = {r['episode_id']:r for r in read_ranges(full)}
    assert set(full_rows) == ids
    for name,key in (('train','train'),('holdout','val')):
        original = old_ranges[name]
        for row in original:
            assert {k:v for k,v in row.items() if k!='ordinal'} == {k:v for k,v in full_rows[row['episode_id']].items() if k!='ordinal'}
        selected = set(split[key])
        rows = original + [full_rows[r['episode_id']] for r in new_rows if r['episode_id'] in selected]
        target = out/f'robodojo.{name}.rsi_f12_r2.ranges.sqlite'
        shutil.copy2(old/f'robodojo.{name}.rsi_f12_r2.ranges.sqlite',target)
        with sqlite3.connect(target) as db:
            db.execute('DELETE FROM episode_ranges')
            columns = list(rows[0])
            for ordinal,row in enumerate(rows):
                values = dict(row,ordinal=ordinal)
                db.execute('INSERT INTO episode_ranges ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',[values[k] for k in columns])
            for k,v in dict(catalog_id=out.name,episodes=str(len(rows)),split_manifest=str(out/'robodojo.split.json'),
                            split_policy='frozen_original_plus_exact_per_new_task').items():
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(k,v))
            assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    # Retained holdout order is stable; new episodes are appended deterministically.
    prior_order_path = old/'robodojo.validation_episodes.json'
    prior_order = json.loads(prior_order_path.read_text()) if prior_order_path.exists() else sorted(old_split['val'])
    order = [eid for eid in prior_order if eid in val_ids] + sorted(val_ids-set(prior_order))
    (out/'robodojo.validation_episodes.json').write_text(json.dumps(order,indent=2)+'\n')
    rows = np.concatenate(training_values)
    diagnostic = dict(policy='diagnostic_only_not_used_by_model', rows=len(rows), train_episodes=len(train_ids),
                      mean=rows.mean(axis=0).tolist(),std=rows.std(axis=0).tolist())
    (out/'train_stats_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    report = dict(episodes=len(records),train=len(train_ids),holdout=len(val_ids),validation_candidate_windows=len(val_ids)*8,
        rgb_depth_check=f'first/middle/last of all {len(new_rows)} new episodes; not every video frame',
        numeric_check=f'every state/action row in all {len(records)} episodes',
        normalization='frozen checkpoint-compatible three-task train statistics',normalization_values=stats,
        tasks={task:dict(episodes=len(items),train=sum(x['episode_id'] in train_ids for x in items),
                        holdout=sum(x['episode_id'] in val_ids for x in items),rows=sum(x['rows'] for x in items),
                        max_abs_z=max(x['max_abs_z'] for x in items)) for task,items in audit.items()},
        original_validation_episode_order=sorted(old_split['val']))
    (out/'build_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','output','data-root','field-map'):p.add_argument('--'+name,required=True)
    p.add_argument('--split-seed',type=int,required=True)
    p.add_argument('--tasks',nargs='+',required=True)
    extend(p.parse_args())
