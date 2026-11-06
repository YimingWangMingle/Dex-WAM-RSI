"""Real release checks without model loading or pixel decoding."""
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from omegaconf import OmegaConf
from thinker_wm.data_prepare.finite_validation import FiniteValidationPipeline
from thinker_wm.data_prepare.task_cycled_ranges import TaskCycledRangesPool
from motus_data.sample_pool import EpisodeGroupedBatchSampler

cfg=OmegaConf.load(sys.argv[1])
old=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
source=old.dataset.sources[0]
for key in ('catalog_path','index_path'):
 source[key]=str(source[key]).replace('robodojo_6task_joint24_v1','robodojo_3task_v1')
for key in ('pool','eval_pool'):
 source[key].path=str(source[key].path).replace('robodojo_6task_joint24_v1','robodojo_3task_v1')
old.system.val_num_batches=15
old.system.val_episode_order_path=None
old.system.val_windows_per_task=None
seed=int(cfg.training.seed)
base=FiniteValidationPipeline(old,rank=0,world_size=1,seed=seed)
full=FiniteValidationPipeline(cfg,rank=0,world_size=1,seed=seed)
try:
 old_sel=[base.dataset.sample_pool.generate(i) for i in base.indices]
 new_sel=[full.dataset.sample_pool.generate(i) for i in full.indices]
 assert len(new_sel)==120
 def identity(s):return (s['episode_id'],tuple(s['video_frame_indices']),tuple(s['action_indices']))
 assert len({identity(s) for s in new_sel})==120
 old_ids={identity(s) for s in old_sel}
 assert sum(identity(s) in old_ids for s in new_sel)==60
 catalog={r['episode_id']:r for r in map(json.loads,Path(cfg.dataset.sources[0].catalog_path).read_text().splitlines())}
 split=json.loads((Path(cfg.dataset.sources[0].catalog_path).parent/'robodojo.split.json').read_text())
 train,holdout=set(split['train']),set(split['val'])
 assert len(train)==570 and len(holdout)==30 and not train&holdout
 assert {s['episode_id'] for s in new_sel}==holdout
 counts=Counter(catalog[s['episode_id']]['meta']['task_name'] for s in new_sel)
 assert counts=={task:20 for task in cfg.system.val_tasks}
 episode_counts=Counter(s['episode_id'] for s in new_sel)
 for task in cfg.system.val_tasks:
  values=[n for eid,n in episode_counts.items() if catalog[eid]['meta']['task_name']==task]
  assert max(values)-min(values)<=1
 repeat=FiniteValidationPipeline(cfg,rank=0,world_size=1,seed=seed)
 try:assert [repeat.dataset.sample_pool.generate(i) for i in repeat.indices]==new_sel
 finally:repeat.close()
 for bad_quota in (1,100):
  invalid=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
  invalid.system.val_windows_per_task=bad_quota
  try:
   bad=FiniteValidationPipeline(invalid,rank=0,world_size=1,seed=seed)
  except ValueError:pass
  else:
   bad.close()
   raise AssertionError('invalid quota accepted')
 assert all(len(s['action_indices'])==24 and s['state_indices'][0]==s['action_indices'][0] for s in new_sel)
 ids=[]
 for rank in range(16):
  shard=FiniteValidationPipeline(cfg,rank=rank,world_size=16,seed=seed)
  try:
   assert len(shard.global_ids)==(8 if rank<8 else 7)
   for ident,index in zip(shard.global_ids,shard.indices):
    assert shard.dataset.sample_pool.generate(index)==new_sel[ident]
   ids+=shard.global_ids
  finally:shard.close()
 assert sorted(ids)==list(range(120))
 pool=TaskCycledRangesPool(cfg.dataset.sources[0].pool.path,full.repository,
     group_size=8,cycle_base_step=0,seed=seed,action_start_offset=0,text_choice='random')
 assert len(pool._tasks)==6
 # Task sequence is balanced independently of episode length.
 assert Counter(pool._tasks[pool.deal_at(i)[0]] for i in range(600))=={t:100 for t in pool._tasks}
 dataset=NS(episode_groups=pool.episode_groups,set_epoch=pool.set_epoch)
 batches=[]
 for rank in range(16):
  sampler=EpisodeGroupedBatchSampler(dataset,batch_size=8,num_replicas=16,rank=rank,seed=seed,drop_last=True)
  batches.append(list(sampler))
 assert len({len(x) for x in batches})==1
 groups=[b[0]//8 for part in batches for b in part]
 assert len(groups)==len(set(groups))
 initial=[pool.generate(i) for i in range(64)]
 pool.set_epoch(1)
 assert initial!=[pool.generate(i) for i in range(64)]
 pool.set_epoch(0)
 assert initial==[pool.generate(i) for i in range(64)]
 report=dict(train_episodes=570,holdout_episodes=30,validation_windows=120,windows_by_task=dict(counts),
  distinct_fixed_windows=True,all_holdout_episodes_covered=True,
  validation_windows_per_rank=[8]*8+[7]*8,episode_window_counts=dict(episode_counts),
  train_steps_per_cycle_16gpu=len(batches[0]),tasks=list(pool._tasks),equal_task_sampling=True)
 print(json.dumps(report,indent=2))
finally:
 base.close();full.close()
