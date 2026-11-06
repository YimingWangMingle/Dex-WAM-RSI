"""Decode real 8-window training batches for all newly added tasks, CPU only."""
from pathlib import Path
import sys,gc,time,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from omegaconf import OmegaConf
from thinker_wm.data_prepare.finite_validation import FiniteValidationPipeline
from thinker_wm.data_prepare.task_cycled_ranges import TaskCycledRangesPool
from thinker_wm.data_prepare.factory import IndexedEpisodeBatch, build_depth_attacher, _producer_transform
from thinker_wm.data_prepare.history_attach import build_history_attacher
import torch

cfg=OmegaConf.load(sys.argv[1])
pipe=FiniteValidationPipeline(cfg,rank=0,world_size=1,seed=int(cfg.training.seed))
pool=TaskCycledRangesPool(cfg.dataset.sources[0].pool.path,pipe.repository,group_size=8,
    cycle_base_step=0,seed=int(cfg.training.seed),action_start_offset=0,text_choice='random')
pipe.dataset.sample_pool=pool
transform=_producer_transform(build_depth_attacher(cfg),build_history_attacher(cfg))
tasks={r['episode_id']:r['meta']['task_name'] for r in map(json.loads,Path(cfg.dataset.sources[0].catalog_path).read_text().splitlines())}
wanted={'plug_in_charger','insert_key','fold_clothes'}
try:
 for group in range(len(pool)//8):
  i=group*8
  episode=pool.generate(i)['episode_id']
  task=tasks[episode]
  if task not in wanted:continue
  start=time.monotonic()
  samples=pipe.dataset.get_grouped_samples(list(range(i,i+8)))
  batch=dict(pipe.adapter.adapt(IndexedEpisodeBatch(info=samples[0].info,samples=samples)))
  batch=transform(batch)
  assert tuple(batch['action_sequence'].shape)==(8,24,14)
  assert tuple(batch['video_frames'].shape)==(8,12,3,480,640)
  assert 'history_recent_rgb' in batch
  assert any('depth' in key for key in batch)
  assert all(torch.isfinite(v).all() for v in batch.values() if torch.is_tensor(v) and v.is_floating_point())
  print('PASS',task,'batch8 actions24/native14 frames12 history/depth',round(time.monotonic()-start,2),'seconds',flush=True)
  del samples,batch
  gc.collect()
  wanted.remove(task)
  if not wanted:break
 assert not wanted,wanted
finally:pipe.close()
