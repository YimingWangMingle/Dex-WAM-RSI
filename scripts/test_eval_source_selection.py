from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from thinker_wm.data_prepare.factory import _indexed_sources

class EvalSourceSelection(unittest.TestCase):
 def config(self):
  return OmegaConf.create(dict(common=dict(num_video_frames=16),training=dict(batch_size=8),dataset=dict(sources=[
   dict(name='train_only',dataset='robotwin_3d',pool=dict(type='task_balanced_ranges',path='thinker_wm.train.sqlite')),
   dict(name='heldout',dataset='robotwin_3d',pool=dict(type='task_balanced_ranges',path='train2.sqlite'),eval_pool=dict(type='sampling_ranges',path='holdout.sqlite'))])))
 def test_selected_eval_does_not_validate_unused_training_pool(self):
  cfg=self.config();before=OmegaConf.to_container(cfg)
  result=_indexed_sources(cfg,eval=True,source_name='heldout')
  self.assertEqual([s['name'] for s in result],['heldout'])
  self.assertEqual(OmegaConf.to_container(cfg),before)
 def test_selected_invalid_source_still_rejected(self):
  with self.assertRaises(ValueError): _indexed_sources(self.config(),eval=True,source_name='train_only')
 def test_full_training_unchanged(self):
  self.assertEqual(len(_indexed_sources(self.config())),2)
 def test_full_eval_validation_not_silently_weakened(self):
  with self.assertRaises(ValueError): _indexed_sources(self.config(),eval=True)
 def test_missing_source_rejected(self):
  with self.assertRaises(ValueError): _indexed_sources(self.config(),eval=True,source_name='missing')
if __name__=='__main__': unittest.main()
