"""CPU checks for bounded concurrency, failure propagation and action time alignment."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch
from threading import Event
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.runtime.async_replan import AsyncReplan, snapshot_history
from thinker_wm.runtime.robodojo_policy import RoboDojoPolicy

class AsyncTests(unittest.TestCase):
    def test_worker_does_not_block_and_never_overlaps(self):
        gate=Event(); worker=AsyncReplan()
        try:
            worker.start(8, lambda: gate.wait(5))
            self.assertTrue(worker.pending)
            with self.assertRaises(RuntimeError): worker.start(9, lambda: None)
            gate.set(); self.assertEqual(worker.take(), (8, True))
        finally: gate.set(); worker.close()
        worker.start(0, lambda: 'next episode')
        self.assertEqual(worker.take(), (0, 'next episode')); worker.close()

    def test_failure_is_not_silently_ignored(self):
        w=AsyncReplan()
        def fail(): raise ValueError('inference failed')
        w.start(0,fail)
        with self.assertRaisesRegex(ValueError,'inference failed'): w.close()
        self.assertFalse(w.pending); self.assertIsNone(w.executor)

    def test_snapshot_survives_live_history_pruning(self):
        h=NS(detector=NS(records=[1]), _cache_key='old',_cache_val='old',_sparse_vae_cache='old',_encoding_identity='old')
        for name in ('_rgb','_state','_times','_event_rgb','_event_state','_event_times'):setattr(h,name,{0:np.zeros(2)})
        snap=snapshot_history(h);h._rgb.clear();h.detector.records.append(2)
        self.assertIn(0,snap._rgb);self.assertEqual(snap.detector.records,[1]);self.assertIsNone(snap._cache_val)

    def test_time_aligned_tail_not_replayed_prefix(self):
        p=RoboDojoPolicy.__new__(RoboDojoPolicy)
        p.horizon=NS(predict_steps=24);p.block_size=16;p.observation=NS(step=16,state=np.zeros(14));p.next_decision=16
        p.history=NS(describe=lambda:{});p.instruction='stack';p.episode_id='test';p.dims=None;p.continuity=None
        records=[];p._record=records.append;p.async_replan=AsyncReplan()
        raw=np.repeat(np.arange(24,dtype=np.float32)[:,None],14,axis=1)
        p.async_replan.start(8,lambda:NS(actions=raw))
        module=ModuleType('XPolicyLab.utils.process_data');module.unpack_robot_state=lambda x,*a,**k:x
        try:
            with patch.dict(sys.modules,{'XPolicyLab.utils.process_data':module}):out=p.get_action()
            np.testing.assert_array_equal(out[:,0],np.arange(8,24))
            self.assertEqual(p.next_decision,32);self.assertEqual(records[0]['skipped_elapsed_actions'],8)
        finally:p.async_replan.close()

if __name__=='__main__':unittest.main()
