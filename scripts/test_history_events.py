"""Memory fixes: real geometry, causal D0, event arrival, train/runtime parity."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import _rsi_cpu_stubs
import unittest
from dataclasses import replace
import numpy as np
import torch
from thinker_wm.data_prepare.history_memory import (EventSpec, EventDetector, select_event_history,
                                         pad_recent_pixels, history_metrics)
from thinker_wm.data_prepare.temporal_windows import WindowSpec, select_window, check_window
from thinker_wm.data_prepare.history_attach import HistoryAttacher
from thinker_wm.runtime.history_buffer import HistoryBuffer
from thinker_wm.models.inference_api import Observation
from thinker_wm.models.history_encoder import HistoryEncoder
from thinker_wm.models.temporal_layout import (TimeGrid, Segment, ROLE_TARGET, ROLE_OBS,
                                   history_frame_positions, history_segment, token_timesteps)
from thinker_wm.models.temporal_attention import visibility, rope_apply_pos
from thinker_wm.models.temporal_training import recent_clip_latent_valid
from thinker_wm.wan.modules.model import rope_params
import test_packed_ar as P

SPEC = WindowSpec(control_hz=50, num_video_frames=16, sparse_jitter=0)
EVENT = EventSpec(enabled=True, gripper_indices=(0,), after_s=.08)
GRID = TimeGrid(4, 1, control_hz=50)


def state(row):
    return np.array([1.0 if row < 20 or row >= 60 else 0.0], dtype=np.float32)


def observe(row):
    return Observation(rgb=torch.full((3,8,8), (row % 255)/255),
                       state=torch.from_numpy(state(row)), step=row, time_s=row/50,
                       robot_name=None, source="measured")


class MemoryFixes(unittest.TestCase):
    def test_padding_leading_and_internal(self):
        px = torch.arange(5.).view(5,1,1,1).expand(5,3,2,2)
        valid = torch.tensor([False,True,False,True,True])
        out = pad_recent_pixels(px, valid)
        self.assertEqual(out[:,0,0,0].tolist(), [1,1,1,3,4])
        self.assertEqual(valid.tolist(), [False,True,False,True,True])
        self.assertEqual(recent_clip_latent_valid(valid[None]).tolist(), [[False,False]])
        self.assertEqual(float(pad_recent_pixels(px, torch.zeros(5,dtype=torch.bool)).sum()), 0)

    def test_runtime_leading_padding(self):
        hb = HistoryBuffer(SPEC, None)
        for row in range(17):
            obs = observe(row)
            obs.rgb.fill_(.5)
            hb.add(obs)
        h = hb.history_inputs(16)
        self.assertEqual(h["history_recent_valid"].tolist(), [[False,True,True,True,True]])
        self.assertTrue(torch.all(h["history_recent_rgb"] > .49))
        self.assertEqual(hb.metrics()["history_recent_valid_frames"], 4)

    def test_odd_pool_and_edge_positions(self):
        enc = HistoryEncoder(12, 1, pool=2)
        self.assertEqual(enc.grid(15,20), (8,10))
        patch = torch.nn.Conv3d(4,12,(1,2,2),stride=(1,2,2))
        z = torch.ones(1,4,2,30,40)
        times = torch.tensor([[-.4,-.08]])
        tok = enc(patch,z,times,torch.ones(1,2,dtype=torch.bool),kind=torch.zeros(1,2,dtype=torch.long))
        self.assertEqual(tok.shape, (1,160,12))
        seg = history_segment(times,torch.ones(1,2,dtype=torch.bool),GRID,8,10,2,(15,20))
        self.assertEqual(float(seg.pos[...,1].max()),14)
        self.assertEqual(float(seg.pos[...,2].max()),18.5)
        tok.sum().backward()
        self.assertIsNotNone(enc.kind_embed.weight.grad)

    def test_old_history_kind_checkpoint(self):
        old = HistoryEncoder(12,1,num_kinds=3).state_dict()
        new = HistoryEncoder(12,1)
        new.load_state_dict(old,strict=True)
        self.assertEqual(tuple(new.kind_embed.weight.shape),(5,12))

    def test_rope_physical_time_and_integer_equivalence(self):
        times = torch.tensor([[-.4,-.08]])
        p = history_frame_positions(times,GRID)
        torch.testing.assert_close(p, times.double()/.32)
        inserted = history_frame_positions(torch.tensor([[-.4,-.24,-.08]]),GRID)
        torch.testing.assert_close(p, inserted[:,[0,2]])
        freqs = torch.cat([rope_params(1024,4),rope_params(1024,4),rope_params(1024,4)],dim=1)
        x = torch.randn(1,3,2,12)
        pos = torch.tensor([[-4,0,0],[0,1,2],[2,2,1]])
        torch.testing.assert_close(rope_apply_pos(x,pos,freqs),rope_apply_pos(x,pos.double(),freqs))
        self.assertTrue(torch.isfinite(rope_apply_pos(x,pos.double()-2000.25,freqs)).all())

    def test_event_arrival_budget_and_causality(self):
        det = EventDetector(EVENT,50)
        for row in range(24): det.update(row,state(row))
        self.assertEqual(det.records, [])
        det.update(24,state(24))
        self.assertEqual(det.records, [(19,24)])
        w = select_window(SPEC,400,300,jitter=False)
        records = [(i,i+4) for i in range(10,290,10)]
        a = select_event_history(w,SPEC,EVENT,records)
        check_window(SPEC,a)
        self.assertLessEqual(int(((a.sparse_kind>=3)&a.sparse_valid).sum()),8)
        self.assertEqual(len(set(a.sparse_rows[a.sparse_valid])),int(a.sparse_valid.sum()))
        b = select_event_history(w,SPEC,EVENT,records+[(299,303),(400,404)])
        np.testing.assert_array_equal(a.sparse_rows,b.sparse_rows)

    def test_event_survives_recent_eviction_and_reset(self):
        hb=HistoryBuffer(SPEC,None,events=EVENT)
        for row in range(1001): hb.add(observe(row))
        self.assertNotIn(19,hb._rgb)
        self.assertIn(19,hb._event_rgb)
        h=hb.history_inputs(1000)
        self.assertEqual(hb.metrics()["history_event_valid_frames"],4)
        self.assertTrue((h["history_sparse_times"][h["history_sparse_kind"]>=3] < -18).all())
        hb.reset("new")
        self.assertFalse(hb._event_rgb)
        self.assertFalse(hb.detector.records)

    def test_attacher_matches_runtime_events_and_pixels(self):
        class Reader:
            def fps(self,ep):return 50
            def rgb(self,ep,rows,hw):return torch.stack([observe(int(r)).rgb for r in rows])
            def state(self,ep,rows):return torch.tensor(np.array([state(int(r)) for r in rows]))
        att=HistoryAttacher(SPEC,{"test":Reader()},seed=5,events=EVENT,strict_sources=True)
        d=120
        batch=dict(video_frames=torch.zeros(1,16,3,8,8),episode_ids=["ep"],
                   frame_indices=[list(range(d,d+65,4))],source_datasets=["test"])
        a=att.attach(batch,0)
        hb=HistoryBuffer(SPEC,None,events=EVENT)
        for row in range(d+1):hb.add(observe(row))
        b=hb.history_inputs(d)
        for key in b:
            torch.testing.assert_close(a[key],b[key])
        self.assertEqual(att.last_metrics,hb.metrics())
        batch["source_datasets"]=["wrong"]
        with self.assertRaises(ValueError):att.attach(batch,0)

    def test_resolved_sparse_duplicates_keep_event_pair(self):
        from dataclasses import replace
        hb=HistoryBuffer(SPEC,None,events=EVENT)
        for row in range(121):hb.add(observe(row))
        w=select_window(hb.spec,121,120,jitter=False)
        rows=np.zeros(SPEC.history_slots,dtype=np.int64)
        rows[:3]=[19,24,20]
        valid=np.zeros_like(rows,dtype=bool);valid[:3]=True
        kinds=np.ones_like(rows);kinds[:2]=[3,4]
        w=replace(w,sparse_rows=rows,sparse_valid=valid,sparse_kind=kinds)
        del hb._rgb[20]
        _,resolved,_,valid=hb._resolve(w,120)
        self.assertEqual(resolved[:3].tolist(),[19,24,19])
        self.assertEqual(valid[:3].tolist(),[True,True,False])

    def test_sparse_fallback_keeps_time_order(self):
        import thinker_wm.runtime.history_buffer as HBM
        from dataclasses import replace
        hb=HistoryBuffer(SPEC,None,events=EVENT)
        for row in range(121):hb.add(observe(row))
        rows=np.zeros(SPEC.history_slots,dtype=np.int64)
        rows[:3]=[19,24,26]
        valid=np.zeros_like(rows,dtype=bool);valid[:3]=True
        kinds=np.ones_like(rows);kinds[:2]=[3,4]
        crafted=lambda w,*a:replace(w,sparse_rows=rows,sparse_valid=valid,sparse_kind=kinds)
        for r in (24,25,26):del hb._rgb[r]
        real=HBM.select_event_history
        HBM.select_event_history=crafted
        try:w,_,sr,st,sv=hb.window(120)
        finally:HBM.select_event_history=real
        self.assertEqual(sr[:3].tolist(),[19,23,24])
        self.assertEqual(w.sparse_kind[:3].tolist(),[3,1,4])
        self.assertTrue((np.diff(st[sv])>0).all())

    def test_rejected_add_leaves_no_frame(self):
        hb=HistoryBuffer(SPEC,None,events=EVENT)
        hb.add(observe(0))
        bad=observe(1);bad.state[0]=float('nan')
        with self.assertRaises(ValueError):hb.add(bad)
        self.assertEqual(sorted(hb._rgb),[0])
        self.assertEqual(sorted(hb._times),[0])
        hb.add(observe(1))

    def test_attacher_events_reject_subsampled_video_clock(self):
        class Reader:
            def fps(self,ep):return 12.5
            def rgb(self,ep,rows,hw):return torch.zeros(len(rows),3,*hw)
            def state(self,ep,rows):return torch.ones(len(rows),1)
        att=HistoryAttacher(SPEC,{"test":Reader()},seed=5,events=EVENT,strict_sources=True)
        batch=dict(video_frames=torch.zeros(1,16,3,8,8),episode_ids=["ep"],
                   frame_indices=[list(range(40,57))],source_datasets=["test"])
        with self.assertRaisesRegex(ValueError,"history events"):att.attach(batch,0)

    def test_zero_sparse_group_skips_vae(self):
        from thinker_wm.models.ubt_wm import UBT_WM
        class Fake:
            device=torch.device('cpu')
            dtype=torch.float32
            config=type('Config',(),dict(history_enable=True))()
            def encode_video(self,x):
                self.seen.append(x.shape)
                return torch.zeros(x.shape[0],4,1+(x.shape[2]-1)//4,2,2)
        m=Fake();m.seen=[]
        raw=dict(history_sparse_rgb=torch.zeros(1,0,3,8,8),
                 history_recent_rgb=torch.ones(1,5,3,8,8),
                 history_recent_valid=torch.ones(1,5,dtype=torch.bool),
                 history_recent_times=torch.tensor([[-.4,-.32,-.24,-.16,-.08]]))
        out=UBT_WM._prepare_history(m,raw,None)
        self.assertEqual(len(m.seen),1)
        self.assertEqual(out['latent'].shape[2],2)

    def test_two_arm_events_not_dropped(self):
        det = EventDetector(EventSpec(enabled=True, gripper_indices=(0, 1), after_s=.08, cooldown_s=.16), 50)
        arms = {10: [0, 1], 16: [0, 0], 40: [1, 0], 42: [1, 1]}
        s = [1, 1]
        for row in range(60):
            s = arms.get(row, s)
            det.update(row, s)
        # Arm 1 closes inside arm 0's cooldown: its own pair. Arm 1 reopens
        # while arm 0's pair is pending: merged into that pair.
        self.assertEqual(det.records, [(9, 14), (15, 20), (39, 44)])
        self.assertEqual(det.last_trigger.tolist(), [.8, .84])

    def test_hysteresis_middle_band(self):
        det = EventDetector(EVENT,50)
        for row,v in enumerate([1,.5,.9,.5,1]):det.update(row,[v])
        for _ in range(10):det.update(det.last_row+1,[1])
        self.assertEqual(det.records,[])
        det.update(20,[.5]);det.update(21,[.1])
        for row in range(22,30):det.update(row,[.1])
        self.assertEqual(det.records,[(20,25)])

    def test_event_capacity_and_newest_pairs_first(self):
        det = EventDetector(EventSpec(enabled=True,gripper_indices=(0,),capacity=2,max_frames=4),50)
        for row in range(200):det.update(row,[0. if (row//40)%2 else 1.])
        self.assertEqual(det.records,[(119,124),(159,164)])
        spec2 = EventSpec(enabled=True,gripper_indices=(0,),max_frames=2)
        w = select_window(SPEC,400,300,jitter=False)
        a = select_event_history(w,SPEC,spec2,[(10,14),(50,54),(90,94)])
        ev = sorted(int(r) for r,k,v in zip(a.sparse_rows,a.sparse_kind,a.sparse_valid) if v and k>=3)
        self.assertEqual(ev,[90,94])

    def test_template_config_paths_and_eight_slots(self):
        import tempfile
        from dataclasses import replace
        from omegaconf import OmegaConf
        from thinker_wm.data_prepare.history_attach import build_history_attacher
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/'tests/fixtures/training_temporal.yaml')
        with tempfile.NamedTemporaryFile('w',suffix='.jsonl') as f:
            cfg.history.robotwin_catalog = f.name
            att = build_history_attacher(cfg)
        hb = HistoryBuffer.from_config(cfg,None)
        self.assertEqual(att.spec,hb.spec.__class__(**{**hb.spec.__dict__,'require_full_future':att.spec.require_full_future}))
        self.assertEqual((att.spec.control_hz,att.spec.history_slots,att.spec.include_episode_start),(50.,8,False))
        self.assertEqual(att.events,hb.events)
        self.assertEqual(att.events.gripper_indices,(6,13))
        # 8-slot layout: attacher and runtime still agree bit-for-bit.
        spec8 = replace(SPEC,include_episode_start=False)
        class Reader:
            def fps(self,ep):return 50
            def rgb(self,ep,rows,hw):return torch.stack([observe(int(r)).rgb for r in rows])
            def state(self,ep,rows):return torch.tensor(np.array([state(int(r)) for r in rows]))
        att=HistoryAttacher(spec8,{"test":Reader()},seed=5,events=EVENT,strict_sources=True)
        d=120
        a=att.attach(dict(video_frames=torch.zeros(1,16,3,8,8),episode_ids=["ep"],
                          frame_indices=[list(range(d,d+65,4))],source_datasets=["test"]),0)
        hb=HistoryBuffer(spec8,None,events=EVENT)
        for row in range(d+1):hb.add(observe(row))
        b=hb.history_inputs(d)
        self.assertEqual(a["history_sparse_valid"].shape[1],8)
        for key in b:torch.testing.assert_close(a[key],b[key])

    def test_event_after_uses_elapsed_time_not_row_gap(self):
        det = EventDetector(EVENT,50)
        det.update(0,[1],10.0)
        det.update(1,[0],10.02)
        det.update(20,[0],10.04)
        self.assertEqual(det.records, [])
        det.update(21,[0],10.10)
        self.assertEqual(det.records, [(0,21)])

    def test_runtime_real_offsets_and_off_grid_frames(self):
        hb = HistoryBuffer(SPEC,None,events=EVENT)
        for row in range(121):
            obs=observe(row)
            obs.time_s += 100
            if row==116:obs.time_s += .015
            hb.add(obs)
        h=hb.history_inputs(120)
        self.assertFalse(bool(h["history_recent_valid"][0,-1]))
        torch.testing.assert_close(h["history_recent_times"][0,-1],torch.tensor(-.065))
        self.assertTrue((h["history_sparse_times"][h["history_sparse_valid"]]<0).all())
        self.assertEqual(h["history_recent_valid"][0].tolist(), [True]*4+[False])

    def test_late_current_frame_keeps_recent_clip(self):
        hb = HistoryBuffer(SPEC,None,events=EVENT)
        for row in range(121):
            obs=observe(row)
            if row==120:obs.time_s += .015
            hb.add(obs)
        h=hb.history_inputs(120)
        self.assertTrue(bool(h["history_recent_valid"].all()))
        torch.testing.assert_close(h["history_recent_times"][0,-1],torch.tensor(-.095))

    def test_no_gripper_events_keeps_sparse_history(self):
        for recent in (5, 9, 13):
            from dataclasses import replace
            spec = replace(SPEC, recent_frames=recent)
            hb = HistoryBuffer(spec, None, events=EVENT)
            for row in range(501):
                obs = observe(row)
                obs.state.fill_(1)
                hb.add(obs)
            self.assertEqual(hb.detector.records, [])
            h = hb.history_inputs(500)
            self.assertEqual(int(h["history_recent_valid"].sum()), recent)
            self.assertGreater(int(h["history_sparse_valid"].sum()), 0)
            self.assertEqual(hb.metrics()["history_event_valid_frames"], 0)

    def test_images_without_usable_latents_metric(self):
        h = dict(history_recent_valid=torch.tensor([[False,True,False,False,False]]))
        m = history_metrics(h)
        self.assertEqual(m["history_empty_fraction"], 0)
        self.assertEqual(m["history_frames_without_latents_fraction"], 1)
        self.assertEqual(history_metrics(None)["history_frames_without_latents_fraction"], 0)

    def test_d0_mask_and_noisy_time(self):
        def seg(blocks,roles):return Segment(torch.tensor(blocks),torch.tensor(roles),torch.zeros(len(blocks),3,dtype=torch.long))
        q=seg([0,1,2],[ROLE_TARGET]*3)
        k=seg([0,1,1],[ROLE_TARGET,ROLE_TARGET,ROLE_OBS])
        self.assertEqual(visibility(q,k,target_reads_earlier_targets=False).tolist(),
                         [[True,False,False],[True,True,True],[False,False,True]])
        self.assertEqual(token_timesteps(torch.tensor([500.]),q).tolist(),[[500.,500.,500.]])

    def test_d0_forward_ignores_future_including_given_commands(self):
        clean,noisy=P.window(0),P.window(1)
        st=torch.randn(P.B,8)
        for role in ("policy","world"):
            r=P.train_req(action_role=role,**P.RGB_ONLY)
            base=P.run_packed(r,clean,noisy,st)[2][:,:,0]
            c={k:v.clone() for k,v in clean.items()}
            n={k:v.clone() for k,v in noisy.items()}
            for data in (c,n):
                data["video"][:,:,1:]+=10
                data["depth"][:,:,1:]-=10
                data["action"]+=10
            changed=P.run_packed(r,c,n,st)[2][:,:,0]
            torch.testing.assert_close(base,changed,atol=1e-5,rtol=0)

class ClosedSettle(unittest.TestCase):
    """Closing stalls at the object width; only a settled reading is closed."""
    SPEC = EventSpec(enabled=True, gripper_indices=(0,), closed_threshold=0.9, open_threshold=0.97,
                     after_s=0.08, cooldown_s=0.16, closed_settle_s=0.08, closed_settle_delta=0.01)

    @staticmethod
    def run_detector(spec, trace, hz=25):
        det = EventDetector(spec, hz)
        for row, g in enumerate(trace):
            det.update(row, np.array([g], dtype=np.float32))
        return det.records

    def test_settled_close_at_any_width(self):
        closing = [1.0] * 10 + [0.95, 0.85, 0.75, 0.68, 0.655] + [0.65] * 10 + [0.8, 0.95, 1.0] + [1.0] * 6
        self.assertEqual(self.run_detector(EVENT, closing), [])
        rec = self.run_detector(self.SPEC, closing)
        # Settled at row 16 (|0.65 - 0.655| <= 0.01 over 2 rows); before = last open row 9.
        self.assertEqual(rec[0], (9, 18))
        # Re-opening: before = last settled-closed row 24; open confirmed at row 27.
        self.assertEqual(rec[1], (24, 29))

    def test_slow_drift_is_not_closed(self):
        drift = [1.0] * 5 + [1.0 - 0.02 * k for k in range(1, 30)]
        self.assertEqual(self.run_detector(self.SPEC, drift), [])

    def test_default_spec_unchanged(self):
        trace = [state(r)[0] for r in range(100)]
        det = EventDetector(EVENT, 50)
        for row, g in enumerate(trace):
            det.update(row, np.array([g], dtype=np.float32))
        self.assertEqual(det.records, [(19, 24), (59, 64)])

    def test_runtime_buffer_keeps_before_frame(self):
        closing = [1.0] * 10 + [0.95, 0.85, 0.75, 0.68, 0.655] + [0.65] * 10
        hb = HistoryBuffer(replace(SPEC, control_hz=25), None, events=self.SPEC)
        for row, g in enumerate(closing):
            hb.add(Observation(rgb=torch.full((3, 8, 8), row / 255), state=torch.tensor([g]), step=row,
                               time_s=row / 25, robot_name=None, source="measured"))
        self.assertEqual(hb.detector.records, [(9, 18)])
        self.assertIn(9, hb._event_rgb)


if __name__=="__main__":unittest.main(verbosity=2)
