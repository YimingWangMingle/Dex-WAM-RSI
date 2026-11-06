"""Exact cached/uncached AR denoising, all three modes, CFG and bounded memory."""
from pathlib import Path
import sys
import unittest
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_cpu_stubs
import _rsi_harness as H
import torch
from thinker_wm.models import temporal_training as TT
from thinker_wm.models.prefix_cache import PrefixKVCache
from test_packed_ar import packed_config, hist


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.m = H.Harness(packed_config(num_video_frames=16, action_t_downsample=1,
             enable_depth=True, depth_condition_mode='rgb_only', history_enable=True)).eval()
        self.b = H.batch(frames=16)
        g = torch.Generator().manual_seed(317)
        self.kw = dict(first_frame=self.b['first_frame'], state=self.b['state'],
            vlm_input_messages=self.b['vlm_input_messages'], num_inference_steps=3,
            action_num_inference_steps=2, decode_outputs=False, num_blocks=4,
            history_context=hist(),
            init_video_noise=torch.randn(2,8,5,4,4,generator=g),
            init_depth_noise=torch.randn(2,8,5,4,4,generator=g),
            init_action_noise=torch.randn(2,64,8,generator=g))

    def run_model(self, mode='off', budget=64, **extra):
        cap={}
        with torch.no_grad():
            out=self.m.inference_step(**(self.kw | extra), prefix_cache_mode=mode,
                prefix_cache_max_mb=budget, latent_capture=cap)
        return out,cap

    def assert_same(self, a, b):
        torch.testing.assert_close(a[0][1],b[0][1],atol=2e-5,rtol=1e-5)
        for key in ('video_latent','action_latent','depth_latent'):
            torch.testing.assert_close(a[1][key],b[1][key],atol=2e-5,rtol=1e-5)

    def test_all_modes_four_blocks(self):
        req=TT.sample_step_decision(self.m.config,0,False,False).request
        for mode in ('joint','idm','fdm'):
            with self.subTest(mode=mode):
                kw={}
                if mode=='idm':
                    kw=dict(temporal_request=replace(req,video_role='goal'),
                            condition_video_frames=self.b['video_frames'])
                if mode=='fdm':
                    kw=dict(temporal_request=replace(req,action_role='world'),
                            condition_actions=self.b['actions'])
                off=self.run_model(**kw);on=self.run_model('fixed',**kw)
                self.assert_same(off,on)
                blocks=on[1]['prefix_cache']['blocks']
                self.assertEqual([x['block'] for x in blocks],[1,2,3,4])
                self.assertTrue(all(x['cond']['hits']>0 for x in blocks))
                if mode=='fdm':
                    torch.testing.assert_close(on[0][1],self.b['actions'])

    def test_cfg_and_fresh_call_after_observation_change(self):
        self.kw.update(num_blocks=2,guidance_scale=2.,action_guidance_scale=1.5)
        self.assert_same(self.run_model(),self.run_model('fixed'))
        on=self.run_model('fixed')[1]['prefix_cache']['blocks']
        self.assertTrue(all(x['cond']['hits']>0 and x['uncond']['hits']>0 for x in on))
        self.kw['first_frame']=self.kw['first_frame']+0.1
        self.kw['history_context']=hist(seed=93)
        self.kw['vlm_input_messages']=['different instruction']*2
        self.assert_same(self.run_model(),self.run_model('fixed'))

    def test_depth_clone_and_direct_depth_visibility(self):
        import copy
        uni=self.m.world_model.uni_branch
        uni.depth_uni_clone_layers=1
        uni.depth_uni_first_layer=len(uni.uni_blocks)-1
        uni.depth_uni_blocks=torch.nn.ModuleList([copy.deepcopy(uni.uni_blocks[-1])])
        self.m.config=replace(self.m.config,depth_read_by_video=True,
                             depth_read_by_action=True,depth_read_by_und=True)
        self.assert_same(self.run_model(num_blocks=2),self.run_model('fixed',num_blocks=2))

    def test_without_history_and_d0_stays_dynamic(self):
        self.kw['history_context']=None
        off=self.run_model(num_blocks=1)
        on=self.run_model('fixed',num_blocks=1)
        self.assert_same(off,on)
        self.kw['init_depth_noise']=self.kw['init_depth_noise']+0.3
        changed=self.run_model('fixed',num_blocks=1)
        self.assertFalse(torch.allclose(on[1]['depth_latent'][:,:,0],changed[1]['depth_latent'][:,:,0]))
        self.assert_same(self.run_model(num_blocks=1),changed)

    def test_zero_budget_recomputes_without_dropping_context(self):
        off=self.run_model(num_blocks=2)
        on=self.run_model('fixed',budget=0,num_blocks=2)
        self.assert_same(off,on)
        self.assertTrue(all(x['cond']['bytes']==0 and x['cond']['budget_skips']>0
                            for x in on[1]['prefix_cache']['blocks']))

    def test_branch_ffn_processes_fewer_rows(self):
        counts=[]
        module=self.m.world_model.video_branch.encode_blocks[0].ffn
        hook=module.register_forward_pre_hook(lambda m,args: counts.append(args[0].size(1)))
        try:
            self.run_model(num_blocks=2);off=sum(counts);counts.clear()
            self.run_model('fixed',num_blocks=2);on=sum(counts)
        finally:
            hook.remove()
        self.assertLess(on,off)

    def test_reject_invalid_usage(self):
        for budget in (-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError): self.run_model('fixed',budget=budget)
        with self.assertRaises(ValueError): self.run_model('typo')
        self.m.train()
        with self.assertRaises(ValueError): self.run_model('fixed')

    def test_public_api_exposes_cache_stats(self):
        from thinker_wm.models.inference_api import WAMInference, GenerationSettings, Observation
        b=H.batch(b=1,frames=16)
        api=WAMInference(self.m,checkpoint='test',action_stats_version='test',camera_transform='test',
                         settings=GenerationSettings(num_inference_steps=2,prefix_cache_mode='fixed'))
        obs=Observation(b['first_frame'][0],b['state'][0],0,0.,None)
        joint=api.generate_action(obs,None,'pick')
        idm=api.generate_action_for_goal(obs,None,'pick',b['video_frames'][0,:4])
        fdm=api.predict_outcome(obs,None,'pick',joint.actions)
        for stats in (joint.forecast.cache_stats,idm.cache_stats,fdm.cache_stats):
            self.assertEqual(stats['mode'],'fixed')
            self.assertGreater(stats['blocks'][0]['cond']['hits'],0)

    def test_budget_is_bounded(self):
        c=PrefixKVCache(16)
        x=torch.arange(8.).reshape(1,4,2)
        fixed=torch.tensor([True,False,True,False])
        c.rows('a',x,fixed,lambda y:y*2)
        c.rows('b',x,fixed,lambda y:y*2)
        self.assertEqual(c.bytes,16)
        self.assertEqual(c.budget_skips,1)
        y=c.rows('a',x+torch.tensor([[[0.],[1.],[0.],[1.]]]),fixed,lambda y:y*2)
        torch.testing.assert_close(y,2*(x+torch.tensor([[[0.],[1.],[0.],[1.]]])))

if __name__=='__main__': unittest.main(verbosity=2)
