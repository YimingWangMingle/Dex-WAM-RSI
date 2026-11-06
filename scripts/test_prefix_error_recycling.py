"""CPU contracts for RGB-prefix error replay in Joint and A2V."""
from pathlib import Path
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_cpu_stubs
import _rsi_harness as H
import torch
from thinker_wm.models import temporal_training as TT
from thinker_wm.models.temporal_layout import ROLE_CLEAN, ROLE_TARGET
from thinker_wm.models.prefix_error_recycling import RGBPrefixErrorBuffer
from test_packed_ar import packed_config


def config(**kw):
    base = dict(num_video_frames=16, action_t_downsample=1, control_hz=50.,
                enable_depth=True, depth_condition_mode='rgb_only', history_enable=True,
                training_mode_probs=dict(joint=.6,a2v=.4), rgb_prefix_error_enable=True,
                rgb_prefix_error_prob=1., rgb_prefix_error_warmup_steps=0)
    base.update(kw)
    return packed_config(**base)


class BufferContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.req = TT.sample_step_decision(self.cfg,0,False,False).request
        self.entries = TT.pack_plan(self.req).video
        self.clean = torch.ones(1,2,5,2,2)
        self.noisy = self.clean * 2
        self.velocity = torch.ones_like(self.clean)*4
        self.sigma = torch.full((1,1,1,1,1), .5)
        self.groups = [('robotwin','aloha')]

    def observe(self, buf, mode=TT.MODE_JOINT, valid=True):
        buf.observe(self.noisy,self.velocity,self.clean,self.sigma,self.entries,
                    mode,self.groups,[valid])

    def test_error_formula_and_prefix_only(self):
        buf = RGBPrefixErrorBuffer(self.cfg)
        self.observe(buf)
        self.assertEqual(len(buf.items),3)
        for _,err in buf.items:
            torch.testing.assert_close(err.rgb,torch.full_like(err.rgb,-1))
            self.assertFalse(err.rgb.requires_grad)
            self.assertEqual(err.rgb.device.type,'cpu')
        packed = TT.pack(self.clean,self.noisy,self.entries,2)
        original = packed.clone()
        changed,stats = buf.perturb(packed,self.entries,TT.MODE_JOINT,5,self.groups,[True])
        self.assertEqual(stats['rgb_prefix_error_applied_blocks'],3)
        self.assertAlmostEqual(stats['rgb_prefix_error_mean_rms_ratio'],.1,places=5)
        self.assertAlmostEqual(stats['rgb_prefix_error_raw_rms_ratio'],.5,places=5)
        self.assertEqual(stats['rgb_prefix_error_clipped_frac'],1.)
        for i,(_,_,role) in enumerate(self.entries):
            if role==ROLE_CLEAN:
                torch.testing.assert_close(changed[:,:,i],original[:,:,i]-.1)
            else:
                torch.testing.assert_close(changed[:,:,i],original[:,:,i])
        torch.testing.assert_close(packed,original)

    def test_each_block_replays_its_own_error(self):
        buf = RGBPrefixErrorBuffer(self.cfg)
        clean = torch.full_like(self.clean, 100.)
        velocity = torch.zeros_like(clean)
        for t in range(clean.shape[2]):
            velocity[:,:,t] = -2.*t   # x0_hat - x0 = -sigma*v = t at latent t
        buf.observe(clean,velocity,clean,self.sigma,self.entries,TT.MODE_JOINT,self.groups,[True])
        packed = TT.pack(clean,clean,self.entries,2)
        changed,stats = buf.perturb(packed,self.entries,TT.MODE_JOINT,5,self.groups,[True])
        self.assertEqual(stats['rgb_prefix_error_clipped_frac'],0.)
        for i,(src,block,role) in enumerate(self.entries):
            want = 100. + (.5*src if role==ROLE_CLEAN and block>0 else 0.)
            torch.testing.assert_close(changed[:,:,i],torch.full_like(changed[:,:,i],want))

    def test_true_flow_velocity_gives_zero_error(self):
        g = torch.Generator().manual_seed(0)
        noise = torch.randn(self.clean.shape,generator=g)
        clean = torch.randn(self.clean.shape,generator=g)
        sigma = torch.full((1,1,1,1,1),.7)
        noisy = (1-sigma)*clean + sigma*noise
        buf = RGBPrefixErrorBuffer(self.cfg)
        buf.observe(noisy,noise-clean,clean,sigma,self.entries,TT.MODE_JOINT,self.groups,[True])
        self.assertEqual(len(buf.items),3)
        for _,err in buf.items:
            self.assertLess(float(err.rgb.abs().max()),1e-5)

    def test_mode_source_warmup_and_probability_isolation(self):
        buf = RGBPrefixErrorBuffer(config(rgb_prefix_error_warmup_steps=2))
        packed = TT.pack(self.clean,self.noisy,self.entries,2)
        self.observe(buf)
        self.assertIs(buf.perturb(packed,self.entries,TT.MODE_JOINT,5,self.groups,[True])[0],packed)
        self.observe(buf)
        self.assertGreater(buf.perturb(packed,self.entries,TT.MODE_JOINT,5,self.groups,[True])[1]['rgb_prefix_error_applied_blocks'],0)
        self.assertIs(buf.perturb(packed,self.entries,TT.MODE_WORLD,5,self.groups,[True])[0],packed)
        self.assertIs(buf.perturb(packed,self.entries,TT.MODE_JOINT,5,[('other','aloha')],[True])[0],packed)
        buf.prob=0.
        self.assertIs(buf.perturb(packed,self.entries,TT.MODE_JOINT,5,self.groups,[True])[0],packed)

    def test_invalid_samples_and_memory_budget(self):
        buf = RGBPrefixErrorBuffer(config(rgb_prefix_error_capacity=2,rgb_prefix_error_max_mb=.0001))
        self.observe(buf,valid=False)
        self.assertEqual(len(buf.items),0)
        self.velocity.fill_(float('nan'))
        self.observe(buf)
        self.assertEqual(len(buf.items),0)
        self.velocity.fill_(4)
        for _ in range(20): self.observe(buf)
        self.assertLessEqual(buf.nbytes,buf.max_bytes)
        self.assertGreater(len(buf.items),0)

    def test_capacity_is_per_key(self):
        from collections import Counter
        buf = RGBPrefixErrorBuffer(config(rgb_prefix_error_capacity=2))
        for _ in range(3): self.observe(buf,mode=TT.MODE_WORLD)
        for _ in range(20): self.observe(buf)
        counts = Counter(k[:2] for k,_ in buf.items)
        self.assertEqual(set(counts.values()),{2})
        self.assertEqual(sum(1 for k in counts if k[0]==TT.MODE_WORLD),3)

    def test_yaml_and_configuration_validation(self):
        from omegaconf import OmegaConf
        cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'tests/fixtures/training_temporal.yaml')
        values=TT.motus_temporal_kwargs(cfg)
        self.assertTrue(values['rgb_prefix_error_enable'])
        self.assertEqual(values['rgb_prefix_error_prob'],.25)
        TT.validate_temporal_config(config(**{k:v for k,v in values.items() if k.startswith('rgb_prefix_error_')}))
        for kw in (dict(rgb_prefix_error_prob=1.1),dict(rgb_prefix_error_scale=float('nan')),
                   dict(rgb_prefix_error_capacity=0),dict(rgb_prefix_error_warmup_steps=-1),
                   dict(rgb_prefix_error_max_mb=0),dict(ar_packed_teacher_forcing=False)):
            with self.assertRaises(ValueError): TT.validate_temporal_config(replace(config(),**kw))


class RealTrainingContracts(unittest.TestCase):
    def test_both_modes_replay_without_mutating_targets_or_d0(self):
        for mode in (TT.MODE_JOINT,TT.MODE_WORLD):
            with self.subTest(mode=mode):
                m=H.Harness(config()).train()
                b=H.batch(depth=True,frames=16,history_slots=3)
                ctx=m.prepare_training_context(**b)
                clean_original=ctx['clean_full_latent'].clone()
                depth_original=ctx['clean_depth_latent'].clone()
                r=TT.sample_step_decision(m.config,0,False,False).request
                r=replace(r,action_role='world' if mode==TT.MODE_WORLD else 'policy')
                d=TT.StepDecision(r,mode)
                calls=[]
                real=m.world_model.forward
                def spy(*args,**kw):
                    out=real(*args,**kw)
                    calls.append((args[3].detach().clone(),args[4].detach().clone(),
                                  kw['depth'].detach().clone(),out))
                    return out
                m.world_model.forward=spy
                def run(seed,step):
                    torch.manual_seed(seed)
                    return m.training_step(**b,training_context=ctx,temporal_decision=d,dropout_seed_step=step)
                first=run(73,0)
                self.assertEqual(first['rgb_prefix_error_applied_blocks'],0)
                second=run(73,1)
                self.assertEqual(second['rgb_prefix_error_applied_blocks'],6) # B=2, 3 prefix blocks
                self.assertLessEqual(second['rgb_prefix_error_max_rms_ratio'],.10001)
                tag='joint' if mode==TT.MODE_JOINT else 'a2v'
                self.assertNotIn(f'prefix_error_{tag}/clipped_frac',first)
                self.assertEqual(second[f'prefix_error_{tag}/mean_rms_ratio'],
                                 second['rgb_prefix_error_mean_rms_ratio'])
                entries=TT.pack_plan(r).video
                for i,(_,_,role) in enumerate(entries):
                    if role==ROLE_CLEAN:
                        self.assertFalse(torch.equal(calls[0][0][:,:,i],calls[1][0][:,:,i]))
                    else:
                        torch.testing.assert_close(calls[0][0][:,:,i],calls[1][0][:,:,i])
                torch.testing.assert_close(calls[0][1],calls[1][1])
                torch.testing.assert_close(calls[0][2],calls[1][2])
                torch.testing.assert_close(ctx['clean_full_latent'],clean_original)
                torch.testing.assert_close(ctx['clean_depth_latent'],depth_original)
                # D0 and first target cannot read any perturbed CLEAN block.
                torch.testing.assert_close(calls[0][3][2][:,:,0],calls[1][3][2][:,:,0])
                first_target=next(i for i,(s,_,role) in enumerate(entries) if s==1 and role==ROLE_TARGET)
                torch.testing.assert_close(calls[0][3][0][:,:,first_target],calls[1][3][0][:,:,first_target])
                self.assertEqual(float(second['action_loss'])==0,mode==TT.MODE_WORLD)
                second['total_loss'].backward()
                grads=[p.grad for p in m.world_model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                self.assertIsNotNone(m.world_model.history_encoder.logit_bias.grad)
                torch.optim.AdamW(m.world_model.parameters(),lr=1e-5).step()
                self.assertTrue(all(torch.isfinite(p).all() for p in m.world_model.parameters()))

    def test_eval_and_disabled_are_unchanged_and_restart_rewarms(self):
        m=H.Harness(config()).train()
        b=H.batch(depth=True,frames=16)
        m.training_step(**b,dropout_seed_step=0)
        count=m._rgb_prefix_error_buffer.nbytes
        seen=dict(m._rgb_prefix_error_buffer.seen)
        m.eval()
        a=m.training_step(**b,dropout_seed_step=1)
        self.assertEqual(a['rgb_prefix_error_active'],0)
        self.assertEqual(m._rgb_prefix_error_buffer.nbytes,count)
        self.assertEqual(m._rgb_prefix_error_buffer.seen,seen)
        # Buffer is not silently serialized into the checkpoint or EMA weights.
        state=m.state_dict()
        self.assertFalse(any('prefix_error' in k for k in state))
        fresh=H.Harness(config()).train()
        fresh.load_state_dict(state)
        out=fresh.training_step(**b,dropout_seed_step=5)
        self.assertEqual(out['rgb_prefix_error_applied_blocks'],0)
        # Inference ignores the feature and existing training buffer.
        kw=dict(first_frame=b['first_frame'],state=b['state'],vlm_input_messages=b['vlm_input_messages'],
                num_inference_steps=2,decode_outputs=False,num_blocks=2)
        def infer():
            cap={}
            torch.manual_seed(93)
            with torch.no_grad(): _,actions=m.inference_step(**kw,latent_capture=cap)
            return actions,cap
        a,ca=infer()
        m.config=replace(m.config,rgb_prefix_error_enable=False)
        aa,cb=infer()
        torch.testing.assert_close(a,aa)
        for k in ('video_latent','depth_latent'):
            torch.testing.assert_close(ca[k],cb[k])
        m.train()
        self.assertEqual(m.training_step(**b,dropout_seed_step=8)['rgb_prefix_error_active'],0)


class JointActionReplay(unittest.TestCase):
    def cfg(self,**kw):
        values=dict(joint_action_prefix_error_enable=True,
                    joint_action_prefix_error_dims=[0,1,2,3,4,5])
        values.update(kw)
        return config(**values)

    def test_joint_rgb_action_and_a2v_rgb_only(self):
        for mode in (TT.MODE_JOINT,TT.MODE_WORLD):
            with self.subTest(mode=mode):
                m=H.Harness(self.cfg()).train()
                b=H.batch(depth=True,frames=16,history_slots=3)
                ctx=m.prepare_training_context(**b)
                actual_actions=ctx['actions'].clone()
                # Exercise both dimension-wide and individual-step validity.
                ctx['action_dim_mask']=torch.ones(2,8,dtype=torch.bool)
                ctx['action_dim_mask'][:,2]=False
                ctx['action_validity_mask']=torch.ones(2,64,8,dtype=torch.bool)
                ctx['action_validity_mask'][:,4,0]=False
                r=TT.sample_step_decision(m.config,0,False,False).request
                r=replace(r,action_role='world' if mode==TT.MODE_WORLD else 'policy')
                d=TT.StepDecision(r,mode)
                calls=[]
                real=m.world_model.forward
                def spy(*args,**kw):
                    out=real(*args,**kw)
                    calls.append((args[3].detach().clone(),args[4].detach().clone(),
                                  kw['depth'].detach().clone(),out))
                    return out
                m.world_model.forward=spy
                def run(step,weights=None):
                    torch.manual_seed(73)
                    return m.training_step(**b,training_context=ctx,temporal_decision=d,
                                           dropout_seed_step=step,action_loss_weights=weights)
                out=run(0)
                self.assertEqual(out['joint_action_prefix_error_applied_blocks'],0)
                out=run(1)
                self.assertEqual(out['rgb_prefix_error_applied_blocks'],6)
                self.assertEqual(out['joint_action_prefix_error_applied_blocks'],6 if mode==TT.MODE_JOINT else 0)
                self.assertLessEqual(out['joint_action_prefix_error_max_abs'],.020001)
                plan=TT.pack_plan(r)
                for i,(src,_,role) in enumerate(plan.action):
                    if mode==TT.MODE_JOINT and role==ROLE_CLEAN:
                        # At least one eligible coordinate changes per step.
                        self.assertFalse(torch.equal(calls[0][1][:,i,[1,3,4,5]],calls[1][1][:,i,[1,3,4,5]]))
                        torch.testing.assert_close(calls[0][1][:,i,[2,6,7]],calls[1][1][:,i,[2,6,7]])
                        if src==4:
                            torch.testing.assert_close(calls[0][1][:,i,0],calls[1][1][:,i,0])
                    else:
                        torch.testing.assert_close(calls[0][1][:,i],calls[1][1][:,i])
                for i,(_,_,role) in enumerate(plan.video):
                    if role!=ROLE_CLEAN:
                        torch.testing.assert_close(calls[0][0][:,:,i],calls[1][0][:,:,i])
                torch.testing.assert_close(calls[0][2],calls[1][2])
                torch.testing.assert_close(ctx['actions'],actual_actions)
                torch.testing.assert_close(calls[0][3][2][:,:,0],calls[1][3][2][:,:,0])
                target_idx=[i for i,(src,_,role) in enumerate(plan.action) if src<16 and role==ROLE_TARGET]
                if target_idx:
                    torch.testing.assert_close(calls[0][3][1][:,target_idx],calls[1][3][1][:,target_idx])
                out['total_loss'].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                torch.optim.AdamW(m.world_model.parameters(),lr=1e-5).step()
                self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
                if mode==TT.MODE_WORLD:
                    self.assertTrue(all(error.action is None for _,error in m._rgb_prefix_error_buffer.items))
                else:
                    self.assertTrue(all(error.action is not None for _,error in m._rgb_prefix_error_buffer.items))
                    # A sample whose action supervision is disabled cannot replay action errors.
                    run(2,weights=torch.tensor([0.,1.]))
                    for i,(_,_,role) in enumerate(plan.action):
                        if role==ROLE_CLEAN:
                            torch.testing.assert_close(calls[-1][1][0,i],calls[0][1][0,i])

    def test_paired_residual_source_and_joint_budget(self):
        cfg=self.cfg(rgb_prefix_error_scale=1.,rgb_prefix_error_max_rms_ratio=1.,
                     joint_action_prefix_error_scale=1.,joint_action_prefix_error_max_abs=1.)
        buf=RGBPrefixErrorBuffer(cfg)
        r=TT.sample_step_decision(cfg,0,False,False).request
        plan=TT.pack_plan(r)
        clean=torch.ones(1,2,5,2,2)
        actions=torch.ones(1,64,8)
        valid=torch.ones_like(actions,dtype=torch.bool)
        sigma=torch.full((1,1,1,1,1),.5)
        # Two paired records with different signs. A chosen pair must retain
        # the RGB/action sign relationship, not draw from independent queues.
        for err in (.1,-.3):
            buf.observe(clean,torch.full_like(clean,-2*err),clean,sigma,plan.video,
                        TT.MODE_JOINT,['x'],[True],action_noisy=actions,
                        action_velocity=torch.full_like(actions,-4*err),action_clean=actions,
                        action_sigma=torch.full((1,1,1),.5),action_entries=plan.action,action_valid=valid)
        v=TT.pack(clean,clean,plan.video,2);a=TT.pack(actions,actions,plan.action,1)
        vv,aa,stats=buf.perturb_paired(v,plan.video,TT.MODE_JOINT,7,['x'],[True],
                                     action_packed=a,action_entries=plan.action,action_valid=valid)
        self.assertEqual(stats['joint_action_prefix_error_applied_blocks'],3)
        for block in (1,2,3):
            vi=next(i for i,(_,b,role) in enumerate(plan.video) if b==block and role==ROLE_CLEAN)
            ai=next(i for i,(_,b,role) in enumerate(plan.action) if b==block and role==ROLE_CLEAN)
            torch.testing.assert_close(aa[0,ai,0]-1,2*(vv[0,0,vi,0,0]-1))
        self.assertEqual(buf.nbytes,sum(e.nbytes for _,e in buf.items))
        for _,e in buf.items:
            self.assertFalse(e.action.requires_grad)
            self.assertEqual(e.action.device.type,'cpu')
        buf.max_bytes=100
        buf.observe(clean,clean,clean,sigma,plan.video,TT.MODE_JOINT,['x'],[True])
        self.assertLessEqual(buf.nbytes,buf.max_bytes)

    def test_allowlist_validation_and_template(self):
        from omegaconf import OmegaConf
        c=OmegaConf.load(Path(__file__).resolve().parents[1]/'tests/fixtures/training_temporal.yaml')
        kwargs=TT.motus_temporal_kwargs(c)
        self.assertTrue(kwargs['joint_action_prefix_error_enable'])
        self.assertNotIn(6,kwargs['joint_action_prefix_error_dims'])
        self.assertNotIn(13,kwargs['joint_action_prefix_error_dims'])
        # Real RoboTwin configuration has 14 unified dimensions (tiny test uses 8).
        TT.validate_temporal_config(replace(config(),unified_action_dim=14,**kwargs))
        for kw in (dict(joint_action_prefix_error_dims=None),dict(joint_action_prefix_error_dims=[8]),
                   dict(joint_action_prefix_error_dims=[1,1]),dict(joint_action_prefix_error_dims=[1.5]),
                   dict(joint_action_prefix_error_max_abs=0),dict(rgb_prefix_error_enable=False),
                   dict(action_target_repr='relative_eef_rot6'),dict(action_norm_policy='gripper_hand_minmax')):
            with self.assertRaises(ValueError): TT.validate_temporal_config(replace(self.cfg(),**kw))


if __name__=='__main__':
    unittest.main(verbosity=2)
