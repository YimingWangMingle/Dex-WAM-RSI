"""Real tiny WAM forward/backward plus verified-pair data/loss contracts (CPU)."""
from pathlib import Path
import json
import tempfile
import unittest
from dataclasses import replace

import _rsi_cpu_stubs
import _rsi_harness as H
import torch
from thinker_wm.models import capability_training as CT, temporal_training as TT
from thinker_wm.models.history_optimization import split_history_groups
from thinker_wm.data_prepare.capability_annotations import load_annotations, CapabilityAnnotations


def records(kind="language", failure=False):
    return [dict(episode_id=f"episode_{i}", start_frame=0, context_id="snapshot_and_history_1",
                 instruction_id=f"goal_{i}" if kind == "language" else "goal_0",
                 branch_id=f"branch_{i}", outcome_source="simulator_recording",
                 joint_valid=not failure, history_required=True) for i in range(2)]


def model_batch(kind="language", failure=False, checkpoint=False):
    cfg = H.config(enable_depth=True, depth_condition_mode="rgb_only", history_enable=True,
                   ar_packed_teacher_forcing=True, ar_target_reads_earlier_targets=False, num_video_frames=16,
                   training_mode_probs=dict(joint=.6, a2v=.4),
                   capability_language_weight=.1 if kind == "language" else 0.,
                   capability_history_weight=.1 if kind == "history" else 0.,
                   capability_a2v_weight=.1 if kind == "a2v" else 0.,
                   capability_min_sigma=0., capability_aux_interval=1,
                   gradient_checkpointing=checkpoint)
    model = H.Harness(cfg).train()
    batch = H.batch(depth=True, history_slots=2, frames=16)
    batch["first_frame"][1] = batch["first_frame"][0]
    batch["state"][1] = batch["state"][0]
    batch["depth_first"][1] = batch["depth_first"][0]
    if kind == "a2v":
        batch["vlm_input_messages"][1] = batch["vlm_input_messages"][0]
    for value in batch["history_inputs"].values():
        value[1] = value[0]
    context = model.prepare_training_context(**batch)
    context["capability_records"] = records(kind, failure)
    desired = TT.MODE_WORLD if kind == "a2v" else TT.MODE_JOINT
    decision = next(TT.sample_step_decision(cfg, s, True, False) for s in range(100)
                    if TT.sample_step_decision(cfg, s, True, False).mode == desired)
    return model, batch, context, decision


class CapabilityTraining(unittest.TestCase):
    def test_three_objectives_backward(self):
        for kind in ("language", "history", "a2v"):
            with self.subTest(kind=kind):
                model, batch, context, decision = model_batch(kind)
                torch.manual_seed(31)
                result = model.training_step(**batch, training_context=context,
                                             temporal_decision=decision, dropout_seed_step=0)
                self.assertTrue(torch.isfinite(result["total_loss"]))
                self.assertEqual(result[f"capability_{kind}_active_samples"], 2)
                self.assertGreaterEqual(result[f"capability_{kind}_loss"], 0)
                result["total_loss"].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertIsNotNone(model.world_model.history_encoder.logit_bias.grad)
                self.assertEqual(float(result["action_loss"]) == 0, kind == "a2v")

    def test_a2v_matched_noise_and_d0(self):
        model, batch, context, decision = model_batch("a2v")
        calls = []
        original = model.world_model.forward
        def capture(*args, **kw):
            out = original(*args, **kw)
            kw = dict(kw, t=args[0], video=args[3], action=args[4])
            calls.append((kw, out))
            return out
        model.world_model.forward = capture
        model.training_step(**batch, training_context=context, temporal_decision=decision)
        self.assertEqual(len(calls), 2)
        kw, out = calls[0]
        torch.testing.assert_close(kw["t_video"][0], kw["t_video"][1], rtol=0, atol=0)
        plan = TT.pack_plan(decision.request)
        sigma = kw["t_video"].reshape(2,1,1,1,1) / 1000
        noisy = TT.unpack_targets(kw["video"], plan.video, 5, 2)
        noise = (noisy - context["clean_full_latent"] * (1-sigma)) / sigma
        torch.testing.assert_close(noise[0,:,1:], noise[1,:,1:], rtol=1e-5, atol=1e-5)
        # Same observed context and same D0 noise: future commands may change
        # future depth, but cannot change current depth.
        d = TT.unpack_targets(out[2], plan.depth, 5, 2)
        torch.testing.assert_close(d[0,:,0], d[1,:,0], rtol=0, atol=1e-5)
        swapped_depth = TT.unpack_targets(calls[1][1][2], plan.depth, 5, 2)
        torch.testing.assert_close(d[:,:,0], swapped_depth[:,:,0], rtol=0, atol=1e-5)
        self.assertTrue(torch.equal(calls[0][0]["video"], calls[1][0]["video"]))

    def test_reference_no_gradient_and_rng_restored(self):
        model, batch, context, decision = model_batch("language")
        calls = []
        original = model.world_model.forward
        def capture(*args, **kw):
            captured = dict(kw, video=args[3], action=args[4])
            calls.append((torch.is_grad_enabled(), captured["video"].clone(), captured["action"].clone()))
            return original(*args, **kw)
        model.world_model.forward = capture
        torch.manual_seed(13)
        model.training_step(**batch, training_context=context, temporal_decision=decision)
        after = torch.get_rng_state()
        self.assertEqual([c[0] for c in calls], [True, False])
        self.assertTrue(torch.equal(calls[0][1], calls[1][1]))
        self.assertTrue(torch.equal(calls[0][2], calls[1][2]))
        model.config.capability_language_weight = 0.
        torch.manual_seed(13)
        model.training_step(**batch, training_context=context, temporal_decision=decision)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_reject_bad_pair_context(self):
        model, batch, context, decision = model_batch("language")
        context["state"][1,0] += 1
        with self.assertRaisesRegex(ValueError, "different state"):
            model.training_step(**batch, training_context=context, temporal_decision=decision)

    def test_failed_data_not_joint_targets(self):
        model, batch, context, decision = model_batch("language", failure=True)
        out = model.training_step(**batch, training_context=context, temporal_decision=decision)
        for key in ("action_loss", "video_loss", "depth_loss", "total_loss"):
            self.assertEqual(float(out[key]), 0., key)
        model, batch, context, decision = model_batch("a2v", failure=True)
        out = model.training_step(**batch, training_context=context, temporal_decision=decision)
        self.assertGreater(float(out["video_loss"]), 0)
        self.assertGreater(float(out["depth_loss"]), 0)

    def test_missing_history_and_prompt_dropout_skip_aux(self):
        model, batch, context, decision = model_batch("history")
        context["history"]["valid"].zero_()
        out = model.training_step(**batch, training_context=context, temporal_decision=decision)
        self.assertEqual(out["capability_history_active_samples"], 0)
        self.assertEqual(out["capability_history_loss"], 0)
        model.config.modality_dropout_prompt = 1.
        out = model.training_step(**batch, training_context=context, temporal_decision=decision)
        self.assertEqual(out["capability_history_loss"], 0)


class Contracts(unittest.TestCase):
    def test_ranking_does_not_degrade_reference(self):
        pos = torch.tensor([.5], requires_grad=True)
        ref = torch.tensor([.3], requires_grad=True)
        CT.ranking_loss(pos, ref, torch.tensor([True]), .01).backward()
        self.assertEqual(pos.grad.item(), 1)
        self.assertIsNone(ref.grad)

    def test_effect_loss_excludes_d0_invalid_depth(self):
        target = torch.randn(2,2,3,2,2)
        prediction = target.clone().requires_grad_()
        mask = torch.ones_like(target,dtype=torch.bool)
        mask[:,:,0] = False
        mask[:,:,:,0,0] = False
        shifted = prediction + (~mask).float() * 100
        value = CT.paired_effect_loss(shifted,target,mask,torch.tensor([1,0]),torch.tensor([True,True]))
        self.assertEqual(value.item(),0)
        value.backward()
        self.assertTrue(torch.equal(prediction.grad,torch.zeros_like(prediction)))

    def test_annotations_lookup_and_reject_prediction_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"windows.jsonl"
            rows = records()
            p.write_text("\n".join(json.dumps(r) for r in rows))
            a = CapabilityAnnotations([p])
            self.assertEqual(a.lookup_batch(dict(episode_ids=["episode_1"],frame_indices=[[0,4]])),[rows[1]])
            rows[0]["outcome_source"] = "model_prediction"
            p.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError): load_annotations(p)

    def test_history_optimizer_groups(self):
        m,_,_,_ = model_batch()
        params = list(m.parameters())
        groups = [dict(params=params,lr=1e-5,name="backbone")]
        h = m.world_model.history_encoder
        self.assertIs(split_history_groups(groups,h),groups)
        new = split_history_groups(groups,h,adapter_lr=3e-5,bias_lr=1e-3)
        ids = [id(p) for g in new for p in g["params"]]
        self.assertEqual(set(ids),{id(p) for p in params})
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(new[-1]["name"],"history_bias")
        self.assertEqual(new[-1]["weight_decay"],0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
