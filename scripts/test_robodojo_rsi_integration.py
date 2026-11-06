"""RoboDojo training/deployment contracts with native-shaped CPU observations.

No trained checkpoint, GPU, LLM or simulator success is substituted by these tests.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from thinker_wm.data_prepare.robodojo_core import FieldMap
from thinker_wm.models.action_stats import ACTION_STATS
from thinker_wm.models.inference_api import Forecast, action_hash
from thinker_wm.runtime.history_buffer import HistoryBuffer
from thinker_wm.runtime.robodojo_adapter import ObservationBinding, RoboDojoObservationAdapter
from environments.robodojo.runtime.robodojo_execution import NativeStep, RoboDojoExecutor, RoboDojoObservationStore
from environments.robodojo.training import MARKER, RoboDojoProfile, inspect_checkpoint, load_training, seal_checkpoint, verify_bundle
from skills.wam.thinker import ThinkerWAMBackend
from skills.contracts import ControlSkill, ExecutionFeedback, SkillIdentity, SkillRegistry, SkillRequest
from agents.context import AgentContext, TaskState
from execution.runner import CallSkill, TaskSkillRunner, Stop
from execution.trajectory import TrajectoryLogger, read_trajectory
from data.recovery import segments_from_trajectory
import test_task_skill_runner as runner_fixtures

CONFIG = ROOT / 'third_party/Thinker-WM/configs/rsi_robodojo_v1.yaml'
FIELD = ROOT / 'third_party/Thinker-WM/configs/robodojo/arx_x5_joint14_v1.field_map.yaml'


def live(row: int = 0) -> dict:
    return dict(head=np.full((32, 48, 3), row, dtype=np.uint8),
                left=np.full((24, 24, 3), 100, dtype=np.uint8),
                right=np.full((24, 24, 3), 200, dtype=np.uint8),
                left_j=np.full(6, row / 100, dtype=np.float32), left_g=[0.5],
                right_j=np.full(6, -row / 100, dtype=np.float32), right_g=[0.5],
                success='must not enter the model or Agent')


class InferenceDouble:
    num_frames, ratio = 4, 2

    def __init__(self):
        self.calls = []

    def generate_action(self, obs, history, instruction):
        self.calls.append((obs.step, history.last_step, history.episode_id, instruction))
        actions = np.full((8, 14), .1, dtype=np.float32)
        actions[:, [6, 13]] = .5
        forecast = Forecast('joint', obs.step, obs.time_s, actions, action_hash(actions),
                            'stats-fixture', 'camera-fixture', 'checkpoint-fixture', {},
                            [obs.step + i for i in (2, 4, 6, 8)], [obs.step + 8])
        return SimpleNamespace(actions=actions, forecast=forecast)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.fm, self.profile, _ = load_training(CONFIG, FIELD)

    def store(self, episode='ep'):
        binding = ObservationBinding(dict(head='head', left_wrist='left', right_wrist='right'),
            state_keys=('left_j', 'left_g', 'right_j', 'right_g'))
        adapter = RoboDojoObservationAdapter(self.fm, binding, model_hw=self.profile.model_hw,
            resize_mode='pad', robot_name=self.profile.robot)
        return RoboDojoObservationStore(self.root, episode, adapter, self.profile)

    def backend(self, store):
        return ThinkerWAMBackend(InferenceDouble(), HistoryBuffer.from_config(self.config, self.profile.robot),
                                store, self.profile)

    def checkpoint(self):
        ckpt = self.root / 'checkpoint_step_1000'; ckpt.mkdir()
        (ckpt / MARKER).write_text(json.dumps(dict(status='complete', step=1000)))
        (ckpt / 'pytorch_model.bin').write_bytes(b'contract fixture, not actual model weights')
        (ckpt / 'config.json').write_text(json.dumps(dict(common=self.config['common'])))
        return ckpt

    def test_profile_uses_trained_time_units_and_statistics(self):
        p = self.profile
        self.assertEqual((p.control_hz, p.action_dim, p.actions_per_block, p.training_actions), (25., 14, 8, 24))
        self.assertEqual(p.actions_per_block / p.control_hz, .32)
        different = copy.deepcopy(ACTION_STATS)
        different[p.robot]['mean'][0] += 1
        self.assertNotEqual(p.action_stats_version,
            RoboDojoProfile.from_config(self.config, self.fm, different).action_stats_version)

    def test_joint24_profile_is_one_24_step_block(self):
        config, _, p, _ = load_training(ROOT / 'third_party/Thinker-WM/configs/robodojo_joint24_v1.yaml', FIELD)
        self.assertEqual((p.frames_per_block, p.actions_per_block, p.training_actions), (12, 24, 24))
        self.assertAlmostEqual(p.actions_per_block / p.control_hz, .96)
        self.assertEqual(p.action_contract, self.profile.action_contract)
        self.assertEqual(p.action_stats_version, self.profile.action_stats_version)
        for latents in (2, 4):
            cfg = copy.deepcopy(config); cfg['model']['wm']['ar_latents_per_block'] = latents
            with self.assertRaises(ValueError):
                RoboDojoProfile.from_config(cfg, self.fm, ACTION_STATS)
        cfg = copy.deepcopy(config); cfg['model']['wm']['rgb_prefix_error_enable'] = True
        with self.assertRaises(ValueError):
            RoboDojoProfile.from_config(cfg, self.fm, ACTION_STATS)
        double = InferenceDouble()
        with self.assertRaises(ValueError):  # an 8-row AR inference cannot serve the Joint24 profile
            ThinkerWAMBackend(double, HistoryBuffer.from_config(config, p.robot), None, p)
        double.num_frames = 12
        ThinkerWAMBackend(double, HistoryBuffer.from_config(config, p.robot), None, p)

    def test_wrong_frequency_order_or_representation_is_rejected(self):
        for section, key, value in [('common', 'control_hz', 50), ('common', 'action_dim', 7)]:
            cfg = copy.deepcopy(self.config); cfg[section][key] = value
            with self.assertRaises(ValueError):
                RoboDojoProfile.from_config(cfg, self.fm, ACTION_STATS)
        with self.assertRaises(ValueError):
            RoboDojoProfile.from_config(self.config, replace(self.fm, action_names=self.fm.action_names[::-1]), ACTION_STATS)
        cfg = copy.deepcopy(self.config); cfg['model']['wm']['action_target_repr'] = 'relative_eef_rot6'
        with self.assertRaises(ValueError):
            RoboDojoProfile.from_config(cfg, self.fm, ACTION_STATS)

    def test_incomplete_checkpoint_and_missing_weights_are_rejected(self):
        ckpt = self.root / 'partial'; ckpt.mkdir()
        with self.assertRaises(ValueError): inspect_checkpoint(ckpt)
        (ckpt / MARKER).write_text('{"status":"complete","step":1000}')
        with self.assertRaises(ValueError): inspect_checkpoint(ckpt)

    def test_seal_preserves_full_config_and_detects_mutation(self):
        ckpt = self.checkpoint(); path = self.root / 'candidate.json'
        doc = seal_checkpoint(CONFIG, ckpt, path, FIELD)
        self.assertEqual(verify_bundle(path)['bundle_id'], doc['bundle_id'])
        for source in ('thinker_wm.py', 'third_party/Thinker-WM/models/inference_api.py',
                       'third_party/Thinker-WM/utils/output_paths.py',
                       'third_party/Thinker-WM/data_prepare/robodojo_core.py'):
            self.assertIn(source, doc['code_sha256'])
        self.assertEqual(doc['config']['history'], self.config['history'])
        self.assertEqual(doc['config']['model']['wm']['depth_condition_mode'], 'rgb_only')
        with self.assertRaises(FileExistsError): seal_checkpoint(CONFIG, ckpt, path, FIELD)
        (ckpt / 'pytorch_model.bin').write_bytes(b'modified')
        with self.assertRaises(ValueError): verify_bundle(path)

    def test_checkpoint_common_config_must_match(self):
        ckpt = self.checkpoint(); (ckpt / 'config.json').write_text('{"common":{}}')
        with self.assertRaises(ValueError): seal_checkpoint(CONFIG, ckpt, self.root / 'no.json', FIELD)

    def test_explicit_state_fields_and_native_layout(self):
        store = self.store(); ref = store.add(live(1), 0); obs = store.load(ref)
        self.assertEqual(tuple(obs.rgb.shape), (3, 480, 640))
        self.assertEqual(tuple(obs.state.shape), (14,))
        self.assertAlmostEqual(float(obs.state[6]), .5)
        self.assertAlmostEqual(float(obs.state[7]), -.01, places=5)
        self.assertFalse(hasattr(obs, 'success'))
        with self.assertRaises(ValueError):
            ObservationBinding({'head': 'head'}, state_key='all', state_keys=('one',))
        bad = live(); bad['left_j'][0] = np.nan
        with self.assertRaises(ValueError): store.add(bad, 1)

    def test_observation_identity_is_immutable(self):
        store = self.store(); ref = store.add(live(), 0)
        with self.assertRaises(ValueError): store.load(replace(ref, episode_id='foreign'))
        Path(ref.payload_ref).write_bytes(b'modified')
        with self.assertRaises(ValueError): store.load(ref)

    def test_native_verdict_is_separate_and_termination_stops_prefix(self):
        store = self.store(); calls = []
        def step(row):
            calls.append(row)
            return NativeStep(live(len(calls)), len(calls) == 3, {'success': len(calls) == 3})
        executor = RoboDojoExecutor(store, live(), step, max_control_steps=20)
        rows = tuple(tuple([.5] * 14) for _ in range(8))
        result = executor.execute(rows, action_contract=self.profile.action_contract)
        self.assertEqual(len(result.executed_actions), 3)
        self.assertTrue(result.terminated)
        self.assertEqual(result.measured, {})
        self.assertEqual(executor.native_result, {'success': True})
        self.assertEqual(len(list(store.since('ep', None, 3))), 4)

    def test_native_rollout_reaches_data_rsi_only_with_exact_segment_verification(self):
        store = self.store(); calls = []
        logger = TrajectoryLogger(self.root / 'rollouts', 'ep', dict(checkpoint='fixture',
            action_stats_version=self.profile.action_stats_version,
            camera_transform=store.camera_transform, runtime='robodojo-executor/v1'))
        self.addCleanup(logger.close)
        def step(row):
            calls.append(row)
            return NativeStep(live(len(calls)), len(calls) == 3, {'success': len(calls) == 3})
        ex = RoboDojoExecutor(store, live(), step, max_control_steps=20, trajectory=logger)
        ex.execute(tuple(tuple([.5] * 14) for _ in range(8)), action_contract=self.profile.action_contract)
        segs = segments_from_trajectory(logger.dir, outcome='success', split='development', task='fixture')
        self.assertEqual((segs[0].decision_step, segs[0].end_step), (0, 2))
        self.assertFalse(segs[0].action_supervision)
        self.assertTrue(segs[0].consequence_supervision)
        records = read_trajectory(logger.dir)
        self.assertEqual([r['executed_row'] for r in records if r['role'] == 'measured'], [None, 0, 1, 2])
        self.assertEqual(records[-1]['outcome'], 'success')
        # A separately versioned verifier must explicitly attest the executed prefix.
        logger.write('segment_verification', episode_id='ep', decision_step=0, end_step=2,
                     command_hash=segs[0].command_hash, verdict='passed',
                     verifier_revision='independent-fixture/v1', evidence_steps=[3])
        verified = segments_from_trajectory(logger.dir, outcome='success', split='development', task='fixture')
        self.assertTrue(verified[0].action_supervision)
        with self.assertRaises(ValueError):
            segments_from_trajectory(logger.dir, outcome='success', split='holdout', task='fixture')

    def test_rejects_budget_bad_contract_and_gripper_without_dispatch(self):
        store = self.store(); calls = []
        ex = RoboDojoExecutor(store, live(), lambda x: calls.append(x), max_control_steps=3)
        rows = tuple(tuple([.5] * 14) for _ in range(4))
        for plan, contract in [(rows, self.profile.action_contract), (rows[:1], 'wrong'),
                               ((tuple([2.] * 14),), self.profile.action_contract)]:
            with self.assertRaises(ValueError): ex.execute(plan, action_contract=contract)
        self.assertEqual(calls, [])

    def test_ambiguous_dispatch_is_not_replayed(self):
        store = self.store(); calls = []
        def failed(row):
            calls.append(row)
            raise OSError('feedback lost after dispatch')
        ex = RoboDojoExecutor(store, live(), failed, max_control_steps=10)
        with self.assertRaises(OSError): ex.execute((tuple([.5] * 14),), action_contract=self.profile.action_contract)
        with self.assertRaises(RuntimeError): ex.execute((tuple([.5] * 14),), action_contract=self.profile.action_contract)
        self.assertEqual(len(calls), 1)
        self.assertIn('execution_fault', (store.root / 'executor.jsonl').read_text())

    def test_backend_rebuilds_every_measured_row_after_partial_execution_and_switch(self):
        store = self.store(); first = store.add(live(), 0)
        backend = self.backend(store); backend.reset('ep')
        req = SkillRequest(first, 'move to bowl', self.profile.action_contract, 8)
        proposal = backend.propose(req)
        self.assertEqual(len(proposal.actions), 8)
        self.assertEqual(len(backend.history), 1)
        for step in range(1, 4): last = store.add(live(step), step)
        backend.observe(ExecutionFeedback('fixture', proposal.actions[:3], last, False))
        self.assertEqual(backend.history.last_step, 3)
        self.assertEqual(backend.prediction_valid_steps, 3)
        for step in range(4, 6): last = store.add(live(step), step)  # direct executed two rows
        backend.reset('ep')
        proposal = backend.propose(SkillRequest(last, 'new subgoal', self.profile.action_contract, 2))
        self.assertEqual(len(proposal.actions), 2)
        self.assertEqual(len(backend.history), 6)
        backend.observe(ExecutionFeedback('fixture', (), last, False))
        self.assertIsNone(backend.last_prediction)

    def test_runner_calls_native_backend_and_keeps_scores_out_of_agent(self):
        store = self.store(); calls = []
        def step(row):
            calls.append(row)
            return NativeStep(live(len(calls)), len(calls) == 8, {'success': True, 'score': 1})
        ex = RoboDojoExecutor(store, live(), step, max_control_steps=40)
        backend = self.backend(store)
        registry = SkillRegistry()
        registry.register(ControlSkill(SkillIdentity('wam_control', 'native/v1', self.profile.action_contract,
            14, 'fixture-checkpoint'), backend))
        agent = runner_fixtures.Scripted([CallSkill('wam_control', 'grasp cup', 8), runner_fixtures.accept(), Stop('done')])
        events = []
        runner = TaskSkillRunner(registry, ex, agent, AgentContext(TaskState('grasp cup')), sink=events.append)
        result = runner.run('ep')
        self.assertEqual(result['counters']['executed_steps'], 8)
        self.assertEqual(backend.history.last_step, 8)
        for event in events:
            if event['event'] == 'execution':
                self.assertNotIn('success', event['measured'])
                self.assertNotIn('score', event['measured'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
