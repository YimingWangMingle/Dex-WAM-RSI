"""CPU contract tests for Joint24 deployment and the retained AR defaults."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environments.robodojo.training import load_training
from thinker_wm.runtime.robodojo_policy import RoboDojoPolicy, load_runtime


class RuntimeTests(unittest.TestCase):
    def profile(self, name):
        return load_training(ROOT / 'third_party/Thinker-WM/configs' / (name + '.yaml'),
            ROOT / 'third_party/Thinker-WM/configs/robodojo/arx_x5_joint14_v1.field_map.yaml')[2]

    def test_joint24_profile_and_default(self):
        p = self.profile('robodojo_joint24_v1')
        runtime, horizon = load_runtime(p)
        self.assertEqual((p.frames_per_block, p.actions_per_block), (12, 24))
        self.assertEqual((horizon.num_blocks, horizon.predict_steps, horizon.execute_steps), (1, 24, 16))
        self.assertEqual(runtime['continuity'], {'mode': 'ema', 'ema_alpha': .5})

    def test_old_ar_default_and_explicit_runtime(self):
        p = self.profile('rsi_robodojo_v1')
        self.assertEqual(load_runtime(p)[1].predict_steps, 8)
        self.assertEqual(load_runtime(p)[1].execute_steps, 8)
        h = load_runtime(p, ROOT / 'third_party/Thinker-WM/configs/runtime/robodojo_ar2_joint_continuity.json')[1]
        self.assertEqual((h.predict_steps, h.execute_steps), (16, 8))

    def test_joint24_rejects_serial_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp, 'runtime.json')
            config.write_text(json.dumps({'num_blocks': 2, 'execute_steps': 16}))
            with self.assertRaises(ValueError): load_runtime(self.profile('robodojo_joint24_v1'), config)

    def test_only_first16_commands_then_replan(self):
        policy = RoboDojoPolicy.__new__(RoboDojoPolicy)
        _, policy.horizon = load_runtime(self.profile('robodojo_joint24_v1'))
        policy.block_size = policy.horizon.execute_steps
        policy.observation = NS(step=0, state=np.zeros(14))
        policy.next_decision = 0
        policy.history = NS(describe=lambda: {})
        policy.instruction, policy.episode_id = 'stack', 'test'
        policy.dims = None
        policy.continuity = None
        policy._record = lambda data: None
        policy.joint_indices = np.array([0,1,2,3,4,5,7,8,9,10,11,12])
        calls = []
        def generate(obs, *args):
            calls.append(obs.step)
            return NS(actions=np.full((24, 14), len(calls) * .1, dtype=np.float32))
        policy.inf = NS(generate_action=generate)
        module = ModuleType('XPolicyLab.utils.process_data')
        module.unpack_robot_state = lambda commands, *a, **k: commands
        with patch.dict(sys.modules, {'XPolicyLab.utils.process_data': module}):
            first = policy.get_action()
            self.assertEqual(first.shape, (16,14))
            self.assertEqual(policy.next_decision, 16)
            with self.assertRaises(RuntimeError): policy.get_action()
            policy.observation = NS(step=16, state=np.ones(14) * .1)
            second = policy.get_action()
        self.assertEqual(calls, [0,16])
        np.testing.assert_allclose(second, .2)
        self.assertEqual(policy.next_decision, 32)


if __name__ == '__main__': unittest.main(verbosity=2)
