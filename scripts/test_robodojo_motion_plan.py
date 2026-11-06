"""CPU contract tests; no simulator or model API required."""
from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.robodojo.runtime.robodojo_motion_plan import checked_plan, NativeMotionPlanner, PlanRejected


class MotionPlanTests(unittest.TestCase):
    def plan(self, path, **kw):
        return checked_plan(dict(status='Success', position=path), np.zeros(2),
                            [[-3, 3], [-3, 3]], [1, 2], 25, **kw)

    def test_dense_sparse_same_motion_time(self):
        dense = np.linspace([0, 0], [1, 2], 501)
        sparse = [[0, 0], [1, 2]]
        a, b = self.plan(dense), self.plan(sparse)
        self.assertLessEqual(abs(len(a.commands) - len(b.commands)), 1)
        self.assertGreaterEqual(len(a.commands), 25)
        for plan in (a, b):
            self.assertTrue(np.all(np.abs(np.diff(np.vstack([[0, 0], plan.commands]), axis=0)) <= plan.max_step + 1e-6))
            np.testing.assert_allclose(plan.commands[-1], [1, 2])

    def test_reversal_and_duplicates_preserve_travel(self):
        plan = self.plan([[0, 0], [1, 0], [1, 0], [0, 0]])
        self.assertEqual(len(plan.commands), 50)
        self.assertAlmostEqual(plan.commands[:, 0].max(), 1)
        np.testing.assert_allclose(plan.commands[-1], [0, 0])

    def test_stationary_and_one_point(self):
        for path in ([[0, 0]], [[0, 0], [0, 0]]):
            np.testing.assert_array_equal(self.plan(path).commands, [[0, 0]])

    def test_bad_paths_rejected_before_playout(self):
        for path in ([], [[.01, 0], [1, 1]], [[0, 0], [4, 0]], [[0, 0], [float('nan'), 1]], [[0, 0, 0]]):
            with self.subTest(path=path), self.assertRaises(PlanRejected):
                self.plan(path)
        with self.assertRaisesRegex(PlanRejected, 'duration'):
            self.plan([[0, 0], [2, 0]], max_duration_s=.5)
        with self.assertRaisesRegex(PlanRejected, 'no_path'):
            checked_plan({'status': 'Fail'}, [0, 0], [[-3, 3]]*2, [1, 1], 25)

    def test_named_adapter_passes_measured_start_and_pose(self):
        calls = []
        class Planner:
            active_joints_name = ['j1', 'j2']
            def plan_path(self, start, goal, base):
                calls.append(('pose', start, goal, base))
                return dict(status='Success', position=[start, [.2, .3]])
            def plan_joint(self, start, goal):
                calls.append(('joint', start, goal))
                return dict(status='Success', position=[start, goal])
        robot = SimpleNamespace(robot_name='left', arm_joints_name=['j1', 'j2'],
                                arm_joint_indices=[0, 1], entity_origin_pose=[0, 0, 0, 1, 0, 0, 0])
        asset = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=np.array([[[-3, 3], [-3, 3]]]),
                                                     joint_vel_limits=np.array([[1, 2]])))
        env = SimpleNamespace(robot_manager=SimpleNamespace(planner={'left': Planner()}, robot_list=[robot], robot_key=[asset]),
                              obs_manager=SimpleNamespace(collect_freq=25))
        planner = NativeMotionPlanner(env, robot)
        planner.prepare([.1, .1], flange=[0, 0, 1, 1, 0, 0, 0])
        self.assertEqual(calls[0][1], [.1, .1])
        planner.prepare([.1, .1], goal_joints=[0, 0])
        self.assertEqual(calls[1][0], 'joint')

    def test_pilot_rejection_executes_nothing_and_return_uses_planner(self):
        from environments.robodojo.runtime import robodojo_deepseek_ik as pilot
        state = {a+'_arm_joint_state': np.zeros(6) for a in ('left', 'right')}
        state.update({a+'_ee_joint_state': np.ones(1) for a in ('left', 'right')})
        class Planner:
            active_joints_name = [f'joint{i}' for i in range(1, 7)]
            fail = False
            joint_calls = 0
            def plan_path(self, start, pose, base):
                if self.fail:
                    return {'status': 'Fail'}
                goal = np.zeros(6); goal[:3] = np.asarray(pose[:3]) - [0, 0, 1]
                return dict(status='Success', position=[start, goal])
            def plan_joint(self, start, goal):
                self.joint_calls += 1
                return dict(status='Success', position=[start, goal])
        planner = Planner()
        robots = [SimpleNamespace(type='target', arm_name=a+'_arm', robot_name=a, gripper_bias=.145,
                    arm_joints_name=planner.active_joints_name, arm_joint_indices=list(range(6)),
                    entity_origin_pose=[0, 0, 0, 1, 0, 0, 0]) for a in ('left', 'right')]
        def flange(robot, *args, **kwargs):
            a = robot.robot_name
            return {0: np.r_[state[a+'_arm_joint_state'][:3]+[0, 0, 1], [1, 0, 0, 0]]}
        asset = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=np.array([[[-3, 3]]*6]),
                                                     joint_vel_limits=np.array([[5]*6])))
        class Env:
            env_seeds = [1]
            success = [False]
            count = 0
            robot_manager = SimpleNamespace(robot_list=robots, robot_key=[asset, asset],
                planner={'left': planner, 'right': planner}, get_real_endpose=flange,
                get_end_effector_real_val=lambda *a: {0: [0.044, 0.044]})
            obs_manager = SimpleNamespace(collect_freq=25)
            def get_obs(self):
                return dict(state=state, vision={c: {'color': np.zeros((4, 4, 3), np.uint8)} for c in pilot.CAMERAS})
            def take_action(self, action):
                self.count += 1
                state.update({k: np.asarray(v).copy() for k, v in action.items()})
            def validate_action_dict(self, action): pass
            def is_episode_end(self): return False
        class Client:
            last_metadata = {}
            calls = 0
            def complete(self, *args):
                self.calls += 1
                if self.calls == 1:
                    name, data = 'return_initial', dict(arm='left', reason='test')
                elif self.calls == 2:
                    planner.fail = True
                    name, data = 'move_tcp', dict(arm='left', position=[.155, 0, 1], orientation='keep', reason='test')
                else:
                    name, data = 'stop', dict(reason='test finished')
                return {'choices': [{'message': {'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(data)}}]}}]}
        old = (pilot.CLIENT, pilot.ROOT, pilot.EXPERIENCE)
        try:
            with tempfile.TemporaryDirectory() as directory:
                pilot.CLIENT, pilot.ROOT, pilot.EXPERIENCE = Client(), directory, 'test'
                env = Env()
                with patch.dict(os.environ,{'RSI_CONTACT_SCENE':'0'}), patch.object(NativeMotionPlanner, 'robot_spheres_world', return_value=[[0,0,0,.1]]), patch.object(NativeMotionPlanner, 'update_other_arm_scene', return_value={'test_double':True}):
                    pilot.eval_one_episode(env, None)
                rows = [json.loads(s) for s in (Path(directory)/'events.jsonl').read_text().splitlines()]
                rejected = [r for r in rows if r['kind'] == 'motion_result' and r['status'] == 'plan_rejected']
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0]['executed_steps'], 0)
                self.assertEqual(env.count, 5)  # Four calibration moves and one joint return.
                self.assertEqual(planner.joint_calls, 1)
        finally:
            pilot.CLIENT, pilot.ROOT, pilot.EXPERIENCE = old


if __name__ == '__main__':
    unittest.main()
