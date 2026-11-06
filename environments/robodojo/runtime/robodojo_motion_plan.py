"""Validate and time complete native CuRobo paths before issuing any action.

Timing follows WorldScape PR162's joint-budget path metric. This adapter uses
RoboDojo's existing planner/model; it does not import a task's object poses.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from copy import deepcopy
import math
import os
from typing import Any

import numpy as np


class PlanRejected(ValueError):
    """No motion has been submitted for this planning request."""
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

    def report(self):
        return dict(status='plan_rejected', reason=str(self), executed_steps=0,
                    diagnostics=self.diagnostics)


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass
class TimedPlan:
    commands: np.ndarray
    source_waypoints: int
    control_hz: float
    max_step: np.ndarray
    source_path: Any = None

    def report(self) -> dict:
        return dict(source_waypoints=self.source_waypoints,
                    planned_steps=len(self.commands), control_hz=self.control_hz,
                    planned_seconds=len(self.commands) / self.control_hz,
                    joint_max_step=self.max_step.tolist())


def checked_plan(result: dict, start: Any, limits: Any, velocity: Any,
                 control_hz: float, max_duration_s: float = 10.0) -> TimedPlan:
    """Reject malformed paths; resample without compressing their joint budget."""
    if result.get("status") != "Success" or result.get("position") is None:
        raise PlanRejected("planner_no_path: no action executed; choose another pose")
    start, limits, velocity = map(numpy, (start, limits, velocity))
    n = start.size
    if (start.shape != (n,) or n == 0 or limits.shape != (n, 2)
            or velocity.shape != (n,) or not all(np.isfinite(x).all() for x in (start, limits, velocity))
            or np.any(limits[:, 0] >= limits[:, 1]) or np.any(velocity <= 0)
            or not math.isfinite(control_hz) or control_hz <= 0
            or not math.isfinite(max_duration_s) or max_duration_s <= 0):
        raise PlanRejected("invalid_robot_motion_contract")
    path = numpy(result["position"]).copy()
    if path.ndim != 2 or path.shape[1] != n or not 1 <= len(path) <= 100000 or not np.isfinite(path).all():
        raise PlanRejected("invalid_planner_path")
    if np.max(np.abs(path[0] - start)) > 1e-3:
        raise PlanRejected("planner_start_discontinuity")
    path[0] = start
    if np.any(path < limits[:, 0] - 1e-6) or np.any(path > limits[:, 1] + 1e-6):
        raise PlanRejected("planner_joint_limit_violation")
    max_step = np.minimum(limits[:, 1] - limits[:, 0], velocity / control_hz)
    travel = np.max(np.abs(np.diff(path, axis=0)) / max_step, axis=1)
    distance = np.r_[0., np.cumsum(travel)]
    steps = max(1, math.ceil(float(distance[-1])))
    if steps > math.floor(max_duration_s * control_hz):
        raise PlanRejected(f"plan_duration_exceeded: needs {steps / control_hz:.3f}s; split the motion")
    keep = np.r_[True, np.diff(distance) > 0]
    ticks = np.linspace(0., distance[-1], steps + 1)[1:]
    commands = np.stack([np.interp(ticks, distance[keep], path[keep, j]) for j in range(n)], axis=1)
    commands[-1] = path[-1]
    delta = np.abs(np.diff(np.vstack([start, commands]), axis=0))
    if np.any(delta > max_step + 1e-6):
        raise PlanRejected("resampled_step_limit_violation")
    return TimedPlan(commands, len(path), float(control_hz), max_step, path)


@contextmanager
def planner_diagnostics(native):
    """Scope instrumentation to one serialized call; never change solver criteria.

    Native RoboDojo discards failure details. Read the exact attempt's metrics
    before they are discarded; unsupported solver versions report unknown.
    """
    report = dict(code='planner_no_path', ik_attempts=[])
    motion = getattr(native, 'motion_planner', None)
    solver = getattr(motion, 'ik_solver', None)
    if solver is None or not hasattr(solver, '_get_result'):
        report['detail'] = 'native diagnostic hooks unavailable'
        yield report
        return
    original = solver._get_result

    def capture(opt, metrics, *args, **kwargs):
        result = original(opt, metrics, *args, **kwargs)
        try:
            pos = numpy(result.position_error).reshape(-1)
            k = int(np.argmin(pos))
            seed = int(numpy(result.debug_info['seed_idx']).reshape(-1)[k])
            costs = metrics.costs_and_constraints.constraints
            penalties = {name: float(np.max(numpy(value)[seed]))
                         for name, value in zip(costs.names, costs.values)}
            attempt = dict(success=bool(numpy(result.success).any()),
                           best_position_error_m=float(pos[k]),
                           best_rotation_error_rad=float(numpy(result.rotation_error).reshape(-1)[k]),
                           candidate_constraint_costs=penalties)
            if penalties.get('self_collision', 0) > 0:
                kin = solver.kinematics
                cfg = kin.get_self_collision_config()
                params = kin.config.kinematics_config
                names = {v: name for name, v in params.link_name_to_idx_map.items()}
                links = numpy(params.link_sphere_idx_map).astype(int)
                pairs = numpy(cfg.collision_pairs).astype(int)
                pad = numpy(cfg.sphere_padding)
                spheres = numpy(metrics.state.cuda_robot_model_state.robot_spheres).reshape(-1, cfg.num_spheres, 4)[seed]
                i, j = pairs[:, 0], pairs[:, 1]
                gaps = np.linalg.norm(spheres[i, :3]-spheres[j, :3], axis=1)-spheres[i, 3]-spheres[j, 3]-pad[i]-pad[j]
                attempt['candidate_self_collision_pairs'] = [dict(
                    links=[names[int(links[i[n]])], names[int(links[j[n]])]],
                    surface_gap_m=float(gaps[n])) for n in np.argsort(gaps)[:2] if gaps[n] < 0]
            report['ik_attempts'].append(attempt)
        except Exception as error:
            report['diagnostic_error'] = type(error).__name__ + ': ' + str(error)
        return result

    solver._get_result = capture
    try:
        yield report
    finally:
        solver._get_result = original
        attempts = report.pop('ik_attempts')
        report['ik_result_batches'] = len(attempts)
        if attempts:
            report['best_candidate'] = min(attempts, key=lambda a:a['best_position_error_m'])
            if any(a['success'] for a in attempts):
                report['code'] = 'trajectory_plan_failed'
            else:
                report['code'] = 'goal_ik_failed'
                report['self_collision_candidates_seen'] = any(
                    a['candidate_constraint_costs'].get('self_collision', 0) > 0 for a in attempts)
        report['interpretation'] = 'Candidate constraint evidence; not a proof that every IK solution is impossible.'


class NativeMotionPlanner:
    """Read live robot limits and use the already configured native planner."""
    def __init__(self, env: Any, robot: Any):
        self.env, self.robot = env, robot
        manager = env.robot_manager
        self.planner = manager.planner[robot.robot_name]
        if list(self.planner.active_joints_name) != list(robot.arm_joints_name):
            raise ValueError("planner/robot joint ordering mismatch")
        self.asset = manager.robot_key[manager.robot_list.index(robot)]
        self.hz = float(env.obs_manager.collect_freq)

    def contract(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self.robot.arm_joint_indices
        return (numpy(self.asset.data.soft_joint_pos_limits[0, ids]),
                numpy(self.asset.data.joint_vel_limits[0, ids]))

    def robot_spheres_world(self, joints):
        """Robot-only geometry from public model + measured joints, not object GT.

        Coverage follows the public model. The current X5 collision model has
        no finger links; report that limitation, rather than inventing a mesh.
        """
        from scipy.spatial.transform import Rotation
        native = self.planner
        if not hasattr(native, 'motion_planner'):
            raise PlanRejected('robot_collision_geometry_unavailable')
        kin = native.motion_planner.compute_kinematics(native._build_joint_state(joints))
        spheres = numpy(kin.robot_spheres).reshape(-1, 4).copy()
        pose = numpy(self.robot.entity_origin_pose).reshape(-1)
        rotation = Rotation.from_quat(pose[[4,5,6,3]])
        spheres[:, :3] = rotation.apply(spheres[:, :3]-numpy(native.frame_bias)) + pose[:3]
        return spheres[spheres[:, 3] > 0]

    def update_other_arm_scene(self, spheres_world, extra_boxes=None):
        """Keep native table; add the other arm as conservative sphere AABBs.

        Only one arm moves per tool call. Refresh from measured state before
        every plan. The fixed other arm is also checked during execution.
        """
        from scipy.spatial.transform import Rotation
        from curobo.scene import Scene
        native = self.planner
        if not np.allclose(native.robot_origin_pose, [0,0,0,1,0,0,0]):
            raise PlanRejected('unsupported_planner_origin_for_scene_transform')
        spheres = numpy(spheres_world)
        if spheres.ndim != 2 or spheres.shape[1] != 4 or not np.isfinite(spheres).all() or np.any(spheres[:,3] <= 0):
            raise PlanRejected('invalid_other_arm_spheres')
        pose = numpy(self.robot.entity_origin_pose).reshape(-1)
        rotation = Rotation.from_quat(pose[[4,5,6,3]])
        centres = rotation.inv().apply(spheres[:,:3]-pose[:3])+numpy(native.frame_bias)
        scene = deepcopy(native.scene_model)
        boxes = scene.setdefault('cuboid', {})
        for i, (centre, sphere) in enumerate(zip(centres, spheres)):
            radius = float(sphere[3])+.005
            boxes['other_arm_'+str(i)] = dict(dims=[2*radius]*3, pose=[*centre,1,0,0,0])
        for name,bounds in (extra_boxes or {}).items():
            low,high=numpy(bounds)
            if not np.isfinite([low,high]).all() or np.any(high<=low):
                raise PlanRejected('invalid_observed_collision_bounds')
            centre=rotation.inv().apply((low+high)/2-pose[:3])+numpy(native.frame_bias)
            q=rotation.inv().as_quat()
            boxes['observed_'+name]=dict(dims=(high-low).tolist(),pose=[*centre,q[3],*q[:3]])
        if len(boxes) > 128:
            raise PlanRejected('collision_scene_capacity_exceeded')
        self.ensure_scene_capacity()
        world = Scene.create(scene)
        native.motion_planner.update_world(world)
        native.ik_solver.update_world(world)
        return dict(other_arm_spheres=len(spheres), additional_margin_m=.005,
                    source='measured_joints_and_public_robot_collision_model',
                    observed_boxes=list(extra_boxes or {}),
                    bowl_geometry_included=any(k.startswith('bowl_') for k in (extra_boxes or {})),
                    stationary_finger_geometry_included=any(k.startswith('finger_') for k in (extra_boxes or {})))

    def flange_world(self, joints):
        from scipy.spatial.transform import Rotation
        native=self.planner
        kin=native.motion_planner.compute_kinematics(native._build_joint_state(joints))
        tool=kin.tool_poses.to_dict()[native.motion_planner.tool_frames[0]]
        local_p=numpy(tool.position).reshape(-1,3)[0]-numpy(native.frame_bias)
        local_q=numpy(tool.quaternion).reshape(-1,4)[0]
        mount=numpy(self.robot.entity_origin_pose)
        r=Rotation.from_quat(mount[[4,5,6,3]])
        q=(r*Rotation.from_quat(local_q[[1,2,3,0]])).as_quat()
        return np.r_[r.apply(local_p)+mount[:3],q[3],q[:3]]

    def table_boxes_world(self):
        from scipy.spatial.transform import Rotation
        from itertools import product
        mount=numpy(self.robot.entity_origin_pose)
        r=Rotation.from_quat(mount[[4,5,6,3]])
        result=[]
        for box in self.planner.scene_model.get('cuboid',{}).values():
            pose=numpy(box['pose']); dims=numpy(box['dims'])
            corners=numpy(list(product([-1,1],repeat=3)))*dims/2
            corners=Rotation.from_quat(pose[[4,5,6,3]]).apply(corners)+pose[:3]
            corners=r.apply(corners-numpy(self.planner.frame_bias))+mount[:3]
            result.append(np.array([corners.min(axis=0),corners.max(axis=0)]))
        if not result: raise PlanRejected('table_collision_geometry_unavailable')
        return result

    def ensure_scene_capacity(self):
        """Allocate fixed capacity before capturing graphs, not during updates.

        The native table-only planner allocates ONE cuboid. Rebuild its single
        query solvers once with the same settings and an explicit cache. Leave
        the unused batch solver alone. Never fall back to the stale scene.
        """
        native = self.planner
        if getattr(native, '_rsi_scene_capacity', 0) >= 128:
            return
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        common = dict(robot=native.robot_cfg, scene_model=native.scene_model,
                      device_cfg=native.device_cfg, self_collision_check=True,
                      use_cuda_graph=native.use_cuda_graph, max_batch_size=1,
                      multi_env=False, max_goalset=1, collision_cache={'obb':128})
        cfg = MotionPlannerCfg.create(**common,num_ik_seeds=32,num_trajopt_seeds=4)
        if native.dt is not None:
            cfg.trajopt_solver_config.interpolation_dt = float(native.dt)
        if not native.use_graph_planner:
            cfg.graph_planner_config = None
        motion = MotionPlanner(cfg)
        ik = None
        try:
            motion.warmup(enable_graph=native.use_graph_planner,num_warmup_iterations=5)
            ik = InverseKinematics(InverseKinematicsCfg.create(**common,num_seeds=32,
                seed_solver_num_seeds=32,position_tolerance=.001,orientation_tolerance=.02))
        except Exception:
            motion.destroy()
            if ik is not None: ik.destroy()
            raise
        old_motion, old_ik = native.motion_planner, native.ik_solver
        native.motion_planner, native.ik_solver = motion, ik
        native._rsi_scene_capacity = 128
        old_motion.destroy(); old_ik.destroy()

    def prepare(self, start: Any, *, flange: Any = None, goal_joints: Any = None) -> TimedPlan:
        if (flange is None) == (goal_joints is None):
            raise ValueError("provide exactly one pose or joint goal")
        limits, velocity = self.contract()
        command_cap = float(os.environ.get('RSI_JOINT_STEP_RAD', '.05'))
        if not math.isfinite(command_cap) or not 0 < command_cap <= .2:
            raise PlanRejected('RSI_JOINT_STEP_RAD must be in (0,0.2]')
        velocity = np.minimum(velocity, self.hz*command_cap)
        if flange is not None:
            with planner_diagnostics(self.planner) as diagnostic:
                result = self.planner.plan_path(numpy(start).tolist(), numpy(flange).tolist(),
                                               self.robot.entity_origin_pose)
        else:
            goal = numpy(goal_joints)
            if goal.shape != (len(limits),) or not np.isfinite(goal).all() or np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
                raise PlanRejected("invalid_joint_goal")
            result = self.planner.plan_joint(numpy(start).tolist(), goal.tolist())
            diagnostic = dict(code='joint_path_failed')
        try:
            return checked_plan(result, start, limits, velocity, self.hz)
        except PlanRejected as error:
            raise PlanRejected(str(error), diagnostic) from error

    def check_orientation_path(self, plan, start, max_deviation_deg=5.):
        """Check dense source path AND submitted points against start orientation.

        Fixed endpoint orientation alone does not constrain intermediate poses.
        """
        from environments.robodojo.runtime.robodojo_pose_control import quaternion_angle
        native = self.planner
        if not hasattr(native, 'motion_planner'):
            raise PlanRejected('orientation_path_check_unavailable')
        def orientation(joints):
            kin = native.motion_planner.compute_kinematics(native._build_joint_state(joints))
            pose = kin.tool_poses.to_dict()[native.motion_planner.tool_frames[0]]
            return numpy(pose.quaternion).reshape(-1, 4)[0]
        reference = orientation(start)
        for q in np.vstack([plan.source_path, plan.commands]):
            if quaternion_angle(reference, orientation(q)) > math.radians(max_deviation_deg):
                raise PlanRejected('held_object_orientation_path_violation')
