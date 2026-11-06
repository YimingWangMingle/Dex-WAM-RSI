"""Offline planning of recorded requests. Never starts or steps a simulator."""
import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.robodojo.runtime.robodojo_motion_plan import checked_plan, PlanRejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--robodojo-root', type=Path, default=Path('/workspace/RoboDojo'))
    ap.add_argument('--yaw-sweep', action='store_true', help='Offline only: try alternate vertical-axis orientations for rejected goals')
    ap.add_argument('--old-jump-replay', action='store_true', help='Recompute the first rejected interpolation target; not historical solver seeds')
    ap.add_argument('--pose-sweep', action='store_true', help='Offline only: probe goal orientation/position constraints, never execute')
    ap.add_argument('--production-upgrade', action='store_true', help='Exercise runtime diagnostics, other-arm scene and bounded pose candidates')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.robodojo_root
    # The two repositories both contain utils; native simulation owns this import.
    code_root = str(Path(__file__).resolve().parents[1])
    sys.path[:] = [entry for entry in sys.path if entry != code_root]
    sys.path.insert(0, str(root))
    from env.planner_manager.curobo_planner import CuroboPlanner

    config = yaml.safe_load((root/'env_cfg/robot/dual_x5.yml').read_text())['robots']
    hz = float(yaml.safe_load((root/'env_cfg/arx_x5.yml').read_text())['observation']['collect_freq'])
    dt = float(yaml.safe_load((root/'env_cfg/sim/sim_config.yml').read_text())['dt'])
    names = [f'joint{i}' for i in range(1, 7)]
    joints = {j.get('name'): j for j in ET.parse(root/'Assets/Robots/x5/X5A.urdf').getroot().findall('joint')}
    limits = [[float(joints[n].find('limit').get(k)) for k in ('lower', 'upper')] for n in names]
    source = (root/'env/robot_manager/robot_config/x5.py').read_text()
    speeds = re.findall(r'velocity_limit_sim\s*=\s*([0-9.]+)', source)
    if len(speeds) != 1:
        raise ValueError('Expected one explicit X5 arm actuator velocity limit')
    velocity = [float(speeds[0])] * 6
    # Match the existing native robot-manager configuration, including its table.
    mounts = {}
    for arm, cfg in zip(('left', 'right'), config):
        q = np.asarray(cfg['default_root_rot'], float)
        mounts[arm] = [*cfg['default_root_pos'], *(q / np.linalg.norm(q))]
    if abs(mounts['left'][2]-mounts['right'][2]) > 1e-8:
        raise ValueError('This replay requires equal robot mounting heights')
    planner = CuroboPlanner(robot_origin_pose=[0, 0, 0, 1, 0, 0, 0],
                            active_joints_name=names, all_joints=names, dt=dt,
                            yml_path=str(root/'Assets/Robots/x5/curobo.yml'),
                            table_height=.74-mounts['left'][2])
    rows = [json.loads(s) for s in args.events.read_text().splitlines()]
    action_states = {r['control_step']: r['reported_state'] for r in rows if r['kind'] == 'action'}
    cases, pending = [], None
    for row in rows:
        if row['kind'] == 'agent':
            pending = row if row['name'] == 'move_tcp' else None
        elif row['kind'] == 'motion_result' and pending:
            cases.append((pending, row))
            pending = None
    results = []
    for index, (request, old) in enumerate(cases):
        a = request['arguments']; arm = a['arm']
        public = json.loads((Path(request['observation'])/'state.json').read_text())
        current = public['current_robot'][arm]
        start = public['reported_state'][arm+'_arm_joint_state']
        q = np.asarray([.5, -.5, .5, .5] if a['orientation'] == 'down' else current['quaternion_wxyz'])
        rot = Rotation.from_quat(q[[1, 2, 3, 0]])
        target = np.asarray(a['position']) - current['tool_offset_m'] * rot.apply([1, 0, 0])
        entry = dict(index=index, old_status=old['status'], source_control_step=request['control_step'],
                     arm=arm, target_tcp=a['position'], executed_steps=0)
        kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(start))
        fk_pose = kin.tool_poses.to_dict()[planner.motion_planner.tool_frames[0]]
        local_p = fk_pose.position.detach().cpu().numpy().reshape(-1, 3)[0] - np.asarray(planner.frame_bias)
        local_q = fk_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        mount = np.asarray(mounts[arm]); mount_r = Rotation.from_quat(mount[[4, 5, 6, 3]])
        measured_flange = np.asarray(current['flange_world'])
        entry['fk_position_error_m'] = float(np.linalg.norm(mount_r.apply(local_p) + mount[:3] - measured_flange[:3]))
        predicted_r = mount_r * Rotation.from_quat(local_q[[1, 2, 3, 0]])
        measured_r = Rotation.from_quat(measured_flange[[4, 5, 6, 3]])
        entry['fk_orientation_error_rad'] = float((predicted_r * measured_r.inv()).magnitude())
        same = planner.solve_ik_to_joint(start, measured_flange.tolist(), mounts[arm])
        entry['current_pose_ik_status'] = same['status']
        if args.old_jump_replay and old['status'] == 'joint_jump_rejected':
            q0 = np.asarray(current['quaternion_wxyz'])
            p0 = np.asarray(current['tcp_world'])
            angle = 2*math.acos(min(1., abs(float(q0 @ q))))
            count = max(8, math.ceil(np.linalg.norm(np.asarray(a['position'])-p0)/.005), math.ceil(angle/.04))
            fraction = (old['executed_steps'] + 1)/count
            ir = Slerp([0, 1], Rotation.from_quat(np.stack([q0, q])[:, [1, 2, 3, 0]]))([fraction])[0]
            ip = p0+(np.asarray(a['position'])-p0)*fraction
            iq = ir.as_quat()[[3, 0, 1, 2]]
            attempted_pose = [*(ip-current['tool_offset_m']*ir.apply([1, 0, 0])), *iq]
            at_failure = action_states.get(old['control_step'], public['reported_state'])[arm+'_arm_joint_state']
            planner.ik_solver.reset_seed()
            replay = planner.solve_ik_to_joint(at_failure, attempted_pose, mounts[arm])
            entry['old_jump_replay'] = dict(status=replay['status'], historical_rng_unavailable=True)
            if replay['status'] == 'Success':
                raw = np.asarray(replay['joint_value']).reshape(-1)
                nearest = np.array([min([v+2*math.pi*k for k in range(-4, 5) if lo <= v+2*math.pi*k <= hi], key=lambda z: abs(z-now))
                                    for v, now, (lo, hi) in zip(raw, at_failure, limits)])
                entry['old_jump_replay'].update(raw_delta_max=float(np.max(np.abs(raw-at_failure))),
                    equivalent_delta_max=float(np.max(np.abs(nearest-at_failure))),
                    raw_solution=raw.tolist(), equivalent_solution=nearest.tolist(), measured_joints=at_failure)
        if args.production_upgrade:
            from environments.robodojo.runtime.robodojo_motion_plan import NativeMotionPlanner
            from environments.robodojo.runtime.robodojo_pose_control import orientation_candidates
            adapter=NativeMotionPlanner.__new__(NativeMotionPlanner)
            adapter.planner=planner; adapter.hz=hz
            adapter.robot=SimpleNamespace(entity_origin_pose=mounts[arm])
            adapter.contract=lambda: (np.asarray(limits),np.asarray(velocity))
            other=NativeMotionPlanner.__new__(NativeMotionPlanner)
            other.planner=planner
            other_arm='right' if arm=='left' else 'left'
            other.robot=SimpleNamespace(entity_origin_pose=mounts[other_arm])
            entry['scene']=adapter.update_other_arm_scene(other.robot_spheres_world(
                public['reported_state'][other_arm+'_arm_joint_state']))
            entry['runtime_trials']=[]
            for candidate in orientation_candidates(q,45):
                rr=Rotation.from_quat(candidate[[1,2,3,0]])
                flange=np.r_[np.asarray(a['position'])-current['tool_offset_m']*rr.apply([1,0,0]),candidate]
                try:
                    planned=adapter.prepare(start,flange=flange)
                except PlanRejected as error:
                    entry['runtime_trials'].append(dict(quaternion_wxyz=candidate.tolist(),**error.report()))
                else:
                    entry['runtime_trials'].append(dict(quaternion_wxyz=candidate.tolist(),status='planned',**planned.report()))
                    break
            entry['status']=entry['runtime_trials'][-1]['status']
            results.append(entry)
            (args.output/'results.json').write_text(json.dumps(dict(diagnostic_only=True,simulator_steps=0,cases=results),indent=2))
            print('RUNTIME_CASE',index,entry['status'],flush=True)
            continue
        begin = time.monotonic()
        result = planner.plan_path(start, [*target, *q], mounts[arm])
        try:
            plan = checked_plan(result, start, limits, velocity, hz)
        except PlanRejected as error:
            entry.update(status='rejected', reason=str(error))
            if args.pose_sweep:
                entry['pose_trials'] = []
                variants = []
                for degrees in [15, -15, 30, -30, 45, -45]:
                    for axis in ('x', 'y'):
                        variants.append((dict(axis=axis, degrees=degrees), np.asarray(a['position']),
                                         Rotation.from_euler(axis, degrees, degrees=True)*rot))
                toward = np.asarray(mounts[arm][:3])-np.asarray(a['position']); toward[2] = 0
                toward /= np.linalg.norm(toward)
                for shift in [.02, .04, .06]:
                    variants.append((dict(toward_base_m=shift), np.asarray(a['position'])+toward*shift, rot))
                    variants.append((dict(lower_m=shift), np.asarray(a['position'])-[0, 0, shift], rot))
                for label, tcp, alternative in variants:
                    aq = alternative.as_quat()[[3, 0, 1, 2]]
                    ap = tcp-current['tool_offset_m']*alternative.apply([1, 0, 0])
                    trial = planner.plan_path(start, [*ap, *aq], mounts[arm])
                    try:
                        alternative_plan = checked_plan(trial, start, limits, velocity, hz)
                    except PlanRejected as reason:
                        entry['pose_trials'].append(dict(**label, status='rejected', reason=str(reason)))
                    else:
                        entry['pose_trials'].append(dict(**label, status='planned', **alternative_plan.report()))
                        break
            if args.yaw_sweep:
                entry['yaw_trials'] = []
                for degrees in [15, -15, 30, -30, 45, -45, 90, -90]:
                    alternative = Rotation.from_euler('z', degrees, degrees=True) * rot
                    aq = alternative.as_quat()[[3, 0, 1, 2]]
                    ap = np.asarray(a['position']) - current['tool_offset_m'] * alternative.apply([1, 0, 0])
                    trial = planner.plan_path(start, [*ap, *aq], mounts[arm])
                    try:
                        alternative_plan = checked_plan(trial, start, limits, velocity, hz)
                    except PlanRejected as reason:
                        entry['yaw_trials'].append(dict(degrees=degrees, status='rejected', reason=str(reason)))
                    else:
                        entry['yaw_trials'].append(dict(degrees=degrees, status='planned', **alternative_plan.report()))
                        break
        else:
            entry.update(status='planned', **plan.report())
            np.save(args.output/f'plan-{index:02d}.npy', plan.commands)
        entry['wall_seconds'] = time.monotonic()-begin
        results.append(entry)
        (args.output/'results.json').write_text(json.dumps(dict(
            diagnostic_only=True, simulator_steps=0, limits_source='public URDF (live runtime uses measured soft limits)',
            control_hz=hz, cases=results), indent=2))
        print(json.dumps(entry), flush=True)
    print('Completed offline request comparison; no task success assessment.', flush=True)


if __name__ == '__main__':
    main()
