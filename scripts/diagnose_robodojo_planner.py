"""Capture native planner diagnostics while replaying saved requests offline."""
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, '/workspace/RoboDojo')
from env.planner_manager.curobo_planner import CuroboPlanner

output = Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir(parents=True, exist_ok=True)


def compact(x, depth=0):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if hasattr(x, 'detach') and hasattr(x, 'numel'):
        x = x.detach().cpu()
        if x.numel() <= 32:
            return x.tolist()
        return dict(shape=list(x.shape), minimum=float(x.min()), maximum=float(x.max()))
    if depth > 3:
        return type(x).__name__
    if isinstance(x, dict):
        return {str(k): compact(v, depth+1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [compact(v, depth+1) for v in x[:10]]
    if hasattr(x, '__dict__'):
        return {'class': type(x).__name__, **{k: compact(v, depth+1) for k, v in vars(x).items() if not k.startswith('_')}}
    return str(x)


original_init = CuroboPlanner.__init__


def instrument(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    original_plan = self.motion_planner.plan_pose
    original_ik = self.motion_planner.ik_solver.solve_pose
    attempts = []
    constraint_metrics = []
    original_result = self.motion_planner.ik_solver._get_result
    def traced_result(opt_result, metrics_result, *a, **kw):
        constraint_metrics.append(compact(metrics_result.costs_and_constraints))
        r = original_result(opt_result, metrics_result, *a, **kw)
        kin = self.motion_planner.ik_solver.kinematics
        cfg = kin.get_self_collision_config()
        params = kin.config.kinematics_config
        names = {v: k for k, v in params.link_name_to_idx_map.items()}
        sphere_links = params.link_sphere_idx_map.detach().cpu().numpy()
        pairs = cfg.collision_pairs.detach().cpu().numpy().astype(int)
        pad = cfg.sphere_padding.detach().cpu().numpy()
        spheres = metrics_result.state.cuda_robot_model_state.robot_spheres.detach().cpu().numpy()
        spheres = spheres.reshape(-1, cfg.num_spheres, 4)
        contacts = []
        for idx in r.debug_info['seed_idx'].reshape(-1).tolist():
            ss = spheres[idx]
            i, j = pairs[:, 0], pairs[:, 1]
            gap = np.linalg.norm(ss[i, :3]-ss[j, :3], axis=1) - ss[i, 3] - ss[j, 3] - pad[i] - pad[j]
            contacts.append([dict(links=[names[int(sphere_links[i[k]])], names[int(sphere_links[j[k]])]],
                                  surface_gap_m=float(gap[k])) for k in np.argsort(gap)[:3]])
        r._diagnostic_contacts = contacts
        return r
    self.motion_planner.ik_solver._get_result = traced_result
    def traced_ik(*a, **kw):
        r = original_ik(*a, **kw)
        item = compact(r)
        item['closest_self_collision_pairs'] = r._diagnostic_contacts
        attempts.append(item)
        return r
    self.motion_planner.ik_solver.solve_pose = traced_ik
    def traced(*a, **kw):
        attempts.clear()
        constraint_metrics.clear()
        result = original_plan(*a, **kw)
        entry = dict(result=compact(result), current=compact(kw.get('current_state')),
                     goal=compact(kw.get('goal_tool_poses')), ik_attempts=list(attempts),
                     constraints=list(constraint_metrics))
        with (output/'raw-planner-results.jsonl').open('a') as f:
            f.write(json.dumps(entry, default=str)+'\n')
        print('RAW_PLANNER', json.dumps(dict(trajectory_returned=result is not None,
              ik_attempts=len(attempts), last_ik_success=attempts[-1]['success'] if attempts else None), default=str), flush=True)
        return result
    self.motion_planner.plan_pose = traced


CuroboPlanner.__init__ = instrument
world = os.environ.get('RSI_DIAGNOSTIC_WORLD', 'native')
if world == 'robot_only':
    original_config = CuroboPlanner._build_robot_and_scene_cfg
    def robot_only(self, *a, **kw):
        robot, scene = original_config(self, *a, **kw)
        return robot, None
    CuroboPlanner._build_robot_and_scene_cfg = robot_only
elif world != 'native':
    raise ValueError('unknown diagnostic world')
from probe_robodojo_motion_plans import main
main()
