"""No-API simulator smoke: calibration projection and one empty-hand pose move."""
import getpass
import json
import os
from pathlib import Path
import socket
import sys
import time
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.deepseek import DeepSeekClient
from environments.robodojo.runtime import robodojo_deepseek_ik as pilot

root = Path(os.environ["RSI_LIVE_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)
class ScriptedClient:
    last_metadata = {'scripted_smoke': True, 'api_calls': 0}
    count = 0
    def complete(self, messages, tools):
        import numpy as np
        from environments.robodojo.runtime.robodojo_pose_control import orientation_candidates
        public = json.loads(messages[1]['content'][0]['text'])
        if self.count == 0:
            arm = public['current_robot']['left']
            q = list(orientation_candidates(arm['quaternion_wxyz'], 15))[1]
            target = (np.asarray(arm['tcp_world']) + [0,0,.01]).tolist()
            if os.environ.get('RSI_SMOKE_TARGET') == 'recorded_approach':
                name,args='move_tcp',dict(arm='left',position=[-.2,0,.95],orientation='down',
                    max_tilt_deg=30,reason='recorded empty approach with bounded tilt search')
            else:
                name, args = 'move_tcp', dict(arm='left',position=target,orientation='explicit',
                    quaternion_wxyz=q.tolist(),max_tilt_deg=0,reason='empty-hand 15 degree orientation smoke')
        else:
            name,args='stop',dict(reason='smoke complete; not a task-success evaluation')
        self.count += 1
        return {'choices':[{'message':{'tool_calls':[{'function':{'name':name,'arguments':json.dumps(args)}}]}}]}

pilot.CLIENT=ScriptedClient()
pilot.ROOT=root
pilot.EXPERIENCE='Smoke test only: no grasp or bowl contact.'
original_episode=pilot.eval_one_episode
def smoke_episode(TASK_ENV, model_client):
    import numpy as np
    from itertools import combinations
    from environments.robodojo.runtime.robodojo_local_geometry import LocalGeometry
    from tools.perception.geometry import project, triangulate
    geometry=LocalGeometry(TASK_ENV)
    obs=TASK_ENV.get_obs()
    frames={c:geometry.camera(c,obs,0) for c in pilot.CAMERAS}
    checks=[]
    # Check algebra/axis/resolution binding against the actual camera's projection.
    # These are robot TCP references, not bowl/object ground truth.
    for robot in TASK_ENV.robot_manager.robot_list:
        if robot.type!='target': continue
        flange=TASK_ENV.robot_manager.get_real_endpose(robot,[0],is_relative=False)[0]
        xyz=pilot.to_tcp(flange,robot.gripper_bias)
        for a,b in combinations(frames,2):
            uv,front=project(frames[a],xyz[None]); uv2,front2=project(frames[b],xyz[None])
            manager=TASK_ENV.obs_manager.camera_manager
            camera=manager.cameras[0][manager.camera_names[0].index(a)]
            import torch
            intrinsic=camera.get_intrinsics_matrix()
            probe=torch.as_tensor(xyz[None],dtype=intrinsic.dtype,device=intrinsic.device)
            sdk=camera.get_image_coords_from_world_points(probe).detach().cpu().numpy()
            projection_error=float(np.max(np.abs(np.asarray(sdk)-uv)))
            try: measured=triangulate(frames[a],uv[0],frames[b],uv2[0])
            except ValueError as error:
                checks.append(dict(cameras=[a,b],status='not_visible_or_degenerate',reason=str(error),projection_error_px=projection_error))
            else:
                error=float(np.linalg.norm(np.asarray(measured['point_world_m'])-xyz))
                if error>1e-4 or projection_error>.05: raise RuntimeError('camera algebra check failed')
                checks.append(dict(cameras=[a,b],status='passed',reconstruction_error_m=error,projection_error_px=projection_error))
    (root/'camera-checks.json').write_text(json.dumps(checks,indent=2))
    if not any(c['status']=='passed' for c in checks): raise RuntimeError('no camera pair passed')
    original_episode(TASK_ENV,model_client)
pilot.eval_one_episode=smoke_episode
log = open(root / "simulation.log", "a", buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
(root / "simulator.pid").write_text(str(os.getpid()))
# Both repositories have a top-level utils package. Native simulation must own
# that import; the pilot's rsi/runtime modules above are already loaded.
code_root = str(Path(__file__).resolve().parents[1])
sys.path[:] = [entry for entry in sys.path if entry != code_root]
sys.path.insert(0, "/workspace/RoboDojo")
from src.eval_client import main as native
from src.eval_client import eval_env

class NoPolicyServer:
    def __init__(self, **kwargs): pass
    def call(self, func_name, **kwargs):
        if func_name not in ("reset", "finish_episode"):
            raise RuntimeError("Direct-only trial must not call WAM")
    def close(self): pass

eval_env.WsModelClient = NoPolicyServer
original_create_env = native.create_eval_env
def create_selected_env(*args, **kwargs):
    env = original_create_env(*args, **kwargs)
    layout = int(os.environ.get("RSI_LAYOUT_ID", "0"))
    if layout not in env.seed_manager.seed_info:
        raise ValueError("requested native layout does not exist")
    env.seed_manager.seed_list = [layout]
    env.seed_manager.idx = 0
    env.seed_manager.st_idx = 0
    env.seed_manager.ed_idx = 1
    env.seed_manager._current_batch_seeds = None
    print("Pilot selected native layout", layout, flush=True)
    return env
native.create_eval_env = create_selected_env
sys.path.insert(0, "/rsi/runs/robodojo-step4000-20261022")
from local_material_assets import install, remove
original_close = native.simulation_app.close
def close_with_material_cleanup(*args, **kwargs):
    # Native result/videos are finalized before this normal close. Some Isaac
    # shutdown failures otherwise retain the GPU indefinitely.
    watchdog = threading.Timer(30, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    remove()
    return original_close(*args, **kwargs)
native.simulation_app.close = close_with_material_cleanup

try:
    install()
    native.main()
    if not (root/'episode-summary.json').is_file():
        raise RuntimeError('native evaluator returned without completing smoke episode; inspect simulation.log')
except BaseException as error:
    (root / "launcher-error.json").write_text(json.dumps(dict(type=type(error).__name__, error=str(error))))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
