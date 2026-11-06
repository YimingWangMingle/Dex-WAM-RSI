"""Opt-in native launcher for the frozen matrix protocol; never run on active trials.

Required environment: RSI_EXPERIMENT_GROUP, RSI_LIVE_RUN_DIR (new directory),
RSI_LAYOUT_ID, ROBODOJO_ROOT, RSI_MATERIAL_ADAPTER_DIR, and RSI_PI05_URL for PI0.5.
Pass native RoboDojo task/env/device/policy/host/port arguments unchanged.
"""
import json
import os
from pathlib import Path
import sys
import threading
import time

REPO_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT))
from environments.robodojo.experiments.layout3 import bridge_episode, group_environment
root=Path(os.environ['RSI_LIVE_RUN_DIR']).resolve()
if root.exists():raise FileExistsError('refuse existing experiment output: '+str(root))
os.environ.update(group_environment(os.environ['RSI_EXPERIMENT_GROUP'],root,
    int(os.environ.get('RSI_EPISODE_STEPS','800'))))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',EVAL_NUM='1',
    ROBODOJO_RUN_ID='pi05-high-0-4-'+os.environ['RSI_EXPERIMENT_GROUP']+'-layout-'+os.environ['RSI_LAYOUT_ID'],
    OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',OMNI_KIT_ACCEPT_EULA='YES')
root.mkdir(parents=True)
(root/'tmp').mkdir()
os.environ['TMPDIR']=str(root/'tmp')
os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
import environments.robodojo.runtime  # Pin platform namespace before simulator path changes.
root = Path(os.environ["RSI_LIVE_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)

print('RSI PI0.5 paired evaluation: Astra high, max 800 steps',flush=True)
log = open(root / "simulation.log", "a", buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
(root / "simulator.pid").write_text(str(os.getpid()))
# Both repositories have a top-level utils package. Native simulation must own
# that import; the pilot's rsi/runtime modules above are already loaded.
code_root = str(REPO_ROOT)
sys.path[:] = [entry for entry in sys.path if entry not in (code_root,)]
for name, module in list(sys.modules.items()):
    if (name == 'utils' or name.startswith('thinker_wm.utils.')) and '/ThinkerWAM-RSI' in str(getattr(module, '__file__', '')):
        del sys.modules[name]
sys.path.insert(0, os.environ["ROBODOJO_ROOT"])
from src.eval_client import main as native
from src.eval_client import eval_env

import importlib
policy_module=importlib.import_module('XPolicyLab.policy.ThinkerWAM_RSI_Joint24_Step10000.deploy')
policy_module.eval_one_episode=bridge_episode
class PolicyInfrastructureError(BaseException): pass
original_client=eval_env.WsModelClient
class AuditedClient(original_client):
    def __init__(self, **kwargs):
        kwargs['request_timeout_s']=1800.0
        super().__init__(**kwargs)
    def call(self,*args,**kwargs):
        try: return super().call(*args,**kwargs)
        except Exception as error:
            (root/'rpc-error.json').write_text(json.dumps({'time':time.time(),'method':kwargs.get('func_name'),'error':repr(error)}))
            raise PolicyInfrastructureError(str(error)) from error
eval_env.WsModelClient=AuditedClient
if os.environ['RSI_MATRIX_MODE'] in ('pi05_baseline','pi05_action1','pi05_action3','direct_memory','eef_file'):
    class NoWAM:
        def __init__(self,**kwargs): pass
        def call(self,func_name,**kwargs):
            if func_name not in ('reset','finish_episode'):raise RuntimeError('Unexpected WAM call')
        def close(self):pass
    eval_env.WsModelClient=NoWAM
original_create_env = native.create_eval_env
def create_selected_env(*args, **kwargs):
    env = original_create_env(*args, **kwargs)
    import numpy as np
    official_limit=int(np.asarray(env.step_lim).max())
    if official_limit!=800:raise ValueError('Unexpected official task limit')
    (root/'official-step-limit.json').write_text(json.dumps({'step_limit':official_limit}))
    layout = int(os.environ["RSI_LAYOUT_ID"])
    if layout not in env.seed_manager.seed_info:
        raise ValueError("requested native layout does not exist")
    env.seed_manager.seed_list = [layout]
    env.seed_manager.idx = 0
    env.seed_manager.st_idx = 0
    env.seed_manager.ed_idx = 1
    env.seed_manager._current_batch_seeds = None
    print("Pilot selected native layout", layout, flush=True)
    save_dir = root/'results'
    env.save_dir = str(save_dir)
    env._stream_dir = str(save_dir/'_stream')
    env.obs_config['save_dir'] = str(save_dir)
    return env
native.create_eval_env = create_selected_env
sys.path.insert(0, os.environ["RSI_MATERIAL_ADAPTER_DIR"])
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
        raise RuntimeError('native evaluator returned without completing episode; inspect simulation.log')
except BaseException as error:
    (root / "launcher-error.json").write_text(json.dumps(dict(type=type(error).__name__, error=str(error))))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
