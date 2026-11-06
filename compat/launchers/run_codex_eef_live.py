"""Run the native task with the current Codex conversation as EEF controller."""
import getpass
import json
import os
from pathlib import Path
import socket
import sys
import time
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from environments.robodojo.runtime.robodojo_codex_bridge import eval_one_episode as bridge_episode
from environments.robodojo.runtime import robodojo_deepseek_ik as pilot

root = Path(os.environ["RSI_LIVE_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)
pilot.eval_one_episode=bridge_episode
provider=os.environ.get('RSI_LLM_PROVIDER','file')
print('Unified controller provider: '+provider+'. Logs: '+str(root/'simulation.log'),flush=True)
log = open(root / "simulation.log", "a", buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
(root / "simulator.pid").write_text(str(os.getpid()))
# Both repositories have a top-level utils package. Native simulation must own
# that import; the pilot's rsi/runtime modules above are already loaded.
code_root = str(Path(__file__).resolve().parents[2])
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

if os.environ.get('RSI_CONTROL_MODE','direct') == 'direct':
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
        raise RuntimeError('native evaluator returned without completing episode; inspect simulation.log')
except BaseException as error:
    (root / "launcher-error.json").write_text(json.dumps(dict(type=type(error).__name__, error=str(error))))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
