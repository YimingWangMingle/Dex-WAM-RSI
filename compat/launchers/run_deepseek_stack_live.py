"""Run one native stack_bowls episode; read credential from terminal only."""
import getpass
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.deepseek import DeepSeekClient
from environments.robodojo.runtime import robodojo_deepseek_live as pilot

root = Path(os.environ["RSI_LIVE_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)
pilot.CLIENT = DeepSeekClient(getpass.getpass("DeepSeek credential (memory only): "), timeout=90)
pilot.ROOT = root
pilot.EXPERIENCE = Path(os.environ["RSI_TASK_EXPERIENCE"]).read_text()[:12000]
print("Credential received; simulator output goes to " + str(root / "simulation.log"), flush=True)
log = open(root / "simulation.log", "a", buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
(root / "simulator.pid").write_text(str(os.getpid()))
print("Waiting for the step2000 WAM service", flush=True)
receipt = Path("/rsi/runs/stack-bowls-deepseek-live-20261022/policy/strict-load.json")
deadline = time.monotonic() + 7200
while time.monotonic() < deadline:
    if receipt.exists():
        info = json.loads(receipt.read_text())
        if info["checkpoint"]["step"] != 2000 or not info["strict_load"]["strict"]:
            raise RuntimeError("wrong or non-strict WAM checkpoint")
        if info["strict_load"]["missing_keys"] or info["strict_load"]["unexpected_keys"]:
            raise RuntimeError("incomplete WAM weights")
        try:
            with socket.create_connection(("thinkerwam-rsi-wam", 19000), timeout=2):
                pass
            break
        except OSError:
            pass
    time.sleep(5)
else:
    raise TimeoutError("WAM server did not become ready")

# Both repositories have a top-level utils package. Native simulation must own
# that import; the pilot's rsi/runtime modules above are already loaded.
code_root = str(Path(__file__).resolve().parents[2])
sys.path[:] = [entry for entry in sys.path if entry != code_root]
sys.path.insert(0, "/workspace/RoboDojo")
from src.eval_client import main as native
from src.eval_client import eval_env

original_client = eval_env.WsModelClient
class AuditedClient(original_client):
    def __init__(self, **kwargs):
        kwargs.update(request_timeout_s=600, max_connect_attempts=2)
        super().__init__(**kwargs)

eval_env.WsModelClient = AuditedClient
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
    remove()
    return original_close(*args, **kwargs)
native.simulation_app.close = close_with_material_cleanup

try:
    install()
    native.main()
except BaseException as error:
    (root / "launcher-error.json").write_text(json.dumps(dict(type=type(error).__name__, error=str(error))))
    raise
finally:
    if native.simulation_app.is_running():
        native.simulation_app.close()
