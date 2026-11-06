"""Capture initial native stack_bowls RGB/calibration; never invoke a policy.

Run with the native evaluator's arguments (see the handoff command). This uses
the same launcher, layout selection and verified material aliases as the live
IK launcher, but replaces the environment's episode callback before reset.
Native initialization/settling still runs; this callback issues zero actions.

Output: cam_head.png, cam_left_wrist.png, cam_right_wrist.png, cameras.json,
capture-summary.json and simulation.log. cameras.json is a camera-name mapping,
not a wrapper object. time_s follows LocalGeometry's episode-relative clock:
the initial, single rendered observation is time zero, not a wall-clock stamp.
Capture success says nothing about bowl detection, geometry or task success.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from types import MethodType


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def capture_episode(env: object, root: Path, layout: int) -> None:
    """Read one RGB snapshot and its optical calibration, with no task poses."""
    from PIL import Image
    from environments.robodojo.runtime.robodojo_local_geometry import LocalGeometry

    if [int(value) for value in env.env_seeds] != [layout]:
        raise RuntimeError("native episode did not select the requested layout")
    if (root / "cameras.json").exists():
        raise RuntimeError("fixture already captured; refusing a second episode")
    # Isolated renderer qualification only: public calibration markers, never
    # task-object ground truth. This diagnostic is not a native benchmark run.
    import numpy as np
    from pxr import UsdGeom, UsdShade, Sdf, Gf
    from isaacsim.core.utils.stage import get_current_stage
    stage=get_current_stage()
    probes=[[-.2,-.05,.90],[0,0,.85],[.18,.1,.90],[0,.2,1.0],[.15,-.05,1.05],[-.15,.15,1.05]]
    colors=[[1,0,1],[0,1,0],[0,1,1],[1,0,0],[0,0,1],[1,1,0]]
    for i,(xyz,color) in enumerate(zip(probes,colors)):
        path=f'/World/RSICalibrationOnly/marker_{i}'
        sphere=UsdGeom.Sphere.Define(stage,path)
        sphere.GetRadiusAttr().Set(.006)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        material=UsdShade.Material.Define(stage,path+'/material')
        shader=UsdShade.Shader.Define(stage,path+'/material/shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    for _ in range(8): env.render()
    obs = env.get_obs()
    (root/'calibration-markers.json').write_text(json.dumps(dict(points_world_m=probes,colors=colors,
        scope='Isolated renderer calibration fixture, not task observations or benchmark')))

    geometry = LocalGeometry(env)
    # Resolve every CameraFrame before writing any image. Missing/invalid
    # calibration is an error, never an identity matrix or estimated fallback.
    frames = {name: geometry.camera(name, obs, 0) for name in CAMERAS}
    metadata = {}
    for name, frame in frames.items():
        Image.fromarray(frame.rgb).save(root / (name + ".png"))
        metadata[name] = {
            "K": frame.K.tolist(),
            "T_world_camera": frame.T_world_camera.tolist(),
            "time_s": frame.time_s,
            "calibration_revision": frame.calibration_revision,
            "rectified_native": frame.rectified_native,
            "calibration_image_hw": list(frame.calibration_image_hw),
        }
    # Publish metadata only when all three PNGs have been written successfully.
    with (root / "cameras.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    stage.RemovePrim('/World/RSICalibrationOnly')
    # End explicitly without calling the reward-based is_episode_end(). Native
    # evaluator bookkeeping may still run; no scores enter this fixture.
    env.success[0] = False
    env.end_flag[0] = True
    summary = {
        "status": "captured",
        "task": "stack_bowls",
        "layout_id": layout,
        "control_step": 0,
        "executed_actions": 0,
        "policy_calls": 0,
        "task_success": False,
        "perception_status": "unknown",
        "cameras": list(CAMERAS),
        "world_frame": "native_simulation_world",
        "length_unit": "m",
        "camera_convention": "ROS optical +Z forward; T_world_camera",
        "time_basis": "episode_relative_initial_snapshot; not sensor wall clock",
        "source": "native RGB and LocalGeometry.camera optical calibration",
    }
    with (root / "capture-summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Captured initial calibrated RGB fixture:", root, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New fixture directory; existing directories are refused")
    parser.add_argument("--layout-id", type=int, default=1)
    args, native_args = parser.parse_known_args()
    # Validate the relevant native options before importing AppLauncher.
    contract = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    contract.add_argument("--task_name", required=True, choices=("stack_bowls",))
    contract.add_argument("--num_envs", type=int, default=1, choices=(1,))
    contract.add_argument("--policy_name", required=True, choices=("ThinkerDeepSeek_IK",))
    contract.parse_known_args(native_args)
    if args.layout_id < 0:
        parser.error("--layout-id must be nonnegative")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    print("Capture log:", root / "simulation.log", flush=True)
    log = (root / "simulation.log").open("a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    (root / "simulator.pid").write_text(str(os.getpid()) + "\n")
    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    # Preload only geometry; no DeepSeek client, credential or pilot import.
    from environments.robodojo.runtime.robodojo_local_geometry import LocalGeometry  # noqa: F401

    # Native utils must win over the training repository's top-level utils.
    sys.path[:] = [entry for entry in sys.path
                   if entry and Path(entry).resolve() != code_root]
    sys.path.insert(0, "/workspace/RoboDojo")
    sys.argv = [sys.argv[0], *native_args]
    os.environ["EVAL_NUM"] = "1"
    from src.eval_client import main as native
    from src.eval_client import eval_env

    class NoPolicyServer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def call(self, func_name: str, **kwargs: object) -> None:
            if func_name not in ("reset", "finish_episode"):
                raise RuntimeError("capture must not call a policy: " + func_name)

        def close(self) -> None:
            pass

    eval_env.WsModelClient = NoPolicyServer
    original_create_env = native.create_eval_env

    def create_selected_env(*positional: object, **keywords: object) -> object:
        env = original_create_env(*positional, **keywords)
        if env.eval_batch:
            raise RuntimeError("capture requires native eval_batch=false")
        if args.layout_id not in env.seed_manager.seed_info:
            raise ValueError("requested native layout does not exist")
        env.seed_manager.seed_list = [args.layout_id]
        env.seed_manager.idx = 0
        env.seed_manager.st_idx = 0
        env.seed_manager.ed_idx = 1
        env.seed_manager._current_batch_seeds = None

        def capture(self: object) -> None:
            capture_episode(self, root, args.layout_id)

        # Replace the native method itself so deploy.py and the pilot are never
        # imported, even though the native policy/config name stays unchanged.
        env.eval_one_episode = MethodType(capture, env)
        return env

    native.create_eval_env = create_selected_env
    sys.path.insert(0, "/rsi/runs/robodojo-step4000-20261022")
    from local_material_assets import install, remove

    original_close = native.simulation_app.close
    closing = False

    def close_with_material_cleanup(*positional: object, **keywords: object) -> object:
        nonlocal closing
        if closing:
            return None
        closing = True
        # Bound the same Isaac shutdown hang handled by the existing launcher;
        # unlike that launcher, a shutdown timeout remains a nonzero exit.
        watchdog = threading.Timer(30, lambda: os._exit(1))
        watchdog.daemon = True
        watchdog.start()
        try:
            remove()
        finally:
            result = original_close(*positional, **keywords)
        watchdog.cancel()
        return result

    native.simulation_app.close = close_with_material_cleanup
    try:
        install()
        native.main()
        if not (root / "capture-summary.json").is_file():
            raise RuntimeError("native evaluator returned without a complete fixture; inspect simulation.log")
        return 0
    except BaseException as error:
        (root / "launcher-error.json").write_text(json.dumps({
            "type": type(error).__name__, "error": str(error),
        }, indent=2) + "\n")
        traceback.print_exc()
        return 1
    finally:
        close_with_material_cleanup()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    # Avoid Isaac atexit hangs after the explicit native close, as in the live
    # launcher. Importing this module alone never initializes the simulator.
    os._exit(main())
