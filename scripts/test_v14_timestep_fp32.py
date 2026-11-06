#!/usr/bin/env python3
"""Execute production timestep code on CPU without loading VLM/VAE weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "third_party/Thinker-WM/models/ubt_wm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UBT_WM")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_flow_timestep")
    scope = {"torch": torch}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "production_sampler", "exec"), scope)
    sigma = torch.tensor([0.999, 0.7773, 0.1234567])
    scheduler = SimpleNamespace(num_train_timesteps=3, sigmas=sigma, timesteps=sigma*1000)
    model = SimpleNamespace(dtype=torch.bfloat16, device="cpu", fm_train_scheduler=scheduler)
    for explicit in (False, True):
        torch.manual_seed(19)
        t, sv, sa = scope["_sample_flow_timestep"](model, 64, scheduler if explicit else None)
        assert t.dtype == sa.dtype == torch.float32
        assert sv.dtype == torch.bfloat16
        torch.testing.assert_close(t, sa.reshape(64, 1)*1000, rtol=0, atol=0)
        assert (t != t.bfloat16().float()).any()
    count = 0
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in ("t_video", "t_action", "t_scaled") and "1000" in ast.unparse(n.value):
                env = {"torch": torch, "B": 6, "sigma": sigma[0], "sigma_action": sigma[1], "self": model}
                exec(compile(ast.Module(body=[n], type_ignores=[]), "production_inference_label", "exec"), env)
                expected = sigma[1] if name == "t_action" else sigma[0]
                assert env[name].dtype == torch.float32
                torch.testing.assert_close(env[name], (expected*1000).expand(6, 1), rtol=0, atol=0)
                count += 1
    assert count == 3, count
    print("PASS training shared/independent scheduler and all 3 inference labels: FP32")


if __name__ == "__main__":
    main()
