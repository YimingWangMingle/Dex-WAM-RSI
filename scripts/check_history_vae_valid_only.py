#!/usr/bin/env python3
"""history_encode_valid_only: numeric difference on real data (off vs on).

Real history windows (RobotwinHistoryReader + HistoryAttacher from the launch
config) are encoded both ways.

  --mode encode  Wan2.2 VAE + checkpoint patch_embedding/history_encoder only:
                 valid sparse latents and final history tokens (CPU or GPU).
  --mode policy  Full model, same input and seed, switch off vs on: history
                 tokens, actions, video/depth latents and wall time (GPU).

Example (CPU): CUDA_VISIBLE_DEVICES= python scripts/check_history_vae_valid_only.py \
  --mode encode --device cpu --config <config_at_launch.yaml> --checkpoint <ckpt_dir> --output <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from thinker_wm.data_prepare.history_attach import RobotwinHistoryReader, build_history_attacher
from thinker_wm.models import temporal_training as TT
from thinker_wm.scripts.eval_video_ckpt import load_config


def _stat(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return dict(max_abs=float(d.max()) if d.numel() else 0.0, mean_abs=float(d.mean()) if d.numel() else 0.0,
                ref_rms=float(a.pow(2).mean().sqrt()) if a.numel() else 0.0,
                bitwise_equal=bool(torch.equal(a, b)))


def _windows(cfg, episodes, per_episode):
    """Real batches: per episode, several decisions spread over the episode."""
    attacher = build_history_attacher(cfg)
    reader = attacher.readers["robotwin_3d"]
    h, w = int(cfg.common.video_height), int(cfg.common.video_width)
    ratio, frames = int(cfg.common.video_action_freq_ratio), int(cfg.common.num_video_frames)
    out = []
    for ep in episodes:
        n = int(reader.records[ep]["state"]["count"])
        last = n - 1 - ratio * frames
        for d in np.linspace(0, max(last, 0), per_episode).astype(int).tolist():
            batch = dict(video_frames=torch.zeros(1, frames, 3, h, w), episode_ids=[ep],
                         frame_indices=[list(range(d, d + ratio * frames + 1, ratio))],
                         source_datasets=["robotwin_3d"])
            out.append((ep, d, attacher.attach(batch, step=0), reader))
    return out


def _cat(items):
    keys = [k for k in items[0] if k.startswith("history_")]
    return {k: torch.cat([x[k] for x in items], 0) for k in keys}


def _load_subset(ckpt, prefixes):
    f = Path(ckpt)
    f = f / "pytorch_model_0.bin" if f.is_dir() else f
    sd = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("module", sd.get("state_dict", sd))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    return {k: v.clone() for k, v in sd.items() if k.startswith(prefixes)}


def mode_encode(args, cfg, wins):
    if args.device == "cpu":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _rsi_cpu_stubs  # noqa: F401  # skips wan/__init__ (T5 needs CUDA); VAE stays real
    from thinker_wm.models.history_encoder import HistoryEncoder
    from thinker_wm.wan.modules.vae2_2 import Wan2_2_VAE
    dev = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = Wan2_2_VAE(vae_pth=str(cfg.model.wan.vae_path), device=dev)
    if dev.type == "cpu":
        vae.model = vae.model.float()
    sub = _load_subset(args.checkpoint, ("world_model.video_branch.patch_embedding.",
                                         "world_model.history_encoder."))
    pw = sub["world_model.video_branch.patch_embedding.weight"]
    patch = torch.nn.Conv3d(pw.shape[1], pw.shape[0], tuple(pw.shape[2:]), stride=tuple(pw.shape[2:]))
    patch.load_state_dict({k.split("patch_embedding.")[1]: v for k, v in sub.items() if "patch_embedding." in k})
    he_sd = {k.split("history_encoder.")[1]: v for k, v in sub.items() if "history_encoder." in k}
    enc = HistoryEncoder(pw.shape[0], he_sd["state_proj.weight"].shape[1],
                         freq_dim=he_sd["time_proj.weight"].shape[1],
                         pool=int(cfg.model.wm.history_spatial_pool))
    enc.load_state_dict(he_sd, strict=True)
    patch, enc = patch.to(dev, dtype).eval(), enc.to(dev, dtype).eval()
    raw = _cat([x[2] for x in wins])
    px = raw["history_sparse_rgb"].to(dev)
    v = raw["history_sparse_valid"].to(dev)
    times = raw["history_sparse_times"].to(dev)
    kind = raw["history_sparse_kind"].to(dev)
    res = dict(samples=int(px.shape[0]), slots=int(px.shape[1]), valid_slots=int(v.sum()))
    lat = {}
    for flag in (False, True):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lat[flag] = TT.encode_history_frames(vae.encode, px, v, dtype, valid_only=flag)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        res[f"encode_s_valid_only_{flag}"] = time.perf_counter() - t0
    m = v[:, None, :, None, None].expand_as(lat[False])
    res["valid_latent"] = _stat(lat[False][m], lat[True][m])
    res["invalid_latent_nonzero_new"] = int(lat[True][~m].count_nonzero())
    with torch.no_grad():
        tok = {f: enc(patch, lat[f].to(dtype), times, v, kind=kind) for f in (False, True)}
    hp = tok[False].shape[1] // v.shape[1]
    tv = v.repeat_interleave(hp, dim=1)
    res["history_token_valid"] = _stat(tok[False][tv], tok[True][tv])
    res["history_token_invalid_nonzero"] = [int(tok[f][~tv].count_nonzero()) for f in (False, True)]
    return res


def mode_policy(args, cfg, wins):
    from thinker_wm.scripts.eval_video_ckpt import _load_model_weights, create_eval_model
    from thinker_wm.models.inference_api import GenerationSettings, Observation, WAMInference, make_message_builder
    model = create_eval_model(cfg)
    _load_model_weights(model, Path(args.checkpoint), strict=True)
    model.to(torch.device("cuda")).eval()
    h, w = int(cfg.common.video_height), int(cfg.common.video_width)
    inf = WAMInference(model, checkpoint=str(args.checkpoint), action_stats_version="robotwin_3d",
                       camera_transform=f"robotwin_3d/motus_t/pad/{h}x{w}",
                       settings=GenerationSettings(num_inference_steps=10, action_num_inference_steps=4,
                                                   guidance_scale=5.0, action_guidance_scale=1.0,
                                                   seed=args.seed, return_depth=True, num_blocks=1),
                       message_builder=make_message_builder(cfg, "robotwin_3d"))
    rows = []
    for ep, d, raw, reader in wins:
        rgb = reader.rgb(ep, [d], (h, w))[0]
        obs = Observation(rgb=rgb, state=reader.state(ep, [d])[0].float(), step=d,
                          time_s=d / float(cfg.common.control_hz), robot_name="aloha")
        hist = {k: v for k, v in raw.items() if k.startswith("history_")}
        out = {}
        for flag in (False, True):
            model.config.history_encode_valid_only = flag
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ctx = model._prepare_history({k: v.cuda() for k, v in hist.items()}, ["aloha"])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            po = inf.generate_action(obs, hist, "Complete the task.")
            torch.cuda.synchronize()
            out[flag] = dict(ctx=ctx, po=po, prepare_s=t1 - t0, generate_s=time.perf_counter() - t1)
        model.config.history_encode_valid_only = False
        a, b = out[False], out[True]
        sm = a["ctx"]["valid"][:, :int(raw["history_sparse_valid"].shape[1])]
        lat_m = sm[:, None, :, None, None].expand(-1, a["ctx"]["latent"].shape[1], -1,
                                                  *a["ctx"]["latent"].shape[3:])
        rec = dict(episode=ep, decision=d, sparse_valid=int(raw["history_sparse_valid"].sum()),
                   sparse_latent_valid=_stat(a["ctx"]["latent"][:, :, :sm.shape[1]][lat_m],
                                             b["ctx"]["latent"][:, :, :sm.shape[1]][lat_m]),
                   actions=_stat(torch.from_numpy(a["po"].actions), torch.from_numpy(b["po"].actions)),
                   prepare_s=[a["prepare_s"], b["prepare_s"]], generate_s=[a["generate_s"], b["generate_s"]])
        for name in ("video_latent", "depth_latent"):
            x, y = getattr(a["po"].forecast, name), getattr(b["po"].forecast, name)
            if x is not None and y is not None:
                rec[name] = _stat(x, y)
        print(json.dumps(rec), flush=True)
        rows.append(rec)
    return dict(samples=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("encode", "policy"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--episodes", default="robotwin_3d:017497,robotwin_3d:010611")
    p.add_argument("--per-episode", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    wins = _windows(cfg, args.episodes.split(","), args.per_episode)
    res = dict(mode=args.mode, device=args.device, dtype=args.dtype, checkpoint=args.checkpoint,
               windows=[(ep, d) for ep, d, _, _ in wins], torch=torch.__version__)
    res.update(mode_encode(args, cfg, wins) if args.mode == "encode" else mode_policy(args, cfg, wins))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
