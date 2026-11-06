#!/usr/bin/env python3
"""Encode family-holdout 18 clips into RynnWorld-4D inference latents.

Reloads the same batches as eval (source + shard + seed). Resamples to
Stage-3 train size 25x480x832, encodes RGB/depth/white-flow with the
Diffusers Wan VAE, and T5-embeds the task prompt.

  CUDA_VISIBLE_DEVICES=0 python scripts/encode_rynnworld_holdout18.py
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import ftfy
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import save_file
from transformers import T5TokenizerFast, UMT5EncoderModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(REPO / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import _build_depth_attacher  # noqa: E402
from eval_video_ckpt import load_config  # noqa: E402
from publish_eval_to_monitor import _meta_first  # noqa: E402

META_ROOT = Path(
    "/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs"
    "/depth_ablation_v5_step20k_motion_baseline/eval/step_0020000"
)
CFG = REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v5.yaml"
MODEL = Path("/code1/zxw_data/checkpoint/Wan-2.2")
OUT = REPO / "tmp/rynnworld4d_holdout18"
NUM_FRAMES, HEIGHT, WIDTH = 25, 480, 832
MAX_TEXT = 226


def _prompt_of(batch) -> str:
    text = _meta_first(batch, "t5_input_messages")
    if text is None:
        text = _meta_first(batch, "task_text")
    return str(text or "").strip()


def _clean(text: str) -> str:
    text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
    return re.sub(r"\s+", " ", text).strip()


def _resample_time(x: torch.Tensor, n: int) -> torch.Tensor:
    t = int(x.shape[0])
    if t == n:
        return x
    idx = torch.linspace(0, t - 1, n).round().long()
    return x[idx]


def _pad_width(x: torch.Tensor, width: int, fill: float) -> torch.Tensor:
    # x: [T,C,H,W]
    w = int(x.shape[-1])
    if w == width:
        return x
    if w > width:
        extra = w - width
        left = extra // 2
        return x[..., left : left + width]
    pad = width - w
    left = pad // 2
    right = pad - left
    return F.pad(x, (left, right, 0, 0), value=fill)


def _to_vae(frames: torch.Tensor) -> torch.Tensor:
    # [T,3,H,W] in [0,1] -> [1,3,T,H,W] in [-1,1]
    return (frames * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@torch.no_grad()
def _encode_video(vae, frames: torch.Tensor) -> torch.Tensor:
    video = _to_vae(frames).to(device=vae.device, dtype=vae.dtype)
    lat = vae.encode(video).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
    return ((lat - mean) / std).squeeze(0).float().cpu().contiguous()


@torch.no_grad()
def _encode_text(tokenizer, encoder, prompt: str) -> torch.Tensor:
    prompt = _clean(prompt) or "A robot performs a manipulation task."
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=MAX_TEXT,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    hidden = encoder(inputs.input_ids.to(device), inputs.attention_mask.to(device)).last_hidden_state
    seq = int(inputs.attention_mask.gt(0).sum().item())
    hidden = hidden[0, :seq].to(dtype)
    if hidden.shape[0] < MAX_TEXT:
        hidden = torch.cat([hidden, hidden.new_zeros(MAX_TEXT - hidden.shape[0], hidden.shape[1])], dim=0)
    return hidden.unsqueeze(0).float().cpu().contiguous()


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metas = sorted(META_ROOT.glob("rank*/sample_*/meta.json"))
    if len(metas) != 18:
        print(f"expected 18 metas, found {len(metas)} under {META_ROOT}")
        return 1

    config = load_config(str(CFG))
    OmegaConf.set_struct(config, False)
    attacher = _build_depth_attacher(config)
    seed = int(config.periodic_eval.seed)

    from diffusers import AutoencoderKLWan

    print(f"loading VAE/T5 from {MODEL} on {device}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(str(MODEL), subfolder="vae").to(device)
    vae.eval()
    tokenizer = T5TokenizerFast.from_pretrained(str(MODEL), subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(str(MODEL), subfolder="text_encoder").to(device)
    text_encoder.eval()

    lat_dir = OUT / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    items = []
    catalog = []

    for i, path in enumerate(metas):
        payload = json.loads(path.read_text())
        source = str(payload["source"])
        split = str(payload["split"])
        shard = int(payload["shard_ordinal"])
        print(f"[{i+1}/18] {source} {split} shard={shard}", flush=True)
        want_eid = str(payload.get("episode_id") or "")
        batch = create_diverse_eval_batch(
            config,
            source,
            batch_size=1,
            shard_ordinal=shard,
            seed=seed,
            use_eval_pool=(split == "holdout"),
            episode_id=want_eid or None,
        )
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        eid = str(_meta_first(batch, "episode_ids") or payload.get("episode_id") or "")
        if want_eid and eid != want_eid:
            raise RuntimeError(f"episode mismatch: want {want_eid} got {eid}")
        prompt = _prompt_of(batch)
        rgb = batch["video_frames"][0].float().clamp(0, 1)
        depth = batch["depth_frames"][0].float().clamp(0, 1)
        n = min(rgb.shape[0], depth.shape[0])
        rgb, depth = rgb[:n], depth[:n]
        rgb = _pad_width(_resample_time(rgb, NUM_FRAMES), WIDTH, 0.0)
        depth = _pad_width(_resample_time(depth, NUM_FRAMES), WIDTH, 1.0)
        if rgb.shape[-2] != HEIGHT:
            rgb = F.interpolate(rgb, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
            depth = F.interpolate(depth, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
        white = torch.ones_like(rgb)

        video_latents = _encode_video(vae, rgb)
        depth_latents = _encode_video(vae, depth)
        flow_latents = _encode_video(vae, white)
        text_embeds = _encode_text(tokenizer, text_encoder, prompt)
        print(
            f"  eid={eid} rgb={tuple(rgb.shape)} lat={tuple(video_latents.shape)} text={tuple(text_embeds.shape)}",
            flush=True,
        )

        stem = f"{i:02d}_{source}_{split}_{shard}"
        rgb_p = lat_dir / f"{stem}_rgb.safetensors"
        fd_p = lat_dir / f"{stem}_flow_depth.safetensors"
        save_file({"video_latents": video_latents, "text_embeds": text_embeds}, str(rgb_p))
        save_file({"depth_latents": depth_latents, "flow_latents": flow_latents}, str(fd_p))
        preview = (rgb[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        np.save(lat_dir / f"{stem}_gt_rgb.npy", (rgb.numpy() * 255).clip(0, 255).astype(np.uint8))
        np.save(lat_dir / f"{stem}_gt_depth.npy", (depth.numpy() * 255).clip(0, 255).astype(np.uint8))
        items.append(
            {
                "rgb_latents": str(rgb_p),
                "flow_depth_latents": str(fd_p),
                "prompt": prompt,
            }
        )
        catalog.append(
            {
                "index": i,
                "source": source,
                "split": split,
                "shard": shard,
                "episode_id": eid,
                "prompt": prompt,
                "latent_shape": list(video_latents.shape),
                "preview_hw": list(preview.shape),
            }
        )

    json_path = OUT / "holdout18.json"
    json_path.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"wrote {json_path} and {len(items)} latent pairs under {lat_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
