#!/usr/bin/env python3
"""Teacher-force GT video + GT action, generate depth only. Visual check."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from eval_official_holdout_clips import (  # noqa: E402
    _build_depth_attacher,
    _depth_to_luma3,
    _rewrite_vlm,
)
from eval_video_ckpt import create_eval_model, load_config, _load_model_weights  # noqa: E402
from thinker_wm.periodic_eval import render  # noqa: E402

LOG = logging.getLogger("viz_depth_tf")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=8)
    args = p.parse_args()

    config = load_config(args.config)
    clips = json.loads(Path(args.clips).read_text())[: int(args.limit)]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = create_eval_model(config)
    _load_model_weights(model, Path(args.ckpt), strict=True)
    model.to(device).eval()
    attacher = _build_depth_attacher(config)
    inf = config.model.inference
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, clip in enumerate(clips):
        batch = create_diverse_eval_batch(
            config,
            clip["source"],
            batch_size=1,
            shard_ordinal=int(clip["shard"]),
            seed=int(config.periodic_eval.seed),
            use_eval_pool=(clip["split"] == "holdout"),
            episode_id=clip.get("episode_id"),
        )
        batch = _rewrite_vlm(config, dict(batch))
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        present = batch.get("depth_present")
        if batch.get("depth_first") is None or (
            present is not None and not bool(present.reshape(-1)[0].item())
        ):
            LOG.warning("skip no depth %s", clip.get("episode_id"))
            continue
        first = batch["first_frame"].to(device=device, dtype=torch.bfloat16)
        state = batch.get("initial_state")
        if state is not None:
            state = state.to(device=device, dtype=torch.bfloat16)
        depth_first = batch["depth_first"].to(device=device, dtype=torch.bfloat16)
        gt_video = batch["video_frames"].to(device=device, dtype=torch.bfloat16)
        gt_act = batch["action_sequence"].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            pred_v, pred_a, pred_d = model.inference_step(
                first_frame=first,
                state=state,
                num_inference_steps=int(getattr(inf, "num_inference_timesteps", 50)),
                first_frame_pils=batch.get("first_frame_pils"),
                t5_input_messages=batch.get("t5_input_messages"),
                vlm_input_messages=batch.get("vlm_input_messages"),
                robot_names=batch.get("robot_names"),
                guidance_scale=float(getattr(inf, "guidance_scale", 1.0)),
                action_guidance_scale=1.0,
                action_num_inference_steps=int(
                    getattr(inf, "action_num_inference_timesteps", 10) or 10
                ),
                action_smooth_alpha=0.0,
                depth_smooth_alpha=0.0,
                depth_first=depth_first,
                return_depth=True,
                condition_video_frames=gt_video,
                condition_actions=gt_act,
            )
        eid = clip.get("episode_id") or f"clip{i}"
        safe = str(eid).replace(":", "_").replace("/", "_")[-80:]
        sample_dir = out_root / f"{i:02d}_{clip['split']}_{safe}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        gt_tchw = batch["video_frames"][0].detach().cpu().float()
        first_u8 = render.image_to_uint8(batch["first_frame"][0].detach().cpu().float())
        pred_d = pred_d[0].detach().cpu().permute(1, 0, 2, 3).float()
        gt_d = batch["depth_frames"][0].detach().cpu().float()
        n = min(pred_d.shape[0], gt_d.shape[0], gt_tchw.shape[0])
        rgb = render.tensor_to_uint8_frames(gt_tchw[:n])
        pred_du = render.tensor_to_uint8_frames(_depth_to_luma3(pred_d[:n]))
        gt_du = render.tensor_to_uint8_frames(_depth_to_luma3(gt_d[:n]))
        # 左 GT RGB，中 pred depth，右 GT depth
        stacked = np.concatenate([rgb, pred_du, gt_du], axis=2)
        render.write_video(sample_dir / "tf_rgb_preddepth_gtdepth.mp4", stacked, fps=6)
        render.save_png(sample_dir / "first_frame.png", first_u8)
        meta = {
            "episode_id": eid,
            "source": clip["source"],
            "split": clip["split"],
            "teacher_force": "video+action",
            "pred_video_is_vae_recon": pred_v is not None,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        rows.append(meta)
        LOG.info("wrote %s", sample_dir)
    (out_root / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    LOG.info("done n=%d out=%s", len(rows), out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
