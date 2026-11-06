#!/usr/bin/env python3
"""Teacher-force video+action, dump AbsRel by range bin and motion/static."""
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
from eval_official_holdout_clips import _build_depth_attacher, _rewrite_vlm  # noqa: E402
from eval_video_ckpt import _load_model_weights, create_eval_model, load_config  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from thinker_wm.periodic_eval.runner import _depth_motion_mask  # noqa: E402
from viz_motion_mask_pose10 import (  # noqa: E402
    _k_train,
    _load_ego_geom,
    _mask_pose,
    _pose7_to_T,
)

LOG = logging.getLogger("depth_tf_bins")
BINS = [(0.15, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]


def _mean(rel, mask):
    if mask is None or int(mask.sum()) == 0:
        return None, 0
    return float(rel[mask].mean().item()), int(mask.sum())


def _rmse(diff, mask):
    if mask is None or int(mask.sum()) == 0:
        return None
    return float((diff[mask] ** 2).mean().sqrt().item())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--no-tf", action="store_true", help="joint-gen only (no GT video/action)")
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
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))

    rows = []
    for clip in clips:
        batch = create_diverse_eval_batch(
            config, clip["source"], batch_size=1,
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
            continue
        first = batch["first_frame"].to(device=device, dtype=torch.bfloat16)
        state = batch.get("initial_state")
        if state is not None:
            state = state.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            _, _, pred_d = model.inference_step(
                first_frame=first,
                state=state,
                num_inference_steps=int(getattr(inf, "num_inference_timesteps", 50)),
                first_frame_pils=batch.get("first_frame_pils"),
                t5_input_messages=batch.get("t5_input_messages"),
                vlm_input_messages=batch.get("vlm_input_messages"),
                robot_names=batch.get("robot_names"),
                guidance_scale=float(getattr(inf, "guidance_scale", 1.0)),
                action_guidance_scale=1.0,
                action_num_inference_steps=int(getattr(inf, "action_num_inference_timesteps", 10) or 10),
                depth_first=batch["depth_first"].to(device=device, dtype=torch.bfloat16),
                return_depth=True,
                **({} if args.no_tf else dict(
                    condition_video_frames=batch["video_frames"].to(device=device, dtype=torch.bfloat16),
                    condition_actions=batch["action_sequence"].to(device=device, dtype=torch.bfloat16),
                )),
            )
        pred = pred_d[0].detach().cpu().permute(1, 0, 2, 3).float()
        gt = batch["depth_frames"][0].detach().cpu().float()
        rgb = batch["video_frames"][0].detach().cpu().float()
        n = min(pred.shape[0], gt.shape[0], rgb.shape[0])
        pred, gt, rgb = pred[:n], gt[:n], rgb[:n]
        pred_m, _ = decode_to_depth_m(pred, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
        gt_m, valid = decode_to_depth_m(gt, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
        rel = (pred_m - gt_m).abs() / gt_m.clamp_min(1e-6)
        bias = pred_m - gt_m
        motion = _depth_motion_mask(gt_m, valid, thresh_m=0.10, rgb=rgb)
        mot = valid & motion
        bg = valid & ~motion
        rec = {
            "episode_id": clip.get("episode_id"),
            "source": clip["source"],
            "family": "ego" if str(clip["source"]).startswith("egoscale") else "agi",
            "split": clip.get("split"),
        }
        rec["absrel"], rec["n"] = _mean(rel, valid)
        rec["absrel_motion"], rec["n_motion"] = _mean(rel, mot)
        rec["absrel_bg"], rec["n_bg"] = _mean(rel, bg)
        rec["bias_m"], _ = _mean(bias, valid)
        rec["rmse_m"] = _rmse(bias, valid)
        rec["rmse_motion"] = _rmse(bias, mot)
        rec["rmse_bg"] = _rmse(bias, bg)
        rec["motion_frac"] = float(mot.sum() / valid.sum().clamp_min(1))

        # Ego: same three gates, but frame pairs aligned by head/pose first.
        rec["pose_mask"] = False
        if str(clip["source"]).startswith("egoscale"):
            raw_fi = batch["frame_indices"][0]
            fi = [int(x) for x in list(raw_fi)]
            # video/depth_frames omit the first (condition) frame
            if len(fi) == n + 1:
                fi = fi[1:]
            fi = fi[:n]
            try:
                raw, k_left, _, _ = _load_ego_geom(
                    config, clip["source"], clip["episode_id"], fi
                )
                k_train, _ = _k_train(k_left)
                head = raw["head/pose"].numpy()
                head_ok = raw["head/valid"].numpy().reshape(-1).astype(bool)
                Ts = np.stack([_pose7_to_T(pp) for pp in head], 0)
                pmask = _mask_pose(
                    gt_m.numpy(), valid.numpy(), rgb.mean(dim=1).numpy(), Ts, head_ok, k_train
                )
                pmot = valid & torch.from_numpy(pmask)
                pbg = valid & ~torch.from_numpy(pmask)
                rec["pose_mask"] = True
                rec["absrel_motion_pose"], rec["n_motion_pose"] = _mean(rel, pmot)
                rec["absrel_bg_pose"], rec["n_bg_pose"] = _mean(rel, pbg)
                rec["rmse_motion_pose"] = _rmse(bias, pmot)
                rec["rmse_bg_pose"] = _rmse(bias, pbg)
                rec["motion_frac_pose"] = float(pmot.sum() / valid.sum().clamp_min(1))
            except Exception as error:  # geometry missing -> report base only
                LOG.warning("pose mask failed for %s: %s", clip.get("episode_id"), error)
        rec["bins"] = {}
        for lo, hi in BINS:
            mb = valid & (gt_m >= lo) & (gt_m < hi)
            a, c = _mean(rel, mb)
            rec["bins"][f"{lo}-{hi}"] = {
                "absrel": a, "n": c,
                "frac": float(mb.sum() / valid.sum().clamp_min(1)),
                "rmse_m": _rmse(bias, mb),
                "bias_m": _mean(bias, mb)[0],
            }
        rec["cross"] = {}
        for name, m in [
            ("near<1_motion", valid & mot & (gt_m < 1.0)),
            ("near<1_static", valid & bg & (gt_m < 1.0)),
            ("far>1.5_motion", valid & mot & (gt_m > 1.5)),
            ("far>1.5_static", valid & bg & (gt_m > 1.5)),
        ]:
            a, c = _mean(rel, m)
            rec["cross"][name] = {"absrel": a, "n": c, "rmse_m": _rmse(bias, m)}
        rows.append(rec)
        LOG.info(
            "%s ov=%.3f mot=%.3f bg=%.3f f=%.3f | pose mot=%s bg=%s f=%s",
            rec["episode_id"], rec["absrel"], rec["absrel_motion"] or -1,
            rec["absrel_bg"] or -1, rec["motion_frac"],
            f"{rec['absrel_motion_pose']:.3f}" if rec.get("absrel_motion_pose") else "-",
            f"{rec['absrel_bg_pose']:.3f}" if rec.get("absrel_bg_pose") else "-",
            f"{rec['motion_frac_pose']:.3f}" if rec.get("motion_frac_pose") else "-",
        )

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    LOG.info("wrote %s n=%d", args.out, len(rows))
    _log_summary(rows)
    return 0


def _wmean(rows, key, nkey):
    num = den = 0.0
    for r in rows:
        v, n = r.get(key), r.get(nkey)
        if v is None or not n:
            continue
        num += float(v) * int(n)
        den += int(n)
    return (num / den) if den else None


def _log_summary(rows: list) -> None:
    for fam in ("agi", "ego", "all"):
        sub = rows if fam == "all" else [r for r in rows if r.get("family") == fam]
        if not sub:
            continue
        ov = _wmean(sub, "absrel", "n")
        mot = _wmean(sub, "absrel_motion", "n_motion")
        bg = _wmean(sub, "absrel_bg", "n_bg")
        f = _wmean(sub, "motion_frac", "n")
        pmot = _wmean(sub, "absrel_motion_pose", "n_motion_pose")
        pbg = _wmean(sub, "absrel_bg_pose", "n_bg_pose")
        pf = _wmean(sub, "motion_frac_pose", "n")
        LOG.info(
            "SUM %s n=%d  ov=%.3f  base mot/bg/f=%.3f/%.3f/%.3f  "
            "pose mot/bg/f=%s/%s/%s",
            fam,
            len(sub),
            ov or -1,
            mot or -1,
            bg or -1,
            f or -1,
            f"{pmot:.3f}" if pmot is not None else "-",
            f"{pbg:.3f}" if pbg is not None else "-",
            f"{pf:.3f}" if pf is not None else "-",
        )


if __name__ == "__main__":
    raise SystemExit(main())
