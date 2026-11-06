#!/usr/bin/env python3
"""Verify EgoScale head/pose == base_0 camera, then render 10 comparison videos.

Panels: RGB | base | pose+cc
Agi has no camera pose — pose panel is a copy of base (not computed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
sys.path.insert(0, str(REPO / "scripts"))

from thinker_wm.data_prepare import create_diverse_eval_batch  # noqa: E402
from thinker_wm.data_prepare.factory import _indexed_sources  # noqa: E402
from eval_official_holdout_clips import _build_depth_attacher  # noqa: E402
from eval_video_ckpt import load_config  # noqa: E402
from thinker_wm.models.depth_codec import decode_to_depth_m  # noqa: E402
from motus_data.indexed_pipeline import open_dataset  # noqa: E402
from motus_data.releases import get_episode_dataset_release  # noqa: E402
from thinker_wm.periodic_eval.runner import _depth_motion_mask  # noqa: E402
from exp_ego_motion_mask import RGB_THRESH, THRESH_M, _drop_small, _residual_gate  # noqa: E402

CAT = Path(
    "/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000"
    "/motion_mask_30seq/catalog.json"
)
CFG = REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v6_1.yaml"
OUT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/depth_tf_v61_3000/motion_mask_pose10")
MASK_DIR = REPO / "mask" / "pose"
PICK = [0, 5, 7, 18, 4, 8, 10, 12, 13, 20]
MIN_AREA = 400
TRAIN_H, TRAIN_W = 480, 640
NATIVE_H, NATIVE_W = 1080, 1920


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose7_to_T(p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_xyzw_to_R(p[3:7])
    T[:3, 3] = np.asarray(p[:3], dtype=np.float64)
    return T


def _inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _k_train(k_native: np.ndarray) -> np.ndarray:
    scale = min(TRAIN_H / NATIVE_H, TRAIN_W / NATIVE_W)
    new_h, new_w = int(NATIVE_H * scale), int(NATIVE_W * scale)
    y0, x0 = (TRAIN_H - new_h) // 2, (TRAIN_W - new_w) // 2
    K = k_native.astype(np.float64).copy()
    K[0, :] *= new_w / NATIVE_W
    K[1, :] *= new_h / NATIVE_H
    K[0, 2] += x0
    K[1, 2] += y0
    return K, (y0, new_h, x0, new_w)


def _uv_native_to_train(uv: np.ndarray, pad_box) -> np.ndarray:
    y0, nh, x0, nw = pad_box
    out = np.asarray(uv, dtype=np.float64).copy()
    out[..., 0] = out[..., 0] * (nw / NATIVE_W) + x0
    out[..., 1] = out[..., 1] * (nh / NATIVE_H) + y0
    return out


def _project(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * xyz[..., 0] / z + K[0, 2]
    v = K[1, 1] * xyz[..., 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def _warp_by_T(
    z: np.ndarray, valid: np.ndarray, luma: np.ndarray, T_rel: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = z.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    Z = z.astype(np.float64)
    P = np.stack([(us - cx) * Z / fx, (vs - cy) * Z / fy, Z], axis=-1)
    R, t = T_rel[:3, :3], T_rel[:3, 3]
    P2 = P @ R.T + t
    Z2 = P2[..., 2]
    u2 = fx * P2[..., 0] / np.clip(Z2, 1e-6, None) + cx
    v2 = fy * P2[..., 1] / np.clip(Z2, 1e-6, None) + cy
    m = valid & (Z > 1e-4) & (Z2 > 1e-4)
    uu = np.rint(u2[m]).astype(np.int32)
    vv = np.rint(v2[m]).astype(np.int32)
    zz = Z2[m].astype(np.float32)
    lu = luma[m].astype(np.float32)
    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    uu, vv, zz, lu = uu[ok], vv[ok], zz[ok], lu[ok]
    out_z = np.full((h, w), np.inf, np.float32)
    out_l = np.zeros((h, w), np.float32)
    out_v = np.zeros((h, w), dtype=bool)
    if uu.size:
        order = np.argsort(-zz)
        uu, vv, zz, lu = uu[order], vv[order], zz[order], lu[order]
        out_z[vv, uu] = zz
        out_l[vv, uu] = lu
        out_v[vv, uu] = True
    out_z[~out_v] = 0.0
    return out_z, out_v, out_l


def _mask_pose(
    z: np.ndarray,
    valid: np.ndarray,
    luma: np.ndarray,
    Ts: np.ndarray,
    head_ok: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    n, h, w = z.shape
    out = np.zeros((n, h, w), dtype=bool)
    for t in range(n - 1):
        if not (head_ok[t] and head_ok[t + 1]):
            continue
        T_rel = _inv_T(Ts[t + 1]) @ Ts[t]
        za, va, la = _warp_by_T(z[t], valid[t], luma[t], T_rel, K)
        dz = np.abs(z[t + 1] - za)
        moving = (valid[t + 1] & va) & (dz > THRESH_M) & _residual_gate(dz)
        moving &= np.abs(luma[t + 1] - la) > RGB_THRESH
        out[t + 1] |= moving
        out[t] |= moving
    return out


def _open_repo(config, source_name: str):
    sources = _indexed_sources(config, eval=True)
    source = next((s for s in sources if s["name"] == source_name), None)
    if source is None:
        sources = _indexed_sources(config, eval=False)
        source = next(s for s in sources if s["name"] == source_name)
    release = get_episode_dataset_release(source["dataset"], source.get("release"))
    return open_dataset(
        source["dataset"],
        release=release,
        catalog_root=source.get("catalog_root"),
        catalog_path=source.get("catalog_path"),
        index_path=source.get("index_path"),
        readers=source.get("readers"),
    )


def _calib_path(episode) -> Path | None:
    rec = episode._catalog_record
    for blob in (rec.get("meta"), rec.get("metadata"), episode.info.metadata):
        if not isinstance(blob, dict):
            continue
        cal = blob.get("calibration")
        if isinstance(cal, dict):
            for key in ("intrinsics", "undistorted_intrinsics"):
                p = cal.get(key)
                if p and Path(p).is_file():
                    return Path(p)
        for key in ("intrinsics", "undistorted_intrinsics"):
            p = blob.get(key)
            if p and Path(p).is_file():
                return Path(p)
    return None


def _load_ego_geom(config, source_name: str, episode_id: str, indices: list[int]):
    repo = _open_repo(config, source_name)
    try:
        ep = repo.get_episode(episode_id)
        raw = ep.read_source_modalities(
            {
                "head/pose": indices,
                "head/valid": indices,
                "left_hand/eepose": indices,
                "right_hand/eepose": indices,
                "left_hand/world_eepose": indices,
                "right_hand/world_eepose": indices,
                "left_hand/mano/valid": indices,
                "right_hand/mano/valid": indices,
                "left_hand/tracks": indices,
                "right_hand/tracks": indices,
            }
        )
        cal = _calib_path(ep)
        if cal is None:
            raise FileNotFoundError(f"no intrinsics for {episode_id}")
        intr = json.loads(cal.read_text())
        k_left = np.asarray(intr["cameras"]["left"]["K"], dtype=np.float64)
        k_right = np.asarray(intr["cameras"]["right"]["K"], dtype=np.float64)
        return raw, k_left, k_right, str(cal)
    finally:
        repo.close()


def _verify_identity(raw) -> dict:
    head = raw["head/pose"].numpy()
    hv = raw["head/valid"].numpy().reshape(-1).astype(bool)
    rows = []
    for side in ("left", "right"):
        cam = raw[f"{side}_hand/eepose"].numpy()
        world = raw[f"{side}_hand/world_eepose"].numpy()
        ok = raw[f"{side}_hand/mano/valid"].numpy().reshape(-1).astype(bool) & hv
        dts, angs = [], []
        for i in np.where(ok)[0]:
            T_pred = _inv_T(_pose7_to_T(head[i])) @ _pose7_to_T(world[i])
            T_cam = _pose7_to_T(cam[i])
            dts.append(float(np.linalg.norm(T_pred[:3, 3] - T_cam[:3, 3])))
            angs.append(_rot_deg(T_pred[:3, :3], T_cam[:3, :3]))
        rows.append(
            {
                "side": side,
                "n": int(ok.sum()),
                "dt_med_m": float(np.median(dts)) if dts else float("nan"),
                "dt_p95_m": float(np.percentile(dts, 95)) if dts else float("nan"),
                "ang_med_deg": float(np.median(angs)) if angs else float("nan"),
                "ang_p95_deg": float(np.percentile(angs, 95)) if angs else float("nan"),
            }
        )
    return {"hands": rows, "head_valid_frac": float(hv.mean())}


def _hand_proj_err(raw, K: np.ndarray, pad_box) -> dict:
    """eepose (camera xyz) vs tracks[:,0,:2] after train resize — wrist-ish."""
    out = {}
    for side in ("left", "right"):
        cam = raw[f"{side}_hand/eepose"].numpy()
        tracks = raw[f"{side}_hand/tracks"].numpy()
        ok = raw[f"{side}_hand/mano/valid"].numpy().reshape(-1).astype(bool)
        ok &= cam[:, 2] > 0.05
        uv_pose = _project(cam[:, :3], K)
        # tracks (21,3): first two channels look like native pixels if large
        uv_tr = _uv_native_to_train(tracks[:, 0, :2], pad_box)
        d = np.linalg.norm(uv_pose - uv_tr, axis=1)
        d = d[ok]
        out[side] = {
            "n": int(ok.sum()),
            "track0_med_px": float(np.median(d)) if d.size else float("nan"),
            "ee_uv_med": [float(x) for x in np.median(uv_pose[ok], axis=0)]
            if ok.any()
            else [float("nan"), float("nan")],
            "tr_uv_med": [float(x) for x in np.median(uv_tr[ok], axis=0)]
            if ok.any()
            else [float("nan"), float("nan")],
            "track_xy_med_raw": [float(x) for x in np.median(tracks[ok, 0, :2], axis=0)]
            if ok.any()
            else [float("nan"), float("nan")],
        }
    return out


def _overlay(rgb: np.ndarray, mot: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    tint = np.array([1.0, 0.25, 0.08], dtype=np.float32)
    out[mot] = 0.35 * out[mot] + 0.65 * tint
    return out.clip(0, 1)


def _mark(rgb: np.ndarray, uv: np.ndarray, color) -> np.ndarray:
    img = np.ascontiguousarray((rgb * 255).astype(np.uint8))
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.circle(img, (u, v), 8, color, 2)
        cv2.circle(img, (u, v), 2, color, -1)
    return img.astype(np.float32) / 255.0


def main() -> int:
    clips = {int(c["i"]): c for c in json.loads(CAT.read_text())}
    config = load_config(str(CFG))
    attacher = _build_depth_attacher(config)
    z_near = float(getattr(config.depth, "z_near", 0.15))
    z_far = float(getattr(config.depth, "z_far", 4.0))
    mode = str(getattr(config.depth, "mode", "log"))
    seed = int(config.periodic_eval.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    font_p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_p, 18) if Path(font_p).is_file() else ImageFont.load_default()
    import imageio.v2 as imageio

    summary = []
    for k, idx in enumerate(PICK):
        clip = clips[idx]
        batch = create_diverse_eval_batch(
            config,
            clip["source"],
            batch_size=1,
            shard_ordinal=int(clip["shard"]),
            seed=seed,
            use_eval_pool=(clip["split"] == "holdout"),
            episode_id=clip.get("episode_id"),
        )
        if attacher is not None:
            batch = dict(attacher.attach(batch))
        rgb = torch.cat(
            [batch["first_frame"][0].float().unsqueeze(0), batch["video_frames"][0].float()], 0
        )
        gt = torch.cat(
            [batch["depth_first"][0].float().unsqueeze(0), batch["depth_frames"][0].float()], 0
        )
        n = min(rgb.shape[0], gt.shape[0])
        rgb, gt = rgb[:n], gt[:n]
        zt, vt = decode_to_depth_m(gt, z_near=z_near, z_far=z_far, mode=mode, in_range="01")
        z, valid = zt.numpy(), vt.numpy()
        luma = rgb.mean(dim=1).numpy()
        rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
        base = (vt & _depth_motion_mask(zt, vt, thresh_m=THRESH_M, rgb=rgb)).numpy()
        pose = np.zeros_like(base)
        verify = None
        K = None
        uv_l = uv_r = None
        if clip["family"] == "ego":
            raw_fi = batch["frame_indices"][0]
            fi = [int(x) for x in (raw_fi.tolist() if hasattr(raw_fi, "tolist") else list(raw_fi))[:n]]
            raw, k_left, k_right, cal = _load_ego_geom(
                config, clip["source"], clip["episode_id"], fi
            )
            K, pad_box = _k_train(k_left)
            ident = _verify_identity(raw)
            proj = _hand_proj_err(raw, K, pad_box)
            head = raw["head/pose"].numpy()
            hv = raw["head/valid"].numpy().reshape(-1).astype(bool)
            Ts = np.stack([_pose7_to_T(p) for p in head], 0)
            pose = _drop_small(_mask_pose(z, valid, luma, Ts, hv, K) & valid, MIN_AREA) & valid
            uv_l = _project(raw["left_hand/eepose"].numpy()[:, :3], K)
            uv_r = _project(raw["right_hand/eepose"].numpy()[:, :3], K)
            verify = {
                "calib": cal,
                "k_left_native": k_left.tolist(),
                "k_right_native": k_right.tolist(),
                "k_train": K.tolist(),
                "identity": ident,
                "proj": proj,
                "k_left_vs_right_cx": float(k_left[0, 2] - k_right[0, 2]),
            }
        den = max(int(valid.sum()), 1)
        rec = {
            "k": k,
            "i": idx,
            "family": clip["family"],
            "episode_id": clip["episode_id"],
            "base": float(base.sum() / den),
            "posecc": float(pose.sum() / den) if clip["family"] == "ego" else None,
            "verify": verify,
        }
        summary.append(rec)
        print(
            f"{k:02d} {clip['family']} base={rec['base']:.3f} "
            f"posecc={rec['posecc'] if rec['posecc'] is not None else 'n/a'} "
            f"{clip['episode_id'][:42]}",
            flush=True,
        )
        if verify is not None:
            for h in verify["identity"]["hands"]:
                print(
                    f"     ident {h['side']}: n={h['n']} dt_med={h['dt_med_m']*1000:.1f}mm "
                    f"ang_med={h['ang_med_deg']:.2f}deg",
                    flush=True,
                )
            for side, p in verify["proj"].items():
                print(
                    f"     proj  {side}: n={p['n']} ee→track0={p['track0_med_px']:.1f}px "
                    f"ee_uv={p['ee_uv_med']} tr_uv={p['tr_uv_med']}",
                    flush=True,
                )

        frames = []
        for t in range(n):
            pose_panel = pose if clip["family"] == "ego" else base
            tiles = [
                rgb_np[t],
                _overlay(rgb_np[t], base[t]),
                _overlay(rgb_np[t], pose_panel[t]),
            ]
            if uv_l is not None and np.isfinite(uv_l[t]).all():
                tiles[0] = _mark(tiles[0], uv_l[t], (0, 220, 80))
            if uv_r is not None and np.isfinite(uv_r[t]).all():
                tiles[0] = _mark(tiles[0], uv_r[t], (40, 120, 255))
            row = np.concatenate(tiles, axis=1)
            img = Image.fromarray((row * 255).clip(0, 255).astype(np.uint8))
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, img.width, 26), fill=(255, 255, 255))
            pose_txt = (
                f"pose+cc {rec['posecc']:.1%}"
                if rec["posecc"] is not None
                else "pose n/a (agi)"
            )
            d.text(
                (6, 3),
                f"{clip['family']} {clip['episode_id'][:36]} t={t}/{n-1}  "
                f"RGB | base {rec['base']:.1%} | {pose_txt}",
                fill=(20, 20, 20),
                font=font,
            )
            frames.append(np.asarray(img))
        name = f"{k:02d}_{clip['family']}_{clip['episode_id'].replace('/', '_')[:48]}.mp4"
        path = OUT / name
        try:
            imageio.mimsave(str(path), frames, fps=5, codec="libx264")
        except Exception:
            path = path.with_suffix(".gif")
            imageio.mimsave(str(path), frames, fps=5)
        dest = MASK_DIR / name
        dest.write_bytes(path.read_bytes())
        print(f"     wrote {dest}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ego = [r for r in summary if r["family"] == "ego"]
    agi = [r for r in summary if r["family"] == "agi"]
    print(
        f"agi  med base={np.median([r['base'] for r in agi]):.3f}",
        flush=True,
    )
    print(
        "ego  med base/pose = "
        f"{np.median([r['base'] for r in ego]):.3f}/"
        f"{np.median([r['posecc'] for r in ego]):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
