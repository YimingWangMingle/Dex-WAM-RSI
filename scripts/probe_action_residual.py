#!/usr/bin/env python3
"""Action-residual probe 2: does VAE depth/XYZ explain error that state already missed?

Cheap form (no 5B forward) — the gate we can run today:

  1. AgiBot clips that have depth sidecars + catalog proprio.
  2. Encode the same mm frames two ways (log-gray vs FLEX XYZ) through the
     frozen Wan VAE; mean-pool each latent to a vector.
  3. Fit Linear(state) → future action chunk. Residual r = action − pred_state.
  4. Fit Linear(depth_latent) → r  and  Linear(xyz_latent) → r  (ridge).
  5. Held-out R² / MAE, plus the z / xy groups (table height / reach).

  R²≈0 → that stream is redundant with state.
  R²>0 and the gain sits on z/xy → worth a train regularizer, not yet a deploy input.

This is NOT yet "current WM action error". That needs --wm-pred-pt (a saved
tensor of Motus action predictions on the same clips). If you pass it, step 3
uses r = action − wm_pred instead of the linear state baseline.

  CUDA_VISIBLE_DEVICES=1 /home/fengodchen/miniconda3/envs/motus/bin/python \\
      scripts/probe_action_residual.py --n 48

`--wm-pred-pt` contract: a dict with keys
  episode_ids: list[str]
  frame0:     list[int]
  pred:       FloatTensor [N, H, A]
  action:     FloatTensor [N, H, A]   # optional, else loaded from disk
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "third_party/Thinker-WM/libs" / "motus-data-factory"))

from thinker_wm.models.depth_codec import encode_depth_mm  # noqa: E402
from thinker_wm.models.depth_reader import DepthSidecarReader  # noqa: E402
from probe_depth_codec_vae import (  # noqa: E402
    AGI_ROOT,
    FLEX_BOUNDS,
    FLEX_MAX_DEPTH,
    T_FRAMES,
    VAE_PTH,
    Z_FAR,
    Z_NEAR,
    _load_k,
    _norm_xyz,
    _pick_clips,
    _unproject,
)

CATALOGS = [
    Path(
        "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
        "agibot_beta_parallel_gripper.20260805_full_v1.episode_catalog.jsonl"
    ),
    Path(
        "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
        "agibot_beta_dexterous_hand.20260805_full_v1.episode_catalog.jsonl"
    ),
]

# This catalog is already 1 video frame : 1 proprio row.
HORIZON = 16
# 26-D = 2 arms × [xyz(3) + rotvec(3) + grip(1) + pad(6)]
DIM_XY = (0, 1, 13, 14)
DIM_Z = (2, 15)
DIM_GRIP = (6, 19)


def _skip(msg: str) -> int:
    print(f"SKIP: {msg}")
    return 0


def _pool_latent(lat: torch.Tensor) -> torch.Tensor:
    """[C,T,H,W] or [B,C,T,H,W] → [C] mean-pool."""
    if lat.dim() == 5:
        lat = lat[0]
    return lat.float().mean(dim=(1, 2, 3))


def _ridge_fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-1) -> np.ndarray:
    n = x.shape[0]
    xb = np.concatenate([x, np.ones((n, 1), dtype=x.dtype)], axis=1)
    xtx = xb.T @ xb
    xtx[np.diag_indices_from(xtx)] += l2
    return np.linalg.solve(xtx, xb.T @ y)


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _lookup_catalog(wanted: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    still = set(wanted)
    for path in CATALOGS:
        if not still or not path.is_file():
            continue
        with path.open() as handle:
            for line in handle:
                if not still:
                    break
                rec = json.loads(line)
                eid = str(rec.get("episode_id", ""))
                if eid in still:
                    found[eid] = rec
                    still.discard(eid)
    return found


def _read_proprio(rec: dict) -> torch.Tensor | None:
    from motus_data.datasets.agibot_common import (
        normalize_agibot_eef_actions,
        read_agibot_eef_actions,
    )

    st = rec.get("state", {}).get("storage") or {}
    kind = st.get("type")
    try:
        if kind == "tar":
            path, offset, size = st["path"], int(st["offset"]), int(st["size"])
            if not Path(path).is_file():
                return None
            with open(path, "rb") as handle:
                handle.seek(offset)
                blob = handle.read(size)
            if len(blob) != size:
                return None
            raw = read_agibot_eef_actions(io.BytesIO(blob))
        elif kind == "file":
            raw = read_agibot_eef_actions(st["path"])
        else:
            return None
        return normalize_agibot_eef_actions(raw)
    except Exception as exc:
        print(f"  proprio fail {rec.get('episode_id')}: {exc}")
        return None


def _slice_sa(timeline: torch.Tensor, frame0: int, horizon: int):
    # catalog is 1:1 with video frames
    if frame0 < 0 or frame0 + horizon >= timeline.shape[0]:
        return None
    state = timeline[frame0].float()
    action = timeline[frame0 + 1 : frame0 + 1 + horizon].float()
    return state, action


def _group_r2(y: np.ndarray, pred: np.ndarray, dims: tuple[int, ...], horizon: int, adim: int) -> float:
    y3 = y.reshape(-1, horizon, adim)
    p3 = pred.reshape(-1, horizon, adim)
    return _r2(y3[..., list(dims)].reshape(len(y), -1), p3[..., list(dims)].reshape(len(y), -1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--vae", type=Path, default=VAE_PTH)
    p.add_argument("--wm-pred-pt", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("/tmp/probe_action_residual.json"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        return _skip("no CUDA")
    if not args.vae.is_file():
        return _skip(f"VAE not found: {args.vae}")
    if not AGI_ROOT.is_dir():
        return _skip(f"no AgiBot depth root: {AGI_ROOT}")

    from thinker_wm.wan.modules.vae2_2 import Wan2_2_VAE

    device = "cuda"
    print(f"loading Wan 2.2 VAE from {args.vae} ...")
    vae = Wan2_2_VAE(vae_pth=str(args.vae), device=device)
    reader = DepthSidecarReader([AGI_ROOT])
    clips = _pick_clips(AGI_ROOT, args.n, T_FRAMES)
    if not clips:
        return _skip("no AgiBot prod_keep clips")

    wanted = {eid for eid, _, _ in clips}
    catalog = _lookup_catalog(wanted)
    print(f"clips={len(clips)}  catalog hits={len(catalog)}/{len(wanted)}")

    wm = None
    if args.wm_pred_pt and args.wm_pred_pt.is_file():
        wm = torch.load(args.wm_pred_pt, map_location="cpu")
        print(f"loaded WM preds from {args.wm_pred_pt}")

    rows = []
    n_sa = 0
    n_xyz = 0
    for eid, clip, ep_dir in clips:
        mm, ok = reader.read_mm(eid, clip)
        if not ok or mm is None:
            continue
        x_gray = encode_depth_mm(mm, Z_NEAR, Z_FAR, mode="log", out_range="pm1")
        x_gray = x_gray.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            lat_g = vae.encode(x_gray)
        feat_g = _pool_latent(lat_g).cpu()

        feat_x = None
        k = _load_k(ep_dir)
        if k is not None:
            xyz = _unproject(mm.float() / 1000.0, k, FLEX_MAX_DEPTH)
            x_xyz = _norm_xyz(xyz, FLEX_BOUNDS).permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad():
                lat_x = vae.encode(x_xyz.to(device=device, dtype=torch.bfloat16))
            feat_x = _pool_latent(lat_x).cpu()
            n_xyz += 1

        rec = catalog.get(eid)
        timeline = _read_proprio(rec) if rec is not None else None
        sa = _slice_sa(timeline, clip[0], HORIZON) if timeline is not None else None
        if sa is None:
            continue
        state, action = sa
        n_sa += 1
        rows.append(
            {
                "eid": eid,
                "frame0": int(clip[0]),
                "feat_g": feat_g.numpy(),
                "feat_x": None if feat_x is None else feat_x.numpy(),
                "state": state.numpy(),
                "action": action.numpy().reshape(-1),
            }
        )
        print(f"  {eid} f0={clip[0]} xyz={'Y' if feat_x is not None else 'N'}")

    print(f"usable={len(rows)}  xyz={n_xyz}  horizon={HORIZON}")
    if len(rows) < 8:
        print("NOT ENOUGH (state, action) pairs. Check catalog / proprio tars.")
        return 1

    rng = np.random.default_rng(0)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_tr = max(6, int(0.7 * len(idx)))
    tr, te = idx[:n_tr], idx[n_tr:]
    if len(te) < 2:
        print("held-out split too small; using last 2 as test")
        tr, te = idx[:-2], idx[-2:]

    def stack(key, ids, skip_none=False):
        vals = []
        keep = []
        for i in ids:
            v = rows[i][key]
            if v is None and skip_none:
                continue
            vals.append(v)
            keep.append(i)
        return np.stack(vals), np.asarray(keep)

    y_tr, _ = stack("action", tr)
    y_te, _ = stack("action", te)
    s_tr, _ = stack("state", tr)
    s_te, _ = stack("state", te)
    adim = rows[0]["state"].shape[0]

    if wm is not None:
        # optional: replace residual with WM error on matching (eid, frame0)
        key = {(str(e), int(f)): i for i, (e, f) in enumerate(zip(wm["episode_ids"], wm["frame0"]))}
        pred = wm["pred"].numpy() if torch.is_tensor(wm["pred"]) else np.asarray(wm["pred"])
        r_tr_list, r_te_list, keep_tr, keep_te = [], [], [], []
        for split, r_list, keep in ((tr, r_tr_list, keep_tr), (te, r_te_list, keep_te)):
            for i in split:
                j = key.get((rows[i]["eid"], rows[i]["frame0"]))
                if j is None:
                    continue
                r_list.append(rows[i]["action"] - pred[j].reshape(-1))
                keep.append(i)
        if len(r_tr_list) < 6 or len(r_te_list) < 2:
            print("WM pred overlap too small; falling back to state baseline")
            wm = None
        else:
            r_tr, r_te = np.stack(r_tr_list), np.stack(r_te_list)
            tr, te = np.asarray(keep_tr), np.asarray(keep_te)
            y_tr, y_te = stack("action", tr)[0], stack("action", te)[0]
            print(f"using WM residual  n_tr={len(tr)} n_te={len(te)}")

    if wm is None:
        w_s = _ridge_fit(s_tr, y_tr)
        pred_s_te = np.c_[s_te, np.ones((len(te), 1))] @ w_s
        pred_s_tr = np.c_[s_tr, np.ones((len(tr), 1))] @ w_s
        r_tr = y_tr - pred_s_tr
        r_te = y_te - pred_s_te
        print(
            f"\nstate → action   held-out R²={_r2(y_te, pred_s_te):.3f}  "
            f"MAE={float(np.abs(y_te - pred_s_te).mean()):.4f}  "
            f"xy={_group_r2(y_te, pred_s_te, DIM_XY, HORIZON, adim):.3f}  "
            f"z={_group_r2(y_te, pred_s_te, DIM_Z, HORIZON, adim):.3f}  "
            f"grip={_group_r2(y_te, pred_s_te, DIM_GRIP, HORIZON, adim):.3f}"
        )

    summary = {"n_tr": int(len(tr)), "n_te": int(len(te)), "horizon": HORIZON, "streams": {}}
    print(f"residual MAE (before depth) = {float(np.abs(r_te).mean()):.4f}")

    for name, key in (("gray_vae", "feat_g"), ("xyz_flex_vae", "feat_x")):
        x_tr, tr_keep = stack(key, tr, skip_none=True)
        x_te, te_keep = stack(key, te, skip_none=True)
        if len(x_tr) < 6 or len(x_te) < 2:
            print(f"{name:14} skipped (missing features)")
            continue
        # align residuals to the rows that actually have this feature
        tr_map = {int(i): k for k, i in enumerate(tr)}
        te_map = {int(i): k for k, i in enumerate(te)}
        rtr = np.stack([r_tr[tr_map[int(i)]] for i in tr_keep])
        rte = np.stack([r_te[te_map[int(i)]] for i in te_keep])
        w = _ridge_fit(x_tr, rtr)
        pred_r = np.c_[x_te, np.ones((len(x_te), 1))] @ w
        mae0 = float(np.abs(rte).mean())
        mae1 = float(np.abs(rte - pred_r).mean())
        line = (
            f"{name:14} residual R²={_r2(rte, pred_r):.3f}  "
            f"ΔMAE={mae0 - mae1:+.4f}  "
            f"xy={_group_r2(rte, pred_r, DIM_XY, HORIZON, adim):.3f}  "
            f"z={_group_r2(rte, pred_r, DIM_Z, HORIZON, adim):.3f}  "
            f"grip={_group_r2(rte, pred_r, DIM_GRIP, HORIZON, adim):.3f}"
        )
        print(line)
        summary["streams"][name] = {
            "residual_r2": _r2(rte, pred_r),
            "delta_mae": mae0 - mae1,
            "xy_r2": _group_r2(rte, pred_r, DIM_XY, HORIZON, adim),
            "z_r2": _group_r2(rte, pred_r, DIM_Z, HORIZON, adim),
            "grip_r2": _group_r2(rte, pred_r, DIM_GRIP, HORIZON, adim),
            "n_te": int(len(x_te)),
        }

    print("\nresidual R²≈0 → VAE feature adds nothing beyond state.")
    print("residual R²>0.05 and ΔMAE>0 on z/xy → worth a train-time regularizer.")
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
