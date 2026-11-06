#!/usr/bin/env python3
"""QC smoke DA3 depth vs AgiBot-Beta sensor head_depth PNGs (same tar)."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SMOKE = Path("/data/xiongwei.zhao/agibot_da3_depth/smoke")
TAR = "/data/agibot_beta/observations/327/648642-685046.tar"
EP = "648642"
N = 49
H, W = 480, 640
NEAR, FAR = 0.1, 10.0  # meters, sensor gate


def colorize_m(d_m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip((d_m - lo) / max(hi - lo, 1e-6), 0, 1)
    u8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)[:, :, ::-1]


def load_sensor() -> np.ndarray:
    needed = {f"{EP}/depth/head_depth_{i:06d}.png": i for i in range(N)}
    got: dict[int, np.ndarray] = {}
    with tarfile.open(TAR, mode="r|") as tf:
        for m in tf:
            if m.name not in needed:
                continue
            raw = tf.extractfile(m).read()
            d = np.asarray(Image.open(io.BytesIO(raw)))
            if d.dtype != np.uint16:
                raise RuntimeError(f"{m.name} dtype={d.dtype}")
            got[needed[m.name]] = d
            if len(got) == N:
                break
    missing = [i for i in range(N) if i not in got]
    if missing:
        raise FileNotFoundError(f"missing sensor frames {missing[:8]}")
    stack = np.stack([got[i] for i in range(N)], 0)
    if stack.shape[1:] != (H, W):
        out = np.empty((N, H, W), np.uint16)
        for i in range(N):
            out[i] = cv2.resize(stack[i], (W, H), interpolation=cv2.INTER_NEAREST)
        stack = out
    return stack


def main() -> int:
    da3_mm = np.load(SMOKE / "depth_uint16_mm.npy")
    assert da3_mm.shape == (N, H, W), da3_mm.shape
    sensor_mm = load_sensor()
    da3 = da3_mm.astype(np.float32) / 1000.0
    sen = sensor_mm.astype(np.float32) / 1000.0

    m_s = np.isfinite(sen) & (sen > NEAR) & (sen < FAR) & (sensor_mm > 0)
    m_d = np.isfinite(da3) & (da3 > 0) & (da3_mm > 0)
    both = m_s & m_d
    n_both = int(both.sum())
    if n_both < 1000:
        raise RuntimeError(f"too few overlap pixels: {n_both}")

    ratio = da3[both] / np.clip(sen[both], 1e-6, None)
    alpha = float(np.median(ratio))
    da3_al = da3 / alpha
    absrel = np.abs(da3_al - sen) / np.clip(sen, 1e-6, None)
    absrel_med = float(np.median(absrel[both]))
    absrel_mean = float(np.mean(absrel[both]))
    delta125 = float(np.mean(np.maximum(da3_al[both] / sen[both], sen[both] / da3_al[both]) < 1.25))
    log_a = float(np.median(np.abs(np.log(np.clip(ratio, 1e-6, None)))))

    # per-frame scale breathing
    alphas = []
    absrels = []
    for t in range(N):
        b = both[t]
        if b.sum() < 500:
            continue
        a = float(np.median(da3[t][b] / np.clip(sen[t][b], 1e-6, None)))
        alphas.append(a)
        d_al = da3[t] / a
        absrels.append(float(np.median(np.abs(d_al[b] - sen[t][b]) / np.clip(sen[t][b], 1e-6, None))))
    a_arr = np.asarray(alphas)
    breath = float(np.median(np.abs(np.log(a_arr / np.median(a_arr))))) if len(a_arr) else float("nan")

    # temporal jitter on DA3 (valid pixels)
    dt = np.abs(da3[1:] - da3[:-1])
    mdt = m_d[1:] & m_d[:-1]
    temporal_med = float(np.median(dt[mdt])) if mdt.any() else float("nan")

    report = {
        "n_frames": N,
        "sensor_hw": list(sensor_mm.shape[1:]),
        "sensor_valid_frac": float(m_s.mean()),
        "da3_valid_frac": float(m_d.mean()),
        "overlap_frac": float(both.mean()),
        "overlap_n": n_both,
        "sensor_p02_p50_p98_m": [float(np.percentile(sen[m_s], p)) for p in (2, 50, 98)],
        "da3_p02_p50_p98_m": [float(np.percentile(da3[m_d], p)) for p in (2, 50, 98)],
        "scale_alpha_da3_over_sensor": alpha,
        "absrel_med_after_scale": absrel_med,
        "absrel_mean_after_scale": absrel_mean,
        "delta_1.25": delta125,
        "log_alpha_med": log_a,
        "per_frame_scale_breath_med": breath,
        "per_frame_absrel_med": float(np.median(absrels)),
        "da3_temporal_med_m": temporal_med,
        "gate": {
            "sensor_valid_ge_0.40": bool(m_s.mean() >= 0.40),
            "absrel_le_0.15": bool(absrel_med <= 0.15),
            "scale_breath_le_0.05": bool(breath <= 0.05),
        },
    }
    (SMOKE / "qc.json").write_text(json.dumps(report, indent=2))

    lo = min(np.percentile(da3[m_d], 2), np.percentile(sen[m_s], 2))
    hi = max(np.percentile(da3[m_d], 98), np.percentile(sen[m_s], 98))
    rows = []
    for t in (0, 24, 48):
        err = np.clip(absrel[t] / 0.30, 0, 1)
        err_u8 = (err * 255).astype(np.uint8)
        err_col = cv2.applyColorMap(err_u8, cv2.COLORMAP_HOT)[:, :, ::-1]
        err_col[~both[t]] = 0
        row = np.concatenate(
            [colorize_m(da3[t], lo, hi), colorize_m(sen[t], lo, hi), err_col],
            axis=1,
        )
        cv2.putText(row, f"t={t} DA3 | sensor | AbsRel", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(SMOKE / "qc_da3_vs_sensor.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))

    print(json.dumps(report, indent=2))
    g = report["gate"]
    print(
        f"\nGATES  sensor_valid={g['sensor_valid_ge_0.40']}  "
        f"absrel={g['absrel_le_0.15']}  breath={g['scale_breath_le_0.05']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
