#!/usr/bin/env python3
"""Smoke test for models/depth_codec.py.

Checks: value range, 3-channel identity, invalid handling, analytic round-trip,
8-bit quantization round-trip error by distance, and prints a resolution table.

  conda run -n motus python scripts/smoke_depth_codec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.models.depth_codec import decode_to_depth_m, encode_depth_mm

Z_NEAR, Z_FAR = 0.1, 10.0
MODES = ("log", "inverse", "linear")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def test_range_shape_invalid() -> None:
    # [B, T, H, W] mm with a mix of invalid(0), near, mid, far, beyond-far.
    depth = torch.tensor([[0, 300, 2000, 8000, 50000]], dtype=torch.int32).view(1, 1, 1, 5)
    x = encode_depth_mm(depth, Z_NEAR, Z_FAR, mode="inverse", invalid_to="far")
    if tuple(x.shape) != (1, 1, 3, 1, 5):
        _fail(f"shape {tuple(x.shape)} != (1,1,3,1,5)")
    if float(x.min()) < -1.0 - 1e-6 or float(x.max()) > 1.0 + 1e-6:
        _fail(f"range [{float(x.min())}, {float(x.max())}] outside [-1,1]")
    if not torch.allclose(x[:, :, 0], x[:, :, 1]) or not torch.allclose(x[:, :, 0], x[:, :, 2]):
        _fail("3 channels not identical")
    if abs(float(x[0, 0, 0, 0, 0]) - (-1.0)) > 1e-6:
        _fail(f"invalid pixel -> {float(x[0,0,0,0,0])}, expected -1.0")
    # near should be brighter (larger) than far
    if not (x[0, 0, 0, 0, 1] > x[0, 0, 0, 0, 3]):
        _fail("near not brighter than far")
    print("ok  range / shape / invalid / near-bright")


def test_analytic_roundtrip() -> None:
    zs_m = torch.tensor([0.5, 1.0, 2.0, 5.0, 9.9])
    depth_mm = (zs_m * 1000).to(torch.int32).view(1, -1)
    for mode in MODES:
        x = encode_depth_mm(depth_mm, Z_NEAR, Z_FAR, mode=mode)
        z_rec, valid = decode_to_depth_m(x, Z_NEAR, Z_FAR, mode=mode)
        z_rec = z_rec.view(-1)
        if not bool(valid.all()):
            _fail(f"[{mode}] valid mask dropped a finite depth")
        err = (z_rec - zs_m).abs().max().item()
        if err > 1e-3:
            _fail(f"[{mode}] analytic round-trip err {err:.2e} m > 1e-3")
        # [0,1] boundary path (what the training batch uses): same numbers.
        x01 = encode_depth_mm(depth_mm, Z_NEAR, Z_FAR, mode=mode, out_range="01")
        if float(x01.min()) < -1e-6 or float(x01.max()) > 1.0 + 1e-6:
            _fail(f"[{mode}] out_range=01 escaped [0,1]")
        z01, _ = decode_to_depth_m(x01, Z_NEAR, Z_FAR, mode=mode, in_range="01")
        if (z01.view(-1) - z_rec).abs().max().item() > 1e-4:
            _fail(f"[{mode}] out_range 01 vs pm1 disagree")
        print(f"ok  analytic round-trip [{mode}] max_err={err:.2e} m (pm1+01)")


def test_quant_roundtrip_table() -> None:
    # Simulate the 8-bit path: v in [0,1] -> uint8 -> back, then decode.
    zs_m = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    depth_mm = (zs_m * 1000).to(torch.int32).view(1, -1)
    print(f"\n  8-bit quant round-trip (z_near={Z_NEAR}, z_far={Z_FAR})")
    print(f"  {'mode':8} {'z_true(m)':>10} {'z_rec(m)':>10} {'abs_err(m)':>11}")
    for mode in MODES:
        x = encode_depth_mm(depth_mm, Z_NEAR, Z_FAR, mode=mode)
        v = (x + 1.0) / 2.0
        v_q = (v * 255.0).round() / 255.0
        x_q = v_q * 2.0 - 1.0
        z_rec, _ = decode_to_depth_m(x_q, Z_NEAR, Z_FAR, mode=mode)
        z_rec = z_rec.view(-1)
        for zt, zr in zip(zs_m.tolist(), z_rec.tolist()):
            print(f"  {mode:8} {zt:10.2f} {zr:10.3f} {abs(zr - zt):11.4f}")


if __name__ == "__main__":
    test_range_shape_invalid()
    test_analytic_roundtrip()
    test_quant_roundtrip_table()
    print("\nsmoke_depth_codec OK")
