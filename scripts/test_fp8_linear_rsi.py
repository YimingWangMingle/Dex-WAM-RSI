"""MOTUS_FP8_LINEAR wiring in UniWorldModel: conversion scope, checkpoint keys,
CPU bit-exactness (FP8Linear falls back to F.linear off-GPU) and CUDA numerics.

Run: python scripts/test_fp8_linear_rsi.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_cpu_stubs as S  # noqa: E402

import torch  # noqa: E402

from thinker_wm.models.fp8_linear import FP8Linear  # noqa: E402
from thinker_wm.models.temporal_layout import TemporalRequest  # noqa: E402
from test_temporal_causality import B, GRID, history, inputs  # noqa: E402


def build(fp8, **kw):
    old = os.environ.get("MOTUS_FP8_LINEAR")
    os.environ["MOTUS_FP8_LINEAR"] = "1" if fp8 else ""
    try:
        return S.tiny_world(enable_depth=True, enable_history=True, depth_uni_clone_layers=1, **kw)
    finally:
        if old is None:
            os.environ.pop("MOTUS_FP8_LINEAR")
        else:
            os.environ["MOTUS_FP8_LINEAR"] = old


def forward(m, device, dtype):
    x = {k: v.to(device, dtype) for k, v in inputs().items()}
    h = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in history().items()}
    with torch.no_grad(), torch.autocast(device, torch.bfloat16, enabled=dtype == torch.bfloat16):
        return m(torch.tensor([[600.0], [800.0]], device=device), None, None, x["video"], x["action"],
                 und_cache=tuple(t.to(device, dtype) if t.is_floating_point() else t.to(device)
                                 for t in S.und_cache(B)),
                 depth=x["depth"], state=x["state"], temporal=TemporalRequest(GRID, cut=1), history=h)


class Fp8Wiring(unittest.TestCase):
    def test_scope_and_keys(self):
        ref, m = build(False), build(True)
        names = sorted(n for n, mod in m.named_modules() if isinstance(mod, FP8Linear))
        self.assertTrue(names)
        for n in names:
            self.assertTrue(n.startswith(("uni_branch.uni_blocks.", "video_branch.decode_block.")), n)
            self.assertTrue(".self_attn." in n or ".ffn." in n, n)
        blocks = list(m.uni_branch.uni_blocks) + list(m.video_branch.decode_block)
        for blk in blocks:
            self.assertIsInstance(blk.self_attn.q, FP8Linear)
            self.assertNotIsInstance(blk.cross_attn.q, FP8Linear)
        for blk in m.uni_branch.depth_uni_blocks:
            self.assertNotIsInstance(blk.self_attn.q, FP8Linear)
        rs, ms = ref.state_dict(), m.state_dict()
        self.assertEqual(list(rs), list(ms))
        for k in rs:
            self.assertTrue(torch.equal(rs[k], ms[k]), k)

    def test_cpu_forward_unchanged(self):
        for a, b in zip(forward(build(False), "cpu", torch.float32), forward(build(True), "cpu", torch.float32)):
            self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
    def test_cuda_close_to_bf16(self):
        outs = [forward(build(fp8).cuda().to(torch.bfloat16), "cuda", torch.bfloat16) for fp8 in (False, True)]
        for a, b in zip(*outs):
            rel = float((b.float() - a.float()).norm() / a.float().norm())
            self.assertLess(rel, 5e-2)


if __name__ == "__main__":
    unittest.main()
