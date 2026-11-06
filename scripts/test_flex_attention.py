"""flex_attention path == dense bool-mask SDPA on the real RSI layout (CUDA only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchmark_flex_attention import build_layout, cases  # noqa: E402
from thinker_wm.models import temporal_attention as TA  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "flex path is CUDA-only")
class FlexMatchesDense(unittest.TestCase):
    def tearDown(self):
        TA.ATTN_IMPL = "flex"

    def test_outputs_and_grads(self):
        batch = 2
        layout = build_layout(batch, device="cuda")
        for name, get in cases(layout, batch):
            mask = get()[0]
            lq, lk = mask.shape[-2:]
            if lq < TA.FLEX_MIN_QUERIES:
                continue
            g = torch.Generator(device="cuda").manual_seed(0)
            q, k, v = (torch.randn(batch, n, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
                       .requires_grad_() for n in (lq, lk, lk))
            res = {}
            for impl in ("dense", "flex"):
                TA.ATTN_IMPL = impl
                out = TA.masked_attention(q, k, v, *get())
                grads = torch.autograd.grad(out.float().square().mean(),
                                            (q, k, v, layout.history_logit_bias), allow_unused=True)
                res[impl] = [out.float()] + [torch.zeros((), device="cuda") if x is None else x.float()
                                             for x in grads]
            for d, f in zip(res["dense"], res["flex"]):
                rel = float((f - d).norm() / d.norm().clamp_min(1e-12))
                self.assertLess(rel, 2e-2, name)

    def test_tiny_world_with_checkpointing(self):
        """Flex inside non-reentrant checkpoint recompute, shared [1,Lq,Lk] masks
        and unaligned key lengths; fp32 so the scalar bias grad is comparable."""
        import _rsi_cpu_stubs as S
        from thinker_wm.models.temporal_layout import TemporalRequest
        from test_temporal_causality import B, GRID, history, inputs
        old = TA.FLEX_MIN_QUERIES
        TA.FLEX_MIN_QUERIES = 0
        try:
            res = {}
            for impl in ("dense", "flex"):
                TA.ATTN_IMPL = impl
                m = S.tiny_world(enable_depth=True, enable_history=True, gradient_checkpointing=True).cuda().train()
                x = {k: v.cuda() for k, v in inputs().items()}
                v, a, d = m(torch.tensor([[600.0], [800.0]], device="cuda"), None, None, x["video"], x["action"],
                            und_cache=tuple(t.cuda() for t in S.und_cache(B)), depth=x["depth"],
                            state=x["state"], temporal=TemporalRequest(GRID, cut=1),
                            history={k: v.cuda() for k, v in history().items()})
                (v.square().mean() + a.square().mean() + d.square().mean()).backward()
                res[impl] = [v, a, d, m.history_encoder.logit_bias.grad,
                             torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])]
            for x, y in zip(res["dense"], res["flex"]):
                self.assertLess(float((y - x).norm() / x.norm()), 1e-2)
        finally:
            TA.FLEX_MIN_QUERIES = old


if __name__ == "__main__":
    unittest.main()
