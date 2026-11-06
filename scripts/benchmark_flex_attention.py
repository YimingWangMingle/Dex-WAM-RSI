"""Dense bool-mask SDPA vs block-sparse flex_attention on the real RSI layout.

480x640 (15x20 tokens/latent), 4 packed AR blocks, 10 history slots pooled 2x2.
Run: python scripts/benchmark_flex_attention.py [--batch 4]   (CUDA)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from thinker_wm.models import temporal_attention as TA  # noqa: E402
from thinker_wm.models.temporal_layout import (  # noqa: E402
    Segment, TemporalLayout, TemporalRequest, TimeGrid, entries_action_segment,
    entries_video_segment, history_segment, packed_action_entries, packed_latent_entries,
)


def build_layout(batch, invalid_slots=3, device="cpu"):
    grid = TimeGrid(video_action_freq_ratio=4, action_t_downsample=1)
    req = TemporalRequest(grid, target_reads_earlier_targets=False, packed=True, clean_blocks=3,
                          first_target=1, last_target=4, depth_first_frame_conditioned=False)
    times = -torch.tensor([9., 8., 7., 6., 5., 4., 3., 2., 1.25, .25]).repeat(batch, 1)
    valid = torch.ones(batch, 10, dtype=torch.bool)
    valid[0, :invalid_slots] = False   # one sample early in its episode
    hseg = history_segment(times.to(device), valid.to(device), grid, 8, 10, pool=2, source_hw=(15, 20))
    layout = TemporalLayout(req)
    layout.video = Segment.cat([hseg, entries_video_segment(packed_latent_entries(req), 15, 20, device)],
                               batch)
    layout.depth = entries_video_segment(packed_latent_entries(req, depth=True), 15, 20, device)
    layout.action = entries_action_segment(packed_action_entries(req), grid, True, device=device)
    layout.history_logit_bias = torch.tensor(-6., device=device, requires_grad=True)
    return layout


def cases(layout, batch):
    # Callables: the bias column depends on a parameter and is rebuilt per call.
    yield "video self", lambda: TA.self_mask(layout, "video", batch)
    yield "uni video<-video,action", lambda: TA.reader_mask(layout, "video", ["video", "action"], batch)
    yield "uni depth<-video,action,depth", lambda: TA.reader_mask(layout, "depth", ["video", "action", "depth"], batch)
    yield "uni action<-video,action", lambda: TA.reader_mask(layout, "action", ["video", "action"], batch)


def run(impl, q, k, v, get):
    TA.ATTN_IMPL = impl
    return TA.masked_attention(q, k, v, *get())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    dev, dt = a.device, (torch.bfloat16 if a.device == "cuda" else torch.float32)
    layout = build_layout(a.batch, device=dev)
    for name, get in cases(layout, a.batch):
        mask = get()[0]
        lq, lk = mask.shape[-2:]
        g = torch.Generator(device=dev).manual_seed(0)
        mk = lambda n: torch.randn(a.batch, n, a.heads, 128, device=dev, dtype=dt, generator=g)  # noqa: E731
        q, k, v = mk(lq).requires_grad_(), mk(lk).requires_grad_(), mk(lk).requires_grad_()
        res = {}
        for impl in ("dense", "flex"):
            out = run(impl, q, k, v, get)
            grads = torch.autograd.grad(out.float().square().mean(), (q, k, v, layout.history_logit_bias),
                                        allow_unused=True)
            grads = [torch.zeros(()) if x is None else x for x in grads]
            res[impl] = (out.float(), [x.float() for x in grads])
            run(impl, q, k, v, get).float().square().mean().backward()  # warm-up / compile
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                out = run(impl, q, k, v, get)
                out.float().square().mean().backward()
            if dev == "cuda":
                torch.cuda.synchronize()
            res[impl] += ((time.perf_counter() - t0) / a.iters * 1e3,)
        (od, gd, td), (og, gg, tg) = res["dense"], res["flex"]
        rel = lambda x, y: float((x - y).norm() / y.norm().clamp_min(1e-12))  # noqa: E731
        print(f"{name:32s} Lq={lq:5d} Lk={lk:5d} "
              f"dense {td:7.2f} ms  flex {tg:7.2f} ms  x{td / tg:4.2f}  "
              f"out rel {rel(og, od):.1e}  grad rel {max(rel(x, y) for x, y in zip(gg, gd)):.1e}")


if __name__ == "__main__":
    main()
