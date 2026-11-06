#!/usr/bin/env python3
"""本机实验：对各数据集源，查看当前数据流中 batch 经 VLM 处理后的输出内容。

复现训练侧真实路径（models/und_branch.py UndBranch.extract_und_features）：

    batch 的 vlm_input_messages（stage2 配置 unified_vlm_task_prompt: true 时
    由 utils.vlm_utils.build_vlm_task_message 用 t5 原始任务文本重建）
    -> preprocess_vlm_messages（首帧图像 + 文本 -> chat template -> processor）
    -> Qwen3-VL language_model forward -> last_hidden_state（即 und tokens）

支持三种用法：

1) 本地管线端到端（micro 配置 8 源）：
    python scripts/inspect_vlm_per_dataset.py \
        --config configs/micro_train_debug.yaml --vlm-path <Qwen3-VL 路径>

2) 仅导出样本（无 GPU/无 VLM 环境，如 H200 CPU 节点；PNG + JSON 很轻量）：
    python scripts/inspect_vlm_per_dataset.py \
        --config configs/export_vlm_probe_h200.yaml --export-samples <out_dir>

3) 用已导出的样本跑 VLM 检查（样本可来自另一台机器）：
    python scripts/inspect_vlm_per_dataset.py \
        --samples-dir <out_dir> --vlm-path <Qwen3-VL 路径>

对每个源报告：原始任务文本、管线 vlm 消息、统一 task prompt、渲染后 chat
template、token 构成、und tokens（last_hidden_state）统计、以及同一输入下
VLM 生成的文本（仅供人工检查；训练不生成文本，只取 hidden state）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from thinker_wm.data_prepare import create_training_data_pipeline  # noqa: E402
from thinker_wm.utils.output_paths import resolve_output_path  # noqa: E402
from thinker_wm.utils.vlm_utils import (  # noqa: E402
    build_vlm_task_message,
    preprocess_vlm_messages,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="third_party/Thinker-WM/configs/micro_train_debug.yaml")
    parser.add_argument("--vlm-path", default=None,
                        help="Qwen3-VL-2B-Instruct 本地权重目录（export 模式不需要）")
    parser.add_argument("--output-dir", default=resolve_output_path("outputs/vlm_inspect"))
    parser.add_argument("--export-samples", default=None, metavar="DIR",
                        help="只导出各源样本（首帧 PNG + 文本 JSON）到 DIR 后退出")
    parser.add_argument("--samples-dir", default=None, metavar="DIR",
                        help="从 DIR 读取已导出的样本（跳过数据管线）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=128,
                        help="集齐全部源之前最多消费的 batch 数")
    parser.add_argument("--gen-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def collect_one_batch_per_source(config_path: str, seed: int, max_batches: int):
    """经训练同款管线消费 batch，每个源保留第一个 batch。"""
    config = OmegaConf.load(config_path)
    pipeline = create_training_data_pipeline(config, rank=0, world_size=1, seed=seed)
    collected: dict[str, dict] = {}
    with pipeline:
        for _ in range(max_batches):
            if collected and len(collected) >= len(pipeline.snapshot().get("mix_sources", [])):
                break
            batch, stats = pipeline.next()
            # 注意：batch["source_datasets"] 是数据集族名（robotwin/robotmind），
            # 区分 YAML 里的多个源要用 mix 统计里的 mix_source_name。
            source = stats.get("mix_source_name") or batch["source_datasets"][0]
            if source not in collected:
                collected[source] = batch
                print(f"[collect] source={source} ({len(collected)} 个源已收集)", flush=True)
    return collected


def batch_to_sample(source: str, batch: dict, idx: int = 0) -> dict:
    return {
        "source": source,
        "robot": batch["robot_names"][idx],
        "t5": batch["t5_input_messages"][idx],
        "pipeline_msg": batch["vlm_input_messages"][idx],
        "pil": batch["first_frame_pils"][idx],
    }


def export_samples(collected: dict[str, dict], export_dir: Path) -> None:
    """把收集到的各源样本导出为 首帧 PNG + samples.json（跨机器搬运用）。"""
    export_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(collected):
        sample = batch_to_sample(source, collected[source])
        frame_file = f"frame_{source}.png"
        sample["pil"].save(export_dir / frame_file)
        records.append({
            "source": source,
            "robot": sample["robot"],
            "t5": sample["t5"],
            "pipeline_msg": sample["pipeline_msg"],
            "frame_file": frame_file,
        })
        print(f"[export] {source}: frame={frame_file} t5={str(sample['t5'])[:60]!r}", flush=True)
    (export_dir / "samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2))
    print(f"[export] 共 {len(records)} 个源 -> {export_dir}", flush=True)


def load_samples(samples_dir: Path) -> list[dict]:
    records = json.loads((samples_dir / "samples.json").read_text())
    samples = []
    for r in records:
        samples.append({
            **r,
            "pil": Image.open(samples_dir / r["frame_file"]).convert("RGB"),
        })
    return samples


def _token_breakdown(inputs, image_token_id: int):
    input_ids = inputs["input_ids"][0]
    n_total = int(input_ids.shape[0])
    n_image = int((input_ids == image_token_id).sum().item())
    return n_total, n_image, n_total - n_image


def _stats_line(name: str, t: torch.Tensor) -> str:
    t = t.float()
    return (
        f"{name}: shape={tuple(t.shape)} mean={t.mean().item():+.4f} "
        f"std={t.std().item():+.4f} min={t.min().item():+.4f} max={t.max().item():+.4f} "
        f"norm(均值)={t.norm(dim=-1).mean().item():.2f}"
    )


@torch.no_grad()
def inspect_sample(sample: dict, processor, und_branch, vlm_model,
                   device: str, gen_max_new_tokens: int, out_dir: Path) -> dict:
    source = sample["source"]
    pil = sample["pil"]
    t5_text = sample["t5"]
    pipeline_msg = sample["pipeline_msg"]
    unified_msg = build_vlm_task_message(t5_text)
    robot = sample["robot"]

    frame_path = out_dir / f"frame_{source}.png"
    pil.save(frame_path)

    image_token_id = vlm_model.config.image_token_id

    report: dict = {
        "source": source,
        "robot": robot,
        "first_frame_size": list(pil.size),
        "frame_file": frame_path.name,
        "t5_input_message(原始任务文本)": t5_text,
        "vlm_input_message(数据管线拼接)": pipeline_msg,
        "unified_vlm_message(stage2 训练实际使用)": unified_msg,
    }

    for tag, msg in (("pipeline", pipeline_msg), ("unified", unified_msg)):
        inputs = preprocess_vlm_messages(msg, pil, processor)
        n_total, n_image, n_text = _token_breakdown(inputs, image_token_id)

        # 渲染后的 chat template 文本（VLM 实际看到的字符串）
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": msg}]}]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # —— 训练路径：与 UndBranch.extract_und_features 完全一致的 forward ——
        und_tokens, _real_lens = und_branch.extract_und_features([msg], [pil])  # [1, L, 2048]
        visual_seg = und_tokens[0, :n_image] if n_image else None
        text_seg = und_tokens[0, n_image:n_total]
        stats_lines = [_stats_line("und_tokens(全部)", und_tokens[0])]
        if visual_seg is not None:
            stats_lines.append(_stats_line("und_tokens(视觉段)", visual_seg))
        stats_lines.append(_stats_line("und_tokens(文本段)", text_seg))

        # —— 对照：同一输入下 VLM 生成的文本（训练不用，仅供查看内容）——
        gen_inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_ids = vlm_model.generate(
            **gen_inputs, max_new_tokens=gen_max_new_tokens, do_sample=False)
        gen_text = processor.tokenizer.decode(
            gen_ids[0, gen_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)

        report[f"[{tag}] rendered_chat_template"] = rendered
        report[f"[{tag}] token统计"] = {
            "总长度": n_total, "视觉token": n_image, "文本token": n_text,
        }
        report[f"[{tag}] und_tokens(last_hidden_state)"] = stats_lines
        report[f"[{tag}] VLM生成文本(对照)"] = gen_text

    return report


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.samples_dir:
        print(f"[setup] 从 {args.samples_dir} 读取已导出样本...", flush=True)
        samples = load_samples(Path(args.samples_dir))
        print(f"[setup] 共 {len(samples)} 个源: "
              f"{sorted(s['source'] for s in samples)}", flush=True)
    else:
        print(f"[setup] 收集各源 batch（config={args.config}）...", flush=True)
        collected = collect_one_batch_per_source(
            args.config, args.seed, args.max_batches)
        print(f"[setup] 共收集 {len(collected)} 个源: {sorted(collected)}", flush=True)
        if args.export_samples:
            export_samples(collected, Path(args.export_samples))
            return 0
        samples = [batch_to_sample(src, collected[src]) for src in sorted(collected)]

    if not args.vlm_path:
        raise SystemExit("--vlm-path 不能为空（export 模式除外）")

    print(f"[setup] 加载 VLM: {args.vlm_path}", flush=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from thinker_wm.models.und_branch import UndBranch

    processor = AutoProcessor.from_pretrained(args.vlm_path)
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.vlm_path, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    for p in vlm_model.parameters():
        p.requires_grad = False
    vlm_model.eval()

    # 与训练一致的 UndBranch（dim/ffn 不影响 extract_und_features 的 VLM forward）
    # 注意 dtype 必须是 bf16：_process_vlm_inputs_to_tokens 会把 image_embeds
    # 转成 und_encoder 权重的 dtype，与 VLM 的 bf16 inputs_embeds 不一致会报错。
    und_branch = UndBranch(
        vlm_preprocess=processor, vlm_model=vlm_model,
        dim=256, ffn_dim=512, num_layers=1,
        vlm_dim=vlm_model.config.text_config.hidden_size,
    ).to(device=device, dtype=torch.bfloat16)

    reports = []
    for sample in samples:
        print(f"[inspect] source={sample['source']} ...", flush=True)
        reports.append(inspect_sample(
            sample, processor, und_branch, vlm_model,
            device, args.gen_max_new_tokens, out_dir))

    report_path = out_dir / "vlm_inspect_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2))

    # 终端可读摘要
    for r in reports:
        print("=" * 100, flush=True)
        print(f"源: {r['source']}  机器人: {r['robot']}  首帧: {r['first_frame_size']} -> {r['frame_file']}", flush=True)
        print(f"原始任务文本(t5): {r['t5_input_message(原始任务文本)']}", flush=True)
        print(f"管线 vlm 消息   : {r['vlm_input_message(数据管线拼接)']}", flush=True)
        print(f"统一 task prompt: {r['unified_vlm_message(stage2 训练实际使用)']}", flush=True)
        for tag in ("pipeline", "unified"):
            tok = r[f"[{tag}] token统计"]
            print(f"  [{tag}] token: 总={tok['总长度']} 视觉={tok['视觉token']} 文本={tok['文本token']}", flush=True)
            for line in r[f"[{tag}] und_tokens(last_hidden_state)"]:
                print(f"    {line}", flush=True)
            print(f"  [{tag}] VLM 生成文本(对照): {r[f'[{tag}] VLM生成文本(对照)']}", flush=True)
    print("=" * 100, flush=True)
    print(f"报告已写入 {report_path}，首帧图像在 {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
