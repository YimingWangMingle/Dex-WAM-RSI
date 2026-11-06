#!/usr/bin/env python3
"""VLM 统一 task prompt 调参脚本。

目标：让 agibot_beta（Scene/Task/Action 三段式标注）、robotwin（三视角 preamble
+ 任务句）、ewmbench（裸指令）三类输入经 Qwen3-VL 后输出结构一致的格式：
Scene / Task / Action（机器人动作）/ Outcome（预期后续物体状态）。

样本来源：
- outputs/vlm_probe_h200_samples/（H200 生产管线导出的 5 源样本，PNG+JSON）
- /save/ewmbench_stage2_20260731_v1_step6000_officialprompt_all_videos/
  （每 task 取 1 个 episode：gt.mp4 首帧 + prompt.txt 的 cleaned_instruction）

用法：
    python scripts/tune_vlm_prompt.py --vlm-path pretrained_models/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from thinker_wm.utils.output_paths import resolve_output_path  # noqa: E402
from thinker_wm.utils.vlm_utils import build_vlm_task_message  # noqa: E402  # V0 基线

# ---------------- 候选 prompt ----------------

V1_FORMAT = (
    "Scene: <one sentence describing the environment and objects involved, "
    "grounded in the first frame image>. "
    "Task: <one sentence describing the robot's goal>. "
    "Action: <one sentence describing how the robot's arm(s) will move>. "
    "Outcome: <one sentence describing the expected state of the involved "
    "objects after the action>."
)


def build_v1(task_text: str) -> str:
    return (
        "Please describe the current robot task in the following format: "
        f"{V1_FORMAT}\n"
        "Here is the data:\n"
        "First frame: <the first frame image provided with this message>.\n"
        f"Task: {str(task_text).strip()}"
    )


V2_FORMAT = (
    "Scene: <one sentence describing the environment and the key objects, "
    "described from the first frame image>. "
    "Task: <one sentence stating the robot's goal>. "
    "Action: <one sentence predicting which arm(s) the robot will use and "
    "how they will move (grasp / lift / move / place / open ... )>. "
    "Outcome: <one sentence predicting where the key objects will be and "
    "what state they will be in when the task is finished>."
)


def build_v2(task_text: str) -> str:
    return (
        "You are given the first frame of a robot episode and a task text. "
        "The task text may be a bare instruction, or may contain fields like "
        "'Scene Description' / 'Task' / 'Action'.\n"
        "Based on BOTH the image and the task text, output exactly the "
        f"following four lines:\n{V2_FORMAT}\n"
        "Rules: one sentence per line; describe the Scene from the image "
        "instead of copying the task text; do not output anything else.\n"
        f"Task text: {str(task_text).strip()}"
    )


V3_FORMAT = (
    "Scene: <one sentence describing the environment and the key objects, "
    "described from the first frame image>. "
    "Task: <one sentence stating the robot's goal>. "
    "Action: <one sentence predicting which arm(s) the robot will use and "
    "how they will move (grasp / lift / move / place / open ... )>. "
    "Outcome: <one sentence predicting where the key objects will be and "
    "what state they will be in when the task is finished>."
)


def build_v3(task_text: str) -> str:
    return (
        "You are given the first frame of a robot episode and a task text. "
        "The task text may be a bare instruction, or may contain fields like "
        "'Scene Description' / 'Task' / 'Action'. Based on BOTH the image and "
        "the task text, output exactly in the following format (one sentence "
        f"per field): {V3_FORMAT}\n"
        "Describe the Scene from the image instead of copying the task text; "
        "do not output anything else.\n"
        "First frame: <the first frame image provided with this message>.\n"
        f"Task text: {str(task_text).strip()}"
    )


VARIANTS = {
    "v0_current": build_vlm_task_message,
    "v1_4field": build_v1,
    "v2_4field_strict": build_v2,
    "v3_4field_inline_strict": build_v3,
}

EWMBENCH_DIR = Path(
    "/save/ewmbench_stage2_20260731_v1_step6000_officialprompt_all_videos")
H200_SAMPLES_DIR = _REPO_ROOT / "outputs/vlm_probe_h200_samples"


def load_h200_samples() -> list[dict]:
    records = json.loads((H200_SAMPLES_DIR / "samples.json").read_text())
    return [
        {
            "name": r["source"],
            "kind": r["source"],
            "t5": r["t5"],
            "pil": Image.open(H200_SAMPLES_DIR / r["frame_file"]).convert("RGB"),
        }
        for r in records
    ]


def _read_cleaned_instruction(prompt_file: Path) -> str:
    text = prompt_file.read_text()
    m = re.search(r"\[cleaned_instruction\]\s*\n(.*?)\n", text)
    return m.group(1).strip()


def load_ewmbench_samples() -> list[dict]:
    import decord
    samples = []
    for task_dir in sorted(EWMBENCH_DIR.glob("task_*")):
        episode_dir = sorted(task_dir.glob("episode_*"))[0]
        trial_dir = sorted(episode_dir.glob("trial_*"))[0]
        instruction = _read_cleaned_instruction(trial_dir / "prompt.txt")
        vr = decord.VideoReader(str(trial_dir / "gt.mp4"))
        frame = vr[0].asnumpy()  # 首帧 = 推理条件帧
        samples.append({
            "name": f"ewmbench_{task_dir.name}",
            "kind": "ewmbench(agibot_beta裸指令)",
            "t5": instruction,
            "pil": Image.fromarray(frame).convert("RGB"),
        })
    return samples


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    parser.add_argument("--gen-max-new-tokens", type=int, default=200)
    parser.add_argument("--output", default=resolve_output_path("outputs/vlm_prompt_tuning.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[setup] 加载样本...", flush=True)
    samples = load_h200_samples() + load_ewmbench_samples()
    print(f"[setup] 共 {len(samples)} 个样本: "
          f"{[s['name'] for s in samples]}", flush=True)

    print(f"[setup] 加载 VLM: {args.vlm_path}", flush=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.vlm_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.vlm_path, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    model.eval()

    variants = {k: VARIANTS[k] for k in args.variants.split(",")}
    results = []
    for sample in samples:
        print(f"[gen] {sample['name']} ...", flush=True)
        entry = {"name": sample["name"], "kind": sample["kind"],
                 "input_text": sample["t5"], "outputs": {}}
        for vname, builder in variants.items():
            msg = builder(sample["t5"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": sample["pil"]},
                {"type": "text", "text": msg}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs,
                               videos=video_inputs, padding=True,
                               return_tensors="pt").to(device)
            gen_ids = model.generate(**inputs, do_sample=False,
                                     max_new_tokens=args.gen_max_new_tokens)
            gen_text = processor.tokenizer.decode(
                gen_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
            entry["outputs"][vname] = gen_text
        results.append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    for entry in results:
        print("=" * 100, flush=True)
        print(f"【{entry['name']}】({entry['kind']})", flush=True)
        print(f"输入文本: {entry['input_text']}", flush=True)
        for vname, gen_text in entry["outputs"].items():
            print(f"--- [{vname}] ---", flush=True)
            print(gen_text, flush=True)
    print(f"\n结果已写入 {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
