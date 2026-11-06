# 构建、运行与测试命令

无构建步骤、无 pytest/unittest 套件；验证靠冒烟脚本（独立 CLI，`--help` 看参数）：

```bash
python scripts/smoke_indexed_mix_cpu.py --config configs/micro_train_debug.yaml --steps 8   # 数据管线端到端（CPU，本机 8 源）
python scripts/smoke_indexed_mix_cpu.py --config configs/smoke_indexed_mix_cpu.yaml --steps 8  # 同上（H200 集群 10 源）
python scripts/smoke_indexed_mix_cpu.py --config configs/smoke_stage2_20260812_v1_cpu.yaml --steps 32  # 同上（H200 集群 stage2_20260812_v1 五族 17 源；13 新源 ranges 用 scripts/build_smoke_ranges_20260812_v1.sh 冒烟抽样构建）
python scripts/smoke_per_source_cpu.py --config configs/smoke_stage2_20260812_v1_cpu.yaml --batches 2  # 逐源冒烟：每个 source 单独建管线强制消费（混合按权重采样覆盖不到低权重源时用）
python scripts/validate_motus_manifest.py ...        # manifest 校验（仅 build 期用）
python scripts/smoke_monitor.py                      # monitor 冒烟（无需 GPU）
python libs/motus-data-factory/tools/audit_ten_source_indexed_mix.py --weighted-batches 8  # 库级 10 源冒烟（H200，CPU）
```

训练启动（全部环境变量见 `train_mix_elastic.sh` 头部注释）：

```bash
TASK=my_run CONFIG_FILE=configs/<config>.yaml bash train_mix_elastic.sh            # 单机 8 卡
NODE_RANK=0 NNODES=2 NPROC_PER_NODE=8 RDZV_ENDPOINT=<node0_ip>:29601 bash train_mix_elastic.sh  # 多机：各机 RDZV_ID/ENDPOINT 同、NODE_RANK 异
```

**新建启动脚本 seed 规则（20260818 起）**：seed 由 `CONFIG_FILE` 的 `training.seed` 提供（见 `03_config.md` 训练 seed 规则），启动脚本不得把 seed 覆盖/硬编码为 0；如需为同一配置跑不同数据序列，脚本可传显式非零新值并在脚本头部注明。当前 `train_mix_elastic.sh`/k8s 脚本无 seed 参数，只认配置。

## 本机极简 debug

不加载预训练权重、单进程；真实 UBT_WM 架构，`MotusConfig.micro_train=True` 时 VLM/VAE 换成 `train_debug/micro_backbones.py` 随机初始化替身（T5 已移除无替身）；Wan DiT 用 tiny config：

```bash
python scripts/fetch_h200_micro_dataset.py --num-records 32 --output-dir micro_data   # 1) 从 H200 采数据（默认 32 条 RoboTwin aloha，1GB 上限）
python data_prepare/align_robotmind_micro_windows.py --micro-data-dir micro_data      # 2) robotmind slim hdf5 → 窗口对齐（幂等）
python data_prepare/build_micro_catalogs.py --micro-data-dir micro_data --output-dir micro_data/catalogs  # 3) 本地 Catalog（robotwin×4+robotmind×4；agibot/agibot_beta 本地不可建，H200 冒烟覆盖）
for n in robotwin_aloha robotwin_ur5 robotwin_arx robotwin_piper robotmind_tianyi robotmind_arkmobile robotmind_ark robotmind_tienkung; do
  python data_prepare/build_sampling_ranges.py --catalog micro_data/catalogs/$n.episode_catalog.jsonl --dataset $n --output micro_data/catalogs/$n.ranges.sqlite --num-video-frames 48
done                                                                                  # 4) 8 源 ranges
bash train_micro.sh                                                                    # 5) 训练（configs/micro_train_debug.yaml）
```

`train_micro.sh`：`CONFIG_FILE`/`MAX_STEPS`/`FETCH_DATA=1`（先采集）/`PREP_DATA=0`（跳过自动准备，默认开启且幂等）；日志 tee 到 `outputs/motus-micro/logs/`；`micro_data/` 已 git 忽略；换源用 `--manifest`/`--dataset-name`。micro 配置为 8 源等权混合，每源 4-32 条（~300MB）。`fetch_h200_micro_dataset.py` 特殊源：**agibot_beta** 用 `--seek-samples 2000` 字节偏移采样 + `--transcode-video-h264 --remote-ffmpeg <imageio_ffmpeg 自带 ffmpeg>`（AV1 本机无法解码，需在 H200 转码 H264）；**robotmind×4** 用 `--hdf5-slim --remote-python /home/fengodchen/miniconda3/envs/motus311/bin/python` 生成 slim hdf5（`video.frame_indices` 重写为 0..N），再跑步骤 2 align 切片对齐；超上限记录跳过（每源 `--max-bytes`）。

## 关键环境变量

`TASK`、`CONFIG_FILE`、`DEEPSPEED_CONFIG`（默认 `configs/zero1.json`）、`NNODES`、`NPROC_PER_NODE`、`NODE_RANK`、`RDZV_ENDPOINT`、`RDZV_ID`、`MAX_STEPS`、`SAVE_INTERVAL`、`LOG_INTERVAL`、`RESUME_CHECKPOINT`（完整续训）、`FINETUNE_CHECKPOINT`（仅权重）、`AUTO_RESUME_LATEST`（默认 1）。输出写 `<输出根>/outputs/motus-$TASK/`，日志 tee 到其 `logs/`。

## 输出根约定

（`utils/output_paths.py`）相对路径经 `resolve_output_path` 解析——`MOTUS_OUTPUT_ROOT` > 仓库上一级目录（探针探测可读写，`tempfile.mkstemp` 唯一文件名）> `/tmp/<仓库目录名>`（默认 raise 报错，仅 `MOTUS_ALLOW_TMP_OUTPUT=1` 放行；`train_micro.sh` 已自动置 1）。**本机不经 `train_micro.sh` 直接跑 `train/train.py`/`train_mix_elastic.sh`/分析脚本（`scripts/inspect_vlm_per_dataset.py`、`scripts/tune_vlm_prompt.py` 等）时需显式加 `MOTUS_ALLOW_TMP_OUTPUT=1`**，否则启动即报错——防止本机误跑把产物散落 /tmp。绝对路径（`--checkpoint_dir`、periodic_eval `cache_dir`）原样通过。**生产 ckpt 例外（20260816 起）**：集群仓库已迁 `/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain`（outputs/monitor 数据随上一级解析落 `ThinkerWM/` 下），但 ckpt 要留 `/data`，故生产 YAML 的 `system.checkpoint_dir` 写绝对路径 `/data/chen.liao/thinkerwm-pretrain/checkpoints`，不再用相对 `./checkpoints`。两启动脚本 `OUTPUT_DIR` 默认遵循同一规则。

## 跨节点数续训

靠 DeepSpeed universal checkpoint。`configs/zero1.json` 保持 `checkpoint.load_universal: false`——先试 legacy 分片格式加载；仅当报 ZeRO DP world-size mismatch（并行度变化）时，`UniDiffuserTrainer.load_checkpoint` 调 `_ensure_universal_checkpoint`：主进程用 `ds_to_universal`（带 `--inject_missing_state`——0.18.x 保存路径不写 `universal_checkpoint_info`，缺该 flag 转换必然断言失败）把 `pytorch_model/` 转成 universal `zero/` 目录，其余 rank 轮询就绪标记，随后置 `load_universal_checkpoint=True` 重试全量恢复（模型/optimizer/scheduler/iteration）；转换或重试失败直接 raise abort（不再降级为仅加载权重）。universal 加载不校验 DP world size，optimizer state 按当前 world size 重分片（要求保存端 bf16 optimizer state 带 `PARAM_SLICE_MAPPINGS`，DeepSpeed 0.18.x 无条件写入；Accelerate 以 tag `"model"` 存取）。**续训默认不重置 scheduler**（`resume.reset_scheduler` 默认 False，`step_count` 同步 `global_step`，LR 接着原计划走；想热启动新 LR 计划才置 True）。**续训/热启动跳过预训练 backbone 加载（20260817 起生效）**：`RESUME_CHECKPOINT`/`FINETUNE_CHECKPOINT`/`AUTO_RESUME_LATEST` 命中且 ckpt 路径真实存在时，`main()` 置 `config.model.load_pretrained_backbones=False` 并透传进 `MotusConfig`，`UBT_WM.__init__` 只建 VLM/Wan 架构（随机初始化，VAE 权重仍加载）、不读预训练权重——全部权重（含冻结的 VLM/VAE，ZeRO-1 ckpt 存完整 `module.state_dict()`）由 checkpoint 全量恢复；路径不存在则保持加载预训练权重兜底。

## checkpoint 离线评估

单进程单卡，`create_eval_data_pipeline` 顺序逐样本：

```bash
python scripts/eval_video_ckpt.py --config configs/<config>.yaml --checkpoint <ckpt_dir> [--start-index 0 --num-samples 8]
```

train/holdout 由 split ranges 表达（`eval_pool` 指向 holdout ranges；评估 pool seed 固定、epoch=0、不 shuffle，跨次可复现）；不再支持 `--dataset-split` 与按索引/manifest-id 随机访问。

## manifest 工具

`python scripts/clean_agibot_beta_manifest_by_action_guard.py --input <in.jsonl> --output <clean.jsonl> --guard-scan`；`python scripts/mark_manifest_holdout_split.py --input <in.jsonl> --output <out.jsonl> --holdout-fraction 0.01 --seed 20260713`。Motus manifest 只是**构建期**格式（转换脚本产物 → Episode Catalog），运行时不再读取。
