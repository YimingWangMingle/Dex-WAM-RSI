# 项目概览、仓库结构与运行环境

## 项目概览

**ThinkerWM Pretrain**：世界模型风格视频/动作模型（"Motus"）的分布式预训练代码库。

- **数据管线**：统一走 `libs/motus-data-factory`（包名 `motus_data`，git submodule）的 indexed（Episode Catalog + SamplePool）管线，经 `data_prepare/` 调 `open_mixed_training_data`：batch 内单源、按权重调度、内建就绪批次队列。SamplePool 经库的 `SupportsSamplePool` 注入点接入。**库只负责统一格式/加载/打包/传输；索引选择策略（当前为 `data_prepare/sampling_ranges.py` 的采样范围 + 内存动态随机索引）归 `data_prepare/` 所有。**旧 `data/` 包已删除；Ego4D 暂不支持。
- **模型**（`models/ubt_wm.py`，类 `UBT_WM`）：Wan2.2 TI2V-5B 视频主干（vendor 在 `wan/`，flow matching）+ 冻结的 Qwen3-VL-2B-Instruct VLM（语言/视觉条件）+ Motus 世界模型分支（`models/{video_branch,uni_branch,und_branch,action_branch,world_model}.py`）。训练经 Accelerate + DeepSpeed（ZeRO-1）、多机弹性 `torchrun`，在混合机器人数据（RoboTwin、Agibot、Agibot_beta、RobotMind HDF5）上训练统一模型。
- **运行**：无打包文件，脚本方式就地运行（入口脚本把仓库根加入 `sys.path`）。依赖（未声明，从 import 推断）：`torch`、`accelerate`、`deepspeed`、`transformers`、`diffusers`、`omegaconf`、`wandb`、`decord`、`cv2`、`h5py`、`imageio`、`safetensors`、`qwen_vl_utils`、`scipy`、`matplotlib`、`PIL`、`numpy`。

## 仓库结构

```text
configs/            训练 YAML、zero1.json、坏视频列表、CPU 冒烟配置
data_prepare/       数据接入层 + 索引构建工具（索引选择策略归本包；factory/sampling_ranges/pool_providers/source_readers + 4 个 build CLI，见 02_commands.md）
libs/motus-data-factory/  数据读取库（submodule，包名 motus_data）
docs/               MOTUS_MANIFEST_V1.md（manifest 规范，仅 build 期用）
models/             UBT_WM 及 video/action/uni/und 分支；action_stats.py（归一化统计量）
periodic_eval/      周期性定集评测（产物进 monitor「eval 评估」tab，见 06_periodic_eval.md）
scripts/            manifest 转换器、清洗/缓存工具、冒烟测试、评估
train/              train.py（入口）、sample.py（评估辅助）
train_debug/        micro_train.py（本机 debug 训练）、micro_backbones.py（替身）
libs/motus-monitor/  实时监控库（包名 motus_monitor；见 libs/motus-monitor/README.md 与 07_monitor.md）
utils/              checkpointer、scheduler、vlm_utils、通用辅助
wan/                vendor 阿里巴巴 Wan2.1/2.2 代码（T5、VAE、DiT、FM 调度器）
train_mix_elastic.sh  多机弹性 torchrun 启动脚本（规范入口）
train_micro.sh      本机极简 debug 训练脚本（单进程，无预训练权重）
H-Server-ssh/       H 集群 SSH/rsync 辅助脚本（含个人跳板机信息）
k8s_job_script/     生产 K8s Indexed Job 提交脚本（每 JOB 专属 headless service）
```

## 运行环境

- **本地**：本机只做代码开发、micro 训练（`train_micro.sh`）与冒烟验证。**生产**：经 `k8s_job_script/` 提交 K8s Indexed Job（`NNODES`×每节点 `GPU_NUM` 卡），容器内 `cd /code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain`（20260816 起；旧路径 `/data/chen.liao/thinkerwm-pretrain/thinkerwm-pretrain` 停用）跑 `bash train_mix_elastic.sh`；集群按 `JOB_COMPLETION_INDEX` 给 `NODE_RANK`、按 `${JOB}-headless` service 给 `RDZV_ENDPOINT`。**集群目录布局（20260816 起）**：代码与日志类产物（`outputs/`、`motus_monitor_data/`、`dataset_timing/`）在 `/code1/llm_team/chen.liao/ThinkerWM/` 下各自独立目录；checkpoint（`/data/chen.liao/thinkerwm-pretrain/checkpoints`）、ranges（`training_dataset_config/`）、`periodic_eval_cache` 仍留 `/data/chen.liao/thinkerwm-pretrain`。
- **Python 环境（20260812 起）**：默认环境为 conda `motus311`（Python 3.11 + zarr==3.1.5，支持 EgoVerse Zarr v3 数据源；由 `motus2` 改名而来，与 v0.0.4-py311-egoverse-test 镜像内环境名一致）。旧 `motus`（Python 3.10）保留作回退，但**无法读 EgoVerse**（`motus_data.egoverse` 有 Python>=3.11 且 zarr==3.1.5 的 fail-closed 门控）。本机与集群同镜像、同绝对路径 `/home/fengodchen/miniconda3/envs/motus311`，整目录 rsync 即可迁移；集群侧需在 motus311 里 `pip install -e <仓库>/libs/motus-data-factory --no-deps --no-build-isolation`（H200 无外网）。monitor dataset 预览 helper 默认 remote python 已指向 motus311；新 k8s job 脚本 `conda activate motus311`（历史脚本未改，勿回退）。
- 服务器代码副本需手动同步（`H-Server-ssh/push-to-h200.sh`）；本地改完代码正式跑前要推送。
- **禁令：未经用户明确允许，严禁向 H 集群同步任何代码或运行会改动其文件系统的命令**；需要时先在对话中征得同意。
- **集群临时运行改动代码的约定（用户 20260815 拍板）**：需要临时在集群跑尚未正式同步的改动脚本（如调试/统计工具）时，把文件 scp 到集群 **tmp 目录**（`/data/chen.liao/tmp/`）运行（配合 `PYTHONPATH` 指向正式仓库解析包导入），**不要直接覆盖正式仓库 `/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain` 里的训练代码**；正式同步仍走 `push-to-h200.sh` 整仓推送并需单独征得同意。

### H 集群无 rsync 时

同步先直接跑 `push-to-h200.sh`（H200 已装 rsync 3.4.4，脚本用 `--rsync-path=/home/fengodchen/opt/rsync/bin/rsync`）。报 `rsync: command not found`/`connection unexpectedly closed` 时才需重编译（本机 `/save/rsync-3.4.4.tar.gz` 有源码包，无则从 <https://download.samba.org/pub/rsync/src/> 下载；H200 无外网）：

```bash
scp -o "ProxyJump chen.liao@10.10.94.35" -P 10023 /save/rsync-3.4.4.tar.gz fengodchen@localhost:~/
ssh -J chen.liao@10.10.94.35 fengodchen@localhost -p 10023 \
  "cd ~ && tar xzf rsync-3.4.4.tar.gz && cd rsync-3.4.4 && \
   ./configure --prefix=\$HOME/opt/rsync --disable-lz4 --disable-zstd \
     --disable-xxhash --disable-openssl --disable-acl-support --disable-xattr-support && \
   make -j8 && make install && \$HOME/opt/rsync/bin/rsync --version"
```

装好前少量文件用 tar-over-ssh：`tar czf - <文件...> | ssh -J chen.liao@10.10.94.35 fengodchen@localhost -p 10023 "cd /code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain && tar xzf -"`
