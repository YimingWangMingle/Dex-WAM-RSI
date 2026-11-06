# OpenVid 数据工厂接入说明

## 1. 数据结论

OpenVid-1M 是纯视频文本数据源：

- 权威表为 `OpenVid-1M.csv`，一行对应一个完整 MP4。
- 主数据由 186 个逻辑 ZIP 构成，其中 170 个是普通 `.zip`，16 个由
  `partaa/partab` 两个物理文件拼成一个逻辑 ZIP。
- `OpenVidHD` 是完整 OpenVid-1M 的重复子集，不重复写入 Catalog。
- 数据没有真实 `state/action`，Catalog 中两者均显式标记为 absent。

旧版两个 Manifest 都是固定 window 文件，每行不是完整 Episode：

- 旧转换日志记录 CSV 共 1,019,957 行。
- 旧 Manifest 共 4,238,995 个固定窗口。
- 旧扫描只识别普通 `.zip`，因此漏掉分片 ZIP 中的 82,000 个视频。
- 旧 converter 因固定窗口要求丢弃的 284 个短视频，在新 Catalog 中仍作为
  完整 Episode 保留；是否满足某种训练窗口长度由 SamplePool 决定。
- `with_test` 版本只是在旧窗口上写 split；新 Catalog 不继承该划分。

因此，新 Catalog 不从 18 GB Manifest 复制窗口，而是以
`OpenVid-1M.csv + ZIP central directory` 重新建立完整视频枚举。
旧 Manifest 仅用于理解历史协议和结果审计，不再是运行时依赖。

## 2. Catalog

每个 MP4 只写一条 split-neutral Episode。主要字段示意：

```json
{
  "episode_id": "openvid:---_iRTHryQ_13_0to241.mp4",
  "robot": "openvid-null",
  "video": {
    "count": 241,
    "storage": {
      "type": "zip",
      "path": "/data/OpenVid/OpenVid_part31.zip",
      "member": "mnt/bn/.../---_iRTHryQ_13_0to241.mp4"
    },
    "meta": {
      "encoding": "mp4",
      "member_size": 1234567,
      "crc32": 123456789,
      "frame_count_source": "OpenVid-1M.csv"
    }
  },
  "state": {
    "count": 0,
    "storage": {"type": "none", "path": "none"}
  },
  "action": {
    "count": 0,
    "storage": {"type": "none", "path": "none"}
  },
  "texts": [
    {"index": 0, "kind": "caption_en", "text": "In the video, ..."}
  ],
  "meta": {
    "fps": 29.97,
    "seconds": 8.04,
    "aesthetic_score": 5.42,
    "motion_score": 1.57,
    "temporal_consistency_score": 0.99,
    "camera_motion": "static",
    "state_action_policy": "absent"
  }
}
```

分片 ZIP 仅把 storage 改成：

```json
{
  "type": "zip_split",
  "path": "/data/OpenVid/OpenVid_part73.zip",
  "member": "mnt/bn/.../example.mp4",
  "parts": [
    "/data/OpenVid/OpenVid_part73_partaa",
    "/data/OpenVid/OpenVid_part73_partab"
  ]
}
```

reader 将 parts 暴露为只读、可 seek 的逻辑字节流，不会重建几十 GB ZIP。

## 3. 全量构建

集群默认输出：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid/
```

构建脚本可重复执行。Catalog 阶段每 10,000 行 fsync 并保存恢复状态；
进程中断后重新执行同一命令会从最近检查点继续。Catalog 完成后再原子生成
SQLite offset index。

```bash
cd /data/jiyuan.wang/thinkerwm-datafactory-dev-20260729

LOG_DIR=outputs/openvid_catalog
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/openvid_full_catalog_$(date +%Y%m%d_%H%M%S).out"

nohup bash tools/run_openvid_full_catalog_build.sh \
  >"$LOG" 2>&1 &

echo "PID=$! LOG=$LOG"
```

查看日志：

```bash
tail -f "$LOG"
```

预期最终产物：

```text
openvid_1m_full_20260730.episode_catalog.jsonl
openvid_1m_full_20260730.episode_catalog.meta.json
openvid_1m_full_20260730.episode_catalog.quarantine.jsonl
openvid_1m_full_20260730.episode_catalog.index.sqlite
```

全量验收重点：

- 逻辑 ZIP 数为 186，普通 ZIP 为 170，分片 ZIP 为 16。
- Catalog Episode 数应与有效 CSV 行数一致，目标为 1,019,957。
- `error_counts` 和 quarantine 行数为 0。
- `matched_zip_members`、`unmatched_zip_members` 与 CSV/ZIP 总量一致。
- Catalog metadata SHA256、行数和 SQLite index 绑定一致。

## 4. 模型侧 SamplePool

Catalog 不做 train/test 划分，也不固定采样窗口。模型侧生成 SamplePool，
明确指定每个样本的帧索引。OpenVid 的 state/action 索引必须为空：

```json
{"format":"motus_sample_pool_v2","pool_id":"openvid_train_v1","catalog_id":"openvid_1m_full_20260730","samples":1}
{"episode_id":"openvid:example.mp4","video_frame_indices":[0,8,16,24,32],"state_indices":[],"action_indices":[],"text_indices":[0]}
```

train/test 互斥、帧间隔、窗口长度、质量分阈值等采样策略全部由模型侧
SamplePool 决定，不修改 Catalog。

## 5. 读取与训练前准备

完整数据流为：

1. SQLite index 按 `episode_id` 定位 Catalog 行，不全量载入 Catalog。
2. reader 打开对应普通或分片 ZIP，并将目标 MP4 member 解压到临时文件。
3. 同一个 Episode 的多个 SamplePool 切片先分组，合并所需帧索引。
4. MP4 只提取一次，联合解码所需帧，再还原各个 sample。
5. DataLoader 组 batch。
6. ReadyBatchQueue 在 CPU 侧预取；训练进程从 queue 消费。

通用 indexed 输出保留真实语义：

```text
video: [T, C, H, W]
state: [0, 0]
action: [0, 0]
text: caption_en
```

当前 Motus 模型 batch 协议仍要求定长 state/action 张量。仅在显式配置以下
选项时，adapter 才会补零以兼容旧模型接口：

```python
adapter_options = {
    "num_video_frames": 48,
    "action_video_fps_ratio": 1,
    "video_loss_weight": 1.0,
    "action_loss_weight": 0.0,
    "state_action_policy": "absent",
    "allow_missing_state_action": True,
    "placeholder_state_action_dim": 26,
}
```

补零值不是数据集真实模态，batch metadata 会写
`synthetic_state_action=true`。Stage1 必须保持 action loss 为 0。若后续模型
原生支持无 state/action 输入，应使用自定义 adapter，直接消费空模态而不补零。

## 6. Queue 闭环验收

Catalog 和 index 生成完成后运行：

```bash
cd /data/jiyuan.wang/thinkerwm-datafactory-dev-20260729

ROOT=/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid
OUT=outputs/openvid_pipeline_smoke

PYTHONPATH="$PWD" python tools/smoke_openvid_pipeline.py \
  --catalog "$ROOT/openvid_1m_full_20260730.episode_catalog.jsonl" \
  --index "$ROOT/openvid_1m_full_20260730.episode_catalog.index.sqlite" \
  --output-dir "$OUT" \
  --batch-size 2 \
  --num-video-frames 4 \
  --frame-stride 8 \
  --height 240 \
  --width 320 \
  --queue-size 2 \
  --num-workers 0
```

成功标志是 `OPENVID_PIPELINE_SMOKE_OK`，并同时检查：

- 英文 caption 非空；
- 同 Episode 多切片可组成 batch；
- video shape 正确；
- action/state placeholder 全零；
- action loss 全 0；
- queue 能正常生产和消费 batch。

## 7. 已注册 Release 与标准 API

完整 Catalog、索引、真实 Episode、双源/十一源混训、多 rank 和 queue 门禁通过
后，数据工厂已注册：

```text
openvid:20260730_full_v1
```

普通数据访问不需要再传 Catalog/index：

```python
from motus_data import open_dataset

repository = open_dataset("openvid")
episode = repository.get_episode("openvid:---_iRTHryQ_13_0to241.mp4")
frames = episode.read_video([0, 8, 16, 24, 32])
caption = episode.read_text([0])
repository.close()
```

训练数据入口沿用既有公开调用方式：

```python
from motus_data import open_training_data

pipeline = open_training_data(
    "openvid",
    sample_pool="/path/to/openvid.sample_pool.jsonl",
    batch_size=2,
    output_adapter="motus_model_batch_v1",
    adapter_options={
        "num_video_frames": 48,
        "action_video_fps_ratio": 1,
        "video_loss_weight": 1.0,
        "action_loss_weight": 0.0,
        "state_action_policy": "absent",
        "allow_missing_state_action": True,
        "placeholder_state_action_dim": 26,
    },
    num_workers=0,
    queue_size=2,
)

batch, stats = pipeline.next()
pipeline.close()
```

这里的 SamplePool 仍由模型侧提供。注册 release 只冻结完整 Episode Catalog、
reader 和默认工程参数，不定义生产训练/验证采样策略。
