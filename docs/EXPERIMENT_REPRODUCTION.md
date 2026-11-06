# 现有实验调用链的兼容与对照

兼容目标是 `layout3-sixgroups-20261104` 的冻结代码与审核协议，以及其使用的
`wam-recovery-high-20261103` 服务补丁。来源文件摘要记录在
`environments/robodojo/experiments/source_manifest.json`。

## 保留的调用过程

实测观测 → PI0.5/WAM 提出候选 → 按实验组预测 → 文件桥审核 →
审核者选择执行、重采样、编辑或停止 → 公共执行器执行批准前缀 → 实测反馈。

编辑与重采样不移动机器人。编辑后的动作重新提出；启用预测的组重新预测、重新审核，
旧预测不授权新动作。ADAPT 可以先审核动作，选择请求预测后，再审核同一组动作。
它沿用原实验的额外审核计费，不转换成新的 `CallTool` 流程。

该入口关闭主管、外层 RSI、HMR/ICL 和额外几何工具。要比较新的自主选择能力，
应使用另一实验配置，不将新选择策略混入这批基线。

## 已移入的实验逻辑

- PI0.5 原生50步输出取前24步，夹爪裁剪、动作哈希校验、相同观测重采样种子推进。
- S83 采样8个候选，按原生 FK 的位置、姿态、夹爪距离确定性筛出3个，保留索引与哈希映射。
- execute / resample / edit / stop 协议；动作编辑只修改明确指定的行区间与通道。
- 非严格模式允许在当前实际提供的动作长度内选任意整数前缀。PI0.5 通常是1–24；
  WAM 仍受原服务执行窗口和当前提案长度约束，不擅自扩成24步。严格 Mode3 保持原规则。
- 审核包含最近两次决策时的实测观测、当前观测、两条简要执行反馈与初始关节状态。
- RGB 预测默认抽取偏移2、6、10、16、20、24；灰度深度、训练 log codec 米制深度展示，
  均保留原画面顺序和哈希。预测深度不作为实测输入。
- 原审核 schema、提示词、任务完成要求和审核配置（60次、high、独立会话）保留。

主组含义：V0 为 PI0.5 动作审核；V1 加 RGB/深度预测；R6 仅加6帧 RGB 预测；
ADAPT 按需请求 RGB 预测；D6M 加米制图例的预测深度；S83 在 R6 基础上采用8选3。
另支持 W0/W1 的 WAM 动作审核/预测审核入口。P0/WBASE 是附带的单候选基线，
不属于六组对照验收；P0 显式设 pool_size=1，避免旧启动脚本固定导出3导致候选数量冲突。

## 在新副本中调用

已有 native 环境及独立服务客户端时：

```python
from environments.robodojo.experiments.layout3 import run_experiment

def run_r6(env, world_client, pi05_client, action_spec, reviewer, output_dir):
    return run_experiment(
        env, group="R6", root=output_dir,
        model_client=world_client, pi05_client=pi05_client,
        action_spec=action_spec, exchange=reviewer, max_steps=800,
    )
```

`env` 由调用者 reset；`reviewer(public_packet)` 返回原协议字典。
省略 exchange 使用原 request/response 文件桥，等待600秒；回复必须匹配 request_id。
输出目录必须是新目录；调用者负责环境及客户端的关闭。环境变量沿用原实验，
因此一进程只跑一个 episode，不并发修改这些标志。

直接装配原有 `run_framework_episode` 时，显式传 `review_protocol="experiment-v3"`，
再配置经验包外的原实验资源。默认 `"standard"` 不会自动改变原有审核前缀规则。

原生启动入口是 `compat/launchers/run_layout3_experiment.py`，通过环境变量指定：

- `RSI_EXPERIMENT_GROUP`、`RSI_LIVE_RUN_DIR`、`RSI_LAYOUT_ID`、可选 `RSI_EPISODE_STEPS`。
- `ROBODOJO_ROOT`：原生仿真仓库；`RSI_MATERIAL_ADAPTER_DIR`：匹配部署的 local_material_assets 模块。
- PI0.5 组还需 `RSI_PI05_URL`。task/env/device/policy/host/port 参数仍使用原生 launcher 参数。

该入口保留原 layout 选择、官方800步限制检查、原生执行计时、异常记录和视频导出路径。
模型端需另启匹配 checkpoint/config/runtime 的服务；该脚本不会启动或替换任何现有服务。
本次没有启动真实仿真验证这个部署入口。

单次审核可用以下命令准备原提示词、schema、附件顺序和审核设置：

```bash
python scripts/prepare_layout3_review.py --group R6 --request /path/to/new-run/review-request.json --output /path/to/new-review
```

需要在能读取请求所指图片的位置运行（或先按原协议传输并核验图片）。此工具只准备输入，
不调用模型或写回实验。外部审核器应使用相同模型/推理配置与附件顺序，返回：
`{"request_id": "对应请求", "result": {"operation": "execute", "candidate_id": 0, "execute_steps": 7, "edits": [], "reason": "审核依据"}}`。
执行本身仍由公共执行器控制。

## 对照验证范围

```bash
python scripts/compare_experiment_protocol.py --reference-code /path/to/frozen/code --reference-launchers /path/to/frozen/experiment --output /tmp/new-protocol-comparison
```

对照器先校验冻结源文件摘要，然后启动两个独立子进程，分别加载旧快照和新框架。
它只从旧 launcher 抽取 bridge 函数与提示词生成语句，不执行启动器顶层代码。
模型、原生环境与 FK 使用确定性 CPU 替身，文件写入全新的验证目录，不连接真实服务。

比较包括：推理种子/状态/图像哈希、候选动作、RPC 顺序与参数、预测及审核图片哈希、
完整审核 schema/提示词、附件顺序、编辑/重采样、批准及实际执行动作、反馈、停止原因和预算。
只归一化输出目录、随机请求/提案标识和耗时，不忽略动作、图片内容、审核内容或预算。

这种对照验证的是调用协议与确定性输入输出等价；真实 GPU 推理、在线审核模型和仿真
重跑是否得到同一轨迹，仍需独立配对实验。不能把 CPU 对照直接称为真实成功率复现。


2026-11-04 验证记录：8组各5种基本场景，加 ADAPT 请求预测、预测后编辑、
R6 预测哈希损坏，共43项旧/新协议对照全部一致；33组集成回归和23项模块测试通过。
原实验与服务快照共192个 Python 文件在本次合入前后哈希未变化。
历史生成报告已按要求删除；验证结论保留在本文，脚本可重新运行。
