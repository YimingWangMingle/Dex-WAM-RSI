# 外层 RSI 选择策略与内层 Tool 调用

本页描述已实现的接口。现有固定 WAM/PI0.5+A2V 审核链路仍可直接运行；
只有显式配置 SelectionPolicy/TaskTool 才启用新路径。

## 分工

- **Skill**：Direct、WAM、VLA 等动作提案能力，由公共执行器执行审核通过的动作。
- **Experience**：HMR/ICL 内容；读取经验是工具操作，不产生动作。
- **Tool**：A2V 对已提出的动作预测后果。预测成功不是批准，更不是任务成功。
- **内层 Agent**：在本轮冻结的可用技能、工具中选择，必要时预测，再提交审核。
- **外层 RSI**：修改候选策略文件，经 development/validation 比较后继承或保留版本。
  单次 Agent 调用工具不等于 RSI。

当前 `TaskTool` 注册接口支持 `action_prediction`。经验仍用 `ReadExperience`，
测量仍用 `Measure`，均受选择策略控制；没有把所有历史接口改名为 CallTool。

## 策略文件

将 `skill_selection/policy.json` 纳入 harness 的 `skill_selection` 文件清单：

```json
{
  "skills": ["wam_control", "vla_control"],
  "tools": ["experience", "a2v"],
  "automatic_tools": [],
  "decision_mode": "agent"
}
```

`skills` 和 `tools` 必须是宿主已注册能力的子集。测量工具使用
`measurement:<注册名称>`；不开放经验时不得同时配置技能经验注入。
`automatic_tools: ["a2v"]` 表示每份技能提案产生后自动预测全部候选；空列表表示按需调用。
`decision_mode: "agent"` 在 RoboDojo 中将 pending 阶段交给 selector/supervisor；
`"fixed"` 保留既有专用审核回调。通用 `run_episode` 的 agent_factory 始终负责命令，
该字段不会替换用户提供的 Agent。

宿主创建 `FrameworkSuite(..., selection_policy_file="skill_selection/policy.json")`。
每次 trial 都从验证过的候选目录读取该文件，冻结为 `SelectionPolicy`，
记录配置摘要和候选 revision。`FailureEdits` 可以编辑这个文件。
候选 `configure` 不得返回 `selection_policy`、`tool_limits`、`task_tools` 或
`required_prediction_skills`；后面三者由宿主 `resources` 提供。

宿主约束示意：

```python
from tools.contracts import ToolLimits

host_options = dict(
    tool_limits=ToolLimits(max_calls=32, max_candidates=3, max_action_rows=24),
    required_prediction_skills=("wam_control", "vla_control"),
)
```

这意味着即使候选取消自动调用，也必须按需取得对应 A2V 预测才能执行。
候选策略直接隐藏必需 A2V 会在初始化时失败。预算耗尽会停止 episode；失败推理也计次。
`max_seconds` 是调用返回后的结果有效性检查，不是抢占式中断，RPC 超时需由宿主客户端配置。
候选 Python 仍是可信代码执行环境，并非针对恶意代码的安全沙箱。

## Agent 按需调用

技能提出动作后，`AgentView.pending` 提供 proposal/candidates；
`available_tools`、`tool_interfaces`、`tool_usage` 提供可用能力和预算状态。

```python
from execution.runner import CallTool

def request_prediction(view):
    p = view.pending
    ids = [c["candidate_id"] for c in p["candidates"]] or [0]
    return CallTool("a2v", dict(
        proposal_id=p["proposal_id"],
        observation_id=view.latest_observation.observation_id,
        candidate_ids=ids,
    ))
```

调用结果出现在下一轮 `pending["tool_predictions"]` 和 prediction 上下文中。
随后提交原有 `Review(ReviewDecision(...))`。执行器检查动作边界、观测是否过期、
审核前缀，并在宿主要求时验证预测的 tool、proposal、候选 ID、观测、动作内容。
预测不能授权另一候选，也不能跨 proposal 或观测复用。切换技能/解决提案后清除预测收据。

DeepSeekAgent 已提供 `call_tool` schema。预测图片若需要给多模态模型看，宿主需提供
`prediction_loader(receipts)`，返回 `[{"label": "candidate/frame", "data_url": "data:image/png;base64,..."}]`。
加载器应验证服务产物路径并限制帧数；返回路径文字本身不会自动变成模型可见图片。
所有预测图片带“预测而非实测”标签，不能进入 measured history。

## RoboDojo 的 A2V 服务

`run_framework_episode` 支持 `a2v_tool_client`，使用**独立的通用审核服务连接**，
与动作服务及已有 `pi05_a2v_client` 分开。它会同步实测历史、调用 `framework_predict`，
验证返回候选和动作哈希；不会调用 `framework_select` 或向环境发动作。
请用现有服务启动流程启动另一个通用服务，并设置不同端口/日志目录。

```python
from rsi.selection_policy import SelectionPolicy
from environments.robodojo.framework import run_framework_episode

# env、actor_client、predictor_client、selector 由宿主构建。
def run_with_tools(env, actor_client, predictor_client, selector, root):
    return run_framework_episode(
        env, root=root, client=actor_client, mode3=False, predict=False,
        selector=selector, a2v_tool_client=predictor_client,
        selection_policy=SelectionPolicy(
            ("wam_control",), ("a2v",), decision_mode="agent"),
        required_prediction_skills=("wam_control",),
    )
```

A2V 要求完整模型 horizon 的最终原生动作（当前常用 24×14），不补齐短动作、
不平滑、不把 EEF 坐标冒充原生关节动作。WAM 工具模式会保留可用完整提案；
Agent 应请求完整 horizon，再审核较短执行前缀。剩余预算不足完整 horizon 时应停止，
不能靠填充动作绕过检查。Direct 需要先有兼容的完整原生动作提案才能接此工具。
严格 Mode3 视觉盲审保持原流程，不开放独立工具入口。

工具 revision 绑定服务报告的 checkpoint、生成设置、动作/相机版本、horizon 和实现摘要；
宿主仍须使用不可变模型产物，并在 driver_revision 中标识外部资源。连接生命周期由宿主管理。

## 可运行验证

在已配置 RSI 的副本环境中：

```bash
python -m examples.modular.selection_campaign_demo --output /tmp/rsi-selection-demo
python -m unittest discover -s tests -v
python scripts/test_a2v_tool.py
python scripts/test_framework_native.py
```

示例父策略选择 small 技能并按需预测，子策略选择 large 技能并自动预测；
技能实现、预测器和宿主预算均不改。真实 RSI 执行发展集评估与验证比较，
预期两轮结果为 inherited、retained。这是确定性 CPU fixture，证明接口与继承链路，
不代表真实 RoboDojo 成功率或学习出的策略提升。
