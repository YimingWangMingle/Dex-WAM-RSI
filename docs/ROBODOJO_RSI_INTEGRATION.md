# RoboDojo 训练代码与 RSI 运行系统接入

更新：2026-10-21。目标目录：`zxw-pod:/code1/zxw_data/pretrain/ThinkerWAM-RSI`。

本页覆盖 RoboDojo ARX-X5 版本。旧 RoboTwin 文档的 50 Hz、16 步执行窗口以及 aloha 统计不适用于此版本。此次保留 pod 上正在使用的模型、数据读取器、训练配置和训练启动程序，在其外部接入 RSI 控制层。训练配置以现有 `third_party/Thinker-WM/configs/rsi_robodojo_v1.yaml` 为依据。

## 1. 训练与部署的共同约定

- 机器人 `robodojo_arx_x5`，robot token 17；状态和动作均为 14 维，统一维度映射为 identity。
- 25 Hz；原生绝对关节动作，左右各 6 个关节（弧度）加 1 个夹爪，顺序为左臂、左夹爪、右臂、右夹爪。夹爪原生范围 `[0,1]`。
- 训练侧使用此机器人的 z-score 统计，运行端接收/下发原生值。WAM 推理接口负责反归一化，执行器不会再次归一化或静默裁剪。
- 视频与动作比例为 1:2。训练窗口 12 个未来视频帧／24 个动作，即 3 个 AR 块；部署每次生成 1 块，4 帧／8 步／0.32 秒，执行已批准前缀后重新观测。
- 三视图采用同一 field map：头部 320×426，上下两层中的下层为两个 160×213 腕部视图；整体居中放入 480×640 画布。
- `depth_condition_mode=rgb_only`：策略读取 RGB 与关节状态；模型内部深度预测分支保留。它不等于输入实测深度，也不自动提供经过标定的米制几何。
- 历史使用实际训练配置的最近连续帧、稀疏帧和夹爪事件。观测步号为原生控制步；`state[t]` 后执行 `action[t]`，反馈为 `state[t+1]`。
- 当前配置训练 joint 和 action-conditioned world 分支，没有据此宣称 IDM、在线权重更新或跨任务示范 ICL 已训练完成。

`environments/robodojo/training.py` 检查这些条件并生成动作协议、统计版本和 field-map 身份。不同协议在推理或下发前报错。

## 2. 整体结构与执行流程

```text
现有 RoboDojo 训练 → 完整 checkpoint + 原始训练 YAML
                             ↓ 显式封装、内容校验
                      只读候选 bundle
                             ↓ 严格加载
Agent（任务状态、经验、测量工具）→ TaskSkillRunner
                             ↓ 子目标 + 最多8步
                      ThinkerWAMBackend
                             ↓ 动作提案 + 独立预测文件
                      审核批准实际前缀
                             ↓
                      RoboDojoExecutor → 原生 step 回调
                             ↓ 每一步都记录
实测 RGB/关节状态 → ObservationStore → WAM HistoryBuffer
原生 success/score → 独立评估记录，不交给 Agent/WAM
                             ↓
trajectory + 独立逐段验证 → Data-RSI manifest → 后续训练候选
```

`ThinkerWAMBackend` 复用现有 `WAMInference.generate_action`、模型配置构造和严格权重加载。不会模拟另一个模型的 latent commit 协议。切换技能后，从观测存储重建每个已经执行步的历史；只执行三步时，剩余五步预测不会成为历史。

`RoboDojoExecutor` 是唯一动作下发者。整个请求先检查维度、动作协议、夹爪范围与剩余步数，再逐行调用传入的 `step_fn`。环境终止即停止；下发后回包丢失视为执行状态不明确，该执行器报错并拒绝重试。真实机器人使用时，`step_fn` 还须负责设备节拍与硬件限制；本模块不提供实时调度保证。

Agent 只获得许可的观测引用和任务反馈。原生分数存于执行日志/评估日志，不能用于生成动作或让 Agent 自判成功。`RoboDojoExecutor.native_result` 仅供试验记录器与独立评估器使用。

## 3. 训练产物接入命令

在目标代码根目录使用已有 `motus` Python 环境：

```bash
PY=/home/fengodchen/miniconda3/envs/motus/bin/python
$PY scripts/robodojo_rsi.py inspect --config third_party/Thinker-WM/configs/rsi_robodojo_v1.yaml
```

检查器只读取带 `.motus_checkpoint_complete.json` 标记、且具有支持格式权重文件的目录，不自动选择 latest，也不根据训练 loss 晋升。检查器不加载模型、不占用 GPU。完整标记只能证明保存完成，不能证明权重质量。

拿到明确的 checkpoint 后封装；以下变量由实验调用方指定实际路径：

```bash
$PY scripts/robodojo_rsi.py seal --config third_party/Thinker-WM/configs/rsi_robodojo_v1.yaml \
  --checkpoint "$CHECKPOINT" --output "$NEW_BUNDLE_JSON"
$PY scripts/robodojo_rsi.py verify --bundle "$NEW_BUNDLE_JSON"
```

`seal` 写入新的 JSON，引用原始权重，不复制或覆盖大模型。记录完整解析配置、field map、统计、源码指纹、完整标记和权重 SHA256。现有训练保存的 `config.json` 缺少 wm/history 等字段，不能单独用来重建模型，所以必须提供该运行实际使用的完整训练 YAML。封装会检查已有 common 配置，但**不能仅凭权重反向证明 YAML 来源**；调用方仍需保证运行出处正确。封装后变更对应源码或权重会使验证失败，需要重建并重新评估候选。

`acceptance` 初始全部为 false。JSON 完整性验证不等于 strict-load、闭环成功或配对评估通过。`load_robodojo_skill` 真正执行模型构造和 strict 权重加载后才返回技能；它本身不操控机器人，也不会把候选变成正式版本。

## 4. 接入现有环境的接口

训练 HDF5 字段与在线环境字段不是同一个接口，不猜测 XPL 的键名。调用方明确提供：

1. `ObservationBinding`：三路 RGB 的在线字段，以及四个有序关节/夹爪字段，或一个完整 14 维状态字段。
2. `RoboDojoObservationStore(root, episode_id, adapter, profile)`：每局新目录，保存许可的 RGB/状态和不可变内容引用。
3. `step_fn(native_joint14) -> NativeStep(live, terminated, native_result)`：下发一次，取得对应实测反馈；`live` 放许可输入，`native_result` 单独放评估信号。
4. `RoboDojoExecutor(..., max_control_steps=明确预算, trajectory=logger)`：预算由试验计划给定。
5. `load_robodojo_skill(bundle, store, device=...)` 返回 `ControlSkill`，注册进 `SkillRegistry`；`TaskSkillRunner` 连接具体 `AgentPolicy`、上下文与执行器。

以下是宿主程序中已有 `adapter`、`initial_live`、`step_fn`、`agent` 和显式试验参数后的组装片段；这些依赖必须由真实环境提供：

```python
from environments.robodojo.training import load_training
from skills.wam.thinker import load_robodojo_skill
from skills.contracts import SkillRegistry
from agents.context import AgentContext, TaskState
from execution.runner import TaskSkillRunner, JsonlEventLog
from runtime.robodojo_execution import RoboDojoObservationStore, RoboDojoExecutor
from runtime.trajectory_logger import TrajectoryLogger

_, _, profile, _ = load_training(training_yaml)
store = RoboDojoObservationStore(observation_root, episode_id, adapter, profile)
skill = load_robodojo_skill(bundle_json, store, device="cuda:0")
logger = TrajectoryLogger(trajectory_root, episode_id, {
    "checkpoint": skill.identity.checkpoint,
    "action_stats_version": profile.action_stats_version,
    "camera_transform": store.camera_transform,
    "runtime": "robodojo-executor/v1",
})
try:
    executor = RoboDojoExecutor(store, initial_live, step_fn,
                               max_control_steps=episode_budget, trajectory=logger)
    registry = SkillRegistry()
    registry.register(skill)
    runner = TaskSkillRunner(registry, executor, agent,
                            AgentContext(TaskState(task_instruction)),
                            sink=JsonlEventLog(store.root / "agent-events.jsonl"))
    result = runner.run(episode_id)
    # 原生 verdict 与逐段物理验证交给宿主评估器，不能从 result 的停止原因推断成功。
finally:
    logger.close()
```

Agent 的经验文件/上下文能力保留在 Agent 层。WAM 接收任务文本与实测历史；此次没有把经验 Markdown 拼进 WAM，也没有把历史条件重新命名为示范 ICL。

## 5. 几何工具与直接控制的边界

本次同步了之前 RSI 的测量工具、标定 RGB 几何计算、操作诊断、经验文件访问和技能切换模块。细节见 [PERCEPTION_TOOLS.md](PERCEPTION_TOOLS.md) 与 [CONTROL_SKILLS.md](CONTROL_SKILLS.md)。

RoboDojo 的新观测存储只提供模型所需拼接 RGB/状态，不自动产生几何工具所需的独立相机内外参、目标 mask、关键点或双视图对应点。宿主需以许可的原始图像和实际标定数据构建测量输入。没有可靠尺度或对应点时，应返回不可测/不确定，不能把预测深度当实测证据。

几何工具给出的笛卡尔位姿也不能直接作为 joint14 动作下发。启用直接控制前，需要 RoboDojo 对应的 FK/IK、关节限位、碰撞/可达性检查和原生关节动作协议适配。此次提供统一控制边界，但没有宣称已完成 ARX-X5 实机 IK 或三个任务的专用感知器。

## 6. 自我改进与数据边界

- Agent 层：读取版本化经验；记录测量、失败原因和子目标进展；通过明确审核后的代码/工具候选改进执行策略。
- WAM 层：完整训练产物固定成候选，以相同任务和预算比较。现有训练器继续作为权重训练入口；此次没有新增未经验证的自动 TrainingProvider 或后台训练任务。
- 数据层：执行器可写 `trajectory.jsonl`，包含初始实测、批准命令、每个确认执行行的实测 RGB/状态。真实原生终止时记录独立 outcome；Agent 停止或执行异常时由宿主评估器完成最终记录。
- 默认只使用实测后果监督。即使整局成功，动作监督仍要求独立 `segment_verification` 精确绑定 episode、命令 hash、实际执行末行和末行反馈。验证失败的动作不能变成正例。
- 仅 `train` 与 `development` 可进入收集。validation、holdout、test 和未知 split 均拒绝。原有 15 个 holdout episode 的开环损失不能当闭环任务成功率，也不进入训练数据。
- `build_training_manifest` 固定样本角色、监督模式与 lineage；从 manifest 到新的训练数据 release/实际训练任务仍需宿主训练管线接入和验收。本次没有启动重训或自动晋升。

## 7. 已验证与下一步验收

2026-10-21：相关 168 项 CPU 回归测试通过，其中新训练/执行接入测试 13 项。覆盖动作协议、部分执行、技能切换历史、训练产物封装、真实执行日志接入数据角色、原生结果隔离，以及原有历史/几何/控制逻辑。模型生成使用测试替身，不是已训练 WAM 的质量评估。

对三个任务（insert_tubes、pour_liquid_into_cup、stack_bowls）各一条真实 HDF5 episode 的首/中/末段共 9 个时刻运行只读审计：训练 reader 与在线 adapter 的 RGB 最大绝对误差为 0、状态误差为 0，所抽样的 `action[t]` 与 `state[t+1]` 误差为 0。复现入口：

```bash
$PY scripts/audit_robodojo_rsi_inputs.py --config third_party/Thinker-WM/configs/rsi_robodojo_v1.yaml \
  --catalog tmp/catalogs/robodojo_3task_v1/robodojo.episode_catalog.jsonl
```

本轮检查时，新 RoboDojo 训练仍运行，目标 checkpoint 根下未发现 step > 0 的完整训练权重；发现的 step 0 preflight 快照已被排除，不能当作已训练候选。尚未执行新权重 strict-load/GPU 推理，也未完成 XPL 环境键名和动作回调绑定、真实 Agent 交互或闭环基准。

后续验收顺序：明确完成权重 → 封装并严格加载 → 一个真实观测的 8×14 原生动作推理 → 接通明确的环境 step/许可观测接口 → 三任务闭环小样本 → 固定种子与预算的基线/RSI 配对实验 → 验证后收集数据并训练下一候选。每项单独记录证据，不以输入一致或 CPU 测试替代任务成功率。

## Joint24 六任务数据的当前约定

配置入口 third_party/Thinker-WM/configs/robodojo_joint24_v1.yaml。六任务为 insert_tubes、pour_liquid_into_cup、stack_bowls、
plug_in_charger、insert_key、fold_clothes。既有划分为570条训练、30条holdout；固定验证每任务20个窗口，共120个。
数据/配置变更后以实际 catalog 与配置审计为准，不能以历史数字代替检查。
沿用checkpoint对应的动作归一化，新增数据诊断统计不得静默覆盖。
检查入口 scripts/check_robodojo_six_task_release.py、scripts/test_robodojo_validation_inputs.py。
评测应固定任务、种子、观测、交互预算与开发数据权限，分别记录成功、失败、reset失败和不确定；
程序版本比较使用配对案例，heldout不能回流到候选提议或训练。
