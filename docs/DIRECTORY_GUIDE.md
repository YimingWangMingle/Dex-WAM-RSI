# 当前目录逐项说明

更新：2026-11-05。本文描述实际目录，不把未来规划当作已完成迁移。
根目录已经没有 runtime/，也没有 adapters.py。以下路径均相对于仓库根。

## 1. 正式框架

### rsi/：外层改进

- framework_campaign.py：组织开发任务、父子候选配对验证、继承与保留。
- core_adapter.py：把任务和评测生命周期接到外层核心。
- proposal_adapter.py：整理失败证据，约束修改范围并物化候选。
- version_manifest.py：候选文件闭包、模型身份、内容哈希与冻结校验。
- selection_policy.py：允许候选修改的技能、工具和自动工具选择策略。
- __init__.py / __main__.py：包和 CLI 入口，不自动启用 compat。

### execution/：一次任务怎样运行

- episode.py：通用环境的单局装配、观测快照与资源生命周期。
- runner.py：Agent 指令处理、技能提案、工具调用、审核、执行与交接。
- review.py：审核决策、动作边界、执行前缀检查。
- feedback.py：执行后的实测操作反馈。
- trajectory.py：轨迹日志、附件、数据角色和动作哈希。

### agents/：模型调用与上下文

- llm_clients.py：模型服务客户端接口及提供方适配。
- deepseek.py：模型驱动的决策接线。
- context.py：共享任务状态、上下文预算和摘要。
- budget.py：控制、主管、审核、摘要等角色的调用计费。

这些是能力模块，不表示每次运行必然创建多个 Agent。固定技能流程可以不启用主管。

### skills/：产生动作的技能

- contracts.py：控制技能、后端、动作提案及反馈契约。
- direct/guarded.py：Direct 控制的约束包装；具体策略由装配时注入。
- wam/backend.py：WAM 服务后端；wam/thinker.py：Thinker-WM 推理接线。
- wam/core.py：WAM 技能与外层核心接口的适配。
- pi05/backend.py：PI0.5 客户端、动作契约、候选和反馈。

技能提出动作，实际执行统一经过 execution，不各自操纵环境。

新增 `skills/chunk_backend.py` 提供后端无关的 commit/discard 生命周期；contracts.py 中的
ModelControlSkill 显式要求模型身份。版本与安装说明见 [通用核心](HARNESS_CORE.md)。

### tools/：辅助决策的工具

- contracts.py：通用工具接口、预算、动作与预测绑定收据。
- prediction/a2v.py：A2V 动作后果预测入口。
- measurement.py：测量工具装配。
- perception/camera.py：原始相机数据适配。
- perception/geometry.py：几何计算；direct_target.py：Direct 目标处理。
- perception/tools.py：对外感知工具；interface.md / notes.md：工具说明。

### experience/：经验代码与经验材料

- packages.py：封装、校验、catalog 选择、单局冻结工作区与受限读取。
- README.md：经验包格式与接入方法。
- templates/hmr.md、templates/icl.md：供填写的经验模板。
- examples/hmr/、examples/icl/：人工构造的教学正文。
- examples/packages/：带 manifest、payload 和文件哈希的封存示例。
- examples/catalog.json：将示例任务映射到包的可加载索引。

HMR/ICL 是经验内容，读取它们是工具能力。真实实验的新经验放独立输出或经验库，
不能直接改动正在评测的冻结包。

### data/：框架轨迹与训练清单

- recovery.py：恢复过程分段、成功证据和数据角色。
- training_manifest.py：将满足条件的记录整理为训练清单。
- robocoliseum.py：离线数据与经验适配，不是在线环境。

此目录处理框架输出如何进入数据流程；模型原始数据读取器在 data_prepare/ 对应源码中。

## 2. 环境与平台实现

### environments/

- contracts.py：环境接口、动作契约、实测观测、原生结果。
- registry.py：平台注册与创建；当前内置平台仅 RoboDojo。
- robodojo/environment.py：将 RoboDojo 原生实例包装为通用环境。
- robodojo/framework.py：RoboDojo 的 Direct、WAM、VLA、审核与工具完整装配。
- robodojo/training.py：平台训练/部署契约、checkpoint 身份及封存。
- robodojo/prediction_tool.py：该平台的预测工具接线。
- robodojo/review_context.py：审核输入；action_revision.py：候选编辑和重采样。
- robodojo/metric_depth.py：预测深度展示；pi05_diversity.py：候选多样性筛选。
- robodojo/experiments/：冻结实验组、审核协议、来源摘要与复现入口。

### environments/robodojo/runtime/：RoboDojo 运行实现

这里的 runtime 表示平台执行时使用的具体组件，属于正式代码，不能随 compat 删除。

- robodojo_execution.py：把批准动作下发到原生环境并记录实测反馈。
- robodojo_a2v.py：候选预测、审核绑定与反馈 RPC。
- robodojo_mode3.py、robodojo_mode3_events.py：视觉审核流程与事件。
- robodojo_codex_bridge.py：文件或模型提供方与 Direct/EEF 策略的桥接。
- robodojo_deepseek_live.py、robodojo_deepseek_ik.py：模型控制与 IK 接线。
- robodojo_motion_plan.py：运动规划；robodojo_pose_control.py：位姿控制。
- robodojo_observation_pose.py：观测位姿；robodojo_run_limits.py：运行限制。
- robodojo_bowl_vision.py：碗视觉检测；robodojo_bowl_geometry.py：碗几何。
- robodojo_bowl_perception.py：综合感知；robodojo_local_geometry.py：局部几何。
- robodojo_contact_scene.py：接触/碰撞几何；robodojo_episode_memory.py：单局操作记忆。

单局记忆是运行状态，不自动等同于已发布的 HMR 经验包。

## 3. 模型与第三方代码

### third_party/Thinker-WM/

世界模型实际源码：模型、训练、推理、数据处理与相关依赖。
其中 runtime/ 是模型运行组件，包括历史缓存、VAE 缓存、动作后处理、预测记录、
恢复判断、原模型执行循环及模型侧 RoboDojo 策略。它与平台 runtime 的职责不同。
UPSTREAM.json 记录源快照哈希、已移除组件及本地导入路径修改。

### thinker_wm.py（根目录）

唯一模型加载入口，将 thinker_wm.models、thinker_wm.runtime 等模块定位到
third_party/Thinker-WM/。只设置包搜索目录，不加载 Torch、不注册旧别名、不修改 sys.path。
source_path() 提供配置和资源定位。third_party 下不再保留第二个 thinker_wm 目录。

### third_party/RoboProbe/

- policy/RoboDojo_Agent_L3_Inspect/：Joint 控制、动作类型、提示词、轨迹与 recipes。
- policy/RoboDojo_Agent_L3_Inspect_EEF/：EEF 控制、规划器接线与位姿处理。
- utils/：模型通信及实测画面处理。
- SOURCE.md、LICENSE：来源与许可证。

这是当前 Direct/EEF 仍在使用的策略源码，不是待删除的历史包。

### Thinker-WM 内部目录

以下目录只存在于 third_party/Thinker-WM/ 中，根目录不保留软链接：

- models/：模型结构与 inference_api。
- data_prepare/：字段转换、时间窗口和采样。
- configs/：模型训练、推理、数据与运行配置。
- train/、train_debug/：训练及微型网络调试。
- utils/：模型公共工具、配置和输出路径。
- monitor/、periodic_eval/：训练监控与周期评测。
- libs/、wan/：依赖库与视频模型源码。

RoboTwin 在线策略和传输已移除；历史训练数据读取器和配置仍保留。

## 4. 可选历史兼容

### compat/

- __init__.py、__main__.py：显式启用与旧脚本启动器。
- imports.py、module_aliases.json：旧模块名映射，保持新旧模块身份一致。
- legacy/adapters.py：旧 adapters 导入入口。
- legacy/runtime/：旧 runtime 包入口，没有业务实现。
- vendor.py：旧源码加载接口，正常代码已不使用。
- experiment_protocol_worker.py：与旧实验对照时使用的独立进程 worker。
- launchers/：旧 layout3、Codex/DeepSeek 原生实验启动脚本。
- tests/：兼容层专属验证。

只有这个 compat/legacy/runtime 是历史入口。正常发行可以不包含整个 compat/；
平台和模型各自的 runtime 是正式实现，必须保留。详见 [兼容说明](../compat/README.md)。

## 5. 使用、开发与生成物

- pyproject.toml：通用框架安装描述；执行依赖通过 execution extra 选择，模型/仿真仍独立提供。
- framework_cli.py：episode / campaign / platforms 命令实现，python -m rsi 转发到这里。
- examples/modular/：CPU 环境 driver、经验模式、外层候选继承与工具选择示例。
- scripts/：模型服务、原生任务启动、数据工具、回归与诊断；具体脚本依赖不同。
- tests/：正常框架的接口与行为测试，不要求兼容层。
- docs/：架构、用法、接入契约、经验与实验复现说明。
- __pycache__/：Python 自动生成缓存，不属于源码。
- .AGENTS_CONTENT/ 与 AGENTS.md：仓库维护约定；.git/：版本历史。

## 6. 本次运行检查

```bash
python scripts/smoke_stack_bowls_modes.py --output /absolute/path/new-smoke-output
```

该检查使用 stack_bowls 格式的 CPU 环境替身，调用正式单局执行器和正式 RSI 外层，
归档 Direct、WAM、VLA、交接、A2V 与父子候选验证证据。模型和修复器为确定性替身。
结果只说明接口和流程可运行，不代表原生 RoboDojo 叠碗成功或模型性能。
真实仿真需要独立显存、独立服务/会话及新输出目录，不能占用正在跑的实验会话。

### 2026-11-05 CPU 流程实测

已运行上述检查；生成报告与轨迹已按要求删除，以下为当次结果：

- Direct 单任务：6 步，分两段执行，反馈为 4+2 步。
- Direct → WAM → Direct：32 步，完成两次技能交接。
- VLA 单任务：27 步，执行前缀为 24+3 步。
- Direct → VLA → Direct：32 步，后续技能使用交接后的实测状态。
- WAM → VLA → WAM：12 步，三个审核/执行阶段。
- Agent 调用 A2V 后审核 WAM：16 步，预测工具和执行器链路贯通。
- Direct 外层 RSI：两个轮次为 inherited、retained；创建 6 次环境，每局 2 步；
  恢复运行不重复执行。候选修复器和成功条件均为测试夹具，不证明机器人策略提升。

本轮没有运行原生 RoboDojo 仿真或真实 WAM/PI0.5 推理：A6000 显存接近占满，已有叠碗实验在运行。
本检查禁用 GPU，不重置原实验服务，不改动原实验快照。
