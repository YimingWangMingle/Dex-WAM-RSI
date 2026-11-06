> 当前重构副本：RoboDojo 优先。最新完整文档从 [docs/README.md](docs/README.md) 开始；
> [架构](docs/ARCHITECTURE.md)、[运行](docs/RSI_USAGE.md)、[HMR/ICL](docs/EXPERIENCE_GUIDE.md)、
> [A2V](docs/INFERENCE_INTERFACES.md)、[新环境](docs/BENCHMARK_ADAPTATION.md)、[新技能](docs/CONTROL_SKILLS.md)。

# ThinkerWAM-RSI

ThinkerWAM-RSI 包含 **WAM 模型训练与推理**，以及 **机器人任务执行与改进框架**。框架通过统一任务循环组织 WAM、PI0.5、direct/EEF 和扩展技能，共享观测、ICL/HMR、动作审核与实测反馈；外层使用 RSI 提出程序或经验修改，通过配对评估选择并继承版本。

本项目面向研究与实验复现。运行需要自行准备模型权重、模型服务、原生模拟器和数据；当前不是克隆后即可运行任意 benchmark 的一键软件包。

## 包含哪些能力

- **WAM 训练与推理**：模型实现、训练入口、训练配置、数据处理、推理服务及评测工具。保留 Joint24 六任务 v1 和后续八任务 v2 配置。
- **统一任务执行**：技能选择、动作提案、审核、执行前缀、实测反馈、技能交接和轨迹记录。执行器是机器人动作的统一出口。
- **动作技能**：WAM、PI0.5、direct/EEF，以及符合当前执行器契约的自定义技能。支持动作审核模式；`action_only` 表示审核动作但不使用 A2V 预测，不代表跳过审核。
- **感知与测量**：通过只读工具提供几何和状态证据，供主管、技能及审核者使用。
- **辅助工具与经验**：读取 HMR/ICL、感知测量、A2V 后果预测；其中 HMR/ICL 内容是经验材料。A2V 支持固定审核链路与 Agent 独立工具调用，外层 RSI 可评估和继承技能／工具选择策略。历史经验不替代当前实测状态。
- **模型提供方**：DeepSeek/Kimi 客户端，以及无 API 会话使用的文件桥接。主管、EEF、审核与外层修复角色需要分别装配。
- **RSI 外层**：根据开发集失败提出候选修改，在相同验证案例上比较父子版本，选择并继承胜出版本。

当前外层改进保持 WAM/PI0.5 权重固定，修改对象是允许范围内的控制代码、技能逻辑、提示词和经验。**模型训练是单独配置与启动的流程，不会由当前外层自动触发。**

## 运行流程

单局任务：

```text
任务与初始经验 → 实测观测 → 选择技能 → 动作提案
                                      ↓
下一轮观测 ← 执行与实测反馈 ← 批准前缀 ← 审核
```

外层改进：

```text
开发集运行 → 失败分析与候选修改 → 父子版本配对验证 → 选择与继承
                                                        ↓
                                             冻结版本 → 独立测试
```

固定版本评测期间保持代码、模型、初始经验和规则不变；开发与验证运行不能混入最终测试成功率。不同初始场景下，固定版本仍可依据观测选择不同技能和恢复动作。

## 从哪里开始

获取代码：

```bash
git clone git@gitlab.rd.ubtrobot.com:world-model/ThinkerWM-RSI.git
cd ThinkerWM-RSI
```

若最终修正尚未合入默认分支，先确认维护者指定的实验分支或提交。GitLab 访问权限不包含模型、数据及内部第三方依赖的访问权限。

建议按以下顺序准备：

1. 阅读 [使用与交付说明](docs/RSI_USAGE.md)，准备框架、模拟器和模型服务所需环境。`requirements-core.txt` 提供轻量核心依赖；模拟器和模型环境仍需分别准备。
2. 按 [DeepSeek/Kimi 接口说明](docs/LLM_INTERFACES.md) 配置提供方、兼容的模型 ID 和本机凭据。切换主管客户端不会自动替换审核角色；无 API 文件桥接需要会话持续处理请求。
3. 按 [WAM/PI0.5 推理与经验设置](docs/INFERENCE_INTERFACES.md) 准备服务、权重、动作契约和经验包。
4. 按 [任务运行说明](docs/RSI_USAGE.md) 准备环境 driver，先验证 reset、原生评分与短步执行，再运行完整单任务。
5. 单任务验证通过后，固定任务、seed、资源及预算，组织批量评测或外层改进。

可先运行客户端接口测试：

```bash
python scripts/test_llm_clients.py
```

该测试使用本机 HTTP 替身，不调用付费 API，也不启动模拟器。通过它不代表真实模型请求或机器人任务已验证。其他回归测试及依赖要求见交付说明。

当前 `compat/launchers/run_codex_eef_live.py` 仍包含原实验环境的固定路径和外部资产辅助模块依赖，新机器需先适配。不要直接把历史实验命令当成通用启动命令。

## 自定义技能、数据与 benchmark

**添加感知能力**：通过感知工具或 `measurement_tools` 接入，读取当前观测并返回测量结果及证据，不直接下发机器人动作。

**添加控制技能**：通过 `skill_factories` 注册 ControlSkill。RoboDojo 完整入口通过 `skill_reviewers` 审核；通用入口通过 `agent_factory` 提供选择与审核。两者工厂签名不同，见技能接口。

**添加训练或经验数据**：适配字段、动作语义、采样率、时间对齐和数据划分。训练数据还需与加载器及批组装兼容；ICL/HMR 数据通过封存经验包和任务映射接入。生成清单不等于已完成训练。

**添加在线 benchmark**：提供环境创建/reset、观测、动作执行、原生评分及资源生命周期适配。通用入口使用 Environment/ActionContract；外层通过 FrameworkSuite.episode_runner 注入执行入口。当前只内置 RoboDojo，新平台仍需实现并验证适配器。

详细契约与验收标准见 [扩展适配说明](docs/BENCHMARK_ADAPTATION.md) 和 [技能接口](docs/CONTROL_SKILLS.md)。扩展时复用核心任务循环、审核、经验和反馈机制。

## WAM 训练

训练代码集中于 `third_party/Thinker-WM/` 下的 models、train、data_prepare 和 configs。Joint24 配置与入口为：

- 六任务 v1：`third_party/Thinker-WM/configs/robodojo_joint24_v1.yaml`、`scripts/launch_robodojo_joint24_v1.sh`。
- 八任务 v2：`third_party/Thinker-WM/configs/robodojo_joint24_v2.yaml`、`scripts/launch_robodojo_joint24_v2.sh`。

启动前核对数据 catalog、采样范围、字段映射、归一化、初始化权重和分布式资源。上述配置含原实验环境路径，需要按实际环境准备依赖和数据，不应直接照搬启动。

训练数据、预训练权重、checkpoint、视频及运行输出独立存放，不随源码分发。数据契约见 [RoboDojo 训练与数据适配](docs/ROBODOJO_RSI_INTEGRATION.md)。

## 当前验证范围

- RoboDojo 已有在线任务装配；RoboColiseum 已有离线数据读取和经验接口，不能当作在线 benchmark 执行器。
- WAM 已验证真实推理和短步执行；PI0.5 已验证真实推理与 A2V 对接，此前部分执行被动作约束阻止。
- 已完成框架与接口回归测试，但尚未完成完整叠碗成功率、真实多轮外层改进收益或其他在线平台的统一验收。
- DeepSeek/Kimi 客户端已经过替身测试，真实账号、指定模型及所需视觉/工具调用能力仍需验证。现有 Kimi 参数适配不兼容所有模型系列。
- 内置 WAM/PI0.5 的 ICL/HMR 主要提供给主管与审核逻辑，不会自动转成策略原生演示输入或更新权重。

测试通过、真实推理通过、完整任务成功和外层改进有效是不同层次的结果，应分别报告。

## 代码结构（2026-11-05）

- `rsi/`：仅外层改进、版本、候选评估及技能/工具选择策略。
- `execution/`、`agents/`：单局执行与审核；模型客户端、上下文及预算。
- `skills/`、`tools/`：控制技能；A2V、感知、测量等辅助工具。
- `experience/`、`data/`：HMR/ICL 经验包；轨迹与训练清单。
- `environments/`、`environments/robodojo/`：通用环境契约与 RoboDojo 实现。
- `compat/`：可选旧导入支持，正常框架不依赖；显式启用方式见 [说明](compat/README.md)。
- `framework_cli.py`：通用命令实现，保留 `python -m rsi` 入口。
- `third_party/RoboProbe/`：现用 Joint/EEF 策略源码。
- `thinker_wm.py`：唯一模型加载入口；不加载模型、不注册旧导入别名。
- `third_party/Thinker-WM/`：世界模型训练、推理及依赖的实际源码。
- `scripts/`：当前服务启动、评测、数据工具和回归测试。旧实验启动脚本位于 `compat/launchers/`。
- `docs/`：使用、架构和适配说明。

RoboTwin 在线组件已删除；其训练数据读取器与历史训练配置保留。`adapters/` 实现目录已合并到 `environments/`，旧 runtime 和 adapters 入口仅保留在可选 `compat/` 内。

完整文档入口：[文档导航](docs/README.md) · [整体架构](docs/ARCHITECTURE.md) · [感知工具](docs/PERCEPTION_TOOLS.md) · [技能／工具选择策略](docs/SELECTION_TOOLS.md)。
