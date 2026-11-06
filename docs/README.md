# RSI 文档导航

更新：2026-11-05。文档对应当前框架；内置平台为 RoboDojo。命令默认在仓库根执行。
CPU 合成检查、真实模型推理、原生任务成功和外层改进收益分别说明。

## 理解和使用框架

1. [目录内容指南](DIRECTORY_GUIDE.md)：每个目录和主要文件的实际职责，以及不同 runtime 的区别。
2. [整体架构](ARCHITECTURE.md)：单局执行、外层改进、技能、工具和平台之间的关系。
3. [使用与交付说明](RSI_USAGE.md)：依赖、启动入口、模式、预算、单任务、外层 campaign 和源码交付。
4. [通用核心与候选校验](HARNESS_CORE.md)：路径防护、后端版本、模型技能、核心安装及 WAM 兼容。

## 技能、工具与经验

- [控制技能](CONTROL_SKILLS.md)：Direct、WAM、VLA 及新增技能的接口和接线。
- [选择策略与工具调用](SELECTION_TOOLS.md)：Agent 选技能、独立工具调用、自动 A2V 及外层策略。
- [HMR / ICL 使用](EXPERIENCE_GUIDE.md)：经验制作、冻结、仅 HMR、仅 ICL、混合输入和对照。
- [推理与 A2V](INFERENCE_INTERFACES.md)：模型服务、动作预测、审核和 PI0.5 接入。
- [感知与测量](PERCEPTION_TOOLS.md)：只读测量工具和实测操作证据。
- [模型客户端与文件桥](LLM_INTERFACES.md)：提供方协议、角色装配和通信。

## 环境与实验

- [新增环境](BENCHMARK_ADAPTATION.md)：环境、动作、观测、执行器、评分及外层接口。
- [RoboDojo 数据与部署](ROBODOJO_RSI_INTEGRATION.md)：训练/部署契约及字段映射模板。
- [原实验复现](EXPERIMENT_REPRODUCTION.md)：冻结实验组、审核协议和新旧对照方法。

## 包内说明与示例

- [经验模板和封存示例](../experience/README.md)。
- [可选兼容层](../compat/README.md)：显式启用旧导入、无兼容层验证。
- examples/modular/harness_core_demo.py：仅标准库的 29 维合成动作/恢复交接。
- examples/modular/fixture_driver.py：CPU 单任务接口示例。
- examples/modular/experience_demo.py：none / hmr / icl / both 四种经验装配。
- examples/modular/campaign_demo.py：通用后端版本的候选继承与保留。
- examples/modular/selection_campaign_demo.py：技能/工具选择策略的外层修改。

训练回归使用的历史配置夹具位于 tests/fixtures/，不作为用户部署模板。
运行报告与轨迹不保存在源码目录。
