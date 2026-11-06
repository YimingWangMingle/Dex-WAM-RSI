# 通用后端与候选校验

更新：2026-11-05。本次选择性吸收 `feat/clean-harness` 提交 `2e8856d` 的通用化设计与回归案例，
保留本框架的 RoboDojo、WAM、VLA、A2V、经验输入及实际外层 campaign。
来源：[clean-harness](https://gitlab.rd.ubtrobot.com/world-model/ThinkerWM-RSI/-/tree/2e8856dbad7663dba1162365e610557993684665)。

## 候选文件边界

候选文件必须使用规范相对路径：不接受绝对路径、`..`、`./`、重复斜线或尾部斜线。
一个文件只能属于一个组件；新增文件必须归入可编辑组件并提供内容。
文件和目录前缀冲突在写入前拒绝。生成后再次比较固定组件与父版本。
这些检查在调用上游 verifier 之前执行，因此安装外层核心不会跳过本地边界检查。

正常候选仍可修改 `skill_selection / skills / tools / control / prompts / memory_rules`。
检查不等于操作系统沙箱，执行器、资源权限与独立任务评估的责任不变。

## 后端版本

新后端使用 `configuration/backend_version.json`，并纳入 configuration 闭包：

```python
from rsi.version_manifest import write_backend_version, build_harness

metadata = write_backend_version(
    candidate_root,
    backend_revision="my-controller/v1",
    action_contract="my-robot-joints/v1",
    control_hz=50,
    runtime_dependencies={"controller": "pinned-revision"},
)
files["configuration"].append(str(metadata.relative_to(candidate_root)))
harness = build_harness(candidate_root, "parent", files)
```

`files` 仍须声明完整组件闭包。后端版本与动作契约必须是非空字符串；频率必须有限且大于零；
运行依赖必须为 JSON 对象。读取及验证时也会校验这些字段，不仅在写入时检查。

默认 `build_harness` 接受通用版本或原有 `wam_version.json`。显式 `require_wam_version=True`
仍要求旧 WAM 文件；`False` 保留已有无版本文件的兼容入口，但新接入应使用通用版本。
若已有通用版本文件，它必须属于 configuration 并通过校验。

原 `write_wam_version` / `wam_version` 接口保留，已有文件、字段及哈希不自动改写。
`backend_version` 只读取已声明的通用版本，不猜测或隐式转换 WAM 的动作频率和依赖。
示例 campaign 已改用通用版本；已有 WAM 实验继续使用原版本信息。

## 代码技能与模型技能

- `ControlSkill`：代码策略、恢复程序或通用后端包装。
- `ModelControlSkill`：显式模型技能，任意技能名都必须声明有效 checkpoint 身份。
- 原 `wam_control` / `vla_control` 的身份检查保留，避免削弱既有校验。
- 自定义动作维度与技能名继续由版本化契约声明，不限定为 RoboDojo 的 14 维。

通用显式提交生命周期在 `skills/chunk_backend.py` 的 `CommittedChunkBackend`。
`skills.wam.backend.WAMControlBackend` 是同一类的兼容名称，没有第二份实现。
只有真实支持 commit/discard 的生成后端才使用这个适配器，其他后端直接实现 ControlBackend。

## 通用证据

`rsi.proposal_adapter.evidence_bundle` 接受 JSONL 记录，新增汇总：

- `review_decisions`：审核决定。
- `skill_handoffs`：切换目标技能及次数。
- `execution_stop_reasons`：执行停止原因。

保留原 schema、`recovery_decisions` 和 `visual_mse` 字段，兼容 WAM 日志。
没有预测误差时 visual_mse 为 null。数值汇总使用标准库；该模块不再依赖 NumPy。
这是可供修复器使用的汇总接口，不自动替换所有修复器的上下文或调用策略。

## 核心安装与依赖边界

```bash
python -m pip install -e .
python -S -m examples.modular.harness_core_demo
```

技能协议、runner、审核、上下文、经验包、候选版本及证据整理可仅用标准库运行。
示例为 29 维合成提案与代码恢复，执行 3 行并切换一次，不是真实机器人验证。

通用 episode 的图像/数组处理还需要执行依赖：

```bash
python -m pip install -e '.[execution]'
```

pyproject 不打包 compat、模型/仿真源码或权重。wheel 适合通用框架和协议调用；
真实 RoboDojo、WAM、RoboProbe 路径仍需完整源码副本、第三方依赖和对应运行环境。
外层 campaign 仍需另行提供实际核心依赖。不能把核心标准库化理解为全框架零依赖。

## 回归

```bash
python -S -m unittest discover -s tests -p test_harness_generalization.py
python scripts/test_without_compat.py --output /absolute/path/outside-repo/regression.json
python scripts/smoke_stack_bowls_modes.py --output /absolute/path/outside-repo/new-modes-run
```

报告及临时打包目录放在仓库外，不创建 validation/。
测试使用 CPU 替身及独立临时目录，不连接运行中的模型服务、不重置仿真或修改实验快照。
