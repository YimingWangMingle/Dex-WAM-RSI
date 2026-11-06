# HMR 与 ICL：制作、配置、消费和对照实验

新增：[外层 RSI 选择策略、Agent 按需工具和自动 A2V](SELECTION_TOOLS.md)。

更新：2026-11-04。入口与控制模式见 [使用说明](RSI_USAGE.md)。

## 1. 先区分经验、控制和预测

ICL 是示范/任务材料等上下文；HMR 是历史尝试、失败、执行反馈和经验记录。
读取/检索 HMR、读取 ICL 示范的操作称为 **经验工具（Tool）**，材料本身称为 Experience。
当前经验工具支持列目录、按路径读文本/图片，没有自动的向量检索器或跨任务语义搜索。
两者都在 `experience/packages.py` 管理的经验包中保存。本框架不会仅凭文件名判断它是 HMR 还是 ICL，
通道由 skill_experience 的 icl/hmr 键指定。经验不能证明本局已抓稳或任务已成功。

Direct、WAM、VLA 是控制技能；HMR/ICL 是它们或 Agent/审核者可使用的材料。
A2V 预测候选动作的后果，和经验是否开启独立。可以用 HMR+Direct，也可以用 ICL+WAM+A2V。
不存在内置的 `CallSkill("hmr", ...)` 或全局 `enable_hmr=True` 参数。

“只做 HMR”有两种意思：只制作 HMR 材料（不运行机器人），或者运行任务时只开放 HMR 经验。
以下分别说明。仅 ICL 同理。

## 2. 只制作经验包，不运行环境

先从已经允许使用的开发轨迹整理材料。HMR 可记录任务、来源版本/布局、实测过程、失败点、
修复假设、适用条件与不确定性；ICL 可记录示范步骤、图像索引、动作语义和结果。
缺少证据时标明未知，不把模型推测写成已确认事实。保留独立测试集不参与经验制作。

假设准备目录如下，文本由使用者撰写或由修复提供方根据开发证据生成：

```text
experience-source/
  HMR.md
  ICL.md
  images/demo-001.png
```

下面是制作 **仅 HMR** 封存包的完整代码。它只复制 HMR.md，没有把 ICL 混进去；out 必须是新目录。

```python
from pathlib import Path
import shutil
from experience.packages import build_package, write_catalog

source = Path("experience-source").resolve()
out = Path("outputs/hmr-only-v1").resolve()
out.mkdir(parents=True, exist_ok=False)
staging = out / "source"
staging.mkdir()
shutil.copyfile(source / "HMR.md", staging / "HMR.md")
package = build_package(staging, out / "packages" / "stack_bowls", "stack_bowls",
                        metadata={"role": "hmr", "source_split": "development"})
write_catalog(
    out / "catalog.json", "hmr-only-v1", "packages", [package],
    {"stack_bowls": ["HMR.md"]},
    {"stack_bowls": {"bundle": "stack_bowls", "kind": "original_preparation",
                     "base_task": "stack_bowls", "reason": "开发证据整理，仅 HMR"}},
)
```

catalog 的 packages_root 相对 catalog 所在目录解析。包包含 manifest.json 和 payload/，
manifest 保存文件哈希，catalog 保存 manifest 哈希及任务映射。`kind` 可用值为
original_preparation、base_task_transfer、rsi_candidate；任务不存在时拒绝装载，不猜测相似任务。

**仅 ICL**：只将 ICL.md 和它引用的示范图像复制进 staging，entry_points 只列 ICL.md，
更换 revision 与 metadata。**两者组合**：封存两份文本及所需材料，entry_points 显式列两份文本。
entry_points 是自动读取的文本入口；不要把 PNG 直接当 read_text 入口，应在文本中索引图像。
需要图像时由 Agent 的 read_image 或自定义多模态装配器加载。

build_package 不是经验总结器，也不自动抽帧、筛选成功轨迹或学习策略。
封存后不要直接改 payload；修改经验应产生新包并更新 catalog。

## 3. 可直接运行的四组 CPU 示例

```bash
python -m examples.modular.experience_demo --mode none --output outputs/experience-none
python -m examples.modular.experience_demo --mode hmr  --output outputs/experience-hmr
python -m examples.modular.experience_demo --mode icl  --output outputs/experience-icl
python -m examples.modular.experience_demo --mode both --output outputs/experience-both
```

每个命令使用独立目录，创建对应经验包并通过真实 run_episode 接口注入技能。
`consumed-experience.json` 保存技能收到的经验，`episode/` 保存运行日志。
示例文字明确标记为教学占位材料，动作策略固定；它验证“传到了哪里”，不测量经验收益，
也不是完整 RSI 改进实验。none 模式不创建 catalog、不传 experience。

## 4. 运行任务时只用 HMR 或只用 ICL

RoboDojo 入口按 `native_env.task_name` 选择包。以下 kwargs 加到已配置的控制调用中：

```python
# 仅 HMR：包本身也应仅开放 HMR 材料。
hmr_options = {
    "experience_catalog": "/absolute/path/hmr-only-v1/catalog.json",
    "skill_experience": {"wam_control": {"hmr": ["HMR.md"]}},
}

# 仅 ICL：使用独立的 ICL 包。
icl_options = {
    "experience_catalog": "/absolute/path/icl-only-v1/catalog.json",
    "skill_experience": {"wam_control": {"icl": ["ICL.md"]}},
}

# 两者组合：文件均需实际存在于该包 payload。
both_options = {
    "experience_catalog": "/absolute/path/both-v1/catalog.json",
    "skill_experience": {
        "wam_control": {"icl": ["ICL.md"], "hmr": ["HMR.md"]},
    },
}
```

PI0.5 将技能名换成 vla_control，自定义技能换成注册名。不能配置未注册的技能。
每通道最多8个路径，每文件在技能专属注入时读取前4000字符；长文应提供简洁入口摘要。
这些是包内相对路径，不是任意主机文件路径。

通用入口还需显式 task 身份：

```python
result = run_episode(
    environment, root=run_dir,
    skill_factories=skill_factories, agent_factory=agent_factory,
    experience_catalog=catalog_path, experience_task="stack_bowls",
    skill_experience={"recovery_control": {"hmr": ["HMR.md"]}},
)
```

如果 environment.identity() 已包含 task，可省 experience_task。当前内置 RoboDojoEnvironment
的 identity 只含 platform/revision，因此通用入口中使用它时应显式传 experience_task。

## 5. 经验在哪里被消费

### 自定义技能后端

```python
def propose(self, request):
    experience = request.experience
    icl = list(experience.icl) if experience else []
    hmr = list(experience.hmr) if experience else []
    # 每项为 (包内路径, 文本)。显式传给你的 planner/model，不能只读取后丢弃。
    rows = self.planner(self.load(request.observation), request.subgoal, icl=icl, hmr=hmr)
    return BackendProposal(tuple(tuple(float(v) for v in row) for row in rows))
```

planner 是新增后端的实现，不是仓库自动提供的函数。它必须遵守预算与动作契约。
框架负责把选定经验传给 request，实际如何影响决策由后端定义。

### WAM / PI0.5 审核者

审核 packet 的 `skill_experience` 包含 identity、icl、hmr；每通道是 `{"path": ..., "text": ...}` 列表。
把它连同候选预测/动作、当前实测和历史反馈一起传给审核模型或人工界面。
框架同时提供 `framework_context`；其中可能含自动加载的 entry_points。

当前 WAM 和 PI0.5 原始模型输入不会自动拼接这些文本，也不会因此训练或更新权重。
经验可通过主管的子目标选择或审核者起作用。模型不支持示范输入时，不能宣称开了 ICL 就完成了
模型级 in-context action learning。需要这项能力时，应扩展模型适配器并验证训练输入契约。

### 共享任务上下文

RoboDojo 的 ExperimentAgent 在调用技能前自动 read_text 各 entry_point，然后提供给 selector/审核者。
通用 agent_factory 没有这一自动预读逻辑；它可以发出 ReadExperience，或使用技能专属注入。
ReadExperience 的普通文本窗口默认最多16000字符，技能专属注入的4000字符限制是另一层。

### ICL 图像

skill_experience 当前只自动注入文本。多模态示范需通过 `ReadExperience("read_image", path)`
或 ExperienceWorkspace.read_image 读取，交给实际支持图像的客户端。不要把经验图像当作本局实测相机帧。

## 6. Direct 如何真正读取经验

内置 EEFBackend.propose 不直接消费 request.experience，只调用 policy.act(observation)。
因此单独设置 `skill_experience={"direct_control": ...}` **不会自动把文本传进 EEF 模型**。
当前已有接线方式是 ContextClient，将框架共享上下文加入 EEF 的 complete 请求。

```python
from environments.robodojo.framework import ContextClient, run_framework_episode

context_client = ContextClient(raw_eef_client)
# build_eef_policy 是调用方的装配函数，应把 client 传给现有 EefAgentPolicy 构造器。
eef_policy = build_eef_policy(client=context_client)
result = run_framework_episode(
    native_env, root=run_dir, action_spec=action_spec,
    direct_policy=eef_policy, context_client=context_client,
    supervisor_client=None,
    experience_catalog="/absolute/path/hmr-only-v1/catalog.json",
    max_steps=800, max_model_calls=170,
)
```

这里 HMR 通过 catalog 的 HMR-only entry_points 进入共享上下文，再由 ContextClient 送给
实际 EEF 客户端；无需额外启动主管。只提供 context_client 参数但 policy 没有使用同一个
wrapper，仍不会生效。现有构造参考 `environments/robodojo/environments/robodojo/runtime/robodojo_codex_bridge.py`。

已有 RoboProbe 的 ICL loader（如 L3_INSPECT_ICL_TEXT_ROOT / L3_INSPECT_ICL_ROOT）
是另一条上下文入口。做严格 HMR-only 或 no-experience 对照时，同时检查这些设置、
recipe、外部文件桥拼装的提示词和历史会话。不要重复注入同一份经验。

## 7. 与 A2V 组合

```python
result = run_framework_episode(
    native_env, root=run_dir, action_spec=action_spec,
    client=wam_client, mode3=False, predict=True, candidate_count=3,
    reviewer=review_predictions,
    experience_catalog="/absolute/path/hmr-only-v1/catalog.json",
    skill_experience={"wam_control": {"hmr": ["HMR.md"]}},
    max_steps=800, max_model_calls=170,
)
```

HMR 帮助审核者理解历史失败，A2V 提供这次候选的预测后果；两者都要结合当前实测判断。
PI0.5+A2V 改用 pi05_client、pi05_a2v_client、vla_reviewer，技能名改为 vla_control。
严格视觉 mode3 的动作隐藏约束不因开启经验而取消；不要经经验或自定义日志向审核者泄露当前隐藏动作。
完整服务配置见 [A2V 接口](INFERENCE_INTERFACES.md)。

## 8. 无经验、HMR-only、ICL-only 的可解释对照

至少固定：模型/推理设置、控制模式、工具、布局、任务说明、动作和调用预算、评分标准。
每组使用独立输出目录与新会话，经验在所有评测局中冻结。

- 无经验：不传 experience 或 experience_catalog，取消技能专属注入，并检查控制器原生 loader/外部提示词。
- HMR-only：工作区只含允许的 HMR 材料，entry_points 只指向 HMR，技能注入只选 hmr。
- ICL-only：工作区只含允许的 ICL 材料，entry_points 只指向 ICL，技能注入只选 icl。
- Both：显式允许两种材料，记录准确版本。

`skill_experience={}` 只关闭专属注入，不隔离共享上下文和经验读取工具。
只改通道但继续加载含两种材料的同一包，不能保证严格 HMR-only/ICL-only。
不要为了关闭历史经验而删除本局执行反馈；本局实测历史是闭环控制的必要输入。

核验三层证据：catalog/manifest 哈希是否正确；日志中的 experience_read 和 proposal.skill_experience
是否符合本组；最终客户端请求是否包含期望材料。仅有读取日志不能证明模型实际使用了经验。

## 9. 只改进 HMR 或 ICL 的 RSI

如果目标是“让系统自己改 HMR”，需要启动外层 campaign，而不是只运行冻结经验 episode。
建议限制修复器仅编辑明确声明的经验文件，冻结控制、工具、模型和评分：

1. 父版本在开发案例运行，保留失败与真实反馈。
2. repair 基于开发证据输出新的 HMR/ICL 文本；不要使用验证反馈写回本轮经验。
3. 重建包 manifest 与 catalog 哈希，并把实际文件纳入候选闭包；如配置路径变化，同步 configure。
4. 父子版本在同一组验证案例配对运行，版本在整轮内冻结。
5. RSI 决定继承或保留，再从被保留版本开始下一轮。

当前可编辑闭包种类：skill_selection、skills、tools、control、prompts、memory_rules。
weights、configuration、dependencies 等默认冻结。可将经验包及 catalog 放入候选的
memory_rules 组件；catalog 的 packages_root 用相对路径，configure 使用候选 root
定位它，避免子版本仍指向外部父包。修改内容时必须同时更新包内部哈希；外层 closure 哈希
不能替代经验包自己的 manifest/catalog 校验。

FailureEdits/materialize_candidate 负责落盘与候选闭包，不自动生成经验总结或修复包内部哈希。
repair 提供方需要完成这些工作。ModelFailureRepair 只能修改显式暴露的文件，不会自动扫描全部轨迹。
若使用新文件，必须通过 new_files 声明所属可编辑组件。

“冻结经验对照通过”“经验修复链路跑通”“真实任务上反复改进有效”分别报告，不能混为一个验收结论。


仓库提供可直接校验的 [经验模板与示例](../experience/README.md)，包含 HMR、ICL 正文和封存 catalog。
