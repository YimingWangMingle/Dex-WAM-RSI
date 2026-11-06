# 控制技能与辅助工具：接口和扩展

新增：[外层 RSI 选择策略、Agent 按需工具和自动 A2V](SELECTION_TOOLS.md)。

更新：2026-11-04。[架构](ARCHITECTURE.md) · [环境适配](BENCHMARK_ADAPTATION.md) · [经验工具](EXPERIENCE_GUIDE.md)

## 1. 命名约定

- **Skill（控制技能）**：根据任务/子目标提出机器人动作，并接收执行反馈。
  内置名称为 direct_control、wam_control、vla_control。
- **Tool（辅助工具）**：读取/检索经验、感知测量、预测动作后果等，不直接执行机器人动作。
- **Experience（经验材料）**：HMR 记录和 ICL 示范本身。读取这些材料的操作才是工具。
- **Backend（后端）**：实现上述能力的模型、服务或代码。WAM/VLA 是模型类型，包装后成为技能。

同一个 WAM 可支持动作生成技能，也可支持 A2V 预测工具。当前 A2V 已接在 WAM/PI0.5
候选审核流程中，尚未注册成 Agent 可自由调用的通用 Tool；不能直接调用不存在的 Measure("a2v", ...)。
详见 [A2V](INFERENCE_INTERFACES.md)。

## 2. ControlBackend 的三个基本方法

```python
from skills.contracts import BackendProposal

class RecoveryBackend:
    def __init__(self, load_observation, planner):
        self.load = load_observation
        self.planner = planner
        self.last_feedback = None

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.last_feedback = None

    def propose(self, request):
        measured = self.load(request.observation)
        memory = request.experience.to_dict() if request.experience else None
        # planner 由调用方实现；根据平台 state 结构、契约、预算生成动作。
        rows = self.planner(measured, request.subgoal, request.max_actions,
                            memory, self.last_feedback)
        return BackendProposal(tuple(tuple(float(v) for v in row) for row in rows))

    def observe(self, feedback):
        self.last_feedback = feedback
```

这是接入骨架，不是已实现的抓取恢复算法。propose 不能调用 env.step/take_action。
request 提供当前观测、子目标、动作契约、max_actions、历史和可选经验。
feedback 提供实际执行行、最终实测观测、终止状态和可选操作验证结果，不是整个原始提案都已执行。

SkillIdentity 必须声明唯一 name、backend_revision、action_contract、action_dim；
内置 wam_control/vla_control 名称还要求 checkpoint 身份。
ControlSkill 检查提案维度、有限值、长度、episode/观测绑定和待决提案生命周期。

## 3. 在 RoboDojo 完整入口注册

工厂只接收 **一个参数** load_observation；NativeSession.load 返回的 state 是原生通道字典。
planner 必须按 native joint14 的顺序转换，不把字典当成通用 state 向量。

```python
from skills.contracts import ControlSkill, SkillIdentity
from environments.robodojo.training import NATIVE_ACTION_CONTRACT
from execution.runner import CallSkill

def make_recovery(load_observation):
    return ControlSkill(
        SkillIdentity("recovery_control", "recovery/v1", NATIVE_ACTION_CONTRACT, 14),
        RecoveryBackend(load_observation, planner=plan_recovery),
    )

result = run_framework_episode(
    native_env, root=run_dir, action_spec=action_spec,
    skill_factories=[make_recovery],
    skill_reviewers={"recovery_control": review_recovery},
    selector=lambda view: CallSkill("recovery_control", "根据当前观测恢复抓取", 24),
    max_steps=800, max_model_calls=170,
)
```

plan_recovery、review_recovery、native_env 等由调用方提供；示例不自动加载模型。
此入口扩展技能不能占用三个内置名称，且必须匹配 NATIVE_ACTION_CONTRACT 和14维。
审核者必须与注册扩展技能一一对应。

审核回调签名是 `review_recovery(proposal, view) -> ReviewDecision`。这里 proposal 是**字典**，
不是 SkillProposal 对象：包含 proposal_id、subgoal、actions/candidates、violations、skill_experience 等。
审核不通过可返回以下类型的决定：

```python
from execution.review import ReviewDecision

def reject_recovery(proposal, view):
    return ReviewDecision(proposal["proposal_id"], "reject", "需要更多实测证据",
                          proposal["subgoal"])
```

批准时 decision="accept"，显式填 approved_steps，候选提案还需对应 candidate_id。
公共层仍会核对当前观测、提案身份、允许前缀和动作边界。审核者不能改写候选动作。

RoboDojo 会包装扩展审核回调的调用预算；一参数扩展工厂不会获得 ModelBudget。
如果新后端内部还发起模型请求，不能假定它自动纳入 max_model_calls。需要自建计量接线，
或优先使用下面的通用工厂接口。内置 Direct/主管/审核已有其现有计量路径。

## 4. 在通用入口注册

工厂接收 **两个参数** `(load_snapshot, model_budget)`，返回 ControlSkill。
通用快照的 state 是有序数值列表，images 是相机名到图片路径的映射。
选择和审核由 agent_factory 返回的 AgentPolicy.next(view) 完成；此入口没有 skill_reviewers 参数。

```python
from skills.contracts import ControlSkill, SkillIdentity
from execution.episode import run_episode

def make_skill(load_snapshot, budget):
    client = budget.client(raw_model_client, "recovery_controller")
    # make_planner 由调用方实现，必须真正通过这个 client 发请求。
    planner = make_planner(client)
    return ControlSkill(
        SkillIdentity("recovery_control", "recovery/v1",
                      environment.contract.name, environment.contract.dimension),
        RecoveryBackend(load_snapshot, planner),
    )

def make_agent(load_snapshot, budget):
    # build_agent 由调用方实现；返回 next(view) 可产出 CallSkill/Review/Stop 等的对象。
    return build_agent(load_snapshot, budget)

result = run_episode(
    environment, root=run_dir, action_spec=environment.contract,
    skill_factories=[make_skill], agent_factory=make_agent,
    max_steps=800, max_model_calls=170,
)
```

工厂示例里的 environment 只用于读取固定 contract；实际后端不应保留执行环境引用。
client.complete 的重试须走相同预算包装器；文件桥或审核函数可使用 budget.callback。
纯代码选择器不消耗模型调用数。工厂是可信代码，框架不拦截任意网络调用。

可运行的单技能/审核示例见 `examples/modular/fixture_driver.py`，经验消费见 experience_demo.py。
这些示例的确定性审核器只用于 CPU 教学，不是生产视觉审核策略。

## 5. 让技能使用 HMR / ICL

两个入口都支持同名配置，前提是已经装载对应任务的经验工作区：

```python
skill_experience = {
    "recovery_control": {"hmr": ["HMR.md"], "icl": ["ICL.md"]},
}
```

后端显式消费 request.experience；仅配置路径不保证 planner 使用文本。
内置 WAM/PI0.5 的原始推理输入不自动接受这些文本，主要由主管和审核者消费。
内置 EEFBackend 不直接读取 request.experience；Direct 通过 ContextClient 接入共享经验，
详见 [经验使用](EXPERIENCE_GUIDE.md#6-direct-如何真正读取经验)。

## 6. 预测、多候选与 A2V

BackendProposal 可携带 ActionCandidate 列表，每个候选包含 candidate_id、actions、prediction_ref。
预测引用必须对应候选的确切动作，不能在审核后再后处理成不同动作。
如果模型有显式 commit/discard 生命周期，在 prepare_execution/observe 等协议钩子中处理，
只将实际执行的前缀写入模型历史。不要在 propose 阶段把预测当作真实反馈。

已实现的 WAM/PI0.5+A2V 适配见 [推理说明](INFERENCE_INTERFACES.md)，新模型可复用控制协议，
但仍要实现自己的输入、预测、候选选择和实测反馈转换。

## 7. 新增辅助工具

经验读取已有 `ReadExperience("list"|"read_text"|"read_image", ...)`；它访问当前任务封存包。
只读感知工具通过 measurement_tools 注册 MeasurementTool，并由 Agent 发出 Measure 调用。
接口与示例见 [感知工具](PERCEPTION_TOOLS.md)。

相机转动、探测性接触等会移动机器人的操作应作为动作技能，不伪装成只读测量工具。
工具结果不能直接代替环境原生评分。A2V 的未来独立 Tool 接口也应遵守“只预测、不执行”的边界；
本次文档更新不新增该工具注册实现。

## 8. 反馈、切换与验证

切换技能时先解决旧待决提案、确认没有在途动作，再用实测观测重建状态。
拒绝提案产生零执行反馈；执行中异常不自动重试，也不抹掉已执行动作。
GuardedDirectBackend 的前后置验证区分 passed/failed/unknown，未知状态不能解锁依赖阶段。

新增技能至少验证：维度/契约不符时零执行；拒绝/越界时零执行；实际执行前缀正确反馈；
手动停止和预算停止；切换前后状态；经验关闭和开启；模型错误/超时；原生任务结果。
CPU 协议通过不替代真实仿真验收。


新模型技能推荐使用 ModelControlSkill，代码技能使用 ControlSkill；提交生命周期和后端版本见 [通用核心](HARNESS_CORE.md)。
