# 新环境与新仿真平台适配

更新：2026-11-04。当前内置平台仅 RoboDojo；RoboTwin 暂不作为接入目标。
本版已支持通用 Environment、run_episode 和 FrameworkSuite.episode_runner 注入。

## 1. 选择扩展层次

- 同一 RoboDojo 机器人和动作契约的新任务：提供场景创建/reset、任务指令、布局/seed 与原生评分。
- 不同机器人或仿真平台：实现下面的 Environment 和 ActionContract，再接通用 run_episode。
- 新离线数据集：处理字段、动作语义、时间对齐、来源与划分；不是在线环境接入。

不应为每个平台复制 TaskSkillRunner，也不把不同机器人简单补零/截断成 joint14。
模型能否服务新机器人还取决于训练动作空间与观测格式，环境接口一致不表示模型通用。

## 2. Environment 的必需方法

定义位于 `environments/contracts.py`：

- identity()：返回稳定的平台/场景/机器人/版本身份；加载经验时可包含 task。
- reset(case)：按固定任务和 seed/layout 开新一局，步号从0开始。
- observe()：返回 Observation(instruction, state, images, step, time_s)。
- execute(row)：恰好执行一行统一动作；只由公共执行器调用。
- is_terminal()：是否已到原生结束条件。
- outcome()：返回 Outcome(success, terminated, score)，无可靠成功判据时 success=None。
- close()：释放本局拥有的环境资源；driver 还须释放自建客户端。

观测要求：state 为契约顺序的有限数值，图像为 uint8 HWC RGB；step 连续且与实际执行行一致，
time_s=step/control_hz。不要把物体真值、原生 success/score 放进 Agent 可读观测。
模型有单独的观测编码器时，转换放在后端适配器，不改共享观测含义。

当前通用执行器要求 state 维度与动作维度一致，用于动作变化检查；不直接支持任意异维的
proprioception/action 组合。如机器人还需额外状态，应显式设计扩展字段与检查器，而不是静默混入。
当前每个 execute 是一个控制周期；多子步仿真需由适配器内部转换并保证时间语义一致。

## 3. 显式动作契约

下面是1维 CPU 教学契约，不是实际机器人参数：

```python
from environments import ActionContract

contract = ActionContract(
    name="fixture-position/v1", channels=("x",), units=("m",),
    control_hz=10, representation="absolute", frame="fixture",
    lower=(-1,), upper=(1,), max_delta=(0.2,),
)
```

真实环境需声明有序通道、单位、absolute/delta、坐标系、频率和每维界限。
name 是版本化契约标识；顺序或语义改变时必须升级，不能复用同名假装兼容。
技能声明相同 name/维度，固定资源提供对应边界。

注意：现有 BoundsChecker/max_delta 比较的是候选行与上一状态/行的数值差。
对于增量指令，这不自动等于绝对关节位置限制。新增 delta 平台要确认该检查语义是否适用，
必要时在受控适配层转换为绝对命令或新增正确的约束实现，再验收。

## 4. 创建与注册平台

可直接用 Environment 实例调用 run_episode，也可注册懒加载工厂：

```python
from environments.registry import register, create

# MyEnvironment 由你的平台适配实现，构造函数接收实际资源配置。
register("my_sim", MyEnvironment)
environment = create("my_sim", **platform_options)
```

register 是当前进程内注册，不持久化配置文件。新进程必须导入注册逻辑。
不能重复注册已有名称。`python -m rsi platforms` 的默认新进程仅显示内置 RoboDojo。

RoboDojo 已有新绑定，其场景构造由可信 driver 提供：

```python
from environments.robodojo.environment import RoboDojoEnvironment

environment = RoboDojoEnvironment(
    native_env, contract=contract,
    reset=reset_native, close=close_native, revision=platform_revision,
)
```

reset_native(native_env, case)、close_native(native_env)、native_env 的创建都由调用方实现，
以匹配实际模拟器版本。绑定只支持14维绝对关节，复用原生 ActionSpace 通道定义。
频率以真实观测与传入 contract 一致为准。通用 contract 不会自动替换旧入口的原生 action_spec。

CLI driver 实现这两个函数：

```python
def create_environment(environment_config):
    # 创建并返回尚待 CLI reset 的 Environment。
    return build_platform(environment_config)

def build_episode(environment, controller_config):
    # 准备模型/技能，返回通用 run_episode 支持的关键字参数。
    return {
        "skill_factories": skill_factories,
        "agent_factory": agent_factory,
    }
```

build_platform、skill_factories、agent_factory 是调用方实现，不是自动可用的库函数。
CLI 用 config["case"] reset，finally close；max_steps/max_model_calls/root/action_spec 由固定配置提供，
driver 的返回选项不能覆盖它们。可运行骨架参照 examples/modular/fixture_driver.py。

## 5. 接入 RSI 外层

单局通过后，再由固定 host driver 实现：

```python
from execution.episode import run_episode
from rsi.framework_campaign import FrameworkSuite
from rsi.core_adapter import TaskSpec

def build_suite(initial, work_root, config, revision):
    return FrameworkSuite(
        name="my-simulation-suite", initial=initial,
        tasks={"task_name": TaskSpec("完整任务指令", [1], [2, 3], 2, [4])},
        open_env=open_and_reset, resources=fixed_resources, close_env=close_trial,
        driver_revision=revision, work_root=work_root,
        max_steps=800, episode_runner=run_episode,
    )
```

这段是 host driver 接口骨架；open_and_reset、fixed_resources、close_trial 需实现：

- open_env(task, case)：创建并 reset 独立 Environment；无法准备返回 None。
- resources(env, case)：返回固定单局资源，必须含 action_spec；通用入口填 env.contract。
- close_env(env)：释放环境及该局拥有的客户端，即使 episode 抛异常。
- driver_revision：必须等于 CLI 传入值，并在 driver 配置中绑定真实外部资源/版本。

已有 RoboDojo 完整装配省略 episode_runner 即可；通用入口必须显式提供 run_episode。
外层一次 action 表示一个完整 episode，不能把核心 max_steps=1 理解为机器人只走一步。

候选文件 `control/framework.py` 的 configure(root, case) 返回允许选项：

```text
direct_policy, direct_backend_factory, selector, reviewer, vla_reviewer,
experience_catalog, measurement_tools, context_summarizer, context_policy,
skill_factories, skill_reviewers, skill_experience, agent_factory
```

这是总允许集，不表示两个 episode_runner 都接受每个参数。RoboDojo 入口用
reviewer/skill_reviewers；通用入口用 agent_factory。具体以函数签名为准。
固定资源与候选不能提供重复键，候选不能替换执行入口、动作约束、预算和评分。
候选代码是可信 Python，不是沙箱执行；本框架不替代候选审阅与进程隔离。

repair(parent, verified_development_evaluations) 返回若干
`{"id": ..., "reason": ..., "edits": {相对文件路径: 完整内容}}`，无改进时返回 []。
完整可运行示例见 `examples/modular/campaign_demo.py`。HMR/ICL 候选的文件与哈希处理见
[经验专题](EXPERIENCE_GUIDE.md#9-只改进-hmr-或-icl-的-rsi)。

## 6. 验收顺序

1. 无模型先测试 reset、步号、动作语义和 close；异常不遗留本局资源。
2. 检查图像、state、时间、契约和评分分离；没有评分时保留 unknown。
3. 用受控技能测试拒绝零执行、越界零执行、前缀反馈与执行异常禁止重放。
4. 接实际模型，核对相机/归一化/动作输出与训练契约，先短步再完整任务。
5. 连续多局验证无状态泄漏，记录场景版本、seed、模型身份、预算与结果。
6. 冻结开发/验证/测试划分，运行父子配对评估，再开始多轮 RSI。

布局数字不同不一定代表经验来源独立，应检查实际场景/示范划分。
CPU 替身通过、真实模型推理通过、完整仿真任务通过、批量成功率和 RSI 收益是不同验收层次。
