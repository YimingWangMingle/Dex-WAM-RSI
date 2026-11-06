# ThinkerWAM-RSI 使用说明

新增：[外层 RSI 选择策略、Agent 按需工具和自动 A2V](SELECTION_TOOLS.md)。

更新：2026-11-04。适用于 RoboDojo 优先的独立重构副本。
[文档导航](README.md) · [架构](ARCHITECTURE.md) · [HMR/ICL](EXPERIENCE_GUIDE.md) · [A2V](INFERENCE_INTERFACES.md)

## 1. 选择要做的事情

- 验证框架安装：运行 CPU fixture episode，不需要模型服务。
- 跑现有 RoboDojo Direct/WAM/PI0.5：使用原生 evaluator 回调中的 `run_framework_episode`。
- 接新平台或全新控制技能：使用 `Environment` + `run_episode`，见适配说明。
- 比较无经验/仅 HMR/仅 ICL：冻结控制配置与预算，只切换经验包和消费路径。
- 真正运行 RSI：用 campaign 让候选经开发证据提出、配对验证并继承。

## 2. 代码位置和依赖

A6000 独立副本（已有实验原目录未被替换）：

```text
宿主机：/mnt/llm_team_76/xiongwei_zhao/ThinkerWAM-RSI/code/ThinkerWAM-RSI-modular-20261104
已有挂载容器内：/rsi/code/ThinkerWAM-RSI-modular-20261104
本地同步副本：/home/ubt/tmp/rsi-modular-20261104
```

其他机器使用自己的 checkout 路径，不复制这些主机绝对路径作为通用配置。
Python >=3.10；以下只安装通用执行核心依赖：

```bash
python -m pip install -r requirements-core.txt
python -m rsi platforms
```

预期平台注册列表只有 `robodojo`。该命令不证明仿真器已安装。
真实 RoboDojo 需要对应模拟器、场景资产、机器人、规划器和 XPolicyLab；WAM 需要现有
Torch/CUDA 推理依赖与匹配 checkpoint；PI0.5 使用其独立 JAX/openpi 服务环境。
`requirements-core.txt` 不覆盖这些环境。模型服务、模拟器可以分别部署。

## 3. 先运行一个 CPU 示例

```bash
python -m rsi episode \
  --driver examples/modular/fixture_driver.py \
  --config examples/modular/episode.json \
  --output outputs/first-episode
```

输出是两步替身环境运行，写入实测快照、framework-events.jsonl 和 episode-summary.json。
示例策略为确定性代码，模型调用数为0。每次使用新的输出目录。
这验证入口/协议，不验证真实任务成功率。

`episode` CLI 调用 driver 的 `create_environment(config["environment"])`，然后 reset，
再调用 `build_episode(env, config["controller"])` 得到工厂配置，最后在 finally 中 close。
配置中的 platform 字段由 driver 解释；CLI 不会因为填了字符串就自动创建仿真环境。

## 4. 现有 RoboDojo 原生运行的共同准备

以下代码放在原生 evaluator 的 episode 回调中，环境必须已经初始化并 reset。
`native_env`、输出路径、模型连接与策略实例都由实际实验提供。

```python
from pathlib import Path
from environments.robodojo.framework import run_framework_episode
from runtime.roboprobe.policy.RoboDojo_Agent_L3_Inspect import deploy

# native_env 由 evaluator 提供；不要用 CPU fixture 替代真实环境。
action_spec = deploy._action_spec(native_env)
common = dict(root=Path(run_dir), action_spec=action_spec,
              max_steps=800, max_model_calls=170)
```

`run_framework_episode` 不创建/重置模拟器、不启动模型服务。调用方负责关闭其创建的客户端。
环境动作是绝对 joint14，左臂6关节、左夹爪、右臂6关节、右夹爪，25Hz。
夹爪和关节边界由真实 action_spec 核对，不用文档中的示意数值替代。

## 5. Direct：Agent 直接控制

```python
result = run_framework_episode(
    native_env, **common,
    direct_policy=eef_policy,
    supervisor_client=None,
)
```

`eef_policy` 必须是已有可用的 EEF 策略；它负责生成提案，框架执行并回传实际执行量。
只有 Direct 时无需主管选择技能。若要让 EEF 模型读取框架 HMR/ICL，上面的最小调用还不够，
要接 `ContextClient`，见 [Direct 经验接线](EXPERIENCE_GUIDE.md#6-direct-如何真正读取经验)。

历史 `compat/launchers/run_codex_eef_live.py` 是特定原生实验启动器，支持 direct/hybrid。
其 framework 路径会配置主管、上下文和几何；`RSI_CONTROL_MODE=direct` 不表示主管必定关闭。
希望精确控制角色时，使用上面的 Python 装配参数和独立 driver，不臆造关闭主管的环境变量。

## 6. WAM：基线、动作审核和 A2V

先按 [推理接口第1节](INFERENCE_INTERFACES.md#1-wam-服务配置) 启动匹配的服务并建立 wam_client。
以下片段三选一，每局使用新 root：

```python
# 无预测、单候选基线。
result = run_framework_episode(
    native_env, **common, client=wam_client,
    mode3=False, predict=False, candidate_count=1,
)
```

```python
# 只审核动作：审核者可看实测图像和动作，不生成预测。
result = run_framework_episode(
    native_env, **common, client=wam_client,
    wam_review_mode="action_only", candidate_count=3,
    reviewer=review_actions,
)
```

```python
# 通用 A2V：WAM 产生动作与后果预测，审核者选择候选/执行前缀。
result = run_framework_episode(
    native_env, **common, client=wam_client,
    mode3=False, predict=True, candidate_count=3,
    reviewer=review_predictions,
)
```

reviewer 是你实现的人工、API 或文件桥回调，必须实际消费审核输入；不存在自动的通用视觉评判器。
返回候选编号，或 `{"candidate_id": 0, "execute_steps": 8}`；可选前缀来自本轮
packet["allowed_execute_steps"]，不是任意整数。None 表示停止本次编排。
多候选或开启预测必须有审核者。

## 7. 严格视觉 mode3

服务用 `--mode3` 启动，episode 指定：

```python
result = run_framework_episode(
    native_env, **common, client=wam_client,
    mode3=True, reviewer=review_visual,
)
```

固定三候选；审核输入提供预测 RGB/depth，隐藏具体动作数值。
返回候选编号与允许前缀；不能修改候选动作。
省略 reviewer 时启用默认文件审核器，不是自动批准。独立响应程序需查看公开请求并写响应，
否则默认等待至超时。文件桥位置和 request_id 格式见 [A2V 与文件桥](INFERENCE_INTERFACES.md#6-a2v-审核与文件桥的完整使用顺序)。

## 8. PI0.5 与 PI0.5+A2V

PI05Client 只适配仓库已有的 HTTP/NPZ 协议，不等于任意 openpi 服务都能直接连接。

```python
from skills.pi05.backend import PI05Client
pi05 = PI05Client("http://<service-host>:19036", checkpoint="<server-checkpoint>")

result = run_framework_episode(
    native_env, **common, pi05_client=pi05,
    pi05_candidates=3,
    vla_review_mode="action_only", vla_reviewer=review_actions,
)
```

要启用 A2V，移除 vla_review_mode，并增加 `pi05_a2v_client=wam_a2v_client`；
将 vla_reviewer 换成消费预测的审核者。该 WAM 服务用通用 `--enable-a2v-review`、
同步执行且关闭 continuity；严格 mode3 服务不能用于 PI0.5 外部动作预测。
PI0.5 生成动作，WAM 仅预测这些确切动作的后果。

## 9. 仅 HMR、仅 ICL 和组合

这三项是经验配置，不替代上面的控制后端。详细包制作、Direct 接线与四组对照见
[经验专题](EXPERIENCE_GUIDE.md)。以下展示 WAM 审核使用仅 HMR：

```python
result = run_framework_episode(
    native_env, **common, client=wam_client,
    wam_review_mode="action_only", candidate_count=3,
    reviewer=review_actions,
    experience_catalog="/absolute/path/hmr-only/catalog.json",
    skill_experience={"wam_control": {"hmr": ["HMR.md"]}},
)
```

review_actions 从 packet["skill_experience"] 读取文本，并把内容传给实际审核模型。
仅 ICL 改成独立 ICL 包和 `{"icl": ["ICL.md"]}`；组合时显式列出两类。
关闭 skill_experience 不会清除已进入共享上下文的经验。严格对照使用不同封存包、
匹配 entry_points、新会话和独立输出目录。

## 10. 多技能和新技能

同时注入 direct_policy、client、pi05_client 只会注册多个后端。
要按状态切换，提供 `selector(view)` 或 `supervisor_client`，两者不能同时提供。

```python
from execution.runner import CallSkill

def selector(view):
    return CallSkill("wam_control", "当前完整子目标", 24)
```

这是固定选择的示意，不是智能切换策略。通用入口通过 agent_factory 返回 AgentPolicy。
新增技能按 [技能说明](CONTROL_SKILLS.md) 实现，两入口的工厂签名不同。

## 11. 预算、上下文与输出

- max_steps：真实执行行数上限；环境提前结束可以更少。
- max_model_calls：主管/控制/审核等被包装调用的预算。通用工厂必须使用传入的 ModelBudget 包装实际模型请求。
- WAM/PI0.5 的原始动作推理和预测 HTTP/RPC 不会自动计入上述决策模型预算；需要总服务调用预算时应在对应适配器另行计量，不能把 max_model_calls 当成所有模型网络请求总数。
- max_review_attempts：RoboDojo 同一提案的有限重审，默认3；通用入口没有此参数，agent_factory 自己管理策略。
- context_policy/context_summarizer：上下文保留与摘要；摘要模型也纳入调用预算。
- 原生成功判据、预算停止和工具成功不同，分别查看 native_success 与 stop_reason。

两入口都保存 `episode-summary.json`、`framework-events.jsonl` 和 `observations/`。
RoboDojo 完整入口还记录 trajectories，严格文件审核结束后归档 review-artifacts。
通用入口记录平台 identity、完整 ActionContract、native_score；没有自动生成同名 trajectories 目录。
模拟器创建或装配阶段失败可能早于 summary 写入，不能仅凭缺少 summary 判断动作成功与否。

## 12. 验证和常见问题

```bash
python -m unittest discover -s tests -v
python scripts/test_modular_refactor.py --output /absolute/path/outside-repo/regression.json
```

这些检查需要当前相应依赖；campaign 检查需真实 PhysicalRSI_core，完整模型导入检查需 WAM 环境。
纯核心用户可以先运行第3节示例。核心依赖与可选安装见 [通用核心](HARNESS_CORE.md)。

- 没有经验效果：检查 experience_read、proposal.skill_experience 及真实模型请求，不能只看路径已配置。
- 文件桥等待：检查 responder 是否启动、request_id 是否匹配、图像路径是否可读。
- 模式不匹配：action_only/PI0.5 外部预测误连严格 mode3 服务。
- 动作被拒绝：核对动作单位、顺序、逐步约束及剩余预算，不通过放宽边界掩盖适配错误。
- 切换后端异常：确认只反馈已执行前缀，历史来源是实测观测。

## 13. 真正运行 RSI 外层

先使实际 PhysicalRSI_core 在当前 Python 环境中可导入。用替身环境验证流程：

```bash
export PYTHONPATH="/path/to/RSI:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.modular.campaign_demo --output outputs/first-campaign
```

预期两轮结果为 inherited、retained。修复器是确定性示例，不证明自动发现有效机器人策略。
接真实任务时：

```bash
python -m rsi campaign \
  --driver /absolute/path/driver.py \
  --initial /absolute/path/parent.json \
  --config /absolute/path/campaign.json \
  --output /absolute/path/new-campaign
```

driver 实现 build_suite(initial, work_root, config, revision) 与 repair(parent, evaluations)。
FrameworkSuite 默认使用 RoboDojo 完整入口；通用平台显式传 `episode_runner=run_episode`。
max_rounds、max_candidates、max_trials、seconds_per_episode、profile 为 campaign 配置项；
完整 driver 约定见 [环境与外层适配](BENCHMARK_ADAPTATION.md#5-接入-rsi-外层)。

如果只优化 HMR/ICL：冻结控制代码与模型，把经验文本、manifest、catalog 和选择配置的
实际修改归入候选允许闭包；每轮生成新版本、重算哈希，不原地改父包。
`repair` 不会自动总结轨迹或重建 catalog，这需要修复提供方实现。固定经验对照与经验自我改进
是两种实验，详见 [经验专题第9节](EXPERIENCE_GUIDE.md#9-只改进-hmr-或-icl-的-rsi)。


## 运行中的实验与版本迁移

重构副本保留原有入口及默认固定审核路径；未显式配置选择策略或 A2V Tool 时，
不会自动启用外层策略改进或独立预测调用。接口回归通过不代表真实轨迹逐步相同。

正在运行的实验应继续使用启动时的代码快照、服务、模型和审核协议，不在一批评测中途替换。
layout3 六组实验的候选池筛选、动作修订、按需预测和画面打包补丁已合入显式兼容入口，
见 [现有实验复现与对照](EXPERIMENT_REPRODUCTION.md)。其他快照迁移前仍须与实际代码比较。

复用旧结果需要保存代码版本、初始场景、随机种子、checkpoint、采样设置、经验包、
审核提示词/图片、候选和执行前缀规则。新版本应在独立输出目录做固定输入对照，检查
候选动作哈希、预测输入/输出、审核请求、批准前缀与实测执行记录，再决定能否合并统计。
启用新的技能选择、工具调用时机或审核流程应作为新的实验条件记录。


新后端推荐声明 configuration/backend_version.json；已有 wam_version.json 原样兼容。核心安装与候选校验规则见 [通用核心](HARNESS_CORE.md)。


## 源码交付与运行环境

交付完整框架源码、pyproject.toml、requirements-core.txt，以及实际需要的 third_party 源码。
模型源码只有 third_party/Thinker-WM/ 一份，加载入口为根目录 thinker_wm.py。
不再交付根目录模型软链接。compat 可不交付，旧脚本需要时才显式启用。

通用 episode 使用 NumPy/Pillow 执行依赖；真实 RoboDojo/规划器、WAM 的 Torch/CUDA、
PI0.5 的 JAX/openpi 及外层核心分别准备，记录版本、checkpoint 身份、端口和共享路径。
权重、数据集、视频、运行日志及私人凭据单独管理，不混入框架源码包。
第三方来源与授权见对应 SOURCE、UPSTREAM 和 LICENSE 文件。

旧原生 launcher 可能含部署主机路径和外部场景资源。新机器优先配置通用 driver，
不要把历史实验路径作为平台标准。按 CPU 接口 → 模型短步 → 原生单任务 → 多局评测 →
外层收益的顺序验收，各阶段结果分别记录。报告输出显式指定仓库外路径。


### 模型与历史入口（2026-11-05）

新代码使用 `from thinker_wm.models.inference_api import WAMInference`。
配置和资源可通过 `from thinker_wm import source_path` 定位，例如
`source_path("configs/runtime/robodojo_joint24_v1.json")`。
模型模块从仓库根以 `python -m thinker_wm.train.train` 或
`python -m thinker_wm.scripts.eval_video_ckpt` 启动，并显式传入配置、权重和输出目录。
服务入口仍为 `python scripts/serve_robodojo_policy.py --help`；单局和外层仍使用 `python -m rsi`。

历史原生实验启动脚本位于 compat/launchers/，仅用于冻结协议复现。
根目录 models/utils 等旧导入只在显式启用 compat 后可用；不提供旧物理路径。

源码路径变化会生成新的部署封存哈希。旧实验继续使用原冻结快照和原封存文件；
新副本重新封存，不修改旧记录以伪装版本相同。
