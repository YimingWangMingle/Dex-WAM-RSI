# WAM、PI0.5 推理设置与技能经验接口

更新：2026-11-04。本文针对 RoboDojo 完整装配入口
`environments.robodojo.framework.run_framework_episode`，旧 `environments.robodojo.framework` 导入保持兼容。
服务只推理/提出动作；环境由 evaluator 初始化和 reset，唯一执行器负责实际控制。
以下路径和服务地址是需替换的参数，不能直接当作已有部署。调用方负责关闭自建客户端。

## 0. Skill、Tool 与 A2V

WAM 生成动作时支撑 wam_control 技能；WAM 预测给定动作的后果时支撑 A2V 预测工具能力。
VLA（当前 PI0.5）支撑 vla_control 技能。技能只提出动作，执行由公共执行器负责。
A2V 保留控制后端的候选/审核协议，同时支持独立 TaskTool 调用。
本页介绍原有 predict、mode3、pi05_a2v_client；新增 a2v_tool_client、CallTool 和外层选择策略
见 [选择策略与工具](SELECTION_TOOLS.md)。

## 1. WAM 服务配置

在配置好 WAM 依赖和 XPolicyLab 的运行环境中执行：

```bash
python scripts/serve_robodojo_policy.py \
  --config-path "$TRAIN_CONFIG" \
  --field-map-path "$FIELD_MAP" \
  --checkpoint "$CHECKPOINT" \
  --runtime-config "$RUNTIME_CONFIG" \
  --log-dir "$POLICY_LOG_DIR" \
  --port 19041 --seed 0 --enable-a2v-review
```

- `config-path`：模型配置，必须与权重、机器人及数据归一化匹配。
- `field-map-path`：观测/动作字段映射；三相机和 joint14 的次序必须正确。
- `checkpoint`：本次固定权重路径；换权重应另建可追溯实验。
- `runtime-config`：下面的 JSON；不提供时 Joint24 使用 `third_party/Thinker-WM/configs/runtime/robodojo_joint24_v1.json`。
- `log-dir`：新的服务输出目录；`strict-load.json` 记录实际模型配置、推理参数和运行配置。
- `port`、`seed`：监听端口和推理基础随机种子。
- `enable-a2v-review`：启用通用候选提案协议；动作基线/action_only 也用此协议，是否生成预测由 episode 参数控制。
- `mode3`：启动严格视觉审核服务，替代通用 review 模式；不能用它预测 PI0.5 外部动作。

WAM 采样设置位于配置的 `model.inference`：

```yaml
model:
  inference:
    num_inference_timesteps: 10
    action_num_inference_timesteps: 10
    guidance_scale: 1.0
    action_guidance_scale: 1.0
```

上面只是字段示例，不是针对当前 checkpoint 的最优值。服务将这些字段传给
`models.inference_api.GenerationSettings` 中的 `num_inference_steps`、
`action_num_inference_steps`、`guidance_scale`、`action_guidance_scale`。
当前服务没有对应的独立 CLI 覆盖参数；使用匹配的配置副本并检查 strict-load.json。
这不会训练模型，也不应改动正在训练的配置。

运行设置示例：

```json
{
  "num_blocks": 1,
  "execute_steps": 16,
  "continuity": null,
  "prefix_cache_mode": "off",
  "prefix_cache_max_mb": 1024.0,
  "async_lead_steps": 0
}
```

`num_blocks` 是生成块数，不能超过模型支持的训练窗口；`execute_steps` 是通用模式的执行前缀。
Joint24 一块24行 joint14、12帧预测，25Hz；生成长度与实际执行长度分开。
`continuity` 是动作后处理配置：已有 Joint24 默认文件为 EMA/alpha=0.5，上面的 null 是显式关闭。
审核服务必须同步，`async_lead_steps=0`；PI0.5 外部动作 A2V 还必须关闭 continuity，避免预测另一组动作。
缓存只支持 off/fixed；不接受未知运行设置键。不要将异步配置直接用于审核服务。

## 2. WAM 调用接口

```python
from client_server.ws.model_client import WsModelClient
from environments.robodojo.framework import run_framework_episode

wam = WsModelClient(
    url="ws://<host>:19041", evaluation_id="rsi-eval",
    trial_id="trial-001", request_timeout_s=900,
)
# env 已由原生 evaluator 初始化并 reset；action_spec 为该环境的动作约束。
try:
    result = run_framework_episode(
        env, root=run_dir, client=wam, action_spec=action_spec,
        wam_review_mode="action_only", candidate_count=3,
        reviewer=review_actions, max_steps=800,
    )
finally:
    wam.close()
```

`review_actions(packet)` 返回 `{"candidate_id": 0, "execute_steps": 8}`，或 None 拒绝。
必须选择 packet.allowed_execute_steps 中的长度，不可修改候选动作。

其他模式仍调用同一入口：

- 单候选基线：`mode3=False, predict=False, candidate_count=1`，不设 reviewer/wam_review_mode。
- 通用 A2V：`mode3=False, predict=True, candidate_count=3, reviewer=choose_candidate`；
  choose_candidate 返回整数候选编号或 None，执行前缀由服务配置决定；
  也支持返回 {candidate_id, execute_steps}，长度取 packet.allowed_execute_steps。
  多候选或开启预测时不提供 reviewer 会拒绝装配。
- 严格视觉 mode3：服务使用 `--mode3`，episode 使用 `mode3=True, reviewer=review_visual`；
  review_visual 返回候选编号和允许的执行步数，审核包隐藏具体候选动作。

原生服务协议的 preview、select、feedback 由统一框架调用，不另写执行循环。
每一行实际执行都回传实测观测；未执行尾部和预测不能进入真实历史。

## 3. PI0.5 服务与推理设置

```python
from skills.pi05.backend import PI05Client

pi05 = PI05Client(
    "http://<host>:19036",
    checkpoint="/服务端实际使用的checkpoint路径",
    timeout=900,
)
identity = pi05.describe()
```

这是接入已有 PI0.5 HTTP/NPZ 实验服务的客户端。当前仓库没有统一的 PI0.5 权重加载/服务启动命令，
服务端仍使用已有实验部署；不能把 WAM 启动命令用于 PI0.5。
已核对 A6000 宿主机的服务实现：
`/mnt/llm_team_76/xiongwei_zhao/ThinkerWAM-RSI/runs/framework-matrix-20261101/pi05_server.py`。
该历史部署脚本固定 checkpoint=/pi05-checkpoint、端口19036，通过 RSI_LIVE_RUN_DIR 指定输出目录，
依赖专用 JAX/openpi 环境和权重挂载；它不是通用可参数化 launcher。此次没有启动或替换该服务。


- `GET /` 核对 config、checkpoint、动作长度、维度和归一化声明。
- `POST /infer` 接收 NPZ：state、instruction、seed、cam_high、cam_left_wrist、cam_right_wrist。
- 客户端输入 state 为14维，图像为 HWC uint8 RGB；内部将 cam_head 映射到 cam_high 并转 CHW。
- 服务配置固定为 `pi05_base_aloha_full_sim_arx-x5_seed_0`；内部32维、quantile_q01_q99 归一化，
  内部臂动作相对当前状态，返回已转换的 **50×14 绝对关节目标**。
- 框架取前24行，先将索引6/13夹爪裁剪到[0,1]，再做 A2V/审核。不再次反归一化或累加状态。
- `pi05_candidates`：1或3；`pi05_seed`：uint32基础种子，每次使用基础种子+当前控制步+候选编号。
- `timeout`：单次 HTTP 超时；错误不会自动重试。
- 当前 HTTP 协议不暴露 diffusion steps、temperature 或动态 horizon；不可向客户端传不存在的参数。
  改这些模型内部设置要扩展服务端身份协议并验证，不能静默改变当前实验契约。
- checkpoint 当前按路径比对，历史服务没有权重内容哈希声明。换权重需在实验资源身份中显式记录。

直接调用 `pi05.infer(state=..., images=..., instruction=..., seed=0)` 返回动作数组，**不会执行环境**。

## 4. PI0.5 进入统一框架

```python
result = run_framework_episode(
    env, root=run_dir, pi05_client=pi05, action_spec=action_spec,
    pi05_candidates=3, pi05_seed=0,
    vla_review_mode="action_only", vla_reviewer=review_actions,
    max_steps=800,
)
```

- 基线：只提供 pi05_client，默认1候选，无 vla_reviewer，执行预算内至多24步并继续受公共动作约束检查。
- 只审核动作：上例；不提供 pi05_a2v_client，不生成预测。
- PI0.5+A2V：去掉 vla_review_mode，设置 `pi05_a2v_client=wam_a2v_client` 和 vla_reviewer。
  WAM 只预测 PI0.5 最终候选动作；禁止审核后再改动作。
- 三候选或 A2V 必须提供 vla_reviewer。其返回格式与上面的 review_actions 相同。

WAM 与 PI0.5 均使用当前 ARX-X5 joint14、弧度/夹爪[0,1]、25Hz 契约。
WAM 身份必须严格匹配 NATIVE_ACTION_CONTRACT，不能仅靠相同维度/频率接入。
非法前缀可针对同一提案重新审核，review_feedback 提供原因；每提案默认最多3次审核，
通过 max_review_attempts 配置，不重新生成动作。严格 mode3 文件桥与日志可见范围见 RSI_USAGE.md 的审核约束。
同时配置多个控制后端时，用 selector/supervisor 选技能，不能假设自动轮换。

## 5. 每个技能独立选择 ICL/HMR

```python
result = run_framework_episode(
    env, root=run_dir, pi05_client=pi05,
    vla_review_mode="action_only", vla_reviewer=review_actions,
    action_spec=action_spec,
    experience_catalog="/absolute/path/catalog.json",
    skill_experience={
        "vla_control": {"icl": ["ICL.md"], "hmr": ["HMR.md"]},
    },
)
```

经验包制作、仅 HMR / 仅 ICL 的严格对照及 Direct 的 ContextClient 接线，见
[经验专题](EXPERIENCE_GUIDE.md)。HMR/ICL 是经验材料，读取操作是辅助工具；
下面的 skill_experience 是自动注入路径，不是新增控制技能。

路径必须存在于当前任务显式选定的封存经验包 payload 中，不是服务器任意文件路径。
只有已注册技能能配置；WAM 使用 wam_control，direct 使用 direct_control，扩展技能用自己的注册名。
每个技能可独立配置：`{}` 关闭、`{"icl":[...]}` 仅ICL、`{"hmr":[...]}` 仅HMR、两者都填则都启用。
未提供 skill_experience 的技能默认不注入这份专属上下文。

- **技能后端**：在 `propose(request)` 中通过 `request.experience` 读取；关闭时为 None。
  icl/hmr 为不可变的 `(路径,文本)` 元组，identity_json 保存经验版本来源。
- **WAM/PI0.5 审核回调**：从 `packet["skill_experience"]` 获取身份与 icl/hmr 文本。
- **扩展技能审核**：从 proposal["skill_experience"] 获取相同内容。
- **外层 RSI**：候选 configure 返回同名 skill_experience 参数，按已有评估规则比较经验选择。
- 每个通道最多8个文件，每次读取每文件前4000字符；长文请整理成明确入口摘要。
  当前专属注入是文本接口；已有主管 read_image 工具仍可读取经验图片，不把图片冒充当前实测画面。

这套开关控制技能专属注入，**不隔离已经进入主管共享上下文的经验**。
若做完全无经验对照，还需不传 experience/experience_catalog，并使用没有历史经验的主管/审核上下文。

当前内置 WAM、PI0.5 模型仍使用原始任务指令和实测输入，不会自动将 ICL/HMR 文本拼进模型输入。
其经验效果来自主管选择、子目标规划和实际使用该上下文的审核者；仅配置经验但审核者忽略它，不会产生经验引导。
自定义模型若支持示范输入或显式记忆，需要在模型适配器中消费 request.experience 并验证输入格式。
WAM 自身 AR 实测历史、prefix/KV cache 与 HMR 历史经验包是不同机制。

经验不改写动作契约，不绕过审核，不证明当前操作成功，也不触发 PI0.5/WAM 重新训练。


## 6. A2V 审核与文件桥的完整使用顺序

### 6.1 选定链路

- WAM+A2V：WAM 同时生成候选动作与后果预测，通用服务 + mode3=False/predict=True。
- 严格视觉：WAM 三候选，服务 --mode3，episode mode3=True，隐藏具体动作数值。
- PI0.5+A2V：PI0.5 生成候选，通用 WAM 服务对这些动作预测，vla_reviewer 审核。

先核对服务类型与身份，再 reset 原生环境并装配审核者；每局单独输出目录。
A2V 本身不决定执行，审核者返回选择，公共执行器验证后才移动机器人。
如果动作被裁剪或后处理，应先确定最终候选，再对同一组动作做预测，不能用旧预测审核新动作。

### 6.2 一个可读的人工审核回调

```python
import json

def human_review(packet):
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    # 人工打开本轮提供的实测/预测图像，阅读 HMR/ICL 与执行反馈后再选择。
    answer = input("候选编号和执行步数（例如 0 8）；输入 stop 停止：").strip()
    if answer == "stop":
        return None
    candidate_id, execute_steps = map(int, answer.split())
    if execute_steps not in packet["allowed_execute_steps"]:
        raise ValueError("执行步数不在本轮允许前缀中")
    return {"candidate_id": candidate_id, "execute_steps": execute_steps}
```

这是需要终端与图片查看能力的交互示例，不用于无人值守作业。后台运行需实现模型审核或文件桥。
候选编号的存在性仍由框架检查。不同模式 packet 的图像/候选结构不同，不假设通用 preview
与严格视觉包具有完全相同的字段。审核者必须消费对应结构及其图片，不能只看文件名。

API 审核者也必须完成相同过程：编码图像、加入历史经验/当前证据、解析模型响应、返回明确选择。
HTTP 客户端或 Codex 文件桥只完成通信，不自动理解该协议；返回自然语言不能直接执行。

### 6.3 默认严格视觉文件桥

mode3=True 且未传 reviewer 时，Mode3FileReviewer 在同级 `<run_dir.name>-review/` 工作。
读取 pending.json 获取 request_id 和 directory；该公开目录中的 request.json/messages.json
包含本轮审核输入。响应程序查看材料后，在同一目录原子写入 response.json：

```json
{"request_id":"复制本轮request_id","candidate_id":0,"execute_steps":8}
```

拒绝用同一 envelope，将 candidate_id 设为 null、execute_steps 设为0。
响应必须匹配本轮 ID，不能复用旧回答。默认等待600秒。
建议先写临时文件再 rename 为 response.json，避免读取半个 JSON。
严格桥仅公开规范化图像和审核输入；原动作审计在局末记录。
它只支持严格视觉包，不能直接用来审核 action_only 或 PI0.5 动作包。

### 6.4 结合经验并确认生效

在现有 A2V 调用上增加 experience_catalog 与对应技能的 skill_experience。
审核回调从 packet["skill_experience"] 取 ICL/HMR，从 framework_context 取共享证据。
如果希望 HMR-only，catalog 的 payload 和 entry_points 也只开放该组材料，详见经验专题。

“预测成功”“审核批准”“原生任务成功”分别记录：预测可以错误，批准后也可能失败。
最后以实测反馈和原生评分判断，不让预测帧替代真实执行历史。
