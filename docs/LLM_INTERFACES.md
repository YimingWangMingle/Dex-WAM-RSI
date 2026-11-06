# DeepSeek / Kimi 接口与 ICL、HMR 边界

两个提供方复用 `agents.llm_clients.ChatCompletionsClient`，统一为：

```python
response = client.complete(messages, tools)
message = response['choices'][0]['message']
```

`messages` 使用 Chat Completions 格式，支持文字、base64 图像及工具调用历史。
`tools` 为 function tools 列表。返回值保留完整 assistant message（包括推理字段）。调用方仍负责验证工具名、参数和执行前缀；网络响应不直接执行动作。

## 直接创建

```python
import os
from agents.llm_clients import DeepSeekClient, KimiClient

deepseek = DeepSeekClient(
    os.environ['DEEPSEEK_API_KEY'], model='deepseek-flash',
    thinking=False, max_tokens=4096,
)
kimi = KimiClient(
    os.environ['MOONSHOT_API_KEY'], model='kimi-k3',
    reasoning_effort='low', max_tokens=4096,
)
# 将其中一个注入现有 Agent 的 client；独立对话使用独立实例。
```

默认地址分别是 `https://api.deepseek.com` 与 `https://api.moonshot.cn/v1`。
可显式设置 `base_url`、`model`、`timeout`。base_url 是 API 根地址，不包含 `/chat/completions`。
旧 `from agents.deepseek import DeepSeekClient, DeepSeekError` 保持兼容，已删除该文件内重复的 HTTP 实现。

DeepSeek 思考开关映射为 thinking；思考开启时工具选择用 auto，非思考时用 required。Kimi 本接口采用现有项目的 **K3 参数契约**：始终思考，支持 low/high/max，使用 max_completion_tokens，不发送不适用的 thinking/temperature 字段。K2 系列不能直接套用这个参数契约。具体可用模型与账号权限以提供方为准。

## 通过现有统一启动入口选用

现有 `compat/launchers/run_codex_eef_live.py` 的环境配置：

- `RSI_LLM_PROVIDER=file`：默认，继续使用会话文件控制器。
- `RSI_LLM_PROVIDER=deepseek`：凭据由 `DEEPSEEK_API_KEY` 读取。
- `RSI_LLM_PROVIDER=kimi`：凭据由 `MOONSHOT_API_KEY` 读取。
- 可选 `RSI_LLM_MODEL`、`RSI_LLM_BASE_URL`、`RSI_LLM_TIMEOUT_S`、`RSI_LLM_MAX_TOKENS`、`RSI_LLM_REASONING_EFFORT`。
- DeepSeek 可选 `RSI_LLM_THINKING=enabled|disabled`，默认 disabled。

监督器和 EEF 各使用一个客户端实例，避免对话缓存串用。这里切换的是模型接口，保留现有原生环境与动作审核。模式三的视觉审核器继续通过 reviewer 参数或现有文件审核协议独立注入；本次没有把它隐式改成 API 调用。

接口不自动重试、不跨提供方回退；凭据错误、超时和截断响应均向调用方报错。HTTP 错误不输出服务端可能回显的请求内容。对 EEF 重建 assistant 消息丢失的字段，仅在同一客户端中、工具 ID/名称/参数完全一致时恢复已收到的原消息字段。

## ICL / HMR

当前命名：HMR/ICL 是经验材料，读取它们的能力属于辅助工具。完整封包与注入方式见
[经验专题](EXPERIENCE_GUIDE.md)。本页只说明客户端边界。

- ICL 文本/图像仍由已有 RoboProbe ICL loader 选择与加载，所生成的多模态消息可以原样传入两种接口。`L3_INSPECT_ICL_TEXT_ROOT` / `L3_INSPECT_ICL_ROOT` 等现有参数保持有效。
- HMR 是历史案例与反馈参考，不是本回合实测状态。已有冻结 v2 包包含任务、来源、动作请求、执行证据和总结。框架可通过 experience_catalog 自动读取入口文本，通过 skill_experience 给技能/审核者注入；
  Direct 需使用 ContextClient 将共享上下文传入 EEF 请求。裸 HTTP 客户端不会扫描或加载 HMR 文件。
- 经验工作区 `experience/packages.py`、任务上下文 `agents/context.py`、模型实测历史 `third_party/Thinker-WM/runtime/history_buffer.py` 职责不同，不能因为都叫“记忆”就合并删除。

## 冗余检查结果

1. **已合并**：原 agents/deepseek.py 的专用 HTTP 请求实现。DeepSeek/Kimi 现在共享网络、错误处理、元数据及消息完整性逻辑。
2. **保留兼容**：DeepSeekAgent 的命令解析本来就通过注入 client 与提供方解耦；名称保留以兼容旧调用，不再复制一份 KimiAgent。
3. **清理结果（2026-11-01）**：已删除无人用于生产入口的三套旧候选客户端；普通审核、严格 mode3 和事件审核统一由 TaskSkillRunner 执行。robodojo_deepseek_ik.py / robodojo_deepseek_live.py 仍有历史实验调用，保留其复现入口。DeepSeek/Kimi 的共享 HTTP 实现与不同参数策略继续保留。
4. **不属于冗余**：模型推理、视觉审核、统一执行器、任务上下文和训练数据整理是不同层次；RoboProbe 自带 Responses 客户端还服务其原始入口，未在本轮重写第三方实现。

验证使用本地 HTTP 服务和 CPU 替身，不调用付费 API、不启动新原生实验。

参考：
- https://api-docs.deepseek.com/api/create-chat-completion/
- https://api-docs.deepseek.com/guides/thinking_mode/
- https://platform.kimi.com/docs/guide/kimi-k3-quickstart
- https://platform.kimi.com/docs/api/chat
