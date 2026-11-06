# Motus Monitor — 训练实时监控库

训练侧（集群）埋点写盘 → SSH 字节级增量同步 → 本机/局域网 Web 可视化。训练侧与同步链路仅依赖 Python 标准库（前端为零依赖原生 JS）；**Web server 侧需要 Flask**（`pip install flask`，训练进程不 import）。训练侧关闭时是完全 no-op。

## 架构

```
集群训练进程                     本机
┌──────────────────┐   SSH    ┌──────────────────────────┐
│ MonitorLogger    │ 增量拉取  │ sync daemon (2s 轮询)     │
│  runs/<run_id>/  │ ───────► │  motus_monitor_data/     │
│   metrics/*.jsonl│          │   ↓ Registry             │
│   system/<node>/ │ ◄─────── │  web server :8377        │
│   events/*.jsonl │  回传     │   REST + SSE + 静态 SPA   │
│   slices/*.jsonl │          │                          │
└──────────────────┘ annotations └───────────┬──────────────┘
                                             │ /api/dataset/*
集群 /tmp 预览 helper ◄── SSH 托管 ──────────┘
（remote_preview_helper.py，按需解码 episode，
 由 server 启动时自动传输、拉起，退出自动回收）
```

- **增量**：指标/事件写成 append-only 分块 JSONL（256KB 轮转），同步只拉每个文件新增的字节；小 JSON（run.json/status/annotations）整体重拉/回传。远端只需要 `python3`（不需要 rsync），经跳板机也可工作。连接复用**继承本机 `~/.ssh/config` 的 `ControlMaster`/`ControlPersist` 配置**（例如你已配置的 `Host *` 持久连接）；若目标机接受密钥认证、想自建控制连接，可在 `--ssh-args` 里传 `-o ControlMaster=auto -o ControlPath=... -o ControlPersist=600`。注意同步进程是 `BatchMode` 非交互模式，**无法输密码**，目标机必须能免密登录（已有持久连接或密钥认证）。
- **run_id**：存在 `<checkpoint_dir>/.motus_monitor_run_id`，弹性重启 / auto-resume 续写同一 run；新 run 记录 parent（resume/finetune 来源 ckpt + 父 run），构成 ckpt 树。
- **resume 重跑的指标语义**：训练只会从已保存 ckpt 的 step 续跑。monitor 磁盘记录是 append-only（不删改历史），resume 时 status 会记录 `lineage_base_step`（本次续跑的 ckpt step）与 `attempt_started_ts`；**前端隐藏基点之后来自旧一次执行的记录（"被放弃的 future"）**，同一 step 重复时后写胜出——曲线始终与当前训练谱系一致。**重启但非 resume（从 0 重跑）时基点为 0**：旧 attempt 的指标与 system/GPU 记录（按 `attempt_started_ts` 过滤）全部不再显示；首次运行基点为 null（同时清除 status 里的陈旧值）。
- **数据目录**：默认 `<仓库上一级目录>/motus_monitor_data/`（不可写时回退 `~/.local/share/motus_monitor_data/`）；`MOTUS_MONITOR_DIR` 覆盖。

## 训练侧接入（已完成）

`train/train.py` 与 `train_debug/micro_train.py` 已集成。开关：

```yaml
# configs/*.yaml
monitor:
  enabled: true        # 总开关（默认 false；MOTUS_MONITOR_ENABLED=0/1 覆盖）
  data_dir: null       # 数据根目录（默认见上）
  system_interval: 2.0 # 系统采样间隔（秒，每节点 local_rank0 采样）
  status_interval: 2.0
  collect_system: true
```

记录内容：全部训练配置快照、**初始化阶段进度与耗时（initializing 状态 + init_phases）**、每 log_interval 步的全部 loss/lr/grad_norm/各阶段耗时/数据源统计/queue 统计、checkpoint 事件（step+路径+耗时）、系统负载（CPU/loadavg/内存/shm/每 GPU util/显存/温度/功耗/磁盘，**每节点 local_rank0 各写一份**：目录 `system/node<N>/`，N 取 `NODE_RANK`/`GROUP_RANK`/`JOB_COMPLETION_INDEX`，与 `train_node<N>_*.log` 对齐，并附 `node_info.json` 注册）、run 生命周期与错误事件。

### 耗时统计的结构

耗时页分四组（随数据自动出现）：**初始化**（run.json init_phases：config/model/dataset/prepare 各阶段，只发生一次）、**训练循环**（dataloader/queue/前向/反向/优化器/checkpoint 等每步耗时，**按函数执行顺序排序**）、**模型内部**（前向内部的模块/逐 block 检查点，同属执行顺序）、**自定义 stage**。两点约定：

- **按执行先后排序**，不按耗时大小；已知键有顺序表，未知键自动挂到父键之后，不同版本检查点集合均不会出错。
- **层级展示，占比只算顶层**：`X_seconds` 包含 `X_block_*_seconds`（如 action_encode ⊇ 其全部 block），`checkpoint_total_seconds` 包含 prepare/save_state/config/marker，`und_encode` 包含 `und_vlm`/`und_encoder`。子项缩进显示"占父级 %"、可折叠，占比与堆叠图只统计顶层，避免重复计数。
- **累计计数器自动排除**（如 `ready_queue_*_total_seconds`）：单调不减的累计值不是每步耗时，不进入阶段堆叠（避免初始化期队列预热等累计量掩盖真实分布）。

### 耗时检查点（版本自适应）

耗时检查点**不是固定 schema**：loop 级（dataloader/前向/反向/优化器/…）、模型内部（`model_*_seconds`：T5、VAE encode、各分支 encode/decode、`wm_*` 模块）、自定义（`MonitorLogger.stage()`）。前端耗时页按数据自动发现检查点并分组（训练循环 / 模型内部 / 自定义），不同代码版本产生的不同键名都能正确展示。

模型内部粒度由 YAML `model.timing_level` 控制：

- `1`（默认）：分支级（video/action encode/decode、uni_process、und、T5、VAE）
- `2`：+ 逐 block 检查点（`wm_video_encode_block_00` 等；每个 block 多两次 CUDA 同步，仅调试用）

细粒度埋点用 `monitor/timing.py` 的 `timed_section` / `block_timing`（一行插入任意代码段；`timing=None` 时零开销）。

代码内使用（可选）：

```python
from monitor import create_monitor
monitor = create_monitor(config, run_name=..., checkpoint_dir=..., rank=..., world_size=...)
monitor.log_step_record(step_record)      # 每步指标（rank0）
monitor.log_checkpoint(step, path, timing)
with monitor.stage("action_branch"): ...  # 细粒度耗时，并入下一条 step 记录
monitor.close("completed")
```

## 本机一键启动

```bash
# 纯本地数据（同机训练或只看已同步数据）
python -m monitor serve --port 8377

# 连集群实时同步（示例：H200 经跳板机）
python -m monitor serve --port 8377 \
    --remote fengodchen@localhost:/code1/llm_team/chen.liao/ThinkerWM/motus_monitor_data \
    --ssh-args "-J chen.liao@10.10.94.35 -p 10023" \
    --interval 2

python -m monitor doctor    # 检查 ssh/远端 python3/数据目录
python -m monitor sync-once # 手动同步一次
python -m monitor demo-data # 生成演示数据（UI 开发用）
```

浏览器打开 `http://localhost:8377`。页签：总览（进度/ETA/负载/事件）、曲线（loss/lr/grad_norm/系统曲线，EMA 平滑、log-y、滚轮缩放、拖拽平移、双击复位、按数据源拆分；**选中 run 自动加载全量历史**，长 run 抽稀显示并注明；**GPU 利用率按卡曲线 + 统计表**：每卡平均/P5/最大 util、平均显存、平均功耗与全体均值，覆盖全量 system 历史；曲线默认只显示最近 1 小时并跟随最新数据，"显示全部/仅看近期"按钮或滚轮缩放切换，手动缩放/平移后点"跟随近期"或双击回到默认）、耗时（各阶段堆叠分析，检查点集合随代码版本自动发现）、**对比**（多 run 指标叠加：勾选若干 run + 单选指标，各自全量历史独立点集叠加，支持 EMA/log-y/缩放/平移）、训练树（run→ckpt 谱系 + benchmark 编辑）、基准（ckpt×基准矩阵 + LLM API 说明）、**评估**（训练中周期性评测产物，见下节）、**训练集**（见下节）。右上角 ☀/☾ 按钮切换亮/暗主题（存 localStorage，**默认亮色**，用户显式选择过则尊重 localStorage）。周期刷新有守卫：正在选中文字或操作下拉/输入框时自动渲染推迟到交互结束，不打断选择。

顶栏三个实用控件：

- **同步进度条**：任一远端在拉取时显示百分比/字节进度（`SyncEngine.status` 的 `phase`/`pull_done_*`/`pull_pending_*` 字段驱动），空闲自动隐藏。
- **监控查看路径切换（每用户 workspace）**：顶栏"📂 查看目录：" chip 打开目录选择弹窗，分**本机 / 远端（H-Server）两种模式**（有 ssh 远端时默认远端模式）。**每个登录用户的查看根与自加同步远端都隔离持久化在 `<启动目录>/workspaces/<user>/`（workspace.json + data/），互不影响**。本机模式：`/api/browse` 只读列本机子目录、标记含 `runs/` 的目录（**只读文件树——只列目录名，不提供任何文件内容的读取/下载**），选中后 `POST /api/data_root` 只切换**自己** workspace 的查看根（被查看目录保持只读；sync 守护仍写各自原目录）。**远端模式**：经一次性 SSH（`python3 -c` 短进程，BatchMode 免密复用 ControlMaster，远端零安装）浏览**远端（H-Server）文件系统**（`GET /api/browse?remote=<名>&path=`，同样只列目录），选中目录后 `POST /api/remotes` 把它加为**自己 workspace 的同步远端**——复用参考远端的 target/ssh_args，新 engine（`owner`=当前用户，名称 `<user>--<slug>`，用户范围内按 target+path 去重）拉取进 `workspaces/<user>/data/` 并把**该用户**查看根切到那里（同步到位后 run 自动出现），持久化在该用户 `workspace.json`（serve 重启自动恢复）。旧版全局 `.added_remotes.json`（拉进启动根）仍按全局引擎恢复兼容。两个模式的弹窗顶部各列"已发现的监控数据目录"快捷项：server 后台守护线程低负载浅扫（启动 30s 后首扫、之后每 10 分钟、深度≤2、单轮限量限时、异常静默）本机候选位置（启动目录父级、默认数据根等，缓存 `.known_roots.json`）与**各 ssh 远端**（候选=远端数据根及其父目录，按 target+path 去重后同样一次性 SSH 扫描），`GET /api/known_roots` 返回 `roots`（本机）+ `remote_roots`（远端，带 remote 名），点击本机项=切换查看根、点击远端项=加入同步。注意：任何登录用户都能浏览服务器/远端目录名并把读模型指到任意本机目录（`/api/runs/{id}/file` 仍限制在 `runs/<run_id>` 沙箱内）。
- **登录用户/登出**：登录后顶栏显示当前用户名 chip 与"登出"按钮（`GET /api/auth/whoami` 探测，同时返回当前有效查看根 `view_root`；登出走 `GET /logout` 清 session 回 /login）；无 user_store 的兼容入口（`make_server`/`serve_forever`）自动隐藏。

同步健壮性：拉取按 128 文件/64MB 分块、逐块落盘状态，失败断点续传；**state 只记录已落盘字节**，且每周期对本地真实文件大小做自愈校验（state 脏或本地文件被删/截断都会自动补拉，无需手动重置 `.sync/`）。

## 独立服务部署（局域网 HTTPS + 登录 + 管理后台）

把 monitor 部署到局域网服务器时，用 **public 模式**：HTTPS（内部 CA 签发的服务器证书）+ 账户登录（Flask session）；另有一个**只绑 127.0.0.1 的管理后台**用于证书分发与账户管理。

```bash
pip install flask   # web server 侧唯一新增依赖

python -m monitor serve \
    --config monitor_server.json \
    --public --san <服务器IP> --san <主机名>     # SAN 也是 Host 头白名单
```

- **证书**：public 模式首次启动自动生成内部 CA 与服务器证书（`<data_root>/certs/`；私钥 0600 且**不提供任何下载**；DN 主体 `O=Ubtech / OU=chen.liao / emailAddress=chen.liao@ubtrobot.com`）。SAN 未配置时只签 localhost 证书（本机验证用）。**SAN 漂移自愈**：启动时若现有服务器证书缺少任一当前 SAN（或旧版本证书根本没有 SAN 扩展），自动重签并打 warning——Chrome 报 `NET::ERR_CERT_COMMON_NAME_INVALID` 多因访问地址不在 SAN 中，自愈后重启生效；**CA DN 变更等于换 CA，各客户端需重新导入新 ca.pem**。局域网用户导入 CA 后浏览器即信任：
  1. 在服务器本机打开管理后台 `http://127.0.0.1:8378`（可用 `ssh -L 8378:127.0.0.1:8378 user@server` 转发后访问）；
  2. 「证书」区下载 `ca.pem` 与 `import-ca.sh`（或按页面说明手动操作）；
  3. Ubuntu 客户端：`sudo ./import-ca.sh ca.pem`（等价于 `sudo cp ca.pem /usr/local/share/ca-certificates/motus-monitor.crt && sudo update-ca-certificates`）；Firefox 需在设置里单独导入。
- **账户**：管理后台「账户」区增删登录账户（密码 scrypt 哈希存 `<data_root>/users.json`；禁止删除最后一个 approved 账户）。用户也可在主站 `/register` **自助注册**——**首个注册的账户自动批准**（避免新部署无可用账户的死锁），其后注册写入 `pending` 状态，管理后台「批准」后才能登录（「拒绝」=删除申请）；pending 账户登录会提示"等待审批"。**所有模式（local 与 public）都要求登录**：除 `/login`、`/register` 与静态资源外全部端点要求会话（未登录 API 401、页面 302 到 /login；含 SSE 与 /api/dataset/*）；登录与注册按 IP 限流（10 次/分钟）。
- **管理后台**：仅绑 127.0.0.1（绑定地址即安全边界，无登录）。功能：CA 下载 / Ubuntu 导入说明导出（**按当前部署动态渲染**：验证命令里的地址=第一个非 localhost SAN + public 端口，改 IP/端口后说明自动跟随）/ 按新 SAN 重签服务器证书（重启主站生效）、账户增删与注册审批、**访问地址管理**（见下）、sync 与 dataset helper 状态查看、手动重启 helper、**服务日志**（主进程 root logger 同时写 `<data_root>/logs/server.log`——RotatingFileHandler 10MB×5——与内存 ring buffer 2000 条；页面支持级别/关键字过滤、5s 自动刷新、`GET /admin/api/logs` 与 `/admin/api/logs/download`）。public 模式默认跟随开启（`--admin` 可单独开）。
- **dataset 重渲染并发闸门**：frames/clip/selection 端点并发数 = heavy workers 数，超限 429，避免打满集群解码进程。
- **每用户 workspace**：每个登录用户的查看根与网页自加的同步远端隔离持久化在 `<data_root>/workspaces/<user>/`（互不影响；serve 重启按 workspace.json 恢复各用户引擎）。无 user_store 的兼容入口（`make_server`/`serve_forever`）维持旧的进程全局查看根（`.view_root.json`）。
- **Host 白名单**：public 模式校验 `Host` 头 ∈ 白名单 ∪ localhost，防 DNS rebinding。白名单 = 启动 SAN ∪ 管理后台「访问地址」区追加的地址（追加即时生效并持久化到 `<data_root>/access_addresses.json`，无需重启；添加时会同时按合并 SAN 重签服务器证书，证书本身重启主站后才生效；启动 SAN 不可在线移除）。默认**不发** `Access-Control-Allow-Origin`（SPA 同源无需 CORS）。

### 配置文件（`--config`，JSON；所有默认值 = 当前部署状态，CLI 参数优先）

```json
{
  "host": "127.0.0.1", "port": 8377,
  "data_dir": null,
  "remotes": [
    {
      "name": "h200",
      "remote": "fengodchen@localhost:/code1/llm_team/chen.liao/ThinkerWM/motus_monitor_data",
      "ssh_args": "-J chen.liao@10.10.94.35 -p 10023",
      "interval": 2,
      "preview": true
    }
  ],
  "dataset": {
    "enabled": true, "local": false,
    "repo": "/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain",
    "python": "/home/fengodchen/miniconda3/envs/motus311/bin/python",
    "catalog_root": null,
    "catalogs": [], "workers": 2
  },
  "public": {"enabled": false, "host": "0.0.0.0", "port": 8443, "san": []},
  "admin": {"enabled": false, "host": "127.0.0.1", "port": 8378}
}
```

**添加其他路径以实时监控**：在 `remotes` 里再加一条（`name` 唯一）并重启即可——多个远端各自增量同步、合并进同一 `<data_root>/runs/` 展示（状态文件 `.sync/<name>.json`，首个命名远端首次启动会自动接管旧版单远端 `.sync/state.json` 并将其改名退役；annotations 会推送到所有 ssh 远端）。dataset 预览 helper 挂在 `preview: true` 的远端（缺省第一个 ssh 远端）；该远端的集群侧路径（`repo`/`python`/`catalog_root`）在 `dataset` 段配置，默认值即当前部署。

### 安全注意事项

- monitor 服务器持有到集群的 SSH 免密身份。**生产部署建议为 sync 配专用受限 key**：在集群侧 `~/.ssh/authorized_keys` 给该 key 加 `command="..."`（只允许 monitor helper/sync 命令）+ `no-pty,no-port-forwarding`，把"服务器失陷 → 集群 shell"的爆炸半径压到最小。
- 管理后台无认证，**绝不能绑到非 127.0.0.1**；远程管理一律 `ssh -L` 转发。
- `certs/ca-key.pem` 是内部 CA 的根私钥，泄露 = 内部 HTTPS 体系全灭，请勿复制出服务器。

## 训练集预览（dataset tab）

数据在集群存储上，解码必须发生在集群侧。serve 启动时（配置了 ssh `--remote`）自动把 `monitor/remote_preview_helper.py` 按内容哈希传到集群 `/tmp/motus_preview_helper_<hash>.py` 并以 `--dataset-python` 拉起为持久进程，`/api/dataset/*` 端点经 stdin/stdout 帧协议按需取回解码结果（JPEG/MP4/float32 blob）；**helper 与 motus_data 库都不依赖集群仓库检出**——helper 源码每次启动按内容哈希上传，同时把本地 `libs/motus-data-factory/motus_data` 包打包上传到 `/tmp/motus_preview_motus_data_<hash>/`，helper 的 `sys.path` 优先该副本（本地缺包时才回退集群检出）。改本地代码/库后重启 serve 即自动生效，无需手动同步集群。

- **生命周期（三层自动终止 + 看门狗）**：① 正常退出（Ctrl-C/SIGTERM/atexit）→ 关 stdin，helper 读 EOF 自退，残留再按 pid `ssh kill`；② server 异常死亡（含 kill -9）→ helper/ssh 子进程带 `PR_SET_PDEATHSIG`，内核杀 ssh → 远端 stdin EOF → helper 自退；③ helper 端 `--idle-timeout`（默认 8h）兜底。每次启动还会 `pkill` 清扫 `/tmp/motus_preview_helper_*.py` 残留。**看门狗**：server 内 supervisor 线程每 20s 对 control worker 发 `list_datasets` 探活，失败（含空闲期 helper 被杀/ssh 断线）即整池重建（含重传），指数退避 20s→120s；状态见 `/api/dataset/status` 的 `supervisor` 字段与管理后台。
- **进程池（不堵塞）**：1 个 control 进程（list/info/action 轻请求）+ `--dataset-workers`（默认 2）个 heavy 进程（frames/mp4/selection 重渲染），轻请求在重解码进行中照常响应。
- **实时解码进度**：helper 分块解码并写 interim 进度行；`frames/clip/selection` 端点加 `&stream=1` 时经 HTTP chunked 把进度实时推给前端（`PRG <frac> <stage>` 前缀行 + `LEN` + 成品字节），前端进度条显示真实解码百分比（agibot 等 ffmpeg 全趟解码的源首次解码必然慢，进度可见 + 结果落缓存）。
- **数据集浏览**：release 10 源（catalog 根默认 `/data/jiyuan.wang/EpisodeCatalogs`，`--dataset-catalog-root` 改；扫描 + releases 解析 + `--dataset-catalog name:path` 显式覆盖）→ 每 dataset 的 episode 按**机器人→任务两级层级树**浏览（`/api/dataset/episode_tree` 按 family 规则解析 episode_id 分段聚合，task 为纯数字时取 `meta.task_name` 作 label；点击任务后子id 走 `/api/dataset/episodes?prefix=` 组内分页，搜索框仍为全局扁平查找）→ 预览：视频（filmstrip、逐帧 scrub、窗口 MP4 播放——**按源视频真实帧率编码、帧连续不跳帧**，整集预览取真实时间窗口 `seconds`，短片播全片；训练切片按真实帧索引回放）、action 曲线（按维开关/归一化）、全部 task 文本（kind 着色，带 video_range 的段落落在统一时间轴上可点击跳转）、**时间对齐**（帧 i ⇔ action 区间 (i·r,(i+1)·r] 双向游标）、←/→ 键盘步进；保存：当前帧 PNG、窗口 MP4、action CSV、selection JSON（全部浏览器侧导出）。
- **训练切片**：**所有 rank** 把各自 batch 的切片索引（`episode_ids`/`frame_indices`/`action_video_fps_ratios` + node/gpu/rank）追加写入 `runs/<run_id>/slices/rank%04d.jsonl`（train/train.py、train_debug/micro_train.py，与 step_record 同间隔，随 sync 增量同步）；旧 run 没有 slices/ 目录时回退读 rank0 metrics 流里的 step_record（行为与旧版一致）。`/api/runs/{id}/slices` 新结构按 **step→node→gpu** 树返回（分页单位=step；同一 (rank,step) 后写胜出），并带 `stats`：每源切片数/占比、**实际喂入帧口径**（`frames_fed` 喂入总帧（含重复）/`frames_unique` 同集不重复源帧/`frames_repeat` 重复量；另保留窗口首末帧区间的时间轴跨度 `frames_total`/`frames_union`/`frames_overlap` 供 tooltip；catalog 有 fps 的源附秒）、**覆盖率**（分母=run 配置快照 `dataset.sources` 的源集合；catalog 优先经 **dataset helper 的 `catalog_index` op** 拉取（ssh 远端/本机 local 均可，gzip blob 传输，结果磁盘缓存 `<data_root>/.cache/dataset/coverage/`、TTL 1h），服务器本地 catalog 文件兜底：每源 episodes_fed/total、frames_covered/total（时间轴覆盖）、coverage_pct + 逐 episode 已喂%/未喂明细；拿不到的源列入 `missing_catalogs`，拿不到配置时 denominator=observed）。**历史累计**：`lineage=1` 沿 run.json parent 链聚合本 run + 全部祖先 run（环保护+深度上限），by_ckpt 分段按 run 标注。前端切片面板：step 折叠树（默认仅展开最新 step，展开/折叠全部按钮）、每源统计行（喂入/去重/重复 + 口径解释 tooltip）、覆盖率百分比条（点击展开 episode 明细）、"历史累计"开关。**只存索引不存媒体**，点击切片走 `/api/dataset/selection` 远端解码——视频索引为训练时的真实值（`frames=<json>` 原样回传，stride 布局即跳帧+末帧填充序列），state/action 索引由 helper 重建：consecutive 布局按 (start·r, (start+N)·r]，stride 布局（视频跳帧、ratio>1）按 state=[start]、action=[start+1..start+N·r] 并对 action_count-1 clamp。窗口在时间轴高亮。**ckpt 配比**：端点按 run 的 checkpoint 把 step 分段（ckpt@S 覆盖 (prev_ckpt, S]，末尾为"最新"段），返回 `by_ckpt` 每段各数据源切片数；左栏渲染每 ckpt 一行的 100% 堆叠配比条（带图例/百分比 tooltip），点击行按该段 step 范围过滤切片（再点取消），选中段后 step 下拉（`steps_only=1` 拉取）可精确到单 step；**进入"训练切片"面板或切换 run 时自动加载并默认选中末尾"最新"段**，手动选段/点"全部"行后不再自动跳；当前范围的数据源占比渲染为饼图（canvas + 图例，与配比条同色），top 复用 chips 随所选范围联动。查询参数：`source`、`step_min`/`step_max`（闭区间）、`steps_only`、`lineage`。注意：统计的 source 取值=训练侧 `mix_source_name or source_dataset`，需与 catalog 名一致才能算覆盖率（micro debug 的 `robotwin`/`robotmind` 父名对不上子 catalog，会落入 missing/extra 标注）。
- **缓存**：frames/mp4/selection 按请求哈希落 `<data_root>/.cache/dataset/`（不在 sync 子目录，不会回传）。
- **相关 CLI**：`--no-dataset`（关闭）、`--dataset-local`（本机子进程跑 helper，开发/冒烟用，catalog 默认扫 `micro_data/catalogs`）、`--dataset-repo` / `--dataset-python` / `--dataset-catalog-root` / `--dataset-catalog name:path`。helper 也可独立一次性 dump：`python monitor/remote_preview_helper.py --repo-root . --catalog name:path --dataset X [--episode-id Y|--selection sel.json] --out DIR`。

## 周期性评测（eval tab）

训练侧 `periodic_eval` 包每 N 步做一轮评测（所有 rank 各评一个 batch），rank0 把索引记录经 `ChunkedJsonlWriter` 追加到 `runs/<id>/eval/*.jsonl`（与 `metrics/`、`events/` 平级的 append-only 流，sync 无需特殊处理）；媒体产物写在同一 run 目录下：

```text
runs/<id>/eval/
  000000.jsonl                            # 索引流（分块轮转、只追加）
  step_0001000/rank3/sample_0/
    first_frame.png                       # 输入首帧
    compare.mp4                           # condition|gt|pred 三联屏对比视频
    gt.mp4 / pred.mp4                     # 可选（可能不存在）
    actions.png                           # action 预测 vs GT 逐维曲线（video_only 样本只有预测）
    meta.json                             # 完整元数据
```

索引流一行一条 JSON，两种 `type`：

- `eval_round`：一轮概要 —— `step` / `ts` / `seconds` / `num_samples` / `ready_ranks` / `world_size` / `sources`（源→样本数）/ `metrics`（`video_mse`/`video_psnr`/`action_mse` 均值）。
- `eval_sample`：单样本 —— `step` / `rank` / `sample_index` / `source` / `dataset` / `robot_name` / `episode_id` / `manifest_id` / `video_only` / `task_text` / `vlm_prompt` / `vlm_output`（可能为 null）/ `metrics` / `files`（各媒体**相对 run 目录**的路径，可选文件不存在时为 null）。

`video_only: true` 的样本无 action GT（`actions.png` 只有预测曲线，`metrics.action_mse` 为 null）。resume 重跑会在同一 step 重复写记录：前端按 `step`（eval_round）与 `(step, rank, sample_index)`（eval_sample）去重、**后写胜出**——与 step 记录同一约定。

媒体经通用文件端点取字节：`GET /api/runs/{run_id}/file?path=<相对路径>`（resolve 后必须落在 `runs/<run_id>/` 内，逃逸路径一律 404；Content-Type 按扩展名猜测）。前端「评估」tab：评测轮选择（step 下拉/左右翻页，默认最新轮，新数据到达时停留在最新轮则自动跟随）+ 当轮概要（样本数/源分布/平均指标/耗时）+ 源过滤器 + 每样本一张卡片（任务文本、可折叠 vlm_prompt、高亮 vlm_output、首帧/action 曲线以**缩略图直接展示**（原图 CSS 缩放 + lazy 加载，点击查看原图；加载失败回退占位按钮）、对比/gt/pred 视频为占位按钮点击弹层播放、逐样本指标）。eval 流不存在时显示空态，不影响其他 tab。

## Benchmark 标注（双边同步）

在"训练树"页选择 ckpt 编辑 benchmark 指标，或走 API（本地写入后自动回传集群）：

```bash
curl -X POST http://localhost:8377/api/annotations \
  -H 'Content-Type: application/json' \
  -d '{"ckpt": {"run_id": "<run_id>", "step": 10000},
       "benchmarks": {"simpler_env": {"success_rate": 0.42}},
       "notes": "", "updated_by": "me"}'
```

LLM 代理可 `GET /api/llm/help` 获取机器可读 API 说明，`GET /api/tree` 枚举全部 run/ckpt（ckpt key = `<run_id>__step_<N>`）。注意 POST 是整体替换该 ckpt 的标注，增量编辑请先 GET 合并。

## 冒烟测试

```bash
python scripts/smoke_monitor.py   # 67 项检查：存储/客户端/registry/同步/REST API/dataset 预览
```

## 文件说明

| 文件 | 职责 |
|---|---|
| `monitor/paths.py` | 数据目录解析、run_id 读写、父 run 反查 |
| `monitor/store.py` | 分块 JSONL 写/读（字节增量）、原子 JSON 写 |
| `monitor/client.py` | `MonitorLogger` / `create_monitor` / `NullMonitor`（含全 rank 切片记录 `log_slice_record` → `slices/rank%04d.jsonl`） |
| `monitor/system_stats.py` | CPU/内存/shm/GPU/磁盘后台采样 |
| `monitor/registry.py` | run 列表/详情、ckpt 树、annotations 读写 |
| `monitor/sync.py` | SSH/local 增量双向同步引擎与守护线程（多远端命名实例） |
| `monitor/server.py` | Flask 主站：REST + SSE + 静态 SPA（public 模式 = HTTPS + 登录 + Host 白名单 + 解码并发闸门）；whoami、known_roots 后台扫描、全 rank 切片聚合/统计/覆盖率 |
| `monitor/auth.py` | 账户存储（users.json，scrypt 哈希）、登录限流、session secret key |
| `monitor/certs.py` | 内部 CA / 服务器证书签发（openssl CLI；DN=Ubtech/chen.liao；SAN 漂移自动重签）、Ubuntu 导入说明与脚本导出 |
| `monitor/admin.py` | 管理后台（仅 127.0.0.1）：证书下载/重签、账户增删、状态、helper 重启、服务日志查看/下载 |
| `monitor/logbuf.py` | 内存 ring buffer logging.Handler（2000 条，供 admin 日志 API） |
| `monitor/config.py` | 独立服务 JSON 配置加载（默认值=当前部署，CLI 优先） |
| `monitor/dataset_preview.py` | dataset 预览后端：helper 传输/拉起/回收、看门狗（探活+整池重建）、帧协议客户端、磁盘缓存 |
| `monitor/remote_preview_helper.py` | 集群侧预览 helper（--serve 长驻 + 一次性 dump；stdin EOF 自退） |
| `monitor/cli.py` | `serve` / `sync-once` / `doctor` / `demo-data`；serve 装日志落地（`logs/server.log` 轮转）与 ring buffer |
| `monitor/web/` | 原生 JS 前端（无依赖、离线可用；含 dataset tab `dataset.js`、eval tab `eval.js`、`login.html`、`admin.html`） |
