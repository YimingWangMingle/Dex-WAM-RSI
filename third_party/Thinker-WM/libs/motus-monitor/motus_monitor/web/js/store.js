/* Motus Monitor 前端 —— 跨 tab 响应式 store（状态 + 数据层 + pub/sub）。
 * 从旧 web/app.js 逐字迁出：state（app.js:11-38）、run 数据层
 * （app.js:317-322 currentRun、395-472 loadRuns/selectRun、474-501 lineageKeep/sysKeep、
 * 503-608 ingestSteps/createReaders/refreshDetail/loadSystemHistory）。
 * 渲染调用一律改为 emit 事件，由 main.js 与各 tab 模块订阅路由：
 *   "runs"       runs 列表刷新（loadRuns 成功后）
 *   "run"        selectRun 完成（localStorage 已写、readers 已建）
 *   "detail"     refreshDetail 返回，payload {detail, nodesChanged}
 *   "steps"      ingestSteps 后（metrics 增量/全量历史合并完成）
 *   "events"     events 流增量到达
 *   "system"     system 流增量到达，payload {nodesChanged}
 *   "syshistory" 全量 system 历史加载状态变化，payload {node, loaded}
 *   "history"    metrics 全量历史加载状态变化（historyLoading/historyInfo）
 *   "sync"       loadSyncStatus 后（main.js 发起，syncPulling 已更新）
 */
import { api } from "./api.js";
import { util } from "./util.js";

export const STEP_CAP = 20000;      // step 记录缓存上限（超出丢弃最旧）
export const EVENT_CAP = 2000;
export const SYS_CAP = 5000;
export const GPU_RECENT_SECONDS = 3600; // GPU 图"近期"视窗宽度（跟随最新数据）

export const state = {
  tab: "overview",
  runs: [],
  runId: null,
  runDetail: null,
  steps: [],               // metrics 流 step 记录（按 step 升序、按 step 去重）
  stepMap: {},             // step -> 记录（resume 重跑时后写胜出）
  events: [],              // events 流记录
  sysNodes: [],
  sysNode: null,
  sysLatest: {},           // node -> 最新 system 记录（总览卡片）
  sysHistory: {},          // node -> [system 记录]（系统曲线，增量尾部）
  sysFullHistory: {},      // node -> [system 记录]（全量历史，GPU 统计用）
  sysHistLoading: {},      // node -> true（全量 system 历史加载中）
  readers: null,           // {metrics, events, system}
  sourceColors: {},        // source_dataset -> color
  gpuViewMode: "recent",   // GPU 图视窗：recent（默认近期）| full（全部）；用户缩放/拖拽后由 viewTouched 接管
  historyLoading: false,   // metrics 全量历史加载中
  historyInfo: null,       // {total, returned, decimated}
  syncPulling: false,      // 任一远端同步处于 pulling（驱动 2s 高频刷新）
  updating: false,
  renderDeferred: false,   // 因用户选中文本/操作控件而推迟的自动渲染
  lastDetailAt: 0,
};

/* ------------------------------------------------------------ pub/sub */
var listeners = {};  // evt -> [fn]

function on(evt, fn) {
  (listeners[evt] = listeners[evt] || []).push(fn);
}

function emit(evt, payload) {
  var fns = listeners[evt];
  if (!fns) return;
  for (var i = 0; i < fns.length; i++) {
    try { fns[i](payload); } catch (e) { console.error(e); }
  }
}

/* ------------------------------------------------------------ Run 选择 */
function currentRun() {
  for (var i = 0; i < state.runs.length; i++) {
    if (state.runs[i].run_id === state.runId) return state.runs[i];
  }
  return null;
}

function loadRuns() {
  return api.get("/api/runs").then(function (resp) {
    state.runs = (resp && resp.runs) || [];
    emit("runs");
    return state.runs;
  }).catch(function (err) {
    console.error("load runs failed", err);
    return [];
  });
}

function selectRun(runId) {
  if (runId === state.runId) return;
  state.runId = runId;
  try { localStorage.setItem("motus.monitor.runId", runId); } catch (e) {}
  state.steps = [];
  state.stepMap = {};
  state.events = [];
  state.sysNodes = [];
  state.sysNode = null;
  state.sysLatest = {};
  state.sysHistory = {};
  state.sysFullHistory = {};
  state.sysHistLoading = {};
  state.sourceColors = {};
  state.runDetail = null;
  state.historyLoading = false;
  state.historyInfo = null;
  /* GPU 图视窗是时间轴：换 run 后回到默认"近期"跟随，避免沿用上个 run 的旧区间。
   * gpuChart 实例归 curves tab：由它监听 "run" 自行重置 viewTouched/view。 */
  createReaders();
  /* 时序注意：emit 必须在 createReaders 之后（监听者可能立刻 poll readers） */
  emit("run", runId);
  loadHistory();
  refreshDetail();
}

/* ------------------------------------------------------------ 全量历史加载
 * 选中 run 后异步拉取完整历史（服务端扫描全流、resume 去重、抽稀），
 * 与实时增量流合并（ingestSteps 按 step 去重，重叠处自动一致）。 */
function loadHistory() {
  var runId = state.runId;
  if (!runId) return;
  state.historyLoading = true;
  emit("history");
  api.loadHistory(runId, 3000).then(function (resp) {
    if (state.runId !== runId || !resp) return;
    state.historyLoading = false;
    state.historyInfo = { total: resp.total_records, returned: resp.returned, decimated: !!resp.decimated };
    ingestSteps(resp.records || []);
    emit("history");
  }).catch(function (err) {
    if (state.runId !== runId) return;
    state.historyLoading = false;
    state.historyInfo = null;
    emit("history");
    console.warn("load history failed", err);
  });
}

/* ------------------------------------------------------------ GPU 利用率历史
 * 数据源：全量 system 历史（/history stream=system，一次性加载）
 * + 增量 system 尾部（实时），按 ts 去重合并（合并在 curves tab 做）。 */
function loadSystemHistory(node) {
  if (!state.runId || !node || state.sysHistLoading[node]) return;
  if (state.sysFullHistory[node]) return;
  state.sysHistLoading[node] = true;
  emit("syshistory", { node: node, loaded: false });
  api.loadHistory(state.runId, 3000, { stream: "system", node: node }).then(function (resp) {
    delete state.sysHistLoading[node];
    if (resp && resp.records) state.sysFullHistory[node] = resp.records;
    emit("syshistory", { node: node, loaded: true });
  }).catch(function (err) {
    delete state.sysHistLoading[node];
    emit("syshistory", { node: node, loaded: false });
    console.warn("system history failed", err);
  });
}

/* resume 语义：训练只会从已保存 ckpt 的 step 续跑，被放弃的那段 future
   会在重跑时被重新记录（同一 step 出现两次）。这里按"后写胜出"去重——
   后到的记录（重新执行的）覆盖先到的（被放弃的），图表不会回折。
   state.steps 始终保持按 step 升序。
   另外按谱系基点（lineage_base_step = 本次 resume 的 ckpt step）隐藏
   基点之后的旧记录：它们来自被中断的那次执行，重跑尚未覆盖时不应显示。 */
function lineageKeep(r) {
  var run = currentRun();
  if (!run) return true;
  var base = run.lineage_base_step;
  if (!util.isNum(base)) return true;
  if (util.isNum(r.step) && r.step <= base) return true;
  var t0 = run.attempt_started_ts;
  if (util.isNum(t0) && util.isNum(r.ts) && r.ts >= t0) return true;
  return false;
}

/* system/GPU 流没有 step 概念，只能按时间过滤：从 0 重跑（lineage_base_step
   === 0，训练侧重启且无 resume 时写入）时，本 attempt 之前的记录全部作废；
   resume 续跑（base > 0）保留完整历史曲线。 */
function sysKeep(r) {
  var run = currentRun();
  if (!run) return true;
  if (run.lineage_base_step !== 0) return true;
  var t0 = run.attempt_started_ts;
  if (!util.isNum(t0)) return true;
  return util.isNum(r.ts) && r.ts >= t0;
}

function ingestSteps(records) {
  var map = state.stepMap;
  for (var i = 0; i < records.length; i++) {
    var r = records[i];
    if (!r || (r.type && r.type !== "step")) continue;
    var s = Number(r.step);
    if (!isFinite(s)) continue;
    if (!lineageKeep(r)) continue;
    map[s] = r;
  }
  var keys = Object.keys(map);
  keys.sort(function (a, b) { return Number(a) - Number(b); });
  if (keys.length > STEP_CAP) {
    for (var j = 0; j < keys.length - STEP_CAP; j++) delete map[keys[j]];
    keys = keys.slice(keys.length - STEP_CAP);
  }
  var out = new Array(keys.length);
  for (var k = 0; k < keys.length; k++) out[k] = map[keys[k]];
  state.steps = out;
  emit("steps");
}

function createReaders() {
  if (!state.runId) { state.readers = null; return; }
  state.readers = {
    metrics: api.streamReader(state.runId, "metrics", function (records) {
      ingestSteps(records);
    }),
    events: api.streamReader(state.runId, "events", function (records) {
      for (var i = 0; i < records.length; i++) {
        var r = records[i];
        if (!r) continue;
        state.events.push(r);
      }
      if (state.events.length > EVENT_CAP) {
        state.events.splice(0, state.events.length - EVENT_CAP);
      }
      emit("events");
    }, { tailBytes: 32768 }),
    system: api.streamReader(state.runId, "system", function (records, reader) {
      var node = reader.node || "_";
      if (!state.sysHistory[node]) state.sysHistory[node] = [];
      var arr = state.sysHistory[node];
      for (var i = 0; i < records.length; i++) {
        if (!records[i] || !sysKeep(records[i])) continue;
        arr.push(records[i]);
        state.sysLatest[node] = records[i];
      }
      if (arr.length > SYS_CAP) state.sysHistory[node] = arr.slice(-SYS_CAP);
      state.sysHistory[node] = arr;
      var nodesChanged = false;
      if (reader.nodes && reader.nodes.join() !== state.sysNodes.join()) {
        state.sysNodes = reader.nodes;
        nodesChanged = true;
      }
      if (!state.sysNode && reader.node) state.sysNode = reader.node;
      /* 节点列表刚到（或变化）时，立刻补齐其余节点的最新记录，
       * 避免总览系统卡等到下一个轮询周期才显示（overview tab 监听处理）。 */
      emit("system", { nodesChanged: nodesChanged });
    }, { tailBytes: 65536 }),
  };
  /* 首次加载：tail 拉最近一段 */
  state.readers.metrics.poll();
  state.readers.events.poll();
  state.readers.system.poll();
}

/* 轮询三个增量流（main.js doUpdate 用）；返回 promise 数组 */
function pollReaders() {
  var tasks = [];
  if (state.readers) {
    tasks.push(state.readers.metrics.poll());
    tasks.push(state.readers.events.poll());
    tasks.push(state.readers.system.poll());
  }
  return tasks;
}

function refreshDetail() {
  if (!state.runId) return Promise.resolve(null);
  return api.getOrNull("/api/runs/" + encodeURIComponent(state.runId)).then(function (d) {
    state.runDetail = d;
    var nodesChanged = false;
    if (d && d.streams && d.streams.system && d.streams.system.nodes) {
      var nodes = d.streams.system.nodes;
      nodesChanged = nodes.join() !== state.sysNodes.join();
      state.sysNodes = nodes;
      if (!state.sysNode && state.sysNodes.length) {
        state.sysNode = state.sysNodes[0];
        if (state.readers) state.readers.system.setNode(state.sysNode);
      }
      if (state.sysNode) loadSystemHistory(state.sysNode);
    }
    emit("detail", { detail: d, nodesChanged: nodesChanged });
    return d;
  });
}

export const store = {
  state: state,
  on: on,
  emit: emit,
  currentRun: currentRun,
  loadRuns: loadRuns,
  selectRun: selectRun,
  refreshDetail: refreshDetail,
  loadHistory: loadHistory,
  loadSystemHistory: loadSystemHistory,
  ingestSteps: ingestSteps,
  createReaders: createReaders,
  pollReaders: pollReaders,
  lineageKeep: lineageKeep,
  sysKeep: sysKeep,
};
