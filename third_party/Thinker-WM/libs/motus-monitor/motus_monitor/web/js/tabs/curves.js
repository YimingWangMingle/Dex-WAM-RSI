/* Motus Monitor 前端 —— 曲线 tab。
 * 从旧 web/app.js 逐字迁入：ensureCharts 中 main/source/system/gpu 四个图表
 * （app.js:275-297，改惰性在 enter 创建）、renderMetricChecks/updateCurves
 * （app.js:984-1074）、renderSysNodeTabs（app.js:1076-1097）、renderHistoryInfo
 * （app.js:430-443）、GPU 图全套（renderGpuHint/renderGpuViewToggle/mergedSysRecords/
 * updateGpuViews，app.js:610-752）与控件绑定（ema-slider/logy-toggle/zoom-reset
 * app.js:1809-1819、gpu-view-toggle app.js:1822-1832）。
 * 阶段 2A：四个图表从手搓 canvas（charts/legacy.js LineChart）切到 ECharts
 * wrapper（charts/line.js createLineChart）；序列改为调用方组装 points
 * （{x,y} 或 [x,y]），GPU"近期跟随"状态机留本模块（gpuTouched/gpuViewMode），
 * wrapper 只提供 setWindow/onUserZoom/onViewReset 原语。
 * 阶段 3A：compare tab 并入本 tab（js/tabs/compare.js 已删）——cmp 状态、
 * cmpMetricDef/cmpRunColor、run 列表渲染（含 stateDotEl helper）、
 * loadCompareHistory/updateCompare 全部迁入；控件条（metric chips/EMA/log-y/
 * 复位缩放/#history-info 提示位）两个子视图共用一份，语义随模式切换
 * （chips 单 run 多选 / 对比单选），EMA/log-y 设置两个模式各存一份、切回恢复。
 *
 * 数据联动（在 init 里订阅 store 事件）：
 *   "run"        → 重置 GPU 图 gpuTouched/视窗（旧 app.js:462-464 语义）
 *                  + 本 tab active 时 updateCurves（旧 selectRun 尾部分支）
 *   "history"    → 更新 #history-info（旧 renderHistoryInfo）
 *   "syshistory" → 更新 #gpu-hint；payload.loaded 时守卫后重绘 GPU 视图
 *                  （旧 loadSystemHistory 完成回调）
 * steps/events/system 到达时的 updateCurves 由 main.js 的 onDataChanged 路由到 refresh()。
 */
import { api } from "../api.js";
import { util } from "../util.js";
import { store, GPU_RECENT_SECONDS } from "../store.js";
import { PALETTE, applyTheme } from "../charts/theme.js";
import { createLineChart } from "../charts/line.js";
import { METRIC_DEFS } from "../metrics-schema.js";
import { guardAutoRender } from "../ui/guard.js";

/* ------------------------------------------------------------ 图表实例 */
var mainChart = null, sourceChart = null, sysChart = null, gpuChart = null;
var compareChart = null;
/* GPU 图：用户缩放/平移过（wrapper onUserZoom 驱动），自动"近期"跟随据此停手 */
var gpuTouched = false;

/* ------------------------------------------------------------ 子视图状态（阶段 3A）
 * "single" = 单 run 指标（4 张图卡）；"compare" = 多 run 对比（run 列表 + 叠加图）。
 * 模块内状态，不做 localStorage 持久化。 */
var viewMode = "single";
/* 对比子视图状态（自 compare.js 迁入）：每个被选 run 的全量历史经 /history
 * 端点加载并缓存；指标单选，系列 = run × 当前指标（独立点集） */
var cmp = {
  selected: {},    // run_id -> color
  metric: "total_loss",
  histories: {},   // run_id -> records
  loading: {},     // run_id -> true
};
/* 控件条两个模式共用，EMA/log-y 设置各存一份（切回时恢复并应用到目标图） */
var modeCtl = {
  single: { ema: 0, logy: false },
  compare: { ema: 0, logy: false },
};

function ensureCharts() {
  if (!mainChart) {
    mainChart = createLineChart(document.getElementById("main-chart"), { xLabel: "step" });
    sourceChart = createLineChart(document.getElementById("source-chart"), { xLabel: "step" });
    sysChart = createLineChart(document.getElementById("system-chart"), { xIsTime: true });
    gpuChart = createLineChart(document.getElementById("gpu-chart"), { xIsTime: true });
    /* 双击复位 = 回到默认"近期"视窗 */
    gpuChart.onViewReset = function () {
      gpuTouched = false;
      store.state.gpuViewMode = "recent";
      renderGpuViewToggle();
      updateGpuViews();
    };
    gpuChart.onUserZoom = function () {
      gpuTouched = true;
      renderGpuViewToggle();
    };
  }
}

/* 对比叠加图惰性创建（首次切到对比子视图；wrapper 对 0 尺寸容器会推迟初始化） */
function ensureCompareCharts() {
  if (!compareChart) {
    compareChart = createLineChart(document.getElementById("compare-chart"), { xLabel: "step" });
  }
}

/* 当前模式控件作用的目标图 */
function activeChart() {
  return viewMode === "compare" ? compareChart : mainChart;
}

/* ------------------------------------------------------------ 控件条提示位
 * 单 run 模式：全量历史信息（旧 renderHistoryInfo）；
 * 对比模式：复用同一元素显历史加载中计数（旧 #compare-hint 的 renderCompareHint 语义）。 */
function renderHistoryInfo() {
  var el = document.getElementById("history-info");
  if (!el) return;
  if (viewMode === "compare") {
    var n = Object.keys(cmp.loading).length;
    el.textContent = n ? "（加载历史中… " + n + "）" : "";
    return;
  }
  if (store.state.historyLoading) {
    el.textContent = "全量历史加载中…";
  } else if (store.state.historyInfo) {
    var info = store.state.historyInfo;
    el.textContent = info.decimated
      ? "历史: 全量 " + info.total + " 条（抽稀至 " + info.returned + " 条显示）"
      : "历史: 全量 " + info.total + " 条";
  } else {
    el.textContent = "";
  }
}

/* ------------------------------------------------------------ 曲线页
 * metric chips 两个模式共用 #metric-checks：单 run 模式多选（切 def.on 重绘 4 图），
 * 对比模式单选（选定 cmp.metric，.on 只留一个，重绘叠加图——旧 renderCompareMetricChecks
 * 语义，chips 结构沿用单 run 版）。 */
function renderMetricChecks() {
  var box = document.getElementById("metric-checks");
  box.innerHTML = "";
  METRIC_DEFS.forEach(function (def, i) {
    var color = PALETTE[i % PALETTE.length];
    var on = viewMode === "compare" ? def.key === cmp.metric : def.on;
    var chip = util.el("label", "metric-check" + (on ? " on" : ""));
    var input = util.el("input");
    input.type = "checkbox";
    input.checked = on;
    var dot = util.el("span", "chip");
    dot.style.background = color;
    dot.style.color = color;
    chip.appendChild(input);
    chip.appendChild(dot);
    chip.appendChild(util.el("span", null, { text: def.label }));
    chip.addEventListener("click", function (ev) {
      ev.preventDefault();
      if (viewMode === "compare") {
        cmp.metric = def.key;
        renderMetricChecks();
        updateCompare();
        return;
      }
      def.on = !def.on;
      chip.classList.toggle("on", def.on);
      input.checked = def.on;
      updateCurves();
    });
    box.appendChild(chip);
  });
}

function updateCurves() {
  ensureCharts();
  var steps = store.state.steps;

  /* 主图（非数值点——如 dropout 步的 null——由 wrapper 丢弃，同 legacy） */
  var series = [];
  METRIC_DEFS.forEach(function (def, i) {
    if (!def.on) return;
    series.push({
      name: def.label,
      color: PALETTE[i % PALETTE.length],
      points: steps.map(function (r) { return { x: r.step, y: def.get(r) }; }),
    });
  });
  mainChart.setSeries(series);

  /* 按数据源拆分：散点原始值 + 各源 EMA0.9 折线（两条序列由调用方给，wrapper 不内嵌） */
  var sources = {};
  for (var i = 0; i < steps.length; i++) {
    var s = steps[i].source_dataset;
    if (s) sources[s] = true;
  }
  var names = Object.keys(sources).sort();
  names.forEach(function (name, idx) {
    if (!store.state.sourceColors[name]) {
      store.state.sourceColors[name] = PALETTE[idx % PALETTE.length];
    }
  });
  var srcSeries = [];
  names.forEach(function (name) {
    var pts = [];
    for (var i = 0; i < steps.length; i++) {
      var r = steps[i];
      if (r.source_dataset !== name) continue;
      pts.push({ x: r.step, y: METRIC_DEFS[0].get(r) });   // total_loss（含 dropout 步过滤）
    }
    var color = store.state.sourceColors[name];
    srcSeries.push({ name: name, color: color, points: pts, scatter: true });
    srcSeries.push({ name: name + " EMA", color: color, points: pts, emaAlpha: 0.9, legend: false });
  });
  sourceChart.setSeries(srcSeries);

  /* 系统曲线（当前节点） */
  renderSysNodeTabs();
  var hist = store.state.sysHistory[store.state.sysNode] || [];
  sysChart.setSeries([
    {
      name: "cpu%", color: PALETTE[0],
      points: hist.map(function (r) { return { x: r.ts, y: r.cpu_percent }; }),
    },
    {
      name: "mem%", color: PALETTE[1],
      points: hist.map(function (r) { return { x: r.ts, y: r.memory && r.memory.used_percent }; }),
    },
    {
      name: "gpu_util%", color: PALETTE[3],
      points: hist.map(function (r) {
        var y = null;
        if (Array.isArray(r.gpus) && r.gpus.length) {
          var sum = 0, cnt = 0;
          r.gpus.forEach(function (g) { if (util.isNum(g.util_percent)) { sum += g.util_percent; cnt++; } });
          y = cnt ? sum / cnt : null;
        }
        return { x: r.ts, y: y };
      }),
    },
  ]);

  /* GPU 利用率（按卡）与统计表 */
  updateGpuViews();
}

function renderSysNodeTabs() {
  var box = document.getElementById("sys-node-tabs");
  box.innerHTML = "";
  if (!store.state.sysNodes.length) {
    box.appendChild(util.el("span", "muted", { text: "（无 system 流）" }));
    return;
  }
  store.state.sysNodes.forEach(function (node) {
    var b = util.el("button", "node-tab" + (node === store.state.sysNode ? " active" : ""), { text: node });
    b.addEventListener("click", function () {
      if (node === store.state.sysNode) return;
      store.state.sysNode = node;
      if (store.state.readers) {
        store.state.readers.system.setNode(node);
        store.state.readers.system.poll();
      }
      store.loadSystemHistory(node);
      updateCurves();
    });
    box.appendChild(b);
  });
}

/* ------------------------------------------------------------ GPU 利用率（按卡 + 统计）
 * 数据源：全量 system 历史（/history stream=system，一次性加载）
 * + 增量 system 尾部（实时），按 ts 去重合并。 */
function renderGpuHint() {
  var el = document.getElementById("gpu-hint");
  if (!el) return;
  var node = store.state.sysNode;
  if (node && store.state.sysHistLoading[node]) { el.textContent = "（历史加载中…）"; return; }
  el.textContent = "";
}

function renderGpuViewToggle() {
  var btn = document.getElementById("gpu-view-toggle");
  if (!btn) return;
  var custom = gpuChart && gpuTouched;
  btn.textContent = custom ? "跟随近期"
    : (store.state.gpuViewMode === "recent" ? "显示全部" : "仅看近期");
  btn.title = custom
    ? "已手动缩放/平移，点击回到默认近期视窗"
    : (store.state.gpuViewMode === "recent"
      ? "当前显示最近 " + Math.round(GPU_RECENT_SECONDS / 60) + " 分钟，点击显示全部历史"
      : "当前显示全部历史，点击回到近期视窗");
}

function mergedSysRecords(node) {
  var map = {};
  (store.state.sysFullHistory[node] || []).forEach(function (r) {
    if (util.isNum(r.ts) && store.sysKeep(r)) map[r.ts.toFixed(3)] = r;
  });
  (store.state.sysHistory[node] || []).forEach(function (r) {
    if (util.isNum(r.ts) && store.sysKeep(r)) map[r.ts.toFixed(3)] = r;
  });
  var keys = Object.keys(map);
  keys.sort(function (a, b) { return Number(a) - Number(b); });
  return keys.map(function (k) { return map[k]; });
}

function _gpuOf(rec, index) {
  var gs = rec.gpus;
  if (!Array.isArray(gs)) return null;
  for (var i = 0; i < gs.length; i++) if (gs[i].index === index) return gs[i];
  return null;
}

function _pct(values, p) {
  if (!values.length) return null;
  var s = values.slice().sort(function (a, b) { return a - b; });
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
}

function updateGpuViews() {
  if (!gpuChart) return;
  var node = store.state.sysNode;
  var recs = node ? mergedSysRecords(node) : [];
  var indexes = [];
  for (var i = 0; i < recs.length; i++) {
    var gs = recs[i].gpus;
    if (!Array.isArray(gs)) continue;
    for (var j = 0; j < gs.length; j++) {
      if (indexes.indexOf(gs[j].index) < 0) indexes.push(gs[j].index);
    }
  }
  indexes.sort(function (a, b) { return a - b; });

  /* 按卡 util 曲线（卡数过多时只画均值，避免不可读） */
  var series = [];
  if (indexes.length > 0 && indexes.length <= 16) {
    indexes.forEach(function (gi, k) {
      series.push({
        name: "gpu" + gi, color: PALETTE[k % PALETTE.length],
        points: recs.map(function (r) {
          var g = _gpuOf(r, gi);
          return (g && util.isNum(g.util_percent)) ? { x: r.ts, y: g.util_percent } : null;
        }).filter(Boolean),
      });
    });
  } else if (indexes.length > 16) {
    series.push({
      name: "gpu_util% 均值", color: PALETTE[0],
      points: recs.map(function (r) {
        if (!Array.isArray(r.gpus) || !r.gpus.length) return null;
        var sum = 0, cnt = 0;
        r.gpus.forEach(function (g) { if (util.isNum(g.util_percent)) { sum += g.util_percent; cnt++; } });
        return cnt ? { x: r.ts, y: sum / cnt } : null;
      }).filter(Boolean),
    });
  }
  gpuChart.setSeries(series);

  /* 视窗：默认"近期"（跟随最新 ts）；用户缩放/拖拽过（gpuTouched）则不干预 */
  if (!gpuTouched) {
    if (store.state.gpuViewMode === "recent" && recs.length) {
      var maxTs = recs[recs.length - 1].ts;
      gpuChart.setWindow([maxTs - GPU_RECENT_SECONDS, maxTs]);
    } else {
      gpuChart.resetView();
    }
  }
  renderGpuViewToggle();

  /* 统计表：每卡 平均/P5/最大 util、平均显存、平均功耗 + 全体均值行 */
  var box = document.getElementById("gpu-stats");
  box.innerHTML = "";
  var hint = document.getElementById("gpu-hint");
  if (!indexes.length) {
    box.appendChild(util.el("div", "no-data", { text: "暂无 GPU 数据" }));
    if (hint && !store.state.sysHistLoading[node]) hint.textContent = "";
    return;
  }
  var html = "<thead><tr><th>GPU</th><th>平均 util</th><th>P5 util</th><th>最大 util</th>" +
    "<th>平均显存</th><th>平均功耗</th><th>样本</th></tr></thead><tbody>";
  var allUtils = [];
  indexes.forEach(function (gi) {
    var utils = [], mems = [], powers = [];
    recs.forEach(function (r) {
      var g = _gpuOf(r, gi);
      if (!g) return;
      if (util.isNum(g.util_percent)) utils.push(g.util_percent);
      if (util.isNum(g.mem_used_bytes) && util.isNum(g.mem_total_bytes) && g.mem_total_bytes > 0) {
        mems.push(100 * g.mem_used_bytes / g.mem_total_bytes);
      }
      if (util.isNum(g.power_watts)) powers.push(g.power_watts);
    });
    var mean = utils.length ? utils.reduce(function (a, b) { return a + b; }, 0) / utils.length : null;
    if (mean != null) allUtils.push(mean);
    var memMean = mems.length ? mems.reduce(function (a, b) { return a + b; }, 0) / mems.length : null;
    var powMean = powers.length ? powers.reduce(function (a, b) { return a + b; }, 0) / powers.length : null;
    html += "<tr><td>gpu" + gi + "</td><td>" + (mean != null ? mean.toFixed(1) + "%" : "—") +
      "</td><td>" + (utils.length ? _pct(utils, 0.05).toFixed(1) + "%" : "—") +
      "</td><td>" + (utils.length ? _pct(utils, 1).toFixed(1) + "%" : "—") +
      "</td><td>" + (memMean != null ? memMean.toFixed(1) + "%" : "—") +
      "</td><td>" + (powMean != null ? Math.round(powMean) + "W" : "—") +
      "</td><td>" + utils.length + "</td></tr>";
  });
  if (allUtils.length) {
    var allMean = allUtils.reduce(function (a, b) { return a + b; }, 0) / allUtils.length;
    html += "<tr class='total-row'><td>全体均值</td><td>" + allMean.toFixed(1) + "%</td>" +
      "<td colspan='4'></td><td>" + indexes.length + " 卡</td></tr>";
    if (hint) hint.textContent = " · " + indexes.length + " 卡平均 util " + allMean.toFixed(1) + "%";
  }
  html += "</tbody>";
  var table = util.el("table", "data-table");
  table.innerHTML = html;
  box.appendChild(table);
}

/* ------------------------------------------------------------ 对比子视图（自 compare.js 迁入）
 * stateDotEl 为旧 app.js:389-393 的模块内拷贝（旧实现是全 app 共享的私有 helper）。 */
function stateDotEl(stateName) {
  var s = stateName || "unknown";
  if (["running", "completed", "failed", "initializing"].indexOf(s) < 0) s = "unknown";
  return util.el("span", "state-dot " + s);
}

function cmpMetricDef() {
  for (var i = 0; i < METRIC_DEFS.length; i++) {
    if (METRIC_DEFS[i].key === cmp.metric) return METRIC_DEFS[i];
  }
  return METRIC_DEFS[0];
}

function cmpRunColor(idx) { return PALETTE[idx % PALETTE.length]; }

function renderCompareRuns() {
  var box = document.getElementById("compare-run-list");
  /* 首次进入：默认勾选当前选中的 run */
  if (!Object.keys(cmp.selected).length && store.state.runId) {
    for (var i = 0; i < store.state.runs.length; i++) {
      if (store.state.runs[i].run_id === store.state.runId) {
        cmp.selected[store.state.runId] = cmpRunColor(i);
        break;
      }
    }
  }
  box.innerHTML = "";
  if (!store.state.runs.length) {
    box.appendChild(util.el("div", "no-data", { text: "暂无 run" }));
    return;
  }
  store.state.runs.forEach(function (r, idx) {
    var selected = cmp.selected.hasOwnProperty(r.run_id);
    var item = util.el("div", "compare-run-item" + (selected ? " selected" : ""));
    var color = cmpRunColor(idx);
    item.appendChild(util.el("span", "cmp-color", { attrs: { style: "background:" + (selected ? cmp.selected[r.run_id] : "var(--border)") } }));
    item.appendChild(stateDotEl(r.state));
    item.appendChild(util.el("span", "cmp-name", { text: r.run_name || r.run_id, attrs: { title: r.run_id } }));
    item.appendChild(util.el("span", "cmp-steps", {
      text: (r.last_step != null ? r.last_step : "—") + "/" + (r.max_steps != null ? r.max_steps : "—"),
    }));
    item.addEventListener("click", function () {
      if (cmp.selected.hasOwnProperty(r.run_id)) delete cmp.selected[r.run_id];
      else cmp.selected[r.run_id] = color;
      renderCompareRuns();
      updateCompare();
    });
    box.appendChild(item);
  });
}

function loadCompareHistory(runId) {
  if (cmp.loading[runId]) return;
  cmp.loading[runId] = true;
  renderHistoryInfo();
  api.loadHistory(runId, 3000).then(function (resp) {
    delete cmp.loading[runId];
    if (resp && resp.records) cmp.histories[runId] = resp.records;
    updateCompare();
  }).catch(function (err) {
    delete cmp.loading[runId];
    console.warn("compare history failed", runId, err);
    updateCompare();
  });
}

function updateCompare() {
  renderHistoryInfo();
  if (!compareChart) return;  /* 历史加载回调可能晚于切回单 run 模式 */
  var def = cmpMetricDef();
  var series = [];
  Object.keys(cmp.selected).forEach(function (runId) {
    var recs = cmp.histories[runId];
    if (!recs) {
      loadCompareHistory(runId);
      return;
    }
    var pts = [];
    for (var i = 0; i < recs.length; i++) {
      var r = recs[i];
      var v = def.get(r);
      if (util.isNum(r.step) && util.isNum(v)) pts.push([r.step, v]);
    }
    var name = runId.replace(/__.*/, "");
    series.push({ name: name + " · " + def.label, color: cmp.selected[runId], points: pts });
  });
  compareChart.setSeries(series);
}

/* ------------------------------------------------------------ 子视图切换（阶段 3A）
 * 把控件状态（EMA/log-y）恢复到目标模式的上次设置并应用到目标图。 */
function setViewMode(mode) {
  if (mode === viewMode) return;
  viewMode = mode;
  document.querySelectorAll("#tab-curves [data-cmode]").forEach(function (b) {
    b.classList.toggle("active", b.getAttribute("data-cmode") === mode);
  });
  document.getElementById("curves-single-view").classList.toggle("hidden", mode !== "single");
  document.getElementById("curves-compare-view").classList.toggle("hidden", mode !== "compare");
  var ctl = modeCtl[mode];
  document.getElementById("ema-slider").value = ctl.ema;
  document.getElementById("ema-val").textContent = ctl.ema.toFixed(2);
  document.getElementById("logy-toggle").checked = ctl.logy;
  renderMetricChecks();
  if (mode === "compare") {
    ensureCompareCharts();
    compareChart.setSmoothing(ctl.ema);
    compareChart.setLogY(ctl.logy);
    renderCompareRuns();
    updateCompare();
  } else {
    ensureCharts();
    mainChart.setSmoothing(ctl.ema);
    mainChart.setLogY(ctl.logy);
    /* 对比期间单 run 图未随数据事件刷新（见 refresh），切回补一次全量 */
    updateCurves();
  }
  renderHistoryInfo();
}

export default {
  id: "curves",

  init: function () {
    renderMetricChecks();

    /* 子视图切换条（单 run 指标 / 多 run 对比） */
    document.querySelectorAll("#tab-curves [data-cmode]").forEach(function (btn) {
      btn.addEventListener("click", function () { setViewMode(btn.getAttribute("data-cmode")); });
    });

    /* 控件条两个模式共用：EMA/log-y/复位作用于当前模式的目标图，
     * 设置值记入当前模式的那份（modeCtl），切换时各自恢复 */
    document.getElementById("ema-slider").addEventListener("input", function (ev) {
      var v = parseFloat(ev.target.value);
      document.getElementById("ema-val").textContent = v.toFixed(2);
      modeCtl[viewMode].ema = v;
      var c = activeChart();
      if (c) c.setSmoothing(v);
    });
    document.getElementById("logy-toggle").addEventListener("change", function (ev) {
      modeCtl[viewMode].logy = ev.target.checked;
      var c = activeChart();
      if (c) c.setLogY(ev.target.checked);
    });
    document.getElementById("zoom-reset").addEventListener("click", function () {
      var c = activeChart();
      if (c) c.resetView();
    });

    /* 对比子视图：重新加载全部已选 run 的历史（旧 cmp-reload 绑定） */
    document.getElementById("cmp-reload").addEventListener("click", function () {
      cmp.histories = {};
      Object.keys(cmp.selected).forEach(loadCompareHistory);
      updateCompare();
    });

    /* GPU 图：近期 ⇄ 全部 切换；手动缩放/平移后此按钮用于回到默认近期 */
    document.getElementById("gpu-view-toggle").addEventListener("click", function () {
      if (!gpuChart) return;
      if (gpuTouched) {
        store.state.gpuViewMode = "recent";
      } else {
        store.state.gpuViewMode = store.state.gpuViewMode === "recent" ? "full" : "recent";
      }
      gpuTouched = false;
      renderGpuViewToggle();
      updateGpuViews();
    });

    store.on("run", function () {
      /* GPU 图视窗是时间轴：换 run 后回到默认"近期"跟随，避免沿用上个 run 的旧区间 */
      gpuTouched = false;
      if (gpuChart) gpuChart.resetView();
      if (store.state.tab === "curves" && viewMode === "single") updateCurves();
    });
    store.on("history", renderHistoryInfo);
    store.on("syshistory", function (payload) {
      renderGpuHint();
      if (!payload || !payload.loaded) return;
      if (guardAutoRender(store.state)) return;
      updateGpuViews();
    });
  },

  enter: function () {
    ensureCharts();
    /* 订阅前已加载完的历史/系统历史状态补写一次提示文案（旧实现由
     * loadHistory/loadSystemHistory 直接调用，与 tab 是否初始化无关） */
    renderHistoryInfo();
    renderGpuHint();
    if (viewMode === "compare") {
      /* 旧 compare tab 的 enter：run 列表 + 叠加图（图表惰性创建） */
      ensureCompareCharts();
      renderCompareRuns();
      updateCompare();
    } else {
      updateCurves();
    }
  },

  leave: function () {
    // 无清理
  },

  /* 数据事件到达时的轻量刷新（main.js 已守卫）；
   * 对比模式不随数据事件刷新（旧 compare tab 的 onDataChanged 空实现语义） */
  refresh: function () {
    if (viewMode === "compare") return;
    updateCurves();
  },

  /* 主题切换后重绘（wrapper 经 theme.js 注册表统一刷新） */
  onTheme: function () {
    applyTheme();
  },
};
