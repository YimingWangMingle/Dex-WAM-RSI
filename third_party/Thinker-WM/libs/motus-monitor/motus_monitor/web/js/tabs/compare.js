/* Motus Monitor 前端 —— 对比 tab（多 run 叠加）。
 * 从旧 web/app.js 逐字迁入：cmp 状态（app.js:1231-1236）、cmpMetricDef/cmpRunColor
 * （app.js:1238-1245）、renderCompareMetricChecks（app.js:1247-1259）、compare run 列表
 * （app.js:1261-1295）、loadCompareHistory/renderCompareHint（app.js:1297-1316）、
 * updateCompare（app.js:1318-1338）与 cmp-* 控件绑定（app.js:1835-1850）。
 * 每个被选 run 的全量历史经 /history 端点加载并缓存；
 * 指标为单选，系列 = run × 当前指标（独立点集）。
 * stateDotEl 为旧 app.js:389-393 的模块内拷贝（旧实现是全 app 共享的私有 helper）。
 * 阶段 2A：叠加图从手搓 canvas（charts/legacy.js LineChart）切到 ECharts
 * wrapper（charts/line.js createLineChart）。
 */
import { api } from "../api.js";
import { util } from "../util.js";
import { store } from "../store.js";
import { PALETTE, applyTheme } from "../charts/theme.js";
import { createLineChart } from "../charts/line.js";
import { METRIC_DEFS } from "../metrics-schema.js";

/* ------------------------------------------------------------ 图表实例与私有状态 */
var compareChart = null;

var cmp = {
  selected: {},    // run_id -> color
  metric: "total_loss",
  histories: {},   // run_id -> records
  loading: {},     // run_id -> true
};

function ensureCharts() {
  if (!compareChart) {
    compareChart = createLineChart(document.getElementById("compare-chart"), { xLabel: "step" });
  }
}

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

function renderCompareMetricChecks() {
  var box = document.getElementById("compare-metric-checks");
  box.innerHTML = "";
  METRIC_DEFS.forEach(function (m) {
    var chip = util.el("span", "metric-check" + (m.key === cmp.metric ? " on" : ""), { text: m.label });
    chip.addEventListener("click", function () {
      cmp.metric = m.key;
      renderCompareMetricChecks();
      updateCompare();
    });
    box.appendChild(chip);
  });
}

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
  renderCompareHint();
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

function renderCompareHint() {
  var el = document.getElementById("compare-hint");
  var n = Object.keys(cmp.loading).length;
  el.textContent = n ? "（加载历史中… " + n + "）" : "";
}

function updateCompare() {
  renderCompareHint();
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

export default {
  id: "compare",

  init: function () {
    renderCompareMetricChecks();

    /* 对比页控件 */
    document.getElementById("cmp-ema-slider").addEventListener("input", function (ev) {
      var v = parseFloat(ev.target.value);
      document.getElementById("cmp-ema-val").textContent = v.toFixed(2);
      if (compareChart) compareChart.setSmoothing(v);
    });
    document.getElementById("cmp-logy-toggle").addEventListener("change", function (ev) {
      if (compareChart) compareChart.setLogY(ev.target.checked);
    });
    document.getElementById("cmp-zoom-reset").addEventListener("click", function () {
      if (compareChart) compareChart.resetView();
    });
    document.getElementById("cmp-reload").addEventListener("click", function () {
      cmp.histories = {};
      Object.keys(cmp.selected).forEach(loadCompareHistory);
      updateCompare();
    });
  },

  enter: function () {
    ensureCharts();
    renderCompareRuns();
    updateCompare();
  },

  leave: function () {
    // 无清理
  },

  /* 数据事件联动：旧实现 onDataChanged 不刷新本 tab，保持空实现 */
  refresh: function () {
  },

  /* 主题切换后重绘（wrapper 经 theme.js 注册表统一刷新） */
  onTheme: function () {
    applyTheme();
  },
};
