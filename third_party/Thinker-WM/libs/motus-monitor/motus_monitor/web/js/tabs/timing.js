/* Motus Monitor 前端 —— 耗时 tab。
 * 从旧 web/app.js 逐字迁入：updateTiming/耗时统计表（app.js:1099-1226，
 * innerHTML 拼接 + querySelectorAll 回挂事件逐字保留）、shareBar（app.js:1214）、
 * timing totals、renderTimingScopePills（app.js:1100-1113）、timingChart 堆叠
 * （app.js:1222-1226）、timingScope（app.js:98 迁为本模块私有；
 * app.js:97 的 timingColorCache 已随 colorForKey 归入 js/metrics-schema.js）。
 * 分组发现逻辑用 metrics-schema 的 discoverTimingGroups（签名改为显式传 steps）。
 *
 * 数据联动（在 init 里订阅 store 事件）：
 *   "run" → 本 tab active 时 updateTiming（旧 selectRun 尾部分支）
 * steps/events/system 到达时的 updateTiming 由 main.js 的 onDataChanged 路由到 refresh()。
 * 阶段 2B：堆叠柱图从手搓 canvas（charts/legacy.js StackedBars）切到 ECharts
 * wrapper（charts/stacked.js createStackedBars）。
 */
import { util } from "../util.js";
import { store } from "../store.js";
import { applyTheme } from "../charts/theme.js";
import { createStackedBars } from "../charts/stacked.js";
import { shareBar } from "../ui/bars.js";
import {
  TIMING_TOTAL_KEYS,
  resolveStageTree,
  colorForKey,
  meanOfKey,
  discoverTimingGroups,
} from "../metrics-schema.js";

/* ------------------------------------------------------------ 图表实例与私有状态 */
var timingChart = null;
var timingScope = null;

function ensureCharts() {
  if (!timingChart) {
    timingChart = createStackedBars(document.getElementById("timing-chart"), {
      xLabel: "step",
      maxBars: 120,
    });
  }
}

/* ------------------------------------------------------------ 耗时页 */
function renderTimingScopePills(groups) {
  var box = document.getElementById("timing-scope");
  box.innerHTML = "";
  groups.forEach(function (g) {
    var pill = util.el("span", "node-tab" + (g.id === timingScope ? " active" : ""), {
      text: g.title + " (" + g.stages.length + ")",
    });
    pill.addEventListener("click", function () {
      timingScope = g.id;
      updateTiming();
    });
    box.appendChild(pill);
  });
}

function updateTiming() {
  ensureCharts();
  var table = document.getElementById("timing-table");
  var shareBox = document.getElementById("share-bar");
  var groups = discoverTimingGroups(store.state.steps);

  /* 初始化阶段组（来自 run.json 的 init_phases，仅发生一次；
     与"训练循环 / 模型内部(前向)"共同构成 初始化→loop→前向 三分部。
     过滤 startup.total_*：那是全部阶段的总和，不能与单项同级展示） */
  var initPhases = ((store.state.runDetail && store.state.runDetail.init_phases) || []).filter(function (p) {
    return !String(p.name || "").startsWith("startup.total");
  });
  if (initPhases.length) {
    groups.unshift({
      id: "init",
      title: "初始化",
      stages: initPhases.map(function (p, i) {
        return {
          key: "init_phase_" + i,
          label: p.name,
          color: colorForKey("init_" + String(p.name)),
          mean: util.isNum(p.seconds) ? p.seconds : null,
          get: function () { return null; },
        };
      }),
    });
  }

  /* scope：保留当前选择；失效（新版本没有该分组）时回退到第一个有数据的分组 */
  var active = null;
  groups.forEach(function (g) { if (g.id === timingScope) active = g; });
  if (!active) {
    active = groups.length ? groups[0] : null;
    timingScope = active ? active.id : null;
  }
  renderTimingScopePills(groups);

  /* 合计行（不堆叠，仅参考；随数据自动出现/消失） */
  var totalsBox = document.getElementById("timing-totals");
  var recent = store.state.steps.slice(-100);
  var parts = [];
  TIMING_TOTAL_KEYS.forEach(function (pair) {
    var mean = meanOfKey(recent, function (r) { return r[pair[0]]; });
    if (mean != null) parts.push(pair[1] + " " + util.fmtSeconds(mean));
  });
  totalsBox.textContent = parts.length ? "最近100步均值 · " + parts.join(" · ") : "";

  shareBox.innerHTML = "";
  if (!active) {
    table.innerHTML = "<tbody><tr><td class='muted'>暂无耗时检查点数据</td></tr></tbody>";
    shareBox.appendChild(util.el("div", "no-data", { text: "暂无数据" }));
    timingChart.setData([], []);
    return;
  }

  /* 层级：占比/堆叠只统计顶层（子项并列会重复计数）；子项缩进展示占父级比例 */
  var tops = resolveStageTree(active.stages);
  var sum = 0;
  tops.forEach(function (s) { if (util.isNum(s.mean)) sum += s.mean; });
  var html = "<thead><tr><th>阶段（" + util.esc(active.title) + "）</th><th>均值</th><th>占比</th></tr></thead><tbody>";
  tops.forEach(function (s) {
    var share = util.isNum(s.mean) && sum > 0 ? (s.mean / sum) * 100 : null;
    var kids = s._children || [];
    var toggle = kids.length
      ? "<span class='parent-toggle' data-key='" + util.esc(s.key) + "'>▾</span>"
      : "<span class='parent-toggle-spacer'></span>";
    html += "<tr class='top-row'><td>" + toggle +
      "<span class='stage-chip' style='background:" + s.color + "'></span>" +
      util.esc(s.label) + "</td><td>" + util.esc(util.fmtSeconds(s.mean)) + "</td><td>" +
      (share != null ? share.toFixed(1) + "%" : "—") + "</td></tr>";
    kids.forEach(function (c) {
      var cshare = util.isNum(c.mean) && util.isNum(s.mean) && s.mean > 0
        ? (c.mean / s.mean) * 100 : null;
      html += "<tr class='child-row' data-parent='" + util.esc(s.key) + "'><td>" +
        "<span class='child-marker'>↳</span>" +
        "<span class='stage-chip' style='background:" + c.color + "'></span>" +
        util.esc(c.label) + "</td><td>" + util.esc(util.fmtSeconds(c.mean)) + "</td><td>" +
        (cshare != null ? cshare.toFixed(1) + "% <span class='muted'>of 父级</span>" : "—") +
        "</td></tr>";
    });
  });
  html += "</tbody>";
  table.innerHTML = html;

  /* 父级折叠/展开 */
  table.querySelectorAll(".parent-toggle").forEach(function (t) {
    t.addEventListener("click", function () {
      var key = t.getAttribute("data-key");
      var collapsed = t.textContent === "▾";
      t.textContent = collapsed ? "▸" : "▾";
      table.querySelectorAll(".child-row[data-parent='" + key + "']").forEach(function (row) {
        row.classList.toggle("hidden", collapsed);
      });
    });
  });

  /* 100% 堆叠条（仅顶层） */
  if (sum > 0) {
    shareBox.appendChild(shareBar(tops.map(function (s) {
      return { label: s.label, value: s.mean || 0, color: s.color };
    })));
  } else {
    shareBox.appendChild(util.el("div", "no-data", { text: "暂无数据" }));
  }

  /* 堆叠柱图（仅顶层，系列随当前分组动态切换；窗口=最近 360 步，语义不变） */
  timingChart.setData(store.state.steps.slice(-360), tops.map(function (s) {
    return { key: s.key, label: s.label, color: s.color, get: s.get };
  }));
}

export default {
  id: "timing",

  init: function () {
    store.on("run", function () {
      if (store.state.tab === "timing") updateTiming();
    });
  },

  enter: function () {
    ensureCharts();
    updateTiming();
  },

  leave: function () {
    // 无清理
  },

  /* 数据事件到达时的轻量刷新（main.js 已守卫） */
  refresh: function () {
    updateTiming();
  },

  /* 主题切换：经 charts/theme.js 注册表统一 refreshTheme（ECharts 取色来自 CSS 变量） */
  onTheme: function () {
    applyTheme();
  },
};
