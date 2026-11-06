/* Motus Monitor 前端 —— ECharts 图表主题与调色板（阶段 2A 新基建）。
 * PALETTE/emaSmooth 从 charts/legacy.js 逐字拷贝（legacy 引擎随阶段 2 收尾删除，
 * 这两项是跨图表复用的业务资产，归档在本模块）。
 * chartTheme() 从 CSS 变量取色（canvas/DOM 图表都无法直接用 var()），返回
 * ECharts 基础 option 片段；wrapper（charts/line.js 等）创建时经 registerChart
 * 登记，主题切换由 applyTheme() 统一刷新全部活图表（main.js → tab.onTheme 链路）。
 */
export const PALETTE = [
  "#4f8cff", "#ff6b6b", "#4ecdc4", "#e0a92c", "#a06bff", "#6bd66b",
  "#ff9f43", "#f368e0", "#48dbfb", "#1dd1a1", "#ff9ff3", "#7d9db5",
];

/* EMA 平滑（legacy.js:60-70 逐字）：alpha<=0 原样返回 */
export function emaSmooth(pts, alpha) {
  if (!alpha || alpha <= 0) return pts;
  var out = new Array(pts.length);
  var s = null;
  for (var i = 0; i < pts.length; i++) {
    var v = pts[i][1];
    s = s === null ? v : alpha * s + (1 - alpha) * v;
    out[i] = [pts[i][0], s];
  }
  return out;
}

/* 主题感知取色：读 :root 当前主题的 CSS 变量（同 legacy themeColor） */
export function themeColor(name, fallback) {
  try {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fallback;
  } catch (e) {
    return fallback;
  }
}

/* 当前主题的全部图表用色（wrapper 自定义元素——如空态文案——直接取用） */
export function themeColors() {
  return {
    text: themeColor("--text", "#d7deeb"),
    muted: themeColor("--muted", "#8b96ad"),
    faint: themeColor("--faint", "#5c6880"),
    border: themeColor("--border", "#2a3550"),
    grid: themeColor("--chart-grid", "rgba(255,255,255,0.06)"),
    cross: themeColor("--chart-cross", "rgba(255,255,255,0.35)"),
    tooltipBg: themeColor("--tooltip-bg", "rgba(20,28,44,0.97)"),
  };
}

var AXIS_FONT_FAMILY = "ui-monospace, Menlo, Consolas, monospace";

/* ECharts 基础 option 片段（每次调用按当前主题重新取色） */
export function chartTheme() {
  var c = themeColors();
  function axis() {
    return {
      axisLine: { lineStyle: { color: c.border } },
      axisTick: { lineStyle: { color: c.border } },
      axisLabel: { color: c.muted, fontSize: 10, fontFamily: AXIS_FONT_FAMILY },
      splitLine: { lineStyle: { color: c.grid } },
    };
  }
  return {
    color: PALETTE.slice(),
    textStyle: { color: c.text },
    xAxis: axis(),
    yAxis: axis(),
    legend: {
      textStyle: { color: c.muted },
      inactiveColor: c.faint,
      pageTextStyle: { color: c.muted },
      pageIconColor: c.muted,
      pageIconInactiveColor: c.faint,
    },
    tooltip: {
      backgroundColor: c.tooltipBg,
      borderColor: c.border,
      textStyle: { color: c.text, fontSize: 12 },
      /* 悬停十字虚线（对齐 legacy 的 cross 观感） */
      axisPointer: { type: "line", lineStyle: { color: c.cross, type: "dashed" } },
    },
  };
}

/* 活图表注册表：wrapper 创建时登记、dispose 时注销返回值函数。
 * tab.onTheme → applyTheme() 统一刷新（注册表保证隐藏 tab 的图表同样跟随主题）。 */
var liveCharts = [];

export function registerChart(wrapper) {
  liveCharts.push(wrapper);
  return function () {
    var i = liveCharts.indexOf(wrapper);
    if (i >= 0) liveCharts.splice(i, 1);
  };
}

export function applyTheme(charts) {
  var list = charts || liveCharts.slice();
  list.forEach(function (c) { if (c && c.refreshTheme) c.refreshTheme(); });
}
