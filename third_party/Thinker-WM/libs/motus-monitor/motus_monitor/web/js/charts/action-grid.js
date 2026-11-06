/* Motus Monitor 前端 —— eval action 逐维曲线（阶段 2C：ECharts 重建）。
 *
 * 替换 actiongrid-legacy.js 的手搓 canvas 实现（对外接口 ActionPlot 逐字一致，
 * eval.js 只改 import 路径）。业务资产从 legacy 逐字保留：bandOf 分 band、
 * DEFAULT_BAND_RANGES、localStorage 覆盖合并、extractPlotData（eef_display
 * 优先，raw 回退）、IntersectionObserver 懒加载。
 *
 * 渲染策略：
 *   缩略图 —— 离屏 ECharts 实例（多 grid 一次渲染）→ getDataURL → dispose，
 *     容器放 <img>（避免每卡片一个活实例；样本卡片多时 32 grid × 2 序列的
 *     活实例开销不可接受）。容器尺寸变化 / y 编辑器改动时重渲染换 img。
 *   交互查看器 —— 单个活 ECharts 实例多 grid 小倍数：axisPointer link 联动
 *     十字线 + tooltip/状态栏读数、dataZoom inside 共享 x 视窗（滚轮缩放/
 *     拖拽平移/双击复位）、GT/Pred 显隐、y band/自适应、维度过滤、单击 cell
 *     进单维聚焦（createLineChart，←/→ 切换维度、Esc 返回）、CSV/PNG 导出。
 *
 * 与 legacy 的两处有意偏差（20260821 评审拍板方向）：
 *   1. legacy 网格模式下单击即进单维，导致"双击复位"实际不可达；本实现单击
 *      延迟 240ms（dblclick 取消），双击复位在网格模式真实生效。
 *   2. tooltip/十字线只在绘图区（grid）内触发，cell 标题/边距区不再出读数
 *      （状态栏读数仍按 legacy 几何映射覆盖整个 cell）。
 *
 * y 范围（用户拍板 20260820）：不同 dim 的 y 可不同，但同一 dim 跨 eps/数据
 * 集必须一致——按维度语义 band 固定（pos/rot6/gripper/other），默认值见
 * DEFAULT_BAND_RANGES（保证大部分 dim 不被裁剪）；超出范围的部分裁剪不显示。
 * eval tab 工具栏「y 轴范围」编辑器按 band 覆盖（localStorage 持久化），
 * 修改后全部已挂载图表重绘（redrawAll）。
 */
import { util } from "../util.js";
import { chartTheme, themeColors, themeColor } from "./theme.js";
import { createEChart } from "./base.js";
import { createLineChart } from "./line.js";

/* 默认固定 y 范围（20260820 按生产 eval 数据 |v| 分位数拍板：stage2_20260818_v1
 * run step 6000/7000 全源 meta.json 统计——pos p99.9≈3.7、rot6 max=1.0、
 * gripper p99≈1.5；robotmind ark 系夹爪物理量程 ±32 默认会被裁剪，属"极少
 * 部分维度"，用 y 编辑器恢复）。同一 dim 跨 eps/数据集共用所属 band 的范围。 */
var DEFAULT_BAND_RANGES = {
  pos: [-4, 4],
  rot6: [-1.1, 1.1],
  gripper: [-2, 2],
  other: [-2, 2],
};
var STORAGE_KEY = "motus.monitor.evalActionYRanges";
var MAX_DIMS = 32;
var COLS = 4;
var CELL_H = 96;          // css px / 维（缩略图）
var PAD_L = 36, PAD_R = 4, PAD_T = 14, PAD_B = 6;  // 缩略图 cell 内边距
var GT_COLOR = "#4f8cff";
var PRED_COLOR = "#ff9f43";
var MONO_FONT = "ui-monospace, Menlo, Consolas, monospace";

/* ------------------------------------------------------------ band 分类 */
function bandOf(label) {
  var s = String(label || "");
  if (s.indexOf("_pos[") >= 0) return "pos";
  if (s.indexOf("_rot6[") >= 0) return "rot6";
  if (s.indexOf("_gripper") >= 0) return "gripper";
  return "other";
}

/* ------------------------------------------------------------ 范围存取 */
function loadOverrides() {
  try {
    var raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    var obj = JSON.parse(raw);
    return obj && typeof obj === "object" ? obj : {};
  } catch (e) {
    return {};
  }
}

function saveOverrides(ov) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(ov)); } catch (e) {}
}

function validRange(r) {
  return Array.isArray(r) && r.length === 2 &&
    isFinite(Number(r[0])) && isFinite(Number(r[1])) &&
    Number(r[0]) < Number(r[1]);
}

/* 生效范围 = 默认 ∪ localStorage 覆盖（非法覆盖忽略） */
function getRanges() {
  var ov = loadOverrides();
  var out = {};
  Object.keys(DEFAULT_BAND_RANGES).forEach(function (band) {
    out[band] = validRange(ov[band])
      ? [Number(ov[band][0]), Number(ov[band][1])]
      : DEFAULT_BAND_RANGES[band].slice();
  });
  return out;
}

/* ------------------------------------------------------------ 数据提取 */
function validMatrix(m) {
  return Array.isArray(m) && m.length > 0 && Array.isArray(m[0]) && m[0].length > 0;
}

/* meta.json → {labels, gt, pred}（gt/pred 为 [T][D]，null 值=断点）；
 * 优先 EEF 规范显示空间，缺失回退 raw local；都无数据返回 null。 */
function extractPlotData(meta) {
  if (!meta || typeof meta !== "object") return null;
  var disp = meta.eef_display;
  if (disp && Array.isArray(disp.labels) && disp.labels.length > 0 &&
      (validMatrix(disp.gt) || validMatrix(disp.pred))) {
    return {
      labels: disp.labels.map(function (l) { return String(l); }),
      gt: validMatrix(disp.gt) ? disp.gt : null,
      pred: validMatrix(disp.pred) ? disp.pred : null,
    };
  }
  var gt = validMatrix(meta.gt_actions) ? meta.gt_actions : null;
  var pred = validMatrix(meta.predicted_actions) ? meta.predicted_actions : null;
  if (!gt && !pred) return null;
  var n = (gt || pred)[0].length;
  var raw = Array.isArray(meta.action_dim_labels) ? meta.action_dim_labels : [];
  var labels = [];
  for (var i = 0; i < n; i++) {
    labels.push(raw[i] != null && raw[i] !== "" ? String(raw[i]) : "dim " + i);
  }
  return { labels: labels, gt: gt, pred: pred };
}

/* ------------------------------------------------------------ 共用小工具 */
function fmtTick(v) {
  if (util && typeof util.fmtAxis === "function") return util.fmtAxis(v);
  return Math.abs(v) >= 100 ? String(Math.round(v)) : String(Number(v.toFixed(2)));
}

/* 自适应 y 范围：当前 x 视窗内 gt/pred 的 min/max ±5%（无有效数据回退 band 范围） */
function autoRange(data, dim, xView, fallback) {
  var t0 = 0, t1 = Infinity;
  if (xView) { t0 = Math.max(0, Math.floor(xView[0])); t1 = Math.ceil(xView[1]); }
  var ymin = Infinity, ymax = -Infinity;
  var mats = [data.gt, data.pred];
  for (var k = 0; k < mats.length; k++) {
    var mat = mats[k];
    if (!mat) continue;
    for (var t = t0; t < mat.length && t <= t1; t++) {
      var row = mat[t];
      if (!Array.isArray(row) || dim >= row.length) continue;
      var v = Number(row[dim]);
      if (!isFinite(v)) continue;
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }
  }
  if (ymin === Infinity) return fallback.slice();
  if (ymax - ymin < 1e-9) { ymin -= 0.5; ymax += 0.5; }
  var pad = (ymax - ymin) * 0.05;
  return [ymin - pad, ymax + pad];
}

/* 网格序列点集：非法值给 null（ECharts 断点开洞，对齐 legacy drawSeries 的
 * null/NaN 断线语义）；行结构损坏即截断（legacy 的 break）。 */
function colSeriesData(mat, dim) {
  var out = [];
  if (!mat) return out;
  for (var t = 0; t < mat.length; t++) {
    var row = mat[t];
    if (!Array.isArray(row) || dim >= row.length) break;
    var v = Number(row[dim]);
    out.push([t, isFinite(v) ? v : null]);
  }
  return out;
}

/* ------------------------------------------------------------ 网格 option 构建
 * 缩略图与交互查看器共用。layout：
 *   data/dims/cssW/cols/cellH/padL/padR/padT/padB/fontPx/lineW/textColor/tickColor
 *   yMode "band"|"auto"、xView [t0,t1]|null（仅 auto 取数用）、showGt/showPred
 *   interactive  true 时带 tooltip/axisPointer link/dataZoom（查看器）
 *   dzStart/dzEnd  dataZoom 百分比（查看器状态保持）
 *   filter         空态文案里的过滤词
 * 返回 {option, cells, seriesDims}：cells 供查看器几何拾取（cell 矩形 + plot 矩形
 * + yAxisId），seriesDims 供 tooltip formatter 反查维度。 */
function buildGridOption(layout) {
  var data = layout.data;
  var dims = layout.dims;
  var base = chartTheme();
  var colors = themeColors();
  var ranges = getRanges();
  var ref = data.gt || data.pred;
  var T = ref ? ref.length : 0;
  var xMax = T > 1 ? T - 1 : 1;
  var cellW = layout.cssW / layout.cols;
  var borderSoft = themeColor("--border-soft", "rgba(128,128,128,0.25)");

  var grids = [], xAxes = [], yAxes = [], series = [], graphic = [], cells = [];
  var seriesDims = [];

  for (var i = 0; i < dims.length; i++) {
    var dim = dims[i];
    var band = bandOf(data.labels[dim]);
    var yr = layout.yMode === "auto"
      ? autoRange(data, dim, layout.xView || null, ranges[band])
      : ranges[band];
    var cellX = (i % layout.cols) * cellW;
    var cellY = Math.floor(i / layout.cols) * layout.cellH;
    var gridLeft = cellX + layout.padL;
    var gridTop = cellY + layout.padT;
    var gridW = cellW - layout.padL - layout.padR;
    var gridH = layout.cellH - layout.padT - layout.padB;
    var yaId = "ya" + i;

    grids.push({
      left: gridLeft, top: gridTop, width: gridW, height: gridH,
      show: true, borderWidth: 1, borderColor: borderSoft,
    });
    xAxes.push({
      gridIndex: i, type: "value", min: 0, max: xMax, show: false,
    });
    /* y 只画上下限两个 tick（interval=整段），对齐 legacy 的 ymin/ymax 标注 */
    yAxes.push({
      id: yaId, gridIndex: i, type: "value", min: yr[0], max: yr[1],
      interval: yr[1] - yr[0],
      axisLine: { show: false }, axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: {
        color: layout.tickColor, fontSize: layout.fontPx, fontFamily: MONO_FONT,
        formatter: fmtTick,
      },
    });
    /* 维度名（截断对齐 legacy：按 cell 宽 truncate） */
    graphic.push({
      type: "text", left: cellX + 4,
      top: cellY + Math.round((layout.padT - layout.fontPx) / 2),
      silent: true,
      style: {
        text: data.labels[dim] || ("dim " + dim),
        fill: layout.textColor, fontSize: layout.fontPx, fontFamily: MONO_FONT,
        width: cellW - 8, overflow: "truncate", ellipsis: "…",
      },
    });
    if (layout.showGt !== false && data.gt) {
      seriesDims.push(dim);
      series.push({
        id: "gt" + dim, name: "GT", type: "line",
        xAxisIndex: i, yAxisIndex: i, data: colSeriesData(data.gt, dim),
        showSymbol: false, symbol: "circle",
        lineStyle: { color: GT_COLOR, width: layout.lineW },
        itemStyle: { color: GT_COLOR },
        silent: !layout.interactive, animation: false,
      });
    }
    if (layout.showPred !== false && data.pred) {
      seriesDims.push(dim);
      series.push({
        id: "pred" + dim, name: "Pred", type: "line",
        xAxisIndex: i, yAxisIndex: i, data: colSeriesData(data.pred, dim),
        showSymbol: false, symbol: "circle",
        lineStyle: { color: PRED_COLOR, width: layout.lineW, type: "dashed" },
        itemStyle: { color: PRED_COLOR },
        silent: !layout.interactive, animation: false,
      });
    }
    cells.push({
      dim: dim, cellX: cellX, cellY: cellY,
      gridLeft: gridLeft, gridTop: gridTop, gridW: gridW, gridH: gridH,
      yAxisId: yaId,
    });
  }

  if (!dims.length) {
    graphic.push({
      type: "text", left: "center", top: "middle", silent: true,
      style: {
        text: "无匹配维度（过滤：" + (layout.filter || "") + "）",
        fontSize: 12, fontFamily: "sans-serif", fill: colors.faint,
      },
    });
  }

  var option = {
    animation: false,
    textStyle: base.textStyle,
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series: series,
    graphic: graphic,
  };

  if (layout.interactive && dims.length) {
    option.axisPointer = { link: [{ xAxisIndex: "all" }] };
    option.tooltip = {
      trigger: "axis",
      confine: true,
      backgroundColor: base.tooltip.backgroundColor,
      borderColor: base.tooltip.borderColor,
      textStyle: base.tooltip.textStyle,
      axisPointer: base.tooltip.axisPointer,
      padding: [7, 10],
      extraCssText: "box-shadow:0 6px 20px var(--shadow);white-space:nowrap;",
      formatter: layout.tipFormatter,
    };
    option.dataZoom = [{
      type: "inside",
      xAxisIndex: dims.map(function (_, k) { return k; }),
      filterMode: "none",
      zoomOnMouseWheel: true,
      moveOnMouseMove: true,
      moveOnMouseWheel: false,
      minValueSpan: 4,          // legacy 滚轮缩放的最小视窗 4 帧
      start: layout.dzStart != null ? layout.dzStart : 0,
      end: layout.dzEnd != null ? layout.dzEnd : 100,
    }];
  }
  return { option: option, cells: cells, seriesDims: seriesDims };
}

/* ------------------------------------------------------------ 离屏渲染（缩略图）
 * 每次渲染：隐藏 div 上 echarts.init → setOption → getDataURL → dispose。
 * 主机（host）模块级复用；dispose 保证同节点可重复 init。 */
var _offHost = null;

function offscreenHost() {
  if (_offHost && _offHost.isConnected) return _offHost;
  _offHost = document.createElement("div");
  _offHost.style.cssText =
    "position:fixed;left:-100000px;top:0;pointer-events:none;z-index:-1;";
  document.body.appendChild(_offHost);
  return _offHost;
}

function renderGridImage(data, cssW, cssH) {
  if (typeof window.echarts === "undefined") return null;
  var dims = [];
  for (var d = 0; d < Math.min(data.labels.length, MAX_DIMS); d++) dims.push(d);
  var host = offscreenHost();
  host.style.width = cssW + "px";
  host.style.height = cssH + "px";
  var inst = window.echarts.init(host, null, { renderer: "canvas" });
  try {
    var built = buildGridOption({
      data: data, dims: dims, cssW: cssW, cols: COLS, cellH: CELL_H,
      padL: PAD_L, padR: PAD_R, padT: PAD_T, padB: PAD_B,
      fontPx: 9, lineW: 1.2,
      textColor: themeColor("--muted", "#8b96ad"),
      tickColor: themeColor("--muted", "#8b96ad"),
      yMode: "band", xView: null, interactive: false,
    });
    inst.setOption(built.option);
    return inst.getDataURL({
      type: "png",
      pixelRatio: window.devicePixelRatio || 1,
      backgroundColor: "rgba(0,0,0,0)",
    });
  } finally {
    inst.dispose();
  }
}

/* ------------------------------------------------------------ 缩略图绘制/挂载 */
var mounted = [];  // {container, metaUrl, fallbackImgUrl, data, io, ro}

function fallbackImage(state) {
  var container = state.container;
  if (!container.isConnected) return;
  container.innerHTML = "";
  if (!state.fallbackImgUrl) {
    var hint = document.createElement("div");
    hint.className = "eval-action-plot-hint muted";
    hint.textContent = "无 action 曲线数据";
    container.appendChild(hint);
    return;
  }
  /* 旧 run 回退：训练侧曾落盘的 actions.png 直接显示 */
  var img = document.createElement("img");
  img.className = "eval-media-thumb-img";
  img.src = state.fallbackImgUrl;
  img.alt = "actions（pred vs GT）";
  img.loading = "lazy";
  container.appendChild(img);
}

function draw(state) {
  var container = state.container;
  var data = state.data;
  if (!container.isConnected) return;
  container.innerHTML = "";
  var cssW = container.clientWidth || 480;
  var dims = Math.min(data.labels.length, MAX_DIMS);
  var rows = Math.ceil(dims / COLS);
  var cssH = rows * CELL_H;
  var url = null;
  try {
    url = renderGridImage(data, cssW, cssH);
  } catch (e) {
    console.error("[action-grid] 缩略图离屏渲染失败", e);
  }
  if (!url) { fallbackImage(state); return; }
  var img = document.createElement("img");
  img.className = "eval-action-grid-img";
  img.alt = "actions（pred vs GT）";
  img.src = url;
  /* 内联复刻 .eval-action-plot canvas 的观感（style.css 不动；位图透明底，
   * 卡片底色经 img 背景透出，同 legacy canvas 的透明绘制区） */
  img.style.display = "block";
  img.style.width = "100%";
  img.style.border = "1px solid " + themeColor("--border-soft", "rgba(128,128,128,0.25)");
  img.style.borderRadius = "7px";
  img.style.background = themeColor("--card-2", "transparent");
  container.appendChild(img);
  state.img = img;
}

function load(state) {
  fetch(state.metaUrl, { credentials: "same-origin" })
    .then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return resp.json();
    })
    .then(function (meta) {
      var data = extractPlotData(meta);
      if (!data) { fallbackImage(state); return; }
      state.data = data;
      draw(state);
      /* 容器尺寸变化（侧栏折叠/窗口缩放）时重绘；离屏渲染比重画 canvas 贵，
       * 拖动窗口时 RO 连发——120ms 尾随防抖（reset 后旧容器已 detach，draw 空转） */
      if (typeof ResizeObserver !== "undefined" && !state.ro) {
        state.ro = new ResizeObserver(util.debounce(function () {
          if (state.data) draw(state);
        }, 120));
        state.ro.observe(state.container);
      }
    })
    .catch(function () { fallbackImage(state); });
}

function mount(container, metaUrl, fallbackImgUrl) {
  if (!container || !metaUrl) {
    if (container && fallbackImgUrl) {
      fallbackImage({ container: container, fallbackImgUrl: fallbackImgUrl });
    }
    return;
  }
  var state = {
    container: container, metaUrl: metaUrl,
    fallbackImgUrl: fallbackImgUrl || null,
    data: null, io: null, ro: null, img: null,
  };
  var loading = document.createElement("div");
  loading.className = "eval-action-plot-hint muted";
  loading.textContent = "action 曲线加载中…";
  container.appendChild(loading);
  if (typeof IntersectionObserver !== "undefined") {
    state.io = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        if (entries[i].isIntersecting) {
          state.io.disconnect();
          state.io = null;
          load(state);
          break;
        }
      }
    }, { rootMargin: "300px" });
    state.io.observe(container);
  } else {
    load(state);
  }
  mounted.push(state);
}

/* eval tab 重渲染样本网格前调用：断开 observer、清挂载表（旧 DOM 随之销毁） */
function reset() {
  for (var i = 0; i < mounted.length; i++) {
    if (mounted[i].io) mounted[i].io.disconnect();
    if (mounted[i].ro) mounted[i].ro.disconnect();
  }
  mounted = [];
}

/* ============================================================ 交互查看器
 * openInteractive(container, metaUrl, opts) —— eval.js 弹层内挂载的大图：
 * 网格模式：axisPointer link 联动十字线 + tooltip/状态栏读数、dataZoom
 *   inside 共享 x 视窗（滚轮缩放/拖拽平移/双击复位）、维度过滤、GT/Pred
 *   显隐、y band/自适应、单击 cell 进单维聚焦、CSV/PNG 导出；
 * 单维聚焦：createLineChart（自带 tooltip/缩放/平移/双击复位），
 *   ←/→ 切换维度、Esc 返回网格。
 * 返回 handle {destroy, handleEsc, redraw}；destroy 由弹层关闭时调用。
 */
var activeViewer = null;

/* 查看器网格的视觉参数（对齐 legacy 交互大图的"大字号/粗线/深色文字"） */
var V_PAD_L = 46, V_PAD_R = 8, V_PAD_T = 20, V_PAD_B = 8;
var V_CELL_H = 128;
var V_FONT_PX = 11;
var V_LINE_W = 1.8;
var CLICK_DELAY = 240;  // 单击进单维的延迟（双击复位窗口，见文件头偏差说明 1）

function openInteractive(container, metaUrl, opts) {
  opts = opts || {};
  var esc = util.esc, el2 = util.el;
  var STATUS_HINT =
    "hover 查看数值 · 滚轮缩放 · 拖拽平移 · 双击复位 · 单击 cell 进入单维（←/→ 切换，Esc 返回）";

  var v = {
    destroyed: false,
    data: null,
    xView: null,          // [t0,t1] | null（网格模式共享 x 视窗）
    dzStart: null,        // dataZoom 百分比状态（apply 重建时保持视窗）
    dzEnd: null,
    showGt: true,
    showPred: true,
    yMode: "band",        // "band" | "auto"
    filter: "",
    single: null,         // 单维聚焦的 dim 索引 | null
    cols: COLS,
    cellH: V_CELL_H,
    fileBase: String(opts.title || "actions").replace(/[^\w.-]+/g, "_") || "actions",
    canvasWrap: null, gridChart: null, statusEl: null,
    cells: [], seriesDims: [],
    lineChart: null, singleHost: null, ro: null,
    hoverDim: null, hoverT: null,   // 最近一次几何拾取（tooltip 对齐状态栏用）
    dzBound: false, clickTimer: null, downPos: null, suppressClick: false,
  };

  /* ------------------------------------------------------------ 工具 */
  function timeLen() {
    var ref = v.data && (v.data.gt || v.data.pred);
    return ref ? ref.length : 0;
  }

  function filteredDims() {
    if (!v.data) return [];
    var n = Math.min(v.data.labels.length, MAX_DIMS);
    var f = v.filter;
    var out = [];
    for (var d = 0; d < n; d++) {
      if (f && String(v.data.labels[d]).toLowerCase().indexOf(f) < 0) continue;
      out.push(d);
    }
    return out;
  }

  function valAt(mat, t, d) {
    if (!mat || t == null || t >= mat.length) return null;
    var row = mat[t];
    if (!Array.isArray(row) || d >= row.length) return null;
    var val = Number(row[d]);
    return isFinite(val) ? val : null;
  }

  function fmtV(x) { return x == null ? "—" : util.fmtVal(x); }

  /* ------------------------------------------------------------ 网格 option */
  function buildViewerOption() {
    var cssW = v.canvasWrap.clientWidth || 800;
    var built = buildGridOption({
      data: v.data, dims: filteredDims(), cssW: cssW,
      cols: v.cols, cellH: v.cellH,
      padL: V_PAD_L, padR: V_PAD_R, padT: V_PAD_T, padB: V_PAD_B,
      fontPx: V_FONT_PX, lineW: V_LINE_W,
      textColor: themeColor("--text", "#22304a"),
      tickColor: themeColor("--text", "#22304a"),
      yMode: v.yMode, xView: v.xView,
      showGt: v.showGt, showPred: v.showPred,
      interactive: true, tipFormatter: fmtGridTooltip,
      dzStart: v.dzStart, dzEnd: v.dzEnd,
      filter: v.filter,
    });
    v.cells = built.cells;
    v.seriesDims = built.seriesDims;
    return built.option;
  }

  /* host 高度随过滤后维度数变化（行数 × cellH；0 维留 1 行放空态文案） */
  function syncHostHeight() {
    if (!v.gridChart) return;
    var rows = Math.max(1, Math.ceil(filteredDims().length / v.cols));
    v.gridChart.host.style.height = rows * v.cellH + "px";
  }

  function refreshGrid() {
    if (v.destroyed || !v.data || v.single !== null || !v.gridChart) return;
    syncHostHeight();
    v.gridChart.apply();
    bindDataZoom();
  }

  /* echarts 实例可能推迟初始化（0 尺寸容器）：datazoom 监听见到实例再挂 */
  function bindDataZoom() {
    if (v.dzBound) return;
    var inst = v.gridChart && v.gridChart.chart();
    if (!inst) return;
    v.dzBound = true;
    inst.on("datazoom", onDataZoom);
  }

  /* ------------------------------------------------------------ hover 读数 */
  function readout(dim, t) {
    /* → {html, text} | null */
    if (dim == null || t == null || !v.data) return null;
    var g = valAt(v.data.gt, t, dim), p = valAt(v.data.pred, t, dim);
    var label = String(v.data.labels[dim]);
    var rows = [];
    var parts = ["t=" + t, label];
    if (v.showGt) {
      rows.push(ttRow(GT_COLOR, "GT", fmtV(g)));
      parts.push("GT " + fmtV(g));
    }
    if (v.showPred) {
      rows.push(ttRow(PRED_COLOR, "Pred", fmtV(p)));
      parts.push("Pred " + fmtV(p));
    }
    if (g != null && p != null) {
      var diff = Math.abs(g - p);
      rows.push(ttRow(themeColor("--muted", "#8b96ad"), "|Δ|", fmtV(diff)));
      parts.push("|Δ| " + fmtV(diff));
    }
    return {
      html: '<div style="color:var(--muted);margin-bottom:4px">t = ' + t + " · " +
        esc(label) + "</div>" + rows.join(""),
      text: parts.join(" · "),
    };
  }

  /* 行内样式复刻 .chart-tooltip 观感（同 charts/line.js fmtTooltip 的理由：
   * echarts tooltip 空壳 div 带上 chart-tooltip 类会被 CSS 描边成可见空胶囊） */
  function ttRow(color, name, val) {
    return '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
      '<span style="width:8px;height:8px;border-radius:2px;flex:none;background:' + color + '"></span>' +
      "<span>" + esc(name) + '</span><span style="margin-left:auto;padding-left:12px">' +
      esc(val) + "</span></div>";
  }

  /* echarts tooltip formatter：seriesIndex 反查 cell 维度；t 优先取最近一次
   * 几何拾取（与状态栏同源；zr 的 tooltip 更新早于冒泡到 host 的 mousemove，
   * 最多落后一个事件），否则连续 x 值取整；数值一律回查原始矩阵 */
  function fmtGridTooltip(params) {
    if (!params || !params.length || !v.data) return "";
    var dim = v.seriesDims[params[0].seriesIndex];
    var T = timeLen();
    if (dim == null || T <= 0) return "";
    var t = (v.hoverDim === dim && v.hoverT != null) ? v.hoverT
      : Math.round(Number(params[0].axisValue));
    if (!isFinite(t)) return "";
    t = Math.max(0, Math.min(T - 1, t));
    var ro = readout(dim, t);
    return ro ? ro.html : "";
  }

  /* 光标位置 → (dim, t)；几何映射与 legacy pickCell 逐字一致（覆盖整个 cell
   * 含标题/边距区；无效区域返回 {dim:null, t:null}） */
  function pickCell(mx, my) {
    var T = timeLen();
    if (!v.data || T <= 0) return { dim: null, t: null };
    var cssW = v.canvasWrap.clientWidth || 800;
    var cellW = cssW / v.cols;
    var col = Math.floor(mx / cellW), row = Math.floor(my / v.cellH);
    if (col < 0 || col >= v.cols || row < 0) return { dim: null, t: null };
    var dims = filteredDims();
    var di = row * v.cols + col;
    if (di >= dims.length) return { dim: null, t: null };
    var left = col * cellW + V_PAD_L, w = cellW - V_PAD_L - V_PAD_R;
    var xv = v.xView || [0, Math.max(0, T - 1)];
    var frac = w > 0 ? (mx - left) / w : 0;
    var t = Math.round(xv[0] + frac * (xv[1] - xv[0]));
    if (t < 0) t = 0;
    if (t > T - 1) t = T - 1;
    return { dim: dims[di], t: t };
  }

  /* ------------------------------------------------------------ 网格事件 */
  function onGridMouseMove(e) {
    if (v.single !== null) return;
    var rect = v.gridChart.host.getBoundingClientRect();
    var pick = pickCell(e.clientX - rect.left, e.clientY - rect.top);
    v.hoverDim = pick.dim;
    v.hoverT = pick.t;
    var ro = readout(pick.dim, pick.t);
    if (v.statusEl) v.statusEl.textContent = ro ? ro.text : STATUS_HINT;
  }

  function onGridMouseLeave() {
    v.hoverDim = null;
    v.hoverT = null;
    if (v.statusEl) v.statusEl.textContent = STATUS_HINT;
  }

  function onGridMouseDown(e) {
    if (e.button !== 0) return;
    v.downPos = { x: e.clientX, y: e.clientY };
  }

  /* 拖拽平移（dataZoom inside moveOnMouseMove）后抑制紧随的 click，否则误进
   * 单维；setTimeout 让 click 先派发（legacy onDragEnd 语义） */
  function onGridMouseUp(e) {
    if (!v.downPos) return;
    var dx = e.clientX - v.downPos.x, dy = e.clientY - v.downPos.y;
    v.downPos = null;
    if (dx * dx + dy * dy > 9) {
      v.suppressClick = true;
      setTimeout(function () { v.suppressClick = false; }, 0);
    }
  }

  function onGridClick(e) {
    if (v.suppressClick) { v.suppressClick = false; return; }
    if (v.destroyed || v.single !== null) return;
    var rect = v.gridChart.host.getBoundingClientRect();
    var pick = pickCell(e.clientX - rect.left, e.clientY - rect.top);
    if (pick.dim == null) return;
    /* 延迟进单维：窗口期内 dblclick 取消（legacy 单击即进导致双击复位不可达） */
    if (v.clickTimer) clearTimeout(v.clickTimer);
    v.clickTimer = setTimeout(function () {
      v.clickTimer = null;
      if (v.destroyed || v.single !== null) return;
      enterSingle(pick.dim);
    }, CLICK_DELAY);
  }

  function onGridDblClick() {
    if (v.clickTimer) { clearTimeout(v.clickTimer); v.clickTimer = null; }
    resetGridView();
  }

  /* 复位共享 x 视窗到全量（双击/「重置视图」按钮共用） */
  function resetGridView() {
    var inst = v.gridChart && v.gridChart.chart();
    if (!inst) { v.dzStart = null; v.dzEnd = null; v.xView = null; return; }
    /* 双击复位 = 回全量视窗（同 charts/line.js，不用 restore） */
    inst.dispatchAction({ type: "dataZoom", start: 0, end: 100 });
  }

  /* datazoom 事件 → 同步 xView/百分比状态；auto 模式下按新视窗重算 y 范围 */
  function onDataZoom() {
    var inst = v.gridChart && v.gridChart.chart();
    if (!inst) return;
    var opt = inst.getOption();
    var dz = opt && opt.dataZoom && opt.dataZoom[0];
    if (!dz) return;
    var T = timeLen();
    var ext = [0, Math.max(0, T - 1)];
    var s = util.isNum(dz.start) ? dz.start : 0;
    var e = util.isNum(dz.end) ? dz.end : 100;
    v.dzStart = s;
    v.dzEnd = e;
    v.xView = (s <= 0 && e >= 100) ? null
      : [ext[0] + ((ext[1] - ext[0]) * s) / 100, ext[0] + ((ext[1] - ext[0]) * e) / 100];
    if (v.yMode !== "auto") return;
    var ranges = getRanges();
    var ya = [];
    for (var i = 0; i < v.cells.length; i++) {
      var c = v.cells[i];
      var yr = autoRange(v.data, c.dim, v.xView, ranges[bandOf(v.data.labels[c.dim])]);
      ya.push({ id: c.yAxisId, min: yr[0], max: yr[1], interval: yr[1] - yr[0] });
    }
    /* 仅按 id 合并 yAxis min/max（不动 dataZoom/series，缩放链路无全量重建） */
    inst.setOption({ yAxis: ya });
  }

  function onKey(e) {
    if (v.destroyed || v.single === null) return;
    if (e.target && /^(input|textarea|select)$/i.test(e.target.tagName)) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    var dims = filteredDims();
    if (!dims.length) return;
    e.preventDefault();
    var idx = dims.indexOf(v.single);
    if (idx < 0) idx = 0;
    idx = (idx + (e.key === "ArrowRight" ? 1 : dims.length - 1)) % dims.length;
    enterSingle(dims[idx]);
  }

  /* ------------------------------------------------------------ 单维聚焦 */
  function pointsFor(mat, dim) {
    var pts = [];
    if (!mat) return pts;
    for (var t = 0; t < mat.length; t++) {
      var row = mat[t];
      if (!Array.isArray(row) || dim >= row.length) break;
      var val = Number(row[dim]);
      if (isFinite(val)) pts.push([t, val]);
    }
    return pts;
  }

  function enterSingle(dim) {
    if (v.lineChart) { v.lineChart.dispose(); v.lineChart = null; }
    v.single = dim;
    /* 网格 host 摘下但实例保留（gridChart 不 dispose，返回网格时重新挂回） */
    v.canvasWrap.innerHTML = "";
    var label = String(v.data.labels[dim]);
    var head = el2("div", "ap-single-head");
    var back = el2("button", "btn", { text: "← 返回网格", attrs: { type: "button" } });
    back.addEventListener("click", backToGrid);
    head.appendChild(back);
    head.appendChild(el2("span", "ap-single-title mono", {
      text: label + "（band: " + bandOf(label) + "）· ←/→ 切换维度 · Esc 返回",
    }));
    v.canvasWrap.appendChild(head);
    var chartWrap = el2("div", "ap-single-chart");
    var host = el2("div");
    host.style.width = "100%";
    host.style.height = "100%";
    host.style.cursor = "grab";
    chartWrap.appendChild(host);
    v.canvasWrap.appendChild(chartWrap);
    v.singleHost = host;
    var series = [];
    if (v.showGt) series.push({ name: "GT", color: GT_COLOR, points: pointsFor(v.data.gt, dim) });
    if (v.showPred) series.push({ name: "Pred", color: PRED_COLOR, points: pointsFor(v.data.pred, dim) });
    v.lineChart = createLineChart(host, { xLabel: "frame" });
    v.lineChart.setSeries(series);
    if (v.statusEl) {
      v.statusEl.textContent =
        "单维聚焦：" + label + " · hover 读数 · 滚轮缩放 · 拖拽平移 · 双击复位";
    }
  }

  function backToGrid() {
    if (v.lineChart) { v.lineChart.dispose(); v.lineChart = null; }
    v.singleHost = null;
    v.single = null;
    v.canvasWrap.innerHTML = "";
    v.canvasWrap.appendChild(v.gridChart.host);
    if (v.statusEl) v.statusEl.textContent = STATUS_HINT;
    refreshGrid();
  }

  /* ------------------------------------------------------------ 导出 */
  function exportCsv() {
    if (!v.data) return;
    var dims = filteredDims();
    var T = timeLen();
    var san = function (s) { return String(s).replace(/[^\w.-]+/g, "_"); };
    var head = ["t"];
    dims.forEach(function (d) { head.push("gt_" + san(v.data.labels[d])); });
    dims.forEach(function (d) { head.push("pred_" + san(v.data.labels[d])); });
    var lines = [head.join(",")];
    var cellV = function (mat, t, d) {
      var val = valAt(mat, t, d);
      return val == null ? "" : String(val);
    };
    for (var t = 0; t < T; t++) {
      var row = [String(t)];
      dims.forEach(function (d) { row.push(cellV(v.data.gt, t, d)); });
      dims.forEach(function (d) { row.push(cellV(v.data.pred, t, d)); });
      lines.push(row.join(","));
    }
    util.saveBlob(new Blob([lines.join("\n")], { type: "text/csv" }), v.fileBase + ".csv");
  }

  /* PNG：取当前视图 echarts 实例的 getDataURL（单维聚焦时经 getInstanceByDom
   * 拿 createLineChart 的实例；网格模式取多 grid 实例），透明底与 legacy
   * canvas.toBlob 导出一致 */
  function exportPng() {
    var inst = null;
    if (v.single !== null) {
      if (v.singleHost && typeof window.echarts !== "undefined") {
        inst = window.echarts.getInstanceByDom(v.singleHost);
      }
    } else {
      inst = v.gridChart && v.gridChart.chart();
    }
    if (!inst) return;
    var url = inst.getDataURL({
      type: "png",
      pixelRatio: window.devicePixelRatio || 1,
      backgroundColor: "rgba(0,0,0,0)",
    });
    fetch(url).then(function (resp) { return resp.blob(); }).then(function (blob) {
      util.saveBlob(blob, v.fileBase + ".png");
    });
  }

  /* ------------------------------------------------------------ toolbar */
  function mkChip(text, color, onClick) {
    var chip = el2("button", "ap-chip", { attrs: { type: "button" } });
    if (color) {
      var sw = el2("span", "ap-swatch");
      sw.style.background = color;
      chip.appendChild(sw);
    }
    chip.appendChild(el2("span", "", { text: text }));
    chip.addEventListener("click", function () { onClick(chip); });
    return chip;
  }

  function refreshViz() {
    if (v.single !== null) enterSingle(v.single);
    else refreshGrid();
  }

  function buildToolbar() {
    var bar = el2("div", "ap-toolbar");
    var gtChip = mkChip("GT", GT_COLOR, function (chipEl) {
      v.showGt = !v.showGt;
      chipEl.classList.toggle("ap-chip-off", !v.showGt);
      refreshViz();
    });
    var predChip = mkChip("Pred", PRED_COLOR, function (chipEl) {
      v.showPred = !v.showPred;
      chipEl.classList.toggle("ap-chip-off", !v.showPred);
      refreshViz();
    });
    bar.appendChild(gtChip);
    bar.appendChild(predChip);

    var yBtn = el2("button", "btn", { text: "y: band", attrs: { type: "button", title: "band=按维度语义固定范围（同 dim 跨样本一致）；自适应=按当前视窗数据缩放" } });
    yBtn.addEventListener("click", function () {
      v.yMode = v.yMode === "band" ? "auto" : "band";
      yBtn.textContent = v.yMode === "band" ? "y: band" : "y: 自适应";
      refreshGrid();
    });
    bar.appendChild(yBtn);

    var filter = el2("input", "ap-filter", {
      attrs: { type: "search", placeholder: "过滤维度…（如 gripper）" },
    });
    filter.addEventListener("input", function () {
      v.filter = filter.value.trim().toLowerCase();
      if (v.single === null) refreshGrid();
    });
    bar.appendChild(filter);

    bar.appendChild(el2("span", "ap-spacer"));

    var reset = el2("button", "btn", { text: "重置视图", attrs: { type: "button" } });
    reset.addEventListener("click", function () {
      if (v.single !== null && v.lineChart) {
        v.lineChart.resetView();
      } else {
        resetGridView();
      }
    });
    bar.appendChild(reset);

    var csv = el2("button", "btn", { text: "导出 CSV", attrs: { type: "button", title: "导出当前过滤后维度的 gt/pred 逐帧数据" } });
    csv.addEventListener("click", exportCsv);
    bar.appendChild(csv);

    var png = el2("button", "btn", { text: "导出 PNG", attrs: { type: "button", title: "导出当前画布截图" } });
    png.addEventListener("click", exportPng);
    bar.appendChild(png);
    return bar;
  }

  /* ------------------------------------------------------------ 构建/加载 */
  function build() {
    container.innerHTML = "";
    container.appendChild(buildToolbar());
    v.canvasWrap = el2("div", "ap-canvas-wrap");
    var ghost = el2("div");
    /* 内联复刻 .ap-canvas-wrap > canvas 的观感（style.css 不动） */
    ghost.style.display = "block";
    ghost.style.width = "100%";
    ghost.style.border = "1px solid " + themeColor("--border-soft", "rgba(128,128,128,0.25)");
    ghost.style.borderRadius = "7px";
    ghost.style.background = themeColor("--card-2", "transparent");
    v.canvasWrap.appendChild(ghost);
    container.appendChild(v.canvasWrap);
    v.statusEl = el2("div", "ap-status", { text: STATUS_HINT });
    container.appendChild(v.statusEl);

    v.gridChart = createEChart(ghost, {
      option: buildViewerOption,
      /* grid/graphic 一并替换：维度过滤减少 cell 时清掉残留 */
      replaceMerge: ["series", "xAxis", "yAxis", "grid", "graphic"],
    });
    syncHostHeight();

    ghost.addEventListener("mousemove", onGridMouseMove);
    ghost.addEventListener("mouseleave", onGridMouseLeave);
    ghost.addEventListener("mousedown", onGridMouseDown);
    ghost.addEventListener("mouseup", onGridMouseUp);
    ghost.addEventListener("click", onGridClick);
    ghost.addEventListener("dblclick", onGridDblClick);
    document.addEventListener("keydown", onKey);
    if (typeof ResizeObserver !== "undefined") {
      /* 全量重建比 base.js 的 inst.resize() 贵，100ms 尾随防抖（拖动窗口时
       * 图表先被拉伸、防抖后按新宽度重排 grid 像素位置） */
      v.ro = new ResizeObserver(util.debounce(function () {
        if (v.single === null) refreshGrid();
      }, 100));
      v.ro.observe(v.canvasWrap);
    }
    refreshGrid();
  }

  function showError(msg) {
    container.innerHTML = "";
    var box = el2("div", "eval-action-plot-hint muted");
    box.appendChild(el2("span", "", { text: msg + " " }));
    var retry = el2("button", "btn", { text: "重试", attrs: { type: "button" } });
    retry.addEventListener("click", start);
    box.appendChild(retry);
    container.appendChild(box);
  }

  function start() {
    container.innerHTML = "";
    container.appendChild(el2("div", "eval-action-plot-hint muted", { text: "action 曲线加载中…" }));
    fetch(metaUrl, { credentials: "same-origin" })
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(function (meta) {
        if (v.destroyed) return;
        var data = extractPlotData(meta);
        if (!data) { showError("无 action 曲线数据（meta.json 缺 eef_display / gt_actions）。"); return; }
        v.data = data;
        build();
      })
      .catch(function (err) {
        if (v.destroyed) return;
        showError("加载失败：" + (err && err.message || err));
      });
  }

  start();

  /* ------------------------------------------------------------ handle */
  var handle = {
    destroy: function () {
      if (v.destroyed) return;
      v.destroyed = true;
      document.removeEventListener("keydown", onKey);
      if (v.clickTimer) { clearTimeout(v.clickTimer); v.clickTimer = null; }
      if (v.ro) v.ro.disconnect();
      if (v.lineChart) { v.lineChart.dispose(); v.lineChart = null; }
      if (v.gridChart) { v.gridChart.dispose(); v.gridChart = null; }
      if (activeViewer === handle) activeViewer = null;
    },
    /* Esc：单维聚焦时返回网格并消费事件；否则不消费（交给弹层关闭） */
    handleEsc: function () {
      if (v.destroyed) return false;
      if (v.single !== null) { backToGrid(); return true; }
      return false;
    },
    redraw: function () {
      if (!v.destroyed && v.single === null) refreshGrid();
    },
  };
  activeViewer = handle;
  return handle;
}

/* y 范围变更后重绘全部已挂载图表（缩略图重渲染换 img；不重建挂载 DOM）；
 * 交互查看器网格模式同步重绘 */
function redrawAll() {
  for (var i = 0; i < mounted.length; i++) {
    if (mounted[i].data) draw(mounted[i]);
  }
  if (activeViewer) activeViewer.redraw();
}

function setBandRange(band, range) {
  if (!Object.prototype.hasOwnProperty.call(DEFAULT_BAND_RANGES, band)) return;
  var ov = loadOverrides();
  if (range === null) delete ov[band];
  else if (validRange(range)) ov[band] = [Number(range[0]), Number(range[1])];
  else return;
  saveOverrides(ov);
  redrawAll();
}

function resetBandRanges() {
  try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
  redrawAll();
}

/* ------------------------------------------------------------ 导出 */
export const ActionPlot = {
  DEFAULT_BAND_RANGES: DEFAULT_BAND_RANGES,
  bandOf: bandOf,
  getRanges: getRanges,
  setBandRange: setBandRange,
  resetBandRanges: resetBandRanges,
  extractPlotData: extractPlotData,
  mount: mount,
  reset: reset,
  redrawAll: redrawAll,
  openInteractive: openInteractive,
};
