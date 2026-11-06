/* Motus Monitor 前端 —— ECharts 折线/散点图统一封装（阶段 2A 新基建）。
 * 覆盖旧 charts/legacy.js LineChart 的全部场景（curves 四图、compare 叠加图；
 * dataset 两图需要 markArea/markLine + 点击 seek，走 charts/base.js 裸封装，
 * 未复用本模块）：
 *
 *   createLineChart(el, {xIsTime=false, xLabel="step", yLabel?, legend=true})
 *   → wrapper 对象：
 *     setSeries(list)    [{name, points:[{x,y}|[x,y]…], color?, scatter?, emaAlpha?, legend?}]
 *                        普通序列画折线（showSymbol:false）；scatter:true 画原始值
 *                        散点（小点、alpha 0.45）；emaAlpha 序列把自身 points 经
 *                        emaSmooth 后画 EMA 折线（"散点+EMA"由调用方给两条序列，
 *                        wrapper 不内嵌）；legend:false 不进图例
 *     setSmoothing(a)    全局 EMA 滑块（仅普通折线序列生效；tooltip 仍给原始值）
 *     setLogY(on)        y 轴 log（过滤非正值，legacy charts.js:436-443 语义）
 *     setWindow([x0,x1]) 程序化视窗（GPU"近期跟随"用；x 单位与 points 一致）
 *     resetView()        程序化复位到全量（按钮用，不触发回调）
 *     onUserZoom=fn([x0,x1])  用户滚轮缩放/拖拽平移后回调（状态机留在调用方）
 *     onViewReset=fn()        双击复位后回调（wrapper 已先复位到全量）
 *     refreshTheme()     主题刷新（charts/theme.js 注册表统一调用）
 *     resize()/dispose()
 *
 * 交互对齐 legacy：dataZoom inside（滚轮缩放、左键拖拽平移、触屏捏合/拖动）
 * + 双击复位（dispatchAction dataZoom start:0/end:100，不用 restore——它会回滚数据）。
 */
import { util } from "../util.js";
import { PALETTE, chartTheme, themeColors, emaSmooth, registerChart } from "./theme.js";

export function createLineChart(el, opts) {
  opts = opts || {};
  var xIsTime = !!opts.xIsTime;
  var xLabel = opts.xLabel || "step";
  var legendOn = opts.legend !== false;

  /* echarts 需要块级容器：旧的 <canvas> 占位原位换成同尺寸 div（index.html 不动） */
  var host = el;
  if (host && host.tagName === "CANVAS") {
    var div = document.createElement("div");
    div.style.width = "100%";
    div.style.height = "100%";
    div.style.cursor = "grab";
    host.parentElement.insertBefore(div, host);
    host.parentElement.removeChild(host);
    host = div;
  }

  var chartInst = null;          // echarts 实例（容器 0 尺寸时推迟初始化）
  var rawSeries = [];            // setSeries 传入序列（_pts 已归一化/排序）
  var smoothing = 0;
  var logY = false;
  var pendingWindow = null;      // init 前调 setWindow 的暂存（数据坐标）
  var progZoom = false;          // 程序化 dataZoom 守卫（不触发 onUserZoom）

  /* ------------------------------------------------------------ 数据归一化 */
  function normPoints(points) {
    var out = [];
    (points || []).forEach(function (p) {
      var x, y;
      if (Array.isArray(p)) { x = p[0]; y = p[1]; }
      else if (p) { x = p.x; y = p.y; }
      if (util.isNum(x) && util.isNum(y)) out.push([x, y]);
    });
    out.sort(function (a, b) { return a[0] - b[0]; });
    return out;
  }

  /* 展示点集 + 对齐的 tooltip 原始值（log 模式过滤非正值，同 legacy） */
  function dispOf(s) {
    var raw = s._pts;
    var disp, tt;
    if (s.scatter) {
      disp = raw; tt = raw;
    } else if (s.emaAlpha > 0) {
      disp = emaSmooth(raw, s.emaAlpha); tt = disp;
    } else {
      disp = smoothing > 0 ? emaSmooth(raw, smoothing) : raw;
      tt = raw; /* 平滑开启时 tooltip 仍给原始值 */
    }
    if (logY) {
      var d2 = [], t2 = [];
      for (var i = 0; i < disp.length; i++) {
        if (disp[i][1] > 0) { d2.push(disp[i]); t2.push(tt[i]); }
      }
      disp = d2; tt = t2;
    }
    return { disp: disp, tt: tt };
  }

  /* 数据坐标 → 轴坐标（时间轴秒→毫秒） */
  function toAxisX(x) { return xIsTime ? x * 1000 : x; }
  function fromAxisX(x) { return xIsTime ? x / 1000 : x; }

  /* ------------------------------------------------------------ option 构建 */
  function deepAssign(dst, src) {
    Object.keys(src).forEach(function (k) {
      var v = src[k];
      if (v && typeof v === "object" && !Array.isArray(v) &&
          dst[k] && typeof dst[k] === "object" && !Array.isArray(dst[k])) {
        deepAssign(dst[k], v);
      } else {
        dst[k] = v;
      }
    });
    return dst;
  }

  function fmtTooltip(params) {
    if (!params || !params.length) return "";
    var ax = Number(params[0].axisValue);
    var title = xIsTime ? util.fmtClock(ax / 1000) : xLabel + " " + Math.round(ax);
    var rows = [];
    params.forEach(function (p) {
      /* 原始值嵌在数据点里（dataZoom filterMode:'filter' 下 dataIndex 是过滤后
       * 视图的索引，用并行数组按 dataIndex 反查会错位，故随点携带） */
      var v = p.data && util.isNum(p.data.raw) ? p.data.raw
        : (p.data && p.data.value && util.isNum(p.data.value[1]) ? p.data.value[1] : null);
      if (!util.isNum(v)) return;
      var color = typeof p.color === "string" ? p.color
        : (rawSeries[p.seriesIndex] && rawSeries[p.seriesIndex].color) || "";
      /* 行内样式复刻 .chart-tooltip 的 tt-row/tt-chip/tt-val 观感（不加 className：
       * echarts 初始化时会先建一个无定位的空 tooltip div，带上 chart-tooltip 类
       * 会被 CSS 描边成可见空胶囊，还把容器撑出纵向滚动条） */
      rows.push(
        '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
        '<span style="width:8px;height:8px;border-radius:2px;flex:none;background:' + color + '"></span>' +
        "<span>" + util.esc(p.seriesName) + "</span>" +
        '<span style="margin-left:auto;padding-left:12px">' + util.esc(util.fmtVal(v)) + "</span></div>"
      );
    });
    if (!rows.length) return "";
    return '<div style="color:var(--muted);margin-bottom:4px">' + util.esc(title) + "</div>" + rows.join("");
  }

  function buildOption() {
    var colors = themeColors();
    var series = [];
    var legendNames = [];
    var anyPts = false;
    rawSeries.forEach(function (s) {
      var d = dispOf(s);
      if (d.disp.length) anyPts = true;
      /* 每个点携带原始值（tooltip 用）：EMA/平滑/log 过滤只改 value，不改 raw */
      var data = d.disp.map(function (p, j) {
        return { value: [toAxisX(p[0]), p[1]], raw: d.tt[j][1] };
      });
      if (s.legend !== false) legendNames.push(s.name);
      if (s.scatter) {
        series.push({
          name: s.name, type: "scatter", data: data,
          symbolSize: 3.5,
          itemStyle: { color: s.color, opacity: 0.45 },
        });
      } else {
        series.push({
          name: s.name, type: "line", data: data,
          showSymbol: false,
          lineStyle: { color: s.color, width: 1.6 },
          itemStyle: { color: s.color },
        });
      }
    });
    var opt = deepAssign({}, chartTheme());
    deepAssign(opt, {
      animation: false,
      grid: {
        left: 6, right: 14, bottom: 2, containLabel: true,
        top: legendOn && legendNames.length ? 26 : 10,
      },
      /* scroll 分页：序列多（GPU 16 卡）时不遮挡绘图区（相对 legacy 手绘图例的改进） */
      legend: legendOn
        ? { show: legendNames.length > 0, type: "scroll", data: legendNames, top: 0, left: 2, icon: "rect", itemWidth: 10, itemHeight: 10 }
        : { show: false },
      xAxis: {
        type: xIsTime ? "time" : "value",
        scale: true,
        splitLine: { show: true },
        axisLabel: xIsTime ? {} : {
          formatter: function (v) {
            return Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : util.fmtAxis(v);
          },
        },
      },
      yAxis: {
        type: logY ? "log" : "value",
        name: opts.yLabel || "",
        scale: !logY,
        axisLabel: { formatter: function (v) { return util.fmtAxis(v); } },
      },
      tooltip: {
        trigger: "axis",
        confine: true,
        padding: [7, 10],
        extraCssText: "box-shadow:0 6px 20px var(--shadow);white-space:nowrap;min-width:130px;",
        formatter: fmtTooltip,
      },
      dataZoom: [{
        type: "inside",
        xAxisIndex: 0,
        filterMode: "filter",
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: false,
      }],
      series: series,
      graphic: [{
        id: "nodata",
        type: "text",
        left: "center",
        top: "middle",
        silent: true,
        invisible: rawSeries.length > 0 && anyPts,
        style: { text: "暂无数据", fontSize: 12, fontFamily: "sans-serif", fill: colors.faint },
      }],
    });
    return opt;
  }

  /* ------------------------------------------------------------ 生命周期 */
  function apply() {
    if (!chartInst) { tryInit(); return; }
    /* series/xAxis/yAxis 全量替换（序列数变少时清掉残留），其余组件合并
     * （dataZoom 不带 start/end，用户缩放状态得以保留） */
    chartInst.setOption(buildOption(), { replaceMerge: ["series", "xAxis", "yAxis"] });
  }

  function onDataZoom() {
    if (progZoom || !chartInst || !api.onUserZoom) return;
    var opt = chartInst.getOption();
    var dz = opt && opt.dataZoom && opt.dataZoom[0];
    if (!dz) return;
    var x0 = dz.startValue, x1 = dz.endValue;
    if (!util.isNum(x0) || !util.isNum(x1)) {
      /* 只有百分比时按数据 extent 折算（不读 echarts 内部 scale） */
      var ext = xExtent();
      var start = util.isNum(dz.start) ? dz.start : 0;
      var end = util.isNum(dz.end) ? dz.end : 100;
      x0 = ext[0] + ((ext[1] - ext[0]) * start) / 100;
      x1 = ext[0] + ((ext[1] - ext[0]) * end) / 100;
      api.onUserZoom([x0, x1]);
      return;
    }
    api.onUserZoom([fromAxisX(x0), fromAxisX(x1)]);
  }

  function xExtent() {
    var min = Infinity, max = -Infinity;
    rawSeries.forEach(function (s) {
      var pts = s._pts;
      if (!pts.length) return;
      if (pts[0][0] < min) min = pts[0][0];
      if (pts[pts.length - 1][0] > max) max = pts[pts.length - 1][0];
    });
    if (min === Infinity) { min = 0; max = 1; }
    if (max - min < 1e-12) max = min + 1;
    return [min, max];
  }

  function dispatchWindow(winAxis) {
    progZoom = true;
    chartInst.dispatchAction(winAxis
      ? { type: "dataZoom", startValue: winAxis[0], endValue: winAxis[1] }
      : { type: "dataZoom", start: 0, end: 100 });
    /* datazoom 事件同步派发，但保险起见下一拍再解除守卫 */
    setTimeout(function () { progZoom = false; }, 0);
  }

  function tryInit() {
    if (chartInst || typeof window.echarts === "undefined") return;
    /* 隐藏 tab 容器 0 尺寸：等 ResizeObserver 见到非 0 尺寸再初始化 */
    if (host.clientWidth <= 0 || host.clientHeight <= 0) return;
    chartInst = window.echarts.init(host);
    chartInst.on("datazoom", onDataZoom);
    chartInst.setOption(buildOption(), { replaceMerge: ["series", "xAxis", "yAxis"] });
    if (pendingWindow) dispatchWindow([toAxisX(pendingWindow[0]), toAxisX(pendingWindow[1])]);
  }

  /* 双击复位 = 回全量视窗（不用 dispatchAction restore——它会回滚 option 数据） */
  function onDblClick() {
    api.resetView();
    if (api.onViewReset) api.onViewReset();
  }
  host.addEventListener("dblclick", onDblClick);

  var ro = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(function () {
      if (host.clientWidth <= 0 || host.clientHeight <= 0) return;
      if (!chartInst) tryInit();
      else chartInst.resize();
    });
    ro.observe(host);
  }

  /* ------------------------------------------------------------ 对外 API */
  var api = {
    onUserZoom: null,
    onViewReset: null,

    setSeries: function (list) {
      rawSeries = (list || []).map(function (s, i) {
        return {
          name: s.name != null ? String(s.name) : "series" + i,
          color: s.color || PALETTE[i % PALETTE.length],
          scatter: !!s.scatter,
          emaAlpha: s.emaAlpha || 0,
          legend: s.legend,
          _pts: normPoints(s.points),
        };
      });
      apply();
    },

    setSmoothing: function (alpha) {
      smoothing = Math.min(0.99, Math.max(0, alpha || 0));
      apply();
    },

    setLogY: function (on) {
      logY = !!on;
      apply();
    },

    /* 程序化视窗（GPU"近期跟随"）；null 等价 resetView */
    setWindow: function (win) {
      pendingWindow = win ? [win[0], win[1]] : null;
      if (!chartInst) { tryInit(); return; }
      dispatchWindow(pendingWindow ? [toAxisX(pendingWindow[0]), toAxisX(pendingWindow[1])] : null);
    },

    resetView: function () {
      pendingWindow = null;
      if (!chartInst) { tryInit(); return; }
      dispatchWindow(null);
    },

    refreshTheme: function () {
      apply();
    },

    resize: function () {
      if (!chartInst) { tryInit(); return; }
      chartInst.resize();
    },

    dispose: function () {
      if (ro) ro.disconnect();
      host.removeEventListener("dblclick", onDblClick);
      unregister();
      if (chartInst) { chartInst.dispose(); chartInst = null; }
    },
  };

  var unregister = registerChart(api);
  tryInit();
  return api;
}
