/* Motus Monitor 前端 —— ECharts 堆叠柱状图统一封装（阶段 2B 新基建）。
 * 覆盖旧 charts/legacy.js StackedBars 的全部场景（当前唯一用户：timing tab
 * 的逐步阶段耗时堆叠）：
 *
 *   createStackedBars(el, {xLabel="step", maxBars=120})
 *   → wrapper 对象：
 *     setData(rows, stages)  rows: [{step, ...}]（按 step 升序）；
 *                            stages: [{key, label, color, get?}]，颜色由调用方
 *                            给（timing 用 metrics-schema 的 colorForKey）；
 *                            get(row) 访问器可选（自定义检查点的值在
 *                            stage_seconds 子字典里，legacy _buckets 语义）
 *     refreshTheme()         主题刷新（charts/theme.js 注册表统一调用）
 *     resize()/dispose()
 *
 * 分桶逻辑逐字移植 legacy StackedBars._buckets（charts/legacy.js:617-649）：
 * 最多 maxBars 根柱，连续记录取均值，x 取桶内最后一个 step。
 * x 轴选型 category：旧版柱按索引等距、刻度按值线性定位——均匀 step 下两者
 * 重合，category 轴在 step 不均/缺号时柱位依然正确，刻度按柱位均匀抽稀
 * （约 6 个，对齐旧版 niceStep(range, 6) 的观感）。无缩放/平移（旧版没有，
 * 不新增）。
 */
import { util } from "../util.js";
import { PALETTE, chartTheme, themeColor, themeColors, registerChart } from "./theme.js";

export function createStackedBars(el, opts) {
  opts = opts || {};
  var xLabel = opts.xLabel || "step";
  var maxBars = opts.maxBars || 120;

  /* echarts 需要块级容器：旧的 <canvas> 占位原位换成同尺寸 div（index.html 不动）。
   * 同 line.js 模式，但不设 grab 光标——本图无缩放/平移。 */
  var host = el;
  if (host && host.tagName === "CANVAS") {
    var div = document.createElement("div");
    div.style.width = "100%";
    div.style.height = "100%";
    host.parentElement.insertBefore(div, host);
    host.parentElement.removeChild(host);
    host = div;
  }

  var chartInst = null;          // echarts 实例（容器 0 尺寸时推迟初始化）
  var rawRows = [];              // setData 传入行
  var stages = [];               // setData 传入阶段定义（归一化后）
  var buckets = [];              // 最近一次分桶结果（tooltip 按 dataIndex 反查）

  /* ------------------------------------------------------------ 分桶
   * 逐字移植 legacy StackedBars._buckets：最多 maxBars 根柱，连续记录
   * 取均值，x 取桶内最后一个 step（xKey 恒为 "step"，新 API 不再暴露）。 */
  function computeBuckets() {
    var rows = rawRows;
    var n = rows.length;
    if (!n) return [];
    var bucketCount = Math.min(maxBars, n);
    var per = n / bucketCount;
    var out = [];
    for (var b = 0; b < bucketCount; b++) {
      var start = Math.floor(b * per);
      var end = Math.max(start + 1, Math.floor((b + 1) * per));
      var sums = {};
      var counts = 0;
      var x = rows[Math.min(end - 1, n - 1)].step;
      for (var i = start; i < end && i < n; i++) {
        counts++;
        for (var k = 0; k < stages.length; k++) {
          var st = stages[k];
          /* 支持 get 访问器：自定义检查点的值在 stage_seconds 子字典里 */
          var v = st.get ? st.get(rows[i]) : rows[i][st.key];
          if (util.isNum(v)) sums[st.key] = (sums[st.key] || 0) + v;
        }
      }
      var vals = {};
      var total = 0;
      for (k = 0; k < stages.length; k++) {
        var kk = stages[k].key;
        vals[kk] = counts && util.isNum(sums[kk]) ? sums[kk] / counts : 0;
        total += vals[kk];
      }
      out.push({ x: x, values: vals, total: total, count: counts });
    }
    return out;
  }

  /* ------------------------------------------------------------ option 构建 */
  /* 深合并小工具（同 line.js；各 wrapper 自留一份，不抽公共模块） */
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

  /* tooltip：legacy _draw 735-750 语义——标题 step N（多步桶标注"N 步均值"）、
   * 分阶段各行（含 0 值阶段）+ 末行合计；数值 util.fmtSeconds（µs/ms/s）。
   * 无 dataZoom，dataIndex 即分桶索引，直接反查 buckets。 */
  function fmtTooltip(params) {
    if (!params || !params.length) return "";
    var bk = buckets[params[0].dataIndex];
    if (!bk) return "";
    var title = xLabel + " " + Math.round(bk.x) +
      (bk.count > 1 ? "（" + bk.count + " 步均值）" : "");
    var rows = [];
    stages.forEach(function (st) {
      rows.push(
        '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
        '<span style="width:8px;height:8px;border-radius:2px;flex:none;background:' + st.color + '"></span>' +
        "<span>" + util.esc(st.label) + "</span>" +
        '<span style="margin-left:auto;padding-left:12px">' +
        util.esc(util.fmtSeconds(bk.values[st.key] || 0)) + "</span></div>"
      );
    });
    /* 合计行 chip 用 --text-strong（legacy textStrongColor()） */
    rows.push(
      '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
      '<span style="width:8px;height:8px;border-radius:2px;flex:none;background:' +
      themeColor("--text-strong", "#ffffff") + '"></span>' +
      "<span>合计</span>" +
      '<span style="margin-left:auto;padding-left:12px">' +
      util.esc(util.fmtSeconds(bk.total)) + "</span></div>"
    );
    return '<div style="color:var(--muted);margin-bottom:4px">' + util.esc(title) + "</div>" + rows.join("");
  }

  function buildOption() {
    var colors = themeColors();
    buckets = computeBuckets();
    var cats = buckets.map(function (b) { return String(Math.round(b.x)); });
    var series = stages.map(function (st) {
      return {
        name: st.label,
        type: "bar",
        stack: "total",
        data: buckets.map(function (b) { return b.values[st.key] || 0; }),
        /* 柱宽对齐 legacy：slot-1.5px、上限 40px；非悬停 alpha 0.85 */
        barMaxWidth: 40,
        barCategoryGap: 1.5,
        itemStyle: { color: st.color, opacity: 0.85 },
        emphasis: { itemStyle: { opacity: 1 } },
      };
    });
    var opt = deepAssign({}, chartTheme());
    deepAssign(opt, {
      animation: false,
      grid: {
        left: 6, right: 14, bottom: 2, containLabel: true,
        top: stages.length ? 26 : 10,
      },
      /* scroll 分页：阶段多（逐 block 检查点）时不遮挡绘图区 */
      legend: {
        show: stages.length > 0,
        type: "scroll",
        data: stages.map(function (s) { return s.label; }),
        top: 0, left: 2, icon: "rect", itemWidth: 10, itemHeight: 10,
      },
      xAxis: {
        type: "category",
        data: cats,
        splitLine: { show: false },
        axisTick: { alignWithLabel: true },
        /* 刻度按柱位均匀抽稀到约 6 个（legacy niceStep(xMax-xMin, 6) 观感） */
        axisLabel: { interval: Math.max(0, Math.ceil(cats.length / 6) - 1) },
      },
      yAxis: {
        type: "value",
        /* min 默认含 0（stack 柱必须），顶余量由 echarts 取整给出
         * （legacy 为 max*1.08 后 niceStep，观感等价）；
         * 刻度标签逐字 legacy 689：util.fmtSeconds（µs/ms/s） */
        axisLabel: { formatter: function (v) { return util.fmtSeconds(v); } },
      },
      tooltip: {
        trigger: "axis",
        confine: true,
        padding: [7, 10],
        extraCssText: "box-shadow:0 6px 20px var(--shadow);white-space:nowrap;min-width:130px;",
        /* 悬停列阴影（对齐 legacy 悬停柱的 cross 描边观感） */
        axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,144,168,0.15)" } },
        formatter: fmtTooltip,
      },
      series: series,
      graphic: [{
        id: "nodata",
        type: "text",
        left: "center",
        top: "middle",
        silent: true,
        invisible: buckets.length > 0,
        style: { text: "暂无数据", fontSize: 12, fontFamily: "sans-serif", fill: colors.faint },
      }],
    });
    return opt;
  }

  /* ------------------------------------------------------------ 生命周期 */
  function apply() {
    if (!chartInst) { tryInit(); return; }
    /* series/xAxis 全量替换（阶段组切换时清掉残留序列与旧类目），其余合并 */
    chartInst.setOption(buildOption(), { replaceMerge: ["series", "xAxis", "yAxis"] });
  }

  function tryInit() {
    if (chartInst || typeof window.echarts === "undefined") return;
    /* 隐藏 tab 容器 0 尺寸：等 ResizeObserver 见到非 0 尺寸再初始化 */
    if (host.clientWidth <= 0 || host.clientHeight <= 0) return;
    chartInst = window.echarts.init(host);
    chartInst.setOption(buildOption(), { replaceMerge: ["series", "xAxis", "yAxis"] });
  }

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
    setData: function (rows, stageDefs) {
      rawRows = rows || [];
      stages = (stageDefs || []).map(function (s, i) {
        return {
          key: s.key != null ? String(s.key) : "stage" + i,
          label: s.label != null ? String(s.label) : String(s.key),
          color: s.color || PALETTE[i % PALETTE.length],
          get: s.get || null,
        };
      });
      apply();
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
      unregister();
      if (chartInst) { chartInst.dispose(); chartInst = null; }
    },
  };

  var unregister = registerChart(api);
  tryInit();
  return api;
}
