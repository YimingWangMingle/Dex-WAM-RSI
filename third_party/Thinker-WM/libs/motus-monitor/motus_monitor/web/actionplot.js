/* Motus Monitor 前端 —— eval action 逐维曲线（canvas 自绘）。
 *
 * 训练侧 20260820 起不再落盘 actions.png：曲线数据在样本 meta.json
 * （eef_display {labels, gt, pred}；raw 回退 gt_actions/predicted_actions +
 * action_dim_labels），本模块按需在浏览器端渲染。
 *
 * y 范围（用户拍板 20260820）：不同 dim 的 y 可不同，但同一 dim 跨 eps/数据
 * 集必须一致——按维度语义 band 固定（pos/rot6/gripper/other），默认值见
 * DEFAULT_BAND_RANGES（保证大部分 dim 不被裁剪）；超出范围的部分裁剪不显示。
 * eval tab 工具栏「y 轴范围」编辑器按 band 覆盖（localStorage 持久化），
 * 修改后全部已挂载图表重绘。
 *
 * 懒加载：样本卡片多，IntersectionObserver 滚动到附近才 fetch meta.json。
 *
 * 交互查看器（20260821 起）：缩略图点击进入 eval.js 弹层内的大图
 * （openInteractive）——联动十字线 + tooltip/状态栏读数、共享 x 视窗滚轮
 * 缩放/拖拽平移/双击复位、GT/Pred 显隐、y band/自适应、维度过滤、单击
 * cell 进单维聚焦（复用 MonApp.LineChart，←/→ 切换维度、Esc 返回）、
 * CSV/PNG 导出；y 编辑器改动经 redrawAll 同步重绘查看器。
 */
window.MonApp = window.MonApp || {};

(function () {
  "use strict";

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
  var CELL_H = 96;          // css px / 维
  var PAD_L = 36, PAD_R = 4, PAD_T = 14, PAD_B = 6;  // cell 内边距
  var GT_COLOR = "#4f8cff";
  var PRED_COLOR = "#ff9f43";

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

  /* ------------------------------------------------------------ 绘制 */
  function themeColor(name, fallback) {
    try {
      var v = getComputedStyle(document.documentElement).getPropertyValue(name);
      return (v && v.trim()) || fallback;
    } catch (e) {
      return fallback;
    }
  }

  function fmtTick(v) {
    var util = MonApp.util;
    if (util && typeof util.fmtAxis === "function") return util.fmtAxis(v);
    return Math.abs(v) >= 100 ? String(Math.round(v)) : String(Number(v.toFixed(2)));
  }

  function drawSeries(ctx, mat, dim, T, xPix, yPix, color, dashed, lineW) {
    if (!mat) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW || 1.2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.beginPath();
    var started = false;
    for (var t = 0; t < T && t < mat.length; t++) {
      var row = mat[t];
      if (!Array.isArray(row) || dim >= row.length) break;
      var v = Number(row[dim]);
      if (!isFinite(v)) { started = false; continue; }  // null/NaN 断线
      var x = xPix(t), y = yPix(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /* x 像素映射：xView=[t0,t1] 时按视窗映射，否则全量 [0,T-1]（缩略图旧行为） */
  function makeXPix(left, w, T, xView) {
    if (xView && xView[1] > xView[0]) {
      return function (t) { return left + ((t - xView[0]) / (xView[1] - xView[0])) * w; };
    }
    return T > 1
      ? function (t) { return left + (t / (T - 1)) * w; }
      : function () { return left + w / 2; };
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

  /* 单 cell 绘制。opts（缩略图全部缺省 = 旧行为）：
   *   xView:[t0,t1]|null 共享 x 视窗；hoverT:int|null 联动十字线 + 值标记；
   *   showGt/showPred 序列显隐；
   *   padL/padR/padT/padB 边距、font 字号、lineW 线宽、textColor 文字色
   *   （交互大图用大字号/粗线/深色提高清晰度，缩略图保持默认小参数）。 */
  function drawCell(ctx, cx, cy, cellW, cellH, data, dim, yRange, opts) {
    opts = opts || {};
    var showGt = opts.showGt !== false, showPred = opts.showPred !== false;
    var padL = opts.padL != null ? opts.padL : PAD_L;
    var padR = opts.padR != null ? opts.padR : PAD_R;
    var padT = opts.padT != null ? opts.padT : PAD_T;
    var padB = opts.padB != null ? opts.padB : PAD_B;
    var font = opts.font || "9px ui-monospace, Menlo, Consolas, monospace";
    var lineW = opts.lineW || 1.2;
    var textCol = opts.textColor || themeColor("--muted", "#8b96ad");
    var left = cx + padL, right = cx + cellW - padR;
    var top = cy + padT, bottom = cy + cellH - padB;
    var w = right - left, h = bottom - top;
    if (w <= 10 || h <= 10) return;
    var ymin = yRange[0], ymax = yRange[1];

    /* cell 边框 + 标题 + y 上下限 */
    ctx.strokeStyle = themeColor("--border-soft", "rgba(128,128,128,0.25)");
    ctx.lineWidth = 1;
    ctx.strokeRect(left + 0.5, top + 0.5, w, h);
    ctx.font = font;
    ctx.textBaseline = "middle";
    ctx.fillStyle = textCol;
    ctx.textAlign = "left";
    var label = data.labels[dim] || ("dim " + dim);
    var maxLabelW = cellW - 8;
    if (ctx.measureText(label).width > maxLabelW) {
      while (label.length > 4 && ctx.measureText(label + "…").width > maxLabelW) {
        label = label.slice(0, -1);
      }
      label += "…";
    }
    ctx.fillText(label, cx + 4, cy + padT / 2 + 1);
    ctx.textAlign = "right";
    ctx.fillText(fmtTick(ymax), left - 3, top + 6);
    ctx.fillText(fmtTick(ymin), left - 3, bottom - 2);

    /* 曲线（clip 到 cell 绘图区：超出 y 范围/x 视窗的部分不显示） */
    var ref = data.gt || data.pred;
    var T = ref ? ref.length : 0;
    if (T <= 0) return;
    var xPix = makeXPix(left, w, T, opts.xView || null);
    var yPix = function (v) {
      return top + (1 - (v - ymin) / (ymax - ymin)) * h;
    };
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, w, h);
    ctx.clip();
    if (showGt) drawSeries(ctx, data.gt, dim, T, xPix, yPix, GT_COLOR, false, lineW);
    if (showPred) drawSeries(ctx, data.pred, dim, T, xPix, yPix, PRED_COLOR, true, lineW);
    /* hover 联动十字线 + 当前值标记点 */
    if (opts.hoverT != null && opts.hoverT >= 0 && opts.hoverT < T) {
      var hx = xPix(opts.hoverT);
      ctx.strokeStyle = themeColor("--chart-cross", "rgba(128,128,128,0.55)");
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, top);
      ctx.lineTo(hx, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      var dotR = Math.max(2.2, lineW + 1);
      var marker = function (mat, color) {
        if (!mat || opts.hoverT >= mat.length) return;
        var row = mat[opts.hoverT];
        if (!Array.isArray(row) || dim >= row.length) return;
        var v = Number(row[dim]);
        if (!isFinite(v)) return;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(hx, yPix(v), dotR, 0, Math.PI * 2);
        ctx.fill();
      };
      if (showGt) marker(data.gt, GT_COLOR);
      if (showPred) marker(data.pred, PRED_COLOR);
    }
    ctx.restore();
  }

  /* 网格绘制（缩略图与交互查看器共用）。opts：
   *   cols/cellH 布局；dims=要画的维度索引数组（null=全部，截 MAX_DIMS）；
   *   yMode "band"|"auto"；其余同 drawCell opts。 */
  function renderGrid(ctx, cssW, data, opts) {
    opts = opts || {};
    var cols = opts.cols || COLS;
    var cellH = opts.cellH || CELL_H;
    var dims = opts.dims;
    if (!dims) {
      dims = [];
      for (var d = 0; d < Math.min(data.labels.length, MAX_DIMS); d++) dims.push(d);
    }
    var ranges = getRanges();
    var cellW = cssW / cols;
    for (var i = 0; i < dims.length; i++) {
      var dim = dims[i];
      var band = bandOf(data.labels[dim]);
      var yr = opts.yMode === "auto"
        ? autoRange(data, dim, opts.xView || null, ranges[band])
        : ranges[band];
      drawCell(ctx, (i % cols) * cellW, Math.floor(i / cols) * cellH,
        cellW, cellH, data, dim, yr, opts);
    }
    return dims.length;
  }

  /* HiDPI 尺寸设置 + 2d 上下文（已 setTransform，绘制坐标 = css 像素） */
  function sizeCanvas(canvas, cssW, cssH) {
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function draw(state) {
    var container = state.container;
    var data = state.data;
    if (!container.isConnected) return;
    container.innerHTML = "";
    var canvas = document.createElement("canvas");
    container.appendChild(canvas);
    var cssW = container.clientWidth || 480;
    var dims = Math.min(data.labels.length, MAX_DIMS);
    var rows = Math.ceil(dims / COLS);
    var cssH = rows * CELL_H;
    var ctx = sizeCanvas(canvas, cssW, cssH);
    renderGrid(ctx, cssW, data, {});
    state.canvas = canvas;
  }

  /* ------------------------------------------------------------ 挂载管理 */
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
        /* 容器尺寸变化（侧栏折叠/窗口缩放）时重绘 */
        if (typeof ResizeObserver !== "undefined" && !state.ro) {
          state.ro = new ResizeObserver(function () {
            if (state.data) draw(state);
          });
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
      data: null, io: null, ro: null,
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
   * 网格模式：hover 全 cell 联动十字线 + tooltip/状态栏读数、滚轮缩放共享
   *   x 视窗、左键拖拽平移、双击复位、维度过滤、GT/Pred 显隐、y band/自适应、
   *   单击 cell 进单维聚焦、CSV/PNG 导出；
   * 单维聚焦：复用 MonApp.LineChart（自带 tooltip/缩放/平移/双击复位），
   *   ←/→ 切换维度、Esc 返回网格。
   * 返回 handle {destroy, handleEsc, redraw}；destroy 由弹层关闭时调用。
   */
  var activeViewer = null;

  /* 查看器网格的视觉参数（20260821 用户反馈"放大后模糊"：放大只放大了格子，
   * 文字/线宽没跟上——这里用比缩略图更大的字号/线宽与更深的文字色） */
  var V_PAD_L = 46, V_PAD_R = 8, V_PAD_T = 20, V_PAD_B = 8;
  var V_CELL_H = 128;
  var V_FONT = "11px ui-monospace, Menlo, Consolas, monospace";
  var V_LINE_W = 1.8;

  function openInteractive(container, metaUrl, opts) {
    opts = opts || {};
    var util = MonApp.util;
    var esc = util.esc, el2 = util.el;
    var STATUS_HINT =
      "hover 查看数值 · 滚轮缩放 · 拖拽平移 · 双击复位 · 单击 cell 进入单维（←/→ 切换，Esc 返回）";

    var v = {
      destroyed: false,
      data: null,
      xView: null,          // [t0,t1] | null（网格模式共享 x 视窗）
      hoverT: null,
      hoverDim: null,
      showGt: true,
      showPred: true,
      yMode: "band",        // "band" | "auto"
      filter: "",
      single: null,         // 单维聚焦的 dim 索引 | null
      cols: COLS,
      cellH: V_CELL_H,
      fileBase: String(opts.title || "actions").replace(/[^\w.-]+/g, "_") || "actions",
      canvasWrap: null, gridCanvas: null, tooltip: null, statusEl: null,
      lineChart: null, ro: null, drag: null,
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

    function saveBlob(blob, filename) {
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 30000);
    }

    /* ------------------------------------------------------------ 网格绘制 */
    function drawViewerGrid() {
      if (v.destroyed || !v.data || v.single !== null) return;
      var canvas = v.gridCanvas;
      if (!canvas || !canvas.isConnected) return;
      var cssW = v.canvasWrap.clientWidth || 800;
      var dims = filteredDims();
      var cssH = Math.max(1, Math.ceil(dims.length / v.cols)) * v.cellH;
      var ctx = sizeCanvas(canvas, cssW, cssH);
      ctx.clearRect(0, 0, cssW, cssH);
      var drawn = renderGrid(ctx, cssW, v.data, {
        cols: v.cols, cellH: v.cellH, dims: dims,
        xView: v.xView, hoverT: v.hoverT,
        showGt: v.showGt, showPred: v.showPred, yMode: v.yMode,
        padL: V_PAD_L, padR: V_PAD_R, padT: V_PAD_T, padB: V_PAD_B,
        font: V_FONT, lineW: V_LINE_W,
        textColor: themeColor("--text", "#22304a"),
      });
      if (!drawn) {
        ctx.fillStyle = themeColor("--faint", "#5c6880");
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("无匹配维度（过滤：" + v.filter + "）", cssW / 2, cssH / 2);
        ctx.textAlign = "left";
      }
    }

    /* ------------------------------------------------------------ hover 读数 */
    function hoverReadout() {
      /* → {html, text} | null */
      if (v.hoverDim == null || v.hoverT == null || !v.data) return null;
      var d = v.hoverDim, t = v.hoverT;
      var g = valAt(v.data.gt, t, d), p = valAt(v.data.pred, t, d);
      var label = String(v.data.labels[d]);
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
        html: '<div class="tt-title">t = ' + t + " · " + esc(label) + "</div>" + rows.join(""),
        text: parts.join(" · "),
      };
    }

    function ttRow(color, name, val) {
      return '<div class="tt-row"><span class="tt-chip" style="background:' + color + '"></span>' +
        "<span>" + esc(name) + '</span><span class="tt-val">' + esc(val) + "</span></div>";
    }

    function updateHoverUI(mx, my) {
      var ro = hoverReadout();
      if (v.statusEl) v.statusEl.textContent = ro ? ro.text : STATUS_HINT;
      var tt = v.tooltip;
      if (!tt) return;
      if (!ro || mx == null) { tt.classList.add("hidden"); return; }
      tt.innerHTML = ro.html;
      tt.classList.remove("hidden");
      var wrapW = v.canvasWrap.clientWidth, wrapH = v.canvasWrap.clientHeight;
      var tw = tt.offsetWidth, th = tt.offsetHeight;
      var x = mx + 14, y = my + 12;
      if (x + tw > wrapW - 4) x = mx - tw - 10;
      if (y + th > wrapH - 4) y = my - th - 8;
      tt.style.left = Math.max(2, x) + "px";
      tt.style.top = Math.max(2, y) + "px";
    }

    /* 光标位置 → (hoverDim, hoverT)；无效区域返回 {dim:null, t:null} */
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
      var rect = v.gridCanvas.getBoundingClientRect();
      var mx = e.clientX - rect.left, my = e.clientY - rect.top;
      var pick = pickCell(mx, my);
      v.hoverDim = pick.dim;
      v.hoverT = pick.t;
      updateHoverUI(mx, my);
      drawViewerGrid();
    }

    function onGridMouseLeave() {
      v.hoverDim = null;
      v.hoverT = null;
      updateHoverUI(null, null);
      drawViewerGrid();
    }

    function onGridWheel(e) {
      e.preventDefault();
      var T = timeLen();
      if (T <= 1) return;
      var ext = [0, T - 1];
      var view = v.xView ? v.xView.slice() : ext.slice();
      var rect = v.gridCanvas.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var cssW = v.canvasWrap.clientWidth || 800;
      var cellW = cssW / v.cols;
      var col = Math.min(v.cols - 1, Math.max(0, Math.floor(mx / cellW)));
      var left = col * cellW + V_PAD_L, w = cellW - V_PAD_L - V_PAD_R;
      var frac = Math.min(1, Math.max(0, (mx - left) / Math.max(1, w)));
      var cx = view[0] + frac * (view[1] - view[0]);
      var factor = e.deltaY > 0 ? 1.25 : 0.8;
      var nx0 = cx - (cx - view[0]) * factor;
      var nx1 = cx + (view[1] - cx) * factor;
      if (nx1 - nx0 < 4) { var mid = (nx0 + nx1) / 2; nx0 = mid - 2; nx1 = mid + 2; }
      if (nx0 <= ext[0] && nx1 >= ext[1]) v.xView = null;
      else v.xView = [Math.max(ext[0], nx0), Math.min(ext[1], nx1)];
      drawViewerGrid();
    }

    function onGridMouseDown(e) {
      if (e.button !== 0) return;
      e.preventDefault();
      var T = timeLen();
      if (T <= 1) return;
      var rect = v.gridCanvas.getBoundingClientRect();
      v.drag = {
        startX: e.clientX - rect.left,
        view: v.xView ? v.xView.slice() : [0, T - 1],
        moved: false,
      };
    }

    function onDragMove(e) {
      var d = v.drag;
      if (!d || !v.gridCanvas) return;
      var T = timeLen();
      var ext = [0, Math.max(0, T - 1)];
      var rect = v.gridCanvas.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var cssW = v.canvasWrap.clientWidth || 800;
      var cellW = cssW / v.cols;
      var pw = Math.max(1, cellW - V_PAD_L - V_PAD_R);
      var dx = ((mx - d.startX) / pw) * (d.view[1] - d.view[0]);
      if (Math.abs(mx - d.startX) > 3) d.moved = true;
      var nx0 = d.view[0] - dx, nx1 = d.view[1] - dx;
      if (nx0 < ext[0]) { nx1 += ext[0] - nx0; nx0 = ext[0]; }
      if (nx1 > ext[1]) { nx0 -= nx1 - ext[1]; nx1 = ext[1]; }
      nx0 = Math.max(ext[0], nx0);
      nx1 = Math.min(ext[1], nx1);
      if (nx1 - nx0 >= ext[1] - ext[0]) v.xView = null;
      else v.xView = [nx0, nx1];
      v.hoverDim = null;
      v.hoverT = null;
      updateHoverUI(null, null);
      drawViewerGrid();
    }

    function onDragEnd() {
      if (!v.drag) return;
      /* 拖拽后抑制紧随的 click（否则误进单维）；setTimeout 让 click 先派发 */
      var wasMoved = v.drag.moved;
      v.drag = null;
      if (wasMoved) {
        setTimeout(function () { v.suppressClick = false; }, 0);
        v.suppressClick = true;
      }
    }

    function onGridDblClick() {
      v.xView = null;
      drawViewerGrid();
    }

    function onGridClick() {
      if (v.suppressClick) { v.suppressClick = false; return; }
      if (v.single !== null || v.hoverDim == null) return;
      enterSingle(v.hoverDim);
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

    /* 跨屏拖动/系统缩放改变 devicePixelRatio 时按新 dpr 重建位图
     * （否则位图按旧 dpr 渲染被拉伸——真实的"变模糊"来源之一）。
     * 单维聚焦的 LineChart 由 BaseChart 自带的 dpr 监听处理（charts.js）。 */
    function onDprChange() {
      armDprWatch();
      if (v.destroyed || v.single !== null) return;
      drawViewerGrid();
    }
    function armDprWatch() {
      if (typeof window.matchMedia !== "function") return;
      if (v.dprMq) v.dprMq.removeEventListener("change", onDprChange);
      v.dprMq = window.matchMedia("(resolution: " + (window.devicePixelRatio || 1) + "dppx)");
      v.dprMq.addEventListener("change", onDprChange);
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
      if (v.lineChart) { v.lineChart.destroy(); v.lineChart = null; }
      v.single = dim;
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
      var cv = document.createElement("canvas");
      chartWrap.appendChild(cv);
      v.canvasWrap.appendChild(chartWrap);
      var series = [];
      if (v.showGt) series.push({ key: "gt", label: "GT", color: GT_COLOR, points: pointsFor(v.data.gt, dim) });
      if (v.showPred) series.push({ key: "pred", label: "Pred", color: PRED_COLOR, points: pointsFor(v.data.pred, dim) });
      v.lineChart = new MonApp.LineChart(cv, { series: series, xLabel: "frame" });
      if (v.tooltip) v.tooltip.classList.add("hidden");
      if (v.statusEl) {
        v.statusEl.textContent =
          "单维聚焦：" + label + " · hover 读数 · 滚轮缩放 · 拖拽平移 · 双击复位";
      }
    }

    function backToGrid() {
      if (v.lineChart) { v.lineChart.destroy(); v.lineChart = null; }
      v.single = null;
      v.canvasWrap.innerHTML = "";
      v.canvasWrap.appendChild(v.gridCanvas);
      v.canvasWrap.appendChild(v.tooltip);
      if (v.statusEl) v.statusEl.textContent = STATUS_HINT;
      drawViewerGrid();
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
      saveBlob(new Blob([lines.join("\n")], { type: "text/csv" }), v.fileBase + ".csv");
    }

    function exportPng() {
      var canvas = v.single !== null && v.lineChart ? v.lineChart.canvas : v.gridCanvas;
      if (!canvas || typeof canvas.toBlob !== "function") return;
      canvas.toBlob(function (blob) {
        if (blob) saveBlob(blob, v.fileBase + ".png");
      }, "image/png");
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
      else drawViewerGrid();
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
        drawViewerGrid();
      });
      bar.appendChild(yBtn);

      var filter = el2("input", "ap-filter", {
        attrs: { type: "search", placeholder: "过滤维度…（如 gripper）" },
      });
      filter.addEventListener("input", function () {
        v.filter = filter.value.trim().toLowerCase();
        if (v.single === null) drawViewerGrid();
      });
      bar.appendChild(filter);

      bar.appendChild(el2("span", "ap-spacer"));

      var reset = el2("button", "btn", { text: "重置视图", attrs: { type: "button" } });
      reset.addEventListener("click", function () {
        if (v.single !== null && v.lineChart) {
          v.lineChart.view = null;
          v.lineChart.viewTouched = false;
          v.lineChart.redraw();
        } else {
          v.xView = null;
          drawViewerGrid();
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
      v.gridCanvas = document.createElement("canvas");
      v.tooltip = el2("div", "chart-tooltip hidden");
      v.canvasWrap.appendChild(v.gridCanvas);
      v.canvasWrap.appendChild(v.tooltip);
      container.appendChild(v.canvasWrap);
      v.statusEl = el2("div", "ap-status", { text: STATUS_HINT });
      container.appendChild(v.statusEl);

      v.gridCanvas.addEventListener("mousemove", onGridMouseMove);
      v.gridCanvas.addEventListener("mouseleave", onGridMouseLeave);
      v.gridCanvas.addEventListener("wheel", onGridWheel, { passive: false });
      v.gridCanvas.addEventListener("mousedown", onGridMouseDown);
      v.gridCanvas.addEventListener("dblclick", onGridDblClick);
      v.gridCanvas.addEventListener("click", onGridClick);
      window.addEventListener("mousemove", onDragMove);
      window.addEventListener("mouseup", onDragEnd);
      document.addEventListener("keydown", onKey);
      armDprWatch();
      if (typeof ResizeObserver !== "undefined") {
        v.ro = new ResizeObserver(function () {
          if (v.single === null) drawViewerGrid();
        });
        v.ro.observe(v.canvasWrap);
      }
      drawViewerGrid();
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
        window.removeEventListener("mousemove", onDragMove);
        window.removeEventListener("mouseup", onDragEnd);
        document.removeEventListener("keydown", onKey);
        if (v.dprMq) v.dprMq.removeEventListener("change", onDprChange);
        if (v.ro) v.ro.disconnect();
        if (v.lineChart) { v.lineChart.destroy(); v.lineChart = null; }
        if (activeViewer === handle) activeViewer = null;
      },
      /* Esc：单维聚焦时返回网格并消费事件；否则不消费（交给弹层关闭） */
      handleEsc: function () {
        if (v.destroyed) return false;
        if (v.single !== null) { backToGrid(); return true; }
        return false;
      },
      redraw: function () {
        if (!v.destroyed && v.single === null) drawViewerGrid();
      },
    };
    activeViewer = handle;
    return handle;
  }

  /* y 范围变更后重绘全部已挂载图表（不重建 DOM）；交互查看器网格模式同步重绘 */
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
  MonApp.ActionPlot = {
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
})();
