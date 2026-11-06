/* Motus Monitor 前端 —— Canvas 2D 自绘图表（无任何图表库）。
 *   MonApp.PALETTE      12 色调色板
 *   MonApp.LineChart    多序列折线/散点图（EMA、log-y、hover、滚轮缩放、拖拽平移、双击复位）
 *   MonApp.StackedBars  阶段耗时堆叠柱状图
 *   MonApp.hbar()       DOM 水平百分比条（系统负载用）
 *   MonApp.shareBar()   DOM 100% 堆叠条 + 图例
 */
window.MonApp = window.MonApp || {};

(function () {
  "use strict";

  var util = MonApp.util;

  MonApp.PALETTE = [
    "#4f8cff", "#ff6b6b", "#4ecdc4", "#e0a92c", "#a06bff", "#6bd66b",
    "#ff9f43", "#f368e0", "#48dbfb", "#1dd1a1", "#ff9ff3", "#7d9db5",
  ];

  var AXIS_FONT = "10px ui-monospace, Menlo, Consolas, monospace";

  /* 主题感知取色：canvas 无法使用 CSS 变量，绘制时从 :root 读取当前主题色 */
  function themeColor(name, fallback) {
    try {
      var v = getComputedStyle(document.documentElement).getPropertyValue(name);
      return (v && v.trim()) || fallback;
    } catch (e) {
      return fallback;
    }
  }
  function gridColor() { return themeColor("--chart-grid", "rgba(255,255,255,0.06)"); }
  function axisColor() { return themeColor("--muted", "#8b96ad"); }
  function faintColor() { return themeColor("--faint", "#5c6880"); }
  function crossColor() { return themeColor("--chart-cross", "rgba(255,255,255,0.35)"); }
  function textStrongColor() { return themeColor("--text-strong", "#ffffff"); }

  /* ------------------------------------------------------------ 小工具 */
  function niceStep(range, target) {
    if (!(range > 0)) return 1;
    var raw = range / target;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
    return step * mag;
  }

  function nearestIndex(pts, x) {
    var lo = 0, hi = pts.length - 1;
    if (hi < 0) return -1;
    if (x <= pts[0][0]) return 0;
    if (x >= pts[hi][0]) return hi;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (pts[mid][0] < x) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && Math.abs(pts[lo - 1][0] - x) < Math.abs(pts[lo][0] - x)) return lo - 1;
    return lo;
  }

  function emaSmooth(pts, alpha) {
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

  function fmtXTick(v, isTime) {
    if (!isTime) {
      return Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : util.fmtAxis(v);
    }
    return util.fmtClock(v);
  }

  /* 时间轴 tick 间隔候选（秒） */
  var TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 43200, 86400];
  function timeStep(range, target) {
    var raw = range / target;
    for (var i = 0; i < TIME_STEPS.length; i++) if (TIME_STEPS[i] >= raw) return TIME_STEPS[i];
    return niceStep(range, target);
  }

  /* ------------------------------------------------------------ 基类骨架
   * 负责：HiDPI 尺寸、ResizeObserver、tooltip div、hover 状态、重绘调度。
   */
  function BaseChart(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.wrap = canvas.parentElement;
    this.cssW = 0;
    this.cssH = 0;
    this.hover = null; // {mx, my} css 像素
    this._rafPending = false;
    this._destroyed = false;

    this.tooltip = util.el("div", "chart-tooltip hidden");
    this.wrap.appendChild(this.tooltip);

    var self = this;
    this._onResize = function () { self._resize(); };
    this._onMouseMove = function (ev) {
      var rect = self.canvas.getBoundingClientRect();
      self.hover = { mx: ev.clientX - rect.left, my: ev.clientY - rect.top };
      self.redraw();
    };
    this._onMouseLeave = function () {
      self.hover = null;
      self.tooltip.classList.add("hidden");
      self.redraw();
    };
    canvas.addEventListener("mousemove", this._onMouseMove);
    canvas.addEventListener("mouseleave", this._onMouseLeave);

    if (typeof ResizeObserver !== "undefined") {
      this._ro = new ResizeObserver(this._onResize);
      this._ro.observe(this.wrap);
    }
    this._resize();
  }

  BaseChart.prototype._resize = function () {
    if (this._destroyed) return;
    var w = this.wrap.clientWidth, h = this.wrap.clientHeight;
    if (w <= 0 || h <= 0) return;
    var dpr = window.devicePixelRatio || 1;
    this.cssW = w;
    this.cssH = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.redraw();
  };

  BaseChart.prototype.redraw = function () {
    if (this._rafPending || this._destroyed) return;
    this._rafPending = true;
    var self = this;
    requestAnimationFrame(function () {
      self._rafPending = false;
      if (self._destroyed) return;
      if (self.cssW <= 0 || self.cssH <= 0) self._resize();
      self._draw();
    });
  };

  BaseChart.prototype.destroy = function () {
    this._destroyed = true;
    if (this._ro) this._ro.disconnect();
    this.canvas.removeEventListener("mousemove", this._onMouseMove);
    this.canvas.removeEventListener("mouseleave", this._onMouseLeave);
    if (this.tooltip.parentElement) this.tooltip.parentElement.removeChild(this.tooltip);
  };

  BaseChart.prototype._plotArea = function () {
    return { left: 56, right: 12, top: 10, bottom: 24 };
  };

  BaseChart.prototype._drawLegend = function (items) {
    var ctx = this.ctx;
    var x = this._plotArea().left + 4;
    var y = 12;
    ctx.font = AXIS_FONT;
    ctx.textBaseline = "middle";
    for (var i = 0; i < items.length; i++) {
      var label = items[i].label;
      var w = ctx.measureText(label).width + 18;
      if (x + w > this.cssW - 10) { x = this._plotArea().left + 4; y += 14; }
      ctx.fillStyle = items[i].color;
      ctx.fillRect(x, y - 4, 8, 8);
      ctx.fillStyle = axisColor();
      ctx.textAlign = "left";
      ctx.fillText(label, x + 12, y);
      x += w + 8;
    }
  };

  BaseChart.prototype._showTooltip = function (html, mx, my) {
    var tt = this.tooltip;
    tt.innerHTML = html;
    tt.classList.remove("hidden");
    var tw = tt.offsetWidth, th = tt.offsetHeight;
    var x = mx + 14, y = my + 12;
    if (x + tw > this.cssW - 4) x = mx - tw - 10;
    if (y + th > this.cssH - 4) y = my - th - 8;
    if (x < 2) x = 2;
    if (y < 2) y = 2;
    tt.style.left = x + "px";
    tt.style.top = y + "px";
  };

  /* ============================================================ LineChart
   * opts: { series: [{key,label,color,value(row),scatter,emaAlpha}],
   *         xKey: "step", xIsTime: false }
   */
  function LineChart(canvas, opts) {
    opts = opts || {};
    this.series = opts.series || [];
    this.xKey = opts.xKey || "step";
    this.xIsTime = !!opts.xIsTime;
    this.rows = [];
    this.smoothing = 0;
    this.logY = false;
    this.view = null; // [x0, x1] 或 null（全量）
    this.viewTouched = false; // 用户缩放/平移过（自动跟随类视窗据此停跟随）
    this._drag = null; // {startX, view}
    this._rawCache = {};  // series.key -> [[x,y],...]
    this._disp = {};      // series.key -> {line, scatter}
    BaseChart.call(this, canvas);
    canvas.classList.add("line-chart");

    var self = this;
    this._onWheel = function (ev) {
      ev.preventDefault();
      self._zoom(ev);
    };
    this._onDblClick = function () {
      self.view = null;
      self.viewTouched = false;
      if (self.onViewReset) self.onViewReset();
      self.redraw();
    };
    /* 左键拖拽 = 左右平移（move/up 挂 window，拖出 canvas 不中断） */
    this._onMouseDown = function (ev) {
      if (ev.button !== 0) return;
      ev.preventDefault();
      var ext = self._xExtent();
      var view = self.view ? self.view.slice() : ext.slice();
      self._drag = { startX: ev.offsetX, view: view };
      canvas.classList.add("dragging");
    };
    this._onDragMove = function (ev) {
      var d = self._drag;
      if (!d) return;
      var area = self._plotArea();
      var pw = Math.max(1, self.cssW - area.left - area.right);
      var span = d.view[1] - d.view[0];
      var rect = self.canvas.getBoundingClientRect();
      var mx = ev.clientX - rect.left;
      var dx = ((mx - d.startX) / pw) * span;
      if (!dx) return;
      var ext = self._xExtent();
      var nx0 = d.view[0] - dx, nx1 = d.view[1] - dx;
      if (nx0 < ext[0]) { nx1 += ext[0] - nx0; nx0 = ext[0]; }
      if (nx1 > ext[1]) { nx0 -= nx1 - ext[1]; nx1 = ext[1]; }
      nx0 = Math.max(ext[0], nx0);
      nx1 = Math.min(ext[1], nx1);
      if (nx1 - nx0 >= ext[1] - ext[0]) self.view = null;
      else self.view = [nx0, nx1];
      self.viewTouched = true;
      self.hover = null;
      self.tooltip.classList.add("hidden");
      if (self.onViewChange) self.onViewChange();
      self.redraw();
    };
    this._onDragEnd = function () {
      if (!self._drag) return;
      self._drag = null;
      canvas.classList.remove("dragging");
      self.redraw();
    };
    canvas.addEventListener("wheel", this._onWheel, { passive: false });
    canvas.addEventListener("dblclick", this._onDblClick);
    canvas.addEventListener("mousedown", this._onMouseDown);
    window.addEventListener("mousemove", this._onDragMove);
    window.addEventListener("mouseup", this._onDragEnd);
  }
  LineChart.prototype = Object.create(BaseChart.prototype);
  LineChart.prototype.constructor = LineChart;

  LineChart.prototype.setSeries = function (series) {
    this.series = series || [];
    this._rawCache = {};
    this.redraw();
  };

  LineChart.prototype.setData = function (rows) {
    this.rows = rows || [];
    this._rawCache = {};
    this.redraw();
  };

  LineChart.prototype.setSmoothing = function (alpha) {
    this.smoothing = Math.min(0.99, Math.max(0, alpha || 0));
    this.redraw();
  };

  LineChart.prototype.setLogY = function (on) {
    this.logY = !!on;
    this.redraw();
  };

  LineChart.prototype.destroy = function () {
    this.canvas.removeEventListener("wheel", this._onWheel);
    this.canvas.removeEventListener("dblclick", this._onDblClick);
    this.canvas.removeEventListener("mousedown", this._onMouseDown);
    window.removeEventListener("mousemove", this._onDragMove);
    window.removeEventListener("mouseup", this._onDragEnd);
    BaseChart.prototype.destroy.call(this);
  };

  LineChart.prototype._rawPts = function (s) {
    /* 独立点集（多 run 对比等场景）：系列自带 [[x,y],...]，不走统一 rows */
    if (s.points) return s.points;
    var cached = this._rawCache[s.key];
    if (cached) return cached;
    var pts = [];
    var rows = this.rows;
    var xKey = this.xKey;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var x = r[xKey];
      var y = s.value ? s.value(r) : r[s.key];
      if (!util.isNum(x) || !util.isNum(y)) continue;
      pts.push([x, y]);
    }
    this._rawCache[s.key] = pts;
    return pts;
  };

  /* 展示用点集：折线（应用 EMA）+ 散点（原始值）。log 模式过滤非正值。 */
  LineChart.prototype._dispPts = function (s) {
    var raw = this._rawPts(s);
    var line = null, scatter = null;
    if (s.scatter) {
      scatter = raw;
      if (s.emaAlpha) line = emaSmooth(raw, s.emaAlpha);
    } else {
      line = this.smoothing > 0 ? emaSmooth(raw, this.smoothing) : raw;
    }
    if (this.logY) {
      var pos = function (p) { return p[1] > 0; };
      if (line) line = line.filter(pos);
      if (scatter) scatter = scatter.filter(pos);
    }
    return { line: line, scatter: scatter };
  };

  LineChart.prototype._xExtent = function () {
    var min = Infinity, max = -Infinity;
    for (var i = 0; i < this.series.length; i++) {
      var pts = this._rawPts(this.series[i]);
      if (!pts.length) continue;
      if (pts[0][0] < min) min = pts[0][0];
      if (pts[pts.length - 1][0] > max) max = pts[pts.length - 1][0];
    }
    if (min === Infinity) { min = 0; max = 1; }
    if (max - min < 1e-12) max = min + 1;
    return [min, max];
  };

  LineChart.prototype._zoom = function (ev) {
    var ext = this._xExtent();
    var view = this.view ? this.view.slice() : ext.slice();
    var area = this._plotArea();
    var frac = (ev.offsetX - area.left) / Math.max(1, this.cssW - area.left - area.right);
    frac = Math.min(1, Math.max(0, frac));
    var cx = view[0] + frac * (view[1] - view[0]);
    var factor = ev.deltaY > 0 ? 1.25 : 0.8;
    var nx0 = cx - (cx - view[0]) * factor;
    var nx1 = cx + (view[1] - cx) * factor;
    var minSpan = this.xIsTime ? 5 : 5;
    if (nx1 - nx0 < minSpan) {
      var mid = (nx0 + nx1) / 2;
      nx0 = mid - minSpan / 2;
      nx1 = mid + minSpan / 2;
    }
    if (nx0 <= ext[0] && nx1 >= ext[1]) this.view = null;
    else {
      nx0 = Math.max(ext[0], nx0);
      nx1 = Math.min(ext[1], nx1);
      this.view = [nx0, nx1];
    }
    this.viewTouched = true;
    if (this.onViewChange) this.onViewChange();
    this.redraw();
  };

  LineChart.prototype._draw = function () {
    var ctx = this.ctx;
    var W = this.cssW, H = this.cssH;
    ctx.clearRect(0, 0, W, H);
    var area = this._plotArea();
    var pw = W - area.left - area.right;
    var ph = H - area.top - area.bottom;
    if (pw <= 10 || ph <= 10) return;

    var ext = this._xExtent();
    var xr = this.view || ext;
    var x0 = xr[0], x1 = xr[1];

    /* y 范围：可见 x 区间内的展示点 */
    var ymin = Infinity, ymax = -Infinity;
    var disps = [];
    for (var i = 0; i < this.series.length; i++) {
      var d = this._dispPts(this.series[i]);
      disps.push(d);
      var sets = [d.line, d.scatter];
      for (var k = 0; k < 2; k++) {
        var pts = sets[k];
        if (!pts) continue;
        for (var j = 0; j < pts.length; j++) {
          if (pts[j][0] < x0 || pts[j][0] > x1) continue;
          var y = pts[j][1];
          if (y < ymin) ymin = y;
          if (y > ymax) ymax = y;
        }
      }
    }
    if (ymin === Infinity) { ymin = 0; ymax = 1; }
    if (ymax - ymin < 1e-12) { ymax += 1; ymin -= 1; }

    var ty, iy; // y 变换（log 可选）
    if (this.logY) {
      var lmin = Math.log10(Math.max(ymin, 1e-300));
      var lmax = Math.log10(Math.max(ymax, 1e-300));
      var lpad = (lmax - lmin) * 0.06 || 0.5;
      lmin -= lpad; lmax += lpad;
      ty = function (v) { return Math.log10(Math.max(v, 1e-300)); };
      iy = function (t) { return Math.pow(10, t); };
      ymin = lmin; ymax = lmax;
    } else {
      var pad = (ymax - ymin) * 0.08;
      if (ymin >= 0 && ymin - pad < 0) ymin = 0; else ymin -= pad;
      ymax += pad;
      ty = function (v) { return v; };
      iy = function (t) { return t; };
    }

    var xPix = function (x) { return area.left + ((x - x0) / (x1 - x0)) * pw; };
    var yPix = function (y) { return area.top + (1 - (ty(y) - ymin) / (ymax - ymin)) * ph; };

    /* 网格 + y 轴刻度 */
    ctx.font = AXIS_FONT;
    ctx.textBaseline = "middle";
    var yStep = niceStep(ymax - ymin, 5);
    var yt0 = Math.ceil(ymin / yStep) * yStep;
    ctx.textAlign = "right";
    for (var yt = yt0; yt <= ymax + 1e-12; yt += yStep) {
      var py = area.top + (1 - (yt - ymin) / (ymax - ymin)) * ph;
      ctx.strokeStyle = gridColor();
      ctx.beginPath();
      ctx.moveTo(area.left, py);
      ctx.lineTo(W - area.right, py);
      ctx.stroke();
      ctx.fillStyle = axisColor();
      ctx.fillText(util.fmtAxis(iy(yt)), area.left - 6, py);
    }

    /* x 轴刻度 */
    var xStep = this.xIsTime ? timeStep(x1 - x0, 6) : niceStep(x1 - x0, 6);
    var xt0 = Math.ceil(x0 / xStep) * xStep;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (var xt = xt0; xt <= x1 + 1e-12; xt += xStep) {
      var px = xPix(xt);
      ctx.strokeStyle = gridColor();
      ctx.beginPath();
      ctx.moveTo(px, area.top);
      ctx.lineTo(px, H - area.bottom);
      ctx.stroke();
      ctx.fillStyle = axisColor();
      ctx.fillText(fmtXTick(xt, this.xIsTime), px, H - area.bottom + 6);
    }
    ctx.textBaseline = "middle";

    /* 序列 */
    ctx.save();
    ctx.beginPath();
    ctx.rect(area.left, area.top - 4, pw, ph + 8);
    ctx.clip();
    for (i = 0; i < this.series.length; i++) {
      var s = this.series[i];
      var dd = disps[i];
      if (dd.line && dd.line.length) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        var started = false;
        for (j = 0; j < dd.line.length; j++) {
          var pxx = xPix(dd.line[j][0]);
          var pyy = yPix(dd.line[j][1]);
          if (!started) { ctx.moveTo(pxx, pyy); started = true; }
          else ctx.lineTo(pxx, pyy);
        }
        ctx.stroke();
      }
      if (dd.scatter && dd.scatter.length) {
        ctx.fillStyle = s.color;
        ctx.globalAlpha = 0.45;
        var maxDots = 3000;
        var stride = Math.max(1, Math.floor(dd.scatter.length / maxDots));
        for (j = 0; j < dd.scatter.length; j += stride) {
          if (dd.scatter[j][0] < x0 || dd.scatter[j][0] > x1) continue;
          ctx.fillRect(xPix(dd.scatter[j][0]) - 1.2, yPix(dd.scatter[j][1]) - 1.2, 2.4, 2.4);
        }
        ctx.globalAlpha = 1;
      }
    }

    /* hover 十字线 + tooltip（拖拽平移中抑制） */
    if (this.hover && !this._drag && this.hover.mx >= area.left && this.hover.mx <= W - area.right) {
      var hx = x0 + ((this.hover.mx - area.left) / pw) * (x1 - x0);
      var rowsHtml = [];
      var bestPx = null;
      for (i = 0; i < this.series.length; i++) {
        s = this.series[i];
        var raw = this._rawPts(s);
        var idx = nearestIndex(raw, hx);
        if (idx < 0) continue;
        var pt = raw[idx];
        if (pt[0] < x0 || pt[0] > x1) continue;
        if (bestPx === null || Math.abs(pt[0] - hx) < Math.abs(bestPx[0] - hx)) bestPx = pt;
        rowsHtml.push(
          '<div class="tt-row"><span class="tt-chip" style="background:' + s.color + '"></span>' +
          "<span>" + util.esc(s.label) + '</span><span class="tt-val">' +
          util.esc(util.fmtVal(pt[1])) + "</span></div>"
        );
      }
      if (bestPx) {
        var cxp = xPix(bestPx[0]);
        ctx.strokeStyle = crossColor();
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(cxp, area.top);
        ctx.lineTo(cxp, H - area.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        /* 各序列最近点高亮 */
        for (i = 0; i < this.series.length; i++) {
          s = this.series[i];
          idx = nearestIndex(this._rawPts(s), hx);
          if (idx < 0) continue;
          pt = this._rawPts(s)[idx];
          if (pt[0] < x0 || pt[0] > x1) continue;
          if (this.logY && pt[1] <= 0) continue;
          ctx.fillStyle = s.color;
          ctx.beginPath();
          ctx.arc(cxp, yPix(pt[1]), 3, 0, Math.PI * 2);
          ctx.fill();
        }
        var title = this.xIsTime ? util.fmtClock(bestPx[0]) : "step " + Math.round(bestPx[0]);
        this._showTooltip(
          '<div class="tt-title">' + util.esc(title) + "</div>" + rowsHtml.join(""),
          this.hover.mx, this.hover.my
        );
      } else {
        this.tooltip.classList.add("hidden");
      }
    } else {
      this.tooltip.classList.add("hidden");
    }
    ctx.restore();

    this._drawLegend(this.series.map(function (s) {
      return { label: s.label, color: s.color };
    }));

    var anySeriesPts = false;
    for (var pi = 0; pi < this.series.length; pi++) {
      if (this.series[pi].points && this.series[pi].points.length) { anySeriesPts = true; break; }
    }
    if (!this.series.length || (!this.rows.length && !anySeriesPts)) {
      ctx.fillStyle = faintColor();
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("暂无数据", W / 2, H / 2);
      ctx.textAlign = "left";
    }
  };

  MonApp.LineChart = LineChart;

  /* ============================================================ StackedBars
   * opts: { stages: [{key,label,color}], xKey: "step", maxBars: 120 }
   * rows: [{ <xKey>: N, values: {stageKey: seconds} }]
   */
  function StackedBars(canvas, opts) {
    opts = opts || {};
    this.stages = opts.stages || [];
    this.xKey = opts.xKey || "step";
    this.maxBars = opts.maxBars || 120;
    this.rows = [];
    BaseChart.call(this, canvas);
  }
  StackedBars.prototype = Object.create(BaseChart.prototype);
  StackedBars.prototype.constructor = StackedBars;

  StackedBars.prototype.setStages = function (stages) {
    this.stages = stages || [];
    this.redraw();
  };

  StackedBars.prototype.setData = function (rows) {
    this.rows = rows || [];
    this.redraw();
  };

  /* 分桶：最多 maxBars 根柱，连续记录取均值，x 取桶内最后一个 */
  StackedBars.prototype._buckets = function () {
    var rows = this.rows;
    var n = rows.length;
    if (!n) return [];
    var bucketCount = Math.min(this.maxBars, n);
    var per = n / bucketCount;
    var out = [];
    for (var b = 0; b < bucketCount; b++) {
      var start = Math.floor(b * per);
      var end = Math.max(start + 1, Math.floor((b + 1) * per));
      var sums = {};
      var counts = 0;
      var x = rows[Math.min(end - 1, n - 1)][this.xKey];
      for (var i = start; i < end && i < n; i++) {
        counts++;
        for (var k = 0; k < this.stages.length; k++) {
          var st = this.stages[k];
          /* 支持 get 访问器：自定义检查点的值在 stage_seconds 子字典里 */
          var v = st.get ? st.get(rows[i]) : rows[i][st.key];
          if (util.isNum(v)) sums[st.key] = (sums[st.key] || 0) + v;
        }
      }
      var vals = {};
      var total = 0;
      for (k = 0; k < this.stages.length; k++) {
        var kk = this.stages[k].key;
        vals[kk] = counts && util.isNum(sums[kk]) ? sums[kk] / counts : 0;
        total += vals[kk];
      }
      out.push({ x: x, values: vals, total: total, count: counts });
    }
    return out;
  };

  StackedBars.prototype._draw = function () {
    var ctx = this.ctx;
    var W = this.cssW, H = this.cssH;
    ctx.clearRect(0, 0, W, H);
    var area = this._plotArea();
    var pw = W - area.left - area.right;
    var ph = H - area.top - area.bottom;
    if (pw <= 10 || ph <= 10) return;

    var buckets = this._buckets();
    if (!buckets.length) {
      ctx.fillStyle = faintColor();
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("暂无数据", W / 2, H / 2);
      ctx.textAlign = "left";
      return;
    }

    var ymax = 0;
    var i, k;
    for (i = 0; i < buckets.length; i++) if (buckets[i].total > ymax) ymax = buckets[i].total;
    if (ymax <= 0) ymax = 1;
    ymax *= 1.08;

    /* 网格 + y 轴 */
    ctx.font = AXIS_FONT;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    var yStep = niceStep(ymax, 4);
    for (var yt = 0; yt <= ymax + 1e-12; yt += yStep) {
      var py = area.top + (1 - yt / ymax) * ph;
      ctx.strokeStyle = gridColor();
      ctx.beginPath();
      ctx.moveTo(area.left, py);
      ctx.lineTo(W - area.right, py);
      ctx.stroke();
      ctx.fillStyle = axisColor();
      ctx.fillText(util.fmtSeconds(yt), area.left - 6, py);
    }

    /* x 轴刻度（step） */
    var xMin = buckets[0].x;
    var xMax = buckets[buckets.length - 1].x;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    var xStep = niceStep(Math.max(1, xMax - xMin), 6);
    for (var xt = Math.ceil(xMin / xStep) * xStep; xt <= xMax + 1e-12; xt += xStep) {
      var px = area.left + ((xt - xMin) / Math.max(1e-9, xMax - xMin)) * pw;
      ctx.fillStyle = axisColor();
      ctx.fillText(String(Math.round(xt)), px, H - area.bottom + 6);
    }
    ctx.textBaseline = "middle";

    /* 柱子 */
    var slot = pw / buckets.length;
    var barW = Math.max(1, Math.min(slot - 1.5, 40));
    var hoverIdx = -1;
    if (this.hover && this.hover.mx >= area.left && this.hover.mx <= W - area.right) {
      hoverIdx = Math.floor((this.hover.mx - area.left) / slot);
      if (hoverIdx >= buckets.length) hoverIdx = buckets.length - 1;
    }
    for (i = 0; i < buckets.length; i++) {
      var bk = buckets[i];
      var x = area.left + i * slot + (slot - barW) / 2;
      var yBase = H - area.bottom;
      for (k = 0; k < this.stages.length; k++) {
        var st = this.stages[k];
        var v = bk.values[st.key] || 0;
        if (v <= 0) continue;
        var hgt = (v / ymax) * ph;
        yBase -= hgt;
        ctx.fillStyle = st.color;
        ctx.globalAlpha = i === hoverIdx ? 1 : 0.85;
        ctx.fillRect(x, yBase, barW, Math.max(hgt, 0.5));
      }
      ctx.globalAlpha = 1;
      if (i === hoverIdx) {
        ctx.strokeStyle = crossColor();
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 0.5, area.top, barW + 1, ph);
      }
    }

    if (hoverIdx >= 0) {
      var hb = buckets[hoverIdx];
      var html = '<div class="tt-title">step ' + Math.round(hb.x) +
        (hb.count > 1 ? "（" + hb.count + " 步均值）" : "") + "</div>";
      for (k = 0; k < this.stages.length; k++) {
        var stg = this.stages[k];
        html += '<div class="tt-row"><span class="tt-chip" style="background:' + stg.color + '"></span>' +
          "<span>" + util.esc(stg.label) + '</span><span class="tt-val">' +
          util.esc(util.fmtSeconds(hb.values[stg.key] || 0)) + "</span></div>";
      }
      html += '<div class="tt-row"><span class="tt-chip" style="background:' + textStrongColor() + '"></span><span>合计</span>' +
        '<span class="tt-val">' + util.esc(util.fmtSeconds(hb.total)) + "</span></div>";
      this._showTooltip(html, this.hover.mx, this.hover.my);
    } else {
      this.tooltip.classList.add("hidden");
    }

    this._drawLegend(this.stages.map(function (s) {
      return { label: s.label, color: s.color };
    }));
  };

  MonApp.StackedBars = StackedBars;

  /* ============================================================ DOM 条形件 */

  /* 水平百分比条（系统负载）。opts: {label, pct(0-100|null), text, colorize} */
  MonApp.hbar = function (opts) {
    var row = util.el("div", "hbar-row");
    row.appendChild(util.el("span", "hbar-label", { text: opts.label || "" }));
    var track = util.el("div", "hbar-track");
    var fill = util.el("div", "hbar-fill");
    var pct = opts.pct;
    if (util.isNum(pct)) {
      pct = Math.max(0, Math.min(100, pct));
      fill.style.width = pct + "%";
      if (opts.color) fill.style.background = opts.color;
      else if (pct >= 90) fill.classList.add("crit");
      else if (pct >= 70) fill.classList.add("warn");
      else fill.classList.add("ok");
    } else {
      fill.style.width = "0";
    }
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(util.el("span", "hbar-text", { text: opts.text != null ? opts.text : "暂无数据" }));
    return row;
  };

  /* 100% 堆叠条 + 图例。parts: [{label, value, color}]；返回元素。 */
  MonApp.shareBar = function (parts) {
    var wrapEl = util.el("div");
    var bar = util.el("div", "share-bar");
    var legend = util.el("div", "share-legend");
    var total = 0;
    var i;
    for (i = 0; i < parts.length; i++) total += Math.max(0, parts[i].value || 0);
    for (i = 0; i < parts.length; i++) {
      var p = parts[i];
      var frac = total > 0 ? Math.max(0, p.value || 0) / total : 0;
      var seg = util.el("div", "share-seg");
      seg.style.width = (frac * 100).toFixed(3) + "%";
      seg.style.background = p.color;
      seg.title = p.label + " " + (frac * 100).toFixed(1) + "%";
      bar.appendChild(seg);
      var lg = util.el("span", "lg");
      var chip = util.el("span", "lg-chip");
      chip.style.background = p.color;
      lg.appendChild(chip);
      lg.appendChild(util.el("span", null, {
        text: p.label + " " + (frac * 100).toFixed(1) + "%",
      }));
      legend.appendChild(lg);
    }
    wrapEl.appendChild(bar);
    wrapEl.appendChild(legend);
    return wrapEl;
  };
})();
