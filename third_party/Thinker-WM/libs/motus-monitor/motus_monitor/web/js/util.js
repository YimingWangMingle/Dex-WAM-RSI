/* Motus Monitor 前端 —— 格式化/小工具函数（ES module）。 */
export const util = {};

util.esc = function (s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
};

util.isNum = function (v) { return typeof v === "number" && isFinite(v); };

/* 坐标轴数值缩写：1.2k / 3.4M / 1e-5 */
util.fmtAxis = function (v) {
  if (!util.isNum(v)) return "—";
  var a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
  if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
  if (a === 0) return "0";
  if (a < 1e-4) return v.toExponential(0);
  if (a < 0.01) return v.toExponential(1);
  if (a < 1) return v.toFixed(3);
  if (a < 10) return v.toFixed(2);
  if (a < 100) return v.toFixed(1);
  return v.toFixed(0);
};

/* 卡片/tooltip 数值：保留 4 位有效数字 */
util.fmtVal = function (v) {
  if (v == null) return "—";
  if (!util.isNum(v)) return String(v);
  var a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(2);
  if (a >= 1000) return v.toFixed(1);
  if (a >= 100) return v.toFixed(2);
  if (a >= 10) return v.toFixed(3);
  return v.toFixed(4);
};

util.fmtPct = function (v, digits) {
  if (!util.isNum(v)) return "—";
  return v.toFixed(digits == null ? 1 : digits) + "%";
};

/* 时长：83.5 → "1分23秒"；0.34 → "340ms"；2h+ → "2时13分" */
util.fmtDuration = function (seconds) {
  if (!util.isNum(seconds)) return "—";
  if (seconds < 1) return Math.round(seconds * 1000) + "ms";
  if (seconds < 60) return seconds.toFixed(1) + "秒";
  var m = Math.floor(seconds / 60);
  var s = Math.round(seconds % 60);
  if (m < 60) return m + "分" + (s < 10 ? "0" : "") + s + "秒";
  var h = Math.floor(m / 60);
  m = m % 60;
  return h + "时" + (m < 10 ? "0" : "") + m + "分";
};

util.fmtSeconds = function (v) {
  if (!util.isNum(v)) return "—";
  if (v < 0.001) return (v * 1e6).toFixed(0) + "µs";
  if (v < 1) return (v * 1000).toFixed(1) + "ms";
  return v.toFixed(2) + "s";
};

util.fmtBytes = function (n) {
  if (!util.isNum(n)) return "—";
  var units = ["B", "KB", "MB", "GB", "TB"];
  var i = 0;
  var v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + units[i];
};

util.fmtClock = function (ts) {
  if (!util.isNum(ts)) return "—";
  var d = new Date(ts * 1000);
  function p(x) { return (x < 10 ? "0" : "") + x; }
  return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
};

util.fmtDateTime = function (iso) {
  if (!iso) return "—";
  var d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso);
  function p(x) { return (x < 10 ? "0" : "") + x; }
  return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
    " " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
};

/* 相对时间：ts(秒) → "刚刚"/"3分钟前"/"2小时前"/"5天前" */
util.fmtAgo = function (ts) {
  if (!util.isNum(ts)) return "—";
  var diff = Date.now() / 1000 - ts;
  if (diff < 0) diff = 0;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return Math.floor(diff / 60) + "分钟前";
  if (diff < 86400) return Math.floor(diff / 3600) + "小时前";
  return Math.floor(diff / 86400) + "天前";
};

/* 千分位整数：1234567 → "1,234,567" */
util.fmtInt = function (n) {
  if (!util.isNum(n)) return "—";
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
};

/* 元素创建小助手：el("div", "cls a b", {text, html, attrs}) */
util.el = function (tag, cls, opts) {
  var node = document.createElement(tag);
  if (cls) node.className = cls;
  opts = opts || {};
  if (opts.text != null) node.textContent = opts.text;
  if (opts.html != null) node.innerHTML = opts.html;
  if (opts.attrs) {
    for (var k in opts.attrs) node.setAttribute(k, opts.attrs[k]);
  }
  return node;
};

/* ============================================================ 新增通用 helper */

/* 防抖：最后一次调用 ms 毫秒后才执行（搜索输入等高频事件用） */
util.debounce = function (fn, ms) {
  var timer = null;
  return function () {
    var self = this, args = arguments;
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () { timer = null; fn.apply(self, args); }, ms);
  };
};

/* Blob 落地为下载文件：objectURL → a.download → 延时回收
 * （浏览器原生“另存为”对 no-store 媒体不可靠，统一走这条路径） */
util.saveBlob = function (blob, filename) {
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(url); }, 30000);
};

/* 读 :root 的 CSS 变量（canvas 无法使用 CSS 变量，绘制时按当前主题取色） */
util.cssVar = function (name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
};
