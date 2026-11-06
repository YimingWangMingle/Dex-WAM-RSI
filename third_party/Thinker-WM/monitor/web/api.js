/* Motus Monitor 前端 —— API 层与工具函数。
 * 挂在 window.MonApp 上，无外部依赖，同源 fetch。
 *   MonApp.api.get(path) / post(path, body) / getOrNull(path)
 *   MonApp.api.streamReader(runId, stream, onLines, opts)  增量 JSONL 读取
 *   MonApp.api.events(onMessage)                            SSE 订阅
 *   MonApp.util.*                                           格式化工具
 */
window.MonApp = window.MonApp || {};

(function () {
  "use strict";

  var api = {};

  function request(path, options) {
    return fetch(path, options).then(function (resp) {
      if (resp.status === 401 && !window.location.pathname.startsWith("/login")) {
        /* public 模式会话过期/未登录：跳登录页，登录后回到当前页面 */
        window.location.href = "/login?next=" +
          encodeURIComponent(window.location.pathname + window.location.search);
        return new Promise(function () { /* 跳转中，挂起后续处理 */ });
      }
      if (!resp.ok) {
        var err = new Error("HTTP " + resp.status + " " + path);
        err.status = resp.status;
        throw err;
      }
      return resp.json();
    });
  }

  api.get = function (path) { return request(path); };

  api.getOrNull = function (path) {
    return api.get(path).catch(function (err) {
      if (err && err.status === 404) return null;
      throw err;
    });
  };

  api.post = function (path, body) {
    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  };

  /* 登录状态：{auth_enabled, user}；本机模式 auth_enabled=false */
  api.whoami = function () {
    return api.getOrNull("/api/auth/whoami");
  };

  /* 全量历史：服务端扫描整个流（metrics 做 resume 去重 + 抽稀；system 按到达序抽稀），
   * opts: {stream: "metrics"|"system", node: "<节点名>"} */
  api.loadHistory = function (runId, maxPoints, opts) {
    opts = opts || {};
    var qs = "?max_points=" + (maxPoints || 3000);
    if (opts.stream) qs += "&stream=" + encodeURIComponent(opts.stream);
    if (opts.node) qs += "&node=" + encodeURIComponent(opts.node);
    return api.get("/api/runs/" + encodeURIComponent(runId) + "/history" + qs);
  };

  /* ---------------------------------------------------------- SSE 订阅
   * onMessage 收到 {type:"update"|"annotation", ...}；返回 {close()}。
   * EventSource 自带断线重连；onState 回调连接状态（live/retry）。
   */
  api.events = function (onMessage, onState) {
    var source = null;
    var closed = false;
    var retryTimer = null;

    function connect() {
      if (closed) return;
      source = new EventSource("/api/events");
      source.onopen = function () { if (onState) onState("live"); };
      source.onmessage = function (ev) {
        var msg = null;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        if (msg && msg.type) onMessage(msg);
      };
      source.onerror = function () {
        if (onState) onState("retry");
        // 浏览器原生会自动重连；若长时间未 open，5s 后手动重建兜底。
        if (source && source.readyState === EventSource.CLOSED) {
          source.close();
          if (!closed && !retryTimer) {
            retryTimer = setTimeout(function () { retryTimer = null; connect(); }, 3000);
          }
        }
      };
    }

    connect();
    return {
      close: function () {
        closed = true;
        if (retryTimer) clearTimeout(retryTimer);
        if (source) source.close();
      },
    };
  };

  /* ---------------------------------------------------------- 增量流读取
   * 管理 offset：首次 tail=1&tail_bytes=N 取最近一段，之后 offset 增量。
   * onLines(records, reader)：每次 poll 解析出的 JSON 记录数组。
   * system 流支持 setNode() 切换节点（自动重置 offset）。
   */
  api.streamReader = function (runId, stream, onLines, opts) {
    opts = opts || {};
    var reader = {
      runId: runId,
      stream: stream,
      offset: null,
      node: opts.node || null,
      nodes: [],
      tailBytes: opts.tailBytes || 131072,
      error: null,
      _partial: "",
      _busy: false,

      reset: function () {
        reader.offset = null;
        reader._partial = "";
      },

      setNode: function (node) {
        if (node !== reader.node) {
          reader.node = node;
          reader.reset();
        }
      },

      poll: function () {
        if (reader._busy) return Promise.resolve(false);
        reader._busy = true;
        var url = "/api/runs/" + encodeURIComponent(reader.runId) +
          "/metrics?stream=" + encodeURIComponent(reader.stream);
        if (reader.offset) {
          url += "&offset=" + encodeURIComponent(reader.offset);
        } else {
          url += "&tail=1&tail_bytes=" + (reader.tailBytes | 0);
        }
        if (reader.stream === "system" && reader.node) {
          url += "&node=" + encodeURIComponent(reader.node);
        }
        return api.get(url).then(function (resp) {
          reader.error = null;
          if (resp.offset) reader.offset = resp.offset;
          if (resp.nodes) reader.nodes = resp.nodes;
          if (resp.node) reader.node = resp.node;
          var text = reader._partial + (resp.data || "");
          var records = [];
          if (text) {
            var parts = text.split("\n");
            // 最后一段若不是整行（理论上不会发生，写入端按行落盘），留待下次拼接。
            if (text.charAt(text.length - 1) !== "\n") {
              reader._partial = parts.pop();
            } else {
              reader._partial = "";
            }
            for (var i = 0; i < parts.length; i++) {
              var line = parts[i];
              if (!line) continue;
              line = line.trim();
              if (!line) continue;
              try {
                records.push(JSON.parse(line));
              } catch (e) { /* 跳过坏行 */ }
            }
          }
          if (records.length && typeof onLines === "function") {
            try { onLines(records, reader); } catch (e) { console.error(e); }
          }
          return records.length > 0;
        }).catch(function (err) {
          reader.error = err;
          return false;
        }).then(function (changed) {
          reader._busy = false;
          return changed;
        });
      },
    };
    return reader;
  };

  MonApp.api = api;

  /* ============================================================ 工具函数 */
  var util = {};

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

  MonApp.util = util;
})();
