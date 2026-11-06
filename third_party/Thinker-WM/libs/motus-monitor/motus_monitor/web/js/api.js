/* Motus Monitor 前端 —— API 层（ES module，无外部依赖，同源 fetch）。
 *   api.get(path) / post(path, body) / getOrNull(path)
 *   api.streamReader(runId, stream, onLines, opts)  增量 JSONL 读取
 *   api.events(onMessage)                            SSE 订阅
 */

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

function get(path) { return request(path); }

function getOrNull(path) {
  return get(path).catch(function (err) {
    if (err && err.status === 404) return null;
    throw err;
  });
}

function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/* 登录状态：{auth_enabled, user}；本机模式 auth_enabled=false */
function whoami() {
  return getOrNull("/api/auth/whoami");
}

/* 全量历史：服务端扫描整个流（metrics 做 resume 去重 + 抽稀；system 按到达序抽稀），
 * opts: {stream: "metrics"|"system", node: "<节点名>"} */
function loadHistory(runId, maxPoints, opts) {
  opts = opts || {};
  var qs = "?max_points=" + (maxPoints || 3000);
  if (opts.stream) qs += "&stream=" + encodeURIComponent(opts.stream);
  if (opts.node) qs += "&node=" + encodeURIComponent(opts.node);
  return get("/api/runs/" + encodeURIComponent(runId) + "/history" + qs);
}

/* ---------------------------------------------------------- SSE 订阅
 * onMessage 收到 {type:"update"|"annotation", ...}；返回 {close()}。
 * EventSource 自带断线重连；onState 回调连接状态（live/retry）。
 */
function events(onMessage, onState) {
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
}

/* ---------------------------------------------------------- 增量流读取
 * 管理 offset：首次 tail=1&tail_bytes=N 取最近一段，之后 offset 增量。
 * onLines(records, reader)：每次 poll 解析出的 JSON 记录数组。
 * system 流支持 setNode() 切换节点（自动重置 offset）。
 */
function streamReader(runId, stream, onLines, opts) {
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
      return get(url).then(function (resp) {
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
}

export const api = {
  request: request,
  get: get,
  getOrNull: getOrNull,
  post: post,
  whoami: whoami,
  loadHistory: loadHistory,
  events: events,
  streamReader: streamReader,
};
