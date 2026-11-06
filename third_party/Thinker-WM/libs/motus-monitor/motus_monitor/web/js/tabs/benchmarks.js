/* Motus Monitor 前端 —— 基准 tab。
 * 从旧 web/app.js 逐字迁入：loadAnnotations/renderBenchMatrix（app.js:1340-1400）、
 * jumpToCkpt（app.js:1402-1405）、loadLlmHelp/formatLlmHelp（app.js:1407-1443）、
 * bench-refresh/llm-toggle 控件绑定（app.js:1852-1860）。
 * jumpToCkpt 的 switchTab("tree") 改为点击顶栏 tree tab 按钮（经 main.js 路由），
 * treeTab.selectCheckpoint(key) 由 tabs/tree.js 的默认导出提供（tree 填充时对齐）。
 */
import { api } from "../api.js";
import { util } from "../util.js";
import treeTab from "./tree.js";

/* ------------------------------------------------------------ 基准页 */
function loadAnnotations() {
  return api.get("/api/annotations").then(function (resp) {
    renderBenchMatrix((resp && resp.annotations) || []);
  }).catch(function (err) {
    console.error("load annotations failed", err);
  });
}

function renderBenchMatrix(annotations) {
  var box = document.getElementById("bench-matrix");
  box.innerHTML = "";
  if (!annotations.length) {
    box.appendChild(util.el("div", "no-data", { text: "暂无基准标注。可在训练树页选择 checkpoint 录入。" }));
    return;
  }
  var benchNames = {};
  annotations.forEach(function (a) {
    Object.keys(a.benchmarks || {}).forEach(function (b) { benchNames[b] = true; });
  });
  var cols = Object.keys(benchNames).sort();
  var rows = annotations.slice().sort(function (a, b) {
    return String(a.key).localeCompare(String(b.key));
  });

  var table = util.el("table", "matrix-table");
  var thead = "<thead><tr><th>Checkpoint</th>";
  cols.forEach(function (c) { thead += "<th>" + util.esc(c) + "</th>"; });
  thead += "</tr></thead>";
  table.innerHTML = thead;
  var tbody = util.el("tbody");
  rows.forEach(function (a) {
    var tr = util.el("tr");
    var keyCell = util.el("td", "key-cell", { text: a.key });
    keyCell.style.cursor = "pointer";
    keyCell.addEventListener("click", function () { jumpToCkpt(a.key); });
    tr.appendChild(keyCell);
    cols.forEach(function (c) {
      var metrics = a.benchmarks && a.benchmarks[c];
      if (!metrics) {
        tr.appendChild(util.el("td", "cell empty", { text: "—" }));
        return;
      }
      var keys = Object.keys(metrics);
      var text = keys.slice(0, 2).map(function (k) {
        var v = metrics[k];
        return k + "=" + (util.isNum(v) ? util.fmtVal(v) : String(v));
      }).join("  ");
      if (keys.length > 2) text += " …";
      var td = util.el("td", "cell", {
        text: text,
        attrs: { title: JSON.stringify(metrics, null, 2) },
      });
      td.addEventListener("click", function () { jumpToCkpt(a.key); });
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  box.appendChild(table);
}

function jumpToCkpt(key) {
  /* 旧 switchTab("tree")：经顶栏 tab 按钮走 main.js 的 tab 路由 */
  var btn = document.querySelector('#tabs .tab[data-tab="tree"]');
  if (btn) btn.click();
  if (typeof treeTab.selectCheckpoint === "function") treeTab.selectCheckpoint(key);
}

function loadLlmHelp() {
  var pre = document.getElementById("llm-help");
  if (pre.textContent) return; /* 只加载一次 */
  api.get("/api/llm/help").then(function (doc) {
    pre.textContent = formatLlmHelp(doc);
  }).catch(function (err) {
    pre.textContent = "加载失败: " + (err && err.message ? err.message : err);
  });
}

function formatLlmHelp(doc) {
  var origin = location.origin;
  var out = [];
  out.push("# " + (doc.name || "motus-monitor") + " — " + (doc.purpose || ""));
  out.push("");
  (doc.endpoints || []).forEach(function (ep) {
    out.push("## " + ep.method + " " + ep.path);
    if (ep.description) out.push("  " + ep.description);
    if (ep.method === "GET") {
      out.push("  curl -s '" + origin + ep.path + "'");
    }
    if (ep.body_schema) {
      out.push("  Body schema:");
      out.push("  " + JSON.stringify(ep.body_schema, null, 2).split("\n").join("\n  "));
    }
    if (ep.example && ep.example.curl) {
      out.push("  示例:");
      out.push("  " + String(ep.example.curl).replace(/http:\/\/localhost:\d+/, origin));
    }
    out.push("");
  });
  if (doc.notes && doc.notes.length) {
    out.push("## 注意");
    doc.notes.forEach(function (n) { out.push("  - " + n); });
  }
  return out.join("\n");
}

export default {
  id: "benchmarks",

  init: function () {
    document.getElementById("bench-refresh").addEventListener("click", loadAnnotations);
    var llmBtn = document.getElementById("llm-toggle");
    llmBtn.addEventListener("click", function () {
      var pre = document.getElementById("llm-help");
      var show = pre.classList.contains("hidden");
      pre.classList.toggle("hidden", !show);
      llmBtn.textContent = show ? "收起" : "展开";
      if (show) loadLlmHelp();
    });
  },

  enter: function () {
    loadAnnotations();
  },

  leave: function () {
    // 无清理
  },

  /* 数据事件联动：旧实现 onDataChanged 不刷新本 tab，保持空实现 */
  refresh: function () {
  },

  /* SSE annotation 消息（旧 onServerMessage 的 benchmarks 分支） */
  onAnnotation: function () {
    loadAnnotations();
  },
};
