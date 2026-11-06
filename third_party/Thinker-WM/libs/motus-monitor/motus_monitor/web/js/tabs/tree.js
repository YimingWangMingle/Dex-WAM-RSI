/* Motus Monitor 前端 —— 训练树 tab。
 * 左侧：/api/tree 渲染 run 树（parent 链接）+ 每个 run 的 checkpoints；
 * 右侧：checkpoint 详情（基本信息、run 配置、benchmarks 表格、annotation 编辑器）。
 * POST /api/annotations 是整体替换语义，因此保存前先 GET 旧文档合并。
 *
 * 阶段 1D：从旧 web/tree.js（MonApp.treeTab）逐字迁入 ES module（模块即严格
 * 模式，原 IIFE/"use strict" 去除）。整树仍用 container.innerHTML="" +
 * renderRunNode 递归重建（行为 parity 优先，不改用 ui/tree.js）。
 * selectCheckpoint 同时挂在默认导出上（benchmarks tab 的 jumpToCkpt 经
 * `import treeTab from "./tree.js"` 默认导出调用）并保留命名导出，对应旧
 * app.js:1404 的 MonApp.treeTab.selectCheckpoint。
 * boot 时机：旧 boot 显式调 MonApp.treeTab.init()（旧 app.js:1867）只绑定
 * tree-refresh 按钮（在 #tab-tree 页内，未激活不可见不可点），首次 /api/tree
 * 加载在切入 tab 时才发生（旧 app.js:309）——无 boot 期依赖，新模型的
 * 惰性 init（首次 enter 前调用）语义等价。
 */
import { api } from "../api.js";
import { util } from "../util.js";

var KIND_LABEL = { finetune: "finetune", resume: "resume" };

var treeTab = {
  tree: null,            // /api/tree 响应
  expanded: {},          // run_id -> bool（默认展开）
  selectedKey: null,     // 当前选中的 ckpt key
  _ckptIndex: {},        // key -> {run_id, step, path, created_at, has_annotation}
};

/* ------------------------------------------------------------ 左侧树 */
function stateDot(state) {
  var s = state || "unknown";
  if (["running", "completed", "failed"].indexOf(s) < 0) s = "unknown";
  return util.el("span", "state-dot " + s);
}

function renderRunNode(node, depth) {
  var frag = document.createDocumentFragment();
  var runDiv = util.el("div", "tree-run");

  var row = util.el("div", "tree-run-row");
  var expanded = treeTab.expanded[node.run_id] !== false;
  var toggle = util.el("span", "tree-toggle", { text: expanded ? "▾" : "▸" });
  row.appendChild(toggle);
  row.appendChild(stateDot(node.state));
  row.appendChild(util.el("span", "tree-name", {
    text: node.run_name || node.run_id,
    attrs: { title: node.run_id },
  }));
  var stepText = (node.last_step != null ? node.last_step : "—") +
    "/" + (node.max_steps != null ? node.max_steps : "—");
  row.appendChild(util.el("span", "tree-step", { text: stepText }));
  row.addEventListener("click", function () {
    treeTab.expanded[node.run_id] = !expanded;
    renderTree();
  });
  runDiv.appendChild(row);

  if (node.parent && node.parent.step != null) {
    var kind = KIND_LABEL[node.parent.kind] || node.parent.kind || "ckpt";
    runDiv.appendChild(util.el("div", "tree-edge", {
      text: "← " + kind + " 自 step " + node.parent.step,
      attrs: { title: node.parent.path || "" },
    }));
  }

  if (expanded) {
    var kids = util.el("div", "tree-children");

    var ckpts = (node.checkpoints || []).slice().sort(function (a, b) {
      return (a.step || 0) - (b.step || 0);
    });
    ckpts.forEach(function (ckpt) {
      treeTab._ckptIndex[ckpt.key] = {
        run_id: node.run_id,
        step: ckpt.step,
        path: ckpt.path,
        created_at: ckpt.created_at,
        has_annotation: ckpt.has_annotation,
      };
      var ckRow = util.el("div", "tree-ckpt-row" +
        (treeTab.selectedKey === ckpt.key ? " selected" : ""));
      ckRow.appendChild(util.el("span", "ckpt-icon", { text: "⚑" }));
      ckRow.appendChild(util.el("span", null, { text: "step " + ckpt.step }));
      if (ckpt.has_annotation) {
        ckRow.appendChild(util.el("span", "ann-tag", { text: "已标注" }));
      }
      ckRow.addEventListener("click", function () { showDetail(ckpt.key); });
      kids.appendChild(ckRow);
    });

    (node.children || []).forEach(function (child) {
      kids.appendChild(renderRunNode(child, depth + 1));
    });
    runDiv.appendChild(kids);
  } else {
    /* 折叠时也登记 ckpt 索引，便于 selectCheckpoint 查找 */
    (node.checkpoints || []).forEach(function (ckpt) {
      treeTab._ckptIndex[ckpt.key] = {
        run_id: node.run_id,
        step: ckpt.step,
        path: ckpt.path,
        created_at: ckpt.created_at,
        has_annotation: ckpt.has_annotation,
      };
    });
    (node.children || []).forEach(function (child) { indexOnly(child); });
  }
  frag.appendChild(runDiv);
  return frag;
}

function indexOnly(node) {
  (node.checkpoints || []).forEach(function (ckpt) {
    treeTab._ckptIndex[ckpt.key] = {
      run_id: node.run_id,
      step: ckpt.step,
      path: ckpt.path,
      created_at: ckpt.created_at,
      has_annotation: ckpt.has_annotation,
    };
  });
  (node.children || []).forEach(indexOnly);
}

function renderTree() {
  var container = document.getElementById("tree-view");
  container.innerHTML = "";
  treeTab._ckptIndex = {};
  var roots = (treeTab.tree && treeTab.tree.roots) || [];
  if (!roots.length) {
    container.appendChild(util.el("div", "empty-hint", { text: "暂无训练记录" }));
    return;
  }
  roots.forEach(function (node) { container.appendChild(renderRunNode(node, 0)); });
}

function refreshTree() {
  return api.get("/api/tree").then(function (tree) {
    treeTab.tree = tree;
    renderTree();
  }).catch(function (err) {
    console.error("tree refresh failed", err);
  });
}

/* 基准矩阵跳转入口：选中并展开对应 ckpt */
export function selectCheckpoint(key) {
  var open = function () {
    /* 确保包含该 ckpt 的 run 及其祖先都展开 */
    var meta = treeTab._ckptIndex[key];
    if (meta) expandAncestors((treeTab.tree && treeTab.tree.roots) || [], meta.run_id);
    treeTab.selectedKey = key;
    renderTree();
    showDetail(key);
  };
  if (!treeTab.tree) {
    refreshTree().then(open);
  } else {
    open();
  }
}

function expandAncestors(nodes, runId) {
  for (var i = 0; i < nodes.length; i++) {
    var n = nodes[i];
    if (n.run_id === runId) { treeTab.expanded[n.run_id] = true; return true; }
    if (expandAncestors(n.children || [], runId)) {
      treeTab.expanded[n.run_id] = true;
      return true;
    }
  }
  return false;
}

/* ------------------------------------------------------------ 右侧详情 */
function kv(k, v) {
  return [
    util.el("div", "k", { text: k }),
    util.el("div", "v", { text: v == null || v === "" ? "—" : String(v) }),
  ];
}

function metricsSummary(metrics) {
  var keys = Object.keys(metrics || {});
  if (!keys.length) return "—";
  return keys.map(function (k) {
    var v = metrics[k];
    return k + "=" + (util.isNum(v) ? util.fmtVal(v) : String(v));
  }).join("  ");
}

function parseMetricValue(text) {
  var t = String(text).trim();
  if (t === "") return "";
  try { return JSON.parse(t); } catch (e) { return t; }
}

function showDetail(key) {
  treeTab.selectedKey = key;
  /* 同步左侧高亮 */
  var rows = document.querySelectorAll(".tree-ckpt-row");
  rows.forEach(function (r) { r.classList.remove("selected"); });
  var meta = treeTab._ckptIndex[key];
  var panel = document.getElementById("ckpt-detail");
  panel.innerHTML = "";
  panel.appendChild(util.el("div", "empty-hint", { text: "加载中…" }));

  var annReq = api.getOrNull("/api/annotations?ckpt=" + encodeURIComponent(key));
  var runReq = meta
    ? api.getOrNull("/api/runs/" + encodeURIComponent(meta.run_id))
    : Promise.resolve(null);

  Promise.all([annReq, runReq]).then(function (results) {
    var ann = results[0];
    var run = results[1];
    renderDetail(panel, key, meta, ann, run);
    /* 左侧高亮 */
    renderTreeSelected(key);
  }).catch(function (err) {
    panel.innerHTML = "";
    panel.appendChild(util.el("div", "empty-hint", {
      text: "加载失败: " + (err && err.message ? err.message : err),
    }));
  });
}

function renderTreeSelected(key) {
  var rows = document.querySelectorAll(".tree-ckpt-row");
  var container = document.getElementById("tree-view");
  if (!container) return;
  /* 通过重新渲染树保证索引与行一致（简单可靠） */
  renderTree();
}

function renderDetail(panel, key, meta, ann, run) {
  panel.innerHTML = "";
  var benchmarks = (ann && ann.benchmarks) || {};
  var notes = ann ? ann.notes || "" : "";

  /* 基本信息 */
  panel.appendChild(util.el("div", "panel-head", { text: "Checkpoint 详情" }));
  var info = util.el("div", "detail-kv");
  function add(k, v) { kv(k, v).forEach(function (n) { info.appendChild(n); }); }
  add("Run", meta ? meta.run_id : (ann && ann.ckpt && ann.ckpt.run_id));
  add("Step", meta ? meta.step : (ann && ann.ckpt && ann.ckpt.step));
  add("Path", meta ? meta.path : (ann && ann.ckpt && ann.ckpt.path));
  add("创建时间", util.fmtDateTime(meta && meta.created_at));
  add("Key", key);
  if (ann) {
    add("标注更新", util.fmtClock(ann.updated_at) + " by " + (ann.updated_by || "—"));
  }
  panel.appendChild(info);

  /* run 配置（折叠） */
  if (run && run.config && Object.keys(run.config).length) {
    panel.appendChild(util.el("div", "detail-section", { text: "Run 配置" }));
    var fold = util.el("details", "config-fold");
    fold.appendChild(util.el("summary", null, {
      text: (run.run_name || run.run_id || "") + " 的完整配置（点击展开）",
    }));
    fold.appendChild(util.el("pre", null, {
      text: JSON.stringify(run.config, null, 2),
    }));
    panel.appendChild(fold);
  }

  /* 已有 benchmarks 表格 */
  panel.appendChild(util.el("div", "detail-section", { text: "基准结果" }));
  var names = Object.keys(benchmarks);
  if (!names.length) {
    panel.appendChild(util.el("div", "no-data", { text: "暂无基准数据，可在下方录入。" }));
  } else {
    var table = util.el("table", "bench-table");
    table.innerHTML = "<thead><tr><th>基准</th><th>指标</th><th style='text-align:right'>操作</th></tr></thead>";
    var tbody = util.el("tbody");
    names.forEach(function (name) {
      var tr = util.el("tr");
      tr.appendChild(util.el("td", "bench-name", { text: name }));
      tr.appendChild(util.el("td", null, { text: metricsSummary(benchmarks[name]) }));
      var btns = util.el("td", "row-btns");
      var editBtn = util.el("button", "btn", { text: "编辑" });
      editBtn.addEventListener("click", function () {
        fillForm(name, benchmarks[name], notes);
      });
      var delBtn = util.el("button", "btn danger", { text: "删除" });
      delBtn.addEventListener("click", function () {
        deleteBenchmark(key, meta, name);
      });
      btns.appendChild(editBtn);
      btns.appendChild(delBtn);
      tr.appendChild(btns);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    panel.appendChild(table);
  }

  /* 编辑区 */
  panel.appendChild(util.el("div", "detail-section", { text: "录入 / 编辑基准" }));
  panel.appendChild(buildForm(key, meta, notes));
}

/* ------------------------------------------------------------ 编辑表单 */
function buildForm(key, meta, notes) {
  var form = util.el("div", "edit-form");

  var nameRow = util.el("div", "form-row");
  nameRow.appendChild(util.el("label", null, { text: "基准名" }));
  var nameInput = util.el("input", null, { attrs: { type: "text", placeholder: "如 simpler_env" } });
  nameRow.appendChild(nameInput);
  form.appendChild(nameRow);

  var metricsBox = util.el("div");
  form.appendChild(metricsBox);

  function addMetricRow(k, v) {
    var row = util.el("div", "metric-row");
    var kInput = util.el("input", null, { attrs: { type: "text", placeholder: "指标名，如 success_rate" } });
    var vInput = util.el("input", null, { attrs: { type: "text", placeholder: "数值或字符串" } });
    if (k != null) kInput.value = k;
    if (v != null) vInput.value = typeof v === "string" ? v : JSON.stringify(v);
    var del = util.el("button", "btn danger", { text: "×" });
    del.addEventListener("click", function () { metricsBox.removeChild(row); });
    row.appendChild(kInput);
    row.appendChild(vInput);
    row.appendChild(del);
    metricsBox.appendChild(row);
  }
  addMetricRow();

  var addBtnRow = util.el("div", "form-row");
  var addBtn = util.el("button", "btn", { text: "+ 添加指标" });
  addBtn.addEventListener("click", function () { addMetricRow(); });
  addBtnRow.appendChild(util.el("label", null, { text: "" }));
  addBtnRow.appendChild(addBtn);
  form.appendChild(addBtnRow);

  var notesRow = util.el("div", "form-row");
  notesRow.appendChild(util.el("label", null, { text: "备注" }));
  var notesInput = util.el("textarea");
  notesInput.value = notes || "";
  notesRow.appendChild(notesInput);
  form.appendChild(notesRow);

  var actions = util.el("div", "form-actions");
  var saveBtn = util.el("button", "btn primary", { text: "保存" });
  var clearBtn = util.el("button", "btn", { text: "清空表单" });
  var hint = util.el("span", "save-hint");
  actions.appendChild(saveBtn);
  actions.appendChild(clearBtn);
  actions.appendChild(hint);
  form.appendChild(actions);

  clearBtn.addEventListener("click", function () {
    nameInput.value = "";
    metricsBox.innerHTML = "";
    addMetricRow();
    hint.textContent = "";
    hint.className = "save-hint";
  });

  saveBtn.addEventListener("click", function () {
    var benchName = nameInput.value.trim();
    var newMetrics = {};
    var rows = metricsBox.querySelectorAll(".metric-row");
    for (var i = 0; i < rows.length; i++) {
      var inputs = rows[i].querySelectorAll("input");
      var mk = inputs[0].value.trim();
      if (!mk) continue;
      newMetrics[mk] = parseMetricValue(inputs[1].value);
    }
    if (benchName && !Object.keys(newMetrics).length) {
      hint.textContent = "请至少填写一个指标";
      hint.className = "save-hint err";
      return;
    }
    saveBtn.disabled = true;
    hint.textContent = "保存中…";
    hint.className = "save-hint";

    /* 合并语义：先读旧文档，再整体替换提交 */
    api.getOrNull("/api/annotations?ckpt=" + encodeURIComponent(key)).then(function (old) {
      var merged = {};
      if (old && old.benchmarks) {
        Object.keys(old.benchmarks).forEach(function (b) { merged[b] = old.benchmarks[b]; });
      }
      if (benchName) merged[benchName] = newMetrics;
      var runId = meta ? meta.run_id : (old && old.ckpt && old.ckpt.run_id);
      var step = meta ? meta.step : (old && old.ckpt && old.ckpt.step);
      return api.post("/api/annotations", {
        ckpt: { run_id: runId, step: step },
        benchmarks: merged,
        notes: notesInput.value,
        updated_by: "web-ui",
      });
    }).then(function () {
      hint.textContent = "已保存 ✓";
      hint.className = "save-hint ok";
      saveBtn.disabled = false;
      refreshTree().then(function () { showDetail(key); });
    }).catch(function (err) {
      hint.textContent = "保存失败: " + (err && err.message ? err.message : err);
      hint.className = "save-hint err";
      saveBtn.disabled = false;
    });
  });

  /* 供“编辑”按钮回填 */
  form._fill = function (name, metrics, oldNotes) {
    nameInput.value = name;
    metricsBox.innerHTML = "";
    var keys = Object.keys(metrics || {});
    if (!keys.length) addMetricRow();
    keys.forEach(function (k) { addMetricRow(k, metrics[k]); });
    notesInput.value = oldNotes || "";
    hint.textContent = "";
    hint.className = "save-hint";
  };
  return form;
}

function fillForm(name, metrics, notes) {
  var form = document.querySelector("#ckpt-detail .edit-form");
  if (form && form._fill) {
    form._fill(name, metrics, notes);
    form.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function deleteBenchmark(key, meta, name) {
  api.getOrNull("/api/annotations?ckpt=" + encodeURIComponent(key)).then(function (old) {
    if (!old) return null;
    var merged = {};
    Object.keys(old.benchmarks || {}).forEach(function (b) {
      if (b !== name) merged[b] = old.benchmarks[b];
    });
    var runId = meta ? meta.run_id : old.ckpt && old.ckpt.run_id;
    var step = meta ? meta.step : old.ckpt && old.ckpt.step;
    return api.post("/api/annotations", {
      ckpt: { run_id: runId, step: step },
      benchmarks: merged,
      notes: old.notes || "",
      updated_by: "web-ui",
    });
  }).then(function () {
    refreshTree().then(function () { showDetail(key); });
  }).catch(function (err) {
    console.error("delete benchmark failed", err);
  });
}

/* ------------------------------------------------------------ tab 接口 */
export default {
  id: "tree",

  /* 首次 enter 前惰性调用一次：控件绑定（旧 treeTab.init，boot 期调用见头部注释） */
  init: function () {
    document.getElementById("tree-refresh").addEventListener("click", function () {
      refreshTree();
    });
  },

  /* 切入：旧 switchTab 分支 MonApp.treeTab.refresh()（旧 app.js:309） */
  enter: function () {
    refreshTree();
  },

  leave: function () {
    // 无清理
  },

  /* 数据事件联动：旧实现 onDataChanged 不刷新本 tab，保持空实现 */
  refresh: function () {},

  /* SSE annotation 消息（旧 onServerMessage 的 tree 分支，app.js:1535）：
   * 刷新树与当前详情（旧 treeTab.onAnnotationChanged） */
  onAnnotation: function () {
    refreshTree().then(function () {
      if (treeTab.selectedKey) showDetail(treeTab.selectedKey);
    });
  },

  /* 基准矩阵跳转入口（benchmarks tab 经默认导出调用，见头部注释） */
  selectCheckpoint: selectCheckpoint,
};
