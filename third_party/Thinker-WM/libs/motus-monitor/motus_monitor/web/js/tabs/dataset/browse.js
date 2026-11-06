/* Motus Monitor 前端 —— 训练集 tab · 数据集浏览模式（阶段 1C）。
 * 逻辑逐字迁自旧 web/dataset.js：
 *   ds-status   renderStatus(327-341)
 *   数据集列表  loadDatasets/renderDatasets/selectDataset(424-462)
 *   两级树      loadEpTree/renderEpTree/selectEpGroup(464-511)
 *   分组标签    renderEpScope/showEpHint(513-540)
 *   分页列表    loadEpisodes/renderEpisodes(542-585)
 *   控件绑定    旧 bindOnce 349-361 中属于浏览模式的部分（搜索 + EP_PAGE 分页）
 * 树渲染沿用旧手搓逻辑逐字搬迁（固定两级：机器人→任务），未改用 ui/tree.js；
 * 分页沿用 index.html 里的静态按钮绑定，未改用 ui/pager.js（它是动态创建组件）。
 */
import { api } from "../../api.js";
import { util } from "../../util.js";
import { createProgress } from "../../ui/progress.js";
import { shared, qs } from "./state.js";
import { openEpisode, currentEpisodeId } from "./viewer.js";

var esc = util.esc, el = util.el;

var EP_PAGE = 50;

/* 浏览模式状态（旧 ds 29-76 中属于数据集浏览的部分） */
var b = {
  datasets: [],
  episodes: [],
  epTotal: 0,
  epOffset: 0,
  epQuery: "",
  epTree: null,            // episode_tree payload（机器人→任务 两级）
  epTreeExpanded: {},      // robot name -> bool（单机器人默认展开）
  epGroup: null,           // 当前选中的 task 节点对象（null = 未选；prefix 可为空串=整库）
};

/* 侧栏列表加载中的迷你进度条（旧 283-291 listLoading；浏览模式专用） */
function listLoading(boxId, text) {
  var box = document.getElementById(boxId);
  box.innerHTML = "";
  var prog = createProgress({ label: text });
  prog.el.classList.add("mini");
  prog.indet();
  box.appendChild(prog.el);
}

/* 浏览模式控件绑定（旧 bindOnce 349-361；index.js init 只调一次，替代 bindOnce.done） */
export function bindBrowse() {
  document.getElementById("ds-ep-search-btn").addEventListener("click", function () {
    b.epQuery = document.getElementById("ds-ep-search").value.trim();
    loadEpisodes(0);
  });
  document.getElementById("ds-ep-search").addEventListener("keydown", function (ev) {
    if (ev.key === "Enter") { b.epQuery = ev.target.value.trim(); loadEpisodes(0); }
  });
  document.getElementById("ds-ep-prev").addEventListener("click", function () {
    loadEpisodes(Math.max(0, b.epOffset - EP_PAGE));
  });
  document.getElementById("ds-ep-next").addEventListener("click", function () {
    loadEpisodes(b.epOffset + EP_PAGE);
  });
}

/* ds-status 后端状态行（旧 327-341 renderStatus；enter 时由 index.js 调用） */
export function renderStatus(st) {
  var box = document.getElementById("ds-status");
  box.innerHTML = "";
  if (!st.enabled) {
    box.appendChild(el("div", "ds-status-err", {
      text: "预览后端未启用" + (st.error ? "：" + st.error : "") +
        "（serve 需 --remote ssh 或 --dataset-local）",
    }));
    return;
  }
  box.appendChild(el("div", "ds-status-ok", {
    text: "后端：" + (st.mode === "ssh" ? ("ssh " + (st.target || "")) : "本机") +
      (st.running ? " · 已连接" : " · 未运行"),
  }));
}

/* ------------------------------------------------------------ 数据集浏览 */
export function loadDatasets() {
  listLoading("ds-datasets", "加载数据集…");
  api.get("/api/dataset/datasets").then(function (payload) {
    b.datasets = payload.datasets || [];
    renderDatasets();
  }).catch(function (err) {
    document.getElementById("ds-datasets").innerHTML =
      '<div class="ds-error">' + esc(err.message) + "</div>";
  });
}

function renderDatasets() {
  var box = document.getElementById("ds-datasets");
  box.innerHTML = "";
  b.datasets.forEach(function (d) {
    var row = el("div", "ds-dataset" + (shared.dataset === d.name ? " active" : ""));
    row.appendChild(el("span", "ds-dataset-name", { text: d.name }));
    row.appendChild(el("span", "ds-dataset-count", {
      text: d.episodes != null ? String(d.episodes) : "?",
    }));
    row.addEventListener("click", function () { selectDataset(d.name); });
    box.appendChild(row);
  });
}

function selectDataset(name) {
  shared.dataset = name;
  b.epOffset = 0;
  b.epQuery = "";
  b.epGroup = null;
  b.epTree = null;
  b.epTreeExpanded = {};
  document.getElementById("ds-ep-search").value = "";
  renderDatasets();
  document.getElementById("ds-ep-browser").classList.remove("hidden");
  renderEpScope();
  showEpHint();
  loadEpTree();
}

/* 机器人→任务 两级树；子id 在选中任务后进入下方分页列表 */
function loadEpTree() {
  if (!shared.dataset) return;
  listLoading("ds-eptree", "加载层级树…");
  api.get("/api/dataset/episode_tree" + qs({ dataset: shared.dataset })).then(function (payload) {
    b.epTree = payload;
    var robots = payload.robots || [];
    if (robots.length === 1) b.epTreeExpanded[robots[0].name] = true;
    renderEpTree();
  }).catch(function (err) {
    document.getElementById("ds-eptree").innerHTML =
      '<div class="ds-error">' + esc(err.message) + "</div>";
  });
}

function renderEpTree() {
  var box = document.getElementById("ds-eptree");
  box.innerHTML = "";
  if (!b.epTree) return;
  (b.epTree.robots || []).forEach(function (robot) {
    var expanded = !!b.epTreeExpanded[robot.name];
    var rrow = el("div", "ds-eptree-robot");
    rrow.appendChild(el("span", "ds-eptree-toggle", { text: expanded ? "▾" : "▸" }));
    rrow.appendChild(el("span", "ds-eptree-name", { text: robot.name, attrs: { title: robot.name } }));
    rrow.appendChild(el("span", "ds-dataset-count", { text: String(robot.count) }));
    rrow.addEventListener("click", function () {
      b.epTreeExpanded[robot.name] = !expanded;
      renderEpTree();
    });
    box.appendChild(rrow);
    if (!expanded) return;
    (robot.tasks || []).forEach(function (task) {
      var trow = el("div", "ds-eptree-task" + (b.epGroup === task ? " active" : ""));
      trow.appendChild(el("span", "ds-eptree-name", { text: task.label, attrs: { title: task.label } }));
      trow.appendChild(el("span", "ds-dataset-count", { text: String(task.count) }));
      trow.addEventListener("click", function () { selectEpGroup(task); });
      box.appendChild(trow);
    });
  });
}

function selectEpGroup(task) {
  b.epGroup = (b.epGroup === task) ? null : task;   // 再点一次取消选中
  b.epOffset = 0;
  renderEpTree();
  renderEpScope();
  loadEpisodes(0);
}

/* 列表上方的当前分组标签 + 清除按钮 */
function renderEpScope() {
  var box = document.getElementById("ds-ep-scope");
  box.innerHTML = "";
  if (!b.epGroup) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  box.appendChild(el("span", "ds-ep-scope-label", {
    text: b.epGroup.label, attrs: { title: b.epGroup.prefix },
  }));
  var clear = el("button", "btn ds-ep-scope-clear", { text: "✕", attrs: { title: "清除分组" } });
  clear.addEventListener("click", function () {
    b.epGroup = null;
    b.epOffset = 0;
    renderEpTree();
    renderEpScope();
    showEpHint();
  });
  box.appendChild(clear);
}

function showEpHint() {
  document.getElementById("ds-episodes").innerHTML =
    '<div class="ds-muted ds-ep-hint">在上方选择 机器人→任务 浏览子id，或用搜索全局查找</div>';
  document.getElementById("ds-ep-page-info").textContent = "";
}

function loadEpisodes(offset) {
  if (!shared.dataset) return;
  if (!b.epQuery && !b.epGroup) {   // 未选分组且未搜索：不再罗列全量长列表
    showEpHint();
    return;
  }
  b.epOffset = offset;
  listLoading("ds-episodes", "加载 episode 列表…");
  var url = "/api/dataset/episodes" + qs({
    dataset: shared.dataset, offset: offset, limit: EP_PAGE, q: b.epQuery,
    prefix: b.epQuery ? "" : (b.epGroup ? b.epGroup.prefix : ""),   // 搜索时全局查找（忽略分组）
  });
  api.get(url).then(function (payload) {
    b.episodes = payload.episode_ids || [];
    b.epTotal = payload.total;
    renderEpisodes();
  }).catch(function (err) {
    document.getElementById("ds-episodes").innerHTML =
      '<div class="ds-error">' + esc(err.message) + "</div>";
  });
}

function renderEpisodes() {
  var box = document.getElementById("ds-episodes");
  box.innerHTML = "";
  var prefix = b.epGroup && !b.epQuery ? b.epGroup.prefix : "";
  b.episodes.forEach(function (eid) {
    // 分组浏览时只显示子id（去掉组前缀），搜索时显示完整 id
    var label = prefix && eid.lastIndexOf(prefix, 0) === 0
      ? eid.slice(prefix.length)
      : eid;
    var row = el("div", "ds-episode" + (currentEpisodeId() === eid ? " active" : ""), {
      text: label,
      attrs: { title: eid },
    });
    row.addEventListener("click", function () { openEpisode(shared.dataset, eid); });
    box.appendChild(row);
  });
  var total = b.epTotal;
  var info = total < 0
    ? (b.epOffset + 1) + "–" + (b.epOffset + b.episodes.length)
    : (b.epOffset + 1) + "–" + Math.min(b.epOffset + EP_PAGE, total) + " / " + total;
  document.getElementById("ds-ep-page-info").textContent = b.episodes.length ? info : "无结果";
}
