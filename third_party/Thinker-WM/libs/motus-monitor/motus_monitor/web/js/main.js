/* Motus Monitor 前端 —— 应用骨架（ES module 入口）。
 * 从旧 web/app.js 抽取：boot 流程（app.js:1863-1908）、tab 路由（app.js:299-314）、
 * 数据变更路由（app.js:1483-1528）、loadSyncStatus（app.js:1540-1573）、
 * initAuthChip（app.js:1576-1587）、主题切换（app.js:1763-1781）、
 * 全局控件绑定（app.js:1783-1807 中属于全局的部分）。
 * tab 内部逻辑（图表、渲染、tab 专属控件）归 js/tabs/*.js 各模块。
 */
import { api } from "./api.js";
import { util } from "./util.js";
import { store } from "./store.js";
import { domRenderBlocked, guardAutoRender, flushDeferredRender } from "./ui/guard.js";
import { flushDropdowns } from "./ui/dropdown.js";
import { initRootPicker, refreshDataRootLabel } from "./rootpicker.js";
import overviewTab from "./tabs/overview.js";
import curvesTab from "./tabs/curves.js";
import timingTab from "./tabs/timing.js";
import compareTab from "./tabs/compare.js";
import treeTab from "./tabs/tree.js";
import benchmarksTab from "./tabs/benchmarks.js";
import evalTab from "./tabs/eval.js";
import datasetTab from "./tabs/dataset/index.js";
import samplingTab from "./tabs/sampling.js";
import taskmetricsTab from "./tabs/taskmetrics.js";

/* ------------------------------------------------------------ Tab 注册表
 * 每个 tab 模块默认导出：
 *   {id, init(), enter(), leave(), refresh(), poll?(), onAnnotation?(), onTheme?()}
 *   init()    首次 enter 前自动调一次（惰性）
 *   enter()   每次切入调用（全量加载/渲染）
 *   leave()   切出时调用
 *   refresh() 数据事件到达时的轻量刷新（main.js 已做 guardAutoRender 守卫）；
 *             无数据联动的 tab 保持空实现（对应旧 onDataChanged 只渲染 overview/curves/timing）
 *   poll?()   doUpdate 周期内调用（随 SSE/兜底轮询增量拉取），返回 Promise
 *   onAnnotation?(msg)  SSE annotation 消息（当前 tab 时路由）
 *   onTheme?()  主题切换后调用（canvas 取色来自 CSS 变量，需重绘）
 */
var TABS = [overviewTab, curvesTab, timingTab, compareTab, treeTab, benchmarksTab, evalTab, datasetTab, samplingTab, taskmetricsTab];
var tabById = {};
TABS.forEach(function (t) { tabById[t.id] = t; });
var initedTabs = {};
document.getElementById('open-task-loss').onclick=function(){switchTab('taskmetrics');};
document.getElementById('open-task-eval').onclick=function(){switchTab('taskmetrics');
  const phase=document.getElementById('taskmetrics-phase');phase.value='validation';phase.dispatchEvent(new Event('change'));};

function currentTab() {
  return tabById[store.state.tab] || null;
}

/* 惰性 init 后 enter */
function activateTab(tab) {
  if (!initedTabs[tab.id]) {
    initedTabs[tab.id] = true;
    tab.init();
  }
  tab.enter();
}

function switchTab(name) {
  if (!tabById[name]) return;
  var prev = currentTab();
  store.state.tab = name;
  var tabs = document.querySelectorAll("#tabs .tab");
  tabs.forEach(function (t) { t.classList.toggle("active", t.getAttribute("data-tab") === name); });
  var pages = document.querySelectorAll(".tab-page");
  pages.forEach(function (p) { p.classList.toggle("active", p.id === "tab-" + name); });
  if (prev && prev !== tabById[name] && prev.leave) prev.leave();
  activateTab(tabById[name]);
}

/* ------------------------------------------------------------ 数据变更路由
 * store 的数据事件到达时调当前 tab 的 refresh()（保持 guardAutoRender 守卫）。 */
function onDataChanged() {
  if (guardAutoRender(store.state)) return;
  var tab = currentTab();
  if (tab && tab.refresh) tab.refresh();
}
store.on("steps", onDataChanged);
store.on("events", onDataChanged);
store.on("system", onDataChanged);

var updateScheduled = false;
function scheduleUpdate() {
  if (updateScheduled) return;
  updateScheduled = true;
  setTimeout(function () {
    updateScheduled = false;
    doUpdate();
  }, 800);
}

function doUpdate() {
  if (store.state.updating) return;
  store.state.updating = true;
  var tasks = [store.loadRuns(), store.refreshDetail(), loadSyncStatus()];
  store.pollReaders().forEach(function (t) { tasks.push(t); });
  /* 当前 tab 的增量拉取随同一 SSE/兜底轮询周期进行（如 eval 流、overview 系统卡 tail） */
  var tab = currentTab();
  if (tab && tab.poll) tasks.push(tab.poll());
  Promise.all(tasks).then(function () {
    store.state.updating = false;
    if (store.state.tab === "overview") {
      if (guardAutoRender(store.state)) return;
      tab.refresh();
    }
  }).catch(function () {
    store.state.updating = false;
  });
}

function onServerMessage(msg) {
  if (!msg) return;
  if (msg.type === "update") {
    scheduleUpdate();
  } else if (msg.type === "annotation") {
    /* annotation 路由给当前 tab 的 onAnnotation 钩子 */
    var tab = currentTab();
    if (tab && tab.onAnnotation) tab.onAnnotation(msg);
  }
}

/* ------------------------------------------------------------ 同步状态（顶栏） */
function loadSyncStatus() {
  return api.getOrNull("/api/sync/status").then(function (s) {
    var el = document.getElementById("sync-status");
    var prog = document.getElementById("sync-progress");
    if (!s) { el.textContent = ""; prog.classList.add("hidden"); return; }
    if (s.backend) {
      el.textContent = "sync: " + s.backend + (util.isNum(s.data_version) ? " v" + s.data_version : "");
    } else {
      el.textContent = "本地模式";
    }
    el.title = JSON.stringify(s, null, 2);
    /* 拉取进度条：任一远端处于 pulling 即显示（周期内实时更新，周期结束隐藏） */
    var active = null;
    var remotes = s.remotes || (s.backend ? [s] : []);
    for (var i = 0; i < remotes.length; i++) {
      if (remotes[i] && remotes[i].phase === "pulling") { active = remotes[i]; break; }
    }
    if (active) {
      var done = active.pull_done_bytes || 0;
      var total = active.pull_pending_bytes || 0;
      var pct = total > 0 ? Math.min(100, (done * 100) / total) : 0;
      var label = pct.toFixed(0) + "%  " + util.fmtBytes(done) + "/" + util.fmtBytes(total);
      prog.classList.remove("hidden");
      prog.querySelector(".sync-progress-fill").style.width = pct.toFixed(1) + "%";
      prog.querySelector(".sync-progress-text").textContent = label;
      prog.title = "正在同步 [" + (active.name || "") + "] " + label +
        "\n文件: " + (active.pull_done_files || 0) + "/" + (active.pull_pending_files || 0) +
        (active.pull_current_file ? "\n当前: " + active.pull_current_file : "");
    } else {
      prog.classList.add("hidden");
    }
    store.state.syncPulling = !!active;
    store.emit("sync", s);
  });
}

/* ------------------------------------------------------------ 登录用户名片 */
function initAuthChip() {
  api.whoami().then(function (info) {
    if (!info || !info.auth_enabled) return;  /* 本机模式无登录，保持隐藏 */
    var chip = document.getElementById("user-chip");
    var btn = document.getElementById("logout-btn");
    chip.textContent = info.user || "";
    chip.classList.remove("hidden");
    btn.classList.remove("hidden");
    /* /logout 是 GET 路由（session.clear 后 302 到 /login），直接跳转即可 */
    btn.addEventListener("click", function () { window.location.href = "/logout"; });
  });
}

/* ------------------------------------------------------------ 主题切换（亮/暗） */
function currentTheme() {
  try { return localStorage.getItem("motus.monitor.theme") || "light"; } catch (e) { return "light"; }
}

function applyTheme(name) {
  if (name !== "light" && name !== "dark") name = "light";
  document.documentElement.setAttribute("data-theme", name);
  try { localStorage.setItem("motus.monitor.theme", name); } catch (e) {}
  var btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = name === "light" ? "☾" : "☀";
    btn.title = name === "light" ? "切换到暗主题" : "切换到亮主题";
  }
  /* canvas 取色来自 CSS 变量，由当前 tab 重绘自己的图表 */
  var tab = currentTab();
  if (tab && tab.onTheme) tab.onTheme();
}

/* ------------------------------------------------------------ 全局控件绑定
 * 只保留跨 tab 的部分；属于具体 tab 的控件绑定（ema-slider 等）归各 tab 模块。 */
function bindGlobalControls() {
  applyTheme(currentTheme());
  document.getElementById("theme-toggle").addEventListener("click", function () {
    applyTheme(currentTheme() === "light" ? "dark" : "light");
  });

  /* 点空白收起下拉：原 bindControls 的全局 click 已通用化进 ui/dropdown.js
   * （模块级单例监听，createDropdown 时自动挂上），此处不再重复关闭，
   * 只负责交互结束后的 pending 补建与推迟渲染补刷。 */
  function onFlushDeferred() {
    if (domRenderBlocked()) return;
    flushDropdowns();
    flushDeferredRender(store.state, function () {
      onDataChanged();
      if (store.state.tab === "overview") {
        var tab = currentTab();
        if (tab && tab.poll) tab.poll();
      }
    });
  }

  /* 交互结束（选区收起 / 控件失焦）后补一次被推迟的自动渲染 */
  document.addEventListener("selectionchange", onFlushDeferred);
  document.addEventListener("change", onFlushDeferred);
  document.addEventListener("blur", onFlushDeferred, true);

  document.getElementById("tabs").addEventListener("click", function (ev) {
    var btn = ev.target.closest(".tab");
    if (!btn) return;
    switchTab(btn.getAttribute("data-tab"));
  });
}

/* ------------------------------------------------------------ 启动 */
function boot() {
  /* 初始 tab（overview）先行 init+enter：applyTheme 的 onTheme 需要已初始化的 tab */
  activateTab(tabById[store.state.tab]);
  bindGlobalControls();
  initAuthChip();
  refreshDataRootLabel();
  initRootPicker();

  store.loadRuns().then(function (runs) {
    var initial = null;
    try { initial = localStorage.getItem("motus.monitor.runId"); } catch (e) {}
    var found = null;
    if (initial) {
      for (var i = 0; i < runs.length; i++) {
        if (runs[i].run_id === initial) { found = initial; break; }
      }
    }
    if (!found && runs.length) found = runs[0].run_id;
    if (found) {
      store.state.runId = null;
      store.selectRun(found);
    } else {
      /* 无 run：渲染一次空总览（旧 renderOverview()） */
      var tab = currentTab();
      if (tab) tab.refresh();
    }
  });

  loadSyncStatus();

  api.events(onServerMessage, function (st) {
    var el = document.getElementById("conn-status");
    if (st === "live") {
      el.textContent = "实时";
      el.className = "conn-status live";
    } else {
      el.textContent = "重连中";
      el.className = "conn-status retry";
    }
  });

  /* SSE 之外的 5s 兜底轮询 */
  setInterval(scheduleUpdate, 5000);
  /* 同步进行中 2s 高频刷新进度条（空闲时不发请求） */
  setInterval(function () { if (store.state.syncPulling) loadSyncStatus(); }, 2000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
