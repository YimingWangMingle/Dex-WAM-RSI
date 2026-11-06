/* Motus Monitor 前端 —— 总览 tab。
 * 从旧 web/app.js 逐字迁入：renderOverview/进度卡/状态卡/statCard（app.js:754-838）、
 * 系统负载卡 renderSystemCards/refreshSystemLatest（app.js:840-957）、
 * 事件流 renderEvents（app.js:959-981）、run 选择器 renderRunSelector/stateDotEl/run-meta
 * （app.js:324-393，改用 ui/dropdown.js 实现，runSelSig/runSelPending 机制由 dropdown 内置）。
 *
 * 数据联动（在 init 里订阅 store 事件；渲染触发路径与旧代码一致）：
 *   "runs"   → 重建 run 选择器（旧 loadRuns 尾部 renderRunSelector）
 *   "run"    → 选择器 + renderOverview + per-node tail 拉取（旧 selectRun 尾部调用）
 *   "detail" → payload.nodesChanged 时补拉各节点最新记录（旧 refreshDetail 的 changed 分支）
 *   "system" → payload.nodesChanged 时同上（旧 system reader onLines 的 nodesChanged 分支）
 * steps/events/system 到达时的 renderOverview 由 main.js 的 onDataChanged 路由到 refresh()，
 * 本模块不重复订阅（旧 onDataChanged 的 overview 分支）。
 */
import { api } from "../api.js";
import { util } from "../util.js";
import { store } from "../store.js";
import { createDropdown } from "../ui/dropdown.js";
import { domRenderBlocked } from "../ui/guard.js";
import { hbar } from "../ui/bars.js";

/* ------------------------------------------------------------ Run 选择器 */
var runDropdown = null;
var runSelSig = null;  // run 选择器内容签名（run-meta 随选择器内容一起更新）

function stateDotEl(stateName) {
  var s = stateName || "unknown";
  if (["running", "completed", "failed", "initializing"].indexOf(s) < 0) s = "unknown";
  return util.el("span", "state-dot " + s);
}

function renderRunSelector() {
  /* 签名对比：runs 内容没变化就完全不碰 DOM（轮询周期里重建会打断
     页面上的文字选择）；菜单展开期间有变化则推迟到菜单关闭后重建（dropdown 内置）。 */
  var sig = String(store.state.runId) + "#" + store.state.runs.map(function (r) {
    return [r.run_id, r.state, r.run_name, r.last_step, r.max_steps].join(":");
  }).join("|");
  runDropdown.update({ items: store.state.runs, currentId: store.state.runId, signature: sig });
  if (sig === runSelSig) return;
  runSelSig = sig;

  var run = store.currentRun();
  var meta = document.getElementById("run-meta");
  if (run) {
    meta.textContent = "id: " + run.run_id + " · 创建于 " + util.fmtDateTime(run.created_at) +
      " · world_size " + (run.world_size != null ? run.world_size : "—");
  } else {
    meta.textContent = "";
  }
}

/* ------------------------------------------------------------ 总览渲染 */
function latestStep() {
  return store.state.steps.length ? store.state.steps[store.state.steps.length - 1] : null;
}

function avgStepTime(n) {
  var arr = store.state.steps.slice(-(n || 20));
  var sum = 0, cnt = 0;
  for (var i = 0; i < arr.length; i++) {
    var v = arr[i].step_time_seconds;
    if (util.isNum(v)) { sum += v; cnt++; }
  }
  return cnt ? sum / cnt : null;
}

function statCard(label, value, sub, dot) {
  var card = util.el("div", "stat-card");
  card.appendChild(util.el("div", "stat-label", { text: label }));
  var v = util.el("div", "stat-value");
  if (dot) v.appendChild(stateDotEl(dot));
  v.appendChild(util.el("span", null, { text: value }));
  card.appendChild(v);
  if (sub != null) card.appendChild(util.el("div", "stat-sub", { text: sub }));
  return card;
}

function renderOverview() {
  var run = store.currentRun();
  var cards = document.getElementById("status-cards");
  cards.innerHTML = "";
  var last = latestStep();
  var metrics = (last && last.metrics) || {};

  var lastStep = run && run.last_step != null ? run.last_step : (last && last.step);
  var maxSteps = (run && run.max_steps) || (store.state.runDetail && store.state.runDetail.max_steps);

  /* 进度条 + ETA（初始化阶段显示当前阶段与用时） */
  if (run && run.state === "initializing") {
    document.getElementById("progress-label").textContent =
      "初始化中 · " + (run.init_phase || "…");
    document.getElementById("progress-fill").style.width = "0";
    document.getElementById("progress-eta").textContent =
      run.init_elapsed_seconds != null ? "已用时 " + util.fmtDuration(run.init_elapsed_seconds) : "";
  } else {
    var pct = null;
    if (util.isNum(lastStep) && util.isNum(maxSteps) && maxSteps > 0) {
      pct = Math.min(100, (lastStep / maxSteps) * 100);
    }
    document.getElementById("progress-label").textContent =
      "Step " + (lastStep != null ? lastStep : "—") + " / " + (maxSteps != null ? maxSteps : "—") +
      (pct != null ? "（" + pct.toFixed(1) + "%）" : "");
    document.getElementById("progress-fill").style.width = pct != null ? pct + "%" : "0";
    var etaText = "";
    if (run && run.state === "running" && util.isNum(lastStep) && util.isNum(maxSteps) && maxSteps > lastStep) {
      var avg = avgStepTime(20);
      if (util.isNum(avg)) etaText = "ETA 约 " + util.fmtDuration(avg * (maxSteps - lastStep));
    }
    document.getElementById("progress-eta").textContent = etaText;
  }

  /* 状态卡片 */
  cards.appendChild(statCard("状态", run ? (run.state || "unknown") : "—",
    run && run.status_ts ? "更新于 " + util.fmtClock(run.status_ts) : null, run ? run.state : null));
  cards.appendChild(statCard("Epoch", last && last.epoch != null ? String(last.epoch) : "—"));
  cards.appendChild(statCard("Total Loss", util.fmtVal(metrics.total_loss)));
  cards.appendChild(statCard("Video Loss", util.fmtVal(metrics.video_loss)));
  cards.appendChild(statCard("Action Loss", util.fmtVal(metrics.action_loss)));
  cards.appendChild(statCard("Depth Loss", util.fmtVal(metrics.depth_loss)));
  cards.appendChild(statCard("LR", util.fmtVal(last && last.lr_main),
    util.isNum(last && last.lr_wan) ? "wan " + util.fmtVal(last.lr_wan) : null));
  var avgT = avgStepTime(20);
  cards.appendChild(statCard("速度", util.isNum(avgT) ? (1 / avgT).toFixed(2) + " it/s" : "—",
    util.isNum(avgT) ? util.fmtSeconds(avgT) + "/it" : null));
  var seen = null;
  if (last && last.source_seen && typeof last.source_seen === "object") {
    seen = 0;
    Object.keys(last.source_seen).forEach(function (k) {
      if (util.isNum(last.source_seen[k])) seen += last.source_seen[k];
    });
  }
  cards.appendChild(statCard("已见样本", seen != null ? String(seen) : "—"));
  cards.appendChild(statCard("重启次数", run && run.restarts != null ? String(run.restarts) : "—"));

  renderSystemCards();
  renderEvents();
}

function renderSystemCards() {
  var box = document.getElementById("system-cards");
  box.innerHTML = "";
  var nodes = store.state.sysNodes.length ? store.state.sysNodes : Object.keys(store.state.sysLatest);
  if (!nodes.length) {
    box.appendChild(util.el("div", "no-data", { text: "暂无系统负载数据（该 run 未采集 system 流）" }));
    return;
  }
  nodes.forEach(function (node) {
    var rec = store.state.sysLatest[node];
    var card = util.el("div", "node-card");
    card.appendChild(util.el("div", "node-name", { text: node }));
    if (!rec) {
      card.appendChild(util.el("div", "no-data", { text: "暂无数据" }));
      box.appendChild(card);
      return;
    }
    card.appendChild(util.el("div", "node-ts", { text: "采样时间 " + util.fmtClock(rec.ts) }));
    card.appendChild(hbar({
      label: "CPU",
      pct: rec.cpu_percent,
      text: util.isNum(rec.cpu_percent) ? util.fmtPct(rec.cpu_percent) : "暂无数据",
    }));
    var load = rec.loadavg;
    var loadRow = util.el("div", "hbar-row");
    loadRow.appendChild(util.el("span", "hbar-label", { text: "loadavg" }));
    var loadText = util.el("span", "hbar-text");
    loadText.style.textAlign = "left";
    loadText.style.width = "auto";
    loadText.textContent = Array.isArray(load)
      ? load.map(function (x) { return util.isNum(x) ? x.toFixed(2) : "—"; }).join(" / ") +
        (rec.cpu_count ? "（" + rec.cpu_count + " 核）" : "")
      : "暂无数据";
    loadRow.appendChild(loadText);
    card.appendChild(loadRow);

    var mem = rec.memory || {};
    card.appendChild(hbar({
      label: "内存",
      pct: mem.used_percent,
      text: util.isNum(mem.used_bytes) && util.isNum(mem.total_bytes)
        ? util.fmtBytes(mem.used_bytes) + " / " + util.fmtBytes(mem.total_bytes)
        : "暂无数据",
    }));
    var shm = rec.dev_shm || {};
    card.appendChild(hbar({
      label: "/dev/shm",
      pct: shm.used_percent,
      text: util.isNum(shm.used_bytes) && util.isNum(shm.total_bytes)
        ? util.fmtBytes(shm.used_bytes) + " / " + util.fmtBytes(shm.total_bytes)
        : (util.isNum(shm.used_percent) ? util.fmtPct(shm.used_percent) : "暂无数据"),
    }));
    var disk = rec.data_dir_disk || {};
    if (util.isNum(disk.used_percent)) {
      card.appendChild(hbar({
        label: "数据盘",
        pct: disk.used_percent,
        text: util.isNum(disk.used_bytes) && util.isNum(disk.total_bytes)
          ? util.fmtBytes(disk.used_bytes) + " / " + util.fmtBytes(disk.total_bytes)
          : util.fmtPct(disk.used_percent),
      }));
    }
    var gpus = Array.isArray(rec.gpus) ? rec.gpus : [];
    if (gpus.length) {
      var gpuBox = util.el("div", "gpu-block");
      gpus.forEach(function (g) {
        gpuBox.appendChild(util.el("div", "gpu-title", {
          text: "GPU " + (g.index != null ? g.index : "?") +
            (util.isNum(g.temperature_c) ? " · " + Math.round(g.temperature_c) + "°C" : "") +
            (util.isNum(g.power_watts) ? " · " + Math.round(g.power_watts) + "W" : ""),
        }));
        gpuBox.appendChild(hbar({
          label: "利用率",
          pct: g.util_percent,
          text: util.isNum(g.util_percent) ? util.fmtPct(g.util_percent, 0) : "暂无数据",
        }));
        var memPct = util.isNum(g.mem_used_bytes) && util.isNum(g.mem_total_bytes) && g.mem_total_bytes > 0
          ? (g.mem_used_bytes / g.mem_total_bytes) * 100 : null;
        gpuBox.appendChild(hbar({
          label: "显存",
          pct: memPct,
          text: util.isNum(g.mem_used_bytes) && util.isNum(g.mem_total_bytes)
            ? util.fmtBytes(g.mem_used_bytes) + " / " + util.fmtBytes(g.mem_total_bytes)
            : "暂无数据",
        }));
      });
      card.appendChild(gpuBox);
    }
    box.appendChild(card);
  });
}

function refreshSystemLatest() {
  /* 总览卡片：每个节点取一次 tail，拿最新记录 */
  var nodes = store.state.sysNodes;
  if (!store.state.runId || !nodes.length) { renderSystemCards(); return Promise.resolve(); }
  var tasks = nodes.map(function (node) {
    var url = "/api/runs/" + encodeURIComponent(store.state.runId) +
      "/metrics?stream=system&tail=1&tail_bytes=8192&node=" + encodeURIComponent(node);
    return api.get(url).then(function (resp) {
      var data = resp && resp.data;
      if (!data) return;
      var lines = data.split("\n");
      for (var i = lines.length - 1; i >= 0; i--) {
        var line = lines[i].trim();
        if (!line) continue;
        try {
          store.state.sysLatest[node] = JSON.parse(line);
        } catch (e) { /* ignore */ }
        break;
      }
      if (store.state.tab === "overview") {
        if (domRenderBlocked()) { store.state.renderDeferred = true; return; }
        renderSystemCards();
      }
    }).catch(function () {});
  });
  return Promise.all(tasks);
}

function renderEvents() {
  var box = document.getElementById("event-list");
  box.innerHTML = "";
  if (!store.state.events.length) {
    box.appendChild(util.el("div", "no-data", { text: "暂无事件" }));
    return;
  }
  var recent = store.state.events.slice(-30).reverse();
  recent.forEach(function (ev) {
    var row = util.el("div", "event-row");
    row.appendChild(util.el("span", "event-ts", { text: util.fmtClock(ev.ts) }));
    row.appendChild(util.el("span", "event-kind " + util.esc(ev.kind || "event"), { text: ev.kind || "event" }));
    var summary = {};
    Object.keys(ev || {}).forEach(function (k) {
      if (["type", "kind", "ts"].indexOf(k) >= 0) return;
      summary[k] = ev[k];
    });
    var text = JSON.stringify(summary);
    if (text.length > 160) text = text.slice(0, 157) + "…";
    row.appendChild(util.el("span", "event-summary", { text: text, attrs: { title: JSON.stringify(summary, null, 2) } }));
    box.appendChild(row);
  });
}

export default {
  id: "overview",

  init: function () {
    /* run 选择器（单选下拉：按钮 + 菜单，签名机制在组件内） */
    runDropdown = createDropdown({
      getId: function (r) { return r.run_id; },
      renderButton: function (btn, run) {
        if (run) {
          btn.appendChild(stateDotEl(run.state));
          btn.appendChild(util.el("span", "opt-name", { text: run.run_name || run.run_id }));
          btn.appendChild(util.el("span", "opt-step mono", {
            text: (run.last_step != null ? run.last_step : "—") + "/" + (run.max_steps != null ? run.max_steps : "—"),
          }));
        } else {
          btn.appendChild(util.el("span", "opt-name muted", { text: store.state.runs.length ? "请选择 run" : "暂无 run" }));
        }
      },
      renderOption: function (r) {
        var opt = util.el("div");
        opt.appendChild(stateDotEl(r.state));
        opt.appendChild(util.el("span", "opt-name", { text: r.run_name || r.run_id }));
        opt.appendChild(util.el("span", "opt-step", {
          text: (r.last_step != null ? r.last_step : "—") + "/" + (r.max_steps != null ? r.max_steps : "—"),
        }));
        return opt;
      },
      onSelect: function (id) { store.selectRun(id); },
    });
    document.getElementById("run-selector").appendChild(runDropdown.el);
    /* 首次渲染等 "runs" 事件（旧实现：loadRuns 成功后才 renderRunSelector，
     * boot 期间 #run-selector 保持为空） */

    store.on("runs", renderRunSelector);
    store.on("run", function () {
      renderRunSelector();
      renderOverview();
      refreshSystemLatest();
    });
    store.on("detail", function (payload) {
      if (payload && payload.nodesChanged) refreshSystemLatest();
    });
    store.on("system", function (payload) {
      if (payload && payload.nodesChanged) refreshSystemLatest();
    });
  },

  enter: function () {
    renderOverview();
    refreshSystemLatest();
  },

  leave: function () {
    // 无清理
  },

  /* 数据事件到达时的轻量刷新（main.js 已守卫） */
  refresh: function () {
    renderOverview();
  },

  /* doUpdate 周期增量拉取：每节点 system tail（旧 refreshSystemLatest），返回 Promise */
  poll: function () {
    return refreshSystemLatest();
  },

  /* 主题切换后重绘（canvas 取色来自 CSS 变量） */
  onTheme: function () {
    renderOverview();
    refreshSystemLatest();
  },
};
