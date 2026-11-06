/* Motus Monitor 前端 —— 主应用：tab 导航、状态管理、实时刷新。
 * 依赖加载顺序：api.js → charts.js → tree.js → app.js（全部挂在 window.MonApp）。
 */
(function () {
  "use strict";

  var api = MonApp.api;
  var util = MonApp.util;
  var PALETTE = MonApp.PALETTE;

  var STEP_CAP = 20000;      // step 记录缓存上限（超出丢弃最旧）
  var EVENT_CAP = 2000;
  var SYS_CAP = 5000;
  var GPU_RECENT_SECONDS = 3600; // GPU 图"近期"视窗宽度（跟随最新数据）

  var state = {
    tab: "overview",
    runs: [],
    runId: null,
    runDetail: null,
    steps: [],               // metrics 流 step 记录（按 step 升序、按 step 去重）
    stepMap: {},             // step -> 记录（resume 重跑时后写胜出）
    events: [],              // events 流记录
    sysNodes: [],
    sysNode: null,
    sysLatest: {},           // node -> 最新 system 记录（总览卡片）
    sysHistory: {},          // node -> [system 记录]（系统曲线，增量尾部）
    sysFullHistory: {},      // node -> [system 记录]（全量历史，GPU 统计用）
    sysHistLoading: {},      // node -> true（全量 system 历史加载中）
    readers: null,           // {metrics, events, system}
    sourceColors: {},        // source_dataset -> color
    gpuViewMode: "recent",   // GPU 图视窗：recent（默认近期）| full（全部）；用户缩放/拖拽后由 viewTouched 接管
    updating: false,
    renderDeferred: false,   // 因用户选中文本/操作控件而推迟的自动渲染
    runSelSig: null,         // run 选择器内容签名（无变化不重建 DOM）
    runSelPending: false,    // 菜单展开期间推迟的选择器重建
    lastDetailAt: 0,
  };

  /* ------------------------------------------------------------ 指标定义 */
  /* 模态 dropout 步：被 drop 分支的 loss 置 0、total_loss 只剩单分支，都不具
     可比性，曲线里忽略这些点（返回 null 即被跳过）。
     新记录（20260807 起）带 modality_dropped_* 标志，优先用标志判定——可区分
     "本步被 drop" 与 "单分支训练该分量恒为 0"；无标志的老记录只对分量自身退化
     为 0 判断，total_loss 不做猜测（避免误空单分支 run 的整条曲线）。 */
  function modalityDropFlag(r, mod) {
    var m = r.metrics;
    var flag = m && m["modality_dropped_" + mod];
    return util.isNum(flag) ? flag !== 0 : null;
  }
  function droppedModality(r, mod) {
    var flag = modalityDropFlag(r, mod);
    if (flag !== null) return flag;
    var m = r.metrics;
    return !!(m && m[mod + "_loss"] === 0);
  }
  function nonZeroLoss(v) {
    return util.isNum(v) && v !== 0 ? v : null;
  }
  var METRIC_DEFS = [
    { key: "total_loss", label: "total_loss", on: true, get: function (r) {
        if (modalityDropFlag(r, "video") || modalityDropFlag(r, "action")) return null;
        return nonZeroLoss(r.metrics && r.metrics.total_loss);
      } },
    { key: "video_loss", label: "video_loss", on: true, get: function (r) {
        return droppedModality(r, "video") ? null : nonZeroLoss(r.metrics && r.metrics.video_loss);
      } },
    { key: "action_loss", label: "action_loss", on: true, get: function (r) {
        return droppedModality(r, "action") ? null : nonZeroLoss(r.metrics && r.metrics.action_loss);
      } },
    { key: "lr_main", label: "lr_main", on: false, get: function (r) { return r.lr_main; } },
    { key: "lr_wan", label: "lr_wan", on: false, get: function (r) { return r.lr_wan; } },
    { key: "grad_norm", label: "grad_norm", on: false, get: function (r) { return r.grad_total_norm_before_clip; } },
    { key: "step_time", label: "step_time", on: false, get: function (r) { return r.step_time_seconds; } },
    { key: "dataloader", label: "dataloader", on: false, get: function (r) { return r.dataloader_next_seconds; } },
  ];

  /* ------------------------------------------------------------ 耗时检查点（数据驱动）
     检查点集合随训练代码版本变化，不做硬编码清单：
       loop   = step 记录里的 *_seconds 键（非 model_ 前缀）
       model  = model_*_seconds 键（模型/分支/逐 block 检查点）
       custom = stage_seconds.*（MonitorLogger.stage() 自定义检查点）
     已知的"合计"键（loop_total/step_time 等）单独展示、不进堆叠，
     未知新键会自动出现在对应分组里，保证不同版本不会出错。 */
  var TIMING_TOTAL_KEYS = [
    ["loop_total_seconds", "loop_total"],
    ["step_time_seconds", "step_time"],
    ["train_step_total_seconds", "train_step_total"],
    ["model_training_step_seconds", "前向合计"],
    ["model_world_model_forward_seconds", "world_model合计"],
  ];
  var TIMING_GROUP_DEFS = [
    { id: "loop", title: "训练循环" },
    { id: "model", title: "模型内部" },
    { id: "custom", title: "自定义 stage" },
  ];
  var timingColorCache = {};
  var timingScope = null;

  /* 阶段排序：按函数执行先后（而不是耗时大小）。
     已知键在顺序表中给出执行顺序；未知键（含 *_block_XX）挂到其父键之后，
     再按名字字典序兜底——不同版本的检查点集合都能稳定、正确地展示。 */
  var LOOP_STAGE_ORDER = [
    "dataloader_next_seconds", "dataloader_iter_restart_seconds",
    "ready_queue_get_wait_seconds",
    "ready_queue_producer_batch_seconds", "ready_queue_producer_dataloader_seconds",
    "ready_queue_producer_state_age_seconds", "ready_queue_producer_clone_seconds",
    "ready_queue_producer_put_wait_seconds",
    "optimizer_zero_grad_seconds", "host_to_device_seconds",
    "backward_seconds", "grad_clip_seconds",
    "optimizer_step_seconds", "scheduler_step_seconds",
    "validation_seconds",
    "checkpoint_prepare_seconds", "accelerator_save_state_seconds",
    "checkpoint_config_seconds", "checkpoint_marker_seconds", "checkpoint_total_seconds",
  ];
  var MODEL_STAGE_ORDER = [
    "model_ego4d_prompt_seconds", "model_scheduler_sample_seconds",
    "model_video_prepare_seconds", "model_vae_encode_video_seconds",
    "model_noise_and_condition_seconds", "model_text_prepare_seconds",
    "model_wm_t5_seconds",
    "model_wm_video_encode_seconds",
    "model_wm_und_encode_seconds", "model_wm_und_vlm_seconds", "model_wm_und_encoder_seconds",
    "model_wm_action_encode_seconds",
    "model_wm_uni_process_seconds",
    "model_wm_video_decode_seconds",
    "model_wm_action_decode_seconds",
    "model_loss_compute_seconds",
  ];

  /* 父子包含关系：parent ⊇ children（如 action_encode ⊇ action_encode_block_*，
     checkpoint_total ⊇ prepare/save_state/config/marker）。block 循环按命名
     规则自动识别（X_block_NN_seconds 属于 X_seconds）；少量非 block 关系显式
     声明。未知新键自动成为顶层，不会出错；占比/堆叠只统计顶层（避免重复计数）。 */
  var STAGE_CHILD_PARENT = {
    "model_wm_und_vlm_seconds": "model_wm_und_encode_seconds",
    "model_wm_und_encoder_seconds": "model_wm_und_encode_seconds",
    "checkpoint_prepare_seconds": "checkpoint_total_seconds",
    "accelerator_save_state_seconds": "checkpoint_total_seconds",
    "checkpoint_config_seconds": "checkpoint_total_seconds",
    "checkpoint_marker_seconds": "checkpoint_total_seconds",
  };

  function resolveStageTree(stages) {
    var byKey = {};
    stages.forEach(function (s) { byKey[s.key] = s; });
    var tops = [];
    stages.forEach(function (s) {
      var parentKey = STAGE_CHILD_PARENT[s.key] || null;
      if (!parentKey) {
        var m = s.key.match(/^(.*)_block_\d+_seconds$/);
        if (m) {
          if (byKey[m[1] + "_seconds"]) parentKey = m[1] + "_seconds";
          else if (byKey[m[1] + "_process_seconds"]) parentKey = m[1] + "_process_seconds";
        }
      }
      if (parentKey && byKey[parentKey] && parentKey !== s.key) {
        var parent = byKey[parentKey];
        (parent._children = parent._children || []).push(s);
        return;
      }
      tops.push(s);
    });
    return tops;
  }

  function stageOrderIndex(groupId, key) {
    var list = groupId === "model" ? MODEL_STAGE_ORDER : LOOP_STAGE_ORDER;
    var idx = list.indexOf(key);
    if (idx >= 0) return idx;
    var stem = key.replace(/_block_\d+_seconds$/, "");
    if (stem !== key) {
      idx = list.indexOf(stem + "_seconds");
      if (idx >= 0) return idx + 0.5;
      idx = list.indexOf(stem + "_process_seconds");
      if (idx >= 0) return idx + 0.5;
    }
    return list.length + 1;
  }

  function colorForKey(key) {
    if (!timingColorCache[key]) {
      var h = 0;
      for (var i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
      timingColorCache[key] = PALETTE[h % PALETTE.length];
    }
    return timingColorCache[key];
  }

  function isTimingTotalKey(key) {
    for (var i = 0; i < TIMING_TOTAL_KEYS.length; i++) {
      if (TIMING_TOTAL_KEYS[i][0] === key) return true;
    }
    return false;
  }

  function prettyStageLabel(key) {
    var k = key;
    if (k.indexOf("model_") === 0) k = k.slice(6);
    if (k.slice(-8) === "_seconds") k = k.slice(0, -8);
    if (k.indexOf("wm_") === 0) k = k.slice(3);
    return k;
  }

  function meanOfKey(rows, get) {
    var sum = 0, cnt = 0;
    for (var i = 0; i < rows.length; i++) {
      var v = get(rows[i]);
      if (util.isNum(v)) { sum += v; cnt++; }
    }
    return cnt ? sum / cnt : null;
  }

  function discoverTimingGroups() {
    var seen = { loop: {}, model: {}, custom: {} };
    var steps = state.steps;
    for (var i = 0; i < steps.length; i++) {
      var r = steps[i];
      for (var key in r) {
        if (key.slice(-8) !== "_seconds" || isTimingTotalKey(key) || !util.isNum(r[key])) continue;
        if (key.indexOf("model_") === 0) seen.model[key] = true;
        else seen.loop[key] = true;
      }
      var ss = r.stage_seconds;
      if (ss && typeof ss === "object") {
        for (var sk in ss) if (util.isNum(ss[sk])) seen.custom[sk] = true;
      }
    }
    /* 排除"累计计数器"（如 ready_queue_*_wait_total_seconds）：它们随时间单调
     * 不减，不是每步耗时，混进阶段堆叠会掩盖真实分布（如初始化期的队列预热）。
     * 判定：近窗口内值单调不减且末值大于首值。 */
    var monoWin = steps.slice(-50);
    var cumulative = {};
    Object.keys(seen.loop).forEach(function (key) {
      var prev = null, mono = true, cnt = 0, first = null, last = null;
      for (var j = 0; j < monoWin.length; j++) {
        var v = monoWin[j][key];
        if (!util.isNum(v)) continue;
        cnt++;
        if (first === null) first = v;
        if (prev !== null && v < prev - 1e-9) { mono = false; break; }
        prev = v; last = v;
      }
      if (mono && cnt >= 5 && last > first) cumulative[key] = true;
    });
    var recent = steps.slice(-100);
    var groups = [];
    TIMING_GROUP_DEFS.forEach(function (g) {
      var isCustom = g.id === "custom";
      var keys = Object.keys(seen[g.id]).filter(function (k) {
        return !(g.id === "loop" && cumulative[k]);
      }).sort();
      var stages = keys.map(function (key) {
        var get = isCustom
          ? function (r) { return r.stage_seconds && r.stage_seconds[key]; }
          : function (r) { return r[key]; };
        return {
          key: key, label: prettyStageLabel(key), color: colorForKey(key),
          mean: meanOfKey(recent, get), get: get,
        };
      });
      /* 按函数执行顺序排序（自定义组保持键名字典序） */
      if (g.id !== "custom") {
        stages.sort(function (a, b) {
          var oa = stageOrderIndex(g.id, a.key), ob = stageOrderIndex(g.id, b.key);
          if (oa !== ob) return oa - ob;
          return a.key < b.key ? -1 : 1;
        });
      }
      if (stages.length) groups.push({ id: g.id, title: g.title, stages: stages });
    });
    return groups;
  }

  /* ------------------------------------------------------------ 图表实例 */
  var mainChart = null, sourceChart = null, sysChart = null, timingChart = null, compareChart = null, gpuChart = null;

  function ensureCharts() {
    if (!mainChart) {
      mainChart = new MonApp.LineChart(document.getElementById("main-chart"), { xKey: "step" });
      sourceChart = new MonApp.LineChart(document.getElementById("source-chart"), { xKey: "step" });
      sysChart = new MonApp.LineChart(document.getElementById("system-chart"), { xKey: "ts", xIsTime: true });
      timingChart = new MonApp.StackedBars(document.getElementById("timing-chart"), {
        xKey: "step",
        stages: [],
        maxBars: 120,
      });
      compareChart = new MonApp.LineChart(document.getElementById("compare-chart"), { xKey: "step" });
      gpuChart = new MonApp.LineChart(document.getElementById("gpu-chart"), { xKey: "ts", xIsTime: true });
      /* 双击复位 = 回到默认"近期"视窗 */
      gpuChart.onViewReset = function () {
        state.gpuViewMode = "recent";
        renderGpuViewToggle();
        updateGpuViews();
      };
      gpuChart.onViewChange = renderGpuViewToggle;
    }
  }

  /* ------------------------------------------------------------ Tab 导航 */
  function switchTab(name) {
    state.tab = name;
    var tabs = document.querySelectorAll("#tabs .tab");
    tabs.forEach(function (t) { t.classList.toggle("active", t.getAttribute("data-tab") === name); });
    var pages = document.querySelectorAll(".tab-page");
    pages.forEach(function (p) { p.classList.toggle("active", p.id === "tab-" + name); });
    if (name === "curves") { ensureCharts(); updateCurves(); }
    if (name === "timing") { ensureCharts(); updateTiming(); }
    if (name === "compare") { ensureCharts(); renderCompareRuns(); updateCompare(); }
    if (name === "tree") { MonApp.treeTab.refresh(); }
    if (name === "benchmarks") { loadAnnotations(); }
    if (name === "dataset") { MonApp.datasetTab.refresh(); }
    if (name === "eval") { MonApp.evalTab.refresh(); }
    if (name === "overview") { renderOverview(); refreshSystemLatest(); }
  }

  /* ------------------------------------------------------------ Run 选择器 */
  function currentRun() {
    for (var i = 0; i < state.runs.length; i++) {
      if (state.runs[i].run_id === state.runId) return state.runs[i];
    }
    return null;
  }

  function renderRunSelector() {
    var box = document.getElementById("run-selector");
    /* 签名对比：runs 内容没变化就完全不碰 DOM（轮询周期里重建会打断
       页面上的文字选择）；菜单展开期间有变化则推迟到菜单关闭后重建。 */
    var sig = String(state.runId) + "#" + state.runs.map(function (r) {
      return [r.run_id, r.state, r.run_name, r.last_step, r.max_steps].join(":");
    }).join("|");
    if (sig === state.runSelSig) return;
    if (domRenderBlocked() || box.querySelector(".run-select-menu:not(.hidden)")) {
      state.runSelPending = true;
      return;
    }
    state.runSelSig = sig;
    state.runSelPending = false;
    box.innerHTML = "";
    var run = currentRun();

    var btn = util.el("button", "run-select-btn");
    if (run) {
      btn.appendChild(stateDotEl(run.state));
      btn.appendChild(util.el("span", "opt-name", { text: run.run_name || run.run_id }));
      btn.appendChild(util.el("span", "opt-step mono", {
        text: (run.last_step != null ? run.last_step : "—") + "/" + (run.max_steps != null ? run.max_steps : "—"),
      }));
    } else {
      btn.appendChild(util.el("span", "opt-name muted", { text: state.runs.length ? "请选择 run" : "暂无 run" }));
    }
    btn.appendChild(util.el("span", "caret", { text: "▾" }));
    box.appendChild(btn);

    var menu = util.el("div", "run-select-menu hidden");
    state.runs.forEach(function (r) {
      var opt = util.el("div", "run-option");
      opt.appendChild(stateDotEl(r.state));
      opt.appendChild(util.el("span", "opt-name", { text: r.run_name || r.run_id }));
      opt.appendChild(util.el("span", "opt-step", {
        text: (r.last_step != null ? r.last_step : "—") + "/" + (r.max_steps != null ? r.max_steps : "—"),
      }));
      opt.addEventListener("click", function () {
        menu.classList.add("hidden");
        selectRun(r.run_id);
      });
      menu.appendChild(opt);
    });
    box.appendChild(menu);

    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      menu.classList.toggle("hidden");
      if (menu.classList.contains("hidden") && state.runSelPending) {
        state.runSelPending = false;
        renderRunSelector();
      }
    });
    /* 全局只挂一次（见 bindControls）：点击空白处收起下拉 */

    var meta = document.getElementById("run-meta");
    if (run) {
      meta.textContent = "id: " + run.run_id + " · 创建于 " + util.fmtDateTime(run.created_at) +
        " · world_size " + (run.world_size != null ? run.world_size : "—");
    } else {
      meta.textContent = "";
    }
  }

  function stateDotEl(stateName) {
    var s = stateName || "unknown";
    if (["running", "completed", "failed", "initializing"].indexOf(s) < 0) s = "unknown";
    return util.el("span", "state-dot " + s);
  }

  function loadRuns() {
    return api.get("/api/runs").then(function (resp) {
      state.runs = (resp && resp.runs) || [];
      renderRunSelector();
      return state.runs;
    }).catch(function (err) {
      console.error("load runs failed", err);
      return [];
    });
  }

  /* ------------------------------------------------------------ 全量历史加载
   * 选中 run 后异步拉取完整历史（服务端扫描全流、resume 去重、抽稀），
   * 与实时增量流合并（ingestSteps 按 step 去重，重叠处自动一致）。 */
  function loadHistory() {
    var runId = state.runId;
    if (!runId) return;
    state.historyLoading = true;
    renderHistoryInfo();
    api.loadHistory(runId, 3000).then(function (resp) {
      if (state.runId !== runId || !resp) return;
      state.historyLoading = false;
      state.historyInfo = { total: resp.total_records, returned: resp.returned, decimated: !!resp.decimated };
      ingestSteps(resp.records || []);
      renderHistoryInfo();
      onDataChanged();
    }).catch(function (err) {
      if (state.runId !== runId) return;
      state.historyLoading = false;
      state.historyInfo = null;
      renderHistoryInfo();
      console.warn("load history failed", err);
    });
  }

  function renderHistoryInfo() {
    var el = document.getElementById("history-info");
    if (!el) return;
    if (state.historyLoading) {
      el.textContent = "全量历史加载中…";
    } else if (state.historyInfo) {
      var info = state.historyInfo;
      el.textContent = info.decimated
        ? "历史: 全量 " + info.total + " 条（抽稀至 " + info.returned + " 条显示）"
        : "历史: 全量 " + info.total + " 条";
    } else {
      el.textContent = "";
    }
  }

  function selectRun(runId) {
    if (runId === state.runId) return;
    state.runId = runId;
    try { localStorage.setItem("motus.monitor.runId", runId); } catch (e) {}
    state.steps = [];
    state.stepMap = {};
    state.events = [];
    state.sysNodes = [];
    state.sysNode = null;
    state.sysLatest = {};
    state.sysHistory = {};
    state.sysFullHistory = {};
    state.sysHistLoading = {};
    state.sourceColors = {};
    state.runDetail = null;
    state.historyLoading = false;
    state.historyInfo = null;
    /* GPU 图视窗是时间轴：换 run 后回到默认"近期"跟随，避免沿用上个 run 的旧区间 */
    if (gpuChart) { gpuChart.viewTouched = false; gpuChart.view = null; }
    createReaders();
    renderRunSelector();
    renderOverview();
    loadHistory();
    if (state.tab === "curves") updateCurves();
    if (state.tab === "timing") updateTiming();
    refreshDetail();
    refreshSystemLatest();
  }

  /* resume 语义：训练只会从已保存 ckpt 的 step 续跑，被放弃的那段 future
     会在重跑时被重新记录（同一 step 出现两次）。这里按"后写胜出"去重——
     后到的记录（重新执行的）覆盖先到的（被放弃的），图表不会回折。
     state.steps 始终保持按 step 升序。
     另外按谱系基点（lineage_base_step = 本次 resume 的 ckpt step）隐藏
     基点之后的旧记录：它们来自被中断的那次执行，重跑尚未覆盖时不应显示。 */
  function lineageKeep(r) {
    var run = currentRun();
    if (!run) return true;
    var base = run.lineage_base_step;
    if (!util.isNum(base)) return true;
    if (util.isNum(r.step) && r.step <= base) return true;
    var t0 = run.attempt_started_ts;
    if (util.isNum(t0) && util.isNum(r.ts) && r.ts >= t0) return true;
    return false;
  }

  /* system/GPU 流没有 step 概念，只能按时间过滤：从 0 重跑（lineage_base_step
     === 0，训练侧重启且无 resume 时写入）时，本 attempt 之前的记录全部作废；
     resume 续跑（base > 0）保留完整历史曲线。 */
  function sysKeep(r) {
    var run = currentRun();
    if (!run) return true;
    if (run.lineage_base_step !== 0) return true;
    var t0 = run.attempt_started_ts;
    if (!util.isNum(t0)) return true;
    return util.isNum(r.ts) && r.ts >= t0;
  }

  function ingestSteps(records) {
    var map = state.stepMap;
    for (var i = 0; i < records.length; i++) {
      var r = records[i];
      if (!r || (r.type && r.type !== "step")) continue;
      var s = Number(r.step);
      if (!isFinite(s)) continue;
      if (!lineageKeep(r)) continue;
      map[s] = r;
    }
    var keys = Object.keys(map);
    keys.sort(function (a, b) { return Number(a) - Number(b); });
    if (keys.length > STEP_CAP) {
      for (var j = 0; j < keys.length - STEP_CAP; j++) delete map[keys[j]];
      keys = keys.slice(keys.length - STEP_CAP);
    }
    var out = new Array(keys.length);
    for (var k = 0; k < keys.length; k++) out[k] = map[keys[k]];
    state.steps = out;
  }

  function createReaders() {    if (!state.runId) { state.readers = null; return; }
    state.readers = {
      metrics: api.streamReader(state.runId, "metrics", function (records) {
        ingestSteps(records);
        onDataChanged();
      }),
      events: api.streamReader(state.runId, "events", function (records) {
        for (var i = 0; i < records.length; i++) {
          var r = records[i];
          if (!r) continue;
          state.events.push(r);
        }
        if (state.events.length > EVENT_CAP) {
          state.events.splice(0, state.events.length - EVENT_CAP);
        }
        onDataChanged();
      }, { tailBytes: 32768 }),
      system: api.streamReader(state.runId, "system", function (records, reader) {
        var node = reader.node || "_";
        if (!state.sysHistory[node]) state.sysHistory[node] = [];
        var arr = state.sysHistory[node];
        for (var i = 0; i < records.length; i++) {
          if (!records[i] || !sysKeep(records[i])) continue;
          arr.push(records[i]);
          state.sysLatest[node] = records[i];
        }
        if (arr.length > SYS_CAP) state.sysHistory[node] = arr.slice(-SYS_CAP);
        state.sysHistory[node] = arr;
        var nodesChanged = false;
        if (reader.nodes && reader.nodes.join() !== state.sysNodes.join()) {
          state.sysNodes = reader.nodes;
          nodesChanged = true;
        }
        if (!state.sysNode && reader.node) state.sysNode = reader.node;
        /* 节点列表刚到（或变化）时，立刻补齐其余节点的最新记录，
         * 避免总览系统卡等到下一个轮询周期才显示。 */
        if (nodesChanged) refreshSystemLatest();
        onDataChanged();
      }, { tailBytes: 65536 }),
    };
    /* 首次加载：tail 拉最近一段 */
    state.readers.metrics.poll();
    state.readers.events.poll();
    state.readers.system.poll();
  }

  function refreshDetail() {
    if (!state.runId) return Promise.resolve(null);
    return api.getOrNull("/api/runs/" + encodeURIComponent(state.runId)).then(function (d) {
      state.runDetail = d;
      if (d && d.streams && d.streams.system && d.streams.system.nodes) {
        var nodes = d.streams.system.nodes;
        var changed = nodes.join() !== state.sysNodes.join();
        state.sysNodes = nodes;
        if (!state.sysNode && state.sysNodes.length) {
          state.sysNode = state.sysNodes[0];
          if (state.readers) state.readers.system.setNode(state.sysNode);
        }
        if (state.sysNode) loadSystemHistory(state.sysNode);
        if (changed) refreshSystemLatest();
      }
      return d;
    });
  }

  /* ------------------------------------------------------------ GPU 利用率（按卡 + 统计）
   * 数据源：全量 system 历史（/history stream=system，一次性加载）
   * + 增量 system 尾部（实时），按 ts 去重合并。 */
  function loadSystemHistory(node) {
    if (!state.runId || !node || state.sysHistLoading[node]) return;
    if (state.sysFullHistory[node]) return;
    state.sysHistLoading[node] = true;
    renderGpuHint();
    api.loadHistory(state.runId, 3000, { stream: "system", node: node }).then(function (resp) {
      delete state.sysHistLoading[node];
      if (resp && resp.records) state.sysFullHistory[node] = resp.records;
      renderGpuHint();
      if (guardAutoRender()) return;
      updateGpuViews();
    }).catch(function (err) {
      delete state.sysHistLoading[node];
      renderGpuHint();
      console.warn("system history failed", err);
    });
  }

  function renderGpuHint() {
    var el = document.getElementById("gpu-hint");
    if (!el) return;
    var node = state.sysNode;
    if (node && state.sysHistLoading[node]) { el.textContent = "（历史加载中…）"; return; }
    el.textContent = "";
  }

  function renderGpuViewToggle() {
    var btn = document.getElementById("gpu-view-toggle");
    if (!btn) return;
    var custom = gpuChart && gpuChart.viewTouched;
    btn.textContent = custom ? "跟随近期"
      : (state.gpuViewMode === "recent" ? "显示全部" : "仅看近期");
    btn.title = custom
      ? "已手动缩放/平移，点击回到默认近期视窗"
      : (state.gpuViewMode === "recent"
        ? "当前显示最近 " + Math.round(GPU_RECENT_SECONDS / 60) + " 分钟，点击显示全部历史"
        : "当前显示全部历史，点击回到近期视窗");
  }

  function mergedSysRecords(node) {
    var map = {};
    (state.sysFullHistory[node] || []).forEach(function (r) {
      if (util.isNum(r.ts) && sysKeep(r)) map[r.ts.toFixed(3)] = r;
    });
    (state.sysHistory[node] || []).forEach(function (r) {
      if (util.isNum(r.ts) && sysKeep(r)) map[r.ts.toFixed(3)] = r;
    });
    var keys = Object.keys(map);
    keys.sort(function (a, b) { return Number(a) - Number(b); });
    return keys.map(function (k) { return map[k]; });
  }

  function _gpuOf(rec, index) {
    var gs = rec.gpus;
    if (!Array.isArray(gs)) return null;
    for (var i = 0; i < gs.length; i++) if (gs[i].index === index) return gs[i];
    return null;
  }

  function _pct(values, p) {
    if (!values.length) return null;
    var s = values.slice().sort(function (a, b) { return a - b; });
    return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  }

  function updateGpuViews() {
    if (!gpuChart) return;
    var node = state.sysNode;
    var recs = node ? mergedSysRecords(node) : [];
    var indexes = [];
    for (var i = 0; i < recs.length; i++) {
      var gs = recs[i].gpus;
      if (!Array.isArray(gs)) continue;
      for (var j = 0; j < gs.length; j++) {
        if (indexes.indexOf(gs[j].index) < 0) indexes.push(gs[j].index);
      }
    }
    indexes.sort(function (a, b) { return a - b; });

    /* 按卡 util 曲线（卡数过多时只画均值，避免不可读） */
    var series = [];
    if (indexes.length > 0 && indexes.length <= 16) {
      indexes.forEach(function (gi, k) {
        series.push({
          key: "gpu" + gi, label: "gpu" + gi, color: PALETTE[k % PALETTE.length],
          points: recs.map(function (r) {
            var g = _gpuOf(r, gi);
            return (g && util.isNum(g.util_percent)) ? [r.ts, g.util_percent] : null;
          }).filter(Boolean),
        });
      });
    } else if (indexes.length > 16) {
      series.push({
        key: "mean", label: "gpu_util% 均值", color: PALETTE[0],
        points: recs.map(function (r) {
          if (!Array.isArray(r.gpus) || !r.gpus.length) return null;
          var sum = 0, cnt = 0;
          r.gpus.forEach(function (g) { if (util.isNum(g.util_percent)) { sum += g.util_percent; cnt++; } });
          return cnt ? [r.ts, sum / cnt] : null;
        }).filter(Boolean),
      });
    }
    gpuChart.setSeries(series);

    /* 视窗：默认"近期"（跟随最新 ts）；用户缩放/拖拽过（viewTouched）则不干预 */
    if (!gpuChart.viewTouched) {
      if (state.gpuViewMode === "recent" && recs.length) {
        var maxTs = recs[recs.length - 1].ts;
        gpuChart.view = [maxTs - GPU_RECENT_SECONDS, maxTs];
      } else {
        gpuChart.view = null;
      }
      gpuChart.redraw();
    }
    renderGpuViewToggle();

    /* 统计表：每卡 平均/P5/最大 util、平均显存、平均功耗 + 全体均值行 */
    var box = document.getElementById("gpu-stats");
    box.innerHTML = "";
    var hint = document.getElementById("gpu-hint");
    if (!indexes.length) {
      box.appendChild(util.el("div", "no-data", { text: "暂无 GPU 数据" }));
      if (hint && !state.sysHistLoading[node]) hint.textContent = "";
      return;
    }
    var html = "<thead><tr><th>GPU</th><th>平均 util</th><th>P5 util</th><th>最大 util</th>" +
      "<th>平均显存</th><th>平均功耗</th><th>样本</th></tr></thead><tbody>";
    var allUtils = [];
    indexes.forEach(function (gi) {
      var utils = [], mems = [], powers = [];
      recs.forEach(function (r) {
        var g = _gpuOf(r, gi);
        if (!g) return;
        if (util.isNum(g.util_percent)) utils.push(g.util_percent);
        if (util.isNum(g.mem_used_bytes) && util.isNum(g.mem_total_bytes) && g.mem_total_bytes > 0) {
          mems.push(100 * g.mem_used_bytes / g.mem_total_bytes);
        }
        if (util.isNum(g.power_watts)) powers.push(g.power_watts);
      });
      var mean = utils.length ? utils.reduce(function (a, b) { return a + b; }, 0) / utils.length : null;
      if (mean != null) allUtils.push(mean);
      var memMean = mems.length ? mems.reduce(function (a, b) { return a + b; }, 0) / mems.length : null;
      var powMean = powers.length ? powers.reduce(function (a, b) { return a + b; }, 0) / powers.length : null;
      html += "<tr><td>gpu" + gi + "</td><td>" + (mean != null ? mean.toFixed(1) + "%" : "—") +
        "</td><td>" + (utils.length ? _pct(utils, 0.05).toFixed(1) + "%" : "—") +
        "</td><td>" + (utils.length ? _pct(utils, 1).toFixed(1) + "%" : "—") +
        "</td><td>" + (memMean != null ? memMean.toFixed(1) + "%" : "—") +
        "</td><td>" + (powMean != null ? Math.round(powMean) + "W" : "—") +
        "</td><td>" + utils.length + "</td></tr>";
    });
    if (allUtils.length) {
      var allMean = allUtils.reduce(function (a, b) { return a + b; }, 0) / allUtils.length;
      html += "<tr class='total-row'><td>全体均值</td><td>" + allMean.toFixed(1) + "%</td>" +
        "<td colspan='4'></td><td>" + indexes.length + " 卡</td></tr>";
      if (hint) hint.textContent = " · " + indexes.length + " 卡平均 util " + allMean.toFixed(1) + "%";
    }
    html += "</tbody>";
    var table = util.el("table", "data-table");
    table.innerHTML = html;
    box.appendChild(table);
  }

  /* ------------------------------------------------------------ 总览渲染 */
  function latestStep() {
    return state.steps.length ? state.steps[state.steps.length - 1] : null;
  }

  function avgStepTime(n) {
    var arr = state.steps.slice(-(n || 20));
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
    var run = currentRun();
    var cards = document.getElementById("status-cards");
    cards.innerHTML = "";
    var last = latestStep();
    var metrics = (last && last.metrics) || {};

    var lastStep = run && run.last_step != null ? run.last_step : (last && last.step);
    var maxSteps = (run && run.max_steps) || (state.runDetail && state.runDetail.max_steps);

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
    var nodes = state.sysNodes.length ? state.sysNodes : Object.keys(state.sysLatest);
    if (!nodes.length) {
      box.appendChild(util.el("div", "no-data", { text: "暂无系统负载数据（该 run 未采集 system 流）" }));
      return;
    }
    nodes.forEach(function (node) {
      var rec = state.sysLatest[node];
      var card = util.el("div", "node-card");
      card.appendChild(util.el("div", "node-name", { text: node }));
      if (!rec) {
        card.appendChild(util.el("div", "no-data", { text: "暂无数据" }));
        box.appendChild(card);
        return;
      }
      card.appendChild(util.el("div", "node-ts", { text: "采样时间 " + util.fmtClock(rec.ts) }));
      card.appendChild(MonApp.hbar({
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
      card.appendChild(MonApp.hbar({
        label: "内存",
        pct: mem.used_percent,
        text: util.isNum(mem.used_bytes) && util.isNum(mem.total_bytes)
          ? util.fmtBytes(mem.used_bytes) + " / " + util.fmtBytes(mem.total_bytes)
          : "暂无数据",
      }));
      var shm = rec.dev_shm || {};
      card.appendChild(MonApp.hbar({
        label: "/dev/shm",
        pct: shm.used_percent,
        text: util.isNum(shm.used_bytes) && util.isNum(shm.total_bytes)
          ? util.fmtBytes(shm.used_bytes) + " / " + util.fmtBytes(shm.total_bytes)
          : (util.isNum(shm.used_percent) ? util.fmtPct(shm.used_percent) : "暂无数据"),
      }));
      var disk = rec.data_dir_disk || {};
      if (util.isNum(disk.used_percent)) {
        card.appendChild(MonApp.hbar({
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
          gpuBox.appendChild(MonApp.hbar({
            label: "利用率",
            pct: g.util_percent,
            text: util.isNum(g.util_percent) ? util.fmtPct(g.util_percent, 0) : "暂无数据",
          }));
          var memPct = util.isNum(g.mem_used_bytes) && util.isNum(g.mem_total_bytes) && g.mem_total_bytes > 0
            ? (g.mem_used_bytes / g.mem_total_bytes) * 100 : null;
          gpuBox.appendChild(MonApp.hbar({
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
    var nodes = state.sysNodes;
    if (!state.runId || !nodes.length) { renderSystemCards(); return; }
    nodes.forEach(function (node) {
      var url = "/api/runs/" + encodeURIComponent(state.runId) +
        "/metrics?stream=system&tail=1&tail_bytes=8192&node=" + encodeURIComponent(node);
      api.get(url).then(function (resp) {
        var data = resp && resp.data;
        if (!data) return;
        var lines = data.split("\n");
        for (var i = lines.length - 1; i >= 0; i--) {
          var line = lines[i].trim();
          if (!line) continue;
          try {
            state.sysLatest[node] = JSON.parse(line);
          } catch (e) { /* ignore */ }
          break;
        }
        if (state.tab === "overview") {
          if (domRenderBlocked()) { state.renderDeferred = true; return; }
          renderSystemCards();
        }
      }).catch(function () {});
    });
  }

  function renderEvents() {
    var box = document.getElementById("event-list");
    box.innerHTML = "";
    if (!state.events.length) {
      box.appendChild(util.el("div", "no-data", { text: "暂无事件" }));
      return;
    }
    var recent = state.events.slice(-30).reverse();
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

  /* ------------------------------------------------------------ 曲线页 */
  function renderMetricChecks() {
    var box = document.getElementById("metric-checks");
    box.innerHTML = "";
    METRIC_DEFS.forEach(function (def, i) {
      var color = PALETTE[i % PALETTE.length];
      var chip = util.el("label", "metric-check" + (def.on ? " on" : ""));
      var input = util.el("input");
      input.type = "checkbox";
      input.checked = def.on;
      var dot = util.el("span", "chip");
      dot.style.background = color;
      dot.style.color = color;
      chip.appendChild(input);
      chip.appendChild(dot);
      chip.appendChild(util.el("span", null, { text: def.label }));
      chip.addEventListener("click", function (ev) {
        ev.preventDefault();
        def.on = !def.on;
        chip.classList.toggle("on", def.on);
        input.checked = def.on;
        updateCurves();
      });
      box.appendChild(chip);
    });
  }

  function updateCurves() {
    ensureCharts();
    /* 主图 */
    var series = [];
    METRIC_DEFS.forEach(function (def, i) {
      if (!def.on) return;
      series.push({
        key: def.key,
        label: def.label,
        color: PALETTE[i % PALETTE.length],
        value: def.get,
      });
    });
    mainChart.setSeries(series);
    mainChart.setData(state.steps);

    /* 按数据源拆分 */
    var sources = {};
    for (var i = 0; i < state.steps.length; i++) {
      var s = state.steps[i].source_dataset;
      if (s) sources[s] = true;
    }
    var names = Object.keys(sources).sort();
    names.forEach(function (name, idx) {
      if (!state.sourceColors[name]) {
        state.sourceColors[name] = PALETTE[idx % PALETTE.length];
      }
    });
    var srcSeries = names.map(function (name) {
      return {
        key: "src_" + name,
        label: name,
        color: state.sourceColors[name],
        scatter: true,
        emaAlpha: 0.9,
        value: function (r) {
          if (r.source_dataset !== name) return null;
          return METRIC_DEFS[0].get(r);   // total_loss（含 dropout 步过滤）
        },
      };
    });
    sourceChart.setSeries(srcSeries);
    sourceChart.setData(state.steps);

    /* 系统曲线（当前节点） */
    renderSysNodeTabs();
    var hist = state.sysHistory[state.sysNode] || [];
    sysChart.setSeries([
      { key: "cpu", label: "cpu%", color: PALETTE[0], value: function (r) { return r.cpu_percent; } },
      { key: "mem", label: "mem%", color: PALETTE[1], value: function (r) { return r.memory && r.memory.used_percent; } },
      {
        key: "gpu", label: "gpu_util%", color: PALETTE[3],
        value: function (r) {
          if (!Array.isArray(r.gpus) || !r.gpus.length) return null;
          var sum = 0, cnt = 0;
          r.gpus.forEach(function (g) { if (util.isNum(g.util_percent)) { sum += g.util_percent; cnt++; } });
          return cnt ? sum / cnt : null;
        },
      },
    ]);
    sysChart.setData(hist);

    /* GPU 利用率（按卡）与统计表 */
    updateGpuViews();
  }

  function renderSysNodeTabs() {
    var box = document.getElementById("sys-node-tabs");
    box.innerHTML = "";
    if (!state.sysNodes.length) {
      box.appendChild(util.el("span", "muted", { text: "（无 system 流）" }));
      return;
    }
    state.sysNodes.forEach(function (node) {
      var b = util.el("button", "node-tab" + (node === state.sysNode ? " active" : ""), { text: node });
      b.addEventListener("click", function () {
        if (node === state.sysNode) return;
        state.sysNode = node;
        if (state.readers) {
          state.readers.system.setNode(node);
          state.readers.system.poll();
        }
        loadSystemHistory(node);
        updateCurves();
      });
      box.appendChild(b);
    });
  }

  /* ------------------------------------------------------------ 耗时页 */
  function renderTimingScopePills(groups) {
    var box = document.getElementById("timing-scope");
    box.innerHTML = "";
    groups.forEach(function (g) {
      var pill = util.el("span", "node-tab" + (g.id === timingScope ? " active" : ""), {
        text: g.title + " (" + g.stages.length + ")",
      });
      pill.addEventListener("click", function () {
        timingScope = g.id;
        updateTiming();
      });
      box.appendChild(pill);
    });
  }

  function updateTiming() {
    ensureCharts();
    var table = document.getElementById("timing-table");
    var shareBox = document.getElementById("share-bar");
    var groups = discoverTimingGroups();

    /* 初始化阶段组（来自 run.json 的 init_phases，仅发生一次；
       与"训练循环 / 模型内部(前向)"共同构成 初始化→loop→前向 三分部。
       过滤 startup.total_*：那是全部阶段的总和，不能与单项同级展示） */
    var initPhases = ((state.runDetail && state.runDetail.init_phases) || []).filter(function (p) {
      return !String(p.name || "").startsWith("startup.total");
    });
    if (initPhases.length) {
      groups.unshift({
        id: "init",
        title: "初始化",
        stages: initPhases.map(function (p, i) {
          return {
            key: "init_phase_" + i,
            label: p.name,
            color: colorForKey("init_" + String(p.name)),
            mean: util.isNum(p.seconds) ? p.seconds : null,
            get: function () { return null; },
          };
        }),
      });
    }

    /* scope：保留当前选择；失效（新版本没有该分组）时回退到第一个有数据的分组 */
    var active = null;
    groups.forEach(function (g) { if (g.id === timingScope) active = g; });
    if (!active) {
      active = groups.length ? groups[0] : null;
      timingScope = active ? active.id : null;
    }
    renderTimingScopePills(groups);

    /* 合计行（不堆叠，仅参考；随数据自动出现/消失） */
    var totalsBox = document.getElementById("timing-totals");
    var recent = state.steps.slice(-100);
    var parts = [];
    TIMING_TOTAL_KEYS.forEach(function (pair) {
      var mean = meanOfKey(recent, function (r) { return r[pair[0]]; });
      if (mean != null) parts.push(pair[1] + " " + util.fmtSeconds(mean));
    });
    totalsBox.textContent = parts.length ? "最近100步均值 · " + parts.join(" · ") : "";

    shareBox.innerHTML = "";
    if (!active) {
      table.innerHTML = "<tbody><tr><td class='muted'>暂无耗时检查点数据</td></tr></tbody>";
      shareBox.appendChild(util.el("div", "no-data", { text: "暂无数据" }));
      timingChart.setStages([]);
      timingChart.setData([]);
      return;
    }

    /* 层级：占比/堆叠只统计顶层（子项并列会重复计数）；子项缩进展示占父级比例 */
    var tops = resolveStageTree(active.stages);
    var sum = 0;
    tops.forEach(function (s) { if (util.isNum(s.mean)) sum += s.mean; });
    var html = "<thead><tr><th>阶段（" + util.esc(active.title) + "）</th><th>均值</th><th>占比</th></tr></thead><tbody>";
    tops.forEach(function (s) {
      var share = util.isNum(s.mean) && sum > 0 ? (s.mean / sum) * 100 : null;
      var kids = s._children || [];
      var toggle = kids.length
        ? "<span class='parent-toggle' data-key='" + util.esc(s.key) + "'>▾</span>"
        : "<span class='parent-toggle-spacer'></span>";
      html += "<tr class='top-row'><td>" + toggle +
        "<span class='stage-chip' style='background:" + s.color + "'></span>" +
        util.esc(s.label) + "</td><td>" + util.esc(util.fmtSeconds(s.mean)) + "</td><td>" +
        (share != null ? share.toFixed(1) + "%" : "—") + "</td></tr>";
      kids.forEach(function (c) {
        var cshare = util.isNum(c.mean) && util.isNum(s.mean) && s.mean > 0
          ? (c.mean / s.mean) * 100 : null;
        html += "<tr class='child-row' data-parent='" + util.esc(s.key) + "'><td>" +
          "<span class='child-marker'>↳</span>" +
          "<span class='stage-chip' style='background:" + c.color + "'></span>" +
          util.esc(c.label) + "</td><td>" + util.esc(util.fmtSeconds(c.mean)) + "</td><td>" +
          (cshare != null ? cshare.toFixed(1) + "% <span class='muted'>of 父级</span>" : "—") +
          "</td></tr>";
      });
    });
    html += "</tbody>";
    table.innerHTML = html;

    /* 父级折叠/展开 */
    table.querySelectorAll(".parent-toggle").forEach(function (t) {
      t.addEventListener("click", function () {
        var key = t.getAttribute("data-key");
        var collapsed = t.textContent === "▾";
        t.textContent = collapsed ? "▸" : "▾";
        table.querySelectorAll(".child-row[data-parent='" + key + "']").forEach(function (row) {
          row.classList.toggle("hidden", collapsed);
        });
      });
    });

    /* 100% 堆叠条（仅顶层） */
    if (sum > 0) {
      shareBox.appendChild(MonApp.shareBar(tops.map(function (s) {
        return { label: s.label, value: s.mean || 0, color: s.color };
      })));
    } else {
      shareBox.appendChild(util.el("div", "no-data", { text: "暂无数据" }));
    }

    /* 堆叠柱图（仅顶层，系列随当前分组动态切换） */
    timingChart.setStages(tops.map(function (s) {
      return { key: s.key, label: s.label, color: s.color, get: s.get };
    }));
    timingChart.setData(state.steps.slice(-360));
  }

  /* ------------------------------------------------------------ 对比页（多 run 叠加）
   * 每个被选 run 的全量历史经 /history 端点加载并缓存；
   * 指标为单选，系列 = run × 当前指标（独立点集）。 */
  var cmp = {
    selected: {},    // run_id -> color
    metric: "total_loss",
    histories: {},   // run_id -> records
    loading: {},     // run_id -> true
  };

  function cmpMetricDef() {
    for (var i = 0; i < METRIC_DEFS.length; i++) {
      if (METRIC_DEFS[i].key === cmp.metric) return METRIC_DEFS[i];
    }
    return METRIC_DEFS[0];
  }

  function cmpRunColor(idx) { return PALETTE[idx % PALETTE.length]; }

  function renderCompareMetricChecks() {
    var box = document.getElementById("compare-metric-checks");
    box.innerHTML = "";
    METRIC_DEFS.forEach(function (m) {
      var chip = util.el("span", "metric-check" + (m.key === cmp.metric ? " on" : ""), { text: m.label });
      chip.addEventListener("click", function () {
        cmp.metric = m.key;
        renderCompareMetricChecks();
        updateCompare();
      });
      box.appendChild(chip);
    });
  }

  function renderCompareRuns() {
    var box = document.getElementById("compare-run-list");
    /* 首次进入：默认勾选当前选中的 run */
    if (!Object.keys(cmp.selected).length && state.runId) {
      for (var i = 0; i < state.runs.length; i++) {
        if (state.runs[i].run_id === state.runId) {
          cmp.selected[state.runId] = cmpRunColor(i);
          break;
        }
      }
    }
    box.innerHTML = "";
    if (!state.runs.length) {
      box.appendChild(util.el("div", "no-data", { text: "暂无 run" }));
      return;
    }
    state.runs.forEach(function (r, idx) {
      var selected = cmp.selected.hasOwnProperty(r.run_id);
      var item = util.el("div", "compare-run-item" + (selected ? " selected" : ""));
      var color = cmpRunColor(idx);
      item.appendChild(util.el("span", "cmp-color", { attrs: { style: "background:" + (selected ? cmp.selected[r.run_id] : "var(--border)") } }));
      item.appendChild(stateDotEl(r.state));
      item.appendChild(util.el("span", "cmp-name", { text: r.run_name || r.run_id, attrs: { title: r.run_id } }));
      item.appendChild(util.el("span", "cmp-steps", {
        text: (r.last_step != null ? r.last_step : "—") + "/" + (r.max_steps != null ? r.max_steps : "—"),
      }));
      item.addEventListener("click", function () {
        if (cmp.selected.hasOwnProperty(r.run_id)) delete cmp.selected[r.run_id];
        else cmp.selected[r.run_id] = color;
        renderCompareRuns();
        updateCompare();
      });
      box.appendChild(item);
    });
  }

  function loadCompareHistory(runId) {
    if (cmp.loading[runId]) return;
    cmp.loading[runId] = true;
    renderCompareHint();
    api.loadHistory(runId, 3000).then(function (resp) {
      delete cmp.loading[runId];
      if (resp && resp.records) cmp.histories[runId] = resp.records;
      updateCompare();
    }).catch(function (err) {
      delete cmp.loading[runId];
      console.warn("compare history failed", runId, err);
      updateCompare();
    });
  }

  function renderCompareHint() {
    var el = document.getElementById("compare-hint");
    var n = Object.keys(cmp.loading).length;
    el.textContent = n ? "（加载历史中… " + n + "）" : "";
  }

  function updateCompare() {
    renderCompareHint();
    var def = cmpMetricDef();
    var series = [];
    Object.keys(cmp.selected).forEach(function (runId) {
      var recs = cmp.histories[runId];
      if (!recs) {
        loadCompareHistory(runId);
        return;
      }
      var pts = [];
      for (var i = 0; i < recs.length; i++) {
        var r = recs[i];
        var v = def.get(r);
        if (util.isNum(r.step) && util.isNum(v)) pts.push([r.step, v]);
      }
      var name = runId.replace(/__.*/, "");
      series.push({ key: runId, label: name + " · " + def.label, color: cmp.selected[runId], points: pts });
    });
    compareChart.setSeries(series);
  }

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
    switchTab("tree");
    MonApp.treeTab.selectCheckpoint(key);
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

  /* ------------------------------------------------------------ 数据变更 */
  /* 自动渲染守卫：用户正在选择文字、或焦点在表单控件（下拉/输入框）上时，
     周期刷新造成的 DOM 重建会清掉选区/收起下拉。此时推迟渲染，待交互结束
     （selectionchange/change/blur，见 bindControls）后补一次。 */
  function domRenderBlocked() {
    try {
      var sel = window.getSelection && window.getSelection();
      if (sel && !sel.isCollapsed && String(sel).length) return true;
    } catch (e) {}
    /* 焦点在下拉/文本域上 = 用户正在操作（SELECT 展开时重建选项会收起它）。
       不含 INPUT：滑块/复选框点击后长期持有焦点，误伤会导致图表停止跟随。 */
    var ae = document.activeElement;
    if (ae && /^(SELECT|TEXTAREA)$/.test(ae.tagName || "")) return true;
    return false;
  }

  /* 返回 true = 本次自动渲染已推迟 */
  function guardAutoRender() {
    if (!domRenderBlocked()) return false;
    state.renderDeferred = true;
    return true;
  }

  function flushDeferredRender() {
    if (domRenderBlocked()) return;
    if (state.runSelPending) {
      state.runSelPending = false;
      renderRunSelector();
    }
    if (!state.renderDeferred) return;
    state.renderDeferred = false;
    onDataChanged();
    if (state.tab === "overview") refreshSystemLatest();
  }

  /* 供 eval.js 等其他 tab 复用同一守卫 */
  MonApp.domRenderBlocked = domRenderBlocked;

  function onDataChanged() {
    if (guardAutoRender()) return;
    if (state.tab === "overview") {
      /* 状态卡依赖 runs 列表（state/last_step），轻量刷新 */
      renderOverview();
    } else if (state.tab === "curves") {
      updateCurves();
    } else if (state.tab === "timing") {
      updateTiming();
    }
  }

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
    if (state.updating) return;
    state.updating = true;
    var tasks = [loadRuns(), refreshDetail(), loadSyncStatus()];
    if (state.readers) {
      tasks.push(state.readers.metrics.poll());
      tasks.push(state.readers.events.poll());
      tasks.push(state.readers.system.poll());
    }
    /* 评估 tab 可见时随同一 SSE/兜底轮询周期增量拉 eval 流 */
    if (state.tab === "eval" && MonApp.evalTab) {
      tasks.push(MonApp.evalTab.poll());
    }
    Promise.all(tasks).then(function () {
      state.updating = false;
      if (state.tab === "overview") {
        if (guardAutoRender()) return;
        renderOverview();
        refreshSystemLatest();
      }
    }).catch(function () {
      state.updating = false;
    });
  }

  function onServerMessage(msg) {
    if (!msg) return;
    if (msg.type === "update") {
      scheduleUpdate();
    } else if (msg.type === "annotation") {
      if (state.tab === "tree") MonApp.treeTab.onAnnotationChanged();
      if (state.tab === "benchmarks") loadAnnotations();
    }
  }

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
      state.syncPulling = !!active;
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

  /* ------------------------------------------------------------ 查看路径 */
  function refreshDataRootLabel() {
    return api.getOrNull("/api/data_root").then(function (info) {
      var btn = document.getElementById("data-root-btn");
      if (!info || !info.data_root) { btn.textContent = "…"; return; }
      btn.textContent = "📂 查看目录: " + info.data_root;
      var tips = "监控查看路径（点击切换）\n当前: " + info.data_root;
      if (info.startup_data_root && info.startup_data_root !== info.data_root) {
        tips += "\n启动目录: " + info.startup_data_root;
      }
      if ((info.sync_local_dirs || []).length) {
        tips += "\n同步写入: " + info.sync_local_dirs.join(", ");
      }
      btn.title = tips;
    });
  }

  function initRootPicker() {
    var overlay = document.getElementById("root-picker");
    var listEl = document.getElementById("root-picker-list");
    var curEl = document.getElementById("root-picker-cur");
    var hintEl = document.getElementById("root-picker-hint");
    var descEl = document.getElementById("root-picker-desc");
    var chooseBtn = document.getElementById("root-picker-choose");
    var modeLocalBtn = document.getElementById("root-picker-mode-local");
    var modeRemoteBtn = document.getElementById("root-picker-mode-remote");
    var current = null;
    var mode = "local";           /* local | remote（经 SSH 浏览 H-Server 等远端） */
    var remoteName = null;        /* 参考 ssh 远端（复用其 target/ssh_args） */
    var sshRemotes = [];

    var DESC_LOCAL = "切换查看根 = 选择查看哪个监控数据目录下的 run 记录（只读）。只看你自己 workspace 的视图，不影响其他用户、各同步任务的写入目录与训练。";
    var DESC_REMOTE = "远端模式 = 经 SSH 浏览远端（H-Server）的目录；选中后把该目录加入你自己的 workspace 同步（与其他用户互不影响），查看根随之切换，数据拉取到本机镜像后 run 自动出现。";

    function render(info) {
      current = info.path;
      curEl.textContent = (mode === "remote" ? "[远端] " : "") + info.path;
      document.getElementById("root-picker-up").disabled = !info.parent;
      listEl.innerHTML = "";
      var dirs = info.dirs || [];
      if (!dirs.length) {
        listEl.appendChild(util.el("div", "root-picker-empty", { text: "（没有子目录）" }));
      }
      dirs.forEach(function (d) {
        var item = util.el("div", "root-picker-item");
        item.appendChild(util.el("span", "dir-icon", { text: "▸" }));
        item.appendChild(util.el("span", null, { text: d.name + "/" }));
        if (d.has_runs_dir) item.appendChild(util.el("span", "dir-tag", { text: "含 runs/" }));
        item.addEventListener("click", function () {
          browse(info.path.replace(/\/+$/, "") + "/" + d.name);
        });
        listEl.appendChild(item);
      });
      hintEl.textContent = info.looks_like_data_root
        ? (mode === "remote" ? "该远端目录含 runs/，可加入同步" : "该目录含 runs/，可作为监控数据根")
        : "";
    }

    function browse(path) {
      hintEl.textContent = mode === "remote" ? "远端加载中（SSH）…" : "加载中…";
      var url = "/api/browse?path=" + encodeURIComponent(path);
      if (mode === "remote") url += "&remote=" + encodeURIComponent(remoteName || "1");
      api.get(url)
        .then(render)
        .catch(function (err) { hintEl.textContent = err.message || "无法读取该目录"; });
    }

    /* 本机：切换查看根（整页重载最干净：各 tab 缓存与选中 run 都基于旧根） */
    function switchTo(path) {
      hintEl.textContent = "切换到 " + path + " …";
      api.post("/api/data_root", { path: path }).then(function () {
        window.location.reload();
      }).catch(function (err) {
        hintEl.textContent = err.message || "切换失败";
      });
    }

    /* 远端：把该目录加为自己 workspace 的同步远端（查看根随之切换） */
    function addRemote(path) {
      hintEl.textContent = "加入同步 " + path + " …";
      api.post("/api/remotes", { path: path, remote: remoteName }).then(function (resp) {
        hintEl.textContent = (resp && resp.note) || "已加入同步。";
        /* 服务端已把本用户查看根切到 workspace 数据目录：刷新顶栏标签 */
        if (resp && !resp.already) refreshDataRootLabel();
      }).catch(function (err) {
        hintEl.textContent = err.message || "加入同步失败";
      });
    }

    function knownItem(r, onClick) {
      var item = util.el("div", "root-picker-known-item");
      item.appendChild(util.el("span", "kr-path", { text: r.path }));
      var meta = (r.run_count || 0) + " runs";
      if (r.latest_ts) meta += " · 最近活跃 " + util.fmtAgo(r.latest_ts);
      item.appendChild(util.el("span", "kr-meta", { text: meta }));
      item.addEventListener("click", onClick);
      return item;
    }

    function loadKnownRoots() {
      var localWrap = document.getElementById("root-picker-known");
      var localList = document.getElementById("root-picker-known-list");
      var remoteWrap = document.getElementById("root-picker-remote-known");
      var remoteList = document.getElementById("root-picker-remote-known-list");
      api.getOrNull("/api/known_roots").then(function (info) {
        var roots = (info && info.roots) || [];
        localWrap.classList.toggle("hidden", !roots.length);
        localList.innerHTML = "";
        roots.forEach(function (r) {
          localList.appendChild(knownItem(r, function () { switchTo(r.path); }));
        });
        var remoteRoots = (info && info.remote_roots) || [];
        remoteWrap.classList.toggle("hidden", !remoteRoots.length);
        remoteList.innerHTML = "";
        remoteRoots.forEach(function (r) {
          remoteList.appendChild(knownItem(r, function () { addRemote(r.path); }));
        });
      });
    }

    function setMode(next) {
      mode = next;
      modeLocalBtn.classList.toggle("active", mode === "local");
      modeRemoteBtn.classList.toggle("active", mode === "remote");
      descEl.textContent = mode === "remote" ? DESC_REMOTE : DESC_LOCAL;
      chooseBtn.textContent = mode === "remote" ? "加入同步" : "选择此目录";
      document.getElementById("root-picker-known").classList.toggle("hidden", mode !== "local");
      document.getElementById("root-picker-remote-known").classList.toggle("hidden", mode !== "remote");
      loadKnownRoots();
      if (mode === "remote") {
        api.getOrNull("/api/remotes").then(function (info) {
          var first = ((info && info.remotes) || []).filter(function (r) { return r.ssh; })[0];
          browse((first && first.remote_dir) || "/");
        });
      } else {
        api.getOrNull("/api/data_root").then(function (info) {
          browse((info && info.data_root) || "/");
        });
      }
    }

    document.getElementById("data-root-btn").addEventListener("click", function () {
      overlay.classList.remove("hidden");
      api.getOrNull("/api/remotes").then(function (info) {
        sshRemotes = ((info && info.remotes) || []).filter(function (r) { return r.ssh; });
        remoteName = sshRemotes.length ? sshRemotes[0].name : null;
        /* 有 ssh 远端时默认远端模式（数据在 H-Server 上），否则只能本机 */
        modeRemoteBtn.classList.toggle("hidden", !sshRemotes.length);
        setMode(sshRemotes.length ? "remote" : "local");
      });
    });
    modeLocalBtn.addEventListener("click", function () { setMode("local"); });
    modeRemoteBtn.addEventListener("click", function () { setMode("remote"); });
    document.getElementById("root-picker-close").addEventListener("click", function () {
      overlay.classList.add("hidden");
    });
    overlay.addEventListener("click", function (ev) {
      if (ev.target === overlay) overlay.classList.add("hidden");
    });
    document.getElementById("root-picker-up").addEventListener("click", function () {
      var url = "/api/browse?path=" + encodeURIComponent(current || "/");
      if (mode === "remote") url += "&remote=" + encodeURIComponent(remoteName || "1");
      api.get(url).then(function (info) {
        if (info.parent) browse(info.parent);
      });
    });
    chooseBtn.addEventListener("click", function () {
      if (!current) return;
      if (mode === "remote") addRemote(current);
      else switchTo(current);
    });
  }

  /* ------------------------------------------------------------ 启动 */
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
    /* canvas 取色来自 CSS 变量，重绘当前页的图表 */
    if (state.tab === "curves") updateCurves();
    else if (state.tab === "timing") updateTiming();
    else if (state.tab === "overview") { renderOverview(); refreshSystemLatest(); }
  }

  function bindControls() {
    applyTheme(currentTheme());
    document.getElementById("theme-toggle").addEventListener("click", function () {
      applyTheme(currentTheme() === "light" ? "dark" : "light");
    });

    document.addEventListener("click", function () {
      var menus = document.querySelectorAll(".run-select-menu");
      menus.forEach(function (m) { m.classList.add("hidden"); });
      if (state.runSelPending) {
        state.runSelPending = false;
        renderRunSelector();
      }
    });

    /* 交互结束（选区收起 / 控件失焦）后补一次被推迟的自动渲染 */
    document.addEventListener("selectionchange", flushDeferredRender);
    document.addEventListener("change", flushDeferredRender);
    document.addEventListener("blur", flushDeferredRender, true);

    document.getElementById("tabs").addEventListener("click", function (ev) {
      var btn = ev.target.closest(".tab");
      if (!btn) return;
      switchTab(btn.getAttribute("data-tab"));
    });

    document.getElementById("ema-slider").addEventListener("input", function (ev) {
      var v = parseFloat(ev.target.value);
      document.getElementById("ema-val").textContent = v.toFixed(2);
      if (mainChart) mainChart.setSmoothing(v);
    });
    document.getElementById("logy-toggle").addEventListener("change", function (ev) {
      if (mainChart) mainChart.setLogY(ev.target.checked);
    });
    document.getElementById("zoom-reset").addEventListener("click", function () {
      if (mainChart) { mainChart.view = null; mainChart.viewTouched = false; mainChart.redraw(); }
    });

    /* GPU 图：近期 ⇄ 全部 切换；手动缩放/平移后此按钮用于回到默认近期 */
    document.getElementById("gpu-view-toggle").addEventListener("click", function () {
      if (!gpuChart) return;
      if (gpuChart.viewTouched) {
        state.gpuViewMode = "recent";
      } else {
        state.gpuViewMode = state.gpuViewMode === "recent" ? "full" : "recent";
      }
      gpuChart.viewTouched = false;
      renderGpuViewToggle();
      updateGpuViews();
    });

    /* 对比页控件 */
    document.getElementById("cmp-ema-slider").addEventListener("input", function (ev) {
      var v = parseFloat(ev.target.value);
      document.getElementById("cmp-ema-val").textContent = v.toFixed(2);
      if (compareChart) compareChart.setSmoothing(v);
    });
    document.getElementById("cmp-logy-toggle").addEventListener("change", function (ev) {
      if (compareChart) compareChart.setLogY(ev.target.checked);
    });
    document.getElementById("cmp-zoom-reset").addEventListener("click", function () {
      if (compareChart) { compareChart.view = null; compareChart.viewTouched = false; compareChart.redraw(); }
    });
    document.getElementById("cmp-reload").addEventListener("click", function () {
      cmp.histories = {};
      Object.keys(cmp.selected).forEach(loadCompareHistory);
      updateCompare();
    });

    document.getElementById("bench-refresh").addEventListener("click", loadAnnotations);
    var llmBtn = document.getElementById("llm-toggle");
    llmBtn.addEventListener("click", function () {
      var pre = document.getElementById("llm-help");
      var show = pre.classList.contains("hidden");
      pre.classList.toggle("hidden", !show);
      llmBtn.textContent = show ? "收起" : "展开";
      if (show) loadLlmHelp();
    });
  }

  function boot() {
    bindControls();
    renderMetricChecks();
    renderCompareMetricChecks();
    MonApp.treeTab.init();
    ensureCharts();

    loadRuns().then(function (runs) {
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
        state.runId = null;
        selectRun(found);
      } else {
        renderOverview();
      }
    });

    loadSyncStatus();
    refreshDataRootLabel();
    initRootPicker();
    initAuthChip();

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
    setInterval(function () { if (state.syncPulling) loadSyncStatus(); }, 2000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
