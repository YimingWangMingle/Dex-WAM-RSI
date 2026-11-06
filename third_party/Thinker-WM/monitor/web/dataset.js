/* Motus Monitor 前端 —— 训练集 tab（数据集浏览 + 训练切片）。
 *
 * 数据流：本机 server /api/dataset/* → SSH 托管的集群 helper 按需解码
 * （视频 JPEG/MP4、action/state float32、文本），训练切片只传索引、点击
 * 后走 /api/dataset/selection 解码。二进制端点打包格式：
 *   4 字节大端 JSON 长度 + JSON 头 + blob（见 server.py _send_packed）。
 *
 * 时间对齐模型：视频帧 i ⇔ action 区间 (i*r, (i+1)*r]（r=ratio）；文本的
 * frame_range 落在同一时间轴上，可点击跳转。
 */
window.MonApp = window.MonApp || {};

(function () {
  "use strict";

  var api, util, esc, el, fmtInt;

  var SLICE_PAGE = 100;
  var EP_PAGE = 50;
  var FRAME_CACHE_MAX = 256;
  var KIND_COLORS = {
    episode_context: "#d9a038",
    action_segment: "#3fb96f",
    instruction_en: "#4f8cff",
    instruction_zh: "#58a6ff",
    instruction: "#4f8cff",
  };

  var ds = {
    initialized: false,
    enabled: false,
    mode: "browse",          // browse | slices
    datasets: [],
    dataset: null,
    episodes: [],
    epTotal: 0,
    epOffset: 0,
    epQuery: "",
    epTree: null,            // episode_tree payload（机器人→任务 两级）
    epTreeExpanded: {},      // robot name -> bool（单机器人默认展开）
    epGroup: null,           // 当前选中的 task 节点对象（null = 未选；prefix 可为空串=整库）
    // viewer
    info: null,              // episode_info payload
    ratio: 1,
    curFrame: 0,
    frameCache: new Map(),   // frame idx -> objectURL
    filmUrls: [],
    action: null,            // {indices, d, data: Float32Array}
    actionNames: null,       // 逐维语义名（helper 透传 catalog 元数据；null = 回退 a{N}）
    dimsOn: [],
    normalize: false,
    // slice 模式
    slice: null,             // {episode_id,start,n,ratio,step,source,dataset}
    sliceFrames: [],         // objectURL per selection frame
    sliceAction: null,
    sliceActionNames: null,
    sliceSel: null,
    // slices 面板
    runs: [],
    sliceRun: null,
    sliceOffset: 0,
    sliceTotal: 0,
    sliceSource: "",
    sliceCkpts: [],          // by_ckpt 段（含末尾“最新”段）
    sliceCkptIdx: -1,        // -1 = 全部；否则为 sliceCkpts 下标
    sliceStep: "",           // 精确 step 过滤（"" = 全部）
    sliceLineage: false,     // 历史累计：沿 parent 链聚合祖先 run
    sliceRanked: false,      // 当前响应是否为全 rank（step→node→gpu）结构
    sliceExpanded: {},       // step -> bool（切片树展开状态；未记录时默认仅最新 step 展开）
    lastSliceSteps: null,    // 当前页的 ranked steps（展开/折叠重渲染用）
    covExpanded: {},         // 覆盖率区块：source -> bool（episode 明细展开）
    sliceAutoPick: true,     // 下次 loadSlices 响应后自动选中末尾“最新”段（用户手动选段后失效）
    slicesLoadedFor: null,   // 已成功加载切片的 run（进入面板时据此判断是否需要加载）
    flashRange: null,
    playing: null,           // {url}
  };

  /* 数据源配色（按源名排序稳定取色，ckpt 配比条与图例共用） */
  var SLICE_PALETTE = [
    "#4f8cff", "#3fb96f", "#d9a038", "#e05d5d", "#9b6dd6",
    "#46b8b0", "#d06fa8", "#8ba33e", "#6d7dd8", "#c97b3f",
  ];

  /* ------------------------------------------------------------ 小工具 */
  function parsePacked(buf) {
    var view = new DataView(buf);
    var jsonLen = view.getUint32(0, false);
    var meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
    return { meta: meta, blob: buf.slice(4 + jsonLen) };
  }

  function fetchPacked(url) {
    return fetch(url).then(function (resp) {
      if (!resp.ok) {
        return resp.json().then(function (body) {
          throw new Error(body.error || ("HTTP " + resp.status));
        });
      }
      return resp.arrayBuffer();
    }).then(parsePacked);
  }

  /* 流式端点（stream=1）：前缀行 PRG <frac> <stage>（服务器解码进度，0..N 条）
   * + LEN <n> + n 字节成品；出错为 ERR <json>。handlers:
   *   onServerProgress(frac, stage)  集群解码推进
   *   onDownload(received, total)    成品下载推进
   * resolve 为成品字节的 ArrayBuffer。 */
  function fetchStream(url, handlers) {
    handlers = handlers || {};
    return fetch(url).then(function (resp) {
      if (!resp.ok) {
        return resp.json().then(function (body) {
          throw new Error(body.error || ("HTTP " + resp.status));
        });
      }
      var reader = resp.body.getReader();
      var pending = new Uint8Array(0);
      var mode = "head";
      var bodyTotal = -1;
      var decoder = new TextDecoder();
      function concat(a, b) {
        var c = new Uint8Array(a.length + b.length);
        c.set(a);
        c.set(b, a.length);
        return c;
      }
      function pump() {
        return reader.read().then(function (res) {
          if (res.value) pending = concat(pending, res.value);
          var i;
          while (mode === "head") {
            var nl = -1;
            for (i = 0; i < pending.length; i++) { if (pending[i] === 10) { nl = i; break; } }
            if (nl < 0) break;
            var line = decoder.decode(pending.slice(0, nl));
            pending = pending.slice(nl + 1);
            if (line.lastIndexOf("PRG ", 0) === 0) {
              var parts = line.slice(4).split(" ");
              if (handlers.onServerProgress) {
                handlers.onServerProgress(parseFloat(parts[0]), parts[1] || "");
              }
            } else if (line.lastIndexOf("LEN ", 0) === 0) {
              bodyTotal = parseInt(line.slice(4), 10) || 0;
              mode = "body";
            } else if (line.lastIndexOf("ERR ", 0) === 0) {
              var msg = "preview stream error";
              try { msg = JSON.parse(line.slice(4)).error || msg; } catch (e) { }
              throw new Error(msg);
            }
          }
          if (mode === "body" && handlers.onDownload) {
            handlers.onDownload(pending.length, bodyTotal);
          }
          if (res.done) {
            if (mode !== "body") throw new Error("bad preview stream");
            return pending.buffer.slice(pending.byteOffset, pending.byteOffset + pending.length);
          }
          return pump();
        });
      }
      return pump();
    });
  }

  function jpegUrls(meta, blob) {
    var urls = [];
    var pos = 0;
    for (var i = 0; i < meta.sizes.length; i++) {
      urls.push(URL.createObjectURL(new Blob([blob.slice(pos, pos + meta.sizes[i])], { type: "image/jpeg" })));
      pos += meta.sizes[i];
    }
    return urls;
  }

  /* 进度条组件：确定性 step/set 与不定态 indet；done() 淡出移除。 */
  function makeProgress(total, label) {
    var root = el("div", "ds-prog");
    var lab = el("div", "ds-prog-label", { text: label || "加载中…" });
    var track = el("div", "ds-prog-track");
    var bar = el("div", "ds-prog-bar");
    track.appendChild(bar);
    root.appendChild(lab);
    root.appendChild(track);
    var completed = 0, failed = false;
    var prog = {
      el: root,
      setTotal: function (n) { total = Math.max(1, n); },
      step: function (text) {
        if (failed) return;
        completed++;
        root.classList.remove("indet");
        bar.style.width = Math.min(100, completed / total * 100) + "%";
        if (text) lab.textContent = text;
        if (completed >= total) prog.done();
      },
      set: function (frac, text) {
        if (failed) return;
        root.classList.remove("indet");
        bar.style.width = Math.min(100, Math.max(0, frac) * 100) + "%";
        if (text) lab.textContent = text;
      },
      indet: function (text) {
        if (failed) return;
        root.classList.add("indet");
        if (text) lab.textContent = text;
      },
      done: function () {
        if (failed) return;
        root.classList.remove("indet");
        bar.style.width = "100%";
        root.classList.add("done");
        setTimeout(function () { root.remove(); }, 600);
      },
      fail: function (msg) {
        failed = true;
        root.classList.remove("indet");
        root.classList.add("fail");
        lab.textContent = msg || "加载失败";
      },
    };
    return prog;
  }

  function clearUrls(list) {
    (list || []).forEach(function (u) { try { URL.revokeObjectURL(u); } catch (e) { } });
  }

  function download(urlOrBlob, filename) {
    var a = document.createElement("a");
    a.href = urlOrBlob;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { a.remove(); }, 100);
  }

  function qs(params) {
    var parts = [];
    for (var k in params) {
      if (params[k] == null || params[k] === "") continue;
      parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(params[k]));
    }
    return parts.length ? "?" + parts.join("&") : "";
  }

  function viewerEl() { return document.getElementById("ds-viewer"); }

  function showViewerError(err) {
    viewerEl().innerHTML =
      '<div class="card"><div class="ds-error">加载失败：' + esc(err && err.message || err) + "</div></div>";
  }

  function showViewerLoading(text) {
    var prog = makeProgress(1, text || "解码中…");
    prog.indet();
    var card = el("div", "card placeholder-card");
    card.appendChild(el("div", "placeholder-icon", { text: "▦" }));
    card.appendChild(prog.el);
    viewerEl().innerHTML = "";
    viewerEl().appendChild(card);
    return prog;
  }

  /* renderViewer 会重建 DOM：把进度条挪进 vhead 的常驻槽位 */
  function attachViewerProgress(prog) {
    var slot = document.getElementById("ds-prog");
    if (slot) {
      slot.innerHTML = "";
      slot.appendChild(prog.el);
    }
  }

  /* 侧栏列表加载中的迷你进度条 */
  function listLoading(boxId, text) {
    var box = document.getElementById(boxId);
    box.innerHTML = "";
    var prog = makeProgress(1, text);
    prog.el.classList.add("mini");
    prog.indet();
    box.appendChild(prog.el);
  }

  function setupCanvas(canvas, height) {
    var width = canvas.parentElement.clientWidth - 2;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(50, width) * dpr;
    canvas.height = height * dpr;
    canvas.style.width = Math.max(50, width) + "px";
    canvas.style.height = height + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: Math.max(50, width), h: height };
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ------------------------------------------------------------ 状态/入口 */
  function refresh() {
    api = MonApp.api; util = MonApp.util; esc = util.esc; el = util.el; fmtInt = util.fmtInt;
    bindOnce();
    api.get("/api/dataset/status").then(function (st) {
      ds.enabled = !!st.enabled && !st.error;
      renderStatus(st);
      if (!ds.enabled) return;
      if (!ds.initialized) {
        ds.initialized = true;
        loadDatasets();
        loadRuns();
      }
    }).catch(function (err) {
      renderStatus({ enabled: false, error: err.message });
    });
  }

  function renderStatus(st) {
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

  function bindOnce() {
    if (bindOnce.done) return;
    bindOnce.done = true;
    document.querySelectorAll(".ds-mode-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { setMode(btn.getAttribute("data-dsmode")); });
    });
    document.getElementById("ds-ep-search-btn").addEventListener("click", function () {
      ds.epQuery = document.getElementById("ds-ep-search").value.trim();
      loadEpisodes(0);
    });
    document.getElementById("ds-ep-search").addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") { ds.epQuery = ev.target.value.trim(); loadEpisodes(0); }
    });
    document.getElementById("ds-ep-prev").addEventListener("click", function () {
      loadEpisodes(Math.max(0, ds.epOffset - EP_PAGE));
    });
    document.getElementById("ds-ep-next").addEventListener("click", function () {
      loadEpisodes(ds.epOffset + EP_PAGE);
    });
    document.getElementById("ds-slice-run").addEventListener("change", function (ev) {
      ds.sliceRun = ev.target.value;
      ds.sliceOffset = 0;
      ds.sliceCkptIdx = -1;
      ds.sliceStep = "";
      ds.sliceAutoPick = true;   // 换 run 后重新默认到最新段
      resetSliceSteps();
      loadSlices();
    });
    document.getElementById("ds-slice-source").addEventListener("change", function (ev) {
      ds.sliceSource = ev.target.value; ds.sliceOffset = 0; loadSlices();
    });
    document.getElementById("ds-slice-lineage").addEventListener("change", function (ev) {
      ds.sliceLineage = !!ev.target.checked; ds.sliceOffset = 0; loadSlices();
    });
    document.getElementById("ds-slice-expand-all").addEventListener("click", function () {
      setAllStepsExpanded(true);
    });
    document.getElementById("ds-slice-collapse-all").addEventListener("click", function () {
      setAllStepsExpanded(false);
    });
    document.getElementById("ds-slice-step").addEventListener("change", function (ev) {
      ds.sliceStep = ev.target.value; ds.sliceOffset = 0; loadSlices();
    });
    document.getElementById("ds-slice-prev").addEventListener("click", function () {
      ds.sliceOffset = Math.max(0, ds.sliceOffset - SLICE_PAGE); loadSlices();
    });
    document.getElementById("ds-slice-next").addEventListener("click", function () {
      if (ds.sliceOffset + SLICE_PAGE < ds.sliceTotal) {
        ds.sliceOffset += SLICE_PAGE; loadSlices();
      }
    });
    document.addEventListener("keydown", onKey);
  }

  function setMode(mode) {
    ds.mode = mode;
    document.querySelectorAll(".ds-mode-btn").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-dsmode") === mode);
    });
    document.getElementById("ds-browse-panel").classList.toggle("hidden", mode !== "browse");
    document.getElementById("ds-slices-panel").classList.toggle("hidden", mode !== "slices");
    if (mode === "slices" && ds.runs.length) {
      if (!ds.sliceRun) {
        ds.sliceRun = ds.runs[0].run_id;
        document.getElementById("ds-slice-run").value = ds.sliceRun;
      }
      /* browse 模式预置过 sliceRun 但未加载：进入面板时自动加载（默认最新段） */
      if (ds.slicesLoadedFor !== ds.sliceRun) loadSlices();
    }
  }

  function onKey(ev) {
    if (!ds.info) return;
    var tab = document.getElementById("tab-dataset");
    if (!tab || !tab.classList.contains("active")) return;
    if (/INPUT|TEXTAREA|SELECT/.test((ev.target.tagName || ""))) return;
    if (ev.key === "ArrowLeft") { ev.preventDefault(); seekFrame(ds.curFrame - 1); }
    if (ev.key === "ArrowRight") { ev.preventDefault(); seekFrame(ds.curFrame + 1); }
  }

  /* ------------------------------------------------------------ 数据集浏览 */
  function loadDatasets() {
    listLoading("ds-datasets", "加载数据集…");
    api.get("/api/dataset/datasets").then(function (payload) {
      ds.datasets = payload.datasets || [];
      renderDatasets();
    }).catch(function (err) {
      document.getElementById("ds-datasets").innerHTML =
        '<div class="ds-error">' + esc(err.message) + "</div>";
    });
  }

  function renderDatasets() {
    var box = document.getElementById("ds-datasets");
    box.innerHTML = "";
    ds.datasets.forEach(function (d) {
      var row = el("div", "ds-dataset" + (ds.dataset === d.name ? " active" : ""));
      row.appendChild(el("span", "ds-dataset-name", { text: d.name }));
      row.appendChild(el("span", "ds-dataset-count", {
        text: d.episodes != null ? String(d.episodes) : "?",
      }));
      row.addEventListener("click", function () { selectDataset(d.name); });
      box.appendChild(row);
    });
  }

  function selectDataset(name) {
    ds.dataset = name;
    ds.epOffset = 0;
    ds.epQuery = "";
    ds.epGroup = null;
    ds.epTree = null;
    ds.epTreeExpanded = {};
    document.getElementById("ds-ep-search").value = "";
    renderDatasets();
    document.getElementById("ds-ep-browser").classList.remove("hidden");
    renderEpScope();
    showEpHint();
    loadEpTree();
  }

  /* 机器人→任务 两级树；子id 在选中任务后进入下方分页列表 */
  function loadEpTree() {
    if (!ds.dataset) return;
    listLoading("ds-eptree", "加载层级树…");
    api.get("/api/dataset/episode_tree" + qs({ dataset: ds.dataset })).then(function (payload) {
      ds.epTree = payload;
      var robots = payload.robots || [];
      if (robots.length === 1) ds.epTreeExpanded[robots[0].name] = true;
      renderEpTree();
    }).catch(function (err) {
      document.getElementById("ds-eptree").innerHTML =
        '<div class="ds-error">' + esc(err.message) + "</div>";
    });
  }

  function renderEpTree() {
    var box = document.getElementById("ds-eptree");
    box.innerHTML = "";
    if (!ds.epTree) return;
    (ds.epTree.robots || []).forEach(function (robot) {
      var expanded = !!ds.epTreeExpanded[robot.name];
      var rrow = el("div", "ds-eptree-robot");
      rrow.appendChild(el("span", "ds-eptree-toggle", { text: expanded ? "▾" : "▸" }));
      rrow.appendChild(el("span", "ds-eptree-name", { text: robot.name, attrs: { title: robot.name } }));
      rrow.appendChild(el("span", "ds-dataset-count", { text: String(robot.count) }));
      rrow.addEventListener("click", function () {
        ds.epTreeExpanded[robot.name] = !expanded;
        renderEpTree();
      });
      box.appendChild(rrow);
      if (!expanded) return;
      (robot.tasks || []).forEach(function (task) {
        var trow = el("div", "ds-eptree-task" + (ds.epGroup === task ? " active" : ""));
        trow.appendChild(el("span", "ds-eptree-name", { text: task.label, attrs: { title: task.label } }));
        trow.appendChild(el("span", "ds-dataset-count", { text: String(task.count) }));
        trow.addEventListener("click", function () { selectEpGroup(task); });
        box.appendChild(trow);
      });
    });
  }

  function selectEpGroup(task) {
    ds.epGroup = (ds.epGroup === task) ? null : task;   // 再点一次取消选中
    ds.epOffset = 0;
    renderEpTree();
    renderEpScope();
    loadEpisodes(0);
  }

  /* 列表上方的当前分组标签 + 清除按钮 */
  function renderEpScope() {
    var box = document.getElementById("ds-ep-scope");
    box.innerHTML = "";
    if (!ds.epGroup) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    box.appendChild(el("span", "ds-ep-scope-label", {
      text: ds.epGroup.label, attrs: { title: ds.epGroup.prefix },
    }));
    var clear = el("button", "btn ds-ep-scope-clear", { text: "✕", attrs: { title: "清除分组" } });
    clear.addEventListener("click", function () {
      ds.epGroup = null;
      ds.epOffset = 0;
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
    if (!ds.dataset) return;
    if (!ds.epQuery && !ds.epGroup) {   // 未选分组且未搜索：不再罗列全量长列表
      showEpHint();
      return;
    }
    ds.epOffset = offset;
    listLoading("ds-episodes", "加载 episode 列表…");
    var url = "/api/dataset/episodes" + qs({
      dataset: ds.dataset, offset: offset, limit: EP_PAGE, q: ds.epQuery,
      prefix: ds.epQuery ? "" : (ds.epGroup ? ds.epGroup.prefix : ""),   // 搜索时全局查找（忽略分组）
    });
    api.get(url).then(function (payload) {
      ds.episodes = payload.episode_ids || [];
      ds.epTotal = payload.total;
      renderEpisodes();
    }).catch(function (err) {
      document.getElementById("ds-episodes").innerHTML =
        '<div class="ds-error">' + esc(err.message) + "</div>";
    });
  }

  function renderEpisodes() {
    var box = document.getElementById("ds-episodes");
    box.innerHTML = "";
    var prefix = ds.epGroup && !ds.epQuery ? ds.epGroup.prefix : "";
    ds.episodes.forEach(function (eid) {
      // 分组浏览时只显示子id（去掉组前缀），搜索时显示完整 id
      var label = prefix && eid.lastIndexOf(prefix, 0) === 0
        ? eid.slice(prefix.length)
        : eid;
      var row = el("div", "ds-episode" + (ds.info && ds.info.episode_id === eid ? " active" : ""), {
        text: label,
        attrs: { title: eid },
      });
      row.addEventListener("click", function () { openEpisode(ds.dataset, eid); });
      box.appendChild(row);
    });
    var total = ds.epTotal;
    var info = total < 0
      ? (ds.epOffset + 1) + "–" + (ds.epOffset + ds.episodes.length)
      : (ds.epOffset + 1) + "–" + Math.min(ds.epOffset + EP_PAGE, total) + " / " + total;
    document.getElementById("ds-ep-page-info").textContent = ds.episodes.length ? info : "无结果";
  }

  /* ------------------------------------------------------------ 训练切片面板 */
  function loadRuns() {
    api.get("/api/runs").then(function (payload) {
      ds.runs = payload.runs || [];
      var sel = document.getElementById("ds-slice-run");
      sel.innerHTML = "";
      ds.runs.forEach(function (r) {
        var label = (r.run_name || r.run_id) + " · " + (r.last_step != null ? ("step " + r.last_step) : "");
        sel.appendChild(el("option", "", { text: label, attrs: { value: r.run_id } }));
      });
      if (ds.runs.length && !ds.sliceRun) {
        ds.sliceRun = ds.runs[0].run_id;
        if (ds.mode === "slices") loadSlices();
      }
      /* 重建选项后恢复选中值，避免 select 显示与内部状态不一致 */
      if (ds.sliceRun) sel.value = ds.sliceRun;
    }).catch(function () { });
  }

  /* 当前 ckpt 段 + 精确 step 选择对应的 step_min/step_max（null = 不限） */
  function sliceStepRange() {
    var lo = null, hi = null;
    if (ds.sliceCkptIdx >= 0 && ds.sliceCkpts[ds.sliceCkptIdx]) {
      lo = ds.sliceCkpts[ds.sliceCkptIdx].step_min;
      hi = ds.sliceCkpts[ds.sliceCkptIdx].step_max;
    }
    if (ds.sliceStep !== "") {
      lo = hi = parseInt(ds.sliceStep, 10);
    }
    return { step_min: lo, step_max: hi };
  }

  function loadSlices() {
    if (!ds.sliceRun) return;
    listLoading("ds-slices", "加载训练切片…");
    var range = sliceStepRange();
    var url = "/api/runs/" + encodeURIComponent(ds.sliceRun) + "/slices" + qs({
      offset: ds.sliceOffset, limit: SLICE_PAGE, source: ds.sliceSource,
      step_min: range.step_min, step_max: range.step_max,
      lineage: ds.sliceLineage ? "1" : "",
    });
    api.get(url).then(function (payload) {
      ds.slicesLoadedFor = ds.sliceRun;
      ds.sliceRanked = !!payload.ranked;
      // ranked 响应分页单位为 step，旧响应为扁平 slice
      ds.sliceTotal = ds.sliceRanked ? (payload.total_steps || 0) : (payload.total_slices || 0);
      ds.sliceCkpts = payload.by_ckpt || [];
      /* 自动选中末尾"最新"段：进入面板/换 run 后的首次加载默认只看最新数据，
         用户可点"全部"行或其他段切换（手动选择后 sliceAutoPick 失效） */
      if (ds.sliceAutoPick && ds.sliceCkptIdx === -1 && ds.sliceCkpts.length) {
        ds.sliceAutoPick = false;
        /* 谱系模式下 by_ckpt 含祖先 run 的段：自动选段要找本 run 的末尾"最新"段 */
        var pickIdx = ds.sliceCkpts.length - 1;
        for (var pi = ds.sliceCkpts.length - 1; pi >= 0; pi--) {
          var segRun = ds.sliceCkpts[pi].run_id;
          if (!segRun || segRun === ds.sliceRun) { pickIdx = pi; break; }
        }
        ds.sliceCkptIdx = pickIdx;
        ds.sliceOffset = 0;
        renderSliceCkpts();
        loadSliceSteps();
        loadSlices();
        return;
      }
      renderSliceCkpts();
      renderSliceStats(payload);
      if (ds.sliceRanked) {
        ds.lastSliceSteps = payload.steps || [];
        renderSliceTree(ds.lastSliceSteps);
      } else {
        ds.lastSliceSteps = null;
        renderSlices(payload.slices || []);
      }
      var pageInfo = document.getElementById("ds-slice-page-info");
      if (ds.sliceTotal) {
        pageInfo.textContent = (ds.sliceOffset + 1) + "–" +
          Math.min(ds.sliceOffset + SLICE_PAGE, ds.sliceTotal) + " / " + ds.sliceTotal +
          (ds.sliceRanked ? " step · 共 " + (payload.total_slices || 0) + " 切片" : "");
      } else {
        pageInfo.textContent = "该范围暂无切片记录（需含 episode_ids 的 step 记录）";
      }
    }).catch(function (err) {
      document.getElementById("ds-slices").innerHTML =
        '<div class="ds-error">' + esc(err.message) + "</div>";
    });
  }

  /* 全部源名（跨 ckpt 段求并集，排序稳定 → 配色稳定） */
  function sliceSourceNames() {
    var seen = {};
    ds.sliceCkpts.forEach(function (seg) {
      Object.keys(seg.by_source || {}).forEach(function (name) { seen[name] = true; });
    });
    return Object.keys(seen).sort();
  }

  function sourceColor(name, names) {
    return SLICE_PALETTE[names.indexOf(name) % SLICE_PALETTE.length];
  }

  /* ckpt 配比图：每行一个 ckpt 段的 100% 堆叠条，点击按该段 step 范围过滤 */
  function renderSliceCkpts() {
    var box = document.getElementById("ds-slice-ckpts");
    var legend = document.getElementById("ds-slice-legend");
    box.innerHTML = "";
    legend.innerHTML = "";
    var names = sliceSourceNames();
    var hasCkpt = ds.sliceCkpts.some(function (seg) { return seg.ckpt_step != null; });
    if (!ds.sliceCkpts.length || !names.length || !hasCkpt) {
      box.appendChild(el("div", "ds-muted", {
        text: hasCkpt || !ds.sliceCkpts.length ? "无统计" : "该 run 无 ckpt 记录，无法按 ckpt 分段",
      }));
      return;
    }
    names.forEach(function (name) {
      var item = el("span", "ds-legend-item", { text: name, attrs: { title: name } });
      var sw = el("i", "ds-legend-sw");
      sw.style.background = sourceColor(name, names);
      item.insertBefore(sw, item.firstChild);
      legend.appendChild(item);
    });
    // “全部”行：跨段求和，范围不限
    var allBySource = {};
    var allTotal = 0;
    ds.sliceCkpts.forEach(function (seg) {
      allTotal += seg.total;
      Object.keys(seg.by_source || {}).forEach(function (name) {
        allBySource[name] = (allBySource[name] || 0) + seg.by_source[name];
      });
    });
    box.appendChild(mkCkptRow(-1, "全部", allBySource, allTotal, names));
    /* 谱系（历史累计）模式下分段行按 run 标注（server 端每段带 run_id） */
    var runIds = {};
    ds.sliceCkpts.forEach(function (seg) { if (seg.run_id) runIds[seg.run_id] = true; });
    var multiRun = Object.keys(runIds).length > 1;
    ds.sliceCkpts.forEach(function (seg, idx) {
      var label = seg.ckpt_step != null
        ? ("ckpt " + seg.ckpt_step)
        : ("最新 >" + (seg.step_min != null ? seg.step_min - 1 : "?"));
      if (multiRun && seg.run_id) label = runShortLabel(seg.run_id) + " " + label;
      box.appendChild(mkCkptRow(idx, label, seg.by_source || {}, seg.total, names));
    });
  }

  /* run 短标签（分段条用）：优先 run_name，否则 run_id 截断 */
  function runShortLabel(runId) {
    var name = null;
    ds.runs.forEach(function (r) { if (r.run_id === runId) name = r.run_name; });
    name = name || runId;
    return name.length > 14 ? name.slice(0, 13) + "…" : name;
  }

  function mkCkptRow(idx, label, bySource, total, names) {
    var row = el("div", "ds-ckpt-row" + (ds.sliceCkptIdx === idx ? " active" : ""));
    var tipParts = [label + " · 共 " + total + " 切片"];
    row.appendChild(el("span", "ds-ckpt-label", { text: label }));
    var bar = el("div", "ds-ckpt-bar");
    names.forEach(function (name) {
      var count = bySource[name] || 0;
      if (!count || !total) return;
      var pct = (count / total) * 100;
      var seg = el("i", "ds-ckpt-seg");
      seg.style.width = pct + "%";
      seg.style.background = sourceColor(name, names);
      if (pct >= 9) seg.textContent = Math.round(pct) + "%";
      bar.appendChild(seg);
      tipParts.push(name + " " + pct.toFixed(1) + "%");
    });
    row.appendChild(bar);
    row.appendChild(el("span", "ds-ckpt-total", { text: String(total) }));
    row.setAttribute("title", tipParts.join("\n"));
    row.addEventListener("click", function () {
      ds.sliceAutoPick = false;   // 手动选段后不再自动跳到最新段
      ds.sliceCkptIdx = (ds.sliceCkptIdx === idx) ? -1 : idx;
      ds.sliceStep = "";
      ds.sliceOffset = 0;
      renderSliceCkpts();
      if (ds.sliceCkptIdx >= 0) loadSliceSteps(); else resetSliceSteps();
      loadSlices();
    });
    return row;
  }

  /* 选中 ckpt 段后拉取该段的 step 列表填充下拉（全部时不拉，避免超长列表） */
  function loadSliceSteps() {
    var sel = document.getElementById("ds-slice-step");
    if (!ds.sliceRun || ds.sliceCkptIdx < 0) return resetSliceSteps();
    var range = sliceStepRange();
    var url = "/api/runs/" + encodeURIComponent(ds.sliceRun) + "/slices" + qs({
      steps_only: "1", step_min: range.step_min, step_max: range.step_max,
    });
    sel.disabled = true;
    api.get(url).then(function (payload) {
      sel.innerHTML = '<option value="">全部 step</option>';
      (payload.steps || []).forEach(function (st) {
        sel.appendChild(el("option", "", { text: "step " + st, attrs: { value: st } }));
      });
      sel.disabled = false;
      sel.value = ds.sliceStep;
    }).catch(function () { sel.disabled = false; });
  }

  function resetSliceSteps() {
    var sel = document.getElementById("ds-slice-step");
    sel.innerHTML = '<option value="">全部 step</option>';
    sel.disabled = ds.sliceCkptIdx < 0;
  }

  /* 当前范围（ckpt 段/step）的数据源占比饼图 + 图例（取代原文字百分比 chips） */
  function renderSlicePie(bySource) {
    var wrap = document.getElementById("ds-slice-pie-wrap");
    wrap.innerHTML = "";
    var names = Object.keys(bySource);
    var total = names.reduce(function (acc, n) { return acc + (bySource[n].count || 0); }, 0);
    if (!names.length || !total) return;
    var allNames = sliceSourceNames();
    var canvas = el("canvas", "ds-slice-pie");
    canvas.width = 132;
    canvas.height = 132;
    var ctx = canvas.getContext("2d");
    var cx = 66, cy = 66, r = 62, a0 = -Math.PI / 2;
    names.forEach(function (name) {
      var frac = bySource[name].count / total;
      var a1 = a0 + frac * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, a0, a1);
      ctx.closePath();
      ctx.fillStyle = sourceColor(name, allNames);
      ctx.fill();
      a0 = a1;
    });
    wrap.appendChild(canvas);
    var legend = el("div", "ds-pie-legend");
    names.forEach(function (name) {
      var s = bySource[name];
      var row = el("div", "ds-pie-legend-row", {
        attrs: { title: name + "：" + s.count + " 切片 / " + s.unique_episodes + " 集" },
      });
      var sw = el("i", "ds-legend-sw");
      sw.style.background = sourceColor(name, allNames);
      row.appendChild(sw);
      row.appendChild(el("span", "ds-pie-legend-name", { text: name }));
      row.appendChild(el("span", "ds-pie-legend-val", {
        text: ((s.count / total) * 100).toFixed(1) + "% · " + s.count,
      }));
      legend.appendChild(row);
    });
    wrap.appendChild(legend);
  }

  function renderSliceStats(payload) {
    var box = document.getElementById("ds-slice-stats");
    box.innerHTML = "";
    var bySource = payload.by_source || {};
    renderSlicePie(bySource);
    /* 每源一行：切片数/占比 + 实际喂入帧（含重复/去重/重复），tooltip 解释口径
     * 并附时间轴跨度与秒数（catalog 有 fps 的源） */
    var statsBySource = (payload.stats && payload.stats.by_source) || {};
    var names = Object.keys(bySource);
    var total = names.reduce(function (acc, n) { return acc + (bySource[n].count || 0); }, 0);
    var allNames = sliceSourceNames();
    names.forEach(function (name) {
      var s = bySource[name];
      var st = statsBySource[name] || {};
      var pct = total ? ((s.count / total) * 100).toFixed(1) : "0.0";
      var text = name + " · " + s.count + " 切片 (" + pct + "%)";
      var tip = name + "：" + s.count + " 切片 / " + s.unique_episodes + " 集";
      if (st.frames_fed != null) {
        text += " · 喂入 " + fmtInt(st.frames_fed) + " 帧（去重 " + fmtInt(st.frames_unique) +
          " · 重复 " + fmtInt(st.frames_repeat) + "）";
        tip += "\n喂入 " + fmtInt(st.frames_fed) + " 帧 = 所有切片实际送进模型的视频帧总数（同一帧被多个切片重复喂入会重复计数）";
        tip += "\n去重 " + fmtInt(st.frames_unique) + " 帧 = 同一集内不重复的源帧数（该集真正被模型看过的不同帧）";
        tip += "\n重复 " + fmtInt(st.frames_repeat) + " 帧 = 喂入 - 去重（多次采样造成的重复喂入量）";
        if (st.seconds_fed != null) {
          tip += "\n≈ 喂入 " + st.seconds_fed + "s / 去重 " + st.seconds_unique + "s（按 fps=" + st.fps + " 换算）";
        }
      }
      if (st.frames_total != null) {
        tip += "\n时间轴跨度（窗口首末帧区间）：求和 " + fmtInt(st.frames_total) +
          " / 并集 " + fmtInt(st.frames_union) + " / 重叠 " + fmtInt(st.frames_overlap) +
          " 帧（用于衡量采样窗口在集时间轴上的铺盖程度）";
      }
      var row = el("div", "ds-src-stat", { text: text, attrs: { title: tip } });
      var sw = el("i", "ds-legend-sw");
      sw.style.background = sourceColor(name, allNames);
      row.insertBefore(sw, row.firstChild);
      box.appendChild(row);
    });
    var top = (payload.top_episodes || []).slice(0, 3);
    top.forEach(function (t) {
      var row = el("div", "ds-stat-chip ds-top-ep", {
        text: "复用×" + t.count + " " + t.episode_id,
        attrs: { title: t.episode_id },
      });
      row.addEventListener("click", function () {
        openEpisode(t.source, t.episode_id);
      });
      box.appendChild(row);
    });
    renderSliceCoverage(payload.stats || null);
    // 源过滤下取用跨 ckpt 段的并集（scoped by_source 随范围变化，不适合做选项源）
    var sel = document.getElementById("ds-slice-source");
    var cur = sel.value;
    sel.innerHTML = '<option value="">全部源</option>';
    sliceSourceNames().forEach(function (name) {
      sel.appendChild(el("option", "", { text: name, attrs: { value: name } }));
    });
    sel.value = cur;
  }

  /* 覆盖率区块：每源百分比条（已喂 frames_covered/frames_total），点击展开
     该源 episode 明细（已喂覆盖率% / 未喂）；分母口径见 denominator。 */
  function renderSliceCoverage(stats) {
    var box = document.getElementById("ds-slice-coverage");
    box.innerHTML = "";
    if (!stats) return;
    var cov = stats.coverage;
    if (!cov) {
      if (stats.coverage_note) {
        box.appendChild(el("div", "ds-muted", { text: stats.coverage_note }));
      }
      return;
    }
    box.appendChild(el("div", "ds-subhead", { text: "数据源覆盖率（已喂帧/总帧，点击展开明细）" }));
    if (cov.denominator === "observed") {
      box.appendChild(el("div", "ds-muted", {
        text: "run 配置快照不可用，分母=切片中出现过的源（非训练配置全集）",
      }));
    }
    Object.keys(cov.sources || {}).forEach(function (name) {
      var c = cov.sources[name];
      var row = el("div", "ds-cov-row");
      row.appendChild(el("span", "ds-cov-name", { text: name, attrs: { title: name } }));
      var track = el("div", "ds-cov-track");
      var fill = el("i", "ds-cov-fill");
      fill.style.width = Math.min(100, c.coverage_pct || 0) + "%";
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", "ds-cov-val", {
        text: (c.coverage_pct || 0) + "% · " + c.episodes_fed + "/" + c.episodes_total + " 集",
        attrs: { title: "已喂 " + c.frames_covered + " / 共 " + c.frames_total + " 帧" },
      }));
      row.addEventListener("click", function () {
        ds.covExpanded[name] = !ds.covExpanded[name];
        renderSliceCoverage(stats);
      });
      box.appendChild(row);
      if (ds.covExpanded[name]) box.appendChild(mkCoverageDetail(name, c));
    });
    if (cov.missing_catalogs && cov.missing_catalogs.length) {
      box.appendChild(el("div", "ds-muted", {
        text: "无本地 catalog，覆盖率缺失：" + cov.missing_catalogs.join("、"),
      }));
    }
  }

  var COV_DETAIL_MAX = 300;

  function mkCoverageDetail(source, c) {
    var wrap = el("div", "ds-cov-detail");
    var episodes = c.episodes || [];
    episodes.slice(0, COV_DETAIL_MAX).forEach(function (ep) {
      var fed = ep.covered_frames > 0;
      var row = el("div", "ds-cov-ep" + (fed ? "" : " unfed"), {
        text: ep.episode_id + " · " + (fed
          ? (ep.covered_frames + "/" + ep.total_frames + " 帧 · " + ep.pct + "%")
          : "未喂"),
        attrs: { title: ep.episode_id },
      });
      row.addEventListener("click", function (ev) {
        ev.stopPropagation();
        openEpisode(source, ep.episode_id);
      });
      wrap.appendChild(row);
    });
    if (episodes.length > COV_DETAIL_MAX || c.episodes_truncated) {
      wrap.appendChild(el("div", "ds-muted", {
        text: "明细过长已截断（共 " + c.episodes_total + " 集）",
      }));
    }
    return wrap;
  }

  function mkSliceRow(s) {
    var row = el("div", "ds-episode ds-slice");
    row.appendChild(el("div", "ds-slice-l1", {
      text: "step " + s.step + " · " + s.source + (s.ratio ? " · r" + s.ratio : ""),
    }));
    row.appendChild(el("div", "ds-slice-l2", {
      text: s.episode_id + " @" + (s.window_start != null ? s.window_start : "?"),
      attrs: { title: s.episode_id },
    }));
    row.addEventListener("click", function () { openSlice(s); });
    return row;
  }

  /* 全 rank 切片树：一级=step（可折叠，默认仅最新 step 展开），
     子树 node → gpu，叶子=切片行（复用点击解码逻辑） */
  function renderSliceTree(steps) {
    var box = document.getElementById("ds-slices");
    box.innerHTML = "";
    steps.forEach(function (st, idx) {
      var expanded = (st.step in ds.sliceExpanded) ? ds.sliceExpanded[st.step] : (idx === 0);
      var head = el("div", "ds-step-head" + (expanded ? " open" : ""));
      head.appendChild(el("span", "ds-eptree-toggle", { text: expanded ? "▾" : "▸" }));
      head.appendChild(el("span", "ds-step-title", {
        text: "step " + st.step + " · " + st.total + " 切片",
      }));
      head.addEventListener("click", function () {
        ds.sliceExpanded[st.step] = !expanded;
        renderSliceTree(ds.lastSliceSteps || []);
      });
      box.appendChild(head);
      if (!expanded) return;
      var body = el("div", "ds-step-body");
      Object.keys(st.nodes || {}).sort().forEach(function (node) {
        var gpus = st.nodes[node] || {};
        Object.keys(gpus).sort().forEach(function (gpu) {
          var list = gpus[gpu] || [];
          body.appendChild(el("div", "ds-gpu-head", {
            text: node + " / gpu" + gpu + " · " + list.length + " 条",
          }));
          list.forEach(function (s) { body.appendChild(mkSliceRow(s)); });
        });
      });
      box.appendChild(body);
    });
  }

  function setAllStepsExpanded(expanded) {
    (ds.lastSliceSteps || []).forEach(function (st) { ds.sliceExpanded[st.step] = expanded; });
    if (ds.lastSliceSteps) renderSliceTree(ds.lastSliceSteps);
  }

  function renderSlices(slices) {
    var box = document.getElementById("ds-slices");
    box.innerHTML = "";
    slices.forEach(function (s) { box.appendChild(mkSliceRow(s)); });
  }

  /* ------------------------------------------------------------ Episode 查看器 */
  function openEpisode(dataset, episodeId, slice) {
    ds.slice = slice || null;
    var prog = showViewerLoading(slice ? "解码训练切片…" : "解码 episode…");
    api.get("/api/dataset/episode" + qs({ dataset: dataset, id: episodeId })).then(function (info) {
      ds.dataset = dataset;
      ds.info = info;
      ds.ratio = (slice && slice.ratio) || Math.max(1, Math.round(info.ratio_hint || 1));
      ds.curFrame = slice ? slice.start : 0;
      ds.flashRange = null;
      clearUrls(ds.filmUrls); ds.filmUrls = [];
      clearUrls(ds.sliceFrames); ds.sliceFrames = [];
      ds.sliceAction = null; ds.sliceSel = null;
      ds.actionNames = (info.action_names && info.action_names.length)
        ? info.action_names : null;   // 先拿 episode_info 的，action_state 回来后再校准
      ds.sliceActionNames = null;
      ds.frameCache.forEach(function (u) { URL.revokeObjectURL(u); });
      ds.frameCache.clear();
      renderViewer();
      var total = 3 + (slice ? 1 : 0);
      prog.setTotal(total);
      attachViewerProgress(prog);
      prog.step("episode 信息 ✓（1/" + total + "）");
      // 各区域独立加载/独立报错（不再一处失败整页替换）
      var jobs = [
        loadAction(dataset, episodeId).then(
          function () { prog.step("action 曲线 ✓（2/" + total + "）"); },
          function (err) { prog.step("action 加载失败"); cardError("action", err); }
        ),
        loadFilmstrip(dataset, episodeId, function (frac) {
          prog.set((2 + frac * 0.99) / total, "视频帧解码 " + Math.round(frac * 100) + "%（3/" + total + "）");
        }).then(
          function () { prog.step("视频帧 ✓（3/" + total + "）"); },
          function (err) { prog.step("视频帧加载失败"); cardError("filmstrip", err); }
        ),
      ];
      if (slice) {
        jobs.push(loadSelection(dataset, slice, function (frac) {
          prog.set((3 + frac * 0.99) / total, "切片解码 " + Math.round(frac * 100) + "%（4/" + total + "）");
        }).then(
          function () { prog.step("切片解码 ✓（4/" + total + "）"); },
          function (err) { prog.step("切片解码失败"); cardError("selection", err); }
        ));
      }
      return Promise.all(jobs);
    }).catch(function (err) {
      prog.fail("加载失败：" + (err && err.message || err));
      showViewerError(err);
    });
  }

  /* 单个区域的错误呈现 + 重试（不影响其他区域） */
  function cardError(kind, err) {
    var msg = "加载失败：" + (err && err.message || err);
    if (kind === "action") {
      if (ds.dimToggles) {
        ds.dimToggles.innerHTML = "";
        ds.dimToggles.appendChild(el("div", "ds-error", { text: msg }));
        ds.dimToggles.appendChild(mkBtn("重试 action", function () {
          ds.dimToggles.innerHTML = "";
          drawAction();
          loadAction(ds.dataset, ds.info.episode_id).then(function () { }, function (e) { cardError("action", e); });
        }));
      }
      if (ds.actionCanvas) {
        var env = setupCanvas(ds.actionCanvas, 220);
        env.ctx.fillStyle = cssVar("--card") || "#161e2e";
        env.ctx.fillRect(0, 0, env.w, env.h);
        env.ctx.fillStyle = cssVar("--red-strong") || "#ff8a80";
        env.ctx.font = "12px sans-serif";
        env.ctx.fillText(msg, 10, 24);
      }
      return;
    }
    if (kind === "filmstrip") {
      if (ds.filmstrip) {
        ds.filmstrip.innerHTML = "";
        ds.filmstrip.appendChild(el("div", "ds-error", { text: msg }));
        ds.filmstrip.appendChild(mkBtn("重试视频帧", function () {
          loadFilmstrip(ds.dataset, ds.info.episode_id).then(function () { }, function (e) { cardError("filmstrip", e); });
        }));
      }
      return;
    }
    if (kind === "selection") {
      var slot = document.getElementById("ds-prog");
      if (slot) {
        slot.innerHTML = "";
        slot.appendChild(el("div", "ds-error", { text: msg + "（可重新点击切片）" }));
      }
    }
  }

  function openSlice(s) {
    if (s.window_start == null || !s.num_video_frames) {
      showViewerError(new Error("该切片记录缺少窗口信息（frame_indices）"));
      return;
    }
    openEpisode(s.source, s.episode_id, {
      dataset: s.source,
      episode_id: s.episode_id,
      start: s.window_start,
      n: s.num_video_frames,
      ratio: s.ratio || 1,
      // 训练时的真实视频索引（stride 布局为跳帧+末帧填充）；有则优先于 start/n
      frames: (s.frame_indices && s.frame_indices.length) ? s.frame_indices : null,
      step: s.step,
      source: s.source,
    });
  }

  function loadAction(dataset, episodeId) {
    var url = "/api/dataset/action" + qs({ dataset: dataset, id: episodeId, max_points: 2048 });
    return fetchPacked(url).then(function (packed) {
      var m = packed.meta;
      var nA = m.action_indices.length, dA = m.d_action;
      ds.action = {
        indices: m.action_indices,
        d: dA,
        data: new Float32Array(packed.blob, 0, nA * dA),
        total: m.action_count,
      };
      ds.actionNames = (m.action_names && m.action_names.length === dA) ? m.action_names : null;
      ds.dimsOn = [];
      for (var i = 0; i < dA; i++) ds.dimsOn.push(i);   // 默认显示全部维度
      drawAction();
      renderDimToggles();
    });
  }

  function loadFilmstrip(dataset, episodeId, onProgress) {
    var vc = ds.info.video_count;
    var count = Math.min(24, vc);
    var stride = Math.max(1, Math.floor(vc / count));
    var url = "/api/dataset/frames" + qs({
      dataset: dataset, id: episodeId, start: 0, count: count, stride: stride,
      max_dim: 200, stream: 1,
    });
    return fetchStream(url, {
      onServerProgress: function (frac) { if (onProgress) onProgress(frac); },
    }).then(function (buf) {
      var packed = parsePacked(buf);
      ds.filmIndices = packed.meta.indices;
      ds.filmUrls = jpegUrls(packed.meta, packed.blob);
      renderFilmstrip();
      showFrame(ds.curFrame);
    });
  }

  function loadSelection(dataset, slice, onProgress) {
    var params = {
      dataset: dataset, episode_id: slice.episode_id,
      ratio: slice.ratio, max_dim: 640, stream: 1,
    };
    if (slice.frames && slice.frames.length) {
      // 训练切片真实视频索引（stride 布局跳帧+末帧填充），由 helper 推导 state/action
      params.frames = JSON.stringify(slice.frames);
    } else {
      params.start = slice.start;
      params.n = slice.n;
    }
    var url = "/api/dataset/selection" + qs(params);
    return fetchStream(url, {
      onServerProgress: function (frac) { if (onProgress) onProgress(frac); },
    }).then(function (buf) {
      var packed = parsePacked(buf);
      var m = packed.meta;
      var pos = 0, i;
      var jpegs = [];
      for (i = 0; i < m.jpeg_sizes.length; i++) {
        jpegs.push(packed.blob.slice(pos, pos + m.jpeg_sizes[i]));
        pos += m.jpeg_sizes[i];
      }
      ds.sliceFrames = jpegs.map(function (b) {
        return URL.createObjectURL(new Blob([b], { type: "image/jpeg" }));
      });
      var nA = m.n_action * m.d_action;
      ds.sliceAction = {
        indices: m.selection.action_indices,
        d: m.d_action,
        // jpeg 段长度任意，action 块偏移未必 4 字节对齐，先 copy 出独立 buffer
        data: new Float32Array(packed.blob.slice(pos, pos + nA * 4)),
      };
      ds.sliceActionNames =
        (m.action_names && m.action_names.length === m.d_action) ? m.action_names : null;
      ds.sliceSel = m.selection;
      renderViewerChrome();
      renderFilmstrip();
      drawTimeline();
      drawAction();
      showFrame(ds.curFrame);
    });
  }

  /* ------------------------------------------------------------ 查看器渲染 */
  function renderViewer() {
    var info = ds.info;
    var root = viewerEl();
    root.innerHTML = "";

    var head = el("div", "card ds-vhead");
    var title = el("div", "ds-vtitle");
    title.appendChild(el("div", "ds-vid", { text: info.episode_id, attrs: { title: info.episode_id } }));
    title.appendChild(el("div", "ds-vmeta", {
      text: info.robot_name + " · 视频 " + info.video_count + " 帧 · action " +
        info.action_count + " · state " + info.state_count + " · 文本 " + info.text_count +
        " · ratio " + ds.ratio + (ds.slice ? " · 切片 step " + ds.slice.step : ""),
    }));
    head.appendChild(title);
    var btns = el("div", "ds-vbtns");
    btns.appendChild(mkBtn("下载当前帧 PNG", saveFramePng));
    btns.appendChild(mkBtn("下载窗口 MP4", saveWindowMp4));
    btns.appendChild(mkBtn("下载 action CSV", saveActionCsv));
    if (ds.slice) {
      btns.appendChild(mkBtn("下载 selection JSON", saveSelectionJson));
      btns.appendChild(mkBtn("查看完整 episode", function () {
        openEpisode(ds.dataset, ds.info.episode_id);
      }));
    }
    head.appendChild(btns);
    head.appendChild(el("div", "ds-prog-slot", { attrs: { id: "ds-prog" } }));
    root.appendChild(head);

    var stage = el("div", "card ds-stage");
    var left = el("div", "ds-stage-left");
    var imgWrap = el("div", "ds-img-wrap");
    ds.mainImg = el("img", "ds-main-img loading", { attrs: { alt: "" } });
    imgWrap.appendChild(ds.mainImg);
    ds.frameSpinner = el("div", "ds-spinner");
    imgWrap.appendChild(ds.frameSpinner);
    ds.frameProg = makeProgress(1, "解码当前帧…");
    ds.frameProg.el.classList.add("mini", "ds-frame-prog", "hidden");
    imgWrap.appendChild(ds.frameProg.el);
    ds.videoWrap = el("div", "ds-video-wrap hidden");
    left.appendChild(imgWrap);
    left.appendChild(ds.videoWrap);
    var transport = el("div", "ds-transport");
    transport.appendChild(mkBtn("◀", function () { seekFrame(ds.curFrame - 1); }));
    ds.frameLabel = el("span", "ds-frame-label", { text: "—" });
    transport.appendChild(ds.frameLabel);
    transport.appendChild(mkBtn("▶", function () { seekFrame(ds.curFrame + 1); }));
    transport.appendChild(mkBtn("播放片段", playWindow));
    left.appendChild(transport);
    ds.filmstrip = el("div", "ds-filmstrip");
    left.appendChild(ds.filmstrip);
    stage.appendChild(left);

    var right = el("div", "ds-stage-right");
    ds.textsBox = el("div", "ds-texts");
    right.appendChild(ds.textsBox);
    stage.appendChild(right);
    root.appendChild(stage);

    var tlCard = el("div", "card");
    tlCard.appendChild(el("div", "ds-chart-title", { text: "统一时间轴（帧 ⇔ action 区间；彩色段=文本区间（连续同文段已合并），顶部淡条=全局指令，点击跳转）" }));
    var tlWrap = el("div", "ds-canvas-wrap");
    ds.timeline = el("canvas", "ds-timeline");
    tlWrap.appendChild(ds.timeline);
    ds.tlTip = el("div", "ds-tooltip hidden");
    tlWrap.appendChild(ds.tlTip);
    tlCard.appendChild(tlWrap);
    root.appendChild(tlCard);

    var acCard = el("div", "card");
    var acHead = el("div", "ds-chart-head");
    acHead.appendChild(el("div", "ds-chart-title", { text: "action 曲线" }));
    var norm = el("label", "ds-norm");
    var normCb = el("input", "", { attrs: { type: "checkbox" } });
    normCb.checked = ds.normalize;
    normCb.addEventListener("change", function () { ds.normalize = normCb.checked; drawAction(); });
    norm.appendChild(normCb);
    norm.appendChild(document.createTextNode("按维归一化"));
    acHead.appendChild(norm);
    acCard.appendChild(acHead);
    ds.dimToggles = el("div", "ds-dims");
    acCard.appendChild(ds.dimToggles);
    var acWrap = el("div", "ds-canvas-wrap");
    ds.actionCanvas = el("canvas", "ds-action");
    acWrap.appendChild(ds.actionCanvas);
    acCard.appendChild(acWrap);
    root.appendChild(acCard);

    renderTexts();
    renderViewerChrome();
    bindCanvasEvents();
  }

  function renderViewerChrome() {
    // slice 到达后更新标题提示
    if (ds.slice && ds.sliceSel) {
      var meta = document.querySelector("#ds-viewer .ds-vmeta");
      if (meta) {
        meta.textContent = meta.textContent.replace(/ · 切片 step.*$/, "") +
          " · 切片窗口 [" + ds.slice.start + ", " + (ds.slice.start + ds.slice.n) + "] step " + ds.slice.step;
      }
    }
  }

  function mkBtn(text, onclick) {
    var b = el("button", "btn ds-btn", { text: text });
    b.addEventListener("click", onclick);
    return b;
  }

  /* 连续同 kind+text 的段合并显示（agibot 逐帧标注会并成一条带多个区间）；
   * 无 video_range 的全局指令（robotwin/robotmind 的几十个同义变体）折叠展示。 */
  function mergedTexts(texts) {
    var out = [];
    (texts || []).forEach(function (t) {
      var last = out.length ? out[out.length - 1] : null;
      if (last && last.ranges && t.frame_range &&
          last.kind === t.kind && last.text === t.text) {
        last.ranges.push(t.frame_range);
        return;
      }
      out.push({
        kind: t.kind, text: t.text, index: t.index,
        ranges: t.frame_range ? [t.frame_range] : null,
      });
    });
    return out;
  }

  function kindBadge(kind, nSeg) {
    var badge = el("span", "ds-kind", { text: kind + (nSeg > 1 ? " ×" + nSeg : "") });
    badge.style.background = KIND_COLORS[kind] || "#5c6880";
    return badge;
  }

  function globalTextRow(t, firstInstr) {
    var row = el("div", "ds-text-item");
    row.appendChild(kindBadge(t.kind, 1));
    row.appendChild(el("span", "ds-text-body", { text: t.text }));
    if (t.index === firstInstr) {
      row.appendChild(el("span", "ds-text-range", { text: "训练采用" }));
    }
    return row;
  }

  function renderTexts() {
    var box = ds.textsBox;
    box.innerHTML = "";
    box.appendChild(el("div", "ds-chart-title", {
      text: "文本（kind 着色；连续同文段已合并，区间 chips 可点击跳转）",
    }));
    var merged = mergedTexts(ds.info.texts);
    var firstInstr = null;
    (ds.info.texts || []).forEach(function (t) {
      if (firstInstr === null && (t.kind === "instruction_en" || t.kind === "instruction")) {
        firstInstr = t.index;
      }
    });
    merged.forEach(function (t) {
      if (!t.ranges) return;
      var row = el("div", "ds-text-item");
      row.appendChild(kindBadge(t.kind, t.ranges.length));
      row.appendChild(el("span", "ds-text-body", { text: t.text }));
      var rangesBox = el("span", "ds-text-ranges");
      t.ranges.slice(0, 8).forEach(function (r) {
        var chip = el("span", "ds-range-chip", { text: "[" + r[0] + "," + r[1] + "]" });
        chip.addEventListener("click", function () {
          ds.flashRange = { range: r, until: Date.now() + 1500 };
          seekFrame(r[0]);
        });
        rangesBox.appendChild(chip);
      });
      if (t.ranges.length > 8) {
        rangesBox.appendChild(el("span", "ds-text-range", { text: "+" + (t.ranges.length - 8) }));
      }
      row.appendChild(rangesBox);
      box.appendChild(row);
    });
    var globals = merged.filter(function (t) { return !t.ranges; });
    var shown = globals.length <= 4 ? globals : globals.slice(0, 3);
    shown.forEach(function (t) { box.appendChild(globalTextRow(t, firstInstr)); });
    if (globals.length > shown.length) {
      var more = el("button", "btn ds-btn", {
        text: "展开其余 " + (globals.length - shown.length) + " 条指令变体",
      });
      more.addEventListener("click", function () {
        globals.slice(shown.length).forEach(function (t) {
          box.insertBefore(globalTextRow(t, firstInstr), more);
        });
        more.remove();
      });
      box.appendChild(more);
    }
  }

  function renderFilmstrip() {
    var strip = ds.filmstrip;
    strip.innerHTML = "";
    if (ds.slice && ds.sliceFrames.length) {
      var sel = ds.sliceSel.video_frame_indices;
      ds.sliceFrames.forEach(function (url, i) {
        var img = el("img", "ds-thumb", { attrs: { src: url, title: "帧 " + sel[i] } });
        img.addEventListener("click", function () { seekFrame(sel[i]); });
        strip.appendChild(img);
      });
      return;
    }
    (ds.filmUrls || []).forEach(function (url, i) {
      var idx = ds.filmIndices[i];
      var img = el("img", "ds-thumb", { attrs: { src: url, title: "帧 " + idx } });
      img.addEventListener("click", function () { seekFrame(idx); });
      strip.appendChild(img);
    });
  }

  /* 当前生效的逐维语义名（切片优先；无标注回退 null） */
  function currentActionNames() {
    if (ds.slice && ds.sliceActionNames) return ds.sliceActionNames;
    return ds.actionNames;
  }

  /* 维度显示名：有语义标注用标注，否则回退 "a{N}"（csvFallback=true 时 "dim_{N}"） */
  function dimLabel(dim, csvFallback) {
    var names = currentActionNames();
    if (names && names[dim]) return names[dim];
    return (csvFallback ? "dim_" : "a") + dim;
  }

  function renderDimToggles() {
    var box = ds.dimToggles;
    if (!box) return;
    box.innerHTML = "";
    var d = currentActionDim();
    for (var i = 0; i < d; i++) {
      (function (dim) {
        var label = dimLabel(dim);
        var chip = el("button", "ds-dim" + (ds.dimsOn.indexOf(dim) >= 0 ? " on" : ""), {
          text: label,
          attrs: { title: "a" + dim + (label !== "a" + dim ? " · " + label : "") },
        });
        chip.style.borderColor = MonApp.PALETTE[dim % MonApp.PALETTE.length];
        chip.addEventListener("click", function () {
          var at = ds.dimsOn.indexOf(dim);
          if (at >= 0) ds.dimsOn.splice(at, 1); else ds.dimsOn.push(dim);
          renderDimToggles();
          drawAction();
        });
        box.appendChild(chip);
      })(i);
    }
  }

  function currentActionDim() {
    if (ds.slice && ds.sliceAction) return ds.sliceAction.d;
    return ds.action ? ds.action.d : 0;
  }

  /* ------------------------------------------------------------ 帧显示 */
  function seekFrame(f) {
    if (!ds.info) return;
    f = Math.max(0, Math.min(ds.info.video_count - 1, f));
    ds.curFrame = f;
    showFrame(f);
    drawTimeline();
    drawAction();
  }

  /* 主图换帧：加载完成前隐藏 img（避免破图图标）并显示转圈动画 */
  function setMainFrame(url) {
    if (!ds.mainImg || ds.mainImg.src === url) return;
    ds.mainImg.classList.add("loading");
    if (ds.frameSpinner) ds.frameSpinner.classList.remove("hidden");
    ds.mainImg.onload = function () {
      ds.mainImg.classList.remove("loading");
      if (ds.frameSpinner) ds.frameSpinner.classList.add("hidden");
    };
    ds.mainImg.onerror = function () {
      ds.mainImg.classList.remove("loading");
      if (ds.frameSpinner) ds.frameSpinner.classList.add("hidden");
    };
    ds.mainImg.src = url;
  }

  function showFrame(f) {
    if (!ds.info) return;
    ds.frameLabel.textContent = "帧 " + f + " / " + (ds.info.video_count - 1) +
      " ｜ action (" + f * ds.ratio + ", " + (f + 1) * ds.ratio + "]";
    // slice 窗口内的帧直接用切片解码结果
    if (ds.slice && ds.sliceSel) {
      var at = ds.sliceSel.video_frame_indices.indexOf(f);
      if (at >= 0 && ds.sliceFrames[at]) {
        setMainFrame(ds.sliceFrames[at]);
        return;
      }
    }
    var cached = ds.frameCache.get(f);
    if (cached) {
      setMainFrame(cached);
      return;
    }
    // 单帧解码：图片上方细进度条（集群 ffmpeg 解码可能需要数秒）
    if (ds.frameProg) {
      ds.frameProg.el.classList.remove("hidden");
      ds.frameProg.indet("解码第 " + f + " 帧…");
    }
    var url = "/api/dataset/frames" + qs({
      dataset: ds.dataset, id: ds.info.episode_id, indices: String(f), max_dim: 768,
    });
    fetchPacked(url).then(function (packed) {
      if (ds.frameProg) ds.frameProg.el.classList.add("hidden");
      var urls = jpegUrls(packed.meta, packed.blob);
      var obj = urls[0];
      ds.frameCache.set(f, obj);
      if (ds.frameCache.size > FRAME_CACHE_MAX) {
        var oldest = ds.frameCache.keys().next().value;
        URL.revokeObjectURL(ds.frameCache.get(oldest));
        ds.frameCache.delete(oldest);
      }
      if (ds.curFrame === f) setMainFrame(obj);
      // 预取相邻帧
      [f + 1, f - 1].forEach(function (nf) {
        if (nf >= 0 && nf < ds.info.video_count && !ds.frameCache.has(nf)) {
          fetchPacked("/api/dataset/frames" + qs({
            dataset: ds.dataset, id: ds.info.episode_id, indices: String(nf), max_dim: 768,
          })).then(function (p) {
            ds.frameCache.set(nf, jpegUrls(p.meta, p.blob)[0]);
          }).catch(function () { });
        }
      });
    }).catch(function () {
      if (ds.frameProg) ds.frameProg.el.classList.add("hidden");
    });
  }

  function playWindow() {
    if (!ds.info) return;
    var params = {
      dataset: ds.dataset, id: ds.info.episode_id, start: 0,
      stride: 1, fps: 0, max_dim: 640, stream: 1,   // fps=0 → 后端按源真实帧率编码
    };
    if (ds.slice) {
      params.start = ds.slice.start;
      params.count = ds.slice.n + 1;   // 训练窗口：真实帧、真实速度
    } else {
      params.seconds = 10;             // 整集预览：真实速度 10s 窗口（短片则全片）
    }
    var url = "/api/dataset/clip" + qs(params);
    ds.videoWrap.classList.remove("hidden");
    ds.videoWrap.innerHTML = "";
    var prog = makeProgress(1, "集群解码/编码 MP4 中…");
    prog.indet();
    ds.videoWrap.appendChild(prog.el);
    fetchStream(url, {
      onServerProgress: function (frac) {
        prog.set(frac * 0.95, "集群解码/编码 " + Math.round(frac * 100) + "%…");
      },
      onDownload: function (received, total) {
        if (total > 0) {
          prog.set(0.95 + 0.05 * received / total,
            "下载中 " + util.fmtBytes(received) + " / " + util.fmtBytes(total));
        }
      },
    }).then(function (buf) {
      prog.done();
      var objUrl = URL.createObjectURL(new Blob([buf], { type: "video/mp4" }));
      ds.videoWrap.innerHTML = "";
      var video = el("video", "ds-video", { attrs: { src: objUrl, controls: "controls", autoplay: "autoplay", loop: "loop" } });
      ds.videoWrap.appendChild(video);
      var close = mkBtn("关闭播放", function () {
        ds.videoWrap.classList.add("hidden");
        ds.videoWrap.innerHTML = "";
        URL.revokeObjectURL(objUrl);
      });
      ds.videoWrap.appendChild(close);
    }).catch(function (err) {
      prog.fail("加载失败：" + (err && err.message || err));
    });
  }

  /* ------------------------------------------------------------ 保存 */
  function saveFramePng() {
    if (!ds.mainImg || !ds.mainImg.src) return;
    var img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = function () {
      var canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      canvas.toBlob(function (blob) {
        download(URL.createObjectURL(blob),
          ds.info.episode_id.replace(/[^\w.-]+/g, "_") + "_f" + ds.curFrame + ".png");
      }, "image/png");
    };
    img.src = ds.mainImg.src;
  }

  function saveWindowMp4() {
    if (!ds.info) return;
    var params = {
      dataset: ds.dataset, id: ds.info.episode_id, start: 0,
      stride: 1, fps: 0, max_dim: 640, stream: 1,   // fps=0 → 后端按源真实帧率编码
    };
    if (ds.slice) {
      params.start = ds.slice.start;
      params.count = ds.slice.n + 1;
    } else {
      params.seconds = 20;             // 真实速度 20s 窗口（短片则全片）
    }
    var url = "/api/dataset/clip" + qs(params);
    var slot = document.getElementById("ds-prog");
    var prog = slot ? makeProgress(1, "集群编码 MP4…") : null;
    if (prog) {
      prog.indet();
      slot.innerHTML = "";
      slot.appendChild(prog.el);
    }
    fetchStream(url, {
      onServerProgress: function (frac) {
        if (prog) prog.set(frac * 0.95, "集群解码/编码 " + Math.round(frac * 100) + "%…");
      },
      onDownload: function (received, total) {
        if (prog && total > 0) {
          prog.set(0.95 + 0.05 * received / total, "下载中 " + util.fmtBytes(received));
        }
      },
    }).then(function (buf) {
      if (prog) prog.done();
      download(URL.createObjectURL(new Blob([buf], { type: "video/mp4" })),
        ds.info.episode_id.replace(/[^\w.-]+/g, "_") + "_f" + params.start + ".mp4");
    }).catch(function (err) {
      if (prog) prog.fail("下载失败：" + (err && err.message || err));
    });
  }

  function saveActionCsv() {
    var src = (ds.slice && ds.sliceAction) || ds.action;
    if (!src || !src.data) return;
    var head = ["action_index"];
    for (var i = 0; i < src.d; i++) head.push(dimLabel(i, true).replace(/[,\r\n]+/g, " "));
    var lines = [head.join(",")];
    for (var r = 0; r < src.indices.length; r++) {
      var row = [String(src.indices[r])];
      for (var c = 0; c < src.d; c++) row.push(String(src.data[r * src.d + c]));
      lines.push(row.join(","));
    }
    var blob = new Blob([lines.join("\n")], { type: "text/csv" });
    download(URL.createObjectURL(blob),
      ds.info.episode_id.replace(/[^\w.-]+/g, "_") + "_action.csv");
  }

  function saveSelectionJson() {
    if (!ds.sliceSel) return;
    var blob = new Blob([JSON.stringify(ds.sliceSel, null, 2)], { type: "application/json" });
    download(URL.createObjectURL(blob),
      ds.info.episode_id.replace(/[^\w.-]+/g, "_") + "_selection.json");
  }

  /* ------------------------------------------------------------ 时间轴/曲线 */
  function bindCanvasEvents() {
    var scrubbing = false;
    function seekFromEvent(ev) {
      var rect = ds.timeline.getBoundingClientRect();
      var x = ev.clientX - rect.left;
      var pad = 8, w = rect.width - pad * 2;
      var vc = ds.info.video_count;
      var f = Math.round(((x - pad) / w) * (vc - 1));
      seekFrame(f);
    }
    ds.timeline.addEventListener("mousedown", function (ev) { scrubbing = true; seekFromEvent(ev); });
    window.addEventListener("mousemove", function (ev) {
      if (scrubbing) seekFromEvent(ev);
      else updateTimelineTip(ev);
    });
    window.addEventListener("mouseup", function () { scrubbing = false; });
    ds.actionCanvas.addEventListener("click", function (ev) {
      var src = (ds.slice && ds.sliceAction) || ds.action;
      if (!src || !src.indices.length) return;
      var rect = ds.actionCanvas.getBoundingClientRect();
      var padL = 44, padR = 8;
      var x0 = src.indices[0], x1 = src.indices[src.indices.length - 1];
      var t = (ev.clientX - rect.left - padL) / (rect.width - padL - padR);
      var aidx = Math.round(x0 + t * (x1 - x0));
      seekFrame(Math.floor((aidx - 1) / ds.ratio));
    });
  }

  function updateTimelineTip(ev) {
    if (!ds.info || !ds.timeline) return;
    var rect = ds.timeline.getBoundingClientRect();
    if (ev.clientX < rect.left || ev.clientX > rect.right ||
        ev.clientY < rect.top || ev.clientY > rect.bottom) {
      ds.tlTip.classList.add("hidden");
      return;
    }
    var pad = 8, w = rect.width - pad * 2;
    var vc = ds.info.video_count;
    var f = Math.round(((ev.clientX - rect.left - pad) / w) * (vc - 1));
    var hit = null;
    (ds.info.texts || []).forEach(function (t) {
      if (t.frame_range && f >= t.frame_range[0] && f <= t.frame_range[1]) hit = t;
    });
    if (hit) {
      ds.tlTip.textContent = hit.kind + "：" + hit.text + " [" + hit.frame_range[0] + ", " + hit.frame_range[1] + "]";
      ds.tlTip.style.left = Math.min(rect.width - 160, Math.max(0, ev.clientX - rect.left + 10)) + "px";
      ds.tlTip.style.top = "2px";
      ds.tlTip.classList.remove("hidden");
    } else {
      ds.tlTip.classList.add("hidden");
    }
  }

  function drawTimeline() {
    if (!ds.info || !ds.timeline) return;
    var vc = ds.info.video_count;
    var env = setupCanvas(ds.timeline, 64);
    var ctx = env.ctx, w = env.w, h = env.h;
    var pad = 8;
    function x(f) { return pad + (f / Math.max(1, vc - 1)) * (w - pad * 2); }

    ctx.fillStyle = cssVar("--card") || "#161e2e";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = cssVar("--border") || "#2a3550";
    ctx.strokeRect(x(0), 30, x(vc - 1) - x(0), 14);

    // 文本段（连续同文段合并后绘制）+ 全局指令底纹（无 video_range 的文本）
    var merged = mergedTexts(ds.info.texts);
    var hasGlobal = false;
    merged.forEach(function (t) {
      if (!t.ranges) { hasGlobal = true; return; }
      ctx.fillStyle = KIND_COLORS[t.kind] || "#5c6880";
      ctx.globalAlpha = 0.75;
      t.ranges.forEach(function (r) {
        ctx.fillRect(x(r[0]), 30, Math.max(2, x(r[1]) - x(r[0])), 14);
      });
      ctx.globalAlpha = 1;
    });
    if (hasGlobal) {
      ctx.fillStyle = "#4f8cff";
      ctx.globalAlpha = 0.15;
      ctx.fillRect(x(0), 16, x(vc - 1) - x(0), 8);
      ctx.globalAlpha = 1;
    }

    // 切片窗口高亮
    if (ds.slice) {
      ctx.fillStyle = cssVar("--accent") || "#4f8cff";
      ctx.globalAlpha = 0.22;
      ctx.fillRect(x(ds.slice.start), 24, x(ds.slice.start + ds.slice.n) - x(ds.slice.start), 26);
      ctx.globalAlpha = 1;
    }
    // flash 区间
    if (ds.flashRange && Date.now() < ds.flashRange.until) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(x(ds.flashRange.range[0]), 28, x(ds.flashRange.range[1]) - x(ds.flashRange.range[0]), 18);
      ctx.lineWidth = 1;
    }

    // 刻度
    ctx.fillStyle = cssVar("--faint") || "#5c6880";
    ctx.font = "10px " + (cssVar("--mono") || "monospace");
    var ticks = 8;
    for (var i = 0; i <= ticks; i++) {
      var f = Math.round((vc - 1) * i / ticks);
      ctx.fillText(String(f), x(f) - 6, 58);
      ctx.fillRect(x(f), 44, 1, 4);
    }
    // action 比例尺说明
    ctx.fillText("action idx ×" + ds.ratio, pad, 14);

    // 游标
    ctx.strokeStyle = cssVar("--accent") || "#4f8cff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x(ds.curFrame), 20);
    ctx.lineTo(x(ds.curFrame), 52);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  function drawAction() {
    if (!ds.actionCanvas || !ds.info) return;
    var src = (ds.slice && ds.sliceAction) || ds.action;
    var env = setupCanvas(ds.actionCanvas, 220);
    var ctx = env.ctx, w = env.w, h = env.h;
    ctx.fillStyle = cssVar("--card") || "#161e2e";
    ctx.fillRect(0, 0, w, h);
    if (!src || !src.indices.length || !src.d) {
      ctx.fillStyle = cssVar("--muted") || "#8b96ad";
      ctx.fillText(src ? "无 action 数据" : "action 加载中…", 10, 20);
      return;
    }
    var padL = 44, padR = 8, padT = 10, padB = 18;
    var x0 = src.indices[0], x1 = src.indices[src.indices.length - 1];
    var n = src.indices.length;
    function X(i) { return padL + (i / Math.max(1, n - 1)) * (w - padL - padR); }

    // y 范围
    var mn = Infinity, mx = -Infinity;
    var d, r, c;
    var dimMin = [], dimMax = [];
    for (c = 0; c < src.d; c++) { dimMin.push(Infinity); dimMax.push(-Infinity); }
    for (r = 0; r < n; r++) {
      for (d = 0; d < ds.dimsOn.length; d++) {
        c = ds.dimsOn[d];
        var v = src.data[r * src.d + c];
        if (v < dimMin[c]) dimMin[c] = v;
        if (v > dimMax[c]) dimMax[c] = v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!isFinite(mn)) { mn = -1; mx = 1; }
    if (mx - mn < 1e-9) { mx = mn + 1; }
    function Y(v, c) {
      var lo = ds.normalize ? dimMin[c] : mn;
      var hi = ds.normalize ? dimMax[c] : mx;
      if (hi - lo < 1e-9) hi = lo + 1;
      return padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);
    }

    // 网格 + y 轴
    ctx.strokeStyle = cssVar("--chart-grid") || "rgba(255,255,255,0.06)";
    ctx.fillStyle = cssVar("--faint") || "#5c6880";
    ctx.font = "10px " + (cssVar("--mono") || "monospace");
    for (var g = 0; g <= 4; g++) {
      var gy = padT + (h - padT - padB) * g / 4;
      ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(w - padR, gy); ctx.stroke();
      var val = mx - (mx - mn) * g / 4;
      ctx.fillText(util.fmtAxis(val), 4, gy + 3);
    }
    // x 轴标签（绝对 action index）
    for (var t = 0; t <= 4; t++) {
      var ai = Math.round(x0 + (x1 - x0) * t / 4);
      ctx.fillText(String(ai), padL + (w - padL - padR) * t / 4 - 8, h - 4);
    }

    // 当前帧对应 action 区间着色
    if (!ds.slice) {
      var a0 = ds.curFrame * ds.ratio + 1, a1 = (ds.curFrame + 1) * ds.ratio;
      if (a1 >= x0 && a0 <= x1) {
        var ia = Math.max(0, Math.round((a0 - x0) / Math.max(1, x1 - x0) * (n - 1)));
        var ib = Math.min(n - 1, Math.round((a1 - x0) / Math.max(1, x1 - x0) * (n - 1)));
        ctx.fillStyle = cssVar("--accent") || "#4f8cff";
        ctx.globalAlpha = 0.18;
        ctx.fillRect(X(ia), padT, Math.max(2, X(ib) - X(ia)), h - padT - padB);
        ctx.globalAlpha = 1;
      }
    }

    // 各维折线
    ds.dimsOn.forEach(function (c) {
      ctx.strokeStyle = MonApp.PALETTE[c % MonApp.PALETTE.length];
      ctx.beginPath();
      for (var r2 = 0; r2 < n; r2++) {
        var y = Y(src.data[r2 * src.d + c], c);
        if (r2 === 0) ctx.moveTo(X(r2), y); else ctx.lineTo(X(r2), y);
      }
      ctx.stroke();
    });
  }

  /* ------------------------------------------------------------ 导出 */
  MonApp.datasetTab = {
    refresh: refresh,
    openEpisode: openEpisode,
  };
})();
