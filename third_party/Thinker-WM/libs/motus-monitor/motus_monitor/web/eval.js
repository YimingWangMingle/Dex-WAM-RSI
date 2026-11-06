/* Motus Monitor 前端 —— 评估 tab（训练中周期性评测产物展示）。
 *
 * 数据源：训练侧 periodic_eval 包每 N 步评一轮（所有 rank 各评一个 batch），
 * rank0 把索引记录追加到 runs/<id>/eval/*.jsonl（append-only 分块 JSONL，
 * 与 metrics/events 平级的新流）；媒体文件（首帧 PNG、condition|gt|pred 对比
 * MP4、action 曲线 PNG、meta.json）写在同一 run 目录下，前端经
 * /api/runs/{id}/file?path=<相对路径> 取字节。
 *
 * 流内两种记录：
 *   eval_round  —— 一轮概要（step/样本数/源分布/平均指标/耗时）；
 *   eval_sample —— 单样本（source/robot/episode/任务文本/vlm_prompt/vlm_output/
 *                  逐样本指标/媒体相对路径）。
 * resume 重跑会在同一 step 重复写记录（append-only 不改历史）：eval_round 按
 * step 去重、eval_sample 按 (step, rank, sample_index) 去重，均后写胜出——
 * 与 step 记录同一约定。
 */
window.MonApp = window.MonApp || {};

(function () {
  "use strict";

  var api, util, esc, el;

  var ev = {
    initialized: false,
    runId: null,
    offset: null,        // eval 流增量读取 offset（"chunk:pos"，null = 从 0:0 全量）
    _partial: "",
    _busy: false,
    error: null,
    roundMap: {},        // step -> eval_round（后写胜出）
    sampleMap: {},       // step -> { "rank:idx" -> eval_sample（后写胜出）}
    steps: [],           // 有记录的评测轮 step（升序）
    curStep: null,       // 当前展示的评测轮（null = 暂无数据）
    sourceFilter: "",    // "" = 全部源
  };

  /* ------------------------------------------------------------ 小工具 */
  function fileUrl(relPath) {
    return "/api/runs/" + encodeURIComponent(ev.runId) +
      "/file?path=" + encodeURIComponent(relPath);
  }

  /* 媒体下载：浏览器原生“另存为/视频菜单下载”对 no-store 媒体不可靠
     （Chrome 常报“网络问题”且文件名取 URL 末段 file），统一走
     fetch→blob→a.download（与数据集 tab 下载同一路径），文件名取真实 basename */
  function downloadMedia(relPath, btn) {
    var name = String(relPath || "file").split("/").pop() || "file";
    if (btn) { btn.disabled = true; btn.textContent = "下载中…"; }
    function done() {
      if (btn) { btn.disabled = false; btn.textContent = "下载"; }
    }
    fetch(fileUrl(relPath), { credentials: "same-origin" }).then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return resp.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 30000);
      done();
    }).catch(function (err) {
      done();
      window.alert("下载失败：" + (err && err.message || err));
    });
  }

  function fmtMetric(v) {
    return util.isNum(v) ? util.fmtVal(v) : "-";
  }

  /* 指标对象 → "key value" 片段数组（键排序，null 显示 -） */
  function metricPairs(metrics) {
    var out = [];
    if (!metrics || typeof metrics !== "object") return out;
    Object.keys(metrics).sort().forEach(function (k) {
      out.push({ key: k, value: metrics[k] });
    });
    return out;
  }

  /* vlm_prompt / vlm_output 可能是字符串或消息对象，统一成展示文本 */
  function asText(v) {
    if (v == null) return null;
    if (typeof v === "string") return v;
    try { return JSON.stringify(v, null, 2); } catch (e) { return String(v); }
  }

  /* ------------------------------------------------------------ 数据摄入 */
  function ingest(records) {
    var prevLatest = ev.steps.length ? ev.steps[ev.steps.length - 1] : null;
    var touched = false;
    for (var i = 0; i < records.length; i++) {
      var r = records[i];
      if (!r || typeof r !== "object") continue;
      var step = Number(r.step);
      if (!isFinite(step)) continue;
      if (r.type === "eval_round") {
        ev.roundMap[step] = r;
        touched = true;
      } else if (r.type === "eval_sample") {
        var bucket = ev.sampleMap[step];
        if (!bucket) bucket = ev.sampleMap[step] = {};
        bucket[String(r.rank) + ":" + String(r.sample_index)] = r;
        touched = true;
      }
    }
    if (!touched) return false;
    /* 重建升序 step 列表（round/sample 任一存在即算一轮） */
    var seen = {};
    Object.keys(ev.roundMap).forEach(function (s) { seen[s] = true; });
    Object.keys(ev.sampleMap).forEach(function (s) { seen[s] = true; });
    ev.steps = Object.keys(seen).map(Number).sort(function (a, b) { return a - b; });
    /* 选中轮跟随：未选过 或 之前停在最新轮 → 跳到最新；用户翻到旧轮则不动 */
    if (ev.steps.length &&
        (ev.curStep === null || (prevLatest !== null && ev.curStep === prevLatest))) {
      ev.curStep = ev.steps[ev.steps.length - 1];
    }
    return true;
  }

  /* 增量拉 eval 流（首次从 0:0 全量，之后按 offset 增量）；返回 Promise<bool> */
  function poll() {
    if (!api || !ev.runId || ev._busy) return Promise.resolve(false);
    ev._busy = true;
    var url = "/api/runs/" + encodeURIComponent(ev.runId) +
      "/metrics?stream=eval&offset=" + encodeURIComponent(ev.offset || "0:0");
    return api.get(url).then(function (resp) {
      ev.error = null;
      if (resp.offset) ev.offset = resp.offset;
      var text = ev._partial + (resp.data || "");
      var records = [];
      if (text) {
        var parts = text.split("\n");
        /* 写入端按行落盘，最后一段若不是整行留待下次拼接（同 api.streamReader） */
        if (text.charAt(text.length - 1) !== "\n") {
          ev._partial = parts.pop();
        } else {
          ev._partial = "";
        }
        for (var i = 0; i < parts.length; i++) {
          var line = parts[i];
          if (!line) continue;
          line = line.trim();
          if (!line) continue;
          try { records.push(JSON.parse(line)); } catch (e) { /* 跳过坏行 */ }
        }
      }
      return records.length ? ingest(records) : false;
    }).catch(function (err) {
      ev.error = err;
      return false;
    }).then(function (changed) {
      ev._busy = false;
      if (changed) {
        /* 用户正在选文字/操作下拉时推迟重建（重建会清选区、收起下拉） */
        if (MonApp.domRenderBlocked && MonApp.domRenderBlocked()) ev.renderPending = true;
        else renderAll();
      } else {
        renderHint();
      }
      return changed;
    });
  }

  function reset(runId) {
    ev.runId = runId;
    ev.offset = null;
    ev._partial = "";
    ev.error = null;
    ev.roundMap = {};
    ev.sampleMap = {};
    ev.steps = [];
    ev.curStep = null;
    ev.sourceFilter = "";
  }

  /* ------------------------------------------------------------ 渲染 */
  function roundSamples(step) {
    var bucket = ev.sampleMap[step] || {};
    var out = [];
    Object.keys(bucket).forEach(function (k) { out.push(bucket[k]); });
    out.sort(function (a, b) {
      var ra = Number(a.rank), rb = Number(b.rank);
      if (ra !== rb) return (isFinite(ra) ? ra : 0) - (isFinite(rb) ? rb : 0);
      var sa = Number(a.sample_index), sb = Number(b.sample_index);
      return (isFinite(sa) ? sa : 0) - (isFinite(sb) ? sb : 0);
    });
    return out;
  }

  function renderHint() {
    var hint = document.getElementById("eval-hint");
    if (!hint) return;
    if (ev.error) {
      hint.textContent = "读取失败：" + ev.error.message;
    } else if (!ev.runId) {
      hint.textContent = "";
    } else if (!ev.steps.length) {
      hint.textContent = ev._busy ? "加载中…" : "";
    } else {
      var idx = ev.steps.indexOf(ev.curStep);
      hint.textContent = "第 " + (idx + 1) + " / " + ev.steps.length + " 轮";
    }
  }

  function renderRoundControls() {
    var sel = document.getElementById("eval-round");
    var prev = document.getElementById("eval-round-prev");
    var next = document.getElementById("eval-round-next");
    sel.innerHTML = "";
    /* 下拉最新一轮在最上 */
    for (var i = ev.steps.length - 1; i >= 0; i--) {
      var step = ev.steps[i];
      var round = ev.roundMap[step];
      var label = "step " + step + (round && round.ts ? " · " + util.fmtDateTime(round.ts) : "");
      sel.appendChild(el("option", "", { text: label, attrs: { value: String(step) } }));
    }
    if (ev.curStep !== null) sel.value = String(ev.curStep);
    var idx = ev.steps.indexOf(ev.curStep);
    prev.disabled = idx <= 0;
    next.disabled = idx < 0 || idx >= ev.steps.length - 1;

    /* 源过滤器：取当轮样本与 round.sources 的并集 */
    var srcSel = document.getElementById("eval-source");
    var names = {};
    var samples = roundSamples(ev.curStep);
    samples.forEach(function (s) { if (s.source) names[s.source] = true; });
    var round2 = ev.roundMap[ev.curStep];
    if (round2 && round2.sources) {
      Object.keys(round2.sources).forEach(function (k) { names[k] = true; });
    }
    var sorted = Object.keys(names).sort();
    if (ev.sourceFilter && sorted.indexOf(ev.sourceFilter) < 0) ev.sourceFilter = "";
    srcSel.innerHTML = "";
    srcSel.appendChild(el("option", "", { text: "全部源", attrs: { value: "" } }));
    sorted.forEach(function (name) {
      srcSel.appendChild(el("option", "", { text: name, attrs: { value: name } }));
    });
    srcSel.value = ev.sourceFilter;
    renderHint();
  }

  function metricRowEl(metrics) {
    var row = el("div", "eval-metrics");
    var pairs = metricPairs(metrics);
    if (!pairs.length) {
      row.appendChild(el("span", "eval-metric muted", { text: "无指标" }));
      return row;
    }
    pairs.forEach(function (p) {
      var item = el("span", "eval-metric");
      item.appendChild(el("span", "eval-metric-key", { text: p.key }));
      item.appendChild(el("span", "eval-metric-val mono", { text: fmtMetric(p.value) }));
      row.appendChild(item);
    });
    return row;
  }

  function renderSummary() {
    var box = document.getElementById("eval-summary");
    box.innerHTML = "";
    if (!ev.runId) {
      box.appendChild(el("div", "card", {
        html: '<div class="empty-hint">暂无 run（训练侧开启周期性评测后在此展示）</div>',
      }));
      return;
    }
    if (!ev.steps.length) {
      box.appendChild(el("div", "card", {
        html: '<div class="empty-hint">暂无评测数据' +
          (ev.error ? '<div class="eval-error">' + esc(ev.error.message) + "</div>" : "") +
          "</div>",
      }));
      return;
    }
    var round = ev.roundMap[ev.curStep] || null;
    var samples = roundSamples(ev.curStep);

    var card = el("div", "card eval-summary");
    var head = el("div", "eval-summary-head");
    head.appendChild(el("span", "eval-summary-title", { text: "step " + ev.curStep }));
    if (round && round.ts) {
      head.appendChild(el("span", "muted", { text: util.fmtDateTime(round.ts) }));
    }
    if (round && util.isNum(round.seconds)) {
      head.appendChild(el("span", "muted", { text: "耗时 " + util.fmtDuration(round.seconds) }));
    }
    if (round && round.ready_ranks != null && round.world_size != null) {
      head.appendChild(el("span", "muted", {
        text: "rank " + round.ready_ranks + "/" + round.world_size,
      }));
    }
    card.appendChild(head);

    /* 样本数 + 源分布 chips（round 记录缺省时从样本聚合） */
    var chips = el("div", "eval-chips");
    chips.appendChild(el("span", "eval-chip", {
      text: "样本 " + (round && round.num_samples != null ? round.num_samples : samples.length),
    }));
    var sources = round && round.sources;
    if (!sources) {
      sources = {};
      samples.forEach(function (s) {
        var name = s.source || "unknown";
        sources[name] = (sources[name] || 0) + 1;
      });
    }
    Object.keys(sources).sort().forEach(function (name) {
      chips.appendChild(el("span", "eval-chip eval-chip-src", {
        text: name + " × " + sources[name],
      }));
    });
    card.appendChild(chips);

    /* 平均指标 */
    if (round && round.metrics) {
      card.appendChild(metricRowEl(round.metrics));
    }
    box.appendChild(card);
  }

  /* ------------------------------------------------------------ 媒体弹层 */
  var _modalRoot = null;          // 懒创建单例
  var _modalEscHandler = null;    // 当前打开的 Esc 监听
  var _modalCleanup = null;       // 关闭时的清理钩子（action 交互查看器 destroy）
  var _actionHandle = null;       // 打开中的 ActionPlot 交互查看器 handle

  function ensureModal() {
    if (_modalRoot) return _modalRoot;
    var root = el("div", "eval-modal");
    var backdrop = el("div", "eval-modal-backdrop");
    var box = el("div", "eval-modal-box");
    var head = el("div", "eval-modal-head");
    head.appendChild(el("span", "eval-modal-title"));
    var close = el("button", "eval-modal-close", {
      text: "×",
      attrs: { type: "button", title: "关闭（Esc）" },
    });
    close.addEventListener("click", closeMediaModal);
    head.appendChild(close);
    var body = el("div", "eval-modal-body");
    box.appendChild(head);
    box.appendChild(body);
    backdrop.appendChild(box);
    root.appendChild(backdrop);
    document.body.appendChild(root);

    backdrop.addEventListener("click", function (e) {
      if (e.target === backdrop) closeMediaModal();
    });
    _modalRoot = root;
    return root;
  }

  function closeMediaModal() {
    if (!_modalRoot) return;
    /* 先跑清理钩子（交互查看器摘 window/document 监听、销毁 LineChart） */
    if (_modalCleanup) {
      try { _modalCleanup(); } catch (e) {}
      _modalCleanup = null;
    }
    _actionHandle = null;
    _modalRoot.classList.remove("open");
    var box = _modalRoot.querySelector(".eval-modal-box");
    if (box) box.classList.remove("action-wide");
    /* 媒体专用「下载」按钮不跨弹层残留 */
    var head = _modalRoot.querySelector(".eval-modal-head");
    var dl = head ? head.querySelector(".eval-modal-download") : null;
    if (dl) dl.parentNode.removeChild(dl);
    /* 清空 body 即移除媒体元素：停止解码/下载，释放内存 */
    var body = _modalRoot.querySelector(".eval-modal-body");
    if (body) body.innerHTML = "";
    if (_modalEscHandler) {
      document.removeEventListener("keydown", _modalEscHandler);
      _modalEscHandler = null;
    }
    document.body.classList.remove("eval-modal-open");
  }

  /* Esc：action 查看器单维聚焦时先返回网格（handleEsc 消费），否则关弹层 */
  function ensureEscHandler() {
    if (_modalEscHandler) return;
    _modalEscHandler = function (e) {
      if (e.key !== "Escape") return;
      if (_actionHandle && _actionHandle.handleEsc && _actionHandle.handleEsc()) return;
      closeMediaModal();
    };
    document.addEventListener("keydown", _modalEscHandler);
  }

  function openMediaModal(relPath, kind, title) {
    var root = ensureModal();
    closeMediaModal(); // 清掉可能残留（重试路径），随后重新打开
    root.classList.add("open");
    document.body.classList.add("eval-modal-open");
    root.querySelector(".eval-modal-title").textContent =
      title || (kind === "video" ? "视频" : "图片");

    /* 弹层头部放显式“下载”按钮（走 fetch→blob，见 downloadMedia 注释；
     * 上一个弹层残留的按钮已在 closeMediaModal 清掉） */
    var head = root.querySelector(".eval-modal-head");
    var dl = el("button", "btn eval-modal-download", {
      text: "下载", attrs: { type: "button", title: "下载该媒体文件" },
    });
    dl.addEventListener("click", function () { downloadMedia(relPath, dl); });
    head.insertBefore(dl, head.querySelector(".eval-modal-close"));

    var body = root.querySelector(".eval-modal-body");
    var loading = el("div", "eval-modal-loading");
    body.appendChild(loading);

    function hideError() {
      var err = body.querySelector(".eval-modal-error");
      if (err) err.classList.add("hidden");
    }
    function showError() {
      var err = body.querySelector(".eval-modal-error");
      if (!err) {
        err = el("div", "eval-modal-error");
        err.appendChild(el("span", "", { text: "媒体加载失败（文件缺失或服务异常）" }));
        var retry = el("button", "btn", { text: "重试" });
        retry.addEventListener("click", function () {
          openMediaModal(relPath, kind, title);
        });
        err.appendChild(retry);
        body.appendChild(err);
      }
      err.classList.remove("hidden");
    }

    /* 按需创建媒体元素并设 src —— 只有此时才发起请求 */
    var node = (kind === "video")
      ? el("video", "eval-modal-video", {
          attrs: { controls: "controls", autoplay: "autoplay", muted: "muted",
                   loop: "loop", src: fileUrl(relPath) },
        })
      : el("img", "eval-modal-img", {
          attrs: { src: fileUrl(relPath), alt: title || "" },
        });
    body.appendChild(node);

    var settled = false;
    function onLoad() {
      if (settled) return;
      settled = true;
      loading.style.display = "none";
      hideError();
    }
    function onError() {
      if (settled) return;
      settled = true;
      loading.style.display = "none";
      if (node.parentNode) node.parentNode.removeChild(node);
      showError();
    }
    node.addEventListener("load", onLoad);
    node.addEventListener("error", onError);
    /* video 的 load 在元数据就绪后即触发；loadeddata 表示首帧可播，更稳 */
    if (kind === "video") node.addEventListener("loadeddata", onLoad);

    ensureEscHandler();
  }

  /* action 交互大图弹层：复用弹层壳，body 挂 ActionPlot 交互查看器
   * （联动十字线/缩放平移/单维聚焦等，见 actionplot.js openInteractive）。
   * 无「下载」按钮——导出由查看器 toolbar 的 CSV/PNG 承担。 */
  function openActionModal(metaUrl, title) {
    if (!MonApp.ActionPlot || typeof MonApp.ActionPlot.openInteractive !== "function") return;
    var root = ensureModal();
    closeMediaModal(); // 清残留（含旧 cleanup），随后重新打开
    root.classList.add("open");
    document.body.classList.add("eval-modal-open");
    root.querySelector(".eval-modal-box").classList.add("action-wide");
    root.querySelector(".eval-modal-title").textContent = title || "actions（pred vs GT）";

    var body = root.querySelector(".eval-modal-body");
    var handle = MonApp.ActionPlot.openInteractive(body, metaUrl, { title: title || "actions" });
    _actionHandle = handle;
    _modalCleanup = function () { handle.destroy(); };
    ensureEscHandler();
  }

  /* 占位按钮（视频用；页面打开时不发起媒体请求，点击后在弹层中按需加载播放） */
  function mediaPlaceholder(relPath, kind, title) {
    var btn = el("button", "eval-media-thumb", {
      attrs: { type: "button", title: "点击加载并放大查看" },
    });
    btn.appendChild(el("span", "eval-media-thumb-icon", {
      text: kind === "video" ? "▶" : "▦",
    }));
    if (title) {
      btn.appendChild(el("span", "eval-media-thumb-title", { text: title }));
    }
    btn.appendChild(el("span", "eval-media-thumb-hint", { text: "点击加载并放大查看" }));
    btn.addEventListener("click", function () {
      openMediaModal(relPath, kind, title);
    });
    return btn;
  }

  /* 单个媒体控件：
   * - 图片：直接渲染真实缩略图（原图 + CSS 缩放，loading=lazy 滚动到视口才拉取），
   *   点击弹层查看原图；加载失败（文件缺失）时回退为占位按钮。
   * - 视频：占位按钮不设 src，点击后在弹层中按需加载播放。 */
  function mediaEl(relPath, kind, title) {
    if (!relPath) return null;
    var wrap = el("div", "eval-media-item");
    if (kind === "img") {
      var img = el("img", "eval-media-thumb-img", {
        attrs: {
          src: fileUrl(relPath),
          alt: title || "image",
          loading: "lazy",
          title: "点击查看原图",
        },
      });
      img.addEventListener("click", function () {
        openMediaModal(relPath, "img", title);
      });
      img.addEventListener("error", function () {
        wrap.innerHTML = "";
        wrap.appendChild(mediaPlaceholder(relPath, kind, title));
      });
      wrap.appendChild(img);
      if (title) {
        wrap.appendChild(el("span", "eval-media-thumb-title", { text: title }));
      }
      return wrap;
    }
    wrap.appendChild(mediaPlaceholder(relPath, kind, title));
    return wrap;
  }

  function renderSampleCard(s) {
    var card = el("div", "card eval-card");

    /* 标题行：source / robot / episode / rank 定位 + video-only 标记 */
    var head = el("div", "eval-card-head");
    if (s.source) head.appendChild(el("span", "eval-chip eval-chip-src", { text: s.source }));
    var titleParts = [];
    if (s.robot_name) titleParts.push(s.robot_name);
    if (s.episode_id) titleParts.push(s.episode_id);
    if (titleParts.length) {
      head.appendChild(el("span", "eval-card-title", { text: titleParts.join(" · ") }));
    }
    head.appendChild(el("span", "eval-card-loc mono muted", {
      text: "rank" + (s.rank != null ? s.rank : "?") + "/sample_" +
        (s.sample_index != null ? s.sample_index : "?"),
    }));
    if (s.video_only) {
      head.appendChild(el("span", "eval-badge", { text: "video-only（无 action GT）" }));
    }
    card.appendChild(head);

    card.appendChild(metricRowEl(s.metrics));

    /* 任务文本 */
    var taskText = asText(s.task_text);
    if (taskText) {
      var task = el("div", "eval-block");
      task.appendChild(el("div", "eval-block-label", { text: "任务" }));
      task.appendChild(el("div", "eval-block-text", { text: taskText }));
      card.appendChild(task);
    }

    /* vlm_prompt（可折叠） */
    var prompt = asText(s.vlm_prompt);
    if (prompt) {
      var det = el("details", "eval-prompt");
      det.appendChild(el("summary", "", { text: "vlm_prompt" }));
      det.appendChild(el("pre", "eval-pre", { text: prompt }));
      card.appendChild(det);
    }

    /* vlm_output（高亮块；null = 未生成） */
    var out = el("div", "eval-block");
    out.appendChild(el("div", "eval-block-label", { text: "vlm_output" }));
    var vlmOut = asText(s.vlm_output);
    if (vlmOut) {
      out.appendChild(el("pre", "eval-pre eval-vlm-out", { text: vlmOut }));
    } else {
      out.appendChild(el("div", "eval-block-text muted", { text: "未生成" }));
    }
    card.appendChild(out);

    /* initial_state（可折叠；数组按 4 位小数展示） */
    if (Array.isArray(s.initial_state) && s.initial_state.length) {
      var stateDet = el("details", "eval-prompt");
      stateDet.appendChild(el("summary", "", {
        text: "initial_state（" + s.initial_state.length + " 维）",
      }));
      var stateText = s.initial_state.map(function (v) {
        return typeof v === "number" ? v.toFixed(4) : String(v);
      }).join(", ");
      stateDet.appendChild(el("pre", "eval-pre", { text: stateText }));
      card.appendChild(stateDet);
    }

    /* 媒体：首帧 / 对比视频 / gt / pred */
    var files = s.files || {};
    var media = el("div", "eval-media");
    var items = [
      mediaEl(files.first_frame, "img", "first_frame"),
      mediaEl(files.compare_video, "video", "condition | gt | pred"),
      mediaEl(files.gt_video, "video", "gt"),
      mediaEl(files.pred_video, "video", "pred"),
    ];
    var hasMedia = false;
    items.forEach(function (item) {
      if (item) { media.appendChild(item); hasMedia = true; }
    });
    if (hasMedia) card.appendChild(media);

    /* action 逐维曲线：canvas 渲染（数据在 meta.json；旧 run 回退 actions.png）。
     * 容器挂载在 renderSamples 末尾统一做（ActionPlot.mount 懒加载）。
     * 有 meta.json 时点击进入交互大图弹层（openActionModal）。 */
    if (files.meta || files.actions_plot) {
      var plotWrap = el("div", "eval-block");
      var plotLabel = el("div", "eval-block-label", { text: "actions（pred vs GT）" });
      if (files.meta) {
        plotLabel.appendChild(el("span", "muted", { text: " · 点击放大交互查看" }));
      }
      plotWrap.appendChild(plotLabel);
      var plotBox = el("div", "eval-action-plot");
      if (files.meta) {
        plotBox.setAttribute("data-meta-url", fileUrl(files.meta));
        plotBox.classList.add("clickable");
        plotBox.title = "点击放大交互查看";
        plotBox.addEventListener("click", function () {
          var sampleTitle = titleParts.join(" · ") || s.source || "";
          openActionModal(fileUrl(files.meta), "actions · " + sampleTitle);
        });
      }
      if (files.actions_plot) plotBox.setAttribute("data-fallback-img", fileUrl(files.actions_plot));
      plotWrap.appendChild(plotBox);
      card.appendChild(plotWrap);
    }
    if (files.meta) {
      var link = el("a", "eval-meta-link mono", {
        text: "meta.json",
        attrs: { href: fileUrl(files.meta), target: "_blank", rel: "noopener" },
      });
      card.appendChild(link);
    }
    return card;
  }

  function renderSamples() {
    var box = document.getElementById("eval-samples");
    box.innerHTML = "";
    /* 旧卡片 DOM 已销毁：断开其上 observer 并清空 action 曲线挂载表 */
    if (MonApp.ActionPlot) MonApp.ActionPlot.reset();
    if (!ev.runId || !ev.steps.length) return;
    var samples = roundSamples(ev.curStep);
    if (ev.sourceFilter) {
      samples = samples.filter(function (s) { return s.source === ev.sourceFilter; });
    }
    if (!samples.length) {
      box.appendChild(el("div", "card", {
        html: '<div class="empty-hint">该轮暂无样本记录' +
          (ev.sourceFilter ? "（源过滤：" + esc(ev.sourceFilter) + "）" : "") + "</div>",
      }));
      return;
    }
    var grid = el("div", "eval-sample-grid");
    samples.forEach(function (s) { grid.appendChild(renderSampleCard(s)); });
    box.appendChild(grid);
    /* action 曲线挂载（IntersectionObserver 懒加载 meta.json 后绘制） */
    if (MonApp.ActionPlot) {
      box.querySelectorAll(".eval-action-plot").forEach(function (c) {
        MonApp.ActionPlot.mount(
          c,
          c.getAttribute("data-meta-url"),
          c.getAttribute("data-fallback-img")
        );
      });
    }
  }

  function renderAll() {
    renderRoundControls();
    renderSummary();
    renderSamples();
  }

  /* ------------------------------------------------------------ run 列表 */
  function loadRuns() {
    return api.get("/api/runs").then(function (payload) {
      var runs = (payload && payload.runs) || [];
      var sel = document.getElementById("eval-run");
      var keep = ev.runId;
      sel.innerHTML = "";
      runs.forEach(function (r) {
        var label = (r.run_name || r.run_id) +
          " · " + (r.last_step != null ? ("step " + r.last_step) : "");
        sel.appendChild(el("option", "", { text: label, attrs: { value: r.run_id } }));
      });
      if (!runs.length) {
        reset(null);
        renderAll();
        return false;
      }
      var found = null;
      if (keep) {
        for (var i = 0; i < runs.length; i++) {
          if (runs[i].run_id === keep) { found = keep; break; }
        }
      }
      if (!found) {
        /* 与总览 run 选择器对齐（localStorage 持久化的当前 run），否则取第一个 */
        var stored = null;
        try { stored = localStorage.getItem("motus.monitor.runId"); } catch (e) {}
        for (var j = 0; j < runs.length; j++) {
          if (runs[j].run_id === stored) { found = stored; break; }
        }
        if (!found) found = runs[0].run_id;
      }
      sel.value = found;
      if (found !== ev.runId) {
        reset(found);
        renderAll();
        return poll();
      }
      return false;
    }).catch(function (err) {
      ev.error = err;
      renderAll();
      return false;
    });
  }

  /* ------------------------------------------------------------ 事件绑定 */
  function bindOnce() {
    if (ev.initialized) return;
    ev.initialized = true;

    document.getElementById("eval-run").addEventListener("change", function (e) {
      reset(e.target.value || null);
      renderAll();
      poll();
    });
    document.getElementById("eval-round").addEventListener("change", function (e) {
      var step = Number(e.target.value);
      if (isFinite(step)) {
        ev.curStep = step;
        renderAll();
      }
    });
    document.getElementById("eval-round-prev").addEventListener("click", function () {
      var idx = ev.steps.indexOf(ev.curStep);
      if (idx > 0) {
        ev.curStep = ev.steps[idx - 1];
        renderAll();
      }
    });
    document.getElementById("eval-round-next").addEventListener("click", function () {
      var idx = ev.steps.indexOf(ev.curStep);
      if (idx >= 0 && idx < ev.steps.length - 1) {
        ev.curStep = ev.steps[idx + 1];
        renderAll();
      }
    });
    document.getElementById("eval-source").addEventListener("change", function (e) {
      ev.sourceFilter = e.target.value || "";
      renderSamples();
    });
    document.getElementById("eval-refresh").addEventListener("click", function () {
      loadRuns().then(poll);
    });

    /* y 轴范围编辑器：action 曲线 band 级固定范围覆盖（localStorage 持久化，
     * 应用后 ActionPlot 重绘全部已挂载图表，不重建 DOM） */
    var yEditor = document.getElementById("eval-y-editor");
    if (yEditor && MonApp.ActionPlot) {
      var yBands = ["pos", "rot6", "gripper", "other"];
      var fillYInputs = function () {
        var ranges = MonApp.ActionPlot.getRanges();
        yBands.forEach(function (b) {
          document.getElementById("eval-y-" + b + "-min").value = ranges[b][0];
          document.getElementById("eval-y-" + b + "-max").value = ranges[b][1];
        });
      };
      yEditor.addEventListener("toggle", function () {
        if (yEditor.open) fillYInputs();
      });
      document.getElementById("eval-y-apply").addEventListener("click", function () {
        yBands.forEach(function (b) {
          var lo = parseFloat(document.getElementById("eval-y-" + b + "-min").value);
          var hi = parseFloat(document.getElementById("eval-y-" + b + "-max").value);
          if (isFinite(lo) && isFinite(hi) && lo < hi) {
            MonApp.ActionPlot.setBandRange(b, [lo, hi]);
          }
        });
        fillYInputs();
      });
      document.getElementById("eval-y-reset").addEventListener("click", function () {
        MonApp.ActionPlot.resetBandRanges();
        fillYInputs();
      });
    }

    /* 交互结束后补一次被推迟的渲染 */
    var flushPending = function () {
      if (!ev.renderPending) return;
      if (MonApp.domRenderBlocked && MonApp.domRenderBlocked()) return;
      ev.renderPending = false;
      renderAll();
    };
    document.addEventListener("selectionchange", flushPending);
    document.addEventListener("change", flushPending);
    document.addEventListener("blur", flushPending, true);
  }

  /* ------------------------------------------------------------ 入口 */
  function refresh() {
    api = MonApp.api; util = MonApp.util; esc = util.esc; el = util.el;
    bindOnce();
    loadRuns().then(function () { poll(); });
  }

  /* ------------------------------------------------------------ 导出 */
  MonApp.evalTab = {
    refresh: refresh,
    poll: poll,
  };
})();
