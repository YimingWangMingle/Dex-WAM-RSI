/* Motus Monitor 前端 —— 训练集 tab · episode/切片查看器（阶段 1C）。
 * 逻辑逐字迁自旧 web/dataset.js：
 *   小工具      clearUrls(234-236)、viewerEl/showViewerError/showViewerLoading/
 *               attachViewerProgress(256-281；canvas setupCanvas 随阶段 2C 移除)
 *   打开/错误   openEpisode(1059-1107)、cardError(1110-1149)、openSlice(1151-1167)
 *   数据加载    loadAction/loadFilmstrip/loadSelection(1169-1250)
 *   渲染        renderViewer/renderViewerChrome/mkBtn(1253-1360，textContent 正则
 *               改写本阶段保留，阶段 3 才重构)、mergedTexts/kindBadge/globalTextRow/
 *               renderTexts(1362-1445)、renderFilmstrip(1447-1465)、dim toggles(1467-1507)
 *   帧显示      seekFrame/setMainFrame/showFrame + 帧 LRU 缓存/预取(1510-1584)
 *   播放/保存   playWindow/saveFramePng/saveWindowMp4/saveActionCsv/
 *               saveSelectionJson(1586-1707，下载改用 util.saveBlob)
 *   时间轴/曲线 事件绑定/scrub/tooltip(1710-1761)、drawTimeline(1763-1830)、
 *               drawAction(1832-1912)——阶段 2C 已从 canvas 手画换成 ECharts
 *               （charts/base.js createEChart 裸封装；视觉元素与原手画一一对应，
 *               见 buildTimelineOption/buildActionOption 注释）。不用 charts/line.js：
 *               两图都需要 markArea/markLine（文本区间/当前帧区间/游标），且交互
 *               是 scrub/点击 seek 而非缩放平移，复用其序列 API 反而要再开洞。
 * 状态：旧 ds 29-76 中属于查看器的字段 → 下方 v（DOM 引用按阶段 1C 规则改为
 * 模块级变量，不进状态对象）；ds.dataset 跨块共享 → state.js shared.dataset。
 */
import { api } from "../../api.js";
import { util } from "../../util.js";
import { parsePacked, fetchPacked, fetchStream, jpegUrls } from "../../ui/stream.js";
import { createProgress } from "../../ui/progress.js";
import { PALETTE, chartTheme, themeColor, themeColors } from "../../charts/theme.js";
import { createEChart } from "../../charts/base.js";
import { shared, qs } from "./state.js";

var esc = util.esc, el = util.el;

var FRAME_CACHE_MAX = 256;
var KIND_COLORS = {
  episode_context: "#d9a038",
  action_segment: "#3fb96f",
  instruction_en: "#4f8cff",
  instruction_zh: "#58a6ff",
  instruction: "#4f8cff",
};

/* 查看器数据状态（旧 ds 29-76 中属于查看器的部分） */
var v = {
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
  flashRange: null,
  playing: null,           // {url}（旧 ds 字段，保留未用）
  filmIndices: null,       // filmstrip 帧索引（loadFilmstrip 赋值）
};

/* 查看器 DOM 引用（renderViewer 每次重建 DOM 时重新赋值；旧实现塞在 ds 里） */
var mainImg = null;
var frameSpinner = null;
var frameProg = null;
var videoWrap = null;
var frameLabel = null;
var filmstrip = null;
var textsBox = null;
var timeline = null;       // 时间轴图表宿主 div
var tlTip = null;
var dimToggles = null;
var actionCanvas = null;   // action 曲线图表宿主 div（保留旧变量名）

/* ECharts wrapper（renderViewer 重建 DOM 前 dispose 旧实例防泄漏） */
var tlChart = null;
var acChart = null;
var actionError = null;    // action 加载失败文案（图表空态红字；null=正常）

/* ------------------------------------------------------------ 小工具 */
function clearUrls(list) {
  (list || []).forEach(function (u) { try { URL.revokeObjectURL(u); } catch (e) { } });
}

function viewerEl() { return document.getElementById("ds-viewer"); }

function showViewerError(err) {
  viewerEl().innerHTML =
    '<div class="card"><div class="ds-error">加载失败：' + esc(err && err.message || err) + "</div></div>";
}

function showViewerLoading(text) {
  var prog = createProgress({ label: text || "解码中…" });
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

/* ------------------------------------------------------------ Episode 查看器 */
export function openEpisode(dataset, episodeId, slice) {
  v.slice = slice || null;
  var prog = showViewerLoading(slice ? "解码训练切片…" : "解码 episode…");
  api.get("/api/dataset/episode" + qs({ dataset: dataset, id: episodeId })).then(function (info) {
    shared.dataset = dataset;
    v.info = info;
    v.ratio = (slice && slice.ratio) || Math.max(1, Math.round(info.ratio_hint || 1));
    v.curFrame = slice ? slice.start : 0;
    v.flashRange = null;
    clearUrls(v.filmUrls); v.filmUrls = [];
    clearUrls(v.sliceFrames); v.sliceFrames = [];
    v.sliceAction = null; v.sliceSel = null;
    v.actionNames = (info.action_names && info.action_names.length)
      ? info.action_names : null;   // 先拿 episode_info 的，action_state 回来后再校准
    v.sliceActionNames = null;
    v.frameCache.forEach(function (u) { URL.revokeObjectURL(u); });
    v.frameCache.clear();
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
    actionError = msg;
    drawAction();   // 图表空态红字（buildActionOption 的 graphic note）
    if (dimToggles) {
      dimToggles.innerHTML = "";
      dimToggles.appendChild(el("div", "ds-error", { text: msg }));
      dimToggles.appendChild(mkBtn("重试 action", function () {
        dimToggles.innerHTML = "";
        loadAction(shared.dataset, v.info.episode_id).then(function () { }, function (e) { cardError("action", e); });
      }));
    }
    return;
  }
  if (kind === "filmstrip") {
    if (filmstrip) {
      filmstrip.innerHTML = "";
      filmstrip.appendChild(el("div", "ds-error", { text: msg }));
      filmstrip.appendChild(mkBtn("重试视频帧", function () {
        loadFilmstrip(shared.dataset, v.info.episode_id).then(function () { }, function (e) { cardError("filmstrip", e); });
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

export function openSlice(s) {
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
  actionError = null;
  var url = "/api/dataset/action" + qs({ dataset: dataset, id: episodeId, max_points: 2048 });
  return fetchPacked(url).then(function (packed) {
    var m = packed.meta;
    var nA = m.action_indices.length, dA = m.d_action;
    v.action = {
      indices: m.action_indices,
      d: dA,
      data: new Float32Array(packed.blob, 0, nA * dA),
      total: m.action_count,
    };
    v.actionNames = (m.action_names && m.action_names.length === dA) ? m.action_names : null;
    v.dimsOn = [];
    for (var i = 0; i < dA; i++) v.dimsOn.push(i);   // 默认显示全部维度
    drawAction();
    renderDimToggles();
  });
}

function loadFilmstrip(dataset, episodeId, onProgress) {
  var vc = v.info.video_count;
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
    v.filmIndices = packed.meta.indices;
    v.filmUrls = jpegUrls(packed.meta, packed.blob);
    renderFilmstrip();
    showFrame(v.curFrame);
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
    v.sliceFrames = jpegs.map(function (b) {
      return URL.createObjectURL(new Blob([b], { type: "image/jpeg" }));
    });
    var nA = m.n_action * m.d_action;
    v.sliceAction = {
      indices: m.selection.action_indices,
      d: m.d_action,
      // jpeg 段长度任意，action 块偏移未必 4 字节对齐，先 copy 出独立 buffer
      data: new Float32Array(packed.blob.slice(pos, pos + nA * 4)),
    };
    v.sliceActionNames =
      (m.action_names && m.action_names.length === m.d_action) ? m.action_names : null;
    v.sliceSel = m.selection;
    renderViewerChrome();
    renderFilmstrip();
    drawTimeline();
    drawAction();
    showFrame(v.curFrame);
  });
}

/* ------------------------------------------------------------ 查看器渲染 */
function renderViewer() {
  var info = v.info;
  var root = viewerEl();
  /* 重建 DOM 前 dispose 旧图表实例（echarts 实例/RO/主题注册表防泄漏） */
  if (tlChart) { tlChart.dispose(); tlChart = null; }
  if (acChart) { acChart.dispose(); acChart = null; }
  root.innerHTML = "";

  var head = el("div", "card ds-vhead");
  var title = el("div", "ds-vtitle");
  title.appendChild(el("div", "ds-vid", { text: info.episode_id, attrs: { title: info.episode_id } }));
  title.appendChild(el("div", "ds-vmeta", {
    text: info.robot_name + " · 视频 " + info.video_count + " 帧 · action " +
      info.action_count + " · state " + info.state_count + " · 文本 " + info.text_count +
      " · ratio " + v.ratio + (v.slice ? " · 切片 step " + v.slice.step : ""),
  }));
  head.appendChild(title);
  var btns = el("div", "ds-vbtns");
  btns.appendChild(mkBtn("下载当前帧 PNG", saveFramePng));
  btns.appendChild(mkBtn("下载窗口 MP4", saveWindowMp4));
  btns.appendChild(mkBtn("下载 action CSV", saveActionCsv));
  if (v.slice) {
    btns.appendChild(mkBtn("下载 selection JSON", saveSelectionJson));
    btns.appendChild(mkBtn("查看完整 episode", function () {
      openEpisode(shared.dataset, v.info.episode_id);
    }));
  }
  head.appendChild(btns);
  head.appendChild(el("div", "ds-prog-slot", { attrs: { id: "ds-prog" } }));
  root.appendChild(head);

  var stage = el("div", "card ds-stage");
  var left = el("div", "ds-stage-left");
  var imgWrap = el("div", "ds-img-wrap");
  mainImg = el("img", "ds-main-img loading", { attrs: { alt: "" } });
  imgWrap.appendChild(mainImg);
  frameSpinner = el("div", "ds-spinner");
  imgWrap.appendChild(frameSpinner);
  frameProg = createProgress({ label: "解码当前帧…" });
  frameProg.el.classList.add("mini", "ds-frame-prog", "hidden");
  imgWrap.appendChild(frameProg.el);
  videoWrap = el("div", "ds-video-wrap hidden");
  left.appendChild(imgWrap);
  left.appendChild(videoWrap);
  var transport = el("div", "ds-transport");
  transport.appendChild(mkBtn("◀", function () { seekFrame(v.curFrame - 1); }));
  frameLabel = el("span", "ds-frame-label", { text: "—" });
  transport.appendChild(frameLabel);
  transport.appendChild(mkBtn("▶", function () { seekFrame(v.curFrame + 1); }));
  transport.appendChild(mkBtn("播放片段", playWindow));
  left.appendChild(transport);
  filmstrip = el("div", "ds-filmstrip");
  left.appendChild(filmstrip);
  stage.appendChild(left);

  var right = el("div", "ds-stage-right");
  textsBox = el("div", "ds-texts");
  right.appendChild(textsBox);
  stage.appendChild(right);
  root.appendChild(stage);

  var tlCard = el("div", "card");
  tlCard.appendChild(el("div", "ds-chart-title", { text: "统一时间轴（帧 ⇔ action 区间；彩色段=文本区间（连续同文段已合并），顶部淡条=全局指令，点击跳转）" }));
  var tlWrap = el("div", "ds-canvas-wrap");
  timeline = el("div", "ds-timeline");
  tlWrap.appendChild(timeline);
  tlTip = el("div", "ds-tooltip hidden");
  tlWrap.appendChild(tlTip);
  tlCard.appendChild(tlWrap);
  root.appendChild(tlCard);

  var acCard = el("div", "card");
  var acHead = el("div", "ds-chart-head");
  acHead.appendChild(el("div", "ds-chart-title", { text: "action 曲线" }));
  var norm = el("label", "ds-norm");
  var normCb = el("input", "", { attrs: { type: "checkbox" } });
  normCb.checked = v.normalize;
  normCb.addEventListener("change", function () { v.normalize = normCb.checked; drawAction(); });
  norm.appendChild(normCb);
  norm.appendChild(document.createTextNode("按维归一化"));
  acHead.appendChild(norm);
  acCard.appendChild(acHead);
  dimToggles = el("div", "ds-dims");
  acCard.appendChild(dimToggles);
  var acWrap = el("div", "ds-canvas-wrap");
  actionCanvas = el("div", "ds-action");
  acWrap.appendChild(actionCanvas);
  acCard.appendChild(acWrap);
  root.appendChild(acCard);

  renderTexts();
  renderViewerChrome();
  initViewerCharts();
}

function renderViewerChrome() {
  // slice 到达后更新标题提示
  if (v.slice && v.sliceSel) {
    var meta = document.querySelector("#ds-viewer .ds-vmeta");
    if (meta) {
      meta.textContent = meta.textContent.replace(/ · 切片 step.*$/, "") +
        " · 切片窗口 [" + v.slice.start + ", " + (v.slice.start + v.slice.n) + "] step " + v.slice.step;
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
  var box = textsBox;
  box.innerHTML = "";
  box.appendChild(el("div", "ds-chart-title", {
    text: "文本（kind 着色；连续同文段已合并，区间 chips 可点击跳转）",
  }));
  var merged = mergedTexts(v.info.texts);
  var firstInstr = null;
  (v.info.texts || []).forEach(function (t) {
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
        v.flashRange = { range: r, until: Date.now() + 1500 };
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
  var strip = filmstrip;
  strip.innerHTML = "";
  if (v.slice && v.sliceFrames.length) {
    var sel = v.sliceSel.video_frame_indices;
    v.sliceFrames.forEach(function (url, i) {
      var img = el("img", "ds-thumb", { attrs: { src: url, title: "帧 " + sel[i] } });
      img.addEventListener("click", function () { seekFrame(sel[i]); });
      strip.appendChild(img);
    });
    return;
  }
  (v.filmUrls || []).forEach(function (url, i) {
    var idx = v.filmIndices[i];
    var img = el("img", "ds-thumb", { attrs: { src: url, title: "帧 " + idx } });
    img.addEventListener("click", function () { seekFrame(idx); });
    strip.appendChild(img);
  });
}

/* 当前生效的逐维语义名（切片优先；无标注回退 null） */
function currentActionNames() {
  if (v.slice && v.sliceActionNames) return v.sliceActionNames;
  return v.actionNames;
}

/* 维度显示名：有语义标注用标注，否则回退 "a{N}"（csvFallback=true 时 "dim_{N}"） */
function dimLabel(dim, csvFallback) {
  var names = currentActionNames();
  if (names && names[dim]) return names[dim];
  return (csvFallback ? "dim_" : "a") + dim;
}

function renderDimToggles() {
  var box = dimToggles;
  if (!box) return;
  box.innerHTML = "";
  var d = currentActionDim();
  for (var i = 0; i < d; i++) {
    (function (dim) {
      var label = dimLabel(dim);
      var chip = el("button", "ds-dim" + (v.dimsOn.indexOf(dim) >= 0 ? " on" : ""), {
        text: label,
        attrs: { title: "a" + dim + (label !== "a" + dim ? " · " + label : "") },
      });
      chip.style.borderColor = PALETTE[dim % PALETTE.length];
      chip.addEventListener("click", function () {
        var at = v.dimsOn.indexOf(dim);
        if (at >= 0) v.dimsOn.splice(at, 1); else v.dimsOn.push(dim);
        renderDimToggles();
        drawAction();
      });
      box.appendChild(chip);
    })(i);
  }
}

function currentActionDim() {
  if (v.slice && v.sliceAction) return v.sliceAction.d;
  return v.action ? v.action.d : 0;
}

/* ------------------------------------------------------------ 帧显示 */
function seekFrame(f) {
  if (!v.info) return;
  f = Math.max(0, Math.min(v.info.video_count - 1, f));
  v.curFrame = f;
  showFrame(f);
  drawTimeline();
  updateActionCursor();   // 只合并游标 markArea，避免 scrub 时整图重建
}

/* 键盘左右方向键步进（index.js 的 document keydown 在 tab active 且非输入控件时调用） */
export function nudgeFrame(delta) {
  if (!v.info) return;
  seekFrame(v.curFrame + delta);
}

/* browse.js 的 episode 列表高亮当前打开的集（旧 ds.info && ds.info.episode_id === eid） */
export function currentEpisodeId() {
  return v.info ? v.info.episode_id : null;
}

/* 主图换帧：加载完成前隐藏 img（避免破图图标）并显示转圈动画 */
function setMainFrame(url) {
  if (!mainImg || mainImg.src === url) return;
  mainImg.classList.add("loading");
  if (frameSpinner) frameSpinner.classList.remove("hidden");
  mainImg.onload = function () {
    mainImg.classList.remove("loading");
    if (frameSpinner) frameSpinner.classList.add("hidden");
  };
  mainImg.onerror = function () {
    mainImg.classList.remove("loading");
    if (frameSpinner) frameSpinner.classList.add("hidden");
  };
  mainImg.src = url;
}

function showFrame(f) {
  if (!v.info) return;
  frameLabel.textContent = "帧 " + f + " / " + (v.info.video_count - 1) +
    " ｜ action (" + f * v.ratio + ", " + (f + 1) * v.ratio + "]";
  // slice 窗口内的帧直接用切片解码结果
  if (v.slice && v.sliceSel) {
    var at = v.sliceSel.video_frame_indices.indexOf(f);
    if (at >= 0 && v.sliceFrames[at]) {
      setMainFrame(v.sliceFrames[at]);
      return;
    }
  }
  var cached = v.frameCache.get(f);
  if (cached) {
    setMainFrame(cached);
    return;
  }
  // 单帧解码：图片上方细进度条（集群 ffmpeg 解码可能需要数秒）
  if (frameProg) {
    frameProg.el.classList.remove("hidden");
    frameProg.indet("解码第 " + f + " 帧…");
  }
  var url = "/api/dataset/frames" + qs({
    dataset: shared.dataset, id: v.info.episode_id, indices: String(f), max_dim: 768,
  });
  fetchPacked(url).then(function (packed) {
    if (frameProg) frameProg.el.classList.add("hidden");
    var urls = jpegUrls(packed.meta, packed.blob);
    var obj = urls[0];
    v.frameCache.set(f, obj);
    if (v.frameCache.size > FRAME_CACHE_MAX) {
      var oldest = v.frameCache.keys().next().value;
      URL.revokeObjectURL(v.frameCache.get(oldest));
      v.frameCache.delete(oldest);
    }
    if (v.curFrame === f) setMainFrame(obj);
    // 预取相邻帧
    [f + 1, f - 1].forEach(function (nf) {
      if (nf >= 0 && nf < v.info.video_count && !v.frameCache.has(nf)) {
        fetchPacked("/api/dataset/frames" + qs({
          dataset: shared.dataset, id: v.info.episode_id, indices: String(nf), max_dim: 768,
        })).then(function (p) {
          v.frameCache.set(nf, jpegUrls(p.meta, p.blob)[0]);
        }).catch(function () { });
      }
    });
  }).catch(function () {
    if (frameProg) frameProg.el.classList.add("hidden");
  });
}

function playWindow() {
  if (!v.info) return;
  var params = {
    dataset: shared.dataset, id: v.info.episode_id, start: 0,
    stride: 1, fps: 0, max_dim: 640, stream: 1,   // fps=0 → 后端按源真实帧率编码
  };
  if (v.slice) {
    params.start = v.slice.start;
    params.count = v.slice.n + 1;   // 训练窗口：真实帧、真实速度
  } else {
    params.seconds = 10;            // 整集预览：真实速度 10s 窗口（短片则全片）
  }
  var url = "/api/dataset/clip" + qs(params);
  videoWrap.classList.remove("hidden");
  videoWrap.innerHTML = "";
  var prog = createProgress({ label: "集群解码/编码 MP4 中…" });
  prog.indet();
  videoWrap.appendChild(prog.el);
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
    videoWrap.innerHTML = "";
    var video = el("video", "ds-video", { attrs: { src: objUrl, controls: "controls", autoplay: "autoplay", loop: "loop" } });
    videoWrap.appendChild(video);
    var close = mkBtn("关闭播放", function () {
      videoWrap.classList.add("hidden");
      videoWrap.innerHTML = "";
      URL.revokeObjectURL(objUrl);
    });
    videoWrap.appendChild(close);
  }).catch(function (err) {
    prog.fail("加载失败：" + (err && err.message || err));
  });
}

/* ------------------------------------------------------------ 保存 */
function saveFramePng() {
  if (!mainImg || !mainImg.src) return;
  var img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = function () {
    var canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    canvas.getContext("2d").drawImage(img, 0, 0);
    canvas.toBlob(function (blob) {
      util.saveBlob(blob,
        v.info.episode_id.replace(/[^\w.-]+/g, "_") + "_f" + v.curFrame + ".png");
    }, "image/png");
  };
  img.src = mainImg.src;
}

function saveWindowMp4() {
  if (!v.info) return;
  var params = {
    dataset: shared.dataset, id: v.info.episode_id, start: 0,
    stride: 1, fps: 0, max_dim: 640, stream: 1,   // fps=0 → 后端按源真实帧率编码
  };
  if (v.slice) {
    params.start = v.slice.start;
    params.count = v.slice.n + 1;
  } else {
    params.seconds = 20;            // 真实速度 20s 窗口（短片则全片）
  }
  var url = "/api/dataset/clip" + qs(params);
  var slot = document.getElementById("ds-prog");
  var prog = slot ? createProgress({ label: "集群编码 MP4…" }) : null;
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
    util.saveBlob(new Blob([buf], { type: "video/mp4" }),
      v.info.episode_id.replace(/[^\w.-]+/g, "_") + "_f" + params.start + ".mp4");
  }).catch(function (err) {
    if (prog) prog.fail("下载失败：" + (err && err.message || err));
  });
}

function saveActionCsv() {
  var src = (v.slice && v.sliceAction) || v.action;
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
  util.saveBlob(blob,
    v.info.episode_id.replace(/[^\w.-]+/g, "_") + "_action.csv");
}

function saveSelectionJson() {
  if (!v.sliceSel) return;
  var blob = new Blob([JSON.stringify(v.sliceSel, null, 2)], { type: "application/json" });
  util.saveBlob(blob,
    v.info.episode_id.replace(/[^\w.-]+/g, "_") + "_selection.json");
}

/* ------------------------------------------------------------ 时间轴/曲线（阶段 2C：ECharts）
 * 事件泄漏修复：旧 bindCanvasEvents（dataset.js:1710-1736）每次 openEpisode
 * 都往 window 新挂一对 mousemove/mouseup 且从不摘除——打开 N 个 episode 累积
 * N 组监听，旧监听闭包的 scrubbing 标志已失效，只剩冗余的 tooltip 更新。
 * 现改为 window 级监听模块级只挂一次（scrubbing 提升为模块变量，处理器引用
 * 模块级 tlChart/v，始终指向当前图表/状态）；图表宿主随 renderViewer 重建，
 * 其监听随旧元素一并销毁（wrapper 同时 dispose echarts 实例与主题注册）。
 *
 * 像素→数据坐标一律走 echarts convertFromPixel（轴 min/max 即旧手算的 pad
 * 映射）；图表未初始化（隐藏 tab 0 尺寸）时交互静默跳过。
 */
var scrubbing = false;
var windowEventsBound = false;

function initViewerCharts() {
  tlChart = createEChart(timeline, { height: 64, option: buildTimelineOption });
  tlChart.host.addEventListener("mousedown", function (ev) { scrubbing = true; seekFromEvent(ev); });
  if (!windowEventsBound) {
    windowEventsBound = true;
    window.addEventListener("mousemove", onWindowMouseMove);
    window.addEventListener("mouseup", onWindowMouseUp);
  }
  acChart = createEChart(actionCanvas, { height: 220, option: buildActionOption });
  acChart.host.addEventListener("click", onActionClick);
}

/* 事件 clientX/Y → 时间轴帧号（chart 未初始化或落点出网格时 null） */
function frameFromEvent(ev) {
  var inst = tlChart && tlChart.chart();
  if (!inst) return null;
  var rect = timeline.getBoundingClientRect();
  var pt = inst.convertFromPixel({ xAxisIndex: 0 }, [ev.clientX - rect.left, ev.clientY - rect.top]);
  if (!pt || !isFinite(pt[0])) return null;
  return Math.round(pt[0]);
}

function seekFromEvent(ev) {
  var f = frameFromEvent(ev);
  if (f != null) seekFrame(f);   // seekFrame 自带 clamp
}

function onWindowMouseMove(ev) {
  if (scrubbing) seekFromEvent(ev);
  else updateTimelineTip(ev);
}

function onWindowMouseUp() {
  scrubbing = false;
}

/* action 曲线点击 seek（旧 bindCanvasEvents 的 padL=44/padR=8 手算映射 →
 * convertFromPixel；x 轴 min/max=数据首末 index，映射等价） */
function onActionClick(ev) {
  var src = (v.slice && v.sliceAction) || v.action;
  if (!src || !src.indices.length) return;
  var inst = acChart && acChart.chart();
  if (!inst) return;
  var rect = actionCanvas.getBoundingClientRect();
  var pt = inst.convertFromPixel({ xAxisIndex: 0 }, [ev.clientX - rect.left, ev.clientY - rect.top]);
  if (!pt || !isFinite(pt[0])) return;
  seekFrame(Math.floor((pt[0] - 1) / v.ratio));
}

function updateTimelineTip(ev) {
  if (!v.info || !timeline) return;
  var rect = timeline.getBoundingClientRect();
  if (ev.clientX < rect.left || ev.clientX > rect.right ||
      ev.clientY < rect.top || ev.clientY > rect.bottom) {
    tlTip.classList.add("hidden");
    return;
  }
  var f = frameFromEvent(ev);
  var hit = null;
  if (f != null) {
    (v.info.texts || []).forEach(function (t) {
      if (t.frame_range && f >= t.frame_range[0] && f <= t.frame_range[1]) hit = t;
    });
  }
  if (hit) {
    tlTip.textContent = hit.kind + "：" + hit.text + " [" + hit.frame_range[0] + ", " + hit.frame_range[1] + "]";
    tlTip.style.left = Math.min(rect.width - 160, Math.max(0, ev.clientX - rect.left + 10)) + "px";
    tlTip.style.top = "2px";
    tlTip.classList.remove("hidden");
  } else {
    tlTip.classList.add("hidden");
  }
}

/* 统一时间轴 option（旧 drawTimeline canvas 手画的元素一一对应；y 轴 0..64
 * inverse 复用旧 canvas 像素行坐标）：
 *   markArea：轨道边框 y30-44（透明填充描边）、文本段色块 y30-44（KIND_COLORS
 *     alpha0.75，连续同文段已合并）、全局指令淡条 y16-24 alpha0.15、切片窗口
 *     y24-50 alpha0.22、flash 白框 y28-46（chip 点击后 until 前）；
 *   markLine：游标 y20-52；
 *   x 轴 splitNumber 8 ≈ 旧 9 个手画刻度；graphic="action idx ×N" 标签。 */
function buildTimelineOption() {
  var colors = themeColors();
  var accent = themeColor("--accent", "#4f8cff");
  var vc = v.info.video_count;
  var areas = [
    [{ xAxis: 0, yAxis: 30, itemStyle: { color: "transparent", borderColor: colors.border, borderWidth: 1 } },
     { xAxis: vc - 1, yAxis: 44 }],
  ];
  var merged = mergedTexts(v.info.texts);
  var hasGlobal = false;
  merged.forEach(function (t) {
    if (!t.ranges) { hasGlobal = true; return; }
    var color = KIND_COLORS[t.kind] || "#5c6880";
    t.ranges.forEach(function (r) {
      areas.push([{ xAxis: r[0], yAxis: 30, itemStyle: { color: color, opacity: 0.75 } },
                  { xAxis: r[1], yAxis: 44 }]);
    });
  });
  if (hasGlobal) {
    areas.push([{ xAxis: 0, yAxis: 16, itemStyle: { color: "#4f8cff", opacity: 0.15 } },
                { xAxis: vc - 1, yAxis: 24 }]);
  }
  if (v.slice) {
    areas.push([{ xAxis: v.slice.start, yAxis: 24, itemStyle: { color: accent, opacity: 0.22 } },
                { xAxis: v.slice.start + v.slice.n, yAxis: 50 }]);
  }
  if (v.flashRange && Date.now() < v.flashRange.until) {
    areas.push([{ xAxis: v.flashRange.range[0], yAxis: 28, itemStyle: { color: "transparent", borderColor: "#ffffff", borderWidth: 2 } },
                { xAxis: v.flashRange.range[1], yAxis: 46 }]);
  }
  var base = chartTheme();
  return {
    animation: false,
    textStyle: base.textStyle,
    grid: { left: 8, right: 8, top: 4, bottom: 18 },
    xAxis: {
      type: "value", min: 0, max: Math.max(1, vc - 1), splitNumber: 8,
      axisLine: { show: false },
      axisTick: { show: true, length: 4, lineStyle: base.xAxis.axisTick.lineStyle },
      axisLabel: Object.assign({}, base.xAxis.axisLabel, {
        formatter: function (x) { return String(Math.round(x)); },
      }),
      splitLine: { show: false },
    },
    yAxis: { type: "value", min: 0, max: 64, inverse: true, show: false },
    tooltip: { show: false },   // hover 命中文本段走自定义 ds-tooltip（updateTimelineTip）
    series: [{
      id: "tl", type: "line", data: [], silent: true,
      markArea: { silent: true, data: areas },
      markLine: {
        silent: true, symbol: "none", animation: false,
        lineStyle: { color: accent, width: 2 },
        label: { show: false },
        data: [[{ coord: [v.curFrame, 20] }, { coord: [v.curFrame, 52] }]],
      },
    }],
    graphic: [{
      id: "tl-note", type: "text", left: 8, top: 3, silent: true,
      style: {
        text: "action idx ×" + v.ratio, fontSize: 10,
        fontFamily: "ui-monospace, Menlo, Consolas, monospace", fill: colors.faint,
      },
    }],
  };
}

function drawTimeline() {
  if (!v.info || !tlChart) return;
  tlChart.apply();
}

/* 当前帧对应 action 区间 markArea（旧 drawAction 内联手画：仅非 slice，
 * 区间 (f*ratio, (f+1)*ratio] 与数据范围有交才画，alpha0.18） */
function actionCursorAreas(x0, x1) {
  if (v.slice || !v.info) return [];
  var a0 = v.curFrame * v.ratio + 1, a1 = (v.curFrame + 1) * v.ratio;
  if (a1 < x0 || a0 > x1) return [];
  return [[{ xAxis: a0, itemStyle: { color: themeColor("--accent", "#4f8cff"), opacity: 0.18 } },
           { xAxis: a1 }]];
}

/* action 曲线 option（旧 drawAction canvas 手画的元素一一对应）：
 * 逐维 PALETTE 折线、x 轴绝对 action index、y 仅横向网格（旧 5 线 → echarts
 * 自动分割）、当前帧区间着色（__cur 空序列挂 markArea，seek 时单独合并更新）、
 * 按维归一化 = 各维数据先映射 [0,1] 再画、轴显 0..1（旧版轴标签仍显示全局
 * min/max 的 quirk 随之消除）；维度开关 chips 即图例（legend 不画）。
 * 新增 axis tooltip（旧手画无读数）：数据点 raw 携带原始值，normalize 时
 * tooltip 仍给真实值。 */
function buildActionOption() {
  var colors = themeColors();
  var base = chartTheme();
  var src = (v.slice && v.sliceAction) || v.action;
  var hasData = !!(src && src.indices.length && src.d);
  var note = null, noteColor = colors.muted;
  if (actionError) { note = actionError; noteColor = themeColor("--red-strong", "#ff8a80"); }
  else if (!hasData) { note = src ? "无 action 数据" : "action 加载中…"; }

  var x0 = 0, x1 = 1;
  var series = [];
  if (hasData) {
    x0 = src.indices[0];
    x1 = src.indices[src.indices.length - 1];
    if (x1 - x0 < 1e-9) x1 = x0 + 1;
    var n = src.indices.length;
    /* 按维 min/max（仅 normalize 需要；只统计开启维度，与旧一致） */
    var dimMin = {}, dimMax = {};
    if (v.normalize) {
      v.dimsOn.forEach(function (c) { dimMin[c] = Infinity; dimMax[c] = -Infinity; });
      for (var r = 0; r < n; r++) {
        v.dimsOn.forEach(function (c) {
          var val = src.data[r * src.d + c];
          if (val < dimMin[c]) dimMin[c] = val;
          if (val > dimMax[c]) dimMax[c] = val;
        });
      }
    }
    v.dimsOn.forEach(function (c) {
      var color = PALETTE[c % PALETTE.length];
      var data = new Array(n);
      for (var r2 = 0; r2 < n; r2++) {
        var val2 = src.data[r2 * src.d + c];
        var y = val2;
        if (v.normalize) {
          var span = dimMax[c] - dimMin[c];
          y = span < 1e-9 ? 0 : (val2 - dimMin[c]) / span;
        }
        data[r2] = { value: [src.indices[r2], y], raw: val2 };
      }
      series.push({
        id: "dim" + c, name: dimLabel(c), type: "line", data: data,
        showSymbol: false,
        lineStyle: { color: color, width: 1.2 },
        itemStyle: { color: color },
      });
    });
  }
  series.push({
    id: "__cur", name: "__cur", type: "line", data: [], silent: true,
    markArea: { silent: true, data: hasData ? actionCursorAreas(x0, x1) : [] },
  });

  return {
    animation: false,
    textStyle: base.textStyle,
    grid: { left: 8, right: 8, top: 10, bottom: 4, containLabel: true },
    legend: { show: false },
    xAxis: {
      type: "value", min: x0, max: x1,
      axisLine: base.xAxis.axisLine,
      axisTick: base.xAxis.axisTick,
      axisLabel: Object.assign({}, base.xAxis.axisLabel, {
        formatter: function (x) { return String(Math.round(x)); },
      }),
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      min: v.normalize ? 0 : null,
      max: v.normalize ? 1 : null,
      scale: !v.normalize,
      axisLine: base.yAxis.axisLine,
      axisTick: base.yAxis.axisTick,
      axisLabel: Object.assign({}, base.yAxis.axisLabel, { formatter: util.fmtAxis }),
      splitLine: base.yAxis.splitLine,
    },
    tooltip: Object.assign({}, base.tooltip, {
      trigger: "axis", confine: true, formatter: fmtActionTip,
    }),
    graphic: [{
      id: "ac-note", type: "text", left: "center", top: "middle", silent: true,
      invisible: !note,
      style: { text: note || "", fontSize: 12, fontFamily: "sans-serif", fill: noteColor },
    }],
    series: series,
  };
}

/* tooltip：标题=action index，逐维 raw 原始值（行内样式复刻 line.js 的观感） */
function fmtActionTip(params) {
  if (!params || !params.length) return "";
  var rows = [];
  params.forEach(function (p) {
    if (p.seriesId === "__cur") return;
    var val = p.data && util.isNum(p.data.raw) ? p.data.raw
      : (p.data && p.data.value && util.isNum(p.data.value[1]) ? p.data.value[1] : null);
    if (!util.isNum(val)) return;
    var color = typeof p.color === "string" ? p.color : "";
    rows.push(
      '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
      '<span style="width:8px;height:8px;border-radius:2px;flex:none;background:' + color + '"></span>' +
      "<span>" + esc(p.seriesName) + "</span>" +
      '<span style="margin-left:auto;padding-left:12px">' + esc(util.fmtVal(val)) + "</span></div>"
    );
  });
  if (!rows.length) return "";
  return '<div style="color:var(--muted);margin-bottom:4px">action ' +
    Math.round(Number(params[0].axisValue)) + "</div>" + rows.join("");
}

function drawAction() {
  if (!acChart || !v.info) return;
  acChart.apply();
}

/* seek 链路快路径：只合并 __cur 序列的 markArea（整图重建留给数据/开关变化，
 * 避免 scrub 时每帧重组 2048×d 数据点） */
function updateActionCursor() {
  var inst = acChart && acChart.chart();
  if (!inst) return;
  var src = (v.slice && v.sliceAction) || v.action;
  if (!src || !src.indices.length) return;
  inst.setOption({
    series: [{
      id: "__cur",
      markArea: { silent: true, data: actionCursorAreas(src.indices[0], src.indices[src.indices.length - 1]) },
    }],
  });
}
