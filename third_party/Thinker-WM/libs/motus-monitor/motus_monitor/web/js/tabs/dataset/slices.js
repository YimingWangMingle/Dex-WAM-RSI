/* Motus Monitor 前端 —— 训练集 tab · 训练切片模式（阶段 1C）。
 * 逻辑逐字迁自旧 web/dataset.js：
 *   run 选项    loadRuns(588-604) + 进入面板自动加载（旧 setMode 404-411 的切片分支）
 *   切片加载    sliceStepRange(606-617)、loadSlices(619-702，PRG/PRGJ 流式进度)、
 *               sliceAutoPick 启发式(663-679 逐字保留)
 *   ckpt 配比   sliceSourceNames/sourceColor(704-715)、renderSliceCkpts(717-759)、
 *               runShortLabel(761-767)、mkCkptRow(769-798)
 *   step 过滤   loadSliceSteps/resetSliceSteps(800-823)
 *   统计/饼图   renderSlicePie(825-866，阶段 2C 换 ECharts pie，DOM 图例保留)、
 *               renderSliceStats(868-926)
 *   覆盖率      renderSliceCoverage/mkCoverageDetail(928-999)
 *   切片列表    mkSliceRow(1001-1012)、renderSliceTree+展开/折叠全部(1014-1050)、
 *               renderSlices(1052-1056)
 *   控件绑定    旧 bindOnce 362-393 中属于切片模式的部分（run/源/谱系/step 过滤、
 *               展开/折叠全部、SLICE_PAGE 分页）
 * 切片树沿用旧手搓逻辑逐字搬迁（固定 step→node→gpu 结构 + 默认仅最新 step 展开），
 * 未改用 ui/tree.js；分页沿用 index.html 静态按钮绑定，未改用 ui/pager.js。
 */
import { api } from "../../api.js";
import { util } from "../../util.js";
import { fetchStream } from "../../ui/stream.js";
import { createProgress } from "../../ui/progress.js";
import { chartTheme } from "../../charts/theme.js";
import { createEChart } from "../../charts/base.js";
import { shared, qs } from "./state.js";
import { openEpisode, openSlice } from "./viewer.js";

var esc = util.esc, el = util.el, fmtInt = util.fmtInt;

var SLICE_PAGE = 100;
var COV_DETAIL_MAX = 300;

/* 数据源配色（按源名排序稳定取色，ckpt 配比条与图例共用） */
var SLICE_PALETTE = [
  "#4f8cff", "#3fb96f", "#d9a038", "#e05d5d", "#9b6dd6",
  "#46b8b0", "#d06fa8", "#8ba33e", "#6d7dd8", "#c97b3f",
];

/* 切片面板状态（旧 ds 29-76 中属于训练切片的部分） */
var s = {
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
};

/* 切片模式控件绑定（旧 bindOnce 362-393；index.js init 只调一次，替代 bindOnce.done） */
export function bindSlices() {
  document.getElementById("ds-slice-run").addEventListener("change", function (ev) {
    s.sliceRun = ev.target.value;
    s.sliceOffset = 0;
    s.sliceCkptIdx = -1;
    s.sliceStep = "";
    s.sliceAutoPick = true;   // 换 run 后重新默认到最新段
    resetSliceSteps();
    loadSlices();
  });
  document.getElementById("ds-slice-source").addEventListener("change", function (ev) {
    s.sliceSource = ev.target.value; s.sliceOffset = 0; loadSlices();
  });
  document.getElementById("ds-slice-lineage").addEventListener("change", function (ev) {
    s.sliceLineage = !!ev.target.checked; s.sliceOffset = 0; loadSlices();
  });
  document.getElementById("ds-slice-expand-all").addEventListener("click", function () {
    setAllStepsExpanded(true);
  });
  document.getElementById("ds-slice-collapse-all").addEventListener("click", function () {
    setAllStepsExpanded(false);
  });
  document.getElementById("ds-slice-step").addEventListener("change", function (ev) {
    s.sliceStep = ev.target.value; s.sliceOffset = 0; loadSlices();
  });
  document.getElementById("ds-slice-prev").addEventListener("click", function () {
    s.sliceOffset = Math.max(0, s.sliceOffset - SLICE_PAGE); loadSlices();
  });
  document.getElementById("ds-slice-next").addEventListener("click", function () {
    if (s.sliceOffset + SLICE_PAGE < s.sliceTotal) {
      s.sliceOffset += SLICE_PAGE; loadSlices();
    }
  });
}

/* ------------------------------------------------------------ 训练切片面板 */
export function loadRuns() {
  api.get("/api/runs").then(function (payload) {
    s.runs = payload.runs || [];
    var sel = document.getElementById("ds-slice-run");
    sel.innerHTML = "";
    s.runs.forEach(function (r) {
      var label = (r.run_name || r.run_id) + " · " + (r.last_step != null ? ("step " + r.last_step) : "");
      sel.appendChild(el("option", "", { text: label, attrs: { value: r.run_id } }));
    });
    if (s.runs.length && !s.sliceRun) {
      s.sliceRun = s.runs[0].run_id;
      if (shared.mode === "slices") loadSlices();
    }
    /* 重建选项后恢复选中值，避免 select 显示与内部状态不一致 */
    if (s.sliceRun) sel.value = s.sliceRun;
  }).catch(function () { });
}

/* 进入切片面板（index.js setMode("slices") 时调用；旧 setMode 404-411 的切片分支）：
 * browse 模式预置过 sliceRun 但未加载：进入面板时自动加载（默认最新段） */
export function enterSlicesPanel() {
  if (!s.runs.length) return;
  if (!s.sliceRun) {
    s.sliceRun = s.runs[0].run_id;
    document.getElementById("ds-slice-run").value = s.sliceRun;
  }
  if (s.slicesLoadedFor !== s.sliceRun) loadSlices();
}

/* 当前 ckpt 段 + 精确 step 选择对应的 step_min/step_max（null = 不限） */
function sliceStepRange() {
  var lo = null, hi = null;
  if (s.sliceCkptIdx >= 0 && s.sliceCkpts[s.sliceCkptIdx]) {
    lo = s.sliceCkpts[s.sliceCkptIdx].step_min;
    hi = s.sliceCkpts[s.sliceCkptIdx].step_max;
  }
  if (s.sliceStep !== "") {
    lo = hi = parseInt(s.sliceStep, 10);
  }
  return { step_min: lo, step_max: hi };
}

function loadSlices() {
  if (!s.sliceRun) return;
  /* 流式加载：服务端解析进度（PRG）驱动确定性进度条，成品为 JSON 响应体 */
  var box = document.getElementById("ds-slices");
  box.innerHTML = "";
  var prog = createProgress({ label: "加载训练切片…" });
  prog.el.classList.add("mini");
  prog.indet();
  box.appendChild(prog.el);
  var range = sliceStepRange();
  var url = "/api/runs/" + encodeURIComponent(s.sliceRun) + "/slices" + qs({
    offset: s.sliceOffset, limit: SLICE_PAGE, source: s.sliceSource,
    step_min: range.step_min, step_max: range.step_max,
    lineage: s.sliceLineage ? "1" : "", stream: "1",
  });
  var sawJson = false;  /* 收到 PRGJ 后忽略兼容用 PRG 行，避免覆盖明细 */
  fetchStream(url, {
    onServerProgress: function (frac, stage) {
      if (sawJson) return;
      prog.set(frac, (stage || "解析切片") + " " + Math.round(frac * 100) + "%");
    },
    onServerProgressJson: function (ev) {
      sawJson = true;
      var frac = ev.frac || 0;
      prog.set(frac, (ev.label || "处理中") + " " + Math.round(frac * 100) + "%");
      var bits = [];
      if (ev.detail) bits.push(ev.detail);
      if (ev.eta_s != null && frac < 1) bits.push("剩余约 " + util.fmtDuration(ev.eta_s));
      prog.detail(bits.join(" · "));
    },
    onDownload: function (received, total) {
      if (total > 0) {
        prog.set(received / total, "下载结果 " + Math.round(received / total * 100) + "%");
        prog.detail(util.fmtBytes(received) + " / " + util.fmtBytes(total));
      }
    },
  }).then(function (buf) {
    prog.done();
    var payload = JSON.parse(new TextDecoder().decode(buf));
    s.slicesLoadedFor = s.sliceRun;
    s.sliceRanked = !!payload.ranked;
    // ranked 响应分页单位为 step，旧响应为扁平 slice
    s.sliceTotal = s.sliceRanked ? (payload.total_steps || 0) : (payload.total_slices || 0);
    s.sliceCkpts = payload.by_ckpt || [];
    /* 自动选中末尾"最新"段：进入面板/换 run 后的首次加载默认只看最新数据，
       用户可点"全部"行或其他段切换（手动选择后 sliceAutoPick 失效） */
    if (s.sliceAutoPick && s.sliceCkptIdx === -1 && s.sliceCkpts.length) {
      s.sliceAutoPick = false;
      /* 谱系模式下 by_ckpt 含祖先 run 的段：自动选段要找本 run 的末尾"最新"段 */
      var pickIdx = s.sliceCkpts.length - 1;
      for (var pi = s.sliceCkpts.length - 1; pi >= 0; pi--) {
        var segRun = s.sliceCkpts[pi].run_id;
        if (!segRun || segRun === s.sliceRun) { pickIdx = pi; break; }
      }
      s.sliceCkptIdx = pickIdx;
      s.sliceOffset = 0;
      renderSliceCkpts();
      loadSliceSteps();
      loadSlices();
      return;
    }
    renderSliceCkpts();
    renderSliceStats(payload);
    if (s.sliceRanked) {
      s.lastSliceSteps = payload.steps || [];
      renderSliceTree(s.lastSliceSteps);
    } else {
      s.lastSliceSteps = null;
      renderSlices(payload.slices || []);
    }
    var pageInfo = document.getElementById("ds-slice-page-info");
    if (s.sliceTotal) {
      pageInfo.textContent = (s.sliceOffset + 1) + "–" +
        Math.min(s.sliceOffset + SLICE_PAGE, s.sliceTotal) + " / " + s.sliceTotal +
        (s.sliceRanked ? " step · 共 " + (payload.total_slices || 0) + " 切片" : "");
    } else {
      pageInfo.textContent = "该范围暂无切片记录（需含 episode_ids 的 step 记录）";
    }
  }).catch(function (err) {
    prog.fail("加载失败");
    document.getElementById("ds-slices").innerHTML =
      '<div class="ds-error">' + esc(err.message) + "</div>";
  });
}

/* 全部源名（跨 ckpt 段求并集，排序稳定 → 配色稳定） */
function sliceSourceNames() {
  var seen = {};
  s.sliceCkpts.forEach(function (seg) {
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
  var hasCkpt = s.sliceCkpts.some(function (seg) { return seg.ckpt_step != null; });
  if (!s.sliceCkpts.length || !names.length || !hasCkpt) {
    box.appendChild(el("div", "ds-muted", {
      text: hasCkpt || !s.sliceCkpts.length ? "无统计" : "该 run 无 ckpt 记录，无法按 ckpt 分段",
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
  s.sliceCkpts.forEach(function (seg) {
    allTotal += seg.total;
    Object.keys(seg.by_source || {}).forEach(function (name) {
      allBySource[name] = (allBySource[name] || 0) + seg.by_source[name];
    });
  });
  box.appendChild(mkCkptRow(-1, "全部", allBySource, allTotal, names));
  /* 谱系（历史累计）模式下分段行按 run 标注（server 端每段带 run_id） */
  var runIds = {};
  s.sliceCkpts.forEach(function (seg) { if (seg.run_id) runIds[seg.run_id] = true; });
  var multiRun = Object.keys(runIds).length > 1;
  s.sliceCkpts.forEach(function (seg, idx) {
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
  s.runs.forEach(function (r) { if (r.run_id === runId) name = r.run_name; });
  name = name || runId;
  return name.length > 14 ? name.slice(0, 13) + "…" : name;
}

function mkCkptRow(idx, label, bySource, total, names) {
  var row = el("div", "ds-ckpt-row" + (s.sliceCkptIdx === idx ? " active" : ""));
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
    s.sliceAutoPick = false;   // 手动选段后不再自动跳到最新段
    s.sliceCkptIdx = (s.sliceCkptIdx === idx) ? -1 : idx;
    s.sliceStep = "";
    s.sliceOffset = 0;
    renderSliceCkpts();
    if (s.sliceCkptIdx >= 0) loadSliceSteps(); else resetSliceSteps();
    loadSlices();
  });
  return row;
}

/* 选中 ckpt 段后拉取该段的 step 列表填充下拉（全部时不拉，避免超长列表） */
function loadSliceSteps() {
  var sel = document.getElementById("ds-slice-step");
  if (!s.sliceRun || s.sliceCkptIdx < 0) return resetSliceSteps();
  var range = sliceStepRange();
  var url = "/api/runs/" + encodeURIComponent(s.sliceRun) + "/slices" + qs({
    steps_only: "1", step_min: range.step_min, step_max: range.step_max,
  });
  sel.disabled = true;
  api.get(url).then(function (payload) {
    sel.innerHTML = '<option value="">全部 step</option>';
    (payload.steps || []).forEach(function (st) {
      sel.appendChild(el("option", "", { text: "step " + st, attrs: { value: st } }));
    });
    sel.disabled = false;
    sel.value = s.sliceStep;
  }).catch(function () { sel.disabled = false; });
}

function resetSliceSteps() {
  var sel = document.getElementById("ds-slice-step");
  sel.innerHTML = '<option value="">全部 step</option>';
  sel.disabled = s.sliceCkptIdx < 0;
}

/* 当前范围（ckpt 段/step）的数据源占比饼图 + 图例（取代原文字百分比 chips）。
 * 阶段 2C：132×132 canvas 手画扇形 → ECharts pie（charts/base.js 裸封装）；
 * ds-pie-legend DOM 图例保留，wrap.innerHTML 重建模式不变——重建前 dispose
 * 旧实例防泄漏。 */
var slicePie = null;

function renderSlicePie(bySource) {
  var wrap = document.getElementById("ds-slice-pie-wrap");
  if (slicePie) { slicePie.dispose(); slicePie = null; }
  wrap.innerHTML = "";
  var names = Object.keys(bySource);
  var total = names.reduce(function (acc, n) { return acc + (bySource[n].count || 0); }, 0);
  if (!names.length || !total) return;
  var allNames = sliceSourceNames();
  var pieEl = el("div", "ds-slice-pie");
  wrap.appendChild(pieEl);
  slicePie = createEChart(pieEl, {
    height: 132,
    option: function () {
      var base = chartTheme();
      return {
        animation: false,
        textStyle: base.textStyle,
        tooltip: Object.assign({}, base.tooltip, {
          confine: true,
          /* 与图例行 title 同口径："name：count 切片 / unique_episodes 集" */
          formatter: function (p) {
            var st = bySource[p.name] || {};
            return esc(p.name) + "：" + (st.count || 0) + " 切片 / " + (st.unique_episodes || 0) + " 集";
          },
        }),
        series: [{
          type: "pie", radius: "94%", center: ["50%", "50%"],
          label: { show: false }, labelLine: { show: false },
          data: names.map(function (name) {
            return {
              name: name, value: bySource[name].count,
              itemStyle: { color: sourceColor(name, allNames) },
            };
          }),
        }],
      };
    },
  });
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
      s.covExpanded[name] = !s.covExpanded[name];
      renderSliceCoverage(stats);
    });
    box.appendChild(row);
    if (s.covExpanded[name]) box.appendChild(mkCoverageDetail(name, c));
  });
  if (cov.missing_catalogs && cov.missing_catalogs.length) {
    box.appendChild(el("div", "ds-muted", {
      text: "无本地 catalog，覆盖率缺失：" + cov.missing_catalogs.join("、"),
    }));
  }
}

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
    var expanded = (st.step in s.sliceExpanded) ? s.sliceExpanded[st.step] : (idx === 0);
    var head = el("div", "ds-step-head" + (expanded ? " open" : ""));
    head.appendChild(el("span", "ds-eptree-toggle", { text: expanded ? "▾" : "▸" }));
    head.appendChild(el("span", "ds-step-title", {
      text: "step " + st.step + " · " + st.total + " 切片",
    }));
    head.addEventListener("click", function () {
      s.sliceExpanded[st.step] = !expanded;
      renderSliceTree(s.lastSliceSteps || []);
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
  (s.lastSliceSteps || []).forEach(function (st) { s.sliceExpanded[st.step] = expanded; });
  if (s.lastSliceSteps) renderSliceTree(s.lastSliceSteps);
}

function renderSlices(slices) {
  var box = document.getElementById("ds-slices");
  box.innerHTML = "";
  slices.forEach(function (s) { box.appendChild(mkSliceRow(s)); });
}
