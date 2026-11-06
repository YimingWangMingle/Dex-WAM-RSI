/* Motus Monitor 前端 —— 训练集 tab 本体（阶段 1C：旧 web/dataset.js 拆分）。
 * 逻辑来源：入口/状态（旧 309-325 refresh、343-395 bindOnce、397-412 setMode、
 * 414-421 onKey）；三块业务归同目录模块：
 *   browse.js  数据集浏览模式（旧 327-341 ds-status + 423-585 + 搜索/分页绑定）
 *   slices.js  训练切片模式（旧 587-1056 + ds-slice-* 绑定）
 *   viewer.js  episode/切片查看器（旧 1058-1912 + 查看器工具函数）
 *   state.js   跨块共享状态（mode/dataset）+ qs 小工具
 *
 * 事件监听修复：
 *   - 旧 bindCanvasEvents（1720-1725）每次 openEpisode 往 window 挂 mousemove/mouseup
 *     从不摘除 → viewer.js 改为模块级只挂一次（详见该文件 bindCanvasEvents 注释）；
 *   - 旧常驻 document keydown（394）改为 enter 挂 / leave 摘
 *     （它本就检查 #tab-dataset.active，语义等价）。
 */
import { api } from "../../api.js";
import { applyTheme } from "../../charts/theme.js";
import { shared } from "./state.js";
import { bindBrowse, loadDatasets, renderStatus } from "./browse.js";
import { bindSlices, loadRuns, enterSlicesPanel } from "./slices.js";
import { openEpisode, nudgeFrame } from "./viewer.js";

/* 对外接口保持：旧 MonApp.datasetTab.openEpisode（app.js 无外部调用方，保留导出） */
export { openEpisode };

var initialized = false;
var enabled = false;

/* 旧 MonApp.datasetTab.refresh()（dataset.js:310-325）：/api/dataset/status →
 * 首次 loadDatasets/loadRuns。具名导出保留旧对外接口；enter 时调用。
 * （注意与下方 tab 契约的 refresh() 区分：那个是数据事件轻量刷新，本 tab 保持空实现。） */
export function refresh() {
  api.get("/api/dataset/status").then(function (st) {
    enabled = !!st.enabled && !st.error;
    renderStatus(st);
    if (!enabled) return;
    if (!initialized) {
      initialized = true;
      loadDatasets();
      loadRuns();
    }
  }).catch(function (err) {
    renderStatus({ enabled: false, error: err.message });
  });
}

/* 模式切换（旧 397-412 setMode）：browse/slices 面板显隐 + 进面板自动加载 */
function setMode(mode) {
  shared.mode = mode;
  document.querySelectorAll(".ds-mode-btn").forEach(function (b) {
    b.classList.toggle("active", b.getAttribute("data-dsmode") === mode);
  });
  document.getElementById("ds-browse-panel").classList.toggle("hidden", mode !== "browse");
  document.getElementById("ds-slices-panel").classList.toggle("hidden", mode !== "slices");
  if (mode === "slices") enterSlicesPanel();
}

/* 查看器键盘左右步进（旧 414-421 onKey）：本就检查 #tab-dataset.active，
 * 改为 enter 挂 / leave 摘后语义等价 */
function onKey(ev) {
  var tab = document.getElementById("tab-dataset");
  if (!tab || !tab.classList.contains("active")) return;
  if (/INPUT|TEXTAREA|SELECT/.test((ev.target.tagName || ""))) return;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); nudgeFrame(-1); }
  if (ev.key === "ArrowRight") { ev.preventDefault(); nudgeFrame(1); }
}

export default {
  id: "dataset",

  init: function () {
    /* 控件绑定（旧 bindOnce 343-395；init 由 main.js 保证只调一次，替代 bindOnce.done） */
    document.querySelectorAll(".ds-mode-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { setMode(btn.getAttribute("data-dsmode")); });
    });
    bindBrowse();
    bindSlices();
  },

  enter: function () {
    document.addEventListener("keydown", onKey);
    refresh();
  },

  leave: function () {
    document.removeEventListener("keydown", onKey);
  },

  /* 数据事件联动：旧实现 onDataChanged 不刷新本 tab，保持空实现 */
  refresh: function () {
  },

  /* 主题切换后重绘（viewer/slices 的 ECharts 经 theme.js 注册表统一刷新） */
  onTheme: function () {
    applyTheme();
  },
};
