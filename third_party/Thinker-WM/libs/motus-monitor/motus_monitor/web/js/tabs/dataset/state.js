/* Motus Monitor 前端 —— 训练集 tab 目录内共享状态与小工具（阶段 1C，本目录私有）。
 * 旧 web/dataset.js:29-76 的 ds 上帝对象按 browse/slices/viewer 三块切分后，
 * 仍跨块共享的字段集中于此，由三块显式 import（不再互相塞引用）：
 *   mode     当前面板模式 "browse" | "slices"
 *            （index.js setMode 写；slices.js loadRuns 读）
 *   dataset  当前数据集名
 *            （browse.js selectDataset / viewer.js openEpisode 写；
 *             browse.js renderDatasets 高亮、viewer.js 各请求参数读）
 * 其余字段归各模块私有（模块级状态对象；DOM 引用一律模块级变量，不进状态对象）。
 */
export const shared = {
  mode: "browse",
  dataset: null,
};

/* 查询串拼装（旧 web/dataset.js:247-254 逐字）。browse/slices/viewer 三块共用；
 * js/util.js 是共享文件、本阶段不改，故放本目录私有模块。 */
export function qs(params) {
  var parts = [];
  for (var k in params) {
    if (params[k] == null || params[k] === "") continue;
    parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(params[k]));
  }
  return parts.length ? "?" + parts.join("&") : "";
}
