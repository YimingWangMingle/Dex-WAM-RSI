/* Motus Monitor 前端 —— 流式进度条组件（PRG/PRGJ 消费）。
 * 从旧 web/dataset.js:181-232 makeProgress 逐字抽象：
 * 确定性 step/set 与不定态 indet；done() 淡出移除；
 * detail(text) 在进度条下方显示一行明细（文件/字节/切片计数/ETA）。
 * createProgress({label, total}) → {el, set(frac,label), step(text), setTotal(n),
 *                                   indet(text), detail(text), fail(msg), done()}
 */
import { util } from "../util.js";

export function createProgress(opts) {
  opts = opts || {};
  var total = Math.max(1, opts.total || 1);
  var root = util.el("div", "ds-prog");
  var lab = util.el("div", "ds-prog-label", { text: opts.label || "加载中…" });
  var track = util.el("div", "ds-prog-track");
  var bar = util.el("div", "ds-prog-bar");
  track.appendChild(bar);
  root.appendChild(lab);
  root.appendChild(track);
  var sub = util.el("div", "ds-prog-detail");
  root.appendChild(sub);
  var completed = 0, failed = false;
  var prog = {
    el: root,
    detail: function (text) { sub.textContent = text || ""; },
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
