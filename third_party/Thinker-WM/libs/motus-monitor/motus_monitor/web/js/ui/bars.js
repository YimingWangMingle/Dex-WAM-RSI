/* Motus Monitor 前端 —— DOM 条形件（纯 DOM 组件，从 charts/legacy.js 逐字接生）。
 * legacy 引擎随阶段 2 收尾删除，这两个组件与 canvas 引擎无关、overview/timing
 * 仍在用，归档到 ui/ 下。
 *   hbar()      水平百分比条（系统负载用）
 *   shareBar()  100% 堆叠条 + 图例
 */
import { util } from "../util.js";

/* 水平百分比条（系统负载）。opts: {label, pct(0-100|null), text, colorize} */
export function hbar(opts) {
  var row = util.el("div", "hbar-row");
  row.appendChild(util.el("span", "hbar-label", { text: opts.label || "" }));
  var track = util.el("div", "hbar-track");
  var fill = util.el("div", "hbar-fill");
  var pct = opts.pct;
  if (util.isNum(pct)) {
    pct = Math.max(0, Math.min(100, pct));
    fill.style.width = pct + "%";
    if (opts.color) fill.style.background = opts.color;
    else if (pct >= 90) fill.classList.add("crit");
    else if (pct >= 70) fill.classList.add("warn");
    else fill.classList.add("ok");
  } else {
    fill.style.width = "0";
  }
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(util.el("span", "hbar-text", { text: opts.text != null ? opts.text : "暂无数据" }));
  return row;
}

/* 100% 堆叠条 + 图例。parts: [{label, value, color}]；返回元素。 */
export function shareBar(parts) {
  var wrapEl = util.el("div");
  var bar = util.el("div", "share-bar");
  var legend = util.el("div", "share-legend");
  var total = 0;
  var i;
  for (i = 0; i < parts.length; i++) total += Math.max(0, parts[i].value || 0);
  for (i = 0; i < parts.length; i++) {
    var p = parts[i];
    var frac = total > 0 ? Math.max(0, p.value || 0) / total : 0;
    var seg = util.el("div", "share-seg");
    seg.style.width = (frac * 100).toFixed(3) + "%";
    seg.style.background = p.color;
    seg.title = p.label + " " + (frac * 100).toFixed(1) + "%";
    bar.appendChild(seg);
    var lg = util.el("span", "lg");
    var chip = util.el("span", "lg-chip");
    chip.style.background = p.color;
    lg.appendChild(chip);
    lg.appendChild(util.el("span", null, {
      text: p.label + " " + (frac * 100).toFixed(1) + "%",
    }));
    legend.appendChild(lg);
  }
  wrapEl.appendChild(bar);
  wrapEl.appendChild(legend);
  return wrapEl;
}
