/* Motus Monitor 前端 —— 翻页组件（上一页/下一页 + 页码文案）。
 * DOM 结构与 index.html 的 .ds-pager 一致（249-253 行），逻辑参考
 * 旧 web/dataset.js:349-393 的分页绑定。
 * createPager({onPrev, onNext}) → {el, setInfo(text), set({info, prevDisabled, nextDisabled}),
 *                                  prevBtn, nextBtn}
 */
import { util } from "../util.js";

export function createPager(opts) {
  opts = opts || {};
  var el = util.el("div", "ds-pager");
  var prevBtn = util.el("button", "btn", { text: "上一页" });
  var infoEl = util.el("span", "ds-pager-info");
  var nextBtn = util.el("button", "btn", { text: "下一页" });
  el.appendChild(prevBtn);
  el.appendChild(infoEl);
  el.appendChild(nextBtn);
  if (opts.onPrev) prevBtn.addEventListener("click", opts.onPrev);
  if (opts.onNext) nextBtn.addEventListener("click", opts.onNext);

  function setInfo(text) {
    infoEl.textContent = text || "";
  }

  function set(state) {
    state = state || {};
    if (state.info != null) setInfo(state.info);
    if (state.prevDisabled != null) prevBtn.disabled = !!state.prevDisabled;
    if (state.nextDisabled != null) nextBtn.disabled = !!state.nextDisabled;
  }

  return { el: el, setInfo: setInfo, set: set, prevBtn: prevBtn, nextBtn: nextBtn };
}
