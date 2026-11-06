/* Motus Monitor 前端 —— 统一弹层组件。
 * openModal({content, className, cardClass, onClose, onEsc, backdrop}) → {el, close()}
 *   content   弹层内容元素（放进卡片容器）
 *   className 叠加在 overlay 上的额外 class
 *   cardClass 卡片容器的 class（缺省 "modal-card card"，传字符串则整体替换）
 *   onClose   关闭时清理钩子（只调一次）
 *   onEsc     Esc 前置钩子：返回 true 表示消费事件不关闭
 *             （eval action 查看器单维聚焦时 Esc 先返回网格，见旧 eval.js:385-394）
 *   backdrop  点击遮罩关闭（默认 true）
 * 行为：Esc 关闭栈（多个弹层只关最上层）、backdrop 点击关闭、body 滚动锁。
 * DOM class 与 style.css 匹配：缺省 .modal-overlay > .modal-card.card（190-194 行）；
 * eval 风格用法：openModal({className: "eval-modal-backdrop", cardClass: "eval-modal-box", …}），
 * 结构与 .eval-modal-*（977-1030 行）对应。
 */
import { util } from "../util.js";

var stack = [];        // 打开中的弹层（栈底 → 栈顶）
var escBound = false;
var scrollLocks = 0;

function lockScroll() {
  scrollLocks++;
  document.body.style.overflow = "hidden";
}

function unlockScroll() {
  scrollLocks = Math.max(0, scrollLocks - 1);
  if (!scrollLocks) document.body.style.overflow = "";
}

function onKeydown(ev) {
  if (ev.key !== "Escape") return;
  var top = stack[stack.length - 1];
  if (!top) return;
  if (top.onEsc && top.onEsc()) return;   /* 弹层自己消费（如返回上一层视图） */
  top.close();
}

function ensureEsc() {
  if (escBound) return;
  escBound = true;
  document.addEventListener("keydown", onKeydown);
}

export function openModal(opts) {
  opts = opts || {};
  var overlay = util.el("div", "modal-overlay" + (opts.className ? " " + opts.className : ""));
  var card = util.el("div", opts.cardClass || "modal-card card");
  if (opts.content) card.appendChild(opts.content);
  overlay.appendChild(card);

  var entry = {
    el: overlay,
    onEsc: opts.onEsc || null,
    close: close,
  };

  var closed = false;
  function close() {
    if (closed) return;
    closed = true;
    var idx = stack.indexOf(entry);
    if (idx >= 0) stack.splice(idx, 1);
    if (overlay.parentElement) overlay.parentElement.removeChild(overlay);
    unlockScroll();
    if (opts.onClose) {
      try { opts.onClose(); } catch (e) { console.error(e); }
    }
  }

  if (opts.backdrop !== false) {
    overlay.addEventListener("click", function (ev) {
      if (ev.target === overlay) close();
    });
  }

  document.body.appendChild(overlay);
  stack.push(entry);
  lockScroll();
  ensureEsc();
  return entry;
}
