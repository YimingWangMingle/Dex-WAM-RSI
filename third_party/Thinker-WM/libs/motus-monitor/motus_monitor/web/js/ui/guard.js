/* Motus Monitor 前端 —— 自动渲染守卫。
 * 从旧 web/app.js:1446-1481 逐字迁出（原闭包 state 改为显式传参，逻辑不变）。
 * eval tab 等仍依赖它：周期刷新造成的 DOM 重建会清掉用户选区/收起下拉，
 * 此时推迟渲染，待交互结束（selectionchange/change/blur）后由 main.js 补一次。
 */

/* 用户正在选择文字、或焦点在表单控件（下拉/输入框）上时返回 true */
export function domRenderBlocked() {
  try {
    var sel = window.getSelection && window.getSelection();
    if (sel && !sel.isCollapsed && String(sel).length) return true;
  } catch (e) {}
  /* 焦点在下拉/文本域上 = 用户正在操作（SELECT 展开时重建选项会收起它）。
     不含 INPUT：滑块/复选框点击后长期持有焦点，误伤会导致图表停止跟随。 */
  var ae = document.activeElement;
  if (ae && /^(SELECT|TEXTAREA)$/.test(ae.tagName || "")) return true;
  return false;
}

/* 返回 true = 本次自动渲染已推迟 */
export function guardAutoRender(state) {
  if (!domRenderBlocked()) return false;
  state.renderDeferred = true;
  return true;
}

/* 交互结束后补一次被推迟的渲染。onFlush 由 main.js 注入（触发一次数据刷新路由）。
 * 旧实现里 runSelPending 的补渲染归 ui/dropdown.js 的 pending 机制（见 main.js 调用顺序）。 */
export function flushDeferredRender(state, onFlush) {
  if (domRenderBlocked()) return;
  if (!state.renderDeferred) return;
  state.renderDeferred = false;
  if (onFlush) onFlush();
}
