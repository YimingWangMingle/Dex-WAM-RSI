/* Motus Monitor 前端 —— 通用缩进树渲染组件。
 * 供 dataset epTree/sliceTree 与训练树复用（参考旧 web/dataset.js:479-503、
 * 1016-1050、web/tree.js:30-129 抽象）。组件只负责：展开状态表 + 递归行走 +
 * 展开/折叠后重渲染；行 DOM 完全由回调产出（缩进/样式归调用方 CSS）。
 *
 * createTreeView(opts):
 *   getKey(node)              → 展开状态表的键
 *   getChildren(node)         → 子节点数组 | null（叶子）
 *   renderRow(node, depth, ctl) → 行元素 | null（null = 不产出行仍下钻子节点）
 *     ctl = {expanded, toggle()}：行内挂 toggle 到点击事件即可展开/折叠
 *   defaultExpanded(node)     → 未记录时的默认展开（可选，缺省 false）
 *   expanded                  外部传入的展开状态表（可选，缺省组件自建 {}）
 * 返回 {render(container, roots), redraw(), expanded, setExpanded(key, bool)}
 * render 后组件记住 container/roots，toggle 自动 redraw。
 */
export function createTreeView(opts) {
  opts = opts || {};
  var getKey = opts.getKey || function (node) { return node.key; };
  var expanded = opts.expanded || {};
  var container = null;
  var roots = [];

  function isExpanded(node) {
    var key = getKey(node);
    if (Object.prototype.hasOwnProperty.call(expanded, key)) return !!expanded[key];
    return opts.defaultExpanded ? !!opts.defaultExpanded(node) : false;
  }

  function setExpanded(key, bool) {
    expanded[key] = !!bool;
  }

  function walk(node, depth) {
    var ctl = {
      expanded: isExpanded(node),
      toggle: function () {
        setExpanded(getKey(node), !isExpanded(node));
        redraw();
      },
    };
    var row = opts.renderRow(node, depth, ctl);
    if (row) container.appendChild(row);
    if (!ctl.expanded) return;
    var children = opts.getChildren ? opts.getChildren(node) : null;
    if (!children) return;
    for (var i = 0; i < children.length; i++) walk(children[i], depth + 1);
  }

  function redraw() {
    if (!container) return;
    container.innerHTML = "";
    for (var i = 0; i < roots.length; i++) walk(roots[i], 0);
  }

  function render(box, list) {
    container = box;
    roots = list || [];
    redraw();
  }

  return {
    render: render,
    redraw: redraw,
    expanded: expanded,
    setExpanded: setExpanded,
  };
}
