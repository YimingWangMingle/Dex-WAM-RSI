/* Motus Monitor 前端 —— 通用单选下拉组件（按钮 + 绝对定位菜单）。
 * 把旧 app.js:324-387 renderRunSelector 的签名机制通用化：
 *   update({items, currentId, signature})
 *     - signature 与上次渲染一致 → 完全不碰 DOM（轮询周期里重建会打断文字选择）
 *     - 菜单展开期 / 自动渲染被阻塞时有变化 → 记下 pending，菜单关闭后补建
 *   点击空白 / Esc 收起菜单。
 * createDropdown(opts):
 *   getId(item)            → id（缺省 item.id）
 *   renderButton(btn, currentItem|null)  填充按钮内容（currentItem=null 时给占位）
 *   renderOption(item)     → 菜单项元素（class 由组件加）
 *   onSelect(id, item)
 *   classes                {button, menu, option}（缺省沿用 run-select-*，与 style.css 匹配）
 * 返回 {el, update, isOpen, close, flush}；el 挂到调用方容器。
 */
import { util } from "../util.js";
import { domRenderBlocked } from "./guard.js";

var instances = [];  // 所有实例（flushDropdowns 统一补建 pending）
var globalBound = false;

/* 点空白收起 / Esc 收起：模块级单例监听（原 app.js bindControls 全局 click 的通用化，
 * 统一遍历实例，避免每实例各挂一份 document 监听） */
function ensureGlobalListeners() {
  if (globalBound) return;
  globalBound = true;
  document.addEventListener("click", function () { closeAllDropdowns(); });
  document.addEventListener("keydown", function (ev) {
    if (ev.key !== "Escape") return;
    for (var i = 0; i < instances.length; i++) if (instances[i].isOpen()) instances[i].close();
  });
}

export function createDropdown(opts) {
  opts = opts || {};
  var getId = opts.getId || function (item) { return item.id; };
  var classes = opts.classes || {};
  var clsButton = classes.button || "run-select-btn";
  var clsMenu = classes.menu || "run-select-menu";
  var clsOption = classes.option || "run-option";

  var box = util.el("div");
  var lastSig = null;        // 已渲染内容签名
  var pendingArgs = null;    // 菜单展开/渲染阻塞期间推迟的 update 参数
  var open = false;

  function rebuild(args) {
    lastSig = args.signature;
    pendingArgs = null;
    box.innerHTML = "";

    var current = null;
    for (var i = 0; i < args.items.length; i++) {
      if (getId(args.items[i]) === args.currentId) { current = args.items[i]; break; }
    }

    var btn = util.el("button", clsButton);
    opts.renderButton(btn, current);
    btn.appendChild(util.el("span", "caret", { text: "▾" }));
    box.appendChild(btn);

    var menu = util.el("div", clsMenu + " hidden");
    args.items.forEach(function (item) {
      var opt = opts.renderOption(item);
      opt.classList.add(clsOption);
      opt.addEventListener("click", function () {
        closeMenu();
        opts.onSelect(getId(item), item);
      });
      menu.appendChild(opt);
    });
    box.appendChild(menu);

    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      if (open) closeMenu();
      else openMenu();
    });
  }

  function update(args) {
    /* 签名对比：内容没变化就完全不碰 DOM */
    if (args.signature === lastSig && !pendingArgs) return;
    if (open || domRenderBlocked()) {
      pendingArgs = args;
      return;
    }
    rebuild(args);
  }

  function openMenu() {
    var menu = box.querySelector("." + clsMenu.split(" ")[0]);
    if (!menu) return;
    menu.classList.remove("hidden");
    open = true;
  }

  function closeMenu() {
    var menu = box.querySelector("." + clsMenu.split(" ")[0]);
    if (menu) menu.classList.add("hidden");
    if (!open) return;
    open = false;
    /* 菜单关闭后补建推迟的内容 */
    if (pendingArgs) rebuild(pendingArgs);
  }

  ensureGlobalListeners();

  var inst = {
    el: box,
    update: update,
    isOpen: function () { return open; },
    close: closeMenu,
    /* 交互结束（选区收起/控件失焦）后补一次 pending 重建（main.js flush 路径调用） */
    flush: function () {
      if (pendingArgs && !open && !domRenderBlocked()) rebuild(pendingArgs);
    },
    destroy: function () {
      var idx = instances.indexOf(inst);
      if (idx >= 0) instances.splice(idx, 1);
    },
  };
  instances.push(inst);
  return inst;
}

/* 收起全部展开中的菜单（模块级全局 click 调用） */
export function closeAllDropdowns() {
  for (var i = 0; i < instances.length; i++) if (instances[i].isOpen()) instances[i].close();
}

/* 全部实例补建 pending（main.js 的 flushDeferredRender 路径统一调用，
 * 对应旧 flushDeferredRender 里 runSelPending 的补渲染） */
export function flushDropdowns() {
  for (var i = 0; i < instances.length; i++) instances[i].flush();
}
