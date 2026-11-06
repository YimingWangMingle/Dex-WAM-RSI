/* Motus Monitor 前端 —— 裸 ECharts 实例封装（阶段 2C 新基建）。
 * 面向 line.js/stacked.js 的序列 API 覆盖不了的自定义图（dataset tab 统一时间轴、
 * action 曲线——markArea 当前帧区间 + 点击 seek、切片源占比饼图）：option 完全
 * 由调用方组装，本模块只管生命周期与主题登记。
 *
 *   createEChart(el, {height?, option(), replaceMerge?})
 *     el            块级容器；旧的 <canvas> 占位会原位换成同宽 div（class 保留，
 *                   同 line.js 模式，index.html 不动）
 *     height        可选，宿主行内高度 px（旧 setupCanvas 的固定高：时间轴 64、
 *                   action 220、饼图 132）
 *     option()      构建完整 option；每次 apply 重新调用，读取调用方最新状态。
 *                   应并入 charts/theme.js chartTheme() 以跟随主题
 *     replaceMerge  setOption 的 replaceMerge 组件列表（默认 series/xAxis/yAxis，
 *                   序列数变少时清掉残留；dataZoom/graphic 等不带即按 id 合并）
 *   → {host, apply(), chart(), refreshTheme(), resize(), dispose()}
 *     chart()  原始 echarts 实例（未初始化时为 null：0 尺寸容器推迟初始化，
 *              ResizeObserver 见到非 0 尺寸才 init，同 line.js）；调用方需要
 *              convertFromPixel / 局部 setOption 合并时取用，用前判空
 *
 * 创建即登记 theme.js 注册表：主题切换由 applyTheme() 统一刷新全部活图表。
 */
import { registerChart } from "./theme.js";

export function createEChart(el, opts) {
  opts = opts || {};
  var host = el;
  if (host && host.tagName === "CANVAS") {
    var div = document.createElement("div");
    div.className = host.className;
    div.style.width = "100%";
    host.parentElement.insertBefore(div, host);
    host.parentElement.removeChild(host);
    host = div;
  }
  if (opts.height) host.style.height = opts.height + "px";

  var inst = null;   // echarts 实例（0 尺寸推迟初始化）
  var dead = false;  // dispose 后阻止 RO 再 init

  function replace() { return { replaceMerge: opts.replaceMerge || ["series", "xAxis", "yAxis"] }; }

  function apply() {
    if (!inst) { tryInit(); return; }
    inst.setOption(opts.option(), replace());
  }

  function tryInit() {
    if (dead || inst || typeof window.echarts === "undefined") return;
    if (host.clientWidth <= 0 || host.clientHeight <= 0) return;
    inst = window.echarts.init(host);
    inst.setOption(opts.option(), replace());
  }

  var ro = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(function () {
      if (host.clientWidth <= 0 || host.clientHeight <= 0) return;
      if (!inst) tryInit();
      else inst.resize();
    });
    ro.observe(host);
  }

  var api = {
    host: host,

    apply: apply,

    chart: function () { return inst; },

    refreshTheme: function () { apply(); },

    resize: function () {
      if (!inst) { tryInit(); return; }
      inst.resize();
    },

    dispose: function () {
      dead = true;
      if (ro) ro.disconnect();
      unregister();
      if (inst) { inst.dispose(); inst = null; }
    },
  };

  var unregister = registerChart(api);
  tryInit();
  return api;
}
