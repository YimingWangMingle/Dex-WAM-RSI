/* Motus Monitor 前端 —— 指标定义与耗时检查点分组（纯数据逻辑，无 DOM）。
 * 从旧 web/app.js 逐字迁出（app.js:46-76 指标定义、app.js:85-272 耗时检查点）。
 * 注意：discoverTimingGroups 由"读全局 state.steps"改为显式传参 steps（模块化需要，
 * 逻辑不变）；timing tab 的渲染代码不在此处（归 tabs/timing.js）。
 */
import { util } from "./util.js";
import { PALETTE } from "./charts/theme.js";

/* ------------------------------------------------------------ 指标定义 */
/* 模态 dropout 步：被 drop 分支的 loss 置 0、total_loss 只剩单分支，都不具
   可比性，曲线里忽略这些点（返回 null 即被跳过）。
   新记录（20260807 起）带 modality_dropped_* 标志，优先用标志判定——可区分
   "本步被 drop" 与 "单分支训练该分量恒为 0"；无标志的老记录只对分量自身退化
   为 0 判断，total_loss 不做猜测（避免误空单分支 run 的整条曲线）。 */
export function modalityDropFlag(r, mod) {
  var m = r.metrics;
  var flag = m && m["modality_dropped_" + mod];
  return util.isNum(flag) ? flag !== 0 : null;
}
export function droppedModality(r, mod) {
  var flag = modalityDropFlag(r, mod);
  if (flag !== null) return flag;
  var m = r.metrics;
  return !!(m && m[mod + "_loss"] === 0);
}
export function nonZeroLoss(v) {
  return util.isNum(v) && v !== 0 ? v : null;
}
export const METRIC_DEFS = [
  { key: "total_loss", label: "total_loss", on: true, get: function (r) {
      if (modalityDropFlag(r, "video") || modalityDropFlag(r, "action") ||
          modalityDropFlag(r, "depth")) return null;
      return nonZeroLoss(r.metrics && r.metrics.total_loss);
    } },
  { key: "video_loss", label: "video_loss", on: true, get: function (r) {
      return droppedModality(r, "video") ? null : nonZeroLoss(r.metrics && r.metrics.video_loss);
    } },
  { key: "action_loss", label: "action_loss", on: true, get: function (r) {
      return droppedModality(r, "action") ? null : nonZeroLoss(r.metrics && r.metrics.action_loss);
    } },
  { key: "depth_loss", label: "depth_loss", on: true, get: function (r) {
      return droppedModality(r, "depth") ? null : nonZeroLoss(r.metrics && r.metrics.depth_loss);
    } },
  { key: "lr_main", label: "lr_main", on: false, get: function (r) { return r.lr_main; } },
  { key: "lr_wan", label: "lr_wan", on: false, get: function (r) { return r.lr_wan; } },
  { key: "grad_norm", label: "grad_norm", on: false, get: function (r) { return r.grad_total_norm_before_clip; } },
  { key: "step_time", label: "step_time", on: false, get: function (r) { return r.step_time_seconds; } },
  { key: "dataloader", label: "dataloader", on: false, get: function (r) { return r.dataloader_next_seconds; } },
];

/* ------------------------------------------------------------ 耗时检查点（数据驱动）
   检查点集合随训练代码版本变化，不做硬编码清单：
     loop   = step 记录里的 *_seconds 键（非 model_ 前缀）
     model  = model_*_seconds 键（模型/分支/逐 block 检查点）
     custom = stage_seconds.*（MonitorLogger.stage() 自定义检查点）
   已知的"合计"键（loop_total/step_time 等）单独展示、不进堆叠，
   未知新键会自动出现在对应分组里，保证不同版本不会出错。 */
export const TIMING_TOTAL_KEYS = [
  ["loop_total_seconds", "loop_total"],
  ["step_time_seconds", "step_time"],
  ["train_step_total_seconds", "train_step_total"],
  ["model_training_step_seconds", "前向合计"],
  ["model_world_model_forward_seconds", "world_model合计"],
];
export const TIMING_GROUP_DEFS = [
  { id: "loop", title: "训练循环" },
  { id: "model", title: "模型内部" },
  { id: "custom", title: "自定义 stage" },
];
var timingColorCache = {};

/* 阶段排序：按函数执行先后（而不是耗时大小）。
   已知键在顺序表中给出执行顺序；未知键（含 *_block_XX）挂到其父键之后，
   再按名字字典序兜底——不同版本的检查点集合都能稳定、正确地展示。 */
export const LOOP_STAGE_ORDER = [
  "dataloader_next_seconds", "dataloader_iter_restart_seconds",
  "ready_queue_get_wait_seconds",
  "ready_queue_producer_batch_seconds", "ready_queue_producer_dataloader_seconds",
  "ready_queue_producer_state_age_seconds", "ready_queue_producer_clone_seconds",
  "ready_queue_producer_put_wait_seconds",
  "optimizer_zero_grad_seconds", "host_to_device_seconds",
  "backward_seconds", "grad_clip_seconds",
  "optimizer_step_seconds", "scheduler_step_seconds",
  "validation_seconds",
  "checkpoint_prepare_seconds", "accelerator_save_state_seconds",
  "checkpoint_config_seconds", "checkpoint_marker_seconds", "checkpoint_total_seconds",
];
export const MODEL_STAGE_ORDER = [
  "model_ego4d_prompt_seconds", "model_scheduler_sample_seconds",
  "model_video_prepare_seconds", "model_vae_encode_video_seconds",
  "model_noise_and_condition_seconds", "model_text_prepare_seconds",
  "model_wm_t5_seconds",
  "model_wm_video_encode_seconds",
  "model_wm_und_encode_seconds", "model_wm_und_vlm_seconds", "model_wm_und_encoder_seconds",
  "model_wm_action_encode_seconds",
  "model_wm_uni_process_seconds",
  "model_wm_video_decode_seconds",
  "model_wm_action_decode_seconds",
  "model_loss_compute_seconds",
];

/* 父子包含关系：parent ⊇ children（如 action_encode ⊇ action_encode_block_*，
   checkpoint_total ⊇ prepare/save_state/config/marker）。block 循环按命名
   规则自动识别（X_block_NN_seconds 属于 X_seconds）；少量非 block 关系显式
   声明。未知新键自动成为顶层，不会出错；占比/堆叠只统计顶层（避免重复计数）。 */
export const STAGE_CHILD_PARENT = {
  "model_wm_und_vlm_seconds": "model_wm_und_encode_seconds",
  "model_wm_und_encoder_seconds": "model_wm_und_encode_seconds",
  "checkpoint_prepare_seconds": "checkpoint_total_seconds",
  "accelerator_save_state_seconds": "checkpoint_total_seconds",
  "checkpoint_config_seconds": "checkpoint_total_seconds",
  "checkpoint_marker_seconds": "checkpoint_total_seconds",
};

export function resolveStageTree(stages) {
  var byKey = {};
  stages.forEach(function (s) { byKey[s.key] = s; });
  var tops = [];
  stages.forEach(function (s) {
    var parentKey = STAGE_CHILD_PARENT[s.key] || null;
    if (!parentKey) {
      var m = s.key.match(/^(.*)_block_\d+_seconds$/);
      if (m) {
        if (byKey[m[1] + "_seconds"]) parentKey = m[1] + "_seconds";
        else if (byKey[m[1] + "_process_seconds"]) parentKey = m[1] + "_process_seconds";
      }
    }
    if (parentKey && byKey[parentKey] && parentKey !== s.key) {
      var parent = byKey[parentKey];
      (parent._children = parent._children || []).push(s);
      return;
    }
    tops.push(s);
  });
  return tops;
}

export function stageOrderIndex(groupId, key) {
  var list = groupId === "model" ? MODEL_STAGE_ORDER : LOOP_STAGE_ORDER;
  var idx = list.indexOf(key);
  if (idx >= 0) return idx;
  var stem = key.replace(/_block_\d+_seconds$/, "");
  if (stem !== key) {
    idx = list.indexOf(stem + "_seconds");
    if (idx >= 0) return idx + 0.5;
    idx = list.indexOf(stem + "_process_seconds");
    if (idx >= 0) return idx + 0.5;
  }
  return list.length + 1;
}

export function colorForKey(key) {
  if (!timingColorCache[key]) {
    var h = 0;
    for (var i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
    timingColorCache[key] = PALETTE[h % PALETTE.length];
  }
  return timingColorCache[key];
}

export function isTimingTotalKey(key) {
  for (var i = 0; i < TIMING_TOTAL_KEYS.length; i++) {
    if (TIMING_TOTAL_KEYS[i][0] === key) return true;
  }
  return false;
}

export function prettyStageLabel(key) {
  var k = key;
  if (k.indexOf("model_") === 0) k = k.slice(6);
  if (k.slice(-8) === "_seconds") k = k.slice(0, -8);
  if (k.indexOf("wm_") === 0) k = k.slice(3);
  return k;
}

export function meanOfKey(rows, get) {
  var sum = 0, cnt = 0;
  for (var i = 0; i < rows.length; i++) {
    var v = get(rows[i]);
    if (util.isNum(v)) { sum += v; cnt++; }
  }
  return cnt ? sum / cnt : null;
}

/* 数据驱动分组发现：入参 steps = store.state.steps（旧实现直接读全局 state）。 */
export function discoverTimingGroups(steps) {
  var seen = { loop: {}, model: {}, custom: {} };
  for (var i = 0; i < steps.length; i++) {
    var r = steps[i];
    for (var key in r) {
      if (key.slice(-8) !== "_seconds" || isTimingTotalKey(key) || !util.isNum(r[key])) continue;
      if (key.indexOf("model_") === 0) seen.model[key] = true;
      else seen.loop[key] = true;
    }
    var ss = r.stage_seconds;
    if (ss && typeof ss === "object") {
      for (var sk in ss) if (util.isNum(ss[sk])) seen.custom[sk] = true;
    }
  }
  /* 排除"累计计数器"（如 ready_queue_*_wait_total_seconds）：它们随时间单调
   * 不减，不是每步耗时，混进阶段堆叠会掩盖真实分布（如初始化期的队列预热）。
   * 判定：近窗口内值单调不减且末值大于首值。 */
  var monoWin = steps.slice(-50);
  var cumulative = {};
  Object.keys(seen.loop).forEach(function (key) {
    var prev = null, mono = true, cnt = 0, first = null, last = null;
    for (var j = 0; j < monoWin.length; j++) {
      var v = monoWin[j][key];
      if (!util.isNum(v)) continue;
      cnt++;
      if (first === null) first = v;
      if (prev !== null && v < prev - 1e-9) { mono = false; break; }
      prev = v; last = v;
    }
    if (mono && cnt >= 5 && last > first) cumulative[key] = true;
  });
  var recent = steps.slice(-100);
  var groups = [];
  TIMING_GROUP_DEFS.forEach(function (g) {
    var isCustom = g.id === "custom";
    var keys = Object.keys(seen[g.id]).filter(function (k) {
      return !(g.id === "loop" && cumulative[k]);
    }).sort();
    var stages = keys.map(function (key) {
      var get = isCustom
        ? function (r) { return r.stage_seconds && r.stage_seconds[key]; }
        : function (r) { return r[key]; };
      return {
        key: key, label: prettyStageLabel(key), color: colorForKey(key),
        mean: meanOfKey(recent, get), get: get,
      };
    });
    /* 按函数执行顺序排序（自定义组保持键名字典序） */
    if (g.id !== "custom") {
      stages.sort(function (a, b) {
        var oa = stageOrderIndex(g.id, a.key), ob = stageOrderIndex(g.id, b.key);
        if (oa !== ob) return oa - ob;
        return a.key < b.key ? -1 : 1;
      });
    }
    if (stages.length) groups.push({ id: g.id, title: g.title, stages: stages });
  });
  return groups;
}
