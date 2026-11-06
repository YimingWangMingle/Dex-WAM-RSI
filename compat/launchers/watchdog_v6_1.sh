#!/usr/bin/env bash
# V6.1 16-card watchdog. Run on BOTH pods (only NODE_RANK differs).
# Detects dead/hung training and coordinated-relaunches with a new rdzv id.
# Resume uses auto_resume_latest (own ckpt if any, else yaml finetune V5@20k).
set -u

NODE_RANK="${NODE_RANK:?set NODE_RANK=0 or 1}"
PEER_RANK=$((1 - NODE_RANK))
REPO=/code1/zxw_data/pretrain/ThinkerWAM-Depth-Action
LOG_DIR=/data/xiongwei.zhao/thinkerwam-depth/outputs/motus-depth_ablation_treatment_v6_1/logs
WATCH_DIR=/data/xiongwei.zhao/thinkerwam-depth/outputs/motus-depth_ablation_treatment_v6_1/watchdog
MASTER=zxw-wm-16card-job-v2-0.chen-liao-world-model-stage2-0825-v2-headless.ogi-pre-train.svc.cluster.local
POLL=60
# Process gone this long → restart. Log silent this long (init/eval included) → hung.
DEAD_SEC=120
STALL_SEC=2400
PEER_WAIT_SEC=480
MAX_RESTARTS=8
PATTERN='configs/depth_ablation_treatment_v6_1\.yaml'

mkdir -p "$WATCH_DIR" "$LOG_DIR"
WLOG="$WATCH_DIR/watchdog_node${NODE_RANK}.log"
STATUS="$WATCH_DIR/status_node${NODE_RANK}"
ALARM="$WATCH_DIR/alarm"
RDZV_FILE="$WATCH_DIR/rdzv"
RESTART_COUNT_FILE="$WATCH_DIR/restart_count"

log() { echo "[$(date '+%F %T')] [node${NODE_RANK}] $*" | tee -a "$WLOG"; }

count_train() {
  local n
  n="$(pgrep -fc 'python3.10 -u train/train.py' 2>/dev/null || true)"
  echo "${n:-0}"
}

latest_log() {
  ls -1t "$LOG_DIR"/train_node${1}_*.log 2>/dev/null | head -1
}

log_age_sec() {
  local f
  f="$(latest_log "$1")"
  [[ -n "$f" && -f "$f" ]] || { echo 999999; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$f") ))
}

training_done() {
  local f
  f="$(latest_log "$NODE_RANK")"
  [[ -n "$f" ]] || return 1
  rg -q 'Training completed|Reached max_steps|step 10000/10000' "$f" 2>/dev/null
}

write_status() {
  printf '%s\n' "$1" > "$STATUS"
}

read_count() {
  [[ -f "$RESTART_COUNT_FILE" ]] && cat "$RESTART_COUNT_FILE" || echo 0
}

launch_train() {
  local rdzv_id rdzv_port
  rdzv_id="$(awk -F= '/^id=/{print $2}' "$RDZV_FILE")"
  rdzv_port="$(awk -F= '/^port=/{print $2}' "$RDZV_FILE")"
  [[ -n "$rdzv_id" && -n "$rdzv_port" ]] || { log "ERROR no rdzv file"; return 1; }

  local ts nohup_log
  ts="$(date +%Y%m%d_%H%M%S)"
  nohup_log="$LOG_DIR/nohup_pod${NODE_RANK}_watchdog_${ts}.log"

  log "launch rdzv_id=$rdzv_id port=$rdzv_port nohup=$nohup_log"
  cd "$REPO" || return 1
  # shellcheck disable=SC1091
  source /home/fengodchen/miniconda3/etc/profile.d/conda.sh
  conda activate motus

  env -u MAX_STEPS -u SAVE_INTERVAL -u LOG_INTERVAL \
     -u RESUME_CHECKPOINT -u FINETUNE_CHECKPOINT \
     -u ELASTIC_SIMULATE_FAILURE_STEP -u ELASTIC_SIMULATE_FAILURE_RANK \
     MOTUS_OUTPUT_ROOT=/data/xiongwei.zhao/thinkerwam-depth \
     TASK=depth_ablation_treatment_v6_1 \
     CONFIG_FILE=configs/depth_ablation_treatment_v6_1.yaml \
     NNODES=2 NPROC_PER_NODE=8 NODE_RANK="$NODE_RANK" \
     RDZV_ENDPOINT="${MASTER}:${rdzv_port}" \
     RDZV_ID="$rdzv_id" \
     MAX_RESTARTS=3 AUTO_RESUME_LATEST=1 \
     setsid nohup bash train_mix_elastic.sh > "$nohup_log" 2>&1 < /dev/null &
  echo $! > "$WATCH_DIR/train_launch_pid_node${NODE_RANK}"
  sleep 8
  if pgrep -f "torchrun.*${PATTERN}|train/train.py.*${PATTERN}" >/dev/null; then
    log "launch ok pid=$(cat "$WATCH_DIR/train_launch_pid_node${NODE_RANK}")"
    return 0
  fi
  log "ERROR launch produced no torchrun/train.py; tail:"
  tail -20 "$nohup_log" | tee -a "$WLOG"
  return 1
}

kill_local_train() {
  pkill -f "$PATTERN" || true
  sleep 8
  pkill -9 -f "$PATTERN" || true
  sleep 4
  log "killed local train; remaining=$(count_train)"
}

coordinated_restart() {
  local reason="$1"
  local n
  n="$(read_count)"
  if (( n >= MAX_RESTARTS )); then
    log "ERROR restart cap $MAX_RESTARTS reached; not launching. reason=$reason"
    write_status "DEAD cap reason=$reason"
    return 1
  fi
  echo $((n + 1)) > "$RESTART_COUNT_FILE"
  log "RESTART #$((n + 1)) reason=$reason"

  echo "$reason $(date '+%F %T') node${NODE_RANK}" >> "$ALARM"
  kill_local_train
  : > "$WATCH_DIR/ready_${NODE_RANK}"

  local waited=0
  while [[ ! -f "$WATCH_DIR/ready_${PEER_RANK}" ]]; do
    if (( waited >= PEER_WAIT_SEC )); then
      log "peer ready timeout ${PEER_WAIT_SEC}s — launching anyway (peer watchdog may be down)"
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done

  if [[ "$NODE_RANK" == "0" ]]; then
    local port
    port=$((21140 + n))
    printf 'id=v6_1_wd_%s\nport=%s\n' "$(date +%H%M%S)" "$port" > "$RDZV_FILE"
    log "wrote $RDZV_FILE"
    sleep 2
  else
    local w=0
    while [[ ! -s "$RDZV_FILE" ]]; do
      if (( w >= 90 )); then
        log "ERROR waited 90s for rank0 rdzv file"
        return 1
      fi
      sleep 2
      w=$((w + 2))
    done
  fi

  launch_train || return 1
  rm -f "$WATCH_DIR/ready_0" "$WATCH_DIR/ready_1"
  # keep ALARM as history; new cycle only if we write again
  write_status "RESTARTED $(date '+%F %T') reason=$reason"
}

# --- main ---
log "watchdog start hostname=$(hostname) dead=${DEAD_SEC}s stall=${STALL_SEC}s"
write_status "START $(date '+%F %T')"
dead_since=0
stall_since=0

while true; do
  if training_done; then
    log "training completed; watchdog idle"
    write_status "DONE $(date '+%F %T')"
    sleep 300
    continue
  fi

  n_train="$(count_train)"
  age="$(log_age_sec "$NODE_RANK")"
  peer_age="$(log_age_sec "$PEER_RANK")"
  write_status "OK $(date '+%F %T') train=$n_train log_age=${age}s peer_age=${peer_age}s"

  if (( n_train < 8 )); then
    dead_since=$((dead_since + POLL))
  else
    dead_since=0
  fi

  if (( n_train >= 8 && age > STALL_SEC )); then
    stall_since=$((stall_since + POLL))
  else
    stall_since=0
  fi

  # Peer vanished: their log froze while we still have processes. Yesterday
  # this was 04:47 (pod1) → 05:18 (pod0 NCCL death). Don't wait the full stall.
  if (( n_train >= 8 && peer_age > 900 && age > 300 )); then
    coordinated_restart "peer_log_stale peer_age=${peer_age}s local_age=${age}s"
    dead_since=0
    stall_since=0
    sleep "$POLL"
    continue
  fi

  if (( dead_since >= DEAD_SEC )); then
    coordinated_restart "train_dead count=$n_train for ${dead_since}s"
    dead_since=0
    stall_since=0
    sleep "$POLL"
    continue
  fi

  if (( stall_since >= 60 )); then
    coordinated_restart "log_stall age=${age}s"
    dead_since=0
    stall_since=0
    sleep "$POLL"
    continue
  fi

  sleep "$POLL"
done
