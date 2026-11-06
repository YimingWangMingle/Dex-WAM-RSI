#!/usr/bin/env bash
# 覆盖 launch_v15_full_16.sh 的后台交棒这一层：普通 preflight 只验到第一层，
# 看不到 detach 之后的子进程环境。这里用 stub train_mix_elastic.sh 接棒，
# 断言 init/resume 两种模式真正把参数传到了训练脚本，并验证旧的“后台重跑
# 本脚本”写法会被本测试抓住（子进程撞 FINETUNE_CHECKPOINT 拒收）。
set -euo pipefail

REPO=/code1/zxw_data/pretrain/ThinkerWAM-Depth-Action-RGB-RoboTwin-V14
SRC="$REPO/tmp/launch_v15_full_32.sh"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# 把 launcher 的工作目录和日志根指到 scratch，其余逻辑（hostname 判定、
# checkpoint 完整性校验、模式判定、detach）保持原样。
make_launcher() {  # $1=目标文件
  sed -e "s#^cd /code1/zxw_data/pretrain/ThinkerWAM-Depth-Action-RGB-RoboTwin-V14\$#cd $SCRATCH#" \
      -e "s#^export MOTUS_OUTPUT_ROOT=.*\$#export MOTUS_OUTPUT_ROOT=$SCRATCH/out#" \
      "$SRC" > "$1"
  grep -q "^cd $SCRATCH\$" "$1" || { echo "FAIL: cd 未被重写"; exit 1; }
  grep -q "^export MOTUS_OUTPUT_ROOT=$SCRATCH/out\$" "$1" || { echo "FAIL: 日志根未被重写"; exit 1; }
}

cat > "$SCRATCH/train_mix_elastic.sh" <<'STUB'
{
  echo "TASK=${TASK:-}"
  echo "CONFIG_FILE=${CONFIG_FILE:-}"
  echo "FINETUNE_CHECKPOINT=${FINETUNE_CHECKPOINT:-}"
  echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}"
  echo "AUTO_RESUME_LATEST=${AUTO_RESUME_LATEST:-}"
  echo "MAX_RESTARTS=${MAX_RESTARTS:-}"
  echo "RDZV_ID=${RDZV_ID:-}"
  echo "RDZV_ENDPOINT=${RDZV_ENDPOINT:-}"
  echo "MASTER_ADDR=${MASTER_ADDR:-}"
  echo "MASTER_PORT=${MASTER_PORT:-}"
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}"
  echo "NNODES=${NNODES:-} NPROC_PER_NODE=${NPROC_PER_NODE:-}"
  echo "MAX_STEPS=${MAX_STEPS:-<unset>}"
  echo "PWD=$PWD"
} > "$HANDOFF"
STUB

V14_CKPT=/data/xiongwei.zhao/thinkerwam-depth/checkpoints/v14_robotwin_full/robotwin_3d_full_v14_rgbd/V14_full/checkpoint_step_10000
[[ -f "$V14_CKPT/.motus_checkpoint_complete.json" ]] || { echo "FAIL: V14@10000 不完整"; exit 1; }

wait_for() {  # $1=文件 $2=秒
  for _ in $(seq 1 $(( $2 * 10 ))); do [[ -s "$1" ]] && return 0; sleep 0.1; done
  return 1
}

echo "=== 1. init 模式后台启动，应交棒到 train_mix_elastic.sh ==="
make_launcher "$SCRATCH/launch.sh"
export HANDOFF="$SCRATCH/handoff_init.env"
V15_MODE=init V15_RUN_DIR="$SCRATCH/run_init" HANDOFF="$HANDOFF" \
  bash "$SCRATCH/launch.sh"
wait_for "$HANDOFF" 20 || { echo "FAIL: 后台子进程没走到 train_mix_elastic.sh"; \
  echo "--- nohup ---"; cat "$SCRATCH"/out/outputs/motus-V15_full/logs/*.log; exit 1; }
cat "$HANDOFF"
grep -qx "FINETUNE_CHECKPOINT=$V14_CKPT" "$HANDOFF" || { echo "FAIL: 权重来源没传到"; exit 1; }
grep -qx "RESUME_CHECKPOINT=" "$HANDOFF" || { echo "FAIL: init 不该带 resume"; exit 1; }
grep -qx "AUTO_RESUME_LATEST=0" "$HANDOFF" || { echo "FAIL: init 必须显式 0"; exit 1; }
grep -qx "MAX_RESTARTS=0" "$HANDOFF" || { echo "FAIL: init 应禁自动重启"; exit 1; }
grep -qx "TASK=V15_full" "$HANDOFF" || { echo "FAIL: TASK 错"; exit 1; }
grep -qx "CONFIG_FILE=configs/robotwin_3d_full_v15_rgbd.yaml" "$HANDOFF" || { echo "FAIL: 配置错"; exit 1; }
grep -qx "RDZV_ID=V15_full" "$HANDOFF" || { echo "FAIL: RDZV_ID 错"; exit 1; }
grep -qx "RDZV_ENDPOINT=10.233.66.58:21130" "$HANDOFF" || { echo "FAIL: RDZV 必须是 rank0 Pod IP:21130"; exit 1; }
grep -qx "MASTER_ADDR=10.233.66.58" "$HANDOFF" || { echo "FAIL: MASTER_ADDR 必须是 rank0 Pod IP"; exit 1; }
grep -qx "MASTER_PORT=29501" "$HANDOFF" || { echo "FAIL: MASTER_PORT 必须是 29501"; exit 1; }
grep -qx "NCCL_IB_DISABLE=0" "$HANDOFF" || { echo "FAIL: 必须走 IB"; exit 1; }
grep -qx "NNODES=4 NPROC_PER_NODE=8" "$HANDOFF" || { echo "FAIL: 规模应为 4×8"; exit 1; }
grep -qx "MAX_STEPS=<unset>" "$HANDOFF" || { echo "FAIL: MAX_STEPS 必须留给 YAML"; exit 1; }
grep -qx "PWD=$SCRATCH" "$HANDOFF" || { echo "FAIL: 工作目录错"; exit 1; }
echo "PASS init 后台交棒"

echo "=== 2. resume 模式后台启动 ==="
FAKE_RUN="$SCRATCH/run_resume"
mkdir -p "$FAKE_RUN/checkpoint_step_4000"
echo '{"status":"complete","step":4000}' > "$FAKE_RUN/checkpoint_step_4000/.motus_checkpoint_complete.json"
export HANDOFF="$SCRATCH/handoff_resume.env"
V15_MODE=resume V15_RUN_DIR="$FAKE_RUN" HANDOFF="$HANDOFF" bash "$SCRATCH/launch.sh"
wait_for "$HANDOFF" 20 || { echo "FAIL: resume 没交棒"; exit 1; }
grep -qx "FINETUNE_CHECKPOINT=" "$HANDOFF" || { echo "FAIL: resume 不该带 finetune"; exit 1; }
grep -qx "AUTO_RESUME_LATEST=1" "$HANDOFF" || { echo "FAIL: resume 应为 1"; exit 1; }
grep -qx "MAX_RESTARTS=3" "$HANDOFF" || { echo "FAIL: resume 应允许重启"; exit 1; }
echo "PASS resume 后台交棒"

echo "=== 3. init 仍拒绝外部传入 FINETUNE_CHECKPOINT ==="
if FINETUNE_CHECKPOINT=/tmp/whatever V15_MODE=init V15_RUN_DIR="$SCRATCH/run_x" \
     HANDOFF="$SCRATCH/handoff_x.env" bash "$SCRATCH/launch.sh" 2>/dev/null; then
  echo "FAIL: 外部 FINETUNE_CHECKPOINT 应被拒"; exit 1
fi
echo "PASS 外部覆盖仍被拒"

echo "=== 4. FOREGROUND=1 不 detach，前台直接交棒 ==="
export HANDOFF="$SCRATCH/handoff_fg.env"
FOREGROUND=1 V15_MODE=init V15_RUN_DIR="$SCRATCH/run_fg" HANDOFF="$HANDOFF" \
  bash "$SCRATCH/launch.sh"
[[ -s "$HANDOFF" ]] || { echo "FAIL: FOREGROUND 没交棒"; exit 1; }
echo "PASS FOREGROUND 前台交棒"

echo "=== 5. 反向验证：旧的“后台重跑本脚本”写法会被本测试抓住 ==="
sed 's#^  setsid nohup bash train_mix_elastic.sh > "$NOHUP_LOG" 2>&1 < /dev/null &$#  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" \&\& pwd)/$(basename "${BASH_SOURCE[0]}")"\n  setsid nohup bash "$SCRIPT_PATH" "$@" > "$NOHUP_LOG" 2>\&1 < /dev/null \&#' \
  "$SCRATCH/launch.sh" > "$SCRATCH/launch_old.sh"
grep -q 'SCRIPT_PATH=' "$SCRATCH/launch_old.sh" || { echo "FAIL: 旧写法没构造出来"; exit 1; }
OLD_LOGS="$SCRATCH/out/outputs/motus-V15_full/logs"
rm -f "$OLD_LOGS"/*.log
export HANDOFF="$SCRATCH/handoff_old.env"
V15_MODE=init V15_RUN_DIR="$SCRATCH/run_old" HANDOFF="$HANDOFF" bash "$SCRATCH/launch_old.sh"
sleep 3
[[ -s "$HANDOFF" ]] && { echo "FAIL: 旧写法本应起不来"; exit 1; }
grep -q "does not accept generic" "$OLD_LOGS"/*.log || { echo "FAIL: 旧写法失败原因不符"; cat "$OLD_LOGS"/*.log; exit 1; }
echo "PASS 旧写法复现原故障（子进程拒收自己 export 的 FINETUNE_CHECKPOINT）"

echo
echo "ALL PASS"
