#!/usr/bin/env bash
# Everything still owed, in value order, strictly sequential.
#
# Waits for the W2S extraction already running, then:
#   1. task-level bounds on BioSR (4 caches, 29 specs, task readouts)
#   2. Phase 3 background-aware unrolled control (both arms, matched budget)
#
# Sequential because the machine has 8 GB and 6 torch threads; two extractions
# in parallel would swap rather than go faster. Measured: task metrics add no
# detectable overhead over Phase 2, so the task pass costs what Phase 2 cost.
set -u

# Resolve the repository root from this script's own location, so the
# driver runs from anywhere rather than only on the machine it was
# written on.
cd "$(dirname "$0")/.." || exit 1
LOG=ieee/logs
TASK_OUT=experiments/features29_task
W2S_OUT=experiments/features29_w2s
mkdir -p "$LOG" "$TASK_OUT"

echo "=== waiting for the W2S extraction to finish ==="
for i in $(seq 1 720); do   # up to 4 h
  if [ -f "$W2S_OUT/rl_photons_15.npz" ]; then
    echo "[$(date +%H:%M:%S)] W2S complete"
    break
  fi
  sleep 20
done

# ---------------------------------------------------------------- task bounds
task () {
  local name="$1"; shift
  if [ -f "$TASK_OUT/$name.npz" ]; then
    echo "[$(date +%H:%M:%S)] SKIP task $name"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] TASK $name"
  local t0=$SECONDS
  python scripts/extract_features.py "$@" --include-physical --task-metrics \
    --source data/biosr_gt.npz --haze-frac 0.31 --crop 256 \
    --out "$TASK_OUT/$name.npz" > "$LOG/task_$name.log" 2>&1
  local rc=$? dt=$(( SECONDS - t0 ))
  echo "[$(date +%H:%M:%S)] task $name rc=$rc in $((dt/60))m$((dt%60))s"
  [ $rc -ne 0 ] && tail -5 "$LOG/task_$name.log"
  return $rc
}

echo "=== 1. task-level bounds, BioSR, 29 specs ==="
task rl_photons_100 --photons 100 --n-iter 75
task rl_photons_15  --photons 15  --n-iter 30
task tv_photons_100 --photons 100 --reconstructor tv --tv-iter 100 --tv-prox-iter 10
task tv_photons_15  --photons 15  --reconstructor tv --tv-iter 100 --tv-prox-iter 10

# ------------------------------------------------------------------- phase 3
echo
echo "=== 2. Phase 3 background-aware unrolled control ==="
bash ieee/run_phase3.sh

echo
echo "=== all remaining compute attempted ==="
ls -la "$TASK_OUT" "$W2S_OUT" 2>&1
