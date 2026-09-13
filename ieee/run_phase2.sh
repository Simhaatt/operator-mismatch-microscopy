#!/usr/bin/env bash
# Phase 2 re-extraction: relative-l2 target + the physical mismatch families.
#
# Strictly sequential. One extraction saturates the available cores and the
# machine has 8 GB of RAM, so running these in parallel would swap rather than
# go faster. Wall-clock therefore equals compute.
#
# Parameters match docs/reproduction.md exactly, with --include-physical added:
# 29 specs instead of 17. Outputs go to features29/ so the 17-spec caches that
# every current result depends on are left untouched.
set -u

# Resolve the repository root from this script's own location, so the
# driver runs from anywhere rather than only on the machine it was
# written on.
cd "$(dirname "$0")/.." || exit 1
OUT=experiments/features29
LOG=ieee/logs
mkdir -p "$OUT" "$LOG"

run () {
  local name="$1"; shift
  if [ -f "$OUT/$name.npz" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name (already present)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $name"
  local t0=$SECONDS
  python scripts/extract_features.py "$@" --include-physical \
    --source data/biosr_gt.npz --haze-frac 0.31 --crop 256 \
    --out "$OUT/$name.npz" > "$LOG/extract29_$name.log" 2>&1
  local rc=$?
  local dt=$(( SECONDS - t0 ))
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] DONE  $name in $((dt/60))m$((dt%60))s"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name rc=$rc after $((dt/60))m$((dt%60))s"
    tail -5 "$LOG/extract29_$name.log"
  fi
  return $rc
}

echo "=== Phase 2 extraction, 29 specs, haze 0.31, CPU ==="

run rl_photons_100 --photons 100 --n-iter 75
run rl_photons_15  --photons 15  --n-iter 30
run tv_photons_100 --photons 100 --reconstructor tv --tv-iter 100 --tv-prox-iter 10
run tv_photons_15  --photons 15  --reconstructor tv --tv-iter 100 --tv-prox-iter 10

echo "=== all Phase 2 simulated extractions attempted ==="
ls -la "$OUT"
