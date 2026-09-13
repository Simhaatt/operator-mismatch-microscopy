#!/usr/bin/env bash
# W2S real-widefield calibration, queued behind the task extraction.
#
# Replaces Phase 3 in the night's plan. Where Phase 3 was a control on one
# reconstructor's feature interpretation, this replicates the paper's real-data
# result on a second instrument -- and on a genuine widefield acquisition rather
# than one synthesised by summing SIM frames, which is the weakest
# methodological point in the current draft.
set -u
# Resolve the repository root from this script's own location, so the
# driver runs from anywhere rather than only on the machine it was
# written on.
cd "$(dirname "$0")/.." || exit 1

echo "=== waiting for the task extraction to finish ==="
for i in $(seq 1 900); do
  if [ -f experiments/features29_task/tv_photons_15.npz ]; then
    echo "[$(date +%H:%M:%S)] task extraction complete"; break
  fi
  sleep 20
done

if [ -f experiments/features19/real_w2s_rl.npz ]; then
  echo "[$(date +%H:%M:%S)] SKIP W2S real (already present)"; exit 0
fi

echo "[$(date +%H:%M:%S)] START W2S real calibration"
t0=$SECONDS
python scripts/real_calibration_w2s.py --n-iter 75 \
  --out experiments/features19/real_w2s_rl.npz > ieee/logs/w2s_real.log 2>&1
rc=$?; dt=$(( SECONDS - t0 ))
echo "[$(date +%H:%M:%S)] W2S real rc=$rc in $((dt/60))m$((dt%60))s"
[ $rc -ne 0 ] && tail -6 ieee/logs/w2s_real.log
