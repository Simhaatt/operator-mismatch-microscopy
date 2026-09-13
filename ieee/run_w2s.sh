#!/usr/bin/env bash
# W2S simulated arm: replicate the bound and operator-shift results on a second
# instrument.
#
# The haze fraction is read from the measurement, not assumed. Carrying BioSR's
# 0.31 across to a different microscope would repeat the error that cost a full
# re-extraction to find the first time: at haze 0.048 the nominal residual
# correlates +0.647 with reconstruction error, and at 0.310 it correlates
# -0.251.
#
# Optics are W2S's own: NA 1.49 (supplementary material), 32 nm SIM grid
# (inferred from the Nikon N-SIM configuration -- 16 um iXon3 pitch behind 100x
# and a 2.5x relay). Emission wavelength is NOT documented by W2S; the three
# channels are excited at 488/561/640 nm and a single representative emission is
# used for all of them, since the simulated arm applies our own forward model to
# the SIM reconstruction as object and is therefore self-consistent regardless.
# The cutoff varies by about 13% across the three channels; that is recorded
# rather than hidden.
#
# Richardson-Lucy at two photon levels only. TV would add ~7 h for a second
# reconstructor on a replication that RL already establishes.
set -u

# Resolve the repository root from this script's own location, so the
# driver runs from anywhere rather than only on the machine it was
# written on.
cd "$(dirname "$0")/.." || exit 1
OUT=experiments/features29_w2s
LOG=ieee/logs
HAZE_JSON=experiments/realism/w2s_haze.json
mkdir -p "$OUT" "$LOG"

if [ ! -f "$HAZE_JSON" ]; then
  echo "FATAL: $HAZE_JSON missing -- run scripts/haze_fraction_w2s.py first"
  exit 1
fi

HAZE=$(python -c "import json;print(f\"{json.load(open('$HAZE_JSON'))['haze_median']:.3f}\")")
if [ -z "$HAZE" ]; then echo "FATAL: could not read haze median"; exit 1; fi

NA=1.49
LAM=0.590      # representative emission; channels span 0.515-0.670
PIXEL=0.032

echo "=== W2S simulated arm ==="
echo "    measured haze fraction : $HAZE   (BioSR was 0.31)"
echo "    NA $NA, lambda $LAM um, pixel $PIXEL um"
echo

run () {
  local name="$1"; shift
  if [ -f "$OUT/$name.npz" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $name"
  local t0=$SECONDS
  python scripts/extract_features.py "$@" --include-physical \
    --source data/w2s_gt.npz --haze-frac "$HAZE" --crop 256 \
    --na "$NA" --wavelength-um "$LAM" --pixel-um "$PIXEL" \
    --held-out ch2_640nm \
    --out "$OUT/$name.npz" > "$LOG/w2s_$name.log" 2>&1
  local rc=$? dt=$(( SECONDS - t0 ))
  echo "[$(date +%H:%M:%S)] $name rc=$rc in $((dt/60))m$((dt%60))s"
  [ $rc -ne 0 ] && tail -5 "$LOG/w2s_$name.log"
  return $rc
}

run rl_photons_100 --photons 100 --n-iter 75
run rl_photons_15  --photons 15  --n-iter 30

echo "=== W2S extraction done ==="
ls -la "$OUT"
