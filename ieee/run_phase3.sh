#!/usr/bin/env bash
# Phase 3 is CUT. It was a control on the unrolled network's background-blind
# fidelity term -- worth doing eventually, but it changes no headline claim and
# the manuscript already states the confound as a limitation. The ~9 h it needs
# is better spent replicating the real-measurement result on a second
# instrument (ieee/run_w2s_real.sh).
#
# The original driver is preserved as run_phase3.sh.disabled; both partial
# checkpoints are renamed .partial.pt so a future resume retrains both arms at a
# matched budget rather than silently reusing an undertrained one.
echo "[$(date +%H:%M:%S)] Phase 3 intentionally skipped (see run_phase3.sh.disabled)"
exit 0
