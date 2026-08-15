#!/bin/bash
# Phase 1b: firm up the statistics. 20 seeds instead of 3, blocks sized above the
# diagnostic-site floor (50/65/80 residues instead of 10/20/30/50, since section 4
# showed 10/20/30 were unwinnable by construction). --tag power20 keeps these
# results in separate files from the original 3-seed run, so nothing already
# written up in RESULTS.md is overwritten.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4

echo "=== selection segment sweep, 20 seeds, widths 50/65/80 ==="
.venv/bin/python experiments/sweeps.py --sweep segment --model selection \
  --seeds 20 --widths 50 65 80 --start 13 --n-perm 200 --n-orders 16 --tag power20

echo "=== f81 segment sweep, 20 seeds, same widths (the control at equal power) ==="
.venv/bin/python experiments/sweeps.py --sweep segment --model f81 \
  --seeds 20 --widths 50 65 80 --start 13 --n-perm 200 --n-orders 16 --tag power20

echo "=== ablation, 20 seeds, widths 50/80 ==="
.venv/bin/python experiments/ablation.py --model selection \
  --seeds 20 --widths 50 80 --n-perm 200 --n-orders 16 --tag power20

echo "### POWER SWEEPS DONE ###"
