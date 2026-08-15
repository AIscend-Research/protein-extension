#!/bin/bash
# Drive run_experiments.py one seed per invocation, so a crash costs one seed
# rather than the whole sweep. Usage: run_all_seeds.sh <model> <nseeds>
set -u
cd "$(dirname "$0")/.."
MODEL="$1"
NSEEDS="${2:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

for s in $(seq 0 $((NSEEDS - 1))); do
  echo "=== ${MODEL} seed ${s} ==="
  .venv/bin/python experiments/run_experiments.py \
    --exp all --models "$MODEL" --seeds 1 --seed-start "$s" \
    --device cpu --name "${MODEL}_s${s}"
  echo "=== ${MODEL} seed ${s} exit=$? ==="
done
