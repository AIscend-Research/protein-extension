#!/bin/bash
# Runs the 20-seed power sweep to completion, folds it into the docs, and
# pushes — unattended. Each of the three sweep stages is checkpointed (see
# sweeps.py/ablation.py), so a crash or a stall costs minutes, not the whole
# run: this script retries a failed/stalled stage in place, and because the
# underlying scripts skip already-completed (seed, width) pairs on restart,
# a retry is cheap rather than starting over.
#
# What "on its own" means concretely here:
#   - a stage that crashes is retried, up to MAX_RETRIES times
#   - a stage that stops writing its checkpoint for STALL_SECS is presumed
#     hung, killed, and retried (it resumes from its last checkpoint)
#   - after all three stages finish, apply_power_results.py folds the numbers
#     into RESULTS.md/README.md mechanically (arithmetic, not prose I'd need
#     to be present to write)
#   - a final `summarize.py` run is required to succeed, error-free, before
#     anything is committed — a broken doc or a broken script does not get
#     silently pushed
#   - if apply_power_results.py raises a review flag (a stat that would
#     reverse an existing headline claim), the run still completes and still
#     commits — the flag is visible in the doc itself, not swallowed
#
# What it will NOT do: paper over a genuine failure. If a stage exhausts its
# retries, or the final consistency check fails, the script stops and leaves
# a clear FAILED marker instead of pushing something broken.

set -u
cd "$(dirname "$0")/.."
LOG=/tmp/autopilot.log
STATUS=/tmp/autopilot_status.txt
STALL_SECS=600      # no checkpoint progress for 10 min = presumed hung
MAX_RETRIES=5
RETRY_BACKOFF=20

echo "started" > "$STATUS"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# run_stage <description> <checkpoint file> <command...>
# Retries on nonzero exit; kills and retries on a stale checkpoint.
run_stage() {
  local desc="$1" ckpt="$2"; shift 2
  local attempt=1
  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    log "$desc: attempt $attempt/$MAX_RETRIES"
    "$@" >> "$LOG" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      if [ -f "$ckpt" ]; then
        local now age
        now=$(date +%s)
        age=$(( now - $(stat -f %m "$ckpt" 2>/dev/null || echo "$now") ))
        if [ "$age" -gt "$STALL_SECS" ]; then
          log "$desc: no checkpoint progress in ${age}s, killing pid $pid"
          kill "$pid" 2>/dev/null
          sleep 3
          kill -9 "$pid" 2>/dev/null
          break
        fi
      fi
    done
    wait "$pid" 2>/dev/null
    local status=$?
    if [ "$status" -eq 0 ]; then
      log "$desc: succeeded"
      return 0
    fi
    log "$desc: exited $status, retrying in ${RETRY_BACKOFF}s"
    attempt=$((attempt + 1))
    sleep "$RETRY_BACKOFF"
  done
  log "$desc: GAVE UP after $MAX_RETRIES attempts"
  return 1
}

export OMP_NUM_THREADS=4

run_stage "selection segment sweep" \
  "experiments/results/sweep_segment_selection_power20.json" \
  .venv/bin/python experiments/sweeps.py --sweep segment --model selection \
    --seeds 20 --widths 50 65 80 --start 13 --n-perm 200 --n-orders 16 --tag power20
if [ $? -ne 0 ]; then
  echo "FAILED: selection segment sweep exhausted retries" > "$STATUS"
  log "### AUTOPILOT FAILED ###"
  exit 1
fi

run_stage "f81 segment sweep" \
  "experiments/results/sweep_segment_f81_power20.json" \
  .venv/bin/python experiments/sweeps.py --sweep segment --model f81 \
    --seeds 20 --widths 50 65 80 --start 13 --n-perm 200 --n-orders 16 --tag power20
if [ $? -ne 0 ]; then
  echo "FAILED: f81 segment sweep exhausted retries" > "$STATUS"
  log "### AUTOPILOT FAILED ###"
  exit 1
fi

run_stage "ablation" \
  "experiments/results/ablation_selection_power20.json" \
  .venv/bin/python experiments/ablation.py --model selection \
    --seeds 20 --widths 50 80 --n-perm 200 --n-orders 16 --tag power20
if [ $? -ne 0 ]; then
  echo "FAILED: ablation exhausted retries" > "$STATUS"
  log "### AUTOPILOT FAILED ###"
  exit 1
fi

log "all three stages complete; folding results into the docs"
.venv/bin/python experiments/apply_power_results.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "FAILED: apply_power_results.py — docs not touched, nothing committed" > "$STATUS"
  log "### AUTOPILOT FAILED (docs unmodified) ###"
  exit 1
fi

log "running the final consistency check before committing anything"
.venv/bin/python experiments/summarize.py > /tmp/autopilot_summary_check.txt 2>&1
if grep -qiE "error|traceback" /tmp/autopilot_summary_check.txt; then
  echo "FAILED: summarize.py errored after the doc update — not committing" > "$STATUS"
  log "### AUTOPILOT FAILED (post-update summarize.py errored) ###"
  exit 1
fi

log "consistency check passed; committing and pushing"
git add -A
git commit -q -m "Firm up the statistics: 20-seed rerun, blocks above the diagnostic-site floor

Runs experiments/run_power_sweeps.sh to completion (segment sweep at 20 seeds
x widths 50/65/80 for both selection and f81, plus the ablation at 20 seeds x
widths 50/80) and folds the result into RESULTS.md/README.md via
experiments/apply_power_results.py, which computes Wilson 95% CIs on every
detection rate mechanically and inserts them as a new section rather than
editing the original anecdote-scale sections in place.

Produced end-to-end by experiments/autopilot.sh, unattended."
git push origin contamination-detector-audit >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "PARTIAL: committed locally but push failed — see $LOG" > "$STATUS"
  log "### AUTOPILOT: commit ok, push FAILED ###"
  exit 1
fi

echo "done" > "$STATUS"
log "### AUTOPILOT DONE ###"
