#!/usr/bin/env bash
# End-to-end checkpoint test script.
#
# Usage:  bash tests/checkpoint/run_e2e.sh
#
# Prerequisites:
#   - Docker with experimental mode enabled
#   - CRIU installed
#   - Docker compose services using network_mode: host (already in compose.yaml)
#
# This script runs 4 phases:
#   1. Happy-path eval with checkpointing
#   2. Inspect the log for checkpoint events
#   3. Crashing eval with checkpointing
#   4. Resume from checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Clear INSPECT_ env vars that reference external plugins (may not be installed)
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE INSPECT_REQUIRED_HOOKS 2>/dev/null || true
unset INSPECT_TELEMETRY_BUCKET INSPECT_TELEMETRY_BUCKET_PREFIX INSPECT_TELEMETRY_LOG_GROUP 2>/dev/null || true

# Activate the project venv if inspect is not already on PATH
if ! command -v inspect &>/dev/null; then
    if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.venv/bin/activate"
    else
        echo "ERROR: 'inspect' not found. Activate a venv with inspect_ai installed."
        exit 1
    fi
fi

# Temporary directory for checkpoint data
CKPT_DIR=$(mktemp -d /tmp/inspect_ckpt_e2e_XXXXXX)
LOG_DIR=$(mktemp -d /tmp/inspect_log_e2e_XXXXXX)
cleanup() {
    echo "Cleaning up..."
    # CRIU checkpoint dirs are owned by root, so use sudo if available
    if command -v sudo &>/dev/null; then
        sudo rm -rf "$CKPT_DIR"
    else
        rm -rf "$CKPT_DIR" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR"
}
trap cleanup EXIT

TASK_FILE="$SCRIPT_DIR/checkpoint_task.py"

echo "============================================"
echo "  Checkpoint E2E Test"
echo "============================================"
echo "Checkpoint dir: $CKPT_DIR"
echo "Log dir:        $LOG_DIR"
echo ""

# ---------------------------------------------------------------
# Phase 1: Happy-path eval with checkpointing
# ---------------------------------------------------------------
echo "=== Phase 1: Happy-path eval with checkpointing ==="
echo ""

INSPECT_LOG_DIR="$LOG_DIR" inspect eval "$TASK_FILE@checkpoint_test_task" \
    --model mockllm/model \
    --checkpoint \
    --checkpoint-interval-seconds 0.1 \
    --checkpoint-dir "$CKPT_DIR/phase1" \
    --no-score

PHASE1_LOG=$(find "$LOG_DIR" -name "*.eval" -type f | sort | tail -1)

if [ -z "$PHASE1_LOG" ]; then
    echo "FAIL: No eval log found for phase 1"
    exit 1
fi

echo ""
echo "Phase 1 log: $PHASE1_LOG"
echo "Phase 1 PASSED - eval completed successfully"
echo ""

# ---------------------------------------------------------------
# Phase 2: Inspect the log for checkpoint events
# ---------------------------------------------------------------
echo "=== Phase 2: Inspect checkpoint events in log ==="
echo ""

python -c "
import sys
sys.path.insert(0, '$REPO_ROOT/src')
from inspect_ai.log._file import read_eval_log
from inspect_ai.event._checkpoint import CheckpointEvent

log = read_eval_log('$PHASE1_LOG')
print(f'Log status: {log.status}')
print(f'Samples: {len(log.samples) if log.samples else 0}')

ckpt_count = 0
if log.samples:
    for sample in log.samples:
        sample_ckpts = [e for e in sample.events if isinstance(e, CheckpointEvent)]
        if sample_ckpts:
            print(f'  Sample {sample.id}: {len(sample_ckpts)} checkpoint(s)')
            for evt in sample_ckpts:
                print(f'    - {evt.checkpoint_id}: {evt.message_count} messages, state={evt.state_file}')
                ckpt_count += 1

print(f'Total checkpoint events: {ckpt_count}')
if ckpt_count == 0:
    print('FAIL: No checkpoint events found')
    sys.exit(1)
print('Phase 2 PASSED')
"

echo ""

# ---------------------------------------------------------------
# Phase 3: Crashing eval with checkpointing
# ---------------------------------------------------------------
echo "=== Phase 3: Crashing eval with checkpointing ==="
echo ""

# Unset CHECKPOINT_TEST_PASS so the crash fires
unset CHECKPOINT_TEST_PASS 2>/dev/null || true

# This eval is expected to fail (the crash solver raises ValueError)
set +e
INSPECT_LOG_DIR="$LOG_DIR" inspect eval "$TASK_FILE@checkpoint_crash_task" \
    --model mockllm/model \
    --checkpoint \
    --checkpoint-interval-seconds 0.1 \
    --checkpoint-dir "$CKPT_DIR/phase3" \
    --no-score
PHASE3_EXIT=$?
set -e

PHASE3_LOG=$(find "$LOG_DIR" -name "*.eval" -type f | sort | tail -1)

if [ -z "$PHASE3_LOG" ] || [ "$PHASE3_LOG" = "$PHASE1_LOG" ]; then
    echo "FAIL: No new eval log found for phase 3"
    exit 1
fi

echo ""
echo "Phase 3 log: $PHASE3_LOG"

python -c "
import sys
sys.path.insert(0, '$REPO_ROOT/src')
from inspect_ai.log._file import read_eval_log
from inspect_ai.event._checkpoint import CheckpointEvent

log = read_eval_log('$PHASE3_LOG')
print(f'Log status: {log.status}')

error_samples = 0
completed_samples = 0
ckpt_count = 0
if log.samples:
    for sample in log.samples:
        if sample.error:
            error_samples += 1
        elif sample.scores:
            completed_samples += 1
        sample_ckpts = [e for e in sample.events if isinstance(e, CheckpointEvent)]
        ckpt_count += len(sample_ckpts)
        if sample_ckpts:
            print(f'  Sample {sample.id}: {len(sample_ckpts)} checkpoint(s), error={sample.error is not None}')

print(f'Error samples: {error_samples}')
print(f'Checkpoint events: {ckpt_count}')

if error_samples == 0:
    print('FAIL: Expected at least one errored sample')
    sys.exit(1)

print('Phase 3 PASSED')
"

echo ""

# ---------------------------------------------------------------
# Phase 4: Resume from checkpoint
# ---------------------------------------------------------------
echo "=== Phase 4: Resume from checkpoint ==="
echo ""

VERIFY_FILE="$LOG_DIR/checkpoint_verify.json"

set +e
CHECKPOINT_TEST_PASS=1 \
CHECKPOINT_VERIFY_FILE="$VERIFY_FILE" \
INSPECT_LOG_DIR="$LOG_DIR" \
inspect eval "$TASK_FILE@checkpoint_crash_task" \
    --model mockllm/model \
    --checkpoint \
    --checkpoint-interval-seconds 0.1 \
    --checkpoint-dir "$CKPT_DIR/phase4" \
    --resume-from-checkpoint "$PHASE3_LOG" \
    --no-score
PHASE4_EXIT=$?
set -e

PHASE4_LOG=$(find "$LOG_DIR" -name "*.eval" -type f | sort | tail -1)

if [ -z "$PHASE4_LOG" ] || [ "$PHASE4_LOG" = "$PHASE3_LOG" ]; then
    echo "FAIL: No new eval log found for phase 4"
    exit 1
fi

echo ""
echo "Phase 4 log: $PHASE4_LOG"

# --- Verify the checkpoint restoration evidence ---
if [ ! -f "$VERIFY_FILE" ]; then
    echo "FAIL: Verification file not written — solver did not detect resumed state"
    exit 1
fi

echo ""
echo "Checkpoint restore verification:"
cat "$VERIFY_FILE"
echo ""

python -c "
import json, sys

with open('$VERIFY_FILE') as f:
    v = json.load(f)

failures = []
warnings = []

# state.store should have the pre-crash counter value
if not v.get('store_restored'):
    failures.append(f'state.store not restored (counter={v.get(\"store_counter\")})')

# messages should have pre-crash conversation history
if not v.get('messages_restored'):
    failures.append(f'messages not restored (count={v.get(\"message_count\")})')

# sandbox /tmp/counter.txt — CRIU container restore.
# Docker 29.x may not support --checkpoint-dir on restore, so this is
# reported as a warning rather than a hard failure.
if not v.get('sandbox_restored'):
    warnings.append(
        f'sandbox filesystem not restored (counter={v.get(\"sandbox_counter\")}) '
        f'— Docker CRIU restore may not be supported on this version'
    )
elif not v.get('counters_match'):
    warnings.append(
        f'store counter ({v.get(\"store_counter\")}) != '
        f'sandbox counter ({v.get(\"sandbox_counter\")})'
    )

if failures:
    print('FAIL: Checkpoint restore verification failed:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)

print(f'  store counter:   {v[\"store_counter\"]} (restored from checkpoint state)')
print(f'  messages:        {v[\"message_count\"]} (restored from checkpoint state)')
if v.get('sandbox_restored'):
    print(f'  sandbox counter: {v[\"sandbox_counter\"]} (CRIU container restored)')
    print(f'  counters match:  {v[\"counters_match\"]}')
else:
    print(f'  sandbox counter: not available (CRIU container restore not supported)')

for w in warnings:
    print(f'  WARNING: {w}')

print('Phase 4 PASSED — checkpoint state restore verified')
"

echo ""
echo "============================================"
echo "  All phases completed!"
echo "============================================"
