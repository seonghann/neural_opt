#!/bin/bash
# Parallel launcher for riemannian_data_sampling.py
#
# Usage:
#   bash run_parallel.sh --num_workers 8 --save_xyz /path/to/output --dataloader train [other args...]
#
# This script splits the dataloader batches across N workers.
# Each worker processes a range of batches independently.
# XYZ files are saved to the same directory (no conflicts since filenames include idx).
# CSV files get a batch-range suffix and can be merged afterward.

set -e

# Parse --num_workers from args (extract it, pass the rest to the python script)
NUM_WORKERS=8
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Determine total number of batches by doing a dry-run query
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "================================================"
echo "Parallel Riemannian Data Sampling"
echo "  Workers: $NUM_WORKERS"
echo "  Project: $PROJECT_DIR"
echo "  Args: ${REMAINING_ARGS[@]}"
echo "================================================"

# Count total batches
TOTAL_BATCHES=$(PYTHONPATH="$PROJECT_DIR" python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from omegaconf import OmegaConf
from src.dataset.data_module import load_datamodule
# Find config_yaml from args
args = '${REMAINING_ARGS[@]}'.split()
config_yaml = None
for i, a in enumerate(args):
    if a == '--config_yaml' and i+1 < len(args):
        config_yaml = args[i+1]
        break
if config_yaml is None:
    config_yaml = '$SCRIPT_DIR/riemannian_data_sampling.yaml'
config = OmegaConf.load(config_yaml)
dm = load_datamodule(config)
dl_name = 'train'
for i, a in enumerate(args):
    if a == '--dataloader' and i+1 < len(args):
        dl_name = args[i+1]
        break
if dl_name == 'train':
    dl = dm.train_dataloader()
elif dl_name == 'val':
    dl = dm.val_dataloader()
else:
    dl = dm.test_dataloader()
print(len(dl))
")

echo "Total batches: $TOTAL_BATCHES"

# Calculate batch ranges per worker
BATCHES_PER_WORKER=$(( (TOTAL_BATCHES + NUM_WORKERS - 1) / NUM_WORKERS ))

# Launch workers
PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
    BATCH_START=$((i * BATCHES_PER_WORKER))
    BATCH_END=$(( (i + 1) * BATCHES_PER_WORKER ))
    if [ $BATCH_END -gt $TOTAL_BATCHES ]; then
        BATCH_END=$TOTAL_BATCHES
    fi
    if [ $BATCH_START -ge $TOTAL_BATCHES ]; then
        break
    fi

    echo "[Worker $i] Batches [$BATCH_START, $BATCH_END)"
    PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_DIR" python3 \
        "$SCRIPT_DIR/riemannian_data_sampling.py" \
        "${REMAINING_ARGS[@]}" \
        --batch_start $BATCH_START \
        --batch_end $BATCH_END \
        > "/tmp/riemannian_worker_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} workers. PIDs: ${PIDS[@]}"
echo "Logs: /tmp/riemannian_worker_*.log"
echo ""

# Wait for all workers
FAILED=0
for ((i=0; i<${#PIDS[@]}; i++)); do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "[Worker $i] (PID $PID) DONE"
    else
        echo "[Worker $i] (PID $PID) FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All workers completed successfully."
else
    echo "WARNING: $FAILED worker(s) failed. Check logs."
fi

# Merge CSV files if --save_csv was specified
SAVE_CSV=""
for ((i=0; i<${#REMAINING_ARGS[@]}; i++)); do
    if [ "${REMAINING_ARGS[$i]}" == "--save_csv" ] && [ $((i+1)) -lt ${#REMAINING_ARGS[@]} ]; then
        SAVE_CSV="${REMAINING_ARGS[$((i+1))]}"
        break
    fi
done

if [ -n "$SAVE_CSV" ]; then
    echo "Merging CSV files..."
    PYTHONPATH="$PROJECT_DIR" python3 -c "
import pandas as pd
import glob, os
base, ext = os.path.splitext('$SAVE_CSV')
parts = sorted(glob.glob(f'{base}_b*{ext}'))
if parts:
    df = pd.concat([pd.read_csv(p, index_col=0) for p in parts], ignore_index=True)
    df.to_csv('$SAVE_CSV')
    print(f'Merged {len(parts)} CSV files -> $SAVE_CSV ({len(df)} rows)')
else:
    print('No CSV parts found to merge.')
"
fi
