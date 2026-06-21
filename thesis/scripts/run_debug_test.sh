#!/bin/bash
# Script to run debug version and compare with original results

echo "================================================================================"
echo "RESEEDING DEBUG COMPARISON"
echo "================================================================================"

cd /system/user/studentwork/nemeth

echo ""
echo "[1/2] Running debug version with verbose reseeding output..."
echo "      This will show EXACTLY why reseeding doesn't happen for DRD2"
echo ""
conda run -n divopt python thesis/scripts/DE_dist_samp_debug_reseeding.py 2>&1 | tee thesis/results/debug_reseed/debug_output.log

echo ""
echo "================================================================================"
echo "[2/2] Comparing results"
echo "================================================================================"

DEBUG_DIR="thesis/results/debug_reseed/drd2/sample"
ORIGINAL_DIR="thesis/results/reseed/drd2/hyperparameter_search/sample"

echo ""
echo "Debug trial results:"
if [ -d "$DEBUG_DIR" ]; then
    TRIAL=$(ls -d $DEBUG_DIR/*/ | head -1)
    echo "  Directory: $TRIAL"
    if [ -f "$TRIAL/debug_metrics.json" ]; then
        echo "  Metrics:"
        cat "$TRIAL/debug_metrics.json" | jq .
    fi
    if [ -f "$TRIAL/reseeding_seen.txt" ]; then
        echo "  Reseeding file:"
        wc -l "$TRIAL/reseeding_seen.txt"
    else
        echo "  Reseeding file: NOT CREATED (this is the problem!)"
    fi
else
    echo "  Directory not found!"
fi

echo ""
echo "Original trial (for comparison):"
ORIG_TRIAL=$(ls -d $ORIGINAL_DIR/drd2_sample_trial1*/ | head -1)
if [ -d "$ORIG_TRIAL" ]; then
    echo "  Directory: $ORIG_TRIAL"
    if [ -f "$ORIG_TRIAL/metrics.json" ]; then
        echo "  Metrics (excerpt):"
        cat "$ORIG_TRIAL/metrics.json" | jq '{eval_count, elapsed_time, budget_hit, budget_error}'
    fi
    if [ -f "$ORIG_TRIAL/reseeding_seen.txt" ]; then
        echo "  Reseeding file:"
        wc -l "$ORIG_TRIAL/reseeding_seen.txt"
    else
        echo "  Reseeding file: NOT CREATED (expected for original)"
    fi
fi

echo ""
echo "================================================================================"
echo "CSV ANALYSIS"
echo "================================================================================"

echo ""
echo "Debug version CSV (high-reward molecules by epoch):"
if [ -f "$DEBUG_DIR"*/rs_training_molecules.csv ]; then
    CSV=$(ls $DEBUG_DIR*/rs_training_molecules.csv)
    echo "  Epoch 1 (≤2000): $(head -2000 $CSV | tail -1 | awk -F, '{print "reward=" $2 ", eval_count=" $4}')"
    echo "  Epoch 2 (≤4000): $(head -4000 $CSV | tail -1 | awk -F, '{print "reward=" $2 ", eval_count=" $4}')"
    echo "  Epoch 4 (≤8000): $(head -8000 $CSV | tail -1 | awk -F, '{print "reward=" $2 ", eval_count=" $4}')"
fi

echo ""
echo "================================================================================"
echo "DEBUG LOG"
echo "================================================================================"
echo "See full debug output in: thesis/results/debug_reseed/debug_output.log"
echo ""
