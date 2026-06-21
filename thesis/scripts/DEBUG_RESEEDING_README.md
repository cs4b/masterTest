# Debug Reseeding - Single Trial Analysis

This is a debugging version based on `DE_dist_samp_gt_expreplay.py` that runs **ONLY ONE trial for DRD2 sample constraint** with extensive logging.

## What It Does

1. **Uses same seed (seed=0)** as expreplay.py, so it samples the same hyperparameters
2. **Runs only 1 trial** instead of 15, for faster debugging
3. **Prints detailed debug output** at each step:
   - When reseeding code is called
   - How many molecules are loaded from CSV
   - How many pass the threshold filter
   - How many are deduplicated
   - How many are already seeded
   - How many are actually novel
   - What gets written to `reseeding_seen.txt`
4. **Saves output in same format** as original for easy comparison

## How to Run

### Option 1: Simple Python execution
```bash
cd /system/user/studentwork/nemeth
conda activate divopt
python thesis/scripts/DE_dist_samp_debug_reseeding.py
```

### Option 2: Full comparison with bash script
```bash
bash thesis/scripts/run_debug_test.sh
```

This runs the debug version AND shows comparison with original results.

## What to Look For

Watch the output for messages like:
```
▶ RESEEDING CHECK (epoch 1)
  [_load_high_scoring_novel (epoch 1)] Loading from ...
  [_load_high_scoring_novel (epoch 1)] Loaded 2105 rows from CSV
  [_load_high_scoring_novel (epoch 1)] Reward stats:
    min=0.0000, mean=0.0750, max=1.0000
    Molecules >= 0.7: 31 / 2105
  [_load_high_scoring_novel (epoch 1)] After filtering: 31 molecules
  [_canonicalize_smiles] Input: 31 SMILES
  [_canonicalize_smiles] Output: 28 valid SMILES (3 failed)
  [_canonicalize_smiles] Deduplicated: 27 unique SMILES
  [_load_high_scoring_novel (epoch 1)] Novel molecules: 27
  ✓ Found 27 novel molecules! Adding to training...
  ✓ Writing 27 SMILES to ...reseeding_seen.txt
```

If you see:
```
✗ NO novel molecules found above threshold 0.7
```

Then we've found the problem!

## Output Location

Results saved to:
```
thesis/results/debug_reseed/drd2/sample/debug_trial_lr*.../
  - debug_metrics.json (metrics with reseeding info)
  - rs_training_molecules.csv (logged scores)
  - reseeding_seen.txt (seeded molecules - if created)
```

## Comparison

Compare the debug run with the original:
```bash
# Debug
cat thesis/results/debug_reseed/drd2/sample/*/debug_metrics.json | jq .

# Original
cat thesis/results/reseed/drd2/hyperparameter_search/sample/drd2_sample_trial1*/metrics.json | jq .
```

## Key Differences from Original

- `RESEED_EVERY_EPOCHS = 1` (reseed after every epoch)
- `RESEED_THRESHOLD = 0.7` (threshold for high-reward molecules)
- Only runs `epochs=10` (can complete 5 epochs in 10,000 molecule budget)
- Prints debug output at each reseeding check
- Prints epoch boundaries and eval_count tracking
