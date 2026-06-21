"""
Check: How many molecules per epoch?
If JNK3 is slower, it could have multiple epochs before budget is hit
"""
import os
import pandas as pd

print("="*80)
print("HYPOTHESIS: JNK3 processes fewer molecules/epoch → allows reseeding")
print("="*80)

targets = ["drd2", "gsk3", "jnk3"]

for target in targets:
    print(f"\n{target.upper()}")
    print("-" * 80)
    
    # Get CSV from first expreplay trial
    csv_path = f"thesis/results/reseed/{target}/hyperparameter_search/sample"
    trials = sorted([d for d in os.listdir(csv_path) if os.path.isdir(os.path.join(csv_path, d))])
    
    if trials:
        first_trial = trials[0]
        csv_file = os.path.join(csv_path, first_trial, "rs_training_molecules.csv")
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"Trial: {first_trial}")
                print(f"  Total molecules scored: {len(df)}")
                print(f"  Elapsed time: {df['elapsed_time'].iloc[-1]:.1f} seconds")
                print(f"  Molecules/second: {len(df) / df['elapsed_time'].iloc[-1]:.1f}")
                
                # Estimate molecules per epoch (assuming 10 epochs total and 10k budget)
                # If all 10,000 hit budget, how many epochs completed?
                # If molecules are scored uniformly over time...
                time_total = df['elapsed_time'].iloc[-1]
                molecules_total = len(df)
                   
                # Rough estimate: if training uses ~10 seconds per epoch base
                # 680 seconds / ~10 sec per epoch = ~68 epochs possible
                # But budget caps at 10,000 molecules
                # If n_samples=2000/epoch, then 10k/2000 = 5 epochs
                
                print(f"\n  Analysis:")
                print(f"    If n_samples=2000/epoch (from code):")
                print(f"    Molecules before budget hit: {molecules_total}")
                print(f"    Epochs possible: {molecules_total / 2000:.1f}")
                
                # More detailed: check eval_count from metrics
                metrics_file = os.path.join(csv_path, first_trial, "metrics.json")
                import json
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    print(f"\n  From metrics.json:")
                    print(f"    Eval count: {metrics['eval_count']}")
                    print(f"    Budget hit at: {metrics['budget_error'].split('(')[1].split(')')[0]}")
                    
            except Exception as e:
                print(f"  Error: {e}")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
If all targets complete only PARTIAL training before
sample budget of 10,000 is hit:

  - During training, molecules are scored
  - LoggingScorer tracks eval_count
  - When eval_count >= 10,000, raises RuntimeError
  - Explorer.fit() throws exception

In current expreplay.py code:
  try:
    explorer.fit(...)
  except (TimeoutError, RuntimeError):
    break  # ← BREAKS BEFORE reseeding code!

So reseeding_seen.txt should NOT exist for ANY target.

BUT JNK3 HAS IT! This suggests either:
  1) JNK3 was run with DIFFERENT CODE (older version)
  2) The exception ISN'T being raised the same way
  3) Code has been modified since runs completed
""")
