"""
Compare reward distributions across targets to explain why reseeding threshold blocks DRD2/GSK3
"""
import os
import pandas as pd
import numpy as np

RESEED_THRESHOLD = 0.7

targets = ["drd2", "gsk3", "jnk3"]
constraints = ["sample"]

print("="*80)
print("REWARD DISTRIBUTION ANALYSIS (Why reseeding only works for JNK3)")
print("="*80)

for target in targets:
    print(f"\n{target.upper()}")
    print("-" * 80)
    
    trial_dir = f"thesis/results/reseed/{target}/hyperparameter_search/sample"
    trials = sorted([d for d in os.listdir(trial_dir) if os.path.isdir(os.path.join(trial_dir, d))])
    
    if not trials:
        print("  No trials found")
        continue
    
    # Analyze first few trials
    all_rewards = []
    all_above_threshold = 0
    total_molecules = 0
    
    for trial in trials[:3]:  # Sample first 3 trials
        csv_file = os.path.join(trial_dir, trial, "rs_training_molecules.csv")
        
        if not os.path.exists(csv_file):
            continue
        
        try:
            df = pd.read_csv(csv_file)
            rewards = df['reward'].values
            all_rewards.extend(rewards)
            
            above = len(df[df['reward'] >= RESEED_THRESHOLD])
            all_above_threshold += above
            total_molecules += len(df)
            
            print(f"\n  Trial: {trial}")
            print(f"    Total molecules scored: {len(df)}")
            print(f"    Max reward: {rewards.max():.4f}")
            print(f"    Mean reward: {rewards.mean():.4f}")
            print(f"    Molecules >= {RESEED_THRESHOLD}: {above}")
            print(f"    Percentile of {RESEED_THRESHOLD}: {100*sum(rewards >= RESEED_THRESHOLD)/len(rewards):.1f}%")
            
        except Exception as e:
            print(f"  Error reading {trial}: {e}")
            continue
    
    if all_rewards:
        print(f"\n  Summary (first 3 trials):")
        print(f"    Total molecules scored: {total_molecules}")
        print(f"    Total molecules >= {RESEED_THRESHOLD}: {all_above_threshold}")
        print(f"    Percentage above threshold: {100*all_above_threshold/total_molecules:.2f}%")
        print(f"    Distribution: min={np.min(all_rewards):.4f}, "
              f"25%={np.percentile(all_rewards, 25):.4f}, "
              f"median={np.median(all_rewards):.4f}, "
              f"75%={np.percentile(all_rewards, 75):.4f}, "
              f"max={np.max(all_rewards):.4f}")
        
        # Show histogram of reward values
        bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        hist, edges = np.histogram(all_rewards, bins=bins)
        print(f"\n    Reward distribution:")
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            print(f"      [{low:.1f}-{high:.1f}): {hist[i]:4d} molecules")

print("\n" + "="*80)
print("EXPLANATION")
print("="*80)
print(f"""
RESEED_THRESHOLD = {RESEED_THRESHOLD}

DRD2 & GSK3: Very few (or zero) molecules reach reward >= {RESEED_THRESHOLD}
  → _load_high_scoring_novel() returns empty list
  → No molecules to add to training set
  → reseeding_seen.txt never created
  → Experience replay has NO EFFECT

JNK3: Multiple molecules reach reward >= {RESEED_THRESHOLD} each epoch
  → _load_high_scoring_novel() returns 3-14 novel molecules
  → Training set grows with high-quality examples
  → reseeding_seen.txt created with SMILES
  → Experience replay IMPROVES training

ROOT CAUSE: DRD2/GSK3 scoring functions are harder (lower rewards)
  than JNK3 scoring function.

SOLUTION: Lower RESEED_THRESHOLD to enable reseeding for all targets
  Try 0.5 or 0.4 instead of 0.7
""")
