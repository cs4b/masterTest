"""
Check CSV content at epoch markers to see when high-reward molecules appear
"""
import pandas as pd
import os

targets = ["drd2", "gsk3", "jnk3"]

print("="*80)
print("CHECKING: When do high-reward molecules appear in CSV?")
print("="*80)

THRESHOLD = 0.7

for target in targets:
    print(f"\n{target.upper()}")
    print("-" * 80)
    
    csv_file = f"thesis/results/reseed/{target}/hyperparameter_search/sample/{os.listdir(f'thesis/results/reseed/{target}/hyperparameter_search/sample')[0]}/rs_training_molecules.csv"
    
    df = pd.read_csv(csv_file, dtype={'reward': float})
    
    # Check reward distribution at different epoch boundaries
    checkpoints = [2000, 4000, 6000, 8000, 10000]
    
    for cp in checkpoints:
        if cp <= len(df):
            subset = df.iloc[:cp]
            above_threshold = len(subset[subset['reward'] >= THRESHOLD])
            print(f"  Epoch at {cp} molecules:")
            print(f"    Molecules >= {THRESHOLD}: {above_threshold}")
            print(f"    % above threshold: {100*above_threshold/len(subset):.2f}%")
            if above_threshold > 0:
                print(f"    Max reward: {subset['reward'].max():.4f}")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
If JNK3 has molecules >= 0.7 early BUT DRD2/GSK3 don't:
  → Reseeding threshold is too high for DRD2/GSK3
  → This explains why reseeding_seen.txt is empty for them

If all three have similar patterns:
  → Something else is preventing DRD2/GSK3 reseeding
  → Possible: already_seeded_set filtering issue
  → Possible: canonicalization removing molecules
  → Possible: a bug in the code logic""")
