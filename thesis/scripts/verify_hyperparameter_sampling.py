"""
Verify: Do both scripts actually sample the SAME hyperparameters?
And if so, do they appear in the trial definitions?
"""
import json
import os
import re

def extract_hp_from_filename(trial_dir_name):
    """Extract LR, BS, Eps from trial directory name"""
    match = re.search(r'lr([0-9.e+-]+)_bs(\d+)_eps([0-9.e+-]+)', trial_dir_name)
    if match:
        return {
            'lr': float(match.group(1)),
            'bs': int(match.group(2)),
            'eps': float(match.group(3))
        }
    return None

print("="*80)
print("HYPOTHESIS: Both scripts sample identical hyperparameters in same order")
print("="*80)

# Get hyperparameters from JNK3 SAMPLE (where they differ)
full_dir = "thesis/results/jnk3/hyperparameter_search/sample"
reseed_dir = "thesis/results/reseed/jnk3/hyperparameter_search/sample"

if not os.path.exists(full_dir):
    print(f"Directory not found: {full_dir}")
else:
    print(f"\nFull.py trials (unordered):")
    trials_full = sorted(os.listdir(full_dir))
    hp_full = {}
    for i, trial in enumerate(trials_full, 1):
        hp = extract_hp_from_filename(trial)
        if hp:
            hp_full[i] = hp
            print(f"  Trial {i}: LR={hp['lr']:.4g}, BS={hp['bs']}, Eps={hp['eps']:.3f}")

if not os.path.exists(reseed_dir):
    print(f"\nDirectory not found: {reseed_dir}")
else:
    print(f"\nExpreplay.py trials (unordered):")
    trials_reseed = sorted(os.listdir(reseed_dir))
    hp_reseed = {}
    for i, trial in enumerate(trials_reseed, 1):
        hp = extract_hp_from_filename(trial)
        if hp:
            hp_reseed[i] = hp
            print(f"  Trial {i}: LR={hp['lr']:.4g}, BS={hp['bs']}, Eps={hp['eps']:.3f}")

# Compare
print("\n" + "="*80)
print("COMPARISON: Are they sampling the same hyperparameters?")
print("="*80)

if hp_full == hp_reseed:
    print("\n✓ YES - Both scripts sample IDENTICAL hyperparameters in same order")
    print("  This confirms the RNG seed=0 determinism")
else:
    print("\n✗ NO - Scripts sample DIFFERENT hyperparameters")
    print("  This would be unexpected!")
    
    # Find differences
    all_trials = max(len(hp_full), len(hp_reseed))
    for trial_num in range(1, all_trials + 1):
        hp_f = hp_full.get(trial_num)
        hp_r = hp_reseed.get(trial_num)
        if hp_f != hp_r:
            print(f"  Trial {trial_num} differs:")
            print(f"    Full.py:     {hp_f}")
            print(f"    Expreplay.py: {hp_r}")

# Now load the actual metrics to show how reseeding affects the ranking
print("\n" + "="*80)
print("Why different best hyperparameters are selected:")
print("="*80)

print("\nFull.py results:")
full_summary = "thesis/results/jnk3/hyperparameter_search/sample/trial_summaries.json"
with open(full_summary) as f:
    trials_full_data = json.load(f)
    sorted_by_circles = sorted(trials_full_data, key=lambda x: x['#CIRCLES'], reverse=True)
    print(f"  Top 5 trials by #CIRCLES:")
    for i, trial in enumerate(sorted_by_circles[:5], 1):
        print(f"    #{i}: #CIRCLES={trial['#CIRCLES']}, "
              f"LR={trial['learning_rate']:.4g}, BS={trial['batch_size']}, Eps={trial['epsilon']:.3f}")

print("\nExpreplay.py results:")
reseed_summary = "thesis/results/reseed/jnk3/hyperparameter_search/sample/trial_summaries.json"
with open(reseed_summary) as f:
    trials_reseed_data = json.load(f)
    sorted_by_circles = sorted(trials_reseed_data, key=lambda x: x['#CIRCLES'], reverse=True)
    print(f"  Top 5 trials by #CIRCLES:")
    for i, trial in enumerate(sorted_by_circles[:5], 1):
        print(f"    #{i}: #CIRCLES={trial['#CIRCLES']}, "
              f"LR={trial['learning_rate']:.4g}, BS={trial['batch_size']}, Eps={trial['epsilon']:.3f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If hyperparameters are identical but rankings differ:
→ Reseeding improves SOME trials more than others
→ Different trials become "best"
→ Different hyperparameters are revealed from the same sampled sequence

This is EXACTLY what we expect when experience replay works!
""")
