"""
Diagnostic script to compare full.py vs expreplay.py results.
Helps understand why they might select identical hyperparameters.
"""

import json
import os
from pathlib import Path

def compare_results(full_results_dir, expreplay_results_dir):
    """
    Compare trial summaries and best configurations from both approaches.
    """
    
    print("="*70)
    print("COMPARING TRAINING APPROACHES: full.py vs expreplay.py")
    print("="*70)
    
    targets = ["drd2", "gsk3", "jnk3"]
    constraints = ["sample", "time"]
    
    for target in targets:
        print(f"\n{target.upper()}")
        print("-" * 70)
        
        for constraint in constraints:
            full_trials_file = os.path.join(
                full_results_dir, target, "hyperparameter_search", 
                constraint, "trial_summaries.json"
            )
            expreplay_trials_file = os.path.join(
                expreplay_results_dir, target, "hyperparameter_search",
                constraint, "trial_summaries.json"
            )
            
            if not os.path.exists(full_trials_file):
                print(f"  {constraint}: full.py trials not found")
                continue
                
            with open(full_trials_file) as f:
                full_trials = json.load(f)
            
            expreplay_trials = []
            if os.path.exists(expreplay_trials_file):
                with open(expreplay_trials_file) as f:
                    expreplay_trials = json.load(f)
            
            # Find best from each approach
            full_best = max(full_trials, key=lambda x: x["#CIRCLES"])
            expreplay_best = max(expreplay_trials, key=lambda x: x["#CIRCLES"]) if expreplay_trials else None
            
            print(f"\n  {constraint.upper()} constraint:")
            print(f"    Full.py best:       LR={full_best['learning_rate']:.4g}, "
                  f"BS={full_best['batch_size']}, Eps={full_best['epsilon']:.3f}")
            print(f"                        #CIRCLES={full_best['#CIRCLES']}, "
                  f"best_value={full_best['best_value']:.3f}")
            
            if expreplay_best:
                print(f"    Expreplay.py best:  LR={expreplay_best['learning_rate']:.4g}, "
                      f"BS={expreplay_best['batch_size']}, Eps={expreplay_best['epsilon']:.3f}")
                print(f"                        #CIRCLES={expreplay_best['#CIRCLES']}, "
                      f"best_value={expreplay_best['best_value']:.3f}")
                
                # Check if hyperparameters are identical
                params_equal = (
                    abs(full_best['learning_rate'] - expreplay_best['learning_rate']) < 1e-6 and
                    full_best['batch_size'] == expreplay_best['batch_size'] and
                    abs(full_best['epsilon'] - expreplay_best['epsilon']) < 1e-6
                )
                
                if params_equal:
                    print(f"    ✓ SAME HYPERPARAMETERS SELECTED")
                    circles_diff = expreplay_best['#CIRCLES'] - full_best['#CIRCLES']
                    if circles_diff > 0:
                        print(f"      → Expreplay improves #CIRCLES by {circles_diff} "
                              f"({100*circles_diff/full_best['#CIRCLES']:.1f}%)")
                    elif circles_diff < 0:
                        print(f"      → Full.py has better #CIRCLES by {-circles_diff} "
                              f"({-100*circles_diff/expreplay_best['#CIRCLES']:.1f}%)")
                    else:
                        print(f"      → IDENTICAL #CIRCLES values (likely same training outcome)")
                else:
                    print(f"    ✗ DIFFERENT HYPERPARAMETERS SELECTED")
                    
                # Show ranking distribution
                print(f"\n    Trial ranking distribution (by #CIRCLES):")
                full_circles = sorted([t["#CIRCLES"] for t in full_trials], reverse=True)
                expreplay_circles = sorted([t["#CIRCLES"] for t in expreplay_trials], reverse=True)
                
                print(f"      Full.py:     Top 5 #CIRCLES: {full_circles[:5]}")
                print(f"      Expreplay.py: Top 5 #CIRCLES: {expreplay_circles[:5]}")
            else:
                print(f"    Expreplay.py trials not found")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("""
1. SAME HYPERPARAMETERS + SAME #CIRCLES
   → Experience replay provides NO benefit, or benefits are negligible
   → Question: Is reseeding actually happening? Check trial logs for:
      "Reseeded X molecules" messages

2. SAME HYPERPARAMETERS + DIFFERENT #CIRCLES
   → Hyperparameter sensitivity is ORTHOGONAL to training method
   → Good hyperparameters remain good regardless of training approach
   → Check if expreplay consistently outperforms full for this hyperparameter

3. DIFFERENT HYPERPARAMETERS + EXPREPLAY BETTER
   → Experience replay changes the hyperparameter landscape
   → Different LRs/etc are optimal for different training methods
   → Expected behavior if reseeding is working

4. Check trial logs for reseeding events:
   grep "Reseeded" <trial_run_dir>/training_molecules.csv
   grep "No novel molecules" <trial_run_dir>/training_molecules.csv
    """)

if __name__ == "__main__":
    import sys
    
    # modify these paths to point to your results directories
    full_dir = "/system/user/studentwork/nemeth/thesis/results"  # from full.py
    expreplay_dir = "/system/user/studentwork/nemeth/thesis/results/reseed"  # from expreplay.py
    
    if os.path.exists(full_dir) and os.path.exists(expreplay_dir):
        compare_results(full_dir, expreplay_dir)
    else:
        print(f"Error: Results directories not found")
        print(f"  Full.py results: {full_dir} (exists: {os.path.exists(full_dir)})")
        print(f"  Expreplay.py results: {expreplay_dir} (exists: {os.path.exists(expreplay_dir)})")
