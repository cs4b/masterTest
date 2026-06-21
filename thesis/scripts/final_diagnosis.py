"""
Final analysis: Why DRD2/GSK3 don't have reseeding_seen.txt but JNK3 does
"""
import os
import json
from datetime import datetime

print("="*80)
print("FINAL DIAGNOSIS: Reseeding Mystery")
print("="*80)

print("\nSCRIPT TIMESTAMPS:")
print("-" * 80)
for script in ["de_dist_samp_gt_full.py", "DE_dist_samp_gt_expreplay.py", "reseeding.py", "reseeding_from_base.py"]:
    path = f"thesis/scripts/{script}"
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime)
        print(f"  {script:40s} : {dt}")

print("\n\nRESULT TIMESTAMPS & RESEEDING STATUS:")
print("-" * 80)

targets = ["drd2", "gsk3", "jnk3"]
for target in targets:
    print(f"\n{target.upper()}:")
    
    # Check full.py results (from main results dir)
    full_trial = f"thesis/results/{target}/hyperparameter_search/sample"
    if os.path.exists(full_trial):
        trials_full = [d for d in os.listdir(full_trial) if os.path.isdir(os.path.join(full_trial, d))]
        if trials_full:
            first = trials_full[0]
            mtime = os.path.getmtime(os.path.join(full_trial, first, "metrics.json"))
            dt = datetime.fromtimestamp(mtime)
            print(f"  Full.py (thesis/results/{target}/...):        {dt}")
            print(f"    Has reseeding_seen.txt? NO (full.py has no reseeding code)")
    
    # Check expreplay results
    expreplay_trial = f"thesis/results/reseed/{target}/hyperparameter_search/sample"
    if os.path.exists(expreplay_trial):
        trials_reseed = [d for d in os.listdir(expreplay_trial) if os.path.isdir(os.path.join(expreplay_trial, d))]
        if trials_reseed:
            first = sorted(trials_reseed)[0]
            mtime = os.path.getmtime(os.path.join(expreplay_trial, first, "metrics.json"))
            dt = datetime.fromtimestamp(mtime)
            print(f"  Expreplay.py (thesis/results/reseed/{target}/...): {dt}")
            
            # Check for reseeding_seen.txt
            reseed_count = 0
            for trial in trials_reseed:
                reseed_file = os.path.join(expreplay_trial, trial, "reseeding_seen.txt")
                if os.path.exists(reseed_file):
                    reseed_count += 1
            
            if reseed_count > 0:
                print(f"    Has reseeding_seen.txt? YES ({reseed_count}/{len(trials_reseed)} trials)")
            else:
                print(f"    Has reseeding_seen.txt? NO (0/{len(trials_reseed)} trials)")

print("\n\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)
print("""
1. reseeding.py and reseeding_from_base.py are OLDER (August 2025)
2. DE_dist_samp_gt_expreplay.py is NEWER (February 2025)
3. If JNK3 was run with older script, it could have had working reseeding

4. CURRENT CODE (expreplay.py):
   → Budget is hit during first epoch
   → Loop breaks BEFORE reseeding code executes
   → reseeding_seen.txt should NOT be created for ANY target

5. BUT JNK3 HAS reseeding_seen.txt!
   This means:
   a) Older reseeding.py was used for JNK3 results, OR
   b) DRD2/GSK3 code was the current expreplay.py (no reseeding), OR  
   c) Different code paths/conditions for JNK3

RECOMMENDATION:
Check which script actually created each result:
  grep -r "class LoggingScorer" thesis/results*/...
  Check for presence of _append_smiles_to_train function calls
""")

print("\n" + "="*80)
print("CHECKING: Does LoggingScorer exist in reseeding_from_base.py?")
print("="*80)

if os.path.exists("thesis/scripts/reseeding_from_base.py"):
    with open("thesis/scripts/reseeding_from_base.py") as f:
        content = f.read()
        if "LoggingScorer" in content:
            print("✓ YES - reseeding_from_base.py has LoggingScorer")
            if "reseeding_seen" in content:
                print("✓ YES - reseeding_from_base.py has reseeding_seen.txt logic")
        else:
            print("✗ NO - reseeding_from_base.py doesn't have LoggingScorer")
