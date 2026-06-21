"""
Check if DRD2/GSK3 are hitting budget constraints before reseeding can trigger
"""
import json
import os

print("="*80)
print("INVESTIGATING: Why reseeding code never triggers for DRD2/GSK3")
print("="*80)

targets = ["drd2", "gsk3", "jnk3"]
constraints = ["sample"]

for target in targets:
    print(f"\n{target.upper()}")
    print("-" * 80)
    
    trial_dir = f"thesis/results/reseed/{target}/hyperparameter_search/sample"
    trials = sorted([d for d in os.listdir(trial_dir) if os.path.isdir(os.path.join(trial_dir, d))])
    
    # Check first trial metrics
    first_trial = trials[0]
    metrics_file = os.path.join(trial_dir, first_trial, "metrics.json")
    
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        print(f"\nFirst trial: {first_trial}")
        print(f"  Budget hit: {metrics.get('budget_hit', False)}")
        print(f"  Budget error: {metrics.get('budget_error', 'N/A')}")
        print(f"  Eval count: {metrics.get('eval_count', 'N/A')}")
        print(f"  Elapsed time: {metrics.get('elapsed_time', 'N/A'):.2f}s")
        
        # Check if budget_hit is true - this would explain why reseeding never happens
        if metrics.get('budget_hit'):
            print(f"\n  ⚠️  BUDGET WAS HIT - Reseeding loop breaks before next reseed cycle")
        else:
            print(f"\n  ✓ Budget was NOT hit - should have allowed reseeding")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
If DRD2/GSK3 trials show budget_hit=True:
  → Training loop exits immediately when budget is hit
  → Reseeding code in while loop never gets a chance to run
  → This explains why reseeding_seen.txt doesn't exist

If budget_hit=False:
  → Budget wasn't hit, so code should reach reseeding
  → Something else is preventing reseeding from triggering
""")
