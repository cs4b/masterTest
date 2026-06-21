"""
Extract metrics from results/missing_values and output in same format as results_metrics.csv

Reads summary.json files from best_variance_* directories and outputs individual
run metrics matching the column structure of the existing results_metrics.csv
"""
import os
import json
import pandas as pd
from pathlib import Path

def extract_metrics_from_missing_values(results_root):
    """
    Extract metrics from results/missing_values directory.
    
    Outputs one row per individual repeat (from the repeats array in summary.json),
    matching the format of results_metrics.csv columns.
    """
    records = []
    
    # Iterate through targets
    for target_entry in os.listdir(results_root):
        target_dir = os.path.join(results_root, target_entry)
        if not os.path.isdir(target_dir):
            continue
        
        target = target_entry.lower()
        
        # Look for best_variance_* directories
        for variance_entry in os.listdir(target_dir):
            variance_dir = os.path.join(target_dir, variance_entry)
            if not os.path.isdir(variance_dir):
                continue
            
            # Match best_variance_<constraint> pattern
            if not variance_entry.startswith("best_variance_"):
                continue
            
            constraint = variance_entry.replace("best_variance_", "")
            optimization_type = "Sample" if constraint == "sample" else "Time"
            
            summary_path = os.path.join(variance_dir, "summary.json")
            
            if not os.path.isfile(summary_path):
                continue
            
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                # Extract repeats array
                repeats = summary.get("repeats", [])
                
                for repeat in repeats:
                    record = {
                        "Model": "Reseeded Graph Transformer",
                        "Target": target.upper(),
                        "Optimization_Type": optimization_type,
                        "Run_Directory": repeat.get("run_dir", ""),
                        "best_value": repeat.get("best_value"),
                        "unique_hits": repeat.get("unique_hits"),
                        "hit_count_raw": repeat.get("hit_count_raw"),
                        "total_molecules": repeat.get("total_molecules"),
                        "elapsed_time": repeat.get("elapsed_time"),
                        "eval_count": repeat.get("eval_count"),
                        "learning_rate": repeat.get("learning_rate"),
                        "batch_size": repeat.get("batch_size"),
                        "epsilon": repeat.get("epsilon"),
                        "#CIRCLES": repeat.get("#CIRCLES"),
                        "seed": repeat.get("seed"),
                        "budget_hit": repeat.get("budget_hit"),
                        "budget_error": repeat.get("budget_error", ""),
                        "status": repeat.get("budget_error") if repeat.get("budget_error") else "",
                    }
                    
                    records.append(record)
                    print(f"✓ Extracted repeat: {target}/{constraint}/{repeat.get('seed')}")
                
            except Exception as e:
                print(f"✗ Error reading {summary_path}: {e}")
                continue
    
    return records

def main():
    # Determine results path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    results_root = os.path.join(project_root, "results/missing_values")
    
    if not os.path.isdir(results_root):
        print(f"Results directory not found: {results_root}")
        return
    
    print(f"Reading metrics from: {results_root}\n")
    
    # Extract metrics
    records = extract_metrics_from_missing_values(results_root)
    
    if not records:
        print("\nNo metrics found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Ensure column order matches results_metrics.csv
    column_order = [
        "Model", "Target", "Optimization_Type", "Run_Directory",
        "best_value", "unique_hits", "hit_count_raw", "total_molecules",
        "elapsed_time", "eval_count", "learning_rate", "batch_size", "epsilon",
        "#CIRCLES", "seed", "budget_hit", "budget_error", "status"
    ]
    
    # Reorder columns (only keep columns that exist)
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    # Sort by target and optimization type
    df = df.sort_values(['Target', 'Optimization_Type', 'seed']).reset_index(drop=True)
    
    # Save to CSV in the results directory
    output_path = os.path.join(results_root, "results_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Extracted metrics for {len(records)} individual runs")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Display summary
    display_df = df[['Model', 'Target', 'Optimization_Type', '#CIRCLES', 'unique_hits', 'seed']].copy()
    print(display_df.to_string(index=False))
    print()

if __name__ == "__main__":
    main()
