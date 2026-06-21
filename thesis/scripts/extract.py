import os
import pandas as pd

#top‐level
RESULTS_ROOT = os.path.join(os.getcwd(), "thesis/backresults/reseed")

#threshold 
SCORE_THRESHOLD = 0.7

run_records = []

# target / constraint / combo / runX
for target in os.listdir(RESULTS_ROOT):
    target_dir = os.path.join(RESULTS_ROOT, target)
    if not os.path.isdir(target_dir):
        continue
    for constraint in os.listdir(target_dir):
        if constraint == "preprocessing":
            continue
        constraint_dir = os.path.join(target_dir, constraint)
        if not os.path.isdir(constraint_dir):
            continue
        for combo in os.listdir(constraint_dir):
            combo_dir = os.path.join(constraint_dir, combo)
            if not os.path.isdir(combo_dir):
                continue
            for run in os.listdir(combo_dir):
                run_dir = os.path.join(combo_dir, run)
                if not os.path.isdir(run_dir):
                    continue
                csv_path = os.path.join(run_dir, "rs_training_molecules.csv") #rnn_ prefix for rnn
                if not os.path.isfile(csv_path):
                    continue
                
                df = pd.read_csv(csv_path, header=0, engine="python", on_bad_lines="skip")
                # drop the exception footer if present
                df = df[~df.iloc[:, 0].astype(str).str.startswith("EXCEPTION:")]
                # ensure numeric rewards
                df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
                df = df.dropna(subset=["reward"])

                hit_count = (df["reward"] >= 0.7).sum()
                #count hits / run
                #df = pd.read_csv(csv_path,
                #                 names=["smiles", "reward", "elapsed"], #step removed for rnn from 1st place
                #                comment="#",skipfooter=1,engine="python")
                #hit_count = (df["reward"] >= SCORE_THRESHOLD).sum()

                
                run_records.append({
                    "target":    target,
                    "constraint":constraint,
                    "combo":     combo,
                    "run":       run,
                    "hit_count": int(hit_count)
                })


run_summary_df = pd.DataFrame(run_records, 
                              columns=["target", "constraint", "combo", "run", "hit_count"])

#save
output_csv = os.path.join(RESULTS_ROOT, "reseed_results.csv")
run_summary_df.to_csv(output_csv, index=False)

print(f"Per‐run summary written to: {output_csv}")