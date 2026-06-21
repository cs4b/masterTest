import os
import shutil
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import itertools
import pandas as pd
import numpy as np
from rdkit import Chem

from divopt.scoring import BenchmarkScoringFunction
from drugex.training.rewards import SingleReward
from drugex.training.environment import DrugExEnvironment

# DrugEx & Diverse-Hits imports
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.processing import Standardization
from drugex.data.datasets import SmilesDataSet
from drugex.training.generators import SequenceRNN
from drugex.training.explorers import SequenceExplorer
from torch.optim import Adam
from drugex.utils import ScheduledOptim

from ModelScorer import ModelScorer

# Configuration
CLEAR_RESULTS = True
SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results/rnn")
BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'

TARGET_CONFIG = {
    "drd2": {
        "data_file":  "drd2_smiles.csv",
        "vocab_file": "Papyrus05.5_smiles_rnn_PT.vocab",
        "model_pkg":  "Papyrus05.5_smiles_rnn_PT.pkg",
        "scoring_dir":"diverse-hits/data/scoring_functions/drd2"
    },
    "gsk3": {
        "data_file":  "gsk3_smiles.csv",
        "vocab_file": "Papyrus05.5_smiles_rnn_PT.vocab",
        "model_pkg":  "Papyrus05.5_smiles_rnn_PT.pkg",
        "scoring_dir":"diverse-hits/data/scoring_functions/gsk3"
    },
    "jnk3": {
        "data_file":  "jnk3_smiles.csv",
        "vocab_file": "Papyrus05.5_smiles_rnn_PT.vocab",
        "model_pkg":  "Papyrus05.5_smiles_rnn_PT.pkg",
        "scoring_dir":"diverse-hits/data/scoring_functions/jnk3"
    },
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Preprocess SMILES data for each target
def preprocess_target(target: str, config: dict, n_proc=4):    
    ds_path = BASE_DATASETS_PATH
    data_path = os.path.join(ds_path, config["data_file"])
    df = pd.read_csv(data_path, header=None, names=['smiles'])
    smiles = set(Standardization(n_proc=n_proc).apply(df['smiles'].tolist()))
    
    # Build vocabulary from pretrained vocab file
    vocab = VocSmiles.fromFile(os.path.join(ds_path, config["vocab_file"]))

    # Split into train/test
    split_list = list(smiles)
    np.random.shuffle(split_list)
    split_idx = int(0.9 * len(split_list))
    train_smiles = split_list[:split_idx]
    test_smiles  = split_list[split_idx:]

    # Save split files
    data_dir = os.path.join(RESULTS_ROOT, target, "preprocessing", "data")
    os.makedirs(data_dir, exist_ok=True)
    train_file = os.path.join(data_dir, f"{target}_train.csv")
    test_file  = os.path.join(data_dir, f"{target}_test.csv")
    pd.Series(train_smiles).to_csv(train_file, index=False, header=False)
    pd.Series(test_smiles).to_csv(test_file,  index=False, header=False)

    # Create dataset objects
    train_ds = SmilesDataSet(train_file, vocab=vocab)
    test_ds  = SmilesDataSet(test_file,  vocab=vocab)

    return {
        "train_ds":   train_ds,
        "test_ds":    test_ds,
        "vocab":      vocab,
        "model_pkg":  config["model_pkg"],
        "scoring_dir":config["scoring_dir"]
    }

# Prepare data for all targets
TARGET_DATA = {}
for tgt, conf in TARGET_CONFIG.items():
    TARGET_DATA[tgt] = preprocess_target(tgt, conf)

# Utility: Cartesian product of parameter grid
def dict_product(param_grid):
    keys = list(param_grid.keys())
    for vals in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, vals))

# Single training run
def train_one_run(target, td, time_budget, sample_budget,
                  learning_rate, batch_size, epsilon,
                  run_dir, epochs=10):
    os.makedirs(run_dir, exist_ok=True)

    # Reload agent
    ds_path = BASE_DATASETS_PATH
    agent = SequenceRNN(voc=td["vocab"], use_gpus=[0])
    agent.loadStatesFromFile(os.path.join(ds_path, td["model_pkg"]))
    agent.to(device)

    train_ds = td["train_ds"]
    test_ds  = td["test_ds"]

    # Initialize diverse-hits based scoring
    bh = BenchmarkScoringFunction(
        scoring_function_dir=td["scoring_dir"],
        time_budget=time_budget,
        sample_budget=sample_budget,
        memory_distance_threshold=None,
        memory_score_threshold=None,
        memory_known_active_init=False,
        use_property_constraints=True,
        n_jobs=8, print_progress=False
    )
    bh.start_timer_and_reset()

    ms = ModelScorer(bh, time_budget=time_budget, sample_budget=sample_budget)
    ms.reset_budgets()

    # Setup RL environment
    env = DrugExEnvironment(
        scorers=[ms],
        thresholds=[0.7],
        reward_scheme=SingleReward()
    )

    # Explorer for sequence RNN
    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        mutate=agent,
        n_samples=2000,
        epsilon=epsilon,
        device=device
    )
    explorer.train_log_path = os.path.join(run_dir, "training_molecules.csv")
    explorer.optim = ScheduledOptim(
        Adam(agent.parameters(), betas=(0.9,0.98), eps=1e-9),
        learning_rate,
        128
    )

    # Fit model
    explorer.fit(
        train_loader=train_ds.asDataLoader(batch_size=batch_size),
        valid_loader=test_ds.asDataLoader(batch_size=batch_size),
        epochs=epochs
    )

    # Save metrics
    results = {
        "best_value":   explorer.best_value,
        "eval_count":   ms.eval_count,
        "elapsed_time": ms.elapsed_time
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    return results

# Hyperparameter search
def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if CLEAR_RESULTS:
        for tgt in TARGET_DATA:
            for cname in ["time", "sample"]:
                dir_to_clear = os.path.join(RESULTS_ROOT, tgt, cname)
                if os.path.exists(dir_to_clear):
                    shutil.rmtree(dir_to_clear)

    constraints = {"time": (600, None), "sample": (None, 10000)}
    param_grid  = {
        "learning_rate": [1, 0.5],
        "batch_size":    [64, 128],
        "epsilon":       [0.05, 0.1]
    }
    n_runs = 5

    for tgt, td in TARGET_DATA.items():
        for cname, (tb, sb) in constraints.items():
            for params in dict_product(param_grid):
                combo = "_".join(f"{k}{v}" for k, v in params.items())
                for run_id in range(1, n_runs+1):
                    run_dir = os.path.join(
                        RESULTS_ROOT, tgt, cname, combo, f"run{run_id}"
                    )
                    print(f"[{tgt} | {cname} | {combo} | run{run_id}]")
                    try:
                        train_one_run(
                            target=tgt,
                            td=td,
                            time_budget=tb,
                            sample_budget=sb,
                            learning_rate=params["learning_rate"],
                            batch_size=params["batch_size"],
                            epsilon=params["epsilon"],
                            run_dir=run_dir
                        )
                    except Exception as e:
                        os.makedirs(run_dir, exist_ok=True)
                        with open(os.path.join(run_dir, "training_molecules.csv"), "a") as log_f:
                            log_f.write(
                                f"# EXCEPTION: time_budget={tb}, sample_budget={sb}, error={e}\n"
                            )
                        continue

if __name__ == "__main__":
    main()
