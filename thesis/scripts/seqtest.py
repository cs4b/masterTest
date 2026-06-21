import os
import shutil
import torch
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from drugex.data.processing import Standardization, CorpusEncoder
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.training.monitors import FileMonitor

from ModelScorer import ModelScorer

# Configuration
CLEAR_RESULTS = True
SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results/rnn")
BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'

TARGET_CONFIG = {
    "drd2": {
        #"data_file":  "drd2_fragbase.txt",
        "vocab_file": "Papyrus05.5_smiles_rnn_PT.vocab",
        "model_pkg":  "Papyrus05.5_smiles_rnn_PT.pkg",
        "scoring_dir":"diverse-hits/data/scoring_functions/drd2"
    },
    "gsk3": {
        #"data_file":  "gsk3_fragbase.txt",
        "vocab_file": "Papyrus05.5_smiles_rnn_PT.vocab",
        "model_pkg":  "Papyrus05.5_smiles_rnn_PT.pkg",
        "scoring_dir":"diverse-hits/data/scoring_functions/gsk3"
    },
    "jnk3": {
        #"data_file":  "jnk3_fragbase.txt",
        "vocab_file": "Papyrus05.5_smiles_rnn_PT.vocab",
        "model_pkg":  "Papyrus05.5_smiles_rnn_PT.pkg",
        "scoring_dir":"diverse-hits/data/scoring_functions/jnk3"
    },
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LoggingScorer(ModelScorer):
    """Wraps your ModelScorer so that budgets are enforced *and* every
    molecule/score is appended to rnn_training_molecules.csv.
    Budgets count each scored item (not just unique) to guarantee stopping.
    """
    def __init__(self, bh, time_budget, sample_budget, log_csv):
        super().__init__(bh, time_budget=time_budget, sample_budget=sample_budget)
        self.log_csv = log_csv
        os.makedirs(os.path.dirname(self.log_csv), exist_ok=True)
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w") as f:
                f.write("smiles,reward,elapsed_time,eval_count,time_budget,sample_budget\n")
        # local counters so we can enforce regardless of BH's internal uniqueness rules
        self._tb = time_budget
        self._sb = sample_budget
        self._start = None
        self.eval_count = 0
        self.elapsed_time = 0.0

    def getScores(self, mols, frags=None):
        # start timer on first scoring call
        if self._start is None:
            self._start = time.time()
        # convert mols to SMILES for DH scoring
        smiles = []
        for m in mols:
            try:
                smiles.append(Chem.MolToSmiles(m) if m is not None else None)
            except Exception:
                smiles.append(None)
        # delegate to underlying diverse-hits scorer (returns scalar rewards)
        rewards = self.dh_scorer(smiles)
        # update counters (count everything we attempted to score)
        self.eval_count += len(rewards)
        self.elapsed_time = time.time() - self._start
        # append rows immediately
        try:
            with open(self.log_csv, "a") as f:
                for smi, r in zip(smiles, rewards):
                    f.write(f"{smi},{r},{self.elapsed_time},{self.eval_count},{self._tb},{self._sb}\n")
        except Exception:
            pass
        # enforce budgets by raising (stops RL loop)
        if self._sb is not None and self.eval_count >= self._sb:
            raise RuntimeError(f"Sample budget of {self._sb} reached ({self.eval_count} molecules scored)")
        if self._tb is not None and self.elapsed_time >= self._tb:
            raise TimeoutError(f"Time budget of {self._tb}s reached ({self.elapsed_time:.2f}s)")
        return rewards

def dict_product(param_grid):
    keys = list(param_grid.keys())
    for vals in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, vals))

def train_one_run(target: str, td: dict, time_budget, sample_budget,
                  learning_rate: float, epsilon: float,
                  run_dir: str, epochs: int = 10):
    os.makedirs(run_dir, exist_ok=True)

    # 1) Load pretrained agent + vocab
    ds_path = BASE_DATASETS_PATH
    vocab = VocSmiles.fromFile(os.path.join(ds_path, td["vocab_file"]),encode_frags=False)

    agent = SequenceRNN(vocab, is_lstm=True, use_gpus=[0])
    agent.loadStatesFromFile(os.path.join(ds_path, td["model_pkg"]))
    agent.to(device)

    # 2) Diverse-Hits scorer with budgets (time & sample/evaluation)
    sf_dir = td["scoring_dir"]
    bh = BenchmarkScoringFunction(
        scoring_function_dir=sf_dir,
        time_budget=time_budget,
        sample_budget=sample_budget,
        memory_distance_threshold=None,
        memory_score_threshold=None,
        memory_known_active_init=False,
        use_property_constraints=True,
        n_jobs=8,
        print_progress=False,
    )

    try:
        if hasattr(bh, "start_timer_and_reset"):
            bh.start_timer_and_reset()
        else:
            if hasattr(bh, "reset"):
                bh.reset()
        if not hasattr(bh, "start_time"):
            bh.start_time = time.monotonic()
    except Exception:
        bh.start_time = time.monotonic()
    
    #bh.start_timer_and_reset()
    log_csv = os.path.join(run_dir, "rnn_training_molecules.csv")
    ms = LoggingScorer(bh, time_budget=time_budget, sample_budget=sample_budget, log_csv=log_csv)

    #ms = ModelScorer(bh, time_budget=time_budget, sample_budget=sample_budget)
    #ms.reset_budgets()

    # 3) RL environment + explorer (pure on-policy sampling)
    env = DrugExEnvironment(
        scorers=[ms],
        thresholds=[0.7],  # adjust to your desirability threshold choice
        reward_scheme=SingleReward(),
    )

    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        n_samples=2000,    # molecules generated per RL step (tune as needed)
        epsilon=epsilon,   # exploration rate
        device=device,
    )

    explorer.train_log_path = os.path.join(run_dir, "rnn_training_molecules.csv")
    explorer.optim = ScheduledOptim(
        Adam(agent.parameters(), betas=(0.9, 0.98), eps=1e-9),
        learning_rate,
        512,  # warmup denominator (kept as a constant for ScheduledOptim)
    )
    monitor = FileMonitor(os.path.join(run_dir, "seq_rnn"), save_smiles=True)
    # 4) Fit (no loaders — SequenceExplorer samples internally)
    final_status = "OK"
    try:
        explorer.fit(
            epochs=10**9,
            monitor=monitor,
            patience=10**9,
            min_epochs=1,
            reload_interval=10**9,
        )
    except Exception as e:
        final_status = f"EXCEPTION: time_budget={time_budget}, sample_budget={sample_budget}, error={e}"
        # append explicit stop-line to the training molecules log
        try:
            with open(log_csv, "a") as f:
                f.write(final_status + "\n")
        except Exception:
            pass
        # re-raise so callers know the run terminated via budgets/other errors
        raise
    finally:
        # 5) Save summary metrics always
        results = {
            "best_value":   getattr(explorer, "best_value", None),
            "eval_count":   getattr(ms, "eval_count", None),
            "elapsed_time": getattr(ms, "elapsed_time", None),
            "status":       final_status,
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as fp:
            json.dump(results, fp, indent=2)

    return results

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    if CLEAR_RESULTS:
        for tgt in TARGET_CONFIG:
            for cname in ["time", "sample"]:
                dir_to_clear = os.path.join(RESULTS_ROOT, tgt, cname)
                if os.path.exists(dir_to_clear):
                    shutil.rmtree(dir_to_clear)

    # Two regimes: cap wall-clock or cap eval count
    constraints = {"time": (600, None), "sample": (None, 10000)}

    # Minimal grid; tune as you like
    param_grid  = {
        "learning_rate": [1.0, 0.5],
        "epsilon":       [0.05, 0.1],
    }
    n_runs = 5

    for tgt, td in TARGET_CONFIG.items():
        for cname, (tb, sb) in constraints.items():
            for params in dict_product(param_grid):
                combo = "_".join(f"{k}{v}" for k, v in params.items())
                for run_id in range(1, n_runs + 1):
                    run_dir = os.path.join(RESULTS_ROOT, tgt, cname, combo, f"run{run_id}")
                    print(f"[{tgt} | {cname} | {combo} | run{run_id}]")
                    try:
                        train_one_run(
                            target=tgt,
                            td=td,
                            time_budget=tb,
                            sample_budget=sb,
                            run_dir=run_dir,
                            epochs=10,
                            **params,
                        )
                    except Exception as e:
                        continue


if __name__ == "__main__":
    main()

"""
ds_path = BASE_DATASETS_PATH
data_path = os.path.join(ds_path, "drd2_fragbase.txt")
df = pd.read_csv(data_path, header=None, names=['smiles'])
smiles = set(Standardization(n_proc=4).apply(df['smiles'].tolist()))
vocab = VocSmiles.fromFile(os.path.join(ds_path, "Papyrus05.5_smiles_rnn_PT.vocab"),encode_frags=False)

pretrained = SequenceRNN(vocab, is_lstm=True, use_gpus=[0])
pretrained.loadStatesFromFile(os.path.join(ds_path, "Papyrus05.5_smiles_rnn_PT.pkg"))
generated = pretrained.generate(num_samples=100)
print(generated)
"""