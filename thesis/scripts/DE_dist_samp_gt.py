import os
import shutil
import json
import math
import random
import itertools
from datetime import datetime
from typing import Dict, Any
import time

import numpy as np
import pandas as pd
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from rdkit import Chem

# Diverse-Hits scoring
from divopt.scoring import BenchmarkScoringFunction

# DrugEx imports
from drugex.logs import logger
from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.generators import GraphTransformer
from drugex.data.processing import Standardization
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.data.fragments import (
    FragmentCorpusEncoder,
    GraphFragmentEncoder,
    FragmentPairsSplitter
)
from drugex.data.datasets import GraphFragDataSet
from drugex.training.rewards import SingleReward
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from torch.optim import Adam
from drugex.utils import ScheduledOptim

from ModelScorer import ModelScorer

# ------------------------- Benchmark config -------------------------

CLEAR_RESULTS = True
N_TRIALS_PER_COMBO = 15          # Diverse-Hits used 15 HP trials
N_REPEATS_BEST = 5               # Re-run best config with 5 seeds

SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results")

BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'
TARGET_CONFIG = {
    "drd2": {"frag_file": "drd2_fragbase.txt",
             "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
             "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
             "scoring_dir":"diverse-hits/data/scoring_functions/drd2"},
    "gsk3": {"frag_file": "gsk3_fragbase.txt",
             "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
             "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
             "scoring_dir":"diverse-hits/data/scoring_functions/gsk3"},
    "jnk3": {"frag_file": "jnk3_fragbase.txt",
             "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
             "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
             "scoring_dir":"diverse-hits/data/scoring_functions/jnk3"},
}

# Two compute-constraint regimes (sample-limited and time-limited)
CONSTRAINTS = {
    "sample": {"time_budget": None, "sample_budget": 10_000},
    "time":   {"time_budget": 600,  "sample_budget": None},
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------------- Utilities -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------------- Preprocessing -------------------------

def preprocess_target(target: str, config: dict, n_proc=4):
    ds_path = BASE_DATASETS_PATH
    frag_path = os.path.join(ds_path, config["frag_file"])
    df = pd.read_csv(frag_path, header=None)
    smiles = set(Standardization(n_proc=n_proc).apply(df.iloc[:, 0]))

    encoder = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4, 4, 'brics'),
        encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
        pairs_splitter=FragmentPairsSplitter(0.1, len(smiles)),
        n_proc=n_proc
    )
    enc_dir = os.path.join(RESULTS_ROOT, target, "preprocessing", "encoded")
    os.makedirs(enc_dir, exist_ok=True)
    train_file = os.path.join(enc_dir, f"{target}_train.tsv")
    test_file  = os.path.join(enc_dir, f"{target}_test.tsv")

    train_ds = GraphFragDataSet(train_file, rewrite=True)
    test_ds  = GraphFragDataSet(test_file,  rewrite=True)
    encoder.apply(list(smiles), encodingCollectors=[test_ds, train_ds])

    # reload for later use
    train_ds = GraphFragDataSet(train_file, rewrite=False)
    test_ds  = GraphFragDataSet(test_file,  rewrite=False)

    return {
        "train_ds":    train_ds,
        "test_ds":     test_ds,
        "vocab_file":  config["vocab_file"],
        "model_pkg":   config["model_pkg"],
        "scoring_dir": config["scoring_dir"]
    }

logger.info("Preprocessing all targets...")

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

    # Optional: ensure we can reset between runs if ever reused
    def reset_budgets(self):
        self._start = None
        self.eval_count = 0
        self.elapsed_time = 0.0

# ------------------------- Distribution-based HP sampling -------------------------
# Mirrors Diverse-Hits: sample from distributions, not full grids.

def sample_hyperparams(rng: np.random.Generator) -> Dict[str, Any]:
    """
    Returns a dict with:
      - learning_rate: log-uniform sample (see notes below)
      - batch_size: randint in [64, 512], then round to multiple of 16
      - epsilon: uniform in [0.05, 0.5]
    Notes:
      * LR: DrugEx examples often use Transformer-style schedulers. Log-uniform search
        is standard; we search base LR in [1e-4, 1.0]. If using ScheduledOptim, the
        effective LR is scaled internally.
      * Batch size: practical range for GPU memory; we keep it broad.
      * ε: exploration strength for FragGraphExplorer.
    """
    lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(1.0))))
    bs = int(rng.integers(64, 513))                  # inclusive of 512
    bs = max(64, min(512, bs - (bs % 16)))          # snap to multiple of 16
    eps = float(rng.uniform(0.05, 0.5))
    return {"learning_rate": lr, "batch_size": bs, "epsilon": eps}

# ------------------------- Single training run -------------------------

def train_one_run(target: str, td: dict, time_budget, sample_budget,
                  learning_rate: float, batch_size: int, epsilon: float,
                  run_dir: str, seed: int, epochs: int = 10) -> Dict[str, Any]:
    ensure_dir(run_dir)
    set_seed(seed)

    # reload agent each run
    ds_path = BASE_DATASETS_PATH
    vocab = VocGraph.fromFile(os.path.join(ds_path, td["vocab_file"]))
    agent = GraphTransformer(voc_trg=vocab, use_gpus=[0])
    agent.loadStatesFromFile(os.path.join(ds_path, td["model_pkg"]))
    agent.to(device)

    train_ds = td["train_ds"]
    test_ds  = td["test_ds"]
    sf_path  = td["scoring_dir"]

    # scoring function (Diverse-Hits)
    bh = BenchmarkScoringFunction(
        scoring_function_dir=sf_path,
        time_budget=time_budget,
        sample_budget=sample_budget,
        memory_distance_threshold=None,
        memory_score_threshold=None,
        memory_known_active_init=False,
        use_property_constraints=True,
        n_jobs=8, print_progress=False
    )
    bh.start_timer_and_reset()
    log_csv = os.path.join(run_dir, "rs_training_molecules.csv")
    ms = LoggingScorer(bh, time_budget=time_budget, sample_budget=sample_budget,log_csv=log_csv)
    ms.reset_budgets()

    # RL env + explorer
    env = DrugExEnvironment(
        scorers=[ms],
        thresholds=[0.7],
        reward_scheme=SingleReward()
    )
    explorer = FragGraphExplorer(
        agent=agent, env=env, mutate=agent,
        n_samples=2000, epsilon=epsilon, device=device
    )
    explorer.train_log_path = os.path.join(run_dir, "training_molecules.csv")
    explorer.optim = ScheduledOptim(
        Adam(agent.parameters(), betas=(0.9, 0.98), eps=1e-9),
        learning_rate, 512
    )

    explorer.fit(
        train_loader=train_ds.asDataLoader(batch_size=batch_size),
        valid_loader=test_ds.asDataLoader(batch_size=batch_size),
        epochs=epochs,
        monitor=None
    )

    results = {
        "best_value":   explorer.best_value,
        "eval_count":   ms.eval_count,
        "elapsed_time": ms.elapsed_time,
        "seed":         seed,
        "learning_rate": learning_rate,
        "batch_size":    batch_size,
        "epsilon":       epsilon
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    return results

# ------------------------- Benchmark orchestration -------------------------

def run_trials_for_combo(tgt: str, td: dict, constraint_name: str,
                         time_budget, sample_budget, base_out: str,
                         num_trials: int, rng: np.random.Generator):
    """
    Create a 'hyperparameter_search' directory per (task, constraint),
    run N sampled trials, each labeled with its trial index + sampled HPs.
    """
    hp_root = os.path.join(base_out, tgt, "hyperparameter_search", constraint_name)
    ensure_dir(hp_root)

    trial_summaries = []
    for idx in range(1, num_trials + 1):
        hp = sample_hyperparams(rng)
        # label: <tgt>_<constraint>_trial<idx>_lr{...}_bs{...}_eps{...}
        tag = f"{tgt}_{constraint_name}_trial{idx}_lr{hp['learning_rate']:.4g}_bs{hp['batch_size']}_eps{hp['epsilon']:.3f}"
        run_dir = os.path.join(hp_root, tag)
        print(f"[{now()}] Trial {idx}/{num_trials} :: {tag}")

        try:
            res = train_one_run(
                target=tgt, td=td,
                time_budget=time_budget, sample_budget=sample_budget,
                learning_rate=hp["learning_rate"], batch_size=hp["batch_size"], epsilon=hp["epsilon"],
                run_dir=run_dir, seed=idx  # deterministic per trial index
            )
            res["run_dir"] = run_dir
            trial_summaries.append(res)
        except Exception as e:
            ensure_dir(run_dir)
            with open(os.path.join(run_dir, "training_molecules.csv"), "a") as log_f:
                log_f.write(f"# EXCEPTION: time_budget={time_budget}, sample_budget={sample_budget}, error={e}\n")
            continue

    # pick best by highest best_value (proxy for “most hits” under the constraint)
    if trial_summaries:
        best = max(trial_summaries, key=lambda x: x["best_value"])
    else:
        best = None

    # save index file
    with open(os.path.join(hp_root, "trial_summaries.json"), "w") as fp:
        json.dump(trial_summaries, fp, indent=2)

    return best

def repeat_best_config(best: dict, tgt: str, constraint_name: str, td: dict,
                       time_budget, sample_budget, base_out: str,
                       num_repeats: int):
    """
    Re-run best HPs with 5 independent seeds and store mean ± range.
    """
    if best is None:
        return None

    repeat_root = os.path.join(base_out, tgt, f"best_variance_{constraint_name}")
    ensure_dir(repeat_root)

    lr = best["learning_rate"]; bs = best["batch_size"]; eps = best["epsilon"]
    seeds = list(range(1001, 1001 + num_repeats))

    repeat_results = []
    for i, seed in enumerate(seeds, 1):
        tag = f"{tgt}_{constraint_name}_best_lr{lr:.4g}_bs{bs}_eps{eps:.3f}_rep{i}"
        run_dir = os.path.join(repeat_root, tag)
        print(f"[{now()}] Repeat {i}/{num_repeats} :: {tag}")

        res = train_one_run(
            target=tgt, td=td,
            time_budget=time_budget, sample_budget=sample_budget,
            learning_rate=lr, batch_size=bs, epsilon=eps,
            run_dir=run_dir, seed=seed
        )
        res["run_dir"] = run_dir
        repeat_results.append(res)

    # compute mean ± range for the main metric (best_value)
    best_values = [r["best_value"] for r in repeat_results]
    mean_val = float(np.mean(best_values))
    rng_val  = float(np.max(best_values) - np.min(best_values))
    summary = {
        "learning_rate": lr,
        "batch_size": bs,
        "epsilon": eps,
        "seeds": seeds,
        "metric": "best_value",
        "mean": mean_val,
        "range": rng_val,
        "repeats": repeat_results
    }
    with open(os.path.join(repeat_root, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    return summary

def main():
    if CLEAR_RESULTS:
        shutil.rmtree(RESULTS_ROOT, ignore_errors=True)
        
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # global RNG for HP sampling
    rng = np.random.default_rng(seed=0)
    TARGET_DATA: Dict[str, Dict[str, Any]] = {t: preprocess_target(t, c) for t, c in TARGET_CONFIG.items()}
    for tgt, td in TARGET_DATA.items():
        for cname, lims in CONSTRAINTS.items():
            print(f"\n=== {tgt.upper()} | constraint={cname} ===")

            # 1) HP trials (distribution sampling)
            best = run_trials_for_combo(
                tgt=tgt, td=td, constraint_name=cname,
                time_budget=lims["time_budget"], sample_budget=lims["sample_budget"],
                base_out=RESULTS_ROOT, num_trials=N_TRIALS_PER_COMBO, rng=rng
            )

            # 2) Re-run best with 5 seeds; label folders like the paper
            _ = repeat_best_config(
                best=best, tgt=tgt, constraint_name=cname, td=td,
                time_budget=lims["time_budget"], sample_budget=lims["sample_budget"],
                base_out=RESULTS_ROOT, num_repeats=N_REPEATS_BEST
            )

if __name__ == "__main__":
    main()
