import os
import shutil
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import json
import itertools
import pandas as pd

from rdkit import Chem
import numpy as np

from divopt.scoring import BenchmarkScoringFunction
from drugex.training.rewards import SingleReward
from drugex.training.environment import DrugExEnvironment

# DrugEx & Diverse-Hits imports
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
from ModelScorer import ModelScorer
from drugex.training.explorers import FragGraphExplorer
from torch.optim import Adam
from drugex.utils import ScheduledOptim

CLEAR_RESULTS = True

SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results")

#config
BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'
TARGET_CONFIG = {
    "drd2": {
        "frag_file":  "drd2_fragbase.txt",
        "model_pkg":  "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir":"diverse-hits/data/scoring_functions/drd2"
    },
    "gsk3": {
        "frag_file":  "gsk3_fragbase.txt",
        "model_pkg":  "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir":"diverse-hits/data/scoring_functions/gsk3"
    },
    "jnk3": {
        "frag_file":  "jnk3_fragbase.txt",
        "model_pkg":  "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir":"diverse-hits/data/scoring_functions/jnk3"
    },
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#preprocessing
def preprocess_target(target: str, config: dict, n_proc=4):    
    ds_path = BASE_DATASETS_PATH
    frag_path = os.path.join(ds_path, config["frag_file"])
    df = pd.read_csv(frag_path, header=None)
    smiles = set(Standardization(n_proc=n_proc).apply(df.iloc[:,0]))
    
    #encoder
    encoder = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4,4,'brics'),
        encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
        pairs_splitter=FragmentPairsSplitter(0.1, len(smiles)),
        n_proc=n_proc
    )
    #encoded folder
    enc_dir = os.path.join(RESULTS_ROOT, target, "preprocessing", "encoded")
    os.makedirs(enc_dir, exist_ok=True)
    train_file = os.path.join(enc_dir, f"{target}_train.tsv")
    test_file  = os.path.join(enc_dir, f"{target}_test.tsv")

    #save to disk
    train_ds = GraphFragDataSet(train_file, rewrite=True)
    test_ds  = GraphFragDataSet(test_file,  rewrite=True)
    encoder.apply(list(smiles), encodingCollectors=[test_ds, train_ds])

    # reload
    train_ds = GraphFragDataSet(train_file, rewrite=False)
    test_ds  = GraphFragDataSet(test_file,  rewrite=False)

    # return paths so we can reload agent per run
    return {
        "train_ds":    train_ds,
        "test_ds":     test_ds,
        "vocab_file":  config["vocab_file"],
        "model_pkg":   config["model_pkg"],
        "scoring_dir": config["scoring_dir"]
    }

#per target setup
TARGET_DATA = {}
for tgt, conf in TARGET_CONFIG.items():
    logger.info(f"Preprocessing target {tgt}...")
    TARGET_DATA[tgt] = preprocess_target(tgt, conf)

def dict_product(param_grid):
    keys = list(param_grid.keys())
    for vals in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, vals))

def train_one_run(target, td, time_budget, sample_budget,
                  learning_rate, batch_size, epsilon,
                  run_dir, epochs=10):
    os.makedirs(run_dir, exist_ok=True)

    #correction - reload agent
    ds_path = BASE_DATASETS_PATH
    vocab = VocGraph.fromFile(os.path.join(ds_path, td["vocab_file"]))
    agent = GraphTransformer(voc_trg=vocab, use_gpus=[0])
    agent.loadStatesFromFile(os.path.join(ds_path, td["model_pkg"]))
    agent.to(device)

    train_ds = td["train_ds"]
    test_ds  = td["test_ds"]
    sf_path  = td["scoring_dir"]

    #scoring function
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
    ms = ModelScorer(bh, time_budget=time_budget, sample_budget=sample_budget)
    ms.reset_budgets()

    # env
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
        Adam(agent.parameters(), betas=(0.9,0.98), eps=1e-9),
        learning_rate, 512
    )

    # fit
    explorer.fit(
        train_loader=train_ds.asDataLoader(batch_size=batch_size),
        valid_loader=test_ds.asDataLoader(batch_size=batch_size),
        epochs=epochs,
        monitor=None
    )

    # results save
    results = {
        "best_value":   explorer.best_value,
        "eval_count":   ms.eval_count,
        "elapsed_time": ms.elapsed_time
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    return results

#hyperparam search
def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if CLEAR_RESULTS:
        for tgt in TARGET_DATA:
            for cname in ["time","sample"]:
                dir_to_clear = os.path.join(RESULTS_ROOT, tgt, cname)
                if os.path.exists(dir_to_clear):
                    shutil.rmtree(dir_to_clear)

    constraints = {"time":(600,None), "sample":(None,10000)}
    param_grid  = {
        "learning_rate":[1, 0.5],
        "batch_size":   [64, 128],
        "epsilon":     [0.05, 0.1]
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
                            run_dir=run_dir,
                            **params
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