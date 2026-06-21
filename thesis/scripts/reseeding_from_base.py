import os
import shutil
import json
import itertools
import time
import pandas as pd
import torch
from rdkit import Chem
import numpy as np

# DrugEx & Diverse-Hits imports
from divopt.scoring import BenchmarkScoringFunction
from drugex.training.rewards import SingleReward
from drugex.training.environment import DrugExEnvironment
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
from ModelScorer import ModelScorer  # base class (we’ll wrap/extend it)
from drugex.training.explorers import FragGraphExplorer
from torch.optim import Adam
from drugex.utils import ScheduledOptim

# -----------------------------------------------------------------------------
# ENV / GLOBALS
# -----------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLEAR_RESULTS = True

SCRIPT_DIR    = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results/reseed")

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


# -----------------------------------------------------------------------------
# HYPERPARAMS MATRIX (exactly as requested)
# -----------------------------------------------------------------------------

HYPERPARAMS = {
    "drd2": {
        "time":   {"learning_rate": 1.0, "batch_size": 128, "epsilon": 0.1},
        "sample": {"learning_rate": 1.0, "batch_size":  64, "epsilon": 0.1},
    },
    "gsk3": {
        "time":   {"learning_rate": 0.5, "batch_size":  64, "epsilon": 0.05},
        "sample": {"learning_rate": 1.0, "batch_size": 128, "epsilon": 0.05},
    },
    "jnk3": {
        "time":   {"learning_rate": 1.0, "batch_size":  64, "epsilon": 0.1},
        "sample": {"learning_rate": 1.0, "batch_size": 128, "epsilon": 0.05},
    },
}

# -----------------------------------------------------------------------------
# LoggingScorer (wraps ModelScorer and enforces budgets + CSV logging)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helpers for reseeding
# -----------------------------------------------------------------------------

def _canonicalize_smiles(smiles_list):
    """Return canonical RDKit SMILES; filter out invalid/None."""
    canon = []
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi:
            continue
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            canon.append(Chem.MolToSmiles(m))
        except Exception:
            continue
    # deduplicate while preserving order
    seen = set()
    out = []
    for s in canon:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _load_high_scoring_novel(log_csv, threshold, already_seeded_set):
    """Read rs_training_molecules.csv, filter reward>=threshold,
    canonicalize, and return only those not already reseeded."""
    if not os.path.exists(log_csv):
        return []
    try:
        df = pd.read_csv(log_csv, comment="#")
    except Exception:
        # fallback: manual parse to skip bad lines
        rows = []
        with open(log_csv, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                rows.append(parts[:2])  # smiles, reward
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=["smiles","reward"])
    # coerce reward
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df = df[df["reward"] >= threshold]
    smiles = _canonicalize_smiles(df["smiles"].tolist())
    # novel w.r.t. ALL prior reseeds
    novel = [s for s in smiles if s not in already_seeded_set]
    return novel


def _append_smiles_to_train(smiles, train_file, n_proc=4):
    if not smiles:
        return 0, GraphFragDataSet(train_file, rewrite=False)

    std_smiles = list(Standardization(n_proc=n_proc).apply(pd.Series(smiles)))

    m = len(std_smiles)
    eff_n_proc = max(1, min(n_proc, m))
    eff_chunk  = max(1, m // eff_n_proc)

    encoder = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4,4,'brics'),
        encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
        pairs_splitter=None,          # <-- no splitter -> exactly 1 split
        n_proc=n_proc,
        chunk_size=eff_chunk
    )

    train_ds = GraphFragDataSet(train_file, rewrite=False)
    encoder.apply(std_smiles, encodingCollectors=[train_ds])   # <-- 1 collector, 1 split

    new_train_ds = GraphFragDataSet(train_file, rewrite=False)
    return len(std_smiles), new_train_ds

# -----------------------------------------------------------------------------
# Preprocessing (same flow as original script)
# -----------------------------------------------------------------------------
def preprocess_target(target: str, config: dict, n_proc=4):
    ds_path = BASE_DATASETS_PATH
    frag_path = os.path.join(ds_path, config["frag_file"])
    df = pd.read_csv(frag_path, header=None)
    smiles = set(Standardization(n_proc=n_proc).apply(df.iloc[:,0]))

    encoder = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4,4,'brics'),
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

    # reload
    train_ds = GraphFragDataSet(train_file, rewrite=False)
    test_ds  = GraphFragDataSet(test_file,  rewrite=False)

    return {
        "train_ds":    train_ds,
        "test_ds":     test_ds,
        "train_file":  train_file,
        "test_file":   test_file,
        "vocab_file":  config["vocab_file"],
        "model_pkg":   config["model_pkg"],
        "scoring_dir": config["scoring_dir"]
    }

# Build per-target preprocessed data up-front
TARGET_DATA = {}
for tgt, conf in TARGET_CONFIG.items():
    logger.info(f"Preprocessing target {tgt}...")
    TARGET_DATA[tgt] = preprocess_target(tgt, conf)

# -----------------------------------------------------------------------------
# Training for a single (target, constraint) with fixed HYPERPARAMS row
# -----------------------------------------------------------------------------
def train_one_run(target, td, time_budget, sample_budget,
                  learning_rate, batch_size, epsilon,
                  run_dir, epochs=10, reseed_every_epochs=1, reseed_threshold=0.7):
    os.makedirs(run_dir, exist_ok=True)

    # Reload agent
    ds_path = BASE_DATASETS_PATH
    vocab = VocGraph.fromFile(os.path.join(ds_path, td["vocab_file"]))
    agent = GraphTransformer(voc_trg=vocab, use_gpus=[0] if torch.cuda.is_available() else [])
    agent.loadStatesFromFile(os.path.join(ds_path, td["model_pkg"]))
    agent.to(device)

    train_ds = td["train_ds"]
    test_ds  = td["test_ds"]
    train_file = td["train_file"]
    sf_path  = td["scoring_dir"]

    # scoring function (BenchmarkScoringFunction)
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

    # our logging scorer (wraps dh scorer)
    log_csv = os.path.join(run_dir, "rs_training_molecules.csv")
    ms = LoggingScorer(bh, time_budget=time_budget, sample_budget=sample_budget, log_csv=log_csv)
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
    explorer.train_log_path = os.path.join(run_dir, "rs_training_molecules.csv")
    explorer.optim = ScheduledOptim(
        Adam(agent.parameters(), betas=(0.9,0.98), eps=1e-9),
        learning_rate, 512
    )

    # fit with reseed
    seen_file = os.path.join(run_dir, "reseeding_seen.txt")
    already_seeded = set()
    if os.path.exists(seen_file):
        with open(seen_file, "r") as f:
            for line in f:
                smi = line.strip()
                if smi:
                    already_seeded.add(smi)

    epochs_done = 0
    while epochs_done < epochs:
        step = min(reseed_every_epochs, epochs - epochs_done)
        explorer.fit(
            train_loader=train_ds.asDataLoader(batch_size=batch_size),
            valid_loader=test_ds.asDataLoader(batch_size=batch_size),  # avoid duplicate logging
            epochs=step,
            monitor=None
        )
        epochs_done += step

        # reseed after this block
        novel = _load_high_scoring_novel(log_csv, threshold=reseed_threshold, already_seeded_set=already_seeded)
        if novel:
            added_count, new_train_ds = _append_smiles_to_train(novel, train_file)
            train_ds = new_train_ds  # swap to reloaded dataset
            # update bookkeeping and persist
            already_seeded.update(novel)
            with open(seen_file, "a") as f:
                for smi in novel:
                    f.write(smi + "\n")
            print(f"[{target}] Reseeded {added_count} molecules (reward >= {reseed_threshold}) into training set.")
        else:
            print(f"[{target}] No novel molecules above threshold {reseed_threshold} to reseed this period.")


    # results save
    results = {
        "best_value":   explorer.best_value,
        "eval_count":   ms.eval_count,
        "elapsed_time": ms.elapsed_time
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    return results

# -----------------------------------------------------------------------------
# Main: EXACTLY two runs per target per constraint using HYPERPARAMS
# -----------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if CLEAR_RESULTS:
        for tgt in TARGET_DATA:
            for cname in ["time","sample"]:
                dir_to_clear = os.path.join(RESULTS_ROOT, tgt, cname)
                if os.path.exists(dir_to_clear):
                    shutil.rmtree(dir_to_clear)

    # fixed budgets (same as original script)
    constraints = {"time":(600,None), "sample":(None,10000)}
    n_runs = 2  # <-- exactly two runs

    for tgt, td in TARGET_DATA.items():
        for cname, (tb, sb) in constraints.items():
            # fetch the one hyperparam row requested
            params = HYPERPARAMS[tgt][cname]
            combo = "_".join(f"{k}{v}" for k, v in params.items())
            for run_id in range(1, n_runs+1):
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
                        reseed_every_epochs=1,      # check every epoch
                        reseed_threshold=0.7,
                        **params
                    )
                except Exception as e:
                        os.makedirs(run_dir, exist_ok=True)
                        with open(os.path.join(run_dir, "rs_training_molecules.csv"), "a") as log_f:
                            log_f.write(
                                f"# EXCEPTION: time_budget={tb}, sample_budget={sb}, error={e}\n"
                            )
                        continue

if __name__ == "__main__":
    main()