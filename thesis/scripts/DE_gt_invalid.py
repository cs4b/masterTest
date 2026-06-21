#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diverse-Hits protocol runner for DrugEx Graph Transformer
- 15-trial random search per (target × constraint) with distributions
- select best config by #Circles (sphere-exclusion over accepted hits)
- freeze and run 5 independent seeds for reporting
"""

import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# Torch / RDKit
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# === External libs (adjust to your install) ==================================
# DrugEx / Diverse-Hits APIs you already use in your codebase
from divopt.scoring import BenchmarkScoringFunction
from drugex.training.rewards import SingleReward
from drugex.training.environment import DrugExEnvironment

# DrugEx graph + fragment pipeline (adjust imports to your layout)
from drugex.training.schedulers import ScheduledOptim
from torch.optim import Adam
from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.generators import GraphTransformer
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.datasets.fragments import (
    FragmentCorpusEncoder, GraphFragmentEncoder,
    FragmentPairsSplitter
)
from drugex.data.processing import Standardization
from drugex.training.explorers import FragGraphExplorer
from drugex.data.datasets import GraphFragDataSet
from ModelScorer import ModelScorer
from drugex.training.explorers import FragGraphExplorer
from torch.optim import Adam
from drugex.utils import ScheduledOptim

# ============================================================================
# Global config
# ============================================================================

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Root folders (edit to your setup)
SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/DEGT")
CLEAR_RESULTS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dh_protocol")

# Target-specific files (adjust if paths differ)
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

# ============================================================================
# Data preparation per target
# ============================================================================

def preprocess_target(target: str, config: dict, n_proc: int = 4):
    """
    Build fragment dataset (train/test loaders) for a target.
    Reuses your BRICS + GraphFragmentEncoder pipeline.
    """
    ds_path = BASE_DATASETS_PATH
    frag_path = os.path.join(ds_path, config["frag_file"])
    df = pd.read_csv(frag_path, header=None)

    # Standardize and unique SMILES
    smiles = set(Standardization(n_proc=n_proc).apply(df.iloc[:, 0]))

    # Encoder: BRICS fragments, graph encoder, scaffold/molecule pairs
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

    #save to disk
    train_ds = GraphFragDataSet(train_file, rewrite=True)
    test_ds  = GraphFragDataSet(test_file,  rewrite=True)
    encoder.apply(list(smiles), encodingCollectors=[test_ds, train_ds])

    # reload
    train_ds = GraphFragDataSet(train_file, rewrite=False)
    test_ds  = GraphFragDataSet(test_file,  rewrite=False)
    return {
        "train_ds": train_ds,
        "test_ds":  test_ds,
        "model_pkg":  config["model_pkg"],
        "vocab_file": config["vocab_file"],
        "scoring_dir": config["scoring_dir"],
    }

# Preprocess once for all targets
TARGET_DATA = {}
for tgt, conf in TARGET_CONFIG.items():
    logger.info(f"Preprocessing target {tgt} ...")
    TARGET_DATA[tgt] = preprocess_target(tgt, conf)

# ============================================================================
# Random-search distributions (not a grid)
# ============================================================================

def _log_uniform(rng: np.random.RandomState, low: float, high: float) -> float:
    """Sample from log-uniform(low, high)."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def _categorical(rng: np.random.RandomState, values, probs=None):
    """Sample from a categorical distribution over 'values'."""
    values = np.asarray(values)
    if probs is None:
        idx = rng.choice(len(values))
    else:
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum()
        idx = rng.choice(len(values), p=probs)
    return values[idx]

def sample_config(rng: np.random.RandomState) -> dict:
    """
    Draw one hyperparameter configuration from method-specific distributions.
    These priors are sensible for DrugEx v3 under Diverse-Hits budgets.
    """
    # Learning-rate scale for ScheduledOptim (log-uniform)
    learning_rate = _log_uniform(rng, 1e-5, 5e-3)

    # Batch size (categorical with mild preference for mid-range)
    batch_vals  = [64, 96, 128, 192, 256]
    batch_probs = [0.10, 0.20, 0.40, 0.20, 0.10]
    batch_size  = int(_categorical(rng, batch_vals, batch_probs))

    # Exploration rate ε (continuous; small values favored). Clip to [0, 0.5].
    epsilon = float(np.clip(rng.beta(a=2.0, b=8.0), 0.0, 0.5))
    epsilon = float(np.round(epsilon, 2))  # log readability

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epsilon": epsilon,
    }

# ============================================================================
# #Circles computation from the CSV log (no re-scoring)
# ============================================================================

def greedy_circles(smiles_list, D: float = 0.7) -> int:
    """
    Greedy sphere-exclusion count (Diverse-Hits SI).
    Keep the first hit as a center; skip any subsequent hit within distance < D.
    Distance is 1 - Tanimoto(ECFP4).
    """
    fps = []
    keep = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ok = True
        for f in fps:
            sim = DataStructs.TanimotoSimilarity(fp, f)
            if (1.0 - sim) < D:
                ok = False
                break
        if ok:
            fps.append(fp)
            keep += 1
    return keep

def circles_from_log(log_csv: str, D: float = 0.7) -> int:
    """
    Compute #Circles from the scorer log only.
    Requires either an 'accepted' column, or enough fields to infer acceptance.
    """
    if not os.path.exists(log_csv):
        return 0
    df = pd.read_csv(log_csv)

    if "accepted" in df.columns:
        hits = df.loc[df["accepted"] == True, "smiles"].dropna().astype(str).tolist()
    else:
        cols = {c.lower() for c in df.columns}
        if {"smiles", "rf_prob", "prop_pass", "df_pass"}.issubset(cols):
            # Normalize column names in case of case differences
            rename = {c: c.lower() for c in df.columns}
            dfl = df.rename(columns=rename)
            mask = (dfl["rf_prob"] >= 0.5) & (dfl["prop_pass"] == 1) & (dfl["df_pass"] == 1)
            hits = dfl.loc[mask, "smiles"].astype(str).tolist()
        elif "reward" in df.columns and "rf_prob" in df.columns:
            hits = df[(df["reward"] > 0) & (df["rf_prob"] >= 0.5)]["smiles"].astype(str).tolist()
        else:
            # Cannot infer acceptance robustly; return 0 to avoid bias
            return 0

    return greedy_circles(hits, D=D)

# ============================================================================
# One training run (budget-faithful)
# ============================================================================

def train_one_run(target: str, td: dict,
                  time_budget: int | None,
                  sample_budget: int | None,
                  learning_rate: float,
                  batch_size: int,
                  epsilon: float,
                  run_dir: str,
                  seed: int = 0,
                  epochs: int = 10):
    """
    Execute one run under given budgets; write training_molecules.csv
    and metrics.json in run_dir. Returns a small dict with meta.
    """
    os.makedirs(run_dir, exist_ok=True)

    # Deterministic seeding
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load prior (Papyrus Graph Transformer) and vocab
    ds_path = BASE_DATASETS_PATH
    vocab = VocGraph.fromFile(os.path.join(ds_path, td["vocab_file"]))
    agent = GraphTransformer(voc_trg=vocab, use_gpus=[0] if device.type == "cuda" else [])
    agent.loadStatesFromFile(os.path.join(ds_path, td["model_pkg"]))
    agent.to(device)

    # Data and scorer
    train_ds = td["train_ds"]; test_ds = td["test_ds"]; sf_path = td["scoring_dir"]

    bh = BenchmarkScoringFunction(
        scoring_function_dir=sf_path,
        time_budget=time_budget,
        sample_budget=sample_budget,
        # use benchmark defaults for DF thresholds and property gates
        memory_distance_threshold=None,
        memory_score_threshold=None,
        memory_known_active_init=False,
        use_property_constraints=True,
        n_jobs=8,
        print_progress=False
    )
    bh.start_timer_and_reset()  # timer starts on first call internally

    # Budget-aware wrapper (assumed in your codebase)
    from divopt.scoring import ModelScorer
    ms = ModelScorer(bh, time_budget=time_budget, sample_budget=sample_budget)
    ms.reset_budgets()

    env = DrugExEnvironment(scorers=[ms], thresholds=[0.7], reward_scheme=SingleReward())

    explorer = FragGraphExplorer(agent=agent, env=env, mutate=agent,
                                 n_samples=2000, epsilon=epsilon, device=device)

    # Per-eval log (must include 'smiles' and 'accepted' if possible)
    explorer.train_log_path = os.path.join(run_dir, "training_molecules.csv")

    # Optimizer + warmup scheduler
    explorer.optim = ScheduledOptim(
        Adam(agent.parameters(), betas=(0.9, 0.98), eps=1e-9),
        learning_rate, 512
    )

    # Train until budgets stop the scorer; validation should not consume budget
    explorer.fit(
        train_loader=train_ds.asDataLoader(batch_size=batch_size),
        valid_loader=test_ds.asDataLoader(batch_size=batch_size),
        epochs=epochs,
        monitor=None
    )

    # Minimal metrics; #Circles is computed separately from the CSV
    results = {
        "target": target,
        "time_budget": time_budget,
        "sample_budget": sample_budget,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epsilon": float(epsilon),
        "seed": int(seed),
        "eval_count": int(getattr(ms, "eval_count", -1)),
        "elapsed_time": float(getattr(ms, "elapsed_time", -1.0)),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    return results

# ============================================================================
# Main: 15-trial random search, then 5 seeds
# ============================================================================

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if CLEAR_RESULTS:
        shutil.rmtree(RESULTS_ROOT, ignore_errors=True)
        os.makedirs(RESULTS_ROOT, exist_ok=True)

    constraints = {
        "time":   (600, None),     # 600 s wall time
        "sample": (None, 10000),   # 10,000 scorer calls
    }
    N_TRIALS = 15
    N_SEEDS  = 5

    rng_global = np.random.RandomState(1337)

    for tgt, td in TARGET_DATA.items():
        for cname, (tb, sb) in constraints.items():
            combo_root = os.path.join(RESULTS_ROOT, tgt, cname)
            os.makedirs(combo_root, exist_ok=True)

            # ---------------- Phase 1: random search (15 trials) ----------------
            trial_records = []
            for trial_idx in range(1, N_TRIALS + 1):
                cfg = sample_config(rng_global)
                trial_dir = os.path.join(combo_root, f"search_trial{trial_idx:02d}")
                logger.info(f"[search] {tgt} | {cname} | trial {trial_idx:02d} | cfg={cfg}")
                try:
                    _ = train_one_run(
                        target=tgt, td=td,
                        time_budget=tb, sample_budget=sb,
                        run_dir=trial_dir, seed=trial_idx,
                        **cfg
                    )
                    log_csv = os.path.join(trial_dir, "training_molecules.csv")
                    circles = circles_from_log(log_csv, D=0.7)
                    rec = {"trial": trial_idx, "circles": int(circles), **cfg}
                    trial_records.append(rec)
                    with open(os.path.join(trial_dir, "trial_summary.json"), "w") as fp:
                        json.dump(rec, fp, indent=2)
                except Exception as e:
                    # Keep going; write a marker for debugging
                    os.makedirs(trial_dir, exist_ok=True)
                    with open(os.path.join(trial_dir, "ERROR.txt"), "w") as fp:
                        fp.write(str(e))
                    continue

            # Select best configuration by #Circles (tie-breakers optional)
            if not trial_records:
                logger.warning(f"[warn] no successful trials for {tgt} | {cname}")
                continue
            df_trials = pd.DataFrame(trial_records).sort_values(
                by=["circles"], ascending=[False]
            )
            best = df_trials.iloc[0].to_dict()
            best_cfg = {k: best[k] for k in ("learning_rate", "batch_size", "epsilon")}
            with open(os.path.join(combo_root, "best_config.json"), "w") as fp:
                json.dump({"best": best, "trials": trial_records}, fp, indent=2)
            logger.info(f"[select] {tgt} | {cname} | best cfg = {best_cfg} (circles={best['circles']})")

            # ---------------- Phase 2: 5 independent seeds ----------------------
            for seed_id in range(1, N_SEEDS + 1):
                run_dir = os.path.join(combo_root, "final", f"seed{seed_id}")
                logger.info(f"[final]  {tgt} | {cname} | seed {seed_id} | cfg={best_cfg}")
                try:
                    _ = train_one_run(
                        target=tgt, td=td,
                        time_budget=tb, sample_budget=sb,
                        run_dir=run_dir, seed=1000 + seed_id,
                        **best_cfg
                    )
                    log_csv = os.path.join(run_dir, "training_molecules.csv")
                    circles = circles_from_log(log_csv, D=0.7)
                    with open(os.path.join(run_dir, "final_summary.json"), "w") as fp:
                        json.dump({"seed": seed_id, "circles": int(circles), **best_cfg}, fp, indent=2)
                except Exception as e:
                    os.makedirs(run_dir, exist_ok=True)
                    with open(os.path.join(run_dir, "ERROR.txt"), "w") as fp:
                        fp.write(str(e))
                    continue

if __name__ == "__main__":
    main()
