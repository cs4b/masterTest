"""
DEBUG VERSION: Single DRD2 sample trial with extensive reseeding debug output
Based on DE_dist_samp_gt_expreplay.py
"""
import os
import shutil
import json
import math
import random
import itertools
from datetime import datetime
from typing import Dict, Any, List
import time

import numpy as np
import pandas as pd
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

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

# =====================================================================
# CONFIG - SINGLE TRIAL DEBUG
# =====================================================================

SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT  = os.path.join(PROJECT_ROOT, "results/debug_reseed")

BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'

TARGET_CONFIG = {
    "drd2": {
        "frag_file": "drd2_fragbase.txt",
        "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir": "diverse-hits/data/scoring_functions/drd2"
    }
}

CONSTRAINT = {"time_budget": None, "sample_budget": 10_000}

# Reseeding config
RESEED_EVERY_EPOCHS = 1
RESEED_THRESHOLD = 0.7

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# =====================================================================
# UTILITIES
# =====================================================================

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

# =====================================================================
# RESEEDING HELPERS (WITH DEBUG OUTPUT)
# =====================================================================

def _canonicalize_smiles(smiles_list):
    """Return canonical RDKit SMILES; filter out invalid/None."""
    print(f"    [_canonicalize_smiles] Input: {len(smiles_list)} SMILES")
    canon = []
    failed = 0
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi:
            failed += 1
            continue
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                failed += 1
                continue
            canon.append(Chem.MolToSmiles(m))
        except Exception as e:
            print(f"      Exception canonicalizing {smi}: {e}")
            failed += 1
            continue
    
    print(f"    [_canonicalize_smiles] Output: {len(canon)} valid SMILES ({failed} failed)")
    
    seen = set()
    out = []
    for s in canon:
        if s not in seen:
            out.append(s)
            seen.add(s)
    
    print(f"    [_canonicalize_smiles] Deduplicated: {len(out)} unique SMILES")
    return out

def _load_high_scoring_novel(log_csv, threshold, already_seeded_set, epoch_num=None):
    """Read CSV, filter reward>=threshold, deduplicate, return novel molecules."""
    epoch_str = f"(epoch {epoch_num})" if epoch_num else ""
    print(f"    [_load_high_scoring_novel {epoch_str}] Loading from {log_csv}")
    
    if not os.path.exists(log_csv):
        print(f"    [_load_high_scoring_novel {epoch_str}] CSV not found!")
        return []
    
    try:
        df = pd.read_csv(log_csv)
        print(f"    [_load_high_scoring_novel {epoch_str}] Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"    [_load_high_scoring_novel {epoch_str}] Failed to load CSV: {e}")
        return []
    
    # Check raw reward distribution
    print(f"    [_load_high_scoring_novel {epoch_str}] Reward stats:")
    print(f"      min={df['reward'].min():.4f}, mean={df['reward'].mean():.4f}, max={df['reward'].max():.4f}")
    above_threshold = len(df[df['reward'] >= threshold])
    print(f"      Molecules >= {threshold}: {above_threshold} / {len(df)}")
    
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df_filtered = df[df["reward"] >= threshold]
    print(f"    [_load_high_scoring_novel {epoch_str}] After filtering: {len(df_filtered)} molecules")
    
    smiles_raw = df_filtered["smiles"].tolist()
    print(f"    [_load_high_scoring_novel {epoch_str}] Got {len(smiles_raw)} SMILES to canonicalize")
    
    smiles = _canonicalize_smiles(smiles_raw)
    print(f"    [_load_high_scoring_novel {epoch_str}] Canonical SMILES: {len(smiles)}")
    print(f"    [_load_high_scoring_novel {epoch_str}] Already seeded: {len(already_seeded_set)}")
    
    novel = [s for s in smiles if s not in already_seeded_set]
    print(f"    [_load_high_scoring_novel {epoch_str}] Novel molecules: {len(novel)}")
    
    if len(novel) > 0:
        print(f"    [_load_high_scoring_novel {epoch_str}] First 3 novel: {novel[:3]}")
    
    return novel

def _append_smiles_to_train(smiles, train_file, n_proc=4):
    """Append SMILES to training dataset."""
    print(f"    [_append_smiles_to_train] Adding {len(smiles)} molecules to {train_file}")
    
    if not smiles:
        return 0, GraphFragDataSet(train_file, rewrite=False)

    std_smiles = list(Standardization(n_proc=n_proc).apply(pd.Series(smiles)))
    print(f"    [_append_smiles_to_train] After standardization: {len(std_smiles)} molecules")

    m = len(std_smiles)
    eff_n_proc = max(1, min(n_proc, m))
    eff_chunk  = max(1, m // eff_n_proc)

    encoder = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4,4,'brics'),
        encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
        pairs_splitter=None,
        n_proc=n_proc,
        chunk_size=eff_chunk
    )

    train_ds = GraphFragDataSet(train_file, rewrite=False)
    encoder.apply(std_smiles, encodingCollectors=[train_ds])

    new_train_ds = GraphFragDataSet(train_file, rewrite=False)
    print(f"    [_append_smiles_to_train] Encoding and dataset reload complete")
    return len(std_smiles), new_train_ds

# =====================================================================
# LOGGING SCORER
# =====================================================================

class LoggingScorer(ModelScorer):
    """Wraps ModelScorer: enforces budgets AND logs every molecule/score to CSV."""
    def __init__(self, bh, time_budget, sample_budget, log_csv):
        super().__init__(bh, time_budget=time_budget, sample_budget=sample_budget)
        self.log_csv = log_csv
        os.makedirs(os.path.dirname(self.log_csv), exist_ok=True)
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w") as f:
                f.write("smiles,reward,elapsed_time,eval_count,time_budget,sample_budget\n")
        self._tb = time_budget
        self._sb = sample_budget
        self._start = None
        self.eval_count = 0
        self.elapsed_time = 0.0

    def getScores(self, mols, frags=None):
        if self._start is None:
            self._start = time.time()
        
        smiles = []
        for m in mols:
            try:
                smiles.append(Chem.MolToSmiles(m) if m is not None else None)
            except Exception:
                smiles.append(None)
        
        rewards = self.dh_scorer(smiles)
        self.eval_count += len(rewards)
        self.elapsed_time = time.time() - self._start
        
        try:
            with open(self.log_csv, "a") as f:
                for smi, r in zip(smiles, rewards):
                    f.write(f"{smi},{r},{self.elapsed_time},{self.eval_count},{self._tb},{self._sb}\n")
        except Exception:
            pass
        
        if self._sb is not None and self.eval_count >= self._sb:
            raise RuntimeError(f"Sample budget of {self._sb} reached ({self.eval_count} molecules scored)")
        if self._tb is not None and self.elapsed_time >= self._tb:
            raise TimeoutError(f"Time budget of {self._tb}s reached ({self.elapsed_time:.2f}s)")
        
        return rewards

    def reset_budgets(self):
        self._start = None
        self.eval_count = 0
        self.elapsed_time = 0.0

# =====================================================================
# PREPROCESSING
# =====================================================================

def preprocess_target(target: str, config: dict, n_proc=4):
    """Preprocess target: standardize SMILES, fragment, split into train/test."""
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

# =====================================================================
# SINGLE DEBUG RUN
# =====================================================================

def train_debug_run(target: str, td: dict, time_budget, sample_budget,
                    learning_rate: float, batch_size: int, epsilon: float,
                    run_dir: str, seed: int, epochs: int = 10) -> Dict[str, Any]:
    """Single run with extensive reseeding debug output."""
    ensure_dir(run_dir)
    set_seed(seed)
    
    print(f"\n{'='*80}")
    print(f"DEBUG RUN: {target.upper()} | LR={learning_rate:.4g}, BS={batch_size}, Eps={epsilon:.3f}")
    print(f"{'='*80}")

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

    # Scoring function
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
    print(f"\n[{now()}] Initializing LoggingScorer → {log_csv}")
    ms = LoggingScorer(bh, time_budget=time_budget, sample_budget=sample_budget, log_csv=log_csv)
    ms.reset_budgets()

    # Environment
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

    # Reseeding bookkeeping
    seen_file = os.path.join(run_dir, "reseeding_seen.txt")
    already_seeded = set()
    if os.path.exists(seen_file):
        with open(seen_file, "r") as f:
            for line in f:
                smi = line.strip()
                if smi:
                    already_seeded.add(smi)

    budget_hit = False
    budget_error_msg = None
    epochs_done = 0

    # =====================================================================
    # TRAINING LOOP WITH DEBUG OUTPUT
    # =====================================================================
    print(f"\n[{now()}] Starting training loop...")
    print(f"  RESEED_EVERY_EPOCHS = {RESEED_EVERY_EPOCHS}")
    print(f"  RESEED_THRESHOLD = {RESEED_THRESHOLD}")
    print(f"  Epochs to run: {epochs}")
    
    while epochs_done < epochs:
        step = min(RESEED_EVERY_EPOCHS, epochs - epochs_done)
        epoch_num = epochs_done // RESEED_EVERY_EPOCHS + 1
        
        print(f"\n[{now()}] ⬤ EPOCH {epoch_num} (step={step}, epochs_done={epochs_done})")
        print(f"  Before fit: eval_count={ms.eval_count}, elapsed_time={ms.elapsed_time:.1f}s")
        
        try:
            print(f"  → Calling explorer.fit(epochs={step})...")
            explorer.fit(
                train_loader=train_ds.asDataLoader(batch_size=batch_size),
                valid_loader=test_ds.asDataLoader(batch_size=batch_size),
                epochs=step,
                monitor=None
            )
            print(f"  ✓ explorer.fit() completed successfully")
        except (TimeoutError, RuntimeError) as e:
            budget_hit = True
            budget_error_msg = str(e)
            print(f"  ✗ EXCEPTION in explorer.fit(): {type(e).__name__}")
            print(f"    Message: {e}")
            print(f"  → Breaking training loop")
            break
        
        epochs_done += step
        print(f"  After fit: eval_count={ms.eval_count}, elapsed_time={ms.elapsed_time:.1f}s")

        # ===== RESEEDING LOGIC =====
        print(f"\n  ▶ RESEEDING CHECK (epoch {epoch_num})")
        print(f"    CSV file: {log_csv}")
        print(f"    CSV exists: {os.path.exists(log_csv)}")
        if os.path.exists(log_csv):
            print(f"    CSV size: {os.path.getsize(log_csv)} bytes")
        
        novel = _load_high_scoring_novel(log_csv, threshold=RESEED_THRESHOLD, 
                                         already_seeded_set=already_seeded, epoch_num=epoch_num)
        
        if novel:
            print(f"    ✓ Found {len(novel)} novel molecules! Adding to training...")
            added_count, new_train_ds = _append_smiles_to_train(novel, train_file)
            train_ds = new_train_ds
            already_seeded.update(novel)
            
            print(f"    ✓ Writing {len(novel)} SMILES to {seen_file}")
            with open(seen_file, "a") as f:
                for smi in novel:
                    f.write(smi + "\n")
            print(f"    ✓ Reseeding complete! File now has {len(already_seeded)} total SMILES")
        else:
            print(f"    ✗ NO novel molecules found above threshold {RESEED_THRESHOLD}")
            print(f"       _load_high_scoring_novel returned empty list")

    # =====================================================================
    # RESULTS
    # =====================================================================
    print(f"\n[{now()}] Training loop ended")
    print(f"  Total epochs: {epochs_done}")
    print(f"  Budget hit: {budget_hit}")
    if budget_error_msg:
        print(f"  Budget error: {budget_error_msg}")
    print(f"  Final eval_count: {ms.eval_count}")
    print(f"  Final elapsed_time: {ms.elapsed_time:.1f}s")
    print(f"  Final already_seeded size: {len(already_seeded)}")
    print(f"  Reseeding file exists: {os.path.exists(seen_file)}")
    if os.path.exists(seen_file):
        with open(seen_file) as f:
            lines = f.readlines()
        print(f"  Reseeding file has {len(lines)} lines")

    results = {
        "eval_count":         ms.eval_count,
        "elapsed_time":       ms.elapsed_time,
        "seed":               seed,
        "learning_rate":      learning_rate,
        "batch_size":         batch_size,
        "epsilon":            epsilon,
        "budget_hit":         budget_hit,
        "budget_error":       budget_error_msg,
        "total_molecules_reseeded": len(already_seeded)
    }
    
    with open(os.path.join(run_dir, "debug_metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    
    print(f"\n✓ Debug metrics saved to {os.path.join(run_dir, 'debug_metrics.json')}")
    return results

def main():
    # Clear results
    shutil.rmtree(RESULTS_ROOT, ignore_errors=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Preprocess
    print(f"\n[{now()}] Preprocessing DRD2...")
    target_data = preprocess_target("drd2", TARGET_CONFIG["drd2"])

    # Sample one hyperparameter with seed=0 (should match expreplay.py)
    print(f"\n[{now()}] Sampling hyperparameters with seed=0...")
    rng = np.random.default_rng(seed=0)
    
    lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(1.0))))
    bs = int(rng.integers(64, 513))
    bs = max(64, min(512, bs - (bs % 16)))
    eps = float(rng.uniform(0.05, 0.5))
    
    print(f"  Sampled: LR={lr:.4g}, BS={bs}, Eps={eps:.3f}")
    
    # Single debug run
    run_dir = os.path.join(RESULTS_ROOT, "drd2", "sample", f"debug_trial_lr{lr:.4g}_bs{bs}_eps{eps:.3f}")
    
    train_debug_run(
        target="drd2",
        td=target_data,
        time_budget=CONSTRAINT["time_budget"],
        sample_budget=CONSTRAINT["sample_budget"],
        learning_rate=lr,
        batch_size=bs,
        epsilon=eps,
        run_dir=run_dir,
        seed=1,
        epochs=10
    )
    
    print(f"\n{'='*80}")
    print(f"DEBUG RUN COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {run_dir}")
    print(f"\nCompare with:")
    print(f"  thesis/results/reseed/drd2/hyperparameter_search/sample/")

if __name__ == "__main__":
    main()
