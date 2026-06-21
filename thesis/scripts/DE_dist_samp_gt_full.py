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

# ------------------------- Benchmark config -------------------------

CLEAR_RESULTS = True
N_TRIALS_PER_COMBO = 15          # Diverse-Hits used 15 HP trials
N_REPEATS_BEST = 5              # Re-run best config with 5 seeds

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

# Circles evaluation thresholds (RENAMED for clarity)
CIRCLES_HIT_THRESHOLD = 0.5           # Minimum reward score (S >= threshold)
CIRCLES_DISTANCE_THRESHOLD = 0.7      # Minimum Tanimoto distance (D >= threshold)

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

# ------------------------- Circles Evaluation -------------------------

def calculate_morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048):
    """
    Calculate Morgan fingerprint for a SMILES string.
    Returns RDKit ExplicitBitVect (not np.array).
    Returns None if molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        return fp
    except Exception:
        return None

def tanimoto_distance(fp1, fp2) -> float:
    """
    Calculate Tanimoto distance = 1 - Tanimoto similarity.
    Uses RDKit's DataStructs.TanimotoSimilarity directly.
    
    Args:
        fp1, fp2: RDKit ExplicitBitVect objects (or None)
    
    Returns:
        float: Tanimoto distance [0, 1]
    """
    if fp1 is None or fp2 is None:
        return 0.0
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 - similarity

def find_largest_diverse_subset(molecules: List[Dict], 
                               hit_threshold: float = CIRCLES_HIT_THRESHOLD,
                               distance_threshold: float = CIRCLES_DISTANCE_THRESHOLD) -> List[Dict]:
    """
    Find the largest subset of molecules (circles) where:
    - All molecules have reward >= hit_threshold
    - All pairwise distances >= distance_threshold (Tanimoto distance on Morgan FP)
    
    Uses MaxMin approximation algorithm:
    1. Filter hits (reward >= hit_threshold)
    2. Deduplicate by canonical SMILES
    3. Start with highest-reward molecule
    4. Iteratively add molecule farthest from all selected
    
    This approximates the maximum independent set better than simple greedy.
    
    Args:
        molecules: List of dicts with 'smiles' and 'reward' keys
        hit_threshold: Minimum reward score
        distance_threshold: Minimum pairwise distance
    
    Returns:
        List of selected molecule dicts (the "circles")
    """
    # Filter molecules meeting hit threshold
    hits = [m for m in molecules if m.get('reward', 0) >= hit_threshold]
    
    if not hits:
        return []
    
    # Deduplicate by canonical SMILES (avoid counting duplicates)
    seen_smiles = {}
    for m in hits:
        smiles = m.get('smiles')
        if smiles:
            try:
                canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                if canonical not in seen_smiles:
                    seen_smiles[canonical] = m
            except Exception:
                pass
    
    dedup_hits = list(seen_smiles.values())
    
    if not dedup_hits:
        return []
    
    # Pre-compute fingerprints
    fingerprints = {}
    for m in dedup_hits:
        smiles = m.get('smiles')
        if smiles:
            fp = calculate_morgan_fingerprint(smiles)
            fingerprints[smiles] = fp
    
    # MaxMin algorithm: start with highest reward, then add farthest point iteratively
    sorted_hits = sorted(dedup_hits, key=lambda x: x.get('reward', 0), reverse=True)
    
    selected = []
    remaining = sorted_hits.copy()
    
    # Start with highest reward
    if remaining:
        selected.append(remaining.pop(0))
    
    # Iteratively add molecule farthest from all selected
    while remaining:
        best_candidate = None
        best_min_dist = -1.0
        best_idx = -1
        
        for idx, candidate in enumerate(remaining):
            cand_smiles = candidate.get('smiles')
            if not cand_smiles or fingerprints.get(cand_smiles) is None:
                continue
            
            # Find minimum distance to any selected molecule
            min_dist_to_selected = float('inf')
            for selected_mol in selected:
                sel_smiles = selected_mol.get('smiles')
                if not sel_smiles or fingerprints.get(sel_smiles) is None:
                    continue
                
                dist = tanimoto_distance(fingerprints[cand_smiles], fingerprints[sel_smiles])
                min_dist_to_selected = min(min_dist_to_selected, dist)
            
            # Keep candidate with largest minimum distance
            if min_dist_to_selected > best_min_dist:
                best_min_dist = min_dist_to_selected
                best_candidate = candidate
                best_idx = idx
        
        # Add only if sufficiently distant from all selected
        if best_candidate is not None and best_min_dist >= distance_threshold:
            selected.append(best_candidate)
            remaining.pop(best_idx)
        else:
            # No more candidates meet distance threshold
            break
    
    return selected

def evaluate_circles_from_csv(csv_path: str) -> Dict[str, Any]:
    """
    Load molecules from CSV and calculate #CIRCLES count using optimal algorithm.
    """
    try:
        df = pd.read_csv(csv_path)
        
        molecules = []
        for _, row in df.iterrows():
            molecules.append({
                'smiles': row.get('smiles'),
                'reward': row.get('reward', 0)
            })
        
        # Find LARGEST diverse subset using branch-and-bound
        circles = find_largest_diverse_subset_optimal(molecules)
        
        unique_hits = len(set(
            Chem.MolToSmiles(Chem.MolFromSmiles(m['smiles']))
            for m in molecules if m['reward'] >= CIRCLES_HIT_THRESHOLD
            if m.get('smiles') and Chem.MolFromSmiles(m['smiles']) is not None
        ))
        
        return {
            'circles_count': len(circles),
            'total_molecules': len(molecules),
            'unique_hits': unique_hits,
            'hit_count_raw': len([m for m in molecules if m['reward'] >= CIRCLES_HIT_THRESHOLD])
        }
    except Exception as e:
        print(f"Error evaluating circles from {csv_path}: {e}")
        return {
            'circles_count': 0,
            'total_molecules': 0,
            'unique_hits': 0,
            'hit_count_raw': 0
        }

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
    molecule/score is appended to rs_training_molecules.csv.
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
    ms = LoggingScorer(bh, time_budget=time_budget, sample_budget=sample_budget, log_csv=log_csv)
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

    budget_hit = False
    budget_error_msg = None
    
    try:
        explorer.fit(
            train_loader=train_ds.asDataLoader(batch_size=batch_size),
            valid_loader=test_ds.asDataLoader(batch_size=batch_size),
            epochs=epochs,
            monitor=None
        )
    except (TimeoutError, RuntimeError) as e:
        # Budget was hit during training - this is expected and acceptable
        budget_hit = True
        budget_error_msg = str(e)
        print(f"    → Budget constraint triggered: {e}")

    # Evaluate circles from the logged CSV
    # (This works even if training was cut short by budget)
    circles_eval = evaluate_circles_from_csv(log_csv)

    results = {
        "#CIRCLES":           circles_eval['circles_count'],
        "best_value":         explorer.best_value,
        "eval_count":         ms.eval_count,
        "elapsed_time":       ms.elapsed_time,
        "seed":               seed,
        "learning_rate":      learning_rate,
        "batch_size":         batch_size,
        "epsilon":            epsilon,
        "total_molecules":    circles_eval['total_molecules'],
        "unique_hits":        circles_eval['unique_hits'],
        "hit_count_raw":      circles_eval['hit_count_raw'],
        "budget_hit":         budget_hit,
        "budget_error":       budget_error_msg
    }
    
    # Save metrics REGARDLESS of whether budget was hit
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
    Evaluate each trial by #CIRCLES metric and return the best configuration.
    """
    hp_root = os.path.join(base_out, tgt, "hyperparameter_search", constraint_name)
    ensure_dir(hp_root)

    trial_summaries = []
    for idx in range(1, num_trials + 1):
        hp = sample_hyperparams(rng)
        tag = f"{tgt}_{constraint_name}_trial{idx}_lr{hp['learning_rate']:.4g}_bs{hp['batch_size']}_eps{hp['epsilon']:.3f}"
        run_dir = os.path.join(hp_root, tag)
        print(f"[{now()}] Trial {idx}/{num_trials} :: {tag}")

        try:
            res = train_one_run(
                target=tgt, td=td,
                time_budget=time_budget, sample_budget=sample_budget,
                learning_rate=hp["learning_rate"], batch_size=hp["batch_size"], epsilon=hp["epsilon"],
                run_dir=run_dir, seed=idx
            )
            res["run_dir"] = run_dir
            trial_summaries.append(res)
            
            budget_status = " (budget hit)" if res.get("budget_hit") else " (completed)"
            print(f"    → #CIRCLES: {res['#CIRCLES']}, best_value: {res['best_value']:.3f}{budget_status}")
                
        except Exception as e:
            # Truly unexpected error (not budget-related)
            ensure_dir(run_dir)
            with open(os.path.join(run_dir, "error_log.txt"), "w") as log_f:
                log_f.write(f"UNEXPECTED ERROR:\n{type(e).__name__}: {e}\n")
            print(f"    ✗ Unexpected error: {type(e).__name__}: {e}")
            continue

    # pick best by highest #CIRCLES
    if trial_summaries:
        best = max(trial_summaries, key=lambda x: x["#CIRCLES"])
        budget_info = " (budget hit)" if best.get("budget_hit") else " (completed normally)"
        print(f"\n[{now()}] Best trial selected: #CIRCLES={best['#CIRCLES']}, best_value={best['best_value']:.3f}{budget_info}")
        print(f"  LR: {best['learning_rate']:.4g}, BS: {best['batch_size']}, Eps: {best['epsilon']:.3f}")
    else:
        best = None
        print(f"\n[{now()}] WARNING: No successful trials for {tgt}/{constraint_name}")

    # save summary of all trials
    with open(os.path.join(hp_root, "trial_summaries.json"), "w") as fp:
        json.dump(trial_summaries, fp, indent=2)

    return best

def repeat_best_config(best: dict, tgt: str, constraint_name: str, td: dict,
                       time_budget, sample_budget, base_out: str,
                       num_repeats: int):
    """
    Re-run best HPs with N independent seeds and store mean ± range.
    Naming convention: <tgt>_<constraint>_best_lr{...}_bs{...}_eps{...}_run<i>
    
    Selection is based on #CIRCLES (primary metric for diversity).
    """
    if best is None:
        return None

    repeat_root = os.path.join(base_out, tgt, f"best_variance_{constraint_name}")
    ensure_dir(repeat_root)

    lr = best["learning_rate"]
    bs = best["batch_size"]
    eps = best["epsilon"]
    seeds = list(range(1001, 1001 + num_repeats))

    repeat_results = []
    for i, seed in enumerate(seeds, 1):
        tag = f"{tgt}_{constraint_name}_best_lr{lr:.4g}_bs{bs}_eps{eps:.3f}_run{i}"
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
        print(f"    → #CIRCLES: {res['#CIRCLES']}, best_value: {res['best_value']:.3f}, unique_hits: {res['unique_hits']}")

    # compute mean ± range for #CIRCLES (PRIMARY METRIC)
    circles_counts = [r["#CIRCLES"] for r in repeat_results]
    best_values = [r["best_value"] for r in repeat_results]
    
    mean_circles = float(np.mean(circles_counts))
    rng_circles  = float(np.max(circles_counts) - np.min(circles_counts))
    mean_best = float(np.mean(best_values))
    rng_best  = float(np.max(best_values) - np.min(best_values))
    
    summary = {
        "learning_rate": lr,
        "batch_size": bs,
        "epsilon": eps,
        "seeds": seeds,
        "primary_metric": "#CIRCLES",
        "algorithm": "MaxMin approximation",
        "circles": {
            "mean": mean_circles,
            "range": rng_circles,
            "min": float(np.min(circles_counts)),
            "max": float(np.max(circles_counts)),
            "values": [int(c) for c in circles_counts]
        },
        "best_value": {
            "mean": mean_best,
            "range": rng_best,
            "min": float(np.min(best_values)),
            "max": float(np.max(best_values)),
            "values": [float(b) for b in best_values]
        },
        "repeats": repeat_results
    }
    with open(os.path.join(repeat_root, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    
    print(f"\n[{now()}] Best config summary:")
    print(f"  #CIRCLES: mean={mean_circles:.1f} ± {rng_circles:.1f} (range [{np.min(circles_counts)}, {np.max(circles_counts)}])")
    print(f"  Best value: mean={mean_best:.3f} ± {rng_best:.3f}")
    return summary

def find_largest_diverse_subset_optimal(molecules: List[Dict], 
                                       hit_threshold: float = CIRCLES_HIT_THRESHOLD,
                                       distance_threshold: float = CIRCLES_DISTANCE_THRESHOLD) -> List[Dict]:
    """
    Find the LARGEST subset of molecules (circles) where:
    - All molecules have reward >= hit_threshold
    - All pairwise distances >= distance_threshold
    
    Uses branch-and-bound with pruning for small sets (n <= 30).
    Falls back to greedy for larger sets (more practical).
    
    Args:
        molecules: List of dicts with 'smiles' and 'reward' keys
        hit_threshold: Minimum reward score
        distance_threshold: Minimum pairwise distance
    
    Returns:
        List of selected molecule dicts (the "circles")
    """
    # Filter molecules meeting hit threshold
    hits = [m for m in molecules if m.get('reward', 0) >= hit_threshold]
    
    if not hits:
        return []
    
    # Deduplicate by canonical SMILES
    seen_smiles = {}
    for m in hits:
        smiles = m.get('smiles')
        if smiles:
            try:
                canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                if canonical not in seen_smiles:
                    seen_smiles[canonical] = m
            except Exception:
                pass
    
    dedup_hits = list(seen_smiles.values())
    
    if not dedup_hits:
        return []
    
    # Pre-compute fingerprints
    fingerprints = {}
    for m in dedup_hits:
        smiles = m.get('smiles')
        if smiles:
            fp = calculate_morgan_fingerprint(smiles)
            fingerprints[smiles] = fp
    
    n = len(dedup_hits)
    
    # For very small sets, use exact branch-and-bound
    # For larger sets, use greedy (much faster and reasonable approximation)
    if n <= 30:
        # Build distance matrix only if we need it
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                smiles_i = dedup_hits[i].get('smiles')
                smiles_j = dedup_hits[j].get('smiles')
                
                if fingerprints.get(smiles_i) is None or fingerprints.get(smiles_j) is None:
                    distance_matrix[i][j] = 0.0
                    distance_matrix[j][i] = 0.0
                else:
                    dist = tanimoto_distance(fingerprints[smiles_i], fingerprints[smiles_j])
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist
        
        best_subset = find_maximum_independent_set(dedup_hits, distance_matrix, distance_threshold)
        return best_subset
    else:
        # Use greedy for n > 30 (much faster, reasonable quality)
        if n > 500:
            print(f"  Note: {n} unique hits; using greedy approximation (exact algorithm too slow)")
        return find_largest_diverse_subset_greedy(dedup_hits, fingerprints, distance_threshold)

def find_maximum_independent_set(molecules: List[Dict], 
                                  distance_matrix: np.ndarray,
                                  distance_threshold: float) -> List[Dict]:
    """
    Branch-and-bound algorithm to find the maximum independent set.
    An "independent set" here = all pairwise distances >= threshold.
    
    Uses pruning: if current + upper bound <= best found, skip branch.
    """
    n = len(molecules)
    best_size = 0
    best_set = []
    
    def is_compatible(selected_indices: List[int], new_idx: int) -> bool:
        """Check if new_idx is far enough from all selected molecules."""
        for sel_idx in selected_indices:
            if distance_matrix[new_idx][sel_idx] < distance_threshold:
                return False
        return True
    
    def branch_and_bound(selected_indices: List[int], remaining_indices: List[int], depth: int = 0):
        nonlocal best_size, best_set
        
        # Pruning: if we can't beat current best even if we take all remaining, skip
        upper_bound = len(selected_indices) + len(remaining_indices)
        if upper_bound <= best_size:
            return
        
        # Base case: no more candidates
        if not remaining_indices:
            if len(selected_indices) > best_size:
                best_size = len(selected_indices)
                best_set = selected_indices.copy()
            return
        
        # Try adding each remaining molecule
        for i in range(len(remaining_indices)):
            idx = remaining_indices[i]
            
            if is_compatible(selected_indices, idx):
                # Add this molecule and recurse
                new_selected = selected_indices + [idx]
                new_remaining = remaining_indices[i + 1:]  # Only consider molecules after this one
                branch_and_bound(new_selected, new_remaining, depth + 1)
    
    branch_and_bound([], list(range(n)))
    return [molecules[i] for i in best_set]

def find_largest_diverse_subset_greedy(molecules: List[Dict],
                                      fingerprints: Dict,
                                      distance_threshold: float) -> List[Dict]:
    """
    Greedy MaxMin fallback for very large sets.
    Start with highest reward, then add farthest point iteratively.
    """
    sorted_hits = sorted(molecules, key=lambda x: x.get('reward', 0), reverse=True)
    
    selected = []
    remaining = sorted_hits.copy()
    
    if remaining:
        selected.append(remaining.pop(0))
    
    while remaining:
        best_candidate = None
        best_min_dist = -1.0
        best_idx = -1
        
        for idx, candidate in enumerate(remaining):
            cand_smiles = candidate.get('smiles')
            if not cand_smiles or fingerprints.get(cand_smiles) is None:
                continue
            
            min_dist_to_selected = float('inf')
            for selected_mol in selected:
                sel_smiles = selected_mol.get('smiles')
                if not sel_smiles or fingerprints.get(sel_smiles) is None:
                    continue
                
                dist = tanimoto_distance(fingerprints[cand_smiles], fingerprints[sel_smiles])
                min_dist_to_selected = min(min_dist_to_selected, dist)
            
            if min_dist_to_selected > best_min_dist:
                best_min_dist = min_dist_to_selected
                best_candidate = candidate
                best_idx = idx
        
        if best_candidate is not None and best_min_dist >= distance_threshold:
            selected.append(best_candidate)
            remaining.pop(best_idx)
        else:
            break
    
    return selected

def main():
    if CLEAR_RESULTS:
        shutil.rmtree(RESULTS_ROOT, ignore_errors=True)
        
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # global RNG for HP sampling
    rng = np.random.default_rng(seed=0)
    TARGET_DATA: Dict[str, Dict[str, Any]] = {t: preprocess_target(t, c) for t, c in TARGET_CONFIG.items()}
    for tgt, td in TARGET_DATA.items():
        for cname, lims in CONSTRAINTS.items():
            print(f"\n{'='*60}")
            print(f"=== {tgt.upper()} | constraint={cname} ===")
            print(f"{'='*60}")

            # 1) HP trials (distribution sampling) - pick best by #CIRCLES
            best = run_trials_for_combo(
                tgt=tgt, td=td, constraint_name=cname,
                time_budget=lims["time_budget"], sample_budget=lims["sample_budget"],
                base_out=RESULTS_ROOT, num_trials=N_TRIALS_PER_COMBO, rng=rng
            )

            # 2) Re-run best with N seeds; measure variance in #CIRCLES
            _ = repeat_best_config(
                best=best, tgt=tgt, constraint_name=cname, td=td,
                time_budget=lims["time_budget"], sample_budget=lims["sample_budget"],
                base_out=RESULTS_ROOT, num_repeats=N_REPEATS_BEST
            )

if __name__ == "__main__":
    main()