import os
import shutil
import torch
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Scoring / RL imports
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
    FragmentPairsSplitter,
)
from drugex.data.datasets import GraphFragDataSet
from ModelScorer import ModelScorer
from drugex.training.explorers import FragGraphExplorer
from torch.optim import Adam
from drugex.utils import ScheduledOptim

# ----------------------
# Basic project settings
# ----------------------

SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/reseed")

# config copied from original script
BASE_DATASETS_PATH = "diverse-hits/optimizers/drugex"
TARGET_CONFIG = {
    "drd2": {
        "frag_file": "drd2_fragbase.txt",
        "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir": "diverse-hits/data/scoring_functions/drd2",
    },
    "gsk3": {
        "frag_file": "gsk3_fragbase.txt",
        "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir": "diverse-hits/data/scoring_functions/gsk3",
    },
    "jnk3": {
        "frag_file": "jnk3_fragbase.txt",
        "model_pkg": "Papyrus05.5_graph_trans_PT.pkg",
        "vocab_file": "Papyrus05.5_graph_trans_PT.vocab",
        "scoring_dir": "diverse-hits/data/scoring_functions/jnk3",
    },
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ----------------------
# Simple hyperparam matrix — one combo per (target, constraint)
# We'll run 2 repeats for each of the 6 combos (3 targets × 2 constraints).
# ----------------------

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

CONSTRAINTS = {"time": (600, None), "sample": (None, 10000)}

# ----------------------
# Preprocessing / datasets
# ----------------------

def _base_encoded_dir(target: str) -> str:
    return os.path.join(RESULTS_ROOT, target, "preprocessing", "encoded")

def _combo_encoded_dir(target: str, constraint: str, combo: str) -> str:
    return os.path.join(RESULTS_ROOT, target, constraint, combo, "encoded")

def _seed_registry_path(target: str, constraint: str, combo: str) -> str:
    enc_dir = _combo_encoded_dir(target, constraint, combo)
    os.makedirs(enc_dir, exist_ok=True)
    return os.path.join(enc_dir, "seeds.smi")  # per-combo registry

def _load_seed_registry(target: str, constraint: str, combo: str) -> set:
    path = _seed_registry_path(target, constraint, combo)
    if os.path.exists(path):
        with open(path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def _append_seed_registry(target: str, constraint: str, combo: str, smiles_list):
    path = _seed_registry_path(target, constraint, combo)
    with open(path, "a") as f:
        for sm in smiles_list:
            f.write(sm + "\n")

def preprocess_target_base(target: str, config: dict, n_proc=4):
    """Create (or reuse) a BASE encoded train/test set for the target from the fragbase.
    We'll copy this base into each combo folder so reseeding stays isolated per combo.
    """
    enc_dir = _base_encoded_dir(target)
    os.makedirs(enc_dir, exist_ok=True)
    train_tsv = os.path.join(enc_dir, f"{target}_train.tsv")
    test_tsv  = os.path.join(enc_dir, f"{target}_test.tsv")

    if os.path.exists(train_tsv) and os.path.exists(test_tsv):
        train_ds = GraphFragDataSet(train_tsv, rewrite=False)
        test_ds  = GraphFragDataSet(test_tsv,  rewrite=False)
        return {
            "train_tsv": train_tsv,
            "test_tsv":  test_tsv,
            "vocab_file": config["vocab_file"],
            "model_pkg":  config["model_pkg"],
            "scoring_dir":config["scoring_dir"],
        }

    # Build base from fragbase (can replace with your own seed SMILES list)
    ds_path   = BASE_DATASETS_PATH
    frag_path = os.path.join(ds_path, config["frag_file"])
    """
    df = pd.read_csv(frag_path, header=None)
    smiles = list(dict.fromkeys(Standardization(n_proc=n_proc).apply(df.iloc[:, 0])))
    """

    smiles = []
    with open(frag_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split(",", 1)[0].split()[0]
            smiles.append(token)
    smiles = list(dict.fromkeys(Standardization(n_proc=n_proc).apply(smiles)))

    encoder = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4, 4, "brics"),
        encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
        pairs_splitter=FragmentPairsSplitter(0.1, len(smiles)),
        n_proc=n_proc,
    )

    train_ds = GraphFragDataSet(train_tsv, rewrite=True)
    test_ds  = GraphFragDataSet(test_tsv,  rewrite=True)
    encoder.apply(smiles, encodingCollectors=[test_ds, train_ds])

    return {
        "train_tsv": train_tsv,
        "test_tsv":  test_tsv,
        "vocab_file": config["vocab_file"],
        "model_pkg":  config["model_pkg"],
        "scoring_dir":config["scoring_dir"],
    }

# One-time base preprocessing per target
TARGET_BASE = {}
for tgt, conf in TARGET_CONFIG.items():
    logger.info(f"Preprocessing base for target {tgt}...")
    TARGET_BASE[tgt] = preprocess_target_base(tgt, conf)


def ensure_combo_datasets(target: str, constraint: str, combo: str):
    """Copy base encoded TSVs into this combo's encoded folder (once),
    then return GraphFragDataSet objects for this combo.
    """
    base = TARGET_BASE[target]
    enc_dir = _combo_encoded_dir(target, constraint, combo)
    os.makedirs(enc_dir, exist_ok=True)
    train_tsv = os.path.join(enc_dir, f"{target}_train.tsv")
    test_tsv  = os.path.join(enc_dir, f"{target}_test.tsv")

    if not (os.path.exists(train_tsv) and os.path.exists(test_tsv)):
        shutil.copy2(base["train_tsv"], train_tsv)
        shutil.copy2(base["test_tsv"],  test_tsv)

    train_ds = GraphFragDataSet(train_tsv, rewrite=False)
    test_ds  = GraphFragDataSet(test_tsv,  rewrite=False)
    return train_ds, test_ds

# ----------------------
# Seed updating (top hits -> seeds), per COMBO
# ----------------------

def _detect_smiles_column(df: pd.DataFrame) -> str | None:
    if "smiles" in df.columns:
        return "smiles"
    for c in df.columns:
        if "smile" in c.lower():
            return c
    return None

def _pick_score_column(df: pd.DataFrame) -> str | None:
    if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
        return "score"
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def add_top_hits_as_seeds(target: str, constraint: str, combo: str, run_dir: str,
                           topk: int = 100, threshold: float = 0.7) -> int:
    """Read the run's training log, take top-K (>= threshold), standardize & dedupe
    against this combo's seed registry, and append to this combo's train.tsv.
    """
    log_path = os.path.join(run_dir, "rs_training_molecules.csv")
    if not os.path.exists(log_path):
        logger.warning(f"No training log found: {log_path}")
        return 0

    df = pd.read_csv(log_path, comment="#", engine="python", on_bad_lines="skip")
    smiles_col = _detect_smiles_column(df)
    score_col  = _pick_score_column(df)
    if smiles_col is None or score_col is None:
        logger.warning("Missing smiles or numeric score column; skip seeding.")
        return 0

    hits = df.dropna(subset=[smiles_col]).copy()
    hits = hits[hits[score_col] >= threshold].sort_values(score_col, ascending=False)
    top  = hits.head(topk)
    if top.empty:
        return 0

    standardized = list(dict.fromkeys(Standardization(n_proc=4).apply(top[smiles_col].astype(str))))

    # Combo-level dedupe across epochs and runs
    already = _load_seed_registry(target, constraint, combo)
    new = [sm for sm in standardized if sm not in already]
    if not new:
        return 0

    enc = FragmentCorpusEncoder(
        fragmenter=Fragmenter(4, 4, "brics"),
        encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
        pairs_splitter=FragmentPairsSplitter(0.0, len(new)),
        n_proc=4,
    )
    enc_dir   = _combo_encoded_dir(target, constraint, combo)
    train_tsv = os.path.join(enc_dir, f"{target}_train.tsv")
    train_ds  = GraphFragDataSet(train_tsv, rewrite=False)
    enc.apply(new, encodingCollectors=[train_ds])

    _append_seed_registry(target, constraint, combo, new)
    return len(new)

# ----------------------
# One training run (per combo)
# ----------------------

def train_one_run(
    target: str,
    constraint: str,
    combo: str,
    time_budget,
    sample_budget,
    learning_rate,
    batch_size,
    epsilon,
    run_dir,
    epochs=10,
    seed_topk=100,
    seed_threshold=0.7,
):
    os.makedirs(run_dir, exist_ok=True)

    # reload agent per run
    base = TARGET_BASE[target]
    ds_path = BASE_DATASETS_PATH
    vocab = VocGraph.fromFile(os.path.join(ds_path, base["vocab_file"]))
    agent = GraphTransformer(voc_trg=vocab, use_gpus=[0])

    # scoring
    sf_path = base["scoring_dir"]
    bh = BenchmarkScoringFunction(
        scoring_function_dir=sf_path,
        time_budget=time_budget,
        sample_budget=sample_budget,
        memory_score_threshold=None,
        memory_distance_threshold=None,
        memory_known_active_init=False,
        use_property_constraints=True,
        n_jobs=8,
        print_progress=False,
    )
    bh.start_timer_and_reset()
    ms = ModelScorer(bh, time_budget=time_budget, sample_budget=sample_budget)
    ms.reset_budgets()

    # env
    env = DrugExEnvironment(scorers=[ms], thresholds=[0.7], reward_scheme=SingleReward())

    # datasets for THIS combo
    train_ds, test_ds = ensure_combo_datasets(target, constraint, combo)

    # explorer
    explorer = FragGraphExplorer(
        agent=agent, env=env, mutate=agent, n_samples=2000, epsilon=epsilon, device=device
    )
    explorer.train_log_path = os.path.join(run_dir, "rs_training_molecules.csv")
    explorer.optim = ScheduledOptim(Adam(agent.parameters(), betas=(0.9, 0.98), eps=1e-9), learning_rate, 512)

    # fit loop, reseed after each epoch (into THIS combo's dataset)
    for ep in range(epochs):
        explorer.fit(
            train_loader=train_ds.asDataLoader(batch_size=batch_size),
            valid_loader=test_ds.asDataLoader(batch_size=batch_size),
            epochs=1,
            monitor=None,
        )
        added = add_top_hits_as_seeds(target, constraint, combo, run_dir, topk=seed_topk, threshold=seed_threshold)
        logger.info(f"[{target} | {constraint} | {combo}] epoch {ep+1}: added {added} seeds")

    # minimal metrics
    metrics = {
        "best_value": getattr(explorer, "best_value", None),
        "eval_count": ms.eval_count,
        "elapsed_time": ms.elapsed_time,
    }
    pd.Series(metrics).to_json(os.path.join(run_dir, "metrics.json"))

# ----------------------
# Main: 2 runs for each of the 6 combos (3 targets × 2 constraints)
# ----------------------

def main():
    n_runs = 2
    for tgt in TARGET_CONFIG.keys():
        for cname, (tb, sb) in CONSTRAINTS.items():
            p = HYPERPARAMS[tgt][cname]
            lr, bs, eps = p["learning_rate"], p["batch_size"], p["epsilon"]
            combo = f"learning_rate{lr}_batch_size{bs}_epsilon{eps}"

            for run_id in range(1, n_runs + 1):
                run_dir = os.path.join(RESULTS_ROOT, tgt, cname, combo, f"run{run_id}")
                print(f"[{tgt} | {cname} | {combo} | run{run_id}]")
                try:
                    train_one_run(
                        target=tgt,
                        constraint=cname,
                        combo=combo,
                        time_budget=tb,
                        sample_budget=sb,
                        learning_rate=lr,
                        batch_size=bs,
                        epsilon=eps,
                        run_dir=run_dir,
                        epochs=10,
                        seed_topk=100,
                        seed_threshold=0.7,
                    )
                except Exception as e:
                    os.makedirs(run_dir, exist_ok=True)
                    with open(os.path.join(run_dir, "rs_training_molecules.csv"), "a") as log_f:
                        log_f.write(
                            f"# EXCEPTION: time_budget={tb}, sample_budget={sb}, error={e}"
                        )
                    continue

if __name__ == "__main__":
    main()
