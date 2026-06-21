#!/usr/bin/env python3
"""
Compute validity and uniqueness from a scorer log CSV.

Expected CSV columns:
  smiles,reward,elapsed_time,eval_count,time_budget,sample_budget

By default, reports:
  - validity = (# valid molecules) / (# parsed rows)
  - uniqueness = (# unique canonical valid) / (# valid)

Optionally, with --hits-threshold, also reports uniqueness among hits:
  - uniqueness_hits = (# unique canonical valid hits) / (# valid hits)

Usage:
  python compute_validity_uniqueness.py path/to/log.csv
  python compute_validity_uniqueness.py log.csv --hits-threshold 0.5
"""

import argparse
import math
import sys
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except Exception as e:
    sys.stderr.write(
        "RDKit is required (e.g., conda install -c conda-forge rdkit)\n"
    )
    raise

def load_log(path: str) -> pd.DataFrame:
    # Read as strings to avoid dtype issues; we'll coerce numeric when needed
    df = pd.read_csv(path, dtype=str)
    # Normalize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]
    # Basic sanity: require 'smiles' column
    if "smiles" not in df.columns:
        raise ValueError("CSV must contain a 'smiles' column.")
    # Drop empty/NaN smiles
    df = df[~df["smiles"].isna()]
    df = df[df["smiles"].str.strip() != ""]
    # Drop EXCEPTION lines (often the last row)
    df = df[~df["smiles"].str.startswith("EXCEPTION:", na=False)]
    # Coerce reward if present
    if "reward" in df.columns:
        # Non-numeric rewards become NaN; that’s fine
        df["reward_num"] = pd.to_numeric(df["reward"], errors="coerce")
    else:
        df["reward_num"] = pd.NA
    return df.reset_index(drop=True)

def mol_from_smiles(smi: str):
    try:
        # Default RDKit sanitization; invalid SMILES -> None
        mol = Chem.MolFromSmiles(smi)
        return mol
    except Exception:
        return None

def canonical_isomeric_smiles(mol: rdchem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def compute_stats(df: pd.DataFrame, hits_threshold: float | None = None):
    total = len(df)
    # Build molecules (validity)
    mols = []
    valid_flags = []
    for smi in df["smiles"]:
        mol = mol_from_smiles(smi)
        mols.append(mol)
        valid_flags.append(mol is not None)
    df["valid"] = valid_flags
    valid_count = int(df["valid"].sum())
    validity = valid_count / total if total > 0 else float("nan")

    # Uniqueness among valid
    unique_set = set()
    for mol in (m for m in mols if m is not None):
        try:
            can = canonical_isomeric_smiles(mol)
        except Exception:
            # If canonicalization fails, treat as invalid
            continue
        unique_set.add(can)
    unique_valid = len(unique_set)
    uniqueness = unique_valid / valid_count if valid_count > 0 else float("nan")

    results = {
        "rows_total": total,
        "rows_after_cleaning": total,  # we cleaned in load_log
        "valid_count": valid_count,
        "validity": validity,
        "unique_valid": unique_valid,
        "uniqueness": uniqueness,
    }

    # Optional: uniqueness among hits (valid & reward >= threshold)
    if hits_threshold is not None and "reward_num" in df.columns:
        hits_mask = (df["valid"]) & (df["reward_num"].notna()) & (df["reward_num"] >= float(hits_threshold))
        df_hits = df[hits_mask].copy()
        mols_hits = [mol_from_smiles(s) for s in df_hits["smiles"]]
        unique_hits_set = set()
        valid_hits_count = 0
        for mol in mols_hits:
            if mol is None:
                continue
            valid_hits_count += 1
            try:
                can = canonical_isomeric_smiles(mol)
            except Exception:
                continue
            unique_hits_set.add(can)
        results.update({
            "valid_hits_count": valid_hits_count,
            "unique_valid_hits": len(unique_hits_set),
            "uniqueness_hits": (len(unique_hits_set) / valid_hits_count) if valid_hits_count > 0 else float("nan"),
            "hits_threshold": hits_threshold,
        })

    return results

def main():
    ap = argparse.ArgumentParser(description="Compute validity and uniqueness from scorer logs.")
    ap.add_argument("csv", help="Path to CSV (smiles,reward,elapsed_time,eval_count,time_budget,sample_budget)")
    ap.add_argument("--hits-threshold", type=float, default=None,
                    help="If set, also compute uniqueness among valid hits with reward >= threshold (e.g., 0.5).")
    args = ap.parse_args()

    df = load_log(args.csv)
    stats = compute_stats(df, hits_threshold=args.hits_threshold)

    # Pretty print
    print("=== Validity & Uniqueness ===")
    print(f"Rows (after cleaning):   {stats['rows_after_cleaning']}")
    print(f"Valid molecules:         {stats['valid_count']} "
          f"({stats['validity']*100:.2f}% of rows)")
    print(f"Unique (among valid):    {stats['unique_valid']} "
          f"({stats['uniqueness']*100:.2f}% of valid)")

    if "hits_threshold" in stats and stats["hits_threshold"] is not None:
        print(f"\n=== Among hits (reward >= {stats['hits_threshold']}) ===")
        print(f"Valid hits:              {stats.get('valid_hits_count', 0)}")
        uh = stats.get("uniqueness_hits", float('nan'))
        uvh = stats.get("unique_valid_hits", 0)
        if isinstance(uh, float) and not math.isnan(uh):
            print(f"Unique (valid hits):     {uvh} ({uh*100:.2f}% of valid hits)")
        else:
            print("Unique (valid hits):     n/a (no valid hits or reward column missing)")

if __name__ == "__main__":
    main()
