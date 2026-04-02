from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_CACHE_PATH = Path(__file__).parent.parent / "data" / "train_fps_cache.pkl"
_TOX21_CSV = Path(__file__).parent.parent / "tox21.csv"

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

                                                                                 

def _build_cache() -> dict:
    """
    Parse tox21.csv and compute ECFP fingerprints for all valid compounds.
    Cached to data/train_fps_cache.pkl for fast subsequent lookups.
    """
    from config import TOX21_TASKS
    df = pd.read_csv(_TOX21_CSV).dropna(subset=["smiles"]).reset_index(drop=True)

    smiles_list   = df["smiles"].tolist()
    label_matrix  = df[TOX21_TASKS].fillna(-1).values                

    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _morgan_gen.GetFingerprint(mol)
        fps.append(fp)
        valid_idx.append(i)

    cache = {
        "fps":          fps,                                                          
        "smiles":       [smiles_list[i] for i in valid_idx],
        "labels":       label_matrix[valid_idx],                
    }
    os.makedirs(_CACHE_PATH.parent, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    return cache


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return _build_cache()


                                                                                

def find_similar_compounds(
    query_smiles: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Find the top-k most similar training compounds by Tanimoto similarity.

    Returns a list of dicts with keys:
        smiles       : str
        similarity   : float  (0–1)
        labels       : dict   {task: label_value}  (-1 = not measured)
    """
    from config import TOX21_TASKS

    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        return []

    query_fp = _morgan_gen.GetFingerprint(mol)
    cache    = _load_cache()
    db_fps   = cache["fps"]
    db_smi   = cache["smiles"]
    db_lbl   = cache["labels"]

                    
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, db_fps)
    sims = np.array(sims, dtype=np.float32)

    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_idx:
        label_dict = {}
        for j, task in enumerate(TOX21_TASKS):
            v = db_lbl[i, j]
            label_dict[task] = int(v) if v >= 0 else None                       
        results.append({
            "smiles":     db_smi[i],
            "similarity": float(sims[i]),
            "labels":     label_dict,
        })
    return results


def rebuild_cache() -> None:
    """Force-rebuild the fingerprint cache (call after re-training)."""
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()
    _build_cache()
