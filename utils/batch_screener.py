from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable

from config import TOX21_TASKS, HIGH_RISK, MEDIUM_RISK


def parse_smiles_block(text: str) -> list[str]:
   
    seen, out = set(), []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
                                                                    
        parts = line.split()
        smi = parts[-1]                                           
        if smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def screen_batch(
    smiles_list: list[str],
    predict_fn: Callable[[str], dict],
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Run ensemble prediction on each SMILES.

    Args:
        smiles_list:       List of SMILES strings
        predict_fn:        predictor.predict  (returns dict with 'ensemble', 'overall_risk')
        progress_callback: Optional callback(done, total) for Streamlit progress bar

    Returns:
        DataFrame sorted by overall_risk descending with columns:
            SMILES | Overall Risk | Risk Level | NR-AR | NR-AR-LBD | … (12 endpoint cols)
    """
    from utils.mol_utils import validate_smiles

    rows = []
    total = len(smiles_list)

    for i, smi in enumerate(smiles_list):
        if progress_callback:
            progress_callback(i, total)

        valid, err = validate_smiles(smi)
        if not valid:
            row = {
                "SMILES":       smi,
                "Status":       f"❌ Invalid: {err}",
                "Overall Risk": np.nan,
                "Risk Level":   "Invalid",
            }
            for task in TOX21_TASKS:
                row[task] = np.nan
            rows.append(row)
            continue

        try:
            result    = predict_fn(smi)
            ensemble  = result["ensemble"]
            overall   = result["overall_risk"]
            risk_lvl  = (
                "🔴 High"   if overall >= HIGH_RISK   else
                "🟡 Medium" if overall >= MEDIUM_RISK else
                "🟢 Low"
            )

            row = {
                "SMILES":       smi,
                "Status":       "✅ OK",
                "Overall Risk": round(overall * 100, 1),
                "Risk Level":   risk_lvl,
            }
            for j, task in enumerate(TOX21_TASKS):
                row[task] = round(float(ensemble[j]) * 100, 1)
            rows.append(row)

        except Exception as e:
            row = {
                "SMILES":       smi,
                "Status":       f"⚠️ Error: {e}",
                "Overall Risk": np.nan,
                "Risk Level":   "Error",
            }
            for task in TOX21_TASKS:
                row[task] = np.nan
            rows.append(row)

    if progress_callback:
        progress_callback(total, total)

    df = pd.DataFrame(rows)
                                                                 
    valid_rows   = df[df["Status"] == "✅ OK"].sort_values("Overall Risk", ascending=False)
    invalid_rows = df[df["Status"] != "✅ OK"]
    return pd.concat([valid_rows, invalid_rows], ignore_index=True)


def dataframe_to_csv(df: pd.DataFrame) -> str:
    """Return CSV string for Streamlit download_button."""
    return df.to_csv(index=False)
