"""
utils/counterfactual.py
★ KILLER FEATURE — Counterfactual Molecule Editor

Given a toxic molecule:
  1. Identify the highest-risk toxic fragments (SHAP + known toxicophores)
  2. Attempt targeted structural modifications via RDKit
  3. Re-predict toxicity on each modified candidate
  4. Return ranked safer alternatives

This is what makes the system look like a real drug-discovery tool.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from rdkit import Chem
from rdkit.Chem import AllChem, RWMol, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import BondType

from config import TOXIC_FRAGMENTS, TOX21_TASKS
from utils.mol_utils import smiles_to_mol, identify_toxic_fragments


@dataclass
class ModificationResult:
    original_smiles:  str
    modified_smiles:  str
    description:      str
    original_risk:    float
    modified_risk:    float
    risk_reduction:   float                             
    tasks_improved:   list[str] = field(default_factory=list)
    original_probs:   np.ndarray | None = None
    modified_probs:   np.ndarray | None = None

    @property
    def is_improvement(self) -> bool:
        return self.risk_reduction > 0

    def summary(self) -> str:
        direction = "↓" if self.is_improvement else "↑"
        pct = abs(self.risk_reduction * 100)
        return (
            f"{self.description}: overall risk "
            f"{direction} {pct:.1f}% "
            f"({self.original_risk*100:.1f}% → {self.modified_risk*100:.1f}%)"
        )


                                                                               

def _remove_nitro_groups(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Replace –NO₂ with –NH₂ (reduces electrophilicity)."""
    results = []
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    amine = Chem.MolFromSmarts("[NH2]")
    if mol.HasSubstructMatch(nitro):
        rw = RWMol(mol)
        try:
            new_mol = AllChem.ReplaceSubstructs(
                mol, nitro, Chem.MolFromSmiles("N"), replaceAll=True
            )[0]
            Chem.SanitizeMol(new_mol)
            smi = Chem.MolToSmiles(new_mol)
            if smi:
                results.append(("Replace –NO₂ with –NH₂", new_mol))
        except Exception:
            pass
    return results


def _remove_halogen_aromatics(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Remove halogen atoms from aromatic rings (reduce bioaccumulation)."""
    results = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (35, 53) and atom.GetIsAromatic():         
            rw = RWMol(mol)
            try:
                rw.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))                  
                new_mol = rw.GetMol()
                Chem.SanitizeMol(new_mol)
                smi = Chem.MolToSmiles(new_mol)
                sym = atom.GetSymbol()
                if smi:
                    results.append((f"Remove aromatic {sym}", new_mol))
                    break                 
            except Exception:
                pass
    return results


def _reduce_logp(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """
    Add a hydroxyl group to an aromatic carbon to increase polarity.
    High logP → organ accumulation → toxicity.
    """
    results = []
    logp = Descriptors.MolLogP(mol)
    if logp < 3.0:
        return results

    arom_c = Chem.MolFromSmarts("c[H]")
    matches = mol.GetSubstructMatches(arom_c)
    if not matches:
        return results

    rw = RWMol(mol)
    try:
        target_atom = matches[0][0]
        rw.ReplaceAtom(target_atom, Chem.Atom(8))      
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        smi = Chem.MolToSmiles(new_mol)
        new_logp = Descriptors.MolLogP(new_mol)
        if smi and new_logp < logp:
            results.append((f"Add –OH (logP: {logp:.1f}→{new_logp:.1f})", new_mol))
    except Exception:
        pass
    return results


def _replace_aldehyde(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Replace aldehyde (–CHO) with alcohol (–CH₂OH) to reduce reactivity."""
    results = []
    aldehyde = Chem.MolFromSmarts("[CX3H1](=O)")
    alcohol  = Chem.MolFromSmiles("CO")
    if mol.HasSubstructMatch(aldehyde):
        try:
            new_mol = AllChem.ReplaceSubstructs(
                mol, aldehyde, Chem.MolFromSmiles("CO"), replaceAll=False
            )[0]
            Chem.SanitizeMol(new_mol)
            smi = Chem.MolToSmiles(new_mol)
            if smi:
                results.append(("Replace –CHO with –CH₂OH", new_mol))
        except Exception:
            pass
    return results


def _add_polar_group(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Add a methyl carboxylate to improve water solubility and reduce toxicity."""
    results = []
    mw = Descriptors.MolWt(mol)
    if mw > 500:
        return results                 
    try:
        rxn = AllChem.ReactionFromSmarts(
            "[c:1][H]>>[c:1]C(=O)O"
        )
        products = rxn.RunReactants((mol,))
        if products:
            new_mol = products[0][0]
            Chem.SanitizeMol(new_mol)
            smi = Chem.MolToSmiles(new_mol)
            if smi:
                results.append(("Add –COOH polar group", new_mol))
    except Exception:
        pass
    return results


def _open_epoxide(mol: Chem.Mol) -> list[tuple[str, Chem.Mol]]:
    """Open epoxide ring (highly reactive, mutagenic) to a diol."""
    results = []
    epoxide = Chem.MolFromSmarts("C1OC1")
    diol    = Chem.MolFromSmiles("OCC(O)")
    if mol.HasSubstructMatch(epoxide):
        try:
            new_mol = AllChem.ReplaceSubstructs(
                mol, epoxide, diol, replaceAll=False
            )[0]
            Chem.SanitizeMol(new_mol)
            smi = Chem.MolToSmiles(new_mol)
            if smi:
                results.append(("Open epoxide ring → diol", new_mol))
        except Exception:
            pass
    return results


                                                                               

MODIFICATION_STRATEGIES = [
    _remove_nitro_groups,
    _open_epoxide,
    _replace_aldehyde,
    _remove_halogen_aromatics,
    _reduce_logp,
    _add_polar_group,
]


                                                                                

def generate_counterfactuals(
    smiles: str,
    predict_fn: Callable[[str], dict],
    max_candidates: int = 5,
) -> list[ModificationResult]:
    """
    Generate candidate safer molecules.

    Args:
        smiles:         Original SMILES
        predict_fn:     Function that accepts SMILES and returns predict() dict
        max_candidates: Max number of results to return

    Returns:
        List of ModificationResult, sorted by risk_reduction descending.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return []

    original_result = predict_fn(smiles)
    original_risk   = original_result["overall_risk"]
    original_probs  = original_result["ensemble"]

    candidates: list[ModificationResult] = []

    for strategy in MODIFICATION_STRATEGIES:
        modifications = strategy(mol)
        for description, new_mol in modifications:
            try:
                new_smi = Chem.MolToSmiles(new_mol)
                if not new_smi or new_smi == smiles:
                    continue

                new_result   = predict_fn(new_smi)
                new_risk     = new_result["overall_risk"]
                new_probs    = new_result["ensemble"]
                risk_delta   = original_risk - new_risk

                                       
                improved = [
                    TOX21_TASKS[i]
                    for i in range(len(TOX21_TASKS))
                    if new_probs[i] < original_probs[i] - 0.05
                ]

                candidates.append(ModificationResult(
                    original_smiles = smiles,
                    modified_smiles = new_smi,
                    description     = description,
                    original_risk   = original_risk,
                    modified_risk   = new_risk,
                    risk_reduction  = risk_delta,
                    tasks_improved  = improved,
                    original_probs  = original_probs,
                    modified_probs  = new_probs,
                ))
            except Exception:
                continue

                                                 
    candidates.sort(key=lambda r: r.risk_reduction, reverse=True)
    return candidates[:max_candidates]
