from __future__ import annotations
import os, pickle, warnings
import numpy as np
import torch
import joblib

warnings.filterwarnings("ignore")

from config import (
    TOX21_TASKS, GNN_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH,
    GNN_WEIGHT, XGB_WEIGHT,
)
from models.gnn_model import ToxicityGATv2, smiles_to_graph, smiles_to_stacked_fp
from utils.mol_utils import smiles_to_mol


class ToxicityPredictor:
  
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model  = None
        self.xgb_models = None
        self.scaler     = None
        self._load_models()

                                                                               

    def _load_models(self):
        """Load all saved models. Raises if files don't exist."""
        if not os.path.exists(GNN_MODEL_PATH):
            raise FileNotFoundError(
                f"GNN model not found at {GNN_MODEL_PATH}. Run train.py first."
            )
        if not os.path.exists(XGB_MODEL_PATH):
            raise FileNotFoundError(
                f"XGB model not found at {XGB_MODEL_PATH}. Run train.py first."
            )

             
        ckpt = torch.load(GNN_MODEL_PATH, map_location=self.device)
        self.gnn_model = ToxicityGATv2(
            in_channels=ckpt["in_channels"],
            num_tasks=ckpt["num_tasks"],
        ).to(self.device)
        self.gnn_model.load_state_dict(ckpt["model_state"])
        self.gnn_model.eval()

                 
        self.xgb_models = joblib.load(XGB_MODEL_PATH)

                
        self.scaler = joblib.load(SCALER_PATH)

                                                                               

    def _predict_gnn(self, smiles: str) -> np.ndarray | None:
        """Run single SMILES through GNN. Returns (12,) or None."""
        try:
            graph = smiles_to_graph(smiles)
            if graph is None:
                return None
            graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            graph = graph.to(self.device)

            with torch.no_grad():
                pred = self.gnn_model(graph)
            return pred.cpu().numpy().flatten()
        except Exception as e:
            print(f"GNN inference error: {e}")
            return None

                                                                               

    def _predict_xgb(self, smiles: str) -> np.ndarray | None:
        """Run stacked fingerprint through XGBoost ensemble. Returns (12,) or None."""
        try:
            fp = smiles_to_stacked_fp(smiles)
            if fp is None or len(fp) == 0:
                return None
            X_full = self.scaler.transform(fp.reshape(1, -1).astype(np.float32))

            preds = []
            for task in TOX21_TASKS:
                entry = self.xgb_models.get(task)
                if entry is None:
                    preds.append(0.5)
                    continue
                                                                              
                                                                 
                if isinstance(entry, dict):
                    clf  = entry["clf"]
                    sidx = entry.get("selector_idx")
                    X    = X_full[:, sidx] if sidx is not None else X_full
                else:
                    clf = entry
                    X   = X_full
                p = clf.predict_proba(X)[0, 1]
                preds.append(float(p))
            return np.array(preds, dtype=np.float32)
        except Exception as e:
            print(f"XGB inference error: {e}")
            return None

                                                                               

    def predict(self, smiles: str) -> dict:
        gnn_probs = self._predict_gnn(smiles)
        xgb_probs = self._predict_xgb(smiles)

        if gnn_probs is not None and xgb_probs is not None:
            ensemble = GNN_WEIGHT * gnn_probs + XGB_WEIGHT * xgb_probs
        elif gnn_probs is not None:
            ensemble = gnn_probs
        elif xgb_probs is not None:
            ensemble = xgb_probs
        else:
            ensemble = np.full(len(TOX21_TASKS), 0.5, dtype=np.float32)

        return {
            "gnn_probs":   gnn_probs,
            "xgb_probs":   xgb_probs,
            "ensemble":    ensemble,
            "task_names":  TOX21_TASKS,
            "overall_risk": float(np.mean(ensemble)),
        }

                                                                               

    def explain_xgb_shap(self, smiles: str, task_idx: int = 0) -> dict|None:
        try:
            import shap

            task  = TOX21_TASKS[task_idx]
            entry = self.xgb_models.get(task)
            if entry is None:
                return None

            fp = smiles_to_stacked_fp(smiles)
            if fp is None or len(fp) == 0:
                return None
            X_full = self.scaler.transform(fp.reshape(1, -1).astype(np.float32))

                                                          
            if isinstance(entry, dict):
                clf  = entry["clf"]
                sidx = entry.get("selector_idx")
                X    = X_full[:, sidx] if sidx is not None else X_full
            else:
                clf  = entry
                sidx = None
                X    = X_full

                                                                         
            raw_clf = clf
            if hasattr(clf, "estimator"):
                raw_clf = clf.estimator
            elif hasattr(clf, "calibrated_classifiers_"):
                raw_clf = clf.calibrated_classifiers_[0].estimator

            explainer = shap.TreeExplainer(raw_clf)
            shap_vals = explainer.shap_values(X)
            sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

                                                    
            abs_shap = np.abs(sv)
            top_idx  = np.argsort(abs_shap)[::-1][:20]

                                                                    
            global_top = sidx[top_idx] if sidx is not None else top_idx

                                                                 
            all_shap_full = np.zeros(4263, dtype=np.float32)
            if sidx is not None:
                all_shap_full[sidx] = sv
            else:
                all_shap_full[:len(sv)] = sv

            ev = explainer.expected_value
            base_val = float(ev[0] if hasattr(ev, "__len__") else ev)

            return {
                "task":          task,
                "feature_names": [f"FP_bit_{int(g)}" for g in global_top],
                "shap_values":   sv[top_idx],
                "base_value":    base_val,
                "all_shap":      all_shap_full,
                "top_indices":   global_top,
            }
        except Exception as e:
            print(f"SHAP error: {e}")
            return None

                                                                                

    def get_attention_weights(self, smiles: str) -> dict | None:
        
        try:
            graph = smiles_to_graph(smiles)
            if graph is None:
                return None
            graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            data = graph.to(self.device)

            with torch.no_grad():
                ei, attn = self.gnn_model.get_attention_weights(data)

                                                      
            n_atoms = graph.x.shape[0]
            atom_scores = np.zeros(n_atoms)
            ei_np   = ei.cpu().numpy()
            attn_np = attn.cpu().numpy()
            for (_, dst), a in zip(ei_np.T, attn_np):
                atom_scores[dst] += float(a)

                       
            if atom_scores.max() > 0:
                atom_scores /= atom_scores.max()

            return {"atom_scores": atom_scores}
        except Exception as e:
            print(f"Attention weight error: {e}")
            return None


                                                                                
_predictor_instance: ToxicityPredictor | None = None

def get_predictor() -> ToxicityPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ToxicityPredictor()
    return _predictor_instance