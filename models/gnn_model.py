from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data


                                                                                
ATOM_FEAT_DIM = 39
BOND_FEAT_DIM = 8

ATOM_TYPES = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]                         


def atom_features(atom) -> list[float]:
    """39-dimensional atom feature vector."""
    from rdkit.Chem import rdchem

                            
    atype = atom.GetAtomicNum()
    atype_oh = [float(atype == t) for t in ATOM_TYPES] + [float(atype not in ATOM_TYPES)]

                                       
    deg = min(atom.GetDegree(), 10)
    deg_oh = [float(deg == i) for i in range(11)]

                               
    hyb_map = {
        rdchem.HybridizationType.SP:    1,
        rdchem.HybridizationType.SP2:   2,
        rdchem.HybridizationType.SP3:   3,
        rdchem.HybridizationType.SP3D:  4,
        rdchem.HybridizationType.SP3D2: 5,
    }
    hyb_val = hyb_map.get(atom.GetHybridization(), 0)
    hyb_oh  = [float(hyb_val == i) for i in range(6)]

                          
    scalars = [
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.IsInRingSize(3)),
        float(atom.IsInRingSize(4)),
        float(atom.IsInRingSize(5)),
        float(atom.IsInRingSize(6)),
        float(atom.IsInRingSize(7)),
        float(atom.GetMass() / 100.0),
    ]

    return atype_oh + deg_oh + hyb_oh + scalars                   


def bond_features(bond) -> list[float]:
    """8-dimensional bond feature vector."""
    from rdkit.Chem import rdchem

    bt = bond.GetBondType()
    return [
        float(bt == rdchem.BondType.SINGLE),
        float(bt == rdchem.BondType.DOUBLE),
        float(bt == rdchem.BondType.TRIPLE),
        float(bt == rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != rdchem.BondStereo.STEREONONE),
        0.0,            
    ]


def smiles_to_stacked_fp(smiles: str):
    """
    ECFP4(2048) + ECFP6(2048) + MACCS(167) = 4263-dim fingerprint.
    Single source of truth — imported by both train.py and predictor.py.
    Returns np.ndarray or None.
    """
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem import rdFingerprintGenerator
    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        ecfp4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol).astype(np.float32)
        ecfp6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048).GetFingerprintAsNumPy(mol).astype(np.float32)
        maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        return np.concatenate([ecfp4, ecfp6, maccs])
    except Exception:
        return None


def smiles_to_graph(smiles: str):
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    except Exception:
        return None

    edge_src, edge_dst, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_src += [i, j]
        edge_dst += [j, i]
        edge_attrs += [bf, bf]

    if len(edge_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


                                                                                

class ToxicityGATv2(nn.Module):
    """
    GATv2 with bond features, residuals, 3-way pooling, per-task heads.
    Target: val AUC ~0.75–0.80 on Tox21 (vs 0.62 in v1).
    """

    def __init__(
        self,
        in_channels:     int   = ATOM_FEAT_DIM,
        edge_dim:        int   = BOND_FEAT_DIM,
        hidden_channels: int   = 128,
        num_heads:       int   = 4,
        num_tasks:       int   = 12,
        dropout:         float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        H = hidden_channels
        head_dim = H // num_heads

                                             
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
        )

                                                        
        self.conv1 = GATv2Conv(H, head_dim, heads=num_heads, edge_dim=edge_dim,
                               concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(H, head_dim, heads=num_heads, edge_dim=edge_dim,
                               concat=True, dropout=dropout)
        self.conv3 = GATv2Conv(H, head_dim, heads=num_heads, edge_dim=edge_dim,
                               concat=True, dropout=dropout)

        self.bn1 = nn.BatchNorm1d(H)
        self.bn2 = nn.BatchNorm1d(H)
        self.bn3 = nn.BatchNorm1d(H)

                                          
        pool_dim = H * 3
        self.trunk = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

                                  
        self.task_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(num_tasks)])

    def encode(self, data: Data) -> torch.Tensor:
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_proj(x)

                     
        h = self.conv1(x, ei, ea)
        h = self.bn1(h); h = F.elu(h)
        x = x + h

                     
        h = self.conv2(x, ei, ea)
        h = self.bn2(h); h = F.elu(h)
        x = x + h

                     
        h = self.conv3(x, ei, ea)
        h = self.bn3(h); h = F.elu(h)
        x = x + h

                             
        xg = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch),
        ], dim=1)

        return self.trunk(xg)            

    def forward(self, data: Data) -> torch.Tensor:
        emb = self.encode(data)
        return torch.cat([
            torch.sigmoid(h(emb)) for h in self.task_heads
        ], dim=1)                  

    def get_attention_weights(self, data: Data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)

        h = self.conv1(x, ei, ea); x = x + self.bn1(h); x = F.elu(x)
        h = self.conv2(x, ei, ea); x = x + self.bn2(h); x = F.elu(x)

        out, (ret_ei, attn) = self.conv3(x, ei, ea, return_attention_weights=True)
        return ret_ei, attn.mean(dim=-1)


def build_gnn_model(num_tasks: int = 12, in_channels: int = ATOM_FEAT_DIM) -> ToxicityGATv2:
    return ToxicityGATv2(in_channels=in_channels, num_tasks=num_tasks)
