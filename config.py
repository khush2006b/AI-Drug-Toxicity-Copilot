import os
from dotenv import load_dotenv
load_dotenv()


                                           
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-3-flash-preview"

                                                                                
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

                                                                               
                                              
                                                                 
GNN_WEIGHT = 0.35
XGB_WEIGHT = 0.65

                                                                                
RANDOM_SEED      = 42
TEST_SPLIT       = 0.1
VALID_SPLIT      = 0.1
GNN_EPOCHS       = 80                                               
GNN_BATCH_SIZE   = 64
GNN_LR           = 3e-4                         
XGB_N_ESTIMATORS = 800
XGB_MAX_DEPTH    = 5

                                                                                
DATA_DIR       = "data"
MODEL_DIR      = "models"
GNN_MODEL_PATH = f"{MODEL_DIR}/gnn_model.pt"
XGB_MODEL_PATH = f"{MODEL_DIR}/xgb_model.pkl"
SCALER_PATH    = f"{MODEL_DIR}/scaler.pkl"

                                                                                
MOL_IMG_SIZE = (400, 300)
HIGH_RISK    = 0.7
MEDIUM_RISK  = 0.4

                                                                                
TOXIC_FRAGMENTS = {
    "Nitro group":          "[N+](=O)[O-]",
    "Aromatic amine":       "c[NH2]",
    "Epoxide":              "C1OC1",
    "Aldehyde":             "[CX3H1](=O)",
    "Michael acceptor":     "C=CC=O",
    "Quinone":              "O=C1C=CC(=O)C=C1",
    "Halogenated aromatic": "[cBr,cI]",
}

                                                                                
ENDPOINT_MECHANISMS = {
    "NR-AR":       "Androgen Receptor: Disruption causes reproductive & developmental toxicity.",
    "NR-AR-LBD":   "AR Ligand-Binding Domain: Direct competitive hormone binding.",
    "NR-AhR":      "Aryl Hydrocarbon Receptor: Mediates toxicity of dioxins/PCBs (carcinogenic).",
    "NR-Aromatase":"Aromatase: Blocks estrogen synthesis; endocrine disruption.",
    "NR-ER":       "Estrogen Receptor: Causes reproductive issues and hormone-driven cancers.",
    "NR-ER-LBD":   "ER Ligand-Binding Domain: Direct competitive estrogen binding.",
    "NR-PPAR-gamma":"PPAR-Gamma: Affects lipid metabolism and insulin sensitivity.",
    "SR-ARE":      "Antioxidant Response Element: Activated by oxidative stress/ROS damage.",
    "SR-ATAD5":    "ATAD5: Marker for genotoxicity and DNA damage response.",
    "SR-HSE":      "Heat-Shock Element: Triggered by proteotoxic stress (misfolded proteins).",
    "SR-MMP":      "Mitochondrial Membrane Potential: Disruption causes mitochondrial toxicity.",
    "SR-p53":      "p53: The guardian of the genome; activated by DNA damage/cancer risk.",
}

                                                                                
EXAMPLE_MOLECULES = {
    "Aspirin (safe NSAID)":       "CC(=O)Oc1ccccc1C(=O)O",
    "Doxorubicin (toxic chemo)":  "COc1cccc2C(=O)c3c(O)c4c(c(O)c3C(=O)c12)C[C@@](O)(C(=O)CO)C[C@H]4O[C@H]1C[C@@H](N)[C@H](O)[C@@H](C)O1",
    "Tamoxifen (moderate)":       "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Benzo[a]pyrene (carcinogen)":"c1ccc2ccc3cccc4ccc(c1)c2c34",
    "Nitrobenzene (toxic)":       "O=[N+]([O-])c1ccccc1",
    "Ibuprofen (safe NSAID)":     "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
}