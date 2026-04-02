from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import streamlit as st
import numpy as np
import pandas as pd
 
st.set_page_config(
    page_title="AI Drug Toxicity Copilot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
                                                                                
     
                                                                                
st.markdown("""
<style>
:root {
    --accent:    #00F4B9;
    --hi:        #EF4444;
    --med:       #F59E0B;
    --lo:        #10B981;
    --muted:     #94A3B8;
    --card-bg:   rgba(28,33,44,0.85);
    --card-bdr:  rgba(255,255,255,0.09);
}

/* ── Force dark background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section.main > div {
    background-color: #0E1117 !important;
    color: #E2E8F0 !important;
}

/* ── Sidebar dark ── */
[data-testid="stSidebar"] {
    background-color: #141824 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

/* ── Input fields dark ── */
textarea, input[type="text"] {
    background-color: #1E2330 !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
 
/* ── hero ── */
.hero-title {
    font-size: 2.5rem; font-weight: 800;
    background: -webkit-linear-gradient(45deg, #00F4B9, #3B82F6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: var(--muted); font-size: 1rem; margin-top: 4px; }
 
/* ── glass card ── */
.gc {
    background: var(--card-bg);
    border: 1px solid var(--card-bdr);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
}
 
/* ── kpi ── */
.kpi-val  { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
.kpi-lbl  { font-size: 0.75rem; text-transform: uppercase;
            letter-spacing: 1.2px; color: var(--muted); margin-top: 2px; }
 
/* ── risk badge ── */
.rb { padding: 5px 14px; border-radius: 20px; font-weight: 700;
      font-size: 0.88rem; display: inline-block; border-width: 1px; border-style: solid; }
.rb-hi  { background:rgba(239,68,68,.18);  color:#FCA5A5; border-color:rgba(239,68,68,.5);}
.rb-med { background:rgba(245,158,11,.18); color:#FCD34D; border-color:rgba(245,158,11,.5);}
.rb-lo  { background:rgba(16,185,129,.18); color:#6EE7B7; border-color:rgba(16,185,129,.5);}
 
/* ── endpoint card ── */
.epc {
    background: rgba(18,22,34,.8);
    border-left: 4px solid rgba(255,255,255,.12);
    padding: 10px 14px; margin-bottom: 9px; border-radius: 8px;
}
.epc-hi  { border-left-color: var(--hi); }
.epc-med { border-left-color: var(--med); }
.epc-lo  { border-left-color: var(--lo); }
.epc-row { display:flex; justify-content:space-between; align-items:flex-start; }
.epc-name{ font-weight:600; font-size:.95rem; }
.epc-desc{ font-size:.78rem; color:var(--muted); margin-top:3px; }
.epc-right{ text-align:right; min-width:110px; }
.epc-prob{ font-family:monospace; font-size:1.15rem; font-weight:700; }
.epc-conf{ font-size:.72rem; color:var(--muted); margin-top:2px; }
 
/* ── before/after compare ── */
.cmp-box {
    border: 1px solid var(--card-bdr); border-radius: 12px;
    padding: 14px; text-align: center; background: var(--card-bg);
}
.cmp-risk-val { font-size: 2rem; font-weight: 800; }
.cmp-label    { font-size: .78rem; color: var(--muted); margin-bottom: 6px; }
.cmp-arrow    { font-size: 2.8rem; font-weight: 700; line-height: 1; }
 
/* ── heatmap legend ── */
.hm-legend {
    display: flex; align-items: center; gap: 8px; margin-top: 6px;
    font-size: .8rem; color: var(--muted);
}
.hm-grad {
    width: 160px; height: 14px; border-radius: 7px;
    background: linear-gradient(to right, #10B981, #F59E0B, #EF4444);
}
 
/* ── pulse ── */
@keyframes pulse-hi {
    0%  { box-shadow: 0 0 0 0 rgba(239,68,68,.4); }
    70% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
    100%{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.pulse { animation: pulse-hi 2s infinite; }
 
/* ── section divider ── */
.sdiv { border-top: 1px solid var(--card-bdr); margin: 18px 0; }
</style>
""", unsafe_allow_html=True)
 
                                                                                
         
                                                                                
from utils.mol_utils import (
    smiles_to_mol, compute_key_properties, mol_to_image, mol_to_atom_heatmap,
    shap_to_atom_scores, identify_toxic_fragments, validate_smiles,
    get_substructure_atoms_bonds,
)
from utils.visualizations import (
    plot_toxicity_bars, plot_model_comparison, plot_shap_values,
    plot_counterfactual_comparison, plot_risk_gauge, plot_properties_comparison,
    plot_toxicity_radar,
)
from utils.counterfactual import generate_counterfactuals
from utils.gemini_api import (
    explain_toxicity, explain_counterfactual, generate_full_report,
    ask_about_molecule, design_molecule_from_description, get_risk_summary,
)
from utils.similarity_search import find_similar_compounds
from utils.batch_screener import screen_batch, parse_smiles_block, dataframe_to_csv
from config import (
    TOX21_TASKS, HIGH_RISK, MEDIUM_RISK, ENDPOINT_MECHANISMS, EXAMPLE_MOLECULES,
)
 
                                                                                
               
                                                                                
_DEFAULTS = {
    "pred_done": False, "last_smi": "", "result": None,
    "mol": None, "props": {}, "frags": {}, "cfs": [],
    "chat": [], "ai_exp": "", "report": "", "analogs": [],
    "shap_data": None, "atom_scores": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
                                                                                
              
                                                                                
@st.cache_resource(show_spinner="Loading GATv2 + XGBoost …")
def _load_predictor():
    from models.predictor import ToxicityPredictor
    return ToxicityPredictor()
 
try:
    predictor   = _load_predictor()
    MODEL_OK    = True
    _MODEL_ERR  = ""
except Exception as _e:
    predictor   = None
    MODEL_OK    = False
    _MODEL_ERR  = str(_e)
 
                                                                                
         
                                                                                
with st.sidebar:
    st.markdown("## 🧬 Drug Toxicity\n### Copilot")
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
 
    mode = st.radio(
        "**Mode**",
        ["🔬 Predict", "🔍 Analyze", "⚗️ Improve", "🧠 AI Design", "📋 Batch"],
        label_visibility="visible",
    )
 
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown("**Load Example**")
    ex = st.selectbox("Select example", [""] + list(EXAMPLE_MOLECULES.keys()), label_visibility="collapsed")
 
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown(
        "✅ Models loaded" if MODEL_OK
        else "❌ Run `python train.py`",
        unsafe_allow_html=False,
    )
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    with st.expander("Stack"):
        st.markdown("""
- **GATv2** — 39-dim atoms + 8-dim bonds  
- **XGBoost** — ECFP4+ECFP6+MACCS (4263-dim)  
- **Focal Loss** + probability calibration  
- **SHAP** + GAT attention heatmap  
- Counterfactual editor  
- Tanimoto analog search  
        """)
    st.caption("CodeCure Hackathon — Track A")
 
                                                                                
        
                                                                                
st.markdown('<p class="hero-title">AI Drug Toxicity Copilot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">GATv2 + XGBoost ensemble &nbsp;·&nbsp; '
    'Atom-level heatmap &nbsp;·&nbsp; Counterfactual safer-molecule design &nbsp;·&nbsp; '
    'Structured AI analysis</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
 
 
                                                                                
                                                                                
                                                                                
if mode == "🧠 AI Design":
    st.markdown("### 🗣️ Natural Language → Molecule Design")
    st.markdown(
        "Describe the properties you want. AI generates a SMILES, "
        "screens it for toxicity, and explains the design reasoning."
    )
    desc = st.text_area(
        "Description", height=90,
        placeholder="e.g. A COX-2 selective NSAID with low GI toxicity, "
                    "similar to celecoxib but without the sulfonamide group.",
    )
    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        go = st.button("🧬 Design & Screen", type="primary", use_container_width=True)
    if go and desc.strip():
        with st.spinner("Designing molecule …"):
            smi, reason = design_molecule_from_description(desc)
        if not smi:
            st.error(f"Design failed: {reason}")
        else:
            mol_d = smiles_to_mol(smi)
            d1, d2 = st.columns([2, 4])
            with d1:
                st.success(f"**Generated SMILES:**\n`{smi}`")
                if mol_d:
                    st.image(mol_to_image(mol_d, size=(280, 200), dark_mode=True))
            with d2:
                st.info(f"**AI Design Reasoning:**\n\n{reason}")
                                      
            st.session_state["_inject"] = smi
            if st.button("→ Run Full Toxicity Screen on This Molecule"):
                st.rerun()
    st.stop()
 
 
                                                                                
                                                                                
                                                                                
elif mode == "📋 Batch":
    st.markdown("### 📋 High-Throughput Batch Screening")
    st.markdown("Paste up to 100 SMILES (one per line). Optional name prefix: `Aspirin CC(=O)Oc1ccccc1C(=O)O`")
    batch_txt = st.text_area("SMILES input", height=200,
                             placeholder="CC(=O)Oc1ccccc1C(=O)O\nO=[N+]([O-])c1ccccc1")
    if st.button("🚀 Run Screen", type="primary"):
        if not MODEL_OK:
            st.error("Models not loaded.")
        else:
            lst = parse_smiles_block(batch_txt)
            if not lst:
                st.warning("No valid SMILES found.")
            else:
                prog, stat = st.progress(0), st.empty()
                def _cb(d, t): prog.progress(d/t); stat.text(f"Screening {d}/{t} …")
                df = screen_batch(lst, predictor.predict, _cb)
                stat.text("Done!")
                st.dataframe(
                    df.style.background_gradient(cmap="RdYlGn_r", subset=["Overall Risk"]),
                    use_container_width=True, hide_index=True,
                )
                st.download_button("⬇️ Download CSV", dataframe_to_csv(df),
                                   "batch_toxicity.csv", "text/csv", use_container_width=True)
    st.stop()
 
 
                                                                                
                                                        
                                                                                
 
                        
if "_inject" in st.session_state:
    _default_smi = st.session_state.pop("_inject")
elif ex:
    _default_smi = EXAMPLE_MOLECULES.get(ex, "")
else:
    _default_smi = st.session_state.last_smi
 
                                                                               
col_in, col_prev = st.columns([3, 2])
with col_in:
    st.markdown("#### 🔬 Input Molecule")
    smi_input = st.text_area(
        "SMILES", value=_default_smi, height=76,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O  (Aspirin)",
        label_visibility="collapsed",
    )
    run_btn = st.button("🚀 Analyse", type="primary", use_container_width=True)
 
with col_prev:
    if smi_input:
        ok, err = validate_smiles(smi_input)
        if ok:
            _pm = smiles_to_mol(smi_input)
            if _pm:
                st.image(mol_to_image(_pm, size=(300, 180), dark_mode=True),
                         caption="Preview", use_container_width=True)
        else:
            st.error(f"❌ {err}")
 
                                                                                
if run_btn and smi_input:
    ok, err = validate_smiles(smi_input)
    if not ok:
        st.error(f"Invalid SMILES: {err}")
    elif not MODEL_OK:
        st.error(f"Models not loaded: {_MODEL_ERR}")
    else:
        _mol   = smiles_to_mol(smi_input)
        _props = compute_key_properties(_mol)
        _frags = identify_toxic_fragments(_mol)
 
        with st.spinner("🧠 GATv2 + XGBoost inference …"):
            _res = predictor.predict(smi_input)
 
        with st.spinner("🔍 Tanimoto analog search …"):
            try:    _analogs = find_similar_compounds(smi_input, top_k=5)
            except: _analogs = []
 
        with st.spinner("⚗️ Counterfactual generation …"):
            try:    _cfs = generate_counterfactuals(smi_input, predictor.predict, max_candidates=4)
            except: _cfs = []
 
        st.session_state.update({
            "pred_done": True, "last_smi": smi_input, "result": _res,
            "mol": _mol, "props": _props, "frags": _frags,
            "cfs": _cfs, "analogs": _analogs,
            "ai_exp": "", "report": "", "chat": [], "shap_data": None, "atom_scores": None,
        })
        st.rerun()
 
                                                                                
if not st.session_state.pred_done or not st.session_state.result:
    st.info("Enter a SMILES string above and click **Analyse** to begin.")
    st.stop()
 
                                                                                
res      = st.session_state.result
mol      = st.session_state.mol
props    = st.session_state.props
frags    = st.session_state.frags
cfs      = st.session_state.cfs
analogs  = st.session_state.analogs
ens      = res["ensemble"]
overall  = res["overall_risk"]
smi_now  = st.session_state.last_smi
risk_sum = get_risk_summary(ens)
 
hi_ct    = int(np.sum(ens >= HIGH_RISK))
med_ct   = int(np.sum((ens >= MEDIUM_RISK) & (ens < HIGH_RISK)))
frag_ct  = sum(1 for v in frags.values() if v)
 
risk_lbl = "CRITICAL" if overall >= HIGH_RISK else ("MODERATE" if overall >= MEDIUM_RISK else "SAFE")
rb_cls   = "rb-hi pulse" if overall >= HIGH_RISK else ("rb-med" if overall >= MEDIUM_RISK else "rb-lo")
 
                                                                                
         
                                                                                
k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(
    f'<div class="kpi-lbl">Assessment</div>'
    f'<div class="rb {rb_cls}" style="margin-top:8px">{risk_lbl}</div>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<div class="kpi-lbl">Overall Risk</div>'
    f'<div class="kpi-val">{overall*100:.1f}%</div>',
    unsafe_allow_html=True,
)
k3.markdown(
    f'<div class="kpi-lbl">High-Risk Endpoints</div>'
    f'<div class="kpi-val" style="color:#EF4444">{hi_ct}/12</div>',
    unsafe_allow_html=True,
)
k4.markdown(
    f'<div class="kpi-lbl">Toxicophores</div>'
    f'<div class="kpi-val">{frag_ct}</div>',
    unsafe_allow_html=True,
)
k5.markdown(
    f'<div class="kpi-lbl">Drug-likeness (QED)</div>'
    f'<div class="kpi-val">{props.get("QED (Drug-likeness)", "—")}</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
 
 
                                                                                
                                                                                
                                                                                
if mode == "🔬 Predict":
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(plot_toxicity_bars(ens), use_container_width=True)
    with c2:
        st.plotly_chart(plot_risk_gauge(overall), use_container_width=True)
 
    st.markdown("#### Model Agreement: GATv2 vs XGBoost vs Ensemble")
    st.plotly_chart(plot_model_comparison(res["gnn_probs"], res["xgb_probs"], ens), use_container_width=True)
 
                                                                              
    st.markdown("#### All 12 Tox21 Endpoints — Risk & Confidence")
    st.caption("Each card shows predicted probability, risk category, and model confidence level.")
    cols_ep = st.columns(3)
    for idx, r in enumerate(risk_sum):
        ep_cls  = "epc-hi" if r["prob"] >= HIGH_RISK else ("epc-med" if r["prob"] >= MEDIUM_RISK else "epc-lo")
        pc_col  = "#FCA5A5" if r["prob"] >= HIGH_RISK else ("#FCD34D" if r["prob"] >= MEDIUM_RISK else "#6EE7B7")
        desc    = ENDPOINT_MECHANISMS.get(r["task"], "")
        html = f"""<div class="epc {ep_cls}">
  <div class="epc-row">
    <div>
      <div class="epc-name">{r["emoji"]} {r["task"]}</div>
      <div class="epc-desc">{desc}</div>
    </div>
    <div class="epc-right">
      <div class="epc-prob" style="color:{pc_col}">{r["pct"]}</div>
      <div class="epc-conf">{r["category"]}<br><span style="font-size:.68rem">{r["confidence"]} confidence</span></div>
    </div>
  </div>
</div>"""
        cols_ep[idx % 3].markdown(html, unsafe_allow_html=True)
 
 
                                                                                
                                                                                
                                                                                
elif mode == "🔍 Analyze":
    tab_struct, tab_heatmap, tab_shap, tab_analogs, tab_ai = st.tabs([
        "🏗️ Structure", "🌡️ Atom Heatmap", "⚙️ SHAP", "🔍 Analogs", "📄 AI Report"
    ])
 
                                                                                
    with tab_struct:
        s1, s2 = st.columns([2, 3])
        with s1:
            st.markdown("#### Molecular Structure")
            st.image(mol_to_image(mol, size=(360, 270), dark_mode=True), use_container_width=True)
            st.code(smi_now, language=None)
 
        with s2:
            st.markdown("#### Physicochemical Profile")
            pdf = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
            st.dataframe(pdf, use_container_width=True, hide_index=True)
            mw, logp, hbd, hba = (props.get(k, 0) for k in
                ["Molecular Weight", "logP (Lipophilicity)", "H-bond Donors", "H-bond Acceptors"])
            ro5 = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
            st.markdown(f"**Lipinski Ro5:** {'✅ Pass — oral bioavailability likely' if ro5 else '❌ Fail — bioavailability concerns'}")
 
        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### Toxic Fragment Scan")
        fc = st.columns(4)
        for i, (nm, present) in enumerate(frags.items()):
            fc[i % 4].markdown(f"{'🔴' if present else '🟢'} **{nm}**")
 
        present_frags = [k for k, v in frags.items() if v]
        if present_frags:
            st.warning(f"⚠️ {len(present_frags)} toxic fragment(s) detected")
            sel_frag = st.selectbox("Highlight on structure", present_frags)
            from config import TOXIC_FRAGMENTS
            smt = TOXIC_FRAGMENTS.get(sel_frag, "").split(",")[0].strip()
            hl_a, hl_b = get_substructure_atoms_bonds(mol, smt)
            if hl_a:
                st.image(mol_to_image(mol, size=(420, 300), dark_mode=True,
                                      highlight_atoms=hl_a, highlight_bonds=hl_b),
                         caption=f"Red highlight: {sel_frag}")
 
                                                                                
    with tab_heatmap:
        st.markdown("#### 🌡️ Atom-Level Toxicity Heatmap")
        st.markdown(
            "Every atom is colored on a **green → yellow → red** scale by its contribution "
            "to the toxicity prediction. This shows *where* the danger lives in the molecule."
        )
 
        hm_src = st.radio(
            "Score source",
            ["GAT Attention Weights", "SHAP Fingerprint Mapping"],
            horizontal=True,
        )
 
        if st.button("🎨 Generate Heatmap", type="primary"):
            with st.spinner("Computing atom scores …"):
                if hm_src == "GAT Attention Weights":
                    attn = predictor.get_attention_weights(smi_now)
                    if attn:
                        st.session_state.atom_scores = attn["atom_scores"]
                    else:
                        st.warning("Attention weights unavailable.")
                else:
                                                       
                    shap_d = st.session_state.shap_data
                    if shap_d is None:
                        shap_d = predictor.explain_xgb_shap(smi_now, task_idx=0)
                        st.session_state.shap_data = shap_d
                    if shap_d and "all_shap" in shap_d:
                        import numpy as _np
                        st.session_state.atom_scores = shap_to_atom_scores(
                            mol, shap_d["all_shap"], shap_d["top_indices"]
                        )
                    else:
                        st.warning("SHAP data unavailable.")
 
        if st.session_state.atom_scores is not None:
            scores = st.session_state.atom_scores
            hm_img = mol_to_atom_heatmap(mol, scores, size=(520, 380))
            st.image(hm_img, width=520)
 
                    
            st.markdown(
                '<div class="hm-legend">'
                '<span>Safe</span>'
                '<div class="hm-grad"></div>'
                '<span>Toxic</span>'
                '</div>',
                unsafe_allow_html=True,
            )
 
                                          
            import plotly.express as px
            atom_syms = [
                mol.GetAtomWithIdx(i).GetSymbol() + str(i)
                for i in range(mol.GetNumAtoms())
            ]
            fig_sc = px.bar(
                x=atom_syms, y=scores,
                color=scores, color_continuous_scale="RdYlGn_r",
                labels={"x": "Atom", "y": "Toxicity score (0–1)"},
                title=f"Per-atom toxicity scores — {hm_src}",
            )
            fig_sc.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sc, use_container_width=True)
 
            top3_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
            st.markdown("**Most Toxic Atoms:**")
            for rank, ai in enumerate(top3_idx, 1):
                sym = mol.GetAtomWithIdx(ai).GetSymbol()
                st.markdown(f"  {rank}. **{sym}{ai}** — score {scores[ai]:.3f}")
        else:
            st.info("Click **Generate Heatmap** above to see the atom-level toxicity map.")
 
                                                                                
    with tab_shap:
        st.markdown("#### ⚙️ SHAP — XGBoost Feature Attribution")
        task_sel = st.selectbox("Endpoint", range(12),
                                format_func=lambda i: f"{TOX21_TASKS[i]}  ({ens[i]*100:.1f}%)")
        if st.button("Compute SHAP", key="shap_btn"):
            with st.spinner("Computing …"):
                sd = predictor.explain_xgb_shap(smi_now, task_idx=task_sel)
            if sd:
                st.session_state.shap_data = sd
            else:
                st.warning("SHAP not available for this endpoint.")
        if st.session_state.shap_data:
            st.plotly_chart(plot_shap_values(st.session_state.shap_data), use_container_width=True)
            st.caption("Red = fingerprint bits that **increase** toxicity. Green = bits that **reduce** it.")
 
                                                                                
    with tab_analogs:
        st.markdown("#### 🔍 Real NIH Tox21 Lab Analogs")
        st.markdown("Closest compounds actually tested in Tox21 wet-lab assays — grounding predictions in reality.")
        if not analogs:
            st.info("No analogs found.")
        else:
            for i, hit in enumerate(analogs):
                with st.expander(f"Analog #{i+1} — Tanimoto {hit['similarity']*100:.1f}%", expanded=(i == 0)):
                    a1, a2, a3 = st.columns([2, 2, 4])
                    with a1:
                        st.metric("Tanimoto", f"{hit['similarity']*100:.1f}%")
                        hm = smiles_to_mol(hit["smiles"])
                        if hm:
                            st.image(mol_to_image(hm, size=(200, 140), dark_mode=True))
                    with a2:
                        st.code(hit["smiles"], language=None)
                    with a3:
                        lbl = hit["labels"]
                        act = [k for k, v in lbl.items() if v == 1]
                        ina = [k for k, v in lbl.items() if v == 0]
                        if act: st.markdown(f"🔴 **Toxic:** {', '.join(act)}")
                        if ina: st.markdown(f"🟢 **Safe:** {', '.join(ina)}")
                        if not act and not ina: st.markdown("⚪ No experimental data")
 
                                                                                
    with tab_ai:
        st.markdown("#### 📄 Structured AI Analysis")
        st.caption("AI outputs structured sections: Assessment → Risk Factors → Biological Impact → Modification")
 
        if st.button("🔬 Generate Structured Explanation", type="primary"):
            with st.spinner("Analysing …"):
                st.session_state.ai_exp = explain_toxicity(
                    smi_now, ens, props, st.session_state.shap_data, frags
                )
 
        if st.session_state.ai_exp:
            st.markdown(
                f'<div class="gc">{st.session_state.ai_exp}</div>',
                unsafe_allow_html=True,
            )
 
        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
 
        if st.button("📋 Generate Full Research Report"):
            with st.spinner("Writing …"):
                st.session_state.report = generate_full_report(
                    smi_now, ens, props, frags, cfs, st.session_state.shap_data
                )
        if st.session_state.report:
            st.download_button(
                "⬇️ Download Report (.md)", st.session_state.report,
                "toxicity_report.md", "text/markdown", use_container_width=True,
            )
            st.markdown(st.session_state.report)
 
        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### 💬 Ask the AI Chemist")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_q = st.chat_input("Ask about mechanisms, drug design, endpoints …")
        if user_q:
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("…"):
                    ans, hist = ask_about_molecule(user_q, smi_now, ens, props, st.session_state.chat)
                st.markdown(ans)
            st.session_state.chat = hist
 
 
                                                                                
                                                                                
                                                                                
elif mode == "⚗️ Improve":
    st.markdown("### ⚗️ Lead Optimisation — Safer Molecule Generation")
    st.markdown(
        "The system applies targeted structural modifications (remove –NO₂, "
        "open epoxide, reduce logP, etc.) and re-screens each candidate "
        "through the full ensemble to find safer alternatives."
    )
 
    if not cfs:
        st.info("No counterfactuals generated for this molecule. "
                "Try a more toxic compound — **Nitrobenzene** or **Doxorubicin** "
                "from the sidebar examples.")
    else:
                                                                               
        st.plotly_chart(
            plot_counterfactual_comparison(ens, cfs, max_show=3),
            use_container_width=True,
        )
 
                                                                               
        for i, cf in enumerate(cfs, 1):
            reduction = cf.risk_reduction * 100
            is_better = cf.is_improvement
            arrow     = "↓" if is_better else "↑"
            arw_col   = "#6EE7B7" if is_better else "#FCA5A5"
            badge_cls = "rb-lo" if is_better else "rb-hi"
 
            with st.expander(
                f"Option {i}: {cf.description}   "
                f"{'▼' if is_better else '▲'} {abs(reduction):.1f}% risk",
                expanded=(i == 1),
            ):
                                                                              
                left, mid, right = st.columns([5, 1, 5])
                new_mol = smiles_to_mol(cf.modified_smiles)
 
                with left:
                    st.markdown('<div class="cmp-box">', unsafe_allow_html=True)
                    st.markdown('<div class="cmp-label">ORIGINAL</div>', unsafe_allow_html=True)
                    st.image(mol_to_image(mol, size=(280, 210), dark_mode=True), use_container_width=True)
                    orig_rb = "rb-hi" if cf.original_risk >= HIGH_RISK else ("rb-med" if cf.original_risk >= MEDIUM_RISK else "rb-lo")
                    st.markdown(
                        f'<div class="cmp-risk-val" style="color:#EF4444">{cf.original_risk*100:.1f}%</div>'
                        f'<div class="rb {orig_rb}" style="margin-top:6px">{"HIGH" if cf.original_risk>=HIGH_RISK else "MODERATE" if cf.original_risk>=MEDIUM_RISK else "SAFE"} RISK</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
 
                with mid:
                    st.markdown(
                        f'<div style="text-align:center; padding-top:80px;">'
                        f'<div class="cmp-arrow" style="color:{arw_col}">{arrow}</div>'
                        f'<div style="font-size:.9rem; color:{arw_col}; font-weight:700">'
                        f'{abs(reduction):.1f}%</div></div>',
                        unsafe_allow_html=True,
                    )
 
                with right:
                    st.markdown('<div class="cmp-box">', unsafe_allow_html=True)
                    st.markdown('<div class="cmp-label">MODIFIED</div>', unsafe_allow_html=True)
                    if new_mol:
                        st.image(mol_to_image(new_mol, size=(280, 210), dark_mode=True), use_container_width=True)
                    mod_rb = "rb-hi" if cf.modified_risk >= HIGH_RISK else ("rb-med" if cf.modified_risk >= MEDIUM_RISK else "rb-lo")
                    st.markdown(
                        f'<div class="cmp-risk-val" style="color:#6EE7B7">{cf.modified_risk*100:.1f}%</div>'
                        f'<div class="rb {mod_rb}" style="margin-top:6px">{"HIGH" if cf.modified_risk>=HIGH_RISK else "MODERATE" if cf.modified_risk>=MEDIUM_RISK else "SAFE"} RISK</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
 
                                                                                
                st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
                d1, d2 = st.columns([3, 3])
                with d1:
                    st.markdown(f"**Modification:** {cf.description}")
                    if cf.tasks_improved:
                        st.markdown(f"**Improved endpoints:** {', '.join(cf.tasks_improved)}")
                    st.code(cf.modified_smiles, language=None)
 
                with d2:
                    if new_mol:
                        st.plotly_chart(
                            plot_properties_comparison(props, compute_key_properties(new_mol)),
                            use_container_width=True,
                        )
 
                if st.button("🤖 Ask AI: Why does this modification work?", key=f"cf_ai_{i}"):
                    with st.spinner("…"):
                        ai_exp = explain_counterfactual(
                            smi_now, cf.modified_smiles, cf.description,
                            cf.original_risk, cf.modified_risk, cf.tasks_improved,
                            props, compute_key_properties(new_mol) if new_mol else {},
                        )
                    st.markdown(f'<div class="gc">{ai_exp}</div>', unsafe_allow_html=True)