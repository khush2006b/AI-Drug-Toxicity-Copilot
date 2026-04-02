from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
import google.generativeai as genai
import numpy as np
from config import GEMINI_API_KEY, TOX21_TASKS, HIGH_RISK, MEDIUM_RISK, GEMINI_MODEL
 
 
def _get_model(system_instruction: str | None = None) -> genai.GenerativeModel:
    genai.configure(api_key=GEMINI_API_KEY)
    kwargs = {"model_name": GEMINI_MODEL}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    return genai.GenerativeModel(**kwargs)
 
 
def _confidence_label(prob: float) -> str:
    if prob >= 0.85: return "Very Strong"
    if prob >= 0.70: return "Strong"
    if prob >= 0.55: return "Moderate"
    if prob >= 0.40: return "Weak"
    return "Very Weak"
 
def _risk_category(prob: float) -> tuple[str, str]:
    if prob >= HIGH_RISK:   return "High Risk", "🔴"
    if prob >= MEDIUM_RISK: return "Moderate Risk", "🟡"
    return "Low Risk", "🟢"
 
 
def get_risk_summary(ensemble_probs: np.ndarray) -> list[dict]:
    """Per-endpoint structured risk + confidence — used by UI cards."""
    results = []
    for i, task in enumerate(TOX21_TASKS):
        p   = float(ensemble_probs[i])
        cat, emoji = _risk_category(p)
        results.append({
            "task":       task,
            "prob":       p,
            "pct":        f"{p*100:.1f}%",
            "category":   cat,
            "emoji":      emoji,
            "confidence": _confidence_label(p),
        })
    return results
 
def explain_toxicity(
    smiles: str,
    ensemble_probs: np.ndarray,
    key_properties: dict,
    shap_summary: dict | None,
    toxic_fragments: dict,
) -> str:
    high_risk_tasks   = [TOX21_TASKS[i] for i, p in enumerate(ensemble_probs) if p >= HIGH_RISK]
    medium_risk_tasks = [TOX21_TASKS[i] for i, p in enumerate(ensemble_probs) if MEDIUM_RISK <= p < HIGH_RISK]
    present_fragments = [k for k, v in toxic_fragments.items() if v]
 
    shap_text = ""
    if shap_summary:
        top3 = list(zip(shap_summary["feature_names"][:3], shap_summary["shap_values"][:3]))
        shap_text = "Top SHAP features: " + ", ".join(f"{n} (Δ={v:+.3f})" for n, v in top3)
 
                                                                             
    logp_val = key_properties.get('logP (Lipophilicity)', '?')

    prompt = (
        "You are a medicinal chemistry expert. Analyse this molecule and respond ONLY "
        "in the exact structured format below — no preamble, no markdown headers outside the template.\n\n"
        f"MOLECULE: {smiles}\n"
        f"Properties: {key_properties}\n"
        f"High-risk endpoints (>=70%): {high_risk_tasks or 'None'}\n"
        f"Medium-risk endpoints (40-70%): {medium_risk_tasks or 'None'}\n"
        f"Toxic fragments present: {present_fragments or 'None'}\n"
        f"{shap_text}\n\n"
        "Respond in EXACTLY this format:\n\n"
        "**Overall Assessment**\n"
        "[1 sentence verdict]\n\n"
        "**Key Risk Factors**\n"
        f"- High lipophilicity (logP={logp_val}) promotes membrane accumulation\n"
        "- [Risk factor 2]\n"
        "- [Risk factor 3 if applicable]\n\n"
        "**Biological Impact**\n"
        "- [Mechanism 1 — cite specific endpoint e.g. NR-AR]\n"
        "- [Mechanism 2]\n\n"
        "**Structural Liabilities**\n"
        "- [Fragment/property causing toxicity]\n"
        "- [Second liability if present]\n\n"
        "**Recommended Modification**\n"
        "[1 concrete structural change]\n\n"
        "Keep each bullet to 1 concise sentence. Total <=200 words."
    )
 
    try:
        return _get_model().generate_content(prompt).text
    except Exception as e:
        return f"AI explanation unavailable: {e}"
 
 
                                                                                
 
def explain_counterfactual(
    original_smiles: str,
    modified_smiles: str,
    description: str,
    original_risk: float,
    modified_risk: float,
    tasks_improved: list[str],
    key_props_original: dict,
    key_props_modified: dict,
) -> str:
    """
    Ask Gemini to explain WHY a structural modification reduced toxicity.
    """
    risk_pct = (original_risk - modified_risk) * 100
 
    prompt = f"""You are a medicinal chemistry expert.
 
Original molecule: {original_smiles}
Modified molecule: {modified_smiles}
Modification applied: {description}
 
Risk change: {original_risk*100:.1f}% → {modified_risk*100:.1f}% (Δ = -{risk_pct:.1f}%)
Toxicity endpoints improved: {tasks_improved if tasks_improved else 'marginal improvements across multiple'}
 
Original properties: {key_props_original}
Modified properties: {key_props_modified}
 
Explain in 3–4 sentences:
1. Why this specific modification reduces toxicity at a mechanistic level
2. Which biological targets or pathways are less activated in the modified molecule
3. Whether this modification is chemically feasible and drug-like (Lipinski rules)
 
Be specific and scientific. ≤150 words."""
 
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"
 
 
                                                                                
 
def generate_full_report(
    smiles: str,
    ensemble_probs: np.ndarray,
    key_properties: dict,
    toxic_fragments: dict,
    counterfactuals: list,
    shap_summary: dict | None,
) -> str:
    """
    Generate a complete structured research report (Markdown format).
    Suitable for submitting as part of hackathon deliverable.
    """
    high_risk = [(TOX21_TASKS[i], p) for i, p in enumerate(ensemble_probs) if p >= HIGH_RISK]
    medium_risk = [(TOX21_TASKS[i], p) for i, p in enumerate(ensemble_probs) if MEDIUM_RISK <= p < HIGH_RISK]
    low_risk = [(TOX21_TASKS[i], p) for i, p in enumerate(ensemble_probs) if p < MEDIUM_RISK]
 
    cf_text = ""
    for i, cf in enumerate(counterfactuals[:3], 1):
        cf_text += f"\n  Candidate {i}: {cf.modified_smiles}\n"
        cf_text += f"    Modification: {cf.description}\n"
        cf_text += f"    Risk reduction: {cf.risk_reduction*100:.1f}%\n"
        cf_text += f"    Improved endpoints: {cf.tasks_improved}\n"
 
    prompt = f"""You are writing a scientific report for a drug discovery hackathon.
 
COMPOUND: {smiles}
 
PHYSICOCHEMICAL PROFILE:
{chr(10).join(f"{k}: {v}" for k, v in key_properties.items())}
 
TOX21 PREDICTIONS (GNN + XGBoost ensemble):
High-risk (≥70%): {high_risk}
Medium-risk (40–70%): {medium_risk}
Low-risk (<40%): {low_risk}
 
STRUCTURAL TOXICOPHORES DETECTED:
{chr(10).join(f"  {'✓' if v else '✗'} {k}" for k, v in toxic_fragments.items())}
 
COUNTERFACTUAL SAFER ALTERNATIVES:
{cf_text if cf_text else "No viable modifications found."}
 
Write a complete structured scientific report in Markdown with these sections:
# Compound Toxicity Analysis Report
## 1. Executive Summary (3 sentences)
## 2. Physicochemical Profile & Drug-Likeness
## 3. Toxicity Risk Assessment (discuss each high/medium endpoint)
## 4. Structural Liability Analysis (discuss each detected toxicophore)
## 5. Safer Structural Alternatives (explain the counterfactuals)
## 6. Recommendations for Lead Optimisation
## 7. Conclusion
 
Use scientific language. Include specific endpoint names. Be concrete about mechanisms.
Target length: 600–800 words."""
 
    try:
        model = _get_model()
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 2048}
        )
        return response.text
    except Exception as e:
        return f"# Report Generation Error\n\n{str(e)}"
 
 
                                                                                
 
def ask_about_molecule(
    question: str,
    smiles: str,
    ensemble_probs: np.ndarray,
    key_properties: dict,
    conversation_history: list[dict],
) -> tuple[str, list[dict]]:
    """
    Conversational Q&A about the analysed molecule.
    Maintains multi-turn history.
    Returns (answer, updated_history).
    """
    system_instruction = f"""You are an expert medicinal chemist and toxicologist.
The user is asking about this molecule: {smiles}
 
Key properties: {key_properties}
Tox21 predictions: {dict(zip(TOX21_TASKS, [round(float(p),3) for p in ensemble_probs]))}
 
Answer concisely and scientifically. If you don't know something from the data provided, say so."""
 
                                                                  
    gemini_history = []
    for msg in conversation_history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
 
    history_for_app = list(conversation_history)                                          
 
    try:
        model = _get_model(system_instruction=system_instruction)
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(question)
        answer = response.text
        history_for_app.append({"role": "user", "content": question})
        history_for_app.append({"role": "assistant", "content": answer})
        return answer, history_for_app
    except Exception as e:
        err = f"Error: {str(e)}"
        history_for_app.append({"role": "user", "content": question})
        history_for_app.append({"role": "assistant", "content": err})
        return err, history_for_app
 
 
                                                                               
 
def design_molecule_from_description(description: str) -> tuple[str, str]:
    """
    ★ HACKATHON FEATURE — Natural Language → SMILES generator.
 
    Ask Gemini to design a molecule matching a plain-English description.
    Returns (smiles, reasoning).
    The SMILES is validated with RDKit; if invalid, returns ("", error_message).
    """
    prompt = f"""You are an expert medicinal chemist and drug designer.
 
A researcher has requested a molecule with the following description:
"{description}"
 
Your task:
1. Design a chemically valid molecule that best matches this description.
2. Return ONLY a single valid SMILES string on the first line (no label, no prefix).
3. On the following lines, provide 3–5 sentences of scientific reasoning explaining:
   - Why this structure satisfies the description
   - Key functional groups present and their roles
   - Expected drug-likeness (Lipinski compliance)
   - Any known concerns or trade-offs
 
SMILES must be valid RDKit-parseable. Do NOT use isotope labels or unusual notation.
Format:
<SMILES>
<Reasoning>"""
 
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        text = response.text.strip()
 
        lines = text.splitlines()
                                            
        smiles = ""
        reasoning_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if not smiles:
                                                          
                smiles = line.strip("`").strip()
            else:
                reasoning_lines.append(line)
 
        reasoning = " ".join(reasoning_lines) if reasoning_lines else "No reasoning provided."
 
                                    
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "", f"Gemini returned an invalid SMILES: `{smiles}`. Try rephrasing your description."
 
                      
        smiles = Chem.MolToSmiles(mol)
        return smiles, reasoning
 
    except Exception as e:
        return "", f"AI design unavailable: {str(e)}"