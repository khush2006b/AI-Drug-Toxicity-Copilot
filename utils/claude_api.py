
from __future__ import annotations
import anthropic
import numpy as np
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOX21_TASKS, HIGH_RISK, MEDIUM_RISK


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


                                                                                

def explain_toxicity(
    smiles: str,
    ensemble_probs: np.ndarray,
    key_properties: dict,
    shap_summary: dict | None,
    toxic_fragments: dict,
) -> str:
    """
    Ask Claude to explain the toxicity prediction in plain English,
    citing the molecular properties and SHAP insights.
    """
    high_risk_tasks = [
        TOX21_TASKS[i] for i, p in enumerate(ensemble_probs) if p >= HIGH_RISK
    ]
    medium_risk_tasks = [
        TOX21_TASKS[i] for i, p in enumerate(ensemble_probs)
        if MEDIUM_RISK <= p < HIGH_RISK
    ]

    present_fragments = [k for k, v in toxic_fragments.items() if v]

    shap_text = ""
    if shap_summary:
        top3 = list(zip(
            shap_summary["feature_names"][:3],
            shap_summary["shap_values"][:3]
        ))
        shap_text = "Top SHAP features: " + ", ".join(
            f"{n} (Δ={v:+.3f})" for n, v in top3
        )

    prompt = f"""You are a medicinal chemistry expert reviewing an AI toxicity prediction.

Molecule SMILES: {smiles}

Physicochemical properties:
{chr(10).join(f"  {k}: {v}" for k, v in key_properties.items())}

Toxicity predictions (Tox21 endpoints, probability 0–1):
{chr(10).join(f"  {TOX21_TASKS[i]}: {p:.3f}" for i, p in enumerate(ensemble_probs))}

High-risk endpoints (≥0.7): {high_risk_tasks if high_risk_tasks else 'None'}
Medium-risk endpoints (0.4–0.7): {medium_risk_tasks if medium_risk_tasks else 'None'}

Known toxic structural fragments detected: {present_fragments if present_fragments else 'None'}
{shap_text}

Please provide:
1. A 2–3 sentence plain-English summary of this molecule's overall toxicity profile
2. The most likely biological mechanisms causing the predicted toxicity (cite specific endpoints)
3. Which structural features are most responsible (refer to the fragments and properties)
4. One concrete structural modification a medicinal chemist would recommend

Keep the tone scientific but accessible. Use ≤250 words total."""

    try:
        client = _client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"


                                                                                

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
    Ask Claude to explain WHY a structural modification reduced toxicity.
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
        client = _client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
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
        client = _client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
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
    system = f"""You are an expert medicinal chemist and toxicologist.
The user is asking about this molecule: {smiles}

Key properties: {key_properties}
Tox21 predictions: {dict(zip(TOX21_TASKS, [round(float(p),3) for p in ensemble_probs]))}

Answer concisely and scientifically. If you don't know something from the data provided, say so."""

    history = conversation_history + [{"role": "user", "content": question}]

    try:
        client = _client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system,
            messages=history,
        )
        answer = message.content[0].text
        history.append({"role": "assistant", "content": answer})
        return answer, history
    except Exception as e:
        err = f"Error: {str(e)}"
        history.append({"role": "assistant", "content": err})
        return err, history
