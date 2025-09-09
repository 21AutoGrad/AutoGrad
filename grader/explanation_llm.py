# autograd/grader/explanation_llm.py
import os
from dotenv import load_dotenv; load_dotenv()
from textwrap import dedent

SYSTEM = """You are a fair teaching assistant. Explain the grade concisely.
- Cite concrete evidence: which tests passed/failed, what edge-cases, structure cues.
- Offer 2-3 actionable tips.
- Keep under 150 words.
"""

def build_context(question, features: dict, test_brief: str):
    # FiLM-lite: we inject evidence as a compact table in the prompt
    return dedent(f"""
    [Problem] {question.prompt[:600]}
    [Evidence]
    - pass_rate: {features.get('pass_rate'):.2f}
    - edge_pass_rate: {features.get('edge_pass_rate'):.2f}
    - timeouts: {features.get('timeouts')}
    - loops/branches: {features.get('loops')}/{features.get('branches')}
    - inferred_complexity: {features.get('inferred_complexity')}
    - tests: {test_brief[:600]}
    Explain the grade and next steps.
    """)

def explain(question, features: dict, test_brief: str) -> str:
    from groq import Groq
    client = Groq()
    content = build_context(question, features, test_brief)
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_NAME","llama-3.1-70b-versatile"),
        messages=[{"role":"system", "content":SYSTEM},
                  {"role":"user", "content":content}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
