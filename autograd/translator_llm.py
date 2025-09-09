# autograd/translator_llm.py
import os
from dotenv import load_dotenv
load_dotenv()

PROVIDER = os.getenv("PROVIDER", "groq")

SYSTEM = """You convert algorithmic pseudocode into SAFE, pure-Python3 code.
- No network/file access.
- Pure functions.
- Respect provided I/O signature if given.
Return ONLY the code."""
USER_TMPL = """Problem:
{prompt}

I/O format (if any):
{io}

Student pseudocode:
{pseudo}

Produce a Python3 function solve(**inputs) returning a dict of outputs. Avoid prints.
"""

def pseudocode_to_python(prompt: str, io: str|None, pseudocode: str) -> str:
    text = USER_TMPL.format(prompt=prompt, io=io or "N/A", pseudo=pseudocode)
    if PROVIDER == "groq":
        from groq import Groq
        client = Groq()
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME","llama-3.1-70b-versatile"),
            messages=[{"role":"system", "content":SYSTEM},
                      {"role":"user", "content":text}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    else:
        raise NotImplementedError
