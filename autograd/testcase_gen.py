# autograd/testcase_gen.py
from typing import List
from .models import IOFormat, TestCase

BASIC_RULES = {
  # fallback heuristics if LLM is unavailable
  "int": ["0", "1", "2", "10", "1000"],
}

def generate_tests(prompt: str, io: IOFormat|None) -> List[TestCase]:
    cases: List[TestCase] = []
    if io:
        # simple seeds; extend with LLM-based generator later
        seeds = [{"kind":"base"}, {"kind":"edge"}, {"kind":"adversarial"}]
        for s in seeds:
            # you’ll write bespoke generators per task family; placeholder below
            tc = TestCase(input={}, output={}, kind=s["kind"])  # fill via LLM or rules
            cases.append(tc)
    return cases
