# autograd/features/structure_features.py
import re

def structural_signals(pseudocode: str) -> dict:
    loops = len(re.findall(r"\b(for|while)\b", pseudocode.lower()))
    branches = len(re.findall(r"\b(if|else if|elif|switch)\b", pseudocode.lower()))
    uses_rec = bool(re.search(r"\brecurse|recursion|call\s+self\b", pseudocode.lower()))
    return {"loops": loops, "branches": branches, "uses_recursion": uses_rec}
