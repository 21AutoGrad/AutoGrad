# autograd/features/complexity_probe.py
# very light heuristic placeholder; you can upgrade to LLM probe
def infer_complexity(pseudocode: str) -> str|None:
    if "nested" in pseudocode.lower() or "for i" in pseudocode and "for j" in pseudocode:
        return "O(n^2)"
    return None
