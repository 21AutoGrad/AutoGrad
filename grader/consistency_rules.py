# autograd/grader/consistency_rules.py
def check(explanation: str, features: dict) -> list[str]:
    issues = []
    if "all tests passed" in explanation.lower() and features.get("pass_rate",0)<0.99:
        issues.append("Explanation claims 'all tests passed' but pass_rate < 0.99.")
    # add more cross-checks
    return issues
