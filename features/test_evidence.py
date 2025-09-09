# autograd/features/test_evidence.py
from ..models import TestSummary, Features

def from_test_summary(ts: TestSummary) -> dict:
    pass_rate = ts.passed / max(1, ts.total)
    edge_pass = sum(r.passed for r in ts.results[-max(1, ts.total//3):]) / max(1, ts.total//3)  # toy
    timeouts = sum(1 for r in ts.results if (r.stderr or "").lower().find("timeout")>=0)
    return {
        "pass_rate": pass_rate,
        "edge_pass_rate": edge_pass,
        "timeouts": timeouts,
        "runtime_ms_p50": 0.0,  # fill if you log timings
    }
