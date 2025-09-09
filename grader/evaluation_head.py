# autograd/grader/evaluation_head.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class EvalHead:
    def __init__(self):
        self.clf = LogisticRegression()
        self.fnames = ["pass_rate","edge_pass_rate","timeouts","runtime_ms_p50","loops","branches"]

    def fit(self, X: list[dict], y: list[int]):
        M = np.array([[x.get(f,0.0) for f in self.fnames] for x in X])
        self.clf.fit(M, y)

    def predict_score(self, x: dict) -> float:
        M = np.array([[x.get(f,0.0) for f in self.fnames]])
        prob = self.clf.predict_proba(M)[0,1] if hasattr(self.clf, "predict_proba") else self.clf.decision_function(M)
        return float(prob*100.0)
