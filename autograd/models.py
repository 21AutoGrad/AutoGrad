# autograd/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class IOFormat(BaseModel):
    inputs: List[str]  # names, e.g., ["n", "arr"]
    outputs: List[str] # names, e.g., ["answer"]

class Question(BaseModel):
    id: str
    prompt: str
    io_format: Optional[IOFormat] = None
    references: Optional[List[str]] = None  # reference solutions (pseudo/code)

class StudentAnswer(BaseModel):
    student_id: str
    pseudocode: str

class TestCase(BaseModel):
    input: Dict[str, str]   # {"n":"5","arr":"1 2 3 4 5"}
    output: Dict[str, str]  # {"answer":"15"}
    kind: Literal["base","edge","adversarial"] = "base"

class ExecResult(BaseModel):
    passed: bool
    trace: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

class TestSummary(BaseModel):
    total: int
    passed: int
    results: List[ExecResult]

class Features(BaseModel):
    pass_rate: float
    edge_pass_rate: float
    timeouts: int
    runtime_ms_p50: float
    loops: int
    branches: int
    uses_recursion: bool
    inferred_complexity: Optional[str] = None
    cosine_to_refs: Optional[float] = None  # embedding similarity

class GradeOutput(BaseModel):
    score: float
    breakdown: Dict[str, float]
    explanation: str
