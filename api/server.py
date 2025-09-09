# autograd/api/server.py
from fastapi import FastAPI
from ..models import Question, StudentAnswer, GradeOutput
from ..pipeline import grade_once

app = FastAPI(title="AutoGrad API")

@app.post("/grade", response_model=GradeOutput)
def grade(question: Question, answer: StudentAnswer):
    return grade_once(question, answer)


# uvicorn autograd.api.server:app --reload --port 8080
