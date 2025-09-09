# autograd/pipeline.py
from .models import Question, StudentAnswer, GradeOutput, TestSummary, ExecResult, IOFormat
from .translator_llm import pseudocode_to_python
from .testcase_gen import generate_tests
from .executor.local_executor import run_python_solution
from .features.test_evidence import from_test_summary
from .features.structure_features import structural_signals
from .features.complexity_probe import infer_complexity
from .grader.evaluation_head import EvalHead
from .grader.explanation_llm import explain

def grade_once(question: Question, answer: StudentAnswer) -> GradeOutput:
    io = question.io_format
    code = pseudocode_to_python(question.prompt, io.model_dump() if io else None, answer.pseudocode)

    tests = generate_tests(question.prompt, io or IOFormat(inputs=[], outputs=[]))
    results = []
    passed = 0
    brief = []
    for tc in tests:
        ok, out, err, rt = run_python_solution(code, tc.input)
        # compare out with tc.output here (you’ll implement robust matchers)
        match = ok and (out == tc.output)
        passed += int(match)
        brief.append(f"{tc.kind}: {'PASS' if match else 'FAIL'} {rt or ''} {err[:60] if err else ''}")
        results.append(ExecResult(passed=match, stdout=str(out), stderr=err, trace=None))
    summary = TestSummary(total=len(tests), passed=passed, results=results)

    f = from_test_summary(summary)
    f.update(structural_signals(answer.pseudocode))
    f["inferred_complexity"] = infer_complexity(answer.pseudocode)

    # baseline head (in practice, fit on weakly supervised data; here a naive score)
    head = EvalHead()
    # dummy calibration: map pass_rate to score when no model yet
    score = f["pass_rate"]*100.0

    exp = explain(question, f, "; ".join(brief))
    breakdown = {"tests": f["pass_rate"]*70, "structure": (f["loops"]>0 or f["branches"]>0)*20, "robustness": f["edge_pass_rate"]*10}
    return GradeOutput(score=score, breakdown=breakdown, explanation=exp)
