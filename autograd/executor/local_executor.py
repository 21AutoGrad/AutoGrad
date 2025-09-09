# autograd/executor/local_executor.py
import textwrap, time, traceback
from AutoGrad.autograd import sandbox_snippets

def run_python_solution(code: str, test_input: dict, timeout_sec: float=2.0):
    namespace = {}
    start = time.time()
    try:
        compiled = compile(code, "<student>", "exec")
        exec(compiled, namespace)
        res = namespace["solve"](**test_input)
        dt = (time.time()-start)*1000
        return True, res, "", f"{dt:.2f}ms"
    except Exception as e:
        return False, None, traceback.format_exc(), None
