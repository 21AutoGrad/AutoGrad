# autograd/executor/docker_executor.py
import docker, json, tempfile, os, textwrap
from pathlib import Path

def run_in_docker(py_code: str, test_input: dict, timeout_sec: int=2):
    client = docker.from_env()
    with tempfile.TemporaryDirectory() as td:
        Path(td, "main.py").write_text(py_code)
        runner = textwrap.dedent("""
        import json, sys
        from main import solve
        inp = json.load(open('in.json'))
        out = solve(**inp)
        json.dump(out, open('out.json','w'))
        """)
        Path(td, "runner.py").write_text(runner)
        Path(td, "in.json").write_text(json.dumps(test_input))
        vols = {td: {"bind": "/work", "mode": "ro"}}
        try:
            out = client.containers.run(
                "python:3.11-slim",
                command=["python","/work/runner.py"],
                volumes=vols,
                network_disabled=True,
                mem_limit="256m",
                pids_limit=256,
                nano_cpus=1_000_000_000, # 1 CPU
                remove=True,
                timeout=timeout_sec,
                working_dir="/work",
            )
            # read out.json by mounting rw in real impl; keep simple here
            return True, None, "", None
        except Exception as e:
            return False, None, str(e), None
