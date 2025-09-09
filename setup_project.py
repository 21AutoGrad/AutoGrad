#!/usr/bin/env python3
"""
setup_project.py — One-command bootstrap for the AutoGrad research repo.

Usage:
  Windows (PowerShell):  python setup_project.py
  Linux/macOS:           python3 setup_project.py
"""

from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import itertools
import venv
from pathlib import Path
from typing import List

# -----------------------------
# Config
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"

DEFAULT_REQUIREMENTS: List[str] = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "python-dotenv>=1.0",
    "sentence-transformers>=3.0",
    "scikit-learn>=1.4",
    "numpy>=1.26",
    "pandas>=2.2",
    "tiktoken>=0.7",
    "groq>=0.9",
    "docker>=7.0",
    # Optional alternative provider:
    # "openai>=1.40",
]

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PIP_BASE_ARGS = ["-m", "pip", "install", "--disable-pip-version-check"]

# -----------------------------
# Minimal progress UI (stdlib)
# -----------------------------

def _term_supports_cr() -> bool:
    return True  # conservative; works for most shells

def draw_step_bar(done: int, total: int, width: int = 26) -> None:
    total = max(total, 1)
    filled = int(width * min(done, total) / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {done}/{total}", end="\r" if _term_supports_cr() else "\n", flush=True)

class Spinner:
    FRAMES = "|/-\\"
    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            print(f"{frame} {self.label}...", end="\r" if _term_supports_cr() else "\n", flush=True)
            time.sleep(0.1)

    def start(self): self._thr.start()
    def stop(self, ok: bool = True):
        self._stop.set()
        self._thr.join(timeout=0.2)
        # clear line
        print(" " * (len(self.label) + 8), end="\r", flush=True)
        print(("✔" if ok else "✖") + f" {self.label}")

def run_step(label: str, func, *args, **kwargs):
    sp = Spinner(label); sp.start()
    try:
        out = func(*args, **kwargs)
        sp.stop(True)
        return out
    except Exception:
        sp.stop(False)
        raise

# -----------------------------
# Core helpers
# -----------------------------

def log(msg: str) -> None:
    print(f"▶ {msg}")

def create_venv() -> None:
    if not VENV_DIR.exists():
        venv.EnvBuilder(with_pip=True, clear=False, upgrade=False, with_wheel=True).create(str(VENV_DIR))

def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if platform.system().lower().startswith("win") else "bin/python")

def ensure_pip_up_to_date(py: Path) -> None:
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], check=True)

def python_expr(py: Path, code: str) -> str:
    cp = subprocess.run([str(py), "-c", code], check=False, capture_output=True, text=True)
    return (cp.stdout or "").strip()

def ensure_torch(py: Path) -> None:
    have = python_expr(py, "import importlib.util as u; print('1' if u.find_spec('torch') else '0')") == "1"
    if have: return
    # Try CPU wheel index first, then fallback
    try:
        subprocess.run([str(py), *PIP_BASE_ARGS, "--index-url", CPU_TORCH_INDEX, "torch", "torchvision", "torchaudio"], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([str(py), *PIP_BASE_ARGS, "torch"], check=True)

def parse_requirements_file(path: Path) -> List[str]:
    """
    Very small parser for simple, one-per-line reqs.
    Skips comments/blank lines and options (-r, -f, --extra-index-url, etc.).
    """
    pkgs: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", "--")):
            # skip option lines; bulk install will still read the file if needed
            continue
        pkgs.append(line)
    return pkgs

def run_streaming(cmd: List[str]) -> int:
    """
    Run a command and stream stdout/stderr line-by-line so the user
    sees progress (useful during long pip downloads/builds).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        # print pip's own progress lines as-is
        print(line.rstrip())
    return proc.wait()

def install_one(py: Path, requirement: str) -> None:
    """
    Install a single requirement and stream its output.
    Shows clear start/finish markers for that package.
    """
    print(f"→ Installing {requirement}")
    code = run_streaming([str(py), *PIP_BASE_ARGS, requirement])
    if code != 0:
        raise subprocess.CalledProcessError(code, requirement)
    print(f"✔ Installed {requirement}\n")

def pip_install_individual(py: Path, packages: List[str]) -> None:
    """
    Install a list of top-level requirements one-by-one with streaming output.
    Note: per-package installs can be slower than a single resolver pass,
    but give visible progress.
    """
    for pkg in packages:
        install_one(py, pkg)

def pip_install_requirements(py: Path) -> None:
    """
    If requirements.txt exists, install each top-level line individually with streaming output.
    After that, run a bulk '-r requirements.txt' to let the resolver finalize transitive deps
    (very quick if already satisfied).
    Otherwise, install the DEFAULT_REQUIREMENTS individually.
    """
    req_file = REPO_ROOT / "requirements.txt"
    if req_file.exists():
        top = parse_requirements_file(req_file)
        if top:
            print("— Installing packages listed in requirements.txt (one by one) —")
            pip_install_individual(py, top)
            # finalize with a quick resolver pass (mostly no-ops now)
            print("— Finalizing dependency resolution (-r requirements.txt) —")
            run_streaming([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req_file)])
        else:
            # empty/special file: fall back to bulk
            run_streaming([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req_file)])
    else:
        print("— Installing default requirements (one by one) —")
        pip_install_individual(py, DEFAULT_REQUIREMENTS)

def ensure_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if env_path.exists():
        return
    if example.exists():
        shutil.copyfile(example, env_path)
    else:
        env_path.write_text(
            "PROVIDER=groq\nGROQ_API_KEY=your_groq_key_here\nMODEL_NAME=llama-3.1-70b-versatile\n"
            "# OPENAI_API_KEY=your_openai_key_here\n# MODEL_NAME=gpt-4o-mini\n"
        )

def check_docker() -> None:
    try:
        cp = subprocess.run(["docker", "--version"], check=False, capture_output=True, text=True)
        if cp.returncode == 0:
            print(f"→ Docker detected: {cp.stdout.strip()}")
        else:
            print("→ Docker not found (optional).")
    except FileNotFoundError:
        print("→ Docker not found (optional).")

def print_next_steps() -> None:
    win = platform.system().lower().startswith("win")
    activation = r".\.venv\Scripts\Activate.ps1" if win else "source .venv/bin/activate"
    uvicorn = "uvicorn autograd.api.server:app --reload --port 8080"
    tips = {
        "activate": activation,
        "run_api": uvicorn,
        "env_hint": "Edit .env to add your GROQ_API_KEY (or OPENAI_API_KEY).",
    }
    print("\nSetup complete. Next steps:")
    print(json.dumps(tips, indent=2))

# -----------------------------
# Main
# -----------------------------

def main() -> int:
    print("AutoGrad repository bootstrap starting…\n")
    steps = [
        ("Create virtualenv", create_venv),
        ("Locate venv Python", lambda: None),
        ("Upgrade pip/setuptools/wheel", None),
        ("Install requirements (visible, one by one)", None),
        ("Ensure PyTorch (CPU)", None),
        ("Prepare .env", ensure_env_file),
        ("Check Docker (optional)", check_docker),
    ]
    total = len(steps); done = 0

    draw_step_bar(done, total)
    run_step(steps[0][0], create_venv); done += 1; draw_step_bar(done, total)

    # locate python
    def _locate():
        py_ = venv_python()
        if not py_.exists():
            raise RuntimeError(f"venv python not found at {py_}")
        _locate.py = py_
    run_step(steps[1][0], _locate); py = _locate.py  # type: ignore[attr-defined]
    done += 1; draw_step_bar(done, total)

    run_step(steps[2][0], ensure_pip_up_to_date, py); done += 1; draw_step_bar(done, total)

    # per-package visible install (streams)
    run_step(steps[3][0], pip_install_requirements, py); done += 1; draw_step_bar(done, total)

    # torch (if still missing)
    run_step(steps[4][0], ensure_torch, py); done += 1; draw_step_bar(done, total)

    run_step(steps[5][0], ensure_env_file); done += 1; draw_step_bar(done, total)
    run_step(steps[6][0], check_docker); done += 1; draw_step_bar(done, total); print()

    print_next_steps()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
