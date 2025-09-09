# AutoGrad
This repository contains a research-grade framework for evaluating student pseudocode solutions to algorithmic design questions. Unlike keyword or rubric-only approaches, our system reasons about the semantic intent of a solution and supports multiple correct implementations.

Here’s a ready-to-paste **README** section that documents the one-command setup script you added.

---

## 🚀 One-Command Setup

This repo ships with a cross-platform bootstrap script that prepares everything for you (virtualenv, dependencies, CPU-only PyTorch if needed, and a `.env` file).

### Prerequisites

* **Python** 3.10+ (3.11 recommended)
* **Git**
* **(Optional)** Docker Engine / Docker Desktop (for sandboxed execution)

### Clone

```bash
git clone https://github.com/sahan-vishwajith/AutoGrad.git
cd AutoGrad
```

### Run the setup script

**Linux / macOS**

```bash
python3 setup_project.py
```

**Windows (PowerShell)**

```powershell
python setup_project.py
```

The script will:

* create `.venv/`
* upgrade `pip/setuptools/wheel`
* install all dependencies from `requirements.txt`
* ensure **PyTorch (CPU)** is available (for `sentence-transformers`)
* create `.env` from `.env.example` if missing
* detect Docker (optional)

### Configure environment variables

Edit `.env` and add your keys:

```env
# Choose provider: groq | openai
PROVIDER=groq

# If using Groq:
GROQ_API_KEY=your_groq_key_here
MODEL_NAME=llama-3.1-70b-versatile

# If using OpenAI instead:
# OPENAI_API_KEY=your_openai_key_here
# MODEL_NAME=gpt-4o-mini
```

### Activate the virtualenv

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
. .\.venv\Scripts\Activate.ps1
```

### Run the API

```bash
uvicorn autograd.api.server:app --reload --port 8080
```

### Quick smoke test

In another terminal:

```bash
curl -X POST "http://127.0.0.1:8080/grade" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {"id":"q1","prompt":"Sum array","io_format":{"inputs":["arr"],"outputs":["sum"]}},
    "answer": {"student_id":"s1","pseudocode":"set total=0; for each x in arr: total=total+x; return total as sum"}
  }'
```

You should receive a JSON response with a `score`, `breakdown`, and `explanation`.

---

## 🧰 Troubleshooting

* **PyTorch install issues**
  The setup script auto-installs CPU-only PyTorch from the official index. If your network blocks it, rerun:

  ```bash
  . .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  python -m pip install -r requirements.txt
  ```

* **Windows execution policy**
  If PowerShell blocks activation: run PowerShell as Administrator and execute:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```

* **Docker not found (optional)**
  Only needed for hardened, sandboxed code execution. You can use the local executor for quick dev.

---

## 📁 Script reference

* **`setup_project.py`** — One-file bootstrapper (cross-platform).
  Re-run it anytime after pulling changes to ensure your environment is up to date.

