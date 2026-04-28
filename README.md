# Pseudoscore-X Flask Backend

HTTP API wrapping the **Criterion-Wise Neural-LLM Hybrid Grading System (v4)**
from `pseudoscore-x.ipynb`. Given a question, a student answer, and a list
of rubric criteria, it returns per-criterion scores, attention-derived
signals, and a generated explanation.

---

## 1. Project layout

```
pseudoscorex_backend/
├── app.py                  # Flask entry point
├── requirements.txt
├── test_request.py         # Smoke test / example client
├── README.md
├── best_model_v5.pt        # ← YOU provide this (trained checkpoint)
└── model/
    ├── __init__.py
    ├── architecture.py     # CriterionWiseScoringSystem (matches notebook)
    ├── inference.py        # Load encoder + checkpoint, run predict()
    ├── signals.py          # Attention-signal extraction
    └── explainer.py        # Rule-based explanation generator
```

---

## 2. Setup

### 2.1. Python + dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` pins `transformers==4.38.2` to match the notebook.
For `torch`, install the build that matches your CUDA version (or CPU):

```bash
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 example
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2.2. Provide the trained checkpoint

Copy the `best_model_v5.pt` file produced by the training notebook into the
project root. If you want it somewhere else, set the `CHECKPOINT_PATH`
environment variable.

The architecture in `model/architecture.py` is a byte-for-byte port of the
classes in the notebook, so `load_state_dict` will succeed without
`strict=False` tricks.

### 2.3. Start the server

```bash
python app.py
```

On first run the CodeT5-large encoder (~770M params) downloads from
HuggingFace — this can take a few minutes and ~3 GB of disk. Subsequent
runs reuse the HuggingFace cache.

Useful environment variables:

| Variable          | Default                    | Description                                 |
|-------------------|----------------------------|---------------------------------------------|
| `HOST`            | `0.0.0.0`                  | Bind address                                |
| `PORT`            | `8000`                     | Bind port                                   |
| `CHECKPOINT_PATH` | `best_model_v5.pt`         | Path to the trained scoring-head weights    |
| `ENCODER_NAME`    | `Salesforce/codet5-large`  | HuggingFace encoder (don't change unless you retrained) |
| `DEVICE`          | `cuda` if available, else `cpu` | Force a specific device                |
| `MAX_LENGTH`      | `512`                      | Max tokens per text field                   |

---

## 3. API

### `GET /health`

Liveness probe.

```json
{"status": "ok"}
```

### `GET /info`

Returns the loaded encoder name, checkpoint path, device, and max length.

### `POST /predict`

**Content-Type:** `application/json` (or multipart with a `file` field
containing the JSON payload).

#### Request schema

```json
{
  "question": "string — the problem statement",
  "answer":   "string — the student's answer / pseudocode",
  "criteria": [
    {
      "name":        "string — short criterion label",
      "max_score":   2,
      "description": "string — full rubric text; include level descriptors (what earns 0 / 1 / 2 marks). Falls back to `name` if omitted."
    }
  ]
}
```

Notes:
- `criteria` must be a non-empty list.
- `description` is strongly recommended — the notebook's v4 improvements
  specifically rely on the encoder seeing the full rubric text, not just
  the criterion name.

#### Response schema

```json
{
  "question": "...",
  "answer": "...",
  "total_score": 3.4,
  "max_total_score": 5,
  "percentage": 68.0,
  "results": [
    {
      "criterion": "Input parsing",
      "criterion_description": "...",
      "score": 1.8,
      "max_score": 2,
      "pred_norm": 0.9012,
      "signals": {
        "top_answer_tokens": [
          {"token": "split", "importance": 0.0821},
          {"token": "toInt", "importance": 0.0654}
        ],
        "missed_answer_tokens": ["..."],
        "active_rubric_concepts": ["parsing", "integer", "commas"],
        "confidence": 0.74,
        "source": "cross_attention"
      },
      "explanation": "'Input parsing': excellent (1.80 / 2, 90.1% of the max). Key rubric concepts..."
    }
  ],
  "overall_explanation": "Overall strong performance: 3.40 / 5 (68.0%). Strongest on: Input parsing. Needs work on: Even detection."
}
```

#### Error responses

- `400` — malformed JSON, missing fields, or non-numeric `max_score`
- `500` — anything else (traceback is logged server-side)

---

## 4. Example client

### curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

### Python

```python
import json, urllib.request

payload = {
    "question": "Write pseudocode that sums the even numbers in a CSV string.",
    "answer":   "function sumEvens(s): ...",
    "criteria": [
        {"name": "Input parsing", "max_score": 2,
         "description": "2 marks: splits and converts. 1 mark: one of the two. 0: no attempt."}
    ]
}

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print(json.load(resp))
```

Or just run the included smoke test once the server is up:

```bash
python test_request.py
```

---

## 5. Production deployment

Flask's built-in server is fine for dev. For production use gunicorn with a
**single worker** — the encoder is ~770M params and you don't want to load
it multiple times:

```bash
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 300 --preload app:app
```

- `-w 1`: one worker. If you need concurrency, use `--threads 4` (the model
  is read-only at inference, so threaded is safe).
- `--preload`: loads the model before forking, so the encoder weights live
  in shared memory if you ever bump `-w`.
- `--timeout 300`: inference over many criteria on CPU can exceed the
  default 30s timeout.

### Docker sketch

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
ENV CHECKPOINT_PATH=/app/best_model_v5.pt
EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "300", "--preload", "app:app"]
```

GPU container: start from `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
instead of `python:3.10-slim` and drop the torch pip line.

---

## 6. Plugging in an LLM explainer

The notebook was designed so that the attention signals feed into a
downstream Llama-based explainer. The current `model/explainer.py` is
deliberately rule-based so the API works out of the box, but swapping in
an LLM is a one-function change. Open `model/explainer.py` and replace the
body of `build_explanation()` with a call to your LLM, passing it the
`signals` dict plus the score. Everything else stays the same.

---

## 7. Troubleshooting

**`FileNotFoundError: Checkpoint not found at 'best_model_v5.pt'`**
You haven't placed the trained weights yet. Copy the `.pt` file produced
by the notebook into the project root, or set `CHECKPOINT_PATH`.

**`RuntimeError: Error(s) in loading state_dict ... Missing key(s) / Unexpected key(s)`**
The architecture hyperparameters don't match what you trained. Check that
the training used `embedding_dim=1024`, `n_heads=8`, `n_cross_layers=2`,
and that `cross_attn_qc` is 1 layer while `cross_attn_qa` / `cross_attn_ac`
are 2 layers — which is what `CriterionWiseScoringSystem.__init__`
reproduces. If you changed any of these during training, override the
matching env var before starting the server.

**Predictions look random / uniformly around 0.5**
The encoder loaded but the scoring head didn't. Confirm the checkpoint path
in the startup logs — the line `[pipeline] Loading checkpoint: ...`.

**First request is very slow, subsequent requests fast**
Expected — model + encoder are loaded lazily on the first `/predict` call
only if you disable `with app.app_context(): load_pipeline()` at the top of
`app.py`. By default they load at startup, so only the initial server boot
is slow.

**CPU inference is too slow**
CodeT5-large is large. Options in order of impact: (1) run on GPU,
(2) shorten `MAX_LENGTH` if your inputs are short, (3) batch criteria
(the current `predict()` loops — easy to extend to batch all criteria in
one forward pass since `q_hidden` / `a_hidden` are already shared).
