# Deployment guide

Goal: deploy the Flask backend somewhere small (Render / Railway / a VPS)
while keeping the heavy models off that server.

```
┌──────────────────────┐      gradio_client       ┌────────────────────────┐
│  Render / Railway    │ ───────────────────────▶ │  HF Space (Gradio)     │
│  Flask + scoring .pt │                          │  CodeT5-large encoder  │
│                      │ ───────────────────────▶ │  HF Inference API      │
│                      │   huggingface_hub        │  Qwen2.5-1.5B-Instruct │
└──────────────────────┘                          └────────────────────────┘
```

The server only loads `best_model_v5.pt` (~150–200 MB) and the cross-attention
layers it needs. No CodeT5, no Qwen.

---

## 1. Deploy the encoder Space

The folder `hf_space_encoder/` is a self-contained Gradio app.

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Gradio**
   - Hardware: **CPU basic** (free) is enough — the encoder is frozen
2. Push the contents of `hf_space_encoder/` to the Space repo:
   ```bash
   cd hf_space_encoder
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USER/pseudoscorex-encoder
   git add . && git commit -m "encoder space"
   git push origin main
   ```
3. Wait for the build (first boot downloads CodeT5-large — a few minutes).
4. The public URL will be `https://YOUR_USER-pseudoscorex-encoder.hf.space`.
   Copy it.

Quick sanity check from your laptop:
```python
from gradio_client import Client
client = Client("YOUR_USER/pseudoscorex-encoder")
out = client.predict("def add(a, b): return a + b", api_name="/encode")
print(out["shape"], len(out["clean_tokens"]))   # → [512, 1024], N
```

## 2. Get a Hugging Face token

https://huggingface.co/settings/tokens → "New token" → role **read**.
Save it somewhere safe — you'll need it for both the LLM call and (optionally)
private-Space access.

## 3. Deploy the Flask backend

The repo root is your deployment artefact. Drop the `.venv/`, `.idea/`,
`__pycache__/` and the `hf_space_encoder/` subfolder out of the deploy bundle
(add to `.gitignore` / `.dockerignore`).

### Required env vars on the host

```
ENCODER_SPACE_URL=https://YOUR_USER-pseudoscorex-encoder.hf.space
HF_TOKEN=hf_xxx
```

### Render

1. New → Web Service → connect your repo
2. Environment: **Docker** (uses the bundled `Dockerfile`)
   *or* **Python 3.12** with start command:
   ```
   gunicorn --preload --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app
   ```
3. Add env vars above
4. Plan: Starter (512 MB) is tight but workable since only the .pt is loaded.
   Standard (2 GB) is safer.

### Railway

1. New project → deploy from repo
2. Railway auto-detects the `Procfile`
3. Add env vars
4. Deploy

### Generic VPS

```bash
git clone <your-repo>
cd pseudoscorex_backend
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export ENCODER_SPACE_URL=...
export HF_TOKEN=...
gunicorn --preload --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:8000 app:app
```

## 4. Smoke-test the deployed backend

```bash
curl https://your-backend.example.com/info
curl -X POST https://your-backend.example.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Write a function that adds two numbers.",
    "answer":   "def add(a, b): return a + b",
    "criteria": [{"name": "correctness", "max_score": 5,
                  "description": "Returns the sum of two numbers"}]
  }'
```

## 5. Cost & latency notes

- **Encoder Space (free CPU)**: ~1–3 s per encode call. Each `/predict`
  request makes `2 + N` encode calls (question, answer, one per criterion).
  On the free tier the Space sleeps after ~48 h of inactivity — first call
  after a sleep takes 30–60 s to wake.
- **HF Inference API for Qwen**: free tier is rate-limited but plenty for
  light traffic. Pin a paid provider with `HF_PROVIDER=together` (or similar)
  if you need consistent latency.
- **Server**: stays small, no GPU needed.

If the Space wake-up latency hurts you, upgrade to Spaces *Always-on* (paid)
or move the encoder to an HF Inference Endpoint.
