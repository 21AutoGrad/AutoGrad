"""
Flask HTTP API for the Criterion-Wise Neural-LLM Hybrid Grading System.

Endpoints
---------
GET  /health   → liveness probe
GET  /info     → model / pipeline metadata
POST /predict  → score an (answer, criteria) payload against a question

POST /predict expects JSON of shape:
{
  "question": "…",
  "answer":   "…",
  "criteria": [
    {"name": "Input handling", "max_score": 2, "description": "full rubric text"},
    {"name": "Logic",          "max_score": 3, "description": "…"}
  ]
}

It returns JSON of shape:
{
  "total_score": 3.4,
  "max_total_score": 5,
  "percentage": 68.0,
  "results": [ {criterion, score, max_score, pred_norm, signals, explanation}, ... ],
  "overall_explanation": "…"
}
"""
import logging
import os
import time
import uuid
from flask import Flask, request, jsonify, g

# Configure logging BEFORE importing anything that uses it
from model.logging_config import setup_logging
setup_logging()

from model import load_pipeline, predict

logger = logging.getLogger("app")


app = Flask(__name__)


# ── Load model once at process start ───────────────────────────────────────
# For production, prefer gunicorn with --preload so this happens pre-fork.
with app.app_context():
    load_pipeline()


# ── Per-request logging hooks ──────────────────────────────────────────────
@app.before_request
def _log_request_start():
    g.req_id = uuid.uuid4().hex[:8]
    g.req_start = time.perf_counter()
    logger.info("→ %s %s [req=%s]", request.method, request.path, g.req_id)


@app.after_request
def _log_request_end(response):
    if hasattr(g, "req_start"):
        elapsed_ms = (time.perf_counter() - g.req_start) * 1000
        logger.info(
            "← %s %s [req=%s] status=%d in %.1fms",
            request.method, request.path,
            getattr(g, "req_id", "?"),
            response.status_code, elapsed_ms,
        )
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/info", methods=["GET"])
def info():
    import torch
    return jsonify({
        "encoder_space": os.getenv("ENCODER_SPACE_URL", "(unset)"),
        "llm_model": os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "llm_disabled": os.getenv("DISABLE_LLM_EXPLAINER", "0") == "1",
        "checkpoint": os.getenv("CHECKPOINT_PATH", "best_model_v5.pt"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # Accept either application/json or a raw file upload named "file"
    payload = None
    if request.is_json:
        payload = request.get_json(silent=True)
    elif "file" in request.files:
        import json as _json
        try:
            payload = _json.load(request.files["file"])
        except Exception as e:
            return jsonify({"error": f"Invalid JSON file: {e}"}), 400
    else:
        # Try raw body as JSON anyway
        payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({
            "error": "Request body must be JSON with keys: question, answer, criteria"
        }), 400

    question = payload.get("question")
    answer = payload.get("answer")
    criteria = payload.get("criteria")

    try:
        result = predict(question=question, answer=answer, criteria=criteria)
        return jsonify(result), 200
    except ValueError as ve:
        # Validation errors → 400
        logger.warning("Validation error: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Unhandled error in /predict")
        return jsonify({"error": "Internal error", "detail": str(e)}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # debug=False because we don't want the reloader to load the encoder twice
    app.run(host=host, port=port, debug=False, threaded=False)