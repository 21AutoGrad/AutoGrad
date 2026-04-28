"""
Thin client for the encoder Hugging Face Space.

Uses the official `gradio_client` so the URL contract isn't fragile across
Gradio versions. Reconstructs torch tensors from the base64 float16 payload.

Env vars
--------
ENCODER_SPACE_URL  Public URL of the Space, e.g.
                   https://USER-pseudoscorex-encoder.hf.space  OR  USER/SPACE-NAME
                   Required.
HF_TOKEN           Required only if the Space is private.
"""
import base64
import logging
import os
import threading
import time

import numpy as np
import torch
from gradio_client import Client

logger = logging.getLogger("encoder_client")

ENCODER_SPACE_URL = os.getenv("ENCODER_SPACE_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN")

_client = None
_client_lock = threading.Lock()


def _get_client() -> Client:
    """Lazily build the Gradio client, thread-safe."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not ENCODER_SPACE_URL:
            raise RuntimeError(
                "ENCODER_SPACE_URL is not set. Point it at your encoder Space, "
                "e.g. https://USER-pseudoscorex-encoder.hf.space"
            )
        logger.info("Connecting to encoder Space: %s", ENCODER_SPACE_URL)
        # gradio_client renamed/removed the auth kwarg between versions.
        # Public Spaces don't need a token, so fall back to no-auth on TypeError.
        if HF_TOKEN:
            for kw in ("hf_token", "token"):
                try:
                    _client = Client(ENCODER_SPACE_URL, **{kw: HF_TOKEN})
                    break
                except TypeError:
                    continue
            else:
                logger.warning(
                    "gradio_client did not accept hf_token/token; "
                    "connecting without auth (Space must be public)."
                )
                _client = Client(ENCODER_SPACE_URL)
        else:
            _client = Client(ENCODER_SPACE_URL)
        return _client


def encode_text(text: str, device: torch.device):
    """
    Returns (hidden, attention_mask, clean_tokens):
        hidden          torch.float32 tensor of shape (1, seq_len, 1024)
        attention_mask  torch.long    tensor of shape (1, seq_len)
        clean_tokens    list[str]
    """
    client = _get_client()

    t0 = time.perf_counter()
    out = client.predict(text, api_name="/encode")
    logger.debug("Space round-trip: %.2fs", time.perf_counter() - t0)

    if not isinstance(out, dict) or "hidden_b64" not in out:
        raise RuntimeError(f"Unexpected encoder Space response: {out!r}")

    arr = np.frombuffer(base64.b64decode(out["hidden_b64"]), dtype=np.float16)
    arr = arr.reshape(out["shape"])  # (seq_len, 1024)

    hidden = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.tensor(out["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
    return hidden, mask, out["clean_tokens"]
