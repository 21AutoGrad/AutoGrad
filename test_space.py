"""
Quick smoke test for the encoder HF Space.

Run from your laptop after the Space says "Running":
    python test_space.py

Verifies the Space is alive, the API contract matches what the backend
expects (hidden_b64 / shape / attention_mask / clean_tokens), and that
the hidden state has the right dimensions.

Delete or keep — not used by the deployed backend.
"""
from gradio_client import Client

SPACE = "sahanwickramasinghe/pseudoscorex-encoder"

print(f"Connecting to {SPACE}...")
client = Client(SPACE)

text = "def add(a, b): return a + b"
print(f"Encoding: {text!r}")
out = client.predict(text, api_name="/encode")

print("\nResponse keys:", list(out.keys()))
print("Hidden shape: ", out["shape"])
print("Mask length:  ", len(out["attention_mask"]))
print("Clean tokens: ", out["clean_tokens"][:8], "...")

# Sanity check: shape should be [512, 1024]
assert out["shape"] == [512, 1024], f"unexpected shape: {out['shape']}"
assert len(out["attention_mask"]) == 512, "mask length mismatch"
print("\nOK — Space is healthy.")
