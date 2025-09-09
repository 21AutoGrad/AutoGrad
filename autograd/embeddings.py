# autograd/embeddings.py
from sentence_transformers import SentenceTransformer
_model = None

def embed_text(text: str):
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/e5-base-v2")
    return _model.encode([text], normalize_embeddings=True)[0]
