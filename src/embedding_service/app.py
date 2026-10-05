"""Embedding microservice (OpenAI-compatible /v1/embeddings).

Serves an open-source embedding model (sentence-transformers) in a Docker
container. Exposes the standard `/v1/embeddings` endpoint so the pipeline
can use it as if it were OpenAI, but running locally.

The vector dimension is the model's native dimension (e.g. 384 for MiniLM,
1024 for bge-m3). The database adapts to this dimension.
"""
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedding Service")

# Configuration via environment variables
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

print(f"🔄 Loading embedding model: {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
NATIVE_DIM = _model.get_sentence_embedding_dimension()
print(f"✅ Model loaded. Dimension: {NATIVE_DIM}")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "dim": NATIVE_DIM}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embed(req: EmbeddingRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    vectors = _model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    data = [
        EmbeddingData(index=i, embedding=np.asarray(v, dtype=np.float32).tolist())
        for i, v in enumerate(vectors)
    ]
    return EmbeddingResponse(
        data=data,
        model=MODEL_NAME,
        usage={"prompt_tokens": len(texts), "total_tokens": len(texts)},
    )

