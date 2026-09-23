from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from eval_harness.config import get_settings

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
COLLECTION_NAME = "fastapi_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TOKENS = 64
CHUNK_OVERLAP_TOKENS = 8

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: int
    text: str
    source_file: str
    chunk_index: int


def load_corpus_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    return sorted(p for p in corpus_dir.glob("*.md") if p.name != "SOURCES.md")


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Token-based chunking with overlap so context isn't lost at chunk boundaries."""
    tokens = _encoding.encode(text)
    if not tokens:
        return []
    pieces = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        pieces.append(_encoding.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return pieces


def build_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = 0
    for path in load_corpus_files(corpus_dir):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(id=next_id, text=piece, source_file=path.name, chunk_index=i))
            next_id += 1
    return chunks


_embedder: SentenceTransformer | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    """Thread-safe lazy singleton. The Phase 9 runner scores questions concurrently across
    a thread pool -- without this lock, multiple threads hitting this on first use each
    started their own `SentenceTransformer(...)` load simultaneously, which segfaulted
    (native PyTorch/BLAS state during concurrent weight loading is not thread-safe)."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(path=settings.qdrant_local_path)


def ingest(client: QdrantClient | None = None, corpus_dir: Path = CORPUS_DIR) -> int:
    """Chunk the corpus, embed it, and (re)upsert into Qdrant. Returns chunk count indexed."""
    chunks = build_chunks(corpus_dir)
    if not chunks:
        raise ValueError(f"No chunks produced from corpus at {corpus_dir}")

    vectors = embed_texts([c.text for c in chunks])
    vector_size = len(vectors[0])

    owns_client = client is None
    client = client or get_qdrant_client()
    try:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=c.id,
                vector=vec,
                payload={"text": c.text, "source_file": c.source_file, "chunk_index": c.chunk_index},
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":
    count = ingest()
    print(f"Indexed {count} chunks into Qdrant collection '{COLLECTION_NAME}'")
