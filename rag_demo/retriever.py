from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from rag_demo.ingest import COLLECTION_NAME, embed_texts, get_qdrant_client

RRF_K = 60


@dataclass
class RetrievedChunk:
    id: int
    text: str
    source_file: str
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Merge multiple ranked id lists into one score per id: score = sum(1 / (k + rank)).

    Standard hybrid-search fusion technique -- robust to vector cosine scores and BM25
    scores living on completely different scales, unlike naive score averaging.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """Vector similarity (Qdrant) + BM25 keyword search over the same corpus, merged via RRF."""

    def __init__(self, client: QdrantClient | None = None):
        self.client = client or get_qdrant_client()
        self._bm25: BM25Okapi | None = None
        self._chunks_by_id: dict[int, RetrievedChunk] = {}
        self._bm25_ids: list[int] = []
        self._load_index()

    def _load_index(self) -> None:
        points, _ = self.client.scroll(
            collection_name=COLLECTION_NAME, limit=10_000, with_payload=True, with_vectors=False
        )
        # Qdrant types point ids as int | str | UUID because it accepts all three, and
        # payloads as optional. Ingest only ever writes integer ids with both payload keys
        # present, so narrowing here is safe -- and asserting it means a malformed collection
        # fails loudly at load instead of surfacing as a confusing error mid-run.
        self._chunks_by_id = {}
        for point in points:
            assert isinstance(point.id, int), f"expected integer point id, got {point.id!r}"
            assert point.payload is not None, f"point {point.id} has no payload"
            self._chunks_by_id[point.id] = RetrievedChunk(
                id=point.id,
                text=point.payload["text"],
                source_file=point.payload["source_file"],
                score=0.0,
            )
        self._bm25_ids = list(self._chunks_by_id.keys())
        tokenized = [_tokenize(self._chunks_by_id[cid].text) for cid in self._bm25_ids]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _vector_ranking(self, question: str, pool_size: int) -> list[int]:
        query_vector = embed_texts([question])[0]
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=pool_size
        ).points
        # Same int-id narrowing as _load_index; ingest only writes integer ids.
        return [hit.id for hit in hits if isinstance(hit.id, int)]

    def _bm25_ranking(self, question: str, pool_size: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(question))
        ranked = sorted(zip(self._bm25_ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [cid for cid, _ in ranked[:pool_size]]

    def retrieve(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        pool_size = max(k * 3, 10)
        vector_ranking = self._vector_ranking(question, pool_size)
        bm25_ranking = self._bm25_ranking(question, pool_size)

        fused = _reciprocal_rank_fusion([vector_ranking, bm25_ranking])
        # reverse=True is load-bearing: RRF scores are higher-is-better, so flipping this
        # silently returns the *worst* candidates. Nothing crashes and no test fails -- it
        # only shows up as a quality drop, which is what the eval gate exists to catch.
        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]

        return [
            RetrievedChunk(
                id=cid,
                text=self._chunks_by_id[cid].text,
                source_file=self._chunks_by_id[cid].source_file,
                score=fused[cid],
            )
            for cid in top_ids
        ]
