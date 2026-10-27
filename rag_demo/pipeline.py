from __future__ import annotations

from rag_demo.generator import generate_answer
from rag_demo.retriever import HybridRetriever


class DemoRAGPipeline:
    """The harness's test subject. Satisfies eval_harness.interfaces.pipeline.RAGPipelineInterface
    structurally -- no inheritance, just a matching `query` method -- exactly like any real
    adapter (e.g. a future VarnueVed adapter) would.
    """

    def __init__(self, retriever: HybridRetriever | None = None, top_k: int = 4):
        self.retriever = retriever or HybridRetriever()
        self.top_k = top_k

    def query(self, question: str) -> dict:
        chunks = self.retriever.retrieve(question, k=self.top_k)
        # Order is meaningful: contexts[0] is the best match. The harness's hallucination
        # metric reports citations as positional indices into this list, so reordering or
        # filtering here would silently change what a cited_chunk_id refers to.
        contexts = [c.text for c in chunks]
        answer = generate_answer(question, contexts)
        return {"answer": answer, "contexts": contexts}
