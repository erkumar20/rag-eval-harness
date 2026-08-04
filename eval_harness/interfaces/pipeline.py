from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class PipelineOutput(TypedDict):
    answer: str
    contexts: list[str]


@runtime_checkable
class RAGPipelineInterface(Protocol):
    """The only contract any RAG pipeline must satisfy to be evaluated by this harness.

    Structural, not nominal: an adapter does not need to inherit from this class, it just
    needs a `query` method with this shape. That's what lets a 10-20 line adapter plug an
    arbitrary pipeline (LangChain chain, REST API, in-process function) into the harness.
    """

    def query(self, question: str) -> PipelineOutput: ...
