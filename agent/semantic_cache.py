"""A lightweight embedding-similarity cache in front of retrieval --
hand-rolled, not GPTCache, same "understand the mechanism" bias as
agent/rag.py's hand-rolled BM25 (a standard-library term-frequency ranker
instead of importing one). The teaching point is semantic vs. exact-match
caching, not the specific library.

Wraps any retriever exposing retrieve(query, top_k) -- the interface
agent/rag.py's HybridPolicyRetriever and agent/rag_pgvector.py's
PgVectorPolicyRetriever already share, and agent/rag_bedrock_kb.py's
BedrockKBRetriever (once it exists) will too. Reuses the SAME embedding
model already used for retrieval (all-MiniLM-L6-v2, already a dependency)
to embed queries for the cache's own similarity check -- no new model,
no extra cost beyond the cache's own bookkeeping.

Intended to sit in front of the Bedrock Knowledge Base retrieval path
specifically (see agent/harness.py's _shared_retriever(), USE_BEDROCK_KB
branch) once that exists -- caching only pays off in front of a real
network-hop retrieval call, which local FAISS/pgvector paths don't
meaningfully need it for at this corpus size.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.rag import _EMBED_MODEL_NAME


# Calibrated empirically against genuine paraphrase pairs vs. genuinely
# different queries (same discipline as agent/rag.py's DEFAULT_MIN_FUSED_SCORE
# calibration) -- all-MiniLM-L6-v2 is a small, fast model whose cosine
# similarity for real paraphrases ("when will my order arrive?" vs.
# "what's the status of my delivery?") lands around 0.54-0.85, while
# genuinely different queries land around 0.10-0.32. 0.5 gives a clean
# margin on both sides; a naively "safe-sounding" threshold like 0.9 (this
# module's first draft) turned out to reject every genuine paraphrase
# tested, catching only near-identical strings -- not what "semantic"
# caching is supposed to mean.
DEFAULT_SIMILARITY_THRESHOLD = 0.5


class SemanticCache:
    def __init__(self, wrapped_retriever, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD, max_entries: int = 500):
        self._wrapped = wrapped_retriever
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._model = SentenceTransformer(_EMBED_MODEL_NAME)
        self._entries: list[tuple[np.ndarray, str, list]] = []  # (query_vec, query_text, result)
        self.hits = 0
        self.misses = 0

    def retrieve(self, query: str, top_k: int = 2):
        qvec = self._model.encode([query], normalize_embeddings=True)[0]
        for cached_vec, cached_text, cached_result in self._entries:
            similarity = float(np.dot(qvec, cached_vec))
            if similarity >= self.similarity_threshold:
                self.hits += 1
                return cached_result

        self.misses += 1
        result = self._wrapped.retrieve(query, top_k)
        self._entries.append((qvec, query, result))
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)  # simple FIFO eviction, fine at this scale
        return result
