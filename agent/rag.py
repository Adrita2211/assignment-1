"""Hybrid retrieval over the policy docs: BM25 (lexical) fused with dense
embedding similarity (semantic), so a question that shares zero words with a
doc's actual wording -- a true synonym case, e.g. "compensate" vs. "refund"
-- can still be found. BM25 alone has a hard ceiling here: it can only ever
rank docs that share vocabulary with the query.

- Lexical signal: BM25, standard library only. Term-frequency saturation
  (repeating a word doesn't linearly inflate its score) and corpus-average
  length normalization, both of which plain TF-IDF cosine lacks.
- Semantic signal: a small local sentence-transformers model
  (all-MiniLM-L6-v2, ~90MB, downloaded once and cached by huggingface_hub)
  indexed in FAISS (IndexFlatIP -- exact inner-product search; embeddings
  are L2-normalized, so inner product equals cosine similarity). Runs
  locally, no external API call, no per-query cost or latency to a
  provider. If FAISS or sentence-transformers isn't installed, or the model
  can't be loaded (offline with no cache yet), the retriever logs a warning
  once and falls back to BM25-only -- the agent stays usable, just
  lexical-only, rather than crashing.

Anti-hallucination: retrieve() only returns docs whose FUSED score clears
MIN_FUSED_SCORE. Below that bar, it returns nothing, and the harness treats
an empty result as "no policy covers this" -- the system prompt then tells
the model to say so explicitly rather than answer from its own guess. That
threshold decision happens here, at retrieval time, not left to the model's
judgment about whether it "feels" grounded.
"""
import math
import re
from collections import Counter
from pathlib import Path

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "for", "to", "in", "on", "at", "and", "or", "but", "if", "so",
    "this", "that", "these", "those", "it", "its", "i", "my", "me", "you",
    "your", "do", "does", "did", "can", "could", "will", "would", "should",
    "has", "have", "had", "with", "as", "by", "not", "no", "any", "all",
    "what", "when", "how", "still", "get", "got", "out", "something",
    "turn", "am", "im",
    # domain-generic terms that appear in nearly every policy doc in this
    # small corpus and would otherwise dominate lexical scoring on the
    # strength of a single shared word
    "order", "orders", "customer", "customers", "item", "items",
}

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# Weighted toward the vector signal deliberately: on this corpus, BM25 (even
# saturated) still lets an off-topic query through whenever it happens to
# share a single real word with a doc (e.g. "what's the weather" vs.
# shipping_delay.md's "weather-related delays") -- lexical overlap alone
# can't tell "shares a word" from "is actually about the same topic." The
# embedding model can: calibrated against a held-out set of 10 genuine
# questions vs. 5 adversarial-but-plausible off-topic ones (see README),
# lexical=0.2/vector=0.8 gave a clean gap (every genuine match >=0.40,
# every off-topic query <=0.37) that a 50/50 split did not.
DEFAULT_LEXICAL_WEIGHT = 0.2
DEFAULT_VECTOR_WEIGHT = 0.8
DEFAULT_MIN_FUSED_SCORE = 0.38
# BM25_SATURATION_K calibrates raw BM25 scores into [0, 1) via raw/(raw+K)
# instead of dividing by the max score in this query's ranking. Max-based
# normalization has a real flaw on a small corpus: if only one doc shares
# even a single incidental word with the query, it gets normalized to a
# full 1.0 regardless of how weak that overlap actually is. Saturating
# against a fixed reference (tuned so raw scores from a genuine multi-word
# match land around 0.6-0.7, and a single-weak-word match lands well below
# 0.5) keeps weak lexical matches weak instead of always rescaling one to
# "perfect."
BM25_SATURATION_K = 3.0


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class _BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tokens = [_tokenize(d["text"]) for d in docs]
        self._lengths = [len(toks) for toks in self._tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df = Counter()
        for toks in self._tokens:
            self._df.update(set(toks))
        self.n_docs = len(docs)

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        # "+1" inside the log keeps common terms non-negative on a small corpus.
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1)

    def scores(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        out = []
        for i in range(len(self.docs)):
            tf = Counter(self._tokens[i])
            doc_len = self._lengths[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_len))
                s += idf * (f * (self.k1 + 1)) / denom
            out.append(s)
        return out


class _VectorIndex:
    """Lazy, best-effort local FAISS embedding index. `available` is False
    if the backend couldn't be set up, in which case every score is 0.0 and
    the hybrid retriever should shift full weight onto BM25."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.available = False
        self._model = None
        self._index = None
        self._np = None
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._np = np
            try:
                # Once the model is cached locally (true after the first
                # run), load straight from disk -- no Hub network call, no
                # "unauthenticated requests" warning, and faster startup.
                self._model = SentenceTransformer(_EMBED_MODEL_NAME, local_files_only=True)
            except Exception:
                # Not cached yet (genuinely first run on this machine) --
                # fall through to a normal load, which downloads it once.
                self._model = SentenceTransformer(_EMBED_MODEL_NAME)
            doc_vecs = self._model.encode([d["text"] for d in docs], normalize_embeddings=True)
            doc_vecs = np.asarray(doc_vecs, dtype="float32")

            # IndexFlatIP = exact (brute-force) inner-product search. With
            # L2-normalized vectors, inner product == cosine similarity.
            # Exact, not approximate, is the right call at this corpus size
            # (a handful of docs) -- an ANN index (IVF/HNSW) only starts
            # paying off at thousands+ vectors.
            self._index = faiss.IndexFlatIP(doc_vecs.shape[1])
            self._index.add(doc_vecs)
            self.available = True
        except Exception as exc:  # missing package, no network, corrupt cache, etc.
            print(
                f"[RAG] vector backend unavailable ({type(exc).__name__}: {exc}) "
                "-- falling back to BM25-only retrieval."
            )

    def scores(self, query: str) -> list[float]:
        if not self.available:
            return [0.0] * len(self.docs)
        qvec = self._model.encode([query], normalize_embeddings=True)
        qvec = self._np.asarray(qvec, dtype="float32")

        # Search for every doc, not just a top-k, so retrieve() has a full
        # score vector to fuse with BM25 -- FAISS returns results sorted by
        # score with their original index, which we scatter back into
        # doc-index order.
        similarities, indices = self._index.search(qvec, len(self.docs))
        out = [0.0] * len(self.docs)
        for score, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            out[idx] = float(score)
        return out


class HybridPolicyRetriever:
    def __init__(
        self,
        policy_dir: str | Path,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        min_fused_score: float = DEFAULT_MIN_FUSED_SCORE,
    ):
        self.docs = []
        for path in sorted(Path(policy_dir).glob("*.md")):
            self.docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
        if not self.docs:
            raise RuntimeError(f"No policy docs found in {policy_dir}")

        self._bm25 = _BM25(self.docs)
        self._vector = _VectorIndex(self.docs)

        if self._vector.available:
            self.lexical_weight = lexical_weight
            self.vector_weight = vector_weight
        else:
            # No semantic signal available -- put all the weight on lexical
            # instead of silently discarding half the score budget.
            self.lexical_weight = 1.0
            self.vector_weight = 0.0

        self.min_fused_score = min_fused_score

    def retrieve(self, query: str, top_k: int = 2):
        """Returns up to top_k (doc, score_breakdown) pairs, best first,
        where score_breakdown = {"bm25": ..., "vector": ..., "fused": ...}.
        Returns [] if nothing clears min_fused_score -- an honest gap."""
        bm25_raw = self._bm25.scores(query)
        bm25_norm = [s / (s + BM25_SATURATION_K) for s in bm25_raw]

        vector_raw = self._vector.scores(query)
        vector_norm = [max(0.0, s) for s in vector_raw]  # clip negative cosine to 0

        fused = [
            self.lexical_weight * l + self.vector_weight * v
            for l, v in zip(bm25_norm, vector_norm)
        ]

        ranked = sorted(
            zip(self.docs, bm25_norm, vector_norm, fused),
            key=lambda row: -row[3],
        )

        results = []
        for doc, lscore, vscore, fscore in ranked[:top_k]:
            if fscore >= self.min_fused_score:
                results.append((doc, {"bm25": round(lscore, 3), "vector": round(vscore, 3), "fused": round(fscore, 3)}))
        return results
