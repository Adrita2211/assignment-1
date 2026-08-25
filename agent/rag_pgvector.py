"""pgvector-backed replacement for agent/rag.py's local FAISS index -- the
vector-store choice this assignment locks in for the deployed agent (RDS
PostgreSQL + pgvector, not a local, in-process store; see the README's AWS
architecture section). Keeps the exact same hybrid-fusion approach (BM25
lexical + dense vector, same weights and score threshold) as
agent/rag.py's HybridPolicyRetriever -- only the vector backend moves from an
in-process FAISS index to a pgvector `<=>` (cosine distance) query against a
Postgres table, so agent/harness.py's retrieve_node doesn't need to know or
care which retriever it's holding; both expose the same retrieve(query,
top_k) -> [(doc, score_breakdown), ...] shape.

Migrating the embeddings themselves (embed each policy doc, insert as a
`vector` column) is a one-time job -- see scripts/seed_pgvector.py -- run
that once against a fresh database before pointing DATABASE_URL at it.
"""
from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer

from agent.rag import (
    _BM25,
    BM25_SATURATION_K,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_MIN_FUSED_SCORE,
    DEFAULT_VECTOR_WEIGHT,
    _EMBED_MODEL_NAME,
)

EMBEDDING_TABLE = "policy_embeddings"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output size


class PgVectorPolicyRetriever:
    """Same public interface as HybridPolicyRetriever so
    agent/harness.py's _shared_retriever() can hand back either one
    interchangeably depending on whether DATABASE_URL is set."""

    def __init__(
        self,
        database_url: str,
        policy_dir: str | Path,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        min_fused_score: float = DEFAULT_MIN_FUSED_SCORE,
    ):
        import psycopg
        from pgvector.psycopg import register_vector

        self.docs = []
        for path in sorted(Path(policy_dir).glob("*.md")):
            self.docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
        if not self.docs:
            raise RuntimeError(f"No policy docs found in {policy_dir}")

        self._bm25 = _BM25(self.docs)
        # Query-time embedding still runs locally (the same small model the
        # local FAISS path uses) -- pgvector stores and searches the doc
        # embeddings, it doesn't replace the embedding model itself.
        self._model = SentenceTransformer(_EMBED_MODEL_NAME)
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.min_fused_score = min_fused_score

        self._conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self._conn)
        self._doc_index = {d["id"]: i for i, d in enumerate(self.docs)}

        with self._conn.cursor() as cur:
            cur.execute(f"SELECT doc_id FROM {EMBEDDING_TABLE}")
            seeded_ids = {row[0] for row in cur.fetchall()}
        missing = set(self._doc_index) - seeded_ids
        if missing:
            raise RuntimeError(
                f"{EMBEDDING_TABLE} is missing embeddings for {sorted(missing)}. "
                "Run scripts/seed_pgvector.py against this database first."
            )

    def close(self):
        self._conn.close()

    def _vector_scores(self, query: str) -> list[float]:
        qvec = self._model.encode([query], normalize_embeddings=True)[0]
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT doc_id, 1 - (embedding <=> %s) AS cosine_sim FROM {EMBEDDING_TABLE}",
                (qvec,),
            )
            rows = cur.fetchall()
        out = [0.0] * len(self.docs)
        for doc_id, cosine_sim in rows:
            idx = self._doc_index.get(doc_id)
            if idx is not None:
                out[idx] = max(0.0, float(cosine_sim))  # clip negative cosine to 0, same as the FAISS path
        return out

    def retrieve(self, query: str, top_k: int = 2):
        """Same fusion math as HybridPolicyRetriever.retrieve -- see
        agent/rag.py for the full rationale behind the weights and
        threshold; kept identical here so switching backends can't silently
        change retrieval behavior, only where the vector search runs."""
        bm25_raw = self._bm25.scores(query)
        bm25_norm = [s / (s + BM25_SATURATION_K) for s in bm25_raw]
        vector_norm = self._vector_scores(query)

        fused = [
            self.lexical_weight * l + self.vector_weight * v
            for l, v in zip(bm25_norm, vector_norm)
        ]
        ranked = sorted(zip(self.docs, bm25_norm, vector_norm, fused), key=lambda row: -row[3])

        results = []
        for doc, lscore, vscore, fscore in ranked[:top_k]:
            if fscore >= self.min_fused_score:
                results.append((doc, {"bm25": round(lscore, 3), "vector": round(vscore, 3), "fused": round(fscore, 3)}))
        return results
