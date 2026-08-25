"""Bedrock Knowledge Bases retriever (Assignment 3 §2.3) -- replaces
agent/rag_pgvector.py's hand-rolled SQL query with the managed
`bedrock-agent-runtime` Retrieve API, backed by Aurora PostgreSQL +
pgvector as the Knowledge Base's own vector store (a KB-managed schema,
different from scripts/seed_pgvector.py's hand-rolled policy_embeddings
table -- ingestion for this backend happens through the Knowledge Base's
own sync job over the policy docs in S3, not that script).

Same retrieve(query, top_k) -> [(doc, score_breakdown), ...] interface as
agent/rag.py's HybridPolicyRetriever and agent/rag_pgvector.py's
PgVectorPolicyRetriever, so agent/harness.py's _shared_retriever() can
select this backend without any other code caring which one it's holding
-- and so agent/semantic_cache.py's SemanticCache can wrap it exactly
like it already wraps the other two.

Request/response shapes verified directly against the installed boto3
botocore service model for `bedrock-agent-runtime`'s Retrieve operation
(not assumed from memory) -- see this module's own development notes in
the README's Bedrock Knowledge Base section.
"""
from __future__ import annotations

import os

# NOT the same scale as agent/rag.py's DEFAULT_MIN_FUSED_SCORE (0.38) --
# Bedrock KB's own relevance score is a different metric entirely,
# produced by its own retrieval/reranking pipeline, not this project's
# hand-tuned BM25+vector fusion. MUST be re-tuned empirically against the
# same held-out genuine-vs-adversarial query set the original threshold
# was calibrated against (see agent/rag.py's module docstring) once a
# real Knowledge Base exists to test against -- 0.38 here is a
# placeholder inherited for continuity, not a value carried over on the
# assumption the scales match.
DEFAULT_MIN_SCORE = 0.38


class BedrockKBRetriever:
    def __init__(self, knowledge_base_id: str, region: str | None = None, min_score: float = DEFAULT_MIN_SCORE):
        self.knowledge_base_id = knowledge_base_id
        self.min_score = min_score
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-agent-runtime", region_name=self._region)
        return self._client

    def retrieve(self, query: str, top_k: int = 2):
        response = self.client.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
        )

        results = []
        for r in response.get("retrievalResults", []):
            score = r["score"]
            if score < self.min_score:
                continue
            doc_id = self._doc_id_from_location(r.get("location", {}))
            doc = {"id": doc_id, "text": r["content"].get("text", "")}
            results.append((doc, {"bedrock_kb": round(score, 3)}))
        return results

    @staticmethod
    def _doc_id_from_location(location: dict) -> str:
        """The Knowledge Base ingests policies/*.md from S3 -- the doc's
        "id" (used the same way agent/rag.py's doc ids are, e.g. for
        [refund_eligibility]-style citation in the system prompt) is the
        S3 object's filename stem."""
        s3_uri = location.get("s3Location", {}).get("uri", "")
        filename = s3_uri.rsplit("/", 1)[-1] if s3_uri else "unknown"
        return filename[:-3] if filename.endswith(".md") else filename
