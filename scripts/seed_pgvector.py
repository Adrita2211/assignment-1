"""One-time migration: embed every policy doc in policies/ and load them into
RDS PostgreSQL + pgvector (agent/rag_pgvector.py's PgVectorPolicyRetriever
reads from the same table this script writes). Run this once against a fresh
database -- or again any time a policy doc changes -- before pointing
DATABASE_URL at that database for the deployed agent.

Usage:
    export DATABASE_URL=postgresql://user:pass@host:5432/dbname
    python -m scripts.seed_pgvector
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from agent.rag_pgvector import EMBEDDING_DIM, EMBEDDING_TABLE
from agent.rag import _EMBED_MODEL_NAME

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


def main():
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set -- point it at your RDS PostgreSQL instance first.")
        sys.exit(1)

    import psycopg
    from pgvector.psycopg import register_vector

    docs = [
        {"id": path.stem, "text": path.read_text(encoding="utf-8")}
        for path in sorted(POLICY_DIR.glob("*.md"))
    ]
    if not docs:
        print(f"No policy docs found in {POLICY_DIR}")
        sys.exit(1)

    print(f"Embedding {len(docs)} policy docs with {_EMBED_MODEL_NAME}...")
    model = SentenceTransformer(_EMBED_MODEL_NAME)
    embeddings = model.encode([d["text"] for d in docs], normalize_embeddings=True)

    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBEDDING_TABLE} (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL
        )
        """
    )

    for doc, vec in zip(docs, embeddings):
        conn.execute(
            f"""
            INSERT INTO {EMBEDDING_TABLE} (doc_id, text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
            """,
            (doc["id"], doc["text"], vec),
        )
        print(f"  seeded: {doc['id']}")

    conn.close()
    print(f"\nDone -- {len(docs)} rows in {EMBEDDING_TABLE}.")


if __name__ == "__main__":
    main()
