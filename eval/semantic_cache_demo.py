"""Demonstrates agent/semantic_cache.py's SemanticCache: fires a query,
then a differently-worded but semantically similar follow-up, and shows
the second one hits the cache (no call to the wrapped retriever at all),
with a real before/after latency number for that specific query pair.

Wraps whatever agent/harness.py's _shared_retriever() currently returns --
provisionally the local/pgvector retriever until agent/rag_bedrock_kb.py
(Assignment 3 Phase 8) exists and the AWS Knowledge Base is provisioned.
The dramatic latency delta this is really meant to demonstrate shows up
once this sits in front of a real network-hop Bedrock Retrieve call, not
an already-fast local BM25 lookup -- see README's semantic cache section
for the re-run against the real KB path once that's live.

Usage:
    python -m eval.semantic_cache_demo
"""
from __future__ import annotations

import time

from agent.harness import _shared_retriever
from agent.semantic_cache import SemanticCache

QUERY_1 = "When will my order arrive?"
QUERY_2 = "What's the status of my delivery?"  # semantically similar, zero shared words with QUERY_1's "arrive"


def main():
    wrapped = _shared_retriever()
    cache = SemanticCache(wrapped)

    t0 = time.perf_counter()
    result_1 = cache.retrieve(QUERY_1)
    latency_1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_2 = cache.retrieve(QUERY_2)
    latency_2 = time.perf_counter() - t0

    print("=" * 70)
    print("SEMANTIC CACHE DEMO")
    print("=" * 70)
    print(f"Query 1: {QUERY_1!r}")
    print(f"  latency: {latency_1 * 1000:.2f}ms, cache hits so far: {cache.hits}, misses: {cache.misses}")
    print(f"Query 2: {QUERY_2!r}  (differently worded, semantically similar)")
    print(f"  latency: {latency_2 * 1000:.2f}ms, cache hits so far: {cache.hits}, misses: {cache.misses}")
    print()
    if cache.hits == 1 and cache.misses == 1:
        print(f"CACHE HIT confirmed on query 2. Latency delta: {(latency_1 - latency_2) * 1000:.2f}ms "
              f"({(latency_1 / latency_2 if latency_2 > 0 else float('inf')):.1f}x faster).")
    else:
        print(f"UNEXPECTED: hits={cache.hits} misses={cache.misses} (expected 1 hit, 1 miss) -- "
              "queries may not be similar enough for the configured threshold, or something else changed.")
    print(f"Confirming results are identical (same cached object, not re-computed): {result_1 == result_2}")
    print(f"Wrapped retriever class: {type(wrapped).__name__}")
    print("=" * 70)


if __name__ == "__main__":
    main()
