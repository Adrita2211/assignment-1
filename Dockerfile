FROM python:3.11-slim

WORKDIR /srv

# Build tools only needed to compile faiss-cpu/sentence-transformers deps on
# some platforms; removed from the final layer via --no-install-recommends
# plus apt cleanup to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ agent/
COPY mcp_server/ mcp_server/
COPY policies/ policies/
COPY data/ data/
COPY server.py .

# Pre-download the embedding model at build time so the first request in a
# freshly-started task doesn't pay a ~90MB download + cold-load penalty (and
# doesn't depend on outbound internet access from inside the ECS task at
# runtime, which the security group may not even allow).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8080
ENV AGENT_REGRESSED=false

CMD ["python", "server.py"]
