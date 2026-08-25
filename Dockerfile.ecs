FROM python:3.11-slim

WORKDIR /srv

# Build tools only needed to compile sentence-transformers' deps (e.g.
# torch) on some platforms; removed from the final layer via
# --no-install-recommends plus apt cleanup to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch is a sentence-transformers dependency; PyPI's default wheel bundles
# full CUDA/GPU libraries (500MB+) this ECS task never uses. Installing the
# CPU-only build from PyTorch's own index first (~150-200MB) satisfies that
# dependency before requirements.txt's normal install would otherwise pull
# the much larger GPU wheel.
#
# --timeout/--retries: a large multi-hundred-MB install is long enough that
# a single transient read timeout on one package (observed in practice on
# this build) would otherwise kill the whole layer -- pip's own defaults
# (15s timeout, no retry on read timeouts mid-download) are too impatient
# for that.
#
# Split into two RUN layers (not chained with &&) so a successful torch
# install is cached independently -- a retry after a later package's
# transient failure resumes from the cached torch layer instead of
# re-downloading it from scratch.
RUN pip install --no-cache-dir --timeout 120 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

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
