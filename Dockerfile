# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

ENV POETRY_HOME="/opt/poetry"
ENV PATH="${POETRY_HOME}/bin:$PATH"

# Install system packages needed for build + dev
RUN apt-get update && apt-get install -y \
    curl build-essential netcat-openbsd git zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install Taskfile binary and move to /usr/local/bin
RUN curl -sL https://taskfile.dev/install.sh | sh && \
    mv ./bin/task /usr/local/bin/task

# Set working directory
WORKDIR /app

# Copy dependency management files
COPY pyproject.toml poetry.lock Taskfile.yml ./

# Install main dependencies without virtualenv
RUN poetry config virtualenvs.create false && poetry install --only main --no-root

# ✅ Now spaCy is installed — so download the language model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY src/ ./src
# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

# Set environment for Poetry and module imports
# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

# Set environment for Poetry, module resolution, and site-packages
ENV POETRY_HOME="/opt/poetry"
ENV PATH="/opt/poetry/bin:/usr/local/bin:$PATH"
ENV PYTHONPATH="/app/src:/usr/local/lib/python3.11/site-packages"
ENV POETRY_VIRTUALENVS_CREATE=false

# Install only runtime system deps
RUN apt-get update && apt-get install -y \
    netcat-openbsd git zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ⬇️ Copy Python site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/lib/python3.11/site-packages/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/bin/task /usr/local/bin/task


# Copy app files and Poetry installation
COPY --from=builder /app /app
COPY --from=builder /opt/poetry /opt/poetry

WORKDIR /app

# Expose FastAPI API port
EXPOSE 8000

# Wait for Neo4j, then serve FastAPI app directly (no poetry run needed)
CMD /bin/sh -c "sleep 5 && until nc -z neo4j 7687; do echo '⏳ Waiting for Neo4j...'; sleep 2; done && echo '✅ Neo4j is up!' && task neo4j:bootstrap && task serve"
