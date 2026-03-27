# mini-rag-pipeline

## Overview

A self-contained RAG pipeline that scrapes the Kubeflow documentation, builds a FAISS
vector index, and serves a LangGraph ReAct agent through FastAPI. It exists to
dogfood Kubeflow Pipeline component concepts locally — each ingestion stage maps
directly to a KFP component — while remaining runnable with a single `docker compose up`
on any machine.

## Architecture

The system has three ingestion stages and one serving layer. The **scraper** crawls
`kubeflow.org/docs` using httpx and BeautifulSoup, following internal links up to a
configurable page limit and returning clean text per page. The **chunker** splits each
page recursively on semantic boundaries — paragraphs first, then sentences, then words
— producing overlapping chunks with deterministic SHA-256 IDs. The **embedder** encodes
every chunk with `all-MiniLM-L6-v2` and writes a normalized FAISS `IndexFlatIP` to
disk alongside a JSON metadata file. The **serving layer** loads both artifacts at
startup, runs a two-node LangGraph graph (retrieve then generate), and exposes the
result through a FastAPI endpoint.

## Quickstart

```bash
git clone https://github.com/zainab-amjed/mini-rag-pipeline.git
cd mini-rag-pipeline

# Set your OpenAI key
export OPENAI_API_KEY=sk-...

# Run the full pipeline (scrape + embed) and start the API
docker compose up

# In a second terminal, send a query
make query
```

For a faster test run that scrapes only 10 pages:

```bash
make ingest   # scrapes 10 pages, builds index
make serve    # starts API on :8000
make query    # sends a sample question
```

## Example Output

```json
{
  "answer": "Kubeflow Pipelines passes artifacts between components using the KFP
artifact store, typically backed by Google Cloud Storage or MinIO. Each component
declares its outputs as typed artifacts (Dataset, Model, Metrics), and the pipeline
runtime resolves the storage URI and injects it into the next component as an input.
The component code itself only reads from a local path; the platform handles the
transfer.",
  "sources": [
    "https://www.kubeflow.org/docs/components/pipelines/user-guides/components/",
    "https://www.kubeflow.org/docs/components/pipelines/concepts/component/"
  ],
  "steps": [
    "Retrieved 5 chunks from FAISS index.",
    "Generated answer using 5 context chunks."
  ]
}
```

## Design Decisions

The chunker uses recursive character splitting rather than a fixed sliding window
because Kubeflow documentation is structured prose — paragraphs, headings, and code
blocks — and slicing it at arbitrary character offsets breaks sentences mid-thought.
Splitting on double newlines first preserves paragraph coherence; the 64-character
overlap ensures a query matching content near a split boundary finds it in at least
one of the adjacent chunks.

FAISS `IndexFlatIP` was chosen over HNSW or IVF because the index holds at most a few
thousand vectors — one scrape of the Kubeflow docs site. At that scale, exact search
with a flat index returns correct results in milliseconds, fits in a few hundred
megabytes of memory, and requires no training or hyperparameter tuning. The upgrade
path to approximate search is a one-line change once the corpus grows.

The LangGraph two-node graph separates retrieval from generation so that each step's
output is inspectable, the graph is extensible without touching existing nodes, and
the execution trace surfaces in the API response. A single function could do the same
work, but adding query rewriting, relevance grading, or tool-calling later would require
a full rewrite rather than adding edges and nodes to an existing graph.
