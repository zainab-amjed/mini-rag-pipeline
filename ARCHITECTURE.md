# Architecture

## Pipeline Structure and the Kubeflow Component Model

This repo is structured to mirror a real Kubeflow Pipelines project without requiring a
running Kubernetes cluster. Each file in `pipeline/components/` maps to a discrete KFP
component: `scraper.py`, `chunker.py`, and `embedder.py` are pure Python functions with
typed signatures, explicit inputs, and explicit outputs. The orchestration in
`run_pipeline.py` plays the role of the pipeline definition, wiring components together
and passing data between them.

The critical difference between this codebase and a production KFP deployment is the
artifact boundary. Here, components communicate through in-memory Python objects:
`scrape_docs` returns a list of dicts that flows directly into `chunk_documents`. In KFP,
each function would carry the `@component` decorator, and those raw Python types would
become `Input` and `Output` artifact types — `Dataset`, `Model`, and so on. Artifact
metadata would be tracked by the KFP metadata store, and each component would run in its
own container. The function bodies stay nearly identical; the interface layer is what
changes. That locality of change is the point — writing components this way makes
migration to KFP mostly mechanical.

## Chunking Strategy

The chunker uses recursive character splitting rather than a fixed sliding window.
Fixed-size windowing is the simplest approach but poorly suited for documentation, where
meaningful units are paragraphs and sentences rather than arbitrary character runs.
Cutting mid-sentence loses context for the current chunk and bleeds irrelevant text into
the next one.

The recursive splitter tries semantic boundaries first: double newlines separate
paragraphs, single newlines separate list items and code lines, period-space separates
sentences, space separates words. The algorithm descends that list until it finds a
separator that produces sub-chunks within the target size. The result is an index where
each chunk contains a complete thought. The 64-character overlap ensures that content
near a split point appears in both neighbors, so a query never misses information that
happened to land on a boundary.

## FAISS Index Choice

The index uses `IndexFlatIP` rather than HNSW or IVF. HNSW gives sub-linear query time
at the cost of approximate results and significant per-vector memory overhead. IVF
requires a training pass and a well-tuned `nlist` parameter, and its accuracy degrades
when the query distribution differs from training data. Both trade-offs only pay off past
tens of millions of vectors. At the scale of a single documentation site, `IndexFlatIP`
performs exact search in milliseconds, fits in memory without tuning, and never returns
a wrong nearest neighbor. Vectors are L2-normalized before insertion, converting inner
product into cosine similarity — magnitude-invariant and measuring semantic direction in
embedding space.

## The LangGraph Two-Node Loop

The agent is a minimal `StateGraph` with two nodes: `search` and `generate`. This is the
smallest structure that qualifies as agentic — shared mutable state, explicit data flow
between steps, and a clean separation between retrieval and generation. Collapsing both
into one function would work, but it would make intermediate state invisible and force a
rewrite to add new capabilities. A fuller agent would extend this graph with a
query-rewriting node, a relevance grader that filters low-scoring chunks, and a
tool-calling node for structured actions. LangGraph makes that extension additive: new
nodes and edges slot in without touching the existing ones.
