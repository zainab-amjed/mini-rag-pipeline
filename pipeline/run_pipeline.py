"""Orchestrates the three-stage RAG ingestion pipeline: scrape, chunk, embed."""

import argparse
import logging
import sys
import time
from pathlib import Path

from pipeline.components.chunker import chunk_documents
from pipeline.components.embedder import embed_and_index
from pipeline.components.scraper import scrape_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_INDEX_PATH = "data/index.faiss"
_META_PATH = "data/meta.json"


def _parse_args() -> argparse.Namespace:
    """Parses CLI arguments for pipeline configuration."""
    parser = argparse.ArgumentParser(description="Run the mini-RAG ingestion pipeline.")
    parser.add_argument("--url", default="https://www.kubeflow.org/docs/", help="Base URL to crawl.")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to scrape.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Target characters per chunk.")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap characters between chunks.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing index if present.")
    return parser.parse_args()


def main() -> None:
    """Runs scrape -> chunk -> embed in sequence, logging wall-clock time per stage."""
    args = _parse_args()

    if Path(_INDEX_PATH).exists() and not args.force:
        logger.warning(
            "Index already exists at %s. Pass --force to overwrite.", _INDEX_PATH
        )
        sys.exit(0)

    t0 = time.perf_counter()
    logger.info("Stage 1/3 — Scraping: %s (max %d pages)", args.url, args.max_pages)
    docs = scrape_docs(base_url=args.url, max_pages=args.max_pages)
    logger.info("Stage 1 done in %.1fs. Pages collected: %d", time.perf_counter() - t0, len(docs))

    t1 = time.perf_counter()
    logger.info("Stage 2/3 — Chunking (size=%d, overlap=%d)", args.chunk_size, args.overlap)
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    logger.info("Stage 2 done in %.1fs. Chunks produced: %d", time.perf_counter() - t1, len(chunks))

    t2 = time.perf_counter()
    logger.info("Stage 3/3 — Embedding and indexing")
    embed_and_index(chunks, index_path=_INDEX_PATH, meta_path=_META_PATH)
    logger.info("Stage 3 done in %.1fs.", time.perf_counter() - t2)

    logger.info("Pipeline complete. Total wall time: %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
