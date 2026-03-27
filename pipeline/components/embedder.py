"""Embeds text chunks and builds a FAISS cosine-similarity index."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_and_index(
    chunks: list[dict],
    index_path: str = "data/index.faiss",
    meta_path: str = "data/meta.json",
) -> None:
    """Embeds chunks with all-MiniLM-L6-v2 and writes a FAISS IndexFlatIP to disk.

    Vectors are L2-normalized before insertion so inner product equals cosine similarity.
    Chunk metadata is saved as a JSON array parallel to the FAISS index.
    """
    if not chunks:
        logger.warning("No chunks provided; skipping indexing.")
        return

    model = SentenceTransformer(_MODEL_NAME)
    texts = [c["text"] for c in chunks]

    logger.info("Encoding %d chunks with %s...", len(texts), _MODEL_NAME)
    vectors: np.ndarray = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)

    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to %s (%d vectors, dim=%d)", index_path, index.ntotal, dimension)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Metadata saved to %s", meta_path)
