"""FastAPI application exposing the RAG agent via HTTP."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serve.agent import RAGAgent

logger = logging.getLogger(__name__)

_agent: RAGAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the RAGAgent on startup and releases it on shutdown."""
    global _agent
    try:
        _agent = RAGAgent()
        logger.info("RAGAgent loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load RAGAgent: %s", exc)
        _agent = None
    yield
    _agent = None


app = FastAPI(
    title="mini-rag-pipeline",
    description="Retrieval-augmented QA over Kubeflow documentation.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Incoming query payload."""

    question: str


class QueryResponse(BaseModel):
    """Structured response from the RAG agent."""

    answer: str
    sources: list[str]
    steps: list[str]


@app.get("/health")
def health() -> dict:
    """Returns service health and whether the index is loaded."""
    return {"status": "ok", "index_loaded": _agent is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Runs a RAG query and returns an answer with source citations."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded. Check logs.")

    try:
        result = _agent.answer(request.question)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        steps=result["steps"],
    )
