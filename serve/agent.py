"""LangGraph ReAct agent that retrieves from FAISS and generates answers via an LLM."""

import json
import logging
import os
from typing import TypedDict

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_INDEX_PATH = "data/index.faiss"
_META_PATH = "data/meta.json"


class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""

    query: str
    retrieved_chunks: list[dict]
    answer: str
    sources: list[str]
    steps: list[str]


class RAGAgent:
    """Retrieval-augmented agent backed by FAISS and a ChatOpenAI LLM."""

    def __init__(
        self,
        index_path: str = _INDEX_PATH,
        meta_path: str = _META_PATH,
    ) -> None:
        """Loads the FAISS index, metadata, and embedding model from disk."""
        self._embedder = SentenceTransformer(_MODEL_NAME)
        self._index = faiss.read_index(index_path)

        with open(meta_path, encoding="utf-8") as f:
            self._metadata: list[dict] = json.load(f)

        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        self._graph = self._build_graph()
        logger.info("RAGAgent ready. Index vectors: %d", self._index.ntotal)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Embeds query, searches the FAISS index, and returns top_k matching chunks."""
        vector = self._embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vector)
        distances, indices = self._index.search(vector, top_k)
        results: list[dict] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def _node_search(self, state: AgentState) -> AgentState:
        """Retrieves relevant chunks for the query and records the step."""
        chunks = self.retrieve(state["query"])
        state["retrieved_chunks"] = chunks
        state["steps"] = state.get("steps", []) + [
            f"Retrieved {len(chunks)} chunks from FAISS index."
        ]
        return state

    def _node_generate(self, state: AgentState) -> AgentState:
        """Builds a grounded prompt from retrieved chunks and calls the LLM."""
        context_parts = []
        sources: list[str] = []
        for chunk in state["retrieved_chunks"]:
            context_parts.append(f"[{chunk['title']}]\n{chunk['text']}")
            if chunk["source_url"] not in sources:
                sources.append(chunk["source_url"])

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"You are a helpful assistant answering questions about Kubeflow.\n\n"
            f"Use only the context below to answer. If the context does not contain "
            f"enough information, say so explicitly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}\n\n"
            f"Answer:"
        )

        response = self._llm.invoke(prompt)
        state["answer"] = response.content
        state["sources"] = sources
        state["steps"] = state.get("steps", []) + [
            f"Generated answer using {len(state['retrieved_chunks'])} context chunks."
        ]
        return state

    def _build_graph(self) -> object:
        """Constructs and compiles the two-node LangGraph StateGraph."""
        builder: StateGraph = StateGraph(AgentState)
        builder.add_node("search", self._node_search)
        builder.add_node("generate", self._node_generate)
        builder.set_entry_point("search")
        builder.add_edge("search", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def answer(self, query: str) -> dict:
        """Runs the full search-then-generate graph and returns a structured result."""
        initial_state: AgentState = {
            "query": query,
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "steps": [],
        }
        final_state: AgentState = self._graph.invoke(initial_state)
        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "steps": final_state["steps"],
        }
