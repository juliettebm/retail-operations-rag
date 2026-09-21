"""Reusable orchestration helpers for the exploratory RAG notebook and applications."""
from __future__ import annotations

import os
import re
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from langchain_core.documents import Document


def load_markdown_documents(data_dir: Path) -> list[Document]:
    """Load every Markdown file while retaining its filename as source metadata."""
    from langchain_core.documents import Document

    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(data_dir.glob("*.md"))
    ]


def split_documents(
    documents: Sequence[Document], chunk_size: int = 400, chunk_overlap: int = 50
) -> list[Document]:
    """Split documents with the chunking strategy used by the demonstrator."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vector_store(
    chunks: Sequence[Document], model_name: str = "intfloat/multilingual-e5-small"
) -> tuple[Any, Any]:
    """Create the embedding model and its in-memory FAISS index."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(list(chunks), embedding_model), embedding_model


def retrieval_hit(sources: Sequence[Document], expected_keywords: Sequence[str]) -> bool | None:
    """Return whether every expected keyword occurs in the retrieved passages."""
    if not expected_keywords:
        return None
    text = " ".join(source.page_content.lower() for source in sources)
    return all(keyword.lower() in text for keyword in expected_keywords)


def evaluate_retrieval(store: Any, items: Iterable[Mapping[str, object]], k: int):
    """Evaluate a retriever with the notebook's keyword-based hit criterion."""
    retriever = store.as_retriever(search_kwargs={"k": k})
    rows: list[dict[str, object]] = []
    hits = 0
    in_scope = 0
    for item in items:
        question = str(item["question"])
        sources = retriever.invoke(question)
        hit = retrieval_hit(sources, item.get("mots_cles_attendus", []))
        if hit is not None:
            in_scope += 1
            hits += int(hit)
        rows.append({"question": question, "hit": hit, "n_sources": len(sources)})
    return hits, in_scope, rows


def get_llm(temperature: float = 0.0, model: str | None = None):
    """Instantiate the configured local Ollama chat model."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
        temperature=temperature,
    )


def build_chain(prompt: str, llm=None):
    """Build a text-generation chain from a prompt and an optional chat model."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template(prompt) | (llm or get_llm()) | StrOutputParser()


def context_from_sources(sources: Sequence[Document]) -> str:
    """Join retrieved passages into the context supplied to a language model."""
    return "\n\n".join(source.page_content for source in sources)


def generate_answer(question: str, retriever, generation_chain):
    """Retrieve context and generate an answer, returning both answer and evidence."""
    sources = retriever.invoke(question)
    context = context_from_sources(sources)
    answer = generation_chain.invoke({"context": context, "question": question})
    return answer, sources


def parse_judge_scores(rows: Iterable[Mapping[str, str]]) -> dict[str, object]:
    """Parse fidelity and accuracy scores emitted by the notebook's LLM judge."""
    fidelity_scores: list[int] = []
    accuracy_scores: list[int] = []
    for row in rows:
        evaluation = row["evaluation"]
        fidelity = re.search(r"FID[ÉE]LIT[ÉE]\s*:\s*(\d+)", evaluation, re.IGNORECASE)
        accuracy = re.search(r"EXACTITUDE\s*:\s*(\d+)", evaluation, re.IGNORECASE)
        if fidelity:
            fidelity_scores.append(int(fidelity.group(1)))
        if accuracy:
            accuracy_scores.append(int(accuracy.group(1)))
    return {
        "fidelity_scores": fidelity_scores,
        "accuracy_scores": accuracy_scores,
        "mean_fidelity": mean(fidelity_scores) if fidelity_scores else None,
        "mean_accuracy": mean(accuracy_scores) if accuracy_scores else None,
    }
