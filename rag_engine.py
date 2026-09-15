"""
rag_engine.py
-------------
The REASONING layer. This is the part the project brief leaves out, and the
part that turns "a tutorial I followed" into "a system I built".

The problem it solves: ten full news articles is roughly 15,000-25,000 words.
You could stuff all of it into a modern long-context model, but you shouldn't —
it is slow, expensive per query, and accuracy degrades when the relevant
sentence is buried in noise. Worse, you get no way to trace a claim back to a
source, which is disqualifying in equity research.

So instead: split -> embed -> index -> retrieve the few passages that actually
bear on the question -> answer over just those, with citations.
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_config import build_rag_chain, get_embeddings


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Cut articles into retrievable passages.

    RecursiveCharacterTextSplitter tries paragraph breaks first, then line
    breaks, then sentences, then words — so it splits at natural seams instead
    of slicing mid-sentence. The overlap keeps a fact from being orphaned when
    it happens to straddle a boundary.

    ~1000 characters is a reasonable default: big enough to hold a complete
    thought, small enough that six of them fit comfortably in a prompt.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(documents: list[Document], **split_kwargs) -> FAISS:
    """Embed the chunks and put them in an in-memory FAISS vector store."""
    chunks = split_documents(documents, **split_kwargs)
    if not chunks:
        raise ValueError("No text could be extracted from these articles.")
    return FAISS.from_documents(chunks, get_embeddings())


def _format_context(chunks: list[Document]) -> tuple[str, list[dict[str, Any]]]:
    """Number the retrieved chunks and build a matching source list.

    Chunks from the same article share one citation number, so the model sees
    [1], [2], [3] rather than five near-duplicate references to one story.
    """
    numbering: dict[str, int] = {}
    sources: list[dict[str, Any]] = []
    blocks: list[str] = []

    for chunk in chunks:
        url = chunk.metadata.get("url", "")
        if url not in numbering:
            numbering[url] = len(numbering) + 1
            sources.append(
                {
                    "n": numbering[url],
                    "title": chunk.metadata.get("title", "Untitled"),
                    "url": url,
                    "source": chunk.metadata.get("source", ""),
                    "published_at": chunk.metadata.get("published_at", ""),
                }
            )
        n = numbering[url]
        header = (
            f"[{n}] {chunk.metadata.get('title', 'Untitled')} — "
            f"{chunk.metadata.get('source', '')}, {chunk.metadata.get('published_at', '')}"
        )
        blocks.append(f"{header}\n{chunk.page_content}")

    return "\n\n---\n\n".join(blocks), sources


def answer_question(
    question: str,
    vectorstore: FAISS,
    k: int = 6,
    model: str | None = None,
) -> dict[str, Any]:
    """Retrieve the k most relevant passages and answer over only those."""
    chunks = vectorstore.similarity_search(question, k=k)
    if not chunks:
        return {"answer": "Nothing in the indexed articles relates to that question.", "sources": []}

    context, sources = _format_context(chunks)
    answer = build_rag_chain(model).invoke({"question": question, "context": context})
    return {"answer": answer, "sources": sources, "chunks_used": len(chunks)}
