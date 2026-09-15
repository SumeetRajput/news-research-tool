"""
langchain_config.py
-------------------
The MODEL layer. Everything in this application runs on Groq:

    Text reasoning   -> openai/gpt-oss-120b   (this file)
    Speech to text   -> whisper-large-v3-turbo (audio_io.py)
    Text to speech   -> canopylabs/orpheus-v1-english (audio_io.py)
    Embeddings       -> all-MiniLM-L6-v2, local, free (this file)

One provider, one key, three modalities. Embeddings stay local because there is
no reason to pay a network round-trip for something a 90 MB model does on your
own CPU in milliseconds.

Notes on how this differs from the project brief:

* The brief imports `OpenAI`, `LLMChain` and `PromptTemplate` from the top-level
  `langchain` package. That path no longer exists and `LLMChain` is deprecated.
  Current LangChain composes chains with the pipe operator (LCEL):
  `prompt | llm | StrOutputParser()`.
* The brief hardcodes the API key in source. Everything here reads from `.env`.
"""

from __future__ import annotations

import os
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


class ConfigError(RuntimeError):
    """Raised when Groq is not properly configured."""


def is_configured() -> tuple[bool, str]:
    if os.getenv("GROQ_API_KEY"):
        return True, "Groq key found in .env"
    return False, "GROQ_API_KEY is not set in .env"


@lru_cache(maxsize=1)
def list_models() -> list[str]:
    """Ask Groq which chat models it is serving right now.

    Groq retires model IDs regularly. Querying the live list means a retirement
    can never silently break the app — the dropdown simply stops offering it.
    """
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"},
            timeout=8,
        )
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        chat = [
            i
            for i in ids
            if not any(x in i.lower() for x in ("whisper", "tts", "guard", "orpheus", "embed"))
        ]
        return sorted(chat) or FALLBACK_MODELS
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return FALLBACK_MODELS


@lru_cache(maxsize=4)
def get_llm(model: str | None = None):
    """Return a Groq chat model. Cached so Streamlit reruns don't rebuild it."""
    ok, reason = is_configured()
    if not ok:
        raise ConfigError(f"{reason}. Get a free key at https://console.groq.com/keys")

    from langchain_groq import ChatGroq

    return ChatGroq(model=model or DEFAULT_MODEL, temperature=0.1)


def test_connection(model: str | None = None) -> tuple[bool, str]:
    """One tiny prompt to confirm key and model ID both work."""
    try:
        reply = get_llm(model).invoke("Reply with the single word: ready")
        text = getattr(reply, "content", str(reply))
        return True, f"Connected. Model replied: {str(text).strip()[:60]}"
    except ConfigError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"


@lru_cache(maxsize=1)
def get_embeddings():
    """Local sentence-transformer embeddings — no key, no cost, works offline.

    Downloads ~90 MB on first run.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        encode_kwargs={"normalize_embeddings": True},
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_ANALYST_ROLE = (
    "You are a research assistant to an equity research analyst. You are precise, "
    "sceptical, and you never invent facts. If the provided material does not "
    "answer the question, you say so plainly instead of guessing."
)

QUICK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _ANALYST_ROLE),
        (
            "human",
            "Query: {query}\n\n"
            "Recent headlines and blurbs:\n{headlines}\n\n"
            "Write a short briefing (5-7 bullets) covering the main themes, then one "
            "line on what is still unclear from these headlines alone. Do not "
            "speculate beyond what is written above.",
        ),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _ANALYST_ROLE + " You cite every factual claim."),
        (
            "human",
            "Answer the question using ONLY the numbered excerpts below.\n\n"
            "Rules:\n"
            "1. Cite sources inline as [1], [2] etc., matching the excerpt numbers.\n"
            "2. If the excerpts disagree, say so and cite both sides.\n"
            "3. If the excerpts do not contain the answer, say exactly what is "
            "missing rather than filling the gap from memory.\n"
            "4. Separate reported fact from opinion or analyst commentary.\n"
            "5. Write for the ear as well as the eye — this answer may be read "
            "aloud, so keep sentences clean and avoid dense nested clauses.\n\n"
            "Question: {question}\n\n"
            "Excerpts:\n{context}\n\n"
            "Write a concise answer (under 250 words), then a short "
            "'What to watch' line naming the key open question.",
        ),
    ]
)


def build_quick_summary_chain(model: str | None = None):
    """prompt -> llm -> string. The LCEL replacement for LLMChain."""
    return QUICK_SUMMARY_PROMPT | get_llm(model) | StrOutputParser()


def build_rag_chain(model: str | None = None):
    return RAG_PROMPT | get_llm(model) | StrOutputParser()
