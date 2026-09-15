"""
news_client.py
--------------
The RETRIEVAL layer of the tool.

Two responsibilities, deliberately kept separate:

1. `search_articles()`  -> ask NewsAPI *which* articles exist for a query.
2. `load_documents()`   -> go to each article's URL and pull the *actual body text*.

Why step 2 exists at all:
NewsAPI never returns full article bodies. The `description` field is a
~150-character SEO blurb and the `content` field is truncated at ~200
characters with a "[+3120 chars]" marker. Summarising those fields gives you a
summary of summaries — confident-sounding and nearly information-free. So we
use NewsAPI purely as a *discovery index* and fetch the text ourselves.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from newsapi import NewsApiClient

load_dotenv()
log = logging.getLogger(__name__)

# NewsAPI's free Developer tier only indexes roughly the last month.
MAX_DAYS_BACK = 28

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags that are almost never article body text.
_JUNK_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "figure"]


class NewsClientError(RuntimeError):
    """Raised when the news source cannot be reached or is misconfigured."""


def _client() -> NewsApiClient:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise NewsClientError(
            "NEWSAPI_KEY is not set. Copy .env.example to .env and add your key "
            "from https://newsapi.org/register"
        )
    return NewsApiClient(api_key=key)


def search_articles(
    query: str,
    days_back: int = 14,
    max_articles: int = 10,
    language: str = "en",
    sort_by: str = "relevancy",
) -> list[dict[str, Any]]:
    """Return article *metadata* (title, url, source, date, blurb) for a query.

    `sort_by` is one of: relevancy, popularity, publishedAt.
    """
    days_back = max(1, min(days_back, MAX_DAYS_BACK))
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()

    try:
        payload = _client().get_everything(
            q=query,
            language=language,
            sort_by=sort_by,
            from_param=from_date,
            page_size=min(max(max_articles, 1), 100),
        )
    except Exception as exc:  # newsapi-python raises its own exception types
        raise NewsClientError(f"NewsAPI request failed: {exc}") from exc

    articles = payload.get("articles", [])[:max_articles]

    cleaned = []
    for art in articles:
        url = art.get("url")
        if not url:
            continue
        cleaned.append(
            {
                "title": (art.get("title") or "Untitled").strip(),
                "url": url,
                "source": (art.get("source") or {}).get("name", "Unknown source"),
                "published_at": (art.get("publishedAt") or "")[:10],
                "blurb": (art.get("description") or "").strip(),
            }
        )
    return cleaned


def _extract_body_text(url: str, timeout: int = 12) -> str:
    """Best-effort extraction of readable article text from a URL."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not fetch %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(_JUNK_TAGS):
        tag.decompose()

    # Real article paragraphs are long; navigation links and captions are short.
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True).split()) >= 8
    ]
    return "\n\n".join(paragraphs)


def load_documents(
    articles: list[dict[str, Any]],
    min_chars: int = 400,
    max_workers: int = 8,
) -> list[Document]:
    """Turn article metadata into LangChain Documents containing real body text.

    Fetches run in parallel because they are network-bound, not CPU-bound.
    Paywalled or JS-only pages will return little or nothing; those fall back to
    the NewsAPI blurb so the article is still represented rather than silently
    dropped. Every Document carries its source metadata so answers can cite it.
    """
    if not articles:
        return []

    docs: list[Document] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        bodies = list(pool.map(lambda a: _extract_body_text(a["url"]), articles))

    for article, body in zip(articles, bodies):
        text = body if len(body) >= min_chars else article.get("blurb", "")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=f"{article['title']}\n\n{text}",
                metadata={
                    "title": article["title"],
                    "url": article["url"],
                    "source": article["source"],
                    "published_at": article["published_at"],
                    "full_text": len(body) >= min_chars,
                },
            )
        )
    return docs
