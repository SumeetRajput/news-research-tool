# Equity Research News Tool

A question-answering tool for news. You give it a topic, it finds recent coverage, reads the actual articles, and answers your questions with inline citations back to the source.

Built with LangChain, FAISS, and Streamlit.

---

## The problem it solves

An equity research analyst covering a company reads 20–40 articles before writing a note. Most of that reading is triage: figuring out which three articles actually contain something new. This tool does the triage and hands back a cited synthesis.

You cannot do this by simply asking a chat model "what's the news on Company X". The model's knowledge has a cutoff, and when asked about events past it, it will produce fluent, plausible, wrong answers. The fix is **grounding**: retrieve real documents first, then let the model reason over only those.

---

## Architecture

```
     query
       │
       ▼
┌──────────────────┐   NewsAPI returns metadata only —
│  news_client.py  │   title, URL, 150-char blurb.
│                  │   So we also scrape each URL for
│  search + scrape │   the real body text.
└────────┬─────────┘
         │  list[Document]  (text + title/url/source/date)
         ▼
┌──────────────────┐   Split into ~1000-char chunks.
│  rag_engine.py   │   Embed each chunk into a vector.
│                  │   Store in FAISS.
│  chunk → embed   │   Retrieve top-k for the question.
│  → index → fetch │
└────────┬─────────┘
         │  6 most relevant passages
         ▼
┌──────────────────┐   Prompt template + LLM + parser,
│langchain_config.py│  composed with LCEL (`|`).
└────────┬─────────┘
         │  cited answer
         ▼
┌──────────────────┐
│     app.py       │   Streamlit UI
└──────────────────┘
```

| File | Layer | Responsibility |
|---|---|---|
| `news_client.py` | Retrieval | Query NewsAPI, fetch and clean article body text |
| `rag_engine.py` | Reasoning | Chunk, embed, index, retrieve, assemble cited answer |
| `langchain_config.py` | Model | Groq client, embeddings, prompt templates, chains |
| `audio_io.py` | Voice | Whisper transcription in, Orpheus speech out |
| `report.py` | Export | Render a session as PDF, Markdown or CSV |
| `app.py` | Interface | Streamlit front end, session state, two modes |

---

## Setup

```bash
git clone <your-repo-url>
cd news-research-tool

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env with your keys
streamlit run app.py
```

### Keys you need

Two, both free, no credit card:

| Key | Where | Used for |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys | Reasoning, transcription, speech |
| `NEWSAPI_KEY` | https://newsapi.org/register | Finding articles |

### One provider, three modalities

Everything AI-shaped in this app runs on Groq:

| Job | Model | Free tier |
|---|---|---|
| Reasoning | `openai/gpt-oss-120b` | 30 req/min |
| Speech to text | `whisper-large-v3-turbo` | 2,000 req/day |
| Text to speech | `canopylabs/orpheus-v1-english` | included |
| Embeddings | `all-MiniLM-L6-v2` | runs locally, unlimited |

Embeddings never leave your machine. There is no reason to pay a network
round-trip for something a 90 MB model does on your own CPU in milliseconds,
and it means the vector index costs nothing however many questions you ask.

The reasoning model is chosen from a dropdown that queries Groq's `/models`
endpoint live, so a retired model ID can never silently break the app.

---

## Voice and exports

**Ask by microphone.** The Speak tab records through `st.audio_input()`, sends
the clip to Whisper on Groq, and drops the transcript into the question box.
Groq's Whisper endpoint is OpenAI-compatible and runs on their LPU hardware, so
a ten-second clip comes back in well under a second.

**Hear the answer.** Orpheus reads it aloud. Citation markers and markdown are
stripped before synthesis — the listener has the text on screen, so having the
voice say "open bracket one close bracket" would be noise.

**Take it with you.** Every question and answer in the session, with all
citations, exports three ways:

- **PDF** — a formatted brief built with ReportLab, ready to attach to an email
- **Markdown** — for pasting into notes or a repo
- **CSV** — just the sources, for a spreadsheet

---

## The two modes

**Quick briefing** — sends headlines and blurbs to the LLM in one call. Fast, cheap, no citations, and fairly shallow because NewsAPI blurbs are marketing copy.

**Deep research** — downloads full article text, indexes it, and answers questions against the index with numbered sources. Slower to build (10–30 seconds), then instant for follow-up questions since the index is cached in session state.

Keep both. Running the same question through each is the clearest demonstration of what retrieval actually buys you.

---

## Concepts worth understanding

**Embeddings.** A model that maps text to a vector such that similar meanings land near each other. "Q3 profit fell" and "quarterly earnings declined" share almost no words but sit close in vector space. Keyword search misses that; vector search doesn't.

**Chunking.** Articles are split into ~1000-character passages before embedding. Two reasons: an embedding of a whole 3000-word article averages away everything specific, and you want to retrieve the paragraph that matters rather than the whole document. The 150-character overlap stops a fact being cut in half at a boundary.

**FAISS.** Facebook AI Similarity Search — a library for fast nearest-neighbour lookup over vectors. Here it lives in memory and disappears when the app restarts, which is fine for this scale. Persisting it (`vectorstore.save_local()`) or moving to Chroma/pgvector is the natural next step.

**LCEL.** LangChain Expression Language. `prompt | llm | StrOutputParser()` composes a runnable pipeline. It replaces the older `LLMChain` class and gives you `.invoke()`, `.stream()`, `.batch()` and async on every chain for free.

**Retrieval-Augmented Generation.** The whole pattern: retrieve relevant documents, put them in the prompt, instruct the model to answer only from them. The instruction to say "the excerpts don't cover this" rather than guessing is doing real work — it is what makes the output auditable.

---

## Known limitations

- **NewsAPI free tier** is 100 requests/day, ~24-hour article delay, roughly one month of archive, and is licensed for development only — not for a deployed public app. For production you would move to a paid tier or a source like GNews, NewsData.io, or the Guardian API.
- **Paywalls and JS-rendered pages** return little or no text. Those articles fall back to their NewsAPI blurb, flagged in metadata as `full_text: False`.
- **No deduplication of syndicated wire copy.** The same Reuters story republished across five outlets will occupy five slots in the index.
- **No evaluation harness.** There is currently no automated check that answers are faithful to the retrieved passages.

---

## Where to take it next

1. **Persist the index** — `FAISS.save_local()` / `load_local()` so a topic survives a restart.
2. **Deduplicate** by cosine similarity between article embeddings before indexing.
3. **Evaluate** — build 20 question/answer pairs by hand and score groundedness and retrieval hit-rate. This is the single thing that most separates a portfolio project from a tutorial.
4. **Add a re-ranker** — retrieve 20 chunks by vector similarity, then re-score with a cross-encoder and keep the best 6.
5. **Sentiment and entity extraction** per article, so you can chart tone over time rather than only answering questions.
6. **Streaming responses** — swap `.invoke()` for `.stream()` and write into `st.write_stream()`.

---

## Differences from the original project brief

The brief was written against an older LangChain and has three issues worth knowing about:

1. **Dead imports.** `from langchain import OpenAI, LLMChain, PromptTemplate` no longer resolves. Provider classes moved to partner packages (`langchain-openai`), and `LLMChain` is deprecated in favour of LCEL.
2. **No actual retrieval.** The brief's `summarize_articles()` joins `article['description']` strings — a summary of ~150-character blurbs. There is no chunking, no embedding, no retrieval, and no citations. This version does all four.
3. **Hardcoded API keys.** The brief assigns keys as string literals in source. This version uses `.env` + `python-dotenv`, with `.env` gitignored.

The brief's structure is otherwise sound and the phase order here matches it.
